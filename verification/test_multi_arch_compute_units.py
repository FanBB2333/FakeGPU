#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path


CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81
CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82
CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106

CUDA_SUCCESS = 0

ARCH_TO_CC_MAJOR: dict[str, int] = {
    "maxwell": 5,
    "pascal": 6,
    "volta": 7,
    "turing": 7,
    "ampere": 8,
    "hopper": 9,
    "ada": 8,
    "blackwell": 10,
}


def _die(msg: str) -> None:
    print(f"[test_multi_arch_compute_units] ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _parse_devices_spec(spec: str) -> list[str]:
    result: list[str] = []
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        if ":" in token:
            preset, count_s = token.split(":", 1)
            preset = preset.strip()
            count = int(count_s.strip())
            if count <= 0:
                continue
            result.extend([preset] * count)
        else:
            result.append(token)
    return result


def _expected_device_profiles() -> list[str]:
    profiles_env = os.environ.get("FAKEGPU_PROFILES")
    if profiles_env:
        profiles = _parse_devices_spec(profiles_env)
        if profiles:
            return profiles

    preset = os.environ.get("FAKEGPU_PROFILE", "a100").strip() or "a100"
    count_s = os.environ.get("FAKEGPU_DEVICE_COUNT", "8").strip()
    try:
        count = int(count_s)
    except ValueError:
        count = 8
    if count <= 0:
        count = 8
    return [preset] * count


def _find_preloaded_path(libname: str) -> str | None:
    preload = os.environ.get("LD_PRELOAD", "")
    for part in preload.split(":"):
        part = part.strip()
        if not part:
            continue
        if Path(part).name == libname:
            return part
    return None


def _load_libcuda() -> ctypes.CDLL:
    path = _find_preloaded_path("libcuda.so.1")
    return ctypes.CDLL(path or "libcuda.so.1", mode=ctypes.RTLD_GLOBAL)


def _parse_simple_yaml(path: Path) -> dict[str, object]:
    scalars: dict[str, str] = {}
    lists: dict[str, list[str]] = {}
    current_list: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("- "):
            if not current_list:
                _die(f"{path}: list item without a key: {raw_line!r}")
            lists.setdefault(current_list, []).append(line[2:].strip())
            continue

        if ":" not in line:
            _die(f"{path}: invalid line (missing colon): {raw_line!r}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            scalars[key] = value
            current_list = None
        else:
            current_list = key
            lists.setdefault(key, [])

    out: dict[str, object] = dict(scalars)
    out.update(lists)
    return out


def _require_int(obj: dict[str, object], key: str, *, ctx: str) -> int:
    if key not in obj:
        _die(f"missing key '{key}' in {ctx}")
    raw = obj[key]
    if not isinstance(raw, str):
        _die(f"{ctx}.{key} must be a scalar string, got {type(raw)}")
    try:
        return int(raw, 0)
    except ValueError:
        _die(f"{ctx}.{key} must be an int, got {raw!r}")


def _require_str(obj: dict[str, object], key: str, *, ctx: str) -> str:
    if key not in obj:
        _die(f"missing key '{key}' in {ctx}")
    raw = obj[key]
    if not isinstance(raw, str):
        _die(f"{ctx}.{key} must be a string, got {type(raw)}")
    return raw


@dataclass(frozen=True)
class ExpectedDevice:
    preset_id: str
    name: str
    cc_major: int
    cc_minor: int
    sm_count: int
    shared_mem_per_sm: int
    regs_per_multiprocessor: int
    max_threads_per_multiprocessor: int
    max_blocks_per_multiprocessor: int


def _load_expected_device(*, repo_root: Path, preset_id: str) -> ExpectedDevice:
    profile_path = repo_root / "profiles" / f"{preset_id}.yaml"
    if not profile_path.is_file():
        _die(f"profile YAML not found: {profile_path}")
    data = _parse_simple_yaml(profile_path)
    ctx = f"profiles/{preset_id}.yaml"

    name = _require_str(data, "name", ctx=ctx)
    arch = _require_str(data, "architecture", ctx=ctx).strip().lower()
    if arch not in ARCH_TO_CC_MAJOR:
        _die(f"{ctx}.architecture: unknown architecture {arch!r}")
    cc_major = ARCH_TO_CC_MAJOR[arch]

    return ExpectedDevice(
        preset_id=preset_id,
        name=name,
        cc_major=cc_major,
        cc_minor=_require_int(data, "compute_minor", ctx=ctx),
        sm_count=_require_int(data, "sm_count", ctx=ctx),
        shared_mem_per_sm=_require_int(data, "shared_mem_per_sm", ctx=ctx),
        regs_per_multiprocessor=_require_int(data, "regs_per_multiprocessor", ctx=ctx),
        max_threads_per_multiprocessor=_require_int(data, "max_threads_per_multiprocessor", ctx=ctx),
        max_blocks_per_multiprocessor=_require_int(data, "max_blocks_per_multiprocessor", ctx=ctx),
    )


def _cu_check(result: int, *, what: str) -> None:
    if int(result) != CUDA_SUCCESS:
        _die(f"{what} failed: CUresult={int(result)}")


def _cu_device_get_attribute(libcuda: ctypes.CDLL, device: int, attrib: int) -> int:
    value = ctypes.c_int()
    _cu_check(
        int(libcuda.cuDeviceGetAttribute(ctypes.byref(value), ctypes.c_int(attrib), ctypes.c_int(device))),
        what=f"cuDeviceGetAttribute(dev={device}, attrib={attrib})",
    )
    return int(value.value)


def _cu_device_get_name(libcuda: ctypes.CDLL, device: int) -> str:
    buf = ctypes.create_string_buffer(256)
    _cu_check(
        int(libcuda.cuDeviceGetName(buf, ctypes.c_int(len(buf)), ctypes.c_int(device))),
        what=f"cuDeviceGetName(dev={device})",
    )
    return buf.value.decode("utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate compute-unit attributes across multiple FakeGPU architectures.")
    parser.add_argument(
        "--repo-root",
        help="Path to FakeGPU repo root (defaults to autodetect from this script location).",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]

    expected_presets = _expected_device_profiles()
    if not expected_presets:
        _die("no expected devices (empty spec?)")

    libcuda = _load_libcuda()
    libcuda.cuInit.argtypes = [ctypes.c_uint]
    libcuda.cuInit.restype = ctypes.c_int
    libcuda.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    libcuda.cuDeviceGetCount.restype = ctypes.c_int
    libcuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    libcuda.cuDeviceGet.restype = ctypes.c_int
    libcuda.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    libcuda.cuDeviceGetName.restype = ctypes.c_int
    libcuda.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    libcuda.cuDeviceGetAttribute.restype = ctypes.c_int

    _cu_check(int(libcuda.cuInit(0)), what="cuInit(0)")

    count = ctypes.c_int()
    _cu_check(int(libcuda.cuDeviceGetCount(ctypes.byref(count))), what="cuDeviceGetCount")
    got_count = int(count.value)
    if got_count != len(expected_presets):
        _die(f"device count mismatch: expected {len(expected_presets)} from spec, got {got_count}")

    failures = 0
    for idx, preset_id in enumerate(expected_presets):
        expected = _load_expected_device(repo_root=repo_root, preset_id=preset_id)
        got_name = _cu_device_get_name(libcuda, idx)
        if got_name != expected.name:
            print(
                f"[FAIL] dev{idx} name mismatch: expected={expected.name!r} got={got_name!r} (preset={preset_id})",
                file=sys.stderr,
            )
            failures += 1

        got_major = _cu_device_get_attribute(libcuda, idx, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        got_minor = _cu_device_get_attribute(libcuda, idx, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        if (got_major, got_minor) != (expected.cc_major, expected.cc_minor):
            print(
                f"[FAIL] dev{idx} CC mismatch: expected={expected.cc_major}.{expected.cc_minor} got={got_major}.{got_minor} (preset={preset_id})",
                file=sys.stderr,
            )
            failures += 1

        checks: list[tuple[str, int, int]] = [
            ("sm_count", CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, expected.sm_count),
            (
                "shared_mem_per_sm",
                CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
                expected.shared_mem_per_sm,
            ),
            (
                "regs_per_multiprocessor",
                CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
                expected.regs_per_multiprocessor,
            ),
            (
                "max_threads_per_multiprocessor",
                CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
                expected.max_threads_per_multiprocessor,
            ),
            (
                "max_blocks_per_multiprocessor",
                CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR,
                expected.max_blocks_per_multiprocessor,
            ),
        ]
        for label, attr, expected_value in checks:
            got_value = _cu_device_get_attribute(libcuda, idx, attr)
            if got_value != expected_value:
                print(
                    f"[FAIL] dev{idx} {label} mismatch: expected={expected_value} got={got_value} (preset={preset_id}, attr={attr})",
                    file=sys.stderr,
                )
                failures += 1

    if failures:
        print(f"\nFAILED: {failures} mismatches", file=sys.stderr)
        return 1

    print(f"OK: validated {len(expected_presets)} FakeGPU device(s) across profiles: {', '.join(expected_presets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

