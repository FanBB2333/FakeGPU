"""Profile-driven CUDA device attributes, memory totals and MIG capability in the native libraries."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from fakegpu.profile_catalog import load_profiles

ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = Path(os.environ.get("FAKEGPU_BUILD_DIR", ROOT / "build"))
SUFFIX = ".dylib" if sys.platform == "darwin" else ".so.1"
CUDA_LIBRARY = BUILD_DIR / f"libcuda{SUFFIX}"
NVML_LIBRARY = BUILD_DIR / f"libnvidia-ml{SUFFIX}"

pytestmark = pytest.mark.skipif(
    not (CUDA_LIBRARY.exists() and NVML_LIBRARY.exists()), reason="native libraries are not built"
)

# CUdevice_attribute values from cuda.h.
ATTRIBUTES = {
    "ecc_enabled": 32,
    "async_engine_count": 40,
    "single_to_double_precision_perf_ratio": 87,
    "pageable_memory_access": 88,
    "can_flush_remote_writes": 98,
    "max_persisting_l2_cache_bytes": 108,
    "max_access_policy_window_bytes": 109,
    "reserved_shared_mem_per_block": 111,
}


def _run(tmp_path: Path, body: str, **environment: str):
    code = textwrap.dedent(
        """
        import ctypes, json
        cuda = ctypes.CDLL({cuda!r})
        nvml = ctypes.CDLL({nvml!r})
        cuda.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        cuda.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        cuda.cuDeviceTotalMem_v2.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
        nvml.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        nvml.nvmlDeviceGetMigMode.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint)]
        nvml.nvmlDeviceGetMaxMigDeviceCount.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
        assert cuda.cuInit(0) == 0
        assert nvml.nvmlInit_v2() == 0

        def attribute(dev, attr):
            value = ctypes.c_int(-1)
            assert cuda.cuDeviceGetAttribute(ctypes.byref(value), attr, dev) == 0
            return value.value

        def handle(index):
            h = ctypes.c_void_p()
            assert nvml.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(h)) == 0
            return h

        result = None
        """
    ).format(cuda=str(CUDA_LIBRARY), nvml=str(NVML_LIBRARY)) + textwrap.dedent(body) + (
        "\nnvml.nvmlShutdown()\nprint(json.dumps(result))\n"
    )
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("FAKEGPU_", "CUDA_VISIBLE_DEVICES"))
    }
    env.update(
        PYTHONPATH=str(ROOT), FAKEGPU_HOME=str(tmp_path), FAKEGPU_MACHINE="off",
        FAKEGPU_TERMINAL_REPORT="0", FAKEGPU_REPORT_PATH=str(tmp_path / "report.json"),
        **environment,
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env, text=True, capture_output=True, timeout=120
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_every_compiled_profile_parses_with_its_reported_memory(tmp_path: Path) -> None:
    # A profile the native parser rejects falls back to a100, so the names and
    # totals of every catalog profile prove each compiled YAML parsed.
    catalog = load_profiles()
    result = _run(
        tmp_path,
        f"""
        result = []
        for dev in range({len(catalog)}):
            name = ctypes.create_string_buffer(256)
            total = ctypes.c_size_t()
            assert cuda.cuDeviceGetName(name, 256, dev) == 0
            assert cuda.cuDeviceTotalMem_v2(ctypes.byref(total), dev) == 0
            result.append([name.value.decode(), total.value])
        """,
        FAKEGPU_PROFILES=",".join(catalog),
    )
    assert result == [[profile.name, profile.device_memory_bytes] for profile in catalog.values()]
    # NVIDIA documents: nvidia-smi 81559MiB for H100 80GB HBM3 (driver
    # 580.95.05), 23034MiB for L4 with ECC on; deviceQuery 42405855232 bytes for
    # A100-PCIE-40GB (CUDA 13.0).
    assert catalog["h100"].device_memory_bytes == 81559 * 2**20
    assert catalog["l4"].device_memory_bytes == 23034 * 2**20
    assert catalog["a100-pcie-40gb"].device_memory_bytes == 42405855232
    assert catalog["a100"].device_memory_bytes == catalog["a100"].memory_bytes


# Expected values: the FP32:FP64 ratio comes from the CUDA Programming Guide per
# compute capability; per-SKU copy-engine counts and ECC states from NVIDIA's
# deviceQuery outputs (A100-PCIE-40GB: 3 engines, ECC on; Jetson AGX Thor: 1
# engine, ECC off). The rest are the documented class defaults.
EXPECTED = {
    # id: ecc, engines, fp32:fp64, pageable, flush, persisting L2 share, window
    "a100": (1, 2, 2, 1, 0, 0.75, 134213632),
    "a100-pcie-40gb": (1, 3, 2, 1, 0, 0.75, 134213632),
    "h100": (1, 2, 2, 1, 0, 0.75, 134213632),
    "gh200-96gb": (1, 2, 2, 1, 0, 0.75, 134213632),
    "l4": (1, 2, 64, 1, 0, 0.75, 134213632),
    "t4": (1, 2, 32, 1, 0, 0, 0),
    "v100": (1, 2, 2, 0, 0, 0, 0),
    "p100": (1, 2, 2, 0, 0, 0, 0),
    "gtx1080": (0, 2, 32, 0, 0, 0, 0),
    "rtx3090": (0, 2, 64, 1, 0, 0.75, 134213632),
    "rtx4090": (0, 2, 64, 1, 0, 0.75, 134213632),
    "rtx5090": (0, 2, 64, 1, 0, 0.75, 134213632),
    "rtx-a6000": (0, 2, 64, 1, 0, 0.75, 134213632),
    "b200": (1, 2, 2, 1, 0, 0.75, 134213632),
    "b300": (1, 2, 64, 1, 0, 0.75, 134213632),
    "jetson-t5000": (0, 1, 64, 0, 0, 0.75, 134213632),
}


def test_device_attributes_follow_the_profile(tmp_path: Path) -> None:
    catalog = load_profiles()
    result = _run(
        tmp_path,
        f"""
        attributes = {list(ATTRIBUTES.values())!r}
        result = [[attribute(dev, attr) for attr in attributes] for dev in range({len(EXPECTED)})]
        """,
        FAKEGPU_PROFILES=",".join(EXPECTED),
    )
    for (profile_id, expected), values in zip(EXPECTED.items(), result):
        ecc, engines, ratio, pageable, flush, persisting_share, window = expected
        persisting = int(catalog[profile_id].l2_cache_bytes * 3 // 4) if persisting_share else 0
        # CUDA Programming Guide, compute capability 8.x and 9.0: 1 KB of
        # shared memory per block is reserved for system use; none before 8.0.
        reserved = 1024 if catalog[profile_id].compute_major >= 8 else 0
        assert values == [ecc, engines, ratio, pageable, flush, persisting, window, reserved], profile_id
        assert persisting < catalog[profile_id].l2_cache_bytes or persisting == 0

