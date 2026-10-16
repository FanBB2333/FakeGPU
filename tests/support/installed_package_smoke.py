#!/usr/bin/env python3
"""Check an installed wheel from outside the checkout with isolated Python.

Run with ``python -I /path/to/installed_package_smoke.py`` after installing
FakeGPU and PyTorch. Native and FakeCUDA checks use separate interpreters.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path


def _installed_paths(expected_profile_count: int | None) -> Path:
    import fakegpu
    from fakegpu.profile_catalog import load_profiles, profile_directory

    package_dir = Path(fakegpu.__file__).resolve().parent
    install_dirs = {
        Path(sysconfig.get_path(kind)).resolve() for kind in ("purelib", "platlib")
    }
    assert any(package_dir.is_relative_to(path) for path in install_dirs), (
        f"FakeGPU must come from installed site-packages: {package_dir}"
    )
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "CMakeLists.txt").is_file():
        assert not Path.cwd().is_relative_to(source_root), (
            "Run the installed-package smoke check outside the source checkout"
        )
    native_dir = fakegpu.library_dir()
    assert native_dir == package_dir / "_native", native_dir
    profile_dir = profile_directory()
    assert profile_dir == package_dir / "_profiles", profile_dir
    profiles = load_profiles()
    assert profiles, "Installed profile catalog is empty"
    if expected_profile_count is not None:
        assert len(profiles) == expected_profile_count, (
            f"Expected {expected_profile_count} packaged profiles, got {len(profiles)}"
        )
    for profile in profiles.values():
        assert (profile_dir / profile.profile_path).is_file(), profile.profile_path
    coordinator = native_dir / "fakegpu-coordinator"
    assert coordinator.is_file() and os.access(coordinator, os.X_OK), coordinator
    result = subprocess.run(
        [str(coordinator), "--help"],
        text=True,
        capture_output=True,
        check=True,
        timeout=15,
    )
    assert "Usage: fakegpu-coordinator" in result.stdout + result.stderr
    print(f"Installed package: {len(profiles)} profiles, native libraries, coordinator")
    return native_dir


def _native_smoke() -> None:
    import fakegpu

    result = fakegpu.init(runtime="native", profile="l4", device_count=2)
    assert result.runtime == "native"
    assert len(result.handles) == 5
    cuda_name = "libcuda.dylib" if sys.platform == "darwin" else "libcuda.so.1"
    cuda = result.handles[cuda_name]
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuInit.restype = ctypes.c_int
    cuda.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cuda.cuDeviceGetCount.restype = ctypes.c_int
    assert cuda.cuInit(0) == 0
    count = ctypes.c_int()
    assert cuda.cuDeviceGetCount(ctypes.byref(count)) == 0
    assert count.value == 2

    pointer = ctypes.c_uint64()
    cuda.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    cuda.cuMemAlloc_v2.restype = ctypes.c_int
    cuda.cuMemFree_v2.argtypes = [ctypes.c_uint64]
    cuda.cuMemFree_v2.restype = ctypes.c_int
    cuda.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
    cuda.cuMemcpyHtoD_v2.restype = ctypes.c_int
    cuda.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
    cuda.cuMemcpyDtoH_v2.restype = ctypes.c_int
    source = ctypes.create_string_buffer(b"installed FakeGPU")
    destination = ctypes.create_string_buffer(len(source))
    assert cuda.cuMemAlloc_v2(ctypes.byref(pointer), len(source)) == 0
    try:
        assert cuda.cuMemcpyHtoD_v2(pointer, source, len(source)) == 0
        assert cuda.cuMemcpyDtoH_v2(destination, pointer, len(source)) == 0
        assert destination.raw == source.raw
    finally:
        assert cuda.cuMemFree_v2(pointer) == 0
    print("Installed native runtime: synthetic device count and memory round-trip passed")


def _fakecuda_smoke() -> None:
    import fakegpu

    result = fakegpu.init(runtime="fakecuda", profile="l4", device_count=2)
    assert result.runtime == "fakecuda"

    import torch

    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 2
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 2).to("cuda:0")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    before = model.weight.detach().cpu().clone()
    inputs = torch.ones(2, 3, device="cuda:0")
    output = model(inputs)
    assert output.is_cuda and output.device.index == 0
    loss = output.square().mean()
    loss.backward()
    assert model.weight.grad is not None
    assert torch.isfinite(model.weight.grad.cpu()).all()
    optimizer.step()
    assert not torch.equal(before, model.weight.detach().cpu())
    assert inputs.cpu().device.type == "cpu"
    assert torch.isfinite(loss.cpu())
    print("Installed FakeCUDA runtime: CPU-backed forward, backward, optimizer passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-profile-count", type=int)
    parser.add_argument("--phase", choices=("native", "fakecuda"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("PYTHONPATH", "FAKEGPU_BUILD_DIR", "FAKEGPU_LIB_DIR", "FAKEGPU_PROFILE_DIR"):
        assert name not in os.environ, f"Unset {name} before checking the installed package"
    assert sys.flags.isolated, "Run this script with python -I"
    _installed_paths(args.expected_profile_count)
    if args.phase:
        (_native_smoke if args.phase == "native" else _fakecuda_smoke)()
        return

    child_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("FAKEGPU_", "TORCH_FAKEGPU_", "DYLD_", "LD_"))
    }
    child_env["FAKEGPU_TERMINAL_REPORT"] = "0"
    script = str(Path(__file__).resolve())
    with tempfile.TemporaryDirectory(prefix="fakegpu-installed-") as temporary:
        for phase in ("native", "fakecuda"):
            child_env["FAKEGPU_REPORT_PATH"] = str(Path(temporary) / f"{phase}.json")
            child_env["FAKEGPU_SMI_STATE_DIR"] = str(Path(temporary) / phase)
            command = [sys.executable, "-I", script, "--phase", phase]
            if args.expected_profile_count is not None:
                command.extend(("--expected-profile-count", str(args.expected_profile_count)))
            subprocess.run(command, env=child_env, cwd=temporary, check=True, timeout=90)


if __name__ == "__main__":
    main()
