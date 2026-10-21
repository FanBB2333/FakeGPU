#!/usr/bin/env python3
"""Kernel launches must reject invalid handles under every compatibility policy."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


CHILD = r"""
import ctypes
import json
import sys

class Dim3(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_uint),
        ("y", ctypes.c_uint),
        ("z", ctypes.c_uint),
    ]

cudart = ctypes.CDLL(sys.argv[1])
driver = ctypes.CDLL(sys.argv[2])
cudart.cudaLaunchKernel.argtypes = [
    ctypes.c_void_p,
    Dim3,
    Dim3,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_size_t,
    ctypes.c_void_p,
]
cudart.cudaLaunchKernel.restype = ctypes.c_int
cudart.cudaGetErrorName.argtypes = [ctypes.c_int]
cudart.cudaGetErrorName.restype = ctypes.c_char_p
driver.cuLaunchKernel.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_void_p),
]
driver.cuLaunchKernel.restype = ctypes.c_int
driver.cuGetErrorName.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_char_p),
]
driver.cuGetErrorName.restype = ctypes.c_int
cudart.cudaLaunch.argtypes = [ctypes.c_void_p]
cudart.cudaLaunch.restype = ctypes.c_int
cudart.cudaLaunchCooperativeKernel.argtypes = cudart.cudaLaunchKernel.argtypes
cudart.cudaLaunchCooperativeKernel.restype = ctypes.c_int
cudart.cudaGraphLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
cudart.cudaGraphLaunch.restype = ctypes.c_int
driver.cuGraphLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
driver.cuGraphLaunch.restype = ctypes.c_int

runtime_results = [
    cudart.cudaLaunchKernel(
        ctypes.c_void_p(1),
        Dim3(1, 1, 1),
        Dim3(1, 1, 1),
        None,
        0,
        None,
    )
    for _ in range(2)
]
driver_results = [
    driver.cuLaunchKernel(
        None,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        None,
        None,
        None,
    )
    for _ in range(2)
]
driver.cuStreamAddCallback.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_uint,
]
driver.cuStreamAddCallback.restype = ctypes.c_int
callback_results = [
    driver.cuStreamAddCallback(None, None, None, 0)
    for _ in range(2)
]
driver_error_name = ctypes.c_char_p()
driver_name_result = driver.cuGetErrorName(
    driver_results[-1],
    ctypes.byref(driver_error_name),
)
unsupported_launch_results = {
    "legacy": cudart.cudaLaunch(ctypes.c_void_p(1)),
    "cooperative": cudart.cudaLaunchCooperativeKernel(
        ctypes.c_void_p(1), Dim3(1, 1, 1), Dim3(1, 1, 1), None, 0, None
    ),
    "runtime_graph": cudart.cudaGraphLaunch(ctypes.c_void_p(1), None),
    "driver_graph": driver.cuGraphLaunch(ctypes.c_void_p(1), None),
}
print(json.dumps({
    "runtime_results": runtime_results,
    "runtime_error_name": cudart.cudaGetErrorName(
        runtime_results[-1]
    ).decode("utf-8"),
    "driver_results": driver_results,
    "driver_name_result": driver_name_result,
    "driver_error_name": driver_error_name.value.decode("utf-8"),
    "unsupported_launch_results": unsupported_launch_results,
    "callback_results": callback_results,
}))
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    args = parser.parse_args(argv)

    build_dir = args.build_dir.resolve()
    library = (
        build_dir / "libcudart.dylib"
        if sys.platform == "darwin"
        else build_dir / "libcudart.so.12"
    )
    driver_library = (
        build_dir / "libcuda.dylib"
        if sys.platform == "darwin"
        else build_dir / "libcuda.so.1"
    )
    if not library.is_file():
        parser.error(f"FakeGPU CUDA Runtime library not found: {library}")
    if not driver_library.is_file():
        parser.error(f"FakeGPU CUDA Driver library not found: {driver_library}")

    for policy in ("allow", "warn", "error"):
        env = dict(os.environ)
        env["FAKEGPU_MODE"] = "simulate"
        env["FAKEGPU_UNSUPPORTED_API"] = policy
        env["FAKEGPU_TERMINAL_REPORT"] = "0"
        env["FAKEGPU_REPORT_PATH"] = os.devnull
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                CHILD,
                str(library),
                str(driver_library),
            ],
            cwd=str(build_dir.parent),
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"policy {policy} failed with {completed.returncode}: "
                f"{completed.stderr}"
            )
        payload = json.loads(completed.stdout)
        callback_result = 801 if policy == "error" else 0
        if payload != {
            "runtime_results": [98, 98],
            "runtime_error_name": "cudaErrorInvalidDeviceFunction",
            "driver_results": [400, 400],
            "driver_name_result": 0,
            "driver_error_name": "CUDA_ERROR_INVALID_HANDLE",
            "unsupported_launch_results": {
                "legacy": 801,
                "cooperative": 801,
                "runtime_graph": 801,
                "driver_graph": 801,
            },
            "callback_results": [callback_result, callback_result],
        }:
            raise AssertionError(
                f"policy {policy} accepted an invalid kernel handle: {payload!r}"
            )
        runtime_warning_count = completed.stderr.count(
            "[FakeGPU] warning: cudaLaunchKernel"
        )
        driver_warning_count = completed.stderr.count(
            "[FakeGPU] warning: cuLaunchKernel"
        )
        callback_warning_count = completed.stderr.count(
            "[FakeGPU] warning: cuStreamAddCallback"
        )
        expected_callback_warnings = 1 if policy == "warn" else 0
        if (
            runtime_warning_count
            or driver_warning_count
            or callback_warning_count != expected_callback_warnings
        ):
            raise AssertionError(
                f"policy {policy} emitted runtime/driver/callback warning "
                f"counts {runtime_warning_count}/{driver_warning_count}/"
                f"{callback_warning_count}, expected "
                f"0/0/{expected_callback_warnings}: {completed.stderr!r}"
            )

    print("invalid kernel handles rejected under allow, warn and error policies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
