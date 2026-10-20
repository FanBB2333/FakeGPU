#!/usr/bin/env python3
"""Check simulated CUDA Driver and Runtime API contracts without a CUDA installation."""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
from pathlib import Path

CUDA_SUCCESS = 0
CUDA_ERROR_INVALID_VALUE = 1

c_void_p = ctypes.c_void_p


def _check(condition: bool, message: str) -> None:
    if not condition:
        print(f"[cuda_api_contracts] ERROR: {message}", file=sys.stderr)
        raise SystemExit(2)


def check_failed_event_record_keeps_stream_idle(driver: ctypes.CDLL) -> None:
    driver.cuStreamCreate.argtypes = [ctypes.POINTER(c_void_p), ctypes.c_uint]
    driver.cuStreamQuery.argtypes = [c_void_p]
    driver.cuEventRecord.argtypes = [c_void_p, c_void_p]
    driver.cuStreamDestroy.argtypes = [c_void_p]
    stream = c_void_p()
    _check(driver.cuStreamCreate(ctypes.byref(stream), 0) == CUDA_SUCCESS, "cuStreamCreate failed")
    for event in (c_void_p(0xDEAD), c_void_p()):
        result = driver.cuEventRecord(event, stream)
        _check(result == CUDA_ERROR_INVALID_VALUE, f"cuEventRecord({event.value}) returned {result}")
        result = driver.cuStreamQuery(stream)
        _check(result == CUDA_SUCCESS, f"stream not idle after a failed cuEventRecord: {result}")
    driver.cuStreamDestroy(stream)


def check_driver_entry_point_status(runtime: ctypes.CDLL) -> None:
    runtime.cudaGetDriverEntryPointByVersion.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(c_void_p),
        ctypes.c_uint,
        ctypes.c_ulonglong,
        ctypes.POINTER(ctypes.c_int),
    ]
    # cudaDriverEntryPointQueryResult: 0 = success, 1 = symbol not found.
    for symbol, expected_status in ((b"cuInit", 0), (b"cuFakeGpuMissingSymbol", 1)):
        function = c_void_p()
        status = ctypes.c_int(-1)
        result = runtime.cudaGetDriverEntryPointByVersion(
            symbol, ctypes.byref(function), 12000, 0, ctypes.byref(status)
        )
        _check(
            result == CUDA_SUCCESS and status.value == expected_status,
            f"cudaGetDriverEntryPointByVersion({symbol!r}) returned {result} with status {status.value}",
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    args = parser.parse_args(argv)

    build_dir = args.build_dir.resolve()
    darwin = sys.platform == "darwin"
    driver_path = build_dir / ("libcuda.dylib" if darwin else "libcuda.so.1")
    runtime_path = build_dir / ("libcudart.dylib" if darwin else "libcudart.so.12")
    for path in (driver_path, runtime_path):
        if not path.is_file():
            parser.error(f"FakeGPU library not found: {path}")

    os.environ["FAKEGPU_TERMINAL_REPORT"] = "0"
    os.environ["FAKEGPU_REPORT_PATH"] = os.devnull
    driver = ctypes.CDLL(str(driver_path), mode=ctypes.RTLD_GLOBAL)
    runtime = ctypes.CDLL(str(runtime_path), mode=ctypes.RTLD_GLOBAL)
    _check(driver.cuInit(0) == CUDA_SUCCESS, "cuInit failed")

    check_failed_event_record_keeps_stream_idle(driver)
    check_driver_entry_point_status(runtime)
    print("CUDA API contract validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
