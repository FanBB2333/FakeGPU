#!/usr/bin/env python3
"""Regression checks for FakeGPU cudart device properties."""

from __future__ import annotations

import ctypes
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fakegpu


class _CudaDevicePropPrefix(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 256),
        ("uuid", ctypes.c_char * 16),
        ("luid", ctypes.c_char * 8),
        ("luidDeviceNodeMask", ctypes.c_uint),
        ("totalGlobalMem", ctypes.c_size_t),
        ("sharedMemPerBlock", ctypes.c_size_t),
        ("regsPerBlock", ctypes.c_int),
        ("warpSize", ctypes.c_int),
        ("memPitch", ctypes.c_size_t),
        ("maxThreadsPerBlock", ctypes.c_int),
        ("maxThreadsDim", ctypes.c_int * 3),
        ("maxGridSize", ctypes.c_int * 3),
    ]


def main() -> None:
    result = fakegpu.init(runtime="native", force=True, update_env=True)
    lib_name = "libcudart.dylib" if sys.platform == "darwin" else "libcudart.so.12"
    cudart = result.handles[lib_name]

    cudart.cudaGetDeviceProperties.argtypes = [ctypes.c_void_p, ctypes.c_int]
    cudart.cudaGetDeviceProperties.restype = ctypes.c_int

    raw = ctypes.create_string_buffer(4096)
    status = cudart.cudaGetDeviceProperties(ctypes.byref(raw), 0)
    assert status == 0, f"cudaGetDeviceProperties failed: {status}"

    props = _CudaDevicePropPrefix.from_buffer(raw)
    assert props.maxThreadsPerBlock > 0
    assert all(dim > 0 for dim in props.maxThreadsDim), tuple(props.maxThreadsDim)
    assert all(dim > 0 for dim in props.maxGridSize), tuple(props.maxGridSize)

    print("cudart device properties regression passed")


if __name__ == "__main__":
    main()
