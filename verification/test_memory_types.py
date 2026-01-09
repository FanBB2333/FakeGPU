#!/usr/bin/env python3
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path


CUDA_SUCCESS = 0

CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2
CU_POINTER_ATTRIBUTE_IS_MANAGED = 8
CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9

CU_MEMORYTYPE_HOST = 1
CU_MEMORYTYPE_DEVICE = 2
CU_MEMORYTYPE_UNIFIED = 4


def _die(msg: str) -> None:
    print(f"[test_memory_types] ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


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


def _cu_check(result: int, *, what: str) -> None:
    if int(result) != CUDA_SUCCESS:
        _die(f"{what} failed: CUresult={int(result)}")


def _cu_get_ptr_attr(libcuda: ctypes.CDLL, ptr: int, attr: int, ctype: object) -> int:
    out = ctype()
    _cu_check(
        int(libcuda.cuPointerGetAttribute(ctypes.byref(out), ctypes.c_int(attr), ctypes.c_ulonglong(ptr))),
        what=f"cuPointerGetAttribute(ptr=0x{ptr:x}, attr={attr})",
    )
    return int(out.value)


def main() -> int:
    libcuda = _load_libcuda()

    libcuda.cuInit.argtypes = [ctypes.c_uint]
    libcuda.cuInit.restype = ctypes.c_int
    libcuda.cuMemAlloc.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_size_t]
    libcuda.cuMemAlloc.restype = ctypes.c_int
    libcuda.cuMemAllocManaged.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_size_t, ctypes.c_uint]
    libcuda.cuMemAllocManaged.restype = ctypes.c_int
    libcuda.cuMemAllocHost.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    libcuda.cuMemAllocHost.restype = ctypes.c_int
    libcuda.cuMemFree.argtypes = [ctypes.c_ulonglong]
    libcuda.cuMemFree.restype = ctypes.c_int
    libcuda.cuMemFreeHost.argtypes = [ctypes.c_void_p]
    libcuda.cuMemFreeHost.restype = ctypes.c_int
    libcuda.cuPointerGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ulonglong]
    libcuda.cuPointerGetAttribute.restype = ctypes.c_int

    _cu_check(int(libcuda.cuInit(0)), what="cuInit(0)")

    dptr = ctypes.c_ulonglong()
    _cu_check(int(libcuda.cuMemAlloc(ctypes.byref(dptr), ctypes.c_size_t(4096))), what="cuMemAlloc")
    mptr = ctypes.c_ulonglong()
    _cu_check(int(libcuda.cuMemAllocManaged(ctypes.byref(mptr), ctypes.c_size_t(4096), ctypes.c_uint(0))), what="cuMemAllocManaged")
    hptr = ctypes.c_void_p()
    _cu_check(int(libcuda.cuMemAllocHost(ctypes.byref(hptr), ctypes.c_size_t(4096))), what="cuMemAllocHost")

    try:
        device_mem_type = _cu_get_ptr_attr(libcuda, int(dptr.value), CU_POINTER_ATTRIBUTE_MEMORY_TYPE, ctypes.c_int)
        device_is_managed = _cu_get_ptr_attr(libcuda, int(dptr.value), CU_POINTER_ATTRIBUTE_IS_MANAGED, ctypes.c_uint)
        if device_mem_type != CU_MEMORYTYPE_DEVICE or device_is_managed != 0:
            _die(f"device allocation attributes mismatch: mem_type={device_mem_type} is_managed={device_is_managed}")

        managed_mem_type = _cu_get_ptr_attr(libcuda, int(mptr.value), CU_POINTER_ATTRIBUTE_MEMORY_TYPE, ctypes.c_int)
        managed_is_managed = _cu_get_ptr_attr(libcuda, int(mptr.value), CU_POINTER_ATTRIBUTE_IS_MANAGED, ctypes.c_uint)
        if managed_mem_type != CU_MEMORYTYPE_UNIFIED or managed_is_managed != 1:
            _die(f"managed allocation attributes mismatch: mem_type={managed_mem_type} is_managed={managed_is_managed}")

        host_mem_type = _cu_get_ptr_attr(libcuda, int(hptr.value), CU_POINTER_ATTRIBUTE_MEMORY_TYPE, ctypes.c_int)
        host_is_managed = _cu_get_ptr_attr(libcuda, int(hptr.value), CU_POINTER_ATTRIBUTE_IS_MANAGED, ctypes.c_uint)
        if host_mem_type != CU_MEMORYTYPE_HOST or host_is_managed != 0:
            _die(f"host allocation attributes mismatch: mem_type={host_mem_type} is_managed={host_is_managed}")

        for label, ptr in (("device", int(dptr.value)), ("managed", int(mptr.value)), ("host", int(hptr.value))):
            ordinal = _cu_get_ptr_attr(libcuda, ptr, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, ctypes.c_int)
            if ordinal != 0:
                _die(f"{label} allocation device ordinal mismatch: expected 0 got {ordinal}")

        print("OK: pointer attribute memory types (device/managed/host) validated")
        return 0
    finally:
        libcuda.cuMemFree(ctypes.c_ulonglong(dptr.value))
        libcuda.cuMemFree(ctypes.c_ulonglong(mptr.value))
        libcuda.cuMemFreeHost(hptr)


if __name__ == "__main__":
    raise SystemExit(main())

