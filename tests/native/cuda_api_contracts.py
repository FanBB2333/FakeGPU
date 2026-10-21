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
    # CUDA 12 cuda_runtime_api.h: cudaGetDriverEntryPoint(symbol, funcPtr, flags,
    # driverStatus) writes the same status as the versioned lookup.
    runtime.cudaGetDriverEntryPoint.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(c_void_p),
        ctypes.c_ulonglong,
        ctypes.POINTER(ctypes.c_int),
    ]
    for symbol, expected_status in ((b"cuInit", 0), (b"cuFakeGpuMissingSymbol", 1)):
        function = c_void_p()
        status = ctypes.c_int(-1)
        result = runtime.cudaGetDriverEntryPoint(symbol, ctypes.byref(function), 0, ctypes.byref(status))
        _check(
            result == CUDA_SUCCESS and status.value == expected_status,
            f"cudaGetDriverEntryPoint({symbol!r}) returned {result} with status {status.value}",
        )


def check_get_proc_address_v2_rejects_null_output(driver: ctypes.CDLL) -> None:
    driver.cuGetProcAddress_v2.argtypes = [
        ctypes.c_char_p,
        c_void_p,
        ctypes.c_int,
        ctypes.c_ulonglong,
        ctypes.POINTER(ctypes.c_int),
    ]
    status = ctypes.c_int(-1)
    result = driver.cuGetProcAddress_v2(b"cuInit", None, 12000, 0, ctypes.byref(status))
    _check(
        result == CUDA_ERROR_INVALID_VALUE and status.value == 1,
        f"cuGetProcAddress_v2 with a NULL pfn returned {result} with status {status.value}",
    )


def check_context_push_switches_runtime_device(runtime: ctypes.CDLL, driver: ctypes.CDLL) -> None:
    # libcudart carries its own copy of the Driver API. It binds to that copy on
    # macOS, but to the first global definition (libcuda) on Linux.
    pusher = runtime if sys.platform == "darwin" else driver
    pusher.cuCtxPushCurrent.argtypes = [c_void_p]
    pusher.cuCtxPopCurrent.argtypes = [ctypes.POINTER(c_void_p)]
    _check(runtime.cudaSetDevice(0) == 0, "cudaSetDevice(0) failed")
    # Simulated contexts encode device ordinal + 1, so context 2 belongs to device 1.
    _check(pusher.cuCtxPushCurrent(c_void_p(2)) == CUDA_SUCCESS, "cuCtxPushCurrent failed")
    device = ctypes.c_int(-1)
    runtime.cudaGetDevice(ctypes.byref(device))
    _check(device.value == 1, f"cudaGetDevice after cuCtxPushCurrent returned {device.value}")
    # Peer access from the pushed device 1 to device 0 is allowed; 0 -> 0 is not.
    result = runtime.cudaDeviceEnablePeerAccess(0, 0)
    _check(result == 0, f"cudaDeviceEnablePeerAccess(0) after cuCtxPushCurrent returned {result}")
    runtime.cudaDeviceDisablePeerAccess(0)
    popped = c_void_p()
    pusher.cuCtxPopCurrent(ctypes.byref(popped))
    runtime.cudaSetDevice(0)


def check_mem_pool_int_attribute_size(runtime: ctypes.CDLL) -> None:
    class IntWithCanary(ctypes.Structure):
        _fields_ = [("value", ctypes.c_int), ("canary", ctypes.c_int)]

    runtime.cudaMemPoolGetAttribute.argtypes = [c_void_p, ctypes.c_int, c_void_p]
    pool = c_void_p()
    _check(runtime.cudaDeviceGetDefaultMemPool(ctypes.byref(pool), 0) == 0, "cudaDeviceGetDefaultMemPool failed")
    # cudaMemPoolReuseFollowEventDependencies (1) is an int attribute.
    pair = IntWithCanary(-1, 0x7777)
    result = runtime.cudaMemPoolGetAttribute(pool, 1, ctypes.byref(pair))
    _check(
        result == 0 and pair.value == 0 and pair.canary == 0x7777,
        f"cudaMemPoolGetAttribute(int attribute) returned {result}, value {pair.value}, canary {pair.canary:#x}",
    )


class PointerAttributes(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("device", ctypes.c_int),
        ("devicePointer", c_void_p),
        ("hostPointer", c_void_p),
    ]


def check_pointer_attribute_memory_types(runtime: ctypes.CDLL) -> None:
    # CUDA 12.4 driver_types.h: cudaMemoryTypeUnregistered = 0, Host = 1,
    # Device = 2, Managed = 3. cuda_runtime_api.h: an alias is NULL when that
    # side cannot access the memory directly.
    runtime.cudaPointerGetAttributes.argtypes = [ctypes.POINTER(PointerAttributes), c_void_p]
    runtime.cudaMalloc.argtypes = [ctypes.POINTER(c_void_p), ctypes.c_size_t]
    runtime.cudaFree.argtypes = [c_void_p]
    host = ctypes.create_string_buffer(64)
    attributes = PointerAttributes()
    result = runtime.cudaPointerGetAttributes(ctypes.byref(attributes), ctypes.addressof(host))
    _check(
        result == 0 and attributes.type == 0 and attributes.devicePointer is None
        and attributes.hostPointer == ctypes.addressof(host),
        f"unregistered host memory: result {result}, type {attributes.type}, "
        f"device alias {attributes.devicePointer}",
    )
    device = c_void_p()
    _check(runtime.cudaMalloc(ctypes.byref(device), 4096) == 0, "cudaMalloc failed")
    result = runtime.cudaPointerGetAttributes(ctypes.byref(attributes), device)
    _check(
        result == 0 and attributes.type == 2 and attributes.devicePointer == device.value
        and attributes.hostPointer is None,
        f"device memory: result {result}, type {attributes.type}, host alias {attributes.hostPointer}",
    )
    runtime.cudaFree(device)


def check_runtime_error_text(runtime: ctypes.CDLL) -> None:
    # Values returned by libcudart.so.12 (CUDA 12.4); cuda_runtime_api.h
    # documents "unrecognized error code" for unknown codes.
    runtime.cudaGetErrorName.restype = ctypes.c_char_p
    runtime.cudaGetErrorName.argtypes = [ctypes.c_int]
    runtime.cudaGetErrorString.restype = ctypes.c_char_p
    runtime.cudaGetErrorString.argtypes = [ctypes.c_int]
    expected = {
        21: (b"cudaErrorInvalidMemcpyDirection", b"invalid copy direction for memcpy"),
        100: (b"cudaErrorNoDevice", b"no CUDA-capable device is detected"),
        600: (b"cudaErrorNotReady", b"device not ready"),
        719: (b"cudaErrorLaunchFailure", b"unspecified launch failure"),
        12345: (b"unrecognized error code", b"unrecognized error code"),
    }
    for code, (name, text) in expected.items():
        actual = (runtime.cudaGetErrorName(code), runtime.cudaGetErrorString(code))
        _check(actual == (name, text), f"cudaGetErrorName/String({code}) returned {actual}")


def check_driver_error_text(driver: ctypes.CDLL) -> None:
    # Values returned by libcuda.so (r550, CUDA 12.4). cuda.h: an unrecognized
    # code returns CUDA_ERROR_INVALID_VALUE and sets the string to NULL.
    for function in (driver.cuGetErrorName, driver.cuGetErrorString):
        function.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    expected = {
        2: (b"CUDA_ERROR_OUT_OF_MEMORY", b"out of memory"),
        100: (b"CUDA_ERROR_NO_DEVICE", b"no CUDA-capable device is detected"),
        719: (b"CUDA_ERROR_LAUNCH_FAILED", b"unspecified launch failure"),
        12345: (None, None),
    }
    for code, (name, text) in expected.items():
        status = CUDA_SUCCESS if name else CUDA_ERROR_INVALID_VALUE
        name_out, text_out = ctypes.c_char_p(b"x"), ctypes.c_char_p(b"x")
        actual = (
            driver.cuGetErrorName(code, ctypes.byref(name_out)),
            name_out.value,
            driver.cuGetErrorString(code, ctypes.byref(text_out)),
            text_out.value,
        )
        _check(actual == (status, name, status, text), f"cuGetErrorName/String({code}) returned {actual}")


def check_last_error_survives_successful_calls(runtime: ctypes.CDLL) -> None:
    # CUDA C++ Programming Guide 12.4, 3.2.12: the per-thread error variable is
    # overwritten only when an error occurs; cudaGetLastError resets it.
    runtime.cudaGetLastError()
    failed = runtime.cudaSetDevice(9999)
    synchronized = runtime.cudaDeviceSynchronize()
    peeked = runtime.cudaPeekAtLastError()
    taken = runtime.cudaGetLastError()
    cleared = runtime.cudaGetLastError()
    actual = (failed, synchronized, peeked, taken, cleared)
    _check(actual == (101, 0, 101, 101, 0), f"last-error sequence returned {actual}")


def check_func_attributes_size(runtime: ctypes.CDLL) -> None:
    # CUDA 12.4 driver_types.h: sizeof(struct cudaFuncAttributes) == 144 on LP64.
    runtime.cudaFuncGetAttributes.argtypes = [c_void_p, c_void_p]
    buffer = (ctypes.c_ubyte * 152)(*([0xA5] * 152))
    result = runtime.cudaFuncGetAttributes(buffer, c_void_p(1))
    _check(
        result == 0 and all(byte == 0 for byte in buffer[:144]) and all(byte == 0xA5 for byte in buffer[144:]),
        f"cudaFuncGetAttributes returned {result}; struct bytes {bytes(buffer).hex()}",
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
    check_get_proc_address_v2_rejects_null_output(driver)
    check_context_push_switches_runtime_device(runtime, driver)
    check_mem_pool_int_attribute_size(runtime)
    check_pointer_attribute_memory_types(runtime)
    check_runtime_error_text(runtime)
    check_driver_error_text(driver)
    check_last_error_survives_successful_calls(runtime)
    check_func_attributes_size(runtime)
    print("CUDA API contract validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
