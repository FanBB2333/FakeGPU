#!/usr/bin/env python3
"""Check simulated CUDA Driver and Runtime API contracts without a CUDA installation."""

from __future__ import annotations

import argparse
import ctypes
import json
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
    # side cannot access the memory directly. Measured on an Ampere GPU with
    # pageable memory access: unregistered memory reports device -1 and the
    # address itself as its device alias.
    runtime.cudaPointerGetAttributes.argtypes = [ctypes.POINTER(PointerAttributes), c_void_p]
    runtime.cudaMalloc.argtypes = [ctypes.POINTER(c_void_p), ctypes.c_size_t]
    runtime.cudaFree.argtypes = [c_void_p]
    host = ctypes.create_string_buffer(64)
    attributes = PointerAttributes()
    result = runtime.cudaPointerGetAttributes(ctypes.byref(attributes), ctypes.addressof(host))
    _check(
        result == 0 and attributes.type == 0 and attributes.device == -1
        and attributes.devicePointer == ctypes.addressof(host)
        and attributes.hostPointer == ctypes.addressof(host),
        f"unregistered host memory: result {result}, type {attributes.type}, "
        f"device {attributes.device}, device alias {attributes.devicePointer}",
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


def check_event_elapsed_time_errors(runtime: ctypes.CDLL) -> None:
    # cuda_runtime_api.h (cudaEventElapsedTime): unrecorded events and events
    # created with cudaEventDisableTiming (0x02) return
    # cudaErrorInvalidResourceHandle (400).
    runtime.cudaEventCreateWithFlags.argtypes = [ctypes.POINTER(c_void_p), ctypes.c_uint]
    runtime.cudaEventRecord.argtypes = [c_void_p, c_void_p]
    runtime.cudaEventSynchronize.argtypes = [c_void_p]
    runtime.cudaEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), c_void_p, c_void_p]
    runtime.cudaEventDestroy.argtypes = [c_void_p]

    def elapsed(flags: int, record: bool) -> int:
        events = [c_void_p(), c_void_p()]
        for event in events:
            _check(runtime.cudaEventCreateWithFlags(ctypes.byref(event), flags) == 0, "cudaEventCreate failed")
            if record:
                runtime.cudaEventRecord(event, None)
        if record:
            runtime.cudaEventSynchronize(events[1])
        milliseconds = ctypes.c_float(-1)
        result = runtime.cudaEventElapsedTime(ctypes.byref(milliseconds), events[0], events[1])
        for event in events:
            runtime.cudaEventDestroy(event)
        return result

    actual = (elapsed(0, False), elapsed(0x02, True), elapsed(0, True))
    runtime.cudaGetLastError()
    _check(actual == (400, 400, 0), f"cudaEventElapsedTime returned {actual}")


def check_grid_dimension_limits(driver: ctypes.CDLL) -> None:
    # CUDA Programming Guide, compute capabilities table: maximum x-dimension
    # of a grid is 2^31-1, maximum y- or z-dimension is 65535.
    driver.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    limits = []
    for attribute in (5, 6, 7):  # CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X/Y/Z
        value = ctypes.c_int()
        _check(driver.cuDeviceGetAttribute(ctypes.byref(value), attribute, 0) == 0, "cuDeviceGetAttribute failed")
        limits.append(value.value)
    _check(limits == [2147483647, 65535, 65535], f"grid dimension limits {limits}")


def check_device_properties_match_attributes(runtime: ctypes.CDLL, driver: ctypes.CDLL) -> None:
    # Field offsets from CUDA 12.4 (cudaGetDeviceProperties_v2, 1032 bytes) and
    # CUDA 11.8 (unversioned cudaGetDeviceProperties, 728 bytes) driver_types.h.
    # Each field equals the device attribute with the same meaning.
    driver.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    driver.cuDeviceGetUuid.argtypes = [ctypes.c_char_p, ctypes.c_int]

    def attribute(number: int) -> int:
        value = ctypes.c_int()
        _check(driver.cuDeviceGetAttribute(ctypes.byref(value), number, 0) == 0, f"attribute {number} failed")
        return value.value

    def read(raw: bytes, offset: int, size: int) -> int:
        return int.from_bytes(raw[offset:offset + size], "little", signed=size == 4)

    v2 = (ctypes.c_ubyte * 1032)()
    legacy = (ctypes.c_ubyte * 728)()
    _check(runtime.cudaGetDeviceProperties_v2(v2, 0) == 0, "cudaGetDeviceProperties_v2 failed")
    _check(runtime.cudaGetDeviceProperties(legacy, 0) == 0, "cudaGetDeviceProperties failed")
    uuid = ctypes.create_string_buffer(16)
    other = (ctypes.c_ubyte * 1032)()
    _check(driver.cuDeviceGetUuid(uuid, 1) == 0, "cuDeviceGetUuid failed")
    _check(runtime.cudaGetDeviceProperties_v2(other, 1) == 0, "cudaGetDeviceProperties_v2(1) failed")
    raw = bytes(v2)
    fields = {
        "totalConstMem": (read(raw, 352, 8), attribute(9)),
        "sharedMemPerMultiprocessor": (read(raw, 640, 8), attribute(81)),
        "regsPerMultiprocessor": (read(raw, 648, 4), attribute(82)),
        "singleToDoublePrecisionPerfRatio": (read(raw, 668, 4), attribute(87)),
        "cooperativeLaunch": (read(raw, 688, 4), attribute(95)),
        "maxBlocksPerMultiProcessor": (read(raw, 712, 4), attribute(106)),
        "legacy.regsPerMultiprocessor": (read(bytes(legacy), 648, 4), attribute(82)),
        "legacy.singleToDoublePrecisionPerfRatio": (read(bytes(legacy), 668, 4), attribute(87)),
        "legacy.cooperativeLaunch": (read(bytes(legacy), 688, 4), attribute(95)),
        "uuid": (bytes(other)[256:272], uuid.raw),
    }
    wrong = {name: pair for name, pair in fields.items() if pair[0] != pair[1] or not pair[1]}
    _check(not wrong, f"device properties differ from attributes: {wrong}")


def check_get_proc_address_versioned_abi(driver: ctypes.CDLL) -> None:
    # cuda.h (cuGetProcAddress) returns the newest ABI introduced at or before
    # cudaVersion; cudaTypedefs.h: PFN_cuGetProcAddress_v12000 takes five
    # arguments and PFN_cuCtxCreate_v11040 takes (pctx, params, numParams, flags, dev).
    driver.cuGetProcAddress.argtypes = [ctypes.c_char_p, ctypes.POINTER(c_void_p), ctypes.c_int, ctypes.c_ulonglong]
    legacy = ctypes.cast(driver.cuGetProcAddress, c_void_p).value
    versioned = ctypes.cast(driver.cuGetProcAddress_v2, c_void_p).value
    found = []
    for version in (11030, 12040):
        function = c_void_p()
        _check(driver.cuGetProcAddress(b"cuGetProcAddress", ctypes.byref(function), version, 0) == 0, "lookup failed")
        found.append(function.value)
    _check(found == [legacy, versioned], f"cuGetProcAddress lookups returned {found}, expected {[legacy, versioned]}")

    function = c_void_p()
    _check(driver.cuGetProcAddress(b"cuCtxCreate", ctypes.byref(function), 12040, 0) == 0, "lookup failed")
    create = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(c_void_p), c_void_p, ctypes.c_int, ctypes.c_uint, ctypes.c_int)(
        function.value
    )
    context = c_void_p()
    result = create(ctypes.byref(context), None, 0, 0, 1)
    # Simulated contexts encode device ordinal + 1.
    _check(result == 0 and context.value == 2, f"cuCtxCreate(v11040 ABI, device 1) returned {result}, {context.value}")
    driver.cuCtxSetCurrent.argtypes = [c_void_p]
    driver.cuCtxSetCurrent(c_void_p(1))


def check_interior_pointers_resolve_to_allocation(runtime: ctypes.CDLL, driver: ctypes.CDLL) -> None:
    # cuda.h (cuMemGetAddressRange): base and size of the allocation that
    # contains dptr, CUDA_ERROR_NOT_FOUND (500) otherwise. Pointer attributes
    # describe any address inside an allocation, e.g. a sub-allocated tensor.
    runtime.cudaMalloc.argtypes = [ctypes.POINTER(c_void_p), ctypes.c_size_t]
    runtime.cudaFree.argtypes = [c_void_p]
    runtime.cudaPointerGetAttributes.argtypes = [ctypes.POINTER(PointerAttributes), c_void_p]
    driver.cuMemGetAddressRange.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_uint64,
    ]
    device = c_void_p()
    _check(runtime.cudaMalloc(ctypes.byref(device), 1 << 20) == 0, "cudaMalloc failed")
    interior = device.value + 4096
    attributes = PointerAttributes()
    result = runtime.cudaPointerGetAttributes(ctypes.byref(attributes), c_void_p(interior))
    _check(result == 0 and attributes.type == 2, f"interior pointer: result {result}, type {attributes.type}")
    base, size = ctypes.c_uint64(), ctypes.c_size_t()
    result = driver.cuMemGetAddressRange(ctypes.byref(base), ctypes.byref(size), interior)
    _check(
        result == 0 and base.value == device.value and size.value == 1 << 20,
        f"cuMemGetAddressRange(interior) returned {result}, base {base.value:#x}, size {size.value}",
    )
    host = ctypes.create_string_buffer(64)
    result = driver.cuMemGetAddressRange(ctypes.byref(base), ctypes.byref(size), ctypes.addressof(host))
    _check(result == 500, f"cuMemGetAddressRange(host memory) returned {result}")
    runtime.cudaFree(device)


def check_versioned_driver_exports(driver: ctypes.CDLL) -> None:
    # CUDA 12.4 cuda.h defines each of these base names to the listed symbol,
    # so programs compiled against it import that name from libcuda.
    imported = (
        "cuMemAllocPitch_v2",
        "cuMemGetAddressRange_v2",
        "cuMemAllocHost_v2",
        "cuMemHostGetDevicePointer_v2",
        "cuMemHostRegister_v2",
        "cuDevicePrimaryCtxRelease_v2",
        "cuDevicePrimaryCtxReset_v2",
        "cuStreamWaitValue32_v2",
        "cuStreamBatchMemOp_v2",
        "cuGraphKernelNodeGetParams_v2",
        "cuGraphExecKernelNodeSetParams_v2",
    )
    missing = [name for name in imported if not hasattr(driver, name)]
    _check(not missing, f"libcuda lacks symbols that cuda.h callers import: {missing}")
    driver.cuMemAllocHost_v2.argtypes = [ctypes.POINTER(c_void_p), ctypes.c_size_t]
    driver.cuMemFreeHost.argtypes = [c_void_p]
    host = c_void_p()
    _check(driver.cuMemAllocHost_v2(ctypes.byref(host), 256) == 0 and host.value, "cuMemAllocHost_v2 failed")
    driver.cuMemFreeHost(host)



def check_zero_size_managed_allocation(runtime: ctypes.CDLL, driver: ctypes.CDLL) -> None:
    # cuda.h (cuMemAllocManaged) and cuda_runtime_api.h (cudaMallocManaged):
    # a size of 0 returns CUDA_ERROR_INVALID_VALUE / cudaErrorInvalidValue.
    driver.cuMemAllocManaged.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_size_t, ctypes.c_uint]
    runtime.cudaMallocManaged.argtypes = [ctypes.POINTER(c_void_p), ctypes.c_size_t, ctypes.c_uint]
    dptr = ctypes.c_ulonglong()
    driver_result = driver.cuMemAllocManaged(ctypes.byref(dptr), 0, 1)
    ptr = c_void_p()
    runtime_result = runtime.cudaMallocManaged(ctypes.byref(ptr), 0, 1)
    runtime.cudaGetLastError()
    actual = (driver_result, runtime_result)
    _check(actual == (1, 1), f"zero-size managed allocation returned {actual}")


# The checks below compare FakeGPU with answers measured once on an Ampere GPU
# (sm_80) with CUDA 12.4 libraries. Per-device numbers (limits, granularity,
# priorities) are FakeGPU's modeled defaults taken from that run.

CU_MEMORYTYPE_HOST = 1
CU_MEMORYTYPE_DEVICE = 2
MiB = 1 << 20
CONTEXT_BYTES_DEFAULT = 630718464  # modeled CUDA context, 601.5 MiB


def _context_bytes() -> int:
    raw = os.environ.get("FAKEGPU_CUDA_CONTEXT_BYTES", "").strip()
    try:
        return max(int(raw), 0) if raw else CONTEXT_BYTES_DEFAULT
    except ValueError:
        return CONTEXT_BYTES_DEFAULT


def check_memory_info_excludes_context(runtime: ctypes.CDLL, driver: ctypes.CDLL) -> None:
    # Runs before any allocation: free memory is total minus the modeled
    # context (FAKEGPU_CUDA_CONTEXT_BYTES), on both API levels.
    for library, name in ((runtime, "cudaMemGetInfo"), (driver, "cuMemGetInfo_v2")):
        function = getattr(library, name)
        function.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        result = function(ctypes.byref(free), ctypes.byref(total))
        _check(
            result == 0 and total.value - free.value == _context_bytes(),
            f"{name} returned {result}, total {total.value}, free {free.value}",
        )


def check_allocation_alignment_and_zero_size(runtime: ctypes.CDLL, driver: ctypes.CDLL) -> None:
    # CUDA Programming Guide: allocations are aligned to at least 256 bytes.
    # Measured: cudaMalloc(0) and cudaMallocHost(0) succeed with NULL;
    # cuMemAlloc(0) returns CUDA_ERROR_INVALID_VALUE and writes 0.
    runtime.cudaMalloc.argtypes = [ctypes.POINTER(c_void_p), ctypes.c_size_t]
    runtime.cudaFree.argtypes = [c_void_p]
    runtime.cudaMallocHost.argtypes = [ctypes.POINTER(c_void_p), ctypes.c_size_t]
    driver.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    driver.cuMemFree_v2.argtypes = [ctypes.c_uint64]
    misaligned = []
    for size in (1, 3, 100, 255, 257, 1000, 4097, 65537):
        pointer = c_void_p()
        _check(runtime.cudaMalloc(ctypes.byref(pointer), size) == 0, f"cudaMalloc({size}) failed")
        address = ctypes.c_uint64()
        _check(driver.cuMemAlloc_v2(ctypes.byref(address), size) == 0, f"cuMemAlloc({size}) failed")
        if pointer.value % 256 or address.value % 256:
            misaligned.append((size, pointer.value, address.value))
        runtime.cudaFree(pointer)
        driver.cuMemFree_v2(address)
    _check(not misaligned, f"allocations not 256-byte aligned: {misaligned}")

    pointer = c_void_p(0x1)
    device_zero = (runtime.cudaMalloc(ctypes.byref(pointer), 0), pointer.value)
    host = c_void_p(0x1)
    host_zero = (runtime.cudaMallocHost(ctypes.byref(host), 0), host.value)
    address = ctypes.c_uint64(0x1)
    driver_zero = (driver.cuMemAlloc_v2(ctypes.byref(address), 0), address.value)
    runtime.cudaGetLastError()
    actual = (device_zero, host_zero, driver_zero)
    _check(actual == ((0, None), (0, None), (1, 0)), f"zero-size allocations returned {actual}")

    # cuda.h: flags must be exactly CU_MEM_ATTACH_GLOBAL (1) or CU_MEM_ATTACH_HOST (2).
    driver.cuMemAllocManaged.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t, ctypes.c_uint]
    runtime.cudaMallocManaged.argtypes = [ctypes.POINTER(c_void_p), ctypes.c_size_t, ctypes.c_uint]
    results = {}
    for flags in (0, 1, 2, 3, 4):
        managed = ctypes.c_uint64()
        driver_result = driver.cuMemAllocManaged(ctypes.byref(managed), 4096, flags)
        if driver_result == 0:
            driver.cuMemFree_v2(managed)
        pointer = c_void_p()
        runtime_result = runtime.cudaMallocManaged(ctypes.byref(pointer), 4096, flags)
        if runtime_result == 0:
            runtime.cudaFree(pointer)
        results[flags] = (driver_result, runtime_result)
    runtime.cudaGetLastError()
    expected = {0: (1, 1), 1: (0, 0), 2: (0, 0), 3: (1, 1), 4: (1, 1)}
    _check(results == expected, f"managed allocation flags returned {results}")


def check_pending_events_are_not_ready(runtime: ctypes.CDLL) -> None:
    # cuda_runtime_api.h: cudaErrorNotReady (600) while either event is still
    # pending. Measured: an end event recorded before the start event gives a
    # negative time.
    runtime.cudaEventCreate.argtypes = [ctypes.POINTER(c_void_p)]
    runtime.cudaEventRecord.argtypes = [c_void_p, c_void_p]
    runtime.cudaEventSynchronize.argtypes = [c_void_p]
    runtime.cudaEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), c_void_p, c_void_p]
    runtime.cudaEventDestroy.argtypes = [c_void_p]
    start, end, later = c_void_p(), c_void_p(), c_void_p()
    for event in (start, end, later):
        _check(runtime.cudaEventCreate(ctypes.byref(event)) == 0, "cudaEventCreate failed")
    runtime.cudaEventRecord(start, None)
    runtime.cudaEventRecord(end, None)
    milliseconds = ctypes.c_float(-1)
    pending = runtime.cudaEventElapsedTime(ctypes.byref(milliseconds), start, end)
    runtime.cudaGetLastError()
    runtime.cudaEventSynchronize(end)
    completed = runtime.cudaEventElapsedTime(ctypes.byref(milliseconds), start, end)
    forward = milliseconds.value
    runtime.cudaEventRecord(later, None)
    runtime.cudaDeviceSynchronize()
    reverse = runtime.cudaEventElapsedTime(ctypes.byref(milliseconds), later, end)
    backward = milliseconds.value
    for event in (start, end, later):
        runtime.cudaEventDestroy(event)
    actual = (pending, completed, forward >= 0, reverse, backward < 0)
    _check(actual == (600, 0, True, 0, True), f"pending event timing returned {actual}")


def check_context_limits(runtime: ctypes.CDLL, driver: ctypes.CDLL) -> None:
    # Modeled defaults and rounding of cudaLimit*/CU_LIMIT_* (0-6 share ids),
    # measured on an Ampere GPU. The runtime and the driver share one set.
    runtime.cudaDeviceGetLimit.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
    runtime.cudaDeviceSetLimit.argtypes = [ctypes.c_int, ctypes.c_size_t]
    driver.cuCtxGetLimit.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
    driver.cuCtxSetLimit.argtypes = [ctypes.c_int, ctypes.c_size_t]
    driver.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    _check(runtime.cudaSetDevice(0) == 0, "cudaSetDevice(0) failed")

    def get(limit: int) -> tuple[int, int, int, int]:
        runtime_value, driver_value = ctypes.c_size_t(0xDEAD), ctypes.c_size_t(0xDEAD)
        runtime_result = runtime.cudaDeviceGetLimit(ctypes.byref(runtime_value), limit)
        driver_result = driver.cuCtxGetLimit(ctypes.byref(driver_value), limit)
        return runtime_result, runtime_value.value, driver_result, driver_value.value

    defaults = {0: 1024, 1: 6291456, 2: 8388608, 3: 2, 4: 2048, 5: 64, 6: 6291456}
    actual = {limit: get(limit) for limit in defaults}
    _check(
        actual == {limit: (0, value, 0, value) for limit, value in defaults.items()},
        f"default limits {actual}",
    )
    optin = ctypes.c_int()
    driver.cuDeviceGetAttribute(ctypes.byref(optin), 97, 0)  # MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
    _check(get(7) == (0, optin.value, 0, optin.value), f"shared memory limit {get(7)}")

    # (limit, requested, set result, value read back)
    cases = [
        (0, 1, 0, 16), (0, 1025, 0, 1040), (1, 1, 0, 524288), (1, 600000, 0, 600064),
        (2, 1, 0, 4194304), (2, 5000000, 0, 5046272), (3, 0, 0, 0), (4, 0, 0, 32), (4, 33, 0, 33),
        (5, 100, 0, 100), (5, 256, 1, 100), (6, 1048576, 0, 2097152), (6, 1 << 40, 1, 2097152),
        (7, 1, 800, optin.value),
    ]
    wrong = []
    for limit, requested, expected_result, expected_value in cases:
        result = runtime.cudaDeviceSetLimit(limit, requested)
        values = get(limit)
        if result != expected_result or values != (0, expected_value, 0, expected_value):
            wrong.append((limit, requested, result, values))
    stack = driver.cuCtxSetLimit(0, 2048)
    stack_value = get(0)[1]
    unknown = (runtime.cudaDeviceSetLimit(99, 1), driver.cuCtxGetLimit(None, 0))
    runtime.cudaGetLastError()
    for limit, value in defaults.items():
        driver.cuCtxSetLimit(limit, value)
    _check(not wrong, f"limit updates {wrong}")
    _check((stack, stack_value, unknown) == (0, 2048, (1, 1)), f"driver limit update {stack, stack_value, unknown}")


def check_stream_priorities_ids_and_attributes(runtime: ctypes.CDLL, driver: ctypes.CDLL) -> None:
    # Modeled priority range (least 0, greatest -5) with clamping, shared by
    # cudaDeviceGetStreamPriorityRange and cuCtxGetStreamPriorityRange.
    lookup = runtime if sys.platform == "darwin" else driver
    least, greatest = ctypes.c_int(7), ctypes.c_int(7)
    runtime.cudaDeviceGetStreamPriorityRange(ctypes.byref(least), ctypes.byref(greatest))
    driver_least, driver_greatest = ctypes.c_int(7), ctypes.c_int(7)
    driver.cuCtxGetStreamPriorityRange(ctypes.byref(driver_least), ctypes.byref(driver_greatest))
    ranges = (least.value, greatest.value, driver_least.value, driver_greatest.value)
    _check(ranges == (0, -5, 0, -5), f"stream priority ranges {ranges}")

    runtime.cudaStreamCreateWithPriority.argtypes = [ctypes.POINTER(c_void_p), ctypes.c_uint, ctypes.c_int]
    runtime.cudaStreamGetPriority.argtypes = [c_void_p, ctypes.POINTER(ctypes.c_int)]
    runtime.cudaStreamDestroy.argtypes = [c_void_p]
    clamped = {}
    streams = []
    for requested in (-100, -3, 0, 7):
        stream = c_void_p()
        _check(runtime.cudaStreamCreateWithPriority(ctypes.byref(stream), 0, requested) == 0, "stream create failed")
        priority = ctypes.c_int(99)
        runtime.cudaStreamGetPriority(stream, ctypes.byref(priority))
        clamped[requested] = priority.value
        streams.append(stream)
    _check(clamped == {-100: -5, -3: -3, 0: 0, 7: 0}, f"stream priorities {clamped}")

    # cudaStreamGetId: unique per stream, the same through the driver API.
    runtime.cudaStreamGetId.argtypes = [c_void_p, ctypes.POINTER(ctypes.c_ulonglong)]
    lookup.cuStreamGetId.argtypes = [c_void_p, ctypes.POINTER(ctypes.c_ulonglong)]
    ids = []
    for stream in (c_void_p(0), c_void_p(0x2), streams[0], streams[1]):
        value, driver_value = ctypes.c_ulonglong(), ctypes.c_ulonglong()
        _check(runtime.cudaStreamGetId(stream, ctypes.byref(value)) == 0, "cudaStreamGetId failed")
        _check(lookup.cuStreamGetId(stream, ctypes.byref(driver_value)) == 0, "cuStreamGetId failed")
        _check(value.value == driver_value.value, f"stream id {value.value} != {driver_value.value}")
        ids.append(value.value)
    legacy = ctypes.c_ulonglong()
    runtime.cudaStreamGetId(c_void_p(0x1), ctypes.byref(legacy))  # cudaStreamLegacy
    null_output = runtime.cudaStreamGetId(None, None)
    runtime.cudaGetLastError()
    _check(len(set(ids)) == 4 and legacy.value == ids[0] and null_output == 1, f"stream ids {ids}, {legacy.value}")

    # cudaStreamAttrValue is a 64-byte union. Ids: access policy window 1,
    # synchronization policy 3, priority 8, mem sync domain map 9, domain 10.
    runtime.cudaStreamGetAttribute.argtypes = [c_void_p, ctypes.c_int, c_void_p]
    runtime.cudaStreamSetAttribute.argtypes = [c_void_p, ctypes.c_int, c_void_p]
    stream = streams[2]

    def attribute(attr: int) -> tuple[int, bytes]:
        value = (ctypes.c_ubyte * 64)(*([0xA5] * 64))
        return runtime.cudaStreamGetAttribute(stream, attr, value), bytes(value)

    def as_int(raw: bytes) -> int:
        return int.from_bytes(raw[:4], "little", signed=True)

    sync_result, sync_raw = attribute(3)
    priority_result, priority_raw = attribute(8)
    window_result, window_raw = attribute(1)
    defaults = (sync_result, as_int(sync_raw), priority_result, as_int(priority_raw), window_result, window_raw[:32])
    _check(defaults == (0, -1, 0, 0, 0, bytes(32)), f"stream attribute defaults {defaults}")
    value = (ctypes.c_ubyte * 64)()
    ctypes.c_int.from_buffer(value).value = 4  # cudaSyncPolicyBlockingSync
    set_sync = runtime.cudaStreamSetAttribute(stream, 3, value)
    ctypes.c_int.from_buffer(value).value = -100
    set_priority = runtime.cudaStreamSetAttribute(stream, 8, value)
    priority = ctypes.c_int()
    runtime.cudaStreamGetPriority(stream, ctypes.byref(priority))
    invalid = attribute(2)[0]  # cooperative is a launch attribute only
    runtime.cudaGetLastError()
    actual = (set_sync, as_int(attribute(3)[1]), set_priority, priority.value, invalid)
    _check(actual == (0, 4, 0, -5, 1), f"stream attribute updates {actual}")
    for stream in streams:
        runtime.cudaStreamDestroy(stream)


def check_host_registration_and_pointer_attributes(runtime: ctypes.CDLL, driver: ctypes.CDLL) -> None:
    # Measured cuMemHostRegister/cuPointerGetAttribute contracts.
    import mmap

    driver.cuMemHostRegister_v2.argtypes = [c_void_p, ctypes.c_size_t, ctypes.c_uint]
    driver.cuMemHostUnregister.argtypes = [c_void_p]
    driver.cuMemHostGetFlags.argtypes = [ctypes.POINTER(ctypes.c_uint), c_void_p]
    driver.cuMemHostGetDevicePointer_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), c_void_p, ctypes.c_uint]
    driver.cuPointerGetAttribute.argtypes = [c_void_p, ctypes.c_int, ctypes.c_uint64]
    driver.cuPointerGetAttributes.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(c_void_p), ctypes.c_uint64]
    runtime.cudaHostGetDevicePointer.argtypes = [ctypes.POINTER(c_void_p), c_void_p, ctypes.c_uint]

    mapping = mmap.mmap(-1, 3 * 4096)
    base = ctypes.addressof(ctypes.c_char.from_buffer(mapping))

    def attribute(attr: int, address: int) -> tuple[int, int]:
        value = ctypes.c_uint64(0x5A5A5A5A5A5A5A5A)
        return driver.cuPointerGetAttribute(ctypes.byref(value), attr, address), value.value

    # Unregistered memory: single queries fail, the batched query describes it.
    unknown = attribute(2, base)[0]
    names = (ctypes.c_int * 4)(1, 3, 4, 9)  # CONTEXT, DEVICE_POINTER, HOST_POINTER, DEVICE_ORDINAL
    slots = [ctypes.c_uint64(0x5A5A5A5A5A5A5A5A) for _ in range(4)]
    data = (c_void_p * 4)(*(ctypes.cast(ctypes.byref(slot), c_void_p) for slot in slots))
    batched = driver.cuPointerGetAttributes(4, names, data, base)
    ordinal = ctypes.c_int.from_address(ctypes.addressof(slots[3])).value
    actual = (unknown, batched, slots[0].value, slots[1].value == base, slots[2].value == base, ordinal)
    _check(actual == (1, 0, 0, True, True, -1), f"unregistered pointer attributes {actual}")

    size = 2 * 4096
    flags = ctypes.c_uint(0xDEAD)
    sequence = [
        driver.cuMemHostRegister_v2(c_void_p(base), size, 0),
        driver.cuMemHostRegister_v2(c_void_p(base), size, 0),
        driver.cuMemHostRegister_v2(c_void_p(base + 4096), size, 0),
        driver.cuMemHostGetFlags(ctypes.byref(flags), c_void_p(base + 100)),
    ]
    _check(sequence == [0, 712, 712, 0] and flags.value == 0x2, f"host registration {sequence}, flags {flags.value:#x}")
    result, value = attribute(2, base + 100)
    memory_type = (result, value & 0xFFFFFFFF)  # CUmemorytype is 4 bytes
    range_start = attribute(11, base + 8191)
    range_size = attribute(12, base + 8191)
    outside = attribute(2, base + size)[0]
    _check(
        memory_type == (0, CU_MEMORYTYPE_HOST) and range_start == (0, base) and range_size == (0, size)
        and outside == 1,
        f"registered pointer attributes {memory_type}, {range_start}, {range_size}, {outside}",
    )
    device_pointer = ctypes.c_uint64()
    mapped = driver.cuMemHostGetDevicePointer_v2(ctypes.byref(device_pointer), c_void_p(base + 64), 0)
    bad_flags = driver.cuMemHostGetDevicePointer_v2(ctypes.byref(device_pointer), c_void_p(base), 1)
    runtime_pointer = c_void_p()
    runtime_mapped = runtime.cudaHostGetDevicePointer(ctypes.byref(runtime_pointer), c_void_p(base), 0)
    _check(
        (mapped, bad_flags, runtime_mapped, runtime_pointer.value) == (0, 1, 0, base),
        f"host device pointers {mapped, bad_flags, runtime_mapped, runtime_pointer.value}",
    )
    teardown = [
        driver.cuMemHostUnregister(c_void_p(base + 100)),
        driver.cuMemHostUnregister(c_void_p(base)),
        driver.cuMemHostUnregister(c_void_p(base)),
        driver.cuMemHostRegister_v2(c_void_p(base), 0, 0),
    ]
    runtime.cudaGetLastError()
    del data
    mapping.close()
    _check(teardown == [1, 0, 713, 1], f"host unregistration {teardown}")


class MemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class MemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", MemLocation),
        ("win32HandleMetaData", c_void_p),
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class MemAccessDesc(ctypes.Structure):
    _fields_ = [("location", MemLocation), ("flags", ctypes.c_int)]


def check_virtual_memory_management(driver: ctypes.CDLL) -> None:
    # The cuMem* VMM flow PyTorch's expandable segments use. Measured: 2 MiB
    # granularity, physical memory is counted from cuMemCreate until the last
    # mapping and handle are gone, and FABRIC handles need permissions.
    c_u64 = ctypes.c_uint64
    driver.cuMemGetAllocationGranularity.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(MemAllocationProp), ctypes.c_int]
    driver.cuMemCreate.argtypes = [ctypes.POINTER(c_u64), ctypes.c_size_t, ctypes.POINTER(MemAllocationProp), ctypes.c_ulonglong]
    driver.cuMemRelease.argtypes = [c_u64]
    driver.cuMemAddressReserve.argtypes = [ctypes.POINTER(c_u64), ctypes.c_size_t, ctypes.c_size_t, c_u64, ctypes.c_ulonglong]
    driver.cuMemAddressFree.argtypes = [c_u64, ctypes.c_size_t]
    driver.cuMemMap.argtypes = [c_u64, ctypes.c_size_t, ctypes.c_size_t, c_u64, ctypes.c_ulonglong]
    driver.cuMemUnmap.argtypes = [c_u64, ctypes.c_size_t]
    driver.cuMemSetAccess.argtypes = [c_u64, ctypes.c_size_t, ctypes.POINTER(MemAccessDesc), ctypes.c_size_t]
    driver.cuMemGetAccess.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(MemLocation), c_u64]
    driver.cuMemRetainAllocationHandle.argtypes = [ctypes.POINTER(c_u64), c_void_p]
    driver.cuMemExportToShareableHandle.argtypes = [c_void_p, c_u64, ctypes.c_int, ctypes.c_ulonglong]
    driver.cuMemGetInfo_v2.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    driver.cuMemsetD8_v2.argtypes = [c_u64, ctypes.c_ubyte, ctypes.c_size_t]
    driver.cuMemcpyDtoH_v2.argtypes = [c_void_p, c_u64, ctypes.c_size_t]
    driver.cuMemFree_v2.argtypes = [c_u64]
    driver.cuMemGetAddressRange_v2.argtypes = [ctypes.POINTER(c_u64), ctypes.POINTER(ctypes.c_size_t), c_u64]
    driver.cuPointerGetAttribute.argtypes = [c_void_p, ctypes.c_int, c_u64]

    def free_bytes() -> int:
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        driver.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total))
        return free.value

    prop = MemAllocationProp(type=1, location=MemLocation(type=1, id=0))  # PINNED on device 0
    granularity = ctypes.c_size_t()
    _check(
        driver.cuMemGetAllocationGranularity(ctypes.byref(granularity), ctypes.byref(prop), 0) == 0
        and granularity.value == 2 * MiB,
        f"allocation granularity {granularity.value}",
    )
    size = 2 * granularity.value
    handle = c_u64()
    before = free_bytes()
    _check(driver.cuMemCreate(ctypes.byref(handle), size, ctypes.byref(prop), 0) == 0, "cuMemCreate failed")
    _check(before - free_bytes() == size, f"cuMemCreate counted {before - free_bytes()} bytes")
    rejected = (
        driver.cuMemCreate(ctypes.byref(c_u64()), size + 1, ctypes.byref(prop), 0),
        driver.cuMemCreate(ctypes.byref(c_u64()), size, ctypes.byref(prop), 1),
        driver.cuMemCreate(ctypes.byref(c_u64()), size, ctypes.byref(MemAllocationProp(
            type=1, requestedHandleTypes=8, location=MemLocation(type=1, id=0))), 0),  # FABRIC
    )
    _check(rejected == (1, 1, 800), f"invalid cuMemCreate calls returned {rejected}")

    va = c_u64()
    _check(driver.cuMemAddressReserve(ctypes.byref(va), size, 0, 0, 0) == 0, "cuMemAddressReserve failed")
    _check(va.value % granularity.value == 0, f"reservation {va.value:#x} is not aligned")
    unmapped_type = ctypes.c_uint(0)
    unmapped = driver.cuPointerGetAttribute(ctypes.byref(unmapped_type), 2, va.value)
    _check(driver.cuMemMap(va.value, size, 0, handle.value, 0) == 0, "cuMemMap failed")
    memory_type = ctypes.c_uint(0)
    mapped = driver.cuPointerGetAttribute(ctypes.byref(memory_type), 2, va.value + 100)
    _check(
        (unmapped, mapped, memory_type.value) == (1, 0, CU_MEMORYTYPE_DEVICE),
        f"VMM pointer attributes {unmapped, mapped, memory_type.value}",
    )
    access = MemAccessDesc(location=MemLocation(type=1, id=0), flags=3)
    _check(driver.cuMemSetAccess(va.value, size, ctypes.byref(access), 1) == 0, "cuMemSetAccess failed")
    flags = ctypes.c_ulonglong()
    location = MemLocation(type=1, id=0)
    _check(
        driver.cuMemGetAccess(ctypes.byref(flags), ctypes.byref(location), va.value) == 0 and flags.value == 3,
        f"cuMemGetAccess flags {flags.value}",
    )
    _check(driver.cuMemsetD8_v2(va.value + 4096, 0x11, 16) == 0, "cuMemsetD8 on a VMM mapping failed")
    readback = (ctypes.c_ubyte * 16)()
    driver.cuMemcpyDtoH_v2(readback, va.value + 4096, 16)
    _check(bytes(readback) == b"\x11" * 16, f"VMM mapping read back {bytes(readback).hex()}")
    base, extent = c_u64(), ctypes.c_size_t()
    address_range = driver.cuMemGetAddressRange_v2(ctypes.byref(base), ctypes.byref(extent), va.value + 100)
    _check(
        (address_range, base.value, extent.value) == (0, va.value, size),
        f"cuMemGetAddressRange on a VMM mapping returned {address_range, base.value, extent.value}",
    )
    retained = c_u64()
    retain = driver.cuMemRetainAllocationHandle(ctypes.byref(retained), c_void_p(va.value))
    _check(retain == 0 and retained.value == handle.value, "cuMemRetainAllocationHandle failed")
    driver.cuMemRelease(retained.value)
    shareable = ctypes.c_int(-1)
    exported = driver.cuMemExportToShareableHandle(ctypes.byref(shareable), handle.value, 1, 0)
    free_mapping = driver.cuMemFree_v2(va.value)
    _check((exported, free_mapping) == (1, 1), f"export/cuMemFree on a mapping returned {exported, free_mapping}")

    # Releasing a mapped handle keeps the memory until the mapping goes away.
    releases = (driver.cuMemRelease(handle.value), before - free_bytes())
    unmaps = (
        driver.cuMemUnmap(va.value, granularity.value),
        driver.cuMemUnmap(va.value, size),
        before - free_bytes(),
        driver.cuMemRelease(handle.value),
    )
    frees = (driver.cuMemAddressFree(va.value, size), driver.cuMemAddressFree(va.value, size))
    actual = (releases, unmaps, frees)
    _check(actual == ((0, size), (1, 0, 0, 1), (0, 1)), f"VMM teardown {actual}")


def check_stream_write_value(runtime: ctypes.CDLL, driver: ctypes.CDLL) -> None:
    # cuda.h (cuStreamWriteValue32): 4-byte aligned device addresses only;
    # CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER (1) is accepted by the v1 entry
    # point, the CUDA 12 _v2 ABI rejects every flag.
    driver.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    driver.cuMemFree_v2.argtypes = [ctypes.c_uint64]
    driver.cuMemcpyDtoH_v2.argtypes = [c_void_p, ctypes.c_uint64, ctypes.c_size_t]
    driver.cuMemsetD8_v2.argtypes = [ctypes.c_uint64, ctypes.c_ubyte, ctypes.c_size_t]
    for name in ("cuStreamWriteValue32", "cuStreamWriteValue32_v2"):
        getattr(driver, name).argtypes = [c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint]
    address = ctypes.c_uint64()
    _check(driver.cuMemAlloc_v2(ctypes.byref(address), 8) == 0, "cuMemAlloc failed")
    driver.cuMemsetD8_v2(address.value, 0, 8)
    results = (
        driver.cuStreamWriteValue32_v2(None, address.value + 4, 0xABCDEF12, 0),
        driver.cuStreamWriteValue32_v2(None, address.value + 2, 1, 0),
        driver.cuStreamWriteValue32_v2(None, address.value, 1, 1),
        driver.cuStreamWriteValue32_v2(None, 0, 1, 0),
        driver.cuStreamWriteValue32(None, address.value, 9, 1),
    )
    driver.cuCtxSynchronize()
    values = (ctypes.c_uint32 * 2)()
    driver.cuMemcpyDtoH_v2(values, address.value, 8)
    driver.cuMemFree_v2(address.value)
    actual = (results, list(values))
    _check(actual == ((0, 1, 1, 1, 0), [9, 0xABCDEF12]), f"cuStreamWriteValue32 returned {actual}")


def check_ipc_is_not_supported(runtime: ctypes.CDLL, driver: ctypes.CDLL) -> None:
    # Simulated memory cannot be shared with another process, so IPC returns
    # not supported (801) instead of handles that could not be opened.
    runtime.cudaMalloc.argtypes = [ctypes.POINTER(c_void_p), ctypes.c_size_t]
    runtime.cudaFree.argtypes = [c_void_p]
    runtime.cudaIpcGetMemHandle.argtypes = [c_void_p, c_void_p]
    pointer = c_void_p()
    _check(runtime.cudaMalloc(ctypes.byref(pointer), 4096) == 0, "cudaMalloc failed")
    handle = (ctypes.c_char * 64)()
    runtime_result = runtime.cudaIpcGetMemHandle(handle, pointer)
    driver.cuIpcGetMemHandle.argtypes = [c_void_p, ctypes.c_uint64]
    driver_result = driver.cuIpcGetMemHandle(handle, pointer.value)
    runtime.cudaFree(pointer)
    runtime.cudaGetLastError()
    _check((runtime_result, driver_result) == (801, 801), f"IPC returned {runtime_result, driver_result}")


def check_primary_context_state_and_device_flags(runtime: ctypes.CDLL, driver: ctypes.CDLL) -> None:
    # Measured: the primary context is inactive until retained, a release
    # without a retain returns CUDA_ERROR_INVALID_CONTEXT, cudaGetDeviceFlags
    # always reports cudaDeviceMapHost (0x08), and cudaSetDeviceFlags rejects
    # two scheduling modes at once.
    driver.cuDevicePrimaryCtxGetState.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_int)]
    driver.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(c_void_p), ctypes.c_int]
    driver.cuDevicePrimaryCtxRelease_v2.argtypes = [ctypes.c_int]
    driver.cuDevicePrimaryCtxSetFlags_v2.argtypes = [ctypes.c_int, ctypes.c_uint]
    device = 3  # untouched by the runtime so far

    def state() -> tuple[int, int, int]:
        flags, active = ctypes.c_uint(0xDEAD), ctypes.c_int(-7)
        result = driver.cuDevicePrimaryCtxGetState(device, ctypes.byref(flags), ctypes.byref(active))
        return result, flags.value, active.value

    context = c_void_p()
    sequence = [state()]
    sequence.append((driver.cuDevicePrimaryCtxRetain(ctypes.byref(context), device),) + state()[1:])
    sequence.append((driver.cuDevicePrimaryCtxRelease_v2(device),) + state()[1:])
    sequence.append((driver.cuDevicePrimaryCtxRelease_v2(device),))
    sequence.append((driver.cuDevicePrimaryCtxSetFlags_v2(device, 4),) + state()[1:])
    sequence.append((driver.cuDevicePrimaryCtxSetFlags_v2(device, 3),))
    driver.cuDevicePrimaryCtxSetFlags_v2(device, 0)
    expected = [(0, 0, 0), (0, 0, 1), (0, 0, 0), (201,), (0, 4, 0), (1,)]
    _check(sequence == expected, f"primary context sequence {sequence}")

    runtime.cudaGetDeviceFlags.argtypes = [ctypes.POINTER(ctypes.c_uint)]
    runtime.cudaSetDeviceFlags.argtypes = [ctypes.c_uint]
    runtime.cudaInitDevice.argtypes = [ctypes.c_int, ctypes.c_uint, ctypes.c_uint]
    _check(runtime.cudaSetDevice(0) == 0, "cudaSetDevice(0) failed")

    def device_flags() -> int:
        flags = ctypes.c_uint(0)
        _check(runtime.cudaGetDeviceFlags(ctypes.byref(flags)) == 0, "cudaGetDeviceFlags failed")
        return flags.value

    results = [
        runtime.cudaSetDeviceFlags(4), device_flags(),   # BlockingSync
        runtime.cudaSetDeviceFlags(0x10), device_flags(),  # LmemResizeToMax
        runtime.cudaSetDeviceFlags(3), runtime.cudaSetDeviceFlags(0x100), runtime.cudaGetDeviceFlags(None),
        runtime.cudaInitDevice(1, 2, 1), runtime.cudaInitDevice(9999, 0, 0),
    ]
    runtime.cudaGetLastError()
    runtime.cudaSetDeviceFlags(0)
    current = ctypes.c_int(-1)
    runtime.cudaGetDevice(ctypes.byref(current))
    expected_flags = [0, 0x0C, 0, 0x18, 1, 1, 1, 0, 101]
    _check(results == expected_flags and current.value == 0, f"device flags {results}, current device {current.value}")


def check_occupancy_and_attribute_bounds(runtime: ctypes.CDLL, driver: ctypes.CDLL) -> None:
    # One occupancy model for both APIs: thread, block-count and shared
    # memory limits of the device (values for the default A100 profile).
    runtime.cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.argtypes = [
        ctypes.POINTER(ctypes.c_int), c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_uint]
    lookup = runtime if sys.platform == "darwin" else driver
    lookup.cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.argtypes = [
        ctypes.POINTER(ctypes.c_int), c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_uint]
    expected = {(1, 0, 0): 32, (128, 0, 0): 16, (96, 0, 0): 21, (1024, 0, 0): 2, (1025, 0, 0): 0,
                (128, 49153, 0): 0, (0, 0, 1): None, (128, 0, 2): None}
    actual = {}
    for (block, shared, flags) in expected:
        blocks, driver_blocks = ctypes.c_int(-1), ctypes.c_int(-1)
        result = runtime.cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
            ctypes.byref(blocks), c_void_p(1), block, shared, flags)
        driver_result = lookup.cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
            ctypes.byref(driver_blocks), c_void_p(1), block, shared, flags)
        actual[(block, shared, flags)] = (
            (blocks.value if result == 0 else None) if result == driver_result else ("mismatch", result, driver_result)
        )
    runtime.cudaGetLastError()
    _check(actual == expected, f"occupancy {actual}")

    # CUDA 12.8 cuda.h: attributes 1 .. CU_DEVICE_ATTRIBUTE_MAX - 1 (143).
    driver.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    results = {}
    for attribute in (-1, 0, 144, 1000, 1):
        value = ctypes.c_int(-99)
        results[attribute] = (driver.cuDeviceGetAttribute(ctypes.byref(value), attribute, 0), value.value != -99)
    _check(
        results == {-1: (1, False), 0: (1, False), 144: (1, False), 1000: (1, False), 1: (0, True)},
        f"device attribute bounds {results}",
    )


def check_versions_and_versioned_lookups(runtime: ctypes.CDLL, driver: ctypes.CDLL) -> None:
    # One modeled driver: CUDA 12.8 for cuDriverGetVersion and
    # cudaDriverGetVersion. cuGetProcAddress returns the _v2 ABI of
    # cuIpcOpenMemHandle from 11000 and of cuStreamWriteValue32 from 12000.
    driver_version, runtime_version = ctypes.c_int(), ctypes.c_int()
    driver.cuDriverGetVersion(ctypes.byref(driver_version))
    runtime.cudaDriverGetVersion(ctypes.byref(runtime_version))
    _check((driver_version.value, runtime_version.value) == (12080, 12080), "driver versions disagree")
    driver.cuGetProcAddress.argtypes = [ctypes.c_char_p, ctypes.POINTER(c_void_p), ctypes.c_int, ctypes.c_ulonglong]
    lookups = {
        (b"cuIpcOpenMemHandle", 12000): "cuIpcOpenMemHandle_v2",
        (b"cuIpcOpenMemHandle", 10000): "cuIpcOpenMemHandle",
        (b"cuStreamWriteValue32", 12000): "cuStreamWriteValue32_v2",
        (b"cuStreamWriteValue32", 11000): "cuStreamWriteValue32",
        (b"cuMemCreate", 12000): "cuMemCreate",
        (b"cuMemAddressReserve", 12000): "cuMemAddressReserve",
    }
    wrong = []
    for (symbol, version), expected in lookups.items():
        function = c_void_p()
        result = driver.cuGetProcAddress(symbol, ctypes.byref(function), version, 0)
        if result != 0 or function.value != ctypes.cast(getattr(driver, expected), c_void_p).value:
            wrong.append((symbol, version, result))
    _check(not wrong, f"cuGetProcAddress lookups {wrong}")


def check_no_visible_device(runtime_path: Path, driver_path: Path) -> None:
    # Measured with CUDA_VISIBLE_DEVICES="" and "-1": runtime calls that need a
    # device, allocations included, return cudaErrorNoDevice (100). Skipped
    # by builds that do not hide devices.
    import subprocess

    code = (
        "import ctypes, json\n"
        f"cuda = ctypes.CDLL({str(driver_path)!r}, mode=ctypes.RTLD_GLOBAL)\n"
        f"rt = ctypes.CDLL({str(runtime_path)!r}, mode=ctypes.RTLD_GLOBAL)\n"
        "count, p, f, t = ctypes.c_int(-1), ctypes.c_void_p(), ctypes.c_size_t(), ctypes.c_size_t()\n"
        "rt.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]\n"
        "first = rt.cudaGetDeviceCount(ctypes.byref(count))\n"
        "print(json.dumps([first, rt.cudaMalloc(ctypes.byref(p), 1024), rt.cudaMalloc(ctypes.byref(p), 0),\n"
        "    rt.cudaMemGetInfo(ctypes.byref(f), ctypes.byref(t)), rt.cudaFree(None)]))\n"
    )
    for hidden in ("", "-1"):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=hidden)
        completed = subprocess.run([sys.executable, "-c", code], env=env, text=True, capture_output=True, timeout=60)
        _check(completed.returncode == 0, f"no-device probe failed: {completed.stderr}")
        results = json.loads(completed.stdout.strip().splitlines()[-1])
        if results[0] == 0:
            print(f"[cuda_api_contracts] CUDA_VISIBLE_DEVICES={hidden!r} does not hide devices here; skipped")
            return
        _check(results == [100, 100, 100, 100, 100], f"no visible device returned {results}")


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

    check_memory_info_excludes_context(runtime, driver)
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
    check_event_elapsed_time_errors(runtime)
    check_grid_dimension_limits(driver)
    check_device_properties_match_attributes(runtime, driver)
    check_get_proc_address_versioned_abi(driver)
    check_interior_pointers_resolve_to_allocation(runtime, driver)
    check_versioned_driver_exports(driver)
    check_zero_size_managed_allocation(runtime, driver)
    check_allocation_alignment_and_zero_size(runtime, driver)
    check_pending_events_are_not_ready(runtime)
    check_context_limits(runtime, driver)
    check_stream_priorities_ids_and_attributes(runtime, driver)
    check_host_registration_and_pointer_attributes(runtime, driver)
    check_virtual_memory_management(driver)
    check_stream_write_value(runtime, driver)
    check_ipc_is_not_supported(runtime, driver)
    check_primary_context_state_and_device_flags(runtime, driver)
    check_occupancy_and_attribute_bounds(runtime, driver)
    check_versions_and_versioned_lookups(runtime, driver)
    check_no_visible_device(runtime_path, driver_path)
    print("CUDA API contract validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
