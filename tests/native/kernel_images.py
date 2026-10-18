#!/usr/bin/env python3
"""Validate PTX image loading and CUDA launch entry points without nvcc."""

from __future__ import annotations

import argparse
import ctypes as ct
import os
import struct
import sys
import tempfile
from pathlib import Path


PTX = b""".version 8.0
.target sm_80
.address_size 64
.global .u32 global_value = 7;
.visible .entry set_value(.param .u64 output) {
    .reg .u64 %address;
    ld.param.u64 %address, [output];
    st.global.u32 [%address], 42;
    ret;
}
"""


def fatbin(payload: bytes, *, kind: int = 1, expanded: int = 0,
           compression: int = 0x2000, header_size: int = 64) -> bytes:
    entry = bytearray(header_size)
    padded = payload + b"\0" * (-len(payload) % 8)
    struct.pack_into("<HHIQ", entry, 0, kind, 0x101, len(entry), len(padded))
    if expanded:
        struct.pack_into("<I", entry, 16, len(payload))
        struct.pack_into("<Q", entry, 40, compression)
        if header_size >= 64:
            struct.pack_into("<Q", entry, 56, expanded)
    return struct.pack("<IHHQ", 0xBA55ED50, 1, 16, len(entry) + len(padded)) + entry + padded


def bind(library: ct.CDLL, name: str, arguments: list, result=ct.c_int):
    function = getattr(library, name)
    function.argtypes = arguments
    function.restype = result
    return function


class DriverLaunchConfig(ct.Structure):
    _fields_ = [
        ("grid", ct.c_uint * 3),
        ("block", ct.c_uint * 3),
        ("shared", ct.c_uint),
        ("stream", ct.c_void_p),
        ("attributes", ct.c_void_p),
        ("attribute_count", ct.c_uint),
    ]


class Dim3(ct.Structure):
    _fields_ = [("x", ct.c_uint), ("y", ct.c_uint), ("z", ct.c_uint)]


class RuntimeLaunchConfig(ct.Structure):
    _fields_ = [
        ("grid", Dim3),
        ("block", Dim3),
        ("shared", ct.c_size_t),
        ("stream", ct.c_void_p),
        ("attributes", ct.c_void_p),
        ("attribute_count", ct.c_uint),
    ]


class FatbinWrapper(ct.Structure):
    _fields_ = [
        ("magic", ct.c_uint32), ("version", ct.c_uint32),
        ("data", ct.c_void_p), ("filename", ct.c_void_p),
    ]


def validate(build_dir: Path) -> None:
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    driver = ct.CDLL(str(build_dir / f"libcuda{suffix}"))
    pointer = ct.c_void_p
    output_pointer = ct.POINTER(pointer)
    load = bind(driver, "cuModuleLoadData", [output_pointer, pointer])
    load_file = bind(driver, "cuModuleLoad", [output_pointer, ct.c_char_p])
    unload = bind(driver, "cuModuleUnload", [pointer])
    get_function = bind(driver, "cuModuleGetFunction", [output_pointer, pointer, ct.c_char_p])

    # An overlapping LZ4 match expands '/' to eleven slashes, then literals
    # supply the newline and program. This is a valid PTX comment prefix.
    tail = b"\n" + PTX + b"\0"
    compressed = b"\x16/\x01\x00\xf0" + bytes([len(tail) - 15]) + tail
    compressed_image = fatbin(compressed, expanded=11 + len(tail))
    raw_image = fatbin(PTX + b"\0")
    zstd = ct.CDLL("libzstd.dylib" if sys.platform == "darwin" else "libzstd.so.1")
    bound = bind(zstd, "ZSTD_compressBound", [ct.c_size_t], ct.c_size_t)
    compress = bind(zstd, "ZSTD_compress", [pointer, ct.c_size_t, pointer, ct.c_size_t, ct.c_int], ct.c_size_t)
    is_error = bind(zstd, "ZSTD_isError", [ct.c_size_t], ct.c_uint)
    zstd_source = PTX + b"\0"
    buffer = ct.create_string_buffer(bound(len(zstd_source)))
    compressed_size = compress(buffer, len(buffer), zstd_source, len(zstd_source), 3)
    assert not is_error(compressed_size)
    zstd_payload = buffer.raw[:compressed_size]
    zstd_image = fatbin(zstd_payload, expanded=len(zstd_source), compression=0x8000)
    zstd_short_header = fatbin(zstd_payload, expanded=len(zstd_source), compression=0x8000, header_size=48)
    for image in (PTX, raw_image, compressed_image, zstd_image, zstd_short_header):
        module = pointer()
        assert load(ct.byref(module), image) == 0
        function = pointer()
        assert get_function(ct.byref(function), module, b"set_value") == 0
        assert unload(module) == 0

    storage = ct.create_string_buffer(compressed_image)
    wrapper = FatbinWrapper(0x466243B1, 1, ct.addressof(storage), None)
    module = pointer()
    assert load(ct.byref(module), ct.byref(wrapper)) == 0
    assert unload(module) == 0
    wrapper.version = 2
    assert load(ct.byref(module), ct.byref(wrapper)) == 801
    assert load(ct.byref(module), fatbin(b"\x7fELF", kind=2)) == 801
    assert load(ct.byref(module), b"\x7fELF") == 801

    malformed = [
        raw_image[:8],                           # incomplete fatbin header
        raw_image[:-1],                          # declared payload exceeds file
        struct.pack("<IHHQ", 0xBA55ED50, 1, 8, 0),
        struct.pack("<IHHQ", 0xBA55ED50, 1, 16, 8) + b"\0" * 8,
        fatbin(compressed, expanded=12 + len(tail)),  # incorrect expanded length
        fatbin(b"\x00\x00\x00", expanded=4),        # zero back-reference distance
        fatbin(b"invalid frame", expanded=1, compression=0x8000),
        fatbin(zstd_payload[:-1], expanded=len(zstd_source), compression=0x8000),
        fatbin(compressed, expanded=11 + len(tail), header_size=48),
    ]
    with tempfile.TemporaryDirectory(prefix="fakegpu-kernel-images-") as directory:
        path = Path(directory) / "module.fatbin"
        assert load_file(ct.byref(module), os.fsencode(path)) == 301
        for image in malformed:
            path.write_bytes(image)
            assert load_file(ct.byref(module), os.fsencode(path)) == 200
        path.write_bytes(compressed_image)
        assert load_file(ct.byref(module), os.fsencode(path)) == 0
        assert unload(module) == 0

    library_load = bind(driver, "cuLibraryLoadData", [output_pointer, pointer, pointer, pointer, ct.c_uint, pointer, pointer, ct.c_uint])
    library_kernel = bind(driver, "cuLibraryGetKernel", [output_pointer, pointer, ct.c_char_p])
    kernel_function = bind(driver, "cuKernelGetFunction", [output_pointer, pointer])
    library_unload = bind(driver, "cuLibraryUnload", [pointer])
    launch = bind(driver, "cuLaunchKernelEx", [ct.POINTER(DriverLaunchConfig), pointer, output_pointer, output_pointer])
    allocate = bind(driver, "cuMemAlloc_v2", [ct.POINTER(ct.c_uint64), ct.c_size_t])
    release = bind(driver, "cuMemFree_v2", [ct.c_uint64])
    copy = bind(driver, "cuMemcpyDtoH_v2", [pointer, ct.c_uint64, ct.c_size_t])
    library, kernel, function = pointer(), pointer(), pointer()
    assert library_load(ct.byref(library), PTX, None, None, 0, None, None, 0) == 0
    assert library_kernel(ct.byref(kernel), library, b"set_value") == 0
    assert kernel_function(ct.byref(function), kernel) == 0
    device = ct.c_uint64()
    assert allocate(ct.byref(device), 4) == 0
    arguments = (pointer * 1)(ct.addressof(device))
    config = DriverLaunchConfig((ct.c_uint * 3)(1, 1, 1), (ct.c_uint * 3)(1, 1, 1), 0, None, None, 0)
    assert launch(ct.byref(config), function, arguments, None) == 0
    result = ct.c_uint()
    assert copy(ct.byref(result), device, 4) == 0 and result.value == 42
    push_context = bind(driver, "cuCtxPushCurrent_v2", [pointer])
    pop_context = bind(driver, "cuCtxPopCurrent_v2", [output_pointer])
    context_device = bind(driver, "cuCtxGetDevice", [ct.POINTER(ct.c_int)])
    popped, other_module = pointer(), pointer()
    selected = ct.c_int()
    assert pop_context(ct.byref(popped)) == 201
    assert push_context(None) == 400
    assert push_context(pointer(3)) == 400
    assert push_context(pointer(2)) == 0
    assert context_device(ct.byref(selected)) == 0 and selected.value == 1
    assert load(ct.byref(other_module), PTX.replace(b"sm_80", b"sm_90")) == 0
    assert push_context(pointer(1)) == 0
    assert pop_context(ct.byref(popped)) == 0 and popped.value == 1
    assert context_device(ct.byref(selected)) == 0 and selected.value == 1
    assert pop_context(ct.byref(popped)) == 0 and popped.value == 2
    assert context_device(ct.byref(selected)) == 0 and selected.value == 0
    assert launch(ct.byref(config), function, arguments, None) == 0
    assert unload(other_module) == 0
    config.attribute_count = 1
    assert launch(ct.byref(config), function, arguments, None) == 801
    assert library_unload(library) == 0
    config.attribute_count = 0
    assert launch(ct.byref(config), function, arguments, None) == 400
    assert release(device) == 0

    runtime = ct.CDLL(str(build_dir / f"libcudart{suffix}"))
    register = bind(runtime, "__cudaRegisterFatBinary", [pointer], pointer)
    register_function = bind(runtime, "__cudaRegisterFunction", [pointer, pointer, ct.c_char_p, ct.c_char_p, ct.c_int] + [pointer] * 5, None)
    register_symbol = bind(runtime, "__cudaRegisterVar", [pointer, pointer, ct.c_char_p, ct.c_char_p, ct.c_int, ct.c_size_t, ct.c_int, ct.c_int], None)
    unregister = bind(runtime, "__cudaUnregisterFatBinary", [pointer], None)
    runtime_launch = bind(runtime, "cudaLaunchKernelExC", [ct.POINTER(RuntimeLaunchConfig), pointer, output_pointer])
    runtime_allocate = bind(runtime, "cudaMalloc", [output_pointer, ct.c_size_t])
    runtime_free = bind(runtime, "cudaFree", [pointer])
    runtime_copy = bind(runtime, "cudaMemcpy", [pointer, pointer, ct.c_size_t, ct.c_int])
    symbol_address = bind(runtime, "cudaGetSymbolAddress", [output_pointer, pointer])
    symbol_size = bind(runtime, "cudaGetSymbolSize", [ct.POINTER(ct.c_size_t), pointer])
    symbol_copy = bind(runtime, "cudaMemcpyToSymbol", [pointer, pointer, ct.c_size_t, ct.c_size_t, ct.c_int])
    wrapper.version = 1
    registered = register(ct.byref(wrapper))
    host_function = pointer(0x1000)
    register_function(registered, host_function, b"set_value", b"set_value", -1, None, None, None, None, None)
    host_symbol = pointer(0x2000)
    register_symbol(registered, host_symbol, b"global_value", b"global_value", 0, 4, 0, 1)
    symbol, size = pointer(), ct.c_size_t()
    assert symbol_address(ct.byref(symbol), host_symbol) == 0
    assert symbol_size(ct.byref(size), host_symbol) == 0 and size.value == 4
    assert runtime_copy(ct.byref(result), symbol, 4, 2) == 0 and result.value == 7
    assert symbol_copy(host_symbol, ct.byref(result), 4, 1, 1) == 1  # offset exceeds storage
    runtime_device = pointer()
    assert runtime_allocate(ct.byref(runtime_device), 4) == 0
    runtime_arguments = (pointer * 1)(ct.addressof(runtime_device))
    runtime_config = RuntimeLaunchConfig(Dim3(1, 1, 1), Dim3(1, 1, 1), 0, None, None, 0)
    assert runtime_launch(ct.byref(runtime_config), host_function, runtime_arguments) == 0
    assert runtime_copy(ct.byref(result), runtime_device, 4, 2) == 0 and result.value == 42
    reset_primary = bind(driver, "cuDevicePrimaryCtxReset", [ct.c_int])
    reset_device = bind(runtime, "cudaDeviceReset", [])
    replacement = ct.c_uint(11)
    assert symbol_copy(host_symbol, ct.byref(replacement), 4, 0, 1) == 0
    assert reset_primary(0) == 801 and reset_device() == 801
    assert runtime_copy(ct.byref(result), symbol, 4, 2) == 0 and result.value == 11
    assert runtime_copy(ct.byref(result), runtime_device, 4, 2) == 0 and result.value == 42
    assert runtime_launch(ct.byref(runtime_config), host_function, runtime_arguments) == 0
    runtime_config.attribute_count = 1
    assert runtime_launch(ct.byref(runtime_config), host_function, runtime_arguments) == 801
    unregister(registered)
    assert symbol_address(ct.byref(symbol), host_symbol) == 13
    assert symbol_size(ct.byref(size), host_symbol) == 13
    runtime_config.attribute_count = 0
    assert runtime_launch(ct.byref(runtime_config), host_function, runtime_arguments) == 98
    assert runtime_free(runtime_device) == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    args = parser.parse_args()
    os.environ["FAKEGPU_MODE"] = "simulate"
    os.environ["FAKEGPU_PROFILES"] = "a100,h100"
    os.environ["FAKEGPU_REPORT_PATH"] = os.devnull
    os.environ["FAKEGPU_TERMINAL_REPORT"] = "0"
    validate(args.build_dir.resolve())
    print("PTX images: fatbins, compressed PTX, malformed images and extended launches passed")


if __name__ == "__main__":
    main()
