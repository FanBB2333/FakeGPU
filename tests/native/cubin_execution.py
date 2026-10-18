#!/usr/bin/env python3
"""Compile and execute SASS-only CUDA images through the Driver API on CPU."""

from __future__ import annotations

import argparse
import ctypes as ct
import os
from pathlib import Path
import struct
import subprocess
import tempfile

from ptx_execution import Driver, check
from kernel_images import FatbinWrapper, bind, fatbin


def compiler_profiles(nvcc: str) -> list[tuple[int, str]]:
    targets = set(subprocess.check_output([nvcc, "--list-gpu-code"], text=True).split())
    profiles = [(80, "a100"), (90, "h100"), (75, "t4"), (86, "rtx3090"), (89, "l40s")]
    for architecture, profile in ((100, "b200"), (120, "rtx5090")):
        if f"sm_{architecture}" in targets:
            profiles.append((architecture, profile))
        else:
            print(f"SASS architecture sm_{architecture}: compiler target unavailable (skipped)", flush=True)
    return profiles


def compile_images(nvcc: str, directory: Path, architectures: list[int]) -> tuple[dict[int, Path], Path]:
    source = Path(__file__).with_name("cubin_kernels.cu")
    images = {architecture: directory / f"sm{architecture}.cubin" for architecture in architectures}
    for architecture, image in images.items():
        subprocess.run([nvcc, "-cubin", f"-arch=sm_{architecture}", str(source),
                        "-o", str(image)], check=True)
        assert image.read_bytes().startswith(b"\x7fELF")
    fatbin = directory / "multiarch.fatbin"
    command = [nvcc, "-fatbin"]
    for architecture in sorted(architectures, reverse=True):
        command.extend(["-gencode", f"arch=compute_{architecture},code=sm_{architecture}"])
    subprocess.run([*command, str(source), "-o", str(fatbin)], check=True)
    image = fatbin.read_bytes()
    offset, entries = 16, 0
    while offset < len(image):
        kind, _, header_size, payload_size = struct.unpack_from("<HHIQ", image, offset)
        assert kind == 2, "multiarchitecture validation requires SASS images without PTX"
        offset += header_size + payload_size
        entries += 1
    assert entries == len(images) and offset == len(image)
    return images, fatbin


def load_image(driver: Driver, image: Path, *, from_file: bool = False) -> ct.c_void_p:
    module = ct.c_void_p()
    if from_file:
        result = driver.api.cuModuleLoad(ct.byref(module), os.fsencode(image))
    else:
        result = driver.api.cuModuleLoadData(ct.byref(module), image.read_bytes())
    check(result, f"load {image.name}")
    driver.modules.append(module)
    return module


def function(driver: Driver, module: ct.c_void_p, name: str) -> ct.c_void_p:
    result = ct.c_void_p()
    check(driver.api.cuModuleGetFunction(ct.byref(result), module, name.encode()), name)
    return result


def test_compressed_cubin(driver: Driver, image: Path) -> None:
    raw = image.read_bytes()
    zstd = ct.CDLL("libzstd.so.1")
    bound = bind(zstd, "ZSTD_compressBound", [ct.c_size_t], ct.c_size_t)
    compress = bind(zstd, "ZSTD_compress", [ct.c_void_p, ct.c_size_t, ct.c_void_p,
                                           ct.c_size_t, ct.c_int], ct.c_size_t)
    is_error = bind(zstd, "ZSTD_isError", [ct.c_size_t], ct.c_uint)
    buffer = ct.create_string_buffer(bound(len(raw)))
    size = compress(buffer, len(buffer), raw, len(raw), 3)
    assert not is_error(size)
    packed = fatbin(buffer.raw[:size], kind=2, expanded=len(raw), compression=0x8000)
    storage = ct.create_string_buffer(packed)
    wrapper = FatbinWrapper(0x466243B1, 1, ct.addressof(storage), None)
    for image_data in (packed, ct.byref(wrapper)):
        module = ct.c_void_p()
        check(driver.api.cuModuleLoadData(ct.byref(module), image_data), "load compressed cubin")
        driver.modules.append(module)
        marker = function(driver, module, "compiled_architecture")
        output = driver.upload((ct.c_int * 1)(0))
        check(driver.launch(marker, [output]), "compressed cubin execution")
        assert driver.download(output, ct.c_int, 1) == [800]


def test_raw_cubin(driver: Driver, image: Path, *, from_file: bool) -> None:
    module = load_image(driver, image, from_file=from_file)
    affine = function(driver, module, "affine_values")
    values = [(i - 8) * 0.5 for i in range(19)]
    source = driver.upload((ct.c_float * 19)(*values))
    output = driver.upload((ct.c_float * 22)(*([-999] * 22)))
    check(driver.launch(affine, [output, source, ct.c_float(2.5), ct.c_int(19)],
                        grid=(3, 1, 1), block=(8, 1, 1)), "cubin vector arguments")
    assert driver.download(output, ct.c_float, 22) == [v * 2.5 + 3 for v in values] + [-999] * 3

    class Packed(ct.Structure):
        _fields_ = [("output", ct.c_uint64), ("input", ct.c_uint64),
                    ("scale", ct.c_float), ("count", ct.c_int)]

    packed = Packed(output.value, source.value, -1.25, 19)
    size = ct.c_size_t(ct.sizeof(packed))
    extra = (ct.c_void_p * 5)(1, ct.addressof(packed), 2, ct.addressof(size), 0)
    check(driver.api.cuLaunchKernel(affine, 3, 1, 1, 8, 1, 1, 0, None, None, extra),
          "cubin packed arguments")
    assert driver.download(output, ct.c_float, 22) == [v * -1.25 + 3 for v in values] + [-999] * 3
    size.value -= 1
    assert driver.api.cuLaunchKernel(affine, 3, 1, 1, 8, 1, 1, 0, None, None, extra) == 1

    counter, counter_size = driver.global_variable("counter")
    bias, bias_size = driver.global_variable("bias")
    assert counter_size == bias_size == 4
    assert driver.download(counter, ct.c_int, 1) == [7]
    assert driver.download(bias, ct.c_int, 1) == [3]
    advance = function(driver, module, "advance_state")
    state_output = driver.upload((ct.c_int * 1)(0))
    for delta, expected in ((4, 11), (-2, 9)):
        check(driver.launch(advance, [state_output, ct.c_int(delta)]), "cubin persistent global")
        assert driver.download(counter, ct.c_int, 1) == [expected]
        assert driver.download(state_output, ct.c_int, 1) == [expected]
    replacement = ct.c_int(20)
    check(driver.api.cuMemcpyHtoD_v2(counter, ct.byref(replacement), 4), "update cubin global")
    check(driver.launch(advance, [state_output, ct.c_int(-3)]), "cubin global after host write")
    assert driver.download(state_output, ct.c_int, 1) == [17]
    replacement.value = 5
    check(driver.api.cuMemcpyHtoD_v2(bias, ct.byref(replacement), 4), "update cubin constant")
    check(driver.launch(affine, [output, source, ct.c_float(1), ct.c_int(19)],
                        grid=(3, 1, 1), block=(8, 1, 1)), "cubin constant after host write")
    assert driver.download(output, ct.c_float, 22) == [v + 5 for v in values] + [-999] * 3

    exported = driver.upload((ct.c_uint64 * 1)(0))
    export_address = function(driver, module, "export_counter_address")
    check(driver.launch(export_address, [exported]), "export cubin global address")
    address = driver.download(exported, ct.c_uint64, 1)[0]
    assert address == counter.value
    assert driver.download(ct.c_uint64(address), ct.c_int, 1) == [17]
    replacement.value = 31
    check(driver.api.cuMemcpyHtoD_v2(address, ct.byref(replacement), 4), "write exported cubin pointer")
    assert driver.download(counter, ct.c_int, 1) == [31]
    initialized_pointer, pointer_size = driver.global_variable("counter_address")
    assert pointer_size == 8
    address = driver.download(initialized_pointer, ct.c_uint64, 1)[0]
    assert address == counter.value
    assert driver.download(ct.c_uint64(address), ct.c_int, 1) == [31]
    export_constant = function(driver, module, "export_constant_address")
    check(driver.launch(export_constant, [exported]), "export cubin constant address")
    address = driver.download(exported, ct.c_uint64, 1)[0]
    value = ct.c_int()
    assert driver.api.cuMemcpyDtoH_v2(ct.byref(value), address, 4) == 801

    fill16 = bind(driver.api, "cuMemsetD16Async", [ct.c_uint64, ct.c_uint16, ct.c_size_t, ct.c_void_p])
    fill32 = bind(driver.api, "cuMemsetD32_v2", [ct.c_uint64, ct.c_uint32, ct.c_size_t])
    check(fill16(counter, 0x1357, 2, None), "16-bit async fill of cubin global")
    assert driver.download(counter, ct.c_uint16, 2) == [0x1357, 0x1357]
    assert fill16(counter, 0, 1 << 20, None) == 700
    check(fill32(counter, 0x1234ABCD, 1), "32-bit fill of cubin global")
    runtime = ct.CDLL(str(Path(driver.api._name).with_name("libcudart.so.12")))
    copy_default = bind(runtime, "cudaMemcpy", [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int])
    check(copy_default(ct.byref(value), counter.value, 4, 4), "cudaMemcpyDefault of cubin global")
    assert value.value == 0x1234ABCD

    store = function(driver, module, "store_word")
    assert driver.launch(store, [bias]) == 719
    assert driver.download(bias, ct.c_int, 1) == [5]
    assert driver.launch(store, [ct.c_uint64(output.value + 22 * 4)]) == 700
    missing = ct.c_void_p()
    assert driver.api.cuModuleGetFunction(ct.byref(missing), module, b"absent_kernel") == 500
    driver.modules.remove(module)
    check(driver.api.cuModuleUnload(module), "unload cubin")
    assert driver.launch(advance, [state_output, ct.c_int(1)]) == 400
    assert driver.api.cuMemcpyDtoH_v2(ct.byref(value), counter, 4) == 700


def test_cublas_interoperability(driver: Driver, image: Path) -> None:
    module = load_image(driver, image)
    vector, size = driver.global_variable("blas_vector")
    initialized_pointer, pointer_size = driver.global_variable("blas_vector_address")
    scale, _ = driver.global_variable("blas_scale")
    device_result, _ = driver.global_variable("blas_result")
    assert size == 16 and pointer_size == 8
    address = driver.download(initialized_pointer, ct.c_uint64, 1)[0]
    assert address == vector.value
    blas = ct.CDLL(str(Path(driver.api._name).with_name("libcublas.so.12")))
    create = bind(blas, "cublasCreate_v2", [ct.POINTER(ct.c_void_p)])
    destroy = bind(blas, "cublasDestroy_v2", [ct.c_void_p])
    set_mode = bind(blas, "cublasSetPointerMode_v2", [ct.c_void_p, ct.c_int])
    dot = bind(blas, "cublasSdot_v2", [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_int,
                                      ct.c_void_p, ct.c_int, ct.c_void_p])
    axpy = bind(blas, "cublasSaxpy_v2", [ct.c_void_p, ct.c_int, ct.c_void_p,
                                       ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_int])
    handle = ct.c_void_p()
    check(create(ct.byref(handle)), "create cuBLAS handle")
    try:
        result = ct.c_float()
        check(dot(handle, 4, address, 1, vector.value, 1, ct.byref(result)),
              "cuBLAS dot reads cubin global and relocated pointer")
        assert result.value == 30
        check(set_mode(handle, 1), "cuBLAS device pointer mode")
        other = driver.upload((ct.c_float * 4)(10, 20, 30, 40))
        check(axpy(handle, 4, scale.value, address, 1, other.value, 1),
              "cuBLAS axpy reads cubin scalar and vector")
        assert driver.download(other, ct.c_float, 4) == [12, 24, 36, 48]
        check(axpy(handle, 4, scale.value, other.value, 1, vector.value, 1),
              "cuBLAS axpy writes cubin global")
        assert driver.download(vector, ct.c_float, 4) == [25, 50, 75, 100]
        check(dot(handle, 4, vector.value, 1, vector.value, 1, device_result.value),
              "cuBLAS dot writes cubin result")
        assert driver.download(device_result, ct.c_float, 1) == [18750]
        affine = function(driver, module, "affine_values")
        check(driver.launch(affine, [other, vector, ct.c_float(1), ct.c_int(4)],
                            block=(4, 1, 1)), "SASS reads cuBLAS global writes")
        assert driver.download(other, ct.c_float, 4) == [28, 53, 78, 103]
    finally:
        check(destroy(handle), "destroy cuBLAS handle")


def test_truncated_cubin(driver: Driver, image: Path) -> None:
    raw = image.read_bytes()
    table = struct.unpack_from("<Q", raw, 40)[0]
    entry_size, count = struct.unpack_from("<HH", raw, 58)
    oversized_section = bytearray(raw)
    for index in range(count):
        entry = table + index * entry_size
        kind = struct.unpack_from("<I", raw, entry + 4)[0]
        size = struct.unpack_from("<Q", raw, entry + 32)[0]
        if kind == 1 and size > 4:
            struct.pack_into("<Q", oversized_section, entry + 24, len(raw) - 4)
            break
    malformed = image.with_name("malformed.cubin")
    for payload in (raw[:63], raw[:table + entry_size - 1], oversized_section):
        malformed.write_bytes(payload)
        module = ct.c_void_p()
        assert driver.api.cuModuleLoad(ct.byref(module), os.fsencode(malformed)) == 200


def test_architecture_selection(driver: Driver, images: dict[int, Path], fatbin: Path) -> None:
    driver.api.cuDevicePrimaryCtxRetain.argtypes = [ct.POINTER(ct.c_void_p), ct.c_int]
    driver.api.cuCtxSetCurrent.argtypes = [ct.c_void_p]
    contexts = []
    for device in range(len(images)):
        context = ct.c_void_p()
        check(driver.api.cuDevicePrimaryCtxRetain(ct.byref(context), device), "retain synthetic context")
        contexts.append(context)
    check(driver.api.cuCtxSetCurrent(contexts[0]), "select Ampere profile")
    module = ct.c_void_p()
    assert driver.api.cuModuleLoadData(ct.byref(module), images[90].read_bytes()) == 209
    assert driver.api.cuModuleLoad(ct.byref(module), os.fsencode(images[90])) == 209
    for device, architecture in enumerate(images):
        check(driver.api.cuCtxSetCurrent(contexts[device]), "select image architecture")
        values = [-2, -0.25, 0.5, 3.75]
        source = driver.upload((ct.c_float * 4)(*values))
        for image in (images[architecture], fatbin):
            module = load_image(driver, image)
            marker = function(driver, module, "compiled_architecture")
            output = driver.upload((ct.c_int * 1)(0))
            check(driver.launch(marker, [output]), "multiarchitecture cubin selection")
            assert driver.download(output, ct.c_int, 1) == [architecture * 10]
            affine = function(driver, module, "affine_values")
            output = driver.upload((ct.c_float * 6)(*([-999] * 6)))
            check(driver.launch(affine, [output, source, ct.c_float(1.5), ct.c_int(4)],
                                block=(8, 1, 1)), f"sm_{architecture} vector numerics")
            assert driver.download(output, ct.c_float, 6) == [v * 1.5 + 3 for v in values] + [-999] * 2
        print(f"SASS architecture sm_{architecture}: raw cubin and fatbin numerics passed", flush=True)
    check(driver.api.cuCtxSetCurrent(contexts[0]), "restore first device")


def test_ptx_child_launch(driver: Driver, nvcc: str, directory: Path) -> None:
    source = Path(__file__).with_name("cuda_child_launch.cu")
    ptx = directory / "child.ptx"
    subprocess.run([nvcc, "-ptx", "-rdc=true", "-arch=sm_80", str(source),
                    "-o", str(ptx)], check=True)
    module = load_image(driver, ptx)
    parent = function(driver, module, "launch_branch")
    output = driver.upload((ct.c_int * 16)(*([0] * 16)))
    status = driver.upload((ct.c_int * 2)(-1, -1))
    check(driver.launch(parent, [output, status]), "PTX parent launches child and grandchild")
    assert driver.download(status, ct.c_int, 2) == [0, 0]
    assert driver.download(output, ct.c_int, 16) == [(i + 1) * 7 for i in range(16)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--nvcc", default="nvcc")
    args = parser.parse_args()
    profiles = compiler_profiles(args.nvcc)
    os.environ["FAKEGPU_MODE"] = "simulate"
    os.environ["FAKEGPU_PROFILES"] = ",".join(profile for _, profile in profiles)
    os.environ["FAKEGPU_UNSUPPORTED_API"] = "error"
    os.environ["FAKEGPU_TERMINAL_REPORT"] = "0"
    os.environ["FAKEGPU_REPORT_PATH"] = os.devnull
    with tempfile.TemporaryDirectory(prefix="fakegpu-cubin-") as directory:
        images, fatbin = compile_images(args.nvcc, Path(directory), [architecture for architecture, _ in profiles])
        sm80 = images[80]
        library = args.build_dir.resolve() / "libcuda.so.1"
        # Match the symbol visibility of FakeGPU's native preloader.
        ct.CDLL(str(library), mode=ct.RTLD_GLOBAL)
        driver = Driver(library)
        try:
            test_truncated_cubin(driver, sm80)
            test_compressed_cubin(driver, sm80)
            test_raw_cubin(driver, sm80, from_file=True)
            test_raw_cubin(driver, sm80, from_file=False)
            test_cublas_interoperability(driver, sm80)
            test_architecture_selection(driver, images, fatbin)
            test_ptx_child_launch(driver, args.nvcc, Path(directory))
        finally:
            driver.close()
    print("CUDA images: SASS numerics, globals, cuBLAS interoperability, architecture selection and PTX child launches passed")


if __name__ == "__main__":
    main()
