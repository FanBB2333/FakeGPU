#!/usr/bin/env python3

import ctypes
import pathlib


CUDA_SUCCESS = 0
CUDA_ERROR_INVALID_VALUE = 1
CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2
CUDA_STREAM_NON_BLOCKING = 1


ROOT = pathlib.Path(__file__).resolve().parents[1]
LIBCUDART = ROOT / "build" / "libcudart.so.12"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    cudart = ctypes.CDLL(str(LIBCUDART), mode=ctypes.RTLD_GLOBAL)

    cudart.cudaStreamCreateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    cudart.cudaStreamCreateWithFlags.restype = ctypes.c_int

    cudart.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
    cudart.cudaStreamDestroy.restype = ctypes.c_int

    cudart.cudaStreamGetPriority.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    cudart.cudaStreamGetPriority.restype = ctypes.c_int

    cudart.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cudart.cudaGetDeviceCount.restype = ctypes.c_int

    cudart.cudaSetDevice.argtypes = [ctypes.c_int]
    cudart.cudaSetDevice.restype = ctypes.c_int

    cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cudart.cudaMalloc.restype = ctypes.c_int

    cudart.cudaMallocAsync.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_void_p]
    cudart.cudaMallocAsync.restype = ctypes.c_int

    cudart.cudaFree.argtypes = [ctypes.c_void_p]
    cudart.cudaFree.restype = ctypes.c_int

    cudart.cudaFreeAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    cudart.cudaFreeAsync.restype = ctypes.c_int

    cudart.cudaMemsetAsync.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p]
    cudart.cudaMemsetAsync.restype = ctypes.c_int

    cudart.cudaMemcpyAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    cudart.cudaMemcpyAsync.restype = ctypes.c_int

    HOST_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    cudart.cudaLaunchHostFunc.argtypes = [ctypes.c_void_p, HOST_FUNC, ctypes.c_void_p]
    cudart.cudaLaunchHostFunc.restype = ctypes.c_int

    callback_hits = []

    @HOST_FUNC
    def host_func(user_data):
        callback_hits.append(ctypes.cast(user_data, ctypes.c_void_p).value)

    device_count = ctypes.c_int()
    require(cudart.cudaGetDeviceCount(ctypes.byref(device_count)) == CUDA_SUCCESS, "cudaGetDeviceCount should succeed")
    require(device_count.value > 0, "at least one fake device should be visible")
    require(cudart.cudaSetDevice(0) == CUDA_SUCCESS, "cudaSetDevice(0) should succeed")

    stream = ctypes.c_void_p()
    require(
        cudart.cudaStreamCreateWithFlags(ctypes.byref(stream), CUDA_STREAM_NON_BLOCKING) == CUDA_SUCCESS,
        "cudaStreamCreateWithFlags should succeed",
    )
    require(stream.value not in (None, 0), "created stream should be non-null")

    priority = ctypes.c_int()
    require(
        cudart.cudaStreamGetPriority(stream, ctypes.byref(priority)) == CUDA_SUCCESS,
        "cudaStreamGetPriority should succeed for a live stream",
    )

    device_ptr = ctypes.c_void_p()
    require(cudart.cudaMalloc(ctypes.byref(device_ptr), 16) == CUDA_SUCCESS, "cudaMalloc should succeed")

    async_ptr = ctypes.c_void_p()
    require(
        cudart.cudaMallocAsync(ctypes.byref(async_ptr), 16, stream) == CUDA_SUCCESS,
        "cudaMallocAsync should accept a live stream",
    )
    require(cudart.cudaMemsetAsync(device_ptr, 7, 16, stream) == CUDA_SUCCESS, "cudaMemsetAsync should accept a live stream")

    host_src = (ctypes.c_ubyte * 16)(*range(16))
    host_dst = (ctypes.c_ubyte * 16)()
    require(
        cudart.cudaMemcpyAsync(device_ptr, ctypes.cast(host_src, ctypes.c_void_p), 16, CUDA_MEMCPY_HOST_TO_DEVICE, stream)
        == CUDA_SUCCESS,
        "cudaMemcpyAsync H2D should accept a live stream",
    )
    require(
        cudart.cudaMemcpyAsync(ctypes.cast(host_dst, ctypes.c_void_p), device_ptr, 16, CUDA_MEMCPY_DEVICE_TO_HOST, stream)
        == CUDA_SUCCESS,
        "cudaMemcpyAsync D2H should accept a live stream",
    )
    require(list(host_dst) == list(range(16)), "cudaMemcpyAsync should copy bytes correctly")

    user_data = ctypes.c_void_p(1234)
    require(
        cudart.cudaLaunchHostFunc(stream, host_func, user_data) == CUDA_SUCCESS,
        "cudaLaunchHostFunc should accept a live stream",
    )
    require(callback_hits == [1234], "cudaLaunchHostFunc should execute the callback for a live stream")

    require(cudart.cudaStreamDestroy(stream) == CUDA_SUCCESS, "cudaStreamDestroy should succeed")

    invalid_async_ptr = ctypes.c_void_p()
    require(
        cudart.cudaMallocAsync(ctypes.byref(invalid_async_ptr), 16, stream) == CUDA_ERROR_INVALID_VALUE,
        "cudaMallocAsync should reject a destroyed stream",
    )
    require(
        cudart.cudaMemsetAsync(device_ptr, 1, 16, stream) == CUDA_ERROR_INVALID_VALUE,
        "cudaMemsetAsync should reject a destroyed stream",
    )
    require(
        cudart.cudaMemcpyAsync(device_ptr, ctypes.cast(host_src, ctypes.c_void_p), 16, CUDA_MEMCPY_HOST_TO_DEVICE, stream)
        == CUDA_ERROR_INVALID_VALUE,
        "cudaMemcpyAsync should reject a destroyed stream",
    )
    require(
        cudart.cudaLaunchHostFunc(stream, host_func, user_data) == CUDA_ERROR_INVALID_VALUE,
        "cudaLaunchHostFunc should reject a destroyed stream",
    )
    require(callback_hits == [1234], "destroyed-stream host launch should not execute the callback")
    require(
        cudart.cudaStreamGetPriority(stream, ctypes.byref(priority)) == CUDA_ERROR_INVALID_VALUE,
        "cudaStreamGetPriority should reject a destroyed stream",
    )
    require(
        cudart.cudaFreeAsync(async_ptr, stream) == CUDA_ERROR_INVALID_VALUE,
        "cudaFreeAsync should reject a destroyed stream",
    )

    require(cudart.cudaFree(async_ptr) == CUDA_SUCCESS, "cudaFree should clean up async allocation")
    require(cudart.cudaFree(device_ptr) == CUDA_SUCCESS, "cudaFree should succeed")
    print("cudart stream lifecycle test passed")


if __name__ == "__main__":
    main()
