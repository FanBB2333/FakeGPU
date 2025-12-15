#!/usr/bin/env python3
"""Test basic CUDA Runtime API functionality"""

import ctypes
import os

print("=== Testing FakeGPU CUDA Runtime API ===\n")

# Load fake cudart library
cudart = ctypes.CDLL('./build/libcudart.so.12', mode=ctypes.RTLD_GLOBAL)

# Define function prototypes
cudart.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
cudart.cudaGetDeviceCount.restype = ctypes.c_int

cudart.cudaSetDevice.argtypes = [ctypes.c_int]
cudart.cudaSetDevice.restype = ctypes.c_int

cudart.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
cudart.cudaGetDevice.restype = ctypes.c_int

cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
cudart.cudaMalloc.restype = ctypes.c_int

cudart.cudaFree.argtypes = [ctypes.c_void_p]
cudart.cudaFree.restype = ctypes.c_int

cudart.cudaGetLastError.argtypes = []
cudart.cudaGetLastError.restype = ctypes.c_int

# Test 1: Get device count
print("1. cudaGetDeviceCount...")
count = ctypes.c_int()
result = cudart.cudaGetDeviceCount(ctypes.byref(count))
print(f"   Result: {result}, Device count: {count.value}")

# Test 2: Set device
print("\n2. cudaSetDevice(0)...")
result = cudart.cudaSetDevice(0)
print(f"   Result: {result}")

# Test 3: Get device
print("\n3. cudaGetDevice...")
device = ctypes.c_int()
result = cudart.cudaGetDevice(ctypes.byref(device))
print(f"   Result: {result}, Current device: {device.value}")

# Test 4: Allocate memory
print("\n4. cudaMalloc (1MB)...")
devPtr = ctypes.c_void_p()
result = cudart.cudaMalloc(ctypes.byref(devPtr), 1024 * 1024)
print(f"   Result: {result}, Pointer: {hex(devPtr.value) if devPtr.value else 'NULL'}")

# Test 5: Free memory
print("\n5. cudaFree...")
result = cudart.cudaFree(devPtr)
print(f"   Result: {result}")

# Test 6: Get last error
print("\n6. cudaGetLastError...")
error = cudart.cudaGetLastError()
print(f"   Last error: {error}")

print("\n=== All tests completed ===")
