import ctypes
import os

print("=== Direct test: FakeGPU CUDA Driver API ===")

# Load FakeGPU CUDA library
fake_cuda = ctypes.CDLL('./build/libcuda.so.1', mode=ctypes.RTLD_GLOBAL)

# Define function prototypes
fake_cuda.cuInit.argtypes = [ctypes.c_uint]
fake_cuda.cuInit.restype = ctypes.c_int

fake_cuda.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
fake_cuda.cuDriverGetVersion.restype = ctypes.c_int

fake_cuda.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
fake_cuda.cuDeviceGetCount.restype = ctypes.c_int

fake_cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
fake_cuda.cuDeviceGet.restype = ctypes.c_int

fake_cuda.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
fake_cuda.cuDeviceGetName.restype = ctypes.c_int

fake_cuda.cuDeviceTotalMem_v2.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
fake_cuda.cuDeviceTotalMem_v2.restype = ctypes.c_int

fake_cuda.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
fake_cuda.cuDevicePrimaryCtxRetain.restype = ctypes.c_int

fake_cuda.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
fake_cuda.cuCtxSetCurrent.restype = ctypes.c_int

fake_cuda.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_size_t]
fake_cuda.cuMemAlloc_v2.restype = ctypes.c_int

fake_cuda.cuMemFree_v2.argtypes = [ctypes.c_ulonglong]
fake_cuda.cuMemFree_v2.restype = ctypes.c_int

fake_cuda.cuMemGetInfo_v2.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
fake_cuda.cuMemGetInfo_v2.restype = ctypes.c_int

# Run tests
print("\n1. cuInit...")
result = fake_cuda.cuInit(0)
print(f"   Result: {result}")

print("\n2. cuDriverGetVersion...")
version = ctypes.c_int()
result = fake_cuda.cuDriverGetVersion(ctypes.byref(version))
print(f"   Result: {result}, version: {version.value}")

print("\n3. cuDeviceGetCount...")
count = ctypes.c_int()
result = fake_cuda.cuDeviceGetCount(ctypes.byref(count))
print(f"   Result: {result}, device_count: {count.value}")

print("\n4. cuDeviceGet...")
device = ctypes.c_int()
result = fake_cuda.cuDeviceGet(ctypes.byref(device), 0)
print(f"   Result: {result}, device: {device.value}")

print("\n5. cuDeviceGetName...")
name = ctypes.create_string_buffer(256)
result = fake_cuda.cuDeviceGetName(name, 256, 0)
print(f"   Result: {result}, name: {name.value.decode()}")

print("\n6. cuDeviceTotalMem_v2...")
total_mem = ctypes.c_size_t()
result = fake_cuda.cuDeviceTotalMem_v2(ctypes.byref(total_mem), 0)
print(f"   Result: {result}, total_memory: {total_mem.value / 1024**3:.2f} GB")

print("\n7. cuDevicePrimaryCtxRetain...")
ctx = ctypes.c_void_p()
result = fake_cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), 0)
print(f"   Result: {result}, ctx: {ctx.value}")

print("\n8. cuCtxSetCurrent...")
result = fake_cuda.cuCtxSetCurrent(ctx)
print(f"   Result: {result}")

print("\n9. cuMemGetInfo_v2...")
free_mem = ctypes.c_size_t()
total_mem = ctypes.c_size_t()
result = fake_cuda.cuMemGetInfo_v2(ctypes.byref(free_mem), ctypes.byref(total_mem))
print(f"   Result: {result}, free: {free_mem.value / 1024**3:.2f} GB, total: {total_mem.value / 1024**3:.2f} GB")

print("\n10. cuMemAlloc_v2...")
dptr = ctypes.c_ulonglong()
result = fake_cuda.cuMemAlloc_v2(ctypes.byref(dptr), 1024 * 1024)  # 1MB
print(f"   Result: {result}, ptr: {hex(dptr.value)}")

print("\n11. cuMemGetInfo_v2 (after allocation)...")
result = fake_cuda.cuMemGetInfo_v2(ctypes.byref(free_mem), ctypes.byref(total_mem))
print(f"   Result: {result}, free: {free_mem.value / 1024**3:.2f} GB, total: {total_mem.value / 1024**3:.2f} GB")

print("\n12. cuMemFree_v2...")
result = fake_cuda.cuMemFree_v2(dptr)
print(f"   Result: {result}")

print("\n=== All tests completed ===")
