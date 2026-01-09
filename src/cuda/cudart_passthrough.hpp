#pragma once

#include "cudart_defs.hpp"
#include "../core/backend_config.hpp"
#include "../core/real_cuda_loader.hpp"
#include "../core/logging.hpp"

namespace fake_gpu {

// Function pointer types for CUDA Runtime API
typedef cudaError_t (*cudaGetDeviceCount_fn)(int*);
typedef cudaError_t (*cudaSetDevice_fn)(int);
typedef cudaError_t (*cudaGetDevice_fn)(int*);
typedef cudaError_t (*cudaGetDeviceProperties_fn)(cudaDeviceProp*, int);
typedef cudaError_t (*cudaDeviceSynchronize_fn)(void);
typedef cudaError_t (*cudaDeviceReset_fn)(void);
typedef cudaError_t (*cudaMalloc_fn)(void**, size_t);
typedef cudaError_t (*cudaFree_fn)(void*);
typedef cudaError_t (*cudaMemcpy_fn)(void*, const void*, size_t, cudaMemcpyKind);
typedef cudaError_t (*cudaMemcpyAsync_fn)(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);
typedef cudaError_t (*cudaMemset_fn)(void*, int, size_t);
typedef cudaError_t (*cudaMemsetAsync_fn)(void*, int, size_t, cudaStream_t);
typedef cudaError_t (*cudaMemGetInfo_fn)(size_t*, size_t*);
typedef cudaError_t (*cudaMallocManaged_fn)(void**, size_t, unsigned int);
typedef cudaError_t (*cudaMallocHost_fn)(void**, size_t);
typedef cudaError_t (*cudaFreeHost_fn)(void*);
typedef cudaError_t (*cudaHostAlloc_fn)(void**, size_t, unsigned int);
typedef cudaError_t (*cudaStreamCreate_fn)(cudaStream_t*);
typedef cudaError_t (*cudaStreamDestroy_fn)(cudaStream_t);
typedef cudaError_t (*cudaStreamSynchronize_fn)(cudaStream_t);
typedef cudaError_t (*cudaEventCreate_fn)(cudaEvent_t*);
typedef cudaError_t (*cudaEventDestroy_fn)(cudaEvent_t);
typedef cudaError_t (*cudaEventRecord_fn)(cudaEvent_t, cudaStream_t);
typedef cudaError_t (*cudaEventSynchronize_fn)(cudaEvent_t);
typedef cudaError_t (*cudaEventElapsedTime_fn)(float*, cudaEvent_t, cudaEvent_t);
typedef cudaError_t (*cudaLaunchKernel_fn)(const void*, dim3, dim3, void**, size_t, cudaStream_t);
typedef cudaError_t (*cudaRuntimeGetVersion_fn)(int*);
typedef cudaError_t (*cudaDriverGetVersion_fn)(int*);
typedef const char* (*cudaGetErrorString_fn)(cudaError_t);
typedef const char* (*cudaGetErrorName_fn)(cudaError_t);
typedef cudaError_t (*cudaGetLastError_fn)(void);
typedef cudaError_t (*cudaPeekAtLastError_fn)(void);
typedef cudaError_t (*cudaPointerGetAttributes_fn)(cudaPointerAttributes*, const void*);

// Passthrough dispatcher for CUDA Runtime API
class CudaRuntimePassthrough {
public:
    static CudaRuntimePassthrough& instance() {
        static CudaRuntimePassthrough pt;
        return pt;
    }

    bool is_available() const { return available_; }

    bool initialize() {
        if (initialized_) return available_;
        initialized_ = true;

        RealCudaLoader& loader = RealCudaLoader::instance();
        if (!loader.initialize() || !loader.has_cudart()) {
            FGPU_LOG("[CudaRuntimePassthrough] Real CUDA runtime not available\n");
            return false;
        }

        #define LOAD_FUNC(name) \
            real_##name = (name##_fn)loader.get_cudart_func(#name)

        LOAD_FUNC(cudaGetDeviceCount);
        LOAD_FUNC(cudaSetDevice);
        LOAD_FUNC(cudaGetDevice);
        LOAD_FUNC(cudaGetDeviceProperties);
        LOAD_FUNC(cudaDeviceSynchronize);
        LOAD_FUNC(cudaDeviceReset);
        LOAD_FUNC(cudaMalloc);
        LOAD_FUNC(cudaFree);
        LOAD_FUNC(cudaMemcpy);
        LOAD_FUNC(cudaMemcpyAsync);
        LOAD_FUNC(cudaMemset);
        LOAD_FUNC(cudaMemsetAsync);
        LOAD_FUNC(cudaMemGetInfo);
        LOAD_FUNC(cudaMallocManaged);
        LOAD_FUNC(cudaMallocHost);
        LOAD_FUNC(cudaFreeHost);
        LOAD_FUNC(cudaHostAlloc);
        LOAD_FUNC(cudaStreamCreate);
        LOAD_FUNC(cudaStreamDestroy);
        LOAD_FUNC(cudaStreamSynchronize);
        LOAD_FUNC(cudaEventCreate);
        LOAD_FUNC(cudaEventDestroy);
        LOAD_FUNC(cudaEventRecord);
        LOAD_FUNC(cudaEventSynchronize);
        LOAD_FUNC(cudaEventElapsedTime);
        LOAD_FUNC(cudaLaunchKernel);
        LOAD_FUNC(cudaRuntimeGetVersion);
        LOAD_FUNC(cudaDriverGetVersion);
        LOAD_FUNC(cudaGetErrorString);
        LOAD_FUNC(cudaGetErrorName);
        LOAD_FUNC(cudaGetLastError);
        LOAD_FUNC(cudaPeekAtLastError);
        LOAD_FUNC(cudaPointerGetAttributes);

        #undef LOAD_FUNC

        available_ = (real_cudaGetDeviceCount != nullptr);
        if (available_) {
            FGPU_LOG("[CudaRuntimePassthrough] Initialized successfully\n");
        }
        return available_;
    }

    // Passthrough functions
    cudaError_t cudaGetDeviceCount(int* count) {
        if (!real_cudaGetDeviceCount) return cudaErrorInitializationError;
        return real_cudaGetDeviceCount(count);
    }

    cudaError_t cudaSetDevice(int device) {
        if (!real_cudaSetDevice) return cudaErrorInitializationError;
        return real_cudaSetDevice(device);
    }

    cudaError_t cudaGetDevice(int* device) {
        if (!real_cudaGetDevice) return cudaErrorInitializationError;
        return real_cudaGetDevice(device);
    }

    cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device) {
        if (!real_cudaGetDeviceProperties) return cudaErrorInitializationError;
        return real_cudaGetDeviceProperties(prop, device);
    }

    cudaError_t cudaDeviceSynchronize() {
        if (!real_cudaDeviceSynchronize) return cudaErrorInitializationError;
        return real_cudaDeviceSynchronize();
    }

    cudaError_t cudaDeviceReset() {
        if (!real_cudaDeviceReset) return cudaErrorInitializationError;
        return real_cudaDeviceReset();
    }

    cudaError_t cudaMalloc(void** devPtr, size_t size) {
        if (!real_cudaMalloc) return cudaErrorInitializationError;
        return real_cudaMalloc(devPtr, size);
    }

    cudaError_t cudaFree(void* devPtr) {
        if (!real_cudaFree) return cudaErrorInitializationError;
        return real_cudaFree(devPtr);
    }

    cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
        if (!real_cudaMemcpy) return cudaErrorInitializationError;
        return real_cudaMemcpy(dst, src, count, kind);
    }

    cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
        if (!real_cudaMemcpyAsync) return cudaErrorInitializationError;
        return real_cudaMemcpyAsync(dst, src, count, kind, stream);
    }

    cudaError_t cudaMemset(void* devPtr, int value, size_t count) {
        if (!real_cudaMemset) return cudaErrorInitializationError;
        return real_cudaMemset(devPtr, value, count);
    }

    cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream) {
        if (!real_cudaMemsetAsync) return cudaErrorInitializationError;
        return real_cudaMemsetAsync(devPtr, value, count, stream);
    }

    cudaError_t cudaMemGetInfo(size_t* free, size_t* total) {
        if (!real_cudaMemGetInfo) return cudaErrorInitializationError;
        return real_cudaMemGetInfo(free, total);
    }

    cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags) {
        if (!real_cudaMallocManaged) return cudaErrorInitializationError;
        return real_cudaMallocManaged(devPtr, size, flags);
    }

    cudaError_t cudaMallocHost(void** ptr, size_t size) {
        if (!real_cudaMallocHost) return cudaErrorInitializationError;
        return real_cudaMallocHost(ptr, size);
    }

    cudaError_t cudaFreeHost(void* ptr) {
        if (!real_cudaFreeHost) return cudaErrorInitializationError;
        return real_cudaFreeHost(ptr);
    }

    cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags) {
        if (!real_cudaHostAlloc) return cudaErrorInitializationError;
        return real_cudaHostAlloc(pHost, size, flags);
    }

    cudaError_t cudaStreamCreate(cudaStream_t* pStream) {
        if (!real_cudaStreamCreate) return cudaErrorInitializationError;
        return real_cudaStreamCreate(pStream);
    }

    cudaError_t cudaStreamDestroy(cudaStream_t stream) {
        if (!real_cudaStreamDestroy) return cudaErrorInitializationError;
        return real_cudaStreamDestroy(stream);
    }

    cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
        if (!real_cudaStreamSynchronize) return cudaErrorInitializationError;
        return real_cudaStreamSynchronize(stream);
    }

    cudaError_t cudaEventCreate(cudaEvent_t* event) {
        if (!real_cudaEventCreate) return cudaErrorInitializationError;
        return real_cudaEventCreate(event);
    }

    cudaError_t cudaEventDestroy(cudaEvent_t event) {
        if (!real_cudaEventDestroy) return cudaErrorInitializationError;
        return real_cudaEventDestroy(event);
    }

    cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
        if (!real_cudaEventRecord) return cudaErrorInitializationError;
        return real_cudaEventRecord(event, stream);
    }

    cudaError_t cudaEventSynchronize(cudaEvent_t event) {
        if (!real_cudaEventSynchronize) return cudaErrorInitializationError;
        return real_cudaEventSynchronize(event);
    }

    cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) {
        if (!real_cudaEventElapsedTime) return cudaErrorInitializationError;
        return real_cudaEventElapsedTime(ms, start, end);
    }

    cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream) {
        if (!real_cudaLaunchKernel) return cudaErrorInitializationError;
        return real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
    }

    cudaError_t cudaRuntimeGetVersion(int* runtimeVersion) {
        if (!real_cudaRuntimeGetVersion) return cudaErrorInitializationError;
        return real_cudaRuntimeGetVersion(runtimeVersion);
    }

    cudaError_t cudaDriverGetVersion(int* driverVersion) {
        if (!real_cudaDriverGetVersion) return cudaErrorInitializationError;
        return real_cudaDriverGetVersion(driverVersion);
    }

    const char* cudaGetErrorString(cudaError_t error) {
        if (!real_cudaGetErrorString) return "unknown error";
        return real_cudaGetErrorString(error);
    }

    const char* cudaGetErrorName(cudaError_t error) {
        if (!real_cudaGetErrorName) return "cudaErrorUnknown";
        return real_cudaGetErrorName(error);
    }

    cudaError_t cudaGetLastError() {
        if (!real_cudaGetLastError) return cudaSuccess;
        return real_cudaGetLastError();
    }

    cudaError_t cudaPeekAtLastError() {
        if (!real_cudaPeekAtLastError) return cudaSuccess;
        return real_cudaPeekAtLastError();
    }

    cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* attributes, const void* ptr) {
        if (!real_cudaPointerGetAttributes) return cudaErrorInitializationError;
        return real_cudaPointerGetAttributes(attributes, ptr);
    }

    // Get any function by name
    void* getRealFunction(const char* name) {
        return RealCudaLoader::instance().get_cudart_func(name);
    }

private:
    CudaRuntimePassthrough() = default;

    bool initialized_ = false;
    bool available_ = false;

    // Cached function pointers
    cudaGetDeviceCount_fn real_cudaGetDeviceCount = nullptr;
    cudaSetDevice_fn real_cudaSetDevice = nullptr;
    cudaGetDevice_fn real_cudaGetDevice = nullptr;
    cudaGetDeviceProperties_fn real_cudaGetDeviceProperties = nullptr;
    cudaDeviceSynchronize_fn real_cudaDeviceSynchronize = nullptr;
    cudaDeviceReset_fn real_cudaDeviceReset = nullptr;
    cudaMalloc_fn real_cudaMalloc = nullptr;
    cudaFree_fn real_cudaFree = nullptr;
    cudaMemcpy_fn real_cudaMemcpy = nullptr;
    cudaMemcpyAsync_fn real_cudaMemcpyAsync = nullptr;
    cudaMemset_fn real_cudaMemset = nullptr;
    cudaMemsetAsync_fn real_cudaMemsetAsync = nullptr;
    cudaMemGetInfo_fn real_cudaMemGetInfo = nullptr;
    cudaMallocManaged_fn real_cudaMallocManaged = nullptr;
    cudaMallocHost_fn real_cudaMallocHost = nullptr;
    cudaFreeHost_fn real_cudaFreeHost = nullptr;
    cudaHostAlloc_fn real_cudaHostAlloc = nullptr;
    cudaStreamCreate_fn real_cudaStreamCreate = nullptr;
    cudaStreamDestroy_fn real_cudaStreamDestroy = nullptr;
    cudaStreamSynchronize_fn real_cudaStreamSynchronize = nullptr;
    cudaEventCreate_fn real_cudaEventCreate = nullptr;
    cudaEventDestroy_fn real_cudaEventDestroy = nullptr;
    cudaEventRecord_fn real_cudaEventRecord = nullptr;
    cudaEventSynchronize_fn real_cudaEventSynchronize = nullptr;
    cudaEventElapsedTime_fn real_cudaEventElapsedTime = nullptr;
    cudaLaunchKernel_fn real_cudaLaunchKernel = nullptr;
    cudaRuntimeGetVersion_fn real_cudaRuntimeGetVersion = nullptr;
    cudaDriverGetVersion_fn real_cudaDriverGetVersion = nullptr;
    cudaGetErrorString_fn real_cudaGetErrorString = nullptr;
    cudaGetErrorName_fn real_cudaGetErrorName = nullptr;
    cudaGetLastError_fn real_cudaGetLastError = nullptr;
    cudaPeekAtLastError_fn real_cudaPeekAtLastError = nullptr;
    cudaPointerGetAttributes_fn real_cudaPointerGetAttributes = nullptr;
};

} // namespace fake_gpu
