#include "cudart_defs.hpp"
#include "cuda_driver_defs.hpp"
#include <cstdio>
#include <cstring>
#include <cstdlib>

// Thread-local error tracking
static __thread cudaError_t last_error = cudaSuccess;

// Helper: Convert CUresult to cudaError_t
static cudaError_t convertDriverError(CUresult result) {
    switch (result) {
        case CUDA_SUCCESS:
            return cudaSuccess;
        case CUDA_ERROR_INVALID_VALUE:
            return cudaErrorInvalidValue;
        case CUDA_ERROR_OUT_OF_MEMORY:
            return cudaErrorMemoryAllocation;
        case CUDA_ERROR_INVALID_DEVICE:
            return cudaErrorInvalidDevice;
        case CUDA_ERROR_NOT_INITIALIZED:
            return cudaErrorInitializationError;
        default:
            return cudaErrorUnknown;
    }
}

extern "C" {

// ============================================================================
// Device Management
// ============================================================================

cudaError_t cudaGetDeviceCount(int *count) {
    if (!count) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    CUresult result = cuDeviceGetCount(count);
    last_error = convertDriverError(result);
    printf("[FakeCUDART] cudaGetDeviceCount returning %d\n", *count);
    return last_error;
}

cudaError_t cudaSetDevice(int device) {
    // Get device handle
    CUdevice cuDevice;
    CUresult result = cuDeviceGet(&cuDevice, device);
    if (result != CUDA_SUCCESS) {
        last_error = convertDriverError(result);
        return last_error;
    }

    // Retain primary context
    CUcontext context;
    result = cuDevicePrimaryCtxRetain(&context, cuDevice);
    if (result != CUDA_SUCCESS) {
        last_error = convertDriverError(result);
        return last_error;
    }

    // Set as current context
    result = cuCtxSetCurrent(context);
    last_error = convertDriverError(result);
    printf("[FakeCUDART] cudaSetDevice(%d)\n", device);
    return last_error;
}

cudaError_t cudaGetDevice(int *device) {
    if (!device) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    // Get current context
    CUcontext context;
    CUresult result = cuCtxGetCurrent(&context);
    if (result != CUDA_SUCCESS) {
        last_error = convertDriverError(result);
        return last_error;
    }

    // Get device from context
    CUdevice cuDevice;
    result = cuCtxGetDevice(&cuDevice);
    if (result != CUDA_SUCCESS) {
        last_error = convertDriverError(result);
        return last_error;
    }

    *device = cuDevice;
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device) {
    if (!prop) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    // Zero out the structure
    memset(prop, 0, sizeof(cudaDeviceProp));

    // Get device name
    char name[256];
    CUresult result = cuDeviceGetName(name, 256, device);
    if (result != CUDA_SUCCESS) {
        last_error = convertDriverError(result);
        return last_error;
    }
    strncpy(prop->name, name, 256);

    // Get total memory
    size_t totalMem;
    result = cuDeviceTotalMem(&totalMem, device);
    if (result == CUDA_SUCCESS) {
        prop->totalGlobalMem = totalMem;
    }

    // Get compute capability
    int major, minor;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    prop->major = major;
    prop->minor = minor;

    // Get other attributes
    cuDeviceGetAttribute(&prop->multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
    cuDeviceGetAttribute(&prop->maxThreadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);
    cuDeviceGetAttribute(&prop->maxThreadsPerMultiProcessor, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device);
    cuDeviceGetAttribute(&prop->warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device);
    cuDeviceGetAttribute(&prop->clockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device);
    cuDeviceGetAttribute(&prop->memoryClockRate, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device);
    cuDeviceGetAttribute(&prop->memoryBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device);
    cuDeviceGetAttribute(&prop->l2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device);
    cuDeviceGetAttribute((int*)&prop->sharedMemPerBlock, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device);
    cuDeviceGetAttribute(&prop->regsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device);
    cuDeviceGetAttribute(&prop->pciBusID, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, device);
    cuDeviceGetAttribute(&prop->pciDeviceID, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, device);
    cuDeviceGetAttribute(&prop->pciDomainID, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, device);

    last_error = cudaSuccess;
    printf("[FakeCUDART] cudaGetDeviceProperties(%d)\n", device);
    return last_error;
}

cudaError_t cudaDeviceReset(void) {
    // Get current device
    CUcontext context;
    cuCtxGetCurrent(&context);

    if (context) {
        CUdevice device;
        cuCtxGetDevice(&device);
        cuDevicePrimaryCtxReset(device);
    }

    last_error = cudaSuccess;
    printf("[FakeCUDART] cudaDeviceReset()\n");
    return last_error;
}

cudaError_t cudaDeviceSynchronize(void) {
    CUresult result = cuCtxSynchronize();
    last_error = convertDriverError(result);
    printf("[FakeCUDART] cudaDeviceSynchronize()\n");
    return last_error;
}

// ============================================================================
// Memory Management
// ============================================================================

cudaError_t cudaMalloc(void **devPtr, size_t size) {
    if (!devPtr) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    CUdeviceptr dptr;
    CUresult result = cuMemAlloc(&dptr, size);

    if (result == CUDA_SUCCESS) {
        *devPtr = (void*)dptr;
        last_error = cudaSuccess;
        printf("[FakeCUDART] cudaMalloc allocated %zu bytes at %p\n", size, *devPtr);
    } else {
        last_error = convertDriverError(result);
    }

    return last_error;
}

cudaError_t cudaFree(void *devPtr) {
    if (!devPtr) {
        last_error = cudaSuccess;
        return last_error;
    }

    CUresult result = cuMemFree((CUdeviceptr)devPtr);
    last_error = convertDriverError(result);
    printf("[FakeCUDART] cudaFree(%p)\n", devPtr);
    return last_error;
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind) {
    CUresult result = CUDA_SUCCESS;

    switch (kind) {
        case cudaMemcpyHostToDevice:
            result = cuMemcpyHtoD((CUdeviceptr)dst, src, count);
            break;
        case cudaMemcpyDeviceToHost:
            result = cuMemcpyDtoH(dst, (CUdeviceptr)src, count);
            break;
        case cudaMemcpyDeviceToDevice:
            result = cuMemcpyDtoD((CUdeviceptr)dst, (CUdeviceptr)src, count);
            break;
        case cudaMemcpyHostToHost:
            memcpy(dst, src, count);
            break;
        default:
            last_error = cudaErrorInvalidMemcpyDirection;
            return last_error;
    }

    last_error = convertDriverError(result);
    printf("[FakeCUDART] cudaMemcpy count=%zu kind=%d\n", count, kind);
    return last_error;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
    // Simplified: just do synchronous memcpy
    memcpy(dst, src, count);
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
    memset(devPtr, value, count);
    last_error = cudaSuccess;
    printf("[FakeCUDART] cudaMemset ptr=%p value=%d count=%zu\n", devPtr, value, count);
    return last_error;
}

cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream) {
    // Simplified: just do synchronous memset
    memset(devPtr, value, count);
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaMemGetInfo(size_t *free, size_t *total) {
    CUresult result = cuMemGetInfo(free, total);
    last_error = convertDriverError(result);
    return last_error;
}

// ============================================================================
// Stream Management
// ============================================================================

cudaError_t cudaStreamCreate(cudaStream_t *pStream) {
    if (!pStream) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    CUresult result = cuStreamCreate((CUstream*)pStream, 0);
    last_error = convertDriverError(result);
    printf("[FakeCUDART] cudaStreamCreate\n");
    return last_error;
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    CUresult result = cuStreamDestroy((CUstream)stream);
    last_error = convertDriverError(result);
    printf("[FakeCUDART] cudaStreamDestroy\n");
    return last_error;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    CUresult result = cuStreamSynchronize((CUstream)stream);
    last_error = convertDriverError(result);
    printf("[FakeCUDART] cudaStreamSynchronize\n");
    return last_error;
}

cudaError_t cudaStreamQuery(cudaStream_t stream) {
    CUresult result = cuStreamQuery((CUstream)stream);
    last_error = convertDriverError(result);
    return last_error;
}

// ============================================================================
// Event Management
// ============================================================================

cudaError_t cudaEventCreate(cudaEvent_t *event) {
    if (!event) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    CUresult result = cuEventCreate((CUevent*)event, 0);
    last_error = convertDriverError(result);
    printf("[FakeCUDART] cudaEventCreate\n");
    return last_error;
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
    CUresult result = cuEventDestroy((CUevent)event);
    last_error = convertDriverError(result);
    printf("[FakeCUDART] cudaEventDestroy\n");
    return last_error;
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    CUresult result = cuEventRecord((CUevent)event, (CUstream)stream);
    last_error = convertDriverError(result);
    return last_error;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    CUresult result = cuEventSynchronize((CUevent)event);
    last_error = convertDriverError(result);
    return last_error;
}

cudaError_t cudaEventQuery(cudaEvent_t event) {
    CUresult result = cuEventQuery((CUevent)event);
    last_error = convertDriverError(result);
    return last_error;
}

cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
    if (!ms) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    CUresult result = cuEventElapsedTime(ms, (CUevent)start, (CUevent)end);
    last_error = convertDriverError(result);
    return last_error;
}

// ============================================================================
// Error Handling
// ============================================================================

cudaError_t cudaGetLastError(void) {
    cudaError_t err = last_error;
    last_error = cudaSuccess;  // Clear error
    printf("[FakeCUDART] cudaGetLastError returning %d\n", err);
    return err;
}

cudaError_t cudaPeekAtLastError(void) {
    printf("[FakeCUDART] cudaPeekAtLastError returning %d\n", last_error);
    return last_error;
}

const char* cudaGetErrorString(cudaError_t error) {
    switch (error) {
        case cudaSuccess:
            return "no error";
        case cudaErrorInvalidValue:
            return "invalid argument";
        case cudaErrorMemoryAllocation:
            return "out of memory";
        case cudaErrorInitializationError:
            return "initialization error";
        case cudaErrorInvalidDevice:
            return "invalid device ordinal";
        case cudaErrorInvalidMemcpyDirection:
            return "invalid memcpy direction";
        default:
            return "unknown error";
    }
}

const char* cudaGetErrorName(cudaError_t error) {
    switch (error) {
        case cudaSuccess:
            return "cudaSuccess";
        case cudaErrorInvalidValue:
            return "cudaErrorInvalidValue";
        case cudaErrorMemoryAllocation:
            return "cudaErrorMemoryAllocation";
        case cudaErrorInitializationError:
            return "cudaErrorInitializationError";
        case cudaErrorInvalidDevice:
            return "cudaErrorInvalidDevice";
        case cudaErrorInvalidMemcpyDirection:
            return "cudaErrorInvalidMemcpyDirection";
        default:
            return "cudaErrorUnknown";
    }
}

// ============================================================================
// Version Management
// ============================================================================

cudaError_t cudaRuntimeGetVersion(int *runtimeVersion) {
    if (!runtimeVersion) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    *runtimeVersion = 12000;  // CUDA 12.0
    last_error = cudaSuccess;
    printf("[FakeCUDART] cudaRuntimeGetVersion returning 12000\n");
    return last_error;
}

cudaError_t cudaDriverGetVersion(int *driverVersion) {
    if (!driverVersion) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    CUresult result = cuDriverGetVersion(driverVersion);
    last_error = convertDriverError(result);
    printf("[FakeCUDART] cudaDriverGetVersion returning %d\n", *driverVersion);
    return last_error;
}

// ============================================================================
// Kernel Launch (stubs - no actual computation)
// ============================================================================

cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem, cudaStream_t stream) {
    printf("[FakeCUDART] cudaLaunchKernel (stub) Grid(%d,%d,%d) Block(%d,%d,%d)\n",
           gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);

    // No actual kernel execution
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream) {
    printf("[FakeCUDART] cudaConfigureCall (stub)\n");
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset) {
    // Just a stub
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaLaunch(const void *func) {
    printf("[FakeCUDART] cudaLaunch (stub)\n");
    last_error = cudaSuccess;
    return last_error;
}

// ============================================================================
// Host Memory
// ============================================================================

cudaError_t cudaMallocHost(void **ptr, size_t size) {
    if (!ptr) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    // Use regular malloc as fallback
    *ptr = malloc(size);
    if (!*ptr) {
        last_error = cudaErrorMemoryAllocation;
        return last_error;
    }

    last_error = cudaSuccess;
    printf("[FakeCUDART] cudaMallocHost allocated %zu bytes\n", size);
    return last_error;
}

cudaError_t cudaFreeHost(void *ptr) {
    free(ptr);
    last_error = cudaSuccess;
    printf("[FakeCUDART] cudaFreeHost\n");
    return last_error;
}

cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags) {
    if (!pHost) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    // Use regular malloc as fallback
    *pHost = malloc(size);
    if (!*pHost) {
        last_error = cudaErrorMemoryAllocation;
        return last_error;
    }

    last_error = cudaSuccess;
    printf("[FakeCUDART] cudaHostAlloc allocated %zu bytes\n", size);
    return last_error;
}

// ============================================================================
// Peer Access
// ============================================================================

cudaError_t cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice) {
    if (!canAccessPeer) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    // Simplified: always allow peer access
    *canAccessPeer = 1;
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) {
    // Simplified: just succeed
    last_error = cudaSuccess;
    printf("[FakeCUDART] cudaDeviceEnablePeerAccess(%d)\n", peerDevice);
    return last_error;
}

cudaError_t cudaDeviceDisablePeerAccess(int peerDevice) {
    // Simplified: just succeed
    last_error = cudaSuccess;
    printf("[FakeCUDART] cudaDeviceDisablePeerAccess(%d)\n", peerDevice);
    return last_error;
}

// ============================================================================
// Driver Entry Point (CUDA 12+)
// ============================================================================

cudaError_t cudaGetDriverEntryPointByVersion(const char *symbol, void **funcPtr, unsigned int cudaVersion, unsigned long long flags, int *driverStatus) {
    if (!symbol || !funcPtr) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    // Use cuGetProcAddress to get the driver function
    CUresult result = cuGetProcAddress(symbol, funcPtr, cudaVersion, flags);

    if (driverStatus) {
        *driverStatus = (result == CUDA_SUCCESS) ? 1 : 0;
    }

    last_error = convertDriverError(result);
    return last_error;
}

cudaError_t cudaGetDriverEntryPoint(const char *symbol, void **funcPtr, unsigned long long flags) {
    return cudaGetDriverEntryPointByVersion(symbol, funcPtr, 12000, flags, NULL);
}

} // extern "C"
