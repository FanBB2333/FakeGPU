#include "cudart_defs.hpp"
#include "cuda_driver_defs.hpp"
#include "../core/global_state.hpp"
#include "../core/logging.hpp"
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
    FGPU_LOG("[FakeCUDART] cudaGetDeviceCount returning %d\n", *count);
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
    FGPU_LOG("[FakeCUDART] cudaSetDevice(%d)\n", device);
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
    FGPU_LOG("[FakeCUDART] cudaGetDeviceProperties(%d)\n", device);
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
    FGPU_LOG("[FakeCUDART] cudaDeviceReset()\n");
    return last_error;
}

cudaError_t cudaDeviceSynchronize(void) {
    CUresult result = cuCtxSynchronize();
    last_error = convertDriverError(result);
    FGPU_LOG("[FakeCUDART] cudaDeviceSynchronize()\n");
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
        FGPU_LOG("[FakeCUDART] cudaMalloc allocated %zu bytes at %p\n", size, *devPtr);
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
    FGPU_LOG("[FakeCUDART] cudaFree(%p)\n", devPtr);
    return last_error;
}

cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t stream) {
    if (!devPtr) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    CUdeviceptr dptr;
    CUresult result = cuMemAlloc(&dptr, size);
    if (result == CUDA_SUCCESS) {
        *devPtr = (void*)dptr;
        last_error = cudaSuccess;
        FGPU_LOG("[FakeCUDART] cudaMallocAsync allocated %zu bytes at %p\n", size, *devPtr);
    } else {
        last_error = convertDriverError(result);
    }

    return last_error;
}

cudaError_t cudaFreeAsync(void *devPtr, cudaStream_t stream) {
    if (!devPtr) {
        last_error = cudaSuccess;
        return last_error;
    }

    CUresult result = cuMemFree((CUdeviceptr)devPtr);
    last_error = convertDriverError(result);
    FGPU_LOG("[FakeCUDART] cudaFreeAsync(%p)\n", devPtr);
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
            fake_gpu::GlobalState::instance().record_memcpy_h2h(count);
            break;
        default:
            last_error = cudaErrorInvalidMemcpyDirection;
            return last_error;
    }

    last_error = convertDriverError(result);
    FGPU_LOG("[FakeCUDART] cudaMemcpy count=%zu kind=%d\n", count, kind);
    return last_error;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
    // Simplified: execute synchronously and ignore stream.
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
            fake_gpu::GlobalState::instance().record_memcpy_h2h(count);
            result = CUDA_SUCCESS;
            break;
        default:
            last_error = cudaErrorInvalidMemcpyDirection;
            return last_error;
    }

    last_error = convertDriverError(result);
    return last_error;
}

cudaError_t cudaMemcpyPeer(void *dst, int dstDevice, const void *src, int srcDevice, size_t count) {
    // Simplified: just do memory copy
    memcpy(dst, src, count);
    last_error = cudaSuccess;
    fake_gpu::GlobalState::instance().record_memcpy_peer(dstDevice, srcDevice, count);
    FGPU_LOG("[FakeCUDART] cudaMemcpyPeer count=%zu from device %d to device %d\n", count, srcDevice, dstDevice);
    return last_error;
}

cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src, int srcDevice, size_t count, cudaStream_t stream) {
    // Simplified: just do memory copy
    memcpy(dst, src, count);
    last_error = cudaSuccess;
    fake_gpu::GlobalState::instance().record_memcpy_peer(dstDevice, srcDevice, count);
    FGPU_LOG("[FakeCUDART] cudaMemcpyPeerAsync count=%zu from device %d to device %d\n", count, srcDevice, dstDevice);
    return last_error;
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
    memset(devPtr, value, count);
    last_error = cudaSuccess;
    fake_gpu::GlobalState::instance().record_memset(devPtr, count);
    FGPU_LOG("[FakeCUDART] cudaMemset ptr=%p value=%d count=%zu\n", devPtr, value, count);
    return last_error;
}

cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream) {
    // Simplified: just do synchronous memset
    memset(devPtr, value, count);
    last_error = cudaSuccess;
    fake_gpu::GlobalState::instance().record_memset(devPtr, count);
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
    FGPU_LOG("[FakeCUDART] cudaStreamCreate\n");
    return last_error;
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    CUresult result = cuStreamDestroy((CUstream)stream);
    last_error = convertDriverError(result);
    FGPU_LOG("[FakeCUDART] cudaStreamDestroy\n");
    return last_error;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    CUresult result = cuStreamSynchronize((CUstream)stream);
    last_error = convertDriverError(result);
    FGPU_LOG("[FakeCUDART] cudaStreamSynchronize\n");
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
    FGPU_LOG("[FakeCUDART] cudaEventCreate\n");
    return last_error;
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
    CUresult result = cuEventDestroy((CUevent)event);
    last_error = convertDriverError(result);
    FGPU_LOG("[FakeCUDART] cudaEventDestroy\n");
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

cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags) {
    CUresult result = cuEventRecord((CUevent)event, (CUstream)stream);
    last_error = convertDriverError(result);
    return last_error;
}

// ============================================================================
// Error Handling
// ============================================================================

cudaError_t cudaGetLastError(void) {
    cudaError_t err = last_error;
    last_error = cudaSuccess;  // Clear error
    FGPU_LOG("[FakeCUDART] cudaGetLastError returning %d\n", err);
    return err;
}

cudaError_t cudaPeekAtLastError(void) {
    FGPU_LOG("[FakeCUDART] cudaPeekAtLastError returning %d\n", last_error);
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
    FGPU_LOG("[FakeCUDART] cudaRuntimeGetVersion returning 12000\n");
    return last_error;
}

cudaError_t cudaDriverGetVersion(int *driverVersion) {
    if (!driverVersion) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    CUresult result = cuDriverGetVersion(driverVersion);
    last_error = convertDriverError(result);
    FGPU_LOG("[FakeCUDART] cudaDriverGetVersion returning %d\n", *driverVersion);
    return last_error;
}

// ============================================================================
// Kernel Launch (stubs - no actual computation)
// ============================================================================

cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem, cudaStream_t stream) {
    FGPU_LOG("[FakeCUDART] cudaLaunchKernel (stub) Grid(%d,%d,%d) Block(%d,%d,%d)\n",
           gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);

    // No actual kernel execution
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream) {
    FGPU_LOG("[FakeCUDART] cudaConfigureCall (stub)\n");
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset) {
    // Just a stub
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaLaunch(const void *func) {
    FGPU_LOG("[FakeCUDART] cudaLaunch (stub)\n");
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaLaunchKernelExC(const void *config, const void *func, void **args) {
    FGPU_LOG("[FakeCUDART] cudaLaunchKernelExC (stub)\n");
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaLaunchHostFunc(cudaStream_t stream, void (*fn)(void *userData), void *userData) {
    // Simplified: just execute the function directly
    if (fn) {
        fn(userData);
    }
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaLaunchHostFunc (executed host function)\n");
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
    FGPU_LOG("[FakeCUDART] cudaMallocHost allocated %zu bytes\n", size);
    return last_error;
}

cudaError_t cudaFreeHost(void *ptr) {
    free(ptr);
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaFreeHost\n");
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
    FGPU_LOG("[FakeCUDART] cudaHostAlloc allocated %zu bytes\n", size);
    return last_error;
}

cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags) {
    // Simplified: just succeed
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaHostRegister ptr=%p size=%zu flags=%u\n", ptr, size, flags);
    return last_error;
}

cudaError_t cudaHostUnregister(void *ptr) {
    // Simplified: just succeed
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaHostUnregister ptr=%p\n", ptr);
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
    FGPU_LOG("[FakeCUDART] cudaDeviceEnablePeerAccess(%d)\n", peerDevice);
    return last_error;
}

cudaError_t cudaDeviceDisablePeerAccess(int peerDevice) {
    // Simplified: just succeed
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaDeviceDisablePeerAccess(%d)\n", peerDevice);
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

// ============================================================================
// Additional Stream Management
// ============================================================================

cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags) {
    if (!pStream) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    CUresult result = cuStreamCreate((CUstream*)pStream, flags);
    last_error = convertDriverError(result);
    FGPU_LOG("[FakeCUDART] cudaStreamCreateWithFlags flags=%u\n", flags);
    return last_error;
}

cudaError_t cudaStreamCreateWithPriority(cudaStream_t *pStream, unsigned int flags, int priority) {
    if (!pStream) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    // Simplified: ignore priority
    CUresult result = cuStreamCreate((CUstream*)pStream, flags);
    last_error = convertDriverError(result);
    FGPU_LOG("[FakeCUDART] cudaStreamCreateWithPriority flags=%u priority=%d\n", flags, priority);
    return last_error;
}

cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int *flags) {
    if (!flags) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    CUresult result = cuStreamGetFlags((CUstream)hStream, flags);
    last_error = convertDriverError(result);
    return last_error;
}

cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int *priority) {
    if (!priority) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    // Simplified: always return 0
    *priority = 0;
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
    CUresult result = cuStreamWaitEvent((CUstream)stream, (CUevent)event, flags);
    last_error = convertDriverError(result);
    return last_error;
}

cudaError_t cudaStreamAddCallback(cudaStream_t stream, void (*callback)(cudaStream_t stream, cudaError_t status, void *userData), void *userData, unsigned int flags) {
    // Simplified: just succeed without actually adding callback
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaStreamAddCallback (stub)\n");
    return last_error;
}

cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t *dependencies, size_t numDependencies, unsigned int flags) {
    // Simplified: just succeed
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaStreamUpdateCaptureDependencies (stub)\n");
    return last_error;
}

// ============================================================================
// Additional Event Management
// ============================================================================

cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags) {
    if (!event) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    CUresult result = cuEventCreate((CUevent*)event, flags);
    last_error = convertDriverError(result);
    FGPU_LOG("[FakeCUDART] cudaEventCreateWithFlags flags=%u\n", flags);
    return last_error;
}

// ============================================================================
// Occupancy Calculation
// ============================================================================

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize) {
    if (!numBlocks) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    // Simplified: return a reasonable fake value
    *numBlocks = 16;
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize, unsigned int flags) {
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize);
}

// ============================================================================
// Device Attributes
// ============================================================================

cudaError_t cudaDeviceGetAttribute(int *value, int attr, int device) {
    if (!value) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    CUresult result = cuDeviceGetAttribute(value, (CUdevice_attribute)attr, device);
    last_error = convertDriverError(result);
    return last_error;
}

cudaError_t cudaDeviceGetLimit(size_t *pValue, int limit) {
    if (!pValue) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    // Simplified: return reasonable defaults
    *pValue = 1024 * 1024 * 1024;  // 1GB
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaDeviceSetLimit(int limit, size_t value) {
    // Simplified: just succeed
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaDeviceGetCacheConfig(int *pCacheConfig) {
    if (!pCacheConfig) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    *pCacheConfig = 0;  // cudaFuncCachePreferNone
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaDeviceSetCacheConfig(int cacheConfig) {
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaDeviceGetSharedMemConfig(int *pConfig) {
    if (!pConfig) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    *pConfig = 0;
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaDeviceSetSharedMemConfig(int config) {
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority) {
    if (!leastPriority || !greatestPriority) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    *leastPriority = 0;
    *greatestPriority = 0;
    last_error = cudaSuccess;
    return last_error;
}

// ============================================================================
// Additional Memory Management
// ============================================================================

cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height) {
    if (!devPtr || !pitch) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    // Align pitch to 512 bytes
    *pitch = (width + 511) & ~511;
    size_t size = (*pitch) * height;

    CUdeviceptr dptr;
    CUresult result = cuMemAlloc(&dptr, size);
    if (result == CUDA_SUCCESS) {
        *devPtr = (void*)dptr;
        last_error = cudaSuccess;
        FGPU_LOG("[FakeCUDART] cudaMallocPitch allocated %zu bytes (pitch=%zu)\n", size, *pitch);
    } else {
        last_error = convertDriverError(result);
    }

    return last_error;
}

cudaError_t cudaMalloc3D(void **devPtr, size_t width, size_t height, size_t depth) {
    if (!devPtr) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    size_t size = width * height * depth;
    CUdeviceptr dptr;
    CUresult result = cuMemAlloc(&dptr, size);
    if (result == CUDA_SUCCESS) {
        *devPtr = (void*)dptr;
        last_error = cudaSuccess;
        FGPU_LOG("[FakeCUDART] cudaMalloc3D allocated %zu bytes\n", size);
    } else {
        last_error = convertDriverError(result);
    }

    return last_error;
}

cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags) {
    if (!devPtr) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    // Simplified: use regular cudaMalloc
    CUdeviceptr dptr;
    CUresult result = cuMemAlloc(&dptr, size);
    if (result == CUDA_SUCCESS) {
        *devPtr = (void*)dptr;
        last_error = cudaSuccess;
        FGPU_LOG("[FakeCUDART] cudaMallocManaged allocated %zu bytes\n", size);
    } else {
        last_error = convertDriverError(result);
    }

    return last_error;
}

cudaError_t cudaMallocArray(void **array, const void *desc, size_t width, size_t height, unsigned int flags) {
    if (!array) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    // Simplified: allocate linear memory
    size_t size = width * height * 4;  // Assume 4 bytes per element
    *array = malloc(size);
    if (!*array) {
        last_error = cudaErrorMemoryAllocation;
        return last_error;
    }

    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaMallocArray allocated %zu bytes\n", size);
    return last_error;
}

cudaError_t cudaFreeArray(void *array) {
    free(array);
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaFreeArray\n");
    return last_error;
}

cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) {
    // Simplified: copy row by row
    for (size_t i = 0; i < height; i++) {
        const char *srcRow = (const char*)src + i * spitch;
        char *dstRow = (char*)dst + i * dpitch;
        memcpy(dstRow, srcRow, width);
    }

    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) {
    // Simplified: do synchronous copy
    return cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
}

cudaError_t cudaMemcpy3D(const void *p) {
    // Simplified: just succeed
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaMemcpy3DAsync(const void *p, cudaStream_t stream) {
    // Simplified: just succeed
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset, cudaMemcpyKind kind) {
    // Simplified: just succeed
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset, cudaMemcpyKind kind) {
    // Simplified: just succeed
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaMemAdvise(const void *devPtr, size_t count, int advice, int device) {
    // Simplified: just succeed
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaMemPrefetchAsync(const void *devPtr, size_t count, int dstDevice, cudaStream_t stream) {
    // Simplified: just succeed
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaMemRangeGetAttribute(void *data, size_t dataSize, int attribute, const void *devPtr, size_t count) {
    // Simplified: just succeed
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaMemRangeGetAttributes(void **data, size_t *dataSizes, int *attributes, size_t numAttributes, const void *devPtr, size_t count) {
    // Simplified: just succeed
    last_error = cudaSuccess;
    return last_error;
}

// ============================================================================
// Unified Memory
// ============================================================================

cudaError_t cudaMemAttachGlobal(void *devPtr) {
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaMemAttach(void *devPtr, unsigned int flags) {
    last_error = cudaSuccess;
    return last_error;
}

// ============================================================================
// Memory Pool Management
// ============================================================================

cudaError_t cudaMemPoolCreate(cudaMemPool_t *memPool, const void *poolProps) {
    if (!memPool) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    // Create a dummy memory pool
    *memPool = (cudaMemPool_t)malloc(sizeof(void*));
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaMemPoolCreate\n");
    return last_error;
}

cudaError_t cudaMemPoolDestroy(cudaMemPool_t memPool) {
    free(memPool);
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaMemPoolDestroy\n");
    return last_error;
}

cudaError_t cudaMallocFromPoolAsync(void **ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream) {
    if (!ptr) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    // Simplified: use regular malloc
    CUdeviceptr dptr;
    CUresult result = cuMemAlloc(&dptr, size);
    if (result == CUDA_SUCCESS) {
        *ptr = (void*)dptr;
        last_error = cudaSuccess;
    } else {
        last_error = convertDriverError(result);
    }

    return last_error;
}

cudaError_t cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep) {
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool, int attr, void *value) {
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool, int attr, void *value) {
    if (!value) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    // Simplified: return default values based on attribute
    // Most attributes are integers or size_t
    *(size_t*)value = 0;
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaMemPoolSetAccess(cudaMemPool_t memPool, const void *descList, size_t count) {
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaMemPoolGetAccess(void *flags, cudaMemPool_t memPool, void *location) {
    if (!flags) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    *(int*)flags = 1;  // Access granted
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t *memPool, int device) {
    if (!memPool) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    // Return a dummy default pool
    static cudaMemPool_t defaultPool = NULL;
    if (!defaultPool) {
        defaultPool = (cudaMemPool_t)malloc(sizeof(void*));
    }
    *memPool = defaultPool;
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaDeviceSetMemPool(int device, cudaMemPool_t memPool) {
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaDeviceGetMemPool(cudaMemPool_t *memPool, int device) {
    return cudaDeviceGetDefaultMemPool(memPool, device);
}

// ============================================================================
// Pointer Attributes
// ============================================================================

cudaError_t cudaPointerGetAttributes(cudaPointerAttributes *attributes, const void *ptr) {
    if (!attributes || !ptr) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    memset(attributes, 0, sizeof(cudaPointerAttributes));

    size_t alloc_size = 0;
    int alloc_device = 0;
    bool found = fake_gpu::GlobalState::instance().get_allocation_info(
        const_cast<void*>(ptr), alloc_size, alloc_device);

    attributes->type = found ? cudaMemoryTypeDevice : cudaMemoryTypeUnregistered;
    attributes->device = found ? alloc_device : 0;
    attributes->devicePointer = const_cast<void*>(ptr);
    attributes->hostPointer = const_cast<void*>(ptr);

    last_error = cudaSuccess;
    return last_error;
}

// ============================================================================
// IPC (Inter-Process Communication)
// ============================================================================

cudaError_t cudaIpcGetMemHandle(void *handle, void *devPtr) {
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaIpcOpenMemHandle(void **devPtr, void *handle, unsigned int flags) {
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaIpcCloseMemHandle(void *devPtr) {
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaIpcGetEventHandle(void *handle, cudaEvent_t event) {
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaIpcOpenEventHandle(cudaEvent_t *event, void *handle) {
    last_error = cudaSuccess;
    return last_error;
}

// ============================================================================
// CUDA Graph API
// ============================================================================

cudaError_t cudaGraphCreate(cudaGraph_t *pGraph, unsigned int flags) {
    if (!pGraph) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    // Create a dummy graph structure
    *pGraph = (cudaGraph_t)malloc(sizeof(void*));
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaGraphCreate\n");
    return last_error;
}

cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
    free(graph);
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaGraphDestroy\n");
    return last_error;
}

cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const void *pNodeParams) {
    if (!pGraphNode) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    *pGraphNode = (cudaGraphNode_t)malloc(sizeof(void*));
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaGraphAddKernelNode\n");
    return last_error;
}

cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const void *pCopyParams) {
    if (!pGraphNode) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    *pGraphNode = (cudaGraphNode_t)malloc(sizeof(void*));
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaGraphAddMemcpyNode\n");
    return last_error;
}

cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const void *pMemsetParams) {
    if (!pGraphNode) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    *pGraphNode = (cudaGraphNode_t)malloc(sizeof(void*));
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaGraphAddMemsetNode\n");
    return last_error;
}

cudaError_t cudaGraphAddHostNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const void *pNodeParams) {
    if (!pGraphNode) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    *pGraphNode = (cudaGraphNode_t)malloc(sizeof(void*));
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaGraphAddHostNode\n");
    return last_error;
}

cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, cudaGraph_t childGraph) {
    if (!pGraphNode) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    *pGraphNode = (cudaGraphNode_t)malloc(sizeof(void*));
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaGraphAddChildGraphNode\n");
    return last_error;
}

cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies) {
    if (!pGraphNode) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    *pGraphNode = (cudaGraphNode_t)malloc(sizeof(void*));
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaGraphAddEmptyNode\n");
    return last_error;
}

cudaError_t cudaGraphAddEventRecordNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, cudaEvent_t event) {
    if (!pGraphNode) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    *pGraphNode = (cudaGraphNode_t)malloc(sizeof(void*));
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaGraphAddEventRecordNode\n");
    return last_error;
}

cudaError_t cudaGraphAddEventWaitNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, cudaEvent_t event) {
    if (!pGraphNode) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    *pGraphNode = (cudaGraphNode_t)malloc(sizeof(void*));
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaGraphAddEventWaitNode\n");
    return last_error;
}

cudaError_t cudaGraphClone(cudaGraph_t *pGraphClone, cudaGraph_t originalGraph) {
    if (!pGraphClone) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    *pGraphClone = (cudaGraph_t)malloc(sizeof(void*));
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaGraphClone\n");
    return last_error;
}

cudaError_t cudaGraphNodeFindInClone(cudaGraphNode_t *pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph) {
    if (!pNode) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    *pNode = originalNode;
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType *pType) {
    if (!pType) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    *pType = cudaGraphNodeTypeKernel;
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t *nodes, size_t *numNodes) {
    if (!numNodes) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    *numNodes = 0;
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t *pRootNodes, size_t *pNumRootNodes) {
    if (!pNumRootNodes) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    *pNumRootNodes = 0;
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t *from, cudaGraphNode_t *to, size_t *numEdges) {
    if (!numEdges) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    *numEdges = 0;
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t *pDependencies, size_t *pNumDependencies) {
    if (!pNumDependencies) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    *pNumDependencies = 0;
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t *pDependentNodes, size_t *pNumDependentNodes) {
    if (!pNumDependentNodes) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    *pNumDependentNodes = 0;
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t *from, const cudaGraphNode_t *to, size_t numDependencies) {
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t *from, const cudaGraphNode_t *to, size_t numDependencies) {
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaGraphInstantiate(cudaGraphExec_t *pGraphExec, cudaGraph_t graph, cudaGraphNode_t *pErrorNode, char *pLogBuffer, size_t bufferSize) {
    if (!pGraphExec) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    *pGraphExec = (cudaGraphExec_t)malloc(sizeof(void*));
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaGraphInstantiate\n");
    return last_error;
}

cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t *pGraphExec, cudaGraph_t graph, unsigned long long flags) {
    if (!pGraphExec) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    *pGraphExec = (cudaGraphExec_t)malloc(sizeof(void*));
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaGraphInstantiateWithFlags flags=%llu\n", flags);
    return last_error;
}

cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec) {
    free(graphExec);
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaGraphExecDestroy\n");
    return last_error;
}

cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream) {
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaGraphLaunch (stub)\n");
    return last_error;
}

cudaError_t cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream) {
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaGraphUpload (stub)\n");
    return last_error;
}

cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph, const char *path, unsigned int flags) {
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaGraphDebugDotPrint path=%s flags=%u (stub)\n", path ? path : "NULL", flags);
    return last_error;
}

cudaError_t cudaStreamBeginCapture(cudaStream_t stream, int mode) {
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaStreamBeginCapture mode=%d\n", mode);
    return last_error;
}

cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *pGraph) {
    if (!pGraph) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    *pGraph = (cudaGraph_t)malloc(sizeof(void*));
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaStreamEndCapture\n");
    return last_error;
}

cudaError_t cudaStreamIsCapturing(cudaStream_t stream, int *pCaptureStatus) {
    if (!pCaptureStatus) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    *pCaptureStatus = 0;  // Not capturing
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream, int *captureStatus, unsigned long long *id) {
    if (!captureStatus) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    *captureStatus = 0;  // Not capturing
    if (id) *id = 0;
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaThreadExchangeStreamCaptureMode(int *mode) {
    if (!mode) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    // Return current mode (simplified: always relaxed mode)
    int oldMode = *mode;
    *mode = 0;  // cudaStreamCaptureModeGlobal
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaThreadExchangeStreamCaptureMode old=%d new=0\n", oldMode);
    return last_error;
}

cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t stream, int *captureStatus, unsigned long long *id, cudaGraph_t *graph, const cudaGraphNode_t **dependencies, size_t *numDependencies) {
    if (!captureStatus) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    *captureStatus = 0;  // Not capturing
    if (id) *id = 0;
    if (graph) *graph = NULL;
    if (numDependencies) *numDependencies = 0;
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaGetDeviceProperties_v2(cudaDeviceProp *prop, int device) {
    // Alias to cudaGetDeviceProperties
    return cudaGetDeviceProperties(prop, device);
}

// ============================================================================
// Texture and Surface Object API
// ============================================================================

cudaError_t cudaCreateTextureObject(cudaTextureObject_t *pTexObject, const cudaResourceDesc *pResDesc, const cudaTextureDesc *pTexDesc, const cudaResourceViewDesc *pResViewDesc) {
    if (!pTexObject) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    static cudaTextureObject_t nextTexId = 1;
    *pTexObject = nextTexId++;
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaCreateTextureObject id=%llu\n", *pTexObject);
    return last_error;
}

cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject) {
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaDestroyTextureObject id=%llu\n", texObject);
    return last_error;
}

cudaError_t cudaGetTextureObjectResourceDesc(cudaResourceDesc *pResDesc, cudaTextureObject_t texObject) {
    if (!pResDesc) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    memset(pResDesc, 0, sizeof(cudaResourceDesc));
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaGetTextureObjectTextureDesc(cudaTextureDesc *pTexDesc, cudaTextureObject_t texObject) {
    if (!pTexDesc) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    memset(pTexDesc, 0, sizeof(cudaTextureDesc));
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc *pResViewDesc, cudaTextureObject_t texObject) {
    if (!pResViewDesc) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    memset(pResViewDesc, 0, sizeof(cudaResourceViewDesc));
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t *pSurfObject, const cudaResourceDesc *pResDesc) {
    if (!pSurfObject) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    static cudaSurfaceObject_t nextSurfId = 1;
    *pSurfObject = nextSurfId++;
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaCreateSurfaceObject id=%llu\n", *pSurfObject);
    return last_error;
}

cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject) {
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaDestroySurfaceObject id=%llu\n", surfObject);
    return last_error;
}

cudaError_t cudaGetSurfaceObjectResourceDesc(cudaResourceDesc *pResDesc, cudaSurfaceObject_t surfObject) {
    if (!pResDesc) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    memset(pResDesc, 0, sizeof(cudaResourceDesc));
    last_error = cudaSuccess;
    return last_error;
}

// ============================================================================
// Cooperative Groups
// ============================================================================

cudaError_t cudaLaunchCooperativeKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) {
    FGPU_LOG("[FakeCUDART] cudaLaunchCooperativeKernel (stub) Grid(%d,%d,%d) Block(%d,%d,%d)\n",
           gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaLaunchCooperativeKernelMultiDevice(void *launchParamsList, unsigned int numDevices, unsigned int flags) {
    FGPU_LOG("[FakeCUDART] cudaLaunchCooperativeKernelMultiDevice (stub) numDevices=%u\n", numDevices);
    last_error = cudaSuccess;
    return last_error;
}

// ============================================================================
// Thread Management (Deprecated but still used)
// ============================================================================

cudaError_t cudaThreadSynchronize(void) {
    // Deprecated, maps to cudaDeviceSynchronize
    return cudaDeviceSynchronize();
}

cudaError_t cudaThreadExit(void) {
    // Deprecated, maps to cudaDeviceReset
    return cudaDeviceReset();
}

// ============================================================================
// Function Attributes
// ============================================================================

cudaError_t cudaFuncGetAttributes(void *attr, const void *func) {
    if (!attr) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    // Simplified: zero out attributes
    memset(attr, 0, 64);  // Assume structure is ~64 bytes
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaFuncSetCacheConfig(const void *func, int cacheConfig) {
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaFuncSetSharedMemConfig(const void *func, int config) {
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaFuncSetAttribute(const void *func, int attr, int value) {
    last_error = cudaSuccess;
    return last_error;
}

// ============================================================================
// Launch Bounds
// ============================================================================

cudaError_t cudaDeviceGetByPCIBusId(int *device, const char *pciBusId) {
    if (!device || !pciBusId) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    // Simplified: always return device 0
    *device = 0;
    last_error = cudaSuccess;
    return last_error;
}

cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device) {
    if (!pciBusId || len < 13) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    snprintf(pciBusId, len, "0000:00:00.0");
    last_error = cudaSuccess;
    return last_error;
}

// ============================================================================
// External Resource Interop
// ============================================================================

cudaError_t cudaImportExternalMemory(void **extMem_out, const void *memHandleDesc) {
    if (!extMem_out) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    *extMem_out = malloc(sizeof(void*));
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaImportExternalMemory (stub)\n");
    return last_error;
}

cudaError_t cudaExternalMemoryGetMappedBuffer(void **devPtr, void *extMem, const void *bufferDesc) {
    if (!devPtr) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    *devPtr = malloc(1024);  // Dummy buffer
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaExternalMemoryGetMappedBuffer (stub)\n");
    return last_error;
}

cudaError_t cudaDestroyExternalMemory(void *extMem) {
    free(extMem);
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaDestroyExternalMemory (stub)\n");
    return last_error;
}

cudaError_t cudaImportExternalSemaphore(void **extSem_out, const void *semHandleDesc) {
    if (!extSem_out) {
        last_error = cudaErrorInvalidValue;
        return last_error;
    }

    *extSem_out = malloc(sizeof(void*));
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaImportExternalSemaphore (stub)\n");
    return last_error;
}

cudaError_t cudaSignalExternalSemaphoresAsync(const void **extSemArray, const void **paramsArray, unsigned int numExtSems, cudaStream_t stream) {
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaSignalExternalSemaphoresAsync (stub)\n");
    return last_error;
}

cudaError_t cudaWaitExternalSemaphoresAsync(const void **extSemArray, const void **paramsArray, unsigned int numExtSems, cudaStream_t stream) {
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaWaitExternalSemaphoresAsync (stub)\n");
    return last_error;
}

cudaError_t cudaDestroyExternalSemaphore(void *extSem) {
    free(extSem);
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaDestroyExternalSemaphore (stub)\n");
    return last_error;
}

// ============================================================================
// Profiling
// ============================================================================

cudaError_t cudaProfilerStart(void) {
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaProfilerStart (stub)\n");
    return last_error;
}

cudaError_t cudaProfilerStop(void) {
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaProfilerStop (stub)\n");
    return last_error;
}

// ============================================================================
// Internal CUDA Runtime Functions (Module/Variable Registration)
// ============================================================================

void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun, const char *deviceName, int thread_limit, void *tid, void *bid, void *bDim, void *gDim, int *wSize) {
    FGPU_LOG("[FakeCUDART] __cudaRegisterFunction hostFun=%s deviceName=%s\n", hostFun ? hostFun : "NULL", deviceName ? deviceName : "NULL");
}

void __cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress, const char *deviceName, int ext, size_t size, int constant, int global) {
    FGPU_LOG("[FakeCUDART] __cudaRegisterVar deviceName=%s size=%zu constant=%d global=%d\n", deviceName ? deviceName : "NULL", size, constant, global);
}

void __cudaRegisterFatBinary(void *fatCubin) {
    FGPU_LOG("[FakeCUDART] __cudaRegisterFatBinary\n");
}

void __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
    FGPU_LOG("[FakeCUDART] __cudaRegisterFatBinaryEnd\n");
}

void __cudaUnregisterFatBinary(void **fatCubinHandle) {
    FGPU_LOG("[FakeCUDART] __cudaUnregisterFatBinary\n");
}

void** __cudaRegisterFatBinaryEnd_v2(void **fatCubinHandle) {
    FGPU_LOG("[FakeCUDART] __cudaRegisterFatBinaryEnd_v2\n");
    return fatCubinHandle;
}

static dim3 saved_gridDim;
static dim3 saved_blockDim;
static size_t saved_sharedMem;
static void *saved_stream;

cudaError_t __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem, void *stream) {
    saved_gridDim = gridDim;
    saved_blockDim = blockDim;
    saved_sharedMem = sharedMem;
    saved_stream = stream;
    return cudaSuccess;
}

cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim, size_t *sharedMem, void **stream) {
    if (gridDim) *gridDim = saved_gridDim;
    if (blockDim) *blockDim = saved_blockDim;
    if (sharedMem) *sharedMem = saved_sharedMem;
    if (stream) *stream = saved_stream;
    return cudaSuccess;
}

} // extern "C"
