#include "cuda_defs.hpp"
#include "../core/global_state.hpp"
#include "../core/logging.hpp"
#include <cstdio>
#include <cstring>
#include <cstdlib>

using namespace fake_gpu;

extern "C" {

cudaError_t cudaGetDeviceCount(int *count) {
    if (!count) return cudaErrorInvalidValue;
    // ensure initialized
    GlobalState::instance().initialize();
    *count = GlobalState::instance().get_device_count();
    FGPU_LOG("[FakeCUDA] cudaGetDeviceCount returning %d\n", *count);
    return cudaSuccess;
}

cudaError_t cudaSetDevice(int device) {
    GlobalState::instance().initialize();
    int count = GlobalState::instance().get_device_count();
    if (device < 0 || device >= count) {
        FGPU_LOG("[FakeCUDA] cudaSetDevice invalid index %d\n", device);
        return cudaErrorInvalidDevice;
    }
    GlobalState::instance().set_current_device(device);
    FGPU_LOG("[FakeCUDA] cudaSetDevice(%d)\n", device);
    return cudaSuccess;
}

cudaError_t cudaMalloc(void **devPtr, size_t size) {
    if (!devPtr) return cudaErrorInvalidValue;
    GlobalState::instance().initialize();
    int device = GlobalState::instance().get_current_device();
    Device& dev = GlobalState::instance().get_device(device);
    if (dev.index < 0) return cudaErrorInvalidDevice;

    // Allocate real RAM
    void* ptr = malloc(size);
    if (!ptr) {
        FGPU_LOG("[FakeCUDA] cudaMalloc failed to allocate %zu bytes\n", size);
        return cudaErrorMemoryAllocation;
    }

    if (!GlobalState::instance().register_allocation(ptr, size, device)) {
        FGPU_LOG("[FakeCUDA] cudaMalloc denied %zu bytes on device %d (over commitment)\n", size, device);
        free(ptr);
        return cudaErrorMemoryAllocation;
    }

    *devPtr = ptr;
    FGPU_LOG("[FakeCUDA] cudaMalloc allocated %zu bytes at %p on device %d\n", size, ptr, device);
    return cudaSuccess;
}

cudaError_t cudaFree(void *devPtr) {
    FGPU_LOG("[FakeCUDA] cudaFree(%p)\n", devPtr);
    if (!devPtr) return cudaSuccess;

    size_t size = 0;
    int device = -1;
    bool tracked = GlobalState::instance().release_allocation(devPtr, size, device);
    if (!tracked) {
        FGPU_LOG("[FakeCUDA] cudaFree warning: pointer not tracked\n");
    } else {
        FGPU_LOG("[FakeCUDA] cudaFree released %zu bytes from device %d\n", size, device);
    }
    free(devPtr);
    return cudaSuccess;
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind) {
    FGPU_LOG("[FakeCUDA] cudaMemcpy count=%zu kind=%d\n", count, kind);
    memcpy(dst, src, count);
    return cudaSuccess;
}

cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device) {
    if (!prop) return cudaErrorInvalidValue;
    GlobalState::instance().initialize();

    int count = GlobalState::instance().get_device_count();
    if (device < 0 || device >= count) {
        return cudaErrorInvalidDevice;
    }

    Device& dev = GlobalState::instance().get_device(device);

    // Fill props - zero everything first
    memset(prop, 0, sizeof(cudaDeviceProp));

    // Basic info
    snprintf(prop->name, 256, "%s", dev.name.c_str());

    // UUID (16 bytes)
    memset(prop->uuid, 0, 16);
    snprintf(prop->uuid, 16, "GPU-%08x", device);

    // LUID (8 bytes) - Windows only, zero for Linux
    memset(prop->luid, 0, 8);
    prop->luidDeviceNodeMask = 0;

    prop->totalGlobalMem = dev.total_memory;
    prop->major = 8; // Ampere
    prop->minor = 0;

    // Fill in realistic A100 values
    prop->sharedMemPerBlock = 49152; // 48KB
    prop->regsPerBlock = 65536;
    prop->warpSize = 32;
    prop->memPitch = 2147483647;
    prop->maxThreadsPerBlock = 1024;
    prop->maxThreadsDim[0] = 1024;
    prop->maxThreadsDim[1] = 1024;
    prop->maxThreadsDim[2] = 64;
    prop->maxGridSize[0] = 2147483647;
    prop->maxGridSize[1] = 65535;
    prop->maxGridSize[2] = 65535;
    prop->clockRate = 1410000; // 1.41 GHz
    prop->totalConstMem = 65536; // 64KB
    prop->multiProcessorCount = 108;
    prop->kernelExecTimeoutEnabled = 0;
    prop->integrated = 0;
    prop->canMapHostMemory = 1;
    prop->computeMode = 0;
    prop->concurrentKernels = 1;
    prop->ECCEnabled = 1;
    prop->pciBusID = device;
    prop->pciDeviceID = 0;
    prop->pciDomainID = 0;
    prop->tccDriver = 0;
    prop->asyncEngineCount = 2;
    prop->unifiedAddressing = 1;
    prop->memoryClockRate = 1215000; // 1.215 GHz
    prop->memoryBusWidth = 5120; // 5120-bit
    prop->l2CacheSize = 41943040; // 40MB
    prop->persistingL2CacheMaxSize = 41943040;
    prop->maxThreadsPerMultiProcessor = 2048;
    prop->streamPrioritiesSupported = 1;
    prop->globalL1CacheSupported = 1;
    prop->localL1CacheSupported = 1;
    prop->sharedMemPerMultiprocessor = 167936; // 164KB
    prop->regsPerMultiprocessor = 65536;
    prop->managedMemory = 1;
    prop->isMultiGpuBoard = 0;
    prop->multiGpuBoardGroupID = 0;
    prop->singleToDoublePrecisionPerfRatio = 2;
    prop->pageableMemoryAccess = 1;
    prop->concurrentManagedAccess = 1;
    prop->computePreemptionSupported = 1;
    prop->canUseHostPointerForRegisteredMem = 1;
    prop->cooperativeLaunch = 1;
    prop->cooperativeMultiDeviceLaunch = 1;
    prop->sharedMemPerBlockOptin = 166912;
    prop->directManagedMemAccessFromHost = 1;
    prop->maxBlocksPerMultiProcessor = 32;

    FGPU_LOG("[FakeCUDA] cudaGetDeviceProperties(%d) returning properties\n", device);
    return cudaSuccess;
}

// Stub for kernel launch - simplest signature
// For real/complex apps, they use cudaLaunchKernel (C++ API) which wraps internal C generic calls
// or use the driver API. This is basic runtime stub.
cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) {
    FGPU_LOG("[FakeCUDA] cudaLaunchKernel (stub) called! Grid(%d,%d,%d) Block(%d,%d,%d)\n", 
           gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
    
    // Here lies the main logic: we do NOTHING but log it.
    // If we wanted to fuzz output, we'd need to know which args are pointers.
    
    return cudaSuccess;
}

cudaError_t cudaGetDevice(int *device) {
    if (!device) return cudaErrorInvalidValue;
    *device = GlobalState::instance().get_current_device();
    FGPU_LOG("[FakeCUDA] cudaGetDevice returning %d\n", *device);
    return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
    FGPU_LOG("[FakeCUDA] cudaMemcpyAsync count=%zu kind=%d stream=%p\n", count, kind, stream);
    memcpy(dst, src, count);
    return cudaSuccess;
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
    FGPU_LOG("[FakeCUDA] cudaMemset ptr=%p value=%d count=%zu\n", devPtr, value, count);
    memset(devPtr, value, count);
    return cudaSuccess;
}

cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream) {
    FGPU_LOG("[FakeCUDA] cudaMemsetAsync ptr=%p value=%d count=%zu stream=%p\n", devPtr, value, count, stream);
    memset(devPtr, value, count);
    return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize(void) {
    FGPU_LOG("[FakeCUDA] cudaDeviceSynchronize (no-op)\n");
    return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    FGPU_LOG("[FakeCUDA] cudaStreamSynchronize stream=%p (no-op)\n", stream);
    return cudaSuccess;
}

// Error tracking
static __thread cudaError_t last_error = cudaSuccess;

cudaError_t cudaGetLastError(void) {
    cudaError_t err = last_error;
    last_error = cudaSuccess;  // Clear error
    FGPU_LOG("[FakeCUDA] cudaGetLastError returning %d\n", err);
    return err;
}

cudaError_t cudaPeekAtLastError(void) {
    FGPU_LOG("[FakeCUDA] cudaPeekAtLastError returning %d\n", last_error);
    return last_error;
}

cudaError_t cudaRuntimeGetVersion(int *runtimeVersion) {
    if (!runtimeVersion) return cudaErrorInvalidValue;
    // Report CUDA 12.0 (12000)
    *runtimeVersion = 12000;
    FGPU_LOG("[FakeCUDA] cudaRuntimeGetVersion returning 12000\n");
    return cudaSuccess;
}

cudaError_t cudaDriverGetVersion(int *driverVersion) {
    if (!driverVersion) return cudaErrorInvalidValue;
    // Report CUDA 12.0 (12000)
    *driverVersion = 12000;
    FGPU_LOG("[FakeCUDA] cudaDriverGetVersion returning 12000\n");
    return cudaSuccess;
}

const char* cudaGetErrorString(cudaError_t error) {
    switch (error) {
        case cudaSuccess:
            return "no error";
        case cudaErrorMemoryAllocation:
            return "out of memory";
        case cudaErrorInitializationError:
            return "initialization error";
        case cudaErrorInvalidValue:
            return "invalid argument";
        case cudaErrorInvalidDevice:
            return "invalid device ordinal";
        case cudaErrorNoDevice:
            return "no CUDA-capable device is detected";
        default:
            return "unknown error";
    }
}

const char* cudaGetErrorName(cudaError_t error) {
    switch (error) {
        case cudaSuccess:
            return "cudaSuccess";
        case cudaErrorMemoryAllocation:
            return "cudaErrorMemoryAllocation";
        case cudaErrorInitializationError:
            return "cudaErrorInitializationError";
        case cudaErrorInvalidValue:
            return "cudaErrorInvalidValue";
        case cudaErrorInvalidDevice:
            return "cudaErrorInvalidDevice";
        case cudaErrorNoDevice:
            return "cudaErrorNoDevice";
        default:
            return "cudaErrorUnknown";
    }
}

} // extern C
