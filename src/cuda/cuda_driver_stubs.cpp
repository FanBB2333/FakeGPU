#include "cuda_driver_defs.hpp"
#include "../core/logging.hpp"
#include "../core/logging.hpp"
#include "../core/global_state.hpp"
#include <cstdio>
#include <cstring>
#include <cstdlib>

using namespace fake_gpu;

// Track current context (simplified - just track device)
static int current_context_device = 0;
static bool driver_initialized = false;

extern "C" {

CUresult cuInit(unsigned int Flags) {
    FGPU_LOG("[FakeCUDA-Driver] cuInit called with flags=%u\n", Flags);
    GlobalState::instance().initialize();
    driver_initialized = true;
    FGPU_LOG("[FakeCUDA-Driver] cuInit completed successfully\n");
    return CUDA_SUCCESS;
}

CUresult cuDriverGetVersion(int *driverVersion) {
    if (!driverVersion) return CUDA_ERROR_INVALID_VALUE;
    // Report CUDA 12.0 (12000)
    *driverVersion = 12000;
    FGPU_LOG("[FakeCUDA-Driver] cuDriverGetVersion returning 12000\n");
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetCount(int *count) {
    if (!count) return CUDA_ERROR_INVALID_VALUE;
    GlobalState::instance().initialize();
    *count = GlobalState::instance().get_device_count();
    FGPU_LOG("[FakeCUDA-Driver] cuDeviceGetCount returning %d\n", *count);
    return CUDA_SUCCESS;
}

CUresult cuDeviceGet(CUdevice *device, int ordinal) {
    if (!device) return CUDA_ERROR_INVALID_VALUE;
    GlobalState::instance().initialize();

    int count = GlobalState::instance().get_device_count();
    if (ordinal < 0 || ordinal >= count) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    *device = ordinal;
    FGPU_LOG("[FakeCUDA-Driver] cuDeviceGet(%d) returning device %d\n", ordinal, *device);
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
    if (!name || len <= 0) return CUDA_ERROR_INVALID_VALUE;
    GlobalState::instance().initialize();

    int count = GlobalState::instance().get_device_count();
    if (dev < 0 || dev >= count) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    Device& device = GlobalState::instance().get_device(dev);
    strncpy(name, device.name.c_str(), len - 1);
    name[len - 1] = '\0';
    FGPU_LOG("[FakeCUDA-Driver] cuDeviceGetName(%d) returning '%s'\n", dev, name);
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) {
    if (!pi) return CUDA_ERROR_INVALID_VALUE;
    GlobalState::instance().initialize();

    int count = GlobalState::instance().get_device_count();
    if (dev < 0 || dev >= count) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    // Log attribute queries for debugging
    FGPU_LOG("[FakeCUDA-Driver] cuDeviceGetAttribute(dev=%d, attrib=%d)\n", dev, attrib);

    // Return fake but reasonable values for A100-like GPU
    switch (attrib) {
        case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
            *pi = 1024;
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X:
        case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y:
            *pi = 1024;
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z:
            *pi = 64;
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X:
        case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y:
        case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z:
            *pi = 2147483647;
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK:
            *pi = 49152;  // 48KB
            break;
        case CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY:
            *pi = 65536;  // 64KB
            break;
        case CU_DEVICE_ATTRIBUTE_WARP_SIZE:
            *pi = 32;
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_PITCH:
            *pi = 2147483647;
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK:
            *pi = 65536;
            break;
        case CU_DEVICE_ATTRIBUTE_CLOCK_RATE:
            *pi = 1410000;  // 1.41 GHz
            break;
        case CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT:
            *pi = 512;
            break;
        case CU_DEVICE_ATTRIBUTE_GPU_OVERLAP:
            *pi = 1;
            break;
        case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT:
            *pi = 108;  // A100 has 108 SMs
            break;
        case CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT:
            *pi = 0;  // No timeout
            break;
        case CU_DEVICE_ATTRIBUTE_INTEGRATED:
            *pi = 0;  // Discrete GPU
            break;
        case CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY:
            *pi = 1;
            break;
        case CU_DEVICE_ATTRIBUTE_COMPUTE_MODE:
            *pi = 0;  // Default mode
            break;
        case CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS:
            *pi = 1;
            break;
        case CU_DEVICE_ATTRIBUTE_ECC_ENABLED:
            *pi = 1;
            break;
        case CU_DEVICE_ATTRIBUTE_PCI_BUS_ID:
            *pi = dev;  // Use device index as bus ID
            break;
        case CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID:
            *pi = 0;
            break;
        case CU_DEVICE_ATTRIBUTE_TCC_DRIVER:
            *pi = 0;
            break;
        case CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE:
            *pi = 1215000;  // 1.215 GHz
            break;
        case CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH:
            *pi = 5120;  // 5120-bit for A100
            break;
        case CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE:
            *pi = 41943040;  // 40MB
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR:
            *pi = 2048;
            break;
        case CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT:
            *pi = 2;
            break;
        case CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING:
            *pi = 1;
            break;
        case CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID:
            *pi = 0;
            break;
        case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR:
            *pi = 8;  // Ampere
            break;
        case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR:
            *pi = 0;
            break;
        case CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED:
            *pi = 1;
            break;
        case CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED:
            *pi = 1;
            break;
        case CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED:
            *pi = 1;
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR:
            *pi = 167936;  // 164KB
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR:
            *pi = 65536;
            break;
        case CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY:
            *pi = 1;
            break;
        case CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD:
            *pi = 0;
            break;
        case CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID:
            *pi = 0;
            break;
        case CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO:
            *pi = 2;
            break;
        case CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS:
            *pi = 1;
            break;
        case CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS:
            *pi = 1;
            break;
        case CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED:
            *pi = 1;
            break;
        case CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM:
            *pi = 1;
            break;
        case CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH:
            *pi = 1;
            break;
        case CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH:
            *pi = 1;
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN:
            *pi = 166912;
            break;
        case CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES:
            *pi = 1;
            break;
        case CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED:
            *pi = 1;
            break;
        case CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES:
            *pi = 0;
            break;
        case CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST:
            *pi = 1;
            break;
        case CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED:
            *pi = 1;
            break;
        case CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED:
            *pi = 1;
            break;
        case CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED:
            *pi = 0;
            break;
        case CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED:
            *pi = 0;
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR:
            *pi = 32;
            break;
        case CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED:
            *pi = 1;
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE:
            *pi = 41943040;
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE:
            *pi = 134217728;
            break;
        case CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED:
            *pi = 1;
            break;
        case CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK:
            *pi = 0;
            break;
        case CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED:
            *pi = 1;
            break;
        case CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED:
            *pi = 1;
            break;
        case CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED:
            *pi = 1;
            break;
        case CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED:
            *pi = 1;
            break;
        case CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS:
            *pi = 1;
            break;
        case CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING:
            *pi = 0;
            break;
        case CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES:
            *pi = 1;
            break;
        default:
            // For unknown attributes, return 0 (safe default)
            *pi = 0;
            break;
    }

    return CUDA_SUCCESS;
}

CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
    if (!bytes) return CUDA_ERROR_INVALID_VALUE;
    GlobalState::instance().initialize();

    int count = GlobalState::instance().get_device_count();
    if (dev < 0 || dev >= count) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    Device& device = GlobalState::instance().get_device(dev);
    *bytes = device.total_memory;
    FGPU_LOG("[FakeCUDA-Driver] cuDeviceTotalMem(%d) returning %zu bytes\n", dev, *bytes);
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetUuid(char *uuid, CUdevice dev) {
    if (!uuid) return CUDA_ERROR_INVALID_VALUE;
    GlobalState::instance().initialize();

    int count = GlobalState::instance().get_device_count();
    if (dev < 0 || dev >= count) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    Device& device = GlobalState::instance().get_device(dev);
    strncpy(uuid, device.uuid.c_str(), 64);
    return CUDA_SUCCESS;
}

// Context management (simplified - we don't really need contexts)
CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) {
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    GlobalState::instance().initialize();

    int count = GlobalState::instance().get_device_count();
    if (dev < 0 || dev >= count) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    // Return a fake context pointer (just use device number + 1 to avoid NULL)
    *pctx = (CUcontext)(uintptr_t)(dev + 1);
    current_context_device = dev;
    FGPU_LOG("[FakeCUDA-Driver] cuCtxCreate for device %d, context=%p\n", dev, *pctx);
    return CUDA_SUCCESS;
}

CUresult cuCtxDestroy(CUcontext ctx) {
    FGPU_LOG("[FakeCUDA-Driver] cuCtxDestroy(%p)\n", ctx);
    return CUDA_SUCCESS;
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
    if (ctx == nullptr) {
        current_context_device = 0;
    } else {
        current_context_device = (int)(uintptr_t)ctx - 1;
    }
    FGPU_LOG("[FakeCUDA-Driver] cuCtxSetCurrent(%p) -> device %d\n", ctx, current_context_device);
    return CUDA_SUCCESS;
}

CUresult cuCtxGetCurrent(CUcontext *pctx) {
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    *pctx = (CUcontext)(uintptr_t)(current_context_device + 1);
    return CUDA_SUCCESS;
}

CUresult cuCtxSynchronize(void) {
    FGPU_LOG("[FakeCUDA-Driver] cuCtxSynchronize (no-op)\n");
    return CUDA_SUCCESS;
}

// Memory management
CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
    if (!dptr) return CUDA_ERROR_INVALID_VALUE;

    void* ptr = malloc(bytesize);
    if (!ptr) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    int device = current_context_device;
    if (!GlobalState::instance().register_allocation(ptr, bytesize, device)) {
        free(ptr);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    *dptr = (CUdeviceptr)ptr;
    FGPU_LOG("[FakeCUDA-Driver] cuMemAlloc allocated %zu bytes at 0x%llx on device %d\n",
           bytesize, *dptr, device);
    return CUDA_SUCCESS;
}

CUresult cuMemFree(CUdeviceptr dptr) {
    void* ptr = (void*)dptr;
    size_t size;
    int device;

    if (GlobalState::instance().release_allocation(ptr, size, device)) {
        free(ptr);
        FGPU_LOG("[FakeCUDA-Driver] cuMemFree(0x%llx) released %zu bytes from device %d\n",
               dptr, size, device);
        return CUDA_SUCCESS;
    }

    return CUDA_ERROR_INVALID_VALUE;
}

CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    if (!dstHost || !srcDevice) return CUDA_ERROR_INVALID_VALUE;
    memcpy(dstHost, (void*)srcDevice, ByteCount);
    FGPU_LOG("[FakeCUDA-Driver] cuMemcpyDtoH copied %zu bytes\n", ByteCount);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
    if (!dstDevice || !srcHost) return CUDA_ERROR_INVALID_VALUE;
    memcpy((void*)dstDevice, srcHost, ByteCount);
    FGPU_LOG("[FakeCUDA-Driver] cuMemcpyHtoD copied %zu bytes\n", ByteCount);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
    if (!dstDevice || !srcDevice) return CUDA_ERROR_INVALID_VALUE;
    memcpy((void*)dstDevice, (void*)srcDevice, ByteCount);
    FGPU_LOG("[FakeCUDA-Driver] cuMemcpyDtoD copied %zu bytes\n", ByteCount);
    return CUDA_SUCCESS;
}

// Primary context management
CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    GlobalState::instance().initialize();

    int count = GlobalState::instance().get_device_count();
    if (dev < 0 || dev >= count) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    // Return a fake context pointer (just use device number + 1 to avoid NULL)
    *pctx = (CUcontext)(uintptr_t)(dev + 1);
    current_context_device = dev;
    FGPU_LOG("[FakeCUDA-Driver] cuDevicePrimaryCtxRetain for device %d, context=%p\n", dev, *pctx);
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
    GlobalState::instance().initialize();

    int count = GlobalState::instance().get_device_count();
    if (dev < 0 || dev >= count) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    FGPU_LOG("[FakeCUDA-Driver] cuDevicePrimaryCtxRelease for device %d\n", dev);
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active) {
    GlobalState::instance().initialize();

    int count = GlobalState::instance().get_device_count();
    if (dev < 0 || dev >= count) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    if (flags) *flags = 0;
    if (active) *active = 1;  // Always report as active
    FGPU_LOG("[FakeCUDA-Driver] cuDevicePrimaryCtxGetState for device %d, flags=0, active=1\n", dev);
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags) {
    GlobalState::instance().initialize();

    int count = GlobalState::instance().get_device_count();
    if (dev < 0 || dev >= count) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    FGPU_LOG("[FakeCUDA-Driver] cuDevicePrimaryCtxSetFlags for device %d, flags=%u\n", dev, flags);
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxReset(CUdevice dev) {
    GlobalState::instance().initialize();

    int count = GlobalState::instance().get_device_count();
    if (dev < 0 || dev >= count) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    FGPU_LOG("[FakeCUDA-Driver] cuDevicePrimaryCtxReset for device %d\n", dev);
    return CUDA_SUCCESS;
}

// Context stack management
CUresult cuCtxPushCurrent(CUcontext ctx) {
    if (ctx == nullptr) {
        current_context_device = 0;
    } else {
        current_context_device = (int)(uintptr_t)ctx - 1;
    }
    FGPU_LOG("[FakeCUDA-Driver] cuCtxPushCurrent(%p) -> device %d\n", ctx, current_context_device);
    return CUDA_SUCCESS;
}

CUresult cuCtxPopCurrent(CUcontext *pctx) {
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    *pctx = (CUcontext)(uintptr_t)(current_context_device + 1);
    FGPU_LOG("[FakeCUDA-Driver] cuCtxPopCurrent returning context %p\n", *pctx);
    return CUDA_SUCCESS;
}

// Error handling
CUresult cuGetErrorString(CUresult error, const char **pStr) {
    static const char* error_strings[] = {
        "CUDA_SUCCESS",
        "CUDA_ERROR_INVALID_VALUE",
        "CUDA_ERROR_OUT_OF_MEMORY",
        "CUDA_ERROR_NOT_INITIALIZED",
        "CUDA_ERROR_DEINITIALIZED"
    };

    if (pStr) {
        if (error < 5) {
            *pStr = error_strings[error];
        } else {
            *pStr = "CUDA_ERROR_UNKNOWN";
        }
    }
    return CUDA_SUCCESS;
}

CUresult cuGetErrorName(CUresult error, const char **pStr) {
    return cuGetErrorString(error, pStr);
}

// Stream management
CUresult cuStreamCreate(CUstream *phStream, unsigned int Flags) {
    if (!phStream) return CUDA_ERROR_INVALID_VALUE;
    // Return a fake stream pointer
    *phStream = (CUstream)(uintptr_t)1;
    FGPU_LOG("[FakeCUDA-Driver] cuStreamCreate returning fake stream\n");
    return CUDA_SUCCESS;
}

CUresult cuStreamDestroy(CUstream hStream) {
    FGPU_LOG("[FakeCUDA-Driver] cuStreamDestroy\n");
    return CUDA_SUCCESS;
}

CUresult cuStreamSynchronize(CUstream hStream) {
    FGPU_LOG("[FakeCUDA-Driver] cuStreamSynchronize (no-op)\n");
    return CUDA_SUCCESS;
}

CUresult cuStreamQuery(CUstream hStream) {
    return CUDA_SUCCESS;
}

CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) {
    return CUDA_SUCCESS;
}

CUresult cuStreamGetPriority(CUstream hStream, int *priority) {
    if (priority) *priority = 0;
    return CUDA_SUCCESS;
}

CUresult cuStreamGetFlags(CUstream hStream, unsigned int *flags) {
    if (flags) *flags = 0;
    return CUDA_SUCCESS;
}

CUresult cuStreamGetCtx(CUstream hStream, CUcontext *pctx) {
    if (pctx) *pctx = (CUcontext)(uintptr_t)(current_context_device + 1);
    return CUDA_SUCCESS;
}

// Event management
CUresult cuEventCreate(CUevent *phEvent, unsigned int Flags) {
    if (!phEvent) return CUDA_ERROR_INVALID_VALUE;
    *phEvent = (CUevent)(uintptr_t)1;
    FGPU_LOG("[FakeCUDA-Driver] cuEventCreate returning fake event\n");
    return CUDA_SUCCESS;
}

CUresult cuEventDestroy(CUevent hEvent) {
    FGPU_LOG("[FakeCUDA-Driver] cuEventDestroy\n");
    return CUDA_SUCCESS;
}

CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
    return CUDA_SUCCESS;
}

CUresult cuEventSynchronize(CUevent hEvent) {
    return CUDA_SUCCESS;
}

CUresult cuEventQuery(CUevent hEvent) {
    return CUDA_SUCCESS;
}

CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd) {
    if (pMilliseconds) *pMilliseconds = 0.0f;
    return CUDA_SUCCESS;
}

// Context info
CUresult cuCtxGetDevice(CUdevice *device) {
    if (!device) return CUDA_ERROR_INVALID_VALUE;
    *device = current_context_device;
    return CUDA_SUCCESS;
}

CUresult cuCtxGetFlags(unsigned int *flags) {
    if (flags) *flags = 0;
    return CUDA_SUCCESS;
}

CUresult cuCtxGetLimit(size_t *pvalue, CUlimit limit) {
    if (!pvalue) return CUDA_ERROR_INVALID_VALUE;
    switch (limit) {
        case CU_LIMIT_STACK_SIZE:
            *pvalue = 8192;
            break;
        case CU_LIMIT_PRINTF_FIFO_SIZE:
            *pvalue = 1048576;
            break;
        case CU_LIMIT_MALLOC_HEAP_SIZE:
            *pvalue = 8388608;
            break;
        default:
            *pvalue = 0;
            break;
    }
    return CUDA_SUCCESS;
}

CUresult cuCtxSetLimit(CUlimit limit, size_t value) {
    FGPU_LOG("[FakeCUDA-Driver] cuCtxSetLimit (no-op)\n");
    return CUDA_SUCCESS;
}

CUresult cuCtxGetStreamPriorityRange(int *leastPriority, int *greatestPriority) {
    if (leastPriority) *leastPriority = 0;
    if (greatestPriority) *greatestPriority = -1;
    return CUDA_SUCCESS;
}

CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *version) {
    if (version) *version = 12000;
    return CUDA_SUCCESS;
}

// Memory info
CUresult cuMemGetInfo(size_t *free, size_t *total) {
    GlobalState::instance().initialize();
    Device& dev = GlobalState::instance().get_device(current_context_device);
    if (total) *total = dev.total_memory;
    if (free) *free = dev.total_memory - dev.used_memory;
    FGPU_LOG("[FakeCUDA-Driver] cuMemGetInfo: free=%zu, total=%zu\n",
           free ? *free : 0, total ? *total : 0);
    return CUDA_SUCCESS;
}

// Device UUID (v2)
CUresult cuDeviceGetUuid_v2(CUuuid *uuid, CUdevice dev) {
    if (!uuid) return CUDA_ERROR_INVALID_VALUE;
    GlobalState::instance().initialize();

    int count = GlobalState::instance().get_device_count();
    if (dev < 0 || dev >= count) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    Device& device = GlobalState::instance().get_device(dev);
    // Copy first 16 bytes of UUID string
    memset(uuid->bytes, 0, 16);
    strncpy(uuid->bytes, device.uuid.c_str(), 16);
    return CUDA_SUCCESS;
}

CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev) {
    return cuDeviceTotalMem(bytes, dev);
}

// Additional context functions (v2 versions)
CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev) {
    return cuCtxCreate(pctx, flags, dev);
}

CUresult cuCtxDestroy_v2(CUcontext ctx) {
    return cuCtxDestroy(ctx);
}

CUresult cuCtxPushCurrent_v2(CUcontext ctx) {
    return cuCtxPushCurrent(ctx);
}

CUresult cuCtxPopCurrent_v2(CUcontext *pctx) {
    return cuCtxPopCurrent(pctx);
}

CUresult cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags) {
    return cuDevicePrimaryCtxSetFlags(dev, flags);
}

// Additional stub functions needed by CUDA runtime
CUresult cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev, CUdevice peerDev) {
    if (canAccessPeer) *canAccessPeer = 1;  // Fake: all devices can access each other
    return CUDA_SUCCESS;
}

CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags) {
    return CUDA_SUCCESS;
}

CUresult cuCtxDisablePeerAccess(CUcontext peerContext) {
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev) {
    if (!pciBusId || len <= 0) return CUDA_ERROR_INVALID_VALUE;
    snprintf(pciBusId, len, "0000:%02x:00.0", dev);
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetByPCIBusId(CUdevice *dev, const char *pciBusId) {
    if (!dev) return CUDA_ERROR_INVALID_VALUE;
    *dev = 0;  // Default to device 0
    return CUDA_SUCCESS;
}

CUresult cuModuleLoad(CUmodule *module, const char *fname) {
    if (!module) return CUDA_ERROR_INVALID_VALUE;
    *module = (CUmodule)(uintptr_t)1;
    FGPU_LOG("[FakeCUDA-Driver] cuModuleLoad (fake)\n");
    return CUDA_SUCCESS;
}

CUresult cuModuleLoadData(CUmodule *module, const void *image) {
    if (!module) return CUDA_ERROR_INVALID_VALUE;
    *module = (CUmodule)(uintptr_t)1;
    return CUDA_SUCCESS;
}

CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, void *options, void **optionValues) {
    if (!module) return CUDA_ERROR_INVALID_VALUE;
    *module = (CUmodule)(uintptr_t)1;
    return CUDA_SUCCESS;
}

CUresult cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin) {
    if (!module) return CUDA_ERROR_INVALID_VALUE;
    *module = (CUmodule)(uintptr_t)1;
    return CUDA_SUCCESS;
}

CUresult cuModuleUnload(CUmodule hmod) {
    return CUDA_SUCCESS;
}

CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
    if (!hfunc) return CUDA_ERROR_INVALID_VALUE;
    *hfunc = (CUfunction)(uintptr_t)1;
    return CUDA_SUCCESS;
}

CUresult cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name) {
    if (dptr) *dptr = 0;
    if (bytes) *bytes = 0;
    return CUDA_SUCCESS;
}

CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra) {
    FGPU_LOG("[FakeCUDA-Driver] cuLaunchKernel (no-op)\n");
    return CUDA_SUCCESS;
}

CUresult cuFuncGetAttribute(int *pi, int attrib, CUfunction hfunc) {
    if (pi) *pi = 0;
    return CUDA_SUCCESS;
}

CUresult cuFuncSetAttribute(CUfunction hfunc, int attrib, int value) {
    return CUDA_SUCCESS;
}

CUresult cuFuncSetCacheConfig(CUfunction hfunc, int config) {
    return CUDA_SUCCESS;
}

CUresult cuCtxGetCacheConfig(int *pconfig) {
    if (pconfig) *pconfig = 0;
    return CUDA_SUCCESS;
}

CUresult cuCtxSetCacheConfig(int config) {
    return CUDA_SUCCESS;
}

CUresult cuCtxGetSharedMemConfig(int *pConfig) {
    if (pConfig) *pConfig = 0;
    return CUDA_SUCCESS;
}

CUresult cuCtxSetSharedMemConfig(int config) {
    return CUDA_SUCCESS;
}

CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize, unsigned int flags) {
    return cuMemAlloc(dptr, bytesize);
}

CUresult cuMemAllocHost(void **pp, size_t bytesize) {
    if (!pp) return CUDA_ERROR_INVALID_VALUE;
    *pp = malloc(bytesize);
    if (!*pp) return CUDA_ERROR_OUT_OF_MEMORY;
    return CUDA_SUCCESS;
}

CUresult cuMemFreeHost(void *p) {
    free(p);
    return CUDA_SUCCESS;
}

CUresult cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags) {
    return cuMemAllocHost(pp, bytesize);
}

CUresult cuMemHostRegister(void *p, size_t bytesize, unsigned int Flags) {
    return CUDA_SUCCESS;
}

CUresult cuMemHostUnregister(void *p) {
    return CUDA_SUCCESS;
}

CUresult cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p, unsigned int Flags) {
    if (pdptr) *pdptr = (CUdeviceptr)p;
    return CUDA_SUCCESS;
}

CUresult cuMemHostGetFlags(unsigned int *pFlags, void *p) {
    if (pFlags) *pFlags = 0;
    return CUDA_SUCCESS;
}

CUresult cuPointerGetAttribute(void *data, int attribute, CUdeviceptr ptr) {
    return CUDA_SUCCESS;
}

CUresult cuPointerGetAttributes(unsigned int numAttributes, int *attributes, void **data, CUdeviceptr ptr) {
    return CUDA_SUCCESS;
}

CUresult cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
    memset((void*)dstDevice, uc, N);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD16(CUdeviceptr dstDevice, unsigned short us, size_t N) {
    unsigned short *p = (unsigned short*)dstDevice;
    for (size_t i = 0; i < N; i++) p[i] = us;
    return CUDA_SUCCESS;
}

CUresult cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
    unsigned int *p = (unsigned int*)dstDevice;
    for (size_t i = 0; i < N; i++) p[i] = ui;
    return CUDA_SUCCESS;
}

CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream) {
    return cuMemsetD8(dstDevice, uc, N);
}

CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream) {
    return cuMemsetD32(dstDevice, ui, N);
}

CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream) {
    memcpy((void*)dst, (void*)src, ByteCount);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
    return cuMemcpyDtoH(dstHost, srcDevice, ByteCount);
}

CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream) {
    return cuMemcpyHtoD(dstDevice, srcHost, ByteCount);
}

CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
    return cuMemcpyDtoD(dstDevice, srcDevice, ByteCount);
}

CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
    memcpy((void*)dst, (void*)src, ByteCount);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount) {
    memcpy((void*)dstDevice, (void*)srcDevice, ByteCount);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream) {
    return cuMemcpyPeer(dstDevice, dstContext, srcDevice, srcContext, ByteCount);
}

CUresult cuStreamCreateWithPriority(CUstream *phStream, unsigned int flags, int priority) {
    return cuStreamCreate(phStream, flags);
}

CUresult cuStreamGetId(CUstream hStream, unsigned long long *streamId) {
    if (streamId) *streamId = (unsigned long long)(uintptr_t)hStream;
    return CUDA_SUCCESS;
}

CUresult cuStreamAddCallback(CUstream hStream, void *callback, void *userData, unsigned int flags) {
    return CUDA_SUCCESS;
}

CUresult cuEventDestroy_v2(CUevent hEvent) {
    return cuEventDestroy(hEvent);
}

CUresult cuStreamDestroy_v2(CUstream hStream) {
    return cuStreamDestroy(hStream);
}

CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
    return cuMemAlloc(dptr, bytesize);
}

CUresult cuMemFree_v2(CUdeviceptr dptr) {
    return cuMemFree(dptr);
}

CUresult cuMemGetInfo_v2(size_t *free, size_t *total) {
    return cuMemGetInfo(free, total);
}

CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    return cuMemcpyDtoH(dstHost, srcDevice, ByteCount);
}

CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
    return cuMemcpyHtoD(dstDevice, srcHost, ByteCount);
}

CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
    return cuMemcpyDtoD(dstDevice, srcDevice, ByteCount);
}

CUresult cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
    return cuMemcpyDtoH(dstHost, srcDevice, ByteCount);
}

CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream) {
    return cuMemcpyHtoD(dstDevice, srcHost, ByteCount);
}

CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
    return cuMemcpyDtoD(dstDevice, srcDevice, ByteCount);
}

CUresult cuIpcGetMemHandle(void *pHandle, CUdeviceptr dptr) {
    return CUDA_SUCCESS;
}

CUresult cuIpcOpenMemHandle(CUdeviceptr *pdptr, void *handle, unsigned int Flags) {
    if (pdptr) *pdptr = 0;
    return CUDA_SUCCESS;
}

CUresult cuIpcCloseMemHandle(CUdeviceptr dptr) {
    return CUDA_SUCCESS;
}

CUresult cuIpcGetEventHandle(void *pHandle, CUevent event) {
    return CUDA_SUCCESS;
}

CUresult cuIpcOpenEventHandle(CUevent *phEvent, void *handle) {
    if (phEvent) *phEvent = (CUevent)(uintptr_t)1;
    return CUDA_SUCCESS;
}

// Memory pool functions
CUresult cuDeviceGetDefaultMemPool(CUmemoryPool *pool_out, CUdevice dev) {
    if (pool_out) *pool_out = (CUmemoryPool)(uintptr_t)(dev + 1);
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetMemPool(CUmemoryPool *pool, CUdevice dev) {
    if (pool) *pool = (CUmemoryPool)(uintptr_t)(dev + 1);
    return CUDA_SUCCESS;
}

CUresult cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool) {
    return CUDA_SUCCESS;
}

CUresult cuMemPoolCreate(CUmemoryPool *pool, const void *poolProps) {
    if (pool) *pool = (CUmemoryPool)(uintptr_t)1;
    return CUDA_SUCCESS;
}

CUresult cuMemPoolDestroy(CUmemoryPool pool) {
    return CUDA_SUCCESS;
}

CUresult cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, CUstream hStream) {
    return cuMemAlloc(dptr, bytesize);
}

CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {
    return cuMemFree(dptr);
}

CUresult cuMemAllocFromPoolAsync(CUdeviceptr *dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream) {
    return cuMemAlloc(dptr, bytesize);
}

CUresult cuMemPoolSetAttribute(CUmemoryPool pool, int attr, void *value) {
    return CUDA_SUCCESS;
}

CUresult cuMemPoolGetAttribute(CUmemoryPool pool, int attr, void *value) {
    return CUDA_SUCCESS;
}

CUresult cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep) {
    return CUDA_SUCCESS;
}

CUresult cuCtxResetPersistingL2Cache(void) {
    return CUDA_SUCCESS;
}

CUresult cuMemGetAddressRange(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr) {
    if (pbase) *pbase = dptr;
    if (psize) *psize = 0;
    return CUDA_SUCCESS;
}

CUresult cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes) {
    size_t pitch = (WidthInBytes + 255) & ~255;
    size_t size = pitch * Height;
    CUresult result = cuMemAlloc(dptr, size);
    if (pPitch) *pPitch = pitch;
    return result;
}

CUresult cuDeviceGetP2PAttribute(int *value, int attrib, CUdevice srcDevice, CUdevice dstDevice) {
    if (value) *value = 1;
    return CUDA_SUCCESS;
}

CUresult cuCtxDetach(CUcontext ctx) {
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetTexture1DLinearMaxWidth(size_t *maxWidthInElements, int format, unsigned numChannels, CUdevice dev) {
    if (maxWidthInElements) *maxWidthInElements = 134217728;
    return CUDA_SUCCESS;
}

// cuGetExportTable - internal NVIDIA API used by CUDA Runtime
// This is undocumented and returns internal function tables
// WARNING: Returning NULL causes the real libcudart to crash!
// We need to either provide a real export table or avoid being used with real libcudart
CUresult cuGetExportTable(const void **ppExportTable, const void *pExportTableId) {
    // FGPU_LOG("[FakeCUDA-Driver] cuGetExportTable called with tableId=%p\n", pExportTableId);

    // The CUDA runtime uses various export table IDs to get internal function tables
    // Returning NULL with CUDA_SUCCESS causes segfault in real libcudart
    // Return error to indicate the table is not available
    if (ppExportTable) {
        *ppExportTable = NULL;
    }

    // Return error - this may cause libcudart to fall back to other methods
    // or fail gracefully instead of crashing
    return CUDA_ERROR_NOT_INITIALIZED;
}

// cuGetProcAddress - critical for CUDA runtime to find driver functions
// This is a key function that allows the runtime to dynamically look up driver API functions
CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, unsigned long long flags) {
    if (!symbol || !pfn) return CUDA_ERROR_INVALID_VALUE;

    // Reduce log spam - only log if not found
    // FGPU_LOG("[FakeCUDA-Driver] cuGetProcAddress looking for: %s\n", symbol);

    // Map of function names to their addresses
    #define MAP_FUNC(name) if (strcmp(symbol, #name) == 0) { *pfn = (void*)name; return CUDA_SUCCESS; }

    // Core functions
    MAP_FUNC(cuInit)
    MAP_FUNC(cuDriverGetVersion)
    MAP_FUNC(cuDeviceGetCount)
    MAP_FUNC(cuDeviceGet)
    MAP_FUNC(cuDeviceGetName)
    MAP_FUNC(cuDeviceGetAttribute)
    MAP_FUNC(cuDeviceTotalMem)
    MAP_FUNC(cuDeviceTotalMem_v2)
    MAP_FUNC(cuDeviceGetUuid)
    MAP_FUNC(cuDeviceGetUuid_v2)
    MAP_FUNC(cuDeviceCanAccessPeer)
    MAP_FUNC(cuDeviceGetPCIBusId)
    MAP_FUNC(cuDeviceGetByPCIBusId)

    // Context management
    MAP_FUNC(cuCtxCreate)
    MAP_FUNC(cuCtxCreate_v2)
    MAP_FUNC(cuCtxDestroy)
    MAP_FUNC(cuCtxDestroy_v2)
    MAP_FUNC(cuCtxSetCurrent)
    MAP_FUNC(cuCtxGetCurrent)
    MAP_FUNC(cuCtxSynchronize)
    MAP_FUNC(cuCtxPushCurrent)
    MAP_FUNC(cuCtxPushCurrent_v2)
    MAP_FUNC(cuCtxPopCurrent)
    MAP_FUNC(cuCtxPopCurrent_v2)
    MAP_FUNC(cuCtxGetDevice)
    MAP_FUNC(cuCtxGetFlags)
    MAP_FUNC(cuCtxGetLimit)
    MAP_FUNC(cuCtxSetLimit)
    MAP_FUNC(cuCtxGetStreamPriorityRange)
    MAP_FUNC(cuCtxGetApiVersion)
    MAP_FUNC(cuCtxEnablePeerAccess)
    MAP_FUNC(cuCtxDisablePeerAccess)
    MAP_FUNC(cuCtxGetCacheConfig)
    MAP_FUNC(cuCtxSetCacheConfig)
    MAP_FUNC(cuCtxGetSharedMemConfig)
    MAP_FUNC(cuCtxSetSharedMemConfig)

    // Primary context
    MAP_FUNC(cuDevicePrimaryCtxRetain)
    MAP_FUNC(cuDevicePrimaryCtxRelease)
    MAP_FUNC(cuDevicePrimaryCtxGetState)
    MAP_FUNC(cuDevicePrimaryCtxSetFlags)
    MAP_FUNC(cuDevicePrimaryCtxSetFlags_v2)
    MAP_FUNC(cuDevicePrimaryCtxReset)

    // Memory management
    MAP_FUNC(cuMemAlloc)
    MAP_FUNC(cuMemAlloc_v2)
    MAP_FUNC(cuMemFree)
    MAP_FUNC(cuMemFree_v2)
    MAP_FUNC(cuMemGetInfo)
    MAP_FUNC(cuMemGetInfo_v2)
    MAP_FUNC(cuMemAllocManaged)
    MAP_FUNC(cuMemAllocHost)
    MAP_FUNC(cuMemFreeHost)
    MAP_FUNC(cuMemHostAlloc)
    MAP_FUNC(cuMemHostRegister)
    MAP_FUNC(cuMemHostUnregister)
    MAP_FUNC(cuMemHostGetDevicePointer)
    MAP_FUNC(cuMemHostGetFlags)
    MAP_FUNC(cuPointerGetAttribute)
    MAP_FUNC(cuPointerGetAttributes)

    // Memory copy
    MAP_FUNC(cuMemcpy)
    MAP_FUNC(cuMemcpyDtoH)
    MAP_FUNC(cuMemcpyDtoH_v2)
    MAP_FUNC(cuMemcpyHtoD)
    MAP_FUNC(cuMemcpyHtoD_v2)
    MAP_FUNC(cuMemcpyDtoD)
    MAP_FUNC(cuMemcpyDtoD_v2)
    MAP_FUNC(cuMemcpyAsync)
    MAP_FUNC(cuMemcpyDtoHAsync)
    MAP_FUNC(cuMemcpyDtoHAsync_v2)
    MAP_FUNC(cuMemcpyHtoDAsync)
    MAP_FUNC(cuMemcpyHtoDAsync_v2)
    MAP_FUNC(cuMemcpyDtoDAsync)
    MAP_FUNC(cuMemcpyDtoDAsync_v2)
    MAP_FUNC(cuMemcpyPeer)
    MAP_FUNC(cuMemcpyPeerAsync)

    // Memset
    MAP_FUNC(cuMemsetD8)
    MAP_FUNC(cuMemsetD16)
    MAP_FUNC(cuMemsetD32)
    MAP_FUNC(cuMemsetD8Async)
    MAP_FUNC(cuMemsetD32Async)

    // Stream management
    MAP_FUNC(cuStreamCreate)
    MAP_FUNC(cuStreamCreateWithPriority)
    MAP_FUNC(cuStreamDestroy)
    MAP_FUNC(cuStreamDestroy_v2)
    MAP_FUNC(cuStreamSynchronize)
    MAP_FUNC(cuStreamQuery)
    MAP_FUNC(cuStreamWaitEvent)
    MAP_FUNC(cuStreamGetPriority)
    MAP_FUNC(cuStreamGetFlags)
    MAP_FUNC(cuStreamGetCtx)
    MAP_FUNC(cuStreamGetId)
    MAP_FUNC(cuStreamAddCallback)

    // Event management
    MAP_FUNC(cuEventCreate)
    MAP_FUNC(cuEventDestroy)
    MAP_FUNC(cuEventDestroy_v2)
    MAP_FUNC(cuEventRecord)
    MAP_FUNC(cuEventSynchronize)
    MAP_FUNC(cuEventQuery)
    MAP_FUNC(cuEventElapsedTime)

    // Module/Function
    MAP_FUNC(cuModuleLoad)
    MAP_FUNC(cuModuleLoadData)
    MAP_FUNC(cuModuleLoadDataEx)
    MAP_FUNC(cuModuleLoadFatBinary)
    MAP_FUNC(cuModuleUnload)
    MAP_FUNC(cuModuleGetFunction)
    MAP_FUNC(cuModuleGetGlobal)
    MAP_FUNC(cuLaunchKernel)
    MAP_FUNC(cuFuncGetAttribute)
    MAP_FUNC(cuFuncSetAttribute)
    MAP_FUNC(cuFuncSetCacheConfig)

    // IPC
    MAP_FUNC(cuIpcGetMemHandle)
    MAP_FUNC(cuIpcOpenMemHandle)
    MAP_FUNC(cuIpcCloseMemHandle)
    MAP_FUNC(cuIpcGetEventHandle)
    MAP_FUNC(cuIpcOpenEventHandle)

    // Error handling
    MAP_FUNC(cuGetErrorString)
    MAP_FUNC(cuGetErrorName)
    MAP_FUNC(cuGetProcAddress)
    MAP_FUNC(cuGetProcAddress_v2)
    MAP_FUNC(cuGetExportTable)

    // Memory pool functions
    MAP_FUNC(cuDeviceGetDefaultMemPool)
    MAP_FUNC(cuDeviceGetMemPool)
    MAP_FUNC(cuDeviceSetMemPool)
    MAP_FUNC(cuMemPoolCreate)
    MAP_FUNC(cuMemPoolDestroy)
    MAP_FUNC(cuMemAllocAsync)
    MAP_FUNC(cuMemFreeAsync)
    MAP_FUNC(cuMemAllocFromPoolAsync)
    MAP_FUNC(cuMemPoolSetAttribute)
    MAP_FUNC(cuMemPoolGetAttribute)
    MAP_FUNC(cuMemPoolTrimTo)
    MAP_FUNC(cuCtxResetPersistingL2Cache)
    MAP_FUNC(cuMemGetAddressRange)
    MAP_FUNC(cuMemAllocPitch)
    MAP_FUNC(cuDeviceGetP2PAttribute)
    MAP_FUNC(cuCtxDetach)
    MAP_FUNC(cuDeviceGetTexture1DLinearMaxWidth)

    #undef MAP_FUNC

    // For unknown symbols, return NULL but success (some symbols are optional)
    // FGPU_LOG("[FakeCUDA-Driver] cuGetProcAddress: symbol '%s' NOT FOUND\n", symbol);
    *pfn = NULL;
    return CUDA_SUCCESS;
}

CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion, unsigned long long flags, void *symbolStatus) {
    CUresult result = cuGetProcAddress(symbol, pfn, cudaVersion, flags);
    // symbolStatus is used to indicate if the symbol was found
    // We ignore it for now
    return result;
}

} // extern "C"
