#include "cuda_driver_defs.hpp"
#include "cuda_driver_passthrough.hpp"
#include "../core/backend_config.hpp"
#include "../core/global_state.hpp"
#include "../core/hybrid_memory_manager.hpp"
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

using namespace fake_gpu;

// Track current context (simplified - just track device)
static int current_context_device = 0;
static bool driver_initialized = false;

namespace {

bool real_driver_available() {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() == FakeGpuMode::Simulate) return false;
    return CudaDriverPassthrough::instance().initialize();
}

int map_virtual_device_to_real(int virtual_device) {
    const auto& info = RealCudaLoader::instance().get_real_gpu_info();
    if (!info.valid || info.device_count <= 0) return virtual_device;
    int mapped = virtual_device % info.device_count;
    if (mapped < 0) mapped += info.device_count;
    return mapped;
}

std::mutex g_ctx_mutex;
std::unordered_map<CUcontext, int> g_ctx_to_virtual_device;

void track_context_mapping(CUcontext ctx, int virtual_device) {
    std::lock_guard<std::mutex> lock(g_ctx_mutex);
    g_ctx_to_virtual_device[ctx] = virtual_device;
}

int lookup_context_mapping(CUcontext ctx) {
    std::lock_guard<std::mutex> lock(g_ctx_mutex);
    auto it = g_ctx_to_virtual_device.find(ctx);
    if (it == g_ctx_to_virtual_device.end()) return -1;
    return it->second;
}

} // namespace

extern "C" {

CUresult cuInit(unsigned int Flags) {
    FGPU_LOG("[FakeCUDA-Driver] cuInit called with flags=%u\n", Flags);

    const BackendConfig& config = BackendConfig::instance();

    // Always initialize the fake state: hybrid/simulate virtualize device info using GlobalState.
    GlobalState::instance().initialize();
    driver_initialized = true;

    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        if (config.mode() == FakeGpuMode::Hybrid) {
            HybridMemoryManager::instance().initialize();
        }
        return CudaDriverPassthrough::instance().cuInit(Flags);
    }

    FGPU_LOG("[FakeCUDA-Driver] cuInit completed successfully (fake)\n");
    return CUDA_SUCCESS;
}

CUresult cuDriverGetVersion(int *driverVersion) {
    if (!driverVersion) return CUDA_ERROR_INVALID_VALUE;

    if (real_driver_available()) {
        CUresult result = CudaDriverPassthrough::instance().cuDriverGetVersion(driverVersion);
        if (result == CUDA_SUCCESS) {
            FGPU_LOG("[FakeCUDA-Driver] cuDriverGetVersion (real) returning %d\n", *driverVersion);
            return result;
        }
    }

    // Fallback: Report CUDA 12.0 (12000)
    *driverVersion = 12000;
    FGPU_LOG("[FakeCUDA-Driver] cuDriverGetVersion (fake) returning 12000\n");
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetCount(int *count) {
    if (!count) return CUDA_ERROR_INVALID_VALUE;

    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() == FakeGpuMode::Passthrough && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuDeviceGetCount(count);
    }

    GlobalState::instance().initialize();
    *count = GlobalState::instance().get_device_count();
    FGPU_LOG("[FakeCUDA-Driver] cuDeviceGetCount (fake) returning %d\n", *count);
    return CUDA_SUCCESS;
}

CUresult cuDeviceGet(CUdevice *device, int ordinal) {
    if (!device) return CUDA_ERROR_INVALID_VALUE;

    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() == FakeGpuMode::Passthrough && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuDeviceGet(device, ordinal);
    }

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

    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() == FakeGpuMode::Passthrough && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuDeviceGetName(name, len, dev);
    }

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

    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() == FakeGpuMode::Passthrough && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuDeviceGetAttribute(pi, attrib, dev);
    }

    GlobalState::instance().initialize();

    int count = GlobalState::instance().get_device_count();
    if (dev < 0 || dev >= count) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    Device& device = GlobalState::instance().get_device(dev);

    // Log attribute queries for debugging
    FGPU_LOG("[FakeCUDA-Driver] cuDeviceGetAttribute(dev=%d, attrib=%d)\n", dev, attrib);

    // Return fake but reasonable values pulled from the device profile
    int real_cc_major = -1;
    int real_cc_minor = -1;
    if (config.mode() == FakeGpuMode::Hybrid && real_driver_available()) {
        const auto& real_info = RealCudaLoader::instance().get_real_gpu_info();
        if (real_info.valid && real_info.device_count > 0) {
            const int real_dev = map_virtual_device_to_real(dev);
            real_cc_major = real_info.compute_major[real_dev];
            real_cc_minor = real_info.compute_minor[real_dev];
        }
    }

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
            *pi = device.profile.shared_mem_per_block;
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
            *pi = device.profile.regs_per_block;
            break;
        case CU_DEVICE_ATTRIBUTE_CLOCK_RATE:
            *pi = device.profile.core_clock_mhz * 1000;  // kHz
            break;
        case CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT:
            *pi = 512;
            break;
        case CU_DEVICE_ATTRIBUTE_GPU_OVERLAP:
            *pi = 1;
            break;
        case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT:
            *pi = device.profile.sm_count;
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
            *pi = static_cast<int>(device.profile.pci_device_id);
            break;
        case CU_DEVICE_ATTRIBUTE_TCC_DRIVER:
            *pi = 0;
            break;
        case CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE:
            *pi = device.profile.memory_clock_mhz * 1000;  // kHz
            break;
        case CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH:
            *pi = device.profile.memory_bus_width_bits;
            break;
        case CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE:
            *pi = device.profile.l2_cache_bytes;
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR:
            *pi = device.profile.max_threads_per_multiprocessor;
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
            *pi = device.profile.compute_major;
            if (real_cc_major >= 0 && device.profile.compute_major > real_cc_major) {
                *pi = real_cc_major;
            }
            break;
        case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR:
            *pi = device.profile.compute_minor;
            if (real_cc_major >= 0) {
                if (device.profile.compute_major > real_cc_major) {
                    *pi = real_cc_minor;
                } else if (device.profile.compute_major == real_cc_major &&
                           device.profile.compute_minor > real_cc_minor) {
                    *pi = real_cc_minor;
                }
            }
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
            *pi = device.profile.shared_mem_per_sm;
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR:
            *pi = device.profile.regs_per_multiprocessor;
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
            *pi = device.profile.shared_mem_per_block_optin;
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
            *pi = device.profile.max_blocks_per_multiprocessor;
            break;
        case CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED:
            *pi = 1;
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE:
            *pi = device.profile.l2_cache_bytes;
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

    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() == FakeGpuMode::Passthrough && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuDeviceTotalMem(bytes, dev);
    }

    GlobalState::instance().initialize();

    int count = GlobalState::instance().get_device_count();
    if (dev < 0 || dev >= count) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    Device& device = GlobalState::instance().get_device(dev);
    size_t total = device.total_memory;
    if (config.mode() == FakeGpuMode::Hybrid && config.oom_policy() == OomPolicy::Clamp && real_driver_available()) {
        HybridMemoryManager::instance().initialize();
        const int real_dev = map_virtual_device_to_real(dev);
        const size_t real_total = HybridMemoryManager::instance().get_real_total_memory(real_dev);
        if (real_total > 0) {
            total = std::min(total, real_total);
        }
    }
    *bytes = total;
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

    const BackendConfig& config = BackendConfig::instance();

    if (config.mode() == FakeGpuMode::Passthrough && real_driver_available()) {
        CUresult result = CudaDriverPassthrough::instance().cuCtxCreate(pctx, flags, dev);
        if (result == CUDA_SUCCESS) {
            current_context_device = dev;
            track_context_mapping(*pctx, dev);
        }
        return result;
    }

    GlobalState::instance().initialize();
    int count = GlobalState::instance().get_device_count();
    if (dev < 0 || dev >= count) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    if (config.mode() == FakeGpuMode::Hybrid && real_driver_available()) {
        HybridMemoryManager::instance().initialize();
        const int real_dev = map_virtual_device_to_real(dev);
        CUresult result = CudaDriverPassthrough::instance().cuCtxCreate(pctx, flags, real_dev);
        if (result == CUDA_SUCCESS) {
            current_context_device = dev;
            GlobalState::instance().set_current_device(dev);
            track_context_mapping(*pctx, dev);
        }
        return result;
    }

    // Simulate mode - Return a fake context pointer (just use device number + 1 to avoid NULL)
    *pctx = (CUcontext)(uintptr_t)(dev + 1);
    current_context_device = dev;
    GlobalState::instance().set_current_device(dev);
    FGPU_LOG("[FakeCUDA-Driver] cuCtxCreate for device %d, context=%p\n", dev, *pctx);
    return CUDA_SUCCESS;
}

CUresult cuCtxDestroy(CUcontext ctx) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuCtxDestroy(ctx);
    }
    FGPU_LOG("[FakeCUDA-Driver] cuCtxDestroy(%p)\n", ctx);
    return CUDA_SUCCESS;
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        CUresult result = CudaDriverPassthrough::instance().cuCtxSetCurrent(ctx);
        if (result == CUDA_SUCCESS) {
            if (ctx == nullptr) {
                current_context_device = 0;
            } else {
                const int mapped = lookup_context_mapping(ctx);
                if (mapped >= 0) {
                    current_context_device = mapped;
                }
            }
            if (config.virtualize_device_info()) {
                GlobalState::instance().set_current_device(current_context_device);
            }
        }
        return result;
    }

    if (ctx == nullptr) {
        current_context_device = 0;
    } else {
        current_context_device = (int)(uintptr_t)ctx - 1;
    }
    GlobalState::instance().set_current_device(current_context_device);
    FGPU_LOG("[FakeCUDA-Driver] cuCtxSetCurrent(%p) -> device %d\n", ctx, current_context_device);
    return CUDA_SUCCESS;
}

CUresult cuCtxGetCurrent(CUcontext *pctx) {
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuCtxGetCurrent(pctx);
    }
    *pctx = (CUcontext)(uintptr_t)(current_context_device + 1);
    return CUDA_SUCCESS;
}

CUresult cuCtxSynchronize(void) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuCtxSynchronize();
    }
    FGPU_LOG("[FakeCUDA-Driver] cuCtxSynchronize (no-op)\n");
    return CUDA_SUCCESS;
}

// Memory management
CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
    if (!dptr) return CUDA_ERROR_INVALID_VALUE;

    const BackendConfig& config = BackendConfig::instance();

    if (config.mode() == FakeGpuMode::Passthrough && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuMemAlloc(dptr, bytesize);
    }

    if (config.mode() == FakeGpuMode::Hybrid && real_driver_available()) {
        GlobalState::instance().initialize();
        HybridMemoryManager::instance().initialize();

        const int virtual_device = current_context_device;
        auto decision = HybridMemoryManager::instance().decide_allocation(virtual_device, bytesize);

        constexpr unsigned int kManagedAttachGlobal = 1; // CU_MEM_ATTACH_GLOBAL
        constexpr unsigned int kHostAllocDevicemap = 2;  // CU_MEMHOSTALLOC_DEVICEMAP

        CUresult result = CUDA_SUCCESS;
        HybridAllocation::Type alloc_type = HybridAllocation::Type::RealDevice;
        void* backing_ptr = nullptr;
        CUdeviceptr out_ptr = 0;

        switch (decision) {
            case HybridMemoryManager::AllocationDecision::UseReal: {
                result = CudaDriverPassthrough::instance().cuMemAlloc(&out_ptr, bytesize);
                alloc_type = HybridAllocation::Type::RealDevice;
                break;
            }
            case HybridMemoryManager::AllocationDecision::UseManaged: {
                result = CudaDriverPassthrough::instance().cuMemAllocManaged(&out_ptr, bytesize, kManagedAttachGlobal);
                alloc_type = HybridAllocation::Type::Managed;
                break;
            }
            case HybridMemoryManager::AllocationDecision::UseMappedHost: {
                void* host_ptr = nullptr;
                result = CudaDriverPassthrough::instance().cuMemHostAlloc(&host_ptr, bytesize, kHostAllocDevicemap);
                if (result != CUDA_SUCCESS) return result;
                CUdeviceptr device_ptr = 0;
                result = CudaDriverPassthrough::instance().cuMemHostGetDevicePointer(&device_ptr, host_ptr, 0);
                if (result != CUDA_SUCCESS) {
                    CudaDriverPassthrough::instance().cuMemFreeHost(host_ptr);
                    return result;
                }
                backing_ptr = host_ptr;
                out_ptr = device_ptr;
                alloc_type = HybridAllocation::Type::MappedHost;
                break;
            }
            case HybridMemoryManager::AllocationDecision::SpillToCpu: {
                void* ptr = malloc(bytesize);
                if (!ptr) return CUDA_ERROR_OUT_OF_MEMORY;
                out_ptr = (CUdeviceptr)ptr;
                alloc_type = HybridAllocation::Type::SpilledCpu;
                break;
            }
            case HybridMemoryManager::AllocationDecision::Fail:
                return CUDA_ERROR_OUT_OF_MEMORY;
        }

        if (result != CUDA_SUCCESS) return result;

        bool registered = false;
        if (alloc_type == HybridAllocation::Type::Managed) {
            registered = GlobalState::instance().register_managed_allocation((void*)out_ptr, bytesize, virtual_device);
        } else {
            registered = GlobalState::instance().register_allocation((void*)out_ptr, bytesize, virtual_device);
        }

        if (!registered) {
            switch (alloc_type) {
                case HybridAllocation::Type::RealDevice:
                case HybridAllocation::Type::Managed:
                    CudaDriverPassthrough::instance().cuMemFree(out_ptr);
                    break;
                case HybridAllocation::Type::MappedHost:
                    if (backing_ptr) {
                        CudaDriverPassthrough::instance().cuMemFreeHost(backing_ptr);
                    }
                    break;
                case HybridAllocation::Type::SpilledCpu:
                    free((void*)out_ptr);
                    break;
            }
            return CUDA_ERROR_OUT_OF_MEMORY;
        }

        *dptr = out_ptr;
        HybridMemoryManager::instance().record_allocation((void*)out_ptr, bytesize, virtual_device, alloc_type, backing_ptr);
        return CUDA_SUCCESS;
    }

    // Simulate mode: allocate in host memory.
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
    // Handle NULL pointer
    if (dptr == 0) {
        FGPU_LOG("[FakeCUDA-Driver] cuMemFree(NULL) - ignoring\n");
        return CUDA_SUCCESS;
    }

    const BackendConfig& config = BackendConfig::instance();

    if (config.mode() == FakeGpuMode::Passthrough && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuMemFree(dptr);
    }

    if (config.mode() == FakeGpuMode::Hybrid && real_driver_available()) {
        HybridAllocation alloc;
        if (HybridMemoryManager::instance().release_allocation((void*)dptr, alloc)) {
            size_t size = 0;
            int device = -1;
            GlobalState::instance().release_allocation((void*)dptr, size, device);

            switch (alloc.type) {
                case HybridAllocation::Type::RealDevice:
                case HybridAllocation::Type::Managed:
                    return CudaDriverPassthrough::instance().cuMemFree(dptr);
                case HybridAllocation::Type::MappedHost:
                    return CudaDriverPassthrough::instance().cuMemFreeHost(alloc.backing_ptr);
                case HybridAllocation::Type::SpilledCpu:
                    free((void*)dptr);
                    return CUDA_SUCCESS;
            }
        }
        // Not tracked: fall back to real free if possible.
        return CudaDriverPassthrough::instance().cuMemFree(dptr);
    }

    void* ptr = (void*)dptr;
    size_t size;
    int device;

    if (GlobalState::instance().release_allocation(ptr, size, device)) {
        free(ptr);
        FGPU_LOG("[FakeCUDA-Driver] cuMemFree(0x%llx) released %zu bytes from device %d\n",
               dptr, size, device);
        return CUDA_SUCCESS;
    }

    // PyTorch's CachingAllocator may try to free pointers it doesn't own
    // Return success to avoid crashes, but log a warning
    FGPU_LOG("[FakeCUDA-Driver] cuMemFree(0x%llx) - pointer not tracked, assuming already freed\n", dptr);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    if (!dstHost || !srcDevice) return CUDA_ERROR_INVALID_VALUE;

    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() == FakeGpuMode::Hybrid && real_driver_available()) {
        HybridAllocation alloc;
        if (HybridMemoryManager::instance().get_allocation_info((void*)srcDevice, alloc)) {
            if (alloc.type == HybridAllocation::Type::MappedHost && alloc.backing_ptr) {
                memcpy(dstHost, alloc.backing_ptr, ByteCount);
                GlobalState::instance().record_memcpy_d2h((void*)srcDevice, ByteCount);
                return CUDA_SUCCESS;
            }
            if (alloc.type == HybridAllocation::Type::SpilledCpu) {
                memcpy(dstHost, (void*)srcDevice, ByteCount);
                GlobalState::instance().record_memcpy_d2h((void*)srcDevice, ByteCount);
                return CUDA_SUCCESS;
            }
        }
        CUresult result = CudaDriverPassthrough::instance().cuMemcpyDtoH(dstHost, srcDevice, ByteCount);
        if (result == CUDA_SUCCESS) {
            GlobalState::instance().record_memcpy_d2h((void*)srcDevice, ByteCount);
        }
        return result;
    }

    if (config.mode() == FakeGpuMode::Passthrough && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuMemcpyDtoH(dstHost, srcDevice, ByteCount);
    }

    memcpy(dstHost, (void*)srcDevice, ByteCount);
    GlobalState::instance().record_memcpy_d2h((void*)srcDevice, ByteCount);
    FGPU_LOG("[FakeCUDA-Driver] cuMemcpyDtoH copied %zu bytes\n", ByteCount);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
    if (!dstDevice || !srcHost) return CUDA_ERROR_INVALID_VALUE;

    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() == FakeGpuMode::Hybrid && real_driver_available()) {
        HybridAllocation alloc;
        if (HybridMemoryManager::instance().get_allocation_info((void*)dstDevice, alloc)) {
            if (alloc.type == HybridAllocation::Type::MappedHost && alloc.backing_ptr) {
                memcpy(alloc.backing_ptr, srcHost, ByteCount);
                GlobalState::instance().record_memcpy_h2d((void*)dstDevice, ByteCount);
                return CUDA_SUCCESS;
            }
            if (alloc.type == HybridAllocation::Type::SpilledCpu) {
                memcpy((void*)dstDevice, srcHost, ByteCount);
                GlobalState::instance().record_memcpy_h2d((void*)dstDevice, ByteCount);
                return CUDA_SUCCESS;
            }
        }
        CUresult result = CudaDriverPassthrough::instance().cuMemcpyHtoD(dstDevice, srcHost, ByteCount);
        if (result == CUDA_SUCCESS) {
            GlobalState::instance().record_memcpy_h2d((void*)dstDevice, ByteCount);
        }
        return result;
    }

    if (config.mode() == FakeGpuMode::Passthrough && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuMemcpyHtoD(dstDevice, srcHost, ByteCount);
    }

    memcpy((void*)dstDevice, srcHost, ByteCount);
    GlobalState::instance().record_memcpy_h2d((void*)dstDevice, ByteCount);
    FGPU_LOG("[FakeCUDA-Driver] cuMemcpyHtoD copied %zu bytes\n", ByteCount);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
    if (!dstDevice || !srcDevice) return CUDA_ERROR_INVALID_VALUE;

    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() == FakeGpuMode::Hybrid && real_driver_available()) {
        auto host_ptr_for = [&](CUdeviceptr ptr) -> void* {
            HybridAllocation alloc;
            if (!HybridMemoryManager::instance().get_allocation_info((void*)ptr, alloc)) return nullptr;
            if (alloc.type == HybridAllocation::Type::MappedHost) return alloc.backing_ptr;
            if (alloc.type == HybridAllocation::Type::SpilledCpu) return (void*)ptr;
            return nullptr;
        };

        void* dst_host = host_ptr_for(dstDevice);
        void* src_host = host_ptr_for(srcDevice);

        if (dst_host && src_host) {
            memcpy(dst_host, src_host, ByteCount);
            GlobalState::instance().record_memcpy_d2d((void*)dstDevice, (void*)srcDevice, ByteCount);
            return CUDA_SUCCESS;
        }
        if (dst_host && !src_host) {
            CUresult result = CudaDriverPassthrough::instance().cuMemcpyDtoH(dst_host, srcDevice, ByteCount);
            if (result == CUDA_SUCCESS) {
                GlobalState::instance().record_memcpy_d2d((void*)dstDevice, (void*)srcDevice, ByteCount);
            }
            return result;
        }
        if (!dst_host && src_host) {
            CUresult result = CudaDriverPassthrough::instance().cuMemcpyHtoD(dstDevice, src_host, ByteCount);
            if (result == CUDA_SUCCESS) {
                GlobalState::instance().record_memcpy_d2d((void*)dstDevice, (void*)srcDevice, ByteCount);
            }
            return result;
        }

        CUresult result = CudaDriverPassthrough::instance().cuMemcpyDtoD(dstDevice, srcDevice, ByteCount);
        if (result == CUDA_SUCCESS) {
            GlobalState::instance().record_memcpy_d2d((void*)dstDevice, (void*)srcDevice, ByteCount);
        }
        return result;
    }

    if (config.mode() == FakeGpuMode::Passthrough && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuMemcpyDtoD(dstDevice, srcDevice, ByteCount);
    }

    memcpy((void*)dstDevice, (void*)srcDevice, ByteCount);
    GlobalState::instance().record_memcpy_d2d((void*)dstDevice, (void*)srcDevice, ByteCount);
    FGPU_LOG("[FakeCUDA-Driver] cuMemcpyDtoD copied %zu bytes\n", ByteCount);
    return CUDA_SUCCESS;
}

// Primary context management
CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;

    const BackendConfig& config = BackendConfig::instance();

    if (config.mode() == FakeGpuMode::Passthrough && real_driver_available()) {
        CUresult result = CudaDriverPassthrough::instance().cuDevicePrimaryCtxRetain(pctx, dev);
        if (result == CUDA_SUCCESS) {
            current_context_device = dev;
            track_context_mapping(*pctx, dev);
        }
        return result;
    }

    GlobalState::instance().initialize();
    int count = GlobalState::instance().get_device_count();
    if (dev < 0 || dev >= count) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    if (config.mode() == FakeGpuMode::Hybrid && real_driver_available()) {
        HybridMemoryManager::instance().initialize();
        const int real_dev = map_virtual_device_to_real(dev);
        CUresult result = CudaDriverPassthrough::instance().cuDevicePrimaryCtxRetain(pctx, real_dev);
        if (result == CUDA_SUCCESS) {
            current_context_device = dev;
            GlobalState::instance().set_current_device(dev);
            track_context_mapping(*pctx, dev);
        }
        return result;
    }

    // Simulate mode: Return a fake context pointer (just use device number + 1 to avoid NULL)
    *pctx = (CUcontext)(uintptr_t)(dev + 1);
    current_context_device = dev;
    GlobalState::instance().set_current_device(dev);
    FGPU_LOG("[FakeCUDA-Driver] cuDevicePrimaryCtxRetain for device %d, context=%p\n", dev, *pctx);
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
    const BackendConfig& config = BackendConfig::instance();

    if (config.mode() == FakeGpuMode::Passthrough && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuDevicePrimaryCtxRelease(dev);
    }

    GlobalState::instance().initialize();
    int count = GlobalState::instance().get_device_count();
    if (dev < 0 || dev >= count) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    if (config.mode() == FakeGpuMode::Hybrid && real_driver_available()) {
        const int real_dev = map_virtual_device_to_real(dev);
        return CudaDriverPassthrough::instance().cuDevicePrimaryCtxRelease(real_dev);
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
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        const CUdevice real_dev = (config.mode() == FakeGpuMode::Hybrid) ? map_virtual_device_to_real(dev) : dev;
        void* fn = CudaDriverPassthrough::instance().getRealFunction("cuDevicePrimaryCtxSetFlags_v2");
        if (!fn) fn = CudaDriverPassthrough::instance().getRealFunction("cuDevicePrimaryCtxSetFlags");
        if (fn) {
            typedef CUresult (*fn_t)(CUdevice, unsigned int);
            return ((fn_t)fn)(real_dev, flags);
        }
    }

    GlobalState::instance().initialize();
    int count = GlobalState::instance().get_device_count();
    if (dev < 0 || dev >= count) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    FGPU_LOG("[FakeCUDA-Driver] cuDevicePrimaryCtxSetFlags for device %d, flags=%u\n", dev, flags);
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxReset(CUdevice dev) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        const CUdevice real_dev = (config.mode() == FakeGpuMode::Hybrid) ? map_virtual_device_to_real(dev) : dev;
        return CudaDriverPassthrough::instance().cuDevicePrimaryCtxReset(real_dev);
    }

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

    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuStreamCreate(phStream, Flags);
    }

    // Return a fake stream pointer
    *phStream = (CUstream)(uintptr_t)1;
    FGPU_LOG("[FakeCUDA-Driver] cuStreamCreate returning fake stream\n");
    return CUDA_SUCCESS;
}

CUresult cuStreamDestroy(CUstream hStream) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuStreamDestroy(hStream);
    }
    FGPU_LOG("[FakeCUDA-Driver] cuStreamDestroy\n");
    return CUDA_SUCCESS;
}

CUresult cuStreamSynchronize(CUstream hStream) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuStreamSynchronize(hStream);
    }
    FGPU_LOG("[FakeCUDA-Driver] cuStreamSynchronize (no-op)\n");
    return CUDA_SUCCESS;
}

CUresult cuStreamQuery(CUstream hStream) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuStreamQuery(hStream);
    }
    return CUDA_SUCCESS;
}

CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuStreamWaitEvent(hStream, hEvent, Flags);
    }
    return CUDA_SUCCESS;
}

CUresult cuStreamGetPriority(CUstream hStream, int *priority) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        void* fn = CudaDriverPassthrough::instance().getRealFunction("cuStreamGetPriority");
        if (fn) {
            typedef CUresult (*fn_t)(CUstream, int*);
            return ((fn_t)fn)(hStream, priority);
        }
    }
    if (priority) *priority = 0;
    return CUDA_SUCCESS;
}

CUresult cuStreamGetFlags(CUstream hStream, unsigned int *flags) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuStreamGetFlags(hStream, flags);
    }
    if (flags) *flags = 0;
    return CUDA_SUCCESS;
}

CUresult cuStreamGetCtx(CUstream hStream, CUcontext *pctx) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        void* fn = CudaDriverPassthrough::instance().getRealFunction("cuStreamGetCtx");
        if (fn) {
            typedef CUresult (*fn_t)(CUstream, CUcontext*);
            return ((fn_t)fn)(hStream, pctx);
        }
    }
    if (pctx) *pctx = (CUcontext)(uintptr_t)(current_context_device + 1);
    return CUDA_SUCCESS;
}

// Event management
CUresult cuEventCreate(CUevent *phEvent, unsigned int Flags) {
    if (!phEvent) return CUDA_ERROR_INVALID_VALUE;

    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuEventCreate(phEvent, Flags);
    }

    *phEvent = (CUevent)(uintptr_t)1;
    FGPU_LOG("[FakeCUDA-Driver] cuEventCreate returning fake event\n");
    return CUDA_SUCCESS;
}

CUresult cuEventDestroy(CUevent hEvent) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuEventDestroy(hEvent);
    }
    FGPU_LOG("[FakeCUDA-Driver] cuEventDestroy\n");
    return CUDA_SUCCESS;
}

CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuEventRecord(hEvent, hStream);
    }
    return CUDA_SUCCESS;
}

CUresult cuEventSynchronize(CUevent hEvent) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuEventSynchronize(hEvent);
    }
    return CUDA_SUCCESS;
}

CUresult cuEventQuery(CUevent hEvent) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuEventQuery(hEvent);
    }
    return CUDA_SUCCESS;
}

CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuEventElapsedTime(pMilliseconds, hStart, hEnd);
    }
    if (pMilliseconds) *pMilliseconds = 0.0f;
    return CUDA_SUCCESS;
}

// Context info
CUresult cuCtxGetDevice(CUdevice *device) {
    if (!device) return CUDA_ERROR_INVALID_VALUE;

    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() == FakeGpuMode::Passthrough && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuCtxGetDevice(device);
    }

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
    const BackendConfig& config = BackendConfig::instance();

    if (config.mode() == FakeGpuMode::Passthrough && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuMemGetInfo(free, total);
    }

    GlobalState::instance().initialize();
    Device& dev = GlobalState::instance().get_device(current_context_device);

    if (config.mode() == FakeGpuMode::Hybrid && real_driver_available()) {
        HybridMemoryManager::instance().initialize();

        if (config.oom_policy() == OomPolicy::Clamp) {
            size_t reported_total = 0;
            size_t reported_free = 0;
            HybridMemoryManager::instance().get_clamped_memory_info(
                current_context_device,
                dev.total_memory,
                dev.used_memory,
                reported_total,
                reported_free);
            if (total) *total = reported_total;
            if (free) *free = reported_free;
            return CUDA_SUCCESS;
        }
        // For non-clamp policies, report virtual memory info to allow oversubscription.
    }

    if (total) *total = dev.total_memory;
    if (free) *free = (dev.total_memory > dev.used_memory) ? (dev.total_memory - dev.used_memory) : 0;
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

    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuModuleLoad(module, fname);
    }

    *module = (CUmodule)(uintptr_t)1;
    FGPU_LOG("[FakeCUDA-Driver] cuModuleLoad (fake)\n");
    return CUDA_SUCCESS;
}

CUresult cuModuleLoadData(CUmodule *module, const void *image) {
    if (!module) return CUDA_ERROR_INVALID_VALUE;

    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuModuleLoadData(module, image);
    }

    *module = (CUmodule)(uintptr_t)1;
    return CUDA_SUCCESS;
}

CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, void *options, void **optionValues) {
    if (!module) return CUDA_ERROR_INVALID_VALUE;

    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        using fn_t = CUresult (*)(CUmodule*, const void*, unsigned int, void*, void**);
        static fn_t real_fn =
            reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuModuleLoadDataEx"));
        if (real_fn) {
            return real_fn(module, image, numOptions, options, optionValues);
        }
        // Fallback: ignore options and load the module data.
        return CudaDriverPassthrough::instance().cuModuleLoadData(module, image);
    }

    *module = (CUmodule)(uintptr_t)1;
    return CUDA_SUCCESS;
}

CUresult cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin) {
    if (!module) return CUDA_ERROR_INVALID_VALUE;

    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        using fn_t = CUresult (*)(CUmodule*, const void*);
        static fn_t real_fn =
            reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuModuleLoadFatBinary"));
        if (real_fn) {
            return real_fn(module, fatCubin);
        }
        // Fallback: try to treat the fatbin as module data.
        return CudaDriverPassthrough::instance().cuModuleLoadData(module, fatCubin);
    }

    *module = (CUmodule)(uintptr_t)1;
    return CUDA_SUCCESS;
}

CUresult cuModuleUnload(CUmodule hmod) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuModuleUnload(hmod);
    }
    return CUDA_SUCCESS;
}

CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
    if (!hfunc) return CUDA_ERROR_INVALID_VALUE;

    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuModuleGetFunction(hfunc, hmod, name);
    }

    *hfunc = (CUfunction)(uintptr_t)1;
    return CUDA_SUCCESS;
}

CUresult cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        using fn_t = CUresult (*)(CUdeviceptr*, size_t*, CUmodule, const char*);
        static fn_t real_fn =
            reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuModuleGetGlobal"));
        if (real_fn) {
            return real_fn(dptr, bytes, hmod, name);
        }
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    if (dptr) *dptr = 0;
    if (bytes) *bytes = 0;
    return CUDA_SUCCESS;
}

CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuLaunchKernel(
            f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
    }
    FGPU_LOG("[FakeCUDA-Driver] cuLaunchKernel (no-op)\n");
    return CUDA_SUCCESS;
}

CUresult cuFuncGetAttribute(int *pi, int attrib, CUfunction hfunc) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        using fn_t = CUresult (*)(int*, int, CUfunction);
        static fn_t real_fn =
            reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuFuncGetAttribute"));
        if (real_fn) return real_fn(pi, attrib, hfunc);
    }
    if (pi) *pi = 0;
    return CUDA_SUCCESS;
}

CUresult cuFuncSetAttribute(CUfunction hfunc, int attrib, int value) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        using fn_t = CUresult (*)(CUfunction, int, int);
        static fn_t real_fn =
            reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuFuncSetAttribute"));
        if (real_fn) return real_fn(hfunc, attrib, value);
    }
    return CUDA_SUCCESS;
}

CUresult cuFuncSetCacheConfig(CUfunction hfunc, int config) {
    const BackendConfig& cfg = BackendConfig::instance();
    if (cfg.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        using fn_t = CUresult (*)(CUfunction, int);
        static fn_t real_fn =
            reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuFuncSetCacheConfig"));
        if (real_fn) return real_fn(hfunc, config);
    }
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
    if (!dptr) return CUDA_ERROR_INVALID_VALUE;

    const BackendConfig& config = BackendConfig::instance();

    if (config.mode() == FakeGpuMode::Passthrough && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuMemAllocManaged(dptr, bytesize, flags);
    }

    if (config.mode() == FakeGpuMode::Hybrid && real_driver_available()) {
        GlobalState::instance().initialize();
        HybridMemoryManager::instance().initialize();

        CUdeviceptr out_ptr = 0;
        CUresult result = CudaDriverPassthrough::instance().cuMemAllocManaged(&out_ptr, bytesize, flags);
        if (result != CUDA_SUCCESS) return result;

        const int virtual_device = current_context_device;
        if (!GlobalState::instance().register_managed_allocation((void*)out_ptr, bytesize, virtual_device)) {
            CudaDriverPassthrough::instance().cuMemFree(out_ptr);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }

        *dptr = out_ptr;
        HybridMemoryManager::instance().record_allocation((void*)out_ptr, bytesize, virtual_device, HybridAllocation::Type::Managed);
        return CUDA_SUCCESS;
    }

    void* ptr = malloc(bytesize);
    if (!ptr) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    int device = current_context_device;
    if (!GlobalState::instance().register_managed_allocation(ptr, bytesize, device)) {
        free(ptr);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    *dptr = (CUdeviceptr)ptr;
    FGPU_LOG("[FakeCUDA-Driver] cuMemAllocManaged allocated %zu bytes at 0x%llx on device %d\n",
           bytesize, *dptr, device);
    return CUDA_SUCCESS;
}

CUresult cuMemAllocHost(void **pp, size_t bytesize) {
    if (!pp) return CUDA_ERROR_INVALID_VALUE;

    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuMemAllocHost(pp, bytesize);
    }

    *pp = malloc(bytesize);
    if (!*pp) return CUDA_ERROR_OUT_OF_MEMORY;
    GlobalState::instance().register_host_allocation(*pp, bytesize, current_context_device);
    return CUDA_SUCCESS;
}

CUresult cuMemFreeHost(void *p) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuMemFreeHost(p);
    }

    size_t size = 0;
    int device = 0;
    GlobalState::instance().release_host_allocation(p, size, device);
    free(p);
    return CUDA_SUCCESS;
}

CUresult cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags) {
    if (!pp) return CUDA_ERROR_INVALID_VALUE;

    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuMemHostAlloc(pp, bytesize, Flags);
    }

    return cuMemAllocHost(pp, bytesize);
}

CUresult cuMemHostRegister(void *p, size_t bytesize, unsigned int Flags) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        using fn_t = CUresult (*)(void*, size_t, unsigned int);
        static fn_t real_fn =
            reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuMemHostRegister"));
        if (real_fn) return real_fn(p, bytesize, Flags);
    }
    return CUDA_SUCCESS;
}

CUresult cuMemHostUnregister(void *p) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        using fn_t = CUresult (*)(void*);
        static fn_t real_fn =
            reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuMemHostUnregister"));
        if (real_fn) return real_fn(p);
    }
    return CUDA_SUCCESS;
}

CUresult cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p, unsigned int Flags) {
    if (!pdptr) return CUDA_ERROR_INVALID_VALUE;

    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuMemHostGetDevicePointer(pdptr, p, Flags);
    }

    *pdptr = (CUdeviceptr)p;
    return CUDA_SUCCESS;
}

CUresult cuMemHostGetFlags(unsigned int *pFlags, void *p) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        using fn_t = CUresult (*)(unsigned int*, void*);
        static fn_t real_fn =
            reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuMemHostGetFlags"));
        if (real_fn) return real_fn(pFlags, p);
    }
    if (pFlags) *pFlags = 0;
    return CUDA_SUCCESS;
}

CUresult cuPointerGetAttribute(void *data, CUpointer_attribute attribute, CUdeviceptr ptr) {
    if (!data) return CUDA_ERROR_INVALID_VALUE;

    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() == FakeGpuMode::Passthrough && real_driver_available()) {
        using fn_t = CUresult (*)(void*, CUpointer_attribute, CUdeviceptr);
        static fn_t real_fn =
            reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuPointerGetAttribute"));
        if (real_fn) return real_fn(data, attribute, ptr);
    }

    if (config.mode() == FakeGpuMode::Hybrid && real_driver_available()) {
        using fn_t = CUresult (*)(void*, CUpointer_attribute, CUdeviceptr);
        static fn_t real_fn =
            reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuPointerGetAttribute"));
        if (real_fn) {
            CUresult result = real_fn(data, attribute, ptr);
            if (result == CUDA_SUCCESS) return result;
        }
    }

    GlobalState::instance().initialize();

    size_t alloc_size = 0;
    int alloc_device = current_context_device;
    GlobalState::AllocationKind alloc_kind = GlobalState::AllocationKind::Device;
    bool found = GlobalState::instance().get_allocation_info_ex((void*)ptr, alloc_size, alloc_device, alloc_kind);

    CUmemorytype mem_type = CU_MEMORYTYPE_HOST;
    unsigned int is_managed = 0;
    if (found) {
        switch (alloc_kind) {
            case GlobalState::AllocationKind::Device:
                mem_type = CU_MEMORYTYPE_DEVICE;
                break;
            case GlobalState::AllocationKind::Managed:
                mem_type = CU_MEMORYTYPE_UNIFIED;
                is_managed = 1;
                break;
            case GlobalState::AllocationKind::Host:
                mem_type = CU_MEMORYTYPE_HOST;
                break;
        }
    }

    switch (attribute) {
        case CU_POINTER_ATTRIBUTE_CONTEXT:
            *(CUcontext*)data = (CUcontext)(uintptr_t)(alloc_device + 1);
            break;
        case CU_POINTER_ATTRIBUTE_MEMORY_TYPE:
            *(CUmemorytype*)data = mem_type;
            break;
        case CU_POINTER_ATTRIBUTE_DEVICE_POINTER:
            *(CUdeviceptr*)data = ptr;
            break;
        case CU_POINTER_ATTRIBUTE_HOST_POINTER:
            *(void**)data = (void*)ptr;
            break;
        case CU_POINTER_ATTRIBUTE_IS_MANAGED:
            *(unsigned int*)data = is_managed;
            break;
        case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL:
            *(int*)data = alloc_device;
            break;
        case CU_POINTER_ATTRIBUTE_SYNC_MEMOPS:
            *(int*)data = 0;
            break;
        case CU_POINTER_ATTRIBUTE_BUFFER_ID:
            *(unsigned long long*)data = 0;
            break;
        default:
            break;
    }

    return CUDA_SUCCESS;
}

CUresult cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute *attributes, void **data, CUdeviceptr ptr) {
    if (!attributes || !data) return CUDA_ERROR_INVALID_VALUE;

    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() == FakeGpuMode::Passthrough && real_driver_available()) {
        using fn_t = CUresult (*)(unsigned int, CUpointer_attribute*, void**, CUdeviceptr);
        static fn_t real_fn =
            reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuPointerGetAttributes"));
        if (real_fn) return real_fn(numAttributes, attributes, data, ptr);
    }

    if (config.mode() == FakeGpuMode::Hybrid && real_driver_available()) {
        using fn_t = CUresult (*)(unsigned int, CUpointer_attribute*, void**, CUdeviceptr);
        static fn_t real_fn =
            reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuPointerGetAttributes"));
        if (real_fn) {
            CUresult result = real_fn(numAttributes, attributes, data, ptr);
            if (result == CUDA_SUCCESS) return result;
        }
    }

    for (unsigned int i = 0; i < numAttributes; ++i) {
        cuPointerGetAttribute(data[i], attributes[i], ptr);
    }
    return CUDA_SUCCESS;
}

CUresult cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        if (config.mode() == FakeGpuMode::Hybrid) {
            HybridAllocation alloc;
            if (HybridMemoryManager::instance().get_allocation_info((void*)dstDevice, alloc)) {
                if (alloc.type == HybridAllocation::Type::MappedHost && alloc.backing_ptr) {
                    memset(alloc.backing_ptr, uc, N);
                    GlobalState::instance().record_memset((void*)dstDevice, N);
                    return CUDA_SUCCESS;
                }
                if (alloc.type == HybridAllocation::Type::SpilledCpu) {
                    memset((void*)dstDevice, uc, N);
                    GlobalState::instance().record_memset((void*)dstDevice, N);
                    return CUDA_SUCCESS;
                }
            }
        }

        using fn_t = CUresult (*)(CUdeviceptr, unsigned char, size_t);
        static fn_t real_fn =
            reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuMemsetD8"));
        if (real_fn) {
            CUresult result = real_fn(dstDevice, uc, N);
            if (result == CUDA_SUCCESS) {
                GlobalState::instance().record_memset((void*)dstDevice, N);
            }
            return result;
        }
    }

    memset((void*)dstDevice, uc, N);
    GlobalState::instance().record_memset((void*)dstDevice, N);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD16(CUdeviceptr dstDevice, unsigned short us, size_t N) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        if (config.mode() == FakeGpuMode::Hybrid) {
            HybridAllocation alloc;
            if (HybridMemoryManager::instance().get_allocation_info((void*)dstDevice, alloc)) {
                void* host_ptr = nullptr;
                if (alloc.type == HybridAllocation::Type::MappedHost) host_ptr = alloc.backing_ptr;
                if (alloc.type == HybridAllocation::Type::SpilledCpu) host_ptr = (void*)dstDevice;
                if (host_ptr) {
                    unsigned short* p = static_cast<unsigned short*>(host_ptr);
                    for (size_t i = 0; i < N; i++) p[i] = us;
                    GlobalState::instance().record_memset((void*)dstDevice, N * sizeof(unsigned short));
                    return CUDA_SUCCESS;
                }
            }
        }

        using fn_t = CUresult (*)(CUdeviceptr, unsigned short, size_t);
        static fn_t real_fn =
            reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuMemsetD16"));
        if (real_fn) {
            CUresult result = real_fn(dstDevice, us, N);
            if (result == CUDA_SUCCESS) {
                GlobalState::instance().record_memset((void*)dstDevice, N * sizeof(unsigned short));
            }
            return result;
        }
    }

    unsigned short *p = (unsigned short*)dstDevice;
    for (size_t i = 0; i < N; i++) p[i] = us;
    GlobalState::instance().record_memset((void*)dstDevice, N * sizeof(unsigned short));
    return CUDA_SUCCESS;
}

CUresult cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        if (config.mode() == FakeGpuMode::Hybrid) {
            HybridAllocation alloc;
            if (HybridMemoryManager::instance().get_allocation_info((void*)dstDevice, alloc)) {
                void* host_ptr = nullptr;
                if (alloc.type == HybridAllocation::Type::MappedHost) host_ptr = alloc.backing_ptr;
                if (alloc.type == HybridAllocation::Type::SpilledCpu) host_ptr = (void*)dstDevice;
                if (host_ptr) {
                    unsigned int* p = static_cast<unsigned int*>(host_ptr);
                    for (size_t i = 0; i < N; i++) p[i] = ui;
                    GlobalState::instance().record_memset((void*)dstDevice, N * sizeof(unsigned int));
                    return CUDA_SUCCESS;
                }
            }
        }

        using fn_t = CUresult (*)(CUdeviceptr, unsigned int, size_t);
        static fn_t real_fn =
            reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuMemsetD32"));
        if (real_fn) {
            CUresult result = real_fn(dstDevice, ui, N);
            if (result == CUDA_SUCCESS) {
                GlobalState::instance().record_memset((void*)dstDevice, N * sizeof(unsigned int));
            }
            return result;
        }
    }

    unsigned int *p = (unsigned int*)dstDevice;
    for (size_t i = 0; i < N; i++) p[i] = ui;
    GlobalState::instance().record_memset((void*)dstDevice, N * sizeof(unsigned int));
    return CUDA_SUCCESS;
}

CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        if (config.mode() == FakeGpuMode::Hybrid) {
            HybridAllocation alloc;
            if (HybridMemoryManager::instance().get_allocation_info((void*)dstDevice, alloc)) {
                if (alloc.type == HybridAllocation::Type::MappedHost && alloc.backing_ptr) {
                    memset(alloc.backing_ptr, uc, N);
                    GlobalState::instance().record_memset((void*)dstDevice, N);
                    return CUDA_SUCCESS;
                }
                if (alloc.type == HybridAllocation::Type::SpilledCpu) {
                    memset((void*)dstDevice, uc, N);
                    GlobalState::instance().record_memset((void*)dstDevice, N);
                    return CUDA_SUCCESS;
                }
            }
        }

        using fn_t = CUresult (*)(CUdeviceptr, unsigned char, size_t, CUstream);
        static fn_t real_fn =
            reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuMemsetD8Async"));
        if (real_fn) {
            CUresult result = real_fn(dstDevice, uc, N, hStream);
            if (result == CUDA_SUCCESS) {
                GlobalState::instance().record_memset((void*)dstDevice, N);
            }
            return result;
        }
    }

    return cuMemsetD8(dstDevice, uc, N);
}

CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        if (config.mode() == FakeGpuMode::Hybrid) {
            HybridAllocation alloc;
            if (HybridMemoryManager::instance().get_allocation_info((void*)dstDevice, alloc)) {
                void* host_ptr = nullptr;
                if (alloc.type == HybridAllocation::Type::MappedHost) host_ptr = alloc.backing_ptr;
                if (alloc.type == HybridAllocation::Type::SpilledCpu) host_ptr = (void*)dstDevice;
                if (host_ptr) {
                    unsigned int* p = static_cast<unsigned int*>(host_ptr);
                    for (size_t i = 0; i < N; i++) p[i] = ui;
                    GlobalState::instance().record_memset((void*)dstDevice, N * sizeof(unsigned int));
                    return CUDA_SUCCESS;
                }
            }
        }

        using fn_t = CUresult (*)(CUdeviceptr, unsigned int, size_t, CUstream);
        static fn_t real_fn =
            reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuMemsetD32Async"));
        if (real_fn) {
            CUresult result = real_fn(dstDevice, ui, N, hStream);
            if (result == CUDA_SUCCESS) {
                GlobalState::instance().record_memset((void*)dstDevice, N * sizeof(unsigned int));
            }
            return result;
        }
    }

    return cuMemsetD32(dstDevice, ui, N);
}

CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream) {
    if (!dst || !src) return CUDA_ERROR_INVALID_VALUE;

    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        auto host_ptr_for = [&](CUdeviceptr ptr) -> void* {
            if (config.mode() != FakeGpuMode::Hybrid) return nullptr;
            HybridAllocation alloc;
            if (!HybridMemoryManager::instance().get_allocation_info((void*)ptr, alloc)) return nullptr;
            if (alloc.type == HybridAllocation::Type::MappedHost) return alloc.backing_ptr;
            if (alloc.type == HybridAllocation::Type::SpilledCpu) return (void*)ptr;
            return nullptr;
        };

        void* dst_host = host_ptr_for(dst);
        void* src_host = host_ptr_for(src);

        if (dst_host && src_host) {
            memcpy(dst_host, src_host, ByteCount);
            GlobalState::instance().record_memcpy_d2d((void*)dst, (void*)src, ByteCount);
            return CUDA_SUCCESS;
        }

        if (dst_host && !src_host) {
            using fn_t = CUresult (*)(void*, CUdeviceptr, size_t, CUstream);
            static fn_t real_fn =
                reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuMemcpyDtoHAsync"));
            if (!real_fn) {
                // Fall back to sync copy if async is unavailable.
                CUresult result = CudaDriverPassthrough::instance().cuMemcpyDtoH(dst_host, src, ByteCount);
                if (result == CUDA_SUCCESS) {
                    GlobalState::instance().record_memcpy_d2d((void*)dst, (void*)src, ByteCount);
                }
                return result;
            }
            CUresult result = real_fn(dst_host, src, ByteCount, hStream);
            if (result == CUDA_SUCCESS) {
                GlobalState::instance().record_memcpy_d2d((void*)dst, (void*)src, ByteCount);
            }
            return result;
        }

        if (!dst_host && src_host) {
            using fn_t = CUresult (*)(CUdeviceptr, const void*, size_t, CUstream);
            static fn_t real_fn =
                reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuMemcpyHtoDAsync"));
            if (!real_fn) {
                CUresult result = CudaDriverPassthrough::instance().cuMemcpyHtoD(dst, src_host, ByteCount);
                if (result == CUDA_SUCCESS) {
                    GlobalState::instance().record_memcpy_d2d((void*)dst, (void*)src, ByteCount);
                }
                return result;
            }
            CUresult result = real_fn(dst, src_host, ByteCount, hStream);
            if (result == CUDA_SUCCESS) {
                GlobalState::instance().record_memcpy_d2d((void*)dst, (void*)src, ByteCount);
            }
            return result;
        }

        using fn_t = CUresult (*)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
        static fn_t real_fn =
            reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuMemcpyAsync"));
        if (!real_fn) {
            // Fallback: use sync D2D.
            CUresult result = CudaDriverPassthrough::instance().cuMemcpyDtoD(dst, src, ByteCount);
            if (result == CUDA_SUCCESS) {
                GlobalState::instance().record_memcpy_d2d((void*)dst, (void*)src, ByteCount);
            }
            return result;
        }

        CUresult result = real_fn(dst, src, ByteCount, hStream);
        if (result == CUDA_SUCCESS) {
            GlobalState::instance().record_memcpy_d2d((void*)dst, (void*)src, ByteCount);
        }
        return result;
    }

    memcpy((void*)dst, (void*)src, ByteCount);
    GlobalState::instance().record_memcpy_d2d((void*)dst, (void*)src, ByteCount);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
    if (!dstHost || !srcDevice) return CUDA_ERROR_INVALID_VALUE;

    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() == FakeGpuMode::Hybrid && real_driver_available()) {
        HybridAllocation alloc;
        if (HybridMemoryManager::instance().get_allocation_info((void*)srcDevice, alloc)) {
            if (alloc.type == HybridAllocation::Type::MappedHost && alloc.backing_ptr) {
                memcpy(dstHost, alloc.backing_ptr, ByteCount);
                GlobalState::instance().record_memcpy_d2h((void*)srcDevice, ByteCount);
                return CUDA_SUCCESS;
            }
            if (alloc.type == HybridAllocation::Type::SpilledCpu) {
                memcpy(dstHost, (void*)srcDevice, ByteCount);
                GlobalState::instance().record_memcpy_d2h((void*)srcDevice, ByteCount);
                return CUDA_SUCCESS;
            }
        }

        using fn_t = CUresult (*)(void*, CUdeviceptr, size_t, CUstream);
        static fn_t real_fn =
            reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuMemcpyDtoHAsync"));
        if (real_fn) {
            CUresult result = real_fn(dstHost, srcDevice, ByteCount, hStream);
            if (result == CUDA_SUCCESS) {
                GlobalState::instance().record_memcpy_d2h((void*)srcDevice, ByteCount);
            }
            return result;
        }
    }

    if (config.mode() == FakeGpuMode::Passthrough && real_driver_available()) {
        using fn_t = CUresult (*)(void*, CUdeviceptr, size_t, CUstream);
        static fn_t real_fn =
            reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuMemcpyDtoHAsync"));
        if (real_fn) return real_fn(dstHost, srcDevice, ByteCount, hStream);
        return CudaDriverPassthrough::instance().cuMemcpyDtoH(dstHost, srcDevice, ByteCount);
    }

    return cuMemcpyDtoH(dstHost, srcDevice, ByteCount);
}

CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream) {
    if (!dstDevice || !srcHost) return CUDA_ERROR_INVALID_VALUE;

    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() == FakeGpuMode::Hybrid && real_driver_available()) {
        HybridAllocation alloc;
        if (HybridMemoryManager::instance().get_allocation_info((void*)dstDevice, alloc)) {
            if (alloc.type == HybridAllocation::Type::MappedHost && alloc.backing_ptr) {
                memcpy(alloc.backing_ptr, srcHost, ByteCount);
                GlobalState::instance().record_memcpy_h2d((void*)dstDevice, ByteCount);
                return CUDA_SUCCESS;
            }
            if (alloc.type == HybridAllocation::Type::SpilledCpu) {
                memcpy((void*)dstDevice, srcHost, ByteCount);
                GlobalState::instance().record_memcpy_h2d((void*)dstDevice, ByteCount);
                return CUDA_SUCCESS;
            }
        }

        using fn_t = CUresult (*)(CUdeviceptr, const void*, size_t, CUstream);
        static fn_t real_fn =
            reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuMemcpyHtoDAsync"));
        if (real_fn) {
            CUresult result = real_fn(dstDevice, srcHost, ByteCount, hStream);
            if (result == CUDA_SUCCESS) {
                GlobalState::instance().record_memcpy_h2d((void*)dstDevice, ByteCount);
            }
            return result;
        }
    }

    if (config.mode() == FakeGpuMode::Passthrough && real_driver_available()) {
        using fn_t = CUresult (*)(CUdeviceptr, const void*, size_t, CUstream);
        static fn_t real_fn =
            reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuMemcpyHtoDAsync"));
        if (real_fn) return real_fn(dstDevice, srcHost, ByteCount, hStream);
        return CudaDriverPassthrough::instance().cuMemcpyHtoD(dstDevice, srcHost, ByteCount);
    }

    return cuMemcpyHtoD(dstDevice, srcHost, ByteCount);
}

CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
    if (!dstDevice || !srcDevice) return CUDA_ERROR_INVALID_VALUE;

    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        // Reuse the same mixed-pointer handling as cuMemcpyAsync (Hybrid spill/mapped host).
        return cuMemcpyAsync(dstDevice, srcDevice, ByteCount, hStream);
    }

    return cuMemcpyDtoD(dstDevice, srcDevice, ByteCount);
}

CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
    return cuMemcpyDtoD(dst, src, ByteCount);
}

CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount) {
    if (!dstDevice || !srcDevice) return CUDA_ERROR_INVALID_VALUE;

    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        using fn_t = CUresult (*)(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t);
        static fn_t real_fn =
            reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuMemcpyPeer"));
        if (!real_fn) {
            // Fallback: treat as regular device copy.
            return CudaDriverPassthrough::instance().cuMemcpyDtoD(dstDevice, srcDevice, ByteCount);
        }
        CUresult result = real_fn(dstDevice, dstContext, srcDevice, srcContext, ByteCount);
        if (result == CUDA_SUCCESS) {
            GlobalState::instance().record_memcpy_d2d((void*)dstDevice, (void*)srcDevice, ByteCount);
        }
        return result;
    }

    memcpy((void*)dstDevice, (void*)srcDevice, ByteCount);
    const int dst_device = dstContext ? (static_cast<int>(reinterpret_cast<uintptr_t>(dstContext)) - 1) : current_context_device;
    const int src_device = srcContext ? (static_cast<int>(reinterpret_cast<uintptr_t>(srcContext)) - 1) : current_context_device;
    GlobalState::instance().record_memcpy_peer(dst_device, src_device, ByteCount);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream) {
    if (!dstDevice || !srcDevice) return CUDA_ERROR_INVALID_VALUE;

    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        using fn_t = CUresult (*)(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t, CUstream);
        static fn_t real_fn =
            reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuMemcpyPeerAsync"));
        if (!real_fn) {
            // Fallback: do sync peer copy.
            return cuMemcpyPeer(dstDevice, dstContext, srcDevice, srcContext, ByteCount);
        }
        CUresult result = real_fn(dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream);
        if (result == CUDA_SUCCESS) {
            GlobalState::instance().record_memcpy_d2d((void*)dstDevice, (void*)srcDevice, ByteCount);
        }
        return result;
    }

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
    return cuMemcpyDtoHAsync(dstHost, srcDevice, ByteCount, hStream);
}

CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream) {
    return cuMemcpyHtoDAsync(dstDevice, srcHost, ByteCount, hStream);
}

CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
    return cuMemcpyDtoDAsync(dstDevice, srcDevice, ByteCount, hStream);
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
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        using fn_t = CUresult (*)(CUmemoryPool*, CUdevice);
        static fn_t real_fn =
            reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuDeviceGetDefaultMemPool"));
        if (real_fn) {
            const CUdevice real_dev = (config.mode() == FakeGpuMode::Hybrid) ? map_virtual_device_to_real(dev) : dev;
            return real_fn(pool_out, real_dev);
        }
    }
    if (pool_out) *pool_out = (CUmemoryPool)(uintptr_t)(dev + 1);
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetMemPool(CUmemoryPool *pool, CUdevice dev) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        using fn_t = CUresult (*)(CUmemoryPool*, CUdevice);
        static fn_t real_fn = reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuDeviceGetMemPool"));
        if (real_fn) {
            const CUdevice real_dev = (config.mode() == FakeGpuMode::Hybrid) ? map_virtual_device_to_real(dev) : dev;
            return real_fn(pool, real_dev);
        }
    }
    if (pool) *pool = (CUmemoryPool)(uintptr_t)(dev + 1);
    return CUDA_SUCCESS;
}

CUresult cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        using fn_t = CUresult (*)(CUdevice, CUmemoryPool);
        static fn_t real_fn = reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuDeviceSetMemPool"));
        if (real_fn) {
            const CUdevice real_dev = (config.mode() == FakeGpuMode::Hybrid) ? map_virtual_device_to_real(dev) : dev;
            return real_fn(real_dev, pool);
        }
    }
    return CUDA_SUCCESS;
}

CUresult cuMemPoolCreate(CUmemoryPool *pool, const void *poolProps) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        using fn_t = CUresult (*)(CUmemoryPool*, const void*);
        static fn_t real_fn = reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuMemPoolCreate"));
        if (real_fn) return real_fn(pool, poolProps);
    }
    if (pool) *pool = (CUmemoryPool)(uintptr_t)1;
    return CUDA_SUCCESS;
}

CUresult cuMemPoolDestroy(CUmemoryPool pool) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        using fn_t = CUresult (*)(CUmemoryPool);
        static fn_t real_fn = reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuMemPoolDestroy"));
        if (real_fn) return real_fn(pool);
    }
    return CUDA_SUCCESS;
}

CUresult cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, CUstream hStream) {
    if (!dptr) return CUDA_ERROR_INVALID_VALUE;

    const BackendConfig& config = BackendConfig::instance();

    if (config.mode() == FakeGpuMode::Passthrough && real_driver_available()) {
        using fn_t = CUresult (*)(CUdeviceptr*, size_t, CUstream);
        static fn_t real_fn = reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuMemAllocAsync"));
        if (real_fn) return real_fn(dptr, bytesize, hStream);
        return CudaDriverPassthrough::instance().cuMemAlloc(dptr, bytesize);
    }

    if (config.mode() == FakeGpuMode::Hybrid && real_driver_available()) {
        GlobalState::instance().initialize();
        HybridMemoryManager::instance().initialize();

        const int virtual_device = current_context_device;
        auto decision = HybridMemoryManager::instance().decide_allocation(virtual_device, bytesize);

        constexpr unsigned int kManagedAttachGlobal = 1; // CU_MEM_ATTACH_GLOBAL
        constexpr unsigned int kHostAllocDevicemap = 2;  // CU_MEMHOSTALLOC_DEVICEMAP

        CUresult result = CUDA_SUCCESS;
        HybridAllocation::Type alloc_type = HybridAllocation::Type::RealDevice;
        void* backing_ptr = nullptr;
        CUdeviceptr out_ptr = 0;

        switch (decision) {
            case HybridMemoryManager::AllocationDecision::UseReal: {
                using fn_t = CUresult (*)(CUdeviceptr*, size_t, CUstream);
                static fn_t real_fn =
                    reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuMemAllocAsync"));
                if (real_fn) {
                    result = real_fn(&out_ptr, bytesize, hStream);
                } else {
                    result = CudaDriverPassthrough::instance().cuMemAlloc(&out_ptr, bytesize);
                }
                alloc_type = HybridAllocation::Type::RealDevice;
                break;
            }
            case HybridMemoryManager::AllocationDecision::UseManaged: {
                result = CudaDriverPassthrough::instance().cuMemAllocManaged(&out_ptr, bytesize, kManagedAttachGlobal);
                alloc_type = HybridAllocation::Type::Managed;
                break;
            }
            case HybridMemoryManager::AllocationDecision::UseMappedHost: {
                void* host_ptr = nullptr;
                result = CudaDriverPassthrough::instance().cuMemHostAlloc(&host_ptr, bytesize, kHostAllocDevicemap);
                if (result != CUDA_SUCCESS) return result;
                CUdeviceptr device_ptr = 0;
                result = CudaDriverPassthrough::instance().cuMemHostGetDevicePointer(&device_ptr, host_ptr, 0);
                if (result != CUDA_SUCCESS) {
                    CudaDriverPassthrough::instance().cuMemFreeHost(host_ptr);
                    return result;
                }
                backing_ptr = host_ptr;
                out_ptr = device_ptr;
                alloc_type = HybridAllocation::Type::MappedHost;
                break;
            }
            case HybridMemoryManager::AllocationDecision::SpillToCpu: {
                void* ptr = malloc(bytesize);
                if (!ptr) return CUDA_ERROR_OUT_OF_MEMORY;
                out_ptr = (CUdeviceptr)ptr;
                alloc_type = HybridAllocation::Type::SpilledCpu;
                break;
            }
            case HybridMemoryManager::AllocationDecision::Fail:
                return CUDA_ERROR_OUT_OF_MEMORY;
        }

        if (result != CUDA_SUCCESS) return result;

        bool registered = false;
        if (alloc_type == HybridAllocation::Type::Managed) {
            registered = GlobalState::instance().register_managed_allocation((void*)out_ptr, bytesize, virtual_device);
        } else {
            registered = GlobalState::instance().register_allocation((void*)out_ptr, bytesize, virtual_device);
        }

        if (!registered) {
            switch (alloc_type) {
                case HybridAllocation::Type::RealDevice:
                case HybridAllocation::Type::Managed:
                    CudaDriverPassthrough::instance().cuMemFree(out_ptr);
                    break;
                case HybridAllocation::Type::MappedHost:
                    if (backing_ptr) {
                        CudaDriverPassthrough::instance().cuMemFreeHost(backing_ptr);
                    }
                    break;
                case HybridAllocation::Type::SpilledCpu:
                    free((void*)out_ptr);
                    break;
            }
            return CUDA_ERROR_OUT_OF_MEMORY;
        }

        *dptr = out_ptr;
        HybridMemoryManager::instance().record_allocation((void*)out_ptr, bytesize, virtual_device, alloc_type, backing_ptr);
        return CUDA_SUCCESS;
    }

    // Simulate mode: treat async alloc as sync alloc.
    return cuMemAlloc(dptr, bytesize);
}

CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {
    const BackendConfig& config = BackendConfig::instance();

    if (config.mode() == FakeGpuMode::Passthrough && real_driver_available()) {
        using fn_t = CUresult (*)(CUdeviceptr, CUstream);
        static fn_t real_fn = reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuMemFreeAsync"));
        if (real_fn) return real_fn(dptr, hStream);
        return CudaDriverPassthrough::instance().cuMemFree(dptr);
    }

    if (config.mode() == FakeGpuMode::Hybrid && real_driver_available()) {
        HybridAllocation alloc;
        if (HybridMemoryManager::instance().release_allocation((void*)dptr, alloc)) {
            size_t size = 0;
            int device = -1;
            GlobalState::instance().release_allocation((void*)dptr, size, device);

            switch (alloc.type) {
                case HybridAllocation::Type::RealDevice: {
                    using fn_t = CUresult (*)(CUdeviceptr, CUstream);
                    static fn_t real_fn =
                        reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuMemFreeAsync"));
                    if (real_fn) return real_fn(dptr, hStream);
                    return CudaDriverPassthrough::instance().cuMemFree(dptr);
                }
                case HybridAllocation::Type::Managed:
                    // Managed allocations are freed with cuMemFree.
                    return CudaDriverPassthrough::instance().cuMemFree(dptr);
                case HybridAllocation::Type::MappedHost:
                    return CudaDriverPassthrough::instance().cuMemFreeHost(alloc.backing_ptr);
                case HybridAllocation::Type::SpilledCpu:
                    free((void*)dptr);
                    return CUDA_SUCCESS;
            }
        }
        // Not tracked: try real async free if available.
        using fn_t = CUresult (*)(CUdeviceptr, CUstream);
        static fn_t real_fn = reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuMemFreeAsync"));
        if (real_fn) return real_fn(dptr, hStream);
        return CudaDriverPassthrough::instance().cuMemFree(dptr);
    }

    return cuMemFree(dptr);
}

CUresult cuMemAllocFromPoolAsync(CUdeviceptr *dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream) {
    if (!dptr) return CUDA_ERROR_INVALID_VALUE;

    const BackendConfig& config = BackendConfig::instance();

    if (config.mode() == FakeGpuMode::Passthrough && real_driver_available()) {
        using fn_t = CUresult (*)(CUdeviceptr*, size_t, CUmemoryPool, CUstream);
        static fn_t real_fn =
            reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuMemAllocFromPoolAsync"));
        if (real_fn) return real_fn(dptr, bytesize, pool, hStream);
        return cuMemAllocAsync(dptr, bytesize, hStream);
    }

    if (config.mode() == FakeGpuMode::Hybrid && real_driver_available()) {
        GlobalState::instance().initialize();
        HybridMemoryManager::instance().initialize();

        const int virtual_device = current_context_device;
        auto decision = HybridMemoryManager::instance().decide_allocation(virtual_device, bytesize);

        if (decision == HybridMemoryManager::AllocationDecision::UseReal) {
            using fn_t = CUresult (*)(CUdeviceptr*, size_t, CUmemoryPool, CUstream);
            static fn_t real_fn =
                reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuMemAllocFromPoolAsync"));
            if (real_fn) {
                CUresult result = real_fn(dptr, bytesize, pool, hStream);
                if (result == CUDA_SUCCESS) {
                    if (!GlobalState::instance().register_allocation((void*)*dptr, bytesize, virtual_device)) {
                        cuMemFreeAsync(*dptr, hStream);
                        return CUDA_ERROR_OUT_OF_MEMORY;
                    }
                    HybridMemoryManager::instance().record_allocation(
                        (void*)*dptr, bytesize, virtual_device, HybridAllocation::Type::RealDevice);
                }
                return result;
            }
        }

        // Fallback to the same policy handling as cuMemAllocAsync.
        return cuMemAllocAsync(dptr, bytesize, hStream);
    }

    return cuMemAlloc(dptr, bytesize);
}

CUresult cuMemPoolSetAttribute(CUmemoryPool pool, int attr, void *value) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        using fn_t = CUresult (*)(CUmemoryPool, int, void*);
        static fn_t real_fn =
            reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuMemPoolSetAttribute"));
        if (real_fn) return real_fn(pool, attr, value);
    }
    return CUDA_SUCCESS;
}

CUresult cuMemPoolGetAttribute(CUmemoryPool pool, int attr, void *value) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        using fn_t = CUresult (*)(CUmemoryPool, int, void*);
        static fn_t real_fn =
            reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuMemPoolGetAttribute"));
        if (real_fn) return real_fn(pool, attr, value);
    }
    return CUDA_SUCCESS;
}

CUresult cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        using fn_t = CUresult (*)(CUmemoryPool, size_t);
        static fn_t real_fn =
            reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuMemPoolTrimTo"));
        if (real_fn) return real_fn(pool, minBytesToKeep);
    }
    return CUDA_SUCCESS;
}

CUresult cuCtxResetPersistingL2Cache(void) {
    return CUDA_SUCCESS;
}

CUresult cuMemGetAddressRange(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        using fn_t = CUresult (*)(CUdeviceptr*, size_t*, CUdeviceptr);
        static fn_t real_fn =
            reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuMemGetAddressRange"));
        if (real_fn) return real_fn(pbase, psize, dptr);
    }
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
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() != FakeGpuMode::Simulate && real_driver_available()) {
        using fn_t = CUresult (*)(const void**, const void*);
        static fn_t real_fn =
            reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuGetExportTable"));
        if (real_fn) {
            return real_fn(ppExportTable, pExportTableId);
        }
    }

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

    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() == FakeGpuMode::Passthrough && real_driver_available()) {
        return CudaDriverPassthrough::instance().cuGetProcAddress(symbol, pfn, cudaVersion, flags);
    }

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

    if (config.mode() == FakeGpuMode::Hybrid && real_driver_available()) {
        // Allow CUDA runtime / frameworks to resolve extra driver entry points without us stubbing them all.
        void* real_fn = nullptr;
        if (CudaDriverPassthrough::instance().cuGetProcAddress(symbol, &real_fn, cudaVersion, flags) == CUDA_SUCCESS &&
            real_fn) {
            *pfn = real_fn;
            return CUDA_SUCCESS;
        }

        real_fn = CudaDriverPassthrough::instance().getRealFunction(symbol);
        if (real_fn) {
            *pfn = real_fn;
            return CUDA_SUCCESS;
        }
    }

    // For unknown symbols, return NULL but success (some symbols are optional).
    // FGPU_LOG("[FakeCUDA-Driver] cuGetProcAddress: symbol '%s' NOT FOUND\n", symbol);
    *pfn = NULL;
    return CUDA_SUCCESS;
}

CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion, unsigned long long flags, void *symbolStatus) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() == FakeGpuMode::Passthrough && real_driver_available()) {
        using fn_t = CUresult (*)(const char*, void**, int, unsigned long long, void*);
        static fn_t real_fn =
            reinterpret_cast<fn_t>(CudaDriverPassthrough::instance().getRealFunction("cuGetProcAddress_v2"));
        if (real_fn) return real_fn(symbol, pfn, cudaVersion, flags, symbolStatus);
        return CudaDriverPassthrough::instance().cuGetProcAddress(symbol, pfn, cudaVersion, flags);
    }

    CUresult result = cuGetProcAddress(symbol, pfn, cudaVersion, flags);
    if (symbolStatus) {
        // Best-effort: 0 means found, 1 means not found.
        *static_cast<int*>(symbolStatus) = (*pfn != nullptr) ? 0 : 1;
    }
    return result;
}

} // extern "C"
