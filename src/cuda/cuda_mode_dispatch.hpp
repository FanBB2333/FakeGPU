#pragma once

#include "cuda_driver_defs.hpp"
#include "cuda_driver_passthrough.hpp"
#include "../core/backend_config.hpp"
#include "../core/global_state.hpp"
#include "../core/hybrid_memory_manager.hpp"
#include "../core/logging.hpp"

namespace fake_gpu {

// Mode-aware dispatch layer for CUDA Driver API
// This class decides whether to use fake implementation, passthrough, or hybrid
class CudaModeDispatch {
public:
    static CudaModeDispatch& instance() {
        static CudaModeDispatch dispatch;
        return dispatch;
    }

    // Check current mode
    FakeGpuMode mode() const {
        return BackendConfig::instance().mode();
    }

    // Check if passthrough is available and should be used
    bool should_passthrough() const {
        FakeGpuMode m = mode();
        if (m == FakeGpuMode::Simulate) return false;

        // For passthrough/hybrid, check if real CUDA is available
        return CudaDriverPassthrough::instance().is_available();
    }

    // Check if we should virtualize device info (simulate or hybrid mode)
    bool should_virtualize_device_info() const {
        return BackendConfig::instance().virtualize_device_info();
    }

    // Initialize - called once at startup
    void initialize() {
        if (initialized_) return;
        initialized_ = true;

        FakeGpuMode m = mode();
        FGPU_LOG("[CudaModeDispatch] Mode: %s\n", mode_name(m));

        if (m != FakeGpuMode::Simulate) {
            // Try to initialize passthrough
            if (CudaDriverPassthrough::instance().initialize()) {
                FGPU_LOG("[CudaModeDispatch] Passthrough initialized successfully\n");

                // Initialize hybrid memory manager if in hybrid mode
                if (m == FakeGpuMode::Hybrid) {
                    HybridMemoryManager::instance().initialize();
                    FGPU_LOG("[CudaModeDispatch] Hybrid memory manager initialized\n");
                }
            } else {
                FGPU_LOG("[CudaModeDispatch] WARNING: Passthrough not available, falling back to simulate\n");
                // Note: We don't change the mode here, but should_passthrough() will return false
            }
        }
    }

    // ========================================================================
    // Dispatch functions - these decide which implementation to use
    // ========================================================================

    // cuInit - always initialize both fake and real (if available)
    CUresult dispatch_cuInit(unsigned int flags) {
        initialize();

        // Always initialize fake state
        GlobalState::instance().initialize();

        if (should_passthrough()) {
            CUresult result = CudaDriverPassthrough::instance().cuInit(flags);
            if (result != CUDA_SUCCESS) {
                FGPU_LOG("[CudaModeDispatch] Real cuInit failed with %d, continuing with fake\n", result);
            }
            return result;
        }

        return CUDA_SUCCESS;
    }

    // cuDeviceGetCount - may return fake or real count depending on mode
    CUresult dispatch_cuDeviceGetCount(int* count) {
        if (!count) return CUDA_ERROR_INVALID_VALUE;

        if (should_virtualize_device_info()) {
            // Return fake device count
            *count = GlobalState::instance().get_device_count();
            return CUDA_SUCCESS;
        }

        if (should_passthrough()) {
            return CudaDriverPassthrough::instance().cuDeviceGetCount(count);
        }

        *count = GlobalState::instance().get_device_count();
        return CUDA_SUCCESS;
    }

    // cuDeviceGetName - may return fake or real name
    CUresult dispatch_cuDeviceGetName(char* name, int len, CUdevice dev) {
        if (!name || len <= 0) return CUDA_ERROR_INVALID_VALUE;

        if (should_virtualize_device_info()) {
            // Return fake device name
            GlobalState::instance().initialize();
            int count = GlobalState::instance().get_device_count();
            if (dev < 0 || dev >= count) return CUDA_ERROR_INVALID_DEVICE;

            Device& device = GlobalState::instance().get_device(dev);
            strncpy(name, device.name.c_str(), len - 1);
            name[len - 1] = '\0';
            return CUDA_SUCCESS;
        }

        if (should_passthrough()) {
            return CudaDriverPassthrough::instance().cuDeviceGetName(name, len, dev);
        }

        // Fallback to fake
        GlobalState::instance().initialize();
        int count = GlobalState::instance().get_device_count();
        if (dev < 0 || dev >= count) return CUDA_ERROR_INVALID_DEVICE;

        Device& device = GlobalState::instance().get_device(dev);
        strncpy(name, device.name.c_str(), len - 1);
        name[len - 1] = '\0';
        return CUDA_SUCCESS;
    }

    // cuDeviceTotalMem - may return fake or real memory
    CUresult dispatch_cuDeviceTotalMem(size_t* bytes, CUdevice dev) {
        if (!bytes) return CUDA_ERROR_INVALID_VALUE;

        if (should_virtualize_device_info()) {
            GlobalState::instance().initialize();
            int count = GlobalState::instance().get_device_count();
            if (dev < 0 || dev >= count) return CUDA_ERROR_INVALID_DEVICE;

            Device& device = GlobalState::instance().get_device(dev);
            *bytes = device.total_memory;
            return CUDA_SUCCESS;
        }

        if (should_passthrough()) {
            return CudaDriverPassthrough::instance().cuDeviceTotalMem(bytes, dev);
        }

        GlobalState::instance().initialize();
        int count = GlobalState::instance().get_device_count();
        if (dev < 0 || dev >= count) return CUDA_ERROR_INVALID_DEVICE;

        Device& device = GlobalState::instance().get_device(dev);
        *bytes = device.total_memory;
        return CUDA_SUCCESS;
    }

    // cuMemGetInfo - may apply clamping in hybrid mode
    CUresult dispatch_cuMemGetInfo(size_t* free, size_t* total) {
        FakeGpuMode m = mode();

        if (m == FakeGpuMode::Passthrough && should_passthrough()) {
            // Pure passthrough - return real values
            return CudaDriverPassthrough::instance().cuMemGetInfo(free, total);
        }

        if (m == FakeGpuMode::Hybrid && should_passthrough()) {
            // Hybrid mode - may clamp values
            size_t real_free = 0, real_total = 0;
            CUresult result = CudaDriverPassthrough::instance().cuMemGetInfo(&real_free, &real_total);
            if (result != CUDA_SUCCESS) {
                // Fall back to fake values
                GlobalState::instance().initialize();
                int current_dev = GlobalState::instance().get_current_device();
                Device& dev = GlobalState::instance().get_device(current_dev);
                if (total) *total = dev.total_memory;
                if (free) *free = dev.total_memory - dev.used_memory;
                return CUDA_SUCCESS;
            }

            // Apply clamping if configured
            int current_dev = GlobalState::instance().get_current_device();
            Device& dev = GlobalState::instance().get_device(current_dev);

            size_t reported_total = 0, reported_free = 0;
            HybridMemoryManager::instance().get_clamped_memory_info(
                current_dev, dev.total_memory, dev.used_memory,
                reported_total, reported_free);

            if (total) *total = reported_total;
            if (free) *free = reported_free;
            return CUDA_SUCCESS;
        }

        // Simulate mode - return fake values
        GlobalState::instance().initialize();
        int current_dev = GlobalState::instance().get_current_device();
        Device& dev = GlobalState::instance().get_device(current_dev);
        if (total) *total = dev.total_memory;
        if (free) *free = dev.total_memory - dev.used_memory;
        return CUDA_SUCCESS;
    }

    // cuMemAlloc - may use different allocation strategies in hybrid mode
    CUresult dispatch_cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) {
        if (!dptr) return CUDA_ERROR_INVALID_VALUE;

        FakeGpuMode m = mode();

        if (m == FakeGpuMode::Passthrough && should_passthrough()) {
            // Pure passthrough
            return CudaDriverPassthrough::instance().cuMemAlloc(dptr, bytesize);
        }

        if (m == FakeGpuMode::Hybrid && should_passthrough()) {
            // Hybrid mode - decide allocation strategy
            int current_dev = GlobalState::instance().get_current_device();
            auto decision = HybridMemoryManager::instance().decide_allocation(current_dev, bytesize);

            CUresult result = CUDA_SUCCESS;
            HybridAllocation::Type alloc_type = HybridAllocation::Type::RealDevice;

            switch (decision) {
                case HybridMemoryManager::AllocationDecision::UseReal:
                    result = CudaDriverPassthrough::instance().cuMemAlloc(dptr, bytesize);
                    alloc_type = HybridAllocation::Type::RealDevice;
                    break;

                case HybridMemoryManager::AllocationDecision::UseManaged:
                    result = CudaDriverPassthrough::instance().cuMemAllocManaged(dptr, bytesize, 1);
                    alloc_type = HybridAllocation::Type::Managed;
                    break;

                case HybridMemoryManager::AllocationDecision::UseMappedHost:
                    // Fall through to spill for now (mapped host requires more setup)
                case HybridMemoryManager::AllocationDecision::SpillToCpu:
                    // Allocate in system RAM
                    {
                        void* ptr = malloc(bytesize);
                        if (!ptr) return CUDA_ERROR_OUT_OF_MEMORY;
                        *dptr = (CUdeviceptr)ptr;
                        alloc_type = HybridAllocation::Type::SpilledCpu;
                    }
                    break;

                case HybridMemoryManager::AllocationDecision::Fail:
                    return CUDA_ERROR_OUT_OF_MEMORY;
            }

            if (result == CUDA_SUCCESS) {
                HybridMemoryManager::instance().record_allocation(
                    (void*)*dptr, bytesize, current_dev, alloc_type);
            }
            return result;
        }

        // Simulate mode - use fake allocation (handled by original stub)
        return CUDA_ERROR_NOT_INITIALIZED; // Signal to use fake implementation
    }

    // cuMemFree - handle different allocation types in hybrid mode
    CUresult dispatch_cuMemFree(CUdeviceptr dptr) {
        if (dptr == 0) return CUDA_SUCCESS;

        FakeGpuMode m = mode();

        if (m == FakeGpuMode::Passthrough && should_passthrough()) {
            return CudaDriverPassthrough::instance().cuMemFree(dptr);
        }

        if (m == FakeGpuMode::Hybrid && should_passthrough()) {
            HybridAllocation alloc;
            if (HybridMemoryManager::instance().release_allocation((void*)dptr, alloc)) {
                switch (alloc.type) {
                    case HybridAllocation::Type::RealDevice:
                    case HybridAllocation::Type::Managed:
                        return CudaDriverPassthrough::instance().cuMemFree(dptr);

                    case HybridAllocation::Type::MappedHost:
                    case HybridAllocation::Type::SpilledCpu:
                        free((void*)dptr);
                        return CUDA_SUCCESS;
                }
            }
            // Not tracked - try real free anyway
            return CudaDriverPassthrough::instance().cuMemFree(dptr);
        }

        return CUDA_ERROR_NOT_INITIALIZED; // Signal to use fake implementation
    }

    // cuMemcpy* - passthrough in passthrough/hybrid modes
    CUresult dispatch_cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t byteCount) {
        if (should_passthrough() && mode() != FakeGpuMode::Simulate) {
            // Check if destination is spilled to CPU
            if (mode() == FakeGpuMode::Hybrid) {
                HybridAllocation alloc;
                if (HybridMemoryManager::instance().get_allocation_info((void*)dstDevice, alloc)) {
                    if (alloc.type == HybridAllocation::Type::SpilledCpu) {
                        // Direct memcpy for spilled allocations
                        memcpy((void*)dstDevice, srcHost, byteCount);
                        return CUDA_SUCCESS;
                    }
                }
            }
            return CudaDriverPassthrough::instance().cuMemcpyHtoD(dstDevice, srcHost, byteCount);
        }
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    CUresult dispatch_cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t byteCount) {
        if (should_passthrough() && mode() != FakeGpuMode::Simulate) {
            if (mode() == FakeGpuMode::Hybrid) {
                HybridAllocation alloc;
                if (HybridMemoryManager::instance().get_allocation_info((void*)srcDevice, alloc)) {
                    if (alloc.type == HybridAllocation::Type::SpilledCpu) {
                        memcpy(dstHost, (void*)srcDevice, byteCount);
                        return CUDA_SUCCESS;
                    }
                }
            }
            return CudaDriverPassthrough::instance().cuMemcpyDtoH(dstHost, srcDevice, byteCount);
        }
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    CUresult dispatch_cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t byteCount) {
        if (should_passthrough() && mode() != FakeGpuMode::Simulate) {
            if (mode() == FakeGpuMode::Hybrid) {
                HybridAllocation dst_alloc, src_alloc;
                bool dst_spilled = HybridMemoryManager::instance().get_allocation_info((void*)dstDevice, dst_alloc) &&
                                   dst_alloc.type == HybridAllocation::Type::SpilledCpu;
                bool src_spilled = HybridMemoryManager::instance().get_allocation_info((void*)srcDevice, src_alloc) &&
                                   src_alloc.type == HybridAllocation::Type::SpilledCpu;

                if (dst_spilled && src_spilled) {
                    memcpy((void*)dstDevice, (void*)srcDevice, byteCount);
                    return CUDA_SUCCESS;
                }
                // Mixed case - need to handle carefully
                // For now, if either is spilled, use memcpy (may not work for real GPU pointers)
                if (dst_spilled || src_spilled) {
                    memcpy((void*)dstDevice, (void*)srcDevice, byteCount);
                    return CUDA_SUCCESS;
                }
            }
            return CudaDriverPassthrough::instance().cuMemcpyDtoD(dstDevice, srcDevice, byteCount);
        }
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    // cuLaunchKernel - passthrough for real compute
    CUresult dispatch_cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                      unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                      unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra) {
        if (should_passthrough() && mode() != FakeGpuMode::Simulate) {
            return CudaDriverPassthrough::instance().cuLaunchKernel(
                f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                sharedMemBytes, hStream, kernelParams, extra);
        }
        // Simulate mode - no-op
        FGPU_LOG("[CudaModeDispatch] cuLaunchKernel (no-op in simulate mode)\n");
        return CUDA_SUCCESS;
    }

    // Context management - passthrough in passthrough mode, fake in simulate/hybrid
    CUresult dispatch_cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev) {
        if (mode() == FakeGpuMode::Passthrough && should_passthrough()) {
            return CudaDriverPassthrough::instance().cuCtxCreate(pctx, flags, dev);
        }
        return CUDA_ERROR_NOT_INITIALIZED; // Use fake implementation
    }

    CUresult dispatch_cuCtxDestroy(CUcontext ctx) {
        if (mode() == FakeGpuMode::Passthrough && should_passthrough()) {
            return CudaDriverPassthrough::instance().cuCtxDestroy(ctx);
        }
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    CUresult dispatch_cuCtxSetCurrent(CUcontext ctx) {
        if (mode() == FakeGpuMode::Passthrough && should_passthrough()) {
            return CudaDriverPassthrough::instance().cuCtxSetCurrent(ctx);
        }
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    CUresult dispatch_cuCtxGetCurrent(CUcontext* pctx) {
        if (mode() == FakeGpuMode::Passthrough && should_passthrough()) {
            return CudaDriverPassthrough::instance().cuCtxGetCurrent(pctx);
        }
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    CUresult dispatch_cuCtxSynchronize() {
        if (should_passthrough() && mode() != FakeGpuMode::Simulate) {
            return CudaDriverPassthrough::instance().cuCtxSynchronize();
        }
        return CUDA_SUCCESS; // No-op in simulate mode
    }

    // Stream management
    CUresult dispatch_cuStreamCreate(CUstream* phStream, unsigned int flags) {
        if (should_passthrough() && mode() != FakeGpuMode::Simulate) {
            return CudaDriverPassthrough::instance().cuStreamCreate(phStream, flags);
        }
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    CUresult dispatch_cuStreamDestroy(CUstream hStream) {
        if (should_passthrough() && mode() != FakeGpuMode::Simulate) {
            return CudaDriverPassthrough::instance().cuStreamDestroy(hStream);
        }
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    CUresult dispatch_cuStreamSynchronize(CUstream hStream) {
        if (should_passthrough() && mode() != FakeGpuMode::Simulate) {
            return CudaDriverPassthrough::instance().cuStreamSynchronize(hStream);
        }
        return CUDA_SUCCESS;
    }

    // Event management
    CUresult dispatch_cuEventCreate(CUevent* phEvent, unsigned int flags) {
        if (should_passthrough() && mode() != FakeGpuMode::Simulate) {
            return CudaDriverPassthrough::instance().cuEventCreate(phEvent, flags);
        }
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    CUresult dispatch_cuEventDestroy(CUevent hEvent) {
        if (should_passthrough() && mode() != FakeGpuMode::Simulate) {
            return CudaDriverPassthrough::instance().cuEventDestroy(hEvent);
        }
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    CUresult dispatch_cuEventRecord(CUevent hEvent, CUstream hStream) {
        if (should_passthrough() && mode() != FakeGpuMode::Simulate) {
            return CudaDriverPassthrough::instance().cuEventRecord(hEvent, hStream);
        }
        return CUDA_SUCCESS;
    }

    CUresult dispatch_cuEventSynchronize(CUevent hEvent) {
        if (should_passthrough() && mode() != FakeGpuMode::Simulate) {
            return CudaDriverPassthrough::instance().cuEventSynchronize(hEvent);
        }
        return CUDA_SUCCESS;
    }

    CUresult dispatch_cuEventElapsedTime(float* pMilliseconds, CUevent hStart, CUevent hEnd) {
        if (should_passthrough() && mode() != FakeGpuMode::Simulate) {
            return CudaDriverPassthrough::instance().cuEventElapsedTime(pMilliseconds, hStart, hEnd);
        }
        if (pMilliseconds) *pMilliseconds = 0.0f;
        return CUDA_SUCCESS;
    }

    // Module management
    CUresult dispatch_cuModuleLoad(CUmodule* module, const char* fname) {
        if (should_passthrough() && mode() != FakeGpuMode::Simulate) {
            return CudaDriverPassthrough::instance().cuModuleLoad(module, fname);
        }
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    CUresult dispatch_cuModuleLoadData(CUmodule* module, const void* image) {
        if (should_passthrough() && mode() != FakeGpuMode::Simulate) {
            return CudaDriverPassthrough::instance().cuModuleLoadData(module, image);
        }
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    CUresult dispatch_cuModuleUnload(CUmodule hmod) {
        if (should_passthrough() && mode() != FakeGpuMode::Simulate) {
            return CudaDriverPassthrough::instance().cuModuleUnload(hmod);
        }
        return CUDA_SUCCESS;
    }

    CUresult dispatch_cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) {
        if (should_passthrough() && mode() != FakeGpuMode::Simulate) {
            return CudaDriverPassthrough::instance().cuModuleGetFunction(hfunc, hmod, name);
        }
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    // Primary context
    CUresult dispatch_cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) {
        if (should_passthrough() && mode() != FakeGpuMode::Simulate) {
            return CudaDriverPassthrough::instance().cuDevicePrimaryCtxRetain(pctx, dev);
        }
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    CUresult dispatch_cuDevicePrimaryCtxRelease(CUdevice dev) {
        if (should_passthrough() && mode() != FakeGpuMode::Simulate) {
            return CudaDriverPassthrough::instance().cuDevicePrimaryCtxRelease(dev);
        }
        return CUDA_SUCCESS;
    }

private:
    CudaModeDispatch() = default;
    bool initialized_ = false;
};

} // namespace fake_gpu
