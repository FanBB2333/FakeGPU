#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "backend_config.hpp"
#include "real_cuda_loader.hpp"
#include "logging.hpp"

namespace fake_gpu {

// Allocation tracking for Hybrid mode
struct HybridAllocation {
    void* ptr = nullptr;
    void* backing_ptr = nullptr;  // e.g. host pointer for mapped host allocations
    size_t size = 0;
    int device = 0;

    enum class Type {
        RealDevice,    // Allocated on real GPU
        Managed,       // cudaMallocManaged
        MappedHost,    // cudaHostAllocMapped
        SpilledCpu     // Spilled to CPU (fake memory)
    };
    Type type = Type::RealDevice;
};

// Memory budget tracker for Hybrid mode
class HybridMemoryManager {
public:
    static HybridMemoryManager& instance() {
        static HybridMemoryManager mgr;
        return mgr;
    }

    // Initialize with real GPU info
    void initialize() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (initialized_) return;

        const BackendConfig& config = BackendConfig::instance();
        if (config.mode() != FakeGpuMode::Hybrid) {
            initialized_ = true;
            return;
        }

        // Get real GPU info
        RealCudaLoader::instance().initialize();
        const auto& real_info = RealCudaLoader::instance().get_real_gpu_info();

        if (real_info.valid) {
            real_device_count_ = real_info.device_count;
            for (int i = 0; i < real_device_count_ && i < 16; ++i) {
                real_total_memory_[i] = real_info.total_memory[i];
                real_free_memory_[i] = real_info.free_memory[i];
                real_used_memory_[i] = 0;
            }
            FGPU_LOG("[HybridMemoryManager] Initialized with %d real GPU(s)\n", real_device_count_);
        } else {
            FGPU_LOG("[HybridMemoryManager] WARNING: No real GPUs available, falling back to simulate mode\n");
        }

        initialized_ = true;
    }

    // Get clamped memory info (for Hybrid mode with clamp policy)
    void get_clamped_memory_info(int device, size_t virtual_total, size_t virtual_used,
                                  size_t& reported_total, size_t& reported_free) {
        std::lock_guard<std::mutex> lock(mutex_);

        const BackendConfig& config = BackendConfig::instance();

        if (config.mode() != FakeGpuMode::Hybrid ||
            config.oom_policy() != OomPolicy::Clamp) {
            // Not in clamp mode, return virtual values
            reported_total = virtual_total;
            reported_free = (virtual_total > virtual_used) ? (virtual_total - virtual_used) : 0;
            return;
        }

        if (real_device_count_ <= 0) {
            // No real backing device available; fall back to virtual values.
            reported_total = virtual_total;
            reported_free = (virtual_total > virtual_used) ? (virtual_total - virtual_used) : 0;
            return;
        }

        // Map virtual device to real device
        int real_dev = device % real_device_count_;
        if (real_dev < 0 || real_dev >= real_device_count_) {
            reported_total = virtual_total;
            reported_free = (virtual_total > virtual_used) ? (virtual_total - virtual_used) : 0;
            return;
        }

        // Clamp to real GPU capacity
        size_t real_total = real_total_memory_[real_dev];
        size_t real_used = real_used_memory_[real_dev];

        // Report the minimum of virtual and real
        reported_total = std::min(virtual_total, real_total);
        reported_free = (reported_total > real_used) ? (reported_total - real_used) : 0;

        FGPU_LOG("[HybridMemoryManager] Clamped memory: virtual=%zu/%zu, real=%zu/%zu, reported=%zu/%zu\n",
                virtual_used, virtual_total, real_used, real_total, reported_total - reported_free, reported_total);
    }

    // Decide allocation strategy based on OOM policy
    enum class AllocationDecision {
        UseReal,       // Allocate on real GPU
        UseManaged,    // Use cudaMallocManaged
        UseMappedHost, // Use cudaHostAllocMapped
        SpillToCpu,    // Spill to CPU memory
        Fail           // Cannot allocate
    };

    AllocationDecision decide_allocation(int device, size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);

        const BackendConfig& config = BackendConfig::instance();

        if (config.mode() != FakeGpuMode::Hybrid) {
            return AllocationDecision::UseReal;
        }

        if (real_device_count_ <= 0) {
            // No real GPU available; always spill so callers can continue in a degraded mode.
            return AllocationDecision::SpillToCpu;
        }

        int real_dev = device % real_device_count_;
        if (real_dev < 0 || real_dev >= real_device_count_) {
            return AllocationDecision::SpillToCpu;
        }

        size_t available = real_total_memory_[real_dev] - real_used_memory_[real_dev];

        // Check if we have enough real memory
        if (size <= available) {
            return AllocationDecision::UseReal;
        }

        // We're over the real budget: count this as an OOM fallback attempt (or OOM failure in clamp mode).
        stats_.oom_fallback_count++;

        // Not enough real memory, apply OOM policy
        switch (config.oom_policy()) {
            case OomPolicy::Clamp:
                // In clamp mode, we should have already reported smaller memory
                // If we still get here, fail the allocation
                FGPU_LOG("[HybridMemoryManager] Clamp policy: allocation of %zu bytes would exceed real GPU memory\n", size);
                return AllocationDecision::Fail;

            case OomPolicy::Managed:
                FGPU_LOG("[HybridMemoryManager] Using managed memory for %zu bytes\n", size);
                return AllocationDecision::UseManaged;

            case OomPolicy::MappedHost:
                FGPU_LOG("[HybridMemoryManager] Using mapped host memory for %zu bytes\n", size);
                return AllocationDecision::UseMappedHost;

            case OomPolicy::SpillCpu:
                FGPU_LOG("[HybridMemoryManager] Spilling %zu bytes to CPU\n", size);
                return AllocationDecision::SpillToCpu;
        }

        return AllocationDecision::Fail;
    }

    // Track allocation
    void record_allocation(void* ptr, size_t size, int device, HybridAllocation::Type type, void* backing_ptr = nullptr) {
        std::lock_guard<std::mutex> lock(mutex_);

        HybridAllocation alloc;
        alloc.ptr = ptr;
        alloc.backing_ptr = backing_ptr;
        alloc.size = size;
        alloc.device = device;
        alloc.type = type;
        allocations_[ptr] = alloc;

        // Update memory tracking
        if (type == HybridAllocation::Type::RealDevice) {
            if (real_device_count_ > 0) {
                int real_dev = device % real_device_count_;
                if (real_dev >= 0 && real_dev < real_device_count_) {
                    real_used_memory_[real_dev] += size;
                }
            }
        }

        // Update statistics
        switch (type) {
            case HybridAllocation::Type::RealDevice:
                stats_.real_alloc_count++;
                stats_.real_alloc_bytes += size;
                break;
            case HybridAllocation::Type::Managed:
                stats_.managed_alloc_count++;
                stats_.managed_alloc_bytes += size;
                break;
            case HybridAllocation::Type::MappedHost:
                stats_.mapped_host_alloc_count++;
                stats_.mapped_host_alloc_bytes += size;
                break;
            case HybridAllocation::Type::SpilledCpu:
                stats_.spilled_alloc_count++;
                stats_.spilled_alloc_bytes += size;
                break;
        }
    }

    // Release allocation
    bool release_allocation(void* ptr, HybridAllocation& out_alloc) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = allocations_.find(ptr);
        if (it == allocations_.end()) {
            return false;
        }

        out_alloc = it->second;
        allocations_.erase(it);

        // Update memory tracking
        if (out_alloc.type == HybridAllocation::Type::RealDevice) {
            if (real_device_count_ <= 0) {
                return true;
            }
            int real_dev = out_alloc.device % real_device_count_;
            if (real_dev >= 0 && real_dev < real_device_count_) {
                if (real_used_memory_[real_dev] >= out_alloc.size) {
                    real_used_memory_[real_dev] -= out_alloc.size;
                } else {
                    real_used_memory_[real_dev] = 0;
                }
            }
        }

        return true;
    }

    // Get allocation info
    bool get_allocation_info(void* ptr, HybridAllocation& out_alloc) const {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = allocations_.find(ptr);
        if (it == allocations_.end()) {
            return false;
        }

        out_alloc = it->second;
        return true;
    }

    // Statistics for reporting
    struct Stats {
        uint64_t real_alloc_count = 0;
        uint64_t real_alloc_bytes = 0;
        uint64_t managed_alloc_count = 0;
        uint64_t managed_alloc_bytes = 0;
        uint64_t mapped_host_alloc_count = 0;
        uint64_t mapped_host_alloc_bytes = 0;
        uint64_t spilled_alloc_count = 0;
        uint64_t spilled_alloc_bytes = 0;
        uint64_t oom_fallback_count = 0;
    };

    Stats get_stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stats_;
    }

    // Get real GPU info for reporting
    int get_real_device_count() const { return real_device_count_; }
    size_t get_real_total_memory(int device) const {
        if (device < 0 || device >= real_device_count_) return 0;
        return real_total_memory_[device];
    }
    size_t get_real_used_memory(int device) const {
        if (device < 0 || device >= real_device_count_) return 0;
        return real_used_memory_[device];
    }

private:
    HybridMemoryManager() = default;

    mutable std::mutex mutex_;
    bool initialized_ = false;

    int real_device_count_ = 0;
    size_t real_total_memory_[16] = {0};
    size_t real_free_memory_[16] = {0};
    size_t real_used_memory_[16] = {0};

    std::unordered_map<void*, HybridAllocation> allocations_;
    Stats stats_;
};

} // namespace fake_gpu
