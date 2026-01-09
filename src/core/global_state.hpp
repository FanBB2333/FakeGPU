#pragma once
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <cstddef>
#include <cstdint>
#include <string>
#include "device.hpp"

namespace fake_gpu {

struct HostIoStats {
    uint64_t memcpy_calls = 0;
    uint64_t memcpy_bytes = 0;
};

struct DeviceReportStats {
    int index = -1;
    std::string name;
    std::string uuid;
    uint64_t total_memory = 0;
    uint64_t used_memory_current = 0;
    uint64_t used_memory_peak = 0;

    uint64_t alloc_calls = 0;
    uint64_t alloc_bytes = 0;
    uint64_t free_calls = 0;
    uint64_t free_bytes = 0;

    uint64_t memcpy_h2d_calls = 0;
    uint64_t memcpy_h2d_bytes = 0;
    uint64_t memcpy_d2h_calls = 0;
    uint64_t memcpy_d2h_bytes = 0;
    uint64_t memcpy_d2d_calls = 0;
    uint64_t memcpy_d2d_bytes = 0;
    uint64_t memcpy_peer_tx_calls = 0;
    uint64_t memcpy_peer_tx_bytes = 0;
    uint64_t memcpy_peer_rx_calls = 0;
    uint64_t memcpy_peer_rx_bytes = 0;
    uint64_t memset_calls = 0;
    uint64_t memset_bytes = 0;

    uint64_t cublas_gemm_calls = 0;
    uint64_t cublas_gemm_flops = 0;
    uint64_t cublaslt_matmul_calls = 0;
    uint64_t cublaslt_matmul_flops = 0;
};

class GlobalState {
public:
    static GlobalState& instance();

    enum class AllocationKind {
        Device,
        Managed,
        Host,
    };

    void initialize();
    int get_device_count() const;
    Device& get_device(int index);
    void set_current_device(int device);
    int get_current_device() const;

    // Allocation tracking
    bool register_allocation(void* ptr, size_t size, int device);
    bool register_managed_allocation(void* ptr, size_t size, int device);
    bool register_host_allocation(void* ptr, size_t size, int device);
    bool release_allocation(void* ptr, size_t& size, int& device);
    bool release_host_allocation(void* ptr, size_t& size, int& device);
    bool get_allocation_info(void* ptr, size_t& size, int& device) const;
    bool get_allocation_info_ex(void* ptr, size_t& size, int& device, AllocationKind& kind) const;

    // IO tracking (best-effort)
    void record_memcpy_h2d(const void* dst_device_ptr, size_t bytes);
    void record_memcpy_d2h(const void* src_device_ptr, size_t bytes);
    void record_memcpy_d2d(const void* dst_device_ptr, const void* src_device_ptr, size_t bytes);
    void record_memcpy_peer(int dst_device, int src_device, size_t bytes);
    void record_memcpy_h2h(size_t bytes);
    void record_memset(const void* dst_device_ptr, size_t bytes);

    // Compute tracking (best-effort)
    void record_cublas_gemm(const void* output_device_ptr, uint64_t flops);
    void record_cublaslt_matmul(const void* output_device_ptr, uint64_t flops);

    // Snapshot for reporting (thread-safe)
    std::vector<DeviceReportStats> snapshot_device_report() const;
    HostIoStats snapshot_host_io() const;

private:
    GlobalState();
    ~GlobalState() = default;

    struct DeviceRuntimeStats {
        uint64_t alloc_calls = 0;
        uint64_t alloc_bytes = 0;
        uint64_t free_calls = 0;
        uint64_t free_bytes = 0;

        uint64_t memcpy_h2d_calls = 0;
        uint64_t memcpy_h2d_bytes = 0;
        uint64_t memcpy_d2h_calls = 0;
        uint64_t memcpy_d2h_bytes = 0;
        uint64_t memcpy_d2d_calls = 0;
        uint64_t memcpy_d2d_bytes = 0;
        uint64_t memcpy_peer_tx_calls = 0;
        uint64_t memcpy_peer_tx_bytes = 0;
        uint64_t memcpy_peer_rx_calls = 0;
        uint64_t memcpy_peer_rx_bytes = 0;
        uint64_t memset_calls = 0;
        uint64_t memset_bytes = 0;

        uint64_t cublas_gemm_calls = 0;
        uint64_t cublas_gemm_flops = 0;
        uint64_t cublaslt_matmul_calls = 0;
        uint64_t cublaslt_matmul_flops = 0;
    };

    bool initialized = false;
    std::vector<Device> devices;
    std::vector<DeviceRuntimeStats> device_stats;
    HostIoStats host_io;
    mutable std::mutex mutex;

    int current_device = 0;

    struct AllocationRecord {
        size_t size = 0;
        int device = 0;
        AllocationKind kind = AllocationKind::Device;
    };

    std::unordered_map<void*, AllocationRecord> allocations;

    int resolve_device_for_ptr_nolock(const void* ptr, int fallback_device) const;
    DeviceRuntimeStats* stats_for_device_nolock(int device);
    const DeviceRuntimeStats* stats_for_device_nolock(int device) const;
    static void saturating_add_u64(uint64_t& target, uint64_t value);

    bool register_allocation_nolock(void* ptr, size_t size, int device, AllocationKind kind);
};

} // namespace fake_gpu
