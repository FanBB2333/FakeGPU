#pragma once

#include "cuda_driver_defs.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <vector>

// Simulate-mode state shared by libcuda and libcudart. Both libraries embed
// the Driver API objects; on macOS each image would otherwise keep its own
// copy, so cudaHostRegister followed by cuMemHostUnregister, or a VMM range
// created through one library and queried through the other, would disagree.
// shared_state() resolves one process-wide instance the same way GlobalState
// does (see fakegpu_cuda_driver_state_v1).
namespace fake_gpu::cuda::model {

// The CUDA C++ Programming Guide (12.4, "Device Memory Accesses") guarantees
// that any address returned by the allocation routines is aligned to at least
// 256 bytes. Host-backed pinned and managed allocations are page aligned.
inline constexpr std::size_t kDeviceAlignment = 256;
inline constexpr std::size_t kHostAlignment = 4096;

// Modeled default: the minimum and recommended VMM granularity reported by
// cuMemGetAllocationGranularity, measured once on an Ampere GPU (sm_80).
inline constexpr std::size_t kVmmGranularity = std::size_t{2} << 20;

// Modeled default for the memory the CUDA context holds outside the caller's
// allocations (601.5 MiB, measured once on an Ampere GPU (sm_80) with driver
// 580). FAKEGPU_CUDA_CONTEXT_BYTES overrides it; 0 disables it. The same
// variable drives FakeCUDA's mem_get_info().
inline constexpr std::uint64_t kDefaultContextBytes = 630718464;
std::uint64_t context_bytes();

// cuMemHostRegister ranges and cuMemAllocHost/cuMemHostAlloc allocations.
struct HostRange {
    std::size_t size = 0;
    unsigned int flags = 0;
    int device = 0;
    bool pinned_allocation = false;
};

struct VmmAllocation {
    void* home = nullptr;  // shared host pages every mapping aliases
    std::size_t size = 0;
    int device = -1;       // -1: host location, not counted against a device
    CUmemAllocationProp prop{};
    unsigned int references = 0;
    unsigned int mappings = 0;
};

struct VmmMapping {
    std::size_t size = 0;
    CUmemGenericAllocationHandle handle = 0;
    std::map<int, unsigned long long> access;  // device ordinal -> CUmemAccess_flags
};

struct VmmReservation {
    std::size_t size = 0;
    std::map<std::uintptr_t, VmmMapping> mappings;  // keyed by mapping start
};

// CUlimit 0-6 are settable context limits.
inline constexpr int kSettableLimitCount = 7;

struct DeviceContext {
    unsigned int primary_retains = 0;
    unsigned int primary_flags = 0;
    bool runtime_retained = false;
    bool limits_initialized = false;
    std::array<std::size_t, kSettableLimitCount> limits{};
};

struct SharedState {
    std::mutex mutex;
    bool driver_initialized = false;
    std::map<std::uintptr_t, HostRange> host_ranges;
    std::map<CUmemGenericAllocationHandle, VmmAllocation> vmm_allocations;
    std::map<std::uintptr_t, VmmReservation> vmm_reservations;
    CUmemGenericAllocationHandle next_vmm_handle = 0x10000;
    std::vector<DeviceContext> devices;
};

SharedState& shared_state();

// The caller holds state.mutex for the helpers below.
DeviceContext& device_context_locked(SharedState& state, int device);

// The host range containing address, if any.
const HostRange* find_host_range_locked(const SharedState& state, std::uintptr_t address, std::uintptr_t& base);

struct VmmAddress {
    std::uintptr_t reservation_base = 0;
    std::size_t reservation_size = 0;
    std::uintptr_t mapping_base = 0;        // 0 when the address is reserved but unmapped
    const VmmMapping* mapping = nullptr;
    const VmmAllocation* allocation = nullptr;
};
bool find_vmm_address_locked(const SharedState& state, std::uintptr_t address, VmmAddress& out);

// Host virtual memory used to model VMM reservations and mappings.
void* reserve_address_range(std::size_t size, std::size_t alignment, std::uintptr_t hint);
void release_address_range(void* base, std::size_t size);
void* create_backing(std::size_t size);
void destroy_backing(void* home, std::size_t size);
bool alias_backing(void* home, std::uintptr_t target, std::size_t size);
bool unmap_alias(std::uintptr_t target, std::size_t size);

// Context limits (CUlimit). Defaults and rounding are modeled on the values
// one Ampere GPU (sm_80) reported; see driver_model.cpp.
CUresult get_context_limit(int device, int limit, std::size_t* value);
CUresult set_context_limit(int device, int limit, std::size_t value);

// Stream priorities: [greatest, least] = [-5, 0], measured on one Ampere GPU.
inline constexpr int kLeastStreamPriority = 0;
inline constexpr int kGreatestStreamPriority = -5;
inline int clamp_stream_priority(int priority) {
    return priority < kGreatestStreamPriority ? kGreatestStreamPriority
           : priority > kLeastStreamPriority  ? kLeastStreamPriority
                                              : priority;
}

}  // namespace fake_gpu::cuda::model
