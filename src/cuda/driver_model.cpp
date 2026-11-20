#include "driver_model.hpp"

#include "../core/global_state.hpp"

#include <dlfcn.h>
#include <sys/mman.h>

#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_vm.h>
#endif

#ifndef MAP_NORESERVE
#define MAP_NORESERVE 0
#endif

namespace fake_gpu::cuda::model {
namespace {

SharedState& local_state() {
    static SharedState* state = new SharedState();
    return *state;
}

}  // namespace
}  // namespace fake_gpu::cuda::model

// Every image that embeds the Driver API objects exports this provider; the
// first one the dynamic linker finds owns the state for the whole process.
extern "C" __attribute__((visibility("default"))) void* fakegpu_cuda_driver_state_v1() {
    return &fake_gpu::cuda::model::local_state();
}

namespace fake_gpu::cuda::model {

SharedState& shared_state() {
    static SharedState* selected = []() {
        using provider_t = void* (*)();
        if (void* symbol = dlsym(RTLD_DEFAULT, "fakegpu_cuda_driver_state_v1")) {
            if (void* shared = reinterpret_cast<provider_t>(symbol)()) {
                return static_cast<SharedState*>(shared);
            }
        }
        return &local_state();
    }();
    return *selected;
}

std::uint64_t context_bytes() {
    // Parsed like FakeCUDA's allocator: a decimal integer, negatives clamp to
    // 0, anything else falls back to the default.
    const char* raw = std::getenv("FAKEGPU_CUDA_CONTEXT_BYTES");
    if (!raw) return kDefaultContextBytes;
    std::string text(raw);
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return kDefaultContextBytes;
    const auto last = text.find_last_not_of(" \t\r\n");
    text = text.substr(first, last - first + 1);
    errno = 0;
    char* end = nullptr;
    const long long value = std::strtoll(text.c_str(), &end, 10);
    if (errno != 0 || !end || *end != '\0' || end == text.c_str()) return kDefaultContextBytes;
    return value < 0 ? 0 : static_cast<std::uint64_t>(value);
}

DeviceContext& device_context_locked(SharedState& state, int device) {
    if (device < 0) device = 0;
    if (static_cast<std::size_t>(device) >= state.devices.size()) {
        state.devices.resize(static_cast<std::size_t>(device) + 1);
    }
    return state.devices[static_cast<std::size_t>(device)];
}

const HostRange* find_host_range_locked(const SharedState& state, std::uintptr_t address, std::uintptr_t& base) {
    auto it = state.host_ranges.upper_bound(address);
    if (it == state.host_ranges.begin()) return nullptr;
    --it;
    if (address - it->first >= it->second.size) return nullptr;
    base = it->first;
    return &it->second;
}

bool find_vmm_address_locked(const SharedState& state, std::uintptr_t address, VmmAddress& out) {
    auto reservation = state.vmm_reservations.upper_bound(address);
    if (reservation == state.vmm_reservations.begin()) return false;
    --reservation;
    if (address - reservation->first >= reservation->second.size) return false;
    out = VmmAddress{};
    out.reservation_base = reservation->first;
    out.reservation_size = reservation->second.size;
    const auto& mappings = reservation->second.mappings;
    auto mapping = mappings.upper_bound(address);
    if (mapping == mappings.begin()) return true;
    --mapping;
    if (address - mapping->first >= mapping->second.size) return true;
    out.mapping_base = mapping->first;
    out.mapping = &mapping->second;
    const auto allocation = state.vmm_allocations.find(mapping->second.handle);
    if (allocation != state.vmm_allocations.end()) out.allocation = &allocation->second;
    return true;
}

void* reserve_address_range(std::size_t size, std::size_t alignment, std::uintptr_t hint) {
    if (size == 0 || alignment == 0 || size > std::numeric_limits<std::size_t>::max() - alignment) return nullptr;
    const std::size_t span = size + alignment;
    void* raw = mmap(reinterpret_cast<void*>(hint), span, PROT_NONE,
                     MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0);
    if (raw == MAP_FAILED) return nullptr;
    const auto raw_start = reinterpret_cast<std::uintptr_t>(raw);
    const std::uintptr_t start = (raw_start + alignment - 1) / alignment * alignment;
    if (start > raw_start) munmap(raw, start - raw_start);
    const std::uintptr_t end = start + size;
    const std::uintptr_t raw_end = raw_start + span;
    if (raw_end > end) munmap(reinterpret_cast<void*>(end), raw_end - end);
    return reinterpret_cast<void*>(start);
}

void release_address_range(void* base, std::size_t size) {
    if (base && size) munmap(base, size);
}

void* create_backing(std::size_t size) {
    // Shared pages so every mapping of the handle aliases the same memory, as
    // device mappings of one physical allocation do.
    void* home = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANON, -1, 0);
    return home == MAP_FAILED ? nullptr : home;
}

void destroy_backing(void* home, std::size_t size) {
    if (home && size) munmap(home, size);
}

bool alias_backing(void* home, std::uintptr_t target, std::size_t size) {
#if defined(__APPLE__)
    mach_vm_address_t address = target;
    vm_prot_t current = VM_PROT_NONE;
    vm_prot_t maximum = VM_PROT_NONE;
    const kern_return_t result = mach_vm_remap(
        mach_task_self(), &address, size, 0, VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE,
        mach_task_self(), reinterpret_cast<mach_vm_address_t>(home), FALSE,
        &current, &maximum, VM_INHERIT_SHARE);
    return result == KERN_SUCCESS && address == target;
#else
    // An old size of 0 on a shared mapping creates a second mapping of the
    // same pages (mremap(2)); MREMAP_FIXED replaces the reserved range.
    void* mapped = mremap(home, 0, size, MREMAP_MAYMOVE | MREMAP_FIXED, reinterpret_cast<void*>(target));
    return mapped != MAP_FAILED && reinterpret_cast<std::uintptr_t>(mapped) == target;
#endif
}

bool unmap_alias(std::uintptr_t target, std::size_t size) {
    // Put an inaccessible reservation back in place of the mapping.
    void* replaced = mmap(reinterpret_cast<void*>(target), size, PROT_NONE,
                          MAP_PRIVATE | MAP_ANON | MAP_FIXED | MAP_NORESERVE, -1, 0);
    return replaced != MAP_FAILED;
}

namespace {

std::size_t round_up(std::size_t value, std::size_t multiple) {
    const std::size_t remainder = value % multiple;
    if (remainder == 0) return value;
    const std::size_t add = multiple - remainder;
    return value > std::numeric_limits<std::size_t>::max() - add ? std::numeric_limits<std::size_t>::max()
                                                                 : value + add;
}

// Defaults one Ampere GPU (sm_80) reported through cuCtxGetLimit and
// cudaDeviceGetLimit; the sync depth (2) and pending launch count (2048)
// defaults are also documented in cuda_runtime_api.h (cudaDeviceSetLimit).
constexpr std::array<std::size_t, kSettableLimitCount> kDefaultLimits = {
    1024,     // CU_LIMIT_STACK_SIZE
    6291456,  // CU_LIMIT_PRINTF_FIFO_SIZE
    8388608,  // CU_LIMIT_MALLOC_HEAP_SIZE
    2,        // CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH
    2048,     // CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT
    64,       // CU_LIMIT_MAX_L2_FETCH_GRANULARITY
    6291456,  // CU_LIMIT_PERSISTING_L2_CACHE_SIZE
};

DeviceContext& limits_locked(SharedState& state, int device) {
    DeviceContext& context = device_context_locked(state, device);
    if (!context.limits_initialized) {
        context.limits = kDefaultLimits;
        context.limits_initialized = true;
    }
    return context;
}

}  // namespace

CUresult get_context_limit(int device, int limit, std::size_t* value) {
    if (!value) return CUDA_ERROR_INVALID_VALUE;
    GlobalState& global = GlobalState::instance();
    global.initialize();
    const Device& profile_device = global.get_device(device);
    switch (limit) {
        case CU_LIMIT_SHMEM_SIZE:
            // Reported as the opt-in shared memory per block.
            *value = static_cast<std::size_t>(profile_device.profile.shared_mem_per_block_optin);
            return CUDA_SUCCESS;
        case CU_LIMIT_CIG_ENABLED:
            *value = 0;
            return CUDA_SUCCESS;
        case CU_LIMIT_CIG_SHMEM_FALLBACK_ENABLED:
            return CUDA_ERROR_UNSUPPORTED_LIMIT;
        default:
            break;
    }
    if (limit < 0 || limit >= kSettableLimitCount) return CUDA_ERROR_INVALID_VALUE;
    SharedState& state = shared_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    *value = limits_locked(state, device).limits[static_cast<std::size_t>(limit)];
    return CUDA_SUCCESS;
}

CUresult set_context_limit(int device, int limit, std::size_t value) {
    GlobalState& global = GlobalState::instance();
    global.initialize();
    const Device& profile_device = global.get_device(device);
    std::size_t stored = value;
    switch (limit) {
        case CU_LIMIT_STACK_SIZE:
            stored = round_up(value, 16);
            break;
        case CU_LIMIT_PRINTF_FIFO_SIZE:
            stored = round_up(value < (std::size_t{512} << 10) ? (std::size_t{512} << 10) : value, 256);
            break;
        case CU_LIMIT_MALLOC_HEAP_SIZE:
            stored = round_up(value < (std::size_t{4} << 20) ? (std::size_t{4} << 20) : value, std::size_t{64} << 10);
            break;
        case CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH:
            // cuda_runtime_api.h: compute capability < 9.0 only, up to a depth of 24.
            if (profile_device.profile.compute_major >= 9) return CUDA_ERROR_UNSUPPORTED_LIMIT;
            if (value > 24) return CUDA_ERROR_INVALID_VALUE;
            break;
        case CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT:
            stored = value < 32 ? 32 : value;
            break;
        case CU_LIMIT_MAX_L2_FETCH_GRANULARITY:
            if (value > 128) return CUDA_ERROR_INVALID_VALUE;
            break;
        case CU_LIMIT_PERSISTING_L2_CACHE_SIZE: {
            const std::size_t maximum = static_cast<std::size_t>(profile_device.profile.l2_cache_bytes);
            if (value > maximum) return CUDA_ERROR_INVALID_VALUE;
            stored = round_up(value, std::size_t{2} << 20);
            break;
        }
        case CU_LIMIT_SHMEM_SIZE:
        case CU_LIMIT_CIG_ENABLED:
            return CUDA_ERROR_NOT_PERMITTED;
        case CU_LIMIT_CIG_SHMEM_FALLBACK_ENABLED:
            return CUDA_ERROR_UNSUPPORTED_LIMIT;
        default:
            return CUDA_ERROR_INVALID_VALUE;
    }
    SharedState& state = shared_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    limits_locked(state, device).limits[static_cast<std::size_t>(limit)] = stored;
    return CUDA_SUCCESS;
}

}  // namespace fake_gpu::cuda::model
