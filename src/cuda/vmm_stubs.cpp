// CUDA virtual memory management (cuMemCreate, cuMemAddressReserve, cuMemMap,
// ...), the driver surface PyTorch's expandable segments allocator uses.
//
// Simulate mode backs each allocation handle with shared host pages and each
// reservation with an inaccessible host address range. cuMemMap aliases the
// handle's pages into the reservation, so two mappings of one handle see the
// same bytes, as on a GPU. The handle's size counts against the device from
// cuMemCreate until its last reference and mapping are gone. Access flags are
// recorded for cuMemGetAccess; mapped memory stays readable and writable on
// the host, so a missing cuMemSetAccess is not caught. Shareable handles and
// fabric allocations are not modeled.
#include "cuda_driver_defs.hpp"
#include "driver_model.hpp"
#include "../core/global_state.hpp"

#include <cstdint>
#include <iterator>
#include <mutex>
#include <vector>

using fake_gpu::GlobalState;
namespace model = fake_gpu::cuda::model;

namespace {

#define FORWARD_VMM(name, ...)                                                   \
    if (void* real_fn = fake_gpu::cuda::real_driver_entry(#name)) {              \
        return reinterpret_cast<decltype(&name)>(real_fn)(__VA_ARGS__);          \
    }

constexpr unsigned long long kKnownHandleTypes =
    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR | CU_MEM_HANDLE_TYPE_WIN32 |
    CU_MEM_HANDLE_TYPE_WIN32_KMT | CU_MEM_HANDLE_TYPE_FABRIC;

bool valid_device(int ordinal) {
    GlobalState& state = GlobalState::instance();
    state.initialize();
    return ordinal >= 0 && ordinal < state.get_device_count();
}

bool is_host_location(CUmemLocationType type) {
    return type == CU_MEM_LOCATION_TYPE_HOST || type == CU_MEM_LOCATION_TYPE_HOST_NUMA ||
           type == CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT;
}

// Frees the handle's pages and device accounting once nothing refers to it.
void release_if_unused_locked(model::SharedState& state, CUmemGenericAllocationHandle handle) {
    auto it = state.vmm_allocations.find(handle);
    if (it == state.vmm_allocations.end()) return;
    model::VmmAllocation& allocation = it->second;
    if (allocation.references != 0 || allocation.mappings != 0) return;
    if (allocation.device >= 0) {
        std::size_t size = 0;
        int device = 0;
        GlobalState::instance().release_allocation(allocation.home, size, device);
    }
    model::destroy_backing(allocation.home, allocation.size);
    state.vmm_allocations.erase(it);
}

model::VmmReservation* reservation_for_range_locked(
    model::SharedState& state, std::uintptr_t start, std::size_t size, std::uintptr_t& base) {
    auto it = state.vmm_reservations.upper_bound(start);
    if (it == state.vmm_reservations.begin()) return nullptr;
    --it;
    const std::uintptr_t offset = start - it->first;
    if (offset >= it->second.size || size > it->second.size - offset) return nullptr;
    base = it->first;
    return &it->second;
}

}  // namespace

extern "C" {

CUresult cuMemGetAllocationGranularity(
    size_t *granularity, const CUmemAllocationProp *prop, CUmemAllocationGranularity_flags option) {
    FORWARD_VMM(cuMemGetAllocationGranularity, granularity, prop, option);
    if (!granularity || !prop) return CUDA_ERROR_INVALID_VALUE;
    if (option != CU_MEM_ALLOC_GRANULARITY_MINIMUM && option != CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *granularity = model::kVmmGranularity;
    return CUDA_SUCCESS;
}

CUresult cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size, const CUmemAllocationProp *prop,
                     unsigned long long flags) {
    FORWARD_VMM(cuMemCreate, handle, size, prop, flags);
    if (!handle || !prop || flags != 0) return CUDA_ERROR_INVALID_VALUE;
    if (prop->type != CU_MEM_ALLOCATION_TYPE_PINNED) return CUDA_ERROR_INVALID_VALUE;
    if (size == 0 || size % model::kVmmGranularity != 0) return CUDA_ERROR_INVALID_VALUE;
    const unsigned long long handle_types = static_cast<unsigned int>(prop->requestedHandleTypes);
    if (handle_types & ~kKnownHandleTypes) return CUDA_ERROR_INVALID_VALUE;
    if (handle_types & (CU_MEM_HANDLE_TYPE_WIN32 | CU_MEM_HANDLE_TYPE_WIN32_KMT)) return CUDA_ERROR_INVALID_VALUE;

    int device = -1;
    if (prop->location.type == CU_MEM_LOCATION_TYPE_DEVICE) {
        if (!valid_device(prop->location.id)) return CUDA_ERROR_INVALID_DEVICE;
        device = prop->location.id;
    } else if (!is_host_location(prop->location.type)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // cuda.h: fabric handles need an IMEX channel; without one the driver
    // returns CUDA_ERROR_NOT_PERMITTED, and PyTorch then retries with a file
    // descriptor handle.
    if (handle_types & CU_MEM_HANDLE_TYPE_FABRIC) return CUDA_ERROR_NOT_PERMITTED;

    GlobalState& global = GlobalState::instance();
    global.initialize();
    if (device >= 0) {
        const fake_gpu::Device& target = global.get_device(device);
        if (target.used_memory + size > target.total_memory) return CUDA_ERROR_OUT_OF_MEMORY;
    }
    void* home = model::create_backing(size);
    if (!home) return CUDA_ERROR_OUT_OF_MEMORY;
    if (device >= 0 && !global.register_allocation(home, size, device)) {
        model::destroy_backing(home, size);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    model::SharedState& state = model::shared_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    const CUmemGenericAllocationHandle created = state.next_vmm_handle++;
    model::VmmAllocation& allocation = state.vmm_allocations[created];
    allocation.home = home;
    allocation.size = size;
    allocation.device = device;
    allocation.prop = *prop;
    allocation.references = 1;
    *handle = created;
    return CUDA_SUCCESS;
}

CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
    FORWARD_VMM(cuMemRelease, handle);
    model::SharedState& state = model::shared_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.vmm_allocations.find(handle);
    if (it == state.vmm_allocations.end() || it->second.references == 0) return CUDA_ERROR_INVALID_VALUE;
    --it->second.references;
    release_if_unused_locked(state, handle);
    return CUDA_SUCCESS;
}

CUresult cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment, CUdeviceptr addr,
                             unsigned long long flags) {
    FORWARD_VMM(cuMemAddressReserve, ptr, size, alignment, addr, flags);
    if (!ptr || flags != 0) return CUDA_ERROR_INVALID_VALUE;
    if (size == 0 || size % model::kVmmGranularity != 0) return CUDA_ERROR_INVALID_VALUE;
    if (alignment & (alignment - 1)) return CUDA_ERROR_INVALID_VALUE;
    const std::size_t effective = alignment > model::kVmmGranularity ? alignment : model::kVmmGranularity;
    void* base = model::reserve_address_range(size, effective, static_cast<std::uintptr_t>(addr));
    if (!base) return CUDA_ERROR_OUT_OF_MEMORY;

    model::SharedState& state = model::shared_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.vmm_reservations[reinterpret_cast<std::uintptr_t>(base)].size = size;
    *ptr = reinterpret_cast<CUdeviceptr>(base);
    return CUDA_SUCCESS;
}

CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size) {
    FORWARD_VMM(cuMemAddressFree, ptr, size);
    model::SharedState& state = model::shared_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.vmm_reservations.find(static_cast<std::uintptr_t>(ptr));
    if (it == state.vmm_reservations.end() || it->second.size != size || !it->second.mappings.empty()) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    model::release_address_range(reinterpret_cast<void*>(it->first), it->second.size);
    state.vmm_reservations.erase(it);
    return CUDA_SUCCESS;
}

CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle,
                  unsigned long long flags) {
    FORWARD_VMM(cuMemMap, ptr, size, offset, handle, flags);
    if (flags != 0) return CUDA_ERROR_INVALID_VALUE;
    model::SharedState& state = model::shared_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto allocation = state.vmm_allocations.find(handle);
    if (allocation == state.vmm_allocations.end()) return CUDA_ERROR_INVALID_VALUE;
    // Measured: a non-zero offset or a size other than the handle's is
    // CUDA_ERROR_NOT_SUPPORTED.
    if (offset != 0 || size != allocation->second.size) return CUDA_ERROR_NOT_SUPPORTED;
    const auto start = static_cast<std::uintptr_t>(ptr);
    if (start % model::kVmmGranularity != 0) return CUDA_ERROR_INVALID_VALUE;
    std::uintptr_t base = 0;
    // The real driver also accepted a mapping that ran past the end of its
    // reservation; here it would overwrite unrelated host memory, so the
    // whole range must lie inside one reservation.
    model::VmmReservation* reservation = reservation_for_range_locked(state, start, size, base);
    if (!reservation) return CUDA_ERROR_INVALID_VALUE;
    auto next = reservation->mappings.lower_bound(start);
    if (next != reservation->mappings.end() && next->first < start + size) return CUDA_ERROR_INVALID_VALUE;
    if (next != reservation->mappings.begin()) {
        auto previous = std::prev(next);
        if (previous->first + previous->second.size > start) return CUDA_ERROR_INVALID_VALUE;
    }
    if (!model::alias_backing(allocation->second.home, start, size)) return CUDA_ERROR_OUT_OF_MEMORY;
    model::VmmMapping& mapping = reservation->mappings[start];
    mapping.size = size;
    mapping.handle = handle;
    ++allocation->second.mappings;
    // Kernels executed on the CPU resolve device addresses through the
    // allocation table; a host-kind record exposes the mapping without
    // counting its bytes a second time.
    GlobalState::instance().register_host_allocation(
        reinterpret_cast<void*>(start), size, allocation->second.device);
    return CUDA_SUCCESS;
}

CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) {
    FORWARD_VMM(cuMemUnmap, ptr, size);
    if (size == 0) return CUDA_ERROR_INVALID_VALUE;
    model::SharedState& state = model::shared_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    const auto start = static_cast<std::uintptr_t>(ptr);
    std::uintptr_t base = 0;
    model::VmmReservation* reservation = reservation_for_range_locked(state, start, size, base);
    if (!reservation) return CUDA_ERROR_INVALID_VALUE;
    const std::uintptr_t end = start + size;
    // Measured: unmapping part of a mapping fails; an unmapped range succeeds.
    auto first = reservation->mappings.lower_bound(start);
    if (first != reservation->mappings.begin()) {
        auto previous = std::prev(first);
        if (previous->first + previous->second.size > start) return CUDA_ERROR_INVALID_VALUE;
    }
    auto last = first;
    for (; last != reservation->mappings.end() && last->first < end; ++last) {
        if (last->first + last->second.size > end) return CUDA_ERROR_INVALID_VALUE;
    }
    for (auto it = first; it != last;) {
        model::unmap_alias(it->first, it->second.size);
        std::size_t released = 0;
        int device = 0;
        GlobalState::instance().release_host_allocation(reinterpret_cast<void*>(it->first), released, device);
        const CUmemGenericAllocationHandle handle = it->second.handle;
        auto allocation = state.vmm_allocations.find(handle);
        if (allocation != state.vmm_allocations.end() && allocation->second.mappings > 0) {
            --allocation->second.mappings;
        }
        it = reservation->mappings.erase(it);
        release_if_unused_locked(state, handle);
    }
    return CUDA_SUCCESS;
}

CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc *desc, size_t count) {
    FORWARD_VMM(cuMemSetAccess, ptr, size, desc, count);
    if (!desc || count == 0 || size == 0) return CUDA_ERROR_INVALID_VALUE;
    for (size_t i = 0; i < count; ++i) {
        if (desc[i].location.type != CU_MEM_LOCATION_TYPE_DEVICE || !valid_device(desc[i].location.id)) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        const auto flags = static_cast<unsigned int>(desc[i].flags);
        if (flags != CU_MEM_ACCESS_FLAGS_PROT_NONE && flags != CU_MEM_ACCESS_FLAGS_PROT_READ &&
            flags != CU_MEM_ACCESS_FLAGS_PROT_READWRITE) {
            return CUDA_ERROR_INVALID_VALUE;
        }
    }
    model::SharedState& state = model::shared_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    const auto start = static_cast<std::uintptr_t>(ptr);
    std::uintptr_t base = 0;
    model::VmmReservation* reservation = reservation_for_range_locked(state, start, size, base);
    if (!reservation) return CUDA_ERROR_INVALID_VALUE;
    // Measured: the range must consist of whole, contiguous mappings.
    const std::uintptr_t end = start + size;
    std::uintptr_t cursor = start;
    auto it = reservation->mappings.find(start);
    std::vector<model::VmmMapping*> covered;
    while (cursor < end) {
        if (it == reservation->mappings.end() || it->first != cursor || it->first + it->second.size > end) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        covered.push_back(&it->second);
        cursor += it->second.size;
        ++it;
    }
    for (model::VmmMapping* mapping : covered) {
        for (size_t i = 0; i < count; ++i) {
            mapping->access[desc[i].location.id] = static_cast<unsigned int>(desc[i].flags);
        }
    }
    return CUDA_SUCCESS;
}

CUresult cuMemGetAccess(unsigned long long *flags, const CUmemLocation *location, CUdeviceptr ptr) {
    FORWARD_VMM(cuMemGetAccess, flags, location, ptr);
    if (!flags || !location) return CUDA_ERROR_INVALID_VALUE;
    *flags = 0;
    if (location->type != CU_MEM_LOCATION_TYPE_DEVICE || !valid_device(location->id)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    model::SharedState& state = model::shared_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    model::VmmAddress address;
    if (!model::find_vmm_address_locked(state, static_cast<std::uintptr_t>(ptr), address) || !address.mapping) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const auto access = address.mapping->access.find(location->id);
    *flags = access == address.mapping->access.end() ? CU_MEM_ACCESS_FLAGS_PROT_NONE : access->second;
    return CUDA_SUCCESS;
}

CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle *handle, void *addr) {
    FORWARD_VMM(cuMemRetainAllocationHandle, handle, addr);
    if (!handle) return CUDA_ERROR_INVALID_VALUE;
    model::SharedState& state = model::shared_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    model::VmmAddress address;
    if (!model::find_vmm_address_locked(state, reinterpret_cast<std::uintptr_t>(addr), address) ||
        !address.mapping || !address.allocation) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    ++state.vmm_allocations[address.mapping->handle].references;
    *handle = address.mapping->handle;
    return CUDA_SUCCESS;
}

CUresult cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp *prop, CUmemGenericAllocationHandle handle) {
    FORWARD_VMM(cuMemGetAllocationPropertiesFromHandle, prop, handle);
    if (!prop) return CUDA_ERROR_INVALID_VALUE;
    model::SharedState& state = model::shared_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    const auto it = state.vmm_allocations.find(handle);
    if (it == state.vmm_allocations.end()) return CUDA_ERROR_INVALID_VALUE;
    *prop = it->second.prop;
    return CUDA_SUCCESS;
}

CUresult cuMemExportToShareableHandle(void *shareableHandle, CUmemGenericAllocationHandle handle,
                                      CUmemAllocationHandleType handleType, unsigned long long flags) {
    FORWARD_VMM(cuMemExportToShareableHandle, shareableHandle, handle, handleType, flags);
    if (!shareableHandle || flags != 0) return CUDA_ERROR_INVALID_VALUE;
    model::SharedState& state = model::shared_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    const auto it = state.vmm_allocations.find(handle);
    if (it == state.vmm_allocations.end()) return CUDA_ERROR_INVALID_VALUE;
    const auto requested = static_cast<unsigned int>(it->second.prop.requestedHandleTypes);
    const auto type = static_cast<unsigned int>(handleType);
    if (type == CU_MEM_HANDLE_TYPE_NONE || (requested & type) != type) return CUDA_ERROR_INVALID_VALUE;
    // Sharing simulated memory with another process is not modeled.
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle *handle, void *osHandle,
                                        CUmemAllocationHandleType shHandleType) {
    FORWARD_VMM(cuMemImportFromShareableHandle, handle, osHandle, shHandleType);
    if (!handle) return CUDA_ERROR_INVALID_VALUE;
    return CUDA_ERROR_NOT_SUPPORTED;
}

}  // extern "C"
