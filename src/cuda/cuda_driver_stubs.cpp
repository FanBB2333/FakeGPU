#include "cuda_driver_defs.hpp"
#include "cuda_driver_passthrough.hpp"
#include "../core/backend_config.hpp"
#include "../core/global_state.hpp"
#include "../core/hybrid_memory_manager.hpp"
#include "../core/unsupported_api.hpp"
#include <cstdio>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>

using namespace fake_gpu;

// Track current context (simplified - just track device)
static int current_context_device = 0;

namespace {
bool real_driver_available() {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() == FakeGpuMode::Simulate) return false;
    return CudaDriverPassthrough::instance().initialize();
}

// Optional real Driver entry points are resolved lazily and cached in a
// function-local static.  Resolve only after real_driver_available() returned
// true: before the passthrough is initialized every lookup yields nullptr.
template <class Fn>
Fn real_entry(const char* name) {
    return reinterpret_cast<Fn>(CudaDriverPassthrough::instance().getRealFunction(name));
}

// Prefers the 64-bit name##_v2 entry point and falls back to the legacy name.
template <class Fn>
Fn real_entry_v2(const char* name) {
    const std::string v2_name = std::string(name) + "_v2";
    if (Fn fn = real_entry<Fn>(v2_name.c_str())) return fn;
    return real_entry<Fn>(name);
}

// Forwards the enclosing API to the real entry point of the same name when a
// real driver is loaded; otherwise execution continues with the simulation.
#define FORWARD_TO_REAL(name, ...)                                          \
    if (real_driver_available()) {                                          \
        static const auto real_fn = real_entry<decltype(&name)>(#name);     \
        if (real_fn) return real_fn(__VA_ARGS__);                           \
    }

// CUDA 12.8 resolves most Driver API entry points through cuGetProcAddress.
// Hybrid mode only needs to interpose the surfaces that virtualize device
// identity/capacity and allocations.  Returning wrappers for every symbol
// makes internal CUDA Runtime and cuBLAS state depend on our incomplete
// simulation-only context helpers (for example cuCtxGetApiVersion), which can
// make cublasCreate fail even though real kernels otherwise execute correctly.
bool hybrid_should_interpose_symbol(const char* symbol) {
    static const char* const symbols[] = {
        "cuInit",
        "cuMemAlloc",
        "cuMemAlloc_v2",
        "cuMemFree",
        "cuMemFree_v2",
        "cuMemGetInfo",
        "cuMemGetInfo_v2",
        "cuMemAllocManaged",
        "cuMemAllocAsync",
        "cuMemFreeAsync",
        "cuMemAllocFromPoolAsync",
        "cuMemAllocPitch",
        "cuGetProcAddress",
        "cuGetProcAddress_v2",
    };
    for (const char* candidate : symbols) {
        if (std::strcmp(symbol, candidate) == 0) return true;
    }
    return false;
}

void* real_driver_symbol(const char* symbol, int cuda_version, unsigned long long flags) {
    void* real_fn = nullptr;
    CUresult result = CudaDriverPassthrough::instance().cuGetProcAddress(
        symbol, &real_fn, cuda_version, flags);
    if (result == CUDA_SUCCESS && real_fn) return real_fn;
    return CudaDriverPassthrough::instance().getRealFunction(symbol);
}

int map_virtual_device_to_real(int virtual_device) {
    const auto& info = RealCudaLoader::instance().get_real_gpu_info();
    if (!info.valid || info.device_count <= 0) return virtual_device;
    int mapped = virtual_device % info.device_count;
    if (mapped < 0) mapped += info.device_count;
    return mapped;
}

// Device ordinal to pass to the real driver: virtual devices map onto the
// physical ones in hybrid mode; passthrough already uses physical ordinals.
CUdevice real_device(CUdevice dev) {
    return BackendConfig::instance().mode() == FakeGpuMode::Hybrid ? map_virtual_device_to_real(dev) : dev;
}

// Allocates for a hybrid-mode virtual device according to the memory manager's
// decision.  alloc_real(CUdeviceptr*) performs the real device allocation.
template <class RealAlloc>
CUresult hybrid_allocate(
    CUdeviceptr* dptr,
    size_t bytesize,
    int virtual_device,
    HybridMemoryManager::AllocationDecision decision,
    RealAlloc&& alloc_real) {
    constexpr unsigned int kManagedAttachGlobal = 1; // CU_MEM_ATTACH_GLOBAL
    constexpr unsigned int kHostAllocDevicemap = 2;  // CU_MEMHOSTALLOC_DEVICEMAP

    CUresult result = CUDA_SUCCESS;
    HybridAllocation::Type alloc_type = HybridAllocation::Type::RealDevice;
    void* backing_ptr = nullptr;
    CUdeviceptr out_ptr = 0;

    switch (decision) {
        case HybridMemoryManager::AllocationDecision::UseReal: {
            result = alloc_real(&out_ptr);
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

// Host memory behind a hybrid allocation that lives outside device memory
// (mapped host or CPU spill); nullptr for real device memory and untracked pointers.
void* hybrid_host_view(CUdeviceptr ptr) {
    HybridAllocation alloc;
    if (!HybridMemoryManager::instance().get_allocation_info((void*)ptr, alloc)) return nullptr;
    if (alloc.type == HybridAllocation::Type::MappedHost) return alloc.backing_ptr;
    if (alloc.type == HybridAllocation::Type::SpilledCpu) return (void*)ptr;
    return nullptr;
}

// Fills n elements of host memory and records an n * sizeof(T) byte memset of dst.
template <class T>
CUresult fill_and_record(void* host, CUdeviceptr dst, T value, size_t n) {
    if constexpr (sizeof(T) == 1) {
        memset(host, value, n);
    } else {
        T* p = static_cast<T*>(host);
        for (size_t i = 0; i < n; ++i) p[i] = value;
    }
    GlobalState::instance().record_memset((void*)dst, n * sizeof(T));
    return CUDA_SUCCESS;
}

CUresult record_memset_if_ok(CUresult result, CUdeviceptr dst, size_t bytes) {
    if (result == CUDA_SUCCESS) GlobalState::instance().record_memset((void*)dst, bytes);
    return result;
}

// Stream-ordered real allocation; falls back to a synchronous cuMemAlloc when
// the driver lacks cuMemAllocAsync.  Call only once real_driver_available().
CUresult real_mem_alloc_async(CUdeviceptr* dptr, size_t bytesize, CUstream stream) {
    static const auto real_fn = real_entry<CUresult (*)(CUdeviceptr*, size_t, CUstream)>("cuMemAllocAsync");
    if (real_fn) return real_fn(dptr, bytesize, stream);
    return CudaDriverPassthrough::instance().cuMemAlloc(dptr, bytesize);
}

std::mutex g_ctx_mutex;
std::unordered_map<CUcontext, int> g_ctx_to_virtual_device;
std::atomic<std::uintptr_t> g_next_stream_handle{1};
std::atomic<std::uintptr_t> g_next_event_handle{1};
std::atomic<std::uint64_t> g_next_timeline_tick{1};
std::mutex g_stream_mutex;
std::uint64_t g_default_stream_submitted_tick = 0;
std::uint64_t g_default_stream_completed_tick = 0;
bool g_default_stream_auto_complete_next_event_record = false;
std::mutex g_event_mutex;

struct StreamState {
    unsigned int flags = 0;
    std::uint64_t submitted_tick = 0;
    std::uint64_t completed_tick = 0;
    bool auto_complete_next_event_record = false;
};

std::unordered_map<CUstream, StreamState> g_stream_states;

struct EventState {
    bool recorded = false;
    bool completed = false;
    CUstream last_stream = nullptr;
    std::uint64_t timeline_tick = 0;
};

std::unordered_map<CUevent, EventState> g_event_states;

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

bool lookup_stream_flags(CUstream stream, unsigned int& flags) {
    if (stream == nullptr) {
        flags = CU_STREAM_DEFAULT;
        return true;
    }
    std::lock_guard<std::mutex> lock(g_stream_mutex);
    auto it = g_stream_states.find(stream);
    if (it == g_stream_states.end()) return false;
    flags = it->second.flags;
    return true;
}

bool lookup_stream_state_locked(CUstream stream, StreamState& state) {
    if (stream == nullptr) {
        state.flags = CU_STREAM_DEFAULT;
        state.submitted_tick = g_default_stream_submitted_tick;
        state.completed_tick = g_default_stream_completed_tick;
        state.auto_complete_next_event_record = g_default_stream_auto_complete_next_event_record;
        return true;
    }
    const auto it = g_stream_states.find(stream);
    if (it == g_stream_states.end()) {
        return false;
    }
    state = it->second;
    return true;
}

bool erase_stream_handle(CUstream stream) {
    if (stream == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_stream_mutex);
    return g_stream_states.erase(stream) > 0;
}

void register_stream_handle(CUstream stream, unsigned int flags) {
    std::lock_guard<std::mutex> lock(g_stream_mutex);
    StreamState& state = g_stream_states[stream];
    state = StreamState{};
    state.flags = flags;
}

bool lookup_event_state(CUevent event, EventState& state) {
    if (event == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_event_mutex);
    auto it = g_event_states.find(event);
    if (it == g_event_states.end()) return false;
    state = it->second;
    return true;
}

bool erase_event_handle(CUevent event) {
    if (event == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_event_mutex);
    return g_event_states.erase(event) > 0;
}

void register_event_handle(CUevent event) {
    std::lock_guard<std::mutex> lock(g_event_mutex);
    g_event_states[event] = EventState{};
}

bool stream_is_non_blocking(unsigned int flags) {
    return (flags & CU_STREAM_NON_BLOCKING) != 0;
}

std::uint64_t max_submitted_tick_for_blocking_streams_locked() {
    std::uint64_t max_tick = g_default_stream_submitted_tick;
    for (const auto& entry : g_stream_states) {
        if (!stream_is_non_blocking(entry.second.flags)) {
            max_tick = std::max(max_tick, entry.second.submitted_tick);
        }
    }
    return max_tick;
}

std::uint64_t reserve_next_tick(std::uint64_t floor_tick) {
    const std::uint64_t next_tick = g_next_timeline_tick.fetch_add(1);
    return std::max(next_tick, floor_tick + 1);
}

bool submit_stream_operation_locked(CUstream stream, std::uint64_t& tick) {
    StreamState state;
    if (!lookup_stream_state_locked(stream, state)) {
        return false;
    }

    std::uint64_t floor_tick = state.submitted_tick;
    if (stream == nullptr) {
        floor_tick = std::max(floor_tick, max_submitted_tick_for_blocking_streams_locked());
        tick = reserve_next_tick(floor_tick);
        g_default_stream_submitted_tick = tick;
        return true;
    }

    if (!stream_is_non_blocking(state.flags)) {
        floor_tick = std::max(floor_tick, g_default_stream_submitted_tick);
    }
    tick = reserve_next_tick(floor_tick);
    StreamState& writable = g_stream_states[stream];
    writable.submitted_tick = tick;
    return true;
}

bool mark_stream_synchronized_locked(CUstream stream) {
    if (stream == nullptr) {
        g_default_stream_completed_tick = g_default_stream_submitted_tick;
        return true;
    }
    auto it = g_stream_states.find(stream);
    if (it == g_stream_states.end()) {
        return false;
    }
    it->second.completed_tick = it->second.submitted_tick;
    return true;
}

void mark_stream_completed_through_tick_locked(CUstream stream, std::uint64_t tick) {
    if (stream == nullptr) {
        g_default_stream_submitted_tick = std::max(g_default_stream_submitted_tick, tick);
        g_default_stream_completed_tick = std::max(g_default_stream_completed_tick, tick);
        return;
    }
    StreamState& writable = g_stream_states[stream];
    writable.submitted_tick = std::max(writable.submitted_tick, tick);
    writable.completed_tick = std::max(writable.completed_tick, tick);
}

void mark_stream_auto_complete_next_event_record_locked(CUstream stream) {
    if (stream == nullptr) {
        g_default_stream_auto_complete_next_event_record = true;
        return;
    }
    g_stream_states[stream].auto_complete_next_event_record = true;
}

bool consume_stream_auto_complete_next_event_record_locked(CUstream stream) {
    if (stream == nullptr) {
        const bool should_auto_complete = g_default_stream_auto_complete_next_event_record;
        g_default_stream_auto_complete_next_event_record = false;
        return should_auto_complete;
    }
    auto it = g_stream_states.find(stream);
    if (it == g_stream_states.end()) {
        return false;
    }
    const bool should_auto_complete = it->second.auto_complete_next_event_record;
    it->second.auto_complete_next_event_record = false;
    return should_auto_complete;
}

std::uint64_t stream_completed_tick(CUstream stream) {
    std::lock_guard<std::mutex> lock(g_stream_mutex);
    if (stream == nullptr) {
        return g_default_stream_completed_tick;
    }
    const auto it = g_stream_states.find(stream);
    if (it == g_stream_states.end()) {
        return 0;
    }
    return it->second.completed_tick;
}

bool mark_event_recorded(CUevent event, CUstream stream) {
    EventState existing;
    if (!lookup_event_state(event, existing)) {
        return false;
    }
    std::uint64_t tick = 0;
    bool auto_complete = false;
    {
        std::lock_guard<std::mutex> stream_lock(g_stream_mutex);
        if (!submit_stream_operation_locked(stream, tick)) {
            return false;
        }
        auto_complete = consume_stream_auto_complete_next_event_record_locked(stream);
        if (auto_complete) {
            mark_stream_completed_through_tick_locked(stream, tick);
        }
    }

    std::lock_guard<std::mutex> lock(g_event_mutex);
    auto it = g_event_states.find(event);
    if (it == g_event_states.end()) return false;
    it->second.recorded = true;
    it->second.completed = auto_complete;
    it->second.last_stream = stream;
    it->second.timeline_tick = tick;
    return true;
}

} // namespace

namespace fake_gpu::cuda {

bool is_stream_handle_live(CUstream stream) {
    if (real_driver_available()) {
        return true;
    }
    unsigned int flags = 0;
    return lookup_stream_flags(stream, flags);
}

CUresult submit_stream_operation(CUstream stream) {
    if (real_driver_available()) {
        return CUDA_SUCCESS;
    }
    std::lock_guard<std::mutex> lock(g_stream_mutex);
    std::uint64_t tick = 0;
    return submit_stream_operation_locked(stream, tick) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

CUresult mark_stream_synchronized(CUstream stream) {
    if (real_driver_available()) {
        return CUDA_SUCCESS;
    }
    std::lock_guard<std::mutex> lock(g_stream_mutex);
    return mark_stream_synchronized_locked(stream) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

CUresult mark_stream_host_function_complete(CUstream stream) {
    if (real_driver_available()) {
        return CUDA_SUCCESS;
    }

    std::lock_guard<std::mutex> lock(g_stream_mutex);
    std::uint64_t tick = 0;
    if (!submit_stream_operation_locked(stream, tick)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!mark_stream_synchronized_locked(stream)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    mark_stream_auto_complete_next_event_record_locked(stream);
    return CUDA_SUCCESS;
}

}  // namespace fake_gpu::cuda

extern "C" {

CUresult fakegpuCudaMarkStreamHostFunctionComplete(CUstream stream) {
    return fake_gpu::cuda::mark_stream_host_function_complete(stream);
}

CUresult cuInit(unsigned int Flags) {
    FGPU_LOG("[FakeCUDA-Driver] cuInit called with flags=%u\n", Flags);

    const BackendConfig& config = BackendConfig::instance();

    // Always initialize the fake state: hybrid/simulate virtualize device info using GlobalState.
    GlobalState::instance().initialize();

    if (real_driver_available()) {
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

    // The real CUDA Runtime validates a large set of low-level attributes for
    // internal consistency.  Returning simulated defaults for newer attributes
    // can make CUDA 12.8+ reject an otherwise valid device with error 103
    // ("integrity checks failed").  Hybrid mode executes real kernels, so use
    // the backing device's execution attributes and virtualize only the
    // user-facing identity/memory surfaces.
    if (real_driver_available()) {
        return CudaDriverPassthrough::instance().cuDeviceGetAttribute(pi, attrib, real_device(dev));
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
            break;
        case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR:
            *pi = device.profile.compute_minor;
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

CUresult cuDeviceGetUuid(CUuuid *uuid, CUdevice dev) {
    if (!uuid) return CUDA_ERROR_INVALID_VALUE;

    if (real_driver_available()) {
        return CudaDriverPassthrough::instance().cuDeviceGetUuid(uuid, real_device(dev));
    }

    GlobalState::instance().initialize();

    int count = GlobalState::instance().get_device_count();
    if (dev < 0 || dev >= count) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    memset(uuid->bytes, 0, sizeof(uuid->bytes));
    memcpy(uuid->bytes, &dev, std::min(sizeof(dev), sizeof(uuid->bytes)));
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
    if (real_driver_available()) {
        return CudaDriverPassthrough::instance().cuCtxDestroy(ctx);
    }
    FGPU_LOG("[FakeCUDA-Driver] cuCtxDestroy(%p)\n", ctx);
    return CUDA_SUCCESS;
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
    const BackendConfig& config = BackendConfig::instance();
    if (real_driver_available()) {
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
    if (real_driver_available()) {
        return CudaDriverPassthrough::instance().cuCtxGetCurrent(pctx);
    }
    *pctx = (CUcontext)(uintptr_t)(current_context_device + 1);
    return CUDA_SUCCESS;
}

CUresult cuCtxSynchronize(void) {
    if (real_driver_available()) {
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
        return hybrid_allocate(
            dptr,
            bytesize,
            virtual_device,
            HybridMemoryManager::instance().decide_allocation(virtual_device, bytesize),
            [&](CUdeviceptr* out) { return CudaDriverPassthrough::instance().cuMemAlloc(out, bytesize); });
    }

    // Simulate mode: allocate in host memory.
    GlobalState::instance().initialize();
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
        CUresult result = CUDA_SUCCESS;
        if (void* host = hybrid_host_view(srcDevice)) {
            memcpy(dstHost, host, ByteCount);
        } else {
            result = CudaDriverPassthrough::instance().cuMemcpyDtoH(dstHost, srcDevice, ByteCount);
        }
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
        CUresult result = CUDA_SUCCESS;
        if (void* host = hybrid_host_view(dstDevice)) {
            memcpy(host, srcHost, ByteCount);
        } else {
            result = CudaDriverPassthrough::instance().cuMemcpyHtoD(dstDevice, srcHost, ByteCount);
        }
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
        void* dst_host = hybrid_host_view(dstDevice);
        void* src_host = hybrid_host_view(srcDevice);

        CUresult result = CUDA_SUCCESS;
        if (dst_host && src_host) {
            memcpy(dst_host, src_host, ByteCount);
        } else if (dst_host) {
            result = CudaDriverPassthrough::instance().cuMemcpyDtoH(dst_host, srcDevice, ByteCount);
        } else if (src_host) {
            result = CudaDriverPassthrough::instance().cuMemcpyHtoD(dstDevice, src_host, ByteCount);
        } else {
            result = CudaDriverPassthrough::instance().cuMemcpyDtoD(dstDevice, srcDevice, ByteCount);
        }
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
    if (real_driver_available()) {
        static const auto real_fn =
            real_entry<decltype(&cuDevicePrimaryCtxGetState)>("cuDevicePrimaryCtxGetState");
        if (!real_fn) return CUDA_ERROR_NOT_INITIALIZED;
        return real_fn(real_device(dev), flags, active);
    }

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
    if (real_driver_available()) {
        const CUdevice real_dev = real_device(dev);
        static const auto real_fn =
            real_entry_v2<decltype(&cuDevicePrimaryCtxSetFlags)>("cuDevicePrimaryCtxSetFlags");
        if (real_fn) return real_fn(real_dev, flags);
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
    if (real_driver_available()) {
        return CudaDriverPassthrough::instance().cuDevicePrimaryCtxReset(real_device(dev));
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
    const BackendConfig& config = BackendConfig::instance();
    if (real_driver_available()) {
        static const auto real_fn = real_entry_v2<decltype(&cuCtxPushCurrent)>("cuCtxPushCurrent");
        if (!real_fn) return CUDA_ERROR_NOT_INITIALIZED;
        CUresult result = real_fn(ctx);
        if (result == CUDA_SUCCESS) {
            const int mapped = lookup_context_mapping(ctx);
            if (mapped >= 0) current_context_device = mapped;
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
    FGPU_LOG("[FakeCUDA-Driver] cuCtxPushCurrent(%p) -> device %d\n", ctx, current_context_device);
    return CUDA_SUCCESS;
}

CUresult cuCtxPopCurrent(CUcontext *pctx) {
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;

    if (real_driver_available()) {
        static const auto real_fn = real_entry_v2<decltype(&cuCtxPopCurrent)>("cuCtxPopCurrent");
        if (!real_fn) return CUDA_ERROR_NOT_INITIALIZED;
        return real_fn(pctx);
    }

    *pctx = (CUcontext)(uintptr_t)(current_context_device + 1);
    FGPU_LOG("[FakeCUDA-Driver] cuCtxPopCurrent returning context %p\n", *pctx);
    return CUDA_SUCCESS;
}

// Error handling
CUresult cuGetErrorString(CUresult error, const char **pStr) {
    if (pStr) {
        switch (error) {
            case CUDA_SUCCESS:
                *pStr = "CUDA_SUCCESS";
                break;
            case CUDA_ERROR_INVALID_VALUE:
                *pStr = "CUDA_ERROR_INVALID_VALUE";
                break;
            case CUDA_ERROR_OUT_OF_MEMORY:
                *pStr = "CUDA_ERROR_OUT_OF_MEMORY";
                break;
            case CUDA_ERROR_NOT_INITIALIZED:
                *pStr = "CUDA_ERROR_NOT_INITIALIZED";
                break;
            case CUDA_ERROR_DEINITIALIZED:
                *pStr = "CUDA_ERROR_DEINITIALIZED";
                break;
            case CUDA_ERROR_NOT_READY:
                *pStr = "CUDA_ERROR_NOT_READY";
                break;
            case CUDA_ERROR_NOT_SUPPORTED:
                *pStr = "CUDA_ERROR_NOT_SUPPORTED";
                break;
            case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:
                *pStr = "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED";
                break;
            default:
                *pStr = "CUDA_ERROR_UNKNOWN";
                break;
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

    if (real_driver_available()) {
        return CudaDriverPassthrough::instance().cuStreamCreate(phStream, Flags);
    }

    // Return a stable but distinct opaque stream handle in simulate mode.
    *phStream = reinterpret_cast<CUstream>(g_next_stream_handle.fetch_add(1));
    register_stream_handle(*phStream, Flags);
    FGPU_LOG("[FakeCUDA-Driver] cuStreamCreate returning fake stream\n");
    return CUDA_SUCCESS;
}

CUresult cuStreamDestroy(CUstream hStream) {
    if (real_driver_available()) {
        return CudaDriverPassthrough::instance().cuStreamDestroy(hStream);
    }
    if (!erase_stream_handle(hStream)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    FGPU_LOG("[FakeCUDA-Driver] cuStreamDestroy\n");
    return CUDA_SUCCESS;
}

CUresult cuStreamSynchronize(CUstream hStream) {
    if (real_driver_available()) {
        return CudaDriverPassthrough::instance().cuStreamSynchronize(hStream);
    }
    if (fake_gpu::cuda::mark_stream_synchronized(hStream) != CUDA_SUCCESS) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    FGPU_LOG("[FakeCUDA-Driver] cuStreamSynchronize (no-op)\n");
    return CUDA_SUCCESS;
}

CUresult cuStreamQuery(CUstream hStream) {
    if (real_driver_available()) {
        return CudaDriverPassthrough::instance().cuStreamQuery(hStream);
    }
    std::lock_guard<std::mutex> lock(g_stream_mutex);
    StreamState state;
    if (!lookup_stream_state_locked(hStream, state)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return state.completed_tick >= state.submitted_tick ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
}

CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) {
    if (real_driver_available()) {
        return CudaDriverPassthrough::instance().cuStreamWaitEvent(hStream, hEvent, Flags);
    }
    std::lock_guard<std::mutex> stream_lock(g_stream_mutex);
    EventState event_state;
    if (!lookup_event_state(hEvent, event_state)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!event_state.recorded) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::uint64_t tick = 0;
    if (!submit_stream_operation_locked(hStream, tick)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (hStream == nullptr) {
        g_default_stream_submitted_tick = std::max(g_default_stream_submitted_tick, event_state.timeline_tick);
    } else {
        StreamState& writable = g_stream_states[hStream];
        writable.submitted_tick = std::max(writable.submitted_tick, event_state.timeline_tick);
    }
    (void)Flags;
    return CUDA_SUCCESS;
}

CUresult cuStreamGetPriority(CUstream hStream, int *priority) {
    FORWARD_TO_REAL(cuStreamGetPriority, hStream, priority);
    unsigned int flags = 0;
    if (!lookup_stream_flags(hStream, flags)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (priority) *priority = 0;
    return CUDA_SUCCESS;
}

CUresult cuStreamGetFlags(CUstream hStream, unsigned int *flags) {
    if (real_driver_available()) {
        return CudaDriverPassthrough::instance().cuStreamGetFlags(hStream, flags);
    }
    unsigned int stream_flags = 0;
    if (!lookup_stream_flags(hStream, stream_flags)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!flags) return CUDA_ERROR_INVALID_VALUE;
    *flags = stream_flags;
    return CUDA_SUCCESS;
}

CUresult cuStreamGetCtx(CUstream hStream, CUcontext *pctx) {
    FORWARD_TO_REAL(cuStreamGetCtx, hStream, pctx);
    unsigned int flags = 0;
    if (!lookup_stream_flags(hStream, flags)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (pctx) *pctx = (CUcontext)(uintptr_t)(current_context_device + 1);
    return CUDA_SUCCESS;
}

// Event management
CUresult cuEventCreate(CUevent *phEvent, unsigned int Flags) {
    if (!phEvent) return CUDA_ERROR_INVALID_VALUE;

    if (real_driver_available()) {
        return CudaDriverPassthrough::instance().cuEventCreate(phEvent, Flags);
    }

    *phEvent = reinterpret_cast<CUevent>(g_next_event_handle.fetch_add(1));
    register_event_handle(*phEvent);
    FGPU_LOG("[FakeCUDA-Driver] cuEventCreate returning fake event\n");
    return CUDA_SUCCESS;
}

CUresult cuEventDestroy(CUevent hEvent) {
    if (real_driver_available()) {
        return CudaDriverPassthrough::instance().cuEventDestroy(hEvent);
    }
    if (!erase_event_handle(hEvent)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    FGPU_LOG("[FakeCUDA-Driver] cuEventDestroy\n");
    return CUDA_SUCCESS;
}

CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
    if (real_driver_available()) {
        return CudaDriverPassthrough::instance().cuEventRecord(hEvent, hStream);
    }
    if (!fake_gpu::cuda::is_stream_handle_live(hStream)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!mark_event_recorded(hEvent, hStream)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return CUDA_SUCCESS;
}

CUresult cuEventSynchronize(CUevent hEvent) {
    if (real_driver_available()) {
        return CudaDriverPassthrough::instance().cuEventSynchronize(hEvent);
    }
    EventState state;
    if (!lookup_event_state(hEvent, state)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (state.recorded && fake_gpu::cuda::mark_stream_synchronized(state.last_stream) != CUDA_SUCCESS) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    {
        std::lock_guard<std::mutex> lock(g_event_mutex);
        auto it = g_event_states.find(hEvent);
        if (it != g_event_states.end()) {
            it->second.completed = true;
        }
    }
    return CUDA_SUCCESS;
}

CUresult cuEventQuery(CUevent hEvent) {
    if (real_driver_available()) {
        return CudaDriverPassthrough::instance().cuEventQuery(hEvent);
    }
    EventState state;
    if (!lookup_event_state(hEvent, state)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!state.recorded) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const bool ready = state.completed || stream_completed_tick(state.last_stream) >= state.timeline_tick;
    return ready ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
}

CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd) {
    if (real_driver_available()) {
        return CudaDriverPassthrough::instance().cuEventElapsedTime(pMilliseconds, hStart, hEnd);
    }
    EventState start_state;
    EventState end_state;
    if (!pMilliseconds ||
        !lookup_event_state(hStart, start_state) ||
        !lookup_event_state(hEnd, end_state)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!start_state.recorded || !end_state.recorded) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const std::uint64_t delta_ticks =
        end_state.timeline_tick > start_state.timeline_tick
            ? (end_state.timeline_tick - start_state.timeline_tick)
            : 0;
    if (pMilliseconds) *pMilliseconds = static_cast<float>(delta_ticks) * 0.1f;
    return CUDA_SUCCESS;
}

// Context info
CUresult cuCtxGetDevice(CUdevice *device) {
    if (!device) return CUDA_ERROR_INVALID_VALUE;

    const BackendConfig& config = BackendConfig::instance();
    if (real_driver_available()) {
        CUresult result = CudaDriverPassthrough::instance().cuCtxGetDevice(device);
        if (result == CUDA_SUCCESS && config.mode() == FakeGpuMode::Hybrid) {
            *device = current_context_device;
        }
        return result;
    }

    *device = current_context_device;
    return CUDA_SUCCESS;
}

CUresult cuCtxGetFlags(unsigned int *flags) {
    FORWARD_TO_REAL(cuCtxGetFlags, flags);
    if (flags) *flags = 0;
    return CUDA_SUCCESS;
}

CUresult cuCtxGetLimit(size_t *pvalue, CUlimit limit) {
    if (!pvalue) return CUDA_ERROR_INVALID_VALUE;
    FORWARD_TO_REAL(cuCtxGetLimit, pvalue, limit);
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
    FORWARD_TO_REAL(cuCtxSetLimit, limit, value);
    FGPU_LOG("[FakeCUDA-Driver] cuCtxSetLimit (no-op)\n");
    return CUDA_SUCCESS;
}

CUresult cuCtxGetStreamPriorityRange(int *leastPriority, int *greatestPriority) {
    FORWARD_TO_REAL(cuCtxGetStreamPriorityRange, leastPriority, greatestPriority);
    if (leastPriority) *leastPriority = 0;
    if (greatestPriority) *greatestPriority = -1;
    return CUDA_SUCCESS;
}

CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *version) {
    FORWARD_TO_REAL(cuCtxGetApiVersion, ctx, version);
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
    if (real_driver_available()) {
        const CUdevice real_dev = real_device(dev);
        static const auto real_fn = real_entry<decltype(&cuDeviceGetUuid_v2)>("cuDeviceGetUuid_v2");
        if (real_fn) return real_fn(uuid, real_dev);
        return CudaDriverPassthrough::instance().cuDeviceGetUuid(uuid, real_dev);
    }
    return cuDeviceGetUuid(uuid, dev);
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
    FORWARD_TO_REAL(cuDeviceCanAccessPeer, canAccessPeer, real_device(dev), real_device(peerDev));
    if (!canAccessPeer) return CUDA_ERROR_INVALID_VALUE;
    GlobalState& state = GlobalState::instance();
    state.initialize();
    if (dev < 0 || dev >= state.get_device_count() || peerDev < 0 || peerDev >= state.get_device_count()) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    // Follows the modeled NVLink/PCIe topology; without one every pair has access.
    *canAccessPeer = state.snapshot_device_topology().can_access_peer(dev, peerDev) ? 1 : 0;
    return CUDA_SUCCESS;
}

CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags) {
    FORWARD_TO_REAL(cuCtxEnablePeerAccess, peerContext, Flags);
    if (!peerContext) return CUDA_ERROR_INVALID_CONTEXT;
    int can_access = 0;
    const CUresult result = cuDeviceCanAccessPeer(
        &can_access,
        current_context_device,
        static_cast<CUdevice>(reinterpret_cast<uintptr_t>(peerContext) - 1));
    if (result != CUDA_SUCCESS) return result;
    return can_access ? CUDA_SUCCESS : CUDA_ERROR_PEER_ACCESS_UNSUPPORTED;
}

CUresult cuCtxDisablePeerAccess(CUcontext peerContext) {
    FORWARD_TO_REAL(cuCtxDisablePeerAccess, peerContext);
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev) {
    if (!pciBusId || len <= 0) return CUDA_ERROR_INVALID_VALUE;

    if (real_driver_available()) {
        static const auto real_fn = real_entry<decltype(&cuDeviceGetPCIBusId)>("cuDeviceGetPCIBusId");
        if (!real_fn) return CUDA_ERROR_NOT_INITIALIZED;
        return real_fn(pciBusId, len, real_device(dev));
    }

    snprintf(pciBusId, len, "0000:%02x:00.0", dev);
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetByPCIBusId(CUdevice *dev, const char *pciBusId) {
    if (!dev || !pciBusId) return CUDA_ERROR_INVALID_VALUE;

    if (real_driver_available()) {
        static const auto real_fn = real_entry<decltype(&cuDeviceGetByPCIBusId)>("cuDeviceGetByPCIBusId");
        if (!real_fn) return CUDA_ERROR_NOT_INITIALIZED;
        return real_fn(dev, pciBusId);
    }

    *dev = 0;  // Default to device 0
    return CUDA_SUCCESS;
}

CUresult cuModuleLoad(CUmodule *module, const char *fname) {
    if (!module) return CUDA_ERROR_INVALID_VALUE;

    if (real_driver_available()) {
        return CudaDriverPassthrough::instance().cuModuleLoad(module, fname);
    }

    *module = (CUmodule)(uintptr_t)1;
    FGPU_LOG("[FakeCUDA-Driver] cuModuleLoad (fake)\n");
    return CUDA_SUCCESS;
}

CUresult cuModuleLoadData(CUmodule *module, const void *image) {
    if (!module) return CUDA_ERROR_INVALID_VALUE;

    if (real_driver_available()) {
        return CudaDriverPassthrough::instance().cuModuleLoadData(module, image);
    }

    *module = (CUmodule)(uintptr_t)1;
    return CUDA_SUCCESS;
}

CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, void *options, void **optionValues) {
    if (!module) return CUDA_ERROR_INVALID_VALUE;

    if (real_driver_available()) {
        static const auto real_fn = real_entry<decltype(&cuModuleLoadDataEx)>("cuModuleLoadDataEx");
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

    if (real_driver_available()) {
        static const auto real_fn = real_entry<decltype(&cuModuleLoadFatBinary)>("cuModuleLoadFatBinary");
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
    if (real_driver_available()) {
        return CudaDriverPassthrough::instance().cuModuleUnload(hmod);
    }
    return CUDA_SUCCESS;
}

CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
    if (!hfunc) return CUDA_ERROR_INVALID_VALUE;

    if (real_driver_available()) {
        return CudaDriverPassthrough::instance().cuModuleGetFunction(hfunc, hmod, name);
    }

    *hfunc = (CUfunction)(uintptr_t)1;
    return CUDA_SUCCESS;
}

CUresult cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name) {
    if (real_driver_available()) {
        static const auto real_fn = real_entry_v2<decltype(&cuModuleGetGlobal)>("cuModuleGetGlobal");
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
    if (real_driver_available()) {
        return CudaDriverPassthrough::instance().cuLaunchKernel(
            f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
    }
    FGPU_LOG("[FakeCUDA-Driver] cuLaunchKernel (no-op)\n");
    GlobalState& state = GlobalState::instance();
    state.initialize();
    state.record_kernel_launch("cuLaunchKernel");
    if (record_unsupported_api(state, "cuLaunchKernel")) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return CUDA_SUCCESS;
}

CUresult cuFuncGetAttribute(int *pi, int attrib, CUfunction hfunc) {
    FORWARD_TO_REAL(cuFuncGetAttribute, pi, attrib, hfunc);
    if (pi) *pi = 0;
    return CUDA_SUCCESS;
}

CUresult cuFuncSetAttribute(CUfunction hfunc, int attrib, int value) {
    FORWARD_TO_REAL(cuFuncSetAttribute, hfunc, attrib, value);
    return CUDA_SUCCESS;
}

CUresult cuFuncSetCacheConfig(CUfunction hfunc, int config) {
    FORWARD_TO_REAL(cuFuncSetCacheConfig, hfunc, config);
    return CUDA_SUCCESS;
}

CUresult cuCtxGetCacheConfig(int *pconfig) {
    FORWARD_TO_REAL(cuCtxGetCacheConfig, pconfig);
    if (pconfig) *pconfig = 0;
    return CUDA_SUCCESS;
}

CUresult cuCtxSetCacheConfig(int config) {
    FORWARD_TO_REAL(cuCtxSetCacheConfig, config);
    return CUDA_SUCCESS;
}

CUresult cuCtxGetSharedMemConfig(int *pConfig) {
    FORWARD_TO_REAL(cuCtxGetSharedMemConfig, pConfig);
    if (pConfig) *pConfig = 0;
    return CUDA_SUCCESS;
}

CUresult cuCtxSetSharedMemConfig(int config) {
    FORWARD_TO_REAL(cuCtxSetSharedMemConfig, config);
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

    GlobalState::instance().initialize();
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

    if (real_driver_available()) {
        return CudaDriverPassthrough::instance().cuMemAllocHost(pp, bytesize);
    }

    GlobalState::instance().initialize();
    *pp = malloc(bytesize);
    if (!*pp) return CUDA_ERROR_OUT_OF_MEMORY;
    GlobalState::instance().register_host_allocation(*pp, bytesize, current_context_device);
    return CUDA_SUCCESS;
}

CUresult cuMemFreeHost(void *p) {
    if (real_driver_available()) {
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

    if (real_driver_available()) {
        return CudaDriverPassthrough::instance().cuMemHostAlloc(pp, bytesize, Flags);
    }

    return cuMemAllocHost(pp, bytesize);
}

CUresult cuMemHostRegister(void *p, size_t bytesize, unsigned int Flags) {
    FORWARD_TO_REAL(cuMemHostRegister, p, bytesize, Flags);
    return CUDA_SUCCESS;
}

CUresult cuMemHostUnregister(void *p) {
    FORWARD_TO_REAL(cuMemHostUnregister, p);
    return CUDA_SUCCESS;
}

CUresult cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p, unsigned int Flags) {
    if (!pdptr) return CUDA_ERROR_INVALID_VALUE;

    if (real_driver_available()) {
        return CudaDriverPassthrough::instance().cuMemHostGetDevicePointer(pdptr, p, Flags);
    }

    *pdptr = (CUdeviceptr)p;
    return CUDA_SUCCESS;
}

CUresult cuMemHostGetFlags(unsigned int *pFlags, void *p) {
    FORWARD_TO_REAL(cuMemHostGetFlags, pFlags, p);
    if (pFlags) *pFlags = 0;
    return CUDA_SUCCESS;
}

CUresult cuPointerGetAttribute(void *data, CUpointer_attribute attribute, CUdeviceptr ptr) {
    if (!data) return CUDA_ERROR_INVALID_VALUE;

    const BackendConfig& config = BackendConfig::instance();
    // Hybrid falls back to the simulated answer for pointers the real driver rejects.
    if (real_driver_available()) {
        static const auto real_fn = real_entry<decltype(&cuPointerGetAttribute)>("cuPointerGetAttribute");
        if (real_fn) {
            CUresult result = real_fn(data, attribute, ptr);
            if (config.mode() != FakeGpuMode::Hybrid || result == CUDA_SUCCESS) return result;
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
    // Hybrid falls back to the simulated answer for pointers the real driver rejects.
    if (real_driver_available()) {
        static const auto real_fn = real_entry<decltype(&cuPointerGetAttributes)>("cuPointerGetAttributes");
        if (real_fn) {
            CUresult result = real_fn(numAttributes, attributes, data, ptr);
            if (config.mode() != FakeGpuMode::Hybrid || result == CUDA_SUCCESS) return result;
        }
    }

    for (unsigned int i = 0; i < numAttributes; ++i) {
        cuPointerGetAttribute(data[i], attributes[i], ptr);
    }
    return CUDA_SUCCESS;
}

CUresult cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
    if (real_driver_available()) {
        if (BackendConfig::instance().mode() == FakeGpuMode::Hybrid) {
            if (void* host = hybrid_host_view(dstDevice)) return fill_and_record(host, dstDevice, uc, N);
        }
        static const auto real_fn = real_entry<decltype(&cuMemsetD8)>("cuMemsetD8");
        if (real_fn) return record_memset_if_ok(real_fn(dstDevice, uc, N), dstDevice, N);
    }

    return fill_and_record((void*)dstDevice, dstDevice, uc, N);
}

CUresult cuMemsetD16(CUdeviceptr dstDevice, unsigned short us, size_t N) {
    if (real_driver_available()) {
        if (BackendConfig::instance().mode() == FakeGpuMode::Hybrid) {
            if (void* host = hybrid_host_view(dstDevice)) return fill_and_record(host, dstDevice, us, N);
        }
        static const auto real_fn = real_entry<decltype(&cuMemsetD16)>("cuMemsetD16");
        if (real_fn) return record_memset_if_ok(real_fn(dstDevice, us, N), dstDevice, N * sizeof(us));
    }

    return fill_and_record((void*)dstDevice, dstDevice, us, N);
}

CUresult cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
    if (real_driver_available()) {
        if (BackendConfig::instance().mode() == FakeGpuMode::Hybrid) {
            if (void* host = hybrid_host_view(dstDevice)) return fill_and_record(host, dstDevice, ui, N);
        }
        static const auto real_fn = real_entry<decltype(&cuMemsetD32)>("cuMemsetD32");
        if (real_fn) return record_memset_if_ok(real_fn(dstDevice, ui, N), dstDevice, N * sizeof(ui));
    }

    return fill_and_record((void*)dstDevice, dstDevice, ui, N);
}

CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream) {
    if (real_driver_available()) {
        if (BackendConfig::instance().mode() == FakeGpuMode::Hybrid) {
            if (void* host = hybrid_host_view(dstDevice)) return fill_and_record(host, dstDevice, uc, N);
        }
        static const auto real_fn = real_entry<decltype(&cuMemsetD8Async)>("cuMemsetD8Async");
        if (real_fn) return record_memset_if_ok(real_fn(dstDevice, uc, N, hStream), dstDevice, N);
    }

    return cuMemsetD8(dstDevice, uc, N);
}

CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream) {
    if (real_driver_available()) {
        if (BackendConfig::instance().mode() == FakeGpuMode::Hybrid) {
            if (void* host = hybrid_host_view(dstDevice)) return fill_and_record(host, dstDevice, ui, N);
        }
        static const auto real_fn = real_entry<decltype(&cuMemsetD32Async)>("cuMemsetD32Async");
        if (real_fn) return record_memset_if_ok(real_fn(dstDevice, ui, N, hStream), dstDevice, N * sizeof(ui));
    }

    return cuMemsetD32(dstDevice, ui, N);
}

CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream) {
    if (!dst || !src) return CUDA_ERROR_INVALID_VALUE;

    if (real_driver_available()) {
        const bool hybrid = BackendConfig::instance().mode() == FakeGpuMode::Hybrid;
        void* dst_host = hybrid ? hybrid_host_view(dst) : nullptr;
        void* src_host = hybrid ? hybrid_host_view(src) : nullptr;

        // Without the async entry points fall back to synchronous copies.
        CUresult result = CUDA_SUCCESS;
        if (dst_host && src_host) {
            memcpy(dst_host, src_host, ByteCount);
        } else if (dst_host) {
            static const auto real_fn =
                real_entry_v2<CUresult (*)(void*, CUdeviceptr, size_t, CUstream)>("cuMemcpyDtoHAsync");
            result = real_fn ? real_fn(dst_host, src, ByteCount, hStream)
                             : CudaDriverPassthrough::instance().cuMemcpyDtoH(dst_host, src, ByteCount);
        } else if (src_host) {
            static const auto real_fn =
                real_entry_v2<CUresult (*)(CUdeviceptr, const void*, size_t, CUstream)>("cuMemcpyHtoDAsync");
            result = real_fn ? real_fn(dst, src_host, ByteCount, hStream)
                             : CudaDriverPassthrough::instance().cuMemcpyHtoD(dst, src_host, ByteCount);
        } else {
            static const auto real_fn = real_entry<decltype(&cuMemcpyAsync)>("cuMemcpyAsync");
            result = real_fn ? real_fn(dst, src, ByteCount, hStream)
                             : CudaDriverPassthrough::instance().cuMemcpyDtoD(dst, src, ByteCount);
        }
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

    if (real_driver_available()) {
        const bool hybrid = BackendConfig::instance().mode() == FakeGpuMode::Hybrid;
        if (void* host = hybrid ? hybrid_host_view(srcDevice) : nullptr) {
            memcpy(dstHost, host, ByteCount);
            GlobalState::instance().record_memcpy_d2h((void*)srcDevice, ByteCount);
            return CUDA_SUCCESS;
        }

        static const auto real_fn = real_entry_v2<decltype(&cuMemcpyDtoHAsync)>("cuMemcpyDtoHAsync");
        if (real_fn) {
            CUresult result = real_fn(dstHost, srcDevice, ByteCount, hStream);
            if (hybrid && result == CUDA_SUCCESS) {
                GlobalState::instance().record_memcpy_d2h((void*)srcDevice, ByteCount);
            }
            return result;
        }
    }

    return cuMemcpyDtoH(dstHost, srcDevice, ByteCount);
}

CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream) {
    if (!dstDevice || !srcHost) return CUDA_ERROR_INVALID_VALUE;

    if (real_driver_available()) {
        const bool hybrid = BackendConfig::instance().mode() == FakeGpuMode::Hybrid;
        if (void* host = hybrid ? hybrid_host_view(dstDevice) : nullptr) {
            memcpy(host, srcHost, ByteCount);
            GlobalState::instance().record_memcpy_h2d((void*)dstDevice, ByteCount);
            return CUDA_SUCCESS;
        }

        static const auto real_fn = real_entry_v2<decltype(&cuMemcpyHtoDAsync)>("cuMemcpyHtoDAsync");
        if (real_fn) {
            CUresult result = real_fn(dstDevice, srcHost, ByteCount, hStream);
            if (hybrid && result == CUDA_SUCCESS) {
                GlobalState::instance().record_memcpy_h2d((void*)dstDevice, ByteCount);
            }
            return result;
        }
    }

    return cuMemcpyHtoD(dstDevice, srcHost, ByteCount);
}

CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
    if (!dstDevice || !srcDevice) return CUDA_ERROR_INVALID_VALUE;

    if (real_driver_available()) {
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

    if (real_driver_available()) {
        static const auto real_fn = real_entry<decltype(&cuMemcpyPeer)>("cuMemcpyPeer");
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

    if (real_driver_available()) {
        static const auto real_fn = real_entry<decltype(&cuMemcpyPeerAsync)>("cuMemcpyPeerAsync");
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
    unsigned int flags = 0;
    if (!lookup_stream_flags(hStream, flags)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (streamId) *streamId = (unsigned long long)(uintptr_t)hStream;
    return CUDA_SUCCESS;
}

CUresult cuStreamAddCallback(CUstream hStream, void *callback, void *userData, unsigned int flags) {
    unsigned int stream_flags = 0;
    if (!lookup_stream_flags(hStream, stream_flags)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (record_unsupported_api("cuStreamAddCallback", "callback_not_registered")) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
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
    FORWARD_TO_REAL(cuDeviceGetDefaultMemPool, pool_out, real_device(dev));
    if (pool_out) *pool_out = (CUmemoryPool)(uintptr_t)(dev + 1);
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetMemPool(CUmemoryPool *pool, CUdevice dev) {
    FORWARD_TO_REAL(cuDeviceGetMemPool, pool, real_device(dev));
    if (pool) *pool = (CUmemoryPool)(uintptr_t)(dev + 1);
    return CUDA_SUCCESS;
}

CUresult cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool) {
    FORWARD_TO_REAL(cuDeviceSetMemPool, real_device(dev), pool);
    return CUDA_SUCCESS;
}

CUresult cuMemPoolCreate(CUmemoryPool *pool, const void *poolProps) {
    FORWARD_TO_REAL(cuMemPoolCreate, pool, poolProps);
    if (pool) *pool = (CUmemoryPool)(uintptr_t)1;
    return CUDA_SUCCESS;
}

CUresult cuMemPoolDestroy(CUmemoryPool pool) {
    FORWARD_TO_REAL(cuMemPoolDestroy, pool);
    return CUDA_SUCCESS;
}

CUresult cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, CUstream hStream) {
    if (!dptr) return CUDA_ERROR_INVALID_VALUE;

    if (real_driver_available()) {
        auto alloc_real = [&](CUdeviceptr* out) { return real_mem_alloc_async(out, bytesize, hStream); };
        if (BackendConfig::instance().mode() == FakeGpuMode::Passthrough) return alloc_real(dptr);

        GlobalState::instance().initialize();
        HybridMemoryManager::instance().initialize();

        const int virtual_device = current_context_device;
        return hybrid_allocate(
            dptr,
            bytesize,
            virtual_device,
            HybridMemoryManager::instance().decide_allocation(virtual_device, bytesize),
            alloc_real);
    }

    // Simulate mode: treat async alloc as sync alloc.
    return cuMemAlloc(dptr, bytesize);
}

CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {
    if (real_driver_available()) {
        HybridAllocation alloc;
        if (BackendConfig::instance().mode() == FakeGpuMode::Hybrid &&
            HybridMemoryManager::instance().release_allocation((void*)dptr, alloc)) {
            size_t size = 0;
            int device = -1;
            GlobalState::instance().release_allocation((void*)dptr, size, device);

            switch (alloc.type) {
                case HybridAllocation::Type::RealDevice:
                    break;
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
        // Real device memory, passthrough and untracked pointers: real async free if available.
        static const auto real_fn = real_entry<decltype(&cuMemFreeAsync)>("cuMemFreeAsync");
        if (real_fn) return real_fn(dptr, hStream);
        return CudaDriverPassthrough::instance().cuMemFree(dptr);
    }

    return cuMemFree(dptr);
}

CUresult cuMemAllocFromPoolAsync(CUdeviceptr *dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream) {
    if (!dptr) return CUDA_ERROR_INVALID_VALUE;

    if (real_driver_available()) {
        static const auto real_fn = real_entry<decltype(&cuMemAllocFromPoolAsync)>("cuMemAllocFromPoolAsync");
        if (BackendConfig::instance().mode() == FakeGpuMode::Passthrough) {
            if (real_fn) return real_fn(dptr, bytesize, pool, hStream);
            return cuMemAllocAsync(dptr, bytesize, hStream);
        }

        GlobalState::instance().initialize();
        HybridMemoryManager::instance().initialize();

        const int virtual_device = current_context_device;
        auto decision = HybridMemoryManager::instance().decide_allocation(virtual_device, bytesize);

        if (decision == HybridMemoryManager::AllocationDecision::UseReal) {
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

        // Apply the same decision with cuMemAllocAsync's policy handling.
        return hybrid_allocate(dptr, bytesize, virtual_device, decision, [&](CUdeviceptr* out) {
            return real_mem_alloc_async(out, bytesize, hStream);
        });
    }

    return cuMemAlloc(dptr, bytesize);
}

CUresult cuMemPoolSetAttribute(CUmemoryPool pool, int attr, void *value) {
    FORWARD_TO_REAL(cuMemPoolSetAttribute, pool, attr, value);
    return CUDA_SUCCESS;
}

CUresult cuMemPoolGetAttribute(CUmemoryPool pool, int attr, void *value) {
    FORWARD_TO_REAL(cuMemPoolGetAttribute, pool, attr, value);
    return CUDA_SUCCESS;
}

CUresult cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep) {
    FORWARD_TO_REAL(cuMemPoolTrimTo, pool, minBytesToKeep);
    return CUDA_SUCCESS;
}

CUresult cuCtxResetPersistingL2Cache(void) {
    FORWARD_TO_REAL(cuCtxResetPersistingL2Cache);
    return CUDA_SUCCESS;
}

CUresult cuMemGetAddressRange(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr) {
    FORWARD_TO_REAL(cuMemGetAddressRange, pbase, psize, dptr);
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
    FORWARD_TO_REAL(cuDeviceGetP2PAttribute, value, attrib, real_device(srcDevice), real_device(dstDevice));
    if (!value) return CUDA_ERROR_INVALID_VALUE;
    GlobalState& state = GlobalState::instance();
    state.initialize();
    if (srcDevice < 0 || srcDevice >= state.get_device_count() ||
        dstDevice < 0 || dstDevice >= state.get_device_count() || srcDevice == dstDevice) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    const ModeledDeviceTopology topology = state.snapshot_device_topology();
    const bool access = topology.can_access_peer(srcDevice, dstDevice);
    const bool nvlink = topology.nvlink_links(srcDevice, dstDevice) > 0;
    switch (attrib) {
        case 0x01:  // CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK: lower is faster
            *value = !access || nvlink ? 0 : topology.pcie_relation(srcDevice, dstDevice) == "PIX" ? 1 : 2;
            break;
        case 0x02:  // CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED
        case 0x04:  // CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED
            *value = access ? 1 : 0;
            break;
        case 0x03:  // CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED
            *value = nvlink ? 1 : 0;
            break;
        default:
            return CUDA_ERROR_INVALID_VALUE;
    }
    return CUDA_SUCCESS;
}

CUresult cuCtxDetach(CUcontext ctx) {
    FORWARD_TO_REAL(cuCtxDetach, ctx);
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetTexture1DLinearMaxWidth(size_t *maxWidthInElements, int format, unsigned numChannels, CUdevice dev) {
    FORWARD_TO_REAL(
        cuDeviceGetTexture1DLinearMaxWidth,
        maxWidthInElements,
        format,
        numChannels,
        real_device(dev));
    if (maxWidthInElements) *maxWidthInElements = 134217728;
    return CUDA_SUCCESS;
}

// Newer CUDA libraries also resolve a broad Driver API surface directly with
// dlsym(libcuda_handle, ...), instead of exclusively using cuGetProcAddress.
// These forwarding exports keep hybrid/passthrough ABI-compatible with the
// backing driver.  Simulate mode only needs stable no-op handles because its
// own Runtime/cuBLAS libraries do not execute real kernels.

CUresult cuLinkCreate_v2(unsigned int numOptions, int *options, void **optionValues, void **stateOut) {
    FORWARD_TO_REAL(cuLinkCreate_v2, numOptions, options, optionValues, stateOut);
    if (stateOut) *stateOut = reinterpret_cast<void*>(uintptr_t{1});
    return CUDA_SUCCESS;
}

CUresult cuLinkCreate(unsigned int numOptions, int *options, void **optionValues, void **stateOut) {
    return cuLinkCreate_v2(numOptions, options, optionValues, stateOut);
}

CUresult cuLinkAddData_v2(
    void *state,
    int type,
    void *data,
    size_t size,
    const char *name,
    unsigned int numOptions,
    int *options,
    void **optionValues) {
    FORWARD_TO_REAL(cuLinkAddData_v2, state, type, data, size, name, numOptions, options, optionValues);
    return CUDA_SUCCESS;
}

CUresult cuLinkAddData(
    void *state,
    int type,
    void *data,
    size_t size,
    const char *name,
    unsigned int numOptions,
    int *options,
    void **optionValues) {
    return cuLinkAddData_v2(state, type, data, size, name, numOptions, options, optionValues);
}

CUresult cuLinkAddFile_v2(
    void *state,
    int type,
    const char *path,
    unsigned int numOptions,
    int *options,
    void **optionValues) {
    FORWARD_TO_REAL(cuLinkAddFile_v2, state, type, path, numOptions, options, optionValues);
    return CUDA_SUCCESS;
}

CUresult cuLinkAddFile(
    void *state,
    int type,
    const char *path,
    unsigned int numOptions,
    int *options,
    void **optionValues) {
    return cuLinkAddFile_v2(state, type, path, numOptions, options, optionValues);
}

CUresult cuLinkComplete(void *state, void **cubinOut, size_t *sizeOut) {
    FORWARD_TO_REAL(cuLinkComplete, state, cubinOut, sizeOut);
    if (cubinOut) *cubinOut = nullptr;
    if (sizeOut) *sizeOut = 0;
    return CUDA_SUCCESS;
}

CUresult cuLinkDestroy(void *state) {
    FORWARD_TO_REAL(cuLinkDestroy, state);
    return CUDA_SUCCESS;
}

CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name) {
    return cuModuleGetGlobal(dptr, bytes, hmod, name);
}

CUresult cuKernelGetFunction(CUfunction *pFunc, void *kernel) {
    FORWARD_TO_REAL(cuKernelGetFunction, pFunc, kernel);
    if (pFunc) *pFunc = reinterpret_cast<CUfunction>(uintptr_t{1});
    return CUDA_SUCCESS;
}

CUresult cuKernelGetAttribute(int *value, int attribute, void *kernel, CUdevice device) {
    FORWARD_TO_REAL(cuKernelGetAttribute, value, attribute, kernel, real_device(device));
    if (value) *value = 0;
    return CUDA_SUCCESS;
}

CUresult cuKernelSetAttribute(int attribute, int value, void *kernel, CUdevice device) {
    FORWARD_TO_REAL(cuKernelSetAttribute, attribute, value, kernel, real_device(device));
    return CUDA_SUCCESS;
}

CUresult cuLibraryGetKernel(void **pKernel, void *library, const char *name) {
    FORWARD_TO_REAL(cuLibraryGetKernel, pKernel, library, name);
    if (pKernel) *pKernel = reinterpret_cast<void*>(uintptr_t{1});
    return CUDA_SUCCESS;
}

CUresult cuLibraryLoadData(
    void **library,
    const void *code,
    int *jitOptions,
    void **jitOptionValues,
    unsigned int numJitOptions,
    int *libraryOptions,
    void **libraryOptionValues,
    unsigned int numLibraryOptions) {
    FORWARD_TO_REAL(
        cuLibraryLoadData,
        library,
        code,
        jitOptions,
        jitOptionValues,
        numJitOptions,
        libraryOptions,
        libraryOptionValues,
        numLibraryOptions);
    if (library) *library = reinterpret_cast<void*>(uintptr_t{1});
    return CUDA_SUCCESS;
}

CUresult cuLaunchKernelEx(const void *config, CUfunction f, void **kernelParams, void **extra) {
    FORWARD_TO_REAL(cuLaunchKernelEx, config, f, kernelParams, extra);
    GlobalState& state = GlobalState::instance();
    state.initialize();
    state.record_kernel_launch("cuLaunchKernelEx");
    if (record_unsupported_api(state, "cuLaunchKernelEx")) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return CUDA_SUCCESS;
}

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize) {
    FORWARD_TO_REAL(cuOccupancyMaxActiveBlocksPerMultiprocessor, numBlocks, func, blockSize, dynamicSMemSize);
    if (numBlocks) *numBlocks = 1;
    return CUDA_SUCCESS;
}

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags) {
    FORWARD_TO_REAL(
        cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags,
        numBlocks,
        func,
        blockSize,
        dynamicSMemSize,
        flags);
    return cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize);
}

CUresult cuMemcpy2DAsync_v2(const void *copy, CUstream stream) {
    FORWARD_TO_REAL(cuMemcpy2DAsync_v2, copy, stream);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char value, size_t count) {
    return cuMemsetD8(dstDevice, value, count);
}

CUresult cuMemPrefetchAsync(CUdeviceptr ptr, size_t count, CUdevice dstDevice, CUstream stream) {
    FORWARD_TO_REAL(cuMemPrefetchAsync, ptr, count, real_device(dstDevice), stream);
    return CUDA_SUCCESS;
}

CUresult cuStreamWaitValue32(CUstream stream, CUdeviceptr addr, uint32_t value, unsigned int flags) {
    FORWARD_TO_REAL(cuStreamWaitValue32, stream, addr, value, flags);
    return CUDA_SUCCESS;
}

CUresult cuStreamBatchMemOp(CUstream stream, unsigned int count, void *params, unsigned int flags) {
    FORWARD_TO_REAL(cuStreamBatchMemOp, stream, count, params, flags);
    return CUDA_SUCCESS;
}

CUresult cuStreamBeginCapture(CUstream stream) {
    FORWARD_TO_REAL(cuStreamBeginCapture, stream);
    return CUDA_SUCCESS;
}

CUresult cuStreamBeginCapture_v2(CUstream stream, int mode) {
    FORWARD_TO_REAL(cuStreamBeginCapture_v2, stream, mode);
    return CUDA_SUCCESS;
}

CUresult cuThreadExchangeStreamCaptureMode(int *mode) {
    FORWARD_TO_REAL(cuThreadExchangeStreamCaptureMode, mode);
    return CUDA_SUCCESS;
}

CUresult cuStreamEndCapture(CUstream stream, void **graph) {
    FORWARD_TO_REAL(cuStreamEndCapture, stream, graph);
    if (graph) *graph = reinterpret_cast<void*>(uintptr_t{1});
    return CUDA_SUCCESS;
}

CUresult cuStreamIsCapturing(CUstream stream, int *status) {
    FORWARD_TO_REAL(cuStreamIsCapturing, stream, status);
    if (status) *status = 0;
    return CUDA_SUCCESS;
}

CUresult cuStreamGetCaptureInfo_v2(
    CUstream stream,
    int *status,
    uint64_t *id,
    void **graph,
    const void ***dependencies,
    size_t *dependencyCount) {
    FORWARD_TO_REAL(cuStreamGetCaptureInfo_v2, stream, status, id, graph, dependencies, dependencyCount);
    if (status) *status = 0;
    if (id) *id = 0;
    if (graph) *graph = nullptr;
    if (dependencies) *dependencies = nullptr;
    if (dependencyCount) *dependencyCount = 0;
    return CUDA_SUCCESS;
}

CUresult cuTensorMapEncodeTiled(
    void *tensorMap,
    int tensorDataType,
    uint32_t tensorRank,
    void *globalAddress,
    const uint64_t *globalDim,
    const uint64_t *globalStrides,
    const uint32_t *boxDim,
    const uint32_t *elementStrides,
    int interleave,
    int swizzle,
    int l2Promotion,
    int oobFill) {
    FORWARD_TO_REAL(
        cuTensorMapEncodeTiled,
        tensorMap,
        tensorDataType,
        tensorRank,
        globalAddress,
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        interleave,
        swizzle,
        l2Promotion,
        oobFill);
    return CUDA_SUCCESS;
}

CUresult cuGraphKernelNodeGetParams(void *node, void *params) {
    FORWARD_TO_REAL(cuGraphKernelNodeGetParams, node, params);
    return CUDA_SUCCESS;
}

CUresult cuGraphExecKernelNodeSetParams(void *graphExec, void *node, const void *params) {
    FORWARD_TO_REAL(cuGraphExecKernelNodeSetParams, graphExec, node, params);
    return CUDA_SUCCESS;
}

CUresult cuGraphExecDestroy(void *graphExec) {
    FORWARD_TO_REAL(cuGraphExecDestroy, graphExec);
    return CUDA_SUCCESS;
}

CUresult cuGraphLaunch(void *graphExec, CUstream stream) {
    FORWARD_TO_REAL(cuGraphLaunch, graphExec, stream);
    GlobalState& state = GlobalState::instance();
    state.initialize();
    if (record_unsupported_api(state, "cuGraphLaunch")) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return CUDA_SUCCESS;
}

CUresult cuGraphDestroy(void *graph) {
    FORWARD_TO_REAL(cuGraphDestroy, graph);
    return CUDA_SUCCESS;
}

CUresult cuGraphInstantiate(
    void **graphExec, void *graph, void **errorNode, char *logBuffer, size_t bufferSize) {
    FORWARD_TO_REAL(cuGraphInstantiate, graphExec, graph, errorNode, logBuffer, bufferSize);
    if (graphExec) *graphExec = reinterpret_cast<void*>(uintptr_t{1});
    return CUDA_SUCCESS;
}

CUresult cuGraphGetNodes(void *graph, void **nodes, size_t *numNodes) {
    FORWARD_TO_REAL(cuGraphGetNodes, graph, nodes, numNodes);
    if (numNodes) *numNodes = 0;
    return CUDA_SUCCESS;
}

CUresult cuGraphNodeGetType(void *node, int *type) {
    FORWARD_TO_REAL(cuGraphNodeGetType, node, type);
    if (type) *type = 0;
    return CUDA_SUCCESS;
}

// cuGetExportTable - internal NVIDIA API used by CUDA Runtime
// This is undocumented and returns internal function tables
// WARNING: Returning NULL causes the real libcudart to crash!
// We need to either provide a real export table or avoid being used with real libcudart
CUresult cuGetExportTable(const void **ppExportTable, const void *pExportTableId) {
    FORWARD_TO_REAL(cuGetExportTable, ppExportTable, pExportTableId);

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

    if (config.mode() == FakeGpuMode::Hybrid &&
        real_driver_available() &&
        !hybrid_should_interpose_symbol(symbol)) {
        // Allow CUDA runtime / frameworks to resolve extra driver entry points without us stubbing them all.
        if (void* fn = real_driver_symbol(symbol, cudaVersion, flags)) {
            *pfn = fn;
            return CUDA_SUCCESS;
        }
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
    MAP_FUNC(cuModuleGetGlobal_v2)
    MAP_FUNC(cuLaunchKernel)
    MAP_FUNC(cuLaunchKernelEx)
    MAP_FUNC(cuFuncGetAttribute)
    MAP_FUNC(cuFuncSetAttribute)
    MAP_FUNC(cuFuncSetCacheConfig)
    MAP_FUNC(cuKernelGetFunction)
    MAP_FUNC(cuKernelGetAttribute)
    MAP_FUNC(cuKernelSetAttribute)
    MAP_FUNC(cuLibraryGetKernel)
    MAP_FUNC(cuLibraryLoadData)
    MAP_FUNC(cuLinkCreate)
    MAP_FUNC(cuLinkCreate_v2)
    MAP_FUNC(cuLinkAddData)
    MAP_FUNC(cuLinkAddData_v2)
    MAP_FUNC(cuLinkAddFile)
    MAP_FUNC(cuLinkAddFile_v2)
    MAP_FUNC(cuLinkComplete)
    MAP_FUNC(cuLinkDestroy)
    MAP_FUNC(cuOccupancyMaxActiveBlocksPerMultiprocessor)
    MAP_FUNC(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)

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
    MAP_FUNC(cuMemcpy2DAsync_v2)
    MAP_FUNC(cuMemsetD8_v2)
    MAP_FUNC(cuMemPrefetchAsync)
    MAP_FUNC(cuStreamWaitValue32)
    MAP_FUNC(cuStreamBatchMemOp)
    MAP_FUNC(cuStreamBeginCapture)
    MAP_FUNC(cuStreamBeginCapture_v2)
    MAP_FUNC(cuThreadExchangeStreamCaptureMode)
    MAP_FUNC(cuStreamEndCapture)
    MAP_FUNC(cuStreamIsCapturing)
    MAP_FUNC(cuStreamGetCaptureInfo_v2)
    MAP_FUNC(cuTensorMapEncodeTiled)
    MAP_FUNC(cuGraphKernelNodeGetParams)
    MAP_FUNC(cuGraphExecKernelNodeSetParams)
    MAP_FUNC(cuGraphExecDestroy)
    MAP_FUNC(cuGraphLaunch)
    MAP_FUNC(cuGraphDestroy)
    MAP_FUNC(cuGraphInstantiate)
    MAP_FUNC(cuGraphGetNodes)
    MAP_FUNC(cuGraphNodeGetType)
    MAP_FUNC(cuDeviceGetP2PAttribute)
    MAP_FUNC(cuCtxDetach)
    MAP_FUNC(cuDeviceGetTexture1DLinearMaxWidth)

    #undef MAP_FUNC

    // For unknown symbols, return NULL but success (some symbols are optional).
    // FGPU_LOG("[FakeCUDA-Driver] cuGetProcAddress: symbol '%s' NOT FOUND\n", symbol);
    *pfn = NULL;
    return CUDA_SUCCESS;
}

CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion, unsigned long long flags, void *symbolStatus) {
    const BackendConfig& config = BackendConfig::instance();
    if (config.mode() == FakeGpuMode::Passthrough && real_driver_available()) {
        static const auto real_fn = real_entry<decltype(&cuGetProcAddress_v2)>("cuGetProcAddress_v2");
        if (real_fn) return real_fn(symbol, pfn, cudaVersion, flags, symbolStatus);
        return CudaDriverPassthrough::instance().cuGetProcAddress(symbol, pfn, cudaVersion, flags);
    }

    CUresult result = cuGetProcAddress(symbol, pfn, cudaVersion, flags);
    if (symbolStatus) {
        // Best-effort: 0 means found, 1 means not found.
        *static_cast<int*>(symbolStatus) = (result == CUDA_SUCCESS && *pfn != nullptr) ? 0 : 1;
    }
    return result;
}

} // extern "C"
