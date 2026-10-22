#include "global_state.hpp"
#include "backend_config.hpp"
#include "logging.hpp"
#include "gpu_profile.hpp"
#include "native_smi.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <limits>
#include <sstream>
#include <string>
#include <tuple>

namespace fake_gpu {

namespace {
constexpr int kDefaultDeviceCount = 8;
thread_local bool g_resolving_process_global_state = false;

const char* cuda_dtype_name(int data_type) {
    switch (data_type) {
        case 0: return "FP32";
        case 1: return "FP64";
        case 2: return "FP16";
        case 3: return "INT8";
        case 14: return "BF16";
        default: return "OTHER";
    }
}

std::string trim_copy(const std::string& value) {
    const size_t begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return "";
    const size_t end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

bool parse_positive_int(const std::string& value, int& out) {
    const std::string trimmed = trim_copy(value);
    if (trimmed.empty()) return false;
    char* end = nullptr;
    long parsed = std::strtol(trimmed.c_str(), &end, 10);
    if (end == trimmed.c_str() || (end && *end != '\0')) return false;
    if (parsed <= 0 || parsed > std::numeric_limits<int>::max()) return false;
    out = static_cast<int>(parsed);
    return true;
}

std::vector<GpuProfile> build_profiles_from_spec(const std::string& spec) {
    std::vector<GpuProfile> profiles;
    std::istringstream stream(spec);
    std::string token;

    while (std::getline(stream, token, ',')) {
        token = trim_copy(token);
        if (token.empty()) continue;

        std::string preset_id = token;
        int repeat = 1;

        const size_t colon = token.find(':');
        if (colon != std::string::npos) {
            preset_id = trim_copy(token.substr(0, colon));
            const std::string count_text = trim_copy(token.substr(colon + 1));
            if (!count_text.empty()) {
                int parsed = 0;
                if (parse_positive_int(count_text, parsed)) {
                    repeat = parsed;
                } else {
                    FGPU_LOG("[FakeGPU] Invalid FAKEGPU_PROFILES entry '%s' (bad count '%s')\n", token.c_str(), count_text.c_str());
                }
            }
        }

        std::optional<GpuProfile> maybe_profile = profile_from_preset_id(preset_id);
        if (!maybe_profile.has_value()) {
            FGPU_LOG("[FakeGPU] Unknown GPU preset '%s'; falling back to A100\n", preset_id.c_str());
            maybe_profile = GpuProfile::A100();
        }

        profiles.insert(profiles.end(), repeat, maybe_profile.value());
    }

    return profiles;
}

std::vector<GpuProfile> build_default_profiles() {
    // Default to eight identical A100-class devices to mirror common server setups.
    const char* profiles_env = std::getenv("FAKEGPU_PROFILES");
    if (profiles_env && *profiles_env) {
        std::vector<GpuProfile> profiles = build_profiles_from_spec(profiles_env);
        if (!profiles.empty()) return profiles;
        FGPU_LOG("[FakeGPU] FAKEGPU_PROFILES is set but produced no devices; falling back to defaults\n");
    }

    std::string preset_id = "a100";
    if (const char* preset_env = std::getenv("FAKEGPU_PROFILE"); preset_env && *preset_env) {
        preset_id = trim_copy(preset_env);
    }

    int count = kDefaultDeviceCount;
    if (const char* count_env = std::getenv("FAKEGPU_DEVICE_COUNT"); count_env && *count_env) {
        int parsed = 0;
        if (parse_positive_int(count_env, parsed)) {
            count = parsed;
        } else {
            FGPU_LOG("[FakeGPU] Invalid FAKEGPU_DEVICE_COUNT='%s'; using %d\n", count_env, kDefaultDeviceCount);
        }
    }

    std::optional<GpuProfile> maybe_profile = profile_from_preset_id(preset_id);
    GpuProfile profile = maybe_profile.has_value() ? maybe_profile.value() : GpuProfile::A100();
    return std::vector<GpuProfile>(count, profile);
}

// CUDA_DEVICE_ORDER (CUDA Programming Guide 5.2.1.2). PCI_BUS_ID enumerates
// in ascending PCI bus id. The default, FASTEST_FIRST, orders by a heuristic
// NVIDIA does not publish; FakeGPU does not model it and also enumerates in
// PCI order, which matches PCI_BUS_ID and NVML.
std::vector<int> cuda_enumeration_order(const std::vector<Device>& devices) {
    std::vector<int> order(devices.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
    auto key = [&devices](int index) {
        PciAddress address;
        parse_pci_address(devices[static_cast<std::size_t>(index)].pci_bus_id, address);
        return std::make_tuple(address.domain, address.bus, address.device, address.function);
    };
    std::stable_sort(order.begin(), order.end(), [&key](int a, int b) { return key(a) < key(b); });
    return order;
}

// An index entry reads like strtoul: leading digits count and the rest is
// ignored, so "0abc" selects device 0 (measured on the real driver). A sign
// or no digits makes the entry invalid.
bool parse_visible_index(const std::string& entry, std::size_t& value) {
    std::size_t position = entry[0] == '+' ? 1 : 0;
    if (position >= entry.size() || !std::isdigit(static_cast<unsigned char>(entry[position]))) return false;
    value = 0;
    for (; position < entry.size() && std::isdigit(static_cast<unsigned char>(entry[position])); ++position) {
        value = std::min<std::size_t>(value * 10 + static_cast<std::size_t>(entry[position] - '0'), 1u << 20);
    }
    return true;
}

// The one device whose UUID starts with prefix; -1 when none or several do.
template <class DeviceList>
int unique_uuid_match(const DeviceList& candidates, const std::string& prefix) {
    int match = -1;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (candidates[i].uuid.compare(0, prefix.size(), prefix) != 0) continue;
        if (match >= 0) return -1;
        match = static_cast<int>(i);
    }
    return match;
}

// The devices CUDA_VISIBLE_DEVICES selects, in mask order (CUDA Programming
// Guide 5.2.1.1). Entries are read up to the first one that names no device,
// so "" and "-1" select nothing. Returns false when a device is listed twice,
// which makes CUDA initialization fail (cuInit returns
// CUDA_ERROR_INVALID_DEVICE on the real driver for "0,0").
bool select_visible_devices(
    const char* mask,
    const std::vector<int>& order,
    std::vector<Device>& devices,
    std::vector<Device>& mig_devices,
    std::vector<int>& physical_out,
    std::vector<Device*>& devices_out) {
    std::istringstream entries{std::string(mask)};
    std::string entry;
    while (std::getline(entries, entry, ',')) {
        entry = trim_copy(entry);
        if (entry.empty()) break;
        if (entry.rfind("MIG-", 0) == 0) {
            // "Only single MIG instance enumeration is supported": a MIG id
            // counts as the first entry and ends the list.
            if (!physical_out.empty()) break;
            Device* instance = nullptr;
            if (entry.rfind("MIG-GPU-", 0) == 0) {
                // Legacy MIG-GPU-<uuid>/<gpu instance>/<compute instance>.
                const std::size_t slash = entry.find('/');
                unsigned int gi = 0, ci = 0;
                char tail = 0;
                if (slash == std::string::npos ||
                    std::sscanf(entry.c_str() + slash, "/%u/%u%c", &gi, &ci, &tail) != 2) {
                    break;
                }
                const int parent = unique_uuid_match(devices, entry.substr(4, slash - 4));
                for (Device& candidate : mig_devices) {
                    if (parent >= 0 && candidate.parent_device_index == parent &&
                        candidate.gpu_instance_id == gi && candidate.compute_instance_id == ci) {
                        instance = &candidate;
                    }
                }
            } else {
                const int match = unique_uuid_match(mig_devices, entry);
                if (match >= 0) instance = &mig_devices[static_cast<std::size_t>(match)];
            }
            if (instance) {
                physical_out.push_back(instance->parent_device_index);
                devices_out.push_back(instance);
            }
            break;
        }
        int physical = -1;
        std::size_t position = 0;
        if (entry.rfind("GPU-", 0) == 0) {
            // A UUID, or enough of its beginning to name one GPU.
            physical = unique_uuid_match(devices, entry);
        } else if (parse_visible_index(entry, position) && position < order.size()) {
            physical = order[position];
        }
        if (physical < 0) break;
        if (std::find(physical_out.begin(), physical_out.end(), physical) != physical_out.end()) return false;
        physical_out.push_back(physical);
        devices_out.push_back(&devices[static_cast<std::size_t>(physical)]);
    }
    return true;
}
} // namespace

GlobalState& GlobalState::instance() {
    static GlobalState* local_instance = new GlobalState();
    if (g_resolving_process_global_state) {
        return *local_instance;
    }

    // Every compatibility library embeds the core objects. Resolve a
    // process-global provider explicitly so CUDA Runtime, Driver, cuBLAS, and
    // NVML calls contribute to one report instead of isolated DSO snapshots.
    static GlobalState* process_instance = []() {
        using provider_t = GlobalState* (*)();
        g_resolving_process_global_state = true;
        void* symbol = dlsym(RTLD_DEFAULT, "fakegpu_process_global_state_v1");
        GlobalState* selected = local_instance;
        if (symbol) {
            auto provider = reinterpret_cast<provider_t>(symbol);
            if (GlobalState* shared = provider()) {
                selected = shared;
            }
        }
        g_resolving_process_global_state = false;
        return selected;
    }();
    return *process_instance;
}

extern "C" GlobalState* fakegpu_process_global_state_v1() {
    return &GlobalState::instance();
}

thread_local int GlobalState::current_device = 0;

GlobalState::GlobalState() {
}

void GlobalState::initialize() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        FGPU_LOG("[GlobalState-%p] initialize called. Current devices: %lu\n", this, devices.size());
        if (initialized) {
            return;
        }

        // Pre-allocate to prevent reallocation during emplace_back.
        // nvitop holds Device* pointers returned by NVML.
        std::vector<GpuProfile> profiles = build_default_profiles();
        devices.reserve(profiles.size());

        for (size_t i = 0; i < profiles.size(); ++i) {
            devices.emplace_back(static_cast<int>(i), profiles[i]);
        }
        device_topology = build_modeled_device_topology(devices);
        fault_model = build_modeled_fault_model(devices.size());
        mig_layout = build_modeled_mig_layout(devices);
        mig_devices.reserve(mig_layout.instances.size());
        for (const auto& instance : mig_layout.instances) {
            Device& parent = devices[
                static_cast<std::size_t>(
                    instance.parent_device_index)];
            mig_devices.emplace_back(parent.index, parent.profile);
            Device& mig_device = mig_devices.back();
            mig_device.name =
                parent.name + " MIG " + instance.profile;
            mig_device.uuid = instance.uuid;
            mig_device.total_memory = instance.memory_bytes;
            mig_device.pci_bus_id = parent.pci_bus_id;
            mig_device.is_mig_device = true;
            mig_device.parent_device_index = parent.index;
            mig_device.mig_device_index =
                instance.mig_device_index;
            mig_device.gpu_instance_id =
                instance.gpu_instance_id;
            mig_device.compute_instance_id =
                instance.compute_instance_id;
        }
        device_stats.resize(devices.size());

        // CUDA's view (CUDA Programming Guide 5.2.1.1). Hybrid and
        // passthrough leave the mask to the real driver, which applies it to
        // the physical GPUs backing the modeled ones. FAKEGPU_PHYSICAL_INDICES
        // means a FakeCUDA parent (fakegpu.init or fakegpu run) already
        // narrowed FAKEGPU_PROFILES to the visible GPUs.
        const std::vector<int> order = cuda_enumeration_order(devices);
        const char* mask = std::getenv("CUDA_VISIBLE_DEVICES");
        if (!mask || std::getenv("FAKEGPU_PHYSICAL_INDICES") ||
            BackendConfig::instance().mode() != FakeGpuMode::Simulate) {
            for (int physical : order) {
                cuda_physical.push_back(physical);
                cuda_devices.push_back(&devices[static_cast<std::size_t>(physical)]);
            }
        } else if (!select_visible_devices(mask, order, devices, mig_devices, cuda_physical, cuda_devices)) {
            cuda_physical.clear();
            cuda_devices.clear();
            visibility = CudaVisibility::InvalidDevice;
        }
        if (visibility != CudaVisibility::InvalidDevice && cuda_devices.empty()) {
            visibility = CudaVisibility::NoDevice;
        }
        initialized = true;
        FGPU_LOG("[GlobalState-%p] Valid devices count after init: %lu\n", this, devices.size());
    }
    ensure_native_smi_publisher();
}

namespace {
Device& missing_device() {
    static Device failure_dev(-1, GpuProfile::A100());
    return failure_dev;
}
} // namespace

int GlobalState::get_device_count() const {
    return static_cast<int>(cuda_devices.size());
}

Device& GlobalState::get_device(int ordinal) {
    if (ordinal < 0 || ordinal >= static_cast<int>(cuda_devices.size())) return missing_device();
    return *cuda_devices[static_cast<std::size_t>(ordinal)];
}

GlobalState::CudaVisibility GlobalState::cuda_visibility() const {
    return visibility;
}

int GlobalState::physical_index(int ordinal) const {
    return physical_index_nolock(ordinal);
}

int GlobalState::get_physical_device_count() const {
    return static_cast<int>(devices.size());
}

Device& GlobalState::get_physical_device(int index) {
    if (index < 0 || index >= static_cast<int>(devices.size())) return missing_device();
    return devices[static_cast<std::size_t>(index)];
}

int GlobalState::physical_index_nolock(int ordinal) const {
    if (ordinal < 0 || ordinal >= static_cast<int>(cuda_physical.size())) return -1;
    return cuda_physical[static_cast<std::size_t>(ordinal)];
}

int GlobalState::current_physical_nolock() const {
    return physical_index_nolock(current_device);
}

void GlobalState::set_current_device(int device) {
    std::lock_guard<std::mutex> lock(mutex);
    if (device >= 0 && device < static_cast<int>(cuda_devices.size())) {
        current_device = device;
    }
}

int GlobalState::get_current_device() const {
    std::lock_guard<std::mutex> lock(mutex);
    return current_device;
}

bool GlobalState::register_allocation(void* ptr, size_t size, int device) {
    std::lock_guard<std::mutex> lock(mutex);
    return register_allocation_nolock(ptr, size, device, AllocationKind::Device);
}

bool GlobalState::register_managed_allocation(void* ptr, size_t size, int device) {
    std::lock_guard<std::mutex> lock(mutex);
    return register_allocation_nolock(ptr, size, device, AllocationKind::Managed);
}

bool GlobalState::register_host_allocation(void* ptr, size_t size, int device) {
    std::lock_guard<std::mutex> lock(mutex);
    if (!ptr) return false;
    if (device < 0 || device >= static_cast<int>(cuda_devices.size())) {
        device = current_device;
    }
    if (allocations.find(ptr) != allocations.end()) return false;

    AllocationRecord rec;
    rec.size = size;
    rec.device = physical_index_nolock(device);
    rec.ordinal = device;
    rec.kind = AllocationKind::Host;
    allocations[ptr] = rec;
    return true;
}

bool GlobalState::release_allocation(void* ptr, size_t& size, int& device) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = allocations.find(ptr);
    if (it == allocations.end()) return false;
    if (it->second.kind == AllocationKind::Host) return false;

    size = it->second.size;
    device = it->second.ordinal;
    const int physical = it->second.device;
    allocations.erase(it);

    devices[physical].used_memory -= size;
    if (Device* target = cuda_devices[static_cast<std::size_t>(device)]; target->is_mig_device) {
        target->used_memory -= size;
    }
    DeviceRuntimeStats& stats = device_stats[physical];
    stats.free_calls += 1;
    saturating_add_u64(stats.free_bytes, static_cast<uint64_t>(size));
    return true;
}

bool GlobalState::release_host_allocation(void* ptr, size_t& size, int& device) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = allocations.find(ptr);
    if (it == allocations.end()) return false;
    if (it->second.kind != AllocationKind::Host) return false;

    size = it->second.size;
    device = it->second.ordinal;
    allocations.erase(it);
    return true;
}

bool GlobalState::get_allocation_info(void* ptr, size_t& size, int& device) const {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = allocations.find(ptr);
    if (it == allocations.end()) return false;
    if (it->second.kind == AllocationKind::Host) return false;

    size = it->second.size;
    device = it->second.ordinal;
    return true;
}

bool GlobalState::get_allocation_info_ex(void* ptr, size_t& size, int& device, AllocationKind& kind) const {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = find_allocation_nolock(ptr);
    if (it == allocations.end()) return false;

    size = it->second.size;
    device = it->second.ordinal;
    kind = it->second.kind;
    return true;
}

bool GlobalState::get_allocation_range(const void* ptr, void*& base, size_t& size) const {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = find_allocation_nolock(ptr);
    if (it == allocations.end()) return false;
    base = it->first;
    size = it->second.size;
    return true;
}

std::vector<std::pair<void*, size_t>> GlobalState::snapshot_allocations() const {
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<std::pair<void*, size_t>> result;
    result.reserve(allocations.size());
    for (const auto& [pointer, allocation] : allocations) {
        result.emplace_back(pointer, allocation.size);
    }
    return result;
}

void GlobalState::record_memcpy_h2d(const void* dst_device_ptr, size_t bytes) {
    if (!dst_device_ptr || bytes == 0) return;
    std::lock_guard<std::mutex> lock(mutex);
    const int device = resolve_device_for_ptr_nolock(dst_device_ptr, current_physical_nolock());
    DeviceRuntimeStats* stats = stats_for_device_nolock(device);
    if (!stats) return;
    stats->memcpy_h2d_calls += 1;
    saturating_add_u64(stats->memcpy_h2d_bytes, static_cast<uint64_t>(bytes));
    link_traffic.record(device_topology, -1, device, static_cast<uint64_t>(bytes), monotonic_seconds());
}

void GlobalState::record_memcpy_d2h(const void* src_device_ptr, size_t bytes) {
    if (!src_device_ptr || bytes == 0) return;
    std::lock_guard<std::mutex> lock(mutex);
    const int device = resolve_device_for_ptr_nolock(src_device_ptr, current_physical_nolock());
    DeviceRuntimeStats* stats = stats_for_device_nolock(device);
    if (!stats) return;
    stats->memcpy_d2h_calls += 1;
    saturating_add_u64(stats->memcpy_d2h_bytes, static_cast<uint64_t>(bytes));
    link_traffic.record(device_topology, device, -1, static_cast<uint64_t>(bytes), monotonic_seconds());
}

void GlobalState::record_memcpy_d2d(const void* dst_device_ptr, const void* src_device_ptr, size_t bytes) {
    if (!dst_device_ptr || !src_device_ptr || bytes == 0) return;
    std::lock_guard<std::mutex> lock(mutex);

    const int dst_device = resolve_device_for_ptr_nolock(dst_device_ptr, -1);
    const int src_device = resolve_device_for_ptr_nolock(src_device_ptr, -1);

    if (dst_device >= 0 && src_device >= 0 && dst_device != src_device) {
        DeviceRuntimeStats* src_stats = stats_for_device_nolock(src_device);
        DeviceRuntimeStats* dst_stats = stats_for_device_nolock(dst_device);
        if (src_stats) {
            src_stats->memcpy_peer_tx_calls += 1;
            saturating_add_u64(src_stats->memcpy_peer_tx_bytes, static_cast<uint64_t>(bytes));
        }
        if (dst_stats) {
            dst_stats->memcpy_peer_rx_calls += 1;
            saturating_add_u64(dst_stats->memcpy_peer_rx_bytes, static_cast<uint64_t>(bytes));
        }
        link_traffic.record(device_topology, src_device, dst_device, static_cast<uint64_t>(bytes), monotonic_seconds());
        return;
    }

    const int device = (dst_device >= 0) ? dst_device : (src_device >= 0 ? src_device : current_physical_nolock());
    DeviceRuntimeStats* stats = stats_for_device_nolock(device);
    if (!stats) return;
    stats->memcpy_d2d_calls += 1;
    saturating_add_u64(stats->memcpy_d2d_bytes, static_cast<uint64_t>(bytes));
}

void GlobalState::record_memcpy_peer(int dst_ordinal, int src_ordinal, size_t bytes) {
    if (bytes == 0) return;
    std::lock_guard<std::mutex> lock(mutex);
    const int dst_device = physical_index_nolock(dst_ordinal);
    const int src_device = physical_index_nolock(src_ordinal);
    if (DeviceRuntimeStats* src_stats = stats_for_device_nolock(src_device)) {
        src_stats->memcpy_peer_tx_calls += 1;
        saturating_add_u64(src_stats->memcpy_peer_tx_bytes, static_cast<uint64_t>(bytes));
    }
    if (DeviceRuntimeStats* dst_stats = stats_for_device_nolock(dst_device)) {
        dst_stats->memcpy_peer_rx_calls += 1;
        saturating_add_u64(dst_stats->memcpy_peer_rx_bytes, static_cast<uint64_t>(bytes));
    }
    if (src_device >= 0 && dst_device >= 0) {
        link_traffic.record(device_topology, src_device, dst_device, static_cast<uint64_t>(bytes), monotonic_seconds());
    }
}

void GlobalState::record_memcpy_h2h(size_t bytes) {
    if (bytes == 0) return;
    std::lock_guard<std::mutex> lock(mutex);
    host_io.memcpy_calls += 1;
    saturating_add_u64(host_io.memcpy_bytes, static_cast<uint64_t>(bytes));
}

void GlobalState::record_memset(const void* dst_device_ptr, size_t bytes) {
    if (!dst_device_ptr || bytes == 0) return;
    std::lock_guard<std::mutex> lock(mutex);
    const int device = resolve_device_for_ptr_nolock(dst_device_ptr, current_physical_nolock());
    DeviceRuntimeStats* stats = stats_for_device_nolock(device);
    if (!stats) return;
    stats->memset_calls += 1;
    saturating_add_u64(stats->memset_bytes, static_cast<uint64_t>(bytes));
}

void GlobalState::record_cublas_gemm(const void* output_device_ptr, uint64_t flops) {
    if (!output_device_ptr) return;
    initialize();
    std::lock_guard<std::mutex> lock(mutex);
    const int device = resolve_device_for_ptr_nolock(output_device_ptr, current_physical_nolock());
    DeviceRuntimeStats* stats = stats_for_device_nolock(device);
    if (!stats) return;
    stats->cublas_gemm_calls += 1;
    saturating_add_u64(stats->cublas_gemm_flops, flops);
}

void GlobalState::record_cublaslt_matmul(const void* output_device_ptr, uint64_t flops) {
    if (!output_device_ptr) return;
    initialize();
    std::lock_guard<std::mutex> lock(mutex);
    const int device = resolve_device_for_ptr_nolock(output_device_ptr, current_physical_nolock());
    DeviceRuntimeStats* stats = stats_for_device_nolock(device);
    if (!stats) return;
    stats->cublaslt_matmul_calls += 1;
    saturating_add_u64(stats->cublaslt_matmul_flops, flops);
}

void GlobalState::record_kernel_launch(const std::string& kernel_name) {
    std::lock_guard<std::mutex> lock(mutex);
    DeviceRuntimeStats* stats = stats_for_device_nolock(current_physical_nolock());
    if (!stats) return;
    stats->kernel_launch_total += 1;
    if (stats->kernel_launches.size() < 10000) {
        stats->kernel_launches[kernel_name] += 1;
    } else {
        stats->kernel_launches["_other_"] += 1;
    }
}

void GlobalState::record_cublas_gemm_typed(const void* output_device_ptr, uint64_t flops, int cuda_data_type) {
    if (!output_device_ptr) return;
    initialize();
    std::lock_guard<std::mutex> lock(mutex);
    const int device = resolve_device_for_ptr_nolock(output_device_ptr, current_physical_nolock());
    DeviceRuntimeStats* stats = stats_for_device_nolock(device);
    if (!stats) return;
    stats->cublas_gemm_calls += 1;
    saturating_add_u64(stats->cublas_gemm_flops, flops);
    auto& dtype_stats = stats->gemm_by_dtype[cuda_data_type];
    dtype_stats.calls += 1;
    saturating_add_u64(dtype_stats.flops, flops);
}

void GlobalState::record_cublaslt_matmul_typed(const void* output_device_ptr, uint64_t flops, int cuda_data_type) {
    if (!output_device_ptr) return;
    initialize();
    std::lock_guard<std::mutex> lock(mutex);
    const int device = resolve_device_for_ptr_nolock(output_device_ptr, current_physical_nolock());
    DeviceRuntimeStats* stats = stats_for_device_nolock(device);
    if (!stats) return;
    stats->cublaslt_matmul_calls += 1;
    saturating_add_u64(stats->cublaslt_matmul_flops, flops);
    auto& dtype_stats = stats->gemm_by_dtype[cuda_data_type];
    dtype_stats.calls += 1;
    saturating_add_u64(dtype_stats.flops, flops);
}

void GlobalState::record_compat_event(int ordinal, const std::string& operation, const std::string& dtype) {
    std::lock_guard<std::mutex> lock(mutex);
    DeviceRuntimeStats* stats = stats_for_device_nolock(physical_index_nolock(ordinal));
    if (!stats) return;
    const std::string key = operation + ":" + dtype;
    auto& evt = stats->compat_events[key];
    if (evt.count == 0) {
        evt.operation = operation;
        evt.dtype = dtype;
    }
    evt.count += 1;
}

bool GlobalState::record_unsupported_api(
    int ordinal,
    const std::string& operation,
    const std::string& behavior,
    const std::string& policy) {
    std::lock_guard<std::mutex> lock(mutex);
    DeviceRuntimeStats* stats = stats_for_device_nolock(physical_index_nolock(ordinal));
    if (!stats) return false;
    const std::string key = operation + ":" + behavior + ":" + policy;
    auto& evt = stats->unsupported_api_events[key];
    const bool first_occurrence = evt.count == 0;
    if (first_occurrence) {
        evt.operation = operation;
        evt.behavior = behavior;
        evt.policy = policy;
    }
    evt.count += 1;
    return first_occurrence;
}

DeviceTransferStats GlobalState::snapshot_device_transfer(int device) const {
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<DeviceTransferStats> transfers = link_traffic.snapshot(device_topology, monotonic_seconds());
    if (device < 0 || static_cast<std::size_t>(device) >= transfers.size()) {
        return DeviceTransferStats{};
    }
    return transfers[static_cast<std::size_t>(device)];
}

std::vector<DeviceReportStats> GlobalState::snapshot_device_report() const {
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<DeviceReportStats> snapshot;
    snapshot.reserve(devices.size());
    std::vector<DeviceTransferStats> transfers = link_traffic.snapshot(device_topology, monotonic_seconds());

    for (size_t i = 0; i < devices.size(); ++i) {
        const Device& dev = devices[i];
        DeviceReportStats entry;
        entry.index = dev.index;
        entry.profile_id = dev.profile.id;
        entry.name = dev.name;
        entry.uuid = dev.uuid;
        entry.pci_bus_id = dev.pci_bus_id;
        entry.architecture = to_string(dev.profile.architecture);
        entry.compute_major = dev.profile.compute_major;
        entry.compute_minor = dev.profile.compute_minor;
        entry.sm_count = dev.profile.sm_count;
        entry.memory_bus_width_bits = dev.profile.memory_bus_width_bits;
        entry.core_clock_mhz = dev.profile.core_clock_mhz;
        entry.memory_clock_mhz = dev.profile.memory_clock_mhz;
        entry.l2_cache_bytes = dev.profile.l2_cache_bytes;
        entry.typical_power_usage_mw = dev.profile.typical_power_usage_mw;
        entry.max_power_limit_mw = dev.profile.max_power_limit_mw;
        for (const auto& dtype : dev.profile.supported_types) {
            entry.supported_types.push_back(to_string(dtype));
        }
        entry.total_memory = dev.total_memory;
        entry.used_memory_current = dev.used_memory;
        entry.used_memory_peak = dev.used_memory_peak;
        entry.transfer = transfers[i];

        const DeviceRuntimeStats& stats = device_stats[i];
        static_cast<DeviceCounters&>(entry) = stats;
        for (const auto& [dtype_int, dtype_stats] : stats.gemm_by_dtype) {
            entry.gemm_by_dtype[cuda_dtype_name(dtype_int)] = {
                dtype_stats.calls,
                dtype_stats.flops,
            };
        }
        for (const auto& [key, evt] : stats.compat_events) {
            entry.compat_events.emplace_back(evt.operation, evt.dtype, evt.count);
        }
        for (const auto& [key, evt] : stats.unsupported_api_events) {
            entry.unsupported_api_events.emplace_back(
                evt.operation,
                evt.behavior,
                evt.policy,
                evt.count);
        }

        snapshot.push_back(std::move(entry));
    }

    return snapshot;
}

ModeledDeviceTopology GlobalState::snapshot_device_topology() const {
    std::lock_guard<std::mutex> lock(mutex);
    return device_topology;
}

ModeledFaultModel GlobalState::snapshot_fault_model() const {
    std::lock_guard<std::mutex> lock(mutex);
    return fault_model;
}

ModeledMigLayout GlobalState::snapshot_mig_layout() const {
    std::lock_guard<std::mutex> lock(mutex);
    return mig_layout;
}

unsigned int GlobalState::get_mig_device_count(
    int parent_device) const {
    unsigned int count = 0;
    for (const Device& device : mig_devices) {
        if (device.parent_device_index == parent_device) {
            ++count;
        }
    }
    return count;
}

Device* GlobalState::get_mig_device(
    int parent_device,
    unsigned int index) {
    for (Device& device : mig_devices) {
        if (
            device.parent_device_index == parent_device &&
            device.mig_device_index == index) {
            return &device;
        }
    }
    return nullptr;
}

Device* GlobalState::get_parent_device_for_mig(
    Device* mig_device) {
    if (!mig_device || !mig_device->is_mig_device) {
        return nullptr;
    }
    const int parent_index = mig_device->parent_device_index;
    if (
        parent_index < 0 ||
        parent_index >= static_cast<int>(devices.size())) {
        return nullptr;
    }
    return &devices[static_cast<std::size_t>(parent_index)];
}

HostIoStats GlobalState::snapshot_host_io() const {
    std::lock_guard<std::mutex> lock(mutex);
    return host_io;
}

void GlobalState::ensure_native_smi_publisher() {
    std::lock_guard<std::mutex> lock(smi_mutex);
    if (native_smi_publisher != nullptr) {
        return;
    }
    native_smi_publisher = create_native_smi_publisher(*this);
}

void GlobalState::stop_native_smi_publisher() {
    void* publisher = nullptr;
    {
        std::lock_guard<std::mutex> lock(smi_mutex);
        publisher = native_smi_publisher;
        native_smi_publisher = nullptr;
    }
    if (publisher != nullptr) {
        destroy_native_smi_publisher(publisher);
    }
}

std::map<void*, GlobalState::AllocationRecord>::const_iterator GlobalState::find_allocation_nolock(
    const void* ptr) const {
    // The last allocation starting at or below ptr, if ptr lies inside it.
    auto it = allocations.upper_bound(const_cast<void*>(ptr));
    if (it == allocations.begin()) return allocations.end();
    --it;
    const auto offset = reinterpret_cast<std::uintptr_t>(ptr) - reinterpret_cast<std::uintptr_t>(it->first);
    return offset == 0 || offset < it->second.size ? it : allocations.end();
}

int GlobalState::resolve_device_for_ptr_nolock(const void* ptr, int fallback_device) const {
    if (!ptr) return fallback_device;
    auto it = find_allocation_nolock(ptr);
    if (it == allocations.end()) return fallback_device;
    if (it->second.kind == AllocationKind::Host) return fallback_device;
    return it->second.device;
}

int GlobalState::resolve_device_for_ptr(const void* ptr, int fallback_device) const {
    std::lock_guard<std::mutex> lock(mutex);
    if (!ptr) return fallback_device;
    auto it = find_allocation_nolock(ptr);
    if (it == allocations.end() || it->second.kind == AllocationKind::Host) return fallback_device;
    return it->second.ordinal;
}

GlobalState::DeviceRuntimeStats* GlobalState::stats_for_device_nolock(int device) {
    if (device < 0 || device >= static_cast<int>(device_stats.size())) return nullptr;
    return &device_stats[device];
}

const GlobalState::DeviceRuntimeStats* GlobalState::stats_for_device_nolock(int device) const {
    if (device < 0 || device >= static_cast<int>(device_stats.size())) return nullptr;
    return &device_stats[device];
}

void GlobalState::saturating_add_u64(uint64_t& target, uint64_t value) {
    const uint64_t max_value = std::numeric_limits<uint64_t>::max();
    if (target >= max_value - value) {
        target = max_value;
    } else {
        target += value;
    }
}

bool GlobalState::register_allocation_nolock(void* ptr, size_t size, int ordinal, AllocationKind kind) {
    if (!ptr) return false;
    const int device = physical_index_nolock(ordinal);
    if (device < 0) return false;
    if (allocations.find(ptr) != allocations.end()) return false;

    // A MIG instance named in CUDA_VISIBLE_DEVICES has its own capacity; the
    // bytes also count against the GPU it is part of, as NVML reports them.
    Device& target = *cuda_devices[static_cast<std::size_t>(ordinal)];
    if (target.used_memory + size > target.total_memory) {
        return false; // not enough fake memory
    }
    if (target.is_mig_device) {
        target.used_memory += size;
        target.used_memory_peak = std::max(target.used_memory_peak, target.used_memory);
    }

    Device& dev = devices[device];
    dev.used_memory += size;
    dev.used_memory_peak = std::max(dev.used_memory_peak, dev.used_memory);
    if (DeviceRuntimeStats* stats = stats_for_device_nolock(device)) {
        stats->alloc_calls += 1;
        saturating_add_u64(stats->alloc_bytes, static_cast<uint64_t>(size));
    }

    AllocationRecord rec;
    rec.size = size;
    rec.device = device;
    rec.ordinal = ordinal;
    rec.kind = kind;
    allocations[ptr] = rec;
    return true;
}

} // namespace fake_gpu

// Keep global state lazy-initialized to avoid interfering with real CUDA startup
// when only a subset of FakeGPU libraries is loaded (for example fake NCCL).
