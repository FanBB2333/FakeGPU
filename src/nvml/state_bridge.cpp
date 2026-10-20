#include "state_bridge.hpp"
#include "../core/device.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <vector>

namespace {
struct ProcessV1 { unsigned int pid; unsigned long long used; };
struct ProcessV2 {
    unsigned int pid;
    unsigned long long used;
    unsigned int gpu_instance, compute_instance;
};
static_assert(sizeof(ProcessV1) == 16 && sizeof(ProcessV2) == 24, "NVML process ABI");

struct UtilizationSample {
    unsigned int pid;
    unsigned long long timestamp;
    unsigned int sm, memory, encoder, decoder;
};
static_assert(sizeof(UtilizationSample) == 32, "NVML utilization sample ABI");

struct Entry {
    std::unique_ptr<fake_gpu::Device> device;
    uint64_t total = 0, used = 0;
    std::vector<ProcessV2> processes;
    FakeGpuNvmlTelemetryV1 telemetry{};
    FakeGpuNvmlLinksV1 links{};
    std::vector<UtilizationSample> samples;
};
std::mutex mutex;
std::atomic<bool> active{false};
// Device objects never move or disappear: nvitop caches handles across refreshes.
std::vector<Entry> entries;

Entry* find(nvmlDevice_t handle) {
    for (auto& entry : entries) {
        if (reinterpret_cast<nvmlDevice_t>(entry.device.get()) == handle) return &entry;
    }
    return nullptr;
}
} // namespace

extern "C" nvmlReturn_t fakegpuNvmlSetSnapshot(
    unsigned int version, unsigned int device_count,
    const void* device_data, unsigned int process_count, const void* process_data) {
    if (version < 1 || version > 3 || device_count > 4096 || process_count > 65536 ||
        (device_count && !device_data) || (process_count && !process_data)) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    try {
        std::lock_guard<std::mutex> lock(mutex);
        std::vector<FakeGpuNvmlDeviceV1> devices(device_count);
        std::vector<FakeGpuNvmlProcessV1> processes(process_count);
        std::vector<FakeGpuNvmlTelemetryV1> telemetry(device_count);
        std::vector<FakeGpuNvmlLinksV1> links(device_count);
        std::vector<std::vector<UtilizationSample>> samples(device_count);
        for (unsigned int i = 0; i < device_count; ++i) {
            if (version == 1) devices[i] = static_cast<const FakeGpuNvmlDeviceV1*>(device_data)[i];
            else {
                FakeGpuNvmlDeviceV3 row{};
                if (version == 2) {
                    const auto& v2 = static_cast<const FakeGpuNvmlDeviceV2*>(device_data)[i];
                    row.device = v2.device;
                    row.telemetry = v2.telemetry;
                } else {
                    row = static_cast<const FakeGpuNvmlDeviceV3*>(device_data)[i];
                }
                devices[i] = row.device;
                telemetry[i] = row.telemetry;
                links[i] = row.links;
                const auto& t = row.telemetry;
                // Power uses NVML's unsigned milliwatt ABI. Catalog devices
                // can exceed 1 kW, so do not impose a lower display-only cap.
                if (t.valid > 1 || t.gpu > 100 || t.memory > 100 || t.temperature > 120 ||
                    t.pstate > 15 || t.sm_clock > 10000 || t.memory_clock > 100000)
                    return NVML_ERROR_INVALID_ARGUMENT;
                const auto& l = row.links;
                if (l.valid > 1 || l.pcie_generation > 6 || l.pcie_width > 32 || l.pcie_max_generation > 6 ||
                    l.pcie_max_width > 32 || l.nvlink_count > 18)
                    return NVML_ERROR_INVALID_ARGUMENT;
            }
        }
        for (unsigned int i = 0; i < process_count; ++i) {
            if (version == 1) processes[i] = static_cast<const FakeGpuNvmlProcessV1*>(process_data)[i];
            else {
                const auto& row = static_cast<const FakeGpuNvmlProcessV2*>(process_data)[i];
                processes[i] = row.process;
                if (row.process.device_index >= device_count || row.gpu > 100 || row.memory > 100)
                    return NVML_ERROR_INVALID_ARGUMENT;
                samples[row.process.device_index].push_back({row.process.pid, row.timestamp_us,
                    row.gpu, row.memory, 0, 0});
            }
        }
        std::unordered_set<std::string> seen;
        for (unsigned int i = 0; i < device_count; ++i) {
            const auto& d = devices[i];
            if (!d.uuid || !*d.uuid || !d.name || !d.profile_id || !d.pci_bus_id ||
                !seen.insert(d.uuid).second) return NVML_ERROR_INVALID_ARGUMENT;
        }
        for (unsigned int i = 0; i < process_count; ++i) {
            if (processes[i].device_index >= device_count || !processes[i].pid)
                return NVML_ERROR_INVALID_ARGUMENT;
        }
        // Prepare all allocations before changing the published snapshot.
        std::vector<Entry> additions;
        std::vector<size_t> indices;
        std::vector<std::vector<ProcessV2>> rows(device_count);
        for (unsigned int i = 0; i < device_count; ++i) {
            size_t index = 0;
            while (index < entries.size() && entries[index].device->uuid != devices[i].uuid) ++index;
            if (index == entries.size()) {
                index += additions.size();
                auto profile = fake_gpu::profile_from_preset_id(devices[i].profile_id)
                    .value_or(fake_gpu::GpuProfile{});
                auto device = std::make_unique<fake_gpu::Device>(static_cast<int>(index), profile);
                device->uuid = devices[i].uuid;
                device->name = devices[i].name;
                device->pci_bus_id = devices[i].pci_bus_id;
                device->total_memory = devices[i].total_memory;
                additions.push_back({std::move(device), 0, 0, {}});
            }
            indices.push_back(index);
        }
        if (entries.size() + additions.size() > 4096) return NVML_ERROR_INSUFFICIENT_SIZE;
        for (unsigned int i = 0; i < process_count; ++i) {
            const auto& p = processes[i];
            rows[p.device_index].push_back({p.pid, p.used_memory, ~0U, ~0U});
        }
        entries.reserve(entries.size() + additions.size());
        for (auto& addition : additions) entries.push_back(std::move(addition));
        for (auto& entry : entries) {
            entry.used = 0;
            entry.processes.clear();
            entry.telemetry = {};
            entry.links = {};
            entry.samples.clear();
        }
        for (unsigned int i = 0; i < device_count; ++i) {
            auto& entry = entries[indices[i]];
            entry.total = devices[i].total_memory;
            entry.used = devices[i].used_memory;
            entry.processes = std::move(rows[i]);
            entry.telemetry = telemetry[i];
            entry.links = links[i];
            entry.samples = std::move(samples[i]);
        }
        active.store(true);
        return NVML_SUCCESS;
    } catch (...) {
        return NVML_ERROR_UNKNOWN; // Never unwind C++ exceptions into ctypes.
    }
}

namespace fake_gpu::nvml_bridge {
bool enabled() { return active.load(); }

nvmlReturn_t count(unsigned int* result) {
    std::lock_guard<std::mutex> lock(mutex);
    *result = static_cast<unsigned int>(entries.size());
    return NVML_SUCCESS;
}

nvmlReturn_t handle(unsigned int index, nvmlDevice_t* result) {
    std::lock_guard<std::mutex> lock(mutex);
    if (index >= entries.size()) return NVML_ERROR_INVALID_ARGUMENT;
    *result = reinterpret_cast<nvmlDevice_t>(entries[index].device.get());
    return NVML_SUCCESS;
}

nvmlReturn_t lookup(const char* key, bool by_uuid, nvmlDevice_t* result) {
    std::lock_guard<std::mutex> lock(mutex);
    for (auto& entry : entries) {
        if ((by_uuid ? entry.device->uuid : entry.device->pci_bus_id) == key) {
            *result = reinterpret_cast<nvmlDevice_t>(entry.device.get());
            return NVML_SUCCESS;
        }
    }
    *result = nullptr;
    return NVML_ERROR_NOT_FOUND;
}

nvmlReturn_t memory(nvmlDevice_t device, nvmlMemory_t* result) {
    std::lock_guard<std::mutex> lock(mutex);
    auto* entry = find(device);
    if (!entry) return NVML_ERROR_INVALID_ARGUMENT;
    result->total = entry->total;
    result->used = entry->used;
    result->free = entry->used < entry->total ? entry->total - entry->used : 0;
    return NVML_SUCCESS;
}

nvmlReturn_t processes(nvmlDevice_t device, unsigned int* count, void* infos, bool legacy) {
    std::lock_guard<std::mutex> lock(mutex);
    auto* entry = find(device);
    if (!entry) return NVML_ERROR_INVALID_ARGUMENT;
    const auto required = static_cast<unsigned int>(entry->processes.size());
    const unsigned int capacity = *count;
    *count = required;
    if (required && (!infos || capacity < required)) return NVML_ERROR_INSUFFICIENT_SIZE;
    for (unsigned int i = 0; i < required; ++i) {
        const auto& p = entry->processes[i];
        if (legacy) {
            static_cast<ProcessV1*>(infos)[i] = {p.pid, p.used};
        } else {
            static_cast<ProcessV2*>(infos)[i] = p;
        }
    }
    return NVML_SUCCESS;
}
nvmlReturn_t telemetry(nvmlDevice_t device, FakeGpuNvmlTelemetryV1* result) {
    std::lock_guard<std::mutex> lock(mutex);
    auto* entry = find(device);
    if (!entry) return NVML_ERROR_INVALID_ARGUMENT;
    if (!entry->telemetry.valid) return NVML_ERROR_NOT_SUPPORTED;
    *result = entry->telemetry;
    return NVML_SUCCESS;
}

nvmlReturn_t links(nvmlDevice_t device, FakeGpuNvmlLinksV1* result) {
    std::lock_guard<std::mutex> lock(mutex);
    auto* entry = find(device);
    if (!entry) return NVML_ERROR_INVALID_ARGUMENT;
    if (!entry->links.valid) return NVML_ERROR_NOT_SUPPORTED;
    *result = entry->links;
    return NVML_SUCCESS;
}

nvmlReturn_t process_utilization(nvmlDevice_t device, void* samples, unsigned int* count,
                                unsigned long long last_seen) {
    if (!count) return NVML_ERROR_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(mutex);
    auto* entry = find(device);
    if (!entry) return NVML_ERROR_INVALID_ARGUMENT;
    if (!entry->telemetry.valid) return NVML_ERROR_NOT_SUPPORTED;
    unsigned int required = 0;
    for (const auto& sample : entry->samples) if (sample.timestamp > last_seen) ++required;
    const unsigned int capacity = *count;
    *count = required;
    if (!required) return NVML_ERROR_NOT_FOUND;
    if (!samples || capacity < required) return NVML_ERROR_INSUFFICIENT_SIZE;
    unsigned int i = 0;
    for (const auto& sample : entry->samples) {
        if (sample.timestamp > last_seen) static_cast<UtilizationSample*>(samples)[i++] = sample;
    }
    return NVML_SUCCESS;
}
} // namespace fake_gpu::nvml_bridge
