#pragma once

#include "nvml_defs.hpp"
#include <cstdint>

// Private, versioned ABI used by fakegpu.nvml. Strings/arrays are borrowed only
// for the duration of SetSnapshot; readers see an atomic, owned snapshot.
struct FakeGpuNvmlDeviceV1 {
    const char* uuid;
    const char* name;
    const char* profile_id;
    const char* pci_bus_id;
    uint64_t total_memory;
    uint64_t used_memory;
};

struct FakeGpuNvmlProcessV1 {
    uint32_t device_index;  // Index in this snapshot, not the persistent registry.
    uint32_t pid;
    uint64_t used_memory;
};

struct FakeGpuNvmlTelemetryV1 {
    uint32_t valid, gpu, memory, temperature, power_mw, sm_clock, memory_clock, pstate;
};

struct FakeGpuNvmlDeviceV2 {
    FakeGpuNvmlDeviceV1 device;
    FakeGpuNvmlTelemetryV1 telemetry;
};

struct FakeGpuNvmlProcessV2 {
    FakeGpuNvmlProcessV1 process;
    uint64_t timestamp_us;
    uint32_t gpu, memory;
};

extern "C" nvmlReturn_t fakegpuNvmlSetSnapshot(
    unsigned int version, unsigned int device_count,
    const void* devices, unsigned int process_count, const void* processes);

namespace fake_gpu::nvml_bridge {
bool enabled();
nvmlReturn_t count(unsigned int* result);
nvmlReturn_t handle(unsigned int index, nvmlDevice_t* result);
nvmlReturn_t lookup(const char* key, bool by_uuid, nvmlDevice_t* result);
nvmlReturn_t memory(nvmlDevice_t device, nvmlMemory_t* result);
nvmlReturn_t processes(nvmlDevice_t device, unsigned int* count, void* infos, bool legacy);
nvmlReturn_t telemetry(nvmlDevice_t device, FakeGpuNvmlTelemetryV1* result);
nvmlReturn_t process_utilization(nvmlDevice_t device, void* samples, unsigned int* count,
                                unsigned long long last_seen);
} // namespace fake_gpu::nvml_bridge
