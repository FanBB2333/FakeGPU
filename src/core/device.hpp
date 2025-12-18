#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include "gpu_profile.hpp"

namespace fake_gpu {

struct Device {
    int index;
    GpuProfile profile;
    std::string name;
    std::string uuid;
    uint64_t total_memory;
    uint64_t used_memory;
    uint64_t used_memory_peak;
    std::string pci_bus_id;

    Device(int idx, const GpuProfile& profile);
};

} // namespace fake_gpu
