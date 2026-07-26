#pragma once
#include <cstddef>
#include <string>
#include <vector>
#include <cstdint>
#include "gpu_profile.hpp"

namespace fake_gpu {

constexpr std::size_t kMaximumModeledNvLinksPerDevice = 18;

struct ModeledNvLinkPeer {
    unsigned int link = 0;
    int peer_index = -1;
    double bandwidth_gbps = 0.0;
};

struct ModeledDeviceTopology {
    std::string source = "modeled_none";
    bool configured = false;
    bool valid = true;
    std::string error;
    double nvlink_bandwidth_gbps = 900.0;
    std::vector<std::vector<ModeledNvLinkPeer>> nvlink_peers;
};

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

ModeledDeviceTopology build_modeled_device_topology(
    std::size_t device_count);

} // namespace fake_gpu
