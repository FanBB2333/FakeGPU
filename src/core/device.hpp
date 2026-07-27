#pragma once
#include <cstddef>
#include <string>
#include <vector>
#include <cstdint>
#include "gpu_profile.hpp"

namespace fake_gpu {

constexpr std::size_t kMaximumModeledNvLinksPerDevice = 18;
constexpr std::size_t kMaximumModeledFaultTypes = 128;
constexpr uint64_t kMaximumModeledFaultCount = 1'000'000'000;
constexpr std::size_t kMaximumModeledMigSlicesPerDevice = 8;
constexpr std::size_t kMaximumModeledMigInstancesPerDevice = 8;

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

struct ModeledFaultEvent {
    int device_index = -1;
    std::string code;
    std::string severity;
    uint64_t count = 0;
};

struct ModeledFaultModel {
    std::string source = "modeled_none";
    bool configured = false;
    bool valid = true;
    std::string error;
    std::vector<ModeledFaultEvent> events;
};

struct ModeledMigInstance {
    int parent_device_index = -1;
    unsigned int mig_device_index = 0;
    unsigned int gpu_instance_id = 0;
    unsigned int compute_instance_id = 0;
    unsigned int slice_count = 0;
    std::string profile;
    std::string uuid;
    uint64_t memory_bytes = 0;
};

struct ModeledMigLayout {
    std::string source = "modeled_none";
    bool configured = false;
    bool valid = true;
    std::string error;
    std::vector<ModeledMigInstance> instances;
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
    bool is_mig_device = false;
    int parent_device_index = -1;
    unsigned int mig_device_index = 0;
    unsigned int gpu_instance_id = 0;
    unsigned int compute_instance_id = 0;
    unsigned int mig_slice_count = 0;
    std::string mig_profile;

    Device(int idx, const GpuProfile& profile);
};

ModeledDeviceTopology build_modeled_device_topology(
    std::size_t device_count);
ModeledFaultModel build_modeled_fault_model(
    std::size_t device_count);
ModeledMigLayout build_modeled_mig_layout(
    const std::vector<Device>& devices);
int modeled_fault_severity_rank(const std::string& severity);

} // namespace fake_gpu
