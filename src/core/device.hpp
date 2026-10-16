#pragma once
#include <cstddef>
#include <string>
#include <vector>
#include <cstdint>
#include "gpu_profile.hpp"

namespace fake_gpu {

constexpr std::size_t kMaximumModeledNvLinksPerDevice = 18;
// Remote PCI bus ID reported for NVLink ports that lead into an NVSwitch.
constexpr const char* kNvSwitchPciBusId = "00000000:c0:00.0";
constexpr std::size_t kMaximumModeledFaultTypes = 128;
constexpr uint64_t kMaximumModeledFaultCount = 1'000'000'000;
constexpr std::size_t kMaximumModeledMigSlicesPerDevice = 8;
constexpr std::size_t kMaximumModeledMigInstancesPerDevice = 8;

struct ModeledPcieLink {
    unsigned int generation = 0;  // 0: no PCIe link
    unsigned int width = 0;

    bool present() const { return generation != 0; }
    // Usable GB/s per direction (see fakegpu/data/platforms.json).
    double bandwidth_gbps() const;
};

// Where one GPU sits and how it reaches the CPU. Mirrors GpuSlot in
// fakegpu/link_topology.py, parsed from FAKEGPU_SLOTS.
struct ModeledSlot {
    std::string type = "any";
    ModeledPcieLink pcie{4, 16};
    ModeledPcieLink gpu_pcie;
    ModeledPcieLink host_pcie;
    std::string host_link = "pcie";
    double host_gbps = 0.0;  // 0: derived from the PCIe links
    unsigned int numa = 0;
    unsigned int bridge = 0;
    std::string pcie_switch = "0";
    bool p2p = true;

    ModeledPcieLink gpu_max_pcie() const;
    double host_bandwidth_gbps() const;  // 0 when unknown
};

// One NVLink port of a GPU, in NVML link-index order.
struct ModeledNvLinkPeer {
    unsigned int link = 0;
    int peer_index = -1;  // -1: the port leads into an NVSwitch fabric
    double bandwidth_gbps = 0.0;
};

struct ModeledNvLinkGroup {
    bool nvswitch = false;
    std::vector<int> members;
    unsigned int links = 0;
    double link_gbps = 0.0;
};

struct ModeledNvLinkPeerSummary {
    int index = -1;
    unsigned int links = 0;
    double bandwidth_gbps = 0.0;
    unsigned int first_link = 0;
    std::string via;
};

struct ModeledDeviceTopology {
    std::string source = "modeled_none";
    std::string platform;
    bool configured = false;
    bool valid = true;
    std::string error;
    std::string nvlink_source = "modeled_none";
    double nvlink_bandwidth_gbps = 900.0;
    std::vector<ModeledSlot> slots;
    std::vector<ModeledNvLinkGroup> nvlink_groups;
    std::vector<std::vector<ModeledNvLinkPeer>> nvlink_peers;

    std::vector<ModeledNvLinkPeerSummary> peers(int index) const;
    unsigned int nvlink_links(int source, int target) const;
    std::string pcie_relation(int source, int target) const;
    std::string relation(int source, int target) const;
    bool can_access_peer(int source, int target) const;
    double peer_bandwidth_gbps(int source, int target) const;
    double host_bandwidth_gbps(int index) const;
    const ModeledSlot* slot(int index) const;
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
    const std::vector<Device>& devices);
ModeledFaultModel build_modeled_fault_model(
    std::size_t device_count);
ModeledMigLayout build_modeled_mig_layout(
    const std::vector<Device>& devices);
int modeled_fault_severity_rank(const std::string& severity);

} // namespace fake_gpu
