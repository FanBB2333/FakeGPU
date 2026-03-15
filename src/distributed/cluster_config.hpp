#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fake_gpu::distributed {

enum class DistributedMode {
    Disabled,
    Simulate,
    Proxy,
    Passthrough,
};

enum class CoordinatorTransport {
    Tcp,
    Unix,
};

struct FabricLinkConfig {
    std::string type;
    double bandwidth_gbps = 0.0;
    double latency_us = 0.0;
    double oversubscription = 1.0;
    bool defined = false;
};

struct ClusterNodeConfig {
    std::string id;
    std::string host;
    std::vector<int> ranks;
    std::vector<std::string> gpu_profiles;
};

struct ClusterConfigModel {
    std::string source_path;
    int version = 0;
    std::string name;
    std::string default_backend;
    std::vector<ClusterNodeConfig> nodes;
    FabricLinkConfig intra_node_fabric;
    FabricLinkConfig inter_node_fabric;
    std::size_t world_size = 0;

    bool loaded() const {
        return !source_path.empty();
    }
};

struct DistributedConfig {
    DistributedMode mode = DistributedMode::Disabled;
    CoordinatorTransport coordinator_transport = CoordinatorTransport::Tcp;
    std::string cluster_config_path;
    std::string coordinator_address;
    std::size_t staging_chunk_bytes = 0;
    ClusterConfigModel cluster_config;
    std::string validation_error;

    bool enabled() const {
        return mode != DistributedMode::Disabled;
    }

    bool valid() const {
        return validation_error.empty();
    }
};

const char* distributed_mode_name(DistributedMode mode);
const char* coordinator_transport_name(CoordinatorTransport transport);
DistributedConfig parse_distributed_config_from_env();
bool load_cluster_config_from_yaml_file(const std::string& path, ClusterConfigModel& out, std::string& error);
bool parse_tcp_endpoint(const std::string& endpoint, std::string& host, uint16_t& port, std::string& error);

}  // namespace fake_gpu::distributed
