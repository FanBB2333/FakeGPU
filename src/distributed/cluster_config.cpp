#include "cluster_config.hpp"

#include "../core/gpu_profile.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace fake_gpu::distributed {

namespace {

std::string trim(const std::string& value) {
    const std::size_t begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const std::size_t end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

std::string strip_comment(std::string line) {
    const std::size_t hash = line.find('#');
    if (hash == std::string::npos) {
        return line;
    }
    return line.substr(0, hash);
}

std::string env_or_empty(const char* name) {
    const char* value = std::getenv(name);
    if (!value) {
        return "";
    }
    return value;
}

bool is_digits(std::string_view value) {
    if (value.empty()) {
        return false;
    }
    for (char ch : value) {
        if (ch < '0' || ch > '9') {
            return false;
        }
    }
    return true;
}

bool split_key_value(const std::string& text, std::string& key, std::string& value) {
    const std::size_t colon = text.find(':');
    if (colon == std::string::npos) {
        return false;
    }
    key = trim(text.substr(0, colon));
    value = trim(text.substr(colon + 1));
    return !key.empty();
}

bool parse_int(const std::string& text, int& out) {
    try {
        std::size_t consumed = 0;
        out = std::stoi(text, &consumed, 10);
        return consumed == text.size();
    } catch (...) {
        return false;
    }
}

bool parse_int_list(const std::string& text, std::vector<int>& out, std::string& error) {
    out.clear();
    if (text.size() < 2 || text.front() != '[' || text.back() != ']') {
        error = "expected [a, b, c] list";
        return false;
    }

    const std::string inner = trim(text.substr(1, text.size() - 2));
    if (inner.empty()) {
        return true;
    }

    std::stringstream stream(inner);
    std::string item;
    while (std::getline(stream, item, ',')) {
        item = trim(item);
        if (item.empty()) {
            error = "empty rank entry";
            return false;
        }
        int rank = 0;
        if (!parse_int(item, rank)) {
            error = "rank must be an integer";
            return false;
        }
        if (rank < 0) {
            error = "rank must be >= 0";
            return false;
        }
        out.push_back(rank);
    }
    return true;
}

bool parse_double(const std::string& text, double& out) {
    try {
        std::size_t consumed = 0;
        out = std::stod(text, &consumed);
        return consumed == text.size() && std::isfinite(out);
    } catch (...) {
        return false;
    }
}

bool parse_size_value(const std::string& text, std::size_t& out) {
    if (!is_digits(text)) {
        return false;
    }
    try {
        out = static_cast<std::size_t>(std::stoull(text, nullptr, 10));
        return true;
    } catch (...) {
        return false;
    }
}

bool assign_fabric_field(FabricLinkConfig& fabric, const std::string& key, const std::string& value, std::string& error) {
    if (key == "type") {
        fabric.type = value;
        fabric.defined = true;
        return true;
    }
    if (key == "bandwidth_gbps") {
        if (!parse_double(value, fabric.bandwidth_gbps) || fabric.bandwidth_gbps <= 0.0) {
            error = key + " must be a positive number";
            return false;
        }
        fabric.defined = true;
        return true;
    }
    if (key == "latency_us") {
        if (!parse_double(value, fabric.latency_us) || fabric.latency_us < 0.0) {
            error = key + " must be a non-negative number";
            return false;
        }
        fabric.defined = true;
        return true;
    }
    if (key == "oversubscription") {
        if (!parse_double(value, fabric.oversubscription) || fabric.oversubscription <= 0.0) {
            error = key + " must be a positive number";
            return false;
        }
        fabric.defined = true;
        return true;
    }
    error = "unsupported fabric key: " + key;
    return false;
}

bool validate_cluster_config_model(ClusterConfigModel& model, std::string& error) {
    if (model.version <= 0) {
        error = "cluster config must define a positive version";
        return false;
    }
    if (model.name.empty()) {
        error = "cluster.name must be set";
        return false;
    }
    if (model.default_backend.empty()) {
        error = "cluster.default_backend must be set";
        return false;
    }
    if (model.nodes.empty()) {
        error = "cluster config must contain at least one node";
        return false;
    }
    if (!model.intra_node_fabric.defined || !model.inter_node_fabric.defined) {
        error = "fabric.intra_node and fabric.inter_node must both be defined";
        return false;
    }
    if (model.intra_node_fabric.type.empty() || model.inter_node_fabric.type.empty()) {
        error = "fabric.intra_node.type and fabric.inter_node.type must both be set";
        return false;
    }
    if (model.intra_node_fabric.bandwidth_gbps <= 0.0 || model.inter_node_fabric.bandwidth_gbps <= 0.0) {
        error = "fabric bandwidth_gbps must be positive for both intra_node and inter_node";
        return false;
    }

    std::unordered_set<std::string> node_ids;
    std::unordered_set<int> seen_ranks;
    int max_rank = -1;

    for (const ClusterNodeConfig& node : model.nodes) {
        if (node.id.empty()) {
            error = "each node must define a non-empty id";
            return false;
        }
        if (!node_ids.insert(node.id).second) {
            error = "duplicate node id: " + node.id;
            return false;
        }
        if (node.host.empty()) {
            error = "node " + node.id + " must define host";
            return false;
        }
        if (node.ranks.empty()) {
            error = "node " + node.id + " must define at least one rank";
            return false;
        }
        if (node.gpu_profiles.size() != node.ranks.size()) {
            error = "node " + node.id + " must define one gpu profile per rank";
            return false;
        }

        for (const std::string& profile : node.gpu_profiles) {
            if (!fake_gpu::profile_from_preset_id(profile).has_value()) {
                error = "node " + node.id + " references unknown gpu profile: " + profile;
                return false;
            }
        }

        for (int rank : node.ranks) {
            if (!seen_ranks.insert(rank).second) {
                error = "duplicate rank detected: " + std::to_string(rank);
                return false;
            }
            max_rank = std::max(max_rank, rank);
        }
    }

    const std::size_t expected_world_size = static_cast<std::size_t>(max_rank + 1);
    if (seen_ranks.size() != expected_world_size) {
        error = "ranks must be contiguous from 0 to " + std::to_string(max_rank);
        return false;
    }

    model.world_size = expected_world_size;
    return true;
}

bool parse_cluster_yaml(std::istream& input, ClusterConfigModel& out, std::string& error) {
    enum class TopLevelSection {
        None,
        Cluster,
        Nodes,
        Fabric,
        Policies,
    };
    enum class FabricSection {
        None,
        IntraNode,
        InterNode,
    };

    TopLevelSection top_level = TopLevelSection::None;
    FabricSection fabric_section = FabricSection::None;
    std::optional<ClusterNodeConfig> current_node;
    bool node_gpus_active = false;
    std::string line;
    int line_no = 0;
    auto line_error = [&](const std::string& message) {
        error = "line " + std::to_string(line_no) + ": " + message;
        return false;
    };

    auto flush_node = [&]() {
        if (current_node.has_value()) {
            out.nodes.push_back(std::move(*current_node));
            current_node.reset();
            node_gpus_active = false;
        }
    };

    while (std::getline(input, line)) {
        ++line_no;
        line = strip_comment(std::move(line));
        const std::size_t non_space = line.find_first_not_of(' ');
        if (non_space == std::string::npos) {
            continue;
        }

        const int indent = static_cast<int>(non_space);
        std::string trimmed = trim(line);
        if (trimmed.empty()) {
            continue;
        }

        if (indent == 0) {
            flush_node();
            fabric_section = FabricSection::None;

            if (trimmed == "cluster:") {
                top_level = TopLevelSection::Cluster;
                continue;
            }
            if (trimmed == "nodes:") {
                top_level = TopLevelSection::Nodes;
                continue;
            }
            if (trimmed == "fabric:") {
                top_level = TopLevelSection::Fabric;
                continue;
            }
            if (trimmed == "policies:") {
                top_level = TopLevelSection::Policies;
                continue;
            }

            std::string key;
            std::string value;
            if (!split_key_value(trimmed, key, value)) {
                return line_error("invalid top-level entry");
            }
            if (key == "version") {
                if (!parse_int(value, out.version)) {
                    return line_error("version must be an integer");
                }
                continue;
            }

            return line_error("unsupported top-level key " + key);
        }

        if (top_level == TopLevelSection::Cluster) {
            if (indent != 2) {
                return line_error("cluster fields must be indented by 2 spaces");
            }
            std::string key;
            std::string value;
            if (!split_key_value(trimmed, key, value)) {
                return line_error("invalid cluster field");
            }
            if (key == "name") {
                out.name = value;
            } else if (key == "default_backend") {
                out.default_backend = value;
            } else {
                return line_error("unsupported cluster field " + key);
            }
            continue;
        }

        if (top_level == TopLevelSection::Policies) {
            continue;
        }

        if (top_level == TopLevelSection::Nodes) {
            if (indent == 2) {
                flush_node();
                if (trimmed.rfind("- ", 0) != 0) {
                    return line_error("node entries must begin with '- '");
                }
                std::string key;
                std::string value;
                if (!split_key_value(trim(trimmed.substr(2)), key, value) || key != "id") {
                    return line_error("node entries must start with '- id:'");
                }
                current_node = ClusterNodeConfig{};
                current_node->id = value;
                node_gpus_active = false;
                continue;
            }

            if (!current_node.has_value()) {
                return line_error("node field encountered before any node declaration");
            }

            if (indent == 4 && trimmed == "gpus:") {
                node_gpus_active = true;
                continue;
            }

            if (indent == 6) {
                if (!node_gpus_active || trimmed.rfind("- ", 0) != 0) {
                    return line_error("unsupported node sub-entry");
                }
                std::string key;
                std::string value;
                if (!split_key_value(trim(trimmed.substr(2)), key, value) || key != "profile") {
                    return line_error("gpu entries must be '- profile: ...'");
                }
                current_node->gpu_profiles.push_back(value);
                continue;
            }

            if (indent != 4) {
                return line_error("unsupported node indentation");
            }

            std::string key;
            std::string value;
            if (!split_key_value(trimmed, key, value)) {
                return line_error("invalid node field");
            }
            node_gpus_active = false;
            if (key == "host") {
                current_node->host = value;
            } else if (key == "ranks") {
                if (!parse_int_list(value, current_node->ranks, error)) {
                    return line_error(error);
                }
            } else {
                return line_error("unsupported node field " + key);
            }
            continue;
        }

        if (top_level == TopLevelSection::Fabric) {
            if (indent == 2) {
                if (trimmed == "intra_node:") {
                    fabric_section = FabricSection::IntraNode;
                    continue;
                }
                if (trimmed == "inter_node:") {
                    fabric_section = FabricSection::InterNode;
                    continue;
                }
                return line_error("unsupported fabric section");
            }

            if (indent != 4 || fabric_section == FabricSection::None) {
                return line_error("invalid fabric field indentation");
            }

            std::string key;
            std::string value;
            if (!split_key_value(trimmed, key, value)) {
                return line_error("invalid fabric field");
            }

            FabricLinkConfig* target =
                (fabric_section == FabricSection::IntraNode) ? &out.intra_node_fabric : &out.inter_node_fabric;
            if (!assign_fabric_field(*target, key, value, error)) {
                return line_error(error);
            }
            continue;
        }

        return line_error("unexpected content");
    }

    flush_node();
    return true;
}

}  // namespace

const char* distributed_mode_name(DistributedMode mode) {
    switch (mode) {
        case DistributedMode::Disabled:
            return "disabled";
        case DistributedMode::Simulate:
            return "simulate";
        case DistributedMode::Proxy:
            return "proxy";
        case DistributedMode::Passthrough:
            return "passthrough";
    }
    return "unknown";
}

const char* coordinator_transport_name(CoordinatorTransport transport) {
    switch (transport) {
        case CoordinatorTransport::Tcp:
            return "tcp";
        case CoordinatorTransport::Unix:
            return "unix";
    }
    return "unknown";
}

bool parse_tcp_endpoint(const std::string& endpoint, std::string& host, uint16_t& port, std::string& error) {
    const std::size_t colon = endpoint.rfind(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 >= endpoint.size()) {
        error = "expected host:port";
        return false;
    }

    std::string host_part = endpoint.substr(0, colon);
    std::string_view port_part(endpoint.data() + colon + 1, endpoint.size() - colon - 1);
    if (!is_digits(port_part)) {
        error = "port must be numeric";
        return false;
    }

    unsigned long parsed_port = std::strtoul(std::string(port_part).c_str(), nullptr, 10);
    if (parsed_port == 0 || parsed_port > std::numeric_limits<uint16_t>::max()) {
        error = "port must be within 1..65535";
        return false;
    }

    host = std::move(host_part);
    port = static_cast<uint16_t>(parsed_port);
    error.clear();
    return true;
}

DistributedConfig parse_distributed_config_from_env() {
    DistributedConfig config;

    const std::string raw_mode = env_or_empty("FAKEGPU_DIST_MODE");
    if (!raw_mode.empty()) {
        if (raw_mode == "disabled") {
            config.mode = DistributedMode::Disabled;
        } else if (raw_mode == "simulate") {
            config.mode = DistributedMode::Simulate;
        } else if (raw_mode == "proxy") {
            config.mode = DistributedMode::Proxy;
        } else if (raw_mode == "passthrough") {
            config.mode = DistributedMode::Passthrough;
        } else {
            config.validation_error =
                "Invalid FAKEGPU_DIST_MODE: " + raw_mode + ". Expected one of: disabled, simulate, proxy, passthrough.";
            return config;
        }
    }

    const std::string raw_transport = env_or_empty("FAKEGPU_COORDINATOR_TRANSPORT");
    if (!raw_transport.empty()) {
        if (raw_transport == "tcp") {
            config.coordinator_transport = CoordinatorTransport::Tcp;
        } else if (raw_transport == "unix") {
            config.coordinator_transport = CoordinatorTransport::Unix;
        } else {
            config.validation_error =
                "Invalid FAKEGPU_COORDINATOR_TRANSPORT: " + raw_transport + ". Expected one of: tcp, unix.";
            return config;
        }
    }

    config.coordinator_address = env_or_empty("FAKEGPU_COORDINATOR_ADDR");

    if (!config.enabled()) {
        return config;
    }

    const std::string raw_staging_chunk_bytes = env_or_empty("FAKEGPU_STAGING_CHUNK_BYTES");
    if (!raw_staging_chunk_bytes.empty()) {
        if (!parse_size_value(raw_staging_chunk_bytes, config.staging_chunk_bytes) ||
            config.staging_chunk_bytes == 0) {
            config.validation_error =
                "Invalid FAKEGPU_STAGING_CHUNK_BYTES: " + raw_staging_chunk_bytes +
                ". Expected a positive integer.";
            return config;
        }
    }

    if (config.mode != DistributedMode::Passthrough || !config.coordinator_address.empty()) {
        if (config.coordinator_address.empty()) {
            config.validation_error =
                "FAKEGPU_COORDINATOR_ADDR must be set when FAKEGPU_DIST_MODE requires a coordinator.";
            return config;
        }

        if (config.coordinator_transport == CoordinatorTransport::Tcp) {
            std::string host;
            uint16_t port = 0;
            std::string error;
            if (!parse_tcp_endpoint(config.coordinator_address, host, port, error)) {
                config.validation_error =
                    "Invalid FAKEGPU_COORDINATOR_ADDR: " + config.coordinator_address + " (" + error + ").";
                return config;
            }
        } else if (config.coordinator_address.front() != '/') {
            config.validation_error =
                "FAKEGPU_COORDINATOR_ADDR must be an absolute Unix socket path when FAKEGPU_COORDINATOR_TRANSPORT=unix.";
            return config;
        }
    }

    const std::string cluster_config_path = env_or_empty("FAKEGPU_CLUSTER_CONFIG");
    if (!cluster_config_path.empty()) {
        load_cluster_config_from_yaml_file(cluster_config_path, config.cluster_config, config.validation_error);
    }
    return config;
}

bool load_cluster_config_from_yaml_file(const std::string& path, ClusterConfigModel& out, std::string& error) {
    std::ifstream input(path);
    if (!input.is_open()) {
        error = "Failed to open cluster config: " + path;
        return false;
    }

    ClusterConfigModel parsed;
    parsed.source_path = path;
    if (!parse_cluster_yaml(input, parsed, error)) {
        error = path + ": " + error;
        return false;
    }
    if (!validate_cluster_config_model(parsed, error)) {
        error = path + ": " + error;
        return false;
    }

    out = std::move(parsed);
    return true;
}

}  // namespace fake_gpu::distributed
