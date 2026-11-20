#pragma once

#include "cluster_config.hpp"
#include "collective_executor.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace fake_gpu::distributed {

enum class TopologyLinkScope {
    IntraNode,
    InterNode,
};

struct TopologyLinkEstimate {
    std::string src_node;
    std::string dst_node;
    TopologyLinkScope scope = TopologyLinkScope::IntraNode;
    std::uint64_t hop_count = 0;
    std::uint64_t bytes = 0;
    double bandwidth_gbps = 0.0;
    double latency_us = 0.0;
    double oversubscription = 1.0;
    double estimated_time_us = 0.0;
};

struct CollectiveTopologyEstimate {
    std::vector<TopologyLinkEstimate> links;
};

class TopologyModel {
public:
    static bool build(const ClusterConfigModel& config, TopologyModel& out, std::string& error);

    bool estimate_transfer(
        int src_rank,
        int dst_rank,
        std::uint64_t bytes,
        TopologyLinkEstimate& out,
        std::string& error) const;

    bool estimate_ring_collective(
        CollectiveType type,
        std::uint64_t bytes_per_rank,
        CollectiveTopologyEstimate& out,
        std::string& error) const;

    bool estimate_ring_collective(
        CollectiveType type,
        std::uint64_t bytes_per_rank,
        const std::vector<int>& participant_ranks,
        CollectiveTopologyEstimate& out,
        std::string& error) const;

private:
    FabricLinkConfig intra_node_fabric_;
    FabricLinkConfig inter_node_fabric_;
    std::unordered_map<int, std::string> node_by_rank_;
    std::vector<int> ordered_ranks_;
};

const char* topology_link_scope_name(TopologyLinkScope scope);

}  // namespace fake_gpu::distributed
