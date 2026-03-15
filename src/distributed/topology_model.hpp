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
    bool ok = false;
    std::string error;
    CollectiveType type = CollectiveType::AllReduce;
    std::string algorithm;
    std::size_t world_size = 0;
    std::uint64_t bytes_per_rank = 0;
    double estimated_time_us = 0.0;
    std::vector<TopologyLinkEstimate> links;
};

class TopologyModel {
public:
    static bool build(const ClusterConfigModel& config, TopologyModel& out, std::string& error);

    bool valid() const { return world_size_ > 0 && ordered_ranks_.size() == world_size_; }
    std::size_t world_size() const { return world_size_; }
    std::size_t node_count() const { return node_count_; }
    const ClusterConfigModel& cluster_config() const { return config_; }

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

private:
    ClusterConfigModel config_;
    std::unordered_map<int, std::string> node_by_rank_;
    std::vector<int> ordered_ranks_;
    std::size_t world_size_ = 0;
    std::size_t node_count_ = 0;
};

const char* topology_link_scope_name(TopologyLinkScope scope);

}  // namespace fake_gpu::distributed
