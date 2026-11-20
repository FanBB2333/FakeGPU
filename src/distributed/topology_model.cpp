#include "topology_model.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <utility>

namespace fake_gpu::distributed {

namespace {

double transfer_time_us(std::uint64_t bytes, const FabricLinkConfig& fabric) {
    const double transfer_component_us =
        (static_cast<double>(bytes) * 8.0 * fabric.oversubscription) /
        (fabric.bandwidth_gbps * 1000.0);
    return fabric.latency_us + transfer_component_us;
}

std::string make_link_key(
    const std::string& src_node,
    const std::string& dst_node,
    TopologyLinkScope scope) {
    return src_node + "\n" + dst_node + "\n" + topology_link_scope_name(scope);
}

}  // namespace

const char* topology_link_scope_name(TopologyLinkScope scope) {
    switch (scope) {
        case TopologyLinkScope::IntraNode:
            return "intra_node";
        case TopologyLinkScope::InterNode:
            return "inter_node";
    }
    return "unknown";
}

bool TopologyModel::build(const ClusterConfigModel& config, TopologyModel& out, std::string& error) {
    error.clear();
    if (!config.loaded()) {
        error = "cluster config must be loaded before building topology model";
        return false;
    }
    if (config.world_size == 0) {
        error = "cluster config world_size must be > 0";
        return false;
    }

    TopologyModel built;
    built.intra_node_fabric_ = config.intra_node_fabric;
    built.inter_node_fabric_ = config.inter_node_fabric;
    for (const ClusterNodeConfig& node : config.nodes) {
        for (int rank : node.ranks) {
            built.node_by_rank_.emplace(rank, node.id);
            built.ordered_ranks_.push_back(rank);
        }
    }
    std::sort(built.ordered_ranks_.begin(), built.ordered_ranks_.end());

    out = std::move(built);
    return true;
}

bool TopologyModel::estimate_transfer(
    int src_rank,
    int dst_rank,
    std::uint64_t bytes,
    TopologyLinkEstimate& out,
    std::string& error) const {
    error.clear();
    const auto src_it = node_by_rank_.find(src_rank);
    const auto dst_it = node_by_rank_.find(dst_rank);
    if (src_it == node_by_rank_.end() || dst_it == node_by_rank_.end()) {
        error = "rank is not part of the topology";
        return false;
    }

    out = TopologyLinkEstimate{};
    out.src_node = src_it->second;
    out.dst_node = dst_it->second;
    out.bytes = bytes;
    out.hop_count = 1;

    const bool same_node = src_it->second == dst_it->second;
    const FabricLinkConfig& fabric =
        same_node ? intra_node_fabric_ : inter_node_fabric_;

    out.scope = same_node ? TopologyLinkScope::IntraNode : TopologyLinkScope::InterNode;
    out.bandwidth_gbps = fabric.bandwidth_gbps;
    out.latency_us = fabric.latency_us;
    out.oversubscription = fabric.oversubscription;
    out.estimated_time_us = transfer_time_us(bytes, fabric);
    return true;
}

bool TopologyModel::estimate_ring_collective(
    CollectiveType type,
    std::uint64_t bytes_per_rank,
    CollectiveTopologyEstimate& out,
    std::string& error) const {
    return estimate_ring_collective(
        type,
        bytes_per_rank,
        ordered_ranks_,
        out,
        error);
}

bool TopologyModel::estimate_ring_collective(
    CollectiveType type,
    std::uint64_t bytes_per_rank,
    const std::vector<int>& participant_ranks,
    CollectiveTopologyEstimate& out,
    std::string& error) const {
    error.clear();
    out = CollectiveTopologyEstimate{};

    for (int rank : participant_ranks) {
        if (node_by_rank_.find(rank) == node_by_rank_.end()) {
            error = "participant rank is not part of the topology: " +
                    std::to_string(rank);
            return false;
        }
    }

    if (participant_ranks.size() < 2) {
        return true;
    }

    const double multiplier = type == CollectiveType::AllReduce ? 2.0 : 1.0;
    const std::uint64_t edge_bytes = static_cast<std::uint64_t>(
        std::llround(static_cast<double>(bytes_per_rank) * multiplier));

    std::unordered_map<std::string, TopologyLinkEstimate> aggregates;

    for (std::size_t index = 0; index < participant_ranks.size(); ++index) {
        const int src_rank = participant_ranks[index];
        const int dst_rank =
            participant_ranks[(index + 1) % participant_ranks.size()];

        TopologyLinkEstimate edge;
        if (!estimate_transfer(src_rank, dst_rank, edge_bytes, edge, error)) {
            return false;
        }

        const std::string key = make_link_key(edge.src_node, edge.dst_node, edge.scope);
        auto it = aggregates.find(key);
        if (it == aggregates.end()) {
            aggregates.emplace(key, edge);
        } else {
            it->second.hop_count += edge.hop_count;
            it->second.bytes += edge.bytes;
            it->second.estimated_time_us += edge.estimated_time_us;
        }
    }

    out.links.reserve(aggregates.size());
    for (auto& entry : aggregates) {
        out.links.push_back(std::move(entry.second));
    }
    std::sort(
        out.links.begin(),
        out.links.end(),
        [](const TopologyLinkEstimate& lhs, const TopologyLinkEstimate& rhs) {
            if (lhs.src_node != rhs.src_node) {
                return lhs.src_node < rhs.src_node;
            }
            if (lhs.dst_node != rhs.dst_node) {
                return lhs.dst_node < rhs.dst_node;
            }
            return std::string(topology_link_scope_name(lhs.scope)) <
                   topology_link_scope_name(rhs.scope);
        });
    return true;
}

}  // namespace fake_gpu::distributed
