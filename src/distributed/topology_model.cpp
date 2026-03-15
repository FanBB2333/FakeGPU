#include "topology_model.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
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

double traffic_multiplier_for_collective(CollectiveType type) {
    switch (type) {
        case CollectiveType::AllReduce:
            return 2.0;
        case CollectiveType::Reduce:
            return 1.0;
        case CollectiveType::Broadcast:
            return 1.0;
        case CollectiveType::AllGather:
            return 1.0;
        case CollectiveType::ReduceScatter:
            return 1.0;
    }
    return 1.0;
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
    built.config_ = config;
    built.world_size_ = config.world_size;
    built.node_count_ = config.nodes.size();

    for (const ClusterNodeConfig& node : config.nodes) {
        for (int rank : node.ranks) {
            if (!built.node_by_rank_.emplace(rank, node.id).second) {
                error = "duplicate rank in topology model: " + std::to_string(rank);
                return false;
            }
            built.ordered_ranks_.push_back(rank);
        }
    }

    std::sort(built.ordered_ranks_.begin(), built.ordered_ranks_.end());
    if (built.ordered_ranks_.size() != built.world_size_) {
        error = "topology model rank count does not match world_size";
        return false;
    }

    for (std::size_t index = 0; index < built.ordered_ranks_.size(); ++index) {
        if (built.ordered_ranks_[index] != static_cast<int>(index)) {
            error = "topology model requires contiguous ranks from 0 to world_size - 1";
            return false;
        }
    }

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
    if (!valid()) {
        error = "topology model is not initialized";
        return false;
    }
    if (bytes == 0) {
        error = "bytes must be > 0";
        return false;
    }

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
        same_node ? config_.intra_node_fabric : config_.inter_node_fabric;

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
    error.clear();
    out = CollectiveTopologyEstimate{};
    out.type = type;
    out.algorithm = "ring";
    out.world_size = world_size_;
    out.bytes_per_rank = bytes_per_rank;

    if (!valid()) {
        error = "topology model is not initialized";
        out.error = error;
        return false;
    }
    if (bytes_per_rank == 0) {
        error = "bytes_per_rank must be > 0";
        out.error = error;
        return false;
    }
    if (world_size_ < 2) {
        out.ok = true;
        return true;
    }

    const double multiplier = traffic_multiplier_for_collective(type);
    const std::uint64_t edge_bytes = static_cast<std::uint64_t>(
        std::llround(static_cast<double>(bytes_per_rank) * multiplier));
    if (edge_bytes == 0) {
        error = "edge_bytes must be > 0";
        out.error = error;
        return false;
    }

    std::unordered_map<std::string, TopologyLinkEstimate> aggregates;
    double total_time_us = 0.0;

    for (std::size_t index = 0; index < ordered_ranks_.size(); ++index) {
        const int src_rank = ordered_ranks_[index];
        const int dst_rank = ordered_ranks_[(index + 1) % ordered_ranks_.size()];

        TopologyLinkEstimate edge;
        if (!estimate_transfer(src_rank, dst_rank, edge_bytes, edge, error)) {
            out.error = error;
            return false;
        }
        total_time_us += edge.estimated_time_us;

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

    out.estimated_time_us = total_time_us;
    out.ok = true;
    return true;
}

}  // namespace fake_gpu::distributed
