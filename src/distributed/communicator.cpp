#include "communicator.hpp"

#include "../core/backend_config.hpp"
#include "buffer_transfer.hpp"
#include "topology_model.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fake_gpu::distributed {

namespace {

using SteadyTimePoint = std::chrono::steady_clock::time_point;

struct CollectiveState {
    CollectiveSubmitRequest request;
    bool completed = false;
    bool failed = false;
    std::string failure_code;
    std::string failure_detail;
    std::unordered_map<int, CollectiveExecutionParticipant> participants;
    std::unordered_map<int, CollectiveSubmitResult> results;
    SteadyTimePoint first_submit_at;
    SteadyTimePoint all_participants_at;
    std::condition_variable cv;
};

struct BarrierState {
    BarrierSubmitRequest request;
    bool completed = false;
    bool failed = false;
    std::string failure_code;
    std::string failure_detail;
    std::unordered_map<int, bool> participants;
    SteadyTimePoint first_submit_at;
    SteadyTimePoint all_participants_at;
    std::condition_variable cv;
};

struct BatchState {
    CollectiveBatchPrepareRequest request;
    bool completed = false;
    bool failed = false;
    std::string failure_code;
    std::string failure_detail;
    std::unordered_map<int, CollectiveBatchPrepareRequest> participants;
    std::condition_variable cv;
};

struct SplitParticipantResult {
    bool participating = false;
    int new_comm_id = -1;
    int new_rank = -1;
    int new_world_size = 0;
};

struct SplitState {
    CommunicatorSplitRequest request;
    bool completed = false;
    bool failed = false;
    std::string failure_code;
    std::string failure_detail;
    std::unordered_map<int, CommunicatorSplitRequest> participants;
    std::unordered_map<int, SplitParticipantResult> results;
    std::condition_variable cv;
};

struct ShrinkParticipantResult {
    int new_comm_id = -1;
    int new_rank = -1;
    int new_world_size = 0;
};

struct ShrinkState {
    CommunicatorShrinkRequest request;
    bool completed = false;
    bool failed = false;
    std::string failure_code;
    std::string failure_detail;
    std::unordered_map<int, CommunicatorShrinkRequest> participants;
    std::unordered_map<int, ShrinkParticipantResult> results;
    SteadyTimePoint first_submit_at;
    std::condition_variable cv;
};

struct PointToPointState {
    PointToPointSubmitRequest request;
    bool completed = false;
    bool failed = false;
    std::string failure_code;
    std::string failure_detail;
    std::unordered_map<int, PointToPointSubmitRequest> participants;
    std::unordered_map<int, PointToPointSubmitResult> results;
    SteadyTimePoint first_submit_at;
    SteadyTimePoint all_participants_at;
    std::condition_variable cv;
};

struct CommunicatorState {
    std::string unique_id;
    int world_size = 0;
    int comm_id = -1;
    bool ready = false;
    bool failed = false;
    std::string failure_code;
    std::string failure_detail;
    std::uint64_t next_seqno = 1;
    std::vector<int> global_ranks;
    std::unordered_map<int, bool> participants;
    std::unordered_map<int, bool> destroyed_ranks;
    std::unordered_map<std::uint64_t, std::shared_ptr<CollectiveState>> collectives;
    std::unordered_map<std::uint64_t, std::shared_ptr<BarrierState>> barriers;
    std::unordered_map<std::uint64_t, std::shared_ptr<BatchState>> batches;
    std::unordered_map<std::uint64_t, std::shared_ptr<SplitState>> splits;
    std::unordered_map<std::uint64_t, std::shared_ptr<ShrinkState>> shrinks;
    std::unordered_map<std::uint64_t, std::shared_ptr<PointToPointState>> point_to_points;
    SteadyTimePoint failed_at;
    std::condition_variable cv;
};

struct RegistryImpl {
    std::mutex mutex;
    int next_comm_id = 1;
    std::unordered_map<std::string, std::shared_ptr<CommunicatorState>> pending_by_unique_id;
    std::unordered_map<int, std::shared_ptr<CommunicatorState>> active_by_comm_id;
    struct ClusterReportState {
        std::size_t world_size = 0;
        std::size_t communicator_count = 0;
        ClusterCollectiveReportStats all_reduce;
        ClusterCollectiveReportStats reduce;
        ClusterCollectiveReportStats broadcast;
        ClusterCollectiveReportStats all_gather;
        ClusterCollectiveReportStats reduce_scatter;
        ClusterCollectiveReportStats all_to_all;
        ClusterCollectiveReportStats barrier;
        ClusterPointToPointReportStats point_to_point;
        std::unordered_map<std::string, ClusterLinkReportStats> links;
        std::unordered_map<std::string, ClusterNodePairReportStats> node_pairs;
        std::unordered_map<int, ClusterRankReportStats> ranks;
        std::deque<ClusterOperationTimelineEntry> operation_timeline;
        std::uint64_t operation_timeline_entries = 0;
        std::uint64_t dropped_operation_timeline_entries = 0;
        std::vector<ClusterFailureEvent> failure_events;
        std::vector<ClusterRecoveryEvent> recovery_events;
        std::uint64_t resilience_event_entries = 0;
    } report;
};

RegistryImpl& registry_impl() {
    static RegistryImpl instance;
    return instance;
}

CommunicatorRegistrationResult make_error(std::string code, std::string detail) {
    CommunicatorRegistrationResult result;
    result.ok = false;
    result.error_code = std::move(code);
    result.error_detail = std::move(detail);
    return result;
}

CommunicatorDestroyResult make_destroy_error(std::string code, std::string detail) {
    CommunicatorDestroyResult result;
    result.ok = false;
    result.error_code = std::move(code);
    result.error_detail = std::move(detail);
    return result;
}

CommunicatorSplitResult make_split_error(std::string code, std::string detail) {
    CommunicatorSplitResult result;
    result.ok = false;
    result.error_code = std::move(code);
    result.error_detail = std::move(detail);
    return result;
}

CommunicatorFailureResult make_failure_error(
    std::string code,
    std::string detail) {
    CommunicatorFailureResult result;
    result.ok = false;
    result.error_code = std::move(code);
    result.error_detail = std::move(detail);
    return result;
}

CommunicatorShrinkResult make_shrink_error(
    std::string code,
    std::string detail) {
    CommunicatorShrinkResult result;
    result.ok = false;
    result.error_code = std::move(code);
    result.error_detail = std::move(detail);
    return result;
}

PointToPointSubmitResult make_point_to_point_error(std::string code, std::string detail) {
    PointToPointSubmitResult result;
    result.ok = false;
    result.error_code = std::move(code);
    result.error_detail = std::move(detail);
    return result;
}

CollectiveSubmitResult make_collective_error(std::string code, std::string detail) {
    CollectiveSubmitResult result;
    result.ok = false;
    result.error_code = std::move(code);
    result.error_detail = std::move(detail);
    return result;
}

BarrierSubmitResult make_barrier_error(std::string code, std::string detail) {
    BarrierSubmitResult result;
    result.ok = false;
    result.error_code = std::move(code);
    result.error_detail = std::move(detail);
    return result;
}

CollectiveBatchPrepareResult make_batch_error(std::string code, std::string detail) {
    CollectiveBatchPrepareResult result;
    result.ok = false;
    result.error_code = std::move(code);
    result.error_detail = std::move(detail);
    return result;
}

ClusterRankReportStats& ensure_rank_report_locked(RegistryImpl& registry, int rank) {
    auto [it, inserted] = registry.report.ranks.emplace(rank, ClusterRankReportStats{});
    if (inserted) {
        it->second.rank = rank;
    }
    return it->second;
}

int global_rank_for_local(
    const std::shared_ptr<CommunicatorState>& state,
    int local_rank) {
    if (state &&
        local_rank >= 0 &&
        static_cast<std::size_t>(local_rank) < state->global_ranks.size()) {
        return state->global_ranks[static_cast<std::size_t>(local_rank)];
    }
    return local_rank;
}

std::vector<int> communicator_global_ranks(
    const std::shared_ptr<CommunicatorState>& state) {
    std::vector<int> ranks;
    if (!state || state->world_size <= 0) {
        return ranks;
    }
    ranks.reserve(static_cast<std::size_t>(state->world_size));
    for (int local_rank = 0; local_rank < state->world_size; ++local_rank) {
        ranks.push_back(global_rank_for_local(state, local_rank));
    }
    return ranks;
}

std::size_t operation_timeline_limit() {
    static const std::size_t limit = []() {
        constexpr std::size_t kDefaultLimit = 4096;
        constexpr std::size_t kMaximumLimit = 1000000;
        const char* configured =
            std::getenv("FAKEGPU_CLUSTER_REPORT_MAX_OPERATIONS");
        if (!configured || !*configured) {
            return kDefaultLimit;
        }
        try {
            std::size_t consumed = 0;
            const unsigned long long parsed =
                std::stoull(configured, &consumed, 10);
            if (consumed != std::string(configured).size() ||
                parsed > kMaximumLimit) {
                return kDefaultLimit;
            }
            return static_cast<std::size_t>(parsed);
        } catch (...) {
            return kDefaultLimit;
        }
    }();
    return limit;
}

double elapsed_us(SteadyTimePoint begin, SteadyTimePoint end) {
    if (begin == SteadyTimePoint{} ||
        end == SteadyTimePoint{} ||
        end < begin) {
        return 0.0;
    }
    return std::chrono::duration<double, std::micro>(end - begin).count();
}

const char* buffer_transport_name(BufferTransport transport) {
    return transport == BufferTransport::SocketPayload
        ? "socket_payload"
        : "shared_memory";
}

void append_operation_timeline_locked(
    RegistryImpl& registry,
    const std::shared_ptr<CommunicatorState>& state,
    std::uint64_t seqno,
    std::string kind,
    std::string operation,
    std::string data_type,
    std::string reduce_op,
    std::string buffer_transport,
    std::uint64_t logical_payload_bytes,
    std::uint64_t socket_request_payload_bytes,
    std::uint64_t socket_response_payload_bytes,
    SteadyTimePoint first_submit_at,
    SteadyTimePoint all_participants_at,
    SteadyTimePoint completed_at,
    double modeled_time_us) {
    ClusterOperationTimelineEntry entry;
    entry.index = ++registry.report.operation_timeline_entries;
    entry.comm_id = state ? state->comm_id : -1;
    entry.seqno = seqno;
    entry.kind = std::move(kind);
    entry.operation = std::move(operation);
    entry.data_type = std::move(data_type);
    entry.reduce_op = std::move(reduce_op);
    entry.buffer_transport = std::move(buffer_transport);
    entry.ranks = communicator_global_ranks(state);
    entry.logical_payload_bytes = logical_payload_bytes;
    entry.socket_request_payload_bytes = socket_request_payload_bytes;
    entry.socket_response_payload_bytes = socket_response_payload_bytes;
    entry.rendezvous_wait_us =
        elapsed_us(first_submit_at, all_participants_at);
    entry.execution_time_us =
        elapsed_us(all_participants_at, completed_at);
    entry.coordinator_duration_us =
        elapsed_us(first_submit_at, completed_at);
    entry.modeled_time_us = modeled_time_us;

    const std::size_t limit = operation_timeline_limit();
    if (limit == 0) {
        registry.report.dropped_operation_timeline_entries++;
        return;
    }
    if (registry.report.operation_timeline.size() >= limit) {
        registry.report.operation_timeline.pop_front();
        registry.report.dropped_operation_timeline_entries++;
    }
    registry.report.operation_timeline.push_back(std::move(entry));
}

template <typename PayloadMap>
std::uint64_t total_payload_bytes(const PayloadMap& payloads) {
    std::uint64_t bytes = 0;
    for (const auto& entry : payloads) {
        bytes += static_cast<std::uint64_t>(entry.second.size());
    }
    return bytes;
}

template <typename ParticipantMap>
std::uint64_t total_participant_payload_bytes(
    const ParticipantMap& participants) {
    std::uint64_t bytes = 0;
    for (const auto& entry : participants) {
        bytes += static_cast<std::uint64_t>(entry.second.payload.size());
    }
    return bytes;
}

std::uint64_t point_to_point_logical_payload_bytes(
    const std::shared_ptr<PointToPointState>& point_to_point) {
    std::uint64_t bytes = 0;
    for (const auto& entry : point_to_point->participants) {
        if (entry.second.type == PointToPointType::Send) {
            bytes += static_cast<std::uint64_t>(entry.second.bytes);
        }
    }
    return bytes;
}

void remember_world_size_locked(RegistryImpl& registry, int world_size) {
    if (world_size > 0) {
        registry.report.world_size =
            std::max(registry.report.world_size, static_cast<std::size_t>(world_size));
    }
}

void record_wait_time_locked(
    RegistryImpl& registry,
    const std::shared_ptr<CommunicatorState>& state,
    int rank,
    std::chrono::steady_clock::duration elapsed) {
    const int global_rank = global_rank_for_local(state, rank);
    if (global_rank < 0) {
        return;
    }
    ClusterRankReportStats& stats =
        ensure_rank_report_locked(registry, global_rank);
    stats.wait_time_ms += std::chrono::duration<double, std::milli>(elapsed).count();
}

template <typename ParticipantMap>
void record_timeout_locked(
    RegistryImpl& registry,
    const std::shared_ptr<CommunicatorState>& state,
    const ParticipantMap& participants) {
    for (const auto& entry : participants) {
        ClusterRankReportStats& stats = ensure_rank_report_locked(
            registry,
            global_rank_for_local(state, entry.first));
        stats.timeouts++;
    }
}

ClusterCollectiveReportStats& collective_report_stats_for_type_locked(
    RegistryImpl& registry,
    CollectiveType type) {
    switch (type) {
        case CollectiveType::AllReduce:
            return registry.report.all_reduce;
        case CollectiveType::Reduce:
            return registry.report.reduce;
        case CollectiveType::Broadcast:
            return registry.report.broadcast;
        case CollectiveType::AllGather:
            return registry.report.all_gather;
        case CollectiveType::ReduceScatter:
            return registry.report.reduce_scatter;
        case CollectiveType::AllToAll:
            return registry.report.all_to_all;
    }
    return registry.report.all_reduce;
}

enum class CommunicationOperationKind {
    Collective,
    PointToPoint,
};

struct TopologyOperationTotals {
    double estimated_time_us = 0.0;
    double contention_penalty_us = 0.0;
};

TopologyOperationTotals record_topology_operation_locked(
    RegistryImpl& registry,
    const std::vector<TopologyLinkEstimate>& links,
    CommunicationOperationKind kind) {
    std::unordered_map<std::string, TopologyLinkEstimate> operation_links;
    for (const TopologyLinkEstimate& link : links) {
        const std::string key = link.src_node + "\n" + link.dst_node + "\n" +
                                topology_link_scope_name(link.scope);
        auto [it, inserted] = operation_links.emplace(key, link);
        if (!inserted) {
            it->second.hop_count += link.hop_count;
            it->second.bytes += link.bytes;
            it->second.estimated_time_us += link.estimated_time_us;
        }
    }

    struct PairOperationAccumulator {
        std::uint64_t bytes = 0;
        double estimated_time_us = 0.0;
    };
    std::unordered_map<std::string, PairOperationAccumulator> pair_operations;
    TopologyOperationTotals totals;

    for (const auto& operation_entry : operation_links) {
        const TopologyLinkEstimate& link = operation_entry.second;
        const double transfer_without_penalty_us =
            link.bandwidth_gbps > 0.0
            ? (static_cast<double>(link.bytes) * 8.0) /
                  (link.bandwidth_gbps * 1000.0)
            : 0.0;
        const double contention_penalty_us =
            transfer_without_penalty_us *
            std::max(0.0, link.oversubscription - 1.0);
        const double estimated_throughput_gbps = link.estimated_time_us > 0.0
            ? (static_cast<double>(link.bytes) * 8.0) /
                  (link.estimated_time_us * 1000.0)
            : 0.0;
        totals.estimated_time_us += link.estimated_time_us;
        totals.contention_penalty_us += contention_penalty_us;

        auto [it, inserted] =
            registry.report.links.emplace(
                operation_entry.first,
                ClusterLinkReportStats{});
        ClusterLinkReportStats& link_stats = it->second;
        if (inserted) {
            link_stats.src_node = link.src_node;
            link_stats.dst_node = link.dst_node;
            link_stats.scope = topology_link_scope_name(link.scope);
            link_stats.bandwidth_gbps = link.bandwidth_gbps;
        }
        const std::uint64_t previous_samples = link_stats.samples;
        link_stats.samples += link.hop_count;
        link_stats.operations++;
        if (kind == CommunicationOperationKind::Collective) {
            link_stats.collective_operations++;
        } else {
            link_stats.point_to_point_operations++;
        }
        link_stats.bytes += link.bytes;
        link_stats.peak_bytes_per_operation =
            std::max(link_stats.peak_bytes_per_operation, link.bytes);
        link_stats.avg_latency_us =
            ((link_stats.avg_latency_us *
              static_cast<double>(previous_samples)) +
             (link.latency_us * static_cast<double>(link.hop_count))) /
            static_cast<double>(link_stats.samples);
        link_stats.estimated_time_us_total += link.estimated_time_us;
        link_stats.contention_penalty_us_total += contention_penalty_us;
        link_stats.peak_estimated_throughput_gbps =
            std::max(
                link_stats.peak_estimated_throughput_gbps,
                estimated_throughput_gbps);

        if (link.src_node == link.dst_node) {
            continue;
        }

        const bool source_is_a = link.src_node < link.dst_node;
        const std::string& node_a =
            source_is_a ? link.src_node : link.dst_node;
        const std::string& node_b =
            source_is_a ? link.dst_node : link.src_node;
        const std::string pair_key = node_a + "\n" + node_b;
        auto [pair_it, pair_inserted] =
            registry.report.node_pairs.emplace(
                pair_key,
                ClusterNodePairReportStats{});
        ClusterNodePairReportStats& pair_stats = pair_it->second;
        if (pair_inserted) {
            pair_stats.node_a = node_a;
            pair_stats.node_b = node_b;
        }

        ClusterNodePairDirectionReportStats& direction =
            source_is_a ? pair_stats.a_to_b : pair_stats.b_to_a;
        const std::uint64_t previous_transfers = direction.transfers;
        direction.transfers += link.hop_count;
        direction.bytes += link.bytes;
        direction.peak_bytes_per_operation =
            std::max(direction.peak_bytes_per_operation, link.bytes);
        direction.model_bandwidth_gbps = link.bandwidth_gbps;
        direction.avg_latency_us =
            ((direction.avg_latency_us *
              static_cast<double>(previous_transfers)) +
             (link.latency_us * static_cast<double>(link.hop_count))) /
            static_cast<double>(direction.transfers);
        direction.estimated_time_us_total += link.estimated_time_us;
        direction.contention_penalty_us_total += contention_penalty_us;
        direction.peak_estimated_throughput_gbps =
            std::max(
                direction.peak_estimated_throughput_gbps,
                estimated_throughput_gbps);

        PairOperationAccumulator& operation = pair_operations[pair_key];
        operation.bytes += link.bytes;
        operation.estimated_time_us += link.estimated_time_us;
    }

    for (const auto& entry : pair_operations) {
        ClusterNodePairReportStats& pair_stats =
            registry.report.node_pairs.at(entry.first);
        const PairOperationAccumulator& operation = entry.second;
        pair_stats.operations++;
        if (kind == CommunicationOperationKind::Collective) {
            pair_stats.collective_operations++;
        } else {
            pair_stats.point_to_point_operations++;
        }
        pair_stats.peak_combined_bytes_per_operation =
            std::max(
                pair_stats.peak_combined_bytes_per_operation,
                operation.bytes);
        const double throughput_gbps = operation.estimated_time_us > 0.0
            ? (static_cast<double>(operation.bytes) * 8.0) /
                  (operation.estimated_time_us * 1000.0)
            : 0.0;
        pair_stats.peak_estimated_throughput_gbps =
            std::max(
                pair_stats.peak_estimated_throughput_gbps,
                throughput_gbps);
    }

    return totals;
}

TopologyOperationTotals record_collective_completion_locked(
    RegistryImpl& registry,
    const std::shared_ptr<CommunicatorState>& state,
    const std::shared_ptr<CollectiveState>& collective) {
    remember_world_size_locked(registry, state->world_size);

    ClusterCollectiveReportStats& stats =
        collective_report_stats_for_type_locked(registry, collective->request.type);
    stats.calls++;
    const bool variable_alltoall =
        collective->request.type == CollectiveType::AllToAll &&
        !collective->request.input_splits.empty();
    std::uint64_t logical_payload_bytes = 0;
    if (variable_alltoall) {
        for (const auto& entry : collective->participants) {
            logical_payload_bytes += static_cast<std::uint64_t>(
                entry.second.input_bytes);
        }
    } else {
        logical_payload_bytes =
            static_cast<std::uint64_t>(collective->request.bytes) *
            static_cast<std::uint64_t>(state->world_size);
    }
    stats.bytes += logical_payload_bytes;

    for (const auto& entry : collective->participants) {
        ClusterRankReportStats& rank_stats = ensure_rank_report_locked(
            registry,
            global_rank_for_local(state, entry.first));
        rank_stats.collective_calls++;
    }

    const DistributedConfig& dist_config = fake_gpu::BackendConfig::instance().distributed_config();
    if (!dist_config.cluster_config.loaded()) {
        return {};
    }

    TopologyModel topology_model;
    std::string topology_error;
    if (!TopologyModel::build(dist_config.cluster_config, topology_model, topology_error)) {
        return {};
    }

    if (variable_alltoall) {
        const std::size_t dtype_size = collective_data_type_size(
            collective->request.dtype);
        std::vector<TopologyLinkEstimate> links;
        for (const auto& entry : collective->participants) {
            const CollectiveExecutionParticipant& sender = entry.second;
            for (int peer = 0; peer < state->world_size; ++peer) {
                if (peer == sender.rank) {
                    continue;
                }
                const std::size_t count =
                    sender.input_splits[static_cast<std::size_t>(peer)];
                if (count == 0) {
                    continue;
                }
                TopologyLinkEstimate link;
                if (topology_model.estimate_transfer(
                        global_rank_for_local(state, sender.rank),
                        global_rank_for_local(state, peer),
                        static_cast<std::uint64_t>(count * dtype_size),
                        link,
                        topology_error)) {
                    links.push_back(std::move(link));
                }
            }
        }
        const TopologyOperationTotals totals = record_topology_operation_locked(
            registry,
            links,
            CommunicationOperationKind::Collective);
        stats.estimated_time_us_total += totals.estimated_time_us;
        stats.contention_penalty_us_total += totals.contention_penalty_us;
        return totals;
    }

    CollectiveTopologyEstimate estimate;
    if (!topology_model.estimate_ring_collective(
            collective->request.type,
            static_cast<std::uint64_t>(collective->request.bytes),
            communicator_global_ranks(state),
            estimate,
            topology_error)) {
        return {};
    }

    const TopologyOperationTotals totals = record_topology_operation_locked(
        registry,
        estimate.links,
        CommunicationOperationKind::Collective);
    stats.estimated_time_us_total += totals.estimated_time_us;
    stats.contention_penalty_us_total += totals.contention_penalty_us;
    return totals;
}

TopologyOperationTotals record_point_to_point_completion_locked(
    RegistryImpl& registry,
    const std::shared_ptr<CommunicatorState>& state,
    const std::shared_ptr<PointToPointState>& point_to_point) {
    remember_world_size_locked(registry, state->world_size);
    ClusterPointToPointReportStats& stats = registry.report.point_to_point;
    stats.operations++;

    for (const auto& entry : point_to_point->participants) {
        ClusterRankReportStats& rank_stats = ensure_rank_report_locked(
            registry,
            global_rank_for_local(state, entry.first));
        rank_stats.point_to_point_calls++;

        const PointToPointSubmitRequest& participant = entry.second;
        if (participant.type == PointToPointType::Send) {
            stats.sends++;
            stats.bytes += static_cast<std::uint64_t>(participant.bytes);
        }
    }

    const DistributedConfig& dist_config =
        fake_gpu::BackendConfig::instance().distributed_config();
    if (!dist_config.cluster_config.loaded()) {
        return {};
    }

    TopologyModel topology_model;
    std::string topology_error;
    if (!TopologyModel::build(
            dist_config.cluster_config,
            topology_model,
            topology_error)) {
        return {};
    }

    std::vector<TopologyLinkEstimate> links;
    links.reserve(point_to_point->participants.size() / 2);
    for (const auto& entry : point_to_point->participants) {
        const PointToPointSubmitRequest& participant = entry.second;
        if (participant.type != PointToPointType::Send) {
            continue;
        }

        TopologyLinkEstimate link;
        if (!topology_model.estimate_transfer(
                global_rank_for_local(state, participant.rank),
                global_rank_for_local(state, participant.peer),
                static_cast<std::uint64_t>(participant.bytes),
                link,
                topology_error)) {
            continue;
        }
        links.push_back(std::move(link));
    }

    const TopologyOperationTotals totals = record_topology_operation_locked(
        registry,
        links,
        CommunicationOperationKind::PointToPoint);
    stats.estimated_time_us_total += totals.estimated_time_us;
    stats.contention_penalty_us_total += totals.contention_penalty_us;
    return totals;
}

void record_barrier_completion_locked(
    RegistryImpl& registry,
    const std::shared_ptr<CommunicatorState>& state,
    const std::shared_ptr<BarrierState>& barrier) {
    remember_world_size_locked(registry, state->world_size);
    registry.report.barrier.calls++;

    for (const auto& entry : barrier->participants) {
        ClusterRankReportStats& rank_stats = ensure_rank_report_locked(
            registry,
            global_rank_for_local(state, entry.first));
        rank_stats.barrier_calls++;
    }
}

void fail_pending_group_locked(
    RegistryImpl& registry,
    const std::shared_ptr<CommunicatorState>& state,
    std::string code,
    std::string detail) {
    state->failed = true;
    if (state->failed_at == SteadyTimePoint{}) {
        state->failed_at = std::chrono::steady_clock::now();
    }
    state->failure_code = std::move(code);
    state->failure_detail = std::move(detail);
    registry.pending_by_unique_id.erase(state->unique_id);
    state->cv.notify_all();
}

void fail_collective_locked(
    const std::shared_ptr<CommunicatorState>& state,
    const std::shared_ptr<CollectiveState>& collective,
    std::string code,
    std::string detail) {
    state->failed = true;
    if (state->failed_at == SteadyTimePoint{}) {
        state->failed_at = std::chrono::steady_clock::now();
    }
    state->failure_code = std::move(code);
    state->failure_detail = std::move(detail);
    if (collective) {
        collective->failed = true;
        collective->failure_code = state->failure_code;
        collective->failure_detail = state->failure_detail;
        collective->cv.notify_all();
        state->collectives.erase(collective->request.seqno);
    }
}

void fail_barrier_locked(
    const std::shared_ptr<CommunicatorState>& state,
    const std::shared_ptr<BarrierState>& barrier,
    std::string code,
    std::string detail) {
    state->failed = true;
    if (state->failed_at == SteadyTimePoint{}) {
        state->failed_at = std::chrono::steady_clock::now();
    }
    state->failure_code = std::move(code);
    state->failure_detail = std::move(detail);
    if (barrier) {
        barrier->failed = true;
        barrier->failure_code = state->failure_code;
        barrier->failure_detail = state->failure_detail;
        barrier->cv.notify_all();
        state->barriers.erase(barrier->request.seqno);
    }
}

void fail_batch_locked(
    const std::shared_ptr<CommunicatorState>& state,
    const std::shared_ptr<BatchState>& batch,
    std::string code,
    std::string detail) {
    state->failed = true;
    if (state->failed_at == SteadyTimePoint{}) {
        state->failed_at = std::chrono::steady_clock::now();
    }
    state->failure_code = std::move(code);
    state->failure_detail = std::move(detail);
    if (batch) {
        batch->failed = true;
        batch->failure_code = state->failure_code;
        batch->failure_detail = state->failure_detail;
        batch->cv.notify_all();
        state->batches.erase(batch->request.base_seqno);
    }
}

void fail_split_locked(
    const std::shared_ptr<CommunicatorState>& state,
    const std::shared_ptr<SplitState>& split,
    std::string code,
    std::string detail) {
    state->failed = true;
    if (state->failed_at == SteadyTimePoint{}) {
        state->failed_at = std::chrono::steady_clock::now();
    }
    state->failure_code = std::move(code);
    state->failure_detail = std::move(detail);
    if (split) {
        split->failed = true;
        split->failure_code = state->failure_code;
        split->failure_detail = state->failure_detail;
        split->cv.notify_all();
        state->splits.erase(split->request.seqno);
    }
}

void fail_point_to_point_locked(
    const std::shared_ptr<CommunicatorState>& state,
    const std::shared_ptr<PointToPointState>& p2p,
    std::string code,
    std::string detail) {
    state->failed = true;
    if (state->failed_at == SteadyTimePoint{}) {
        state->failed_at = std::chrono::steady_clock::now();
    }
    state->failure_code = std::move(code);
    state->failure_detail = std::move(detail);
    if (p2p) {
        p2p->failed = true;
        p2p->failure_code = state->failure_code;
        p2p->failure_detail = state->failure_detail;
        p2p->cv.notify_all();
        state->point_to_points.erase(p2p->request.seqno);
    }
}

void fail_shrink_locked(
    const std::shared_ptr<ShrinkState>& shrink,
    std::string code,
    std::string detail) {
    if (shrink) {
        shrink->failed = true;
        shrink->failure_code = std::move(code);
        shrink->failure_detail = std::move(detail);
        shrink->cv.notify_all();
    }
}

void fail_pending_operations_locked(
    const std::shared_ptr<CommunicatorState>& state,
    const std::string& code,
    const std::string& detail) {
    state->failed = true;
    if (state->failed_at == SteadyTimePoint{}) {
        state->failed_at = std::chrono::steady_clock::now();
    }
    state->failure_code = code;
    state->failure_detail = detail;

    for (const auto& entry : state->collectives) {
        entry.second->failed = true;
        entry.second->failure_code = code;
        entry.second->failure_detail = detail;
        entry.second->cv.notify_all();
    }
    for (const auto& entry : state->barriers) {
        entry.second->failed = true;
        entry.second->failure_code = code;
        entry.second->failure_detail = detail;
        entry.second->cv.notify_all();
    }
    for (const auto& entry : state->batches) {
        entry.second->failed = true;
        entry.second->failure_code = code;
        entry.second->failure_detail = detail;
        entry.second->cv.notify_all();
    }
    for (const auto& entry : state->splits) {
        entry.second->failed = true;
        entry.second->failure_code = code;
        entry.second->failure_detail = detail;
        entry.second->cv.notify_all();
    }
    for (const auto& entry : state->point_to_points) {
        entry.second->failed = true;
        entry.second->failure_code = code;
        entry.second->failure_detail = detail;
        entry.second->cv.notify_all();
    }

    state->collectives.clear();
    state->barriers.clear();
    state->batches.clear();
    state->splits.clear();
    state->point_to_points.clear();
    state->cv.notify_all();
}

bool collective_requests_match(
    const CollectiveSubmitRequest& expected,
    const CollectiveSubmitRequest& actual,
    std::string& error_code,
    std::string& error_detail) {
    if (expected.transport != actual.transport) {
        error_code = "buffer_transport_mismatch";
        error_detail = "collective ranks disagreed on buffer transport";
        return false;
    }
    if (expected.type != actual.type) {
        error_code = "collective_type_mismatch";
        error_detail =
            "expected " + std::string(collective_type_name(expected.type)) +
            ", got " + collective_type_name(actual.type);
        return false;
    }
    if (expected.dtype != actual.dtype) {
        error_code = "dtype_mismatch";
        error_detail =
            "expected " + std::string(collective_data_type_name(expected.dtype)) +
            ", got " + collective_data_type_name(actual.dtype);
        return false;
    }
    const bool expected_variable =
        expected.type == CollectiveType::AllToAll &&
        !expected.input_splits.empty();
    const bool actual_variable =
        actual.type == CollectiveType::AllToAll &&
        !actual.input_splits.empty();
    if (expected_variable != actual_variable) {
        error_code = "alltoall_split_mode_mismatch";
        error_detail = "all-to-all ranks disagreed on explicit split mode";
        return false;
    }
    if (!expected_variable && expected.count != actual.count) {
        error_code = "count_mismatch";
        error_detail =
            "expected count=" + std::to_string(expected.count) +
            ", got " + std::to_string(actual.count);
        return false;
    }
    if (expected.root != actual.root) {
        error_code = "root_mismatch";
        error_detail =
            "expected root=" + std::to_string(expected.root) +
            ", got " + std::to_string(actual.root);
        return false;
    }
    if (expected.reduce_op != actual.reduce_op) {
        error_code = "reduce_op_mismatch";
        error_detail =
            "expected reduce_op=" + std::string(collective_reduce_op_name(expected.reduce_op)) +
            ", got " + collective_reduce_op_name(actual.reduce_op);
        return false;
    }
    if (!expected_variable && expected.bytes != actual.bytes) {
        error_code = "bytes_mismatch";
        error_detail =
            "expected bytes=" + std::to_string(expected.bytes) +
            ", got " + std::to_string(actual.bytes);
        return false;
    }
    if (expected.proxy_only != actual.proxy_only) {
        error_code = "proxy_only_mismatch";
        error_detail =
            "expected proxy_only=" + std::string(expected.proxy_only ? "1" : "0") +
            ", got " + (actual.proxy_only ? "1" : "0");
        return false;
    }
    if (!expected_variable && expected.payload_bytes != actual.payload_bytes) {
        error_code = "payload_bytes_mismatch";
        error_detail =
            "expected payload_bytes=" + std::to_string(expected.payload_bytes) +
            ", got " + std::to_string(actual.payload_bytes);
        return false;
    }
    return true;
}

bool point_to_point_requests_match(
    const PointToPointSubmitRequest& expected,
    const PointToPointSubmitRequest& actual,
    std::string& error_code,
    std::string& error_detail) {
    if (expected.transport != actual.transport) {
        error_code = "buffer_transport_mismatch";
        error_detail = "point-to-point peers disagreed on buffer transport";
        return false;
    }
    if (expected.timeout_ms != actual.timeout_ms) {
        error_code = "p2p_timeout_mismatch";
        error_detail =
            "expected timeout_ms=" + std::to_string(expected.timeout_ms) +
            ", got " + std::to_string(actual.timeout_ms);
        return false;
    }
    if (expected.payload_bytes != actual.payload_bytes) {
        error_code = "payload_bytes_mismatch";
        error_detail =
            "expected payload_bytes=" + std::to_string(expected.payload_bytes) +
            ", got " + std::to_string(actual.payload_bytes);
        return false;
    }
    return true;
}

bool split_requests_match(
    const CommunicatorSplitRequest& expected,
    const CommunicatorSplitRequest& actual,
    std::string& error_code,
    std::string& error_detail) {
    if (expected.timeout_ms != actual.timeout_ms) {
        error_code = "split_timeout_mismatch";
        error_detail =
            "expected timeout_ms=" + std::to_string(expected.timeout_ms) +
            ", got " + std::to_string(actual.timeout_ms);
        return false;
    }
    return true;
}

bool shrink_requests_match(
    const CommunicatorShrinkRequest& expected,
    const CommunicatorShrinkRequest& actual,
    std::string& error_code,
    std::string& error_detail) {
    if (expected.excluded_ranks != actual.excluded_ranks) {
        error_code = "shrink_excluded_ranks_mismatch";
        error_detail = "communicator ranks disagreed on the shrink exclusion list";
        return false;
    }
    if (expected.abort_parent != actual.abort_parent) {
        error_code = "shrink_flags_mismatch";
        error_detail = "communicator ranks disagreed on shrink flags";
        return false;
    }
    if (expected.timeout_ms != actual.timeout_ms) {
        error_code = "shrink_timeout_mismatch";
        error_detail =
            "expected timeout_ms=" + std::to_string(expected.timeout_ms) +
            ", got " + std::to_string(actual.timeout_ms);
        return false;
    }
    return true;
}

CollectiveExecutionResult execute_point_to_point_locked(
    const PointToPointSubmitRequest& request,
    const std::shared_ptr<PointToPointState>& p2p) {
    std::string error;
    CollectiveExecutionResult result;
    result.ok = true;

    for (const auto& entry : p2p->participants) {
        const PointToPointSubmitRequest& participant = entry.second;
        if (participant.peer == participant.rank) {
            return CollectiveExecutionResult{
                false,
                "invalid_peer",
                "point-to-point peer must not equal the local rank",
            };
        }

        const auto peer_it = p2p->participants.find(participant.peer);
        if (peer_it == p2p->participants.end()) {
            return CollectiveExecutionResult{
                false,
                "missing_peer",
                "rank " + std::to_string(participant.rank) +
                    " expected peer " + std::to_string(participant.peer) +
                    " to submit the same point-to-point seqno",
            };
        }

        const PointToPointSubmitRequest& peer = peer_it->second;
        if (participant.type == peer.type) {
            return CollectiveExecutionResult{
                false,
                "p2p_direction_mismatch",
                "rank " + std::to_string(participant.rank) +
                    " and peer " + std::to_string(participant.peer) +
                    " submitted the same point-to-point direction",
            };
        }
        if (peer.peer != participant.rank) {
            return CollectiveExecutionResult{
                false,
                "peer_mismatch",
                "rank " + std::to_string(participant.rank) +
                    " expected peer " + std::to_string(participant.peer) +
                    " to target rank " + std::to_string(participant.rank),
            };
        }
        if (participant.dtype != peer.dtype) {
            return CollectiveExecutionResult{
                false,
                "dtype_mismatch",
                "point-to-point dtype mismatch between rank " +
                    std::to_string(participant.rank) + " and peer " +
                    std::to_string(participant.peer),
            };
        }
        if (participant.count != peer.count) {
            return CollectiveExecutionResult{
                false,
                "count_mismatch",
                "point-to-point count mismatch between rank " +
                    std::to_string(participant.rank) + " and peer " +
                    std::to_string(participant.peer),
            };
        }
        if (participant.bytes != peer.bytes) {
            return CollectiveExecutionResult{
                false,
                "bytes_mismatch",
                "point-to-point bytes mismatch between rank " +
                    std::to_string(participant.rank) + " and peer " +
                    std::to_string(participant.peer),
            };
        }
    }

    for (const auto& entry : p2p->participants) {
        const PointToPointSubmitRequest& participant = entry.second;
        if (participant.type != PointToPointType::Send) {
            continue;
        }

        const PointToPointSubmitRequest& receiver =
            p2p->participants.at(participant.peer);

        std::vector<char> sender_bytes;
        if (!load_participant_buffer(
                participant.transport,
                participant.staging_name,
                participant.dtype,
                participant.transport == BufferTransport::SocketPayload
                    ? participant.payload_bytes
                    : participant.bytes,
                {participant.count},
                participant.rank,
                participant.seqno,
                participant.payload,
                sender_bytes,
                error)) {
            return CollectiveExecutionResult{false, "staging_open_failed", error};
        }

        std::vector<char> receiver_output;
        if (!store_participant_buffer(
                receiver.transport,
                receiver.staging_name,
                receiver.dtype,
                {receiver.count},
                receiver.rank,
                receiver.seqno,
                sender_bytes.data(),
                sender_bytes.size(),
                receiver_output,
                error)) {
            return CollectiveExecutionResult{false, "staging_open_failed", error};
        }
        if (receiver.transport == BufferTransport::SocketPayload) {
            result.output_payloads[receiver.rank] = std::move(receiver_output);
        }
        if (participant.transport == BufferTransport::SocketPayload) {
            result.output_payloads.emplace(participant.rank, std::vector<char>{});
        }
    }

    for (const auto& entry : p2p->participants) {
        if (entry.second.transport == BufferTransport::SocketPayload) {
            result.output_payloads.emplace(entry.first, std::vector<char>{});
        }
    }
    return result;
}

bool batch_items_match(
    const CollectiveBatchPlanItem& expected,
    const CollectiveBatchPlanItem& actual,
    std::string& error_code,
    std::string& error_detail,
    std::size_t index) {
    if (expected.type != actual.type) {
        error_code = "group_collective_type_mismatch";
        error_detail =
            "group op " + std::to_string(index) + " expected " +
            collective_type_name(expected.type) + ", got " +
            collective_type_name(actual.type);
        return false;
    }
    if (expected.dtype != actual.dtype) {
        error_code = "group_dtype_mismatch";
        error_detail =
            "group op " + std::to_string(index) + " expected " +
            collective_data_type_name(expected.dtype) + ", got " +
            collective_data_type_name(actual.dtype);
        return false;
    }
    if (expected.count != actual.count) {
        error_code = "group_count_mismatch";
        error_detail =
            "group op " + std::to_string(index) + " expected count=" +
            std::to_string(expected.count) + ", got " + std::to_string(actual.count);
        return false;
    }
    if (expected.root != actual.root) {
        error_code = "group_root_mismatch";
        error_detail =
            "group op " + std::to_string(index) + " expected root=" +
            std::to_string(expected.root) + ", got " + std::to_string(actual.root);
        return false;
    }
    if (expected.reduce_op != actual.reduce_op) {
        error_code = "group_reduce_op_mismatch";
        error_detail =
            "group op " + std::to_string(index) + " expected reduce_op=" +
            collective_reduce_op_name(expected.reduce_op) + ", got " +
            collective_reduce_op_name(actual.reduce_op);
        return false;
    }
    if (expected.bytes != actual.bytes) {
        error_code = "group_bytes_mismatch";
        error_detail =
            "group op " + std::to_string(index) + " expected bytes=" +
            std::to_string(expected.bytes) + ", got " + std::to_string(actual.bytes);
        return false;
    }
    return true;
}

bool batch_requests_match(
    const CollectiveBatchPrepareRequest& expected,
    const CollectiveBatchPrepareRequest& actual,
    std::string& error_code,
    std::string& error_detail) {
    if (expected.operations.empty()) {
        error_code = "empty_group";
        error_detail = "group must contain at least one operation";
        return false;
    }
    if (expected.operations.size() != actual.operations.size()) {
        error_code = "group_size_mismatch";
        error_detail =
            "expected group size=" + std::to_string(expected.operations.size()) +
            ", got " + std::to_string(actual.operations.size());
        return false;
    }
    for (std::size_t index = 0; index < expected.operations.size(); ++index) {
        if (!batch_items_match(expected.operations[index], actual.operations[index], error_code, error_detail, index)) {
            return false;
        }
    }
    return true;
}

CollectiveExecutionResult execute_collective_locked(
    const CollectiveSubmitRequest& request,
    const std::shared_ptr<CollectiveState>& collective) {
    std::vector<CollectiveExecutionParticipant> participants;
    participants.reserve(collective->participants.size());
    for (const auto& entry : collective->participants) {
        participants.push_back(entry.second);
    }
    std::sort(
        participants.begin(),
        participants.end(),
        [](const CollectiveExecutionParticipant& lhs, const CollectiveExecutionParticipant& rhs) {
            return lhs.rank < rhs.rank;
        });

    CollectiveExecutionRequest execution_request;
    execution_request.comm_id = request.comm_id;
    execution_request.seqno = request.seqno;
    execution_request.type = request.type;
    execution_request.dtype = request.dtype;
    execution_request.count = request.count;
    execution_request.root_rank = request.root;
    execution_request.reduce_op = request.reduce_op;
    execution_request.bytes = request.bytes;

    if (request.type == CollectiveType::AllReduce) {
        return execute_allreduce_sum(execution_request, participants);
    }
    if (request.type == CollectiveType::Reduce) {
        return execute_reduce(execution_request, participants);
    }
    if (request.type == CollectiveType::Broadcast) {
        return execute_broadcast(execution_request, participants);
    }
    if (request.type == CollectiveType::AllGather) {
        return execute_allgather(execution_request, participants);
    }
    if (request.type == CollectiveType::ReduceScatter) {
        return execute_reducescatter(execution_request, participants);
    }
    if (request.type == CollectiveType::AllToAll) {
        return execute_alltoall(execution_request, participants);
    }
    return CollectiveExecutionResult{false, "unsupported_collective", "unsupported collective type"};
}

}  // namespace

CommunicatorRegistrationResult CommunicatorRegistry::init_communicator(
    const std::string& unique_id,
    int world_size,
    int rank,
    int timeout_ms) {
    if (unique_id.empty()) {
        return make_error("missing_unique_id", "unique_id must be set");
    }
    if (world_size <= 0) {
        return make_error("invalid_world_size", "world_size must be > 0");
    }
    if (rank < 0 || rank >= world_size) {
        return make_error("invalid_rank", "rank must be within [0, world_size)");
    }
    if (timeout_ms <= 0) {
        return make_error("invalid_timeout", "timeout_ms must be > 0");
    }

    RegistryImpl& registry = registry_impl();
    std::shared_ptr<CommunicatorState> state;

    {
        std::unique_lock<std::mutex> lock(registry.mutex);
        auto it = registry.pending_by_unique_id.find(unique_id);
        if (it == registry.pending_by_unique_id.end()) {
            state = std::make_shared<CommunicatorState>();
            state->unique_id = unique_id;
            state->world_size = world_size;
            state->global_ranks.reserve(static_cast<std::size_t>(world_size));
            for (int global_rank = 0; global_rank < world_size; ++global_rank) {
                state->global_ranks.push_back(global_rank);
            }
            registry.pending_by_unique_id.emplace(unique_id, state);
        } else {
            state = it->second;
        }

        if (state->world_size != world_size) {
            const std::string detail =
                "world_size mismatch for unique_id " + unique_id + ": expected " +
                std::to_string(state->world_size) + ", got " + std::to_string(world_size);
            fail_pending_group_locked(registry, state, "world_size_mismatch", detail);
            return make_error("world_size_mismatch", detail);
        }

        if (state->participants.find(rank) != state->participants.end()) {
            const std::string detail =
                "rank " + std::to_string(rank) + " already registered for unique_id " + unique_id;
            fail_pending_group_locked(registry, state, "duplicate_rank", detail);
            return make_error("duplicate_rank", detail);
        }

        state->participants.emplace(rank, true);
        remember_world_size_locked(registry, world_size);
        ensure_rank_report_locked(registry, rank).communicator_inits++;
        if (static_cast<int>(state->participants.size()) == state->world_size) {
            state->ready = true;
            state->comm_id = registry.next_comm_id++;
            registry.active_by_comm_id.emplace(state->comm_id, state);
            registry.pending_by_unique_id.erase(unique_id);
            registry.report.communicator_count++;
            state->cv.notify_all();
            return CommunicatorRegistrationResult{true, state->comm_id, 0, "", ""};
        }

        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        const auto wait_begin = std::chrono::steady_clock::now();
        while (!state->ready && !state->failed) {
            if (state->cv.wait_until(lock, deadline) == std::cv_status::timeout) {
                record_wait_time_locked(
                    registry,
                    state,
                    rank,
                    std::chrono::steady_clock::now() - wait_begin);
                record_timeout_locked(registry, state, state->participants);
                const std::string detail =
                    "timeout waiting for ranks on unique_id " + unique_id;
                fail_pending_group_locked(registry, state, "timeout_waiting_for_ranks", detail);
                return make_error("timeout_waiting_for_ranks", detail);
            }
        }
        record_wait_time_locked(
            registry,
            state,
            rank,
            std::chrono::steady_clock::now() - wait_begin);

        if (state->failed) {
            return make_error(state->failure_code, state->failure_detail);
        }

        return CommunicatorRegistrationResult{true, state->comm_id, 0, "", ""};
    }
}

CommunicatorDestroyResult CommunicatorRegistry::destroy_communicator(int comm_id, int rank) {
    RegistryImpl& registry = registry_impl();
    std::lock_guard<std::mutex> lock(registry.mutex);

    auto it = registry.active_by_comm_id.find(comm_id);
    if (it == registry.active_by_comm_id.end()) {
        return make_destroy_error("unknown_comm_id", "communicator not found");
    }

    const std::shared_ptr<CommunicatorState>& state = it->second;
    if (state->participants.find(rank) == state->participants.end()) {
        return make_destroy_error("unknown_rank", "rank is not a member of this communicator");
    }
    if (!state->destroyed_ranks.emplace(rank, true).second) {
        return make_destroy_error("duplicate_destroy", "rank already destroyed this communicator");
    }

    if (static_cast<int>(state->destroyed_ranks.size()) == state->world_size) {
        registry.active_by_comm_id.erase(it);
    }

    CommunicatorDestroyResult result;
    result.ok = true;
    return result;
}

CommunicatorSplitResult CommunicatorRegistry::split_communicator(const CommunicatorSplitRequest& request) {
    if (request.comm_id <= 0) {
        return make_split_error("invalid_comm_id", "comm_id must be > 0");
    }
    if (request.rank < 0) {
        return make_split_error("invalid_rank", "rank must be >= 0");
    }
    if (request.seqno == 0) {
        return make_split_error("invalid_seqno", "seqno must be > 0");
    }
    if (request.color < -1) {
        return make_split_error("invalid_color", "color must be >= 0 or -1 for no color");
    }
    if (request.timeout_ms <= 0) {
        return make_split_error("invalid_timeout", "timeout_ms must be > 0");
    }

    RegistryImpl& registry = registry_impl();
    std::shared_ptr<CommunicatorState> state;
    std::shared_ptr<SplitState> split;

    std::unique_lock<std::mutex> lock(registry.mutex);
    auto it = registry.active_by_comm_id.find(request.comm_id);
    if (it == registry.active_by_comm_id.end()) {
        return make_split_error("unknown_comm_id", "communicator not found");
    }

    state = it->second;
    if (state->participants.find(request.rank) == state->participants.end()) {
        return make_split_error("unknown_rank", "rank is not a member of this communicator");
    }
    if (state->destroyed_ranks.find(request.rank) != state->destroyed_ranks.end()) {
        return make_split_error("rank_destroyed", "rank already destroyed this communicator");
    }
    if (state->failed) {
        return make_split_error(state->failure_code, state->failure_detail);
    }
    if (request.rank >= state->world_size) {
        return make_split_error("invalid_rank", "rank must be within [0, world_size)");
    }

    if (request.seqno != state->next_seqno) {
        if (request.seqno < state->next_seqno) {
            return make_split_error(
                "stale_seqno",
                "expected seqno " + std::to_string(state->next_seqno) +
                    ", got " + std::to_string(request.seqno));
        }
        return make_split_error(
            "out_of_order_seqno",
            "expected seqno " + std::to_string(state->next_seqno) +
                ", got " + std::to_string(request.seqno));
    }

    auto split_it = state->splits.find(request.seqno);
    if (split_it == state->splits.end()) {
        split = std::make_shared<SplitState>();
        split->request = request;
        state->splits.emplace(request.seqno, split);
    } else {
        split = split_it->second;
    }

    std::string error_code;
    std::string error_detail;
    if (!split_requests_match(split->request, request, error_code, error_detail)) {
        fail_split_locked(state, split, std::move(error_code), std::move(error_detail));
        return make_split_error(state->failure_code, state->failure_detail);
    }

    if (!split->participants.emplace(request.rank, request).second) {
        fail_split_locked(
            state,
            split,
            "duplicate_split_rank",
            "rank already submitted this split seqno");
        return make_split_error(state->failure_code, state->failure_detail);
    }

    if (static_cast<int>(split->participants.size()) != state->world_size) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(request.timeout_ms);
        const auto wait_begin = std::chrono::steady_clock::now();
        while (!split->completed && !split->failed && !state->failed) {
            if (split->cv.wait_until(lock, deadline) == std::cv_status::timeout) {
                record_wait_time_locked(
                    registry,
                    state,
                    request.rank,
                    std::chrono::steady_clock::now() - wait_begin);
                record_timeout_locked(registry, state, split->participants);
                fail_split_locked(
                    state,
                    split,
                    "timeout_waiting_for_split",
                    "timeout waiting for all ranks to join split seqno " +
                        std::to_string(request.seqno));
                return make_split_error(state->failure_code, state->failure_detail);
            }
        }
        record_wait_time_locked(
            registry,
            state,
            request.rank,
            std::chrono::steady_clock::now() - wait_begin);

        if (split->completed) {
            const auto result_it = split->results.find(request.rank);
            if (result_it == split->results.end()) {
                return make_split_error("internal_error", "split completed without a per-rank result");
            }
            const SplitParticipantResult& per_rank = result_it->second;
            return CommunicatorSplitResult{
                true,
                request.seqno,
                per_rank.participating,
                per_rank.new_comm_id,
                per_rank.new_rank,
                per_rank.new_world_size,
                "",
                "",
            };
        }
        return make_split_error(state->failure_code, state->failure_detail);
    }

    std::unordered_map<int, std::vector<std::pair<int, int>>> groups;
    for (const auto& entry : split->participants) {
        const CommunicatorSplitRequest& participant = entry.second;
        if (participant.color == -1) {
            split->results.emplace(
                participant.rank,
                SplitParticipantResult{false, -1, -1, 0});
            continue;
        }
        groups[participant.color].push_back({participant.key, participant.rank});
    }

    for (auto& entry : groups) {
        const int color = entry.first;
        std::vector<std::pair<int, int>>& members = entry.second;
        std::sort(
            members.begin(),
            members.end(),
            [](const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) {
                if (lhs.first != rhs.first) {
                    return lhs.first < rhs.first;
                }
                return lhs.second < rhs.second;
            });

        auto child = std::make_shared<CommunicatorState>();
        child->unique_id =
            "split-parent" + std::to_string(state->comm_id) +
            "-seq" + std::to_string(request.seqno) +
            "-color" + std::to_string(color);
        child->world_size = static_cast<int>(members.size());
        child->comm_id = registry.next_comm_id++;
        child->ready = true;
        child->global_ranks.resize(members.size());

        for (std::size_t subgroup_rank = 0; subgroup_rank < members.size(); ++subgroup_rank) {
            const int parent_rank = members[subgroup_rank].second;
            const int global_rank = global_rank_for_local(state, parent_rank);
            child->global_ranks[subgroup_rank] = global_rank;
            child->participants.emplace(static_cast<int>(subgroup_rank), true);
            split->results.emplace(
                parent_rank,
                SplitParticipantResult{
                    true,
                    child->comm_id,
                    static_cast<int>(subgroup_rank),
                    child->world_size,
                });
            ensure_rank_report_locked(registry, global_rank).communicator_inits++;
        }

        registry.active_by_comm_id.emplace(child->comm_id, child);
        registry.report.communicator_count++;
    }

    split->completed = true;
    state->next_seqno++;
    state->splits.erase(request.seqno);
    split->cv.notify_all();

    const auto result_it = split->results.find(request.rank);
    if (result_it == split->results.end()) {
        fail_split_locked(state, split, "internal_error", "split completed without a per-rank result");
        return make_split_error(state->failure_code, state->failure_detail);
    }

    const SplitParticipantResult& per_rank = result_it->second;
    return CommunicatorSplitResult{
        true,
        request.seqno,
        per_rank.participating,
        per_rank.new_comm_id,
        per_rank.new_rank,
        per_rank.new_world_size,
        "",
        "",
    };
}

CommunicatorFailureResult CommunicatorRegistry::inject_communicator_failure(
    const CommunicatorFailureRequest& request) {
    if (request.comm_id <= 0) {
        return make_failure_error("invalid_comm_id", "comm_id must be > 0");
    }
    if (request.rank < 0) {
        return make_failure_error("invalid_rank", "rank must be >= 0");
    }
    if (request.seqno == 0) {
        return make_failure_error("invalid_seqno", "seqno must be > 0");
    }
    if (request.operation.empty()) {
        return make_failure_error("invalid_operation", "operation must not be empty");
    }
    if (request.error_code.empty()) {
        return make_failure_error("invalid_error_code", "error_code must not be empty");
    }

    RegistryImpl& registry = registry_impl();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto state_it = registry.active_by_comm_id.find(request.comm_id);
    if (state_it == registry.active_by_comm_id.end()) {
        return make_failure_error("unknown_comm_id", "communicator not found");
    }

    const std::shared_ptr<CommunicatorState>& state = state_it->second;
    if (request.rank >= state->world_size ||
        state->participants.find(request.rank) == state->participants.end()) {
        return make_failure_error(
            "unknown_rank",
            "rank is not a member of this communicator");
    }
    if (state->destroyed_ranks.find(request.rank) !=
        state->destroyed_ranks.end()) {
        return make_failure_error(
            "rank_destroyed",
            "rank already destroyed this communicator");
    }
    if (state->failed) {
        return make_failure_error(state->failure_code, state->failure_detail);
    }
    if (request.seqno != state->next_seqno) {
        const std::string code = request.seqno < state->next_seqno
            ? "stale_seqno"
            : "out_of_order_seqno";
        return make_failure_error(
            code,
            "expected seqno " + std::to_string(state->next_seqno) +
                ", got " + std::to_string(request.seqno));
    }

    ClusterFailureEvent event;
    event.index = ++registry.report.resilience_event_entries;
    event.comm_id = state->comm_id;
    event.seqno = request.seqno;
    event.local_rank = request.rank;
    event.global_rank = global_rank_for_local(state, request.rank);
    event.source = "injected";
    event.operation = request.operation;
    event.error_code = request.error_code;
    event.error_detail =
        "injected failure on rank " + std::to_string(event.global_rank) +
        " during " + request.operation + " seqno " +
        std::to_string(request.seqno);
    event.attempted_payload_bytes = request.attempted_payload_bytes;

    const auto collective_it = state->collectives.find(request.seqno);
    if (collective_it != state->collectives.end()) {
        for (const auto& entry : collective_it->second->participants) {
            event.observed_ranks.push_back(
                global_rank_for_local(state, entry.first));
            event.attempted_payload_bytes +=
                static_cast<std::uint64_t>(entry.second.bytes);
        }
    }
    event.observed_ranks.push_back(event.global_rank);
    std::sort(event.observed_ranks.begin(), event.observed_ranks.end());
    event.observed_ranks.erase(
        std::unique(event.observed_ranks.begin(), event.observed_ranks.end()),
        event.observed_ranks.end());

    const std::uint64_t event_index = event.index;
    const std::string detail = event.error_detail;
    registry.report.failure_events.push_back(std::move(event));
    fail_pending_operations_locked(state, request.error_code, detail);

    CommunicatorFailureResult result;
    result.ok = true;
    result.event_index = event_index;
    return result;
}

CommunicatorShrinkResult CommunicatorRegistry::shrink_communicator(
    const CommunicatorShrinkRequest& request) {
    if (request.comm_id <= 0) {
        return make_shrink_error("invalid_comm_id", "comm_id must be > 0");
    }
    if (request.rank < 0) {
        return make_shrink_error("invalid_rank", "rank must be >= 0");
    }
    if (request.seqno == 0) {
        return make_shrink_error("invalid_seqno", "seqno must be > 0");
    }
    if (request.excluded_ranks.empty()) {
        return make_shrink_error(
            "invalid_excluded_ranks",
            "at least one rank must be excluded");
    }
    if (request.timeout_ms <= 0) {
        return make_shrink_error("invalid_timeout", "timeout_ms must be > 0");
    }
    if (!std::is_sorted(
            request.excluded_ranks.begin(), request.excluded_ranks.end()) ||
        std::adjacent_find(
            request.excluded_ranks.begin(), request.excluded_ranks.end()) !=
            request.excluded_ranks.end()) {
        return make_shrink_error(
            "invalid_excluded_ranks",
            "excluded ranks must be sorted and unique");
    }

    RegistryImpl& registry = registry_impl();
    std::shared_ptr<CommunicatorState> state;
    std::shared_ptr<ShrinkState> shrink;
    std::unique_lock<std::mutex> lock(registry.mutex);

    const auto state_it = registry.active_by_comm_id.find(request.comm_id);
    if (state_it == registry.active_by_comm_id.end()) {
        return make_shrink_error("unknown_comm_id", "communicator not found");
    }
    state = state_it->second;
    if (request.rank >= state->world_size ||
        state->participants.find(request.rank) == state->participants.end()) {
        return make_shrink_error(
            "unknown_rank",
            "rank is not a member of this communicator");
    }
    if (state->destroyed_ranks.find(request.rank) !=
        state->destroyed_ranks.end()) {
        return make_shrink_error(
            "rank_destroyed",
            "rank already destroyed this communicator");
    }
    if (request.excluded_ranks.size() >=
        static_cast<std::size_t>(state->world_size)) {
        return make_shrink_error(
            "invalid_excluded_ranks",
            "shrink must retain at least one rank");
    }
    for (int excluded_rank : request.excluded_ranks) {
        if (excluded_rank < 0 || excluded_rank >= state->world_size) {
            return make_shrink_error(
                "invalid_excluded_ranks",
                "excluded rank must be within [0, world_size)");
        }
    }
    if (std::binary_search(
            request.excluded_ranks.begin(),
            request.excluded_ranks.end(),
            request.rank)) {
        return make_shrink_error(
            "excluded_rank_called_shrink",
            "an excluded rank must not call communicator shrink");
    }
    if (state->failed && !request.abort_parent) {
        return make_shrink_error(
            "parent_failed_requires_abort",
            "NCCL_SHRINK_ABORT is required after a communicator error");
    }
    if (!state->failed && !request.abort_parent &&
        (!state->collectives.empty() || !state->barriers.empty() ||
         !state->batches.empty() || !state->splits.empty() ||
         !state->point_to_points.empty())) {
        return make_shrink_error(
            "outstanding_operations",
            "normal communicator shrink requires no outstanding operations");
    }
    if (request.seqno != state->next_seqno) {
        const std::string code = request.seqno < state->next_seqno
            ? "stale_seqno"
            : "out_of_order_seqno";
        return make_shrink_error(
            code,
            "expected seqno " + std::to_string(state->next_seqno) +
                ", got " + std::to_string(request.seqno));
    }

    if (request.abort_parent && !state->failed) {
        fail_pending_operations_locked(
            state,
            "communicator_shrink_abort",
            "parent communicator was aborted during shrink");
    }

    auto shrink_it = state->shrinks.find(request.seqno);
    if (shrink_it == state->shrinks.end()) {
        shrink = std::make_shared<ShrinkState>();
        shrink->request = request;
        shrink->first_submit_at = std::chrono::steady_clock::now();
        state->shrinks.emplace(request.seqno, shrink);
    } else {
        shrink = shrink_it->second;
    }

    std::string error_code;
    std::string error_detail;
    if (!shrink_requests_match(
            shrink->request,
            request,
            error_code,
            error_detail)) {
        fail_shrink_locked(
            shrink,
            std::move(error_code),
            std::move(error_detail));
        return make_shrink_error(
            shrink->failure_code,
            shrink->failure_detail);
    }
    if (shrink->failed) {
        return make_shrink_error(
            shrink->failure_code,
            shrink->failure_detail);
    }
    if (!shrink->participants.emplace(request.rank, request).second) {
        fail_shrink_locked(
            shrink,
            "duplicate_shrink_rank",
            "rank already submitted this communicator shrink");
        return make_shrink_error(
            shrink->failure_code,
            shrink->failure_detail);
    }

    const int expected_participants = state->world_size -
        static_cast<int>(request.excluded_ranks.size());
    if (static_cast<int>(shrink->participants.size()) !=
        expected_participants) {
        const auto deadline = std::chrono::steady_clock::now() +
            std::chrono::milliseconds(request.timeout_ms);
        const auto wait_begin = std::chrono::steady_clock::now();
        while (!shrink->completed && !shrink->failed) {
            if (shrink->cv.wait_until(lock, deadline) ==
                std::cv_status::timeout) {
                record_wait_time_locked(
                    registry,
                    state,
                    request.rank,
                    std::chrono::steady_clock::now() - wait_begin);
                record_timeout_locked(
                    registry,
                    state,
                    shrink->participants);
                fail_shrink_locked(
                    shrink,
                    "timeout_waiting_for_shrink",
                    "timeout waiting for surviving ranks to join shrink seqno " +
                        std::to_string(request.seqno));
                return make_shrink_error(
                    shrink->failure_code,
                    shrink->failure_detail);
            }
        }
        record_wait_time_locked(
            registry,
            state,
            request.rank,
            std::chrono::steady_clock::now() - wait_begin);

        if (!shrink->completed) {
            return make_shrink_error(
                shrink->failure_code,
                shrink->failure_detail);
        }
        const auto result_it = shrink->results.find(request.rank);
        if (result_it == shrink->results.end()) {
            return make_shrink_error(
                "internal_error",
                "shrink completed without a per-rank result");
        }
        const ShrinkParticipantResult& per_rank = result_it->second;
        return CommunicatorShrinkResult{
            true,
            request.seqno,
            per_rank.new_comm_id,
            per_rank.new_rank,
            per_rank.new_world_size,
            "",
            "",
        };
    }

    auto child = std::make_shared<CommunicatorState>();
    child->unique_id =
        "shrink-parent" + std::to_string(state->comm_id) +
        "-seq" + std::to_string(request.seqno);
    child->world_size = expected_participants;
    child->comm_id = registry.next_comm_id++;
    child->ready = true;
    child->global_ranks.reserve(
        static_cast<std::size_t>(expected_participants));

    std::vector<int> excluded_global_ranks;
    excluded_global_ranks.reserve(request.excluded_ranks.size());
    for (int excluded_rank : request.excluded_ranks) {
        excluded_global_ranks.push_back(
            global_rank_for_local(state, excluded_rank));
    }

    int child_rank = 0;
    for (int parent_rank = 0; parent_rank < state->world_size; ++parent_rank) {
        if (std::binary_search(
                request.excluded_ranks.begin(),
                request.excluded_ranks.end(),
                parent_rank)) {
            continue;
        }
        const int global_rank = global_rank_for_local(state, parent_rank);
        child->global_ranks.push_back(global_rank);
        child->participants.emplace(child_rank, true);
        shrink->results.emplace(
            parent_rank,
            ShrinkParticipantResult{
                child->comm_id,
                child_rank,
                child->world_size,
            });
        ensure_rank_report_locked(registry, global_rank).communicator_inits++;
        child_rank++;
    }

    registry.active_by_comm_id.emplace(child->comm_id, child);
    registry.report.communicator_count++;
    state->next_seqno++;

    ClusterRecoveryEvent recovery;
    recovery.index = ++registry.report.resilience_event_entries;
    recovery.parent_comm_id = state->comm_id;
    recovery.new_comm_id = child->comm_id;
    recovery.seqno = request.seqno;
    recovery.abort_parent = request.abort_parent;
    recovery.excluded_ranks = std::move(excluded_global_ranks);
    recovery.surviving_ranks = child->global_ranks;
    const SteadyTimePoint recovery_begin =
        state->failed_at == SteadyTimePoint{}
        ? shrink->first_submit_at
        : state->failed_at;
    recovery.recovery_time_us = elapsed_us(
        recovery_begin,
        std::chrono::steady_clock::now());
    registry.report.recovery_events.push_back(std::move(recovery));

    shrink->completed = true;
    state->shrinks.erase(request.seqno);
    shrink->cv.notify_all();

    const auto result_it = shrink->results.find(request.rank);
    if (result_it == shrink->results.end()) {
        return make_shrink_error(
            "internal_error",
            "shrink completed without a per-rank result");
    }
    const ShrinkParticipantResult& per_rank = result_it->second;
    return CommunicatorShrinkResult{
        true,
        request.seqno,
        per_rank.new_comm_id,
        per_rank.new_rank,
        per_rank.new_world_size,
        "",
        "",
    };
}

PointToPointSubmitResult CommunicatorRegistry::submit_point_to_point(const PointToPointSubmitRequest& request) {
    if (request.comm_id <= 0) {
        return make_point_to_point_error("invalid_comm_id", "comm_id must be > 0");
    }
    if (request.rank < 0) {
        return make_point_to_point_error("invalid_rank", "rank must be >= 0");
    }
    if (request.seqno == 0) {
        return make_point_to_point_error("invalid_seqno", "seqno must be > 0");
    }
    if (request.peer < 0) {
        return make_point_to_point_error("invalid_peer", "peer must be >= 0");
    }
    if (request.count == 0) {
        return make_point_to_point_error("invalid_count", "count must be > 0");
    }
    if (request.bytes == 0) {
        return make_point_to_point_error("invalid_bytes", "bytes must be > 0");
    }
    if (request.transport == BufferTransport::SharedMemory && request.staging_name.empty()) {
        return make_point_to_point_error("missing_staging_name", "staging_name must be set");
    }
    if (request.transport == BufferTransport::SocketPayload &&
        request.type == PointToPointType::Send &&
        request.payload.size() != request.payload_bytes) {
        return make_point_to_point_error("socket_payload_size_mismatch", "point-to-point socket payload size did not match bytes");
    }
    if (request.timeout_ms <= 0) {
        return make_point_to_point_error("invalid_timeout", "timeout_ms must be > 0");
    }

    RegistryImpl& registry = registry_impl();
    std::shared_ptr<CommunicatorState> state;
    std::shared_ptr<PointToPointState> p2p;

    std::unique_lock<std::mutex> lock(registry.mutex);
    auto it = registry.active_by_comm_id.find(request.comm_id);
    if (it == registry.active_by_comm_id.end()) {
        return make_point_to_point_error("unknown_comm_id", "communicator not found");
    }

    state = it->second;
    if (state->participants.find(request.rank) == state->participants.end()) {
        return make_point_to_point_error("unknown_rank", "rank is not a member of this communicator");
    }
    if (state->destroyed_ranks.find(request.rank) != state->destroyed_ranks.end()) {
        return make_point_to_point_error("rank_destroyed", "rank already destroyed this communicator");
    }
    if (state->failed) {
        return make_point_to_point_error(state->failure_code, state->failure_detail);
    }
    if (request.rank >= state->world_size) {
        return make_point_to_point_error("invalid_rank", "rank must be within [0, world_size)");
    }
    if (request.peer >= state->world_size) {
        return make_point_to_point_error("invalid_peer", "peer must be within [0, world_size)");
    }
    if (request.peer == request.rank) {
        return make_point_to_point_error("invalid_peer", "peer must not equal rank");
    }

    if (request.seqno != state->next_seqno) {
        if (request.seqno < state->next_seqno) {
            return make_point_to_point_error(
                "stale_seqno",
                "expected seqno " + std::to_string(state->next_seqno) +
                    ", got " + std::to_string(request.seqno));
        }
        return make_point_to_point_error(
            "out_of_order_seqno",
            "expected seqno " + std::to_string(state->next_seqno) +
                ", got " + std::to_string(request.seqno));
    }

    auto p2p_it = state->point_to_points.find(request.seqno);
    if (p2p_it == state->point_to_points.end()) {
        p2p = std::make_shared<PointToPointState>();
        p2p->request = request;
        p2p->first_submit_at = std::chrono::steady_clock::now();
        state->point_to_points.emplace(request.seqno, p2p);
    } else {
        p2p = p2p_it->second;
    }

    std::string error_code;
    std::string error_detail;
    if (!point_to_point_requests_match(p2p->request, request, error_code, error_detail)) {
        fail_point_to_point_locked(state, p2p, std::move(error_code), std::move(error_detail));
        return make_point_to_point_error(state->failure_code, state->failure_detail);
    }

    if (!p2p->participants.emplace(request.rank, request).second) {
        fail_point_to_point_locked(
            state,
            p2p,
            "duplicate_p2p_rank",
            "rank already submitted this point-to-point seqno");
        return make_point_to_point_error(state->failure_code, state->failure_detail);
    }

    if (static_cast<int>(p2p->participants.size()) != state->world_size) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(request.timeout_ms);
        const auto wait_begin = std::chrono::steady_clock::now();
        while (!p2p->completed && !p2p->failed && !state->failed) {
            if (p2p->cv.wait_until(lock, deadline) == std::cv_status::timeout) {
                record_wait_time_locked(
                    registry,
                    state,
                    request.rank,
                    std::chrono::steady_clock::now() - wait_begin);
                record_timeout_locked(registry, state, p2p->participants);
                fail_point_to_point_locked(
                    state,
                    p2p,
                    "timeout_waiting_for_p2p",
                    "timeout waiting for all ranks to join point-to-point seqno " +
                        std::to_string(request.seqno));
                return make_point_to_point_error(state->failure_code, state->failure_detail);
            }
        }
        record_wait_time_locked(
            registry,
            state,
            request.rank,
            std::chrono::steady_clock::now() - wait_begin);

        if (p2p->completed) {
            auto result_it = p2p->results.find(request.rank);
            if (result_it == p2p->results.end()) {
                return make_point_to_point_error("internal_error", "point-to-point completed without a per-rank result");
            }
            return result_it->second;
        }
        return make_point_to_point_error(state->failure_code, state->failure_detail);
    }

    p2p->all_participants_at = std::chrono::steady_clock::now();
    lock.unlock();
    CollectiveExecutionResult execution = execute_point_to_point_locked(request, p2p);
    lock.lock();

    if (state->failed) {
        return make_point_to_point_error(state->failure_code, state->failure_detail);
    }
    if (!execution.ok) {
        fail_point_to_point_locked(state, p2p, execution.error_code, execution.error_detail);
        return make_point_to_point_error(state->failure_code, state->failure_detail);
    }

    const TopologyOperationTotals topology_totals =
        record_point_to_point_completion_locked(registry, state, p2p);
    for (const auto& entry : p2p->participants) {
        PointToPointSubmitResult result;
        result.ok = true;
        result.seqno = request.seqno;
        auto payload_it = execution.output_payloads.find(entry.first);
        if (payload_it != execution.output_payloads.end()) {
            result.output_payload = payload_it->second;
        }
        p2p->results[entry.first] = std::move(result);
    }
    const SteadyTimePoint completed_at = std::chrono::steady_clock::now();
    append_operation_timeline_locked(
        registry,
        state,
        request.seqno,
        "point_to_point",
        "send_recv",
        collective_data_type_name(request.dtype),
        collective_reduce_op_name(CollectiveReduceOp::None),
        buffer_transport_name(request.transport),
        point_to_point_logical_payload_bytes(p2p),
        total_participant_payload_bytes(p2p->participants),
        total_payload_bytes(execution.output_payloads),
        p2p->first_submit_at,
        p2p->all_participants_at,
        completed_at,
        topology_totals.estimated_time_us);
    p2p->completed = true;
    state->next_seqno++;
    state->point_to_points.erase(request.seqno);
    p2p->cv.notify_all();
    return p2p->results.at(request.rank);
}

CollectiveSubmitResult CommunicatorRegistry::submit_collective(const CollectiveSubmitRequest& request) {
    if (request.comm_id <= 0) {
        return make_collective_error("invalid_comm_id", "comm_id must be > 0");
    }
    if (request.rank < 0) {
        return make_collective_error("invalid_rank", "rank must be >= 0");
    }
    if (request.seqno == 0) {
        return make_collective_error("invalid_seqno", "seqno must be > 0");
    }
    if (request.count == 0) {
        return make_collective_error("invalid_count", "count must be > 0");
    }
    if (request.bytes == 0) {
        return make_collective_error("invalid_bytes", "bytes must be > 0");
    }
    if (!request.proxy_only &&
        request.transport == BufferTransport::SharedMemory &&
        request.staging_name.empty()) {
        return make_collective_error("missing_staging_name", "staging_name must be set");
    }
    if (!request.proxy_only &&
        request.transport == BufferTransport::SocketPayload &&
        request.payload.size() != request.payload_bytes) {
        return make_collective_error("socket_payload_size_mismatch", "collective socket payload size did not match bytes");
    }
    if (request.timeout_ms <= 0) {
        return make_collective_error("invalid_timeout", "timeout_ms must be > 0");
    }

    RegistryImpl& registry = registry_impl();
    std::shared_ptr<CommunicatorState> state;
    std::shared_ptr<CollectiveState> collective;

    std::unique_lock<std::mutex> lock(registry.mutex);
    auto it = registry.active_by_comm_id.find(request.comm_id);
    if (it == registry.active_by_comm_id.end()) {
        return make_collective_error("unknown_comm_id", "communicator not found");
    }

    state = it->second;
    if (state->participants.find(request.rank) == state->participants.end()) {
        return make_collective_error("unknown_rank", "rank is not a member of this communicator");
    }
    if (state->destroyed_ranks.find(request.rank) != state->destroyed_ranks.end()) {
        return make_collective_error("rank_destroyed", "rank already destroyed this communicator");
    }
    if (state->failed) {
        return make_collective_error(state->failure_code, state->failure_detail);
    }

    if (request.rank >= state->world_size) {
        return make_collective_error("invalid_rank", "rank must be within [0, world_size)");
    }
    const bool variable_alltoall =
        request.type == CollectiveType::AllToAll &&
        !request.input_splits.empty();
    if (variable_alltoall) {
        if (request.input_splits.size() !=
                static_cast<std::size_t>(state->world_size) ||
            request.output_splits.size() !=
                static_cast<std::size_t>(state->world_size)) {
            return make_collective_error(
                "invalid_split_plan",
                "all-to-all split vectors must match world_size");
        }
        const std::size_t dtype_size = collective_data_type_size(request.dtype);
        std::size_t input_count = 0;
        std::size_t output_count = 0;
        for (std::size_t count : request.input_splits) {
            input_count += count;
        }
        for (std::size_t count : request.output_splits) {
            output_count += count;
        }
        if (dtype_size == 0 ||
            request.input_bytes != input_count * dtype_size ||
            request.output_bytes != output_count * dtype_size ||
            request.bytes < std::max(request.input_bytes, request.output_bytes)) {
            return make_collective_error(
                "invalid_split_plan",
                "all-to-all split bytes are inconsistent");
        }
        if (request.transport == BufferTransport::SocketPayload &&
            request.payload_bytes != request.input_bytes) {
            return make_collective_error(
                "invalid_split_plan",
                "all-to-all socket payload must match input bytes");
        }
    } else if (!request.input_splits.empty() || !request.output_splits.empty()) {
        return make_collective_error(
            "invalid_split_plan",
            "explicit splits are only valid for all-to-all");
    }
    if (request.type == CollectiveType::Broadcast || request.type == CollectiveType::Reduce) {
        if (request.root < 0 || request.root >= state->world_size) {
            return make_collective_error(
                "invalid_root",
                std::string(collective_type_name(request.type)) +
                    " root must be within [0, world_size)");
        }
    }

    if (request.seqno != state->next_seqno) {
        if (request.seqno < state->next_seqno) {
            return make_collective_error(
                "stale_seqno",
                "expected seqno " + std::to_string(state->next_seqno) +
                    ", got " + std::to_string(request.seqno));
        }
        return make_collective_error(
            "out_of_order_seqno",
            "expected seqno " + std::to_string(state->next_seqno) +
                ", got " + std::to_string(request.seqno));
    }

    auto collective_it = state->collectives.find(request.seqno);
    if (collective_it == state->collectives.end()) {
        collective = std::make_shared<CollectiveState>();
        collective->request = request;
        collective->first_submit_at = std::chrono::steady_clock::now();
        state->collectives.emplace(request.seqno, collective);
    } else {
        collective = collective_it->second;
    }

    std::string error_code;
    std::string error_detail;
    if (!collective_requests_match(collective->request, request, error_code, error_detail)) {
        fail_collective_locked(state, collective, std::move(error_code), std::move(error_detail));
        return make_collective_error(state->failure_code, state->failure_detail);
    }

    CollectiveExecutionParticipant participant;
    participant.rank = request.rank;
    participant.transport = request.transport;
    participant.staging_name = request.staging_name;
    participant.bytes = request.bytes;
    participant.payload_bytes = request.payload_bytes;
    participant.payload = request.payload;
    participant.input_splits = request.input_splits;
    participant.output_splits = request.output_splits;
    participant.input_bytes = request.input_bytes;
    participant.output_bytes = request.output_bytes;
    if (!collective->participants.emplace(request.rank, participant).second) {
        fail_collective_locked(
            state,
            collective,
            "duplicate_collective_rank",
            "rank already submitted this seqno");
        return make_collective_error(state->failure_code, state->failure_detail);
    }

    if (static_cast<int>(collective->participants.size()) == state->world_size) {
        collective->all_participants_at = std::chrono::steady_clock::now();
        if (request.proxy_only) {
            const TopologyOperationTotals topology_totals =
                record_collective_completion_locked(
                    registry,
                    state,
                    collective);
            for (const auto& entry : collective->participants) {
                CollectiveSubmitResult result;
                result.ok = true;
                result.seqno = request.seqno;
                collective->results[entry.first] = std::move(result);
            }
            const SteadyTimePoint completed_at =
                std::chrono::steady_clock::now();
            append_operation_timeline_locked(
                registry,
                state,
                request.seqno,
                "collective",
                collective_type_name(request.type),
                collective_data_type_name(request.dtype),
                collective_reduce_op_name(request.reduce_op),
                "proxy_only",
                static_cast<std::uint64_t>(request.bytes) *
                    static_cast<std::uint64_t>(state->world_size),
                total_participant_payload_bytes(collective->participants),
                0,
                collective->first_submit_at,
                collective->all_participants_at,
                completed_at,
                topology_totals.estimated_time_us);
            collective->completed = true;
            state->next_seqno++;
            state->collectives.erase(request.seqno);
            collective->cv.notify_all();
            return collective->results.at(request.rank);
        }
    } else {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(request.timeout_ms);
        const auto wait_begin = std::chrono::steady_clock::now();
        while (!collective->completed && !collective->failed && !state->failed) {
            if (collective->cv.wait_until(lock, deadline) == std::cv_status::timeout) {
                record_wait_time_locked(
                    registry,
                    state,
                    request.rank,
                    std::chrono::steady_clock::now() - wait_begin);
                record_timeout_locked(registry, state, collective->participants);
                fail_collective_locked(
                    state,
                    collective,
                    "timeout_waiting_for_collective",
                    "timeout waiting for all ranks to join collective seqno " + std::to_string(request.seqno));
                return make_collective_error(state->failure_code, state->failure_detail);
            }
        }
        record_wait_time_locked(
            registry,
            state,
            request.rank,
            std::chrono::steady_clock::now() - wait_begin);

        if (collective->completed) {
            auto result_it = collective->results.find(request.rank);
            if (result_it == collective->results.end()) {
                return make_collective_error("internal_error", "collective completed without a per-rank result");
            }
            return result_it->second;
        }
        return make_collective_error(state->failure_code, state->failure_detail);
    }

    lock.unlock();
    CollectiveExecutionResult execution = execute_collective_locked(request, collective);
    lock.lock();

    if (state->failed) {
        return make_collective_error(state->failure_code, state->failure_detail);
    }
    if (!execution.ok) {
        fail_collective_locked(state, collective, execution.error_code, execution.error_detail);
        return make_collective_error(state->failure_code, state->failure_detail);
    }

    const TopologyOperationTotals topology_totals =
        record_collective_completion_locked(registry, state, collective);
    for (const auto& entry : collective->participants) {
        CollectiveSubmitResult result;
        result.ok = true;
        result.seqno = request.seqno;
        auto payload_it = execution.output_payloads.find(entry.first);
        if (payload_it != execution.output_payloads.end()) {
            result.output_payload = payload_it->second;
        }
        collective->results[entry.first] = std::move(result);
    }
    const SteadyTimePoint completed_at = std::chrono::steady_clock::now();
    append_operation_timeline_locked(
        registry,
        state,
        request.seqno,
        "collective",
        collective_type_name(request.type),
        collective_data_type_name(request.dtype),
        collective_reduce_op_name(request.reduce_op),
        buffer_transport_name(request.transport),
        variable_alltoall
            ? [&collective]() {
                  std::uint64_t bytes = 0;
                  for (const auto& entry : collective->participants) {
                      bytes += static_cast<std::uint64_t>(
                          entry.second.input_bytes);
                  }
                  return bytes;
              }()
            : static_cast<std::uint64_t>(request.bytes) *
                  static_cast<std::uint64_t>(state->world_size),
        total_participant_payload_bytes(collective->participants),
        total_payload_bytes(execution.output_payloads),
        collective->first_submit_at,
        collective->all_participants_at,
        completed_at,
        topology_totals.estimated_time_us);
    collective->completed = true;
    state->next_seqno++;
    state->collectives.erase(request.seqno);
    collective->cv.notify_all();
    return collective->results.at(request.rank);
}

BarrierSubmitResult CommunicatorRegistry::submit_barrier(const BarrierSubmitRequest& request) {
    if (request.comm_id <= 0) {
        return make_barrier_error("invalid_comm_id", "comm_id must be > 0");
    }
    if (request.rank < 0) {
        return make_barrier_error("invalid_rank", "rank must be >= 0");
    }
    if (request.seqno == 0) {
        return make_barrier_error("invalid_seqno", "seqno must be > 0");
    }
    if (request.timeout_ms <= 0) {
        return make_barrier_error("invalid_timeout", "timeout_ms must be > 0");
    }

    RegistryImpl& registry = registry_impl();
    std::shared_ptr<CommunicatorState> state;
    std::shared_ptr<BarrierState> barrier;

    std::unique_lock<std::mutex> lock(registry.mutex);
    auto it = registry.active_by_comm_id.find(request.comm_id);
    if (it == registry.active_by_comm_id.end()) {
        return make_barrier_error("unknown_comm_id", "communicator not found");
    }

    state = it->second;
    if (state->participants.find(request.rank) == state->participants.end()) {
        return make_barrier_error("unknown_rank", "rank is not a member of this communicator");
    }
    if (state->destroyed_ranks.find(request.rank) != state->destroyed_ranks.end()) {
        return make_barrier_error("rank_destroyed", "rank already destroyed this communicator");
    }
    if (state->failed) {
        return make_barrier_error(state->failure_code, state->failure_detail);
    }
    if (request.rank >= state->world_size) {
        return make_barrier_error("invalid_rank", "rank must be within [0, world_size)");
    }

    if (request.seqno != state->next_seqno) {
        if (request.seqno < state->next_seqno) {
            return make_barrier_error(
                "stale_seqno",
                "expected seqno " + std::to_string(state->next_seqno) +
                    ", got " + std::to_string(request.seqno));
        }
        return make_barrier_error(
            "out_of_order_seqno",
            "expected seqno " + std::to_string(state->next_seqno) +
                ", got " + std::to_string(request.seqno));
    }

    auto barrier_it = state->barriers.find(request.seqno);
    if (barrier_it == state->barriers.end()) {
        barrier = std::make_shared<BarrierState>();
        barrier->request = request;
        barrier->first_submit_at = std::chrono::steady_clock::now();
        state->barriers.emplace(request.seqno, barrier);
    } else {
        barrier = barrier_it->second;
    }

    if (barrier->request.timeout_ms != request.timeout_ms) {
        fail_barrier_locked(
            state,
            barrier,
            "timeout_mismatch",
            "barrier timeout mismatch: expected timeout_ms=" +
                std::to_string(barrier->request.timeout_ms) +
                ", got " + std::to_string(request.timeout_ms));
        return make_barrier_error(state->failure_code, state->failure_detail);
    }

    if (!barrier->participants.emplace(request.rank, true).second) {
        fail_barrier_locked(
            state,
            barrier,
            "duplicate_barrier_rank",
            "rank already submitted this barrier seqno");
        return make_barrier_error(state->failure_code, state->failure_detail);
    }

    if (static_cast<int>(barrier->participants.size()) != state->world_size) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(request.timeout_ms);
        const auto wait_begin = std::chrono::steady_clock::now();
        while (!barrier->completed && !barrier->failed && !state->failed) {
            if (barrier->cv.wait_until(lock, deadline) == std::cv_status::timeout) {
                record_wait_time_locked(
                    registry,
                    state,
                    request.rank,
                    std::chrono::steady_clock::now() - wait_begin);
                record_timeout_locked(registry, state, barrier->participants);
                fail_barrier_locked(
                    state,
                    barrier,
                    "timeout_waiting_for_barrier",
                    "timeout waiting for all ranks to join barrier seqno " +
                        std::to_string(request.seqno));
                return make_barrier_error(state->failure_code, state->failure_detail);
            }
        }
        record_wait_time_locked(
            registry,
            state,
            request.rank,
            std::chrono::steady_clock::now() - wait_begin);

        if (barrier->completed) {
            return BarrierSubmitResult{true, request.seqno, "", ""};
        }
        return make_barrier_error(state->failure_code, state->failure_detail);
    }

    barrier->all_participants_at = std::chrono::steady_clock::now();
    record_barrier_completion_locked(registry, state, barrier);
    append_operation_timeline_locked(
        registry,
        state,
        request.seqno,
        "barrier",
        "barrier",
        "none",
        "none",
        "control",
        0,
        0,
        0,
        barrier->first_submit_at,
        barrier->all_participants_at,
        std::chrono::steady_clock::now(),
        0.0);
    barrier->completed = true;
    state->next_seqno++;
    state->barriers.erase(request.seqno);
    barrier->cv.notify_all();
    return BarrierSubmitResult{true, request.seqno, "", ""};
}

CollectiveBatchPrepareResult CommunicatorRegistry::prepare_collective_batch(
    const CollectiveBatchPrepareRequest& request) {
    if (request.comm_id <= 0) {
        return make_batch_error("invalid_comm_id", "comm_id must be > 0");
    }
    if (request.rank < 0) {
        return make_batch_error("invalid_rank", "rank must be >= 0");
    }
    if (request.base_seqno == 0) {
        return make_batch_error("invalid_seqno", "base_seqno must be > 0");
    }
    if (request.timeout_ms <= 0) {
        return make_batch_error("invalid_timeout", "timeout_ms must be > 0");
    }
    if (request.operations.empty()) {
        return make_batch_error("empty_group", "group must contain at least one operation");
    }

    RegistryImpl& registry = registry_impl();
    std::shared_ptr<CommunicatorState> state;
    std::shared_ptr<BatchState> batch;

    std::unique_lock<std::mutex> lock(registry.mutex);
    auto it = registry.active_by_comm_id.find(request.comm_id);
    if (it == registry.active_by_comm_id.end()) {
        return make_batch_error("unknown_comm_id", "communicator not found");
    }

    state = it->second;
    if (state->participants.find(request.rank) == state->participants.end()) {
        return make_batch_error("unknown_rank", "rank is not a member of this communicator");
    }
    if (state->destroyed_ranks.find(request.rank) != state->destroyed_ranks.end()) {
        return make_batch_error("rank_destroyed", "rank already destroyed this communicator");
    }
    if (state->failed) {
        return make_batch_error(state->failure_code, state->failure_detail);
    }
    if (request.rank >= state->world_size) {
        return make_batch_error("invalid_rank", "rank must be within [0, world_size)");
    }

    if (request.base_seqno != state->next_seqno) {
        if (request.base_seqno < state->next_seqno) {
            return make_batch_error(
                "stale_seqno",
                "expected seqno " + std::to_string(state->next_seqno) +
                    ", got " + std::to_string(request.base_seqno));
        }
        return make_batch_error(
            "out_of_order_seqno",
            "expected seqno " + std::to_string(state->next_seqno) +
                ", got " + std::to_string(request.base_seqno));
    }

    auto batch_it = state->batches.find(request.base_seqno);
    if (batch_it == state->batches.end()) {
        batch = std::make_shared<BatchState>();
        batch->request = request;
        state->batches.emplace(request.base_seqno, batch);
    } else {
        batch = batch_it->second;
    }

    if (batch->request.timeout_ms != request.timeout_ms) {
        fail_batch_locked(
            state,
            batch,
            "timeout_mismatch",
            "group timeout mismatch: expected timeout_ms=" +
                std::to_string(batch->request.timeout_ms) + ", got " +
                std::to_string(request.timeout_ms));
        return make_batch_error(state->failure_code, state->failure_detail);
    }

    std::string error_code;
    std::string error_detail;
    if (!batch_requests_match(batch->request, request, error_code, error_detail)) {
        fail_batch_locked(state, batch, std::move(error_code), std::move(error_detail));
        return make_batch_error(state->failure_code, state->failure_detail);
    }

    if (!batch->participants.emplace(request.rank, request).second) {
        fail_batch_locked(
            state,
            batch,
            "duplicate_group_rank",
            "rank already submitted this group base_seqno");
        return make_batch_error(state->failure_code, state->failure_detail);
    }
    ensure_rank_report_locked(
        registry,
        global_rank_for_local(state, request.rank)).group_prepares++;

    if (static_cast<int>(batch->participants.size()) != state->world_size) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(request.timeout_ms);
        const auto wait_begin = std::chrono::steady_clock::now();
        while (!batch->completed && !batch->failed && !state->failed) {
            if (batch->cv.wait_until(lock, deadline) == std::cv_status::timeout) {
                record_wait_time_locked(
                    registry,
                    state,
                    request.rank,
                    std::chrono::steady_clock::now() - wait_begin);
                record_timeout_locked(registry, state, batch->participants);
                fail_batch_locked(
                    state,
                    batch,
                    "timeout_waiting_for_group",
                    "timeout waiting for all ranks to join group base_seqno " +
                        std::to_string(request.base_seqno));
                return make_batch_error(state->failure_code, state->failure_detail);
            }
        }
        record_wait_time_locked(
            registry,
            state,
            request.rank,
            std::chrono::steady_clock::now() - wait_begin);

        if (batch->completed) {
            return CollectiveBatchPrepareResult{true, request.base_seqno, "", ""};
        }
        return make_batch_error(state->failure_code, state->failure_detail);
    }

    batch->completed = true;
    state->batches.erase(request.base_seqno);
    batch->cv.notify_all();
    return CollectiveBatchPrepareResult{true, request.base_seqno, "", ""};
}

ClusterReportSnapshot snapshot_cluster_report() {
    RegistryImpl& registry = registry_impl();
    std::lock_guard<std::mutex> lock(registry.mutex);

    ClusterReportSnapshot snapshot;
    snapshot.world_size = registry.report.world_size;
    snapshot.communicator_count = registry.report.communicator_count;
    snapshot.all_reduce = registry.report.all_reduce;
    snapshot.reduce = registry.report.reduce;
    snapshot.broadcast = registry.report.broadcast;
    snapshot.all_gather = registry.report.all_gather;
    snapshot.reduce_scatter = registry.report.reduce_scatter;
    snapshot.all_to_all = registry.report.all_to_all;
    snapshot.barrier = registry.report.barrier;
    snapshot.point_to_point = registry.report.point_to_point;
    snapshot.links.reserve(registry.report.links.size());
    snapshot.node_pairs.reserve(registry.report.node_pairs.size());
    snapshot.ranks.reserve(registry.report.ranks.size());
    snapshot.operation_timeline.reserve(
        registry.report.operation_timeline.size());
    snapshot.dropped_operation_timeline_entries =
        registry.report.dropped_operation_timeline_entries;
    snapshot.failure_events = registry.report.failure_events;
    snapshot.recovery_events = registry.report.recovery_events;

    for (const auto& entry : registry.report.links) {
        snapshot.links.push_back(entry.second);
    }
    for (const auto& entry : registry.report.node_pairs) {
        snapshot.node_pairs.push_back(entry.second);
    }
    for (const auto& entry : registry.report.ranks) {
        snapshot.ranks.push_back(entry.second);
    }
    for (const ClusterOperationTimelineEntry& entry :
         registry.report.operation_timeline) {
        snapshot.operation_timeline.push_back(entry);
    }
    std::sort(
        snapshot.links.begin(),
        snapshot.links.end(),
        [](const ClusterLinkReportStats& lhs, const ClusterLinkReportStats& rhs) {
            if (lhs.src_node != rhs.src_node) {
                return lhs.src_node < rhs.src_node;
            }
            if (lhs.dst_node != rhs.dst_node) {
                return lhs.dst_node < rhs.dst_node;
            }
            return lhs.scope < rhs.scope;
        });
    std::sort(
        snapshot.node_pairs.begin(),
        snapshot.node_pairs.end(),
        [](const ClusterNodePairReportStats& lhs, const ClusterNodePairReportStats& rhs) {
            if (lhs.node_a != rhs.node_a) {
                return lhs.node_a < rhs.node_a;
            }
            return lhs.node_b < rhs.node_b;
        });
    std::sort(
        snapshot.ranks.begin(),
        snapshot.ranks.end(),
        [](const ClusterRankReportStats& lhs, const ClusterRankReportStats& rhs) {
            return lhs.rank < rhs.rank;
        });

    snapshot.has_data =
        snapshot.communicator_count > 0 ||
        snapshot.all_reduce.calls > 0 ||
        snapshot.reduce.calls > 0 ||
        snapshot.broadcast.calls > 0 ||
        snapshot.all_gather.calls > 0 ||
        snapshot.reduce_scatter.calls > 0 ||
        snapshot.all_to_all.calls > 0 ||
        snapshot.barrier.calls > 0 ||
        snapshot.point_to_point.operations > 0 ||
        !snapshot.links.empty() ||
        !snapshot.node_pairs.empty() ||
        !snapshot.ranks.empty() ||
        !snapshot.operation_timeline.empty() ||
        !snapshot.failure_events.empty() ||
        !snapshot.recovery_events.empty();
    return snapshot;
}

}  // namespace fake_gpu::distributed
