#include "communicator.hpp"

#include "../core/backend_config.hpp"
#include "topology_model.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fake_gpu::distributed {

namespace {

struct CollectiveState {
    CollectiveSubmitRequest request;
    bool completed = false;
    bool failed = false;
    std::string failure_code;
    std::string failure_detail;
    std::unordered_map<int, CollectiveExecutionParticipant> participants;
    std::condition_variable cv;
};

struct BarrierState {
    BarrierSubmitRequest request;
    bool completed = false;
    bool failed = false;
    std::string failure_code;
    std::string failure_detail;
    std::unordered_map<int, bool> participants;
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

struct CommunicatorState {
    std::string unique_id;
    int world_size = 0;
    int comm_id = -1;
    bool ready = false;
    bool failed = false;
    std::string failure_code;
    std::string failure_detail;
    std::uint64_t next_seqno = 1;
    std::unordered_map<int, bool> participants;
    std::unordered_map<int, bool> destroyed_ranks;
    std::unordered_map<std::uint64_t, std::shared_ptr<CollectiveState>> collectives;
    std::unordered_map<std::uint64_t, std::shared_ptr<BarrierState>> barriers;
    std::unordered_map<std::uint64_t, std::shared_ptr<BatchState>> batches;
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
        ClusterCollectiveReportStats barrier;
        std::unordered_map<std::string, ClusterLinkReportStats> links;
        std::unordered_map<int, ClusterRankReportStats> ranks;
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

void remember_world_size_locked(RegistryImpl& registry, int world_size) {
    if (world_size > 0) {
        registry.report.world_size =
            std::max(registry.report.world_size, static_cast<std::size_t>(world_size));
    }
}

void record_wait_time_locked(
    RegistryImpl& registry,
    int rank,
    std::chrono::steady_clock::duration elapsed) {
    if (rank < 0) {
        return;
    }
    ClusterRankReportStats& stats = ensure_rank_report_locked(registry, rank);
    stats.wait_time_ms += std::chrono::duration<double, std::milli>(elapsed).count();
}

template <typename ParticipantMap>
void record_timeout_locked(RegistryImpl& registry, const ParticipantMap& participants) {
    for (const auto& entry : participants) {
        ClusterRankReportStats& stats = ensure_rank_report_locked(registry, entry.first);
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
    }
    return registry.report.all_reduce;
}

void record_collective_completion_locked(
    RegistryImpl& registry,
    const std::shared_ptr<CommunicatorState>& state,
    const std::shared_ptr<CollectiveState>& collective) {
    remember_world_size_locked(registry, state->world_size);

    ClusterCollectiveReportStats& stats =
        collective_report_stats_for_type_locked(registry, collective->request.type);
    stats.calls++;
    stats.bytes += static_cast<std::uint64_t>(collective->request.bytes) *
                   static_cast<std::uint64_t>(state->world_size);

    for (const auto& entry : collective->participants) {
        ClusterRankReportStats& rank_stats = ensure_rank_report_locked(registry, entry.first);
        rank_stats.collective_calls++;
    }

    const DistributedConfig& dist_config = fake_gpu::BackendConfig::instance().distributed_config();
    if (!dist_config.cluster_config.loaded()) {
        return;
    }

    TopologyModel topology_model;
    std::string topology_error;
    if (!TopologyModel::build(dist_config.cluster_config, topology_model, topology_error)) {
        return;
    }

    CollectiveTopologyEstimate estimate;
    if (!topology_model.estimate_ring_collective(
            collective->request.type,
            static_cast<std::uint64_t>(collective->request.bytes),
            estimate,
            topology_error)) {
        return;
    }

    stats.estimated_time_us_total += estimate.estimated_time_us;

    for (const TopologyLinkEstimate& link : estimate.links) {
        const double transfer_without_penalty_us =
            (static_cast<double>(link.bytes) * 8.0) / (link.bandwidth_gbps * 1000.0);
        const double contention_penalty_us =
            transfer_without_penalty_us * std::max(0.0, link.oversubscription - 1.0);

        stats.contention_penalty_us_total += contention_penalty_us;

        const std::string key = link.src_node + "\n" + link.dst_node + "\n" +
                                topology_link_scope_name(link.scope);
        auto [it, inserted] = registry.report.links.emplace(key, ClusterLinkReportStats{});
        ClusterLinkReportStats& link_stats = it->second;
        if (inserted) {
            link_stats.src_node = link.src_node;
            link_stats.dst_node = link.dst_node;
            link_stats.scope = topology_link_scope_name(link.scope);
            link_stats.bandwidth_gbps = link.bandwidth_gbps;
        }
        link_stats.samples += link.hop_count;
        link_stats.bytes += link.bytes;
        link_stats.avg_latency_us =
            ((link_stats.avg_latency_us * static_cast<double>(link_stats.samples - link.hop_count)) +
             (link.latency_us * static_cast<double>(link.hop_count))) /
            static_cast<double>(link_stats.samples);
        link_stats.estimated_time_us_total += link.estimated_time_us;
        link_stats.contention_penalty_us_total += contention_penalty_us;
    }
}

void record_barrier_completion_locked(
    RegistryImpl& registry,
    const std::shared_ptr<CommunicatorState>& state,
    const std::shared_ptr<BarrierState>& barrier) {
    remember_world_size_locked(registry, state->world_size);
    registry.report.barrier.calls++;

    for (const auto& entry : barrier->participants) {
        ClusterRankReportStats& rank_stats = ensure_rank_report_locked(registry, entry.first);
        rank_stats.barrier_calls++;
    }
}

void fail_pending_group_locked(
    RegistryImpl& registry,
    const std::shared_ptr<CommunicatorState>& state,
    std::string code,
    std::string detail) {
    state->failed = true;
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

bool collective_requests_match(
    const CollectiveSubmitRequest& expected,
    const CollectiveSubmitRequest& actual,
    std::string& error_code,
    std::string& error_detail) {
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
    if (expected.count != actual.count) {
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
    if (expected.bytes != actual.bytes) {
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
    return true;
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
                record_wait_time_locked(registry, rank, std::chrono::steady_clock::now() - wait_begin);
                record_timeout_locked(registry, state->participants);
                const std::string detail =
                    "timeout waiting for ranks on unique_id " + unique_id;
                fail_pending_group_locked(registry, state, "timeout_waiting_for_ranks", detail);
                return make_error("timeout_waiting_for_ranks", detail);
            }
        }
        record_wait_time_locked(registry, rank, std::chrono::steady_clock::now() - wait_begin);

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
    if (!request.proxy_only && request.staging_name.empty()) {
        return make_collective_error("missing_staging_name", "staging_name must be set");
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
    participant.staging_name = request.staging_name;
    participant.bytes = request.bytes;
    if (!collective->participants.emplace(request.rank, participant).second) {
        fail_collective_locked(
            state,
            collective,
            "duplicate_collective_rank",
            "rank already submitted this seqno");
        return make_collective_error(state->failure_code, state->failure_detail);
    }

    if (static_cast<int>(collective->participants.size()) == state->world_size) {
        if (request.proxy_only) {
            record_collective_completion_locked(registry, state, collective);
            collective->completed = true;
            state->next_seqno++;
            state->collectives.erase(request.seqno);
            collective->cv.notify_all();
            return CollectiveSubmitResult{true, request.seqno, "", ""};
        }
    } else {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(request.timeout_ms);
        const auto wait_begin = std::chrono::steady_clock::now();
        while (!collective->completed && !collective->failed && !state->failed) {
            if (collective->cv.wait_until(lock, deadline) == std::cv_status::timeout) {
                record_wait_time_locked(registry, request.rank, std::chrono::steady_clock::now() - wait_begin);
                record_timeout_locked(registry, collective->participants);
                fail_collective_locked(
                    state,
                    collective,
                    "timeout_waiting_for_collective",
                    "timeout waiting for all ranks to join collective seqno " + std::to_string(request.seqno));
                return make_collective_error(state->failure_code, state->failure_detail);
            }
        }
        record_wait_time_locked(registry, request.rank, std::chrono::steady_clock::now() - wait_begin);

        if (collective->completed) {
            return CollectiveSubmitResult{true, request.seqno, "", ""};
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

    record_collective_completion_locked(registry, state, collective);
    collective->completed = true;
    state->next_seqno++;
    state->collectives.erase(request.seqno);
    collective->cv.notify_all();
    return CollectiveSubmitResult{true, request.seqno, "", ""};
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
                record_wait_time_locked(registry, request.rank, std::chrono::steady_clock::now() - wait_begin);
                record_timeout_locked(registry, barrier->participants);
                fail_barrier_locked(
                    state,
                    barrier,
                    "timeout_waiting_for_barrier",
                    "timeout waiting for all ranks to join barrier seqno " +
                        std::to_string(request.seqno));
                return make_barrier_error(state->failure_code, state->failure_detail);
            }
        }
        record_wait_time_locked(registry, request.rank, std::chrono::steady_clock::now() - wait_begin);

        if (barrier->completed) {
            return BarrierSubmitResult{true, request.seqno, "", ""};
        }
        return make_barrier_error(state->failure_code, state->failure_detail);
    }

    record_barrier_completion_locked(registry, state, barrier);
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
    ensure_rank_report_locked(registry, request.rank).group_prepares++;

    if (static_cast<int>(batch->participants.size()) != state->world_size) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(request.timeout_ms);
        const auto wait_begin = std::chrono::steady_clock::now();
        while (!batch->completed && !batch->failed && !state->failed) {
            if (batch->cv.wait_until(lock, deadline) == std::cv_status::timeout) {
                record_wait_time_locked(registry, request.rank, std::chrono::steady_clock::now() - wait_begin);
                record_timeout_locked(registry, batch->participants);
                fail_batch_locked(
                    state,
                    batch,
                    "timeout_waiting_for_group",
                    "timeout waiting for all ranks to join group base_seqno " +
                        std::to_string(request.base_seqno));
                return make_batch_error(state->failure_code, state->failure_detail);
            }
        }
        record_wait_time_locked(registry, request.rank, std::chrono::steady_clock::now() - wait_begin);

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
    snapshot.barrier = registry.report.barrier;
    snapshot.links.reserve(registry.report.links.size());
    snapshot.ranks.reserve(registry.report.ranks.size());

    for (const auto& entry : registry.report.links) {
        snapshot.links.push_back(entry.second);
    }
    for (const auto& entry : registry.report.ranks) {
        snapshot.ranks.push_back(entry.second);
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
        snapshot.barrier.calls > 0 ||
        !snapshot.links.empty() ||
        !snapshot.ranks.empty();
    return snapshot;
}

}  // namespace fake_gpu::distributed
