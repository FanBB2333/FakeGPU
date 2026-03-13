#include "communicator.hpp"

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
    std::condition_variable cv;
};

struct RegistryImpl {
    std::mutex mutex;
    int next_comm_id = 1;
    std::unordered_map<std::string, std::shared_ptr<CommunicatorState>> pending_by_unique_id;
    std::unordered_map<int, std::shared_ptr<CommunicatorState>> active_by_comm_id;
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
    if (request.type == CollectiveType::Broadcast) {
        return execute_broadcast(execution_request, participants);
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
        if (static_cast<int>(state->participants.size()) == state->world_size) {
            state->ready = true;
            state->comm_id = registry.next_comm_id++;
            registry.active_by_comm_id.emplace(state->comm_id, state);
            registry.pending_by_unique_id.erase(unique_id);
            state->cv.notify_all();
            return CommunicatorRegistrationResult{true, state->comm_id, 0, "", ""};
        }

        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (!state->ready && !state->failed) {
            if (state->cv.wait_until(lock, deadline) == std::cv_status::timeout) {
                const std::string detail =
                    "timeout waiting for ranks on unique_id " + unique_id;
                fail_pending_group_locked(registry, state, "timeout_waiting_for_ranks", detail);
                return make_error("timeout_waiting_for_ranks", detail);
            }
        }

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
    if (request.staging_name.empty()) {
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
    if (request.type == CollectiveType::Broadcast) {
        if (request.root < 0 || request.root >= state->world_size) {
            return make_collective_error("invalid_root", "broadcast root must be within [0, world_size)");
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
    } else {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(request.timeout_ms);
        while (!collective->completed && !collective->failed && !state->failed) {
            if (collective->cv.wait_until(lock, deadline) == std::cv_status::timeout) {
                fail_collective_locked(
                    state,
                    collective,
                    "timeout_waiting_for_collective",
                    "timeout waiting for all ranks to join collective seqno " + std::to_string(request.seqno));
                return make_collective_error(state->failure_code, state->failure_detail);
            }
        }

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

    collective->completed = true;
    state->next_seqno++;
    state->collectives.erase(request.seqno);
    collective->cv.notify_all();
    return CollectiveSubmitResult{true, request.seqno, "", ""};
}

}  // namespace fake_gpu::distributed
