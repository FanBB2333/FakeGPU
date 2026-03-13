#include "communicator.hpp"

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace fake_gpu::distributed {

namespace {

struct CommunicatorState {
    std::string unique_id;
    int world_size = 0;
    int comm_id = -1;
    bool ready = false;
    bool failed = false;
    std::string failure_code;
    std::string failure_detail;
    std::unordered_map<int, bool> participants;
    std::unordered_map<int, bool> destroyed_ranks;
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

}  // namespace fake_gpu::distributed
