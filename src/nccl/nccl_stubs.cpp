#include "nccl_defs.hpp"
#include "nccl_mode_dispatch.hpp"

#include "../core/backend_config.hpp"
#include "../distributed/transport.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <random>
#include <sstream>
#include <string>

struct ncclComm {
    int comm_id = -1;
    int world_size = 0;
    int rank = -1;
    bool destroyed = false;
};

namespace {

constexpr int kFakeNcclVersion = 29000;
constexpr int kCoordinatorTimeoutMs = 1000;

std::string unique_id_to_token(const ncclUniqueId& unique_id) {
    const char* begin = unique_id.internal;
    const char* end = std::find(begin, begin + NCCL_UNIQUE_ID_BYTES, '\0');
    return std::string(begin, end);
}

bool parse_int_field(
    const fake_gpu::distributed::CoordinatorResponse& response,
    const char* key,
    int& value) {
    auto it = response.fields.find(key);
    if (it == response.fields.end()) {
        return false;
    }
    try {
        std::size_t consumed = 0;
        value = std::stoi(it->second, &consumed, 10);
        return consumed == it->second.size();
    } catch (...) {
        return false;
    }
}

ncclResult_t map_response_error(const std::string& error_code) {
    if (error_code == "bad_request" ||
        error_code == "invalid_rank" ||
        error_code == "invalid_world_size" ||
        error_code == "missing_unique_id") {
        return ncclInvalidArgument;
    }
    if (error_code == "duplicate_destroy" ||
        error_code == "duplicate_rank" ||
        error_code == "world_size_mismatch" ||
        error_code == "unknown_comm_id") {
        return ncclInvalidUsage;
    }
    if (error_code == "timeout_waiting_for_ranks") {
        return ncclSystemError;
    }
    return ncclInternalError;
}

}  // namespace

extern "C" {

const char* ncclGetErrorString(ncclResult_t result) {
    switch (result) {
        case ncclSuccess:
            return "success";
        case ncclUnhandledCudaError:
            return "unhandled cuda error";
        case ncclSystemError:
            return "system error";
        case ncclInternalError:
            return "internal error";
        case ncclInvalidArgument:
            return "invalid argument";
        case ncclInvalidUsage:
            return "invalid usage";
    }
    return "unknown nccl error";
}

ncclResult_t ncclGetVersion(int* version) {
    if (!version) {
        return ncclInvalidArgument;
    }
    *version = kFakeNcclVersion;
    return ncclSuccess;
}

ncclResult_t ncclGetUniqueId(ncclUniqueId* unique_id) {
    if (!unique_id) {
        return ncclInvalidArgument;
    }

    std::memset(unique_id, 0, sizeof(*unique_id));

    static constexpr char kHexDigits[] = "0123456789abcdef";
    std::array<unsigned char, 16> random_bytes {};
    std::random_device device;
    for (unsigned char& value : random_bytes) {
        value = static_cast<unsigned char>(device());
    }

    std::string token;
    token.reserve(random_bytes.size() * 2);
    for (unsigned char value : random_bytes) {
        token.push_back(kHexDigits[(value >> 4U) & 0x0FU]);
        token.push_back(kHexDigits[value & 0x0FU]);
    }

    std::memcpy(unique_id->internal, token.data(), token.size());
    return ncclSuccess;
}

ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId comm_id, int rank) {
    if (!comm) {
        return ncclInvalidArgument;
    }
    *comm = nullptr;

    if (nranks <= 0) {
        return ncclInvalidArgument;
    }
    if (rank < 0 || rank >= nranks) {
        return ncclInvalidArgument;
    }

    const std::string unique_id = unique_id_to_token(comm_id);
    if (unique_id.empty()) {
        return ncclInvalidArgument;
    }

    const fake_gpu::distributed::DistributedConfig& config =
        fake_gpu::BackendConfig::instance().distributed_config();
    std::string error;
    if (!fake_gpu::nccl::validate_direct_init_config(config, error)) {
        return ncclInvalidUsage;
    }

    std::ostringstream request;
    request << "INIT_COMM"
            << " unique_id=" << unique_id
            << " world_size=" << nranks
            << " rank=" << rank
            << " timeout_ms=" << kCoordinatorTimeoutMs;

    fake_gpu::distributed::CoordinatorResponse response;
    if (!fake_gpu::distributed::request_response_unix_socket(
            config.coordinator_address,
            request.str(),
            response,
            error)) {
        return ncclSystemError;
    }
    if (!response.ok) {
        return map_response_error(response.error_code);
    }

    int coordinator_comm_id = -1;
    if (!parse_int_field(response, "comm_id", coordinator_comm_id) || coordinator_comm_id <= 0) {
        return ncclInternalError;
    }

    ncclComm* state = new ncclComm();
    state->comm_id = coordinator_comm_id;
    state->world_size = nranks;
    state->rank = rank;
    *comm = state;
    return ncclSuccess;
}

ncclResult_t ncclCommDestroy(ncclComm_t comm) {
    if (!comm) {
        return ncclInvalidArgument;
    }
    if (comm->destroyed) {
        return ncclInvalidUsage;
    }

    const fake_gpu::distributed::DistributedConfig& config =
        fake_gpu::BackendConfig::instance().distributed_config();
    std::string error;
    if (!fake_gpu::nccl::validate_direct_init_config(config, error)) {
        return ncclInvalidUsage;
    }

    std::ostringstream request;
    request << "DESTROY_COMM"
            << " comm_id=" << comm->comm_id
            << " rank=" << comm->rank;

    fake_gpu::distributed::CoordinatorResponse response;
    if (!fake_gpu::distributed::request_response_unix_socket(
            config.coordinator_address,
            request.str(),
            response,
            error)) {
        return ncclSystemError;
    }
    if (!response.ok) {
        return map_response_error(response.error_code);
    }

    // Keep the handle allocated so repeated destroy reports a deterministic error.
    comm->destroyed = true;
    return ncclSuccess;
}

}  // extern "C"
