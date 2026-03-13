#include "nccl_defs.hpp"
#include "nccl_mode_dispatch.hpp"

#include "../core/backend_config.hpp"
#include "../distributed/collective_executor.hpp"
#include "../distributed/staging_buffer.hpp"
#include "../distributed/transport.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <random>
#include <sstream>
#include <string>

#include <unistd.h>

struct ncclComm {
    int comm_id = -1;
    int world_size = 0;
    int rank = -1;
    std::uint64_t next_seqno = 1;
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

bool parse_u64_field(
    const fake_gpu::distributed::CoordinatorResponse& response,
    const char* key,
    std::uint64_t& value) {
    auto it = response.fields.find(key);
    if (it == response.fields.end()) {
        return false;
    }
    try {
        std::size_t consumed = 0;
        value = std::stoull(it->second, &consumed, 10);
        return consumed == it->second.size();
    } catch (...) {
        return false;
    }
}

bool map_dtype(
    ncclDataType_t datatype,
    fake_gpu::distributed::CollectiveDataType& out) {
    if (datatype == ncclInt32 || datatype == ncclInt) {
        out = fake_gpu::distributed::CollectiveDataType::Int32;
        return true;
    }
    if (datatype == ncclFloat32 || datatype == ncclFloat) {
        out = fake_gpu::distributed::CollectiveDataType::Float32;
        return true;
    }
    return false;
}

bool map_reduce_op(
    ncclRedOp_t op,
    fake_gpu::distributed::CollectiveReduceOp& out) {
    switch (op) {
        case ncclSum:
            out = fake_gpu::distributed::CollectiveReduceOp::Sum;
            return true;
        case ncclProd:
            out = fake_gpu::distributed::CollectiveReduceOp::Prod;
            return true;
        case ncclMax:
            out = fake_gpu::distributed::CollectiveReduceOp::Max;
            return true;
        case ncclMin:
            out = fake_gpu::distributed::CollectiveReduceOp::Min;
            return true;
    }
    return false;
}

ncclResult_t map_response_error(const std::string& error_code) {
    if (error_code == "bad_request" ||
        error_code == "invalid_rank" ||
        error_code == "invalid_world_size" ||
        error_code == "missing_unique_id" ||
        error_code == "invalid_root" ||
        error_code == "invalid_count" ||
        error_code == "invalid_bytes" ||
        error_code == "invalid_comm_id" ||
        error_code == "invalid_seqno") {
        return ncclInvalidArgument;
    }
    if (error_code == "duplicate_destroy" ||
        error_code == "duplicate_rank" ||
        error_code == "world_size_mismatch" ||
        error_code == "unknown_comm_id" ||
        error_code == "collective_type_mismatch" ||
        error_code == "dtype_mismatch" ||
        error_code == "count_mismatch" ||
        error_code == "root_mismatch" ||
        error_code == "reduce_op_mismatch" ||
        error_code == "bytes_mismatch" ||
        error_code == "duplicate_collective_rank" ||
        error_code == "out_of_order_seqno" ||
        error_code == "stale_seqno" ||
        error_code == "rank_destroyed" ||
        error_code == "unsupported_reduce_op" ||
        error_code == "unsupported_dtype" ||
        error_code == "unsupported_collective") {
        return ncclInvalidUsage;
    }
    if (error_code == "timeout_waiting_for_ranks" ||
        error_code == "timeout_waiting_for_collective" ||
        error_code == "staging_open_failed") {
        return ncclSystemError;
    }
    return ncclInternalError;
}

std::string make_staging_name(
    const ncclComm& comm,
    std::uint64_t seqno,
    const char* op_name) {
    return "/fakegpu-" + std::string(op_name) +
           "-c" + std::to_string(comm.comm_id) +
           "-r" + std::to_string(comm.rank) +
           "-s" + std::to_string(seqno) +
           "-p" + std::to_string(static_cast<long long>(::getpid()));
}

ncclResult_t submit_collective(
    const char* command,
    fake_gpu::distributed::CollectiveType collective_type,
    const void* local_input,
    void* recvbuff,
    std::size_t count,
    ncclDataType_t datatype,
    fake_gpu::distributed::CollectiveReduceOp reduce_op,
    int root,
    ncclComm_t comm) {
    if (!comm) {
        return ncclInvalidArgument;
    }
    if (comm->destroyed) {
        return ncclInvalidUsage;
    }
    if (!local_input || !recvbuff) {
        return ncclInvalidArgument;
    }
    if (count == 0) {
        return ncclInvalidArgument;
    }

    fake_gpu::distributed::CollectiveDataType mapped_dtype;
    if (!map_dtype(datatype, mapped_dtype)) {
        return ncclInvalidArgument;
    }

    const std::size_t dtype_size = fake_gpu::distributed::collective_data_type_size(mapped_dtype);
    const std::size_t bytes = count * dtype_size;

    const fake_gpu::distributed::DistributedConfig& config =
        fake_gpu::BackendConfig::instance().distributed_config();
    std::string error;
    if (!fake_gpu::nccl::validate_direct_init_config(config, error)) {
        return ncclInvalidUsage;
    }

    const std::uint64_t seqno = comm->next_seqno;
    const std::string staging_name = make_staging_name(*comm, seqno, command);

    fake_gpu::distributed::StagingBufferMetadata metadata;
    metadata.name = staging_name;
    metadata.dtype = fake_gpu::distributed::collective_data_type_name(mapped_dtype);
    metadata.bytes = bytes;
    metadata.shape = {count};
    metadata.owner_rank = comm->rank;
    metadata.staging_id = seqno;

    fake_gpu::distributed::StagingBufferManager manager;
    fake_gpu::distributed::StagingBufferHandle handle;
    if (!manager.create(metadata, true, handle, error)) {
        return ncclSystemError;
    }

    std::memcpy(handle.data(), local_input, bytes);

    std::ostringstream request;
    request << command
            << " comm_id=" << comm->comm_id
            << " rank=" << comm->rank
            << " seqno=" << seqno
            << " dtype=" << fake_gpu::distributed::collective_data_type_name(mapped_dtype)
            << " count=" << count
            << " bytes=" << bytes
            << " root=" << root
            << " reduce_op=" << fake_gpu::distributed::collective_reduce_op_name(reduce_op)
            << " staging_name=" << staging_name
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

    int response_comm_id = -1;
    std::uint64_t response_seqno = 0;
    if (!parse_int_field(response, "comm_id", response_comm_id) ||
        !parse_u64_field(response, "seqno", response_seqno) ||
        response_comm_id != comm->comm_id ||
        response_seqno != seqno) {
        return ncclInternalError;
    }

    std::memcpy(recvbuff, handle.data(), bytes);
    comm->next_seqno++;
    return ncclSuccess;
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

    comm->destroyed = true;
    return ncclSuccess;
}

ncclResult_t ncclAllReduce(
    const void* sendbuff,
    void* recvbuff,
    std::size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    cudaStream_t /*stream*/) {
    fake_gpu::distributed::CollectiveReduceOp reduce_op;
    if (!map_reduce_op(op, reduce_op)) {
        return ncclInvalidArgument;
    }
    return submit_collective(
        "ALLREDUCE",
        fake_gpu::distributed::CollectiveType::AllReduce,
        sendbuff,
        recvbuff,
        count,
        datatype,
        reduce_op,
        -1,
        comm);
}

ncclResult_t ncclBroadcast(
    const void* sendbuff,
    void* recvbuff,
    std::size_t count,
    ncclDataType_t datatype,
    int root,
    ncclComm_t comm,
    cudaStream_t /*stream*/) {
    if (!comm) {
        return ncclInvalidArgument;
    }
    const void* local_input = recvbuff;
    if (comm->rank == root) {
        local_input = sendbuff;
    }
    return submit_collective(
        "BROADCAST",
        fake_gpu::distributed::CollectiveType::Broadcast,
        local_input,
        recvbuff,
        count,
        datatype,
        fake_gpu::distributed::CollectiveReduceOp::None,
        root,
        comm);
}

}  // extern "C"
