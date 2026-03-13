#include "nccl_defs.hpp"
#include "nccl_mode_dispatch.hpp"

#include "../core/backend_config.hpp"
#include "../distributed/collective_executor.hpp"
#include "../distributed/staging_buffer.hpp"
#include "../distributed/transport.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

struct ncclComm {
    int comm_id = -1;
    int world_size = 0;
    int rank = -1;
    int device = -1;
    std::uint64_t next_seqno = 1;
    bool destroyed = false;
    bool finalized = false;
    ncclResult_t async_error = ncclSuccess;
    std::string last_error;
};

namespace {

constexpr int kFakeNcclVersion = NCCL_VERSION_CODE;
constexpr int kCoordinatorTimeoutMs = 1000;

thread_local int g_group_depth = 0;
thread_local std::string g_last_error;

struct GroupCollectiveCall {
    fake_gpu::distributed::CollectiveType type = fake_gpu::distributed::CollectiveType::AllReduce;
    const void* sendbuff = nullptr;
    void* recvbuff = nullptr;
    std::size_t count = 0;
    ncclDataType_t datatype = ncclFloat32;
    fake_gpu::distributed::CollectiveReduceOp reduce_op = fake_gpu::distributed::CollectiveReduceOp::None;
    int root = -1;
    ncclComm_t comm = nullptr;
};

thread_local std::vector<GroupCollectiveCall> g_group_operations;

void clear_last_error(ncclComm_t comm) {
    g_last_error.clear();
    if (comm) {
        comm->last_error.clear();
        comm->async_error = ncclSuccess;
    }
}

ncclResult_t fail_with(ncclComm_t comm, ncclResult_t result, std::string message) {
    g_last_error = message;
    if (comm) {
        comm->last_error = std::move(message);
        if (result != ncclSuccess && result != ncclInProgress) {
            comm->async_error = result;
        }
    }
    return result;
}

std::string unique_id_to_token(const ncclUniqueId& unique_id) {
    const char* begin = unique_id.internal;
    const char* end = std::find(begin, begin + NCCL_UNIQUE_ID_BYTES, '\0');
    return std::string(begin, end);
}

int infer_device_for_rank(int rank) {
    const char* local_rank = std::getenv("LOCAL_RANK");
    if (local_rank && *local_rank) {
        try {
            return std::stoi(local_rank);
        } catch (...) {
        }
    }
    return rank;
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
    if (datatype == ncclInt64) {
        out = fake_gpu::distributed::CollectiveDataType::Int64;
        return true;
    }
    if (datatype == ncclFloat32 || datatype == ncclFloat) {
        out = fake_gpu::distributed::CollectiveDataType::Float32;
        return true;
    }
    if (datatype == ncclFloat64 || datatype == ncclDouble) {
        out = fake_gpu::distributed::CollectiveDataType::Float64;
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
        default:
            return false;
    }
}

ncclResult_t map_response_error(const std::string& error_code) {
    if (error_code == "bad_request" ||
        error_code == "invalid_rank" ||
        error_code == "invalid_world_size" ||
        error_code == "missing_unique_id" ||
        error_code == "invalid_root" ||
        error_code == "invalid_count" ||
        error_code == "invalid_bytes" ||
        error_code == "invalid_slice_plan" ||
        error_code == "invalid_collective_size" ||
        error_code == "invalid_comm_id" ||
        error_code == "invalid_seqno" ||
        error_code == "invalid_timeout") {
        return ncclInvalidArgument;
    }
    if (error_code == "duplicate_destroy" ||
        error_code == "duplicate_rank" ||
        error_code == "duplicate_group_rank" ||
        error_code == "world_size_mismatch" ||
        error_code == "unknown_comm_id" ||
        error_code == "collective_type_mismatch" ||
        error_code == "group_collective_type_mismatch" ||
        error_code == "dtype_mismatch" ||
        error_code == "group_dtype_mismatch" ||
        error_code == "count_mismatch" ||
        error_code == "group_count_mismatch" ||
        error_code == "root_mismatch" ||
        error_code == "group_root_mismatch" ||
        error_code == "reduce_op_mismatch" ||
        error_code == "group_reduce_op_mismatch" ||
        error_code == "bytes_mismatch" ||
        error_code == "group_bytes_mismatch" ||
        error_code == "group_size_mismatch" ||
        error_code == "duplicate_collective_rank" ||
        error_code == "duplicate_barrier_rank" ||
        error_code == "staging_size_mismatch" ||
        error_code == "root_rank_missing" ||
        error_code == "timeout_mismatch" ||
        error_code == "out_of_order_seqno" ||
        error_code == "stale_seqno" ||
        error_code == "rank_destroyed" ||
        error_code == "empty_group" ||
        error_code == "unsupported_reduce_op" ||
        error_code == "unsupported_dtype" ||
        error_code == "unsupported_collective") {
        return ncclInvalidUsage;
    }
    if (error_code == "timeout_waiting_for_ranks" ||
        error_code == "timeout_waiting_for_collective" ||
        error_code == "timeout_waiting_for_barrier" ||
        error_code == "timeout_waiting_for_group" ||
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

bool validate_runtime_config(
    ncclComm_t comm,
    fake_gpu::distributed::DistributedConfig& config,
    std::string& error) {
    config = fake_gpu::BackendConfig::instance().distributed_config();
    if (!fake_gpu::nccl::validate_direct_init_config(config, error)) {
        fail_with(comm, ncclInvalidUsage, error);
        return false;
    }
    return true;
}

bool collective_transfer_sizes(
    fake_gpu::distributed::CollectiveType type,
    std::size_t element_count,
    fake_gpu::distributed::CollectiveDataType dtype,
    int world_size,
    std::size_t& input_bytes,
    std::size_t& staging_bytes,
    std::size_t& output_bytes,
    std::string& error) {
    error.clear();
    if (world_size <= 0) {
        error = "world_size must be > 0";
        return false;
    }
    const std::size_t dtype_size = fake_gpu::distributed::collective_data_type_size(dtype);
    if (dtype_size == 0) {
        error = "unsupported dtype";
        return false;
    }
    if (element_count == 0) {
        error = "count must be > 0";
        return false;
    }

    const std::size_t chunk_bytes = element_count * dtype_size;
    switch (type) {
        case fake_gpu::distributed::CollectiveType::AllReduce:
        case fake_gpu::distributed::CollectiveType::Broadcast:
            input_bytes = chunk_bytes;
            staging_bytes = chunk_bytes;
            output_bytes = chunk_bytes;
            return true;
        case fake_gpu::distributed::CollectiveType::AllGather:
            input_bytes = chunk_bytes;
            staging_bytes = chunk_bytes * static_cast<std::size_t>(world_size);
            output_bytes = staging_bytes;
            return true;
        case fake_gpu::distributed::CollectiveType::ReduceScatter:
            input_bytes = chunk_bytes * static_cast<std::size_t>(world_size);
            staging_bytes = input_bytes;
            output_bytes = chunk_bytes;
            return true;
    }

    error = "unsupported collective";
    return false;
}

void clear_group_operations() {
    g_group_operations.clear();
}

bool all_group_operations_share_comm(
    const std::vector<GroupCollectiveCall>& operations,
    ncclComm_t& comm) {
    if (operations.empty()) {
        comm = nullptr;
        return true;
    }
    comm = operations.front().comm;
    if (!comm) {
        return false;
    }
    for (const GroupCollectiveCall& operation : operations) {
        if (operation.comm != comm) {
            return false;
        }
    }
    return true;
}

ncclResult_t prepare_group_batch(
    ncclComm_t comm,
    const std::vector<GroupCollectiveCall>& operations) {
    if (!comm) {
        return fail_with(nullptr, ncclInvalidArgument, "group contains a null communicator");
    }
    if (operations.empty()) {
        clear_last_error(comm);
        return ncclSuccess;
    }

    fake_gpu::distributed::DistributedConfig config;
    std::string error;
    if (!validate_runtime_config(comm, config, error)) {
        return ncclInvalidUsage;
    }

    std::ostringstream request;
    request << "GROUP_PREPARE"
            << " comm_id=" << comm->comm_id
            << " rank=" << comm->rank
            << " base_seqno=" << comm->next_seqno
            << " timeout_ms=" << kCoordinatorTimeoutMs
            << " op_count=" << operations.size();

    for (std::size_t index = 0; index < operations.size(); ++index) {
        const GroupCollectiveCall& operation = operations[index];
        fake_gpu::distributed::CollectiveDataType mapped_dtype;
        if (!map_dtype(operation.datatype, mapped_dtype)) {
            return fail_with(comm, ncclInvalidArgument, "unsupported dtype in grouped collective");
        }

        std::size_t input_bytes = 0;
        std::size_t staging_bytes = 0;
        std::size_t output_bytes = 0;
        if (!collective_transfer_sizes(
                operation.type,
                operation.count,
                mapped_dtype,
                comm->world_size,
                input_bytes,
                staging_bytes,
                output_bytes,
                error)) {
            return fail_with(comm, ncclInvalidArgument, error);
        }
        request << " op" << index << "_type="
                << fake_gpu::distributed::collective_type_name(operation.type)
                << " op" << index << "_dtype="
                << fake_gpu::distributed::collective_data_type_name(mapped_dtype)
                << " op" << index << "_count=" << operation.count
                << " op" << index << "_root=" << operation.root
                << " op" << index << "_reduce_op="
                << fake_gpu::distributed::collective_reduce_op_name(operation.reduce_op)
                << " op" << index << "_bytes=" << staging_bytes;
    }

    fake_gpu::distributed::CoordinatorResponse response;
    if (!fake_gpu::distributed::request_response_unix_socket(
            config.coordinator_address,
            request.str(),
            response,
            error)) {
        return fail_with(comm, ncclSystemError, error);
    }
    if (!response.ok) {
        return fail_with(
            comm,
            map_response_error(response.error_code),
            response.error_detail.empty() ? response.error_code : response.error_detail);
    }

    int response_comm_id = -1;
    std::uint64_t response_seqno = 0;
    if (!parse_int_field(response, "comm_id", response_comm_id) ||
        !parse_u64_field(response, "base_seqno", response_seqno) ||
        response_comm_id != comm->comm_id ||
        response_seqno != comm->next_seqno) {
        return fail_with(comm, ncclInternalError, "group prepare response was inconsistent");
    }

    clear_last_error(comm);
    return ncclSuccess;
}

ncclResult_t buffer_group_allreduce(
    const void* sendbuff,
    void* recvbuff,
    std::size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm) {
    if (!comm) {
        return fail_with(nullptr, ncclInvalidArgument, "communicator must not be null");
    }
    if (!sendbuff || !recvbuff) {
        return fail_with(comm, ncclInvalidArgument, "send/recv buffer must not be null");
    }
    if (count == 0) {
        return fail_with(comm, ncclInvalidArgument, "count must be > 0");
    }

    fake_gpu::distributed::CollectiveReduceOp reduce_op;
    if (!map_reduce_op(op, reduce_op)) {
        return fail_with(comm, ncclInvalidArgument, "unsupported reduce op");
    }
    fake_gpu::distributed::CollectiveDataType mapped_dtype;
    if (!map_dtype(datatype, mapped_dtype)) {
        return fail_with(comm, ncclInvalidArgument, "unsupported dtype for this collective");
    }

    GroupCollectiveCall call;
    call.type = fake_gpu::distributed::CollectiveType::AllReduce;
    call.sendbuff = sendbuff;
    call.recvbuff = recvbuff;
    call.count = count;
    call.datatype = datatype;
    call.reduce_op = reduce_op;
    call.root = -1;
    call.comm = comm;
    g_group_operations.push_back(call);
    clear_last_error(comm);
    return ncclSuccess;
}

ncclResult_t buffer_group_broadcast(
    const void* sendbuff,
    void* recvbuff,
    std::size_t count,
    ncclDataType_t datatype,
    int root,
    ncclComm_t comm) {
    if (!comm) {
        return fail_with(nullptr, ncclInvalidArgument, "communicator must not be null");
    }
    if (!recvbuff) {
        return fail_with(comm, ncclInvalidArgument, "recv buffer must not be null");
    }
    if (comm->rank == root && !sendbuff) {
        return fail_with(comm, ncclInvalidArgument, "root send buffer must not be null");
    }
    if (count == 0) {
        return fail_with(comm, ncclInvalidArgument, "count must be > 0");
    }
    if (root < 0 || root >= comm->world_size) {
        return fail_with(comm, ncclInvalidArgument, "broadcast root must be within [0, world_size)");
    }

    fake_gpu::distributed::CollectiveDataType mapped_dtype;
    if (!map_dtype(datatype, mapped_dtype)) {
        return fail_with(comm, ncclInvalidArgument, "unsupported dtype for this collective");
    }

    GroupCollectiveCall call;
    call.type = fake_gpu::distributed::CollectiveType::Broadcast;
    call.sendbuff = sendbuff;
    call.recvbuff = recvbuff;
    call.count = count;
    call.datatype = datatype;
    call.reduce_op = fake_gpu::distributed::CollectiveReduceOp::None;
    call.root = root;
    call.comm = comm;
    g_group_operations.push_back(call);
    clear_last_error(comm);
    return ncclSuccess;
}

ncclResult_t buffer_group_allgather(
    const void* sendbuff,
    void* recvbuff,
    std::size_t count,
    ncclDataType_t datatype,
    ncclComm_t comm) {
    if (!comm) {
        return fail_with(nullptr, ncclInvalidArgument, "communicator must not be null");
    }
    if (!sendbuff || !recvbuff) {
        return fail_with(comm, ncclInvalidArgument, "send/recv buffer must not be null");
    }
    if (count == 0) {
        return fail_with(comm, ncclInvalidArgument, "count must be > 0");
    }

    fake_gpu::distributed::CollectiveDataType mapped_dtype;
    if (!map_dtype(datatype, mapped_dtype)) {
        return fail_with(comm, ncclInvalidArgument, "unsupported dtype for this collective");
    }

    GroupCollectiveCall call;
    call.type = fake_gpu::distributed::CollectiveType::AllGather;
    call.sendbuff = sendbuff;
    call.recvbuff = recvbuff;
    call.count = count;
    call.datatype = datatype;
    call.reduce_op = fake_gpu::distributed::CollectiveReduceOp::None;
    call.root = -1;
    call.comm = comm;
    g_group_operations.push_back(call);
    clear_last_error(comm);
    return ncclSuccess;
}

ncclResult_t buffer_group_reducescatter(
    const void* sendbuff,
    void* recvbuff,
    std::size_t recvcount,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm) {
    if (!comm) {
        return fail_with(nullptr, ncclInvalidArgument, "communicator must not be null");
    }
    if (!sendbuff || !recvbuff) {
        return fail_with(comm, ncclInvalidArgument, "send/recv buffer must not be null");
    }
    if (recvcount == 0) {
        return fail_with(comm, ncclInvalidArgument, "count must be > 0");
    }

    fake_gpu::distributed::CollectiveReduceOp reduce_op;
    if (!map_reduce_op(op, reduce_op)) {
        return fail_with(comm, ncclInvalidArgument, "unsupported reduce op");
    }
    fake_gpu::distributed::CollectiveDataType mapped_dtype;
    if (!map_dtype(datatype, mapped_dtype)) {
        return fail_with(comm, ncclInvalidArgument, "unsupported dtype for this collective");
    }

    GroupCollectiveCall call;
    call.type = fake_gpu::distributed::CollectiveType::ReduceScatter;
    call.sendbuff = sendbuff;
    call.recvbuff = recvbuff;
    call.count = recvcount;
    call.datatype = datatype;
    call.reduce_op = reduce_op;
    call.root = -1;
    call.comm = comm;
    g_group_operations.push_back(call);
    clear_last_error(comm);
    return ncclSuccess;
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
        return fail_with(nullptr, ncclInvalidArgument, "communicator must not be null");
    }
    if (comm->destroyed) {
        return fail_with(comm, ncclInvalidUsage, "communicator is already destroyed");
    }
    if (!local_input || !recvbuff) {
        return fail_with(comm, ncclInvalidArgument, "send/recv buffer must not be null");
    }
    if (count == 0) {
        return fail_with(comm, ncclInvalidArgument, "count must be > 0");
    }

    fake_gpu::distributed::CollectiveDataType mapped_dtype;
    if (!map_dtype(datatype, mapped_dtype)) {
        return fail_with(comm, ncclInvalidArgument, "unsupported dtype for this collective");
    }

    std::string error;
    std::size_t input_bytes = 0;
    std::size_t staging_bytes = 0;
    std::size_t output_bytes = 0;
    if (!collective_transfer_sizes(
            collective_type,
            count,
            mapped_dtype,
            comm->world_size,
            input_bytes,
            staging_bytes,
            output_bytes,
            error)) {
        return fail_with(comm, ncclInvalidArgument, error);
    }

    fake_gpu::distributed::DistributedConfig config;
    if (!validate_runtime_config(comm, config, error)) {
        return ncclInvalidUsage;
    }

    const std::uint64_t seqno = comm->next_seqno;
    const std::string staging_name = make_staging_name(*comm, seqno, command);

    fake_gpu::distributed::StagingBufferMetadata metadata;
    metadata.name = staging_name;
    metadata.dtype = fake_gpu::distributed::collective_data_type_name(mapped_dtype);
    metadata.bytes = staging_bytes;
    if (collective_type == fake_gpu::distributed::CollectiveType::AllGather ||
        collective_type == fake_gpu::distributed::CollectiveType::ReduceScatter) {
        metadata.shape = {static_cast<std::size_t>(comm->world_size), count};
    } else {
        metadata.shape = {count};
    }
    metadata.owner_rank = comm->rank;
    metadata.staging_id = seqno;

    fake_gpu::distributed::StagingBufferManager manager;
    fake_gpu::distributed::StagingBufferHandle handle;
    if (!manager.create(metadata, true, handle, error)) {
        return fail_with(comm, ncclSystemError, error);
    }

    std::memcpy(handle.data(), local_input, input_bytes);

    std::ostringstream request;
    request << command
            << " comm_id=" << comm->comm_id
            << " rank=" << comm->rank
            << " seqno=" << seqno
            << " dtype=" << fake_gpu::distributed::collective_data_type_name(mapped_dtype)
            << " count=" << count
            << " bytes=" << staging_bytes
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
        return fail_with(comm, ncclSystemError, error);
    }
    if (!response.ok) {
        return fail_with(
            comm,
            map_response_error(response.error_code),
            response.error_detail.empty() ? response.error_code : response.error_detail);
    }

    int response_comm_id = -1;
    std::uint64_t response_seqno = 0;
    if (!parse_int_field(response, "comm_id", response_comm_id) ||
        !parse_u64_field(response, "seqno", response_seqno) ||
        response_comm_id != comm->comm_id ||
        response_seqno != seqno) {
        return fail_with(comm, ncclInternalError, "coordinator returned an inconsistent response");
    }

    std::memcpy(recvbuff, handle.data(), output_bytes);
    comm->next_seqno++;
    clear_last_error(comm);
    return ncclSuccess;
}

ncclResult_t do_destroy(ncclComm_t comm, bool allow_missing) {
    if (!comm) {
        return fail_with(nullptr, ncclInvalidArgument, "communicator must not be null");
    }
    if (comm->destroyed) {
        return allow_missing ? ncclSuccess : fail_with(comm, ncclInvalidUsage, "communicator is already destroyed");
    }

    fake_gpu::distributed::DistributedConfig config;
    std::string error;
    if (comm->comm_id > 0 && validate_runtime_config(comm, config, error)) {
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
            if (!allow_missing) {
                return fail_with(comm, ncclSystemError, error);
            }
        } else if (!response.ok &&
                   !(allow_missing && response.error_code == "unknown_comm_id")) {
            if (!allow_missing) {
                return fail_with(
                    comm,
                    map_response_error(response.error_code),
                    response.error_detail.empty() ? response.error_code : response.error_detail);
            }
        }
    } else if (!allow_missing) {
        return ncclInvalidUsage;
    }

    comm->destroyed = true;
    comm->finalized = true;
    clear_last_error(comm);
    return ncclSuccess;
}

ncclResult_t unsupported_step_api(ncclComm_t comm, const char* api, const char* step) {
    return fail_with(
        comm,
        ncclInvalidUsage,
        std::string(api) + " is not implemented yet; it is planned for " + step);
}

ncclResult_t flush_grouped_operations() {
    if (g_group_operations.empty()) {
        g_last_error.clear();
        return ncclSuccess;
    }

    ncclComm_t comm = nullptr;
    if (!all_group_operations_share_comm(g_group_operations, comm)) {
        clear_group_operations();
        return fail_with(nullptr, ncclInvalidUsage, "group currently requires all operations to use the same communicator");
    }

    ncclResult_t result = prepare_group_batch(comm, g_group_operations);
    if (result != ncclSuccess) {
        clear_group_operations();
        return result;
    }

    for (const GroupCollectiveCall& operation : g_group_operations) {
        if (operation.type == fake_gpu::distributed::CollectiveType::AllReduce) {
            result = submit_collective(
                "ALLREDUCE",
                fake_gpu::distributed::CollectiveType::AllReduce,
                operation.sendbuff,
                operation.recvbuff,
                operation.count,
                operation.datatype,
                operation.reduce_op,
                -1,
                operation.comm);
        } else if (operation.type == fake_gpu::distributed::CollectiveType::Broadcast) {
            const void* local_input = operation.recvbuff;
            if (operation.comm->rank == operation.root) {
                local_input = operation.sendbuff;
            }
            result = submit_collective(
                "BROADCAST",
                fake_gpu::distributed::CollectiveType::Broadcast,
                local_input,
                operation.recvbuff,
                operation.count,
                operation.datatype,
                fake_gpu::distributed::CollectiveReduceOp::None,
                operation.root,
                operation.comm);
        } else if (operation.type == fake_gpu::distributed::CollectiveType::AllGather) {
            result = submit_collective(
                "ALLGATHER",
                fake_gpu::distributed::CollectiveType::AllGather,
                operation.sendbuff,
                operation.recvbuff,
                operation.count,
                operation.datatype,
                fake_gpu::distributed::CollectiveReduceOp::None,
                -1,
                operation.comm);
        } else if (operation.type == fake_gpu::distributed::CollectiveType::ReduceScatter) {
            result = submit_collective(
                "REDUCESCATTER",
                fake_gpu::distributed::CollectiveType::ReduceScatter,
                operation.sendbuff,
                operation.recvbuff,
                operation.count,
                operation.datatype,
                operation.reduce_op,
                -1,
                operation.comm);
        } else {
            result = fail_with(operation.comm, ncclInvalidUsage, "unsupported grouped collective");
        }

        if (result != ncclSuccess) {
            clear_group_operations();
            return result;
        }
    }

    clear_group_operations();
    clear_last_error(comm);
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
        case ncclRemoteError:
            return "remote error";
        case ncclInProgress:
            return "in progress";
        case ncclNumResults:
            return "num results";
    }
    return "unknown nccl error";
}

const char* ncclGetLastError(ncclComm_t comm) {
    if (comm && !comm->last_error.empty()) {
        return comm->last_error.c_str();
    }
    return g_last_error.empty() ? "success" : g_last_error.c_str();
}

ncclResult_t ncclMemAlloc(void** ptr, std::size_t size) {
    if (!ptr || size == 0) {
        return fail_with(nullptr, ncclInvalidArgument, "ncclMemAlloc requires a non-null ptr and size > 0");
    }
    *ptr = std::malloc(size);
    if (!*ptr) {
        return fail_with(nullptr, ncclSystemError, "malloc failed");
    }
    g_last_error.clear();
    return ncclSuccess;
}

ncclResult_t ncclMemFree(void* ptr) {
    std::free(ptr);
    g_last_error.clear();
    return ncclSuccess;
}

ncclResult_t ncclGetVersion(int* version) {
    if (!version) {
        return fail_with(nullptr, ncclInvalidArgument, "version must not be null");
    }
    *version = kFakeNcclVersion;
    g_last_error.clear();
    return ncclSuccess;
}

ncclResult_t ncclGetUniqueId(ncclUniqueId* unique_id) {
    if (!unique_id) {
        return fail_with(nullptr, ncclInvalidArgument, "unique_id must not be null");
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
    g_last_error.clear();
    return ncclSuccess;
}

ncclResult_t ncclCommInitRankConfig(
    ncclComm_t* comm,
    int nranks,
    ncclUniqueId comm_id,
    int rank,
    ncclConfig_t* /*config*/) {
    return ncclCommInitRank(comm, nranks, comm_id, rank);
}

ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId comm_id, int rank) {
    if (!comm) {
        return fail_with(nullptr, ncclInvalidArgument, "comm must not be null");
    }
    *comm = nullptr;

    if (nranks <= 0) {
        return fail_with(nullptr, ncclInvalidArgument, "nranks must be > 0");
    }
    if (rank < 0 || rank >= nranks) {
        return fail_with(nullptr, ncclInvalidArgument, "rank must be within [0, nranks)");
    }

    const std::string unique_id = unique_id_to_token(comm_id);
    if (unique_id.empty()) {
        return fail_with(nullptr, ncclInvalidArgument, "comm_id token must not be empty");
    }

    fake_gpu::distributed::DistributedConfig config;
    std::string error;
    if (!validate_runtime_config(nullptr, config, error)) {
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
        return fail_with(nullptr, ncclSystemError, error);
    }
    if (!response.ok) {
        return fail_with(
            nullptr,
            map_response_error(response.error_code),
            response.error_detail.empty() ? response.error_code : response.error_detail);
    }

    int coordinator_comm_id = -1;
    if (!parse_int_field(response, "comm_id", coordinator_comm_id) || coordinator_comm_id <= 0) {
        return fail_with(nullptr, ncclInternalError, "coordinator did not return a valid comm_id");
    }

    ncclComm* state = new ncclComm();
    state->comm_id = coordinator_comm_id;
    state->world_size = nranks;
    state->rank = rank;
    state->device = infer_device_for_rank(rank);
    *comm = state;
    clear_last_error(*comm);
    return ncclSuccess;
}

ncclResult_t ncclCommInitAll(ncclComm_t* comm, int ndev, const int* /*devlist*/) {
    if (comm && ndev > 0) {
        for (int index = 0; index < ndev; ++index) {
            comm[index] = nullptr;
        }
    }
    return unsupported_step_api(nullptr, "ncclCommInitAll", "a later compatibility pass");
}

ncclResult_t ncclCommInitRankScalable(
    ncclComm_t* comm,
    int nranks,
    int myrank,
    int n_id,
    ncclUniqueId* comm_ids,
    ncclConfig_t* /*config*/) {
    if (!comm_ids || n_id <= 0) {
        return fail_with(nullptr, ncclInvalidArgument, "comm_ids must not be null and n_id must be > 0");
    }
    return ncclCommInitRank(comm, nranks, comm_ids[0], myrank);
}

ncclResult_t ncclCommFinalize(ncclComm_t comm) {
    if (!comm) {
        return fail_with(nullptr, ncclInvalidArgument, "communicator must not be null");
    }
    comm->finalized = true;
    clear_last_error(comm);
    return ncclSuccess;
}

ncclResult_t ncclCommDestroy(ncclComm_t comm) {
    return do_destroy(comm, false);
}

ncclResult_t ncclCommAbort(ncclComm_t comm) {
    return do_destroy(comm, true);
}

ncclResult_t ncclCommSplit(
    ncclComm_t comm,
    int color,
    int /*key*/,
    ncclComm_t* newcomm,
    ncclConfig_t* /*config*/) {
    if (!newcomm) {
        return fail_with(comm, ncclInvalidArgument, "newcomm must not be null");
    }
    *newcomm = nullptr;
    if (color == NCCL_SPLIT_NOCOLOR) {
        clear_last_error(comm);
        return ncclSuccess;
    }
    return unsupported_step_api(comm, "ncclCommSplit", "a later compatibility pass");
}

ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* async_error) {
    if (!comm || !async_error) {
        return fail_with(comm, ncclInvalidArgument, "communicator and async_error must not be null");
    }
    *async_error = comm->async_error;
    g_last_error.clear();
    return ncclSuccess;
}

ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
    if (!comm || !count) {
        return fail_with(comm, ncclInvalidArgument, "communicator and count must not be null");
    }
    *count = comm->world_size;
    clear_last_error(comm);
    return ncclSuccess;
}

ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* device) {
    if (!comm || !device) {
        return fail_with(comm, ncclInvalidArgument, "communicator and device must not be null");
    }
    *device = comm->device;
    clear_last_error(comm);
    return ncclSuccess;
}

ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) {
    if (!comm || !rank) {
        return fail_with(comm, ncclInvalidArgument, "communicator and rank must not be null");
    }
    *rank = comm->rank;
    clear_last_error(comm);
    return ncclSuccess;
}

ncclResult_t ncclCommRegister(const ncclComm_t comm, void* buff, std::size_t size, void** handle) {
    if (!comm || !buff || !handle || size == 0) {
        return fail_with(comm, ncclInvalidArgument, "invalid ncclCommRegister arguments");
    }
    *handle = buff;
    clear_last_error(comm);
    return ncclSuccess;
}

ncclResult_t ncclCommDeregister(const ncclComm_t comm, void* /*handle*/) {
    if (!comm) {
        return fail_with(nullptr, ncclInvalidArgument, "communicator must not be null");
    }
    clear_last_error(comm);
    return ncclSuccess;
}

ncclResult_t ncclCommWindowRegister(
    ncclComm_t comm,
    void* buff,
    std::size_t size,
    ncclWindow_t* window,
    int /*window_flags*/) {
    if (!comm || !buff || !window || size == 0) {
        return fail_with(comm, ncclInvalidArgument, "invalid ncclCommWindowRegister arguments");
    }
    *window = reinterpret_cast<ncclWindow_t>(buff);
    clear_last_error(comm);
    return ncclSuccess;
}

ncclResult_t ncclCommWindowDeregister(ncclComm_t comm, ncclWindow_t /*window*/) {
    if (!comm) {
        return fail_with(nullptr, ncclInvalidArgument, "communicator must not be null");
    }
    clear_last_error(comm);
    return ncclSuccess;
}

ncclResult_t ncclRedOpCreatePreMulSum(
    ncclRedOp_t* op,
    void* scalar,
    ncclDataType_t /*datatype*/,
    ncclScalarResidence_t /*residence*/,
    ncclComm_t comm) {
    if (!op || !scalar) {
        return fail_with(comm, ncclInvalidArgument, "op and scalar must not be null");
    }
    return unsupported_step_api(comm, "ncclRedOpCreatePreMulSum", "a later compatibility pass");
}

ncclResult_t ncclRedOpDestroy(ncclRedOp_t /*op*/, ncclComm_t comm) {
    clear_last_error(comm);
    return ncclSuccess;
}

ncclResult_t ncclReduce(
    const void* /*sendbuff*/,
    void* /*recvbuff*/,
    std::size_t /*count*/,
    ncclDataType_t /*datatype*/,
    ncclRedOp_t /*op*/,
    int /*root*/,
    ncclComm_t comm,
    cudaStream_t /*stream*/) {
    return unsupported_step_api(comm, "ncclReduce", "Step 13");
}

ncclResult_t ncclBcast(
    void* buff,
    std::size_t count,
    ncclDataType_t datatype,
    int root,
    ncclComm_t comm,
    cudaStream_t stream) {
    return ncclBroadcast(buff, buff, count, datatype, root, comm, stream);
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
        return fail_with(nullptr, ncclInvalidArgument, "communicator must not be null");
    }
    if (g_group_depth > 0) {
        return buffer_group_broadcast(sendbuff, recvbuff, count, datatype, root, comm);
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

ncclResult_t ncclAllReduce(
    const void* sendbuff,
    void* recvbuff,
    std::size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    cudaStream_t /*stream*/) {
    if (g_group_depth > 0) {
        return buffer_group_allreduce(sendbuff, recvbuff, count, datatype, op, comm);
    }
    fake_gpu::distributed::CollectiveReduceOp reduce_op;
    if (!map_reduce_op(op, reduce_op)) {
        return fail_with(comm, ncclInvalidArgument, "unsupported reduce op");
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

ncclResult_t ncclReduceScatter(
    const void* sendbuff,
    void* recvbuff,
    std::size_t recvcount,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    cudaStream_t /*stream*/) {
    if (g_group_depth > 0) {
        return buffer_group_reducescatter(sendbuff, recvbuff, recvcount, datatype, op, comm);
    }
    fake_gpu::distributed::CollectiveReduceOp reduce_op;
    if (!map_reduce_op(op, reduce_op)) {
        return fail_with(comm, ncclInvalidArgument, "unsupported reduce op");
    }
    return submit_collective(
        "REDUCESCATTER",
        fake_gpu::distributed::CollectiveType::ReduceScatter,
        sendbuff,
        recvbuff,
        recvcount,
        datatype,
        reduce_op,
        -1,
        comm);
}

ncclResult_t ncclAllGather(
    const void* sendbuff,
    void* recvbuff,
    std::size_t sendcount,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t /*stream*/) {
    if (g_group_depth > 0) {
        return buffer_group_allgather(sendbuff, recvbuff, sendcount, datatype, comm);
    }
    return submit_collective(
        "ALLGATHER",
        fake_gpu::distributed::CollectiveType::AllGather,
        sendbuff,
        recvbuff,
        sendcount,
        datatype,
        fake_gpu::distributed::CollectiveReduceOp::None,
        -1,
        comm);
}

ncclResult_t ncclSend(
    const void* /*sendbuff*/,
    std::size_t /*count*/,
    ncclDataType_t /*datatype*/,
    int /*peer*/,
    ncclComm_t comm,
    cudaStream_t /*stream*/) {
    return unsupported_step_api(comm, "ncclSend", "a later compatibility pass");
}

ncclResult_t ncclRecv(
    void* /*recvbuff*/,
    std::size_t /*count*/,
    ncclDataType_t /*datatype*/,
    int /*peer*/,
    ncclComm_t comm,
    cudaStream_t /*stream*/) {
    return unsupported_step_api(comm, "ncclRecv", "a later compatibility pass");
}

ncclResult_t ncclGroupStart(void) {
    ++g_group_depth;
    g_last_error.clear();
    return ncclSuccess;
}

ncclResult_t ncclGroupEnd(void) {
    if (g_group_depth > 0) {
        --g_group_depth;
    }
    if (g_group_depth > 0) {
        return ncclSuccess;
    }
    return flush_grouped_operations();
}

ncclResult_t ncclGroupSimulateEnd(ncclSimInfo_t* sim_info) {
    if (sim_info) {
        sim_info->estimatedTime = 0.0f;
    }
    if (g_group_depth > 0) {
        --g_group_depth;
    }
    clear_group_operations();
    g_last_error.clear();
    return ncclSuccess;
}

}  // extern "C"
