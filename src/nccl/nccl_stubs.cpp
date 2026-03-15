#include "nccl_defs.hpp"
#include "nccl_mode_dispatch.hpp"
#include "nccl_passthrough.hpp"
#include "staging_adapter.hpp"

#include "../core/backend_config.hpp"
#include "../core/global_state.hpp"
#include "../distributed/communicator.hpp"
#include "../distributed/collective_executor.hpp"
#include "../distributed/staging_chunk_plan.hpp"
#include "../distributed/staging_buffer.hpp"
#include "../distributed/transport.hpp"
#include "../cuda/cuda_driver_defs.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <unistd.h>

struct ncclComm {
    fake_gpu::distributed::DistributedMode dist_mode =
        fake_gpu::distributed::DistributedMode::Simulate;
    int comm_id = -1;
    int world_size = 0;
    int rank = -1;
    int device = -1;
    std::uint64_t next_seqno = 1;
    void* real_comm = nullptr;
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
    std::vector<char> recv_scratch;
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
    int inferred = rank;
    const char* local_rank = std::getenv("LOCAL_RANK");
    if (local_rank && *local_rank) {
        try {
            inferred = std::stoi(local_rank);
        } catch (...) {
        }
    }
    fake_gpu::GlobalState::instance().initialize();
    const int device_count = fake_gpu::GlobalState::instance().get_device_count();
    if (device_count > 0) {
        inferred %= device_count;
        if (inferred < 0) {
            inferred += device_count;
        }
    }
    return inferred;
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

bool uses_real_nccl(fake_gpu::distributed::DistributedMode mode) {
    return mode == fake_gpu::distributed::DistributedMode::Proxy ||
           mode == fake_gpu::distributed::DistributedMode::Passthrough;
}

bool requires_coordinator(fake_gpu::distributed::DistributedMode mode) {
    return mode != fake_gpu::distributed::DistributedMode::Passthrough;
}

ncclResult_t require_real_nccl(std::string& error) {
    if (!fake_gpu::nccl::RealNcclLoader::instance().initialize(error)) {
        return ncclSystemError;
    }
    return ncclSuccess;
}

ncclResult_t real_nccl_error(
    ncclComm_t comm,
    ncclResult_t result,
    std::string error) {
    std::string lookup_error;
    const char* detail =
        fake_gpu::nccl::RealNcclLoader::instance().get_error_string(result, lookup_error);
    if (detail && *detail) {
        if (!error.empty()) {
            error += ": ";
        }
        error += detail;
    } else if (!lookup_error.empty()) {
        if (!error.empty()) {
            error += ": ";
        }
        error += lookup_error;
    }
    if (error.empty()) {
        error = "real NCCL call failed";
    }
    return fail_with(comm, result, std::move(error));
}

bool grouped_collective_supported(ncclComm_t comm) {
    if (comm && comm->dist_mode != fake_gpu::distributed::DistributedMode::Simulate) {
        fail_with(
            comm,
            ncclInvalidUsage,
            "grouped collectives are only implemented for FAKEGPU_DIST_MODE=simulate");
        return false;
    }
    return true;
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
        error_code == "proxy_only_mismatch" ||
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

bool coordinator_request_response(
    const fake_gpu::distributed::DistributedConfig& config,
    const std::string& request_line,
    fake_gpu::distributed::CoordinatorResponse& response,
    std::string& error) {
    return fake_gpu::distributed::request_response(
        config.coordinator_transport,
        config.coordinator_address,
        request_line,
        response,
        error);
}

bool collective_transfer_sizes(
    fake_gpu::distributed::CollectiveType type,
    std::size_t element_count,
    fake_gpu::distributed::CollectiveDataType dtype,
    int world_size,
    std::size_t& input_bytes,
    std::size_t& staging_bytes,
    std::size_t& output_bytes,
    std::string& error);

ncclResult_t submit_proxy_collective_record(
    const fake_gpu::distributed::DistributedConfig& config,
    const char* command,
    fake_gpu::distributed::CollectiveType collective_type,
    std::size_t count,
    ncclDataType_t datatype,
    fake_gpu::distributed::CollectiveReduceOp reduce_op,
    int root,
    ncclComm_t comm) {
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

    std::ostringstream request;
    request << command
            << " comm_id=" << comm->comm_id
            << " rank=" << comm->rank
            << " seqno=" << comm->next_seqno
            << " dtype=" << fake_gpu::distributed::collective_data_type_name(mapped_dtype)
            << " count=" << count
            << " bytes=" << staging_bytes
            << " root=" << root
            << " reduce_op=" << fake_gpu::distributed::collective_reduce_op_name(reduce_op)
            << " proxy_only=1"
            << " timeout_ms=" << kCoordinatorTimeoutMs;

    fake_gpu::distributed::CoordinatorResponse response;
    if (!coordinator_request_response(config, request.str(), response, error)) {
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
        response_seqno != comm->next_seqno) {
        return fail_with(comm, ncclInternalError, "coordinator returned an inconsistent response");
    }

    comm->next_seqno++;
    clear_last_error(comm);
    return ncclSuccess;
}

ncclResult_t submit_real_collective(
    fake_gpu::distributed::CollectiveType collective_type,
    const void* sendbuff,
    void* recvbuff,
    std::size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    int root,
    ncclComm_t comm,
    cudaStream_t stream) {
    if (!comm->real_comm) {
        return fail_with(comm, ncclInvalidUsage, "real NCCL communicator is not initialized");
    }
    std::string error;
    const ncclResult_t require_result = require_real_nccl(error);
    if (require_result != ncclSuccess) {
        return fail_with(comm, require_result, error);
    }

    const ncclComm_t real_comm = reinterpret_cast<ncclComm_t>(comm->real_comm);
    ncclResult_t result = ncclInvalidUsage;
    if (collective_type == fake_gpu::distributed::CollectiveType::AllReduce) {
        result = fake_gpu::nccl::RealNcclLoader::instance().all_reduce(
            sendbuff,
            recvbuff,
            count,
            datatype,
            op,
            real_comm,
            stream,
            error);
    } else if (collective_type == fake_gpu::distributed::CollectiveType::Reduce) {
        result = fake_gpu::nccl::RealNcclLoader::instance().reduce(
            sendbuff,
            recvbuff,
            count,
            datatype,
            op,
            root,
            real_comm,
            stream,
            error);
    } else if (collective_type == fake_gpu::distributed::CollectiveType::Broadcast) {
        result = fake_gpu::nccl::RealNcclLoader::instance().broadcast(
            sendbuff,
            recvbuff,
            count,
            datatype,
            root,
            real_comm,
            stream,
            error);
    } else if (collective_type == fake_gpu::distributed::CollectiveType::AllGather) {
        result = fake_gpu::nccl::RealNcclLoader::instance().all_gather(
            sendbuff,
            recvbuff,
            count,
            datatype,
            real_comm,
            stream,
            error);
    } else if (collective_type == fake_gpu::distributed::CollectiveType::ReduceScatter) {
        result = fake_gpu::nccl::RealNcclLoader::instance().reduce_scatter(
            sendbuff,
            recvbuff,
            count,
            datatype,
            op,
            real_comm,
            stream,
            error);
    }

    if (result != ncclSuccess) {
        return real_nccl_error(comm, result, error);
    }

    clear_last_error(comm);
    return ncclSuccess;
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
        case fake_gpu::distributed::CollectiveType::Reduce:
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

bool build_transfer_chunk_plan(
    fake_gpu::distributed::CollectiveType type,
    std::size_t element_count,
    fake_gpu::distributed::CollectiveDataType dtype,
    int world_size,
    std::size_t chunk_threshold_bytes,
    fake_gpu::distributed::StagingChunkPlan& out,
    std::string& error) {
    if (!fake_gpu::distributed::build_staging_chunk_plan(
            type,
            element_count,
            dtype,
            world_size,
            chunk_threshold_bytes,
            out,
            error)) {
        return false;
    }
    if (out.chunks.empty()) {
        error = "chunk plan must contain at least one chunk";
        return false;
    }
    return true;
}

std::size_t byte_offset_for_elements(std::size_t element_offset, std::size_t dtype_size) {
    return element_offset * dtype_size;
}

char* byte_pointer(void* buffer, std::size_t byte_offset) {
    return static_cast<char*>(buffer) + byte_offset;
}

const char* byte_pointer(const void* buffer, std::size_t byte_offset) {
    return static_cast<const char*>(buffer) + byte_offset;
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
    if (!grouped_collective_supported(comm)) {
        return ncclInvalidUsage;
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
            << " timeout_ms=" << kCoordinatorTimeoutMs;

    std::vector<fake_gpu::distributed::CollectiveBatchPlanItem> prepared_operations;
    for (const GroupCollectiveCall& operation : operations) {
        fake_gpu::distributed::CollectiveDataType mapped_dtype;
        if (!map_dtype(operation.datatype, mapped_dtype)) {
            return fail_with(comm, ncclInvalidArgument, "unsupported dtype in grouped collective");
        }

        fake_gpu::distributed::StagingChunkPlan chunk_plan;
        if (!build_transfer_chunk_plan(
                operation.type,
                operation.count,
                mapped_dtype,
                comm->world_size,
                config.staging_chunk_bytes,
                chunk_plan,
                error)) {
            return fail_with(comm, ncclInvalidArgument, error);
        }

        for (const fake_gpu::distributed::StagingChunkPlanEntry& chunk : chunk_plan.chunks) {
            fake_gpu::distributed::CollectiveBatchPlanItem item;
            item.type = operation.type;
            item.dtype = mapped_dtype;
            item.count = chunk.element_count;
            item.root = operation.root;
            item.reduce_op = operation.reduce_op;
            item.bytes = chunk.staging_bytes;
            prepared_operations.push_back(item);
        }
    }

    request << " op_count=" << prepared_operations.size();

    for (std::size_t index = 0; index < prepared_operations.size(); ++index) {
        const fake_gpu::distributed::CollectiveBatchPlanItem& operation =
            prepared_operations[index];
        request << " op" << index << "_type="
                << fake_gpu::distributed::collective_type_name(operation.type)
                << " op" << index << "_dtype="
                << fake_gpu::distributed::collective_data_type_name(operation.dtype)
                << " op" << index << "_count=" << operation.count
                << " op" << index << "_root=" << operation.root
                << " op" << index << "_reduce_op="
                << fake_gpu::distributed::collective_reduce_op_name(operation.reduce_op)
                << " op" << index << "_bytes=" << operation.bytes;
    }

    fake_gpu::distributed::CoordinatorResponse response;
    if (!coordinator_request_response(config, request.str(), response, error)) {
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
    if (!grouped_collective_supported(comm)) {
        return ncclInvalidUsage;
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

ncclResult_t buffer_group_reduce(
    const void* sendbuff,
    void* recvbuff,
    std::size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    int root,
    ncclComm_t comm) {
    if (!comm) {
        return fail_with(nullptr, ncclInvalidArgument, "communicator must not be null");
    }
    if (!sendbuff) {
        return fail_with(comm, ncclInvalidArgument, "send buffer must not be null");
    }
    if (comm->rank == root && !recvbuff) {
        return fail_with(comm, ncclInvalidArgument, "root recv buffer must not be null");
    }
    if (!grouped_collective_supported(comm)) {
        return ncclInvalidUsage;
    }
    if (count == 0) {
        return fail_with(comm, ncclInvalidArgument, "count must be > 0");
    }
    if (root < 0 || root >= comm->world_size) {
        return fail_with(comm, ncclInvalidArgument, "reduce root must be within [0, world_size)");
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
    call.type = fake_gpu::distributed::CollectiveType::Reduce;
    call.sendbuff = sendbuff;
    call.recvbuff = recvbuff;
    call.count = count;
    call.datatype = datatype;
    call.reduce_op = reduce_op;
    call.root = root;
    call.comm = comm;
    if (comm->rank != root) {
        const std::size_t scratch_bytes =
            fake_gpu::distributed::collective_data_type_size(mapped_dtype) * count;
        call.recv_scratch.assign(scratch_bytes, '\0');
    }
    g_group_operations.push_back(std::move(call));
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
    if (!grouped_collective_supported(comm)) {
        return ncclInvalidUsage;
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
    if (!grouped_collective_supported(comm)) {
        return ncclInvalidUsage;
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
    if (!grouped_collective_supported(comm)) {
        return ncclInvalidUsage;
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

ncclResult_t submit_collective_chunk(
    const fake_gpu::distributed::DistributedConfig& config,
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
    if (!coordinator_request_response(config, request.str(), response, error)) {
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

ncclResult_t submit_collective(
    const char* command,
    fake_gpu::distributed::CollectiveType collective_type,
    const void* local_input,
    void* recvbuff,
    std::size_t count,
    ncclDataType_t datatype,
    fake_gpu::distributed::CollectiveReduceOp reduce_op,
    int root,
    ncclComm_t comm,
    ncclRedOp_t real_reduce_op,
    cudaStream_t stream) {
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

    fake_gpu::distributed::DistributedConfig config;
    std::string error;
    if (!validate_runtime_config(comm, config, error)) {
        return ncclInvalidUsage;
    }

    if (comm->dist_mode == fake_gpu::distributed::DistributedMode::Proxy) {
        ncclResult_t result = submit_proxy_collective_record(
            config,
            command,
            collective_type,
            count,
            datatype,
            reduce_op,
            root,
            comm);
        if (result != ncclSuccess) {
            return result;
        }
        return submit_real_collective(
            collective_type,
            local_input,
            recvbuff,
            count,
            datatype,
            real_reduce_op,
            root,
            comm,
            stream);
    }

    if (comm->dist_mode == fake_gpu::distributed::DistributedMode::Passthrough) {
        return submit_real_collective(
            collective_type,
            local_input,
            recvbuff,
            count,
            datatype,
            real_reduce_op,
            root,
            comm,
            stream);
    }

    fake_gpu::distributed::StagingChunkPlan chunk_plan;
    if (!build_transfer_chunk_plan(
            collective_type,
            count,
            mapped_dtype,
            comm->world_size,
            config.staging_chunk_bytes,
            chunk_plan,
            error)) {
        return fail_with(comm, ncclInvalidArgument, error);
    }

    const std::size_t dtype_size = chunk_plan.dtype_size;
    const char* input_bytes = static_cast<const char*>(local_input);
    char* output_bytes = static_cast<char*>(recvbuff);

    for (const fake_gpu::distributed::StagingChunkPlanEntry& chunk : chunk_plan.chunks) {
        const std::size_t chunk_offset_bytes =
            byte_offset_for_elements(chunk.offset_elements, dtype_size);

        if (collective_type == fake_gpu::distributed::CollectiveType::AllReduce ||
            collective_type == fake_gpu::distributed::CollectiveType::Reduce) {
            std::vector<char> chunk_input;
            if (!fake_gpu::nccl::copy_buffer_to_host(
                    byte_pointer(input_bytes, chunk_offset_bytes),
                    chunk.input_bytes,
                    chunk_input,
                    error)) {
                return fail_with(comm, ncclSystemError, error);
            }
            std::vector<char> chunk_output(chunk.output_bytes, 0);
            ncclResult_t result = submit_collective_chunk(
                config,
                command,
                collective_type,
                chunk_input.data(),
                chunk_output.data(),
                chunk.element_count,
                datatype,
                reduce_op,
                root,
                comm);
            if (result != ncclSuccess) {
                return result;
            }
            if (!fake_gpu::nccl::copy_host_to_buffer(
                    byte_pointer(output_bytes, chunk_offset_bytes),
                    chunk_output.data(),
                    chunk.output_bytes,
                    error)) {
                return fail_with(comm, ncclSystemError, error);
            }
            continue;
        }

        if (collective_type == fake_gpu::distributed::CollectiveType::Broadcast) {
            std::vector<char> chunk_input(chunk.input_bytes, 0);
            if (comm->rank == root) {
                if (!fake_gpu::nccl::copy_buffer_to_host(
                        byte_pointer(input_bytes, chunk_offset_bytes),
                        chunk.input_bytes,
                        chunk_input,
                        error)) {
                    return fail_with(comm, ncclSystemError, error);
                }
            }
            std::vector<char> chunk_output(chunk.output_bytes, 0);
            ncclResult_t result = submit_collective_chunk(
                config,
                command,
                collective_type,
                chunk_input.data(),
                chunk_output.data(),
                chunk.element_count,
                datatype,
                reduce_op,
                root,
                comm);
            if (result != ncclSuccess) {
                return result;
            }
            if (!fake_gpu::nccl::copy_host_to_buffer(
                    byte_pointer(output_bytes, chunk_offset_bytes),
                    chunk_output.data(),
                    chunk.output_bytes,
                    error)) {
                return fail_with(comm, ncclSystemError, error);
            }
            continue;
        }

        if (collective_type == fake_gpu::distributed::CollectiveType::AllGather) {
            std::vector<char> chunk_input;
            if (!fake_gpu::nccl::copy_buffer_to_host(
                    byte_pointer(input_bytes, chunk_offset_bytes),
                    chunk.input_bytes,
                    chunk_input,
                    error)) {
                return fail_with(comm, ncclSystemError, error);
            }
            std::vector<char> chunk_output(chunk.output_bytes, 0);
            ncclResult_t result = submit_collective_chunk(
                config,
                command,
                collective_type,
                chunk_input.data(),
                chunk_output.data(),
                chunk.element_count,
                datatype,
                reduce_op,
                root,
                comm);
            if (result != ncclSuccess) {
                return result;
            }

            for (int rank = 0; rank < comm->world_size; ++rank) {
                const std::size_t src_offset =
                    static_cast<std::size_t>(rank) * chunk.input_bytes;
                const std::size_t dst_offset =
                    byte_offset_for_elements(
                        static_cast<std::size_t>(rank) * count + chunk.offset_elements,
                        dtype_size);
                if (!fake_gpu::nccl::copy_host_to_buffer(
                    byte_pointer(output_bytes, dst_offset),
                    chunk_output.data() + src_offset,
                    chunk.input_bytes,
                    error)) {
                    return fail_with(comm, ncclSystemError, error);
                }
            }
            continue;
        }

        if (collective_type == fake_gpu::distributed::CollectiveType::ReduceScatter) {
            std::vector<char> chunk_input(chunk.input_bytes, 0);
            for (int rank = 0; rank < comm->world_size; ++rank) {
                const std::size_t src_offset =
                    byte_offset_for_elements(
                        static_cast<std::size_t>(rank) * count + chunk.offset_elements,
                        dtype_size);
                const std::size_t dst_offset =
                    static_cast<std::size_t>(rank) * chunk.output_bytes;
                std::vector<char> rank_slice;
                if (!fake_gpu::nccl::copy_buffer_to_host(
                        byte_pointer(input_bytes, src_offset),
                        chunk.output_bytes,
                        rank_slice,
                        error)) {
                    return fail_with(comm, ncclSystemError, error);
                }
                std::memcpy(chunk_input.data() + dst_offset, rank_slice.data(), chunk.output_bytes);
            }

            std::vector<char> chunk_output(chunk.output_bytes, 0);
            ncclResult_t result = submit_collective_chunk(
                config,
                command,
                collective_type,
                chunk_input.data(),
                chunk_output.data(),
                chunk.element_count,
                datatype,
                reduce_op,
                root,
                comm);
            if (result != ncclSuccess) {
                return result;
            }
            if (!fake_gpu::nccl::copy_host_to_buffer(
                    byte_pointer(output_bytes, chunk_offset_bytes),
                    chunk_output.data(),
                    chunk.output_bytes,
                    error)) {
                return fail_with(comm, ncclSystemError, error);
            }
            continue;
        }

        return fail_with(comm, ncclInvalidUsage, "unsupported chunked collective");
    }

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

    std::string error;
    if (comm->real_comm) {
        const ncclResult_t require_result = require_real_nccl(error);
        if (require_result != ncclSuccess) {
            if (!allow_missing) {
                return fail_with(comm, require_result, error);
            }
        } else {
            const ncclComm_t real_comm = reinterpret_cast<ncclComm_t>(comm->real_comm);
            const ncclResult_t result = allow_missing
                ? fake_gpu::nccl::RealNcclLoader::instance().comm_abort(real_comm, error)
                : fake_gpu::nccl::RealNcclLoader::instance().comm_destroy(real_comm, error);
            if (result != ncclSuccess && !allow_missing) {
                return real_nccl_error(comm, result, error);
            }
        }
    }

    fake_gpu::distributed::DistributedConfig config;
    if (comm->comm_id > 0 &&
        requires_coordinator(comm->dist_mode) &&
        validate_runtime_config(comm, config, error)) {
        std::ostringstream request;
        request << "DESTROY_COMM"
                << " comm_id=" << comm->comm_id
                << " rank=" << comm->rank;

        fake_gpu::distributed::CoordinatorResponse response;
        if (!coordinator_request_response(config, request.str(), response, error)) {
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
    } else if (requires_coordinator(comm->dist_mode) && !allow_missing) {
        return ncclInvalidUsage;
    }

    comm->real_comm = nullptr;
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

    for (GroupCollectiveCall& operation : g_group_operations) {
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
                operation.comm,
                ncclSum,
                nullptr);
        } else if (operation.type == fake_gpu::distributed::CollectiveType::Reduce) {
            void* recv_target = operation.recvbuff;
            if (!operation.recv_scratch.empty()) {
                recv_target = operation.recv_scratch.data();
            }
            result = submit_collective(
                "REDUCE",
                fake_gpu::distributed::CollectiveType::Reduce,
                operation.sendbuff,
                recv_target,
                operation.count,
                operation.datatype,
                operation.reduce_op,
                operation.root,
                operation.comm,
                ncclSum,
                nullptr);
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
                operation.comm,
                ncclSum,
                nullptr);
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
                operation.comm,
                ncclSum,
                nullptr);
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
                operation.comm,
                ncclSum,
                nullptr);
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
    const fake_gpu::distributed::DistributedConfig& config =
        fake_gpu::BackendConfig::instance().distributed_config();
    if (uses_real_nccl(config.mode)) {
        std::string error;
        const ncclResult_t require_result = require_real_nccl(error);
        if (require_result != ncclSuccess) {
            return fail_with(nullptr, require_result, error);
        }
        const ncclResult_t result =
            fake_gpu::nccl::RealNcclLoader::instance().get_version(version, error);
        if (result != ncclSuccess) {
            return real_nccl_error(nullptr, result, error);
        }
        g_last_error.clear();
        return ncclSuccess;
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

    const fake_gpu::distributed::DistributedConfig& config =
        fake_gpu::BackendConfig::instance().distributed_config();
    if (uses_real_nccl(config.mode)) {
        std::string error;
        const ncclResult_t require_result = require_real_nccl(error);
        if (require_result != ncclSuccess) {
            return fail_with(nullptr, require_result, error);
        }
        const ncclResult_t result =
            fake_gpu::nccl::RealNcclLoader::instance().get_unique_id(unique_id, error);
        if (result != ncclSuccess) {
            return real_nccl_error(nullptr, result, error);
        }
        g_last_error.clear();
        return ncclSuccess;
    }

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

    int coordinator_comm_id = -1;
    if (requires_coordinator(config.mode)) {
        std::ostringstream request;
        request << "INIT_COMM"
                << " unique_id=" << unique_id
                << " world_size=" << nranks
                << " rank=" << rank
                << " timeout_ms=" << kCoordinatorTimeoutMs;

        fake_gpu::distributed::CoordinatorResponse response;
        if (!coordinator_request_response(config, request.str(), response, error)) {
            return fail_with(nullptr, ncclSystemError, error);
        }
        if (!response.ok) {
            return fail_with(
                nullptr,
                map_response_error(response.error_code),
                response.error_detail.empty() ? response.error_code : response.error_detail);
        }
        if (!parse_int_field(response, "comm_id", coordinator_comm_id) || coordinator_comm_id <= 0) {
            return fail_with(nullptr, ncclInternalError, "coordinator did not return a valid comm_id");
        }
    }

    ncclComm_t real_comm = nullptr;
    if (uses_real_nccl(config.mode)) {
        const ncclResult_t require_result = require_real_nccl(error);
        if (require_result != ncclSuccess) {
            return fail_with(nullptr, require_result, error);
        }
        const ncclResult_t result = fake_gpu::nccl::RealNcclLoader::instance().comm_init_rank(
            &real_comm,
            nranks,
            comm_id,
            rank,
            error);
        if (result != ncclSuccess) {
            if (coordinator_comm_id > 0) {
                std::ostringstream cleanup_request;
                cleanup_request << "DESTROY_COMM"
                                << " comm_id=" << coordinator_comm_id
                                << " rank=" << rank;
                fake_gpu::distributed::CoordinatorResponse cleanup_response;
                std::string cleanup_error;
                coordinator_request_response(
                    config,
                    cleanup_request.str(),
                    cleanup_response,
                    cleanup_error);
            }
            return real_nccl_error(nullptr, result, error);
        }
    }

    ncclComm* state = new ncclComm();
    state->dist_mode = config.mode;
    state->comm_id = coordinator_comm_id;
    state->world_size = nranks;
    state->rank = rank;
    state->device = infer_device_for_rank(rank);
    state->real_comm = reinterpret_cast<void*>(real_comm);
    *comm = state;
    clear_last_error(*comm);
    return ncclSuccess;
}

ncclResult_t ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist) {
    if (!comm) {
        return fail_with(nullptr, ncclInvalidArgument, "comm must not be null");
    }
    if (ndev <= 0) {
        return fail_with(nullptr, ncclInvalidArgument, "ndev must be > 0");
    }

    std::vector<int> devices(static_cast<std::size_t>(ndev), 0);
    for (int index = 0; index < ndev; ++index) {
        comm[index] = nullptr;
        devices[static_cast<std::size_t>(index)] =
            devlist ? devlist[index] : index;
    }

    ncclUniqueId unique_id {};
    ncclResult_t result = ncclGetUniqueId(&unique_id);
    if (result != ncclSuccess) {
        return result;
    }

    std::vector<ncclComm_t> local_comms(static_cast<std::size_t>(ndev), nullptr);
    std::vector<ncclResult_t> results(static_cast<std::size_t>(ndev), ncclInternalError);
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(ndev));

    for (int rank = 0; rank < ndev; ++rank) {
        threads.emplace_back([&, rank]() {
            results[static_cast<std::size_t>(rank)] =
                ncclCommInitRank(&local_comms[static_cast<std::size_t>(rank)], ndev, unique_id, rank);
        });
    }

    for (std::thread& thread : threads) {
        thread.join();
    }

    ncclResult_t first_error = ncclSuccess;
    for (int rank = 0; rank < ndev; ++rank) {
        const ncclResult_t init_result = results[static_cast<std::size_t>(rank)];
        if (init_result != ncclSuccess && first_error == ncclSuccess) {
            first_error = init_result;
        }
    }

    if (first_error != ncclSuccess) {
        for (ncclComm_t local_comm : local_comms) {
            if (local_comm) {
                ncclCommAbort(local_comm);
            }
        }
        return first_error;
    }

    for (int rank = 0; rank < ndev; ++rank) {
        ncclComm_t local_comm = local_comms[static_cast<std::size_t>(rank)];
        if (local_comm) {
            local_comm->device = devices[static_cast<std::size_t>(rank)];
        }
        comm[rank] = local_comm;
    }

    clear_last_error(nullptr);
    return ncclSuccess;
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
    const void* sendbuff,
    void* recvbuff,
    std::size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    int root,
    ncclComm_t comm,
    cudaStream_t stream) {
    if (!comm) {
        return fail_with(nullptr, ncclInvalidArgument, "communicator must not be null");
    }
    if (g_group_depth > 0) {
        return buffer_group_reduce(sendbuff, recvbuff, count, datatype, op, root, comm);
    }
    if (!sendbuff) {
        return fail_with(comm, ncclInvalidArgument, "send buffer must not be null");
    }
    if (comm->rank == root && !recvbuff) {
        return fail_with(comm, ncclInvalidArgument, "root recv buffer must not be null");
    }
    if (count == 0) {
        return fail_with(comm, ncclInvalidArgument, "count must be > 0");
    }
    if (root < 0 || root >= comm->world_size) {
        return fail_with(comm, ncclInvalidArgument, "reduce root must be within [0, world_size)");
    }

    fake_gpu::distributed::CollectiveReduceOp reduce_op;
    if (!map_reduce_op(op, reduce_op)) {
        return fail_with(comm, ncclInvalidArgument, "unsupported reduce op");
    }
    fake_gpu::distributed::CollectiveDataType mapped_dtype;
    if (!map_dtype(datatype, mapped_dtype)) {
        return fail_with(comm, ncclInvalidArgument, "unsupported dtype for this collective");
    }

    fake_gpu::distributed::DistributedConfig config;
    std::string error;
    if (!validate_runtime_config(comm, config, error)) {
        return ncclInvalidUsage;
    }

    if (comm->dist_mode == fake_gpu::distributed::DistributedMode::Proxy) {
        ncclResult_t result = submit_proxy_collective_record(
            config,
            "REDUCE",
            fake_gpu::distributed::CollectiveType::Reduce,
            count,
            datatype,
            reduce_op,
            root,
            comm);
        if (result != ncclSuccess) {
            return result;
        }
        return submit_real_collective(
            fake_gpu::distributed::CollectiveType::Reduce,
            sendbuff,
            recvbuff,
            count,
            datatype,
            op,
            root,
            comm,
            stream);
    }

    if (comm->dist_mode == fake_gpu::distributed::DistributedMode::Passthrough) {
        return submit_real_collective(
            fake_gpu::distributed::CollectiveType::Reduce,
            sendbuff,
            recvbuff,
            count,
            datatype,
            op,
            root,
            comm,
            stream);
    }

    void* output = recvbuff;
    std::vector<char> recv_scratch;
    if (comm->rank != root) {
        const std::size_t scratch_bytes =
            fake_gpu::distributed::collective_data_type_size(mapped_dtype) * count;
        recv_scratch.assign(scratch_bytes, '\0');
        output = recv_scratch.data();
    }

    return submit_collective(
        "REDUCE",
        fake_gpu::distributed::CollectiveType::Reduce,
        sendbuff,
        output,
        count,
        datatype,
        reduce_op,
        root,
        comm,
        op,
        stream);
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
    cudaStream_t stream) {
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
        comm,
        ncclSum,
        stream);
}

ncclResult_t ncclAllReduce(
    const void* sendbuff,
    void* recvbuff,
    std::size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    cudaStream_t stream) {
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
        comm,
        op,
        stream);
}

ncclResult_t ncclReduceScatter(
    const void* sendbuff,
    void* recvbuff,
    std::size_t recvcount,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    cudaStream_t stream) {
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
        comm,
        op,
        stream);
}

ncclResult_t ncclAllGather(
    const void* sendbuff,
    void* recvbuff,
    std::size_t sendcount,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream) {
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
        comm,
        ncclSum,
        stream);
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
