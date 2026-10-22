#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fake_gpu::distributed {

enum class CollectiveType {
    AllReduce,
    Reduce,
    Broadcast,
    AllGather,
    ReduceScatter,
    AllToAll,
};

enum class CollectiveDataType {
    Int8,
    Uint8,
    Int32,
    Uint32,
    Int64,
    Uint64,
    Float16,
    BFloat16,
    Float32,
    Float64,
    // ncclFloat8e4m3 and ncclFloat8e5m2.
    Float8E4M3,
    Float8E5M2,
};

enum class CollectiveReduceOp {
    None,
    Sum,
    Prod,
    Max,
    Min,
    Avg,
};

enum class BufferTransport {
    SharedMemory,
    SocketPayload,
};

struct CollectiveExecutionRequest {
    int comm_id = -1;
    std::uint64_t seqno = 0;
    CollectiveType type = CollectiveType::AllReduce;
    CollectiveDataType dtype = CollectiveDataType::Float32;
    std::size_t count = 0;
    int root_rank = -1;
    CollectiveReduceOp reduce_op = CollectiveReduceOp::None;
    std::size_t bytes = 0;
};

struct CollectiveExecutionParticipant {
    int rank = -1;
    BufferTransport transport = BufferTransport::SharedMemory;
    std::string staging_name;
    std::size_t bytes = 0;
    std::size_t payload_bytes = 0;
    std::vector<char> payload;
    std::vector<std::size_t> input_splits;
    std::vector<std::size_t> output_splits;
    std::size_t input_bytes = 0;
    std::size_t output_bytes = 0;
};

struct CollectiveExecutionResult {
    bool ok = false;
    std::string error_code;
    std::string error_detail;
    std::unordered_map<int, std::vector<char>> output_payloads;
};

inline CollectiveExecutionResult collective_error(std::string code, std::string detail) {
    CollectiveExecutionResult result;
    result.error_code = std::move(code);
    result.error_detail = std::move(detail);
    return result;
}

inline const char* collective_type_name(CollectiveType type) {
    switch (type) {
        case CollectiveType::AllReduce:
            return "allreduce";
        case CollectiveType::Reduce:
            return "reduce";
        case CollectiveType::Broadcast:
            return "broadcast";
        case CollectiveType::AllGather:
            return "allgather";
        case CollectiveType::ReduceScatter:
            return "reducescatter";
        case CollectiveType::AllToAll:
            return "alltoall";
    }
    return "unknown";
}

inline const char* collective_data_type_name(CollectiveDataType dtype) {
    switch (dtype) {
        case CollectiveDataType::Int8:
            return "int8";
        case CollectiveDataType::Uint8:
            return "uint8";
        case CollectiveDataType::Int32:
            return "int32";
        case CollectiveDataType::Uint32:
            return "uint32";
        case CollectiveDataType::Int64:
            return "int64";
        case CollectiveDataType::Uint64:
            return "uint64";
        case CollectiveDataType::Float16:
            return "float16";
        case CollectiveDataType::BFloat16:
            return "bfloat16";
        case CollectiveDataType::Float32:
            return "float32";
        case CollectiveDataType::Float64:
            return "float64";
        case CollectiveDataType::Float8E4M3:
            return "float8_e4m3";
        case CollectiveDataType::Float8E5M2:
            return "float8_e5m2";
    }
    return "unknown";
}

inline const char* collective_reduce_op_name(CollectiveReduceOp op) {
    switch (op) {
        case CollectiveReduceOp::None:
            return "none";
        case CollectiveReduceOp::Sum:
            return "sum";
        case CollectiveReduceOp::Prod:
            return "prod";
        case CollectiveReduceOp::Max:
            return "max";
        case CollectiveReduceOp::Min:
            return "min";
        case CollectiveReduceOp::Avg:
            return "avg";
    }
    return "unknown";
}

inline bool parse_collective_type(const std::string& text, CollectiveType& out) {
    for (CollectiveType value : {
             CollectiveType::AllReduce,
             CollectiveType::Reduce,
             CollectiveType::Broadcast,
             CollectiveType::AllGather,
             CollectiveType::ReduceScatter,
             CollectiveType::AllToAll,
         }) {
        if (text == collective_type_name(value)) {
            out = value;
            return true;
        }
    }
    return false;
}

inline bool parse_collective_data_type(const std::string& text, CollectiveDataType& out) {
    for (CollectiveDataType value : {
             CollectiveDataType::Int8,
             CollectiveDataType::Uint8,
             CollectiveDataType::Int32,
             CollectiveDataType::Uint32,
             CollectiveDataType::Int64,
             CollectiveDataType::Uint64,
             CollectiveDataType::Float16,
             CollectiveDataType::BFloat16,
             CollectiveDataType::Float32,
             CollectiveDataType::Float64,
             CollectiveDataType::Float8E4M3,
             CollectiveDataType::Float8E5M2,
         }) {
        if (text == collective_data_type_name(value)) {
            out = value;
            return true;
        }
    }
    return false;
}

inline bool parse_collective_reduce_op(const std::string& text, CollectiveReduceOp& out) {
    for (CollectiveReduceOp value : {
             CollectiveReduceOp::None,
             CollectiveReduceOp::Sum,
             CollectiveReduceOp::Prod,
             CollectiveReduceOp::Max,
             CollectiveReduceOp::Min,
             CollectiveReduceOp::Avg,
         }) {
        if (text == collective_reduce_op_name(value)) {
            out = value;
            return true;
        }
    }
    return false;
}

inline std::size_t collective_data_type_size(CollectiveDataType dtype) {
    switch (dtype) {
        case CollectiveDataType::Int8:
            return sizeof(std::int8_t);
        case CollectiveDataType::Uint8:
            return sizeof(std::uint8_t);
        case CollectiveDataType::Int32:
            return sizeof(std::int32_t);
        case CollectiveDataType::Uint32:
            return sizeof(std::uint32_t);
        case CollectiveDataType::Int64:
            return sizeof(std::int64_t);
        case CollectiveDataType::Uint64:
            return sizeof(std::uint64_t);
        case CollectiveDataType::Float16:
        case CollectiveDataType::BFloat16:
            return sizeof(std::uint16_t);
        case CollectiveDataType::Float32:
            return sizeof(float);
        case CollectiveDataType::Float64:
            return sizeof(double);
        case CollectiveDataType::Float8E4M3:
        case CollectiveDataType::Float8E5M2:
            return 1;
    }
    return 0;
}

CollectiveExecutionResult execute_allreduce_sum(
    const CollectiveExecutionRequest& request,
    const std::vector<CollectiveExecutionParticipant>& participants);

CollectiveExecutionResult execute_reduce(
    const CollectiveExecutionRequest& request,
    const std::vector<CollectiveExecutionParticipant>& participants);

CollectiveExecutionResult execute_broadcast(
    const CollectiveExecutionRequest& request,
    const std::vector<CollectiveExecutionParticipant>& participants);

CollectiveExecutionResult execute_allgather(
    const CollectiveExecutionRequest& request,
    const std::vector<CollectiveExecutionParticipant>& participants);

CollectiveExecutionResult execute_reducescatter(
    const CollectiveExecutionRequest& request,
    const std::vector<CollectiveExecutionParticipant>& participants);

CollectiveExecutionResult execute_alltoall(
    const CollectiveExecutionRequest& request,
    const std::vector<CollectiveExecutionParticipant>& participants);

}  // namespace fake_gpu::distributed
