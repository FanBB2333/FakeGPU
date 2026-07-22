#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
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
};

struct CollectiveExecutionResult {
    bool ok = false;
    std::string error_code;
    std::string error_detail;
    std::unordered_map<int, std::vector<char>> output_payloads;
};

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
    if (text == "allreduce") {
        out = CollectiveType::AllReduce;
        return true;
    }
    if (text == "reduce") {
        out = CollectiveType::Reduce;
        return true;
    }
    if (text == "broadcast") {
        out = CollectiveType::Broadcast;
        return true;
    }
    if (text == "allgather") {
        out = CollectiveType::AllGather;
        return true;
    }
    if (text == "reducescatter") {
        out = CollectiveType::ReduceScatter;
        return true;
    }
    if (text == "alltoall") {
        out = CollectiveType::AllToAll;
        return true;
    }
    return false;
}

inline bool parse_collective_data_type(const std::string& text, CollectiveDataType& out) {
    if (text == "int8") {
        out = CollectiveDataType::Int8;
        return true;
    }
    if (text == "uint8") {
        out = CollectiveDataType::Uint8;
        return true;
    }
    if (text == "int32") {
        out = CollectiveDataType::Int32;
        return true;
    }
    if (text == "uint32") {
        out = CollectiveDataType::Uint32;
        return true;
    }
    if (text == "int64") {
        out = CollectiveDataType::Int64;
        return true;
    }
    if (text == "uint64") {
        out = CollectiveDataType::Uint64;
        return true;
    }
    if (text == "float16") {
        out = CollectiveDataType::Float16;
        return true;
    }
    if (text == "bfloat16") {
        out = CollectiveDataType::BFloat16;
        return true;
    }
    if (text == "float32") {
        out = CollectiveDataType::Float32;
        return true;
    }
    if (text == "float64") {
        out = CollectiveDataType::Float64;
        return true;
    }
    return false;
}

inline bool parse_collective_reduce_op(const std::string& text, CollectiveReduceOp& out) {
    if (text == "none") {
        out = CollectiveReduceOp::None;
        return true;
    }
    if (text == "sum") {
        out = CollectiveReduceOp::Sum;
        return true;
    }
    if (text == "prod") {
        out = CollectiveReduceOp::Prod;
        return true;
    }
    if (text == "max") {
        out = CollectiveReduceOp::Max;
        return true;
    }
    if (text == "min") {
        out = CollectiveReduceOp::Min;
        return true;
    }
    if (text == "avg") {
        out = CollectiveReduceOp::Avg;
        return true;
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
