#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fake_gpu::distributed {

enum class CollectiveType {
    AllReduce,
    Broadcast,
};

enum class CollectiveDataType {
    Int32,
    Int64,
    Float32,
    Float64,
};

enum class CollectiveReduceOp {
    None,
    Sum,
    Prod,
    Max,
    Min,
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
    std::string staging_name;
    std::size_t bytes = 0;
};

struct CollectiveExecutionResult {
    bool ok = false;
    std::string error_code;
    std::string error_detail;
};

inline const char* collective_type_name(CollectiveType type) {
    switch (type) {
        case CollectiveType::AllReduce:
            return "allreduce";
        case CollectiveType::Broadcast:
            return "broadcast";
    }
    return "unknown";
}

inline const char* collective_data_type_name(CollectiveDataType dtype) {
    switch (dtype) {
        case CollectiveDataType::Int32:
            return "int32";
        case CollectiveDataType::Int64:
            return "int64";
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
    }
    return "unknown";
}

inline bool parse_collective_type(const std::string& text, CollectiveType& out) {
    if (text == "allreduce") {
        out = CollectiveType::AllReduce;
        return true;
    }
    if (text == "broadcast") {
        out = CollectiveType::Broadcast;
        return true;
    }
    return false;
}

inline bool parse_collective_data_type(const std::string& text, CollectiveDataType& out) {
    if (text == "int32") {
        out = CollectiveDataType::Int32;
        return true;
    }
    if (text == "int64") {
        out = CollectiveDataType::Int64;
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
    return false;
}

inline std::size_t collective_data_type_size(CollectiveDataType dtype) {
    switch (dtype) {
        case CollectiveDataType::Int32:
            return sizeof(std::int32_t);
        case CollectiveDataType::Int64:
            return sizeof(std::int64_t);
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

CollectiveExecutionResult execute_broadcast(
    const CollectiveExecutionRequest& request,
    const std::vector<CollectiveExecutionParticipant>& participants);

}  // namespace fake_gpu::distributed
