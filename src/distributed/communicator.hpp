#pragma once

#include "collective_executor.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace fake_gpu::distributed {

struct CommunicatorRegistrationResult {
    bool ok = false;
    int comm_id = -1;
    std::uint64_t seqno = 0;
    std::string error_code;
    std::string error_detail;
};

struct CommunicatorDestroyResult {
    bool ok = false;
    std::string error_code;
    std::string error_detail;
};

struct CollectiveSubmitRequest {
    int comm_id = -1;
    int rank = -1;
    std::uint64_t seqno = 0;
    CollectiveType type = CollectiveType::AllReduce;
    CollectiveDataType dtype = CollectiveDataType::Float32;
    std::size_t count = 0;
    int root = -1;
    CollectiveReduceOp reduce_op = CollectiveReduceOp::None;
    std::string staging_name;
    std::size_t bytes = 0;
    int timeout_ms = 0;
};

struct CollectiveSubmitResult {
    bool ok = false;
    std::uint64_t seqno = 0;
    std::string error_code;
    std::string error_detail;
};

class CommunicatorRegistry {
public:
    CommunicatorRegistrationResult init_communicator(
        const std::string& unique_id,
        int world_size,
        int rank,
        int timeout_ms);

    CommunicatorDestroyResult destroy_communicator(int comm_id, int rank);
    CollectiveSubmitResult submit_collective(const CollectiveSubmitRequest& request);
};

}  // namespace fake_gpu::distributed
