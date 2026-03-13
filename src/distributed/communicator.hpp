#pragma once

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

class CommunicatorRegistry {
public:
    CommunicatorRegistrationResult init_communicator(
        const std::string& unique_id,
        int world_size,
        int rank,
        int timeout_ms);

    CommunicatorDestroyResult destroy_communicator(int comm_id, int rank);
};

}  // namespace fake_gpu::distributed
