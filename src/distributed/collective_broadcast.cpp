#include "collective_executor.hpp"

#include "buffer_transfer.hpp"

#include <string>
#include <vector>

namespace fake_gpu::distributed {

CollectiveExecutionResult execute_broadcast(
    const CollectiveExecutionRequest& request,
    const std::vector<CollectiveExecutionParticipant>& participants) {
    const std::size_t dtype_size = collective_data_type_size(request.dtype);
    if (request.bytes != request.count * dtype_size) {
        return collective_error("invalid_collective_size", "broadcast bytes do not match count * dtype_size");
    }

    std::vector<std::vector<char>> buffers;
    buffers.reserve(participants.size());

    int root_index = -1;
    for (std::size_t index = 0; index < participants.size(); ++index) {
        const CollectiveExecutionParticipant& participant = participants[index];
        if (participant.bytes != request.bytes) {
            return collective_error(
                "staging_size_mismatch",
                participant_size_mismatch(participant, request));
        }

        std::string error;
        std::vector<char> buffer;
        if (!load_participant_input(
                request,
                participant,
                {request.count},
                buffer,
                error)) {
            return collective_error("staging_open_failed", error);
        }
        buffers.push_back(std::move(buffer));

        if (participant.rank == request.root_rank) {
            root_index = static_cast<int>(index);
        }
    }

    if (root_index < 0) {
        return collective_error("root_rank_missing", "broadcast root rank is not part of the collective");
    }

    const auto& root_data = buffers[static_cast<std::size_t>(root_index)];

    CollectiveExecutionResult result;
    result.ok = true;
    for (const CollectiveExecutionParticipant& participant : participants) {
        std::string error;
        if (!store_participant_output(
                request,
                participant,
                {request.count},
                root_data.data(),
                request.bytes,
                result,
                error)) {
            return collective_error("staging_open_failed", error);
        }
    }

    return result;
}

}  // namespace fake_gpu::distributed
