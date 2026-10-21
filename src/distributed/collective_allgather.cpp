#include "collective_executor.hpp"

#include "buffer_transfer.hpp"
#include "collective_slice_plan.hpp"

#include <cstring>
#include <string>
#include <vector>

namespace fake_gpu::distributed {

CollectiveExecutionResult execute_allgather(
    const CollectiveExecutionRequest& request,
    const std::vector<CollectiveExecutionParticipant>& participants) {
    CollectiveSlicePlan plan;
    std::string error;
    if (!build_even_slice_plan(request, participants.size(), plan, error)) {
        return collective_error("invalid_slice_plan", error);
    }
    if (request.bytes != plan.total_bytes) {
        return collective_error("invalid_collective_size", "allgather bytes do not match world_size * count * dtype_size");
    }

    std::vector<unsigned char> gathered(plan.total_bytes, 0);

    for (std::size_t index = 0; index < participants.size(); ++index) {
        const CollectiveExecutionParticipant& participant = participants[index];
        if (participant.bytes != request.bytes) {
            return collective_error(
                "staging_size_mismatch",
                participant_size_mismatch(participant, request));
        }

        std::vector<char> buffer;
        if (!load_participant_input(
                request,
                participant,
                {plan.total_elements},
                buffer,
                error)) {
            return collective_error("staging_open_failed", error);
        }
        std::memcpy(
            gathered.data() + plan.byte_offset_for_rank(participant.rank),
            buffer.data(),
            plan.chunk_bytes);
    }

    CollectiveExecutionResult result;
    result.ok = true;
    for (const CollectiveExecutionParticipant& participant : participants) {
        if (!store_participant_output(
                request,
                participant,
                {plan.total_elements},
                gathered.data(),
                gathered.size(),
                result,
                error)) {
            return collective_error("staging_open_failed", error);
        }
    }

    return result;
}

}  // namespace fake_gpu::distributed
