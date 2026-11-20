#include "collective_executor.hpp"

#include "buffer_transfer.hpp"
#include "collective_slice_plan.hpp"
#include "reduction.hpp"

#include <string>
#include <vector>

namespace fake_gpu::distributed {

CollectiveExecutionResult execute_reducescatter(
    const CollectiveExecutionRequest& request,
    const std::vector<CollectiveExecutionParticipant>& participants) {
    if (!is_supported_reduce_op(request.reduce_op)) {
        return collective_error(
            "unsupported_reduce_op",
            "unsupported reducescatter operation");
    }

    CollectiveSlicePlan plan;
    std::string error;
    if (!build_even_slice_plan(request, participants.size(), plan, error)) {
        return collective_error("invalid_slice_plan", error);
    }
    if (request.bytes != plan.total_bytes) {
        return collective_error("invalid_collective_size", "reducescatter bytes do not match world_size * count * dtype_size");
    }

    std::vector<std::vector<char>> buffers;
    buffers.reserve(participants.size());

    for (const CollectiveExecutionParticipant& participant : participants) {
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
        buffers.push_back(std::move(buffer));
    }

    std::vector<char> reduced;
    reduce_buffers(request, buffers, plan.total_elements, reduced);

    CollectiveExecutionResult result;
    result.ok = true;
    for (std::size_t index = 0; index < participants.size(); ++index) {
        const CollectiveExecutionParticipant& participant = participants[index];
        if (!store_participant_output(
                request,
                participant,
                {plan.chunk_elements},
                reduced.data() + index * plan.chunk_bytes,
                plan.chunk_bytes,
                result,
                error)) {
            return collective_error("staging_open_failed", error);
        }
    }
    return result;
}

}  // namespace fake_gpu::distributed
