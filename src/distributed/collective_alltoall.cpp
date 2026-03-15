#include "collective_executor.hpp"

#include "buffer_transfer.hpp"
#include "collective_slice_plan.hpp"

#include <cstring>
#include <string>
#include <vector>

namespace fake_gpu::distributed {

namespace {

CollectiveExecutionResult make_error(std::string code, std::string detail) {
    CollectiveExecutionResult result;
    result.ok = false;
    result.error_code = std::move(code);
    result.error_detail = std::move(detail);
    return result;
}

}  // namespace

CollectiveExecutionResult execute_alltoall(
    const CollectiveExecutionRequest& request,
    const std::vector<CollectiveExecutionParticipant>& participants) {
    CollectiveSlicePlan plan;
    std::string error;
    if (!build_even_slice_plan(request, participants.size(), plan, error)) {
        return make_error("invalid_slice_plan", error);
    }
    if (request.bytes != plan.total_bytes) {
        return make_error("invalid_collective_size", "alltoall bytes do not match world_size * count * dtype_size");
    }

    std::vector<std::vector<char>> buffers;
    buffers.reserve(participants.size());

    for (const CollectiveExecutionParticipant& participant : participants) {
        if (participant.bytes != request.bytes) {
            return make_error(
                "staging_size_mismatch",
                "participant rank " + std::to_string(participant.rank) +
                    " reported bytes=" + std::to_string(participant.bytes) +
                    ", expected " + std::to_string(request.bytes));
        }

        std::vector<char> buffer;
        if (!load_participant_buffer(
                participant.transport,
                participant.staging_name,
                request.dtype,
                participant.transport == BufferTransport::SocketPayload
                    ? participant.payload_bytes
                    : request.bytes,
                {plan.world_size, plan.chunk_elements},
                participant.rank,
                request.seqno,
                participant.payload,
                buffer,
                error)) {
            return make_error("staging_open_failed", error);
        }
        buffers.push_back(std::move(buffer));
    }

    std::vector<unsigned char> outputs(plan.total_bytes * participants.size(), 0);

    for (const CollectiveExecutionParticipant& sender : participants) {
        const unsigned char* sender_input =
            reinterpret_cast<const unsigned char*>(buffers[static_cast<std::size_t>(sender.rank)].data());
        for (const CollectiveExecutionParticipant& receiver : participants) {
            const std::size_t sender_offset = plan.byte_offset_for_rank(receiver.rank);
            const std::size_t receiver_offset = plan.byte_offset_for_rank(sender.rank);
            std::memcpy(
                outputs.data() + static_cast<std::size_t>(receiver.rank) * plan.total_bytes + receiver_offset,
                sender_input + sender_offset,
                plan.chunk_bytes);
        }
    }

    CollectiveExecutionResult result;
    result.ok = true;
    for (const CollectiveExecutionParticipant& participant : participants) {
        if (!store_participant_buffer(
                participant.transport,
                participant.staging_name,
                request.dtype,
                {plan.world_size, plan.chunk_elements},
                participant.rank,
                request.seqno,
                outputs.data() + static_cast<std::size_t>(participant.rank) * plan.total_bytes,
                plan.total_bytes,
                result.output_payloads[participant.rank],
                error)) {
            return make_error("staging_open_failed", error);
        }
    }

    return result;
}

}  // namespace fake_gpu::distributed
