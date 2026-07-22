#include "collective_executor.hpp"

#include "buffer_transfer.hpp"
#include "collective_slice_plan.hpp"
#include "reduction.hpp"

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

CollectiveExecutionResult execute_reducescatter(
    const CollectiveExecutionRequest& request,
    const std::vector<CollectiveExecutionParticipant>& participants) {
    if (!is_supported_reduce_op(request.reduce_op)) {
        return make_error(
            "unsupported_reduce_op",
            "unsupported reducescatter operation");
    }

    CollectiveSlicePlan plan;
    std::string error;
    if (!build_even_slice_plan(request, participants.size(), plan, error)) {
        return make_error("invalid_slice_plan", error);
    }
    if (request.bytes != plan.total_bytes) {
        return make_error("invalid_collective_size", "reducescatter bytes do not match world_size * count * dtype_size");
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
                {plan.total_elements},
                participant.rank,
                request.seqno,
                participant.payload,
                buffer,
                error)) {
            return make_error("staging_open_failed", error);
        }
        buffers.push_back(std::move(buffer));
    }

    std::vector<char> reduced;
    reduce_buffers(request, buffers, plan.total_elements, reduced);

    std::vector<std::vector<char>> outputs(buffers.size());
    for (std::size_t index = 0; index < buffers.size(); ++index) {
        const std::size_t offset = index * plan.chunk_bytes;
        outputs[index].assign(
            reduced.begin() + static_cast<std::ptrdiff_t>(offset),
            reduced.begin() + static_cast<std::ptrdiff_t>(
                offset + plan.chunk_bytes));
    }

    CollectiveExecutionResult result;
    result.ok = true;
    for (std::size_t index = 0; index < participants.size(); ++index) {
        const CollectiveExecutionParticipant& participant = participants[index];
        if (!store_participant_buffer(
                participant.transport,
                participant.staging_name,
                request.dtype,
                {plan.chunk_elements},
                participant.rank,
                request.seqno,
                outputs[index].data(),
                outputs[index].size(),
                result.output_payloads[participant.rank],
                error)) {
            return make_error("staging_open_failed", error);
        }
    }
    return result;
}

}  // namespace fake_gpu::distributed
