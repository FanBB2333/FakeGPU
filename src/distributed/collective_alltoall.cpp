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

CollectiveExecutionResult execute_variable_alltoall(
    const CollectiveExecutionRequest& request,
    const std::vector<CollectiveExecutionParticipant>& participants) {
    const std::size_t world_size = participants.size();
    const std::size_t dtype_size = collective_data_type_size(request.dtype);
    if (world_size == 0 || dtype_size == 0) {
        return make_error("invalid_split_plan", "invalid all-to-all world size or dtype");
    }

    std::vector<std::vector<char>> inputs(world_size);
    std::string error;
    for (const CollectiveExecutionParticipant& participant : participants) {
        if (participant.rank < 0 ||
            static_cast<std::size_t>(participant.rank) >= world_size ||
            participant.input_splits.size() != world_size ||
            participant.output_splits.size() != world_size) {
            return make_error(
                "invalid_split_plan",
                "all-to-all participant split vectors must match world_size");
        }
        const std::size_t load_bytes =
            participant.transport == BufferTransport::SocketPayload
            ? participant.input_bytes
            : participant.bytes;
        if (!load_participant_buffer(
                participant.transport,
                participant.staging_name,
                request.dtype,
                load_bytes,
                {load_bytes / dtype_size},
                participant.rank,
                request.seqno,
                participant.payload,
                inputs[static_cast<std::size_t>(participant.rank)],
                error)) {
            return make_error("staging_open_failed", error);
        }
    }

    std::vector<std::vector<char>> outputs(world_size);
    for (const CollectiveExecutionParticipant& participant : participants) {
        outputs[static_cast<std::size_t>(participant.rank)].assign(
            participant.output_bytes,
            '\0');
    }

    for (const CollectiveExecutionParticipant& sender : participants) {
        std::size_t sender_offset_elements = 0;
        for (const CollectiveExecutionParticipant& receiver : participants) {
            const std::size_t sender_count =
                sender.input_splits[static_cast<std::size_t>(receiver.rank)];
            const std::size_t receiver_count =
                receiver.output_splits[static_cast<std::size_t>(sender.rank)];
            if (sender_count != receiver_count) {
                return make_error(
                    "invalid_split_plan",
                    "all-to-all sender and receiver split sizes disagree");
            }

            std::size_t receiver_offset_elements = 0;
            for (int source = 0; source < sender.rank; ++source) {
                receiver_offset_elements +=
                    receiver.output_splits[static_cast<std::size_t>(source)];
            }
            const std::size_t transfer_bytes = sender_count * dtype_size;
            if (transfer_bytes > 0) {
                std::memcpy(
                    outputs[static_cast<std::size_t>(receiver.rank)].data() +
                        receiver_offset_elements * dtype_size,
                    inputs[static_cast<std::size_t>(sender.rank)].data() +
                        sender_offset_elements * dtype_size,
                    transfer_bytes);
            }
            sender_offset_elements += sender_count;
        }
    }

    CollectiveExecutionResult result;
    result.ok = true;
    for (const CollectiveExecutionParticipant& participant : participants) {
        const std::vector<char>& output =
            outputs[static_cast<std::size_t>(participant.rank)];
        if (participant.transport == BufferTransport::SocketPayload) {
            if (!store_participant_buffer(
                    participant.transport,
                    participant.staging_name,
                    request.dtype,
                    {participant.output_bytes / dtype_size},
                    participant.rank,
                    request.seqno,
                    output.data(),
                    participant.output_bytes,
                    result.output_payloads[participant.rank],
                    error)) {
                return make_error("staging_open_failed", error);
            }
            continue;
        }

        std::vector<char> staging_output(participant.bytes, '\0');
        std::memcpy(
            staging_output.data(),
            output.data(),
            participant.output_bytes);
        if (!store_participant_buffer(
                participant.transport,
                participant.staging_name,
                request.dtype,
                {participant.bytes / dtype_size},
                participant.rank,
                request.seqno,
                staging_output.data(),
                participant.bytes,
                result.output_payloads[participant.rank],
                error)) {
            return make_error("staging_open_failed", error);
        }
    }
    return result;
}

}  // namespace

CollectiveExecutionResult execute_alltoall(
    const CollectiveExecutionRequest& request,
    const std::vector<CollectiveExecutionParticipant>& participants) {
    if (!participants.empty() && !participants.front().input_splits.empty()) {
        return execute_variable_alltoall(request, participants);
    }
    CollectiveSlicePlan plan;
    std::string error;
    if (!build_even_slice_plan(request, participants.size(), plan, error)) {
        return make_error("invalid_slice_plan", error);
    }
    if (request.bytes != plan.total_bytes) {
        return make_error(
            "invalid_collective_size",
            "alltoall bytes=" + std::to_string(request.bytes) +
                " do not match expected=" + std::to_string(plan.total_bytes) +
                " for participants=" + std::to_string(participants.size()) +
                ", count=" + std::to_string(request.count) +
                ", dtype_size=" + std::to_string(
                    collective_data_type_size(request.dtype)));
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
