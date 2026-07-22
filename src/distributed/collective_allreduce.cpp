#include "collective_executor.hpp"

#include "buffer_transfer.hpp"
#include "reduction.hpp"

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

bool open_participant_buffers(
    const CollectiveExecutionRequest& request,
    const std::vector<CollectiveExecutionParticipant>& participants,
    const std::vector<std::size_t>& shape,
    std::vector<std::vector<char>>& buffers,
    std::string& error) {
    buffers.clear();

    for (const CollectiveExecutionParticipant& participant : participants) {
        if (participant.bytes != request.bytes) {
            error =
                "participant rank " + std::to_string(participant.rank) +
                " reported bytes=" + std::to_string(participant.bytes) +
                ", expected " + std::to_string(request.bytes);
            return false;
        }

        std::vector<char> buffer;
        if (!load_participant_buffer(
                participant.transport,
                participant.staging_name,
                request.dtype,
                participant.transport == BufferTransport::SocketPayload
                    ? participant.payload_bytes
                    : request.bytes,
                shape,
                participant.rank,
                request.seqno,
                participant.payload,
                buffer,
                error)) {
            return false;
        }
        buffers.push_back(std::move(buffer));
    }

    return true;
}

}  // namespace

CollectiveExecutionResult execute_allreduce_sum(
    const CollectiveExecutionRequest& request,
    const std::vector<CollectiveExecutionParticipant>& participants) {
    if (!is_supported_reduce_op(request.reduce_op)) {
        return make_error(
            "unsupported_reduce_op",
            "unsupported allreduce operation");
    }

    const std::size_t dtype_size = collective_data_type_size(request.dtype);
    if (dtype_size == 0) {
        return make_error("unsupported_dtype", "unsupported allreduce dtype");
    }
    if (request.bytes != request.count * dtype_size) {
        return make_error("invalid_collective_size", "allreduce bytes do not match count * dtype_size");
    }

    std::vector<std::vector<char>> buffers;
    std::string error;
    if (!open_participant_buffers(request, participants, {request.count}, buffers, error)) {
        return make_error("staging_open_failed", error);
    }

    std::vector<char> reduced_bytes;
    reduce_buffers(request, buffers, request.count, reduced_bytes);

    CollectiveExecutionResult result;
    result.ok = true;
    for (const CollectiveExecutionParticipant& participant : participants) {
        if (!store_participant_buffer(
                participant.transport,
                participant.staging_name,
                request.dtype,
                {request.count},
                participant.rank,
                request.seqno,
                reduced_bytes.data(),
                request.bytes,
                result.output_payloads[participant.rank],
                error)) {
            return make_error("staging_open_failed", error);
        }
    }
    return result;
}

CollectiveExecutionResult execute_reduce(
    const CollectiveExecutionRequest& request,
    const std::vector<CollectiveExecutionParticipant>& participants) {
    if (!is_supported_reduce_op(request.reduce_op)) {
        return make_error(
            "unsupported_reduce_op",
            "unsupported reduce operation");
    }

    const std::size_t dtype_size = collective_data_type_size(request.dtype);
    if (dtype_size == 0) {
        return make_error("unsupported_dtype", "unsupported reduce dtype");
    }
    if (request.bytes != request.count * dtype_size) {
        return make_error("invalid_collective_size", "reduce bytes do not match count * dtype_size");
    }

    bool has_root = false;
    for (const CollectiveExecutionParticipant& participant : participants) {
        if (participant.rank == request.root_rank) {
            has_root = true;
            break;
        }
    }
    if (!has_root) {
        return make_error("root_rank_missing", "reduce root rank is not part of the collective");
    }

    std::vector<std::vector<char>> buffers;
    std::string error;
    if (!open_participant_buffers(request, participants, {request.count}, buffers, error)) {
        return make_error("staging_open_failed", error);
    }

    std::vector<char> reduced_bytes;
    reduce_buffers(request, buffers, request.count, reduced_bytes);

    CollectiveExecutionResult result;
    result.ok = true;
    for (const CollectiveExecutionParticipant& participant : participants) {
        if (participant.rank != request.root_rank) {
            result.output_payloads[participant.rank].clear();
            continue;
        }
        if (!store_participant_buffer(
                participant.transport,
                participant.staging_name,
                request.dtype,
                {request.count},
                participant.rank,
                request.seqno,
                reduced_bytes.data(),
                request.bytes,
                result.output_payloads[participant.rank],
                error)) {
            return make_error("staging_open_failed", error);
        }
    }
    return result;
}

}  // namespace fake_gpu::distributed
