#include "collective_executor.hpp"

#include "buffer_transfer.hpp"

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

CollectiveExecutionResult execute_broadcast(
    const CollectiveExecutionRequest& request,
    const std::vector<CollectiveExecutionParticipant>& participants) {
    const std::size_t dtype_size = collective_data_type_size(request.dtype);
    if (dtype_size == 0) {
        return make_error("unsupported_dtype", "unsupported broadcast dtype");
    }
    if (request.bytes != request.count * dtype_size) {
        return make_error("invalid_collective_size", "broadcast bytes do not match count * dtype_size");
    }

    std::vector<std::vector<char>> buffers;
    buffers.reserve(participants.size());

    int root_index = -1;
    for (std::size_t index = 0; index < participants.size(); ++index) {
        const CollectiveExecutionParticipant& participant = participants[index];
        if (participant.bytes != request.bytes) {
            return make_error(
                "staging_size_mismatch",
                "participant rank " + std::to_string(participant.rank) +
                    " reported bytes=" + std::to_string(participant.bytes) +
                    ", expected " + std::to_string(request.bytes));
        }

        std::string error;
        std::vector<char> buffer;
        if (!load_participant_buffer(
                participant.transport,
                participant.staging_name,
                request.dtype,
                participant.transport == BufferTransport::SocketPayload
                    ? participant.payload_bytes
                    : request.bytes,
                {request.count},
                participant.rank,
                request.seqno,
                participant.payload,
                buffer,
                error)) {
            return make_error("staging_open_failed", error);
        }
        buffers.push_back(std::move(buffer));

        if (participant.rank == request.root_rank) {
            root_index = static_cast<int>(index);
        }
    }

    if (root_index < 0) {
        return make_error("root_rank_missing", "broadcast root rank is not part of the collective");
    }

    std::vector<unsigned char> root_data(request.bytes, 0);
    std::memcpy(
        root_data.data(),
        buffers[static_cast<std::size_t>(root_index)].data(),
        request.bytes);

    CollectiveExecutionResult result;
    result.ok = true;
    for (const CollectiveExecutionParticipant& participant : participants) {
        std::string error;
        if (!store_participant_buffer(
                participant.transport,
                participant.staging_name,
                request.dtype,
                {request.count},
                participant.rank,
                request.seqno,
                root_data.data(),
                request.bytes,
                result.output_payloads[participant.rank],
                error)) {
            return make_error("staging_open_failed", error);
        }
    }

    return result;
}

}  // namespace fake_gpu::distributed
