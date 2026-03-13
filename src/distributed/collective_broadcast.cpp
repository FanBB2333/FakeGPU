#include "collective_executor.hpp"

#include "staging_buffer.hpp"

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

    StagingBufferManager manager;
    std::vector<StagingBufferHandle> handles;
    handles.reserve(participants.size());

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

        StagingBufferMetadata metadata;
        metadata.name = participant.staging_name;
        metadata.dtype = collective_data_type_name(request.dtype);
        metadata.bytes = request.bytes;
        metadata.shape = {request.count};
        metadata.owner_rank = participant.rank;
        metadata.staging_id = request.seqno;

        StagingBufferHandle handle;
        std::string error;
        if (!manager.open(metadata, false, handle, error)) {
            return make_error("staging_open_failed", error);
        }
        handles.push_back(std::move(handle));

        if (participant.rank == request.root_rank) {
            root_index = static_cast<int>(index);
        }
    }

    if (root_index < 0) {
        return make_error("root_rank_missing", "broadcast root rank is not part of the collective");
    }

    std::vector<unsigned char> root_data(request.bytes, 0);
    std::memcpy(root_data.data(), handles[static_cast<std::size_t>(root_index)].data(), request.bytes);

    for (StagingBufferHandle& handle : handles) {
        std::memcpy(handle.data(), root_data.data(), request.bytes);
    }

    CollectiveExecutionResult result;
    result.ok = true;
    return result;
}

}  // namespace fake_gpu::distributed
