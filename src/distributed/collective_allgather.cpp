#include "collective_executor.hpp"

#include "collective_slice_plan.hpp"
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

CollectiveExecutionResult execute_allgather(
    const CollectiveExecutionRequest& request,
    const std::vector<CollectiveExecutionParticipant>& participants) {
    CollectiveSlicePlan plan;
    std::string error;
    if (!build_even_slice_plan(request, participants.size(), plan, error)) {
        return make_error("invalid_slice_plan", error);
    }
    if (request.bytes != plan.total_bytes) {
        return make_error("invalid_collective_size", "allgather bytes do not match world_size * count * dtype_size");
    }

    StagingBufferManager manager;
    std::vector<StagingBufferHandle> handles;
    handles.reserve(participants.size());
    std::vector<unsigned char> gathered(plan.total_bytes, 0);

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
        metadata.shape = {plan.total_elements};
        metadata.owner_rank = participant.rank;
        metadata.staging_id = request.seqno;

        StagingBufferHandle handle;
        if (!manager.open(metadata, false, handle, error)) {
            return make_error("staging_open_failed", error);
        }
        std::memcpy(
            gathered.data() + plan.byte_offset_for_rank(participant.rank),
            handle.data(),
            plan.chunk_bytes);
        handles.push_back(std::move(handle));
    }

    for (StagingBufferHandle& handle : handles) {
        std::memcpy(handle.data(), gathered.data(), gathered.size());
    }

    CollectiveExecutionResult result;
    result.ok = true;
    return result;
}

}  // namespace fake_gpu::distributed
