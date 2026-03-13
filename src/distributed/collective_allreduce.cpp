#include "collective_executor.hpp"

#include "staging_buffer.hpp"

#include <cstdint>
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

bool open_participant_buffers(
    const CollectiveExecutionRequest& request,
    const std::vector<CollectiveExecutionParticipant>& participants,
    std::vector<StagingBufferHandle>& handles,
    std::string& error) {
    handles.clear();
    StagingBufferManager manager;

    for (const CollectiveExecutionParticipant& participant : participants) {
        if (participant.bytes != request.bytes) {
            error =
                "participant rank " + std::to_string(participant.rank) +
                " reported bytes=" + std::to_string(participant.bytes) +
                ", expected " + std::to_string(request.bytes);
            return false;
        }

        StagingBufferMetadata metadata;
        metadata.name = participant.staging_name;
        metadata.dtype = collective_data_type_name(request.dtype);
        metadata.bytes = request.bytes;
        metadata.shape = {request.count};
        metadata.owner_rank = participant.rank;
        metadata.staging_id = request.seqno;

        StagingBufferHandle handle;
        if (!manager.open(metadata, false, handle, error)) {
            return false;
        }
        handles.push_back(std::move(handle));
    }

    return true;
}

}  // namespace

CollectiveExecutionResult execute_allreduce_sum(
    const CollectiveExecutionRequest& request,
    const std::vector<CollectiveExecutionParticipant>& participants) {
    if (request.reduce_op != CollectiveReduceOp::Sum) {
        return make_error("unsupported_reduce_op", "only allreduce(sum) is implemented");
    }

    const std::size_t dtype_size = collective_data_type_size(request.dtype);
    if (dtype_size == 0) {
        return make_error("unsupported_dtype", "unsupported allreduce dtype");
    }
    if (request.bytes != request.count * dtype_size) {
        return make_error("invalid_collective_size", "allreduce bytes do not match count * dtype_size");
    }

    std::vector<StagingBufferHandle> handles;
    std::string error;
    if (!open_participant_buffers(request, participants, handles, error)) {
        return make_error("staging_open_failed", error);
    }

    if (request.dtype == CollectiveDataType::Float32) {
        std::vector<float> reduced(request.count, 0.0f);
        for (const StagingBufferHandle& handle : handles) {
            const float* values = static_cast<const float*>(handle.data());
            for (std::size_t index = 0; index < request.count; ++index) {
                reduced[index] += values[index];
            }
        }
        for (StagingBufferHandle& handle : handles) {
            std::memcpy(handle.data(), reduced.data(), request.bytes);
        }
        CollectiveExecutionResult result;
        result.ok = true;
        return result;
    }

    std::vector<std::int32_t> reduced(request.count, 0);
    for (const StagingBufferHandle& handle : handles) {
        const std::int32_t* values = static_cast<const std::int32_t*>(handle.data());
        for (std::size_t index = 0; index < request.count; ++index) {
            reduced[index] += values[index];
        }
    }
    for (StagingBufferHandle& handle : handles) {
        std::memcpy(handle.data(), reduced.data(), request.bytes);
    }

    CollectiveExecutionResult result;
    result.ok = true;
    return result;
}

}  // namespace fake_gpu::distributed
