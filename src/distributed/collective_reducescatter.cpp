#include "collective_executor.hpp"

#include "collective_slice_plan.hpp"
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

template <typename T>
void reduce_scatter_sum_into_handles(
    const CollectiveSlicePlan& plan,
    std::vector<StagingBufferHandle>& handles,
    const std::vector<CollectiveExecutionParticipant>& participants) {
    std::vector<T> reduced(plan.total_elements, static_cast<T>(0));
    for (const StagingBufferHandle& handle : handles) {
        const T* values = static_cast<const T*>(handle.data());
        for (std::size_t index = 0; index < plan.total_elements; ++index) {
            reduced[index] += values[index];
        }
    }

    for (std::size_t index = 0; index < handles.size(); ++index) {
        const CollectiveExecutionParticipant& participant = participants[index];
        void* output = handles[index].data();
        std::memcpy(
            output,
            reduced.data() + participant.rank * plan.chunk_elements,
            plan.chunk_bytes);
    }
}

}  // namespace

CollectiveExecutionResult execute_reducescatter(
    const CollectiveExecutionRequest& request,
    const std::vector<CollectiveExecutionParticipant>& participants) {
    if (request.reduce_op != CollectiveReduceOp::Sum) {
        return make_error("unsupported_reduce_op", "only reducescatter(sum) is implemented");
    }

    CollectiveSlicePlan plan;
    std::string error;
    if (!build_even_slice_plan(request, participants.size(), plan, error)) {
        return make_error("invalid_slice_plan", error);
    }
    if (request.bytes != plan.total_bytes) {
        return make_error("invalid_collective_size", "reducescatter bytes do not match world_size * count * dtype_size");
    }

    StagingBufferManager manager;
    std::vector<StagingBufferHandle> handles;
    handles.reserve(participants.size());

    for (const CollectiveExecutionParticipant& participant : participants) {
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
        handles.push_back(std::move(handle));
    }

    switch (request.dtype) {
        case CollectiveDataType::Int32:
            reduce_scatter_sum_into_handles<std::int32_t>(plan, handles, participants);
            break;
        case CollectiveDataType::Int64:
            reduce_scatter_sum_into_handles<std::int64_t>(plan, handles, participants);
            break;
        case CollectiveDataType::Float32:
            reduce_scatter_sum_into_handles<float>(plan, handles, participants);
            break;
        case CollectiveDataType::Float64:
            reduce_scatter_sum_into_handles<double>(plan, handles, participants);
            break;
    }

    CollectiveExecutionResult result;
    result.ok = true;
    return result;
}

}  // namespace fake_gpu::distributed
