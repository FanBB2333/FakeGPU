#include "collective_executor.hpp"

#include "buffer_transfer.hpp"
#include "collective_slice_plan.hpp"

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
    const std::vector<std::vector<char>>& buffers,
    std::vector<std::vector<char>>& outputs) {
    std::vector<T> reduced(plan.total_elements, static_cast<T>(0));
    for (const std::vector<char>& buffer : buffers) {
        const T* values = reinterpret_cast<const T*>(buffer.data());
        for (std::size_t index = 0; index < plan.total_elements; ++index) {
            reduced[index] += values[index];
        }
    }

    outputs.resize(buffers.size());
    for (std::size_t index = 0; index < buffers.size(); ++index) {
        outputs[index].assign(plan.chunk_bytes, '\0');
        std::memcpy(
            outputs[index].data(),
            reduced.data() + index * plan.chunk_elements,
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

    std::vector<std::vector<char>> outputs;
    switch (request.dtype) {
        case CollectiveDataType::Int32:
            reduce_scatter_sum_into_handles<std::int32_t>(plan, buffers, outputs);
            break;
        case CollectiveDataType::Int64:
            reduce_scatter_sum_into_handles<std::int64_t>(plan, buffers, outputs);
            break;
        case CollectiveDataType::Float32:
            reduce_scatter_sum_into_handles<float>(plan, buffers, outputs);
            break;
        case CollectiveDataType::Float64:
            reduce_scatter_sum_into_handles<double>(plan, buffers, outputs);
            break;
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
