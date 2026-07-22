#include "collective_executor.hpp"

#include "buffer_transfer.hpp"
#include "low_precision.hpp"

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

template <typename T>
void sum_or_average_reduce(
    const CollectiveExecutionRequest& request,
    const std::vector<std::vector<char>>& buffers,
    std::vector<char>& reduced_bytes) {
    std::vector<T> reduced(request.count, static_cast<T>(0));
    for (const std::vector<char>& buffer : buffers) {
        const T* values = reinterpret_cast<const T*>(buffer.data());
        for (std::size_t index = 0; index < request.count; ++index) {
            reduced[index] += values[index];
        }
    }
    if (request.reduce_op == CollectiveReduceOp::Avg) {
        const T divisor = static_cast<T>(buffers.size());
        for (T& value : reduced) {
            value = static_cast<T>(value / divisor);
        }
    }
    reduced_bytes.assign(
        reinterpret_cast<const char*>(reduced.data()),
        reinterpret_cast<const char*>(reduced.data()) + request.bytes);
}

template <typename Decode, typename Encode>
void sum_or_average_reduce_16bit(
    const CollectiveExecutionRequest& request,
    const std::vector<std::vector<char>>& buffers,
    Decode decode,
    Encode encode,
    std::vector<char>& reduced_bytes) {
    std::vector<std::uint16_t> reduced(request.count, 0);
    for (const std::vector<char>& buffer : buffers) {
        for (std::size_t index = 0; index < request.count; ++index) {
            std::uint16_t bits = 0;
            std::memcpy(
                &bits,
                buffer.data() + index * sizeof(bits),
                sizeof(bits));
            reduced[index] = encode(decode(reduced[index]) + decode(bits));
        }
    }
    if (request.reduce_op == CollectiveReduceOp::Avg) {
        const float divisor = static_cast<float>(buffers.size());
        for (std::uint16_t& value : reduced) {
            value = encode(decode(value) / divisor);
        }
    }
    reduced_bytes.assign(
        reinterpret_cast<const char*>(reduced.data()),
        reinterpret_cast<const char*>(reduced.data()) + request.bytes);
}

void sum_or_average_reduce_buffers(
    const CollectiveExecutionRequest& request,
    const std::vector<std::vector<char>>& buffers,
    std::vector<char>& reduced_bytes) {
    switch (request.dtype) {
        case CollectiveDataType::Int8:
            sum_or_average_reduce<std::int8_t>(request, buffers, reduced_bytes);
            break;
        case CollectiveDataType::Uint8:
            sum_or_average_reduce<std::uint8_t>(request, buffers, reduced_bytes);
            break;
        case CollectiveDataType::Int32:
            sum_or_average_reduce<std::int32_t>(request, buffers, reduced_bytes);
            break;
        case CollectiveDataType::Uint32:
            sum_or_average_reduce<std::uint32_t>(request, buffers, reduced_bytes);
            break;
        case CollectiveDataType::Int64:
            sum_or_average_reduce<std::int64_t>(request, buffers, reduced_bytes);
            break;
        case CollectiveDataType::Uint64:
            sum_or_average_reduce<std::uint64_t>(request, buffers, reduced_bytes);
            break;
        case CollectiveDataType::Float16:
            sum_or_average_reduce_16bit(
                request,
                buffers,
                float16_bits_to_float,
                float_to_float16_bits,
                reduced_bytes);
            break;
        case CollectiveDataType::BFloat16:
            sum_or_average_reduce_16bit(
                request,
                buffers,
                bfloat16_bits_to_float,
                float_to_bfloat16_bits,
                reduced_bytes);
            break;
        case CollectiveDataType::Float32:
            sum_or_average_reduce<float>(request, buffers, reduced_bytes);
            break;
        case CollectiveDataType::Float64:
            sum_or_average_reduce<double>(request, buffers, reduced_bytes);
            break;
    }
}

}  // namespace

CollectiveExecutionResult execute_allreduce_sum(
    const CollectiveExecutionRequest& request,
    const std::vector<CollectiveExecutionParticipant>& participants) {
    if (request.reduce_op != CollectiveReduceOp::Sum &&
        request.reduce_op != CollectiveReduceOp::Avg) {
        return make_error(
            "unsupported_reduce_op",
            "only allreduce(sum/avg) is implemented");
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
    sum_or_average_reduce_buffers(request, buffers, reduced_bytes);

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
    if (request.reduce_op != CollectiveReduceOp::Sum &&
        request.reduce_op != CollectiveReduceOp::Avg) {
        return make_error(
            "unsupported_reduce_op",
            "only reduce(sum/avg) is implemented");
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
    sum_or_average_reduce_buffers(request, buffers, reduced_bytes);

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
