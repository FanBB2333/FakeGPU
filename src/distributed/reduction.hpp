#pragma once

#include "collective_executor.hpp"
#include "low_precision.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace fake_gpu::distributed {

inline bool is_supported_reduce_op(CollectiveReduceOp op) {
    return op == CollectiveReduceOp::Sum ||
        op == CollectiveReduceOp::Prod ||
        op == CollectiveReduceOp::Max ||
        op == CollectiveReduceOp::Min ||
        op == CollectiveReduceOp::Avg;
}

template <typename T>
T apply_reduce_op(T lhs, T rhs, CollectiveReduceOp op) {
    switch (op) {
        case CollectiveReduceOp::Sum:
        case CollectiveReduceOp::Avg:
            return static_cast<T>(lhs + rhs);
        case CollectiveReduceOp::Prod:
            return static_cast<T>(lhs * rhs);
        case CollectiveReduceOp::Max:
            return std::max(lhs, rhs);
        case CollectiveReduceOp::Min:
            return std::min(lhs, rhs);
        case CollectiveReduceOp::None:
            break;
    }
    return lhs;
}

template <typename T>
void reduce_typed_buffers(
    const CollectiveExecutionRequest& request,
    const std::vector<std::vector<char>>& buffers,
    std::size_t element_count,
    std::vector<char>& reduced_bytes) {
    std::vector<T> reduced(element_count, static_cast<T>(0));
    if (!buffers.empty()) {
        std::memcpy(
            reduced.data(),
            buffers.front().data(),
            element_count * sizeof(T));
    }
    for (std::size_t buffer_index = 1;
         buffer_index < buffers.size();
         ++buffer_index) {
        const T* values = reinterpret_cast<const T*>(
            buffers[buffer_index].data());
        for (std::size_t index = 0; index < element_count; ++index) {
            reduced[index] = apply_reduce_op(
                reduced[index],
                values[index],
                request.reduce_op);
        }
    }
    if (request.reduce_op == CollectiveReduceOp::Avg && !buffers.empty()) {
        const T divisor = static_cast<T>(buffers.size());
        for (T& value : reduced) {
            value = static_cast<T>(value / divisor);
        }
    }
    reduced_bytes.assign(
        reinterpret_cast<const char*>(reduced.data()),
        reinterpret_cast<const char*>(reduced.data()) +
            element_count * sizeof(T));
}

template <typename Decode, typename Encode>
void reduce_16bit_buffers(
    const CollectiveExecutionRequest& request,
    const std::vector<std::vector<char>>& buffers,
    std::size_t element_count,
    Decode decode,
    Encode encode,
    std::vector<char>& reduced_bytes) {
    std::vector<float> reduced(element_count, 0.0f);
    if (!buffers.empty()) {
        for (std::size_t index = 0; index < element_count; ++index) {
            std::uint16_t bits = 0;
            std::memcpy(
                &bits,
                buffers.front().data() + index * sizeof(bits),
                sizeof(bits));
            reduced[index] = decode(bits);
        }
    }
    for (std::size_t buffer_index = 1;
         buffer_index < buffers.size();
         ++buffer_index) {
        for (std::size_t index = 0; index < element_count; ++index) {
            std::uint16_t bits = 0;
            std::memcpy(
                &bits,
                buffers[buffer_index].data() + index * sizeof(bits),
                sizeof(bits));
            reduced[index] = apply_reduce_op(
                reduced[index],
                decode(bits),
                request.reduce_op);
        }
    }
    if (request.reduce_op == CollectiveReduceOp::Avg && !buffers.empty()) {
        const float divisor = static_cast<float>(buffers.size());
        for (float& value : reduced) {
            value /= divisor;
        }
    }

    std::vector<std::uint16_t> encoded(element_count, 0);
    for (std::size_t index = 0; index < element_count; ++index) {
        encoded[index] = encode(reduced[index]);
    }
    reduced_bytes.assign(
        reinterpret_cast<const char*>(encoded.data()),
        reinterpret_cast<const char*>(encoded.data()) +
            element_count * sizeof(std::uint16_t));
}

inline void reduce_buffers(
    const CollectiveExecutionRequest& request,
    const std::vector<std::vector<char>>& buffers,
    std::size_t element_count,
    std::vector<char>& reduced_bytes) {
    switch (request.dtype) {
        case CollectiveDataType::Int8:
            reduce_typed_buffers<std::int8_t>(
                request, buffers, element_count, reduced_bytes);
            break;
        case CollectiveDataType::Uint8:
            reduce_typed_buffers<std::uint8_t>(
                request, buffers, element_count, reduced_bytes);
            break;
        case CollectiveDataType::Int32:
            reduce_typed_buffers<std::int32_t>(
                request, buffers, element_count, reduced_bytes);
            break;
        case CollectiveDataType::Uint32:
            reduce_typed_buffers<std::uint32_t>(
                request, buffers, element_count, reduced_bytes);
            break;
        case CollectiveDataType::Int64:
            reduce_typed_buffers<std::int64_t>(
                request, buffers, element_count, reduced_bytes);
            break;
        case CollectiveDataType::Uint64:
            reduce_typed_buffers<std::uint64_t>(
                request, buffers, element_count, reduced_bytes);
            break;
        case CollectiveDataType::Float16:
            reduce_16bit_buffers(
                request,
                buffers,
                element_count,
                float16_bits_to_float,
                float_to_float16_bits,
                reduced_bytes);
            break;
        case CollectiveDataType::BFloat16:
            reduce_16bit_buffers(
                request,
                buffers,
                element_count,
                bfloat16_bits_to_float,
                float_to_bfloat16_bits,
                reduced_bytes);
            break;
        case CollectiveDataType::Float32:
            reduce_typed_buffers<float>(
                request, buffers, element_count, reduced_bytes);
            break;
        case CollectiveDataType::Float64:
            reduce_typed_buffers<double>(
                request, buffers, element_count, reduced_bytes);
            break;
    }
}

}  // namespace fake_gpu::distributed
