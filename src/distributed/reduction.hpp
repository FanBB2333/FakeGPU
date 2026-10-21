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

// Folds buffers of Storage elements in Acc precision. Native dtypes use
// Storage == Acc; 16-bit floats are decoded to float and encoded back.
template <typename Storage, typename Acc, typename Decode, typename Encode>
void reduce_encoded_buffers(
    const CollectiveExecutionRequest& request,
    const std::vector<std::vector<char>>& buffers,
    std::size_t element_count,
    Decode decode,
    Encode encode,
    std::vector<char>& reduced_bytes) {
    const auto load = [&](const std::vector<char>& buffer, std::size_t index) {
        Storage stored {};
        std::memcpy(&stored, buffer.data() + index * sizeof(Storage), sizeof(Storage));
        return static_cast<Acc>(decode(stored));
    };
    std::vector<Acc> reduced(element_count, static_cast<Acc>(0));
    if (!buffers.empty()) {
        for (std::size_t index = 0; index < element_count; ++index) {
            reduced[index] = load(buffers.front(), index);
        }
    }
    for (std::size_t buffer_index = 1;
         buffer_index < buffers.size();
         ++buffer_index) {
        for (std::size_t index = 0; index < element_count; ++index) {
            reduced[index] = apply_reduce_op(
                reduced[index],
                load(buffers[buffer_index], index),
                request.reduce_op);
        }
    }
    if (request.reduce_op == CollectiveReduceOp::Avg && !buffers.empty()) {
        const Acc divisor = static_cast<Acc>(buffers.size());
        for (Acc& value : reduced) {
            value = static_cast<Acc>(value / divisor);
        }
    }

    reduced_bytes.assign(element_count * sizeof(Storage), '\0');
    for (std::size_t index = 0; index < element_count; ++index) {
        const Storage encoded = encode(reduced[index]);
        std::memcpy(
            reduced_bytes.data() + index * sizeof(Storage),
            &encoded,
            sizeof(Storage));
    }
}

template <typename T>
void reduce_typed_buffers(
    const CollectiveExecutionRequest& request,
    const std::vector<std::vector<char>>& buffers,
    std::size_t element_count,
    std::vector<char>& reduced_bytes) {
    const auto identity = [](T value) { return value; };
    reduce_encoded_buffers<T, T>(
        request, buffers, element_count, identity, identity, reduced_bytes);
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
            reduce_encoded_buffers<std::uint16_t, float>(
                request,
                buffers,
                element_count,
                float16_bits_to_float,
                float_to_float16_bits,
                reduced_bytes);
            break;
        case CollectiveDataType::BFloat16:
            reduce_encoded_buffers<std::uint16_t, float>(
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
