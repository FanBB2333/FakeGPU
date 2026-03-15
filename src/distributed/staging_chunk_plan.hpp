#pragma once

#include "collective_executor.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

namespace fake_gpu::distributed {

struct StagingChunkPlanEntry {
    std::size_t chunk_index = 0;
    std::size_t offset_elements = 0;
    std::size_t element_count = 0;
    std::size_t input_bytes = 0;
    std::size_t staging_bytes = 0;
    std::size_t output_bytes = 0;
};

struct StagingChunkPlan {
    std::size_t total_elements = 0;
    std::size_t dtype_size = 0;
    std::size_t chunk_threshold_bytes = 0;
    std::vector<StagingChunkPlanEntry> chunks;

    bool chunked() const {
        return chunks.size() > 1;
    }
};

inline bool build_staging_chunk_plan(
    CollectiveType type,
    std::size_t element_count,
    CollectiveDataType dtype,
    int world_size,
    std::size_t chunk_threshold_bytes,
    StagingChunkPlan& out,
    std::string& error) {
    out = StagingChunkPlan{};
    error.clear();

    if (world_size <= 0) {
        error = "world_size must be > 0";
        return false;
    }

    const std::size_t dtype_size = collective_data_type_size(dtype);
    if (dtype_size == 0) {
        error = "unsupported dtype";
        return false;
    }
    if (element_count == 0) {
        error = "count must be > 0";
        return false;
    }

    const std::size_t staging_multiplier =
        (type == CollectiveType::AllGather || type == CollectiveType::ReduceScatter)
        ? static_cast<std::size_t>(world_size)
        : 1U;
    const std::size_t minimum_staging_bytes = dtype_size * staging_multiplier;

    std::size_t max_chunk_elements = element_count;
    if (chunk_threshold_bytes > 0) {
        if (chunk_threshold_bytes < minimum_staging_bytes) {
            error =
                "chunk threshold is too small for a single element: threshold_bytes=" +
                std::to_string(chunk_threshold_bytes) +
                ", minimum_bytes=" + std::to_string(minimum_staging_bytes);
            return false;
        }
        max_chunk_elements =
            std::max<std::size_t>(1, chunk_threshold_bytes / minimum_staging_bytes);
        max_chunk_elements = std::min(max_chunk_elements, element_count);
    }

    out.total_elements = element_count;
    out.dtype_size = dtype_size;
    out.chunk_threshold_bytes = chunk_threshold_bytes;

    for (std::size_t offset = 0, chunk_index = 0; offset < element_count; ++chunk_index) {
        const std::size_t chunk_elements =
            std::min(max_chunk_elements, element_count - offset);
        const std::size_t chunk_bytes = chunk_elements * dtype_size;

        StagingChunkPlanEntry chunk;
        chunk.chunk_index = chunk_index;
        chunk.offset_elements = offset;
        chunk.element_count = chunk_elements;

        switch (type) {
            case CollectiveType::AllReduce:
            case CollectiveType::Reduce:
            case CollectiveType::Broadcast:
                chunk.input_bytes = chunk_bytes;
                chunk.staging_bytes = chunk_bytes;
                chunk.output_bytes = chunk_bytes;
                break;
            case CollectiveType::AllGather:
                chunk.input_bytes = chunk_bytes;
                chunk.staging_bytes = chunk_bytes * static_cast<std::size_t>(world_size);
                chunk.output_bytes = chunk.staging_bytes;
                break;
            case CollectiveType::ReduceScatter:
                chunk.input_bytes = chunk_bytes * static_cast<std::size_t>(world_size);
                chunk.staging_bytes = chunk.input_bytes;
                chunk.output_bytes = chunk_bytes;
                break;
        }

        out.chunks.push_back(chunk);
        offset += chunk_elements;
    }

    return true;
}

}  // namespace fake_gpu::distributed
