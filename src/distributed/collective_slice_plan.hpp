#pragma once

#include "collective_executor.hpp"

#include <cstddef>
#include <string>

namespace fake_gpu::distributed {

struct CollectiveSlicePlan {
    std::size_t world_size = 0;
    std::size_t chunk_elements = 0;
    std::size_t chunk_bytes = 0;
    std::size_t total_elements = 0;
    std::size_t total_bytes = 0;

    std::size_t byte_offset_for_rank(int rank) const {
        return static_cast<std::size_t>(rank) * chunk_bytes;
    }
};

inline bool build_even_slice_plan(
    const CollectiveExecutionRequest& request,
    std::size_t world_size,
    CollectiveSlicePlan& out,
    std::string& error) {
    error.clear();
    if (world_size == 0) {
        error = "world_size must be > 0";
        return false;
    }

    const std::size_t dtype_size = collective_data_type_size(request.dtype);
    if (dtype_size == 0) {
        error = "unsupported dtype";
        return false;
    }

    out.world_size = world_size;
    out.chunk_elements = request.count;
    out.chunk_bytes = request.count * dtype_size;
    out.total_elements = out.chunk_elements * out.world_size;
    out.total_bytes = out.chunk_bytes * out.world_size;

    if (out.chunk_elements == 0 || out.chunk_bytes == 0) {
        error = "slice plan requires count > 0";
        return false;
    }
    return true;
}

}  // namespace fake_gpu::distributed
