#pragma once

#include "../cuda/cudart_defs.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace fake_gpu::nccl {

bool copy_buffer_to_host(
    const void* src,
    std::size_t bytes,
    std::vector<char>& out,
    std::string& error,
    cudaStream_t stream = nullptr);

bool copy_host_to_buffer(
    void* dst,
    const void* src,
    std::size_t bytes,
    std::string& error,
    cudaStream_t stream = nullptr);

}  // namespace fake_gpu::nccl
