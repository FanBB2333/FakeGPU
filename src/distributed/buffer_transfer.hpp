#pragma once

#include "collective_executor.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fake_gpu::distributed {

bool load_participant_buffer(
    BufferTransport transport,
    const std::string& staging_name,
    CollectiveDataType dtype,
    std::size_t bytes,
    const std::vector<std::size_t>& shape,
    int owner_rank,
    std::uint64_t staging_id,
    const std::vector<char>& payload,
    std::vector<char>& out,
    std::string& error);

bool store_participant_buffer(
    BufferTransport transport,
    const std::string& staging_name,
    CollectiveDataType dtype,
    const std::vector<std::size_t>& shape,
    int owner_rank,
    std::uint64_t staging_id,
    const void* data,
    std::size_t bytes,
    std::vector<char>& output_payload,
    std::string& error);

}  // namespace fake_gpu::distributed
