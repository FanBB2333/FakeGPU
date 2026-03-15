#include "buffer_transfer.hpp"

#include "staging_buffer.hpp"

#include <cstring>

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
    std::string& error) {
    error.clear();
    if (transport == BufferTransport::SocketPayload) {
        if (payload.size() != bytes) {
            error =
                "socket payload bytes=" + std::to_string(payload.size()) +
                ", expected " + std::to_string(bytes);
            return false;
        }
        out = payload;
        return true;
    }

    StagingBufferMetadata metadata;
    metadata.name = staging_name;
    metadata.dtype = collective_data_type_name(dtype);
    metadata.bytes = bytes;
    metadata.shape = shape;
    metadata.owner_rank = owner_rank;
    metadata.staging_id = staging_id;

    StagingBufferManager manager;
    StagingBufferHandle handle;
    if (!manager.open(metadata, false, handle, error)) {
        return false;
    }

    out.assign(bytes, '\0');
    std::memcpy(out.data(), handle.data(), bytes);
    return true;
}

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
    std::string& error) {
    error.clear();
    if (transport == BufferTransport::SocketPayload) {
        output_payload.assign(
            static_cast<const char*>(data),
            static_cast<const char*>(data) + bytes);
        return true;
    }

    StagingBufferMetadata metadata;
    metadata.name = staging_name;
    metadata.dtype = collective_data_type_name(dtype);
    metadata.bytes = bytes;
    metadata.shape = shape;
    metadata.owner_rank = owner_rank;
    metadata.staging_id = staging_id;

    StagingBufferManager manager;
    StagingBufferHandle handle;
    if (!manager.open(metadata, false, handle, error)) {
        return false;
    }

    std::memcpy(handle.data(), data, bytes);
    output_payload.clear();
    return true;
}

}  // namespace fake_gpu::distributed
