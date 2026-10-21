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

inline std::string participant_size_mismatch(
    const CollectiveExecutionParticipant& participant,
    const CollectiveExecutionRequest& request) {
    return "participant rank " + std::to_string(participant.rank) +
           " reported bytes=" + std::to_string(participant.bytes) +
           ", expected " + std::to_string(request.bytes);
}

inline bool load_participant_input(
    const CollectiveExecutionRequest& request,
    const CollectiveExecutionParticipant& participant,
    const std::vector<std::size_t>& shape,
    std::vector<char>& out,
    std::string& error) {
    return load_participant_buffer(
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
        out,
        error);
}

inline bool store_participant_output(
    const CollectiveExecutionRequest& request,
    const CollectiveExecutionParticipant& participant,
    const std::vector<std::size_t>& shape,
    const void* data,
    std::size_t bytes,
    CollectiveExecutionResult& result,
    std::string& error) {
    return store_participant_buffer(
        participant.transport,
        participant.staging_name,
        request.dtype,
        shape,
        participant.rank,
        request.seqno,
        data,
        bytes,
        result.output_payloads[participant.rank],
        error);
}

}  // namespace fake_gpu::distributed
