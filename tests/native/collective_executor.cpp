#include "collective_executor.hpp"
#include "staging_buffer.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

using namespace fake_gpu::distributed;
using Values = std::vector<float>;
using Executor = CollectiveExecutionResult (*)(
    const CollectiveExecutionRequest&,
    const std::vector<CollectiveExecutionParticipant>&);

namespace {

void require(bool condition, const std::string& detail) {
    if (!condition) throw std::runtime_error(detail);
}

void check_collective(
    Executor execute,
    const CollectiveExecutionRequest& request,
    const std::vector<Values>& inputs,
    const std::vector<Values>& expected,
    BufferTransport transport) {
    static std::uint64_t sequence = 0;
    StagingBufferManager manager;
    std::vector<StagingBufferHandle> handles(inputs.size());
    std::vector<CollectiveExecutionParticipant> participants;
    std::string error;
    for (std::size_t rank = 0; rank < inputs.size(); ++rank) {
        CollectiveExecutionParticipant participant;
        participant.rank = static_cast<int>(rank);
        participant.transport = transport;
        participant.bytes = request.bytes;
        participant.payload_bytes = inputs[rank].size() * sizeof(float);
        participant.payload.resize(participant.payload_bytes);
        std::memcpy(participant.payload.data(), inputs[rank].data(), participant.payload_bytes);
        if (transport == BufferTransport::SharedMemory) {
            StagingBufferMetadata metadata;
            metadata.name = "/fgpu-test-" + std::to_string(getpid()) + "-" + std::to_string(++sequence);
            metadata.dtype = "float32";
            metadata.bytes = request.bytes;
            metadata.owner_rank = participant.rank;
            metadata.staging_id = sequence;
            require(manager.create(metadata, true, handles[rank], error), error);
            std::memset(handles[rank].data(), 0, request.bytes);
            std::memcpy(handles[rank].data(), participant.payload.data(), participant.payload_bytes);
            participant.staging_name = metadata.name;
        }
        participants.push_back(std::move(participant));
    }

    const auto result = execute(request, participants);
    require(result.ok, result.error_code + ": " + result.error_detail);
    for (std::size_t rank = 0; rank < expected.size(); ++rank) {
        Values actual(expected[rank].size());
        const std::size_t bytes = actual.size() * sizeof(float);
        if (transport == BufferTransport::SocketPayload) {
            const auto& payload = result.output_payloads.at(static_cast<int>(rank));
            require(payload.size() == bytes, "incorrect output size");
            if (bytes) std::memcpy(actual.data(), payload.data(), bytes);
        } else if (bytes) {
            std::memcpy(actual.data(), handles[rank].data(), bytes);
        } else {
            // Reduce only writes the root's buffer.
            Values untouched(inputs[rank].size());
            std::memcpy(untouched.data(), handles[rank].data(), untouched.size() * sizeof(float));
            require(untouched == inputs[rank], "non-root reduce buffer changed");
        }
        require(actual == expected[rank], "incorrect collective output on rank " + std::to_string(rank));
    }

    // Verify each executor preserves its diagnostic on inconsistent byte counts.
    auto invalid = request;
    invalid.bytes += sizeof(float);
    const auto failed = execute(invalid, participants);
    require(!failed.ok && failed.error_code == "invalid_collective_size", "missing size error");
    require(!failed.error_detail.empty() && failed.output_payloads.empty(), "invalid error result");
}

void check_transport(BufferTransport transport) {
    CollectiveExecutionRequest request;
    request.dtype = CollectiveDataType::Float32;
    request.count = 2;
    request.bytes = 2 * sizeof(float);
    request.root_rank = 1;
    const std::vector<Values> inputs = {{2, 4}, {4, 8}, {6, 12}};
    const std::vector<std::pair<CollectiveReduceOp, Values>> reductions = {
        {CollectiveReduceOp::Sum, {12, 24}},
        {CollectiveReduceOp::Prod, {48, 384}},
        {CollectiveReduceOp::Min, {2, 4}},
        {CollectiveReduceOp::Max, {6, 12}},
        {CollectiveReduceOp::Avg, {4, 8}},
    };
    for (const auto& [operation, values] : reductions) {
        request.reduce_op = operation;
        request.type = CollectiveType::AllReduce;
        check_collective(execute_allreduce_sum, request, inputs, {values, values, values}, transport);
        request.type = CollectiveType::Reduce;
        check_collective(execute_reduce, request, inputs, {{}, values, {}}, transport);
    }
    request.type = CollectiveType::Broadcast;
    check_collective(execute_broadcast, request, inputs, {inputs[1], inputs[1], inputs[1]}, transport);

    request.bytes = 6 * sizeof(float);
    request.type = CollectiveType::AllGather;
    const Values gathered = {2, 4, 4, 8, 6, 12};
    check_collective(execute_allgather, request, inputs, {gathered, gathered, gathered}, transport);

    request.type = CollectiveType::ReduceScatter;
    request.reduce_op = CollectiveReduceOp::Sum;
    check_collective(execute_reducescatter, request,
        {{1, 2, 3, 4, 5, 6}, {10, 20, 30, 40, 50, 60}, {100, 200, 300, 400, 500, 600}},
        {{111, 222}, {333, 444}, {555, 666}}, transport);

    request.type = CollectiveType::AllToAll;
    check_collective(execute_alltoall, request,
        {{0, 1, 2, 3, 4, 5}, {10, 11, 12, 13, 14, 15}, {20, 21, 22, 23, 24, 25}},
        {{0, 1, 10, 11, 20, 21}, {2, 3, 12, 13, 22, 23}, {4, 5, 14, 15, 24, 25}}, transport);
}

}  // namespace

int main() {
    try {
        check_transport(BufferTransport::SocketPayload);
        check_transport(BufferTransport::SharedMemory);
        std::cout << "PASS: 28 collective cases and 28 size errors across socket/shared memory\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
