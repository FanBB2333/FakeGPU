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

std::vector<char> words(const std::vector<std::uint16_t>& values) {
    std::vector<char> bytes(values.size() * sizeof(std::uint16_t));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

// All-reduces raw element bytes over socket payloads and returns the result,
// which every rank must receive unchanged.
std::vector<char> allreduce_bytes(
    CollectiveDataType dtype,
    CollectiveReduceOp operation,
    const std::vector<std::vector<char>>& inputs) {
    CollectiveExecutionRequest request;
    request.type = CollectiveType::AllReduce;
    request.dtype = dtype;
    request.reduce_op = operation;
    request.bytes = inputs.front().size();
    request.count = request.bytes / collective_data_type_size(dtype);
    std::vector<CollectiveExecutionParticipant> participants;
    for (std::size_t rank = 0; rank < inputs.size(); ++rank) {
        CollectiveExecutionParticipant participant;
        participant.rank = static_cast<int>(rank);
        participant.transport = BufferTransport::SocketPayload;
        participant.bytes = request.bytes;
        participant.payload_bytes = inputs[rank].size();
        participant.payload = inputs[rank];
        participants.push_back(std::move(participant));
    }
    const auto result = execute_allreduce_sum(request, participants);
    require(result.ok, result.error_code + ": " + result.error_detail);
    const std::vector<char>& first = result.output_payloads.at(0);
    for (const auto& entry : result.output_payloads) {
        require(entry.second == first, "ranks received different all-reduce results");
    }
    return first;
}

// NCCL computes floating-point ncclAvg as PreMulSum by 1/n in the element type
// (enqueue.cc, hostToDevRedOp; the premultiply in the element type was
// checked against NCCL 2.27.5 on an Ampere GPU). With two ranks the factor is
// 0.5, so halving the smallest subnormal rounds to zero before the sum, where
// summing first and then dividing would keep it.
void check_floating_average_premultiplies() {
    require(
        allreduce_bytes(CollectiveDataType::Float16, CollectiveReduceOp::Avg,
                        {words({0x0001, 0x3c00}), words({0x0001, 0x4000})}) ==
            words({0x0000, 0x3e00}),
        "float16 ncclAvg did not premultiply by 1/n");
    require(
        allreduce_bytes(CollectiveDataType::BFloat16, CollectiveReduceOp::Avg,
                        {words({0x0001, 0x3f80}), words({0x0001, 0x4000})}) ==
            words({0x0000, 0x3fc0}),
        "bfloat16 ncclAvg did not premultiply by 1/n");
    // Integer ncclAvg stays SumPostDiv: wrap-around sum, then truncation.
    const std::vector<char> int8_avg = allreduce_bytes(
        CollectiveDataType::Int8, CollectiveReduceOp::Avg,
        {{static_cast<char>(100), static_cast<char>(-3)},
         {static_cast<char>(100), static_cast<char>(0)}});
    require(int8_avg == std::vector<char>{static_cast<char>(-28), static_cast<char>(-1)},
            "int8 ncclAvg is not a wrapped sum followed by division");
}

std::vector<char> fp8(const std::vector<unsigned char>& values) {
    return std::vector<char>(values.begin(), values.end());
}

// NCCL's FP8 reductions compute in float16 and store FP8 after every step
// (reduce_kernel.h), saturating to the largest finite value. ncclAvg is
// PreMulSum by the FP8 value of 1/n (enqueue.cc, hostToDevRedOp).
void check_fp8_reductions() {
    using Op = CollectiveReduceOp;
    const auto e4m3 = CollectiveDataType::Float8E4M3;
    const auto e5m2 = CollectiveDataType::Float8E5M2;
    // 1 + 2 + 4 = 7. 16 + 1 rounds to 16 (ties to even at a spacing of 2),
    // and so does adding the second 1, where a single rounding of 18 would not.
    require(allreduce_bytes(e4m3, Op::Sum, {fp8({0x38, 0x58}), fp8({0x40, 0x38}), fp8({0x48, 0x38})}) ==
                fp8({0x4e, 0x58}),
            "float8_e4m3 sum");
    // 448 + 448 saturates to 448 and -2 * 2 * 1 = -4.
    require(allreduce_bytes(e4m3, Op::Sum, {fp8({0x7e}), fp8({0x7e})}) == fp8({0x7e}),
            "float8_e4m3 saturation");
    require(allreduce_bytes(e4m3, Op::Prod, {fp8({0xc0}), fp8({0x40}), fp8({0x38})}) == fp8({0xc8}),
            "float8_e4m3 product");
    require(allreduce_bytes(e4m3, Op::Max, {fp8({0x38}), fp8({0xc0}), fp8({0x48})}) == fp8({0x48}),
            "float8_e4m3 max");
    // (1 + 3) / 2 premultiplied by 0.5: 0.5 + 1.5 = 2.
    require(allreduce_bytes(e4m3, Op::Avg, {fp8({0x38}), fp8({0x44})}) == fp8({0x40}),
            "float8_e4m3 average");
    // e5m2: 1 + 2 = 3; 57344 + 57344 overflows float16 and saturates to 57344.
    require(allreduce_bytes(e5m2, Op::Sum, {fp8({0x3c, 0x7b}), fp8({0x40, 0x7b})}) == fp8({0x42, 0x7b}),
            "float8_e5m2 sum");
    require(allreduce_bytes(e5m2, Op::Avg, {fp8({0x3c}), fp8({0x40})}) == fp8({0x3e}),
            "float8_e5m2 average");
    require(collective_data_type_size(e4m3) == 1 && collective_data_type_size(e5m2) == 1,
            "FP8 element size");
    CollectiveDataType parsed = CollectiveDataType::Float32;
    require(parse_collective_data_type("float8_e4m3", parsed) && parsed == e4m3 &&
                parse_collective_data_type("float8_e5m2", parsed) && parsed == e5m2,
            "FP8 dtype names");
}

}  // namespace

int main() {
    try {
        check_transport(BufferTransport::SocketPayload);
        check_transport(BufferTransport::SharedMemory);
        check_floating_average_premultiplies();
        check_fp8_reductions();
        std::cout << "PASS: 28 collective cases and 28 size errors across socket/shared memory\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
