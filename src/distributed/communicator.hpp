#pragma once

#include "collective_executor.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fake_gpu::distributed {

struct CommunicatorRegistrationResult {
    bool ok = false;
    int comm_id = -1;
    std::uint64_t seqno = 0;
    std::string error_code;
    std::string error_detail;
};

struct CommunicatorDestroyResult {
    bool ok = false;
    std::string error_code;
    std::string error_detail;
};

struct CommunicatorSplitRequest {
    int comm_id = -1;
    int rank = -1;
    std::uint64_t seqno = 0;
    int color = -1;
    int key = 0;
    int timeout_ms = 0;
};

struct CommunicatorSplitResult {
    bool ok = false;
    std::uint64_t seqno = 0;
    bool participating = false;
    int new_comm_id = -1;
    int new_rank = -1;
    int new_world_size = 0;
    std::string error_code;
    std::string error_detail;
};

enum class PointToPointType {
    Send,
    Recv,
};

struct PointToPointSubmitRequest {
    int comm_id = -1;
    int rank = -1;
    std::uint64_t seqno = 0;
    PointToPointType type = PointToPointType::Send;
    int peer = -1;
    CollectiveDataType dtype = CollectiveDataType::Float32;
    std::size_t count = 0;
    BufferTransport transport = BufferTransport::SharedMemory;
    std::string staging_name;
    std::size_t bytes = 0;
    std::size_t payload_bytes = 0;
    std::vector<char> payload;
    int timeout_ms = 0;
};

struct PointToPointSubmitResult {
    bool ok = false;
    std::uint64_t seqno = 0;
    std::string error_code;
    std::string error_detail;
    std::vector<char> output_payload;
};

struct CollectiveSubmitRequest {
    int comm_id = -1;
    int rank = -1;
    std::uint64_t seqno = 0;
    CollectiveType type = CollectiveType::AllReduce;
    CollectiveDataType dtype = CollectiveDataType::Float32;
    std::size_t count = 0;
    int root = -1;
    CollectiveReduceOp reduce_op = CollectiveReduceOp::None;
    BufferTransport transport = BufferTransport::SharedMemory;
    std::string staging_name;
    std::size_t bytes = 0;
    std::size_t payload_bytes = 0;
    std::vector<char> payload;
    int timeout_ms = 0;
    bool proxy_only = false;
};

struct CollectiveSubmitResult {
    bool ok = false;
    std::uint64_t seqno = 0;
    std::string error_code;
    std::string error_detail;
    std::vector<char> output_payload;
};

struct BarrierSubmitRequest {
    int comm_id = -1;
    int rank = -1;
    std::uint64_t seqno = 0;
    int timeout_ms = 0;
};

struct BarrierSubmitResult {
    bool ok = false;
    std::uint64_t seqno = 0;
    std::string error_code;
    std::string error_detail;
};

struct CollectiveBatchPlanItem {
    CollectiveType type = CollectiveType::AllReduce;
    CollectiveDataType dtype = CollectiveDataType::Float32;
    std::size_t count = 0;
    int root = -1;
    CollectiveReduceOp reduce_op = CollectiveReduceOp::None;
    std::size_t bytes = 0;
};

struct CollectiveBatchPrepareRequest {
    int comm_id = -1;
    int rank = -1;
    std::uint64_t base_seqno = 0;
    int timeout_ms = 0;
    std::vector<CollectiveBatchPlanItem> operations;
};

struct CollectiveBatchPrepareResult {
    bool ok = false;
    std::uint64_t base_seqno = 0;
    std::string error_code;
    std::string error_detail;
};

struct ClusterCollectiveReportStats {
    std::uint64_t calls = 0;
    std::uint64_t bytes = 0;
    double estimated_time_us_total = 0.0;
    double contention_penalty_us_total = 0.0;
};

struct ClusterPointToPointReportStats {
    std::uint64_t operations = 0;
    std::uint64_t sends = 0;
    std::uint64_t bytes = 0;
    double estimated_time_us_total = 0.0;
    double contention_penalty_us_total = 0.0;
};

struct ClusterLinkReportStats {
    std::string src_node;
    std::string dst_node;
    std::string scope;
    std::uint64_t samples = 0;
    std::uint64_t operations = 0;
    std::uint64_t collective_operations = 0;
    std::uint64_t point_to_point_operations = 0;
    std::uint64_t bytes = 0;
    std::uint64_t peak_bytes_per_operation = 0;
    double bandwidth_gbps = 0.0;
    double avg_latency_us = 0.0;
    double estimated_time_us_total = 0.0;
    double contention_penalty_us_total = 0.0;
    double peak_estimated_throughput_gbps = 0.0;
};

struct ClusterNodePairDirectionReportStats {
    std::uint64_t transfers = 0;
    std::uint64_t bytes = 0;
    std::uint64_t peak_bytes_per_operation = 0;
    double model_bandwidth_gbps = 0.0;
    double avg_latency_us = 0.0;
    double estimated_time_us_total = 0.0;
    double contention_penalty_us_total = 0.0;
    double peak_estimated_throughput_gbps = 0.0;
};

struct ClusterNodePairReportStats {
    std::string node_a;
    std::string node_b;
    std::uint64_t operations = 0;
    std::uint64_t collective_operations = 0;
    std::uint64_t point_to_point_operations = 0;
    std::uint64_t peak_combined_bytes_per_operation = 0;
    double peak_estimated_throughput_gbps = 0.0;
    ClusterNodePairDirectionReportStats a_to_b;
    ClusterNodePairDirectionReportStats b_to_a;
};

struct ClusterRankReportStats {
    int rank = -1;
    double wait_time_ms = 0.0;
    std::uint64_t timeouts = 0;
    std::uint64_t communicator_inits = 0;
    std::uint64_t collective_calls = 0;
    std::uint64_t point_to_point_calls = 0;
    std::uint64_t barrier_calls = 0;
    std::uint64_t group_prepares = 0;
};

struct ClusterOperationTimelineEntry {
    std::uint64_t index = 0;
    int comm_id = -1;
    std::uint64_t seqno = 0;
    std::string kind;
    std::string operation;
    std::string buffer_transport;
    std::vector<int> ranks;
    std::uint64_t logical_payload_bytes = 0;
    std::uint64_t socket_request_payload_bytes = 0;
    std::uint64_t socket_response_payload_bytes = 0;
    double rendezvous_wait_us = 0.0;
    double execution_time_us = 0.0;
    double coordinator_duration_us = 0.0;
    double modeled_time_us = 0.0;
};

struct ClusterReportSnapshot {
    bool has_data = false;
    std::size_t world_size = 0;
    std::size_t communicator_count = 0;
    ClusterCollectiveReportStats all_reduce;
    ClusterCollectiveReportStats reduce;
    ClusterCollectiveReportStats broadcast;
    ClusterCollectiveReportStats all_gather;
    ClusterCollectiveReportStats reduce_scatter;
    ClusterCollectiveReportStats all_to_all;
    ClusterCollectiveReportStats barrier;
    ClusterPointToPointReportStats point_to_point;
    std::vector<ClusterLinkReportStats> links;
    std::vector<ClusterNodePairReportStats> node_pairs;
    std::vector<ClusterRankReportStats> ranks;
    std::vector<ClusterOperationTimelineEntry> operation_timeline;
    std::uint64_t dropped_operation_timeline_entries = 0;
};

class CommunicatorRegistry {
public:
    CommunicatorRegistrationResult init_communicator(
        const std::string& unique_id,
        int world_size,
        int rank,
        int timeout_ms);

    CommunicatorDestroyResult destroy_communicator(int comm_id, int rank);
    CommunicatorSplitResult split_communicator(const CommunicatorSplitRequest& request);
    PointToPointSubmitResult submit_point_to_point(const PointToPointSubmitRequest& request);
    CollectiveSubmitResult submit_collective(const CollectiveSubmitRequest& request);
    BarrierSubmitResult submit_barrier(const BarrierSubmitRequest& request);
    CollectiveBatchPrepareResult prepare_collective_batch(const CollectiveBatchPrepareRequest& request);
};

ClusterReportSnapshot snapshot_cluster_report();

}  // namespace fake_gpu::distributed
