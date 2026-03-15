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

struct CollectiveSubmitRequest {
    int comm_id = -1;
    int rank = -1;
    std::uint64_t seqno = 0;
    CollectiveType type = CollectiveType::AllReduce;
    CollectiveDataType dtype = CollectiveDataType::Float32;
    std::size_t count = 0;
    int root = -1;
    CollectiveReduceOp reduce_op = CollectiveReduceOp::None;
    std::string staging_name;
    std::size_t bytes = 0;
    int timeout_ms = 0;
};

struct CollectiveSubmitResult {
    bool ok = false;
    std::uint64_t seqno = 0;
    std::string error_code;
    std::string error_detail;
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

struct ClusterLinkReportStats {
    std::string src_node;
    std::string dst_node;
    std::string scope;
    std::uint64_t samples = 0;
    std::uint64_t bytes = 0;
    double bandwidth_gbps = 0.0;
    double avg_latency_us = 0.0;
    double estimated_time_us_total = 0.0;
    double contention_penalty_us_total = 0.0;
};

struct ClusterRankReportStats {
    int rank = -1;
    double wait_time_ms = 0.0;
    std::uint64_t timeouts = 0;
    std::uint64_t communicator_inits = 0;
    std::uint64_t collective_calls = 0;
    std::uint64_t barrier_calls = 0;
    std::uint64_t group_prepares = 0;
};

struct ClusterReportSnapshot {
    bool has_data = false;
    std::size_t world_size = 0;
    std::size_t communicator_count = 0;
    ClusterCollectiveReportStats all_reduce;
    ClusterCollectiveReportStats broadcast;
    ClusterCollectiveReportStats all_gather;
    ClusterCollectiveReportStats reduce_scatter;
    ClusterCollectiveReportStats barrier;
    std::vector<ClusterLinkReportStats> links;
    std::vector<ClusterRankReportStats> ranks;
};

class CommunicatorRegistry {
public:
    CommunicatorRegistrationResult init_communicator(
        const std::string& unique_id,
        int world_size,
        int rank,
        int timeout_ms);

    CommunicatorDestroyResult destroy_communicator(int comm_id, int rank);
    CollectiveSubmitResult submit_collective(const CollectiveSubmitRequest& request);
    BarrierSubmitResult submit_barrier(const BarrierSubmitRequest& request);
    CollectiveBatchPrepareResult prepare_collective_batch(const CollectiveBatchPrepareRequest& request);
};

ClusterReportSnapshot snapshot_cluster_report();

}  // namespace fake_gpu::distributed
