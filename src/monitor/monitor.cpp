#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <atomic>
#include <limits>
#include "../core/global_state.hpp"
#include "../core/logging.hpp"
#include "../core/backend_config.hpp"
#include "../core/hybrid_memory_manager.hpp"
#include "../distributed/communicator.hpp"

namespace fake_gpu {

// Global flag to track if we've already dumped the report
static std::atomic<bool> g_report_dumped{false};
static std::atomic<bool> g_shutting_down{false};

class ResourceMonitor {
public:
    static ResourceMonitor& instance() {
        static ResourceMonitor s_instance;
        return s_instance;
    }

    ResourceMonitor() {
        FGPU_LOG("[Monitor] Initialized\n");
    }

    ~ResourceMonitor() {
        // Try to dump report if not already done
        if (!g_report_dumped.load()) {
            dump_report_internal();
        }
        // Mark as shutting down to prevent further operations
        g_shutting_down.store(true);
    }

    void dump_report() {
        // Only dump report once, and not during shutdown
        if (g_shutting_down.load() || g_report_dumped.exchange(true)) {
            return;
        }
        dump_report_internal();
    }

private:
    static uint64_t saturating_add_u64(uint64_t a, uint64_t b) {
        const uint64_t max_value = std::numeric_limits<uint64_t>::max();
        if (a >= max_value - b) return max_value;
        return a + b;
    }

    static std::string node_name_for_rank(
        const distributed::ClusterConfigModel& config,
        int rank) {
        if (config.loaded()) {
            for (const auto& node : config.nodes) {
                for (int candidate : node.ranks) {
                    if (candidate == rank) {
                        return node.id;
                    }
                }
            }
        }
        return "node0";
    }

    void dump_cluster_report_internal(const BackendConfig& config) {
        const char* cluster_report_path = std::getenv("FAKEGPU_CLUSTER_REPORT_PATH");
        const distributed::ClusterReportSnapshot cluster_snapshot =
            distributed::snapshot_cluster_report();
        if ((!cluster_report_path || !*cluster_report_path) && !config.distributed_enabled()) {
            return;
        }
        if (!cluster_snapshot.has_data) {
            return;
        }
        if (!cluster_report_path || !*cluster_report_path) {
            cluster_report_path = "fake_gpu_cluster_report.json";
        }

        FILE* out = fopen(cluster_report_path, "w");
        if (!out) {
            FGPU_LOG("[Monitor] Failed to open cluster report file\n");
            return;
        }

        const distributed::DistributedConfig& dist_config = config.distributed_config();
        const distributed::ClusterConfigModel& cluster_config = dist_config.cluster_config;
        const std::size_t world_size = cluster_snapshot.world_size > 0
            ? cluster_snapshot.world_size
            : (cluster_config.loaded() ? cluster_config.world_size : cluster_snapshot.ranks.size());
        const std::size_t node_count = cluster_config.loaded()
            ? cluster_config.nodes.size()
            : (world_size > 0 ? 1U : 0U);

        fprintf(out, "{\n");
        fprintf(out, "  \"report_version\": 4,\n");
        fprintf(out, "  \"schema\": \"experimental\",\n");
        fprintf(out, "  \"cluster\": {\n");
        fprintf(out, "    \"mode\": \"%s\",\n", distributed::distributed_mode_name(dist_config.mode));
        fprintf(out, "    \"world_size\": %llu,\n", (unsigned long long)world_size);
        fprintf(out, "    \"node_count\": %llu,\n", (unsigned long long)node_count);
        fprintf(out, "    \"communicators\": %llu,\n",
                (unsigned long long)cluster_snapshot.communicator_count);
        fprintf(out, "    \"coordinator_transport\": \"%s\"",
                distributed::coordinator_transport_name(dist_config.coordinator_transport));
        if (cluster_config.loaded()) {
            fprintf(out, ",\n");
            fprintf(out, "    \"name\": \"%s\",\n", cluster_config.name.c_str());
            fprintf(out, "    \"default_backend\": \"%s\",\n", cluster_config.default_backend.c_str());
            fprintf(out, "    \"config_path\": \"%s\"\n", cluster_config.source_path.c_str());
        } else {
            fprintf(out, "\n");
        }
        fprintf(out, "  },\n");
        fprintf(out, "  \"collectives\": {\n");
        fprintf(out, "    \"all_reduce\": {\"calls\": %llu, \"bytes\": %llu, \"estimated_time_us_total\": %.3f, \"contention_penalty_us_total\": %.3f},\n",
                (unsigned long long)cluster_snapshot.all_reduce.calls,
                (unsigned long long)cluster_snapshot.all_reduce.bytes,
                cluster_snapshot.all_reduce.estimated_time_us_total,
                cluster_snapshot.all_reduce.contention_penalty_us_total);
        fprintf(out, "    \"reduce\": {\"calls\": %llu, \"bytes\": %llu, \"estimated_time_us_total\": %.3f, \"contention_penalty_us_total\": %.3f},\n",
                (unsigned long long)cluster_snapshot.reduce.calls,
                (unsigned long long)cluster_snapshot.reduce.bytes,
                cluster_snapshot.reduce.estimated_time_us_total,
                cluster_snapshot.reduce.contention_penalty_us_total);
        fprintf(out, "    \"broadcast\": {\"calls\": %llu, \"bytes\": %llu, \"estimated_time_us_total\": %.3f, \"contention_penalty_us_total\": %.3f},\n",
                (unsigned long long)cluster_snapshot.broadcast.calls,
                (unsigned long long)cluster_snapshot.broadcast.bytes,
                cluster_snapshot.broadcast.estimated_time_us_total,
                cluster_snapshot.broadcast.contention_penalty_us_total);
        fprintf(out, "    \"all_gather\": {\"calls\": %llu, \"bytes\": %llu, \"estimated_time_us_total\": %.3f, \"contention_penalty_us_total\": %.3f},\n",
                (unsigned long long)cluster_snapshot.all_gather.calls,
                (unsigned long long)cluster_snapshot.all_gather.bytes,
                cluster_snapshot.all_gather.estimated_time_us_total,
                cluster_snapshot.all_gather.contention_penalty_us_total);
        fprintf(out, "    \"reduce_scatter\": {\"calls\": %llu, \"bytes\": %llu, \"estimated_time_us_total\": %.3f, \"contention_penalty_us_total\": %.3f},\n",
                (unsigned long long)cluster_snapshot.reduce_scatter.calls,
                (unsigned long long)cluster_snapshot.reduce_scatter.bytes,
                cluster_snapshot.reduce_scatter.estimated_time_us_total,
                cluster_snapshot.reduce_scatter.contention_penalty_us_total);
        fprintf(out, "    \"barrier\": {\"calls\": %llu, \"bytes\": %llu, \"estimated_time_us_total\": %.3f, \"contention_penalty_us_total\": %.3f}\n",
                (unsigned long long)cluster_snapshot.barrier.calls,
                (unsigned long long)cluster_snapshot.barrier.bytes,
                cluster_snapshot.barrier.estimated_time_us_total,
                cluster_snapshot.barrier.contention_penalty_us_total);
        fprintf(out, "  },\n");
        fprintf(out, "  \"links\": [\n");
        for (std::size_t index = 0; index < cluster_snapshot.links.size(); ++index) {
            const auto& link_stats = cluster_snapshot.links[index];
            fprintf(out, "    {\n");
            fprintf(out, "      \"src\": \"%s\",\n", link_stats.src_node.c_str());
            fprintf(out, "      \"dst\": \"%s\",\n", link_stats.dst_node.c_str());
            fprintf(out, "      \"scope\": \"%s\",\n", link_stats.scope.c_str());
            fprintf(out, "      \"samples\": %llu,\n", (unsigned long long)link_stats.samples);
            fprintf(out, "      \"bytes\": %llu,\n", (unsigned long long)link_stats.bytes);
            fprintf(out, "      \"bandwidth_gbps\": %.3f,\n", link_stats.bandwidth_gbps);
            fprintf(out, "      \"avg_latency_us\": %.3f,\n", link_stats.avg_latency_us);
            fprintf(out, "      \"estimated_time_us_total\": %.3f,\n", link_stats.estimated_time_us_total);
            fprintf(out, "      \"contention_penalty_us_total\": %.3f\n", link_stats.contention_penalty_us_total);
            fprintf(out, "    }%s\n", (index + 1 < cluster_snapshot.links.size() ? "," : ""));
        }
        fprintf(out, "  ],\n");
        fprintf(out, "  \"ranks\": [\n");
        for (std::size_t index = 0; index < cluster_snapshot.ranks.size(); ++index) {
            const auto& rank_stats = cluster_snapshot.ranks[index];
            const std::string node_name = node_name_for_rank(cluster_config, rank_stats.rank);
            fprintf(out, "    {\n");
            fprintf(out, "      \"rank\": %d,\n", rank_stats.rank);
            fprintf(out, "      \"node\": \"%s\",\n", node_name.c_str());
            fprintf(out, "      \"wait_time_ms\": %.3f,\n", rank_stats.wait_time_ms);
            fprintf(out, "      \"timeouts\": %llu,\n",
                    (unsigned long long)rank_stats.timeouts);
            fprintf(out, "      \"communicator_inits\": %llu,\n",
                    (unsigned long long)rank_stats.communicator_inits);
            fprintf(out, "      \"collective_calls\": %llu,\n",
                    (unsigned long long)rank_stats.collective_calls);
            fprintf(out, "      \"barrier_calls\": %llu,\n",
                    (unsigned long long)rank_stats.barrier_calls);
            fprintf(out, "      \"group_prepares\": %llu\n",
                    (unsigned long long)rank_stats.group_prepares);
            fprintf(out, "    }%s\n", (index + 1 < cluster_snapshot.ranks.size() ? "," : ""));
        }
        fprintf(out, "  ]\n");
        fprintf(out, "}\n");
        fclose(out);
    }

    void dump_report_internal() {
        try {
            const char* report_path = std::getenv("FAKEGPU_REPORT_PATH");
            if (!report_path || !*report_path) {
                report_path = "fake_gpu_report.json";
            }
            GlobalState& gs = GlobalState::instance();
            std::vector<DeviceReportStats> devices = gs.snapshot_device_report();
            HostIoStats host_io = gs.snapshot_host_io();
            const int count = static_cast<int>(devices.size());
            FGPU_LOG("[Monitor] Dumping report to %s. GlobalState Addr: %p, Device Count: %d\n", report_path, (void*)&gs, count);

            FILE* out = fopen(report_path, "w");
            if (!out) {
                FGPU_LOG("[Monitor] Failed to open report file\n");
                return;
            }

            uint64_t total_alloc_bytes = 0;
            uint64_t total_free_bytes = 0;
            uint64_t total_h2d_bytes = 0;
            uint64_t total_d2h_bytes = 0;
            uint64_t total_d2d_bytes = 0;
            uint64_t total_peer_tx_bytes = 0;
            uint64_t total_peer_rx_bytes = 0;
            uint64_t total_memset_bytes = 0;
            uint64_t total_cublas_gemm_flops = 0;
            uint64_t total_cublaslt_matmul_flops = 0;

            uint64_t total_alloc_calls = 0;
            uint64_t total_free_calls = 0;
            uint64_t total_h2d_calls = 0;
            uint64_t total_d2h_calls = 0;
            uint64_t total_d2d_calls = 0;
            uint64_t total_peer_tx_calls = 0;
            uint64_t total_peer_rx_calls = 0;
            uint64_t total_memset_calls = 0;
            uint64_t total_cublas_gemm_calls = 0;
            uint64_t total_cublaslt_matmul_calls = 0;

            for (const auto& dev : devices) {
                total_alloc_bytes = saturating_add_u64(total_alloc_bytes, dev.alloc_bytes);
                total_free_bytes = saturating_add_u64(total_free_bytes, dev.free_bytes);
                total_h2d_bytes = saturating_add_u64(total_h2d_bytes, dev.memcpy_h2d_bytes);
                total_d2h_bytes = saturating_add_u64(total_d2h_bytes, dev.memcpy_d2h_bytes);
                total_d2d_bytes = saturating_add_u64(total_d2d_bytes, dev.memcpy_d2d_bytes);
                total_peer_tx_bytes = saturating_add_u64(total_peer_tx_bytes, dev.memcpy_peer_tx_bytes);
                total_peer_rx_bytes = saturating_add_u64(total_peer_rx_bytes, dev.memcpy_peer_rx_bytes);
                total_memset_bytes = saturating_add_u64(total_memset_bytes, dev.memset_bytes);
                total_cublas_gemm_flops = saturating_add_u64(total_cublas_gemm_flops, dev.cublas_gemm_flops);
                total_cublaslt_matmul_flops = saturating_add_u64(total_cublaslt_matmul_flops, dev.cublaslt_matmul_flops);

                total_alloc_calls = saturating_add_u64(total_alloc_calls, dev.alloc_calls);
                total_free_calls = saturating_add_u64(total_free_calls, dev.free_calls);
                total_h2d_calls = saturating_add_u64(total_h2d_calls, dev.memcpy_h2d_calls);
                total_d2h_calls = saturating_add_u64(total_d2h_calls, dev.memcpy_d2h_calls);
                total_d2d_calls = saturating_add_u64(total_d2d_calls, dev.memcpy_d2d_calls);
                total_peer_tx_calls = saturating_add_u64(total_peer_tx_calls, dev.memcpy_peer_tx_calls);
                total_peer_rx_calls = saturating_add_u64(total_peer_rx_calls, dev.memcpy_peer_rx_calls);
                total_memset_calls = saturating_add_u64(total_memset_calls, dev.memset_calls);
                total_cublas_gemm_calls = saturating_add_u64(total_cublas_gemm_calls, dev.cublas_gemm_calls);
                total_cublaslt_matmul_calls = saturating_add_u64(total_cublaslt_matmul_calls, dev.cublaslt_matmul_calls);
            }

            fprintf(out, "{\n");
            fprintf(out, "  \"report_version\": 3,\n");

            // Add mode information
            const BackendConfig& config = BackendConfig::instance();
            fprintf(out, "  \"mode\": \"%s\",\n", mode_name(config.mode()));
            if (config.mode() == FakeGpuMode::Hybrid) {
                fprintf(out, "  \"oom_policy\": \"%s\",\n", policy_name(config.oom_policy()));

                // Add hybrid memory statistics
                const auto& hybrid_stats = HybridMemoryManager::instance().get_stats();
                fprintf(out, "  \"hybrid_stats\": {\n");
                fprintf(out, "    \"real_alloc\": {\"count\": %llu, \"bytes\": %llu},\n",
                        (unsigned long long)hybrid_stats.real_alloc_count,
                        (unsigned long long)hybrid_stats.real_alloc_bytes);
                fprintf(out, "    \"managed_alloc\": {\"count\": %llu, \"bytes\": %llu},\n",
                        (unsigned long long)hybrid_stats.managed_alloc_count,
                        (unsigned long long)hybrid_stats.managed_alloc_bytes);
                fprintf(out, "    \"mapped_host_alloc\": {\"count\": %llu, \"bytes\": %llu},\n",
                        (unsigned long long)hybrid_stats.mapped_host_alloc_count,
                        (unsigned long long)hybrid_stats.mapped_host_alloc_bytes);
                fprintf(out, "    \"spilled_alloc\": {\"count\": %llu, \"bytes\": %llu},\n",
                        (unsigned long long)hybrid_stats.spilled_alloc_count,
                        (unsigned long long)hybrid_stats.spilled_alloc_bytes);
                fprintf(out, "    \"oom_fallback_count\": %llu\n",
                        (unsigned long long)hybrid_stats.oom_fallback_count);
                fprintf(out, "  },\n");

                // Add real GPU info if available
                int real_count = HybridMemoryManager::instance().get_real_device_count();
                if (real_count > 0) {
                    fprintf(out, "  \"backing_gpus\": [\n");
                    for (int i = 0; i < real_count; ++i) {
                        fprintf(out, "    {\n");
                        fprintf(out, "      \"index\": %d,\n", i);
                        fprintf(out, "      \"total_memory\": %llu,\n",
                                (unsigned long long)HybridMemoryManager::instance().get_real_total_memory(i));
                        fprintf(out, "      \"used_memory\": %llu\n",
                                (unsigned long long)HybridMemoryManager::instance().get_real_used_memory(i));
                        fprintf(out, "    }%s\n", (i < real_count - 1 ? "," : ""));
                    }
                    fprintf(out, "  ],\n");
                }
            }

            fprintf(out, "  \"host_io\": {\n");
            fprintf(out, "    \"memcpy_calls\": %llu,\n", (unsigned long long)host_io.memcpy_calls);
            fprintf(out, "    \"memcpy_bytes\": %llu\n", (unsigned long long)host_io.memcpy_bytes);
            fprintf(out, "  },\n");
            fprintf(out, "  \"summary\": {\n");
            fprintf(out, "    \"device_count\": %d,\n", count);
            fprintf(out, "    \"alloc\": {\"calls\": %llu, \"bytes\": %llu},\n",
                    (unsigned long long)total_alloc_calls, (unsigned long long)total_alloc_bytes);
            fprintf(out, "    \"free\": {\"calls\": %llu, \"bytes\": %llu},\n",
                    (unsigned long long)total_free_calls, (unsigned long long)total_free_bytes);
            fprintf(out, "    \"io\": {\n");
            fprintf(out, "      \"h2d\": {\"calls\": %llu, \"bytes\": %llu},\n",
                    (unsigned long long)total_h2d_calls, (unsigned long long)total_h2d_bytes);
            fprintf(out, "      \"d2h\": {\"calls\": %llu, \"bytes\": %llu},\n",
                    (unsigned long long)total_d2h_calls, (unsigned long long)total_d2h_bytes);
            fprintf(out, "      \"d2d\": {\"calls\": %llu, \"bytes\": %llu},\n",
                    (unsigned long long)total_d2d_calls, (unsigned long long)total_d2d_bytes);
            fprintf(out, "      \"peer_tx\": {\"calls\": %llu, \"bytes\": %llu},\n",
                    (unsigned long long)total_peer_tx_calls, (unsigned long long)total_peer_tx_bytes);
            fprintf(out, "      \"peer_rx\": {\"calls\": %llu, \"bytes\": %llu},\n",
                    (unsigned long long)total_peer_rx_calls, (unsigned long long)total_peer_rx_bytes);
            fprintf(out, "      \"memset\": {\"calls\": %llu, \"bytes\": %llu}\n",
                    (unsigned long long)total_memset_calls, (unsigned long long)total_memset_bytes);
            fprintf(out, "    },\n");
            fprintf(out, "    \"compute\": {\n");
            fprintf(out, "      \"cublas_gemm\": {\"calls\": %llu, \"flops\": %llu},\n",
                    (unsigned long long)total_cublas_gemm_calls, (unsigned long long)total_cublas_gemm_flops);
            fprintf(out, "      \"cublaslt_matmul\": {\"calls\": %llu, \"flops\": %llu}\n",
                    (unsigned long long)total_cublaslt_matmul_calls, (unsigned long long)total_cublaslt_matmul_flops);
            fprintf(out, "    }\n");
            fprintf(out, "  },\n");
            fprintf(out, "  \"devices\": [\n");

            for (int i = 0; i < count; ++i) {
                const DeviceReportStats& dev = devices[static_cast<size_t>(i)];
                const uint64_t device_total_io_bytes = saturating_add_u64(
                    saturating_add_u64(saturating_add_u64(dev.memcpy_h2d_bytes, dev.memcpy_d2h_bytes),
                                       saturating_add_u64(dev.memcpy_d2d_bytes, dev.memcpy_peer_tx_bytes)),
                    saturating_add_u64(saturating_add_u64(dev.memcpy_peer_rx_bytes, dev.memset_bytes), 0));
                const uint64_t device_total_flops = saturating_add_u64(dev.cublas_gemm_flops, dev.cublaslt_matmul_flops);

                fprintf(out, "    {\n");
                fprintf(out, "      \"index\": %d,\n", dev.index);
                fprintf(out, "      \"name\": \"%s\",\n", dev.name.c_str());
                fprintf(out, "      \"uuid\": \"%s\",\n", dev.uuid.c_str());
                fprintf(out, "      \"total_memory\": %llu,\n", (unsigned long long)dev.total_memory);
                fprintf(out, "      \"used_memory_peak\": %llu,\n", (unsigned long long)dev.used_memory_peak);
                fprintf(out, "      \"used_memory_current\": %llu,\n", (unsigned long long)dev.used_memory_current);

                fprintf(out, "      \"alloc\": {\"calls\": %llu, \"bytes\": %llu},\n",
                        (unsigned long long)dev.alloc_calls, (unsigned long long)dev.alloc_bytes);
                fprintf(out, "      \"free\": {\"calls\": %llu, \"bytes\": %llu},\n",
                        (unsigned long long)dev.free_calls, (unsigned long long)dev.free_bytes);

                fprintf(out, "      \"io\": {\n");
                fprintf(out, "        \"h2d\": {\"calls\": %llu, \"bytes\": %llu},\n",
                        (unsigned long long)dev.memcpy_h2d_calls, (unsigned long long)dev.memcpy_h2d_bytes);
                fprintf(out, "        \"d2h\": {\"calls\": %llu, \"bytes\": %llu},\n",
                        (unsigned long long)dev.memcpy_d2h_calls, (unsigned long long)dev.memcpy_d2h_bytes);
                fprintf(out, "        \"d2d\": {\"calls\": %llu, \"bytes\": %llu},\n",
                        (unsigned long long)dev.memcpy_d2d_calls, (unsigned long long)dev.memcpy_d2d_bytes);
                fprintf(out, "        \"peer_tx\": {\"calls\": %llu, \"bytes\": %llu},\n",
                        (unsigned long long)dev.memcpy_peer_tx_calls, (unsigned long long)dev.memcpy_peer_tx_bytes);
                fprintf(out, "        \"peer_rx\": {\"calls\": %llu, \"bytes\": %llu},\n",
                        (unsigned long long)dev.memcpy_peer_rx_calls, (unsigned long long)dev.memcpy_peer_rx_bytes);
                fprintf(out, "        \"memset\": {\"calls\": %llu, \"bytes\": %llu},\n",
                        (unsigned long long)dev.memset_calls, (unsigned long long)dev.memset_bytes);
                fprintf(out, "        \"total_bytes\": %llu\n", (unsigned long long)device_total_io_bytes);
                fprintf(out, "      },\n");

                fprintf(out, "      \"compute\": {\n");
                fprintf(out, "        \"cublas_gemm\": {\"calls\": %llu, \"flops\": %llu},\n",
                        (unsigned long long)dev.cublas_gemm_calls, (unsigned long long)dev.cublas_gemm_flops);
                fprintf(out, "        \"cublaslt_matmul\": {\"calls\": %llu, \"flops\": %llu},\n",
                        (unsigned long long)dev.cublaslt_matmul_calls, (unsigned long long)dev.cublaslt_matmul_flops);
                fprintf(out, "        \"total_flops\": %llu\n", (unsigned long long)device_total_flops);
                fprintf(out, "      }\n");

                fprintf(out, "    }%s\n", (i < count - 1 ? "," : ""));
            }

            fprintf(out, "  ]\n");
            fprintf(out, "}\n");
            fclose(out);
            dump_cluster_report_internal(config);
            g_report_dumped.store(true);
        } catch (...) {
            FGPU_LOG("[Monitor] Exception during report dump\n");
        }
    }
};

// Static instance to force construction/destruction
static ResourceMonitor& s_monitor = ResourceMonitor::instance();

// Public function to dump report
void dump_monitor_report() {
    ResourceMonitor::instance().dump_report();
}

} // namespace fake_gpu
