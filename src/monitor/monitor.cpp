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
