#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <atomic>
#include <limits>
#include "../core/version.hpp"
#include "../core/global_state.hpp"
#include "../core/logging.hpp"
#include "../core/backend_config.hpp"
#include "../core/hybrid_memory_manager.hpp"
#include "../distributed/cluster_report_writer.hpp"
#include "../distributed/communicator.hpp"

namespace fake_gpu {

// Global flag to track if we've already dumped the report
static std::atomic<bool> g_report_dumped{false};
static std::atomic<bool> g_shutting_down{false};

namespace {

uint64_t saturating_add_u64(uint64_t a, uint64_t b) {
    const uint64_t max_value = std::numeric_limits<uint64_t>::max();
    if (a >= max_value - b) return max_value;
    return a + b;
}

// One report row: a call counter and its byte or FLOP total.
struct CounterField {
    const char* name;
    uint64_t DeviceCounters::*calls;
    uint64_t DeviceCounters::*amount;
};

constexpr CounterField kMemoryCounters[] = {
    {"alloc", &DeviceCounters::alloc_calls, &DeviceCounters::alloc_bytes},
    {"free", &DeviceCounters::free_calls, &DeviceCounters::free_bytes},
};

constexpr CounterField kIoCounters[] = {
    {"h2d", &DeviceCounters::memcpy_h2d_calls, &DeviceCounters::memcpy_h2d_bytes},
    {"d2h", &DeviceCounters::memcpy_d2h_calls, &DeviceCounters::memcpy_d2h_bytes},
    {"d2d", &DeviceCounters::memcpy_d2d_calls, &DeviceCounters::memcpy_d2d_bytes},
    {"peer_tx", &DeviceCounters::memcpy_peer_tx_calls, &DeviceCounters::memcpy_peer_tx_bytes},
    {"peer_rx", &DeviceCounters::memcpy_peer_rx_calls, &DeviceCounters::memcpy_peer_rx_bytes},
    {"memset", &DeviceCounters::memset_calls, &DeviceCounters::memset_bytes},
};

constexpr CounterField kComputeCounters[] = {
    {"cublas_gemm", &DeviceCounters::cublas_gemm_calls, &DeviceCounters::cublas_gemm_flops},
    {"cublaslt_matmul", &DeviceCounters::cublaslt_matmul_calls, &DeviceCounters::cublaslt_matmul_flops},
};

template <std::size_t N>
bool any_counter(const CounterField (&fields)[N], const DeviceCounters& stats) {
    for (const CounterField& field : fields) {
        if (stats.*field.calls > 0 || stats.*field.amount > 0) return true;
    }
    return false;
}

template <std::size_t N>
uint64_t total_amount(const CounterField (&fields)[N], const DeviceCounters& stats) {
    uint64_t total = 0;
    for (const CounterField& field : fields) {
        total = saturating_add_u64(total, stats.*field.amount);
    }
    return total;
}

template <std::size_t N>
void add_counters(const CounterField (&fields)[N], const DeviceCounters& stats, DeviceCounters& totals) {
    for (const CounterField& field : fields) {
        totals.*field.calls = saturating_add_u64(totals.*field.calls, stats.*field.calls);
        totals.*field.amount = saturating_add_u64(totals.*field.amount, stats.*field.amount);
    }
}

template <std::size_t N>
void print_counter_rows(
    FILE* out,
    const char* indent,
    const CounterField (&fields)[N],
    const DeviceCounters& stats,
    const char* unit,
    bool comma_after_last) {
    for (std::size_t i = 0; i < N; ++i) {
        fprintf(out, "%s\"%s\": {\"calls\": %llu, \"%s\": %llu}%s\n",
                indent,
                fields[i].name,
                (unsigned long long)(stats.*fields[i].calls),
                unit,
                (unsigned long long)(stats.*fields[i].amount),
                (i + 1 < N || comma_after_last) ? "," : "");
    }
}

} // namespace

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
    static bool snapshot_has_meaningful_activity(
        const std::vector<DeviceReportStats>& devices,
        const HostIoStats& host_io) {
        if (host_io.memcpy_calls > 0 || host_io.memcpy_bytes > 0) {
            return true;
        }

        for (const auto& dev : devices) {
            if (dev.used_memory_current > 0 || dev.used_memory_peak > 0) return true;
            if (any_counter(kMemoryCounters, dev) ||
                any_counter(kIoCounters, dev) ||
                any_counter(kComputeCounters, dev)) {
                return true;
            }
            if (dev.kernel_launch_total > 0) return true;
            if (!dev.compat_events.empty()) return true;
            if (!dev.unsupported_api_events.empty()) return true;
        }

        return false;
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

        const distributed::DistributedConfig& dist_config = config.distributed_config();
        std::string error;
        if (!distributed::write_cluster_report_files(
                dist_config,
                cluster_snapshot,
                cluster_report_path,
                error)) {
            FGPU_LOG("[Monitor] Failed to write cluster report: %s\n", error.c_str());
        }
    }

    void dump_terminal_summary(const std::vector<DeviceReportStats>& devices,
                               const HostIoStats& host_io) {
        const char* env = std::getenv("FAKEGPU_TERMINAL_REPORT");
        if (env && std::string(env) == "0") {
            return;
        }

        auto fmt_seconds = [](double seconds) -> std::string {
            char buf[32];
            if (seconds < 1e-3) {
                std::snprintf(buf, sizeof(buf), "%.1f us", seconds * 1e6);
            } else if (seconds < 1.0) {
                std::snprintf(buf, sizeof(buf), "%.1f ms", seconds * 1e3);
            } else {
                std::snprintf(buf, sizeof(buf), "%.2f s", seconds);
            }
            return buf;
        };

        auto fmt_bytes = [](uint64_t bytes) -> std::string {
            char buf[32];
            if (bytes >= 1024ULL * 1024 * 1024) {
                std::snprintf(buf, sizeof(buf), "%.1f GB", bytes / (1024.0 * 1024.0 * 1024.0));
            } else if (bytes >= 1024ULL * 1024) {
                std::snprintf(buf, sizeof(buf), "%.1f MB", bytes / (1024.0 * 1024.0));
            } else {
                std::snprintf(buf, sizeof(buf), "%llu B", (unsigned long long)bytes);
            }
            return buf;
        };

        std::fprintf(stderr, "\n");
        std::fprintf(stderr, "======================================================\n");
        std::fprintf(stderr, "             FakeGPU Report Summary\n");
        std::fprintf(stderr, "======================================================\n");

        for (const auto& dev : devices) {
            const double peak_pct = dev.total_memory > 0
                ? (100.0 * static_cast<double>(dev.used_memory_peak) / static_cast<double>(dev.total_memory))
                : 0.0;

            std::fprintf(stderr, " Device %d: %s", dev.index, dev.name.c_str());
            if (!dev.architecture.empty()) {
                std::fprintf(stderr, " (%s, cc %d.%d)",
                             dev.architecture.c_str(), dev.compute_major, dev.compute_minor);
            }
            std::fprintf(stderr, "\n");

            std::fprintf(stderr, "   Memory: %s / %s peak (%.1f%%)\n",
                         fmt_bytes(dev.used_memory_peak).c_str(),
                         fmt_bytes(dev.total_memory).c_str(),
                         peak_pct);
            std::fprintf(stderr, "   Alloc: %llu calls | Free: %llu calls\n",
                         (unsigned long long)dev.alloc_calls,
                         (unsigned long long)dev.free_calls);

            if (!dev.gemm_by_dtype.empty()) {
                std::fprintf(stderr, "   GEMM:");
                for (const auto& [dtype, stats] : dev.gemm_by_dtype) {
                    std::fprintf(stderr, " %s=%llu", dtype.c_str(), (unsigned long long)stats.first);
                }
                std::fprintf(stderr, "\n");
            }

            if (dev.kernel_launch_total > 0) {
                std::fprintf(stderr, "   Kernels: %llu total",
                             (unsigned long long)dev.kernel_launch_total);
                if (!dev.kernel_launches.empty()) {
                    std::string top_name;
                    uint64_t top_count = 0;
                    for (const auto& [name, count] : dev.kernel_launches) {
                        if (count > top_count) {
                            top_name = name;
                            top_count = count;
                        }
                    }
                    if (top_name.size() > 40) {
                        top_name = top_name.substr(0, 37) + "...";
                    }
                    std::fprintf(stderr, " (top: %s=%llu)",
                                 top_name.c_str(), (unsigned long long)top_count);
                }
                std::fprintf(stderr, "\n");
            }

            const uint64_t total_io = dev.memcpy_h2d_bytes + dev.memcpy_d2h_bytes + dev.memcpy_d2d_bytes;
            if (total_io > 0) {
                std::fprintf(stderr, "   IO: H2D=%s  D2H=%s  D2D=%s\n",
                             fmt_bytes(dev.memcpy_h2d_bytes).c_str(),
                             fmt_bytes(dev.memcpy_d2h_bytes).c_str(),
                             fmt_bytes(dev.memcpy_d2d_bytes).c_str());
            }
            if (dev.transfer.any_copies()) {
                static const char* const labels[kCopyKindCount] = {"H2D", "D2H", "peer out", "peer in"};
                std::string line;
                for (std::size_t kind = 0; kind < kCopyKindCount; ++kind) {
                    const CopyCounter& copy = dev.transfer.copies[kind];
                    if (!copy.bytes) continue;
                    if (!line.empty()) line += ", ";
                    line += std::string(labels[kind]) + " " + fmt_bytes(copy.bytes) + " / " +
                            fmt_seconds(copy.modeled_seconds);
                }
                std::fprintf(stderr, "   Copies (modeled): %s\n", line.c_str());
            }

            if (!dev.compat_events.empty()) {
                std::fprintf(stderr, "   COMPAT WARNINGS:");
                for (const auto& [op, dtype, count] : dev.compat_events) {
                    std::fprintf(stderr, " %s(%s)x%llu",
                                 op.c_str(), dtype.c_str(), (unsigned long long)count);
                }
                std::fprintf(stderr, "\n");
            }
            if (!dev.unsupported_api_events.empty()) {
                std::fprintf(stderr, "   UNSUPPORTED APIS:");
                for (const auto& [op, behavior, policy, count] :
                     dev.unsupported_api_events) {
                    std::fprintf(
                        stderr,
                        " %s(%s,%s)x%llu",
                        op.c_str(),
                        behavior.c_str(),
                        policy.c_str(),
                        (unsigned long long)count);
                }
                std::fprintf(stderr, "\n");
            }

            std::fprintf(stderr, "------------------------------------------------------\n");
        }

        if (host_io.memcpy_calls > 0) {
            std::fprintf(stderr, " Host IO: %llu calls, %s\n",
                         (unsigned long long)host_io.memcpy_calls,
                         fmt_bytes(host_io.memcpy_bytes).c_str());
        }

        std::fprintf(stderr, "======================================================\n\n");
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

            if (!snapshot_has_meaningful_activity(devices, host_io)) {
                std::ifstream existing_report(report_path);
                if (existing_report.good()) {
                    FGPU_LOG("[Monitor] Skipping report overwrite for %s because this library instance has no activity\n", report_path);
                    return;
                }
            }

            FILE* out = fopen(report_path, "w");
            if (!out) {
                FGPU_LOG("[Monitor] Failed to open report file\n");
                return;
            }

            DeviceCounters totals;
            for (const auto& dev : devices) {
                add_counters(kMemoryCounters, dev, totals);
                add_counters(kIoCounters, dev, totals);
                add_counters(kComputeCounters, dev, totals);
            }

            fprintf(out, "{\n");
        fprintf(out, "  \"report_version\": \"%s\",\n", FAKEGPU_VERSION);

            // Add mode information
            const BackendConfig& config = BackendConfig::instance();
            fprintf(out, "  \"mode\": \"%s\",\n", mode_name(config.mode()));
            fprintf(
                out,
                "  \"unsupported_api_policy\": \"%s\",\n",
                unsupported_api_policy_name(config.unsupported_api_policy()));
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
            print_counter_rows(out, "    ", kMemoryCounters, totals, "bytes", true);
            fprintf(out, "    \"io\": {\n");
            print_counter_rows(out, "      ", kIoCounters, totals, "bytes", false);
            fprintf(out, "    },\n");
            fprintf(out, "    \"compute\": {\n");
            print_counter_rows(out, "      ", kComputeCounters, totals, "flops", false);
            fprintf(out, "    }\n");
            fprintf(out, "  },\n");
            fprintf(out, "  \"devices\": [\n");

            for (int i = 0; i < count; ++i) {
                const DeviceReportStats& dev = devices[static_cast<size_t>(i)];
                const uint64_t device_total_io_bytes = total_amount(kIoCounters, dev);
                const uint64_t device_total_flops = total_amount(kComputeCounters, dev);

                fprintf(out, "    {\n");
                fprintf(out, "      \"index\": %d,\n", dev.index);
                fprintf(out, "      \"name\": \"%s\",\n", dev.name.c_str());
                fprintf(out, "      \"uuid\": \"%s\",\n", dev.uuid.c_str());
                fprintf(out, "      \"gpu_profile\": {\n");
                fprintf(out, "        \"architecture\": \"%s\",\n", dev.architecture.c_str());
                fprintf(out, "        \"compute_capability\": \"%d.%d\",\n", dev.compute_major, dev.compute_minor);
                fprintf(out, "        \"supported_types\": [");
                for (size_t t = 0; t < dev.supported_types.size(); ++t) {
                    fprintf(out, "\"%s\"%s",
                            dev.supported_types[t].c_str(),
                            (t + 1 < dev.supported_types.size() ? ", " : ""));
                }
                fprintf(out, "]\n");
                fprintf(out, "      },\n");
                fprintf(out, "      \"total_memory\": %llu,\n", (unsigned long long)dev.total_memory);
                fprintf(out, "      \"used_memory_peak\": %llu,\n", (unsigned long long)dev.used_memory_peak);
                fprintf(out, "      \"used_memory_current\": %llu,\n", (unsigned long long)dev.used_memory_current);

                print_counter_rows(out, "      ", kMemoryCounters, dev, "bytes", true);

                fprintf(out, "      \"io\": {\n");
                print_counter_rows(out, "        ", kIoCounters, dev, "bytes", true);
                fprintf(out, "        \"total_bytes\": %llu\n", (unsigned long long)device_total_io_bytes);
                fprintf(out, "      },\n");

                // Modeled copy time and link use (fakegpu.transfer.v1 counters).
                fprintf(out, "      \"transfer\": {\n");
                fprintf(out, "        \"host_link\": \"%s\",\n", dev.transfer.host_link.c_str());
                fprintf(out, "        \"host_bandwidth_gbps\": %.12g,\n", dev.transfer.host_bandwidth_gbps);
                fprintf(out, "        \"modeled_seconds\": %.12g,\n", dev.transfer.modeled_seconds());
                fprintf(out, "        \"copies\": {");
                for (std::size_t kind = 0; kind < kCopyKindCount; ++kind) {
                    const CopyCounter& copy = dev.transfer.copies[kind];
                    fprintf(out, "%s\"%s\": {\"calls\": %llu, \"bytes\": %llu, \"modeled_seconds\": %.12g}",
                            kind ? ", " : "", copy_kind_name(kind), (unsigned long long)copy.calls,
                            (unsigned long long)copy.bytes, copy.modeled_seconds);
                }
                fprintf(out, "},\n");
                fprintf(out, "        \"links\": {");
                for (std::size_t link = 0; link < kTransferLinkCount; ++link) {
                    const LinkCounter& counter = dev.transfer.links[link];
                    fprintf(out, "%s\"%s\": {\"tx_bytes\": %llu, \"rx_bytes\": %llu}", link ? ", " : "",
                            transfer_link_name(link), (unsigned long long)counter.tx_bytes,
                            (unsigned long long)counter.rx_bytes);
                }
                fprintf(out, "}\n");
                fprintf(out, "      },\n");

                fprintf(out, "      \"compute\": {\n");
                print_counter_rows(out, "        ", kComputeCounters, dev, "flops", true);
                fprintf(out, "        \"total_flops\": %llu\n", (unsigned long long)device_total_flops);
                fprintf(out, "      },\n");

                fprintf(out, "      \"kernel_launches\": {\n");
                fprintf(out, "        \"total\": %llu", (unsigned long long)dev.kernel_launch_total);
                if (!dev.kernel_launches.empty()) {
                    fprintf(out, ",\n");
                    fprintf(out, "        \"by_name\": {\n");
                    size_t ki = 0;
                    for (const auto& [name, count] : dev.kernel_launches) {
                        fprintf(out, "          \"%s\": %llu%s\n",
                                name.c_str(),
                                (unsigned long long)count,
                                (ki + 1 < dev.kernel_launches.size() ? "," : ""));
                        ++ki;
                    }
                    fprintf(out, "        }\n");
                } else {
                    fprintf(out, "\n");
                }
                fprintf(out, "      },\n");

                fprintf(out, "      \"gemm_by_dtype\": {");
                if (!dev.gemm_by_dtype.empty()) {
                    fprintf(out, "\n");
                    size_t gi = 0;
                    for (const auto& [dtype_name, stats_pair] : dev.gemm_by_dtype) {
                        fprintf(out, "        \"%s\": {\"calls\": %llu, \"flops\": %llu}%s\n",
                                dtype_name.c_str(),
                                (unsigned long long)stats_pair.first,
                                (unsigned long long)stats_pair.second,
                                (gi + 1 < dev.gemm_by_dtype.size() ? "," : ""));
                        ++gi;
                    }
                    fprintf(out, "      }\n");
                } else {
                    fprintf(out, "}\n");
                }

                if (!dev.compat_events.empty()) {
                    fprintf(out, "      ,\"compatibility_events\": [\n");
                    for (size_t ci = 0; ci < dev.compat_events.size(); ++ci) {
                        const auto& [op, dtype, count] = dev.compat_events[ci];
                        fprintf(out, "        {\"operation\": \"%s\", \"dtype\": \"%s\", \"count\": %llu}%s\n",
                                op.c_str(),
                                dtype.c_str(),
                                (unsigned long long)count,
                                (ci + 1 < dev.compat_events.size() ? "," : ""));
                    }
                    fprintf(out, "      ]\n");
                }
                if (!dev.unsupported_api_events.empty()) {
                    fprintf(out, "      ,\"unsupported_api_events\": [\n");
                    for (size_t ui = 0; ui < dev.unsupported_api_events.size(); ++ui) {
                        const auto& [op, behavior, policy, count] =
                            dev.unsupported_api_events[ui];
                        fprintf(
                            out,
                            "        {\"operation\": \"%s\", \"behavior\": \"%s\", "
                            "\"policy\": \"%s\", \"count\": %llu}%s\n",
                            op.c_str(),
                            behavior.c_str(),
                            policy.c_str(),
                            (unsigned long long)count,
                            (ui + 1 < dev.unsupported_api_events.size() ? "," : ""));
                    }
                    fprintf(out, "      ]\n");
                }

                fprintf(out, "    }%s\n", (i < count - 1 ? "," : ""));
            }

            fprintf(out, "  ]\n");
            fprintf(out, "}\n");
            fclose(out);
            dump_terminal_summary(devices, host_io);
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
