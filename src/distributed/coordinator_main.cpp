#include "cluster_coordinator.hpp"
#include "cluster_config.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void print_usage() {
    std::cerr << "Usage: fakegpu-coordinator --transport {unix|tcp} --address <endpoint>\n";
}

std::string getenv_or_empty(const char* name) {
    const char* value = std::getenv(name);
    return value ? std::string(value) : std::string();
}

std::string node_name_for_rank(
    const fake_gpu::distributed::ClusterConfigModel& config,
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

bool dump_cluster_report(
    const fake_gpu::distributed::DistributedConfig& config,
    std::string& error) {
    const char* report_path = std::getenv("FAKEGPU_CLUSTER_REPORT_PATH");
    if (!report_path || !*report_path) {
        return true;
    }

    const fake_gpu::distributed::ClusterReportSnapshot snapshot =
        fake_gpu::distributed::snapshot_cluster_report();
    if (!snapshot.has_data) {
        return true;
    }

    FILE* out = std::fopen(report_path, "w");
    if (!out) {
        error = std::string("failed to open cluster report: ") + report_path;
        return false;
    }

    const auto& cluster_config = config.cluster_config;
    const std::size_t world_size = snapshot.world_size > 0
        ? snapshot.world_size
        : (cluster_config.loaded() ? cluster_config.world_size : snapshot.ranks.size());
    const std::size_t node_count = cluster_config.loaded()
        ? cluster_config.nodes.size()
        : (world_size > 0 ? 1U : 0U);

    std::fprintf(out, "{\n");
    std::fprintf(out, "  \"report_version\": 4,\n");
    std::fprintf(out, "  \"schema\": \"experimental\",\n");
    std::fprintf(out, "  \"cluster\": {\n");
    std::fprintf(out, "    \"mode\": \"%s\",\n",
                 fake_gpu::distributed::distributed_mode_name(config.mode));
    std::fprintf(out, "    \"world_size\": %llu,\n", (unsigned long long)world_size);
    std::fprintf(out, "    \"node_count\": %llu,\n", (unsigned long long)node_count);
    std::fprintf(out, "    \"communicators\": %llu,\n",
                 (unsigned long long)snapshot.communicator_count);
    std::fprintf(out, "    \"coordinator_transport\": \"%s\"",
                 fake_gpu::distributed::coordinator_transport_name(config.coordinator_transport));
    if (cluster_config.loaded()) {
        std::fprintf(out, ",\n");
        std::fprintf(out, "    \"name\": \"%s\",\n", cluster_config.name.c_str());
        std::fprintf(out, "    \"default_backend\": \"%s\",\n", cluster_config.default_backend.c_str());
        std::fprintf(out, "    \"config_path\": \"%s\"\n", cluster_config.source_path.c_str());
    } else {
        std::fprintf(out, "\n");
    }
    std::fprintf(out, "  },\n");
    std::fprintf(out, "  \"collectives\": {\n");
    std::fprintf(out, "    \"all_reduce\": {\"calls\": %llu, \"bytes\": %llu, \"estimated_time_us_total\": %.3f, \"contention_penalty_us_total\": %.3f},\n",
                 (unsigned long long)snapshot.all_reduce.calls,
                 (unsigned long long)snapshot.all_reduce.bytes,
                 snapshot.all_reduce.estimated_time_us_total,
                 snapshot.all_reduce.contention_penalty_us_total);
    std::fprintf(out, "    \"broadcast\": {\"calls\": %llu, \"bytes\": %llu, \"estimated_time_us_total\": %.3f, \"contention_penalty_us_total\": %.3f},\n",
                 (unsigned long long)snapshot.broadcast.calls,
                 (unsigned long long)snapshot.broadcast.bytes,
                 snapshot.broadcast.estimated_time_us_total,
                 snapshot.broadcast.contention_penalty_us_total);
    std::fprintf(out, "    \"all_gather\": {\"calls\": %llu, \"bytes\": %llu, \"estimated_time_us_total\": %.3f, \"contention_penalty_us_total\": %.3f},\n",
                 (unsigned long long)snapshot.all_gather.calls,
                 (unsigned long long)snapshot.all_gather.bytes,
                 snapshot.all_gather.estimated_time_us_total,
                 snapshot.all_gather.contention_penalty_us_total);
    std::fprintf(out, "    \"reduce_scatter\": {\"calls\": %llu, \"bytes\": %llu, \"estimated_time_us_total\": %.3f, \"contention_penalty_us_total\": %.3f},\n",
                 (unsigned long long)snapshot.reduce_scatter.calls,
                 (unsigned long long)snapshot.reduce_scatter.bytes,
                 snapshot.reduce_scatter.estimated_time_us_total,
                 snapshot.reduce_scatter.contention_penalty_us_total);
    std::fprintf(out, "    \"barrier\": {\"calls\": %llu, \"bytes\": %llu, \"estimated_time_us_total\": %.3f, \"contention_penalty_us_total\": %.3f}\n",
                 (unsigned long long)snapshot.barrier.calls,
                 (unsigned long long)snapshot.barrier.bytes,
                 snapshot.barrier.estimated_time_us_total,
                 snapshot.barrier.contention_penalty_us_total);
    std::fprintf(out, "  },\n");
    std::fprintf(out, "  \"links\": [\n");
    for (std::size_t index = 0; index < snapshot.links.size(); ++index) {
        const auto& link_stats = snapshot.links[index];
        std::fprintf(out, "    {\n");
        std::fprintf(out, "      \"src\": \"%s\",\n", link_stats.src_node.c_str());
        std::fprintf(out, "      \"dst\": \"%s\",\n", link_stats.dst_node.c_str());
        std::fprintf(out, "      \"scope\": \"%s\",\n", link_stats.scope.c_str());
        std::fprintf(out, "      \"samples\": %llu,\n", (unsigned long long)link_stats.samples);
        std::fprintf(out, "      \"bytes\": %llu,\n", (unsigned long long)link_stats.bytes);
        std::fprintf(out, "      \"bandwidth_gbps\": %.3f,\n", link_stats.bandwidth_gbps);
        std::fprintf(out, "      \"avg_latency_us\": %.3f,\n", link_stats.avg_latency_us);
        std::fprintf(out, "      \"estimated_time_us_total\": %.3f,\n", link_stats.estimated_time_us_total);
        std::fprintf(out, "      \"contention_penalty_us_total\": %.3f\n", link_stats.contention_penalty_us_total);
        std::fprintf(out, "    }%s\n", (index + 1 < snapshot.links.size() ? "," : ""));
    }
    std::fprintf(out, "  ],\n");
    std::fprintf(out, "  \"ranks\": [\n");
    for (std::size_t index = 0; index < snapshot.ranks.size(); ++index) {
        const auto& rank_stats = snapshot.ranks[index];
        const std::string node_name = node_name_for_rank(cluster_config, rank_stats.rank);
        std::fprintf(out, "    {\n");
        std::fprintf(out, "      \"rank\": %d,\n", rank_stats.rank);
        std::fprintf(out, "      \"node\": \"%s\",\n", node_name.c_str());
        std::fprintf(out, "      \"wait_time_ms\": %.3f,\n", rank_stats.wait_time_ms);
        std::fprintf(out, "      \"timeouts\": %llu,\n", (unsigned long long)rank_stats.timeouts);
        std::fprintf(out, "      \"communicator_inits\": %llu,\n",
                     (unsigned long long)rank_stats.communicator_inits);
        std::fprintf(out, "      \"collective_calls\": %llu,\n",
                     (unsigned long long)rank_stats.collective_calls);
        std::fprintf(out, "      \"barrier_calls\": %llu,\n",
                     (unsigned long long)rank_stats.barrier_calls);
        std::fprintf(out, "      \"group_prepares\": %llu\n",
                     (unsigned long long)rank_stats.group_prepares);
        std::fprintf(out, "    }%s\n", (index + 1 < snapshot.ranks.size() ? "," : ""));
    }
    std::fprintf(out, "  ]\n");
    std::fprintf(out, "}\n");
    std::fclose(out);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::string transport = getenv_or_empty("FAKEGPU_COORDINATOR_TRANSPORT");
    std::string address = getenv_or_empty("FAKEGPU_COORDINATOR_ADDR");

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--transport" && i + 1 < argc) {
            transport = argv[++i];
        } else if (arg == "--address" && i + 1 < argc) {
            address = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage();
            return 2;
        }
    }

    if (transport.empty()) {
        transport = "unix";
    }
    if (address.empty()) {
        std::cerr << "--address is required\n";
        return 2;
    }

    fake_gpu::distributed::CoordinatorTransport coordinator_transport =
        fake_gpu::distributed::CoordinatorTransport::Unix;
    if (transport == "unix") {
        coordinator_transport = fake_gpu::distributed::CoordinatorTransport::Unix;
    } else if (transport == "tcp") {
        coordinator_transport = fake_gpu::distributed::CoordinatorTransport::Tcp;
    } else {
        std::cerr << "Unsupported --transport: " << transport << "\n";
        return 2;
    }

    fake_gpu::distributed::ClusterCoordinator coordinator(coordinator_transport, address);
    std::string error;
    if (!coordinator.start(error)) {
        std::cerr << "Failed to start coordinator: " << error << "\n";
        return 1;
    }

    std::cout << "fakegpu-coordinator listening on " << coordinator.address() << "\n";
    std::cout.flush();
    int exit_code = coordinator.run();

    fake_gpu::distributed::DistributedConfig report_config =
        fake_gpu::distributed::parse_distributed_config_from_env();
    if (report_config.mode == fake_gpu::distributed::DistributedMode::Disabled) {
        report_config.mode = fake_gpu::distributed::DistributedMode::Simulate;
    }
    report_config.coordinator_transport = coordinator_transport;
    report_config.coordinator_address = address;

    if (!dump_cluster_report(report_config, error)) {
        std::cerr << "Failed to write cluster report: " << error << "\n";
        if (exit_code == 0) {
            exit_code = 1;
        }
    }

    return exit_code;
}
