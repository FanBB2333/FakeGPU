#include "../src/distributed/cluster_config.hpp"
#include "../src/distributed/topology_model.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

std::uint64_t parse_u64(const char* text, const char* field) {
    if (!text) {
        throw std::runtime_error(std::string("missing value for ") + field);
    }
    try {
        std::size_t consumed = 0;
        const std::uint64_t value = std::stoull(text, &consumed, 10);
        if (consumed != std::string(text).size()) {
            throw std::invalid_argument("trailing");
        }
        return value;
    } catch (...) {
        throw std::runtime_error(std::string("invalid integer value for ") + field);
    }
}

fake_gpu::distributed::CollectiveType parse_collective(const char* text) {
    if (!text) {
        throw std::runtime_error("missing value for --collective");
    }
    fake_gpu::distributed::CollectiveType type;
    if (!fake_gpu::distributed::parse_collective_type(text, type)) {
        throw std::runtime_error(std::string("unsupported collective type: ") + text);
    }
    return type;
}

void print_usage() {
    std::cerr << "Usage: fakegpu_topology_probe --cluster-config path --collective type --bytes-per-rank N\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string cluster_config_path;
        fake_gpu::distributed::CollectiveType collective =
            fake_gpu::distributed::CollectiveType::AllReduce;
        std::uint64_t bytes_per_rank = 0;

        for (int index = 1; index < argc; ++index) {
            const std::string arg = argv[index];
            if (arg == "--cluster-config" && index + 1 < argc) {
                cluster_config_path = argv[++index];
            } else if (arg == "--collective" && index + 1 < argc) {
                collective = parse_collective(argv[++index]);
            } else if (arg == "--bytes-per-rank" && index + 1 < argc) {
                bytes_per_rank = parse_u64(argv[++index], "--bytes-per-rank");
            } else if (arg == "--help" || arg == "-h") {
                print_usage();
                return 0;
            } else {
                throw std::runtime_error("unknown argument: " + arg);
            }
        }

        if (cluster_config_path.empty()) {
            throw std::runtime_error("--cluster-config is required");
        }
        if (bytes_per_rank == 0) {
            throw std::runtime_error("--bytes-per-rank must be > 0");
        }

        fake_gpu::distributed::ClusterConfigModel config;
        std::string error;
        if (!fake_gpu::distributed::load_cluster_config_from_yaml_file(cluster_config_path, config, error)) {
            throw std::runtime_error(error);
        }

        fake_gpu::distributed::TopologyModel model;
        if (!fake_gpu::distributed::TopologyModel::build(config, model, error)) {
            throw std::runtime_error(error);
        }

        fake_gpu::distributed::CollectiveTopologyEstimate estimate;
        if (!model.estimate_ring_collective(collective, bytes_per_rank, estimate, error)) {
            throw std::runtime_error(error);
        }

        std::cout << "{\n";
        std::cout << "  \"report_version\": 1,\n";
        std::cout << "  \"cluster\": {\n";
        std::cout << "    \"config_path\": \"" << cluster_config_path << "\",\n";
        std::cout << "    \"name\": \"" << config.name << "\",\n";
        std::cout << "    \"world_size\": " << config.world_size << ",\n";
        std::cout << "    \"node_count\": " << config.nodes.size() << "\n";
        std::cout << "  },\n";
        std::cout << "  \"collective\": {\n";
        std::cout << "    \"type\": \"" << fake_gpu::distributed::collective_type_name(collective) << "\",\n";
        std::cout << "    \"algorithm\": \"" << estimate.algorithm << "\",\n";
        std::cout << "    \"bytes_per_rank\": " << estimate.bytes_per_rank << ",\n";
        std::cout << "    \"estimated_time_us\": " << estimate.estimated_time_us << "\n";
        std::cout << "  },\n";
        std::cout << "  \"links\": [\n";
        for (std::size_t index = 0; index < estimate.links.size(); ++index) {
            const auto& link = estimate.links[index];
            std::cout << "    {\n";
            std::cout << "      \"src\": \"" << link.src_node << "\",\n";
            std::cout << "      \"dst\": \"" << link.dst_node << "\",\n";
            std::cout << "      \"scope\": \"" << fake_gpu::distributed::topology_link_scope_name(link.scope) << "\",\n";
            std::cout << "      \"hop_count\": " << link.hop_count << ",\n";
            std::cout << "      \"bytes\": " << link.bytes << ",\n";
            std::cout << "      \"bandwidth_gbps\": " << link.bandwidth_gbps << ",\n";
            std::cout << "      \"latency_us\": " << link.latency_us << ",\n";
            std::cout << "      \"oversubscription\": " << link.oversubscription << ",\n";
            std::cout << "      \"estimated_time_us\": " << link.estimated_time_us << "\n";
            std::cout << "    }" << (index + 1 < estimate.links.size() ? "," : "") << "\n";
        }
        std::cout << "  ]\n";
        std::cout << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
