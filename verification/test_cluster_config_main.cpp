#include <iostream>

#include "../src/core/backend_config.hpp"

int main() {
    const fake_gpu::BackendConfig& config = fake_gpu::BackendConfig::instance();
    if (config.has_configuration_error()) {
        std::cerr << config.configuration_error() << "\n";
        return 2;
    }

    const auto& dist = config.distributed_config();
    std::cout << "dist_mode=" << fake_gpu::distributed::distributed_mode_name(dist.mode) << "\n";
    std::cout << "dist_enabled=" << (dist.enabled() ? "true" : "false") << "\n";
    std::cout << "coordinator_transport="
              << fake_gpu::distributed::coordinator_transport_name(dist.coordinator_transport) << "\n";
    std::cout << "coordinator_addr=" << (dist.coordinator_address.empty() ? "<none>" : dist.coordinator_address) << "\n";
    std::cout << "cluster_config=" << (dist.cluster_config_path.empty() ? "<none>" : dist.cluster_config_path) << "\n";
    std::cout << "staging_chunk_bytes=" << dist.staging_chunk_bytes << "\n";
    if (dist.cluster_config.loaded()) {
        std::cout << "cluster_name=" << dist.cluster_config.name << "\n";
        std::cout << "world_size=" << dist.cluster_config.world_size << "\n";
        std::cout << "node_count=" << dist.cluster_config.nodes.size() << "\n";
    }
    return 0;
}
