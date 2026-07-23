#include "cluster_coordinator.hpp"
#include "cluster_config.hpp"
#include "cluster_report_writer.hpp"

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

bool dump_cluster_report(
    const fake_gpu::distributed::DistributedConfig& config,
    std::string& error) {
    const char* report_path = std::getenv("FAKEGPU_CLUSTER_REPORT_PATH");
    if (!report_path || !*report_path) {
        return true;
    }

    const fake_gpu::distributed::ClusterReportSnapshot snapshot =
        fake_gpu::distributed::snapshot_cluster_report();
    return fake_gpu::distributed::write_cluster_report_files(
        config,
        snapshot,
        report_path,
        error);
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
