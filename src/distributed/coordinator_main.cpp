#include "cluster_coordinator.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void print_usage() {
    std::cerr << "Usage: fakegpu-coordinator --transport unix --address /abs/path.sock\n";
}

std::string getenv_or_empty(const char* name) {
    const char* value = std::getenv(name);
    return value ? std::string(value) : std::string();
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
    if (transport != "unix") {
        std::cerr << "Only --transport unix is supported in this stage\n";
        return 2;
    }
    if (address.empty()) {
        std::cerr << "--address is required\n";
        return 2;
    }

    fake_gpu::distributed::ClusterCoordinator coordinator(address);
    std::string error;
    if (!coordinator.start(error)) {
        std::cerr << "Failed to start coordinator: " << error << "\n";
        return 1;
    }

    std::cout << "fakegpu-coordinator listening on " << coordinator.socket_path() << "\n";
    std::cout.flush();
    return coordinator.run();
}
