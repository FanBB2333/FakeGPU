#pragma once

#include "cluster_config.hpp"
#include "communicator.hpp"

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace fake_gpu::distributed {

class ClusterCoordinator {
public:
    ClusterCoordinator(CoordinatorTransport transport, std::string address);
    explicit ClusterCoordinator(std::string socket_path);
    ~ClusterCoordinator();

    bool start(std::string& error);
    int run();
    void request_shutdown();
    const std::string& address() const { return address_; }
    const std::string& socket_path() const { return address_; }
    CoordinatorTransport transport() const { return transport_; }

private:
    void accept_loop();
    void handle_client(int client_fd);

    CoordinatorTransport transport_ = CoordinatorTransport::Unix;
    std::string address_;
    int server_fd_ = -1;
    std::atomic<bool> shutdown_requested_{false};
    std::vector<std::thread> client_threads_;
    std::mutex client_threads_mutex_;
    CommunicatorRegistry communicator_registry_;
};

}  // namespace fake_gpu::distributed
