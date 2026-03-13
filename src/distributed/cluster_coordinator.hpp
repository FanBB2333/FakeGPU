#pragma once

#include "communicator.hpp"

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace fake_gpu::distributed {

class ClusterCoordinator {
public:
    explicit ClusterCoordinator(std::string socket_path);
    ~ClusterCoordinator();

    bool start(std::string& error);
    int run();
    void request_shutdown();
    const std::string& socket_path() const { return socket_path_; }

private:
    void accept_loop();
    void handle_client(int client_fd);

    std::string socket_path_;
    int server_fd_ = -1;
    std::atomic<bool> shutdown_requested_{false};
    std::vector<std::thread> client_threads_;
    std::mutex client_threads_mutex_;
    CommunicatorRegistry communicator_registry_;
};

}  // namespace fake_gpu::distributed
