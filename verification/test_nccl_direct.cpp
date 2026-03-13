#include "../src/distributed/cluster_coordinator.hpp"
#include "../src/nccl/nccl_defs.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

namespace {

using fake_gpu::distributed::ClusterCoordinator;

std::string make_temp_directory() {
    std::string pattern = "/tmp/fakegpu-nccl-direct-XXXXXX";
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back('\0');
    char* path = ::mkdtemp(buffer.data());
    if (!path) {
        throw std::runtime_error("mkdtemp failed");
    }
    return path;
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_result(ncclResult_t actual, ncclResult_t expected, const std::string& message) {
    if (actual != expected) {
        throw std::runtime_error(
            message + ": expected " + ncclGetErrorString(expected) +
            ", got " + ncclGetErrorString(actual));
    }
}

class CoordinatorFixture {
public:
    CoordinatorFixture() {
        temp_dir_ = make_temp_directory();
        socket_path_ = temp_dir_ + "/coordinator.sock";
        coordinator_ = std::make_unique<ClusterCoordinator>(socket_path_);

        std::string error;
        if (!coordinator_->start(error)) {
            throw std::runtime_error("failed to start coordinator: " + error);
        }

        thread_ = std::thread([this]() {
            coordinator_->run();
        });

        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
        while (std::chrono::steady_clock::now() < deadline) {
            if (std::filesystem::exists(socket_path_)) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        require(std::filesystem::exists(socket_path_), "coordinator socket did not appear");

        ::setenv("FAKEGPU_DIST_MODE", "simulate", 1);
        ::setenv("FAKEGPU_COORDINATOR_TRANSPORT", "unix", 1);
        ::setenv("FAKEGPU_COORDINATOR_ADDR", socket_path_.c_str(), 1);
    }

    ~CoordinatorFixture() {
        if (coordinator_) {
            coordinator_->request_shutdown();
        }
        if (thread_.joinable()) {
            thread_.join();
        }
        coordinator_.reset();
        if (!temp_dir_.empty()) {
            std::filesystem::remove_all(temp_dir_);
        }
    }

private:
    std::string temp_dir_;
    std::string socket_path_;
    std::unique_ptr<ClusterCoordinator> coordinator_;
    std::thread thread_;
};

void run_world_size_case(int world_size) {
    ncclUniqueId unique_id {};
    require_result(ncclGetUniqueId(&unique_id), ncclSuccess, "ncclGetUniqueId failed");

    std::vector<ncclComm_t> comms(static_cast<std::size_t>(world_size), nullptr);
    std::vector<ncclResult_t> results(static_cast<std::size_t>(world_size), ncclInternalError);
    std::vector<std::thread> threads;

    for (int rank = 0; rank < world_size; ++rank) {
        threads.emplace_back([&, rank]() {
            results[static_cast<std::size_t>(rank)] =
                ncclCommInitRank(&comms[static_cast<std::size_t>(rank)], world_size, unique_id, rank);
        });
    }

    for (std::thread& thread : threads) {
        thread.join();
    }

    for (int rank = 0; rank < world_size; ++rank) {
        require_result(results[static_cast<std::size_t>(rank)], ncclSuccess, "ncclCommInitRank failed");
        require(comms[static_cast<std::size_t>(rank)] != nullptr, "communicator handle was null");
    }

    for (int rank = 0; rank < world_size; ++rank) {
        require_result(
            ncclCommDestroy(comms[static_cast<std::size_t>(rank)]),
            ncclSuccess,
            "ncclCommDestroy failed");
    }
}

void run_invalid_argument_case() {
    ncclUniqueId unique_id {};
    require_result(ncclGetUniqueId(&unique_id), ncclSuccess, "ncclGetUniqueId failed");

    ncclComm_t comm = nullptr;
    require_result(
        ncclCommInitRank(&comm, 0, unique_id, 0),
        ncclInvalidArgument,
        "world_size=0 should fail");
    require(comm == nullptr, "communicator should stay null for invalid world size");

    require_result(
        ncclCommInitRank(&comm, 2, unique_id, 2),
        ncclInvalidArgument,
        "rank >= world_size should fail");
    require(comm == nullptr, "communicator should stay null for invalid rank");
}

void run_duplicate_destroy_case() {
    ncclUniqueId unique_id {};
    require_result(ncclGetUniqueId(&unique_id), ncclSuccess, "ncclGetUniqueId failed");

    ncclComm_t comm = nullptr;
    require_result(
        ncclCommInitRank(&comm, 1, unique_id, 0),
        ncclSuccess,
        "single-rank init should succeed");
    require(comm != nullptr, "single-rank communicator handle was null");

    require_result(ncclCommDestroy(comm), ncclSuccess, "first destroy should succeed");
    require_result(
        ncclCommDestroy(comm),
        ncclInvalidUsage,
        "second destroy should fail with ncclInvalidUsage");
}

}  // namespace

int main() {
    try {
        CoordinatorFixture fixture;

        int version = 0;
        require_result(ncclGetVersion(&version), ncclSuccess, "ncclGetVersion failed");
        require(version > 0, "ncclGetVersion returned a non-positive version");

        run_world_size_case(2);
        run_world_size_case(4);
        run_invalid_argument_case();
        run_duplicate_destroy_case();

        std::cout << "nccl direct init/destroy test passed" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
