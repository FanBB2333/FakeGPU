#include "../src/distributed/cluster_coordinator.hpp"
#include "../src/nccl/nccl_defs.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

namespace {

using fake_gpu::distributed::ClusterCoordinator;

std::string make_temp_directory() {
    std::string pattern = "/tmp/fakegpu-group-XXXXXX";
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

std::vector<ncclComm_t> init_communicators(int world_size) {
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

    return comms;
}

void destroy_communicators(std::vector<ncclComm_t>& comms) {
    for (ncclComm_t comm : comms) {
        require_result(ncclCommDestroy(comm), ncclSuccess, "ncclCommDestroy failed");
    }
}

void run_success_case(int world_size) {
    std::vector<ncclComm_t> comms = init_communicators(world_size);
    std::vector<float> values(static_cast<std::size_t>(world_size), 0.0f);
    std::vector<ncclResult_t> results(static_cast<std::size_t>(world_size), ncclInternalError);
    std::vector<std::thread> threads;

    const float expected = static_cast<float>(world_size * (world_size + 1) / 2);

    for (int rank = 0; rank < world_size; ++rank) {
        values[static_cast<std::size_t>(rank)] = static_cast<float>(rank + 1);
        threads.emplace_back([&, rank]() {
            float* buffer = &values[static_cast<std::size_t>(rank)];
            require_result(ncclGroupStart(), ncclSuccess, "ncclGroupStart failed");
            require_result(
                ncclAllReduce(buffer, buffer, 1, ncclFloat32, ncclSum, comms[static_cast<std::size_t>(rank)], nullptr),
                ncclSuccess,
                "grouped ncclAllReduce enqueue failed");
            require_result(
                ncclBroadcast(buffer, buffer, 1, ncclFloat32, 0, comms[static_cast<std::size_t>(rank)], nullptr),
                ncclSuccess,
                "grouped ncclBroadcast enqueue failed");
            results[static_cast<std::size_t>(rank)] = ncclGroupEnd();
        });
    }

    for (std::thread& thread : threads) {
        thread.join();
    }

    for (int rank = 0; rank < world_size; ++rank) {
        require_result(results[static_cast<std::size_t>(rank)], ncclSuccess, "grouped collectives failed");
        if (std::fabs(values[static_cast<std::size_t>(rank)] - expected) > 1e-6f) {
            throw std::runtime_error("grouped collectives did not preserve execution order");
        }
    }

    destroy_communicators(comms);
}

void run_second_op_mismatch_case() {
    std::vector<ncclComm_t> comms = init_communicators(2);
    std::vector<float> rank0_values = {1.0f, 10.0f};
    std::vector<float> rank1_values = {2.0f, 20.0f};
    std::vector<ncclResult_t> results(2, ncclInternalError);

    std::thread thread0([&]() {
        require_result(ncclGroupStart(), ncclSuccess, "ncclGroupStart failed");
        require_result(
            ncclAllReduce(rank0_values.data(), rank0_values.data(), 1, ncclFloat32, ncclSum, comms[0], nullptr),
            ncclSuccess,
            "rank 0 grouped allreduce enqueue failed");
        require_result(
            ncclBroadcast(rank0_values.data(), rank0_values.data(), 1, ncclFloat32, 0, comms[0], nullptr),
            ncclSuccess,
            "rank 0 grouped broadcast enqueue failed");
        results[0] = ncclGroupEnd();
    });

    std::thread thread1([&]() {
        require_result(ncclGroupStart(), ncclSuccess, "ncclGroupStart failed");
        require_result(
            ncclAllReduce(rank1_values.data(), rank1_values.data(), 1, ncclFloat32, ncclSum, comms[1], nullptr),
            ncclSuccess,
            "rank 1 grouped allreduce enqueue failed");
        require_result(
            ncclBroadcast(rank1_values.data(), rank1_values.data(), 2, ncclFloat32, 0, comms[1], nullptr),
            ncclSuccess,
            "rank 1 grouped broadcast enqueue failed");
        results[1] = ncclGroupEnd();
    });

    thread0.join();
    thread1.join();

    require_result(results[0], ncclInvalidUsage, "group mismatch should fail on rank 0");
    require_result(results[1], ncclInvalidUsage, "group mismatch should fail on rank 1");
    require(std::fabs(rank0_values[0] - 1.0f) < 1e-6f, "rank 0 allreduce should not execute after group mismatch");
    require(std::fabs(rank1_values[0] - 2.0f) < 1e-6f, "rank 1 allreduce should not execute after group mismatch");

    destroy_communicators(comms);
}

}  // namespace

int main() {
    try {
        CoordinatorFixture fixture;
        run_success_case(2);
        run_success_case(4);
        run_second_op_mismatch_case();
        std::cout << "group semantics test passed" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
