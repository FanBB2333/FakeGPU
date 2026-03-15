#include "../src/distributed/cluster_coordinator.hpp"
#include "../src/cuda/cuda_driver_defs.hpp"
#include "../src/nccl/nccl_defs.hpp"

#include <chrono>
#include <array>
#include <cstdint>
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

void run_comm_init_all_case() {
    std::array<int, 3> devlist = {3, 1, 2};
    std::array<ncclComm_t, 3> comms = {nullptr, nullptr, nullptr};

    require_result(
        ncclCommInitAll(comms.data(), static_cast<int>(comms.size()), devlist.data()),
        ncclSuccess,
        "ncclCommInitAll should succeed");

    for (std::size_t index = 0; index < comms.size(); ++index) {
        ncclComm_t comm = comms[index];
        require(comm != nullptr, "ncclCommInitAll returned a null communicator");

        int count = -1;
        int rank = -1;
        int device = -1;
        require_result(ncclCommCount(comm, &count), ncclSuccess, "ncclCommCount failed");
        require_result(ncclCommUserRank(comm, &rank), ncclSuccess, "ncclCommUserRank failed");
        require_result(ncclCommCuDevice(comm, &device), ncclSuccess, "ncclCommCuDevice failed");

        require(count == static_cast<int>(comms.size()), "ncclCommCount returned the wrong world size");
        require(rank == static_cast<int>(index), "ncclCommUserRank returned the wrong rank");
        require(device == devlist[index], "ncclCommCuDevice returned the wrong device id");
    }

    for (ncclComm_t comm : comms) {
        require_result(ncclCommDestroy(comm), ncclSuccess, "ncclCommDestroy after ncclCommInitAll failed");
    }
}

void run_comm_init_all_invalid_case() {
    std::array<ncclComm_t, 2> comms = {nullptr, nullptr};
    require_result(
        ncclCommInitAll(comms.data(), 0, nullptr),
        ncclInvalidArgument,
        "ndev=0 should fail");
    require(comms[0] == nullptr && comms[1] == nullptr, "invalid ncclCommInitAll should not mutate outputs");
}

void run_comm_split_case() {
    const int world_size = 4;
    ncclUniqueId unique_id {};
    require_result(ncclGetUniqueId(&unique_id), ncclSuccess, "ncclGetUniqueId failed");

    std::array<ncclComm_t, world_size> parents = {nullptr, nullptr, nullptr, nullptr};
    std::array<ncclResult_t, world_size> init_results = {
        ncclInternalError, ncclInternalError, ncclInternalError, ncclInternalError};
    std::vector<std::thread> init_threads;
    for (int rank = 0; rank < world_size; ++rank) {
        init_threads.emplace_back([&, rank]() {
            init_results[static_cast<std::size_t>(rank)] =
                ncclCommInitRank(&parents[static_cast<std::size_t>(rank)], world_size, unique_id, rank);
        });
    }
    for (std::thread& thread : init_threads) {
        thread.join();
    }
    for (int rank = 0; rank < world_size; ++rank) {
        require_result(init_results[static_cast<std::size_t>(rank)], ncclSuccess, "parent init failed");
    }

    const std::array<int, world_size> colors = {0, 1, 0, NCCL_SPLIT_NOCOLOR};
    const std::array<int, world_size> keys = {20, 0, 10, 0};
    std::array<ncclComm_t, world_size> children = {nullptr, nullptr, nullptr, nullptr};
    std::array<ncclResult_t, world_size> split_results = {
        ncclInternalError, ncclInternalError, ncclInternalError, ncclInternalError};
    std::vector<std::thread> split_threads;
    for (int rank = 0; rank < world_size; ++rank) {
        split_threads.emplace_back([&, rank]() {
            split_results[static_cast<std::size_t>(rank)] = ncclCommSplit(
                parents[static_cast<std::size_t>(rank)],
                colors[static_cast<std::size_t>(rank)],
                keys[static_cast<std::size_t>(rank)],
                &children[static_cast<std::size_t>(rank)],
                nullptr);
        });
    }
    for (std::thread& thread : split_threads) {
        thread.join();
    }
    for (int rank = 0; rank < world_size; ++rank) {
        require_result(split_results[static_cast<std::size_t>(rank)], ncclSuccess, "ncclCommSplit failed");
    }

    require(children[3] == nullptr, "NCCL_SPLIT_NOCOLOR should return a null child communicator");

    int child_count = -1;
    int child_rank = -1;
    require_result(ncclCommCount(children[2], &child_count), ncclSuccess, "split child count failed");
    require_result(ncclCommUserRank(children[2], &child_rank), ncclSuccess, "split child rank failed");
    require(child_count == 2, "color=0 subgroup should have size 2");
    require(child_rank == 0, "rank 2 should become subgroup rank 0 because of smaller key");

    require_result(ncclCommCount(children[0], &child_count), ncclSuccess, "split child count failed");
    require_result(ncclCommUserRank(children[0], &child_rank), ncclSuccess, "split child rank failed");
    require(child_count == 2, "color=0 subgroup should have size 2");
    require(child_rank == 1, "rank 0 should become subgroup rank 1 because of larger key");

    require_result(ncclCommCount(children[1], &child_count), ncclSuccess, "single-rank child count failed");
    require_result(ncclCommUserRank(children[1], &child_rank), ncclSuccess, "single-rank child rank failed");
    require(child_count == 1, "color=1 subgroup should have size 1");
    require(child_rank == 0, "single-rank subgroup rank should be 0");

    std::array<float, 2> color0_send = {1.0f, 3.0f};
    std::array<float, 2> color0_recv = {0.0f, 0.0f};
    std::array<ncclResult_t, 2> color0_results = {ncclInternalError, ncclInternalError};
    std::thread color0_thread0([&]() {
        color0_results[0] = ncclAllReduce(&color0_send[0], &color0_recv[0], 1, ncclFloat32, ncclSum, children[0], nullptr);
    });
    std::thread color0_thread1([&]() {
        color0_results[1] = ncclAllReduce(&color0_send[1], &color0_recv[1], 1, ncclFloat32, ncclSum, children[2], nullptr);
    });
    color0_thread0.join();
    color0_thread1.join();
    require_result(color0_results[0], ncclSuccess, "color=0 subgroup allreduce failed");
    require_result(color0_results[1], ncclSuccess, "color=0 subgroup allreduce failed");
    require(color0_recv[0] == 4.0f && color0_recv[1] == 4.0f, "color=0 subgroup allreduce mismatch");

    float color1_send = 7.0f;
    float color1_recv = 0.0f;
    require_result(
        ncclAllReduce(&color1_send, &color1_recv, 1, ncclFloat32, ncclSum, children[1], nullptr),
        ncclSuccess,
        "single-rank subgroup allreduce failed");
    require(color1_recv == 7.0f, "single-rank subgroup allreduce should preserve value");

    std::array<float, world_size> parent_send = {1.0f, 2.0f, 3.0f, 4.0f};
    std::array<float, world_size> parent_recv = {0.0f, 0.0f, 0.0f, 0.0f};
    std::array<ncclResult_t, world_size> parent_results = {
        ncclInternalError, ncclInternalError, ncclInternalError, ncclInternalError};
    std::vector<std::thread> parent_threads;
    for (int rank = 0; rank < world_size; ++rank) {
        parent_threads.emplace_back([&, rank]() {
            parent_results[static_cast<std::size_t>(rank)] = ncclAllReduce(
                &parent_send[static_cast<std::size_t>(rank)],
                &parent_recv[static_cast<std::size_t>(rank)],
                1,
                ncclFloat32,
                ncclSum,
                parents[static_cast<std::size_t>(rank)],
                nullptr);
        });
    }
    for (std::thread& thread : parent_threads) {
        thread.join();
    }
    for (int rank = 0; rank < world_size; ++rank) {
        require_result(parent_results[static_cast<std::size_t>(rank)], ncclSuccess, "parent allreduce after split failed");
        require(parent_recv[static_cast<std::size_t>(rank)] == 10.0f, "parent allreduce after split mismatch");
    }

    for (ncclComm_t child : children) {
        if (child) {
            require_result(ncclCommDestroy(child), ncclSuccess, "child destroy failed");
        }
    }
    for (ncclComm_t parent : parents) {
        require_result(ncclCommDestroy(parent), ncclSuccess, "parent destroy failed");
    }
}

void run_send_recv_case() {
    ncclUniqueId unique_id {};
    require_result(ncclGetUniqueId(&unique_id), ncclSuccess, "ncclGetUniqueId failed");

    std::array<ncclComm_t, 2> comms = {nullptr, nullptr};
    std::array<ncclResult_t, 2> init_results = {ncclInternalError, ncclInternalError};
    std::vector<std::thread> init_threads;
    for (int rank = 0; rank < 2; ++rank) {
        init_threads.emplace_back([&, rank]() {
            init_results[static_cast<std::size_t>(rank)] =
                ncclCommInitRank(&comms[static_cast<std::size_t>(rank)], 2, unique_id, rank);
        });
    }
    for (std::thread& thread : init_threads) {
        thread.join();
    }
    for (int rank = 0; rank < 2; ++rank) {
        require_result(init_results[static_cast<std::size_t>(rank)], ncclSuccess, "p2p init failed");
    }

    std::array<float, 4> send = {1.5f, 2.5f, 3.5f, 4.5f};
    std::array<float, 4> recv = {0.0f, 0.0f, 0.0f, 0.0f};
    std::array<ncclResult_t, 2> results = {ncclInternalError, ncclInternalError};
    std::thread sender([&]() {
        results[0] = ncclSend(send.data(), send.size(), ncclFloat32, 1, comms[0], nullptr);
    });
    std::thread receiver([&]() {
        results[1] = ncclRecv(recv.data(), recv.size(), ncclFloat32, 0, comms[1], nullptr);
    });
    sender.join();
    receiver.join();
    require_result(results[0], ncclSuccess, "ncclSend failed");
    require_result(results[1], ncclSuccess, "ncclRecv failed");
    for (std::size_t index = 0; index < send.size(); ++index) {
        require(send[index] == recv[index], "2-rank send/recv payload mismatch");
    }

    for (ncclComm_t comm : comms) {
        require_result(ncclCommDestroy(comm), ncclSuccess, "destroy after send/recv failed");
    }
}

void run_send_recv_multi_pair_case() {
    const int world_size = 4;
    ncclUniqueId unique_id {};
    require_result(ncclGetUniqueId(&unique_id), ncclSuccess, "ncclGetUniqueId failed");

    std::array<ncclComm_t, world_size> comms = {nullptr, nullptr, nullptr, nullptr};
    std::array<ncclResult_t, world_size> init_results = {
        ncclInternalError, ncclInternalError, ncclInternalError, ncclInternalError};
    std::vector<std::thread> init_threads;
    for (int rank = 0; rank < world_size; ++rank) {
        init_threads.emplace_back([&, rank]() {
            init_results[static_cast<std::size_t>(rank)] =
                ncclCommInitRank(&comms[static_cast<std::size_t>(rank)], world_size, unique_id, rank);
        });
    }
    for (std::thread& thread : init_threads) {
        thread.join();
    }
    for (int rank = 0; rank < world_size; ++rank) {
        require_result(init_results[static_cast<std::size_t>(rank)], ncclSuccess, "multi-pair init failed");
    }

    std::array<int, world_size> peers = {1, 0, 3, 2};
    std::array<bool, world_size> is_send = {true, false, false, true};
    std::array<std::array<std::int32_t, 3>, world_size> send_buffers = {
        std::array<std::int32_t, 3>{11, 12, 13},
        std::array<std::int32_t, 3>{0, 0, 0},
        std::array<std::int32_t, 3>{0, 0, 0},
        std::array<std::int32_t, 3>{31, 32, 33},
    };
    std::array<std::array<std::int32_t, 3>, world_size> recv_buffers = {
        std::array<std::int32_t, 3>{0, 0, 0},
        std::array<std::int32_t, 3>{0, 0, 0},
        std::array<std::int32_t, 3>{0, 0, 0},
        std::array<std::int32_t, 3>{0, 0, 0},
    };
    std::array<ncclResult_t, world_size> results = {
        ncclInternalError, ncclInternalError, ncclInternalError, ncclInternalError};
    std::vector<std::thread> threads;
    for (int rank = 0; rank < world_size; ++rank) {
        threads.emplace_back([&, rank]() {
            if (is_send[static_cast<std::size_t>(rank)]) {
                results[static_cast<std::size_t>(rank)] = ncclSend(
                    send_buffers[static_cast<std::size_t>(rank)].data(),
                    send_buffers[static_cast<std::size_t>(rank)].size(),
                    ncclInt32,
                    peers[static_cast<std::size_t>(rank)],
                    comms[static_cast<std::size_t>(rank)],
                    nullptr);
            } else {
                results[static_cast<std::size_t>(rank)] = ncclRecv(
                    recv_buffers[static_cast<std::size_t>(rank)].data(),
                    recv_buffers[static_cast<std::size_t>(rank)].size(),
                    ncclInt32,
                    peers[static_cast<std::size_t>(rank)],
                    comms[static_cast<std::size_t>(rank)],
                    nullptr);
            }
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    for (int rank = 0; rank < world_size; ++rank) {
        require_result(results[static_cast<std::size_t>(rank)], ncclSuccess, "multi-pair send/recv failed");
    }

    require(recv_buffers[1] == send_buffers[0], "rank 1 recv payload mismatch");
    require(recv_buffers[2] == send_buffers[3], "rank 2 recv payload mismatch");

    for (ncclComm_t comm : comms) {
        require_result(ncclCommDestroy(comm), ncclSuccess, "destroy after multi-pair send/recv failed");
    }
}

void run_grouped_send_recv_case() {
    ncclUniqueId unique_id {};
    require_result(ncclGetUniqueId(&unique_id), ncclSuccess, "ncclGetUniqueId failed");

    std::array<ncclComm_t, 2> comms = {nullptr, nullptr};
    std::array<ncclResult_t, 2> init_results = {ncclInternalError, ncclInternalError};
    std::vector<std::thread> init_threads;
    for (int rank = 0; rank < 2; ++rank) {
        init_threads.emplace_back([&, rank]() {
            init_results[static_cast<std::size_t>(rank)] =
                ncclCommInitRank(&comms[static_cast<std::size_t>(rank)], 2, unique_id, rank);
        });
    }
    for (std::thread& thread : init_threads) {
        thread.join();
    }
    for (int rank = 0; rank < 2; ++rank) {
        require_result(init_results[static_cast<std::size_t>(rank)], ncclSuccess, "grouped p2p init failed");
    }

    std::array<float, 3> send0 = {1.0f, 2.0f, 3.0f};
    std::array<float, 3> send1 = {4.0f, 5.0f, 6.0f};
    std::array<float, 3> recv0 = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> recv1 = {0.0f, 0.0f, 0.0f};
    std::array<ncclResult_t, 2> results = {ncclInternalError, ncclInternalError};

    std::thread rank0([&]() {
        require_result(ncclGroupStart(), ncclSuccess, "rank0 group start failed");
        require_result(
            ncclSend(send0.data(), send0.size(), ncclFloat32, 1, comms[0], nullptr),
            ncclSuccess,
            "rank0 grouped send failed");
        require_result(
            ncclRecv(recv0.data(), recv0.size(), ncclFloat32, 1, comms[0], nullptr),
            ncclSuccess,
            "rank0 grouped recv failed");
        results[0] = ncclGroupEnd();
    });
    std::thread rank1([&]() {
        require_result(ncclGroupStart(), ncclSuccess, "rank1 group start failed");
        require_result(
            ncclRecv(recv1.data(), recv1.size(), ncclFloat32, 0, comms[1], nullptr),
            ncclSuccess,
            "rank1 grouped recv failed");
        require_result(
            ncclSend(send1.data(), send1.size(), ncclFloat32, 0, comms[1], nullptr),
            ncclSuccess,
            "rank1 grouped send failed");
        results[1] = ncclGroupEnd();
    });
    rank0.join();
    rank1.join();

    require_result(results[0], ncclSuccess, "rank0 grouped ncclGroupEnd failed");
    require_result(results[1], ncclSuccess, "rank1 grouped ncclGroupEnd failed");
    require(recv0 == send1, "rank0 grouped recv payload mismatch");
    require(recv1 == send0, "rank1 grouped recv payload mismatch");

    for (ncclComm_t comm : comms) {
        require_result(ncclCommDestroy(comm), ncclSuccess, "destroy after grouped send/recv failed");
    }
}

void run_grouped_send_recv_then_collective_case() {
    ncclUniqueId unique_id {};
    require_result(ncclGetUniqueId(&unique_id), ncclSuccess, "ncclGetUniqueId failed");

    std::array<ncclComm_t, 2> comms = {nullptr, nullptr};
    std::array<ncclResult_t, 2> init_results = {ncclInternalError, ncclInternalError};
    std::vector<std::thread> init_threads;
    for (int rank = 0; rank < 2; ++rank) {
        init_threads.emplace_back([&, rank]() {
            init_results[static_cast<std::size_t>(rank)] =
                ncclCommInitRank(&comms[static_cast<std::size_t>(rank)], 2, unique_id, rank);
        });
    }
    for (std::thread& thread : init_threads) {
        thread.join();
    }
    for (int rank = 0; rank < 2; ++rank) {
        require_result(init_results[static_cast<std::size_t>(rank)], ncclSuccess, "mixed group init failed");
    }

    std::array<std::int32_t, 2> send_payload = {7, 8};
    std::array<std::int32_t, 2> recv_payload = {0, 0};
    std::array<float, 1> send_values0 = {1.25f};
    std::array<float, 1> send_values1 = {2.75f};
    std::array<float, 1> recv_values0 = {0.0f};
    std::array<float, 1> recv_values1 = {0.0f};
    std::array<ncclResult_t, 2> results = {ncclInternalError, ncclInternalError};

    std::thread rank0([&]() {
        require_result(ncclGroupStart(), ncclSuccess, "mixed rank0 group start failed");
        require_result(
            ncclSend(send_payload.data(), send_payload.size(), ncclInt32, 1, comms[0], nullptr),
            ncclSuccess,
            "mixed rank0 grouped send failed");
        require_result(
            ncclAllReduce(send_values0.data(), recv_values0.data(), send_values0.size(), ncclFloat32, ncclSum, comms[0], nullptr),
            ncclSuccess,
            "mixed rank0 grouped allreduce enqueue failed");
        results[0] = ncclGroupEnd();
    });
    std::thread rank1([&]() {
        require_result(ncclGroupStart(), ncclSuccess, "mixed rank1 group start failed");
        require_result(
            ncclRecv(recv_payload.data(), recv_payload.size(), ncclInt32, 0, comms[1], nullptr),
            ncclSuccess,
            "mixed rank1 grouped recv failed");
        require_result(
            ncclAllReduce(send_values1.data(), recv_values1.data(), send_values1.size(), ncclFloat32, ncclSum, comms[1], nullptr),
            ncclSuccess,
            "mixed rank1 grouped allreduce enqueue failed");
        results[1] = ncclGroupEnd();
    });
    rank0.join();
    rank1.join();

    require_result(results[0], ncclSuccess, "mixed rank0 ncclGroupEnd failed");
    require_result(results[1], ncclSuccess, "mixed rank1 ncclGroupEnd failed");
    require(recv_payload == send_payload, "mixed grouped recv payload mismatch");
    require(recv_values0[0] == 4.0f, "mixed grouped allreduce rank0 mismatch");
    require(recv_values1[0] == 4.0f, "mixed grouped allreduce rank1 mismatch");

    for (ncclComm_t comm : comms) {
        require_result(ncclCommDestroy(comm), ncclSuccess, "destroy after mixed grouped send/recv failed");
    }
}

void run_send_recv_timeout_case() {
    ncclUniqueId unique_id {};
    require_result(ncclGetUniqueId(&unique_id), ncclSuccess, "ncclGetUniqueId failed");

    std::array<ncclComm_t, 2> comms = {nullptr, nullptr};
    std::array<ncclResult_t, 2> init_results = {ncclInternalError, ncclInternalError};
    std::vector<std::thread> init_threads;
    for (int rank = 0; rank < 2; ++rank) {
        init_threads.emplace_back([&, rank]() {
            init_results[static_cast<std::size_t>(rank)] =
                ncclCommInitRank(&comms[static_cast<std::size_t>(rank)], 2, unique_id, rank);
        });
    }
    for (std::thread& thread : init_threads) {
        thread.join();
    }
    for (int rank = 0; rank < 2; ++rank) {
        require_result(init_results[static_cast<std::size_t>(rank)], ncclSuccess, "timeout init failed");
    }

    std::array<float, 2> send = {9.0f, 10.0f};
    ncclResult_t result = ncclInternalError;
    const auto begin = std::chrono::steady_clock::now();
    std::thread sender([&]() {
        result = ncclSend(send.data(), send.size(), ncclFloat32, 1, comms[0], nullptr);
    });
    sender.join();
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - begin).count();

    require_result(result, ncclSystemError, "missing recv should time out");
    require(elapsed_ms < 5000, "send timeout exceeded 5 seconds");
    require_result(ncclCommDestroy(comms[0]), ncclSuccess, "destroy after timeout failed");
    require_result(ncclCommDestroy(comms[1]), ncclSuccess, "destroy after timeout failed");
}

void run_premul_redop_api_case() {
    ncclRedOp_t op = ncclSum;
    float scalar = 2.0f;
    require_result(
        ncclRedOpCreatePreMulSum(&op, &scalar, ncclFloat32, ncclScalarHostImmediate, nullptr),
        ncclSuccess,
        "ncclRedOpCreatePreMulSum should succeed");
    require(static_cast<int>(op) != static_cast<int>(ncclSum), "custom redop should not alias ncclSum");
    require_result(ncclRedOpDestroy(op, nullptr), ncclSuccess, "ncclRedOpDestroy should succeed");

    require_result(
        ncclRedOpCreatePreMulSum(nullptr, &scalar, ncclFloat32, ncclScalarHostImmediate, nullptr),
        ncclInvalidArgument,
        "null op should fail");
    require_result(
        ncclRedOpCreatePreMulSum(&op, nullptr, ncclFloat32, ncclScalarHostImmediate, nullptr),
        ncclInvalidArgument,
        "null scalar should fail");
    require_result(
        ncclRedOpCreatePreMulSum(&op, &scalar, ncclFloat16, ncclScalarHostImmediate, nullptr),
        ncclInvalidArgument,
        "unsupported datatype should fail");
    require_result(
        ncclRedOpCreatePreMulSum(&op, &scalar, ncclFloat32, ncclScalarDevice, nullptr),
        ncclInvalidUsage,
        "device scalar residence should fail");
}

void run_async_error_persistence_case() {
    ncclUniqueId unique_id {};
    require_result(ncclGetUniqueId(&unique_id), ncclSuccess, "ncclGetUniqueId failed");

    std::array<ncclComm_t, 2> comms = {nullptr, nullptr};
    std::array<ncclResult_t, 2> init_results = {ncclInternalError, ncclInternalError};
    std::vector<std::thread> init_threads;
    for (int rank = 0; rank < 2; ++rank) {
        init_threads.emplace_back([&, rank]() {
            init_results[static_cast<std::size_t>(rank)] =
                ncclCommInitRank(&comms[static_cast<std::size_t>(rank)], 2, unique_id, rank);
        });
    }
    for (std::thread& thread : init_threads) {
        thread.join();
    }
    for (int rank = 0; rank < 2; ++rank) {
        require_result(init_results[static_cast<std::size_t>(rank)], ncclSuccess, "async-error init failed");
    }

    std::array<float, 4> send0 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::array<float, 4> send1 = {10.0f, 20.0f, 30.0f, 40.0f};
    std::array<float, 4> recv0 = {0.0f, 0.0f, 0.0f, 0.0f};
    std::array<float, 4> recv1 = {0.0f, 0.0f, 0.0f, 0.0f};
    std::array<ncclResult_t, 2> mismatch_results = {ncclInternalError, ncclInternalError};

    std::thread rank0([&]() {
        mismatch_results[0] =
            ncclAllReduce(send0.data(), recv0.data(), send0.size(), ncclFloat32, ncclSum, comms[0], nullptr);
    });
    std::thread rank1([&]() {
        mismatch_results[1] =
            ncclAllReduce(send1.data(), recv1.data(), send1.size(), ncclFloat32, ncclProd, comms[1], nullptr);
    });
    rank0.join();
    rank1.join();

    require_result(mismatch_results[0], ncclInvalidUsage, "rank0 mismatch should fail");
    require_result(mismatch_results[1], ncclInvalidUsage, "rank1 mismatch should fail");

    ncclResult_t async_error0 = ncclSuccess;
    ncclResult_t async_error1 = ncclSuccess;
    require_result(ncclCommGetAsyncError(comms[0], &async_error0), ncclSuccess, "rank0 get async error failed");
    require_result(ncclCommGetAsyncError(comms[1], &async_error1), ncclSuccess, "rank1 get async error failed");
    require(async_error0 == ncclInvalidUsage, "rank0 async error should persist as ncclInvalidUsage");
    require(async_error1 == ncclInvalidUsage, "rank1 async error should persist as ncclInvalidUsage");

    std::array<float, 1> retry_send0 = {2.0f};
    std::array<float, 1> retry_send1 = {3.0f};
    std::array<float, 1> retry_recv0 = {0.0f};
    std::array<float, 1> retry_recv1 = {0.0f};
    std::array<ncclResult_t, 2> retry_results = {ncclInternalError, ncclInternalError};

    const auto begin = std::chrono::steady_clock::now();
    std::thread retry_rank0([&]() {
        retry_results[0] = ncclAllReduce(
            retry_send0.data(), retry_recv0.data(), retry_send0.size(), ncclFloat32, ncclSum, comms[0], nullptr);
    });
    std::thread retry_rank1([&]() {
        retry_results[1] = ncclAllReduce(
            retry_send1.data(), retry_recv1.data(), retry_send1.size(), ncclFloat32, ncclSum, comms[1], nullptr);
    });
    retry_rank0.join();
    retry_rank1.join();
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - begin).count();

    require_result(retry_results[0], ncclInvalidUsage, "rank0 retry should fail fast");
    require_result(retry_results[1], ncclInvalidUsage, "rank1 retry should fail fast");
    require(elapsed_ms < 500, "poisoned communicator retry should fail quickly");

    require_result(ncclCommDestroy(comms[0]), ncclSuccess, "destroy after async error failed");
    require_result(ncclCommDestroy(comms[1]), ncclSuccess, "destroy after async error failed");
}

void run_group_stream_mismatch_case() {
    ncclUniqueId unique_id {};
    require_result(ncclGetUniqueId(&unique_id), ncclSuccess, "ncclGetUniqueId failed");

    std::array<ncclComm_t, 2> comms = {nullptr, nullptr};
    std::array<ncclResult_t, 2> init_results = {ncclInternalError, ncclInternalError};
    std::vector<std::thread> init_threads;
    for (int rank = 0; rank < 2; ++rank) {
        init_threads.emplace_back([&, rank]() {
            init_results[static_cast<std::size_t>(rank)] =
                ncclCommInitRank(&comms[static_cast<std::size_t>(rank)], 2, unique_id, rank);
        });
    }
    for (std::thread& thread : init_threads) {
        thread.join();
    }
    for (int rank = 0; rank < 2; ++rank) {
        require_result(init_results[static_cast<std::size_t>(rank)], ncclSuccess, "stream-mismatch init failed");
    }

    CUstream stream_a = nullptr;
    CUstream stream_b = nullptr;
    require(cuStreamCreate(&stream_a, 0) == CUDA_SUCCESS, "stream-mismatch stream_a create failed");
    require(cuStreamCreate(&stream_b, CU_STREAM_NON_BLOCKING) == CUDA_SUCCESS, "stream-mismatch stream_b create failed");

    std::array<float, 1> send0 = {1.0f};
    std::array<float, 1> send1 = {2.0f};
    std::array<float, 1> recv0 = {0.0f};
    std::array<float, 1> recv1 = {0.0f};
    std::array<ncclResult_t, 2> results = {ncclInternalError, ncclInternalError};

    std::thread rank0([&]() {
        require_result(ncclGroupStart(), ncclSuccess, "rank0 stream group start failed");
        require_result(
            ncclAllReduce(
                send0.data(),
                recv0.data(),
                send0.size(),
                ncclFloat32,
                ncclSum,
                comms[0],
                reinterpret_cast<cudaStream_t>(stream_a)),
            ncclSuccess,
            "rank0 first grouped enqueue failed");
        results[0] = ncclBroadcast(
            send0.data(),
            recv0.data(),
            send0.size(),
            ncclFloat32,
            0,
            comms[0],
            reinterpret_cast<cudaStream_t>(stream_b));
        require_result(ncclGroupSimulateEnd(nullptr), ncclSuccess, "rank0 group cleanup failed");
    });
    std::thread rank1([&]() {
        require_result(ncclGroupStart(), ncclSuccess, "rank1 stream group start failed");
        require_result(
            ncclAllReduce(
                send1.data(),
                recv1.data(),
                send1.size(),
                ncclFloat32,
                ncclSum,
                comms[1],
                reinterpret_cast<cudaStream_t>(stream_a)),
            ncclSuccess,
            "rank1 first grouped enqueue failed");
        results[1] = ncclBroadcast(
            send1.data(),
            recv1.data(),
            send1.size(),
            ncclFloat32,
            0,
            comms[1],
            reinterpret_cast<cudaStream_t>(stream_b));
        require_result(ncclGroupSimulateEnd(nullptr), ncclSuccess, "rank1 group cleanup failed");
    });
    rank0.join();
    rank1.join();

    require_result(results[0], ncclInvalidUsage, "rank0 mismatched group stream should fail");
    require_result(results[1], ncclInvalidUsage, "rank1 mismatched group stream should fail");

    require(cuStreamDestroy(stream_a) == CUDA_SUCCESS, "stream-mismatch stream_a destroy failed");
    require(cuStreamDestroy(stream_b) == CUDA_SUCCESS, "stream-mismatch stream_b destroy failed");

    for (ncclComm_t comm : comms) {
        require_result(ncclCommDestroy(comm), ncclSuccess, "destroy after stream mismatch failed");
    }
}

void run_fake_stream_identity_case() {
    CUstream stream0 = nullptr;
    CUstream stream1 = nullptr;
    require(cuStreamCreate(&stream0, 0) == CUDA_SUCCESS, "first fake cuStreamCreate failed");
    require(cuStreamCreate(&stream1, 0) == CUDA_SUCCESS, "second fake cuStreamCreate failed");
    require(stream0 != nullptr, "first fake stream should not be null");
    require(stream1 != nullptr, "second fake stream should not be null");
    require(stream0 != stream1, "simulate mode should return distinct fake stream handles");
    require(cuStreamDestroy(stream0) == CUDA_SUCCESS, "first fake cuStreamDestroy failed");
    require(cuStreamDestroy(stream1) == CUDA_SUCCESS, "second fake cuStreamDestroy failed");
}

void run_fake_stream_registry_case() {
    CUstream stream = nullptr;
    require(
        cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) == CUDA_SUCCESS,
        "fake cuStreamCreateWithFlags path failed");
    require(stream != nullptr, "registered fake stream should not be null");
    require(cuStreamQuery(stream) == CUDA_SUCCESS, "registered fake stream should query successfully");

    unsigned int flags = 0;
    require(cuStreamGetFlags(stream, &flags) == CUDA_SUCCESS, "registered fake stream flags should be readable");
    require(flags == CU_STREAM_NON_BLOCKING, "registered fake stream flags mismatch");

    unsigned long long stream_id = 0;
    require(cuStreamGetId(stream, &stream_id) == CUDA_SUCCESS, "registered fake stream id should be readable");
    require(stream_id != 0, "registered fake stream id should be non-zero");

    require(cuStreamGetFlags(nullptr, &flags) == CUDA_SUCCESS, "default stream flags should be readable");
    require(flags == CU_STREAM_DEFAULT, "default stream flags should be zero");

    require(cuStreamDestroy(stream) == CUDA_SUCCESS, "registered fake stream destroy failed");
    require(cuStreamQuery(stream) == CUDA_ERROR_INVALID_VALUE, "destroyed fake stream should become invalid");
    require(cuStreamGetFlags(stream, &flags) == CUDA_ERROR_INVALID_VALUE, "destroyed fake stream flags should fail");
}

void run_invalid_stream_collective_case() {
    ncclUniqueId unique_id {};
    require_result(ncclGetUniqueId(&unique_id), ncclSuccess, "invalid-stream ncclGetUniqueId failed");

    std::array<ncclComm_t, 2> comms = {nullptr, nullptr};
    std::array<ncclResult_t, 2> init_results = {ncclInternalError, ncclInternalError};
    std::vector<std::thread> init_threads;
    for (int rank = 0; rank < 2; ++rank) {
        init_threads.emplace_back([&, rank]() {
            init_results[static_cast<std::size_t>(rank)] =
                ncclCommInitRank(&comms[static_cast<std::size_t>(rank)], 2, unique_id, rank);
        });
    }
    for (std::thread& thread : init_threads) {
        thread.join();
    }
    for (int rank = 0; rank < 2; ++rank) {
        require_result(init_results[static_cast<std::size_t>(rank)], ncclSuccess, "invalid-stream init failed");
    }

    CUstream invalid_stream = nullptr;
    require(cuStreamCreate(&invalid_stream, 0) == CUDA_SUCCESS, "invalid-stream create failed");
    require(cuStreamDestroy(invalid_stream) == CUDA_SUCCESS, "invalid-stream destroy failed");

    std::array<float, 1> send0 = {1.0f};
    std::array<float, 1> send1 = {2.0f};
    std::array<float, 1> recv0 = {0.0f};
    std::array<float, 1> recv1 = {0.0f};
    std::array<ncclResult_t, 2> results = {ncclInternalError, ncclInternalError};

    std::thread rank0([&]() {
        results[0] = ncclAllReduce(
            send0.data(),
            recv0.data(),
            send0.size(),
            ncclFloat32,
            ncclSum,
            comms[0],
            reinterpret_cast<cudaStream_t>(invalid_stream));
    });
    std::thread rank1([&]() {
        results[1] = ncclAllReduce(
            send1.data(),
            recv1.data(),
            send1.size(),
            ncclFloat32,
            ncclSum,
            comms[1],
            reinterpret_cast<cudaStream_t>(invalid_stream));
    });
    rank0.join();
    rank1.join();

    require_result(results[0], ncclInvalidArgument, "rank0 invalid stream should fail");
    require_result(results[1], ncclInvalidArgument, "rank1 invalid stream should fail");

    for (ncclComm_t comm : comms) {
        require_result(ncclCommDestroy(comm), ncclSuccess, "destroy after invalid stream failed");
    }
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
        run_comm_init_all_case();
        run_comm_init_all_invalid_case();
        run_comm_split_case();
        run_send_recv_case();
        run_send_recv_multi_pair_case();
        run_grouped_send_recv_case();
        run_grouped_send_recv_then_collective_case();
        run_send_recv_timeout_case();
        run_premul_redop_api_case();
        run_async_error_persistence_case();
        run_fake_stream_identity_case();
        run_fake_stream_registry_case();
        run_invalid_stream_collective_case();
        run_group_stream_mismatch_case();

        std::cout << "nccl direct init/destroy test passed" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
