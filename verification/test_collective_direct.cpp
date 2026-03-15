#include "../src/distributed/cluster_coordinator.hpp"
#include "../src/monitor/monitor.hpp"
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
    std::string pattern = "/tmp/fakegpu-collective-XXXXXX";
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

template <typename T>
std::vector<T> make_rank_values(int rank, std::size_t count) {
    std::vector<T> values(count);
    for (std::size_t index = 0; index < count; ++index) {
        values[index] = static_cast<T>((rank + 1) * 10 + static_cast<int>(index));
    }
    return values;
}

template <>
std::vector<float> make_rank_values<float>(int rank, std::size_t count) {
    std::vector<float> values(count);
    for (std::size_t index = 0; index < count; ++index) {
        values[index] = static_cast<float>((rank + 1) * 1.25 + static_cast<double>(index) * 0.5);
    }
    return values;
}

template <typename T>
std::vector<T> make_allreduce_reference(int world_size, std::size_t count) {
    std::vector<T> reference(count, static_cast<T>(0));
    for (int rank = 0; rank < world_size; ++rank) {
        std::vector<T> values = make_rank_values<T>(rank, count);
        for (std::size_t index = 0; index < count; ++index) {
            reference[index] += values[index];
        }
    }
    return reference;
}

template <typename T>
std::vector<T> scale_reference(const std::vector<T>& values, T scalar) {
    std::vector<T> scaled = values;
    for (T& value : scaled) {
        value = static_cast<T>(value * scalar);
    }
    return scaled;
}

template <typename T>
std::vector<T> make_reduce_untouched_reference(std::size_t count) {
    return std::vector<T>(count, static_cast<T>(-777));
}

template <typename T>
std::vector<T> make_allgather_reference(int world_size, std::size_t count) {
    std::vector<T> reference;
    reference.reserve(static_cast<std::size_t>(world_size) * count);
    for (int rank = 0; rank < world_size; ++rank) {
        std::vector<T> values = make_rank_values<T>(rank, count);
        reference.insert(reference.end(), values.begin(), values.end());
    }
    return reference;
}

template <typename T>
std::vector<T> make_reducescatter_reference(int world_size, std::size_t recvcount, int rank) {
    const std::size_t total_count = recvcount * static_cast<std::size_t>(world_size);
    std::vector<T> reduced(total_count, static_cast<T>(0));
    for (int peer = 0; peer < world_size; ++peer) {
        std::vector<T> values = make_rank_values<T>(peer, total_count);
        for (std::size_t index = 0; index < total_count; ++index) {
            reduced[index] += values[index];
        }
    }

    const std::size_t offset = static_cast<std::size_t>(rank) * recvcount;
    return std::vector<T>(reduced.begin() + static_cast<std::ptrdiff_t>(offset),
                          reduced.begin() + static_cast<std::ptrdiff_t>(offset + recvcount));
}

template <typename T>
void assert_equal_vectors(
    const std::vector<T>& actual,
    const std::vector<T>& expected,
    const std::string& message) {
    require(actual.size() == expected.size(), message + ": size mismatch");
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (actual[index] != expected[index]) {
            throw std::runtime_error(
                message + ": mismatch at index " + std::to_string(index));
        }
    }
}

template <>
void assert_equal_vectors<float>(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    const std::string& message) {
    require(actual.size() == expected.size(), message + ": size mismatch");
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (std::fabs(actual[index] - expected[index]) > 1e-6f) {
            throw std::runtime_error(
                message + ": mismatch at index " + std::to_string(index));
        }
    }
}

template <typename T>
void run_allreduce_case(
    int world_size,
    ncclDataType_t datatype,
    const std::string& label,
    std::size_t count = 8) {
    std::vector<ncclComm_t> comms = init_communicators(world_size);
    const std::vector<T> reference = make_allreduce_reference<T>(world_size, count);

    for (int iteration = 0; iteration < 3; ++iteration) {
        const bool grouped = iteration == 2;
        std::vector<std::vector<T>> send_buffers;
        std::vector<std::vector<T>> recv_buffers(static_cast<std::size_t>(world_size), std::vector<T>(count, {}));
        std::vector<ncclResult_t> results(static_cast<std::size_t>(world_size), ncclInternalError);
        std::vector<std::thread> threads;

        send_buffers.reserve(static_cast<std::size_t>(world_size));
        for (int rank = 0; rank < world_size; ++rank) {
            send_buffers.push_back(make_rank_values<T>(rank, count));
        }

        for (int rank = 0; rank < world_size; ++rank) {
            threads.emplace_back([&, rank]() {
                if (grouped) {
                    require_result(ncclGroupStart(), ncclSuccess, label + " ncclGroupStart failed");
                    ncclResult_t inner = ncclAllReduce(
                        send_buffers[static_cast<std::size_t>(rank)].data(),
                        recv_buffers[static_cast<std::size_t>(rank)].data(),
                        count,
                        datatype,
                        ncclSum,
                        comms[static_cast<std::size_t>(rank)],
                        nullptr);
                    if (inner != ncclSuccess) {
                        ncclGroupSimulateEnd(nullptr);
                        results[static_cast<std::size_t>(rank)] = inner;
                        return;
                    }
                    results[static_cast<std::size_t>(rank)] = ncclGroupEnd();
                    return;
                }
                results[static_cast<std::size_t>(rank)] = ncclAllReduce(
                    send_buffers[static_cast<std::size_t>(rank)].data(),
                    recv_buffers[static_cast<std::size_t>(rank)].data(),
                    count,
                    datatype,
                    ncclSum,
                    comms[static_cast<std::size_t>(rank)],
                    nullptr);
            });
        }

        for (std::thread& thread : threads) {
            thread.join();
        }

        for (int rank = 0; rank < world_size; ++rank) {
            require_result(results[static_cast<std::size_t>(rank)], ncclSuccess, label + " allreduce failed");
            assert_equal_vectors(
                recv_buffers[static_cast<std::size_t>(rank)],
                reference,
                label + " allreduce result mismatch");
        }
    }

    destroy_communicators(comms);
}

template <typename T>
void run_premul_allreduce_case(
    int world_size,
    ncclDataType_t datatype,
    T scalar,
    const std::string& label,
    std::size_t count = 8) {
    std::vector<ncclComm_t> comms = init_communicators(world_size);
    const std::vector<T> reference =
        scale_reference(make_allreduce_reference<T>(world_size, count), scalar);

    ncclRedOp_t op = ncclSum;
    require_result(
        ncclRedOpCreatePreMulSum(&op, &scalar, datatype, ncclScalarHostImmediate, nullptr),
        ncclSuccess,
        label + " ncclRedOpCreatePreMulSum failed");

    for (int iteration = 0; iteration < 3; ++iteration) {
        const bool grouped = iteration == 2;
        std::vector<std::vector<T>> send_buffers;
        std::vector<std::vector<T>> recv_buffers(static_cast<std::size_t>(world_size), std::vector<T>(count, {}));
        std::vector<ncclResult_t> results(static_cast<std::size_t>(world_size), ncclInternalError);
        std::vector<std::thread> threads;

        send_buffers.reserve(static_cast<std::size_t>(world_size));
        for (int rank = 0; rank < world_size; ++rank) {
            send_buffers.push_back(make_rank_values<T>(rank, count));
        }

        for (int rank = 0; rank < world_size; ++rank) {
            threads.emplace_back([&, rank]() {
                if (grouped) {
                    require_result(ncclGroupStart(), ncclSuccess, label + " ncclGroupStart failed");
                    ncclResult_t inner = ncclAllReduce(
                        send_buffers[static_cast<std::size_t>(rank)].data(),
                        recv_buffers[static_cast<std::size_t>(rank)].data(),
                        count,
                        datatype,
                        op,
                        comms[static_cast<std::size_t>(rank)],
                        nullptr);
                    if (inner != ncclSuccess) {
                        ncclGroupSimulateEnd(nullptr);
                        results[static_cast<std::size_t>(rank)] = inner;
                        return;
                    }
                    results[static_cast<std::size_t>(rank)] = ncclGroupEnd();
                    return;
                }
                results[static_cast<std::size_t>(rank)] = ncclAllReduce(
                    send_buffers[static_cast<std::size_t>(rank)].data(),
                    recv_buffers[static_cast<std::size_t>(rank)].data(),
                    count,
                    datatype,
                    op,
                    comms[static_cast<std::size_t>(rank)],
                    nullptr);
            });
        }

        for (std::thread& thread : threads) {
            thread.join();
        }

        for (int rank = 0; rank < world_size; ++rank) {
            require_result(results[static_cast<std::size_t>(rank)], ncclSuccess, label + " preMul allreduce failed");
            assert_equal_vectors(
                recv_buffers[static_cast<std::size_t>(rank)],
                reference,
                label + " preMul allreduce result mismatch");
        }
    }

    require_result(ncclRedOpDestroy(op, nullptr), ncclSuccess, label + " ncclRedOpDestroy failed");
    destroy_communicators(comms);
}

template <typename T>
void run_premul_reduce_case(
    int world_size,
    int root,
    ncclDataType_t datatype,
    T scalar,
    const std::string& label,
    std::size_t count = 7) {
    std::vector<ncclComm_t> comms = init_communicators(world_size);
    const std::vector<T> reference =
        scale_reference(make_allreduce_reference<T>(world_size, count), scalar);
    const std::vector<T> untouched = make_reduce_untouched_reference<T>(count);

    ncclRedOp_t op = ncclSum;
    require_result(
        ncclRedOpCreatePreMulSum(&op, &scalar, datatype, ncclScalarHostImmediate, nullptr),
        ncclSuccess,
        label + " ncclRedOpCreatePreMulSum failed");

    for (int iteration = 0; iteration < 3; ++iteration) {
        const bool grouped = iteration == 2;
        const bool null_non_root_recv = iteration == 1;
        std::vector<std::vector<T>> send_buffers;
        std::vector<std::vector<T>> recv_buffers(
            static_cast<std::size_t>(world_size),
            untouched);
        std::vector<ncclResult_t> results(static_cast<std::size_t>(world_size), ncclInternalError);
        std::vector<std::thread> threads;

        send_buffers.reserve(static_cast<std::size_t>(world_size));
        for (int rank = 0; rank < world_size; ++rank) {
            send_buffers.push_back(make_rank_values<T>(rank, count));
        }

        for (int rank = 0; rank < world_size; ++rank) {
            threads.emplace_back([&, rank]() {
                T* recv_ptr = recv_buffers[static_cast<std::size_t>(rank)].data();
                if (rank != root && null_non_root_recv) {
                    recv_ptr = nullptr;
                }

                if (grouped) {
                    require_result(ncclGroupStart(), ncclSuccess, label + " ncclGroupStart failed");
                    ncclResult_t inner = ncclReduce(
                        send_buffers[static_cast<std::size_t>(rank)].data(),
                        recv_ptr,
                        count,
                        datatype,
                        op,
                        root,
                        comms[static_cast<std::size_t>(rank)],
                        nullptr);
                    if (inner != ncclSuccess) {
                        ncclGroupSimulateEnd(nullptr);
                        results[static_cast<std::size_t>(rank)] = inner;
                        return;
                    }
                    results[static_cast<std::size_t>(rank)] = ncclGroupEnd();
                    return;
                }

                results[static_cast<std::size_t>(rank)] = ncclReduce(
                    send_buffers[static_cast<std::size_t>(rank)].data(),
                    recv_ptr,
                    count,
                    datatype,
                    op,
                    root,
                    comms[static_cast<std::size_t>(rank)],
                    nullptr);
            });
        }

        for (std::thread& thread : threads) {
            thread.join();
        }

        for (int rank = 0; rank < world_size; ++rank) {
            require_result(results[static_cast<std::size_t>(rank)], ncclSuccess, label + " preMul reduce failed");
            if (rank == root) {
                assert_equal_vectors(
                    recv_buffers[static_cast<std::size_t>(rank)],
                    reference,
                    label + " preMul reduce root result mismatch");
            } else {
                assert_equal_vectors(
                    recv_buffers[static_cast<std::size_t>(rank)],
                    untouched,
                    label + " preMul reduce non-root recv buffer should stay untouched");
            }
        }
    }

    require_result(ncclRedOpDestroy(op, nullptr), ncclSuccess, label + " ncclRedOpDestroy failed");
    destroy_communicators(comms);
}

template <typename T>
void run_premul_reducescatter_case(
    int world_size,
    ncclDataType_t datatype,
    T scalar,
    const std::string& label,
    std::size_t recvcount = 4) {
    std::vector<ncclComm_t> comms = init_communicators(world_size);
    const std::size_t total_count = recvcount * static_cast<std::size_t>(world_size);

    ncclRedOp_t op = ncclSum;
    require_result(
        ncclRedOpCreatePreMulSum(&op, &scalar, datatype, ncclScalarHostImmediate, nullptr),
        ncclSuccess,
        label + " ncclRedOpCreatePreMulSum failed");

    for (int iteration = 0; iteration < 3; ++iteration) {
        const bool grouped = iteration == 2;
        std::vector<std::vector<T>> send_buffers;
        std::vector<std::vector<T>> recv_buffers(
            static_cast<std::size_t>(world_size),
            std::vector<T>(recvcount, static_cast<T>(0)));
        std::vector<ncclResult_t> results(static_cast<std::size_t>(world_size), ncclInternalError);
        std::vector<std::thread> threads;

        send_buffers.reserve(static_cast<std::size_t>(world_size));
        for (int rank = 0; rank < world_size; ++rank) {
            send_buffers.push_back(make_rank_values<T>(rank, total_count));
        }

        for (int rank = 0; rank < world_size; ++rank) {
            threads.emplace_back([&, rank]() {
                if (grouped) {
                    require_result(ncclGroupStart(), ncclSuccess, label + " ncclGroupStart failed");
                    ncclResult_t inner = ncclReduceScatter(
                        send_buffers[static_cast<std::size_t>(rank)].data(),
                        recv_buffers[static_cast<std::size_t>(rank)].data(),
                        recvcount,
                        datatype,
                        op,
                        comms[static_cast<std::size_t>(rank)],
                        nullptr);
                    if (inner != ncclSuccess) {
                        ncclGroupSimulateEnd(nullptr);
                        results[static_cast<std::size_t>(rank)] = inner;
                        return;
                    }
                    results[static_cast<std::size_t>(rank)] = ncclGroupEnd();
                    return;
                }

                results[static_cast<std::size_t>(rank)] = ncclReduceScatter(
                    send_buffers[static_cast<std::size_t>(rank)].data(),
                    recv_buffers[static_cast<std::size_t>(rank)].data(),
                    recvcount,
                    datatype,
                    op,
                    comms[static_cast<std::size_t>(rank)],
                    nullptr);
            });
        }

        for (std::thread& thread : threads) {
            thread.join();
        }

        for (int rank = 0; rank < world_size; ++rank) {
            require_result(
                results[static_cast<std::size_t>(rank)],
                ncclSuccess,
                label + " preMul reducescatter failed");
            assert_equal_vectors(
                recv_buffers[static_cast<std::size_t>(rank)],
                scale_reference(make_reducescatter_reference<T>(world_size, recvcount, rank), scalar),
                label + " preMul reducescatter result mismatch");
        }
    }

    require_result(ncclRedOpDestroy(op, nullptr), ncclSuccess, label + " ncclRedOpDestroy failed");
    destroy_communicators(comms);
}

template <typename T>
void run_allgather_case(
    int world_size,
    ncclDataType_t datatype,
    const std::string& label,
    std::size_t sendcount = 5) {
    std::vector<ncclComm_t> comms = init_communicators(world_size);
    const std::vector<T> reference = make_allgather_reference<T>(world_size, sendcount);

    for (int iteration = 0; iteration < 3; ++iteration) {
        const bool grouped = iteration == 2;
        std::vector<std::vector<T>> send_buffers;
        std::vector<std::vector<T>> recv_buffers(
            static_cast<std::size_t>(world_size),
            std::vector<T>(reference.size(), static_cast<T>(0)));
        std::vector<ncclResult_t> results(static_cast<std::size_t>(world_size), ncclInternalError);
        std::vector<std::thread> threads;

        send_buffers.reserve(static_cast<std::size_t>(world_size));
        for (int rank = 0; rank < world_size; ++rank) {
            send_buffers.push_back(make_rank_values<T>(rank, sendcount));
        }

        for (int rank = 0; rank < world_size; ++rank) {
            threads.emplace_back([&, rank]() {
                if (grouped) {
                    require_result(ncclGroupStart(), ncclSuccess, label + " ncclGroupStart failed");
                    ncclResult_t inner = ncclAllGather(
                        send_buffers[static_cast<std::size_t>(rank)].data(),
                        recv_buffers[static_cast<std::size_t>(rank)].data(),
                        sendcount,
                        datatype,
                        comms[static_cast<std::size_t>(rank)],
                        nullptr);
                    if (inner != ncclSuccess) {
                        ncclGroupSimulateEnd(nullptr);
                        results[static_cast<std::size_t>(rank)] = inner;
                        return;
                    }
                    results[static_cast<std::size_t>(rank)] = ncclGroupEnd();
                    return;
                }

                results[static_cast<std::size_t>(rank)] = ncclAllGather(
                    send_buffers[static_cast<std::size_t>(rank)].data(),
                    recv_buffers[static_cast<std::size_t>(rank)].data(),
                    sendcount,
                    datatype,
                    comms[static_cast<std::size_t>(rank)],
                    nullptr);
            });
        }

        for (std::thread& thread : threads) {
            thread.join();
        }

        for (int rank = 0; rank < world_size; ++rank) {
            require_result(results[static_cast<std::size_t>(rank)], ncclSuccess, label + " allgather failed");
            assert_equal_vectors(
                recv_buffers[static_cast<std::size_t>(rank)],
                reference,
                label + " allgather result mismatch");
        }
    }

    destroy_communicators(comms);
}

template <typename T>
void run_reduce_case(
    int world_size,
    int root,
    ncclDataType_t datatype,
    const std::string& label,
    std::size_t count = 7) {
    std::vector<ncclComm_t> comms = init_communicators(world_size);
    const std::vector<T> reference = make_allreduce_reference<T>(world_size, count);
    const std::vector<T> untouched = make_reduce_untouched_reference<T>(count);

    for (int iteration = 0; iteration < 3; ++iteration) {
        const bool grouped = iteration == 2;
        const bool null_non_root_recv = iteration == 1;
        std::vector<std::vector<T>> send_buffers;
        std::vector<std::vector<T>> recv_buffers(
            static_cast<std::size_t>(world_size),
            untouched);
        std::vector<ncclResult_t> results(static_cast<std::size_t>(world_size), ncclInternalError);
        std::vector<std::thread> threads;

        send_buffers.reserve(static_cast<std::size_t>(world_size));
        for (int rank = 0; rank < world_size; ++rank) {
            send_buffers.push_back(make_rank_values<T>(rank, count));
        }

        for (int rank = 0; rank < world_size; ++rank) {
            threads.emplace_back([&, rank]() {
                T* recv_ptr = recv_buffers[static_cast<std::size_t>(rank)].data();
                if (rank != root && null_non_root_recv) {
                    recv_ptr = nullptr;
                }

                if (grouped) {
                    require_result(ncclGroupStart(), ncclSuccess, label + " ncclGroupStart failed");
                    ncclResult_t inner = ncclReduce(
                        send_buffers[static_cast<std::size_t>(rank)].data(),
                        recv_ptr,
                        count,
                        datatype,
                        ncclSum,
                        root,
                        comms[static_cast<std::size_t>(rank)],
                        nullptr);
                    if (inner != ncclSuccess) {
                        ncclGroupSimulateEnd(nullptr);
                        results[static_cast<std::size_t>(rank)] = inner;
                        return;
                    }
                    results[static_cast<std::size_t>(rank)] = ncclGroupEnd();
                    return;
                }

                results[static_cast<std::size_t>(rank)] = ncclReduce(
                    send_buffers[static_cast<std::size_t>(rank)].data(),
                    recv_ptr,
                    count,
                    datatype,
                    ncclSum,
                    root,
                    comms[static_cast<std::size_t>(rank)],
                    nullptr);
            });
        }

        for (std::thread& thread : threads) {
            thread.join();
        }

        for (int rank = 0; rank < world_size; ++rank) {
            require_result(results[static_cast<std::size_t>(rank)], ncclSuccess, label + " reduce failed");
            if (rank == root) {
                assert_equal_vectors(
                    recv_buffers[static_cast<std::size_t>(rank)],
                    reference,
                    label + " reduce root result mismatch");
            } else {
                assert_equal_vectors(
                    recv_buffers[static_cast<std::size_t>(rank)],
                    untouched,
                    label + " reduce non-root recv buffer should stay untouched");
            }
        }
    }

    destroy_communicators(comms);
}

template <typename T>
void run_reducescatter_case(
    int world_size,
    ncclDataType_t datatype,
    const std::string& label,
    std::size_t recvcount = 4) {
    std::vector<ncclComm_t> comms = init_communicators(world_size);
    const std::size_t total_count = recvcount * static_cast<std::size_t>(world_size);

    for (int iteration = 0; iteration < 3; ++iteration) {
        const bool grouped = iteration == 2;
        std::vector<std::vector<T>> send_buffers;
        std::vector<std::vector<T>> recv_buffers(
            static_cast<std::size_t>(world_size),
            std::vector<T>(recvcount, static_cast<T>(0)));
        std::vector<ncclResult_t> results(static_cast<std::size_t>(world_size), ncclInternalError);
        std::vector<std::thread> threads;

        send_buffers.reserve(static_cast<std::size_t>(world_size));
        for (int rank = 0; rank < world_size; ++rank) {
            send_buffers.push_back(make_rank_values<T>(rank, total_count));
        }

        for (int rank = 0; rank < world_size; ++rank) {
            threads.emplace_back([&, rank]() {
                if (grouped) {
                    require_result(ncclGroupStart(), ncclSuccess, label + " ncclGroupStart failed");
                    ncclResult_t inner = ncclReduceScatter(
                        send_buffers[static_cast<std::size_t>(rank)].data(),
                        recv_buffers[static_cast<std::size_t>(rank)].data(),
                        recvcount,
                        datatype,
                        ncclSum,
                        comms[static_cast<std::size_t>(rank)],
                        nullptr);
                    if (inner != ncclSuccess) {
                        ncclGroupSimulateEnd(nullptr);
                        results[static_cast<std::size_t>(rank)] = inner;
                        return;
                    }
                    results[static_cast<std::size_t>(rank)] = ncclGroupEnd();
                    return;
                }

                results[static_cast<std::size_t>(rank)] = ncclReduceScatter(
                    send_buffers[static_cast<std::size_t>(rank)].data(),
                    recv_buffers[static_cast<std::size_t>(rank)].data(),
                    recvcount,
                    datatype,
                    ncclSum,
                    comms[static_cast<std::size_t>(rank)],
                    nullptr);
            });
        }

        for (std::thread& thread : threads) {
            thread.join();
        }

        for (int rank = 0; rank < world_size; ++rank) {
            require_result(
                results[static_cast<std::size_t>(rank)],
                ncclSuccess,
                label + " reducescatter failed");
            assert_equal_vectors(
                recv_buffers[static_cast<std::size_t>(rank)],
                make_reducescatter_reference<T>(world_size, recvcount, rank),
                label + " reducescatter result mismatch");
        }
    }

    destroy_communicators(comms);
}

template <typename T>
void run_broadcast_case(
    int world_size,
    int root,
    ncclDataType_t datatype,
    const std::string& label,
    std::size_t count = 6) {
    std::vector<ncclComm_t> comms = init_communicators(world_size);
    const std::vector<T> root_values = make_rank_values<T>(root, count);

    for (int iteration = 0; iteration < 3; ++iteration) {
        const bool grouped = iteration == 2;
        std::vector<std::vector<T>> recv_buffers(static_cast<std::size_t>(world_size), std::vector<T>(count, {}));
        std::vector<ncclResult_t> results(static_cast<std::size_t>(world_size), ncclInternalError);
        std::vector<std::thread> threads;

        for (int rank = 0; rank < world_size; ++rank) {
            threads.emplace_back([&, rank]() {
                const T* send_ptr = nullptr;
                if (rank == root) {
                    send_ptr = root_values.data();
                }
                if (grouped) {
                    require_result(ncclGroupStart(), ncclSuccess, label + " ncclGroupStart failed");
                    ncclResult_t inner = ncclBroadcast(
                        send_ptr,
                        recv_buffers[static_cast<std::size_t>(rank)].data(),
                        count,
                        datatype,
                        root,
                        comms[static_cast<std::size_t>(rank)],
                        nullptr);
                    if (inner != ncclSuccess) {
                        ncclGroupSimulateEnd(nullptr);
                        results[static_cast<std::size_t>(rank)] = inner;
                        return;
                    }
                    results[static_cast<std::size_t>(rank)] = ncclGroupEnd();
                    return;
                }
                results[static_cast<std::size_t>(rank)] = ncclBroadcast(
                    send_ptr,
                    recv_buffers[static_cast<std::size_t>(rank)].data(),
                    count,
                    datatype,
                    root,
                    comms[static_cast<std::size_t>(rank)],
                    nullptr);
            });
        }

        for (std::thread& thread : threads) {
            thread.join();
        }

        for (int rank = 0; rank < world_size; ++rank) {
            require_result(results[static_cast<std::size_t>(rank)], ncclSuccess, label + " broadcast failed");
            assert_equal_vectors(
                recv_buffers[static_cast<std::size_t>(rank)],
                root_values,
                label + " broadcast result mismatch");
        }
    }

    destroy_communicators(comms);
}

void run_type_mismatch_case() {
    std::vector<ncclComm_t> comms = init_communicators(2);

    std::vector<float> send0 = make_rank_values<float>(0, 4);
    std::vector<float> recv0(4, 0.0f);
    std::vector<float> send1 = make_rank_values<float>(1, 4);
    std::vector<float> recv1(4, 0.0f);
    ncclResult_t result0 = ncclInternalError;
    ncclResult_t result1 = ncclInternalError;

    std::thread thread0([&]() {
        result0 = ncclAllReduce(send0.data(), recv0.data(), send0.size(), ncclFloat32, ncclSum, comms[0], nullptr);
    });
    std::thread thread1([&]() {
        result1 = ncclBroadcast(send1.data(), recv1.data(), send1.size(), ncclFloat32, 0, comms[1], nullptr);
    });
    thread0.join();
    thread1.join();

    require_result(result0, ncclInvalidUsage, "collective type mismatch should fail");
    require_result(result1, ncclInvalidUsage, "collective type mismatch should fail");
    destroy_communicators(comms);
}

void run_dtype_mismatch_case() {
    std::vector<ncclComm_t> comms = init_communicators(2);

    std::vector<float> send0 = make_rank_values<float>(0, 4);
    std::vector<float> recv0(4, 0.0f);
    std::vector<std::int32_t> send1 = make_rank_values<std::int32_t>(1, 4);
    std::vector<std::int32_t> recv1(4, 0);
    ncclResult_t result0 = ncclInternalError;
    ncclResult_t result1 = ncclInternalError;

    std::thread thread0([&]() {
        result0 = ncclAllReduce(send0.data(), recv0.data(), send0.size(), ncclFloat32, ncclSum, comms[0], nullptr);
    });
    std::thread thread1([&]() {
        result1 = ncclAllReduce(send1.data(), recv1.data(), send1.size(), ncclInt32, ncclSum, comms[1], nullptr);
    });
    thread0.join();
    thread1.join();

    require_result(result0, ncclInvalidUsage, "dtype mismatch should fail");
    require_result(result1, ncclInvalidUsage, "dtype mismatch should fail");
    destroy_communicators(comms);
}

void run_count_mismatch_case() {
    std::vector<ncclComm_t> comms = init_communicators(2);

    std::vector<std::int32_t> send0 = make_rank_values<std::int32_t>(0, 4);
    std::vector<std::int32_t> recv0(4, 0);
    std::vector<std::int32_t> send1 = make_rank_values<std::int32_t>(1, 5);
    std::vector<std::int32_t> recv1(5, 0);
    ncclResult_t result0 = ncclInternalError;
    ncclResult_t result1 = ncclInternalError;

    std::thread thread0([&]() {
        result0 = ncclAllReduce(send0.data(), recv0.data(), send0.size(), ncclInt32, ncclSum, comms[0], nullptr);
    });
    std::thread thread1([&]() {
        result1 = ncclAllReduce(send1.data(), recv1.data(), send1.size(), ncclInt32, ncclSum, comms[1], nullptr);
    });
    thread0.join();
    thread1.join();

    require_result(result0, ncclInvalidUsage, "count mismatch should fail");
    require_result(result1, ncclInvalidUsage, "count mismatch should fail");
    destroy_communicators(comms);
}

void run_root_mismatch_case() {
    std::vector<ncclComm_t> comms = init_communicators(2);

    std::vector<std::int32_t> send0 = make_rank_values<std::int32_t>(0, 4);
    std::vector<std::int32_t> recv0(4, 0);
    std::vector<std::int32_t> send1 = make_rank_values<std::int32_t>(1, 4);
    std::vector<std::int32_t> recv1(4, 0);
    ncclResult_t result0 = ncclInternalError;
    ncclResult_t result1 = ncclInternalError;

    std::thread thread0([&]() {
        result0 = ncclBroadcast(send0.data(), recv0.data(), send0.size(), ncclInt32, 0, comms[0], nullptr);
    });
    std::thread thread1([&]() {
        result1 = ncclBroadcast(send1.data(), recv1.data(), send1.size(), ncclInt32, 1, comms[1], nullptr);
    });
    thread0.join();
    thread1.join();

    require_result(result0, ncclInvalidUsage, "root mismatch should fail");
    require_result(result1, ncclInvalidUsage, "root mismatch should fail");
    destroy_communicators(comms);
}

void run_reduce_op_mismatch_case() {
    std::vector<ncclComm_t> comms = init_communicators(2);

    std::vector<float> send0 = make_rank_values<float>(0, 4);
    std::vector<float> recv0(4, 0.0f);
    std::vector<float> send1 = make_rank_values<float>(1, 4);
    std::vector<float> recv1(4, 0.0f);
    ncclResult_t result0 = ncclInternalError;
    ncclResult_t result1 = ncclInternalError;

    std::thread thread0([&]() {
        result0 = ncclAllReduce(send0.data(), recv0.data(), send0.size(), ncclFloat32, ncclSum, comms[0], nullptr);
    });
    std::thread thread1([&]() {
        result1 = ncclAllReduce(send1.data(), recv1.data(), send1.size(), ncclFloat32, ncclProd, comms[1], nullptr);
    });
    thread0.join();
    thread1.join();

    require_result(result0, ncclInvalidUsage, "reduce op mismatch should fail");
    require_result(result1, ncclInvalidUsage, "reduce op mismatch should fail");
    destroy_communicators(comms);
}

void run_timeout_case() {
    std::vector<ncclComm_t> comms = init_communicators(2);

    std::vector<float> send0 = make_rank_values<float>(0, 4);
    std::vector<float> recv0(4, 0.0f);
    ncclResult_t result0 = ncclInternalError;

    const auto start = std::chrono::steady_clock::now();
    std::thread thread0([&]() {
        result0 = ncclAllReduce(send0.data(), recv0.data(), send0.size(), ncclFloat32, ncclSum, comms[0], nullptr);
    });
    thread0.join();
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start)
            .count();

    require_result(result0, ncclSystemError, "missing rank should time out");
    require(elapsed_ms < 5000, "collective timeout exceeded 5 seconds");
    destroy_communicators(comms);
}

void run_allreduce_scenario() {
    run_allreduce_case<float>(2, ncclFloat32, "2-rank float32");
    run_allreduce_case<float>(4, ncclFloat32, "4-rank float32");
    run_allreduce_case<std::int32_t>(2, ncclInt32, "2-rank int32");
    run_allreduce_case<std::int32_t>(4, ncclInt32, "4-rank int32");
    run_premul_allreduce_case<float>(2, ncclFloat32, 2.0f, "2-rank premul float32");
}

void run_broadcast_scenario() {
    run_broadcast_case<float>(2, 0, ncclFloat32, "2-rank root0 float32");
    run_broadcast_case<float>(4, 0, ncclFloat32, "4-rank root0 float32");
    run_broadcast_case<std::int32_t>(4, 3, ncclInt32, "4-rank root-last int32");
}

void run_reduce_scenario() {
    run_reduce_case<float>(2, 0, ncclFloat32, "2-rank root0 float32");
    run_reduce_case<float>(4, 2, ncclFloat32, "4-rank root2 float32");
    run_reduce_case<std::int64_t>(4, 3, ncclInt64, "4-rank root-last int64");
    run_premul_reduce_case<float>(2, 1, ncclFloat32, 2.0f, "2-rank premul reduce");
}

void run_allgather_scenario() {
    run_allgather_case<float>(2, ncclFloat32, "2-rank float32");
    run_allgather_case<float>(4, ncclFloat32, "4-rank float32");
    run_allgather_case<std::int32_t>(4, ncclInt32, "4-rank int32");
}

void run_reducescatter_scenario() {
    run_reducescatter_case<float>(2, ncclFloat32, "2-rank float32");
    run_reducescatter_case<float>(4, ncclFloat32, "4-rank float32");
    run_reducescatter_case<std::int32_t>(4, ncclInt32, "4-rank int32");
    run_premul_reducescatter_case<float>(2, ncclFloat32, 2.0f, "2-rank premul float32");
}

void run_chunked_scenario() {
    run_allreduce_case<float>(4, ncclFloat32, "4-rank chunked float32", 65536);
    run_reduce_case<float>(4, 1, ncclFloat32, "4-rank chunked reduce", 65536);
    run_broadcast_case<float>(4, 3, ncclFloat32, "4-rank chunked broadcast", 65536);
    run_allgather_case<float>(4, ncclFloat32, "4-rank chunked allgather", 16384);
    run_reducescatter_case<float>(4, ncclFloat32, "4-rank chunked reducescatter", 16384);
}

void run_mismatch_scenario() {
    run_type_mismatch_case();
    run_dtype_mismatch_case();
    run_count_mismatch_case();
    run_root_mismatch_case();
    run_reduce_op_mismatch_case();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3 || std::string(argv[1]) != "--scenario") {
            throw std::runtime_error("usage: fakegpu_collective_direct_test --scenario <name>");
        }

        const std::string scenario = argv[2];
        CoordinatorFixture fixture;

        if (scenario == "allreduce") {
            run_allreduce_scenario();
            fake_gpu::dump_monitor_report();
            std::cout << "allreduce scenario passed" << std::endl;
            return 0;
        }
        if (scenario == "broadcast") {
            run_broadcast_scenario();
            fake_gpu::dump_monitor_report();
            std::cout << "broadcast scenario passed" << std::endl;
            return 0;
        }
        if (scenario == "reduce") {
            run_reduce_scenario();
            fake_gpu::dump_monitor_report();
            std::cout << "reduce scenario passed" << std::endl;
            return 0;
        }
        if (scenario == "allgather") {
            run_allgather_scenario();
            fake_gpu::dump_monitor_report();
            std::cout << "allgather scenario passed" << std::endl;
            return 0;
        }
        if (scenario == "reducescatter") {
            run_reducescatter_scenario();
            fake_gpu::dump_monitor_report();
            std::cout << "reducescatter scenario passed" << std::endl;
            return 0;
        }
        if (scenario == "chunked") {
            run_chunked_scenario();
            fake_gpu::dump_monitor_report();
            std::cout << "chunked scenario passed" << std::endl;
            return 0;
        }
        if (scenario == "mismatch") {
            run_mismatch_scenario();
            fake_gpu::dump_monitor_report();
            std::cout << "mismatch scenario passed" << std::endl;
            return 0;
        }
        if (scenario == "timeout") {
            run_timeout_case();
            fake_gpu::dump_monitor_report();
            std::cout << "timeout scenario passed" << std::endl;
            return 0;
        }

        throw std::runtime_error("unknown scenario: " + scenario);
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
