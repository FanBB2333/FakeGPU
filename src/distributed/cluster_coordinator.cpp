#include "cluster_coordinator.hpp"

#include "transport.hpp"

#include <cstdint>
#include <cerrno>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <sys/socket.h>
#include <unistd.h>
#include <unordered_map>

namespace fake_gpu::distributed {

namespace {

bool parse_optional_bool_field(
    const std::unordered_map<std::string, std::string>& fields,
    const char* key,
    bool default_value,
    bool& value,
    std::string& error) {
    value = default_value;
    auto it = fields.find(key);
    if (it == fields.end()) {
        return true;
    }
    if (it->second == "1" || it->second == "true") {
        value = true;
        return true;
    }
    if (it->second == "0" || it->second == "false") {
        value = false;
        return true;
    }
    error = std::string("invalid boolean field: ") + key;
    return false;
}

int parse_required_int(
    const std::unordered_map<std::string, std::string>& fields,
    const char* key,
    bool& ok,
    std::string& error) {
    auto it = fields.find(key);
    if (it == fields.end()) {
        ok = false;
        error = std::string("missing required field: ") + key;
        return 0;
    }
    try {
        std::size_t consumed = 0;
        int value = std::stoi(it->second, &consumed, 10);
        if (consumed != it->second.size()) {
            throw std::invalid_argument("trailing");
        }
        ok = true;
        return value;
    } catch (...) {
        ok = false;
        error = std::string("invalid integer field: ") + key;
        return 0;
    }
}

std::uint64_t parse_required_u64(
    const std::unordered_map<std::string, std::string>& fields,
    const char* key,
    bool& ok,
    std::string& error) {
    auto it = fields.find(key);
    if (it == fields.end()) {
        ok = false;
        error = std::string("missing required field: ") + key;
        return 0;
    }
    try {
        std::size_t consumed = 0;
        std::uint64_t value = std::stoull(it->second, &consumed, 10);
        if (consumed != it->second.size()) {
            throw std::invalid_argument("trailing");
        }
        ok = true;
        return value;
    } catch (...) {
        ok = false;
        error = std::string("invalid uint64 field: ") + key;
        return 0;
    }
}

std::size_t parse_required_size(
    const std::unordered_map<std::string, std::string>& fields,
    const char* key,
    bool& ok,
    std::string& error) {
    const std::uint64_t value = parse_required_u64(fields, key, ok, error);
    if (!ok) {
        return 0;
    }
    if (value > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        ok = false;
        error = std::string("field is too large: ") + key;
        return 0;
    }
    return static_cast<std::size_t>(value);
}

}  // namespace

ClusterCoordinator::ClusterCoordinator(std::string socket_path)
    : socket_path_(std::move(socket_path)) {
}

ClusterCoordinator::~ClusterCoordinator() {
    request_shutdown();
    if (server_fd_ >= 0) {
        ::close(server_fd_);
        server_fd_ = -1;
    }
    {
        std::lock_guard<std::mutex> lock(client_threads_mutex_);
        for (std::thread& thread : client_threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        client_threads_.clear();
    }
    if (!socket_path_.empty()) {
        ::unlink(socket_path_.c_str());
    }
}

bool ClusterCoordinator::start(std::string& error) {
    if (!bind_and_listen_unix_socket(socket_path_, 64, server_fd_, error)) {
        return false;
    }
    return true;
}

int ClusterCoordinator::run() {
    accept_loop();
    return 0;
}

void ClusterCoordinator::request_shutdown() {
    if (!shutdown_requested_.exchange(true) && server_fd_ >= 0) {
        ::shutdown(server_fd_, SHUT_RDWR);
        ::close(server_fd_);
        server_fd_ = -1;
    }
}

void ClusterCoordinator::accept_loop() {
    while (!shutdown_requested_.load()) {
        const int client_fd = ::accept(server_fd_, nullptr, nullptr);
        if (client_fd < 0) {
            if (shutdown_requested_.load()) {
                break;
            }
            if (errno == EINTR) {
                continue;
            }
            break;
        }

        std::lock_guard<std::mutex> lock(client_threads_mutex_);
        client_threads_.emplace_back([this, client_fd]() {
            handle_client(client_fd);
        });
    }
}

void ClusterCoordinator::handle_client(int client_fd) {
    std::string request_line;
    std::string transport_error;
    if (!receive_message_line(client_fd, request_line, transport_error)) {
        send_message_line(client_fd, format_error_response("bad_request", transport_error), transport_error);
        ::close(client_fd);
        return;
    }

    CoordinatorMessage request;
    std::string parse_error;
    if (!parse_message_line(request_line, request, parse_error)) {
        send_message_line(client_fd, format_error_response("bad_request", parse_error), transport_error);
        ::close(client_fd);
        return;
    }

    std::string response;
    if (request.command == "PING") {
        response = format_ok_response({
            {"status", "ready"},
            {"version", "1"},
            {"transport", "unix"},
        });
    } else if (request.command == "HELLO") {
        response = format_ok_response({
            {"status", "ready"},
            {"version", "1"},
        });
    } else if (request.command == "INIT_COMM") {
        auto unique_it = request.fields.find("unique_id");
        if (unique_it == request.fields.end()) {
            response = format_error_response("bad_request", "missing required field: unique_id");
        } else {
            bool ok = false;
            std::string error;
            const int world_size = parse_required_int(request.fields, "world_size", ok, error);
            if (!ok) {
                response = format_error_response("bad_request", error);
            } else {
                const int rank = parse_required_int(request.fields, "rank", ok, error);
                if (!ok) {
                    response = format_error_response("bad_request", error);
                } else {
                    int timeout_ms = 1000;
                    auto timeout_it = request.fields.find("timeout_ms");
                    if (timeout_it != request.fields.end()) {
                        try {
                            std::size_t consumed = 0;
                            timeout_ms = std::stoi(timeout_it->second, &consumed, 10);
                            if (consumed != timeout_it->second.size()) {
                                throw std::invalid_argument("trailing");
                            }
                        } catch (...) {
                            timeout_ms = -1;
                        }
                    }

                    CommunicatorRegistrationResult result =
                        communicator_registry_.init_communicator(unique_it->second, world_size, rank, timeout_ms);
                    if (!result.ok) {
                        response = format_error_response(result.error_code, result.error_detail);
                    } else {
                        response = format_ok_response({
                            {"comm_id", std::to_string(result.comm_id)},
                            {"seqno", std::to_string(result.seqno)},
                            {"rank", std::to_string(rank)},
                            {"world_size", std::to_string(world_size)},
                        });
                    }
                }
            }
        }
    } else if (request.command == "DESTROY_COMM") {
        bool ok = false;
        std::string error;
        const int comm_id = parse_required_int(request.fields, "comm_id", ok, error);
        if (!ok) {
            response = format_error_response("bad_request", error);
        } else {
            const int rank = parse_required_int(request.fields, "rank", ok, error);
            if (!ok) {
                response = format_error_response("bad_request", error);
            } else {
                CommunicatorDestroyResult result = communicator_registry_.destroy_communicator(comm_id, rank);
                if (!result.ok) {
                    response = format_error_response(result.error_code, result.error_detail);
                } else {
                    response = format_ok_response({
                        {"comm_id", std::to_string(comm_id)},
                        {"rank", std::to_string(rank)},
                    });
                }
            }
        }
    } else if (
        request.command == "ALLREDUCE" ||
        request.command == "BROADCAST" ||
        request.command == "ALLGATHER" ||
        request.command == "REDUCESCATTER") {
        CollectiveSubmitRequest collective_request;
        bool ok = false;
        std::string error;

        collective_request.comm_id = parse_required_int(request.fields, "comm_id", ok, error);
        if (!ok) {
            response = format_error_response("bad_request", error);
        } else {
            collective_request.rank = parse_required_int(request.fields, "rank", ok, error);
            if (!ok) {
                response = format_error_response("bad_request", error);
            } else {
                collective_request.seqno = parse_required_u64(request.fields, "seqno", ok, error);
                if (!ok) {
                    response = format_error_response("bad_request", error);
                } else {
                    collective_request.count = parse_required_size(request.fields, "count", ok, error);
                    if (!ok) {
                        response = format_error_response("bad_request", error);
                    } else {
                        collective_request.bytes = parse_required_size(request.fields, "bytes", ok, error);
                        if (!ok) {
                            response = format_error_response("bad_request", error);
                        } else {
                            collective_request.root = parse_required_int(request.fields, "root", ok, error);
                            if (!ok) {
                                response = format_error_response("bad_request", error);
                            } else {
                                collective_request.timeout_ms =
                                    parse_required_int(request.fields, "timeout_ms", ok, error);
                                if (!ok) {
                                    response = format_error_response("bad_request", error);
                                } else {
                                    bool proxy_only = false;
                                    if (!parse_optional_bool_field(
                                            request.fields,
                                            "proxy_only",
                                            false,
                                            proxy_only,
                                            error)) {
                                        response = format_error_response("bad_request", error);
                                    } else {
                                        collective_request.proxy_only = proxy_only;
                                        auto staging_it = request.fields.find("staging_name");
                                        if (!collective_request.proxy_only &&
                                            staging_it == request.fields.end()) {
                                            response = format_error_response(
                                                "bad_request",
                                                "missing required field: staging_name");
                                        } else {
                                            if (staging_it != request.fields.end()) {
                                                collective_request.staging_name = staging_it->second;
                                            }
                                            auto dtype_it = request.fields.find("dtype");
                                            auto reduce_it = request.fields.find("reduce_op");
                                            if (dtype_it == request.fields.end()) {
                                                response = format_error_response(
                                                    "bad_request",
                                                    "missing required field: dtype");
                                            } else if (reduce_it == request.fields.end()) {
                                                response = format_error_response(
                                                    "bad_request",
                                                    "missing required field: reduce_op");
                                            } else if (!parse_collective_data_type(
                                                           dtype_it->second,
                                                           collective_request.dtype)) {
                                                response = format_error_response(
                                                    "bad_request",
                                                    "unsupported dtype");
                                            } else if (!parse_collective_reduce_op(
                                                           reduce_it->second,
                                                           collective_request.reduce_op)) {
                                                response = format_error_response(
                                                    "bad_request",
                                                    "unsupported reduce_op");
                                            } else {
                                                if (request.command == "ALLREDUCE") {
                                                    collective_request.type = CollectiveType::AllReduce;
                                                } else if (request.command == "BROADCAST") {
                                                    collective_request.type = CollectiveType::Broadcast;
                                                } else if (request.command == "ALLGATHER") {
                                                    collective_request.type = CollectiveType::AllGather;
                                                } else {
                                                    collective_request.type = CollectiveType::ReduceScatter;
                                                }
                                                CollectiveSubmitResult result =
                                                    communicator_registry_.submit_collective(collective_request);
                                                if (!result.ok) {
                                                    response = format_error_response(
                                                        result.error_code,
                                                        result.error_detail);
                                                } else {
                                                    response = format_ok_response({
                                                        {"comm_id", std::to_string(collective_request.comm_id)},
                                                        {"seqno", std::to_string(result.seqno)},
                                                        {"rank", std::to_string(collective_request.rank)},
                                                        {"op", collective_type_name(collective_request.type)},
                                                    });
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } else if (request.command == "BARRIER") {
        BarrierSubmitRequest barrier_request;
        bool ok = false;
        std::string error;

        barrier_request.comm_id = parse_required_int(request.fields, "comm_id", ok, error);
        if (!ok) {
            response = format_error_response("bad_request", error);
        } else {
            barrier_request.rank = parse_required_int(request.fields, "rank", ok, error);
            if (!ok) {
                response = format_error_response("bad_request", error);
            } else {
                barrier_request.seqno = parse_required_u64(request.fields, "seqno", ok, error);
                if (!ok) {
                    response = format_error_response("bad_request", error);
                } else {
                    barrier_request.timeout_ms = parse_required_int(request.fields, "timeout_ms", ok, error);
                    if (!ok) {
                        response = format_error_response("bad_request", error);
                    } else {
                        BarrierSubmitResult result =
                            communicator_registry_.submit_barrier(barrier_request);
                        if (!result.ok) {
                            response = format_error_response(result.error_code, result.error_detail);
                        } else {
                            response = format_ok_response({
                                {"comm_id", std::to_string(barrier_request.comm_id)},
                                {"seqno", std::to_string(result.seqno)},
                                {"rank", std::to_string(barrier_request.rank)},
                                {"op", "barrier"},
                            });
                        }
                    }
                }
            }
        }
    } else if (request.command == "GROUP_PREPARE") {
        CollectiveBatchPrepareRequest batch_request;
        bool ok = false;
        std::string error;

        batch_request.comm_id = parse_required_int(request.fields, "comm_id", ok, error);
        if (!ok) {
            response = format_error_response("bad_request", error);
        } else {
            batch_request.rank = parse_required_int(request.fields, "rank", ok, error);
            if (!ok) {
                response = format_error_response("bad_request", error);
            } else {
                batch_request.base_seqno = parse_required_u64(request.fields, "base_seqno", ok, error);
                if (!ok) {
                    response = format_error_response("bad_request", error);
                } else {
                    batch_request.timeout_ms = parse_required_int(request.fields, "timeout_ms", ok, error);
                    if (!ok) {
                        response = format_error_response("bad_request", error);
                    } else {
                        const int op_count = parse_required_int(request.fields, "op_count", ok, error);
                        if (!ok) {
                            response = format_error_response("bad_request", error);
                        } else if (op_count <= 0) {
                            response = format_error_response("bad_request", "op_count must be > 0");
                        } else {
                            batch_request.operations.reserve(static_cast<std::size_t>(op_count));
                            for (int index = 0; index < op_count && ok; ++index) {
                                CollectiveBatchPlanItem item;
                                const std::string prefix = "op" + std::to_string(index) + "_";

                                auto type_it = request.fields.find(prefix + "type");
                                auto dtype_it = request.fields.find(prefix + "dtype");
                                auto reduce_it = request.fields.find(prefix + "reduce_op");
                                if (type_it == request.fields.end()) {
                                    ok = false;
                                    error = "missing required field: " + prefix + "type";
                                    break;
                                }
                                if (dtype_it == request.fields.end()) {
                                    ok = false;
                                    error = "missing required field: " + prefix + "dtype";
                                    break;
                                }
                                if (reduce_it == request.fields.end()) {
                                    ok = false;
                                    error = "missing required field: " + prefix + "reduce_op";
                                    break;
                                }
                                if (!parse_collective_type(type_it->second, item.type)) {
                                    ok = false;
                                    error = "unsupported collective type";
                                    break;
                                }
                                if (!parse_collective_data_type(dtype_it->second, item.dtype)) {
                                    ok = false;
                                    error = "unsupported dtype";
                                    break;
                                }
                                if (!parse_collective_reduce_op(reduce_it->second, item.reduce_op)) {
                                    ok = false;
                                    error = "unsupported reduce_op";
                                    break;
                                }

                                item.count = parse_required_size(request.fields, (prefix + "count").c_str(), ok, error);
                                if (!ok) {
                                    break;
                                }
                                item.bytes = parse_required_size(request.fields, (prefix + "bytes").c_str(), ok, error);
                                if (!ok) {
                                    break;
                                }
                                item.root = parse_required_int(request.fields, (prefix + "root").c_str(), ok, error);
                                if (!ok) {
                                    break;
                                }
                                batch_request.operations.push_back(item);
                            }

                            if (!ok) {
                                response = format_error_response("bad_request", error);
                            } else {
                                CollectiveBatchPrepareResult result =
                                    communicator_registry_.prepare_collective_batch(batch_request);
                                if (!result.ok) {
                                    response = format_error_response(result.error_code, result.error_detail);
                                } else {
                                    response = format_ok_response({
                                        {"comm_id", std::to_string(batch_request.comm_id)},
                                        {"base_seqno", std::to_string(result.base_seqno)},
                                        {"rank", std::to_string(batch_request.rank)},
                                        {"op_count", std::to_string(batch_request.operations.size())},
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    } else if (request.command == "SHUTDOWN") {
        response = format_ok_response({
            {"status", "shutting_down"},
        });
        send_message_line(client_fd, response, transport_error);
        ::close(client_fd);
        request_shutdown();
        return;
    } else {
        response = format_error_response("unknown_command", request.command);
    }

    send_message_line(client_fd, response, transport_error);
    ::close(client_fd);
}

}  // namespace fake_gpu::distributed
