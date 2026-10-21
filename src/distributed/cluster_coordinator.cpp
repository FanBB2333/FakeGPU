#include "cluster_coordinator.hpp"

#include "transport.hpp"

#include <cerrno>
#include <cstdint>
#include <string>
#include <utility>
#include <sys/socket.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace fake_gpu::distributed {

namespace {

using MessageFields = std::unordered_map<std::string, std::string>;

static_assert(sizeof(std::size_t) == sizeof(std::uint64_t));

bool parse_int_text(const std::string& text, int& value) {
    try {
        std::size_t consumed = 0;
        const int parsed = std::stoi(text, &consumed, 10);
        if (consumed != text.size()) {
            return false;
        }
        value = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_u64_text(const std::string& text, std::uint64_t& value) {
    try {
        std::size_t consumed = 0;
        const std::uint64_t parsed = std::stoull(text, &consumed, 10);
        if (consumed != text.size()) {
            return false;
        }
        value = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_size_text(const std::string& text, std::size_t& value) {
    std::uint64_t parsed = 0;
    if (!parse_u64_text(text, parsed)) {
        return false;
    }
    value = static_cast<std::size_t>(parsed);
    return true;
}

template <typename T, typename ParseItem>
bool parse_list(const std::string& text, std::vector<T>& values, ParseItem parse_item) {
    std::size_t begin = 0;
    while (true) {
        const std::size_t end = text.find(',', begin);
        T value{};
        if (!parse_item(
                text.substr(begin, end == std::string::npos ? std::string::npos : end - begin),
                value)) {
            return false;
        }
        values.push_back(value);
        if (end == std::string::npos) {
            return true;
        }
        begin = end + 1;
    }
}

bool read_optional_bool(
    const MessageFields& fields,
    const char* key,
    bool& value,
    std::string& error) {
    value = false;
    auto it = fields.find(key);
    if (it == fields.end()) {
        return true;
    }
    if (it->second == "1" || it->second == "true") {
        value = true;
        return true;
    }
    if (it->second == "0" || it->second == "false") {
        return true;
    }
    error = std::string("invalid boolean field: ") + key;
    return false;
}

bool read_int(
    const MessageFields& fields,
    const std::string& key,
    int& value,
    std::string& error) {
    auto it = fields.find(key);
    if (it == fields.end()) {
        error = "missing required field: " + key;
        return false;
    }
    if (!parse_int_text(it->second, value)) {
        error = "invalid integer field: " + key;
        return false;
    }
    return true;
}

bool read_u64(
    const MessageFields& fields,
    const std::string& key,
    std::uint64_t& value,
    std::string& error) {
    auto it = fields.find(key);
    if (it == fields.end()) {
        error = "missing required field: " + key;
        return false;
    }
    if (!parse_u64_text(it->second, value)) {
        error = "invalid uint64 field: " + key;
        return false;
    }
    return true;
}

bool read_size(
    const MessageFields& fields,
    const std::string& key,
    std::size_t& value,
    std::string& error) {
    std::uint64_t parsed = 0;
    if (!read_u64(fields, key, parsed, error)) {
        return false;
    }
    value = static_cast<std::size_t>(parsed);
    return true;
}

bool read_optional_size(
    const MessageFields& fields,
    const char* key,
    std::size_t default_value,
    std::size_t& value,
    std::string& error) {
    if (fields.find(key) == fields.end()) {
        value = default_value;
        return true;
    }
    return read_size(fields, key, value, error);
}

bool read_optional_size_list(
    const MessageFields& fields,
    const char* key,
    std::vector<std::size_t>& values,
    std::string& error) {
    values.clear();
    auto it = fields.find(key);
    if (it == fields.end()) {
        return true;
    }
    if (it->second.empty()) {
        error = std::string("empty size list field: ") + key;
        return false;
    }
    if (!parse_list(it->second, values, parse_size_text)) {
        error = std::string("invalid size list field: ") + key;
        return false;
    }
    return true;
}

bool read_int_list(
    const MessageFields& fields,
    const char* key,
    std::vector<int>& values,
    std::string& error) {
    values.clear();
    const auto it = fields.find(key);
    if (it == fields.end()) {
        error = std::string("missing required field: ") + key;
        return false;
    }
    if (it->second.empty()) {
        error = std::string("empty integer list field: ") + key;
        return false;
    }
    if (!parse_list(it->second, values, parse_int_text)) {
        error = std::string("invalid integer list field: ") + key;
        return false;
    }
    return true;
}

std::string bad_request(const std::string& detail) {
    return format_error_response("bad_request", detail);
}

bool collective_command_type(const std::string& command, CollectiveType& type) {
    static const std::pair<const char*, CollectiveType> kCollectiveCommands[] = {
        {"ALLREDUCE", CollectiveType::AllReduce},
        {"REDUCE", CollectiveType::Reduce},
        {"BROADCAST", CollectiveType::Broadcast},
        {"ALLGATHER", CollectiveType::AllGather},
        {"REDUCESCATTER", CollectiveType::ReduceScatter},
        {"ALLTOALL", CollectiveType::AllToAll},
    };
    for (const auto& [name, command_type] : kCollectiveCommands) {
        if (command == name) {
            type = command_type;
            return true;
        }
    }
    return false;
}

std::string handle_init_comm(CommunicatorRegistry& registry, const CoordinatorMessage& request) {
    auto unique_it = request.fields.find("unique_id");
    if (unique_it == request.fields.end()) {
        return bad_request("missing required field: unique_id");
    }
    int world_size = 0;
    int rank = 0;
    std::string error;
    if (!read_int(request.fields, "world_size", world_size, error) ||
        !read_int(request.fields, "rank", rank, error)) {
        return bad_request(error);
    }
    int timeout_ms = 1000;
    auto timeout_it = request.fields.find("timeout_ms");
    if (timeout_it != request.fields.end() && !parse_int_text(timeout_it->second, timeout_ms)) {
        timeout_ms = -1;
    }

    CommunicatorRegistrationResult result =
        registry.init_communicator(unique_it->second, world_size, rank, timeout_ms);
    if (!result.ok) {
        return format_error_response(result.error_code, result.error_detail);
    }
    return format_ok_response({
        {"comm_id", std::to_string(result.comm_id)},
        {"seqno", std::to_string(result.seqno)},
        {"rank", std::to_string(rank)},
        {"world_size", std::to_string(world_size)},
    });
}

std::string handle_destroy_comm(CommunicatorRegistry& registry, const CoordinatorMessage& request) {
    int comm_id = 0;
    int rank = 0;
    std::string error;
    if (!read_int(request.fields, "comm_id", comm_id, error) ||
        !read_int(request.fields, "rank", rank, error)) {
        return bad_request(error);
    }
    CommunicatorDestroyResult result = registry.destroy_communicator(comm_id, rank);
    if (!result.ok) {
        return format_error_response(result.error_code, result.error_detail);
    }
    return format_ok_response({
        {"comm_id", std::to_string(comm_id)},
        {"rank", std::to_string(rank)},
    });
}

std::string handle_split_comm(CommunicatorRegistry& registry, const CoordinatorMessage& request) {
    CommunicatorSplitRequest split_request;
    std::string error;
    if (!read_int(request.fields, "comm_id", split_request.comm_id, error) ||
        !read_int(request.fields, "rank", split_request.rank, error) ||
        !read_u64(request.fields, "seqno", split_request.seqno, error) ||
        !read_int(request.fields, "color", split_request.color, error) ||
        !read_int(request.fields, "key", split_request.key, error) ||
        !read_int(request.fields, "timeout_ms", split_request.timeout_ms, error)) {
        return bad_request(error);
    }
    CommunicatorSplitResult result = registry.split_communicator(split_request);
    if (!result.ok) {
        return format_error_response(result.error_code, result.error_detail);
    }
    return format_ok_response({
        {"comm_id", std::to_string(split_request.comm_id)},
        {"seqno", std::to_string(result.seqno)},
        {"rank", std::to_string(split_request.rank)},
        {"participating", result.participating ? "1" : "0"},
        {"new_comm_id", std::to_string(result.new_comm_id)},
        {"new_rank", std::to_string(result.new_rank)},
        {"new_world_size", std::to_string(result.new_world_size)},
    });
}

std::string handle_inject_failure(CommunicatorRegistry& registry, const CoordinatorMessage& request) {
    CommunicatorFailureRequest failure_request;
    std::string error;
    if (!read_int(request.fields, "comm_id", failure_request.comm_id, error) ||
        !read_int(request.fields, "rank", failure_request.rank, error) ||
        !read_u64(request.fields, "seqno", failure_request.seqno, error)) {
        return bad_request(error);
    }
    const auto operation_it = request.fields.find("operation");
    if (operation_it == request.fields.end()) {
        return bad_request("missing required field: operation");
    }
    const auto error_code_it = request.fields.find("error_code");
    if (error_code_it == request.fields.end()) {
        return bad_request("missing required field: error_code");
    }
    failure_request.operation = operation_it->second;
    failure_request.error_code = error_code_it->second;
    if (!read_u64(
            request.fields,
            "attempted_payload_bytes",
            failure_request.attempted_payload_bytes,
            error)) {
        return bad_request(error);
    }
    CommunicatorFailureResult result = registry.inject_communicator_failure(failure_request);
    if (!result.ok) {
        return format_error_response(result.error_code, result.error_detail);
    }
    return format_ok_response({
        {"comm_id", std::to_string(failure_request.comm_id)},
        {"seqno", std::to_string(failure_request.seqno)},
        {"rank", std::to_string(failure_request.rank)},
        {"event_index", std::to_string(result.event_index)},
    });
}

std::string handle_shrink_comm(CommunicatorRegistry& registry, const CoordinatorMessage& request) {
    CommunicatorShrinkRequest shrink_request;
    std::string error;
    if (!read_int(request.fields, "comm_id", shrink_request.comm_id, error) ||
        !read_int(request.fields, "rank", shrink_request.rank, error) ||
        !read_u64(request.fields, "seqno", shrink_request.seqno, error) ||
        !read_int_list(request.fields, "excluded_ranks", shrink_request.excluded_ranks, error) ||
        !read_int(request.fields, "timeout_ms", shrink_request.timeout_ms, error) ||
        !read_optional_bool(request.fields, "abort_parent", shrink_request.abort_parent, error)) {
        return bad_request(error);
    }
    CommunicatorShrinkResult result = registry.shrink_communicator(shrink_request);
    if (!result.ok) {
        return format_error_response(result.error_code, result.error_detail);
    }
    return format_ok_response({
        {"comm_id", std::to_string(shrink_request.comm_id)},
        {"seqno", std::to_string(result.seqno)},
        {"rank", std::to_string(shrink_request.rank)},
        {"new_comm_id", std::to_string(result.new_comm_id)},
        {"new_rank", std::to_string(result.new_rank)},
        {"new_world_size", std::to_string(result.new_world_size)},
    });
}

std::string handle_point_to_point(
    CommunicatorRegistry& registry,
    CoordinatorMessage& request,
    std::vector<char>& response_payload) {
    PointToPointSubmitRequest p2p_request;
    std::string error;
    if (!read_int(request.fields, "comm_id", p2p_request.comm_id, error) ||
        !read_int(request.fields, "rank", p2p_request.rank, error) ||
        !read_u64(request.fields, "seqno", p2p_request.seqno, error) ||
        !read_int(request.fields, "peer", p2p_request.peer, error) ||
        !read_size(request.fields, "count", p2p_request.count, error) ||
        !read_size(request.fields, "bytes", p2p_request.bytes, error) ||
        !read_int(request.fields, "timeout_ms", p2p_request.timeout_ms, error)) {
        return bad_request(error);
    }

    auto staging_it = request.fields.find("staging_name");
    const auto transport_it = request.fields.find("buffer_transport");
    const bool socket_transport =
        transport_it != request.fields.end() &&
        transport_it->second == "socket";
    auto dtype_it = request.fields.find("dtype");
    if (!socket_transport && staging_it == request.fields.end()) {
        return bad_request("missing required field: staging_name");
    }
    if (dtype_it == request.fields.end()) {
        return bad_request("missing required field: dtype");
    }
    if (!parse_collective_data_type(dtype_it->second, p2p_request.dtype)) {
        return bad_request("unsupported dtype");
    }
    if (staging_it != request.fields.end()) {
        p2p_request.staging_name = staging_it->second;
    }
    if (socket_transport) {
        p2p_request.transport = BufferTransport::SocketPayload;
    }
    if (!read_optional_size(
            request.fields,
            "payload_bytes",
            request.payload.size(),
            p2p_request.payload_bytes,
            error)) {
        return bad_request(error);
    }
    p2p_request.payload = std::move(request.payload);
    p2p_request.type = request.command == "SEND"
        ? PointToPointType::Send
        : PointToPointType::Recv;

    PointToPointSubmitResult result = registry.submit_point_to_point(p2p_request);
    if (!result.ok) {
        return format_error_response(result.error_code, result.error_detail);
    }
    response_payload = std::move(result.output_payload);
    return format_ok_response({
        {"comm_id", std::to_string(p2p_request.comm_id)},
        {"seqno", std::to_string(result.seqno)},
        {"rank", std::to_string(p2p_request.rank)},
        {"peer", std::to_string(p2p_request.peer)},
        {"op", request.command == "SEND" ? "send" : "recv"},
    });
}

std::string handle_collective(
    CommunicatorRegistry& registry,
    CoordinatorMessage& request,
    CollectiveType type,
    std::vector<char>& response_payload) {
    CollectiveSubmitRequest collective_request;
    std::string error;
    if (!read_int(request.fields, "comm_id", collective_request.comm_id, error) ||
        !read_int(request.fields, "rank", collective_request.rank, error) ||
        !read_u64(request.fields, "seqno", collective_request.seqno, error) ||
        !read_size(request.fields, "count", collective_request.count, error) ||
        !read_size(request.fields, "bytes", collective_request.bytes, error) ||
        !read_int(request.fields, "root", collective_request.root, error) ||
        !read_int(request.fields, "timeout_ms", collective_request.timeout_ms, error) ||
        !read_optional_bool(request.fields, "proxy_only", collective_request.proxy_only, error)) {
        return bad_request(error);
    }

    auto staging_it = request.fields.find("staging_name");
    const auto transport_it = request.fields.find("buffer_transport");
    const bool socket_transport =
        transport_it != request.fields.end() &&
        transport_it->second == "socket";
    if (!collective_request.proxy_only &&
        !socket_transport &&
        staging_it == request.fields.end()) {
        return bad_request("missing required field: staging_name");
    }
    if (staging_it != request.fields.end()) {
        collective_request.staging_name = staging_it->second;
    }
    if (socket_transport) {
        collective_request.transport = BufferTransport::SocketPayload;
    }
    if (!read_optional_size(
            request.fields,
            "payload_bytes",
            request.payload.size(),
            collective_request.payload_bytes,
            error)) {
        return bad_request(error);
    }
    collective_request.payload = std::move(request.payload);

    const bool has_input_splits =
        request.fields.find("input_splits") != request.fields.end();
    const bool has_output_splits =
        request.fields.find("output_splits") != request.fields.end();
    if (has_input_splits != has_output_splits) {
        return bad_request("input_splits and output_splits must be provided together");
    }
    if (!read_optional_size_list(request.fields, "input_splits", collective_request.input_splits, error) ||
        !read_optional_size_list(request.fields, "output_splits", collective_request.output_splits, error) ||
        !read_optional_size(
            request.fields,
            "input_bytes",
            collective_request.bytes,
            collective_request.input_bytes,
            error) ||
        !read_optional_size(
            request.fields,
            "output_bytes",
            collective_request.bytes,
            collective_request.output_bytes,
            error)) {
        return bad_request(error);
    }

    auto dtype_it = request.fields.find("dtype");
    auto reduce_it = request.fields.find("reduce_op");
    if (dtype_it == request.fields.end()) {
        return bad_request("missing required field: dtype");
    }
    if (reduce_it == request.fields.end()) {
        return bad_request("missing required field: reduce_op");
    }
    if (!parse_collective_data_type(dtype_it->second, collective_request.dtype)) {
        return bad_request("unsupported dtype");
    }
    if (!parse_collective_reduce_op(reduce_it->second, collective_request.reduce_op)) {
        return bad_request("unsupported reduce_op");
    }
    collective_request.type = type;

    CollectiveSubmitResult result = registry.submit_collective(collective_request);
    if (!result.ok) {
        return format_error_response(result.error_code, result.error_detail);
    }
    response_payload = std::move(result.output_payload);
    return format_ok_response({
        {"comm_id", std::to_string(collective_request.comm_id)},
        {"seqno", std::to_string(result.seqno)},
        {"rank", std::to_string(collective_request.rank)},
        {"op", collective_type_name(collective_request.type)},
    });
}

std::string handle_barrier(CommunicatorRegistry& registry, const CoordinatorMessage& request) {
    BarrierSubmitRequest barrier_request;
    std::string error;
    if (!read_int(request.fields, "comm_id", barrier_request.comm_id, error) ||
        !read_int(request.fields, "rank", barrier_request.rank, error) ||
        !read_u64(request.fields, "seqno", barrier_request.seqno, error) ||
        !read_int(request.fields, "timeout_ms", barrier_request.timeout_ms, error)) {
        return bad_request(error);
    }
    BarrierSubmitResult result = registry.submit_barrier(barrier_request);
    if (!result.ok) {
        return format_error_response(result.error_code, result.error_detail);
    }
    return format_ok_response({
        {"comm_id", std::to_string(barrier_request.comm_id)},
        {"seqno", std::to_string(result.seqno)},
        {"rank", std::to_string(barrier_request.rank)},
        {"op", "barrier"},
    });
}

std::string handle_group_prepare(CommunicatorRegistry& registry, const CoordinatorMessage& request) {
    CollectiveBatchPrepareRequest batch_request;
    int op_count = 0;
    std::string error;
    if (!read_int(request.fields, "comm_id", batch_request.comm_id, error) ||
        !read_int(request.fields, "rank", batch_request.rank, error) ||
        !read_u64(request.fields, "base_seqno", batch_request.base_seqno, error) ||
        !read_int(request.fields, "timeout_ms", batch_request.timeout_ms, error) ||
        !read_int(request.fields, "op_count", op_count, error)) {
        return bad_request(error);
    }
    if (op_count <= 0) {
        return bad_request("op_count must be > 0");
    }

    batch_request.operations.reserve(static_cast<std::size_t>(op_count));
    for (int index = 0; index < op_count; ++index) {
        CollectiveBatchPlanItem item;
        const std::string prefix = "op" + std::to_string(index) + "_";

        auto type_it = request.fields.find(prefix + "type");
        auto dtype_it = request.fields.find(prefix + "dtype");
        auto reduce_it = request.fields.find(prefix + "reduce_op");
        if (type_it == request.fields.end()) {
            return bad_request("missing required field: " + prefix + "type");
        }
        if (dtype_it == request.fields.end()) {
            return bad_request("missing required field: " + prefix + "dtype");
        }
        if (reduce_it == request.fields.end()) {
            return bad_request("missing required field: " + prefix + "reduce_op");
        }
        if (!parse_collective_type(type_it->second, item.type)) {
            return bad_request("unsupported collective type");
        }
        if (!parse_collective_data_type(dtype_it->second, item.dtype)) {
            return bad_request("unsupported dtype");
        }
        if (!parse_collective_reduce_op(reduce_it->second, item.reduce_op)) {
            return bad_request("unsupported reduce_op");
        }
        if (!read_size(request.fields, prefix + "count", item.count, error) ||
            !read_size(request.fields, prefix + "bytes", item.bytes, error) ||
            !read_int(request.fields, prefix + "root", item.root, error)) {
            return bad_request(error);
        }
        batch_request.operations.push_back(item);
    }

    CollectiveBatchPrepareResult result = registry.prepare_collective_batch(batch_request);
    if (!result.ok) {
        return format_error_response(result.error_code, result.error_detail);
    }
    return format_ok_response({
        {"comm_id", std::to_string(batch_request.comm_id)},
        {"base_seqno", std::to_string(result.base_seqno)},
        {"rank", std::to_string(batch_request.rank)},
        {"op_count", std::to_string(batch_request.operations.size())},
    });
}

}  // namespace

ClusterCoordinator::ClusterCoordinator(CoordinatorTransport transport, std::string address)
    : transport_(transport), address_(std::move(address)) {
}

ClusterCoordinator::ClusterCoordinator(std::string socket_path)
    : ClusterCoordinator(CoordinatorTransport::Unix, std::move(socket_path)) {
}

ClusterCoordinator::~ClusterCoordinator() {
    request_shutdown();
    {
        std::lock_guard<std::mutex> lock(client_threads_mutex_);
        for (ClientThread& client : client_threads_) {
            if (client.thread.joinable()) {
                client.thread.join();
            }
        }
        client_threads_.clear();
    }
    if (transport_ == CoordinatorTransport::Unix && !address_.empty()) {
        ::unlink(address_.c_str());
    }
}

bool ClusterCoordinator::start(std::string& error) {
    return bind_and_listen(transport_, address_, 64, server_fd_, error);
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
        for (auto it = client_threads_.begin(); it != client_threads_.end();) {
            if (it->done->load()) {
                it->thread.join();
                it = client_threads_.erase(it);
            } else {
                ++it;
            }
        }
        auto done = std::make_shared<std::atomic<bool>>(false);
        client_threads_.push_back(ClientThread{
            std::thread([this, client_fd, done]() {
                handle_client(client_fd);
                done->store(true);
            }),
            done,
        });
    }
}

void ClusterCoordinator::handle_client(int client_fd) {
    std::string request_line;
    std::vector<char> request_payload;
    std::string transport_error;
    if (!receive_message_packet(client_fd, request_line, request_payload, transport_error)) {
        send_message_packet(
            client_fd,
            format_error_response("bad_request", transport_error),
            {},
            transport_error);
        ::close(client_fd);
        return;
    }

    CoordinatorMessage request;
    std::string parse_error;
    if (!parse_message_line(request_line, request, parse_error)) {
        send_message_packet(
            client_fd,
            format_error_response("bad_request", parse_error),
            {},
            transport_error);
        ::close(client_fd);
        return;
    }
    request.payload = std::move(request_payload);

    if (request.command == "SHUTDOWN") {
        send_message_packet(
            client_fd,
            format_ok_response({
                {"status", "shutting_down"},
            }),
            {},
            transport_error);
        ::close(client_fd);
        request_shutdown();
        return;
    }

    std::string response;
    std::vector<char> response_payload;
    CollectiveType collective_type = CollectiveType::AllReduce;
    if (request.command == "PING") {
        response = format_ok_response({
            {"status", "ready"},
            {"version", "1"},
            {"transport", coordinator_transport_name(transport_)},
        });
    } else if (request.command == "HELLO") {
        response = format_ok_response({
            {"status", "ready"},
            {"version", "1"},
        });
    } else if (request.command == "INIT_COMM") {
        response = handle_init_comm(communicator_registry_, request);
    } else if (request.command == "DESTROY_COMM") {
        response = handle_destroy_comm(communicator_registry_, request);
    } else if (request.command == "SPLIT_COMM") {
        response = handle_split_comm(communicator_registry_, request);
    } else if (request.command == "INJECT_FAILURE") {
        response = handle_inject_failure(communicator_registry_, request);
    } else if (request.command == "SHRINK_COMM") {
        response = handle_shrink_comm(communicator_registry_, request);
    } else if (request.command == "SEND" || request.command == "RECV") {
        response = handle_point_to_point(communicator_registry_, request, response_payload);
    } else if (collective_command_type(request.command, collective_type)) {
        response = handle_collective(communicator_registry_, request, collective_type, response_payload);
    } else if (request.command == "BARRIER") {
        response = handle_barrier(communicator_registry_, request);
    } else if (request.command == "GROUP_PREPARE") {
        response = handle_group_prepare(communicator_registry_, request);
    } else {
        response = format_error_response("unknown_command", request.command);
    }

    send_message_packet(client_fd, response, response_payload, transport_error);
    ::close(client_fd);
}

}  // namespace fake_gpu::distributed
