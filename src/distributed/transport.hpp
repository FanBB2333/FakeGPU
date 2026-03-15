#pragma once

#include "cluster_config.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace fake_gpu::distributed {

struct CoordinatorMessage {
    std::string command;
    std::unordered_map<std::string, std::string> fields;
    std::vector<char> payload;
};

struct CoordinatorResponse {
    bool ok = false;
    std::unordered_map<std::string, std::string> fields;
    std::string error_code;
    std::string error_detail;
    std::vector<char> payload;
};

bool bind_and_listen_unix_socket(const std::string& path, int backlog, int& server_fd, std::string& error);
bool bind_and_listen_tcp_socket(const std::string& endpoint, int backlog, int& server_fd, std::string& error);
bool bind_and_listen(
    CoordinatorTransport transport,
    const std::string& address,
    int backlog,
    int& server_fd,
    std::string& error);
bool request_response_unix_socket(
    const std::string& path,
    const std::string& request_line,
    const std::vector<char>& request_payload,
    CoordinatorResponse& response,
    std::string& error);
bool request_response_tcp_socket(
    const std::string& endpoint,
    const std::string& request_line,
    const std::vector<char>& request_payload,
    CoordinatorResponse& response,
    std::string& error);
bool request_response(
    CoordinatorTransport transport,
    const std::string& address,
    const std::string& request_line,
    const std::vector<char>& request_payload,
    CoordinatorResponse& response,
    std::string& error);
bool request_response(
    CoordinatorTransport transport,
    const std::string& address,
    const std::string& request_line,
    CoordinatorResponse& response,
    std::string& error);
bool receive_message_line(int fd, std::string& line, std::string& error);
bool send_message_line(int fd, const std::string& line, std::string& error);
bool receive_message_packet(
    int fd,
    std::string& line,
    std::vector<char>& payload,
    std::string& error);
bool send_message_packet(
    int fd,
    const std::string& line,
    const std::vector<char>& payload,
    std::string& error);
bool parse_message_line(const std::string& line, CoordinatorMessage& message, std::string& error);
bool parse_response_line(const std::string& line, CoordinatorResponse& response, std::string& error);
std::string format_ok_response(const std::unordered_map<std::string, std::string>& fields = {});
std::string format_error_response(const std::string& code, const std::string& detail);

}  // namespace fake_gpu::distributed
