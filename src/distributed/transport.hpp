#pragma once

#include <string>
#include <unordered_map>

namespace fake_gpu::distributed {

struct CoordinatorMessage {
    std::string command;
    std::unordered_map<std::string, std::string> fields;
};

struct CoordinatorResponse {
    bool ok = false;
    std::unordered_map<std::string, std::string> fields;
    std::string error_code;
    std::string error_detail;
};

bool bind_and_listen_unix_socket(const std::string& path, int backlog, int& server_fd, std::string& error);
bool request_response_unix_socket(
    const std::string& path,
    const std::string& request_line,
    CoordinatorResponse& response,
    std::string& error);
bool receive_message_line(int fd, std::string& line, std::string& error);
bool send_message_line(int fd, const std::string& line, std::string& error);
bool parse_message_line(const std::string& line, CoordinatorMessage& message, std::string& error);
bool parse_response_line(const std::string& line, CoordinatorResponse& response, std::string& error);
std::string format_ok_response(const std::unordered_map<std::string, std::string>& fields = {});
std::string format_error_response(const std::string& code, const std::string& detail);

}  // namespace fake_gpu::distributed
