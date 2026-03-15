#include "transport.hpp"

#include <cerrno>
#include <cstring>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <array>
#include <charconv>
#include <sstream>
#include <utility>
#include <vector>

namespace fake_gpu::distributed {

namespace {

std::string trim(const std::string& value) {
    const std::size_t begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const std::size_t end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

std::string sanitize_value(std::string value) {
    for (char& ch : value) {
        if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == '=') {
            ch = '_';
        }
    }
    return value;
}

bool parse_payload_size_header(
    const std::string& line,
    std::size_t& payload_bytes,
    std::string& error) {
    payload_bytes = 0;

    std::istringstream stream(line);
    std::string token;
    if (!(stream >> token)) {
        error = "missing command";
        return false;
    }

    while (stream >> token) {
        const std::size_t equals = token.find('=');
        if (equals == std::string::npos || equals == 0 || equals + 1 >= token.size()) {
            error = "invalid token: " + token;
            return false;
        }
        if (token.substr(0, equals) != "payload_bytes") {
            continue;
        }

        const std::string value = token.substr(equals + 1);
        std::size_t parsed = 0;
        const auto [ptr, ec] =
            std::from_chars(value.data(), value.data() + value.size(), parsed);
        if (ec != std::errc() || ptr != value.data() + value.size()) {
            error = "invalid payload_bytes value";
            return false;
        }
        payload_bytes = parsed;
        return true;
    }

    return true;
}

bool send_all_bytes(int fd, const char* data, std::size_t size, std::string& error) {
    std::size_t offset = 0;
    while (offset < size) {
        const ssize_t rc = ::send(fd, data + offset, size - offset, 0);
        if (rc < 0) {
            if (errno == EINTR) {
                continue;
            }
            error = "send() failed: " + std::string(std::strerror(errno));
            return false;
        }
        offset += static_cast<std::size_t>(rc);
    }
    return true;
}

bool receive_exact_bytes(int fd, std::size_t size, std::vector<char>& out, std::string& error) {
    out.assign(size, '\0');
    std::size_t offset = 0;
    while (offset < size) {
        const ssize_t rc = ::recv(fd, out.data() + offset, size - offset, 0);
        if (rc == 0) {
            error = "peer closed connection before sending the full payload";
            return false;
        }
        if (rc < 0) {
            if (errno == EINTR) {
                continue;
            }
            error = "recv() failed: " + std::string(std::strerror(errno));
            return false;
        }
        offset += static_cast<std::size_t>(rc);
    }
    return true;
}

std::string append_payload_header(
    const std::string& line,
    const std::vector<char>& payload) {
    if (payload.empty()) {
        return line;
    }
    if (line.find("payload_bytes=") != std::string::npos) {
        return line;
    }
    return line + " payload_bytes=" + std::to_string(payload.size());
}

}  // namespace

bool bind_and_listen_unix_socket(const std::string& path, int backlog, int& server_fd, std::string& error) {
    server_fd = -1;

    if (path.empty() || path.front() != '/') {
        error = "unix socket path must be absolute";
        return false;
    }

    const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        error = "socket() failed: " + std::string(std::strerror(errno));
        return false;
    }

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (path.size() >= sizeof(addr.sun_path)) {
        error = "unix socket path is too long";
        ::close(fd);
        return false;
    }

    std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
    ::unlink(path.c_str());

    if (::bind(fd, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) != 0) {
        error = "bind() failed: " + std::string(std::strerror(errno));
        ::close(fd);
        return false;
    }

    if (::listen(fd, backlog) != 0) {
        error = "listen() failed: " + std::string(std::strerror(errno));
        ::close(fd);
        ::unlink(path.c_str());
        return false;
    }

    server_fd = fd;
    return true;
}

bool bind_and_listen(
    CoordinatorTransport transport,
    const std::string& address,
    int backlog,
    int& server_fd,
    std::string& error) {
    if (transport == CoordinatorTransport::Unix) {
        return bind_and_listen_unix_socket(address, backlog, server_fd, error);
    }
    return bind_and_listen_tcp_socket(address, backlog, server_fd, error);
}

bool request_response_unix_socket(
    const std::string& path,
    const std::string& request_line,
    const std::vector<char>& request_payload,
    CoordinatorResponse& response,
    std::string& error) {
    response = CoordinatorResponse{};

    if (path.empty() || path.front() != '/') {
        error = "unix socket path must be absolute";
        return false;
    }

    const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        error = "socket() failed: " + std::string(std::strerror(errno));
        return false;
    }

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (path.size() >= sizeof(addr.sun_path)) {
        error = "unix socket path is too long";
        ::close(fd);
        return false;
    }

    std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
    if (::connect(fd, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) != 0) {
        error = "connect() failed: " + std::string(std::strerror(errno));
        ::close(fd);
        return false;
    }

    if (!send_message_packet(fd, request_line, request_payload, error)) {
        ::close(fd);
        return false;
    }

    std::string response_line;
    if (!receive_message_packet(fd, response_line, response.payload, error)) {
        ::close(fd);
        return false;
    }

    ::close(fd);
    return parse_response_line(response_line, response, error);
}

bool request_response(
    CoordinatorTransport transport,
    const std::string& address,
    const std::string& request_line,
    const std::vector<char>& request_payload,
    CoordinatorResponse& response,
    std::string& error) {
    if (transport == CoordinatorTransport::Unix) {
        return request_response_unix_socket(address, request_line, request_payload, response, error);
    }
    return request_response_tcp_socket(address, request_line, request_payload, response, error);
}

bool request_response(
    CoordinatorTransport transport,
    const std::string& address,
    const std::string& request_line,
    CoordinatorResponse& response,
    std::string& error) {
    static const std::vector<char> kEmptyPayload;
    return request_response(transport, address, request_line, kEmptyPayload, response, error);
}

bool receive_message_line(int fd, std::string& line, std::string& error) {
    line.clear();
    std::array<char, 256> buffer{};

    while (true) {
        const ssize_t rc = ::recv(fd, buffer.data(), buffer.size(), 0);
        if (rc == 0) {
            if (line.empty()) {
                error = "peer closed connection before sending a request";
                return false;
            }
            break;
        }
        if (rc < 0) {
            if (errno == EINTR) {
                continue;
            }
            error = "recv() failed: " + std::string(std::strerror(errno));
            return false;
        }

        line.append(buffer.data(), static_cast<std::size_t>(rc));
        const std::size_t newline = line.find('\n');
        if (newline != std::string::npos) {
            line.resize(newline);
            line = trim(line);
            return !line.empty();
        }

        if (line.size() > 4096) {
            error = "request line is too long";
            return false;
        }
    }

    line = trim(line);
    return !line.empty();
}

bool send_message_line(int fd, const std::string& line, std::string& error) {
    std::string payload = line;
    if (payload.empty() || payload.back() != '\n') {
        payload.push_back('\n');
    }

    return send_all_bytes(fd, payload.data(), payload.size(), error);
}

bool receive_message_packet(
    int fd,
    std::string& line,
    std::vector<char>& payload,
    std::string& error) {
    line.clear();
    payload.clear();

    std::array<char, 1024> buffer{};
    std::vector<char> header_bytes;
    std::vector<char> spill_bytes;

    while (true) {
        const ssize_t rc = ::recv(fd, buffer.data(), buffer.size(), 0);
        if (rc == 0) {
            if (header_bytes.empty()) {
                error = "peer closed connection before sending a request";
                return false;
            }
            error = "peer closed connection before terminating the header";
            return false;
        }
        if (rc < 0) {
            if (errno == EINTR) {
                continue;
            }
            error = "recv() failed: " + std::string(std::strerror(errno));
            return false;
        }

        const char* chunk_begin = buffer.data();
        const char* chunk_end = buffer.data() + static_cast<std::size_t>(rc);
        const char* newline =
            static_cast<const char*>(std::memchr(chunk_begin, '\n', static_cast<std::size_t>(rc)));
        if (!newline) {
            header_bytes.insert(
                header_bytes.end(),
                chunk_begin,
                chunk_end);
            if (header_bytes.size() > 4096) {
                error = "request line is too long";
                return false;
            }
            continue;
        }

        const std::size_t header_chunk =
            static_cast<std::size_t>(newline - chunk_begin);
        header_bytes.insert(header_bytes.end(), chunk_begin, chunk_begin + header_chunk);
        spill_bytes.insert(
            spill_bytes.end(),
            newline + 1,
            chunk_end);
        break;
    }

    line.assign(header_bytes.begin(), header_bytes.end());
    line = trim(line);
    if (line.empty()) {
        error = "empty request line";
        return false;
    }

    std::size_t payload_bytes = 0;
    if (!parse_payload_size_header(line, payload_bytes, error)) {
        return false;
    }
    if (payload_bytes == 0) {
        return true;
    }

    if (spill_bytes.size() > payload_bytes) {
        error = "payload exceeded declared payload_bytes";
        return false;
    }

    payload = std::move(spill_bytes);
    if (payload.size() == payload_bytes) {
        return true;
    }

    std::vector<char> tail;
    if (!receive_exact_bytes(fd, payload_bytes - payload.size(), tail, error)) {
        return false;
    }
    payload.insert(payload.end(), tail.begin(), tail.end());
    return true;
}

bool send_message_packet(
    int fd,
    const std::string& line,
    const std::vector<char>& payload,
    std::string& error) {
    const std::string header = append_payload_header(line, payload);
    if (!send_message_line(fd, header, error)) {
        return false;
    }
    if (payload.empty()) {
        return true;
    }
    return send_all_bytes(fd, payload.data(), payload.size(), error);
}

bool parse_message_line(const std::string& line, CoordinatorMessage& message, std::string& error) {
    const std::vector<char> existing_payload = std::move(message.payload);
    message = CoordinatorMessage{};
    message.payload = existing_payload;
    std::istringstream stream(line);
    if (!(stream >> message.command)) {
        error = "missing command";
        return false;
    }

    std::string token;
    while (stream >> token) {
        const std::size_t equals = token.find('=');
        if (equals == std::string::npos || equals == 0 || equals + 1 >= token.size()) {
            error = "invalid token: " + token;
            return false;
        }
        message.fields[token.substr(0, equals)] = token.substr(equals + 1);
    }
    return true;
}

bool parse_response_line(const std::string& line, CoordinatorResponse& response, std::string& error) {
    const std::vector<char> existing_payload = std::move(response.payload);
    response = CoordinatorResponse{};
    response.payload = existing_payload;

    std::istringstream stream(line);
    std::string status;
    if (!(stream >> status)) {
        error = "missing response status";
        return false;
    }
    if (status != "OK" && status != "ERR") {
        error = "invalid response status: " + status;
        return false;
    }

    std::string token;
    while (stream >> token) {
        const std::size_t equals = token.find('=');
        if (equals == std::string::npos || equals == 0 || equals + 1 >= token.size()) {
            error = "invalid token: " + token;
            return false;
        }
        response.fields[token.substr(0, equals)] = token.substr(equals + 1);
    }

    response.ok = (status == "OK");
    if (!response.ok) {
        auto code_it = response.fields.find("code");
        if (code_it == response.fields.end()) {
            error = "missing error code";
            return false;
        }
        response.error_code = code_it->second;
        auto detail_it = response.fields.find("detail");
        if (detail_it != response.fields.end()) {
            response.error_detail = detail_it->second;
        }
    }

    return true;
}

std::string format_ok_response(const std::unordered_map<std::string, std::string>& fields) {
    std::ostringstream stream;
    stream << "OK";
    for (const auto& entry : fields) {
        stream << ' ' << entry.first << '=' << sanitize_value(entry.second);
    }
    return stream.str();
}

std::string format_error_response(const std::string& code, const std::string& detail) {
    std::ostringstream stream;
    stream << "ERR code=" << sanitize_value(code);
    if (!detail.empty()) {
        stream << " detail=" << sanitize_value(detail);
    }
    return stream.str();
}

}  // namespace fake_gpu::distributed
