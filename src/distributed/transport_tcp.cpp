#include "transport.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <string>

namespace fake_gpu::distributed {

namespace {

bool connect_tcp_socket(const std::string& host, uint16_t port, int& fd, std::string& error) {
    fd = -1;

    addrinfo hints {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo* result = nullptr;
    const std::string service = std::to_string(port);
    const int rc = ::getaddrinfo(host.c_str(), service.c_str(), &hints, &result);
    if (rc != 0) {
        error = "getaddrinfo() failed: " + std::string(::gai_strerror(rc));
        return false;
    }

    for (addrinfo* it = result; it != nullptr; it = it->ai_next) {
        const int candidate = ::socket(it->ai_family, it->ai_socktype, it->ai_protocol);
        if (candidate < 0) {
            continue;
        }
        if (::connect(candidate, it->ai_addr, it->ai_addrlen) == 0) {
            fd = candidate;
            ::freeaddrinfo(result);
            return true;
        }
        error = "connect() failed: " + std::string(std::strerror(errno));
        ::close(candidate);
    }

    ::freeaddrinfo(result);
    if (error.empty()) {
        error = "connect() failed";
    }
    return false;
}

}  // namespace

bool bind_and_listen_tcp_socket(const std::string& endpoint, int backlog, int& server_fd, std::string& error) {
    server_fd = -1;

    std::string host;
    uint16_t port = 0;
    if (!parse_tcp_endpoint(endpoint, host, port, error)) {
        error = "invalid tcp endpoint: " + endpoint + " (" + error + ")";
        return false;
    }

    addrinfo hints {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;

    addrinfo* result = nullptr;
    const std::string service = std::to_string(port);
    const int rc = ::getaddrinfo(host.c_str(), service.c_str(), &hints, &result);
    if (rc != 0) {
        error = "getaddrinfo() failed: " + std::string(::gai_strerror(rc));
        return false;
    }

    for (addrinfo* it = result; it != nullptr; it = it->ai_next) {
        const int fd = ::socket(it->ai_family, it->ai_socktype, it->ai_protocol);
        if (fd < 0) {
            continue;
        }

        int reuse_addr = 1;
        ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse_addr, sizeof(reuse_addr));

        if (::bind(fd, it->ai_addr, it->ai_addrlen) != 0) {
            error = "bind() failed: " + std::string(std::strerror(errno));
            ::close(fd);
            continue;
        }
        if (::listen(fd, backlog) != 0) {
            error = "listen() failed: " + std::string(std::strerror(errno));
            ::close(fd);
            continue;
        }

        server_fd = fd;
        ::freeaddrinfo(result);
        return true;
    }

    ::freeaddrinfo(result);
    if (error.empty()) {
        error = "failed to bind any tcp address for " + endpoint;
    }
    return false;
}

bool request_response_tcp_socket(
    const std::string& endpoint,
    const std::string& request_line,
    CoordinatorResponse& response,
    std::string& error) {
    response = CoordinatorResponse{};

    std::string host;
    uint16_t port = 0;
    if (!parse_tcp_endpoint(endpoint, host, port, error)) {
        error = "invalid tcp endpoint: " + endpoint + " (" + error + ")";
        return false;
    }

    int fd = -1;
    if (!connect_tcp_socket(host, port, fd, error)) {
        return false;
    }

    if (!send_message_line(fd, request_line, error)) {
        ::close(fd);
        return false;
    }

    std::string response_line;
    if (!receive_message_line(fd, response_line, error)) {
        ::close(fd);
        return false;
    }

    ::close(fd);
    return parse_response_line(response_line, response, error);
}

}  // namespace fake_gpu::distributed
