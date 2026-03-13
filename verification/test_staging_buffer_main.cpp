#include "../src/distributed/staging_buffer.hpp"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using fake_gpu::distributed::StagingBufferHandle;
using fake_gpu::distributed::StagingBufferManager;
using fake_gpu::distributed::StagingBufferMetadata;

bool has_flag(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == flag) {
            return true;
        }
    }
    return false;
}

std::string get_arg(int argc, char** argv, const char* flag) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == flag) {
            return argv[i + 1];
        }
    }
    throw std::runtime_error(std::string("missing argument: ") + flag);
}

void wait_for_file(const std::string& path, const std::string& description) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
        if (std::filesystem::exists(path)) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    throw std::runtime_error("timeout waiting for " + description);
}

void write_file(const std::string& path, const std::string& content) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to open file: " + path);
    }
    output << content;
}

std::vector<unsigned char> parse_hex(const std::string& hex) {
    if (hex.size() % 2 != 0) {
        throw std::runtime_error("hex payload must contain an even number of characters");
    }

    auto decode = [](char ch) -> int {
        if (ch >= '0' && ch <= '9') {
            return ch - '0';
        }
        if (ch >= 'a' && ch <= 'f') {
            return 10 + (ch - 'a');
        }
        if (ch >= 'A' && ch <= 'F') {
            return 10 + (ch - 'A');
        }
        return -1;
    };

    std::vector<unsigned char> bytes;
    bytes.reserve(hex.size() / 2);
    for (std::size_t index = 0; index < hex.size(); index += 2) {
        const int high = decode(hex[index]);
        const int low = decode(hex[index + 1]);
        if (high < 0 || low < 0) {
            throw std::runtime_error("hex payload contains an invalid character");
        }
        bytes.push_back(static_cast<unsigned char>((high << 4) | low));
    }
    return bytes;
}

StagingBufferMetadata make_metadata(const std::string& name, std::size_t bytes) {
    StagingBufferMetadata metadata;
    metadata.name = name;
    metadata.dtype = "uint8";
    metadata.bytes = bytes;
    metadata.shape = {bytes};
    metadata.owner_rank = 0;
    metadata.staging_id = 1;
    return metadata;
}

int writer_main(int argc, char** argv) {
    const std::string name = get_arg(argc, argv, "--name");
    const std::string ready_file = get_arg(argc, argv, "--ready-file");
    const std::string done_file = get_arg(argc, argv, "--done-file");
    const std::vector<unsigned char> payload = parse_hex(get_arg(argc, argv, "--payload-hex"));

    StagingBufferManager manager;
    StagingBufferHandle handle;
    std::string error;
    if (!manager.create(make_metadata(name, payload.size()), true, handle, error)) {
        throw std::runtime_error(error);
    }

    std::memcpy(handle.data(), payload.data(), payload.size());
    write_file(ready_file, "ready");
    wait_for_file(done_file, "reader completion marker");
    return 0;
}

int reader_main(int argc, char** argv) {
    const std::string name = get_arg(argc, argv, "--name");
    const std::string ready_file = get_arg(argc, argv, "--ready-file");
    const std::string done_file = get_arg(argc, argv, "--done-file");
    const std::vector<unsigned char> expected = parse_hex(get_arg(argc, argv, "--expected-hex"));

    wait_for_file(ready_file, "writer readiness marker");

    StagingBufferManager manager;
    StagingBufferHandle handle;
    std::string error;
    if (!manager.open(make_metadata(name, expected.size()), true, handle, error)) {
        throw std::runtime_error(error);
    }

    if (std::memcmp(handle.data(), expected.data(), expected.size()) != 0) {
        throw std::runtime_error("reader observed unexpected staging buffer contents");
    }

    write_file(done_file, "done");
    return 0;
}

int probe_missing_main(int argc, char** argv) {
    const std::string name = get_arg(argc, argv, "--name");

    StagingBufferManager manager;
    StagingBufferHandle handle;
    std::string error;
    if (manager.open(make_metadata(name, 1), true, handle, error)) {
        throw std::runtime_error("shared memory segment still exists");
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (has_flag(argc, argv, "--write")) {
            return writer_main(argc, argv);
        }
        if (has_flag(argc, argv, "--read")) {
            return reader_main(argc, argv);
        }
        if (has_flag(argc, argv, "--probe-missing")) {
            return probe_missing_main(argc, argv);
        }
        throw std::runtime_error("expected one of --write, --read or --probe-missing");
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
