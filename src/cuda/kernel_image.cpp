#include "kernel_image.hpp"
#include "cuda_driver_defs.hpp"

#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <new>

namespace fake_gpu::kernel {
namespace {

template <typename T>
T read(const unsigned char* data) {
    T value;
    std::memcpy(&value, data, sizeof(value));
    return value;
}

// Older CUDA fatbins use raw LZ4 blocks, including overlapping back-references.
bool decompress_lz4(const unsigned char* data, std::size_t size,
                std::size_t expected, std::string& output) {
    std::size_t cursor = 0;
    output.clear();
    auto length = [&](std::size_t& value) {
        if (value != 15) return true;
        unsigned extension;
        do {
            if (cursor == size) return false;
            extension = data[cursor++];
            if (extension > expected || value > expected - extension) return false;
            value += extension;
        } while (extension == 255);
        return true;
    };
    while (cursor < size) {
        const unsigned token = data[cursor++];
        std::size_t literals = token >> 4;
        if (!length(literals) || literals > size - cursor ||
            literals > expected - output.size()) return false;
        output.append(reinterpret_cast<const char*>(data + cursor), literals);
        cursor += literals;
        if (cursor == size) break;
        if (size - cursor < 2) return false;
        const std::size_t offset = read<std::uint16_t>(data + cursor);
        cursor += 2;
        std::size_t count = token & 15;
        if (!length(count) || count > std::numeric_limits<std::size_t>::max() - 4)
            return false;
        count += 4;
        if (!offset || offset > output.size() || count > expected - output.size())
            return false;
        for (std::size_t i = 0; i < count; ++i)
            output.push_back(output[output.size() - offset]);
    }
    return output.size() == expected;
}

int decompress_zstd(const unsigned char* data, std::size_t size, std::string& output) {
    struct Codec {
        unsigned long long (*content_size)(const void*, size_t) = nullptr;
        size_t (*decompress)(void*, size_t, const void*, size_t) = nullptr;
        unsigned (*is_error)(size_t) = nullptr;
    };
    static const Codec codec = [] {
#ifdef __APPLE__
        void* library = dlopen("libzstd.dylib", RTLD_NOW | RTLD_LOCAL);
#else
        void* library = dlopen("libzstd.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
        if (!library) return Codec{};
        return Codec{
            reinterpret_cast<decltype(Codec::content_size)>(dlsym(library, "ZSTD_getFrameContentSize")),
            reinterpret_cast<decltype(Codec::decompress)>(dlsym(library, "ZSTD_decompress")),
            reinterpret_cast<decltype(Codec::is_error)>(dlsym(library, "ZSTD_isError")),
        };
    }();
    if (!codec.content_size || !codec.decompress || !codec.is_error)
        return CUDA_ERROR_NOT_SUPPORTED;
    const auto expected = codec.content_size(data, size);
    if (!expected || expected > output.max_size()) return CUDA_ERROR_INVALID_IMAGE;
    output.resize(expected);
    const size_t decoded = codec.decompress(output.data(), output.size(), data, size);
    return codec.is_error(decoded) || decoded != expected ? CUDA_ERROR_INVALID_IMAGE : CUDA_SUCCESS;
}

int extract(const void* image, std::string& ptx, std::size_t size) {
    if (!image) return CUDA_ERROR_INVALID_VALUE;
    const auto* data = static_cast<const unsigned char*>(image);
    if (data[0] == 0xb1) {
        struct Wrapper {
            std::uint32_t magic;
            std::uint32_t version;
            const void* data;
            void* filename;
        };
        if (size && size < sizeof(Wrapper)) return CUDA_ERROR_INVALID_IMAGE;
        const auto* wrapper = static_cast<const Wrapper*>(image);
        if (wrapper->magic != 0x466243b1) return CUDA_ERROR_INVALID_IMAGE;
        if (wrapper->version != 1) return CUDA_ERROR_NOT_SUPPORTED;
        return extract(wrapper->data, ptx, 0);
    }
    if (data[0] == 0x7f) return CUDA_ERROR_NOT_SUPPORTED; // Cubin contains SASS.
    if (data[0] != 0x50) {
        ptx.assign(reinterpret_cast<const char*>(data),
                   size ? size : std::strlen(reinterpret_cast<const char*>(data)));
        return CUDA_SUCCESS;
    }
    if ((size && size < 16) || read<std::uint32_t>(data) != 0xba55ed50)
        return CUDA_ERROR_INVALID_IMAGE;
    if (read<std::uint16_t>(data + 4) != 1) return CUDA_ERROR_NOT_SUPPORTED;
    const std::size_t header = read<std::uint16_t>(data + 6);
    const std::uint64_t payload = read<std::uint64_t>(data + 8);
    if (header < 16 || payload > std::numeric_limits<std::size_t>::max() - header)
        return CUDA_ERROR_INVALID_IMAGE;
    const std::size_t total = header + payload;
    if (size && total > size) return CUDA_ERROR_INVALID_IMAGE;
    for (std::size_t offset = header; offset < total;) {
        if (total - offset < 16) return CUDA_ERROR_INVALID_IMAGE;
        const auto* entry = data + offset;
        const std::size_t entry_header = read<std::uint32_t>(entry + 4);
        const std::uint64_t entry_size = read<std::uint64_t>(entry + 8);
        if (entry_header < 16 || entry_header > total - offset ||
            entry_size > total - offset - entry_header) return CUDA_ERROR_INVALID_IMAGE;
        if (read<std::uint16_t>(entry) == 1) {
            if (entry_header < 48) return CUDA_ERROR_INVALID_IMAGE;
            const auto* contents = entry + entry_header;
            const std::size_t stored_size = read<std::uint32_t>(entry + 16);
            const std::size_t content_size = stored_size ? stored_size : entry_size;
            if (content_size > entry_size) return CUDA_ERROR_INVALID_IMAGE;
            const auto flags = read<std::uint64_t>(entry + 40);
            if (flags & 0x8000) {
                const int result = decompress_zstd(contents, content_size, ptx);
                if (result != CUDA_SUCCESS) return result;
            } else if (flags & 0x2000) {
                if (entry_header < 64) return CUDA_ERROR_INVALID_IMAGE;
                const std::uint64_t expected = read<std::uint64_t>(entry + 56);
                if (expected > ptx.max_size() ||
                    !decompress_lz4(contents, content_size, expected, ptx))
                    return CUDA_ERROR_INVALID_IMAGE;
            } else {
                ptx.assign(reinterpret_cast<const char*>(contents), content_size);
            }
            const auto terminator = ptx.find('\0');
            if (terminator != std::string::npos) ptx.resize(terminator);
            return CUDA_SUCCESS;
        }
        offset += entry_header + entry_size;
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

} // namespace

int extract_ptx(const void* image, std::string& ptx, std::size_t size) {
    try {
        return extract(image, ptx, size);
    } catch (const std::bad_alloc&) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
}

} // namespace fake_gpu::kernel
