#include "kernel_image.hpp"
#include "kernel_executor.hpp"
#include "cuda_driver_defs.hpp"

#include <algorithm>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <new>
#include <vector>

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

int extract_elf(const unsigned char* data, std::string& code, unsigned cc, std::size_t size) {
    if ((size && size < 64) || std::memcmp(data, "\x7f" "ELF", 4) != 0 ||
        data[4] != 2 || data[5] != 1 || read<std::uint16_t>(data + 18) != 190)
        return CUDA_ERROR_INVALID_IMAGE;
    if (read<std::uint16_t>(data + 16) != 2) return CUDA_ERROR_NOT_SUPPORTED;

    // cuModuleLoadData supplies no length. ELF tables describe the image's
    // extent; file/fatbin callers also provide a bound checked before each read.
    std::size_t extent = 64;
    auto include = [&](std::uint64_t offset, std::uint64_t bytes) {
        if (offset > code.max_size() || bytes > code.max_size() - offset ||
            (size && (offset > size || bytes > size - offset))) return false;
        extent = std::max(extent, static_cast<std::size_t>(offset + bytes));
        return true;
    };
    const auto program_offset = read<std::uint64_t>(data + 32);
    const auto section_offset = read<std::uint64_t>(data + 40);
    const auto program_entry = read<std::uint16_t>(data + 54);
    const auto program_count = read<std::uint16_t>(data + 56);
    const auto section_entry = read<std::uint16_t>(data + 58);
    const auto section_count = read<std::uint16_t>(data + 60);
    if ((program_count && (program_entry < 56 ||
         !include(program_offset, std::uint64_t{program_entry} * program_count))) ||
        (section_count && (section_entry < 64 ||
         !include(section_offset, std::uint64_t{section_entry} * section_count))))
        return CUDA_ERROR_INVALID_IMAGE;
    for (unsigned i = 0; i < program_count; ++i) {
        const auto* program = data + program_offset + std::size_t{i} * program_entry;
        if (!include(read<std::uint64_t>(program + 8), read<std::uint64_t>(program + 32)))
            return CUDA_ERROR_INVALID_IMAGE;
    }
    for (unsigned i = 0; i < section_count; ++i) {
        const auto* section = data + section_offset + std::size_t{i} * section_entry;
        if (read<std::uint32_t>(section + 4) != 8 && // SHT_NOBITS has no file bytes.
            !include(read<std::uint64_t>(section + 24), read<std::uint64_t>(section + 32)))
            return CUDA_ERROR_INVALID_IMAGE;
    }
    code.assign(reinterpret_cast<const char*>(data), extent);
    return cubin_compatible(code, cc);
}

struct Entry {
    const unsigned char* data;
    std::size_t header;
    std::size_t size;
    unsigned kind;
    unsigned arch;
};

int decode(const Entry& entry, std::string& code) {
    const auto* contents = entry.data + entry.header;
    const std::size_t stored_size = read<std::uint32_t>(entry.data + 16);
    const std::size_t content_size = stored_size ? stored_size : entry.size;
    if (content_size > entry.size) return CUDA_ERROR_INVALID_IMAGE;
    const auto flags = read<std::uint64_t>(entry.data + 40);
    if (flags & 0x8000) return decompress_zstd(contents, content_size, code);
    if (flags & 0x2000) {
        if (entry.header < 64) return CUDA_ERROR_INVALID_IMAGE;
        const std::uint64_t expected = read<std::uint64_t>(entry.data + 56);
        if (expected > code.max_size() || !decompress_lz4(contents, content_size, expected, code))
            return CUDA_ERROR_INVALID_IMAGE;
    } else {
        code.assign(reinterpret_cast<const char*>(contents), content_size);
    }
    return CUDA_SUCCESS;
}

int extract(const void* image, std::string& code, unsigned cc, std::size_t size) {
    if (!image) return CUDA_ERROR_INVALID_VALUE;
    const auto* data = static_cast<const unsigned char*>(image);
    if (data[0] == 0xb1) {
        // The runtime wrapper holds process pointers, never an on-disk image.
        if (size) return CUDA_ERROR_INVALID_IMAGE;
        struct Wrapper {
            std::uint32_t magic;
            std::uint32_t version;
            const void* data;
            void* filename;
        };
        const auto* wrapper = static_cast<const Wrapper*>(image);
        if (wrapper->magic != 0x466243b1) return CUDA_ERROR_INVALID_IMAGE;
        // Version 2 keeps the linked image in data; its last field points to
        // prelink inputs instead of the version 1 offline filename.
        if (wrapper->version != 1 && wrapper->version != 2) return CUDA_ERROR_NOT_SUPPORTED;
        return extract(wrapper->data, code, cc, 0);
    }
    if (data[0] == 0x7f) return extract_elf(data, code, cc, size);
    if (data[0] != 0x50) {
        code.assign(reinterpret_cast<const char*>(data),
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
    std::vector<Entry> entries;
    for (std::size_t offset = header; offset < total;) {
        if (total - offset < 16) return CUDA_ERROR_INVALID_IMAGE;
        const auto* entry = data + offset;
        const std::size_t entry_header = read<std::uint32_t>(entry + 4);
        const std::uint64_t entry_size = read<std::uint64_t>(entry + 8);
        if (entry_header < 16 || entry_header > total - offset ||
            entry_size > total - offset - entry_header) return CUDA_ERROR_INVALID_IMAGE;
        const auto kind = read<std::uint16_t>(entry);
        if (kind == 1 || kind == 2) {
            if (entry_header < 48) return CUDA_ERROR_INVALID_IMAGE;
            entries.push_back({entry, entry_header, static_cast<std::size_t>(entry_size),
                               kind, read<std::uint32_t>(entry + 28)});
        }
        offset += entry_header + entry_size;
    }
    std::stable_sort(entries.begin(), entries.end(), [](const Entry& left, const Entry& right) {
        return left.arch > right.arch;
    });
    for (const auto& entry : entries) {
        if (entry.kind != 1 || entry.arch > cc) continue;
        int result = decode(entry, code);
        if (result != CUDA_SUCCESS) return result;
        const auto terminator = code.find('\0');
        if (terminator != std::string::npos) code.resize(terminator);
        result = ptx_compatible(code, cc);
        if (result == CUDA_ERROR_NO_BINARY_FOR_GPU) continue;
        return result;
    }
    unsigned best_arch = 0;
    for (const auto& entry : entries) {
        if (entry.kind != 2 ||
            (entry.arch && (entry.arch / 10 != cc / 10 || entry.arch > cc))) continue;
        std::string stored, cubin;
        int result = decode(entry, stored);
        if (result != CUDA_SUCCESS) return result;
        if (stored.size() < 64) return CUDA_ERROR_INVALID_IMAGE;
        result = extract_elf(reinterpret_cast<const unsigned char*>(stored.data()), cubin, cc, stored.size());
        if (result == CUDA_ERROR_NO_BINARY_FOR_GPU) continue;
        if (result != CUDA_SUCCESS) return result;
        const auto flags = read<std::uint32_t>(reinterpret_cast<const unsigned char*>(cubin.data()) + 48);
        const unsigned arch = (cubin[8] >= 8 ? flags >> 8 : flags) & 0xff;
        if (arch > best_arch) {
            best_arch = arch;
            code = std::move(cubin);
        }
    }
    if (best_arch) return CUDA_SUCCESS;
    return entries.empty() ? CUDA_ERROR_NOT_SUPPORTED : CUDA_ERROR_NO_BINARY_FOR_GPU;
}

} // namespace

int extract_image(const void* image, std::string& code, unsigned compute_capability, std::size_t size) {
    try {
        return extract(image, code, compute_capability, size);
    } catch (const std::bad_alloc&) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
}

} // namespace fake_gpu::kernel
