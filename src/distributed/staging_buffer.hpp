#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fake_gpu::distributed {

struct StagingBufferMetadata {
    std::string name;
    std::string dtype;
    std::size_t bytes = 0;
    std::vector<std::size_t> shape;
    int owner_rank = -1;
    std::uint64_t staging_id = 0;
};

std::string default_staging_buffer_name(int owner_rank, std::uint64_t staging_id);
bool is_staging_transport_unavailable_error(const std::string& error);

class StagingBufferHandle {
public:
    StagingBufferHandle() = default;
    ~StagingBufferHandle();

    StagingBufferHandle(const StagingBufferHandle&) = delete;
    StagingBufferHandle& operator=(const StagingBufferHandle&) = delete;
    StagingBufferHandle(StagingBufferHandle&& other) noexcept;
    StagingBufferHandle& operator=(StagingBufferHandle&& other) noexcept;

    void* data() { return mapping_; }
    const void* data() const { return mapping_; }

    bool close(std::string& error);
    bool unlink(std::string& error);

private:
    friend class StagingBufferManager;

    void reset();

    int fd_ = -1;
    void* mapping_ = nullptr;
    bool unlinked_ = false;
    bool unlink_on_close_ = false;
    StagingBufferMetadata metadata_;
};

class StagingBufferManager {
public:
    bool create(
        const StagingBufferMetadata& metadata,
        bool unlink_on_close,
        StagingBufferHandle& out,
        std::string& error) const;

    bool open(
        const StagingBufferMetadata& metadata,
        bool read_only,
        StagingBufferHandle& out,
        std::string& error) const;

    bool release(const std::string& name, std::string& error) const;
};

}  // namespace fake_gpu::distributed
