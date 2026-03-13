#include "staging_buffer.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <utility>

namespace fake_gpu::distributed {

namespace {

bool validate_name(const std::string& name, std::string& error) {
    if (name.empty() || name.front() != '/') {
        error = "staging buffer name must start with /";
        return false;
    }
    if (name.size() == 1) {
        error = "staging buffer name must not be empty";
        return false;
    }
    if (name.find('/', 1) != std::string::npos) {
        error = "staging buffer name must not contain nested slashes";
        return false;
    }
    return true;
}

bool validate_metadata(const StagingBufferMetadata& metadata, std::string& error) {
    if (!validate_name(metadata.name, error)) {
        return false;
    }
    if (metadata.bytes == 0) {
        error = "staging buffer bytes must be > 0";
        return false;
    }
    if (metadata.dtype.empty()) {
        error = "staging buffer dtype must be set";
        return false;
    }
    if (metadata.owner_rank < 0) {
        error = "staging buffer owner_rank must be >= 0";
        return false;
    }
    return true;
}

bool unlink_name(const std::string& name, bool missing_ok, std::string& error) {
    if (::shm_unlink(name.c_str()) == 0) {
        return true;
    }
    if (errno == ENOENT && missing_ok) {
        return true;
    }
    error = "shm_unlink() failed: " + std::string(std::strerror(errno));
    return false;
}

}  // namespace

std::string default_staging_buffer_name(int owner_rank, std::uint64_t staging_id) {
    return "/fakegpu-staging-r" + std::to_string(owner_rank) + "-s" + std::to_string(staging_id);
}

StagingBufferHandle::~StagingBufferHandle() {
    std::string ignored_error;
    close(ignored_error);
}

StagingBufferHandle::StagingBufferHandle(StagingBufferHandle&& other) noexcept {
    *this = std::move(other);
}

StagingBufferHandle& StagingBufferHandle::operator=(StagingBufferHandle&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    std::string ignored_error;
    close(ignored_error);

    fd_ = other.fd_;
    mapping_ = other.mapping_;
    read_only_ = other.read_only_;
    unlinked_ = other.unlinked_;
    unlink_on_close_ = other.unlink_on_close_;
    metadata_ = std::move(other.metadata_);

    other.reset();
    return *this;
}

bool StagingBufferHandle::close(std::string& error) {
    error.clear();

    bool ok = true;
    if (mapping_) {
        if (::munmap(mapping_, metadata_.bytes) != 0 && ok) {
            error = "munmap() failed: " + std::string(std::strerror(errno));
            ok = false;
        }
        mapping_ = nullptr;
    }

    if (fd_ >= 0) {
        if (::close(fd_) != 0 && ok) {
            error = "close() failed: " + std::string(std::strerror(errno));
            ok = false;
        }
        fd_ = -1;
    }

    if (unlink_on_close_ && !metadata_.name.empty() && !unlinked_) {
        std::string unlink_error;
        if (!unlink_name(metadata_.name, true, unlink_error) && ok) {
            error = std::move(unlink_error);
            ok = false;
        } else {
            unlinked_ = true;
        }
    }

    if (ok) {
        reset();
    }

    return ok;
}

bool StagingBufferHandle::unlink(std::string& error) {
    error.clear();
    if (metadata_.name.empty()) {
        error = "staging buffer is not initialized";
        return false;
    }
    if (unlinked_) {
        return true;
    }
    if (!unlink_name(metadata_.name, true, error)) {
        return false;
    }
    unlinked_ = true;
    return true;
}

void StagingBufferHandle::reset() {
    fd_ = -1;
    mapping_ = nullptr;
    read_only_ = false;
    unlinked_ = false;
    unlink_on_close_ = false;
    metadata_ = StagingBufferMetadata{};
}

bool StagingBufferManager::create(
    const StagingBufferMetadata& metadata,
    bool unlink_on_close,
    StagingBufferHandle& out,
    std::string& error) const {
    error.clear();

    if (!validate_metadata(metadata, error)) {
        return false;
    }

    std::string ignored_error;
    out.close(ignored_error);

    const int fd = ::shm_open(metadata.name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0) {
        error = "shm_open() failed: " + std::string(std::strerror(errno));
        return false;
    }

    if (::ftruncate(fd, static_cast<off_t>(metadata.bytes)) != 0) {
        error = "ftruncate() failed: " + std::string(std::strerror(errno));
        ::close(fd);
        ::shm_unlink(metadata.name.c_str());
        return false;
    }

    void* mapping = ::mmap(nullptr, metadata.bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapping == MAP_FAILED) {
        error = "mmap() failed: " + std::string(std::strerror(errno));
        ::close(fd);
        ::shm_unlink(metadata.name.c_str());
        return false;
    }

    out.fd_ = fd;
    out.mapping_ = mapping;
    out.read_only_ = false;
    out.unlinked_ = false;
    out.unlink_on_close_ = unlink_on_close;
    out.metadata_ = metadata;
    return true;
}

bool StagingBufferManager::open(
    const StagingBufferMetadata& metadata,
    bool read_only,
    StagingBufferHandle& out,
    std::string& error) const {
    error.clear();

    if (!validate_name(metadata.name, error)) {
        return false;
    }

    std::string ignored_error;
    out.close(ignored_error);

    const int flags = read_only ? O_RDONLY : O_RDWR;
    const int fd = ::shm_open(metadata.name.c_str(), flags, 0600);
    if (fd < 0) {
        error = "shm_open() failed: " + std::string(std::strerror(errno));
        return false;
    }

    struct stat stats {};
    if (::fstat(fd, &stats) != 0) {
        error = "fstat() failed: " + std::string(std::strerror(errno));
        ::close(fd);
        return false;
    }

    StagingBufferMetadata resolved = metadata;
    if (resolved.bytes == 0) {
        resolved.bytes = static_cast<std::size_t>(stats.st_size);
    }
    if (resolved.bytes == 0) {
        error = "staging buffer bytes must be > 0";
        ::close(fd);
        return false;
    }
    if (static_cast<off_t>(resolved.bytes) > stats.st_size) {
        error = "staging buffer bytes exceed shared memory size";
        ::close(fd);
        return false;
    }

    const int prot = read_only ? PROT_READ : (PROT_READ | PROT_WRITE);
    void* mapping = ::mmap(nullptr, resolved.bytes, prot, MAP_SHARED, fd, 0);
    if (mapping == MAP_FAILED) {
        error = "mmap() failed: " + std::string(std::strerror(errno));
        ::close(fd);
        return false;
    }

    out.fd_ = fd;
    out.mapping_ = mapping;
    out.read_only_ = read_only;
    out.unlinked_ = false;
    out.unlink_on_close_ = false;
    out.metadata_ = std::move(resolved);
    return true;
}

bool StagingBufferManager::release(const std::string& name, std::string& error) const {
    error.clear();
    if (!validate_name(name, error)) {
        return false;
    }
    return unlink_name(name, false, error);
}

}  // namespace fake_gpu::distributed
