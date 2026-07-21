#include "staging_adapter.hpp"

#include "../core/backend_config.hpp"
#include "../core/global_state.hpp"
#include "../cuda/cudart_defs.hpp"

#include <cstdint>
#include <dlfcn.h>
#include <cstring>
#include <string>
#include <vector>

namespace fake_gpu::nccl {

namespace {

using cudaMemcpy_fn = cudaError_t (*)(void*, const void*, size_t, cudaMemcpyKind);
using cudaMemcpyAsync_fn =
    cudaError_t (*)(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);
using cudaStreamSynchronize_fn = cudaError_t (*)(cudaStream_t);
using cudaPointerGetAttributes_fn = cudaError_t (*)(void*, const void*);
using cudaGetErrorString_fn = const char* (*)(cudaError_t);

// CUDA 13 extended cudaPointerAttributes with reserved[8], increasing its
// Linux ABI size from 32 to 96 bytes. FakeGPU builds without CUDA headers, so
// use an intentionally oversized private buffer when calling the physical
// Runtime. The stable prefix remains readable on CUDA 12 and CUDA 13.
struct RuntimeCudaPointerAttributes {
    cudaMemoryType type = cudaMemoryTypeUnregistered;
    int device = -1;
    void* device_pointer = nullptr;
    void* host_pointer = nullptr;
    std::uint64_t reserved[16] {};
};

static_assert(
    sizeof(RuntimeCudaPointerAttributes) >= 96,
    "runtime pointer-attribute storage must cover the CUDA 13 ABI");

struct RuntimeCudaApi {
    cudaMemcpy_fn memcpy_fn = nullptr;
    cudaMemcpyAsync_fn memcpy_async_fn = nullptr;
    cudaStreamSynchronize_fn stream_synchronize_fn = nullptr;
    cudaPointerGetAttributes_fn pointer_get_attributes_fn = nullptr;
    cudaGetErrorString_fn get_error_string_fn = nullptr;
};

const RuntimeCudaApi& runtime_cuda_api() {
    static const RuntimeCudaApi api = [] {
        RuntimeCudaApi value;
        value.memcpy_fn = reinterpret_cast<cudaMemcpy_fn>(::dlsym(RTLD_DEFAULT, "cudaMemcpy"));
        value.memcpy_async_fn = reinterpret_cast<cudaMemcpyAsync_fn>(
            ::dlsym(RTLD_DEFAULT, "cudaMemcpyAsync"));
        value.stream_synchronize_fn = reinterpret_cast<cudaStreamSynchronize_fn>(
            ::dlsym(RTLD_DEFAULT, "cudaStreamSynchronize"));
        value.pointer_get_attributes_fn = reinterpret_cast<cudaPointerGetAttributes_fn>(
            ::dlsym(RTLD_DEFAULT, "cudaPointerGetAttributes"));
        value.get_error_string_fn = reinterpret_cast<cudaGetErrorString_fn>(
            ::dlsym(RTLD_DEFAULT, "cudaGetErrorString"));
        return value;
    }();
    return api;
}

bool is_runtime_device_pointer(const void* ptr) {
    const RuntimeCudaApi& api = runtime_cuda_api();
    if (!api.pointer_get_attributes_fn || !ptr) {
        return false;
    }

    RuntimeCudaPointerAttributes attributes {};
    const cudaError_t result = api.pointer_get_attributes_fn(&attributes, ptr);
    if (result != cudaSuccess) {
        return false;
    }

    return attributes.type == cudaMemoryTypeDevice ||
           attributes.type == cudaMemoryTypeManaged;
}

bool use_device_copy_path(const void* ptr) {
    if (!ptr) {
        return false;
    }
    if (fake_gpu::BackendConfig::instance().mode() != fake_gpu::FakeGpuMode::Hybrid) {
        return false;
    }

    std::size_t alloc_size = 0;
    int alloc_device = -1;
    fake_gpu::GlobalState::AllocationKind alloc_kind =
        fake_gpu::GlobalState::AllocationKind::Device;
    if (!fake_gpu::GlobalState::instance().get_allocation_info_ex(
            const_cast<void*>(ptr), alloc_size, alloc_device, alloc_kind)) {
        return is_runtime_device_pointer(ptr);
    }
    return alloc_kind != fake_gpu::GlobalState::AllocationKind::Host;
}

std::string format_runtime_error(const char* api, cudaError_t result) {
    const RuntimeCudaApi& runtime_api = runtime_cuda_api();
    const char* message = "unknown";
    if (runtime_api.get_error_string_fn) {
        message = runtime_api.get_error_string_fn(result);
    }
    return std::string(api) + " failed: " + message;
}

bool copy_device_buffer(
    void* dst,
    const void* src,
    std::size_t bytes,
    cudaMemcpyKind kind,
    cudaStream_t stream,
    std::string& error) {
    const RuntimeCudaApi& api = runtime_cuda_api();
    if (stream != nullptr) {
        if (!api.memcpy_async_fn || !api.stream_synchronize_fn) {
            error = (
                "cudaMemcpyAsync and cudaStreamSynchronize are required for "
                "hybrid staging on a non-default stream");
            return false;
        }
        cudaError_t result = api.memcpy_async_fn(dst, src, bytes, kind, stream);
        if (result != cudaSuccess) {
            error = format_runtime_error("cudaMemcpyAsync", result);
            return false;
        }
        result = api.stream_synchronize_fn(stream);
        if (result != cudaSuccess) {
            error = format_runtime_error("cudaStreamSynchronize", result);
            return false;
        }
        return true;
    }

    if (!api.memcpy_fn) {
        error = "cudaMemcpy is not available for hybrid staging";
        return false;
    }
    const cudaError_t result = api.memcpy_fn(dst, src, bytes, kind);
    if (result != cudaSuccess) {
        error = format_runtime_error("cudaMemcpy", result);
        return false;
    }
    return true;
}

}  // namespace

bool copy_buffer_to_host(
    const void* src,
    std::size_t bytes,
    std::vector<char>& out,
    std::string& error,
    cudaStream_t stream) {
    error.clear();
    out.assign(bytes, 0);
    if (bytes == 0) {
        return true;
    }
    if (!src) {
        error = "source buffer must not be null";
        return false;
    }

    if (use_device_copy_path(src)) {
        return copy_device_buffer(
            out.data(),
            const_cast<void*>(src),
            bytes,
            cudaMemcpyDeviceToHost,
            stream,
            error);
    }

    std::memcpy(out.data(), src, bytes);
    return true;
}

bool copy_host_to_buffer(
    void* dst,
    const void* src,
    std::size_t bytes,
    std::string& error,
    cudaStream_t stream) {
    error.clear();
    if (bytes == 0) {
        return true;
    }
    if (!dst || !src) {
        error = "destination and source buffers must not be null";
        return false;
    }

    if (use_device_copy_path(dst)) {
        return copy_device_buffer(
            dst,
            const_cast<void*>(src),
            bytes,
            cudaMemcpyHostToDevice,
            stream,
            error);
    }

    std::memcpy(dst, src, bytes);
    return true;
}

}  // namespace fake_gpu::nccl
