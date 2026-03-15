#include "staging_adapter.hpp"

#include "../core/backend_config.hpp"
#include "../core/global_state.hpp"
#include "../cuda/cudart_defs.hpp"

#include <dlfcn.h>
#include <cstring>
#include <string>
#include <vector>

namespace fake_gpu::nccl {

namespace {

using cudaMemcpy_fn = cudaError_t (*)(void*, const void*, size_t, cudaMemcpyKind);
using cudaPointerGetAttributes_fn = cudaError_t (*)(cudaPointerAttributes*, const void*);
using cudaGetErrorString_fn = const char* (*)(cudaError_t);

struct RuntimeCudaApi {
    cudaMemcpy_fn memcpy_fn = nullptr;
    cudaPointerGetAttributes_fn pointer_get_attributes_fn = nullptr;
    cudaGetErrorString_fn get_error_string_fn = nullptr;
};

const RuntimeCudaApi& runtime_cuda_api() {
    static const RuntimeCudaApi api = [] {
        RuntimeCudaApi value;
        value.memcpy_fn = reinterpret_cast<cudaMemcpy_fn>(::dlsym(RTLD_DEFAULT, "cudaMemcpy"));
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

    cudaPointerAttributes attributes {};
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

}  // namespace

bool copy_buffer_to_host(
    const void* src,
    std::size_t bytes,
    std::vector<char>& out,
    std::string& error) {
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
        const RuntimeCudaApi& runtime_api = runtime_cuda_api();
        if (!runtime_api.memcpy_fn) {
            error = "cudaMemcpy is not available for hybrid staging";
            return false;
        }
        const cudaError_t result = runtime_api.memcpy_fn(
            out.data(),
            const_cast<void*>(src),
            bytes,
            cudaMemcpyDeviceToHost);
        if (result != cudaSuccess) {
            error = format_runtime_error("cudaMemcpy(DeviceToHost)", result);
            return false;
        }
        return true;
    }

    std::memcpy(out.data(), src, bytes);
    return true;
}

bool copy_host_to_buffer(
    void* dst,
    const void* src,
    std::size_t bytes,
    std::string& error) {
    error.clear();
    if (bytes == 0) {
        return true;
    }
    if (!dst || !src) {
        error = "destination and source buffers must not be null";
        return false;
    }

    if (use_device_copy_path(dst)) {
        const RuntimeCudaApi& runtime_api = runtime_cuda_api();
        if (!runtime_api.memcpy_fn) {
            error = "cudaMemcpy is not available for hybrid staging";
            return false;
        }
        const cudaError_t result = runtime_api.memcpy_fn(
            dst,
            const_cast<void*>(src),
            bytes,
            cudaMemcpyHostToDevice);
        if (result != cudaSuccess) {
            error = format_runtime_error("cudaMemcpy(HostToDevice)", result);
            return false;
        }
        return true;
    }

    std::memcpy(dst, src, bytes);
    return true;
}

}  // namespace fake_gpu::nccl
