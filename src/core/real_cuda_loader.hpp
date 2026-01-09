#pragma once

#include <dlfcn.h>
#include <cstdio>
#include <string>
#include <mutex>
#include "backend_config.hpp"
#include "logging.hpp"

namespace fake_gpu {

// Real CUDA library loader - loads and provides access to real CUDA functions
class RealCudaLoader {
public:
    static RealCudaLoader& instance() {
        static RealCudaLoader loader;
        return loader;
    }

    // Check if real libraries are available
    bool is_available() const { return cuda_driver_handle_ != nullptr; }
    bool has_cudart() const { return cudart_handle_ != nullptr; }
    bool has_cublas() const { return cublas_handle_ != nullptr; }
    bool has_nvml() const { return nvml_handle_ != nullptr; }

    // Get function pointers from real libraries
    void* get_cuda_driver_func(const char* name) {
        if (!cuda_driver_handle_) return nullptr;
        return dlsym(cuda_driver_handle_, name);
    }

    void* get_cudart_func(const char* name) {
        if (!cudart_handle_) return nullptr;
        return dlsym(cudart_handle_, name);
    }

    void* get_cublas_func(const char* name) {
        if (!cublas_handle_) return nullptr;
        return dlsym(cublas_handle_, name);
    }

    void* get_nvml_func(const char* name) {
        if (!nvml_handle_) return nullptr;
        return dlsym(nvml_handle_, name);
    }

    // Initialize/load real libraries
    bool initialize() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (initialized_) return is_available();
        initialized_ = true;

        const BackendConfig& config = BackendConfig::instance();

        // Only load if we need real libraries
        if (!config.use_real_cuda()) {
            FGPU_LOG("[RealCudaLoader] Mode is simulate, not loading real libraries\n");
            return false;
        }

        FGPU_LOG("[RealCudaLoader] Attempting to load real CUDA libraries...\n");

        // Load CUDA driver library first (libcuda.so)
        cuda_driver_handle_ = load_library(config.real_cuda_driver_path(), "libcuda.so");
        if (!cuda_driver_handle_) {
            // Try system default
            cuda_driver_handle_ = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        }

        if (cuda_driver_handle_) {
            FGPU_LOG("[RealCudaLoader] Loaded real CUDA driver library\n");
        } else {
            FGPU_LOG("[RealCudaLoader] WARNING: Could not load real CUDA driver library\n");
            return false;
        }

        // Load CUDA runtime library (libcudart.so)
        cudart_handle_ = load_library(config.real_cudart_path(), "libcudart.so");
        if (!cudart_handle_) {
            cudart_handle_ = dlopen("libcudart.so.12", RTLD_NOW | RTLD_LOCAL);
        }
        if (!cudart_handle_) {
            cudart_handle_ = dlopen("libcudart.so.11", RTLD_NOW | RTLD_LOCAL);
        }

        if (cudart_handle_) {
            FGPU_LOG("[RealCudaLoader] Loaded real CUDA runtime library\n");
        }

        // Load cuBLAS library
        cublas_handle_ = load_library(config.real_cublas_path(), "libcublas.so");
        if (!cublas_handle_) {
            cublas_handle_ = dlopen("libcublas.so.12", RTLD_NOW | RTLD_LOCAL);
        }
        if (!cublas_handle_) {
            cublas_handle_ = dlopen("libcublas.so.11", RTLD_NOW | RTLD_LOCAL);
        }

        if (cublas_handle_) {
            FGPU_LOG("[RealCudaLoader] Loaded real cuBLAS library\n");
        }

        // Load NVML library
        nvml_handle_ = load_library(config.real_nvml_path(), "libnvidia-ml.so");
        if (!nvml_handle_) {
            nvml_handle_ = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
        }

        if (nvml_handle_) {
            FGPU_LOG("[RealCudaLoader] Loaded real NVML library\n");
        }

        return is_available();
    }

    // Query real GPU information
    struct RealGpuInfo {
        int device_count = 0;
        size_t total_memory[16] = {0};  // Per-device total memory
        size_t free_memory[16] = {0};   // Per-device free memory
        int compute_major[16] = {0};
        int compute_minor[16] = {0};
        char name[16][256] = {{0}};
        bool valid = false;
    };

    const RealGpuInfo& get_real_gpu_info() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!real_gpu_info_cached_) {
            query_real_gpu_info();
            real_gpu_info_cached_ = true;
        }
        return real_gpu_info_;
    }

    // Refresh real GPU info (e.g., after memory changes)
    void refresh_real_gpu_info() {
        std::lock_guard<std::mutex> lock(mutex_);
        query_real_gpu_info();
    }

private:
    RealCudaLoader() = default;
    ~RealCudaLoader() {
        // Don't unload libraries - they may still be in use
    }

    void* load_library(const std::string& path, const char* fallback_name) {
        if (!path.empty()) {
            void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (handle) {
                FGPU_LOG("[RealCudaLoader] Loaded %s from %s\n", fallback_name, path.c_str());
                return handle;
            }
            FGPU_LOG("[RealCudaLoader] Failed to load %s: %s\n", path.c_str(), dlerror());
        }
        return nullptr;
    }

    void query_real_gpu_info() {
        if (!cuda_driver_handle_) {
            real_gpu_info_.valid = false;
            return;
        }

        // Get function pointers
        typedef int (*cuInit_t)(unsigned int);
        typedef int (*cuDeviceGetCount_t)(int*);
        typedef int (*cuDeviceTotalMem_t)(size_t*, int);
        typedef int (*cuDeviceGetName_t)(char*, int, int);
        typedef int (*cuDeviceGetAttribute_t)(int*, int, int);
        typedef int (*cuMemGetInfo_t)(size_t*, size_t*);
        typedef int (*cuCtxCreate_t)(void**, unsigned int, int);
        typedef int (*cuCtxDestroy_t)(void*);

        auto cuInit = (cuInit_t)dlsym(cuda_driver_handle_, "cuInit");
        auto cuDeviceGetCount = (cuDeviceGetCount_t)dlsym(cuda_driver_handle_, "cuDeviceGetCount");
        auto cuDeviceTotalMem = (cuDeviceTotalMem_t)dlsym(cuda_driver_handle_, "cuDeviceTotalMem_v2");
        auto cuDeviceGetName = (cuDeviceGetName_t)dlsym(cuda_driver_handle_, "cuDeviceGetName");
        auto cuDeviceGetAttribute = (cuDeviceGetAttribute_t)dlsym(cuda_driver_handle_, "cuDeviceGetAttribute");
        auto cuMemGetInfo = (cuMemGetInfo_t)dlsym(cuda_driver_handle_, "cuMemGetInfo_v2");
        auto cuCtxCreate = (cuCtxCreate_t)dlsym(cuda_driver_handle_, "cuCtxCreate_v2");
        auto cuCtxDestroy = (cuCtxDestroy_t)dlsym(cuda_driver_handle_, "cuCtxDestroy_v2");

        if (!cuInit || !cuDeviceGetCount) {
            FGPU_LOG("[RealCudaLoader] Missing required CUDA driver functions\n");
            real_gpu_info_.valid = false;
            return;
        }

        // Initialize CUDA
        int result = cuInit(0);
        if (result != 0) {
            FGPU_LOG("[RealCudaLoader] cuInit failed with error %d\n", result);
            real_gpu_info_.valid = false;
            return;
        }

        // Get device count
        result = cuDeviceGetCount(&real_gpu_info_.device_count);
        if (result != 0 || real_gpu_info_.device_count == 0) {
            FGPU_LOG("[RealCudaLoader] No real GPUs found\n");
            real_gpu_info_.valid = false;
            return;
        }

        FGPU_LOG("[RealCudaLoader] Found %d real GPU(s)\n", real_gpu_info_.device_count);

        // Query each device
        for (int i = 0; i < real_gpu_info_.device_count && i < 16; ++i) {
            if (cuDeviceTotalMem) {
                cuDeviceTotalMem(&real_gpu_info_.total_memory[i], i);
            }

            if (cuDeviceGetName) {
                cuDeviceGetName(real_gpu_info_.name[i], 256, i);
            }

            if (cuDeviceGetAttribute) {
                // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
                // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
                cuDeviceGetAttribute(&real_gpu_info_.compute_major[i], 75, i);
                cuDeviceGetAttribute(&real_gpu_info_.compute_minor[i], 76, i);
            }

            // Get free memory (requires context)
            if (cuCtxCreate && cuMemGetInfo && cuCtxDestroy) {
                void* ctx = nullptr;
                if (cuCtxCreate(&ctx, 0, i) == 0) {
                    size_t free_mem = 0, total_mem = 0;
                    cuMemGetInfo(&free_mem, &total_mem);
                    real_gpu_info_.free_memory[i] = free_mem;
                    cuCtxDestroy(ctx);
                }
            }

            FGPU_LOG("[RealCudaLoader] GPU %d: %s, %zu MB total, %zu MB free, CC %d.%d\n",
                    i, real_gpu_info_.name[i],
                    real_gpu_info_.total_memory[i] / (1024 * 1024),
                    real_gpu_info_.free_memory[i] / (1024 * 1024),
                    real_gpu_info_.compute_major[i], real_gpu_info_.compute_minor[i]);
        }

        real_gpu_info_.valid = true;
    }

    std::mutex mutex_;
    bool initialized_ = false;
    bool real_gpu_info_cached_ = false;

    void* cuda_driver_handle_ = nullptr;
    void* cudart_handle_ = nullptr;
    void* cublas_handle_ = nullptr;
    void* nvml_handle_ = nullptr;

    RealGpuInfo real_gpu_info_;
};

} // namespace fake_gpu
