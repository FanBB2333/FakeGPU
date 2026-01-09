#pragma once

#include <string>
#include <cstdlib>
#include <cstring>

namespace fake_gpu {

// FakeGPU operation modes
enum class FakeGpuMode {
    Simulate,    // Default: all APIs return fake data, no real GPU needed
    Passthrough, // Forward all calls to real CUDA libraries
    Hybrid       // Virtual device info + real compute with OOM safety
};

// OOM safety policies for Hybrid mode
enum class OomPolicy {
    Clamp,       // Report memory clamped to real GPU capacity
    Managed,     // Use cudaMallocManaged for oversubscription
    MappedHost,  // Use cudaHostAllocMapped for overflow
    SpillCpu     // Spill to CPU memory, fail unsupported kernels
};

// Backend configuration singleton
class BackendConfig {
public:
    static BackendConfig& instance() {
        static BackendConfig config;
        return config;
    }

    FakeGpuMode mode() const { return mode_; }
    OomPolicy oom_policy() const { return oom_policy_; }

    // Real library paths (can be overridden via env vars)
    const std::string& real_cuda_lib_dir() const { return real_cuda_lib_dir_; }
    const std::string& real_cudart_path() const { return real_cudart_path_; }
    const std::string& real_cuda_driver_path() const { return real_cuda_driver_path_; }
    const std::string& real_cublas_path() const { return real_cublas_path_; }
    const std::string& real_nvml_path() const { return real_nvml_path_; }

    // Check if we should use real libraries
    bool use_real_cuda() const {
        return mode_ == FakeGpuMode::Passthrough || mode_ == FakeGpuMode::Hybrid;
    }

    // Check if device info should be virtualized (Hybrid mode)
    bool virtualize_device_info() const {
        return mode_ == FakeGpuMode::Simulate || mode_ == FakeGpuMode::Hybrid;
    }

    // Check if compute should use real GPU
    bool use_real_compute() const {
        return mode_ == FakeGpuMode::Passthrough || mode_ == FakeGpuMode::Hybrid;
    }

    // For testing: allow mode override
    void set_mode(FakeGpuMode mode) { mode_ = mode; }
    void set_oom_policy(OomPolicy policy) { oom_policy_ = policy; }

private:
    BackendConfig() {
        initialize_from_env();
    }

    void initialize_from_env() {
        // Parse FAKEGPU_MODE
        if (const char* mode_env = std::getenv("FAKEGPU_MODE")) {
            if (strcasecmp(mode_env, "passthrough") == 0) {
                mode_ = FakeGpuMode::Passthrough;
            } else if (strcasecmp(mode_env, "hybrid") == 0) {
                mode_ = FakeGpuMode::Hybrid;
            } else {
                mode_ = FakeGpuMode::Simulate;
            }
        }

        // Parse FAKEGPU_OOM_POLICY
        if (const char* policy_env = std::getenv("FAKEGPU_OOM_POLICY")) {
            if (strcasecmp(policy_env, "managed") == 0) {
                oom_policy_ = OomPolicy::Managed;
            } else if (strcasecmp(policy_env, "mapped_host") == 0) {
                oom_policy_ = OomPolicy::MappedHost;
            } else if (strcasecmp(policy_env, "spill_cpu") == 0) {
                oom_policy_ = OomPolicy::SpillCpu;
            } else {
                oom_policy_ = OomPolicy::Clamp;
            }
        }

        // Parse library paths
        if (const char* dir = std::getenv("FAKEGPU_REAL_CUDA_LIB_DIR")) {
            real_cuda_lib_dir_ = dir;
        }

        // Try to find real library paths
        find_real_library_paths();
    }

    void find_real_library_paths() {
        // Common CUDA installation paths
        static const char* cuda_search_paths[] = {
            "/usr/local/cuda/lib64",
            "/usr/local/cuda-12/lib64",
            "/usr/local/cuda-11/lib64",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib64",
            nullptr
        };

        // If user specified a directory, use it
        if (!real_cuda_lib_dir_.empty()) {
            real_cudart_path_ = real_cuda_lib_dir_ + "/libcudart.so";
            real_cuda_driver_path_ = real_cuda_lib_dir_ + "/libcuda.so";
            real_cublas_path_ = real_cuda_lib_dir_ + "/libcublas.so";
            real_nvml_path_ = real_cuda_lib_dir_ + "/libnvidia-ml.so";
            return;
        }

        // Search for libraries in common paths
        for (const char** path = cuda_search_paths; *path; ++path) {
            std::string dir(*path);

            // Check for libcudart.so
            if (real_cudart_path_.empty()) {
                std::string cudart = dir + "/libcudart.so";
                if (file_exists(cudart)) {
                    real_cudart_path_ = cudart;
                    if (real_cuda_lib_dir_.empty()) {
                        real_cuda_lib_dir_ = dir;
                    }
                }
            }

            // Check for libcuda.so (driver)
            if (real_cuda_driver_path_.empty()) {
                std::string cuda = dir + "/libcuda.so";
                if (file_exists(cuda)) {
                    real_cuda_driver_path_ = cuda;
                }
            }

            // Check for libcublas.so
            if (real_cublas_path_.empty()) {
                std::string cublas = dir + "/libcublas.so";
                if (file_exists(cublas)) {
                    real_cublas_path_ = cublas;
                }
            }

            // Check for libnvidia-ml.so
            if (real_nvml_path_.empty()) {
                std::string nvml = dir + "/libnvidia-ml.so";
                if (file_exists(nvml)) {
                    real_nvml_path_ = nvml;
                }
            }
        }

        // Also check /usr/lib for nvidia-ml (often installed separately)
        if (real_nvml_path_.empty()) {
            if (file_exists("/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1")) {
                real_nvml_path_ = "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1";
            }
        }
    }

    static bool file_exists(const std::string& path) {
        FILE* f = fopen(path.c_str(), "r");
        if (f) {
            fclose(f);
            return true;
        }
        return false;
    }

    FakeGpuMode mode_ = FakeGpuMode::Simulate;
    OomPolicy oom_policy_ = OomPolicy::Clamp;
    std::string real_cuda_lib_dir_;
    std::string real_cudart_path_;
    std::string real_cuda_driver_path_;
    std::string real_cublas_path_;
    std::string real_nvml_path_;
};

// Helper function to get mode name
inline const char* mode_name(FakeGpuMode mode) {
    switch (mode) {
        case FakeGpuMode::Simulate: return "simulate";
        case FakeGpuMode::Passthrough: return "passthrough";
        case FakeGpuMode::Hybrid: return "hybrid";
    }
    return "unknown";
}

// Helper function to get policy name
inline const char* policy_name(OomPolicy policy) {
    switch (policy) {
        case OomPolicy::Clamp: return "clamp";
        case OomPolicy::Managed: return "managed";
        case OomPolicy::MappedHost: return "mapped_host";
        case OomPolicy::SpillCpu: return "spill_cpu";
    }
    return "unknown";
}

} // namespace fake_gpu
