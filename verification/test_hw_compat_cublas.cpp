#include "../src/core/global_state.hpp"
#include "../src/core/gpu_profile.hpp"
#include "../src/cublas/cublas_defs.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <tuple>

namespace {

void require(bool condition, const char* message) {
    if (condition) {
        return;
    }
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
}

bool has_compat_event(
    const fake_gpu::DeviceReportStats& dev,
    const char* op,
    const char* dtype,
    uint64_t expected_count) {
    for (const auto& event : dev.compat_events) {
        if (std::get<0>(event) == op && std::get<1>(event) == dtype && std::get<2>(event) == expected_count) {
            return true;
        }
    }
    return false;
}

} // namespace

int main() {
    auto bf16 = fake_gpu::cuda_dtype_to_gpu_dtype(14);
    require(bf16.has_value(), "missing BF16 mapping");
    require(*bf16 == fake_gpu::GpuDataType::BF16, "unexpected BF16 mapping");

    setenv("FAKEGPU_PROFILES", "v100:1", 1);

    fake_gpu::GlobalState& gs = fake_gpu::GlobalState::instance();
    gs.initialize();

    float alpha = 1.0f;
    float beta = 0.0f;
    void* A = std::malloc(8);
    void* B = std::malloc(8);
    void* C = std::malloc(8);
    require(A && B && C, "malloc failed");
    std::memset(A, 0, 8);
    std::memset(B, 0, 8);
    std::memset(C, 0, 8);

    require(gs.register_allocation(A, 8, 0), "register A failed");
    require(gs.register_allocation(B, 8, 0), "register B failed");
    require(gs.register_allocation(C, 8, 0), "register C failed");

    cublasStatus_t status = cublasGemmEx(
        nullptr,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        1,
        1,
        1,
        &alpha,
        A,
        14,
        1,
        B,
        14,
        1,
        &beta,
        C,
        14,
        1,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);

    const char* strict_env = std::getenv("FAKEGPU_STRICT_COMPAT");
    const bool strict = !(strict_env && std::strcmp(strict_env, "0") == 0);
    if (strict) {
        require(status == CUBLAS_STATUS_NOT_SUPPORTED, "strict mode should reject BF16 on V100");
    } else {
        require(status == CUBLAS_STATUS_SUCCESS, "non-strict mode should allow BF16 on V100");
    }

    auto devices = gs.snapshot_device_report();
    require(!devices.empty(), "snapshot_device_report returned no devices");
    require(has_compat_event(devices.front(), "cublasGemmEx", "bf16", 1), "missing compatibility event");

    size_t released_size = 0;
    int released_device = -1;
    require(gs.release_allocation(A, released_size, released_device), "release A failed");
    require(gs.release_allocation(B, released_size, released_device), "release B failed");
    require(gs.release_allocation(C, released_size, released_device), "release C failed");
    std::free(A);
    std::free(B);
    std::free(C);

    std::puts("hardware compatibility cuBLAS test passed");
    return 0;
}
