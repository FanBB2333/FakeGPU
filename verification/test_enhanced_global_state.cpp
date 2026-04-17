#include "../src/core/global_state.hpp"

#include <cstdio>
#include <cstdlib>
#include <utility>

namespace {

void require(bool condition, const char* message) {
    if (condition) {
        return;
    }
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
}

} // namespace

int main() {
    fake_gpu::GlobalState& gs = fake_gpu::GlobalState::instance();
    gs.initialize();

    void* ptr = std::malloc(4096);
    require(ptr != nullptr, "malloc returned null");
    require(gs.register_allocation(ptr, 4096, 0), "register_allocation failed");

    gs.record_kernel_launch("unit_test_kernel");
    gs.record_kernel_launch("unit_test_kernel");
    gs.record_cublas_gemm_typed(ptr, 128, 0);
    gs.record_cublaslt_matmul_typed(ptr, 64, 14);

    auto devices = gs.snapshot_device_report();
    require(!devices.empty(), "snapshot_device_report returned no devices");

    const auto& dev0 = devices.front();
    require(!dev0.architecture.empty(), "missing architecture");
    require(dev0.compute_major > 0, "missing compute_major");
    require(dev0.kernel_launch_total == 2, "unexpected kernel_launch_total");

    auto kernel_it = dev0.kernel_launches.find("unit_test_kernel");
    require(kernel_it != dev0.kernel_launches.end(), "missing unit_test_kernel entry");
    require(kernel_it->second == 2, "unexpected unit_test_kernel count");

    auto fp32_it = dev0.gemm_by_dtype.find("FP32");
    require(fp32_it != dev0.gemm_by_dtype.end(), "missing FP32 dtype stats");
    require(fp32_it->second.first == 1, "unexpected FP32 call count");
    require(fp32_it->second.second == 128, "unexpected FP32 flop count");

    auto bf16_it = dev0.gemm_by_dtype.find("BF16");
    require(bf16_it != dev0.gemm_by_dtype.end(), "missing BF16 dtype stats");
    require(bf16_it->second.first == 1, "unexpected BF16 call count");
    require(bf16_it->second.second == 64, "unexpected BF16 flop count");

    size_t released_size = 0;
    int released_device = -1;
    require(gs.release_allocation(ptr, released_size, released_device), "release_allocation failed");
    std::free(ptr);

    std::puts("enhanced GlobalState report stats test passed");
    return 0;
}
