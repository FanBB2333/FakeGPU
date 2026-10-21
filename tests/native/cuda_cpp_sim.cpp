// CUDA Runtime and cuBLAS application flow, compiled without a CUDA toolkit.
#include "../../src/cuda/cudart_defs.hpp"
#include "../../src/cublas/cublas_defs.hpp"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace {

void check(bool condition, const char* operation) {
    if (!condition) {
        std::fprintf(stderr, "FAILED: %s\n", operation);
        std::exit(1);
    }
}

#define CUDA_CHECK(call) check((call) == cudaSuccess, #call)
#define BLAS_CHECK(call) check((call) == CUBLAS_STATUS_SUCCESS, #call)

void check_launch_configuration(unsigned int id) {
    dim3 grid{}, block{};
    size_t shared_mem = 0;
    void* stream = nullptr;
    CUDA_CHECK(__cudaPopCallConfiguration(&grid, &block, &shared_mem, &stream));
    check(grid.x == id && grid.y == 2 && grid.z == 3, "launch grid");
    check(block.x == 32 && block.y == id && block.z == 1, "launch block");
    check(shared_mem == id * 128 && stream == nullptr, "launch memory and stream");
}

void test_registration_and_launch_configuration() {
    int image = 0;
    void** module = __cudaRegisterFatBinary(&image);
    check(module != nullptr && *module == &image, "fat binary registration handle");
    check(__cudaInitModule(module) != 0, "module registration initialization");
    __cudaRegisterFatBinaryEnd(module);
    __cudaUnregisterFatBinary(module);

    // Argument evaluation can nest another launch between the push and pop.
    CUDA_CHECK(__cudaPushCallConfiguration({1, 2, 3}, {32, 1, 1}, 128, nullptr));
    CUDA_CHECK(__cudaPushCallConfiguration({2, 2, 3}, {32, 2, 1}, 256, nullptr));
    check_launch_configuration(2);
    check_launch_configuration(1);
    check(__cudaPopCallConfiguration(nullptr, nullptr, nullptr, nullptr) ==
              cudaErrorInvalidValue, "empty launch configuration stack");

    // Both host threads push before either pops: a shared slot loses one launch.
    std::atomic<int> ready{0};
    auto worker = [&ready](unsigned int id) {
        CUDA_CHECK(__cudaPushCallConfiguration(
            {id, 2, 3}, {32, id, 1}, id * 128, nullptr));
        ready.fetch_add(1);
        while (ready.load() != 2) std::this_thread::yield();
        check_launch_configuration(id);
    };
    std::thread first(worker, 3);
    std::thread second(worker, 4);
    first.join();
    second.join();
}

void test_numerical_pipeline() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    check(count > 0, "simulated device discovery");
    CUDA_CHECK(cudaSetDevice(0));
    struct {
        cudaDeviceProp properties;
        uint64_t sentinel = UINT64_C(0x12345678abcdef01);
    } legacy;
    CUDA_CHECK(cudaGetDeviceProperties(&legacy.properties, 0));
    check(legacy.sentinel == UINT64_C(0x12345678abcdef01), "legacy device property ABI bounds");
    cudaDevicePropV2 modern{};
    CUDA_CHECK(cudaGetDeviceProperties_v2(&modern, 0));
    check(modern.totalGlobalMem == legacy.properties.totalGlobalMem &&
          modern.multiProcessorCount == legacy.properties.multiProcessorCount,
          "legacy and v2 device properties agree");
    check(modern.hostRegisterSupported == 1 && modern.memoryPoolsSupported == 1,
          "v2 runtime capability properties");
    size_t free_before = 0, total = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total));

    cudaStream_t compute = nullptr, download = nullptr;
    cudaEvent_t start = nullptr, done = nullptr;
    CUDA_CHECK(cudaStreamCreate(&compute));
    CUDA_CHECK(cudaStreamCreate(&download));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&done));

    constexpr size_t bytes = 4 * sizeof(float);
    const float a[] = {1, 2, 3, 4};  // Column-major matrices.
    const float b[] = {5, 6, 7, 8};
    const float expected[] = {23, 34, 31, 46};
    float *device_a = nullptr, *device_b = nullptr, *device_c = nullptr;
    float* result = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_a), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_b), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_c), bytes));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&result), bytes));
    CUDA_CHECK(cudaEventRecord(start, compute));
    CUDA_CHECK(cudaMemcpyAsync(device_a, a, bytes, cudaMemcpyHostToDevice, compute));
    CUDA_CHECK(cudaMemcpyAsync(device_b, b, bytes, cudaMemcpyHostToDevice, compute));
    CUDA_CHECK(cudaMemsetAsync(device_c, 0, bytes, compute));

    cublasHandle_t blas = nullptr;
    BLAS_CHECK(cublasCreate_v2(&blas));
    BLAS_CHECK(cublasSetStream_v2(blas, compute));
    const float alpha = 1, beta = 0;
    BLAS_CHECK(cublasSgemm_v2(blas, CUBLAS_OP_N, CUBLAS_OP_N,
        2, 2, 2, &alpha, device_a, 2, device_b, 2, &beta, device_c, 2));
    CUDA_CHECK(cudaEventRecord(done, compute));
    CUDA_CHECK(cudaStreamWaitEvent(download, done, 0));
    CUDA_CHECK(cudaMemcpyAsync(result, device_c, bytes, cudaMemcpyDeviceToHost, download));
    CUDA_CHECK(cudaStreamSynchronize(download));
    CUDA_CHECK(cudaEventSynchronize(done));
    CUDA_CHECK(cudaEventQuery(done));
    float milliseconds = -1;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, done));
    check(milliseconds >= 0 && std::isfinite(milliseconds), "logical event elapsed time");
    for (int i = 0; i < 4; ++i) {
        check(std::fabs(result[i] - expected[i]) < 1e-5f, "cuBLAS numerical result");
    }

    BLAS_CHECK(cublasDestroy_v2(blas));
    CUDA_CHECK(cudaFreeHost(result));
    CUDA_CHECK(cudaFree(device_c));
    CUDA_CHECK(cudaFree(device_b));
    CUDA_CHECK(cudaFree(device_a));
    CUDA_CHECK(cudaEventDestroy(done));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaStreamDestroy(download));
    CUDA_CHECK(cudaStreamDestroy(compute));
    size_t free_after = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_after, &total));
    check(free_after == free_before, "device allocations released");
}

}  // namespace

int main() {
    test_registration_and_launch_configuration();
    test_numerical_pipeline();
    std::puts("CUDA C++: registration, host threads, copies, streams, events and CPU cuBLAS passed");
}
