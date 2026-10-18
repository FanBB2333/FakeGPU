// Compile with nvcc -rdc=true --cudart shared for device-side child launches.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

extern "C" __global__ void leaf_values(int* output) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    output[index] = (index + 1) * 7;
}

extern "C" __global__ void launch_leaf(int* output, int* status) {
    leaf_values<<<2, 8>>>(output);
    status[1] = cudaGetLastError();
}

extern "C" __global__ void launch_branch(int* output, int* status) {
    launch_leaf<<<1, 1>>>(output, status);
    status[0] = cudaGetLastError();
}

void check(bool valid, const char* operation) {
    if (!valid) {
        std::fprintf(stderr, "FAILED: %s\n", operation);
        std::exit(1);
    }
}

void check_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "FAILED: %s returned %s (%d)\n", operation,
                     cudaGetErrorString(result), static_cast<int>(result));
        std::exit(1);
    }
}

#define CUDA_CHECK(call) check_cuda(call, #call)

int main() {
    int *output = nullptr, *status = nullptr;
    CUDA_CHECK(cudaMalloc(&output, 16 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&status, 2 * sizeof(int)));
    CUDA_CHECK(cudaMemset(output, 0, 16 * sizeof(int)));
    CUDA_CHECK(cudaMemset(status, 0xff, 2 * sizeof(int)));
    launch_branch<<<1, 1>>>(output, status);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    int values[16], errors[2];
    CUDA_CHECK(cudaMemcpy(values, output, sizeof(values), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(errors, status, sizeof(errors), cudaMemcpyDeviceToHost));
    check(errors[0] == 0 && errors[1] == 0, "device-side launch errors");
    for (int i = 0; i < 16; ++i) check(values[i] == (i + 1) * 7, "nested child launch numerics");
    CUDA_CHECK(cudaFree(status));
    CUDA_CHECK(cudaFree(output));
    std::puts("CUDA dynamic parallelism: parent, child and grandchild numerics passed");
}
