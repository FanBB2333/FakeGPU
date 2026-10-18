// Compile with nvcc --cudart shared. Device kernels are deliberately unsupported.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

__global__ void vector_add(float* output, const float* input) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    output[i] += input[i];
}

void check(bool condition, const char* operation) {
    if (!condition) {
        std::fprintf(stderr, "FAILED: %s\n", operation);
        std::exit(1);
    }
}

#define CUDA_CHECK(call) check((call) == cudaSuccess, #call)

int main(int argc, char** argv) {
    check(argc == 2, "expected policy argument: allow, warn or error");
    const bool strict = std::strcmp(argv[1], "error") == 0;
    // The runner selects the synthetic a100 profile. Use SDK fields, including
    // the later v2 fields, and sentinels to catch an incompatible ABI size.
    struct {
        uint64_t before;
        cudaDeviceProp properties;
        uint64_t after;
    } guarded;
    std::memset(&guarded, 0xa5, sizeof(guarded));
    CUDA_CHECK(cudaGetDeviceProperties(&guarded.properties, 0));
    check(guarded.before == UINT64_C(0xa5a5a5a5a5a5a5a5) &&
          guarded.after == UINT64_C(0xa5a5a5a5a5a5a5a5), "device properties ABI bounds");
    const auto& properties = guarded.properties;
    check(properties.major == 8 && properties.minor == 0, "synthetic compute capability");
    check(properties.totalGlobalMem == 80ULL * 1024 * 1024 * 1024, "synthetic memory capacity");
    check(properties.multiProcessorCount == 108, "synthetic multiprocessor count");
    check(properties.sharedMemPerBlockOptin == 166912, "v2 shared memory field offset");
    check(properties.hostRegisterSupported == 1 && properties.memoryPoolsSupported == 1,
          "v2 host registration and memory pool field offsets");
    check(properties.reserved[59] == 0, "v2 property tail initialized");
    const float input[] = {1, 2, 3, 4};
    float result[4] = {};
    float* device = nullptr;
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(&device, sizeof(input)));
    CUDA_CHECK(cudaMemcpyAsync(device, input, sizeof(input), cudaMemcpyHostToDevice, stream));

    vector_add<<<1, 4, 0, stream>>>(device, device);
    const cudaError_t launch_error = cudaGetLastError();
    check(launch_error == (strict ? cudaErrorNotSupported : cudaSuccess), "kernel launch policy");
    CUDA_CHECK(cudaGetLastError());  // Getting the last error clears it.
    CUDA_CHECK(cudaMemcpyAsync(result, device, sizeof(result), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int i = 0; i < 4; ++i) {
        check(result[i] == input[i], "arbitrary kernel did not execute");
    }
    CUDA_CHECK(cudaFree(device));
    CUDA_CHECK(cudaStreamDestroy(stream));
    std::printf("nvcc CUDA: device properties and launch policy %s passed; arbitrary kernel was not executed\n", argv[1]);
}
