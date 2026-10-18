// Compiled as SASS-only cubins/fatbins by cubin_execution.py.
#include <cuda_runtime.h>

extern "C" {

__device__ int counter = 7;
__constant__ int bias = 3;
__device__ int* counter_address = &counter;
__device__ float blas_vector[4] = {1, 2, 3, 4};
__device__ float* blas_vector_address = blas_vector;
__device__ float blas_scale = 2;
__device__ float blas_result = 0;

__global__ void affine_values(float* output, const float* input, float scale, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = input[index] * scale + bias;
}

__global__ void advance_state(int* output, int delta) {
    counter += delta;
    *output = counter;
}

__global__ void compiled_architecture(int* output) {
    *output = __CUDA_ARCH__;
}

__global__ void store_word(int* output) {
    *output = 99;
}

__global__ void export_counter_address(int** output) {
    *output = &counter;
}

__global__ void export_constant_address(const int** output) {
    *output = &bias;
}

}
