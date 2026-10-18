// Compile with nvcc --cudart shared. The runner also builds SASS-only images.
// Both image formats must execute the same numerical suite without a GPU.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

__device__ __noinline__ float finish_value(float value) {
    return value * 0.5f + 2.0f;
}

__device__ unsigned int device_counter = 10;
__constant__ unsigned int device_increment = 3;

extern "C" __global__ void advance_counter(unsigned int* output) {
    device_counter += device_increment;
    *output = device_counter;
}

extern "C" __global__ void affine_branch(float* output, const float* input,
                                         float scale, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        float value = input[i] * scale;
        for (int step = 0; step < i % 4; ++step) value += 0.25f;
        output[i] = finish_value(i % 2 ? value : -value);
    }
}

extern "C" __global__ void rectangular_product(float* output, const float* left,
                                               const float* right,
                                               int rows, int inner, int columns) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && column < columns) {
        float sum = 0;
        for (int k = 0; k < inner; ++k) {
            sum += left[row * inner + k] * right[k * columns + column];
        }
        output[row * columns + column] = sum;
    }
}

extern "C" __global__ void block_sum(float* output, const float* input) {
    extern __shared__ float scratch[];
    __shared__ float initial[64];
    const unsigned int tid = threadIdx.x;
    initial[tid] = input[blockIdx.x * blockDim.x + tid];
    __syncthreads();
    scratch[tid] = initial[blockDim.x - 1 - tid];
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride; stride /= 2) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    if (!tid) output[blockIdx.x] = scratch[0];
}

extern "C" __global__ void ticket_counts(unsigned int* counts, unsigned int* tickets) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    tickets[i] = atomicAdd(&counts[i % 4], 1u);
    atomicAdd(&counts[4], 1u);
}

extern "C" __global__ void volume_coordinates(unsigned int* output) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
    const unsigned int width = gridDim.x * blockDim.x;
    const unsigned int height = gridDim.y * blockDim.y;
    output[(z * height + y) * width + x] = x + 10 * y + 100 * z;
}

extern "C" __global__ void double_transform(double* output, const double* input,
                                            double scale, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = input[index] * scale + 0.125;
}

extern "C" __global__ void warp_exchange(unsigned int* output, unsigned int* masks) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int mask = __ballot_sync(0xffffffffu, index % 3 == 0);
    masks[index] = mask;
    output[index] = __shfl_xor_sync(0xffffffffu, index, 1) + 1000 * __popc(mask);
}

struct TransformArguments {
    int count;
    const float* input;
    float scale;
    float* output;
};

extern "C" __global__ void structured_transform(TransformArguments arguments) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < arguments.count) {
        arguments.output[index] = arguments.input[index] * arguments.scale - 2;
    }
}

void check(bool condition, const char* operation) {
    if (!condition) {
        std::fprintf(stderr, "FAILED: %s\n", operation);
        std::exit(1);
    }
}

#define CUDA_CHECK(call) check((call) == cudaSuccess, #call)

template <typename T>
T* upload(const std::vector<T>& values) {
    T* device = nullptr;
    CUDA_CHECK(cudaMalloc(&device, values.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(device, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    return device;
}

template <typename T>
std::vector<T> download(T* device, size_t count) {
    std::vector<T> values(count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(values.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device));
    return values;
}

void test_device_properties() {
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
}

void test_vector_branches() {
    constexpr int count = 37;
    constexpr float scale = 1.75f;
    std::vector<float> input(count);
    for (int i = 0; i < count; ++i) input[i] = (i - 13) * 0.5f;
    float* source = upload(input);
    float* output = upload(std::vector<float>(count + 3, -999));
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    affine_branch<<<3, 16, 0, stream>>>(output, source, scale, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto result = download(output, count + 3);
    for (int i = 0; i < count; ++i) {
        const float value = input[i] * scale + (i % 4) * 0.25f;
        const float expected = (i % 2 ? value : -value) * 0.5f + 2.0f;
        check(result[i] == expected, "vector branches, loops, scalar arguments and device calls");
    }
    for (int i = count; i < count + 3; ++i) check(result[i] == -999, "partial block bounds");
    CUDA_CHECK(cudaFree(source));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void test_matrix_product() {
    constexpr int rows = 5, inner = 7, columns = 6;
    std::vector<float> left(rows * inner), right(inner * columns);
    for (int i = 0; i < rows * inner; ++i) left[i] = (i % 9 - 4) * 0.25f;
    for (int i = 0; i < inner * columns; ++i) right[i] = (i % 7 - 3) * 0.5f;
    float* a = upload(left);
    float* b = upload(right);
    float* output = upload(std::vector<float>(rows * columns, -999));
    rectangular_product<<<dim3(2, 2), dim3(4, 3)>>>(output, a, b, rows, inner, columns);
    const auto result = download(output, rows * columns);
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            float expected = 0;
            for (int k = 0; k < inner; ++k) expected += left[row * inner + k] * right[k * columns + column];
            check(std::fabs(result[row * columns + column] - expected) < 1e-5f, "2D rectangular matrix product");
        }
    }
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
}

void test_shared_reduction() {
    constexpr int blocks = 3, threads = 64;
    std::vector<float> input(blocks * threads);
    for (int i = 0; i < blocks * threads; ++i) input[i] = i * 0.5f - 3;
    float* source = upload(input);
    float* output = upload(std::vector<float>(blocks, -999));
    block_sum<<<blocks, threads, threads * sizeof(float)>>>(output, source);
    const auto result = download(output, blocks);
    for (int block = 0; block < blocks; ++block) {
        float expected = 0;
        for (int thread = 0; thread < threads; ++thread) expected += input[block * threads + thread];
        check(result[block] == expected, "static/dynamic shared memory and block barriers");
    }
    CUDA_CHECK(cudaFree(source));
}

void test_atomics() {
    constexpr int count = 128;
    unsigned int* counts = upload(std::vector<unsigned int>(5, 0));
    unsigned int* tickets = upload(std::vector<unsigned int>(count, 999));
    ticket_counts<<<2, 64>>>(counts, tickets);
    const auto totals = download(counts, 5);
    const auto result = download(tickets, count);
    check(totals[4] == count, "atomic reduction without return value");
    for (int bucket = 0; bucket < 4; ++bucket) {
        check(totals[bucket] == count / 4, "atomic add counter");
        std::vector<unsigned int> bucket_tickets;
        for (int i = bucket; i < count; i += 4) bucket_tickets.push_back(result[i]);
        std::sort(bucket_tickets.begin(), bucket_tickets.end());
        for (unsigned int i = 0; i < bucket_tickets.size(); ++i) {
            check(bucket_tickets[i] == i, "atomic add previous values are unique");
        }
    }
}

void test_3d_launch() {
    constexpr int width = 6, height = 4, depth = 6;
    unsigned int* output = upload(std::vector<unsigned int>(width * height * depth, 99999));
    volume_coordinates<<<dim3(2, 2, 3), dim3(3, 2, 2)>>>(output);
    const auto result = download(output, width * height * depth);
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                check(result[(z * height + y) * width + x] == x + 10 * y + 100 * z,
                      "three dimensional grid, block and grid dimensions");
            }
        }
    }
}

void test_double_precision() {
    constexpr int count = 19;
    std::vector<double> input(count);
    for (int i = 0; i < count; ++i) input[i] = (i - 7) * 0.25;
    double* source = upload(input);
    double* output = upload(std::vector<double>(count, -999));
    double_transform<<<2, 16>>>(output, source, 1.5, count);
    const auto result = download(output, count);
    for (int i = 0; i < count; ++i) {
        check(result[i] == input[i] * 1.5 + 0.125, "double precision arithmetic and scalar arguments");
    }
    CUDA_CHECK(cudaFree(source));
}

void test_warp_intrinsics() {
    constexpr int count = 192;
    unsigned int* output = upload(std::vector<unsigned int>(count, 0));
    unsigned int* masks = upload(std::vector<unsigned int>(count, 0));
    warp_exchange<<<3, 64>>>(output, masks);
    const auto values = download(output, count);
    const auto ballots = download(masks, count);
    for (int i = 0; i < count; ++i) {
        unsigned int expected_mask = 0, expected_count = 0;
        const int warp_start = (i / 32) * 32;
        for (int lane = 0; lane < 32; ++lane) {
            if ((warp_start + lane) % 3 == 0) {
                expected_mask |= 1u << lane;
                ++expected_count;
            }
        }
        check(ballots[i] == expected_mask, "warp ballot mask");
        check(values[i] == (i ^ 1) + 1000 * expected_count, "warp shuffle and population count");
    }
}

void test_struct_arguments() {
    constexpr int count = 23;
    std::vector<float> input(count);
    for (int i = 0; i < count; ++i) input[i] = (i - 9) * 0.25f;
    float* source = upload(input);
    float* output = upload(std::vector<float>(count, -999));
    structured_transform<<<2, 16>>>(TransformArguments{count, source, -1.5f, output});
    const auto result = download(output, count);
    for (int i = 0; i < count; ++i) {
        check(result[i] == input[i] * -1.5f - 2, "structure arguments with aligned pointers and scalars");
    }
    CUDA_CHECK(cudaFree(source));
}

void test_device_symbols_and_isolation() {
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    check(devices >= 2, "two synthetic devices for module isolation");
    for (int device = 0; device < 2; ++device) {
        CUDA_CHECK(cudaSetDevice(device));
        unsigned int initial = 0;
        CUDA_CHECK(cudaMemcpyFromSymbol(&initial, device_counter, sizeof(initial)));
        check(initial == 10, "module global initialized on each device");
        CUDA_CHECK(cudaMemcpyFromSymbol(&initial, device_increment, sizeof(initial)));
        check(initial == 3, "module constant initialized on each device");
        const unsigned int increment = 5 + 4 * device;
        CUDA_CHECK(cudaMemcpyToSymbol(device_increment, &increment, sizeof(increment)));
        unsigned int* output = upload(std::vector<unsigned int>(1, 0));
        advance_counter<<<1, 1>>>(output);
        check(download(output, 1)[0] == 10 + increment, "kernel executes on selected device");
    }
    for (int device = 0; device < 2; ++device) {
        CUDA_CHECK(cudaSetDevice(device));
        unsigned int value = 0;
        const unsigned int increment = 5 + 4 * device;
        CUDA_CHECK(cudaMemcpyFromSymbol(&value, device_counter, sizeof(value)));
        check(value == 10 + increment, "global state isolated between devices");
        CUDA_CHECK(cudaMemcpyFromSymbol(&value, device_increment, sizeof(value)));
        check(value == increment, "constant state isolated between devices");
        const unsigned int replacement = 100 + device;
        CUDA_CHECK(cudaMemcpyToSymbol(device_counter, &replacement, sizeof(replacement)));
        unsigned int* output = upload(std::vector<unsigned int>(1, 0));
        advance_counter<<<1, 1>>>(output);
        check(download(output, 1)[0] == replacement + increment, "host update to module global");
    }
    CUDA_CHECK(cudaSetDevice(0));
}

void test_host_thread_devices() {
    unsigned int main_before = 0, main_increment = 0;
    CUDA_CHECK(cudaMemcpyFromSymbol(&main_before, device_counter, sizeof(main_before)));
    CUDA_CHECK(cudaMemcpyFromSymbol(&main_increment, device_increment, sizeof(main_increment)));
    std::thread worker([] {
        CUDA_CHECK(cudaSetDevice(1));
        unsigned int before = 0, increment = 0;
        CUDA_CHECK(cudaMemcpyFromSymbol(&before, device_counter, sizeof(before)));
        CUDA_CHECK(cudaMemcpyFromSymbol(&increment, device_increment, sizeof(increment)));
        unsigned int* output = upload(std::vector<unsigned int>(1, 0));
        advance_counter<<<1, 1>>>(output);
        check(download(output, 1)[0] == before + increment, "host worker launches on its selected device");
    });
    worker.join();
    int selected = -1;
    CUDA_CHECK(cudaGetDevice(&selected));
    check(selected == 0, "host thread device selection is independent");
    unsigned int unchanged = 0;
    CUDA_CHECK(cudaMemcpyFromSymbol(&unchanged, device_counter, sizeof(unchanged)));
    check(unchanged == main_before, "worker preserves other device globals");
    unsigned int* output = upload(std::vector<unsigned int>(1, 0));
    advance_counter<<<1, 1>>>(output);
    check(download(output, 1)[0] == main_before + main_increment, "main thread keeps its selected device");
}

int main() {
    test_device_properties();
    test_vector_branches();
    test_matrix_product();
    test_shared_reduction();
    test_atomics();
    test_3d_launch();
    test_double_precision();
    test_warp_intrinsics();
    test_struct_arguments();
    test_device_symbols_and_isolation();
    test_host_thread_devices();
    std::puts("nvcc CUDA: kernel numerics, device calls, symbols and per-device module isolation passed");
}
