#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dlfcn.h>

// Declare expected symbols (normally from nvml.h / cuda_runtime.h)
typedef enum { NVML_SUCCESS = 0 } nvmlReturn_t;
typedef void* nvmlDevice_t;
typedef struct { unsigned long long total, free, used; } nvmlMemory_t;

typedef enum { cudaSuccess = 0 } cudaError_t;

// Function pointer types
typedef nvmlReturn_t (*nvmlInit_t)();
typedef nvmlReturn_t (*nvmlDeviceGetCount_t)(unsigned int *count);
typedef nvmlReturn_t (*nvmlDeviceGetHandleByIndex_t)(unsigned int index, nvmlDevice_t *device);
typedef nvmlReturn_t (*nvmlDeviceGetMemoryInfo_t)(nvmlDevice_t device, nvmlMemory_t *memory);

typedef cudaError_t (*cudaGetDeviceCount_t)(int *count);
typedef cudaError_t (*cudaMalloc_t)(void **devPtr, size_t size);
typedef cudaError_t (*cudaFree_t)(void *devPtr);

// Function pointers
nvmlInit_t nvmlInit;
nvmlDeviceGetCount_t nvmlDeviceGetCount;
nvmlDeviceGetHandleByIndex_t nvmlDeviceGetHandleByIndex;
nvmlDeviceGetMemoryInfo_t nvmlDeviceGetMemoryInfo;
cudaGetDeviceCount_t cudaGetDeviceCount;
cudaMalloc_t cudaMalloc;
cudaFree_t cudaFree;

int main() {
    printf("--- Verification Start ---\n");

    // Load symbols dynamically
    nvmlInit = (nvmlInit_t)dlsym(RTLD_DEFAULT, "nvmlInit");
    nvmlDeviceGetCount = (nvmlDeviceGetCount_t)dlsym(RTLD_DEFAULT, "nvmlDeviceGetCount");
    nvmlDeviceGetHandleByIndex = (nvmlDeviceGetHandleByIndex_t)dlsym(RTLD_DEFAULT, "nvmlDeviceGetHandleByIndex");
    nvmlDeviceGetMemoryInfo = (nvmlDeviceGetMemoryInfo_t)dlsym(RTLD_DEFAULT, "nvmlDeviceGetMemoryInfo");
    cudaGetDeviceCount = (cudaGetDeviceCount_t)dlsym(RTLD_DEFAULT, "cudaGetDeviceCount");
    cudaMalloc = (cudaMalloc_t)dlsym(RTLD_DEFAULT, "cudaMalloc");
    cudaFree = (cudaFree_t)dlsym(RTLD_DEFAULT, "cudaFree");

    if (!nvmlInit || !cudaGetDeviceCount) {
        printf("Failed to load symbols. Make sure FakeGPU is enabled (LD_PRELOAD/DYLD_INSERT_LIBRARIES or fakegpu.init()).\n");
        return 1;
    }

    // 1. NVML Test
    printf("[Test] calling nvmlInit...\n");
    if (nvmlInit() != NVML_SUCCESS) {
         printf("nvmlInit failed (or symbol not found)\n");
         return 1;
    }
    
    unsigned int nvml_count = 0;
    nvmlDeviceGetCount(&nvml_count);
    printf("[Test] NVML Device Count: %d\n", nvml_count);
    
    if (nvml_count > 0) {
        nvmlDevice_t dev;
        nvmlDeviceGetHandleByIndex(0, &dev);
        nvmlMemory_t mem;
        nvmlDeviceGetMemoryInfo(dev, &mem);
        printf("[Test] Device 0 Memory: %llu bytes\n", mem.total);
    }

    // 2. CUDA Test
    int cuda_count = 0;
    cudaGetDeviceCount(&cuda_count);
    printf("[Test] CUDA Device Count: %d\n", cuda_count);
    
    void* ptr = NULL;
    size_t size = 1024 * 1024 * 64; // 64 MB
    printf("[Test] Allocating 64MB via cudaMalloc...\n");
    if (cudaMalloc(&ptr, size) == cudaSuccess) {
        printf("[Test] Allocation successful at %p. Writing data...\n", ptr);
        memset(ptr, 0xAB, size); // Touch memory
        cudaFree(ptr);
        printf("[Test] Freed.\n");
    } else {
        printf("[Test] cudaMalloc failed\n");
    }

    printf("--- Verification End ---\n");
    return 0;
}
