/*
 * Test program for FakeGPU mode switching
 * Tests simulate, passthrough, and hybrid modes
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

// NVML types
typedef int nvmlReturn_t;
typedef void* nvmlDevice_t;
typedef struct {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
} nvmlMemory_t;

// CUDA types
typedef int CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;

// Function pointer types
typedef nvmlReturn_t (*nvmlInit_t)(void);
typedef nvmlReturn_t (*nvmlShutdown_t)(void);
typedef nvmlReturn_t (*nvmlDeviceGetCount_t)(unsigned int*);
typedef nvmlReturn_t (*nvmlDeviceGetHandleByIndex_t)(unsigned int, nvmlDevice_t*);
typedef nvmlReturn_t (*nvmlDeviceGetName_t)(nvmlDevice_t, char*, unsigned int);
typedef nvmlReturn_t (*nvmlDeviceGetMemoryInfo_t)(nvmlDevice_t, nvmlMemory_t*);

typedef CUresult (*cuInit_t)(unsigned int);
typedef CUresult (*cuDeviceGetCount_t)(int*);
typedef CUresult (*cuDeviceGetName_t)(char*, int, CUdevice);
typedef CUresult (*cuDeviceTotalMem_t)(size_t*, CUdevice);
typedef CUresult (*cuMemAlloc_t)(CUdeviceptr*, size_t);
typedef CUresult (*cuMemFree_t)(CUdeviceptr);
typedef CUresult (*cuMemGetInfo_t)(size_t*, size_t*);

int main(int argc, char** argv) {
    const char* mode = getenv("FAKEGPU_MODE");
    if (!mode) mode = "simulate";

    printf("========================================\n");
    printf("FakeGPU Mode Test\n");
    printf("Mode: %s\n", mode);
    printf("========================================\n\n");

    // Test NVML
    printf("[NVML Test]\n");
    nvmlInit_t nvmlInit = (nvmlInit_t)dlsym(RTLD_DEFAULT, "nvmlInit_v2");
    if (!nvmlInit) nvmlInit = (nvmlInit_t)dlsym(RTLD_DEFAULT, "nvmlInit");

    if (nvmlInit) {
        nvmlReturn_t ret = nvmlInit();
        printf("  nvmlInit: %d\n", ret);

        if (ret == 0) {
            nvmlDeviceGetCount_t nvmlDeviceGetCount =
                (nvmlDeviceGetCount_t)dlsym(RTLD_DEFAULT, "nvmlDeviceGetCount_v2");
            if (!nvmlDeviceGetCount)
                nvmlDeviceGetCount = (nvmlDeviceGetCount_t)dlsym(RTLD_DEFAULT, "nvmlDeviceGetCount");

            if (nvmlDeviceGetCount) {
                unsigned int count = 0;
                ret = nvmlDeviceGetCount(&count);
                printf("  Device count: %u (ret=%d)\n", count, ret);

                nvmlDeviceGetHandleByIndex_t nvmlDeviceGetHandleByIndex =
                    (nvmlDeviceGetHandleByIndex_t)dlsym(RTLD_DEFAULT, "nvmlDeviceGetHandleByIndex_v2");
                if (!nvmlDeviceGetHandleByIndex)
                    nvmlDeviceGetHandleByIndex = (nvmlDeviceGetHandleByIndex_t)dlsym(RTLD_DEFAULT, "nvmlDeviceGetHandleByIndex");

                nvmlDeviceGetName_t nvmlDeviceGetName =
                    (nvmlDeviceGetName_t)dlsym(RTLD_DEFAULT, "nvmlDeviceGetName");

                nvmlDeviceGetMemoryInfo_t nvmlDeviceGetMemoryInfo =
                    (nvmlDeviceGetMemoryInfo_t)dlsym(RTLD_DEFAULT, "nvmlDeviceGetMemoryInfo");

                for (unsigned int i = 0; i < count && i < 2; i++) {
                    nvmlDevice_t device;
                    ret = nvmlDeviceGetHandleByIndex(i, &device);
                    if (ret == 0) {
                        char name[256] = {0};
                        nvmlDeviceGetName(device, name, 256);
                        printf("  Device %u: %s\n", i, name);

                        nvmlMemory_t mem;
                        if (nvmlDeviceGetMemoryInfo && nvmlDeviceGetMemoryInfo(device, &mem) == 0) {
                            printf("    Memory: %.2f GB total, %.2f GB free\n",
                                   mem.total / (1024.0 * 1024.0 * 1024.0),
                                   mem.free / (1024.0 * 1024.0 * 1024.0));
                        }
                    }
                }
            }

            nvmlShutdown_t nvmlShutdown = (nvmlShutdown_t)dlsym(RTLD_DEFAULT, "nvmlShutdown");
            if (nvmlShutdown) nvmlShutdown();
        }
    } else {
        printf("  nvmlInit not found\n");
    }

    // Test CUDA Driver API
    printf("\n[CUDA Driver API Test]\n");
    cuInit_t cuInit = (cuInit_t)dlsym(RTLD_DEFAULT, "cuInit");

    if (cuInit) {
        CUresult ret = cuInit(0);
        printf("  cuInit: %d\n", ret);

        if (ret == 0) {
            cuDeviceGetCount_t cuDeviceGetCount =
                (cuDeviceGetCount_t)dlsym(RTLD_DEFAULT, "cuDeviceGetCount");

            if (cuDeviceGetCount) {
                int count = 0;
                ret = cuDeviceGetCount(&count);
                printf("  Device count: %d (ret=%d)\n", count, ret);

                cuDeviceGetName_t cuDeviceGetName =
                    (cuDeviceGetName_t)dlsym(RTLD_DEFAULT, "cuDeviceGetName");
                cuDeviceTotalMem_t cuDeviceTotalMem =
                    (cuDeviceTotalMem_t)dlsym(RTLD_DEFAULT, "cuDeviceTotalMem_v2");
                if (!cuDeviceTotalMem)
                    cuDeviceTotalMem = (cuDeviceTotalMem_t)dlsym(RTLD_DEFAULT, "cuDeviceTotalMem");

                for (int i = 0; i < count && i < 2; i++) {
                    char name[256] = {0};
                    if (cuDeviceGetName) {
                        cuDeviceGetName(name, 256, i);
                        printf("  Device %d: %s\n", i, name);
                    }

                    size_t totalMem = 0;
                    if (cuDeviceTotalMem && cuDeviceTotalMem(&totalMem, i) == 0) {
                        printf("    Total memory: %.2f GB\n", totalMem / (1024.0 * 1024.0 * 1024.0));
                    }
                }
            }

            // Test memory allocation
            printf("\n[Memory Allocation Test]\n");
            cuMemAlloc_t cuMemAlloc = (cuMemAlloc_t)dlsym(RTLD_DEFAULT, "cuMemAlloc_v2");
            if (!cuMemAlloc) cuMemAlloc = (cuMemAlloc_t)dlsym(RTLD_DEFAULT, "cuMemAlloc");

            cuMemFree_t cuMemFree = (cuMemFree_t)dlsym(RTLD_DEFAULT, "cuMemFree_v2");
            if (!cuMemFree) cuMemFree = (cuMemFree_t)dlsym(RTLD_DEFAULT, "cuMemFree");

            cuMemGetInfo_t cuMemGetInfo = (cuMemGetInfo_t)dlsym(RTLD_DEFAULT, "cuMemGetInfo_v2");
            if (!cuMemGetInfo) cuMemGetInfo = (cuMemGetInfo_t)dlsym(RTLD_DEFAULT, "cuMemGetInfo");

            if (cuMemAlloc && cuMemFree) {
                // Get initial memory info
                size_t free_before = 0, total = 0;
                if (cuMemGetInfo) {
                    cuMemGetInfo(&free_before, &total);
                    printf("  Before alloc: %.2f GB free / %.2f GB total\n",
                           free_before / (1024.0 * 1024.0 * 1024.0),
                           total / (1024.0 * 1024.0 * 1024.0));
                }

                // Allocate 64MB
                CUdeviceptr ptr = 0;
                size_t alloc_size = 64 * 1024 * 1024;
                ret = cuMemAlloc(&ptr, alloc_size);
                printf("  cuMemAlloc(64MB): ret=%d, ptr=0x%llx\n", ret, ptr);

                if (ret == 0 && cuMemGetInfo) {
                    size_t free_after = 0;
                    cuMemGetInfo(&free_after, &total);
                    printf("  After alloc: %.2f GB free (delta: %.2f MB)\n",
                           free_after / (1024.0 * 1024.0 * 1024.0),
                           (free_before - free_after) / (1024.0 * 1024.0));
                }

                if (ret == 0) {
                    ret = cuMemFree(ptr);
                    printf("  cuMemFree: ret=%d\n", ret);
                }
            }
        }
    } else {
        printf("  cuInit not found\n");
    }

    printf("\n========================================\n");
    printf("Test completed!\n");
    printf("========================================\n");

    return 0;
}
