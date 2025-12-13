#include <stdio.h>
#include <stdlib.h>

// NVML types and functions
typedef enum {
    NVML_SUCCESS = 0,
    NVML_ERROR_INVALID_ARGUMENT = 2
} nvmlReturn_t;

typedef struct nvmlDevice_st* nvmlDevice_t;

nvmlReturn_t nvmlInit();
nvmlReturn_t nvmlShutdown();
nvmlReturn_t nvmlDeviceGetCount(unsigned int *deviceCount);
nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t *device);
nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char *name, unsigned int length);

int main() {
    printf("Testing NVML...\n");

    nvmlReturn_t result = nvmlInit();
    printf("nvmlInit: %d\n", result);

    unsigned int count = 0;
    result = nvmlDeviceGetCount(&count);
    printf("nvmlDeviceGetCount: %d, count: %u\n", result, count);

    if (count > 0) {
        nvmlDevice_t device;
        result = nvmlDeviceGetHandleByIndex(0, &device);
        printf("nvmlDeviceGetHandleByIndex: %d, device: %p\n", result, device);

        char name[64];
        result = nvmlDeviceGetName(device, name, sizeof(name));
        printf("nvmlDeviceGetName: %d, name: %s\n", result, name);
    }

    result = nvmlShutdown();
    printf("nvmlShutdown: %d\n", result);

    printf("Done\n");
    return 0;
}
