#include "nvml_defs.hpp"
#include "../core/global_state.hpp"
#include <cstdio>
#include <cstring>

using namespace fake_gpu;

// Helper to check init
static bool check_init() {
    // In a real implementation we would check a flag, but GlobalState handles itself
    return true; 
}

extern "C" {

nvmlReturn_t nvmlInit() {
    printf("[FakeNVML] nvmlInit called\n");
    GlobalState::instance().initialize();
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlInitWithFlags(unsigned int flags) {
    printf("[FakeNVML] nvmlInitWithFlags called with flags=%u\n", flags);
    GlobalState::instance().initialize();
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlShutdown() {
    printf("[FakeNVML] nvmlShutdown called\n");
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetCount(unsigned int *deviceCount) {
    if (!deviceCount) return NVML_ERROR_INVALID_ARGUMENT;
    *deviceCount = GlobalState::instance().get_device_count();
    printf("[FakeNVML] nvmlDeviceGetCount returning %d\n", *deviceCount);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t *device) {
    if (!device) return NVML_ERROR_INVALID_ARGUMENT;
    if (index >= GlobalState::instance().get_device_count()) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    // Just cast index to device pointer for simplicity, or use address of Device object
    Device& dev = GlobalState::instance().get_device(index);
    *device = (nvmlDevice_t)&dev;
    printf("[FakeNVML] nvmlDeviceGetHandleByIndex(%d) returning %p\n", index, *device);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char *name, unsigned int length) {
    if (!device || !name) return NVML_ERROR_INVALID_ARGUMENT;
    Device* dev = (Device*)device;
    snprintf(name, length, "%s", dev->name.c_str());
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char *uuid, unsigned int length) {
    if (!device || !uuid) return NVML_ERROR_INVALID_ARGUMENT;
    Device* dev = (Device*)device;
    snprintf(uuid, length, "%s", dev->uuid.c_str());
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t *memory) {
    if (!device || !memory) return NVML_ERROR_INVALID_ARGUMENT;
    Device* dev = (Device*)device;
    // Mock values
    memory->total = dev->total_memory;
    memory->used = dev->used_memory;
    memory->free = memory->total - memory->used;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t *memory) {
    if (!device || !memory) return NVML_ERROR_INVALID_ARGUMENT;
    Device* dev = (Device*)device;
    memory->version = nvmlMemory_v2;
    memory->total = dev->total_memory;
    memory->reserved = 0;
    memory->used = dev->used_memory;
    memory->free = memory->total - memory->used;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPciInfo(nvmlDevice_t device, nvmlPciInfo_t *pci) {
    if (!device || !pci) return NVML_ERROR_INVALID_ARGUMENT;
    Device* dev = (Device*)device;

    // Fill in PCI info from device
    snprintf(pci->busId, sizeof(pci->busId), "%s", dev->pci_bus_id.c_str());
    pci->domain = 0;
    pci->bus = dev->index + 1;
    pci->device = 0;
    pci->pciDeviceId = 0x20B0;  // A100 device ID
    pci->pciSubSystemId = 0x1450;

    return NVML_SUCCESS;
}

const char* nvmlErrorString(nvmlReturn_t result) {
    switch (result) {
        case NVML_SUCCESS: return "Success";
        case NVML_ERROR_UNINITIALIZED: return "Uninitialized";
        case NVML_ERROR_INVALID_ARGUMENT: return "Invalid Argument";
        default: return "Unknown Error";
    }
}

// Versioned function aliases (NVML v2 API)
nvmlReturn_t nvmlInit_v2() {
    return nvmlInit();
}

nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *deviceCount) {
    return nvmlDeviceGetCount(deviceCount);
}

nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t *device) {
    return nvmlDeviceGetHandleByIndex(index, device);
}

} // extern C
