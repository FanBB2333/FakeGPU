#include "nvml_defs.hpp"
#include "../core/global_state.hpp"
#include "../monitor/monitor.hpp"
#include <cstdio>
#include <cstring>

using namespace fake_gpu;

// Version strings that nvidia-smi may search for
extern "C" {
const char nvml_version_string[] = "12.570.195.03";
const char driver_version_string[] = "570.195.03";
}

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
    // Dump the monitor report before shutdown
    dump_monitor_report();
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

nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_v3_t *pci) {
    if (!device || !pci) return NVML_ERROR_INVALID_ARGUMENT;
    Device* dev = (Device*)device;

    // Fill in PCI info v3 from device
    snprintf(pci->busId, sizeof(pci->busId), "%s", dev->pci_bus_id.c_str());
    snprintf(pci->busIdLegacy, sizeof(pci->busIdLegacy), "%s", dev->pci_bus_id.c_str());
    pci->domain = 0;
    pci->bus = dev->index + 1;
    pci->device = 0;
    pci->pciDeviceId = 0x20B0;  // A100 device ID
    pci->pciSubSystemId = 0x1450;
    pci->reserved0 = 0;
    pci->reserved1 = 0;
    pci->reserved2 = 0;
    pci->reserved3 = 0;

    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t device, nvmlUtilization_t *utilization) {
    if (!device || !utilization) return NVML_ERROR_INVALID_ARGUMENT;

    // Return fake utilization rates
    // For a more realistic simulation, these could be randomized or based on actual allocations
    utilization->gpu = 50;      // 50% GPU utilization
    utilization->memory = 30;   // 30% memory utilization

    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t device, unsigned int *power) {
    if (!device || !power) return NVML_ERROR_INVALID_ARGUMENT;

    // Return fake power usage in milliwatts
    // A100 typical power is around 250-400W, so return 300W = 300000 mW
    *power = 300000;  // 300W in milliwatts

    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPowerManagementLimit(nvmlDevice_t device, unsigned int *limit) {
    if (!device || !limit) return NVML_ERROR_INVALID_ARGUMENT;

    // Return fake power limit in milliwatts
    // A100 max power is 400W
    *limit = 400000;  // 400W in milliwatts

    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetTemperature(nvmlDevice_t device, unsigned int sensorType, unsigned int *temp) {
    if (!device || !temp) return NVML_ERROR_INVALID_ARGUMENT;

    // Return fake temperature in Celsius
    // sensorType: 0 = GPU core temperature
    *temp = 65;  // 65°C

    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t device, unsigned int type, unsigned int *clock) {
    if (!device || !clock) return NVML_ERROR_INVALID_ARGUMENT;

    // Return fake clock speeds in MHz
    // type: 0 = graphics clock, 1 = SM clock, 2 = memory clock
    switch (type) {
        case 0:  // Graphics clock
            *clock = 1410;  // A100 boost clock
            break;
        case 1:  // SM clock
            *clock = 1410;
            break;
        case 2:  // Memory clock
            *clock = 1215;  // A100 memory clock
            break;
        default:
            *clock = 1410;
            break;
    }

    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetClock(nvmlDevice_t device, unsigned int clockType, unsigned int clockId, unsigned int *clockMHz) {
    if (!device || !clockMHz) return NVML_ERROR_INVALID_ARGUMENT;

    // clockType: 0 = graphics, 1 = SM, 2 = memory
    // clockId: typically 0 for current clock
    // Return fake clock speeds in MHz
    switch (clockType) {
        case 0:  // Graphics clock
            *clockMHz = 1410;  // A100 boost clock
            break;
        case 1:  // SM clock
            *clockMHz = 1410;
            break;
        case 2:  // Memory clock
            *clockMHz = 1215;  // A100 memory clock
            break;
        default:
            *clockMHz = 1410;
            break;
    }

    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMaxClockInfo(nvmlDevice_t device, unsigned int type, unsigned int *clock) {
    if (!device || !clock) return NVML_ERROR_INVALID_ARGUMENT;

    // Return fake max clock speeds in MHz
    switch (type) {
        case 0:  // Graphics clock
            *clock = 1410;  // A100 max boost clock
            break;
        case 1:  // SM clock
            *clock = 1410;
            break;
        case 2:  // Memory clock
            *clock = 1215;  // A100 max memory clock
            break;
        default:
            *clock = 1410;
            break;
    }

    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetFanSpeed(nvmlDevice_t device, unsigned int *speed) {
    if (!device || !speed) return NVML_ERROR_INVALID_ARGUMENT;

    // Return fake fan speed percentage
    *speed = 50;  // 50%

    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPerformanceState(nvmlDevice_t device, unsigned int *pState) {
    if (!device || !pState) return NVML_ERROR_INVALID_ARGUMENT;

    // Return fake performance state (P0 = maximum performance)
    *pState = 0;  // P0

    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetEncoderUtilization(nvmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs) {
    if (!device || !utilization || !samplingPeriodUs) return NVML_ERROR_INVALID_ARGUMENT;

    // Return fake encoder utilization
    *utilization = 0;  // 0% encoder utilization
    *samplingPeriodUs = 1000000;  // 1 second sampling period

    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetDecoderUtilization(nvmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs) {
    if (!device || !utilization || !samplingPeriodUs) return NVML_ERROR_INVALID_ARGUMENT;

    // Return fake decoder utilization
    *utilization = 0;  // 0% decoder utilization
    *samplingPeriodUs = 1000000;  // 1 second sampling period

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

// System query functions
nvmlReturn_t nvmlSystemGetDriverVersion(char *version, unsigned int length) {
    if (!version) return NVML_ERROR_INVALID_ARGUMENT;
    snprintf(version, length, "570.195.03");
    printf("[FakeNVML] nvmlSystemGetDriverVersion returning: %s\n", version);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlSystemGetNVMLVersion(char *version, unsigned int length) {
    if (!version) return NVML_ERROR_INVALID_ARGUMENT;
    snprintf(version, length, "12.570.195");
    printf("[FakeNVML] nvmlSystemGetNVMLVersion returning: %s\n", version);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlSystemGetCudaDriverVersion(int *cudaDriverVersion) {
    if (!cudaDriverVersion) return NVML_ERROR_INVALID_ARGUMENT;
    *cudaDriverVersion = 12080;  // CUDA 12.8
    printf("[FakeNVML] nvmlSystemGetCudaDriverVersion returning: %d\n", *cudaDriverVersion);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlSystemGetCudaDriverVersion_v2(int *cudaDriverVersion) {
    return nvmlSystemGetCudaDriverVersion(cudaDriverVersion);
}

} // extern C
