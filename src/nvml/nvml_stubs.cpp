#include "nvml_defs.hpp"
#include "../core/global_state.hpp"
#include "../core/logging.hpp"
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
    FGPU_LOG("[FakeNVML] nvmlInit called\n");
    GlobalState::instance().initialize();
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlInitWithFlags(unsigned int flags) {
    FGPU_LOG("[FakeNVML] nvmlInitWithFlags called with flags=%u\n", flags);
    GlobalState::instance().initialize();
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlShutdown() {
    FGPU_LOG("[FakeNVML] nvmlShutdown called\n");
    // Dump the monitor report before shutdown
    dump_monitor_report();
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetCount(unsigned int *deviceCount) {
    if (!deviceCount) return NVML_ERROR_INVALID_ARGUMENT;
    *deviceCount = GlobalState::instance().get_device_count();
    FGPU_LOG("[FakeNVML] nvmlDeviceGetCount returning %d\n", *deviceCount);
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
    FGPU_LOG("[FakeNVML] nvmlDeviceGetHandleByIndex(%d) returning %p\n", index, *device);
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

    // Fill in both busId fields
    snprintf(pci->busIdLegacy, sizeof(pci->busIdLegacy), "%s", dev->pci_bus_id.c_str());
    snprintf(pci->busId, sizeof(pci->busId), "%s", dev->pci_bus_id.c_str());
    pci->domain = 0;
    pci->bus = dev->index + 1;
    pci->device = 0;
    pci->pciDeviceId = 0x20B010DE;  // A100 device ID (reversed: 0x10DE20B0)
    pci->pciSubSystemId = 0x145010DE;

    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_v3_t *pci) {
    if (!device || !pci) return NVML_ERROR_INVALID_ARGUMENT;
    Device* dev = (Device*)device;

    // Only fill the base fields that exist in nvmlPciInfo_t as well.
    // nvitop loads nvmlDeviceGetPciInfo_v3 but allocates nvmlPciInfo_t (68 bytes),
    // so touching the reserved fields would overflow the caller's buffer.
    memset(pci, 0, sizeof(nvmlPciInfo_t));
    // Fill in both busId fields
    snprintf(pci->busIdLegacy, sizeof(pci->busIdLegacy), "%s", dev->pci_bus_id.c_str());
    snprintf(pci->busId, sizeof(pci->busId), "%s", dev->pci_bus_id.c_str());
    pci->domain = 0;
    pci->bus = dev->index + 1;
    pci->device = 0;
    pci->pciDeviceId = 0x20B010DE;  // A100 device ID
    pci->pciSubSystemId = 0x145010DE;

    return NVML_SUCCESS;
}

// nvmlDeviceGetTupleIndex is not part of public NVML, but nvitop queries it
// when building snapshots for MIG-aware layouts. We just return the device
// index to keep it happy.
nvmlReturn_t nvmlDeviceGetTupleIndex(nvmlDevice_t device, unsigned int *tupleIndex) {
    if (!device || !tupleIndex) return NVML_ERROR_INVALID_ARGUMENT;
    Device* dev = (Device*)device;
    *tupleIndex = dev->index;
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
    FGPU_LOG("[FakeNVML] nvmlSystemGetDriverVersion returning: %s\n", version);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlSystemGetNVMLVersion(char *version, unsigned int length) {
    if (!version) return NVML_ERROR_INVALID_ARGUMENT;
    snprintf(version, length, "12.570.195");
    FGPU_LOG("[FakeNVML] nvmlSystemGetNVMLVersion returning: %s\n", version);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlSystemGetCudaDriverVersion(int *cudaDriverVersion) {
    if (!cudaDriverVersion) return NVML_ERROR_INVALID_ARGUMENT;
    *cudaDriverVersion = 12080;  // CUDA 12.8
    FGPU_LOG("[FakeNVML] nvmlSystemGetCudaDriverVersion returning: %d\n", *cudaDriverVersion);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlSystemGetCudaDriverVersion_v2(int *cudaDriverVersion) {
    return nvmlSystemGetCudaDriverVersion(cudaDriverVersion);
}

// Additional functions required by nvitop

nvmlReturn_t nvmlDeviceGetDisplayActive(nvmlDevice_t device, unsigned int *isActive) {
    if (!device || !isActive) return NVML_ERROR_INVALID_ARGUMENT;
    *isActive = 0;  // Display not active (server GPU)
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPersistenceMode(nvmlDevice_t device, unsigned int *mode) {
    if (!device || !mode) return NVML_ERROR_INVALID_ARGUMENT;
    *mode = 0;  // Not in persistence mode
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetComputeMode(nvmlDevice_t device, unsigned int *mode) {
    if (!device || !mode) return NVML_ERROR_INVALID_ARGUMENT;
    *mode = 0;  // NVML_COMPUTEMODE_DEFAULT
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetEccMode(nvmlDevice_t device, unsigned int *current, unsigned int *pending) {
    if (!device) return NVML_ERROR_INVALID_ARGUMENT;
    if (current) *current = 0;
    if (pending) *pending = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetTotalEccErrors(nvmlDevice_t device, unsigned int errorType, unsigned int counterType, unsigned long long *eccCounts) {
    if (!device || !eccCounts) return NVML_ERROR_INVALID_ARGUMENT;
    *eccCounts = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMigMode(nvmlDevice_t device, unsigned int *currentMode, unsigned int *pendingMode) {
    if (!device) return NVML_ERROR_INVALID_ARGUMENT;
    if (currentMode) *currentMode = 0;  // MIG disabled
    if (pendingMode) *pendingMode = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetDriverModel(nvmlDevice_t device, unsigned int *current, unsigned int *pending) {
    if (!device) return NVML_ERROR_INVALID_ARGUMENT;
    // WDDM (Windows) = 0, TCC = 1, Linux doesn't use this
    if (current) *current = 0;
    if (pending) *pending = 0;
    return NVML_SUCCESS;
}

// Process information - return empty lists (no running processes)
// NVML uses fixed-size arrays, returning count=0 means no processes

nvmlReturn_t nvmlDeviceGetComputeRunningProcesses(nvmlDevice_t device, unsigned int *infoCount, void *infos) {
    if (!device || !infoCount) return NVML_ERROR_INVALID_ARGUMENT;
    *infoCount = 0;  // No running compute processes
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v2(nvmlDevice_t device, unsigned int *infoCount, void *infos) {
    return nvmlDeviceGetComputeRunningProcesses(device, infoCount, infos);
}

nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v3(nvmlDevice_t device, unsigned int *infoCount, void *infos) {
    return nvmlDeviceGetComputeRunningProcesses(device, infoCount, infos);
}

nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses(nvmlDevice_t device, unsigned int *infoCount, void *infos) {
    if (!device || !infoCount) return NVML_ERROR_INVALID_ARGUMENT;
    *infoCount = 0;  // No running graphics processes
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses_v2(nvmlDevice_t device, unsigned int *infoCount, void *infos) {
    return nvmlDeviceGetGraphicsRunningProcesses(device, infoCount, infos);
}

nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses_v3(nvmlDevice_t device, unsigned int *infoCount, void *infos) {
    return nvmlDeviceGetGraphicsRunningProcesses(device, infoCount, infos);
}

nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses(nvmlDevice_t device, unsigned int *infoCount, void *infos) {
    if (!device || !infoCount) return NVML_ERROR_INVALID_ARGUMENT;
    *infoCount = 0;  // No MPS processes
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses_v2(nvmlDevice_t device, unsigned int *infoCount, void *infos) {
    return nvmlDeviceGetMPSComputeRunningProcesses(device, infoCount, infos);
}

nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses_v3(nvmlDevice_t device, unsigned int *infoCount, void *infos) {
    return nvmlDeviceGetMPSComputeRunningProcesses(device, infoCount, infos);
}

// Clock-related functions
nvmlReturn_t nvmlDeviceGetMaxCustomerBoostClock(nvmlDevice_t device, unsigned int clockType, unsigned int *clockMHz) {
    if (!device || !clockMHz) return NVML_ERROR_INVALID_ARGUMENT;
    *clockMHz = 1410;  // A100 boost clock
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMinMaxClockOfPState(nvmlDevice_t device, unsigned int clockType, unsigned int pState, unsigned int *minClockMHz, unsigned int *maxClockMHz) {
    if (!device) return NVML_ERROR_INVALID_ARGUMENT;
    if (minClockMHz) *minClockMHz = 210;
    if (maxClockMHz) *maxClockMHz = 1410;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetApplicationsClock(nvmlDevice_t device, unsigned int clockType, unsigned int *clockMHz) {
    if (!device || !clockMHz) return NVML_ERROR_INVALID_ARGUMENT;
    *clockMHz = 1410;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetDefaultApplicationsClock(nvmlDevice_t device, unsigned int clockType, unsigned int *clockMHz) {
    if (!device || !clockMHz) return NVML_ERROR_INVALID_ARGUMENT;
    *clockMHz = 1410;
    return NVML_SUCCESS;
}

// Power management
nvmlReturn_t nvmlDeviceGetEnforcedPowerLimit(nvmlDevice_t device, unsigned int *limit) {
    if (!device || !limit) return NVML_ERROR_INVALID_ARGUMENT;
    *limit = 400000;  // 400W
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPowerManagementLimitConstraints(nvmlDevice_t device, unsigned int *minLimit, unsigned int *maxLimit) {
    if (!device) return NVML_ERROR_INVALID_ARGUMENT;
    if (minLimit) *minLimit = 100000;   // 100W min
    if (maxLimit) *maxLimit = 400000;   // 400W max
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPowerManagementDefaultLimit(nvmlDevice_t device, unsigned int *defaultLimit) {
    if (!device || !defaultLimit) return NVML_ERROR_INVALID_ARGUMENT;
    *defaultLimit = 300000;  // 300W default
    return NVML_SUCCESS;
}

// Architecture info
nvmlReturn_t nvmlDeviceGetArchitecture(nvmlDevice_t device, unsigned int *arch) {
    if (!device || !arch) return NVML_ERROR_INVALID_ARGUMENT;
    *arch = 8;  // NVML_DEVICE_ARCH_AMPERE
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int *major, int *minor) {
    if (!device) return NVML_ERROR_INVALID_ARGUMENT;
    if (major) *major = 8;
    if (minor) *minor = 0;
    return NVML_SUCCESS;
}

// BAR1 Memory info (for some nvidia-smi compatibility)
nvmlReturn_t nvmlDeviceGetBAR1MemoryInfo(nvmlDevice_t device, void *bar1Memory) {
    if (!device || !bar1Memory) return NVML_ERROR_INVALID_ARGUMENT;
    struct { unsigned long long bar1Total; unsigned long long bar1Free; unsigned long long bar1Used; } *mem = 
        (decltype(mem))bar1Memory;
    mem->bar1Total = 256ULL * 1024 * 1024;  // 256MB BAR1
    mem->bar1Free = 256ULL * 1024 * 1024;
    mem->bar1Used = 0;
    return NVML_SUCCESS;
}

// Board/serial info
nvmlReturn_t nvmlDeviceGetBoardId(nvmlDevice_t device, unsigned int *boardId) {
    if (!device || !boardId) return NVML_ERROR_INVALID_ARGUMENT;
    *boardId = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetSerial(nvmlDevice_t device, char *serial, unsigned int length) {
    if (!device || !serial) return NVML_ERROR_INVALID_ARGUMENT;
    snprintf(serial, length, "N/A");
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetBrand(nvmlDevice_t device, unsigned int *type) {
    if (!device || !type) return NVML_ERROR_INVALID_ARGUMENT;
    *type = 0;  // NVML_BRAND_UNKNOWN
    return NVML_SUCCESS;
}

// VBIOS
nvmlReturn_t nvmlDeviceGetVbiosVersion(nvmlDevice_t device, char *version, unsigned int length) {
    if (!device || !version) return NVML_ERROR_INVALID_ARGUMENT;
    snprintf(version, length, "92.00.00.00.00");
    return NVML_SUCCESS;
}

// Info ROM
nvmlReturn_t nvmlDeviceGetInforomVersion(nvmlDevice_t device, unsigned int object, char *version, unsigned int length) {
    if (!device || !version) return NVML_ERROR_INVALID_ARGUMENT;
    snprintf(version, length, "N/A");
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetInforomImageVersion(nvmlDevice_t device, char *version, unsigned int length) {
    if (!device || !version) return NVML_ERROR_INVALID_ARGUMENT;
    snprintf(version, length, "N/A");
    return NVML_SUCCESS;
}

// NVLink
nvmlReturn_t nvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link, unsigned int *isActive) {
    if (!device || !isActive) return NVML_ERROR_INVALID_ARGUMENT;
    *isActive = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetNvLinkCapability(nvmlDevice_t device, unsigned int link, unsigned int capability, unsigned int *capResult) {
    if (!device || !capResult) return NVML_ERROR_INVALID_ARGUMENT;
    *capResult = 0;
    return NVML_SUCCESS;
}

// Retired pages
nvmlReturn_t nvmlDeviceGetRetiredPages(nvmlDevice_t device, unsigned int cause, unsigned int *pageCount, unsigned long long *addresses) {
    if (!device || !pageCount) return NVML_ERROR_INVALID_ARGUMENT;
    *pageCount = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetRetiredPagesPendingStatus(nvmlDevice_t device, unsigned int *isPending) {
    if (!device || !isPending) return NVML_ERROR_INVALID_ARGUMENT;
    *isPending = 0;
    return NVML_SUCCESS;
}

// PCIe throughput
nvmlReturn_t nvmlDeviceGetPcieThroughput(nvmlDevice_t device, unsigned int counter, unsigned int *value) {
    if (!device || !value) return NVML_ERROR_INVALID_ARGUMENT;
    // counter: 0 = TX bytes/s, 1 = RX bytes/s
    *value = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPcieReplayCounter(nvmlDevice_t device, unsigned int *value) {
    if (!device || !value) return NVML_ERROR_INVALID_ARGUMENT;
    *value = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPcieLinkGeneration(nvmlDevice_t device, unsigned int *currLinkGen) {
    if (!device || !currLinkGen) return NVML_ERROR_INVALID_ARGUMENT;
    *currLinkGen = 4;  // PCIe Gen4
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPcieLinkWidth(nvmlDevice_t device, unsigned int *currLinkWidth) {
    if (!device || !currLinkWidth) return NVML_ERROR_INVALID_ARGUMENT;
    *currLinkWidth = 16;  // x16
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetCurrPcieLinkGeneration(nvmlDevice_t device, unsigned int *currLinkGen) {
    return nvmlDeviceGetPcieLinkGeneration(device, currLinkGen);
}

nvmlReturn_t nvmlDeviceGetCurrPcieLinkWidth(nvmlDevice_t device, unsigned int *currLinkWidth) {
    return nvmlDeviceGetPcieLinkWidth(device, currLinkWidth);
}

nvmlReturn_t nvmlDeviceGetMaxPcieLinkGeneration(nvmlDevice_t device, unsigned int *maxLinkGen) {
    if (!device || !maxLinkGen) return NVML_ERROR_INVALID_ARGUMENT;
    *maxLinkGen = 4;  // PCIe Gen4
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMaxPcieLinkWidth(nvmlDevice_t device, unsigned int *maxLinkWidth) {
    if (!device || !maxLinkWidth) return NVML_ERROR_INVALID_ARGUMENT;
    *maxLinkWidth = 16;  // x16
    return NVML_SUCCESS;
}

// Display mode
nvmlReturn_t nvmlDeviceGetDisplayMode(nvmlDevice_t device, unsigned int *display) {
    if (!device || !display) return NVML_ERROR_INVALID_ARGUMENT;
    *display = 0;  // Display disabled (server GPU)
    return NVML_SUCCESS;
}

// Handle by UUID and PCI Bus ID
nvmlReturn_t nvmlDeviceGetHandleByUUID(const char *uuid, nvmlDevice_t *device) {
    if (!uuid || !device) return NVML_ERROR_INVALID_ARGUMENT;
    // For simplicity, return device 0
    Device& dev = GlobalState::instance().get_device(0);
    *device = (nvmlDevice_t)&dev;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetHandleByPciBusId(const char *pciBusId, nvmlDevice_t *device) {
    if (!pciBusId || !device) return NVML_ERROR_INVALID_ARGUMENT;
    // For simplicity, return device 0
    Device& dev = GlobalState::instance().get_device(0);
    *device = (nvmlDevice_t)&dev;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetHandleByPciBusId_v2(const char *pciBusId, nvmlDevice_t *device) {
    return nvmlDeviceGetHandleByPciBusId(pciBusId, device);
}

nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int *index) {
    if (!device || !index) return NVML_ERROR_INVALID_ARGUMENT;
    Device* dev = (Device*)device;
    *index = dev->index;
    return NVML_SUCCESS;
}

// Unit queries (for compute clusters)
nvmlReturn_t nvmlUnitGetCount(unsigned int *unitCount) {
    if (!unitCount) return NVML_ERROR_INVALID_ARGUMENT;
    *unitCount = 0;
    return NVML_SUCCESS;
}

// Event set (stub implementations)
nvmlReturn_t nvmlEventSetCreate(void **set) {
    if (!set) return NVML_ERROR_INVALID_ARGUMENT;
    *set = (void*)0x1;  // Fake handle
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlEventSetFree(void *set) {
    return NVML_SUCCESS;
}

// Accounting (stub)
nvmlReturn_t nvmlDeviceGetAccountingMode(nvmlDevice_t device, unsigned int *mode) {
    if (!device || !mode) return NVML_ERROR_INVALID_ARGUMENT;
    *mode = 0;  // Disabled
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetAccountingPids(nvmlDevice_t device, unsigned int *count, unsigned int *pids) {
    if (!device || !count) return NVML_ERROR_INVALID_ARGUMENT;
    *count = 0;
    return NVML_SUCCESS;
}

// MIG-related functions
nvmlReturn_t nvmlDeviceGetMaxMigDeviceCount(nvmlDevice_t device, unsigned int *count) {
    if (!device || !count) return NVML_ERROR_INVALID_ARGUMENT;
    *count = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMigDeviceHandleByIndex(nvmlDevice_t device, unsigned int index, nvmlDevice_t *migDevice) {
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlDeviceGetGpuInstanceById(nvmlDevice_t device, unsigned int id, void *gpuInstance) {
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlDeviceGetGpuInstances(nvmlDevice_t device, unsigned int profileId, void *gpuInstances, unsigned int *count) {
    if (count) *count = 0;
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlDeviceIsMigDeviceHandle(nvmlDevice_t device, unsigned int *isMigDevice) {
    if (!device || !isMigDevice) return NVML_ERROR_INVALID_ARGUMENT;
    *isMigDevice = 0;  // Not a MIG device
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetDeviceHandleFromMigDeviceHandle(nvmlDevice_t migDevice, nvmlDevice_t *device) {
    return NVML_ERROR_NOT_SUPPORTED;
}

// Tuple index - used by nvitop for MIG devices
// nvmlDeviceGetTupleIndex doesn't exist in standard NVML, nvitop may construct it
// For non-MIG devices, return NOT_SUPPORTED

// Additional power functions
nvmlReturn_t nvmlDeviceGetPowerState(nvmlDevice_t device, unsigned int *pState) {
    if (!device || !pState) return NVML_ERROR_INVALID_ARGUMENT;
    *pState = 0;  // P0
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPowerSource(nvmlDevice_t device, unsigned int *powerSource) {
    if (!device || !powerSource) return NVML_ERROR_INVALID_ARGUMENT;
    *powerSource = 0;  // AC power
    return NVML_SUCCESS;
}

// Supported/enabled clocks
nvmlReturn_t nvmlDeviceGetSupportedGraphicsClocks(nvmlDevice_t device, unsigned int memoryClockMHz, unsigned int *count, unsigned int *clocksMHz) {
    if (!device || !count) return NVML_ERROR_INVALID_ARGUMENT;
    if (clocksMHz && *count > 0) {
        clocksMHz[0] = 1410;
    }
    *count = 1;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetSupportedMemoryClocks(nvmlDevice_t device, unsigned int *count, unsigned int *clocksMHz) {
    if (!device || !count) return NVML_ERROR_INVALID_ARGUMENT;
    if (clocksMHz && *count > 0) {
        clocksMHz[0] = 1215;
    }
    *count = 1;
    return NVML_SUCCESS;
}

// Lock clocks (no-op)
nvmlReturn_t nvmlDeviceSetGpuLockedClocks(nvmlDevice_t device, unsigned int minGpuClockMHz, unsigned int maxGpuClockMHz) {
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceResetGpuLockedClocks(nvmlDevice_t device) {
    return NVML_SUCCESS;
}

// Auto-boost
nvmlReturn_t nvmlDeviceGetAutoBoostedClocksEnabled(nvmlDevice_t device, unsigned int *isEnabled, unsigned int *defaultIsEnabled) {
    if (!device) return NVML_ERROR_INVALID_ARGUMENT;
    if (isEnabled) *isEnabled = 1;
    if (defaultIsEnabled) *defaultIsEnabled = 1;
    return NVML_SUCCESS;
}

// Remapped rows
nvmlReturn_t nvmlDeviceGetRemappedRows(nvmlDevice_t device, unsigned int *corrRows, unsigned int *uncRows, unsigned int *isPending, unsigned int *failureOccurred) {
    if (!device) return NVML_ERROR_INVALID_ARGUMENT;
    if (corrRows) *corrRows = 0;
    if (uncRows) *uncRows = 0;
    if (isPending) *isPending = 0;
    if (failureOccurred) *failureOccurred = 0;
    return NVML_SUCCESS;
}

// FieldValues (for batch queries)
nvmlReturn_t nvmlDeviceGetFieldValues(nvmlDevice_t device, int valuesCount, void *values) {
    // Return not supported for batch field value queries
    return NVML_ERROR_NOT_SUPPORTED;
}

// Samples
nvmlReturn_t nvmlDeviceGetSamples(nvmlDevice_t device, unsigned int type, unsigned long long lastSeenTimeStamp, unsigned int *sampleValType, unsigned int *sampleCount, void *samples) {
    if (!device || !sampleCount) return NVML_ERROR_INVALID_ARGUMENT;
    *sampleCount = 0;
    return NVML_SUCCESS;
}

} // extern C
