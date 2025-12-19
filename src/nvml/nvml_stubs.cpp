#include "nvml_defs.hpp"
#include "../core/global_state.hpp"
#include "../core/logging.hpp"
#include "../monitor/monitor.hpp"
#include <cstdio>
#include <cstring>
#include <cstdlib>

using namespace fake_gpu;

extern "C" {
nvmlReturn_t nvmlDeviceGetGpuOperationMode(nvmlDevice_t device, unsigned int *current, unsigned int *pending);
nvmlReturn_t nvmlDeviceGetDisplayActive(nvmlDevice_t device, unsigned int *isActive);
nvmlReturn_t nvmlDeviceGetPowerManagementLimitConstraints(nvmlDevice_t device, unsigned int *minLimit, unsigned int *maxLimit);
nvmlReturn_t nvmlDeviceGetPowerManagementDefaultLimit(nvmlDevice_t device, unsigned int *defaultLimit);
nvmlReturn_t nvmlDeviceGetCurrentClocksThrottleReasons(nvmlDevice_t device, unsigned long long *reasons);
nvmlReturn_t nvmlDeviceGetSupportedClocksThrottleReasons(nvmlDevice_t device, unsigned long long *supportedReasons);
nvmlReturn_t nvmlDeviceGetEnforcedPowerLimit(nvmlDevice_t device, unsigned int *limit);
nvmlReturn_t nvmlDeviceGetBoardPartNumber(nvmlDevice_t device, char *partNumber, unsigned int length);
}

namespace {
// Export table layout expected by the nvidia-smi build shipped alongside driver 570.195.03.
// The table size (0x858) is discovered by observing the system NVML export table and is used
// as a feature probe by nvidia-smi before it falls back to the public entry points.
struct FakeNvmlExportTable_v570 {
    size_t size;
    void* reserved0;
    void* reserved1;
    void* functions[264];
};

static_assert(sizeof(FakeNvmlExportTable_v570) == 0x858, "Fake NVML export table size mismatch");

static FakeNvmlExportTable_v570& export_table_v570() {
    static FakeNvmlExportTable_v570 table = []() {
        FakeNvmlExportTable_v570 t{};
        t.size = sizeof(FakeNvmlExportTable_v570);

        // Newer nvidia-smi builds first resolve many entry points via
        // nvmlInternalGetExportTable and do not reliably fall back to dlsym()
        // for all queries. Provide the subset we implement that nvidia-smi
        // depends on for stable XML output.
        //
        // Table indices are in pointer-sized words from the start of the table.
        // The first three words are the header: size + 2 reserved pointers.
        auto set_word = [&](size_t index, void* fn) {
            if (index < 3) return;
            const size_t fn_index = index - 3;
            if (fn_index >= (sizeof(t.functions) / sizeof(t.functions[0]))) return;
            t.functions[fn_index] = fn;
        };

        // Power management and power limits.
        set_word(46, reinterpret_cast<void*>(&nvmlDeviceGetPowerManagementLimitConstraints));
        set_word(47, reinterpret_cast<void*>(&nvmlDeviceGetPowerManagementDefaultLimit));
        set_word(108, reinterpret_cast<void*>(&nvmlDeviceGetEnforcedPowerLimit));

        // Keep other function pointers null so callers fall back to public NVML symbols
        // exported by this library when possible.
        return t;
    }();
    return table;
}
} // namespace

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

// Internal export table lookup used by newer nvidia-smi builds.
// We don't expose any special tables, but we must return success so nvidia-smi
// will proceed to the regular NVML entry points.
nvmlReturn_t nvmlInternalGetExportTable(void **ppExportTable, const void *pExportTableId) {
    if (!ppExportTable || !pExportTableId) return NVML_ERROR_INVALID_ARGUMENT;

    // Log the first/last bytes of the requested GUID for debugging.
    const unsigned char *id = static_cast<const unsigned char*>(pExportTableId);
    FGPU_LOG("[FakeNVML] nvmlInternalGetExportTable id[0]=%02x id[15]=%02x\n", id[0], id[15]);
    if (std::getenv("FAKEGPU_NVML_EXPORTTABLE_TRACE")) {
        std::fprintf(stderr,
                     "[FakeNVML] nvmlInternalGetExportTable id=%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x\n",
                     id[0], id[1], id[2], id[3],
                     id[4], id[5],
                     id[6], id[7],
                     id[8], id[9],
                     id[10], id[11], id[12], id[13], id[14], id[15]);
    }

    *ppExportTable = &export_table_v570();
    return NVML_SUCCESS;
}

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

nvmlReturn_t nvmlDeviceGetAddressingMode(nvmlDevice_t device, nvmlDeviceAddressingMode_t *mode) {
    if (!device || !mode) return NVML_ERROR_INVALID_ARGUMENT;
    mode->version = nvmlDeviceAddressingMode_v1;
    mode->value = NVML_DEVICE_ADDRESSING_MODE_NONE;
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

nvmlReturn_t nvmlDeviceGetConfComputeProtectedMemoryUsage(nvmlDevice_t device, nvmlMemory_t *memory) {
    if (!device || !memory) return NVML_ERROR_INVALID_ARGUMENT;
    // FakeGPU does not emulate confidential computing; report empty usage.
    memory->total = 0;
    memory->used = 0;
    memory->free = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetConfComputeMemSizeInfo(nvmlDevice_t device, void *ccMemSizeInfo) {
    if (!device || !ccMemSizeInfo) return NVML_ERROR_INVALID_ARGUMENT;
    struct { unsigned long long protectedMemSizeKib; unsigned long long unprotectedMemSizeKib; } *info =
        (decltype(info))ccMemSizeInfo;
    info->protectedMemSizeKib = 0;
    info->unprotectedMemSizeKib = 0;
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
    pci->pciDeviceId = dev->profile.pci_device_id;
    pci->pciSubSystemId = dev->profile.pci_device_id;

    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPciInfoExt(nvmlDevice_t device, nvmlPciInfoExt_v1_t *pci) {
    if (!device || !pci) return NVML_ERROR_INVALID_ARGUMENT;
    Device* dev = (Device*)device;

    pci->version = nvmlPciInfoExt_v1;
    pci->domain = 0;
    pci->bus = static_cast<unsigned int>(dev->index + 1);
    pci->device = 0;
    pci->pciDeviceId = dev->profile.pci_device_id;
    pci->pciSubSystemId = dev->profile.pci_device_id;
    pci->baseClass = 3;  // Display controller
    pci->subClass = 0;
    snprintf(pci->busId, sizeof(pci->busId), "%s", dev->pci_bus_id.c_str());
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
    pci->pciDeviceId = dev->profile.pci_device_id;
    pci->pciSubSystemId = dev->profile.pci_device_id;

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

nvmlReturn_t nvmlDeviceGetJpgUtilization(nvmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs) {
    if (!device || !utilization || !samplingPeriodUs) return NVML_ERROR_INVALID_ARGUMENT;
    *utilization = 0;
    *samplingPeriodUs = 1000000;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetOfaUtilization(nvmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs) {
    if (!device || !utilization || !samplingPeriodUs) return NVML_ERROR_INVALID_ARGUMENT;
    *utilization = 0;
    *samplingPeriodUs = 1000000;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t device, unsigned int *power) {
    if (!device || !power) return NVML_ERROR_INVALID_ARGUMENT;
    if (std::getenv("FAKEGPU_NVML_TRACE_POWER")) {
        std::fprintf(stderr, "[FakeNVML] nvmlDeviceGetPowerUsage(dev=%p)\n", device);
    }

    // Return fake power usage in milliwatts.
    // This is the *measured draw*, not the power limit.
    *power = 1000;  // 1W

    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPowerManagementMode(nvmlDevice_t device, unsigned int *mode) {
    if (!device || !mode) return NVML_ERROR_INVALID_ARGUMENT;
    *mode = 1;  // Enabled
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPowerManagementLimit(nvmlDevice_t device, unsigned int *limit) {
    if (!device || !limit) return NVML_ERROR_INVALID_ARGUMENT;
    if (std::getenv("FAKEGPU_NVML_TRACE_POWER")) {
        std::fprintf(stderr, "[FakeNVML] nvmlDeviceGetPowerManagementLimit(dev=%p)\n", device);
    }

    Device* dev = (Device*)device;
    unsigned int default_limit_mw = dev->profile.typical_power_usage_mw;
    if (default_limit_mw > dev->profile.max_power_limit_mw) default_limit_mw = dev->profile.max_power_limit_mw;
    *limit = default_limit_mw;

    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetTemperature(nvmlDevice_t device, unsigned int sensorType, unsigned int *temp) {
    if (!device || !temp) return NVML_ERROR_INVALID_ARGUMENT;

    FGPU_LOG("[FakeNVML] nvmlDeviceGetTemperature sensor=%u\n", sensorType);

    // Return fake temperature in Celsius
    // sensorType: 0 = GPU core temperature
    *temp = 65;  // 65°C

    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetTemperatureThreshold(nvmlDevice_t device, unsigned int thresholdType, unsigned int *temp) {
    if (!device || !temp) return NVML_ERROR_INVALID_ARGUMENT;

    // Threshold constants follow NVML_TEMPERATURE_THRESHOLD_* (0..7).
    // Provide common values so nvidia-smi can populate temperature sections.
    switch (thresholdType) {
        case 0:  // Shutdown
            *temp = 97;
            return NVML_SUCCESS;
        case 1:  // Slowdown
            *temp = 94;
            return NVML_SUCCESS;
        case 3:  // GPU max
            *temp = 92;
            return NVML_SUCCESS;
        case 4:  // Acoustic min (used as target temp min)
            *temp = 65;
            return NVML_SUCCESS;
        case 5:  // Acoustic current (used as target temperature)
            *temp = 83;
            return NVML_SUCCESS;
        case 6:  // Acoustic max (used as target temp max)
            *temp = 90;
            return NVML_SUCCESS;
        default:
            return NVML_ERROR_NOT_SUPPORTED;
    }
}

nvmlReturn_t nvmlDeviceGetMarginTemperature(nvmlDevice_t device, nvmlMarginTemperature_v1_t *marginTempInfo) {
    if (!device || !marginTempInfo) return NVML_ERROR_INVALID_ARGUMENT;
    // Most FakeGPU profiles don't model margin-to-tlimit; report not supported so
    // nvidia-smi renders N/A (matching many real consumer GPUs).
    return NVML_ERROR_NOT_SUPPORTED;
}

// Newer temperature query variant used by some tools (e.g., nvidia-smi)
// The real signature takes a versioned struct pointer; we return NOT_SUPPORTED
// so callers fall back to the basic nvmlDeviceGetTemperature path.
nvmlReturn_t nvmlDeviceGetTemperatureV(nvmlDevice_t device, void *tempInfo) {
    FGPU_LOG("[FakeNVML] nvmlDeviceGetTemperatureV (unsupported path)\n");
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t device, unsigned int type, unsigned int *clock) {
    if (!device || !clock) return NVML_ERROR_INVALID_ARGUMENT;
    Device* dev = (Device*)device;

    // Return fake clock speeds in MHz
    // type: 0 = graphics clock, 1 = SM clock, 2 = memory clock
    switch (type) {
        case 0:  // Graphics clock
            *clock = dev->profile.core_clock_mhz;
            break;
        case 1:  // SM clock
            *clock = dev->profile.core_clock_mhz;
            break;
        case 2:  // Memory clock
            *clock = dev->profile.memory_clock_mhz;
            break;
        default:
            *clock = dev->profile.core_clock_mhz;
            break;
    }

    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetClock(nvmlDevice_t device, unsigned int clockType, unsigned int clockId, unsigned int *clockMHz) {
    if (!device || !clockMHz) return NVML_ERROR_INVALID_ARGUMENT;
    Device* dev = (Device*)device;

    // clockType: 0 = graphics, 1 = SM, 2 = memory
    // clockId: typically 0 for current clock
    // Return fake clock speeds in MHz
    switch (clockType) {
        case 0:  // Graphics clock
            *clockMHz = dev->profile.core_clock_mhz;
            break;
        case 1:  // SM clock
            *clockMHz = dev->profile.core_clock_mhz;
            break;
        case 2:  // Memory clock
            *clockMHz = dev->profile.memory_clock_mhz;
            break;
        default:
            *clockMHz = dev->profile.core_clock_mhz;
            break;
    }

    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMaxClockInfo(nvmlDevice_t device, unsigned int type, unsigned int *clock) {
    if (!device || !clock) return NVML_ERROR_INVALID_ARGUMENT;
    Device* dev = (Device*)device;

    // Return fake max clock speeds in MHz
    switch (type) {
        case 0:  // Graphics clock
            *clock = dev->profile.core_clock_mhz;
            break;
        case 1:  // SM clock
            *clock = dev->profile.core_clock_mhz;
            break;
        case 2:  // Memory clock
            *clock = dev->profile.memory_clock_mhz;
            break;
        default:
            *clock = dev->profile.core_clock_mhz;
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

nvmlReturn_t nvmlDeviceGetEncoderStats(nvmlDevice_t device, unsigned long long *sessionCount, unsigned long long *averageFps, unsigned long long *averageLatency) {
    if (!device || !sessionCount || !averageFps || !averageLatency) return NVML_ERROR_INVALID_ARGUMENT;
    *sessionCount = 0;
    *averageFps = 0;
    *averageLatency = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetFBCStats(nvmlDevice_t device, nvmlFBCStats_t *stats) {
    if (!device || !stats) return NVML_ERROR_INVALID_ARGUMENT;
    stats->sessionsCount = 0;
    stats->averageFPS = 0;
    stats->averageLatency = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetDramEncryptionMode(nvmlDevice_t device, nvmlDramEncryptionInfo_t *current, nvmlDramEncryptionInfo_t *pending) {
    if (!device || !current || !pending) return NVML_ERROR_INVALID_ARGUMENT;
    current->version = nvmlDramEncryptionInfo_v1;
    current->encryptionState = 0;  // Disabled
    pending->version = nvmlDramEncryptionInfo_v1;
    pending->encryptionState = 0;
    return NVML_SUCCESS;
}

const char* nvmlErrorString(nvmlReturn_t result) {
    switch (result) {
        case NVML_SUCCESS: return "Success";
        case NVML_ERROR_UNINITIALIZED: return "Uninitialized";
        case NVML_ERROR_INVALID_ARGUMENT: return "Invalid Argument";
        case NVML_ERROR_NOT_SUPPORTED: return "Not Supported";
        case NVML_ERROR_FUNCTION_NOT_FOUND: return "Function Not Found";
        case NVML_ERROR_NOT_FOUND: return "Not Found";
        case NVML_ERROR_NO_PERMISSION: return "No Permission";
        case NVML_ERROR_INSUFFICIENT_SIZE: return "Insufficient Size";
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

nvmlReturn_t nvmlDeviceGetDriverModel_v2(nvmlDevice_t device, unsigned int *current, unsigned int *pending) {
    return nvmlDeviceGetDriverModel(device, current, pending);
}

nvmlReturn_t nvmlDeviceGetMinorNumber(nvmlDevice_t device, unsigned int *minorNumber) {
    if (!device || !minorNumber) return NVML_ERROR_INVALID_ARGUMENT;
    Device* dev = (Device*)device;
    *minorNumber = static_cast<unsigned int>(dev->index);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMultiGpuBoard(nvmlDevice_t device, unsigned int *multiGpu) {
    if (!device || !multiGpu) return NVML_ERROR_INVALID_ARGUMENT;
    *multiGpu = 0;  // No
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetBoardPartNumber(nvmlDevice_t device, char *partNumber, unsigned int length) {
    if (!device || !partNumber) return NVML_ERROR_INVALID_ARGUMENT;
    snprintf(partNumber, length, "N/A");
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetModuleId(nvmlDevice_t device, unsigned int *moduleId) {
    if (!device || !moduleId) return NVML_ERROR_INVALID_ARGUMENT;
    *moduleId = 1;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetAccountingBufferSize(nvmlDevice_t device, unsigned int *bufferSize) {
    if (!device || !bufferSize) return NVML_ERROR_INVALID_ARGUMENT;
    *bufferSize = 4000;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetLastBBXFlushTime(nvmlDevice_t device, unsigned long long *timestamp, unsigned long *durationUs) {
    if (!device || !timestamp || !durationUs) return NVML_ERROR_INVALID_ARGUMENT;
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlDeviceGetGpuOperationMode(nvmlDevice_t device, unsigned int *current, unsigned int *pending) {
    if (!device) return NVML_ERROR_INVALID_ARGUMENT;
    (void)current;
    (void)pending;
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlDeviceGetC2cModeInfoV(nvmlDevice_t device, nvmlC2cModeInfo_v1_t *info) {
    if (!device || !info) return NVML_ERROR_INVALID_ARGUMENT;
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlDeviceGetHostVgpuMode(nvmlDevice_t device, unsigned int *mode) {
    if (!device || !mode) return NVML_ERROR_INVALID_ARGUMENT;
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlDeviceGetVgpuHeterogeneousMode(nvmlDevice_t device, nvmlVgpuHeterogeneousMode_v1_t *mode) {
    if (!device || !mode) return NVML_ERROR_INVALID_ARGUMENT;
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlDeviceGetGspFirmwareVersion(nvmlDevice_t device, char *version) {
    if (!device || !version) return NVML_ERROR_INVALID_ARGUMENT;
    snprintf(version, NVML_GSP_FIRMWARE_VERSION_BUF_SIZE, "%s", driver_version_string);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetCurrentClocksThrottleReasons(nvmlDevice_t device, unsigned long long *reasons) {
    if (!device || !reasons) return NVML_ERROR_INVALID_ARGUMENT;
    *reasons = 0x1ULL;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetSupportedClocksThrottleReasons(nvmlDevice_t device, unsigned long long *supportedReasons) {
    if (!device || !supportedReasons) return NVML_ERROR_INVALID_ARGUMENT;
    *supportedReasons = 0x1ffULL;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetGpuMaxPcieLinkGeneration(nvmlDevice_t device, unsigned int *maxLinkGen) {
    if (!device || !maxLinkGen) return NVML_ERROR_INVALID_ARGUMENT;
    *maxLinkGen = 4;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetVirtualizationMode(nvmlDevice_t device, nvmlGpuVirtualizationMode_t *pVirtualMode) {
    if (!device || !pVirtualMode) return NVML_ERROR_INVALID_ARGUMENT;
    *pVirtualMode = NVML_GPU_VIRTUALIZATION_MODE_NONE;
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
    Device* dev = (Device*)device;
    *clockMHz = dev->profile.core_clock_mhz;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMinMaxClockOfPState(nvmlDevice_t device, unsigned int clockType, unsigned int pState, unsigned int *minClockMHz, unsigned int *maxClockMHz) {
    if (!device) return NVML_ERROR_INVALID_ARGUMENT;
    if (minClockMHz) *minClockMHz = 210;
    if (maxClockMHz) {
        Device* dev = (Device*)device;
        *maxClockMHz = dev->profile.core_clock_mhz;
    }
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetApplicationsClock(nvmlDevice_t device, unsigned int clockType, unsigned int *clockMHz) {
    if (!device || !clockMHz) return NVML_ERROR_INVALID_ARGUMENT;
    Device* dev = (Device*)device;
    *clockMHz = dev->profile.core_clock_mhz;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetDefaultApplicationsClock(nvmlDevice_t device, unsigned int clockType, unsigned int *clockMHz) {
    if (!device || !clockMHz) return NVML_ERROR_INVALID_ARGUMENT;
    Device* dev = (Device*)device;
    *clockMHz = dev->profile.core_clock_mhz;
    return NVML_SUCCESS;
}

// Power management
nvmlReturn_t nvmlDeviceGetEnforcedPowerLimit(nvmlDevice_t device, unsigned int *limit) {
    if (!device || !limit) return NVML_ERROR_INVALID_ARGUMENT;
    if (std::getenv("FAKEGPU_NVML_TRACE_POWER")) {
        std::fprintf(stderr, "[FakeNVML] nvmlDeviceGetEnforcedPowerLimit(dev=%p)\n", device);
    }
    Device* dev = (Device*)device;
    unsigned int default_limit_mw = dev->profile.typical_power_usage_mw;
    if (default_limit_mw > dev->profile.max_power_limit_mw) default_limit_mw = dev->profile.max_power_limit_mw;
    *limit = default_limit_mw;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPowerManagementLimitConstraints(nvmlDevice_t device, unsigned int *minLimit, unsigned int *maxLimit) {
    if (!device) return NVML_ERROR_INVALID_ARGUMENT;
    if (std::getenv("FAKEGPU_NVML_TRACE_POWER")) {
        std::fprintf(stderr, "[FakeNVML] nvmlDeviceGetPowerManagementLimitConstraints(dev=%p)\n", device);
    }
    if (minLimit) *minLimit = 100000;   // 100W min
    if (maxLimit) {
        Device* dev = (Device*)device;
        *maxLimit = dev->profile.max_power_limit_mw;
    }
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPowerManagementDefaultLimit(nvmlDevice_t device, unsigned int *defaultLimit) {
    if (!device || !defaultLimit) return NVML_ERROR_INVALID_ARGUMENT;
    if (std::getenv("FAKEGPU_NVML_TRACE_POWER")) {
        std::fprintf(stderr, "[FakeNVML] nvmlDeviceGetPowerManagementDefaultLimit(dev=%p)\n", device);
    }
    Device* dev = (Device*)device;
    unsigned int default_limit_mw = dev->profile.typical_power_usage_mw;
    if (default_limit_mw > dev->profile.max_power_limit_mw) default_limit_mw = dev->profile.max_power_limit_mw;
    *defaultLimit = default_limit_mw;
    return NVML_SUCCESS;
}

// Workload power profiles: not supported in FakeGPU.
nvmlReturn_t nvmlDeviceWorkloadPowerProfileGetProfilesInfo(nvmlDevice_t device, void *profilesInfo) {
    (void)device;
    (void)profilesInfo;
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlDeviceWorkloadPowerProfileGetCurrentProfiles(nvmlDevice_t device, void *currentProfiles) {
    (void)device;
    (void)currentProfiles;
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlDeviceWorkloadPowerProfileSetRequestedProfiles(nvmlDevice_t device, void *requestedProfiles) {
    (void)device;
    (void)requestedProfiles;
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlDeviceWorkloadPowerProfileClearRequestedProfiles(nvmlDevice_t device, void *requestedProfiles) {
    (void)device;
    (void)requestedProfiles;
    return NVML_ERROR_NOT_SUPPORTED;
}

// Architecture info
nvmlReturn_t nvmlDeviceGetArchitecture(nvmlDevice_t device, unsigned int *arch) {
    if (!device || !arch) return NVML_ERROR_INVALID_ARGUMENT;
    Device* dev = (Device*)device;
    unsigned int default_limit_mw = dev->profile.typical_power_usage_mw;
    if (default_limit_mw > dev->profile.max_power_limit_mw) default_limit_mw = dev->profile.max_power_limit_mw;
    *arch = static_cast<unsigned int>(dev->profile.architecture);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int *major, int *minor) {
    if (!device) return NVML_ERROR_INVALID_ARGUMENT;
    Device* dev = (Device*)device;
    if (major) *major = dev->profile.compute_major;
    if (minor) *minor = dev->profile.compute_minor;
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

nvmlReturn_t nvmlDeviceValidateInforom(nvmlDevice_t device) {
    if (!device) return NVML_ERROR_INVALID_ARGUMENT;
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

nvmlReturn_t nvmlDeviceGetGpuFabricInfoV(nvmlDevice_t device, void *gpuFabricInfo) {
    if (!device || !gpuFabricInfo) return NVML_ERROR_INVALID_ARGUMENT;

    // Versioned structs: v2/v3 begin with a version field.
    unsigned int version = *static_cast<unsigned int*>(gpuFabricInfo);
    if (version == nvmlGpuFabricInfo_v2) {
        nvmlGpuFabricInfo_v2_t *info = static_cast<nvmlGpuFabricInfo_v2_t*>(gpuFabricInfo);
        memset(info->clusterUuid, 0, sizeof(info->clusterUuid));
        info->status = NVML_ERROR_NOT_SUPPORTED;
        info->cliqueId = 0;
        info->state = 0;      // NOT_SUPPORTED
        info->healthMask = 0; // NOT_SUPPORTED
        return NVML_SUCCESS;
    }
    if (version == nvmlGpuFabricInfo_v3) {
        nvmlGpuFabricInfo_v3_t *info = static_cast<nvmlGpuFabricInfo_v3_t*>(gpuFabricInfo);
        memset(info->clusterUuid, 0, sizeof(info->clusterUuid));
        info->status = NVML_ERROR_NOT_SUPPORTED;
        info->cliqueId = 0;
        info->state = 0;
        info->healthMask = 0;
        info->healthSummary = 0;
        return NVML_SUCCESS;
    }
    return NVML_ERROR_INVALID_ARGUMENT;
}

nvmlReturn_t nvmlDeviceGetBridgeChipInfo(nvmlDevice_t device, nvmlBridgeChipHierarchy_t *bridgeHierarchy) {
    if (!device || !bridgeHierarchy) return NVML_ERROR_INVALID_ARGUMENT;
    bridgeHierarchy->bridgeCount = 0;
    return NVML_SUCCESS;
}

// Retired pages
nvmlReturn_t nvmlDeviceGetRetiredPages(nvmlDevice_t device, unsigned int cause, unsigned int *pageCount, unsigned long long *addresses) {
    if (!device || !pageCount) return NVML_ERROR_INVALID_ARGUMENT;
    *pageCount = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetRetiredPages_v2(nvmlDevice_t device, unsigned int cause, unsigned int *pageCount, unsigned long long *addresses, unsigned long long *timestamps) {
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
        Device* dev = (Device*)device;
        clocksMHz[0] = dev->profile.core_clock_mhz;
    }
    *count = 1;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetSupportedMemoryClocks(nvmlDevice_t device, unsigned int *count, unsigned int *clocksMHz) {
    if (!device || !count) return NVML_ERROR_INVALID_ARGUMENT;
    if (clocksMHz && *count > 0) {
        Device* dev = (Device*)device;
        clocksMHz[0] = dev->profile.memory_clock_mhz;
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

nvmlReturn_t nvmlDeviceGetRowRemapperHistogram(nvmlDevice_t device, nvmlRowRemapperHistogramValues_t *values) {
    if (!device || !values) return NVML_ERROR_INVALID_ARGUMENT;
    values->max = 192;
    values->high = 0;
    values->partial = 0;
    values->low = 0;
    values->none = 0;
    return NVML_SUCCESS;
}

// FieldValues (for batch queries)
nvmlReturn_t nvmlDeviceGetFieldValues(nvmlDevice_t device, int valuesCount, nvmlFieldValue_t *values) {
    if (!device || !values || valuesCount < 0) return NVML_ERROR_INVALID_ARGUMENT;

    // Implement the subset required by nvidia-smi power readings.
    // Field IDs based on NVML_FI_DEV_* constants (see pynvml).
    constexpr uint32_t NVML_FI_DEV_POWER_AVERAGE = 185;
    constexpr uint32_t NVML_FI_DEV_POWER_INSTANT = 186;
    constexpr uint32_t NVML_FI_DEV_POWER_MIN_LIMIT = 187;
    constexpr uint32_t NVML_FI_DEV_POWER_MAX_LIMIT = 188;
    constexpr uint32_t NVML_FI_DEV_POWER_DEFAULT_LIMIT = 189;
    constexpr uint32_t NVML_FI_DEV_POWER_CURRENT_LIMIT = 190;
    constexpr uint32_t NVML_FI_DEV_POWER_REQUESTED_LIMIT = 192;

    Device* dev = (Device*)device;
    unsigned int default_limit_mw = dev->profile.typical_power_usage_mw;
    if (default_limit_mw > dev->profile.max_power_limit_mw) default_limit_mw = dev->profile.max_power_limit_mw;

    for (int i = 0; i < valuesCount; ++i) {
        nvmlFieldValue_t &entry = values[i];
        entry.timestamp = 0;
        entry.latencyUsec = 0;

        // Default: mark field unsupported.
        entry.valueType = NVML_VALUE_TYPE_UNSIGNED_INT;
        entry.nvmlReturn = NVML_ERROR_NOT_SUPPORTED;
        entry.value.uiVal = 0;

        // Scope handling: nvidia-smi may query non-GPU scopes (e.g., module power).
        // Only scopeId==0 is supported for FakeGPU.
        if (entry.scopeId != 0) {
            continue;
        }

        switch (entry.fieldId) {
            case NVML_FI_DEV_POWER_AVERAGE:
            case NVML_FI_DEV_POWER_INSTANT:
                entry.nvmlReturn = NVML_SUCCESS;
                entry.value.uiVal = 1000;  // 1W in mW
                break;
            case NVML_FI_DEV_POWER_MIN_LIMIT:
                entry.nvmlReturn = NVML_SUCCESS;
                entry.value.uiVal = 100000;  // 100W in mW
                break;
            case NVML_FI_DEV_POWER_MAX_LIMIT:
                entry.nvmlReturn = NVML_SUCCESS;
                entry.value.uiVal = dev->profile.max_power_limit_mw;
                break;
            case NVML_FI_DEV_POWER_DEFAULT_LIMIT:
            case NVML_FI_DEV_POWER_CURRENT_LIMIT:
            case NVML_FI_DEV_POWER_REQUESTED_LIMIT:
                entry.nvmlReturn = NVML_SUCCESS;
                entry.value.uiVal = default_limit_mw;
                break;
            default:
                break;
        }

        if (std::getenv("FAKEGPU_NVML_TRACE_FIELD_VALUES")) {
            std::fprintf(stderr,
                         "[FakeNVML] nvmlDeviceGetFieldValues fieldId=%u scopeId=%u -> nvmlReturn=%d uiVal=%u\n",
                         entry.fieldId, entry.scopeId, entry.nvmlReturn, entry.value.uiVal);
        }
    }

    return NVML_SUCCESS;
}

// Samples
nvmlReturn_t nvmlDeviceGetSamples(nvmlDevice_t device, unsigned int type, unsigned long long lastSeenTimeStamp, unsigned int *sampleValType, unsigned int *sampleCount, void *samples) {
    if (!device || !sampleCount) return NVML_ERROR_INVALID_ARGUMENT;
    *sampleCount = 0;
    return NVML_SUCCESS;
}

} // extern C
