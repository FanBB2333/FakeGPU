#include "nvml_defs.hpp"
#include "pci_bus_id.hpp"
#include "state_bridge.hpp"
#include "../core/backend_config.hpp"
#include "../core/global_state.hpp"
#include "../core/hybrid_memory_manager.hpp"
#include "../core/logging.hpp"
#include "../cuda/modeled_driver_version.hpp"
#include "../monitor/monitor.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <unistd.h>

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
        // Table indices are in pointer-sized words from the start of the table;
        // the first three words are the header (size + 2 reserved pointers).
        // Power management and power limits.
        t.functions[46 - 3] = reinterpret_cast<void*>(&nvmlDeviceGetPowerManagementLimitConstraints);
        t.functions[47 - 3] = reinterpret_cast<void*>(&nvmlDeviceGetPowerManagementDefaultLimit);
        t.functions[108 - 3] = reinterpret_cast<void*>(&nvmlDeviceGetEnforcedPowerLimit);

        // Keep other function pointers null so callers fall back to public NVML symbols
        // exported by this library when possible.
        return t;
    }();
    return table;
}
} // namespace

static constexpr const char* kDriverVersion = fake_gpu::kModeledDriverVersion;

// NVML counts nvmlInit calls: every entry point except nvmlInit*,
// nvmlShutdown, nvmlErrorString and nvmlSystemGetNVMLVersion returns
// NVML_ERROR_UNINITIALIZED before the first nvmlInit and after the matching
// last nvmlShutdown, even for invalid arguments (measured on driver 580).
static std::atomic<int> g_nvml_init_count{0};

#define NVML_REQUIRE_INIT()                                                   \
    do {                                                                      \
        if (g_nvml_init_count.load(std::memory_order_acquire) <= 0) {         \
            return NVML_ERROR_UNINITIALIZED;                                  \
        }                                                                     \
    } while (0)

static bool modeled_nvlink_peer(
    const Device* device,
    unsigned int link,
    ModeledNvLinkPeer& result) {
    if (nvml_bridge::enabled() || device->is_mig_device) {
        return false;
    }
    const ModeledDeviceTopology topology =
        GlobalState::instance().snapshot_device_topology();
    const std::size_t index = static_cast<std::size_t>(device->index);
    if (index >= topology.nvlink_peers.size()) {
        return false;
    }
    const auto& peers = topology.nvlink_peers[index];
    if (link >= peers.size()) {
        return false;
    }
    result = peers[link];
    return true;
}

// Slot of a physical GPU (MIG instances use their parent's); null in bridge mode.
static bool modeled_slot(const Device* device, ModeledSlot& result, ModeledDeviceTopology* topology_out = nullptr) {
    if (nvml_bridge::enabled()) {
        return false;
    }
    GlobalState& state = GlobalState::instance();
    const Device* physical = device;
    if (device->is_mig_device) {
        physical = state.get_parent_device_for_mig(const_cast<Device*>(device));
        if (!physical) return false;
    }
    ModeledDeviceTopology topology = state.snapshot_device_topology();
    const ModeledSlot* slot = topology.slot(physical->index);
    if (!slot) return false;
    result = *slot;
    if (topology_out) *topology_out = std::move(topology);
    return true;
}

// PCIe link to report: the modeled slot, the published topology in bridge
// mode, or the profile's own link.
static nvmlReturn_t modeled_pcie(nvmlDevice_t device, ModeledPcieLink& current, ModeledPcieLink& gpu_max) {
    const Device* dev = reinterpret_cast<const Device*>(device);
    FakeGpuNvmlLinksV1 published{};
    if (nvml_bridge::enabled() && nvml_bridge::links(device, &published) == NVML_SUCCESS) {
        if (!published.pcie_generation) return NVML_ERROR_NOT_SUPPORTED;
        current = {published.pcie_generation, published.pcie_width};
        gpu_max = {published.pcie_max_generation, published.pcie_max_width};
        return NVML_SUCCESS;
    }
    ModeledSlot slot;
    if (!modeled_slot(dev, slot)) {
        current.generation = dev->profile.pcie_generation > 0 ? static_cast<unsigned int>(dev->profile.pcie_generation) : 4U;
        current.width = dev->profile.pcie_lanes > 0 ? static_cast<unsigned int>(dev->profile.pcie_lanes) : 16U;
        gpu_max = current;
        return NVML_SUCCESS;
    }
    if (!slot.pcie.present()) return NVML_ERROR_NOT_SUPPORTED;
    current = slot.pcie;
    gpu_max = slot.gpu_max_pcie();
    return NVML_SUCCESS;
}

// Modeled copy counters of a physical GPU (MIG instances use their parent's).
static bool modeled_transfer(const Device* device, DeviceTransferStats& result) {
    if (nvml_bridge::enabled()) return false;
    GlobalState& state = GlobalState::instance();
    const Device* physical = device;
    if (device->is_mig_device) {
        physical = state.get_parent_device_for_mig(const_cast<Device*>(device));
        if (!physical) return false;
    }
    result = state.snapshot_device_transfer(physical->index);
    return true;
}

// Port count and cumulative NVLink bytes (TX, RX) of a GPU.
static bool modeled_nvlink_traffic(nvmlDevice_t device, unsigned int& ports, uint64_t& tx, uint64_t& rx) {
    FakeGpuNvmlLinksV1 published{};
    if (nvml_bridge::enabled()) {
        if (nvml_bridge::links(device, &published) != NVML_SUCCESS) return false;
        ports = published.nvlink_count;
        tx = published.nvlink_tx_bytes;
        rx = published.nvlink_rx_bytes;
        return true;
    }
    const Device* dev = reinterpret_cast<const Device*>(device);
    if (dev->is_mig_device || dev->index < 0) return false;
    const ModeledDeviceTopology topology = GlobalState::instance().snapshot_device_topology();
    const std::size_t index = static_cast<std::size_t>(dev->index);
    ports = index < topology.nvlink_peers.size() ? static_cast<unsigned int>(topology.nvlink_peers[index].size()) : 0;
    DeviceTransferStats transfer;
    if (!modeled_transfer(dev, transfer)) return false;
    tx = transfer.links[static_cast<int>(TransferLink::NvLink)].tx_bytes;
    rx = transfer.links[static_cast<int>(TransferLink::NvLink)].rx_bytes;
    return true;
}

// Writes only the nvmlPciInfo_t prefix: nvitop loads nvmlDeviceGetPciInfo_v3
// but allocates nvmlPciInfo_t (68 bytes), so touching the v3 reserved fields
// would overflow the caller's buffer.
// Bus ids use nvml.h's formats: busId "%08X:%02X:%02X.0", busIdLegacy "%04X:%02X:%02X.0".
template<class PciInfo>
static void fill_pci_info(const Device& device, PciInfo& pci) {
    memset(&pci, 0, sizeof(nvmlPciInfo_t));
    NvmlPciAddress address;
    if (parse_nvml_pci_bus_id(device.pci_bus_id.c_str(), address)) {
        format_nvml_pci_bus_id(address, pci.busId, sizeof(pci.busId), pci.busIdLegacy, sizeof(pci.busIdLegacy));
    } else {
        address.bus = static_cast<unsigned int>(device.index + 1);
        snprintf(pci.busIdLegacy, sizeof(pci.busIdLegacy), "%s", device.pci_bus_id.c_str());
        snprintf(pci.busId, sizeof(pci.busId), "%s", device.pci_bus_id.c_str());
    }
    pci.domain = address.domain;
    pci.bus = address.bus;
    pci.device = address.device;
    pci.pciDeviceId = device.profile.pci_device_id;
    pci.pciSubSystemId = device.profile.pci_device_id;
}

// type: 0 = graphics, 1 = SM, 2 = memory; other types report the core clock.
static unsigned int profile_clock_mhz(const Device* dev, unsigned int type) {
    return type == 2 ? dev->profile.memory_clock_mhz : dev->profile.core_clock_mhz;
}

// Power limit the device boots with and enforces, in mW; 0 when the catalog has
// no power data (a zero catalog field means unknown, not 0 W). The state bridge
// reports the catalog power cap its reference telemetry is drawn against.
static unsigned int default_power_limit_mw(const Device* dev) {
    const unsigned int cap = dev->profile.max_power_limit_mw;
    const unsigned int typical = dev->profile.typical_power_usage_mw;
    if (nvml_bridge::enabled() || !typical) return cap;
    return cap ? std::min(typical, cap) : typical;
}

// The limit must lie within the constraints: [min(100 W, limit), catalog cap].
static nvmlReturn_t power_limit_constraints_mw(const Device* dev, unsigned int& min_limit, unsigned int& max_limit) {
    const unsigned int limit = default_power_limit_mw(dev);
    if (!limit) return NVML_ERROR_NOT_SUPPORTED;
    min_limit = std::min(100000U, limit);
    max_limit = std::max(dev->profile.max_power_limit_mw, limit);
    return NVML_SUCCESS;
}

static void trace_power(const char* api, nvmlDevice_t device) {
    if (std::getenv("FAKEGPU_NVML_TRACE_POWER")) {
        std::fprintf(stderr, "[FakeNVML] %s(dev=%p)\n", api, device);
    }
}

// One published telemetry field of a bridge device.
static nvmlReturn_t read_telemetry(nvmlDevice_t device, unsigned int* out, uint32_t FakeGpuNvmlTelemetryV1::*field) {
    if (!out) return NVML_ERROR_INVALID_ARGUMENT;
    FakeGpuNvmlTelemetryV1 sample{};
    const auto status = nvml_bridge::telemetry(device, &sample);
    if (status == NVML_SUCCESS) *out = sample.*field;
    return status;
}

// Copies a string into a caller buffer of `length` bytes. Like NVML, a buffer
// without room for the terminator is NVML_ERROR_INSUFFICIENT_SIZE, not truncated.
static nvmlReturn_t copy_string(char* out, unsigned int length, const char* value) {
    const size_t size = std::strlen(value) + 1;
    if (length < size) return NVML_ERROR_INSUFFICIENT_SIZE;
    std::memcpy(out, value, size);
    return NVML_SUCCESS;
}

// Video and accelerator engines are never busy.
static nvmlReturn_t idle_engine_utilization(nvmlDevice_t device, unsigned int* utilization, unsigned int* samplingPeriodUs) {
    if (nvml_bridge::enabled()) return NVML_ERROR_NOT_SUPPORTED;
    if (!device || !utilization || !samplingPeriodUs) return NVML_ERROR_INVALID_ARGUMENT;
    *utilization = 0;
    *samplingPeriodUs = 1000000;  // 1 second sampling period
    return NVML_SUCCESS;
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
    if (!nvml_bridge::enabled()) GlobalState::instance().initialize();
    g_nvml_init_count.fetch_add(1, std::memory_order_acq_rel);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlInitWithFlags(unsigned int flags) {
    FGPU_LOG("[FakeNVML] nvmlInitWithFlags called with flags=%u\n", flags);
    return nvmlInit();
}

nvmlReturn_t nvmlShutdown() {
    FGPU_LOG("[FakeNVML] nvmlShutdown called\n");
    int count = g_nvml_init_count.load(std::memory_order_acquire);
    do {
        if (count <= 0) return NVML_ERROR_UNINITIALIZED;
    } while (!g_nvml_init_count.compare_exchange_weak(count, count - 1, std::memory_order_acq_rel));
    // Dump the monitor report when the last user shuts NVML down.
    if (count == 1 && !nvml_bridge::enabled()) dump_monitor_report();
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetCount(unsigned int *deviceCount) {
    NVML_REQUIRE_INIT();
    if (!deviceCount) return NVML_ERROR_INVALID_ARGUMENT;
    if (nvml_bridge::enabled()) return nvml_bridge::count(deviceCount);
    *deviceCount = GlobalState::instance().get_physical_device_count();
    FGPU_LOG("[FakeNVML] nvmlDeviceGetCount returning %d\n", *deviceCount);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t *device) {
    NVML_REQUIRE_INIT();
    if (!device) return NVML_ERROR_INVALID_ARGUMENT;
    if (nvml_bridge::enabled()) return nvml_bridge::handle(index, device);
    if (index >= GlobalState::instance().get_physical_device_count()) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    // Just cast index to device pointer for simplicity, or use address of Device object
    Device& dev = GlobalState::instance().get_physical_device(index);
    *device = (nvmlDevice_t)&dev;
    FGPU_LOG("[FakeNVML] nvmlDeviceGetHandleByIndex(%d) returning %p\n", index, *device);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char *name, unsigned int length) {
    NVML_REQUIRE_INIT();
    if (!device || !name) return NVML_ERROR_INVALID_ARGUMENT;
    return copy_string(name, length, ((Device*)device)->name.c_str());
}

nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char *uuid, unsigned int length) {
    NVML_REQUIRE_INIT();
    if (!device || !uuid) return NVML_ERROR_INVALID_ARGUMENT;
    return copy_string(uuid, length, ((Device*)device)->uuid.c_str());
}

nvmlReturn_t nvmlDeviceGetAddressingMode(nvmlDevice_t device, nvmlDeviceAddressingMode_t *mode) {
    NVML_REQUIRE_INIT();
    if (!device || !mode) return NVML_ERROR_INVALID_ARGUMENT;
    mode->version = nvmlDeviceAddressingMode_v1;
    mode->value = NVML_DEVICE_ADDRESSING_MODE_NONE;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t *memory) {
    NVML_REQUIRE_INIT();
    if (!device || !memory) return NVML_ERROR_INVALID_ARGUMENT;
    if (nvml_bridge::enabled()) return nvml_bridge::memory(device, memory);
    Device* dev = (Device*)device;
    if (!dev->is_mig_device) {
        const BackendConfig& config = BackendConfig::instance();
        if (config.mode() == FakeGpuMode::Hybrid && config.oom_policy() == OomPolicy::Clamp) {
            HybridMemoryManager::instance().initialize();
            size_t reported_total = 0;
            size_t reported_free = 0;
            HybridMemoryManager::instance().get_clamped_memory_info(
                dev->index, static_cast<size_t>(dev->total_memory), static_cast<size_t>(dev->used_memory),
                reported_total, reported_free);
            memory->total = reported_total;
            memory->free = reported_free;
            memory->used = reported_total - reported_free;
            return NVML_SUCCESS;
        }
    }

    // Virtual device info (MIG instances, simulate/hybrid non-clamp).
    memory->total = dev->total_memory;
    memory->used = dev->used_memory;
    memory->free = memory->total - memory->used;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t *memory) {
    NVML_REQUIRE_INIT();
    if (!device || !memory) return NVML_ERROR_INVALID_ARGUMENT;
    nvmlMemory_t snapshot{};
    const auto status = nvmlDeviceGetMemoryInfo(device, &snapshot);
    if (status != NVML_SUCCESS) return status;
    memory->version = nvmlMemory_v2;
    memory->total = snapshot.total;
    memory->used = snapshot.used;
    memory->free = snapshot.free;
    // Allocator reservations are already included in used; this field is
    // NVML driver/firmware reservation, not the PyTorch caching allocator.
    memory->reserved = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetConfComputeProtectedMemoryUsage(nvmlDevice_t device, nvmlMemory_t *memory) {
    NVML_REQUIRE_INIT();
    if (!device || !memory) return NVML_ERROR_INVALID_ARGUMENT;
    // FakeGPU does not emulate confidential computing; report empty usage.
    memory->total = 0;
    memory->used = 0;
    memory->free = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetConfComputeMemSizeInfo(nvmlDevice_t device, void *ccMemSizeInfo) {
    NVML_REQUIRE_INIT();
    if (!device || !ccMemSizeInfo) return NVML_ERROR_INVALID_ARGUMENT;
    struct { unsigned long long protectedMemSizeKib; unsigned long long unprotectedMemSizeKib; } *info =
        (decltype(info))ccMemSizeInfo;
    info->protectedMemSizeKib = 0;
    info->unprotectedMemSizeKib = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPciInfo(nvmlDevice_t device, nvmlPciInfo_t *pci) {
    NVML_REQUIRE_INIT();
    if (!device || !pci) return NVML_ERROR_INVALID_ARGUMENT;
    Device* dev = (Device*)device;
    fill_pci_info(*dev, *pci);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPciInfoExt(nvmlDevice_t device, nvmlPciInfoExt_v1_t *pci) {
    NVML_REQUIRE_INIT();
    if (!device || !pci) return NVML_ERROR_INVALID_ARGUMENT;
    Device* dev = (Device*)device;
    nvmlPciInfo_t info;
    fill_pci_info(*dev, info);

    pci->version = nvmlPciInfoExt_v1;
    pci->domain = info.domain;
    pci->bus = info.bus;
    pci->device = info.device;
    pci->pciDeviceId = info.pciDeviceId;
    pci->pciSubSystemId = info.pciSubSystemId;
    pci->baseClass = 3;  // Display controller
    pci->subClass = 0;
    memcpy(pci->busId, info.busId, sizeof(pci->busId));
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_v3_t *pci) {
    NVML_REQUIRE_INIT();
    if (!device || !pci) return NVML_ERROR_INVALID_ARGUMENT;
    Device* dev = (Device*)device;
    fill_pci_info(*dev, *pci);
    return NVML_SUCCESS;
}

// nvmlDeviceGetTupleIndex is not part of public NVML, but nvitop queries it
// when building snapshots for MIG-aware layouts. We just return the device
// index to keep it happy.
nvmlReturn_t nvmlDeviceGetTupleIndex(nvmlDevice_t device, unsigned int *tupleIndex) {
    NVML_REQUIRE_INIT();
    if (!device || !tupleIndex) return NVML_ERROR_INVALID_ARGUMENT;
    Device* dev = (Device*)device;
    *tupleIndex = dev->is_mig_device
        ? dev->mig_device_index
        : static_cast<unsigned int>(dev->index);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t device, nvmlUtilization_t *utilization) {
    NVML_REQUIRE_INIT();
    if (nvml_bridge::enabled()) {
        if (!utilization) return NVML_ERROR_INVALID_ARGUMENT;
        FakeGpuNvmlTelemetryV1 sample{};
        const auto status = nvml_bridge::telemetry(device, &sample);
        if (status == NVML_SUCCESS) { utilization->gpu = sample.gpu; utilization->memory = sample.memory; }
        return status;
    }
    if (!device || !utilization) return NVML_ERROR_INVALID_ARGUMENT;

    // Return fake utilization rates
    // For a more realistic simulation, these could be randomized or based on actual allocations
    utilization->gpu = 50;      // 50% GPU utilization
    utilization->memory = 30;   // 30% memory utilization

    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetJpgUtilization(nvmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs) {
    NVML_REQUIRE_INIT();
    return idle_engine_utilization(device, utilization, samplingPeriodUs);
}

nvmlReturn_t nvmlDeviceGetOfaUtilization(nvmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs) {
    NVML_REQUIRE_INIT();
    return idle_engine_utilization(device, utilization, samplingPeriodUs);
}

nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t device, unsigned int *power) {
    NVML_REQUIRE_INIT();
    if (nvml_bridge::enabled()) return read_telemetry(device, power, &FakeGpuNvmlTelemetryV1::power_mw);
    if (!device || !power) return NVML_ERROR_INVALID_ARGUMENT;
    trace_power("nvmlDeviceGetPowerUsage", device);

    // Return fake power usage in milliwatts.
    // This is the *measured draw*, not the power limit.
    *power = 1000;  // 1W

    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPowerManagementMode(nvmlDevice_t device, unsigned int *mode) {
    NVML_REQUIRE_INIT();
    if (!device || !mode) return NVML_ERROR_INVALID_ARGUMENT;
    *mode = 1;  // Enabled
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPowerManagementLimit(nvmlDevice_t device, unsigned int *limit) {
    NVML_REQUIRE_INIT();
    if (!device || !limit) return NVML_ERROR_INVALID_ARGUMENT;
    trace_power("nvmlDeviceGetPowerManagementLimit", device);

    const unsigned int value = default_power_limit_mw((Device*)device);
    if (!value) return NVML_ERROR_NOT_SUPPORTED;
    *limit = value;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetTemperature(nvmlDevice_t device, unsigned int sensorType, unsigned int *temp) {
    NVML_REQUIRE_INIT();
    if (nvml_bridge::enabled()) return read_telemetry(device, temp, &FakeGpuNvmlTelemetryV1::temperature);
    if (!device || !temp) return NVML_ERROR_INVALID_ARGUMENT;

    FGPU_LOG("[FakeNVML] nvmlDeviceGetTemperature sensor=%u\n", sensorType);

    // Return fake temperature in Celsius
    // sensorType: 0 = GPU core temperature
    *temp = 65;  // 65°C

    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetTemperatureThreshold(nvmlDevice_t device, unsigned int thresholdType, unsigned int *temp) {
    NVML_REQUIRE_INIT();
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
    NVML_REQUIRE_INIT();
    if (!device || !marginTempInfo) return NVML_ERROR_INVALID_ARGUMENT;
    // Most FakeGPU profiles don't model margin-to-tlimit; report not supported so
    // nvidia-smi renders N/A (matching many real consumer GPUs).
    return NVML_ERROR_NOT_SUPPORTED;
}

// Versioned temperature query (nvmlTemperature_v1_t, version nvmlTemperature_v1).
// It reports the same reading as nvmlDeviceGetTemperature; nvitop prefers it
// whenever the symbol exists.
nvmlReturn_t nvmlDeviceGetTemperatureV(nvmlDevice_t device, void *tempInfo) {
    NVML_REQUIRE_INIT();
    struct Temperature { unsigned int version, sensor; int temperature; };
    auto* value = static_cast<Temperature*>(tempInfo);
    if (!value || value->version != 0x100000C || value->sensor != 0) return NVML_ERROR_INVALID_ARGUMENT;
    unsigned int temperature = 0;
    const auto status = nvmlDeviceGetTemperature(device, value->sensor, &temperature);
    if (status == NVML_SUCCESS) value->temperature = static_cast<int>(temperature);
    return status;
}

nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t device, unsigned int type, unsigned int *clock) {
    NVML_REQUIRE_INIT();
    if (nvml_bridge::enabled()) {
        if (!clock || type > 2) return NVML_ERROR_INVALID_ARGUMENT;
        return read_telemetry(
            device, clock, type == 2 ? &FakeGpuNvmlTelemetryV1::memory_clock : &FakeGpuNvmlTelemetryV1::sm_clock);
    }
    if (!device || !clock) return NVML_ERROR_INVALID_ARGUMENT;
    *clock = profile_clock_mhz((Device*)device, type);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetClock(nvmlDevice_t device, unsigned int clockType, unsigned int clockId, unsigned int *clockMHz) {
    NVML_REQUIRE_INIT();
    if (nvml_bridge::enabled() && clockId == 0) return nvmlDeviceGetClockInfo(device, clockType, clockMHz);
    if (!device || !clockMHz) return NVML_ERROR_INVALID_ARGUMENT;
    // clockId: typically 0 for current clock
    *clockMHz = profile_clock_mhz((Device*)device, clockType);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMaxClockInfo(nvmlDevice_t device, unsigned int type, unsigned int *clock) {
    NVML_REQUIRE_INIT();
    if (!device || !clock) return NVML_ERROR_INVALID_ARGUMENT;
    *clock = profile_clock_mhz((Device*)device, type);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetFanSpeed(nvmlDevice_t device, unsigned int *speed) {
    NVML_REQUIRE_INIT();
    if (nvml_bridge::enabled()) return NVML_ERROR_NOT_SUPPORTED;
    if (!device || !speed) return NVML_ERROR_INVALID_ARGUMENT;

    // Return fake fan speed percentage
    *speed = 50;  // 50%

    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPerformanceState(nvmlDevice_t device, unsigned int *pState) {
    NVML_REQUIRE_INIT();
    if (nvml_bridge::enabled()) return read_telemetry(device, pState, &FakeGpuNvmlTelemetryV1::pstate);
    if (!device || !pState) return NVML_ERROR_INVALID_ARGUMENT;

    // Return fake performance state (P0 = maximum performance)
    *pState = 0;  // P0

    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetEncoderUtilization(nvmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs) {
    NVML_REQUIRE_INIT();
    return idle_engine_utilization(device, utilization, samplingPeriodUs);
}

nvmlReturn_t nvmlDeviceGetDecoderUtilization(nvmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs) {
    NVML_REQUIRE_INIT();
    return idle_engine_utilization(device, utilization, samplingPeriodUs);
}

nvmlReturn_t nvmlDeviceGetEncoderStats(nvmlDevice_t device, unsigned int *sessionCount, unsigned int *averageFps, unsigned int *averageLatency) {
    NVML_REQUIRE_INIT();
    if (!device || !sessionCount || !averageFps || !averageLatency) return NVML_ERROR_INVALID_ARGUMENT;
    *sessionCount = 0;
    *averageFps = 0;
    *averageLatency = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetFBCStats(nvmlDevice_t device, nvmlFBCStats_t *stats) {
    NVML_REQUIRE_INIT();
    if (!device || !stats) return NVML_ERROR_INVALID_ARGUMENT;
    stats->sessionsCount = 0;
    stats->averageFPS = 0;
    stats->averageLatency = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetDramEncryptionMode(nvmlDevice_t device, nvmlDramEncryptionInfo_t *current, nvmlDramEncryptionInfo_t *pending) {
    NVML_REQUIRE_INIT();
    if (!device || !current || !pending) return NVML_ERROR_INVALID_ARGUMENT;
    current->version = nvmlDramEncryptionInfo_v1;
    current->encryptionState = 0;  // Disabled
    pending->version = nvmlDramEncryptionInfo_v1;
    pending->encryptionState = 0;
    return NVML_SUCCESS;
}

// Texts as libnvidia-ml (driver 580) returns them for every defined code.
const char* nvmlErrorString(nvmlReturn_t result) {
    static const char* const kTexts[] = {
        "Success",                                                    // 0
        "Uninitialized",                                              // 1
        "Invalid Argument",                                           // 2
        "Not Supported",                                              // 3
        "Insufficient Permissions",                                   // 4
        "Already Initialized",                                        // 5
        "Not Found",                                                  // 6
        "Insufficient Size",                                          // 7
        "Insufficient External Power",                                // 8
        "Driver Not Loaded",                                          // 9
        "Timeout",                                                    // 10
        "Interrupt Request Issue",                                    // 11
        "NVML Shared Library Not Found",                              // 12
        "Function Not Found",                                         // 13
        "Corrupted infoROM",                                          // 14
        "GPU is lost",                                                // 15
        "GPU requires reset",                                         // 16
        "GPU access blocked by the operating system",                 // 17
        "Driver/library version mismatch",                            // 18
        "In use by another client",                                   // 19
        "Insufficient Memory",                                        // 20
        "No data",                                                    // 21
        "vGPU ECC not supported",                                     // 22
        "Insufficient Resources",                                     // 23
        "Invalid Frequency",                                          // 24
        "Argument version mismatch",                                  // 25
        "Requested functionality has been deprecated",                // 26
        "System is not in ready state",                               // 27
        "No supported GPUs were found",                               // 28
        "Resource not in correct state to perform requested operation",  // 29
        "Reset type is not supported for given device",               // 30
    };
    const int code = static_cast<int>(result);
    if (code >= 0 && code < static_cast<int>(sizeof(kTexts) / sizeof(kTexts[0]))) return kTexts[code];
    return "Unknown Error";
}

// Versioned function aliases (NVML v2 API)
nvmlReturn_t nvmlInit_v2() {
    return nvmlInit();
}

nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *deviceCount) {
    NVML_REQUIRE_INIT();
    return nvmlDeviceGetCount(deviceCount);
}

nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t *device) {
    NVML_REQUIRE_INIT();
    return nvmlDeviceGetHandleByIndex(index, device);
}

// System query functions
nvmlReturn_t nvmlSystemGetDriverVersion(char *version, unsigned int length) {
    NVML_REQUIRE_INIT();
    if (!version) return NVML_ERROR_INVALID_ARGUMENT;
    const char* value = nvml_bridge::enabled() ? "FakeGPU" : kDriverVersion;
    FGPU_LOG("[FakeNVML] nvmlSystemGetDriverVersion returning: %s\n", value);
    return copy_string(version, length, value);
}

nvmlReturn_t nvmlSystemGetNVMLVersion(char *version, unsigned int length) {
    if (!version) return NVML_ERROR_INVALID_ARGUMENT;
    const char* value = nvml_bridge::enabled() ? "FakeGPU state bridge v1" : fake_gpu::kModeledNvmlVersion;
    FGPU_LOG("[FakeNVML] nvmlSystemGetNVMLVersion returning: %s\n", value);
    return copy_string(version, length, value);
}

nvmlReturn_t nvmlSystemGetCudaDriverVersion(int *cudaDriverVersion) {
    NVML_REQUIRE_INIT();
    if (nvml_bridge::enabled()) return NVML_ERROR_NOT_SUPPORTED;
    if (!cudaDriverVersion) return NVML_ERROR_INVALID_ARGUMENT;
    // The same CUDA version cuDriverGetVersion reports.
    *cudaDriverVersion = fake_gpu::kModeledCudaDriverVersion;
    FGPU_LOG("[FakeNVML] nvmlSystemGetCudaDriverVersion returning: %d\n", *cudaDriverVersion);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlSystemGetCudaDriverVersion_v2(int *cudaDriverVersion) {
    NVML_REQUIRE_INIT();
    return nvmlSystemGetCudaDriverVersion(cudaDriverVersion);
}

// Additional functions required by nvitop

nvmlReturn_t nvmlDeviceGetDisplayActive(nvmlDevice_t device, unsigned int *isActive) {
    NVML_REQUIRE_INIT();
    if (!device || !isActive) return NVML_ERROR_INVALID_ARGUMENT;
    *isActive = 0;  // Display not active (server GPU)
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPersistenceMode(nvmlDevice_t device, unsigned int *mode) {
    NVML_REQUIRE_INIT();
    if (!device || !mode) return NVML_ERROR_INVALID_ARGUMENT;
    *mode = 0;  // Not in persistence mode
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetComputeMode(nvmlDevice_t device, unsigned int *mode) {
    NVML_REQUIRE_INIT();
    if (!device || !mode) return NVML_ERROR_INVALID_ARGUMENT;
    *mode = 0;  // NVML_COMPUTEMODE_DEFAULT
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetEccMode(nvmlDevice_t device, unsigned int *current, unsigned int *pending) {
    NVML_REQUIRE_INIT();
    if (!device) return NVML_ERROR_INVALID_ARGUMENT;
    if (current) *current = 0;
    if (pending) *pending = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetTotalEccErrors(nvmlDevice_t device, unsigned int errorType, unsigned int counterType, unsigned long long *eccCounts) {
    NVML_REQUIRE_INIT();
    if (!device || !eccCounts) return NVML_ERROR_INVALID_ARGUMENT;
    *eccCounts = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMigMode(nvmlDevice_t device, unsigned int *currentMode, unsigned int *pendingMode) {
    NVML_REQUIRE_INIT();
    if (nvml_bridge::enabled()) return NVML_ERROR_NOT_SUPPORTED;
    if (!device) return NVML_ERROR_INVALID_ARGUMENT;
    Device* dev = (Device*)device;
    const unsigned int enabled =
        dev->is_mig_device ||
        GlobalState::instance().get_mig_device_count(
            dev->index) > 0
        ? 1U
        : 0U;
    if (currentMode) *currentMode = enabled;
    if (pendingMode) *pendingMode = enabled;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetDriverModel(nvmlDevice_t device, unsigned int *current, unsigned int *pending) {
    NVML_REQUIRE_INIT();
    if (!device || (!current && !pending)) return NVML_ERROR_INVALID_ARGUMENT;
    // Windows only (WDDM = 0, WDM/TCC = 1); nvml.h returns NOT_SUPPORTED on
    // other platforms, and this library is built for Linux and macOS.
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlDeviceGetDriverModel_v2(nvmlDevice_t device, unsigned int *current, unsigned int *pending) {
    NVML_REQUIRE_INIT();
    return nvmlDeviceGetDriverModel(device, current, pending);
}

nvmlReturn_t nvmlDeviceGetMinorNumber(nvmlDevice_t device, unsigned int *minorNumber) {
    NVML_REQUIRE_INIT();
    if (!device || !minorNumber) return NVML_ERROR_INVALID_ARGUMENT;
    Device* dev = (Device*)device;
    *minorNumber = static_cast<unsigned int>(dev->index);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMultiGpuBoard(nvmlDevice_t device, unsigned int *multiGpu) {
    NVML_REQUIRE_INIT();
    if (!device || !multiGpu) return NVML_ERROR_INVALID_ARGUMENT;
    *multiGpu = 0;  // No
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetBoardPartNumber(nvmlDevice_t device, char *partNumber, unsigned int length) {
    NVML_REQUIRE_INIT();
    if (!device || !partNumber) return NVML_ERROR_INVALID_ARGUMENT;
    snprintf(partNumber, length, "N/A");
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetModuleId(nvmlDevice_t device, unsigned int *moduleId) {
    NVML_REQUIRE_INIT();
    if (!device || !moduleId) return NVML_ERROR_INVALID_ARGUMENT;
    *moduleId = 1;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetAccountingBufferSize(nvmlDevice_t device, unsigned int *bufferSize) {
    NVML_REQUIRE_INIT();
    if (!device || !bufferSize) return NVML_ERROR_INVALID_ARGUMENT;
    *bufferSize = 4000;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetLastBBXFlushTime(nvmlDevice_t device, unsigned long long *timestamp, unsigned long *durationUs) {
    NVML_REQUIRE_INIT();
    if (!device || !timestamp || !durationUs) return NVML_ERROR_INVALID_ARGUMENT;
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlDeviceGetGpuOperationMode(nvmlDevice_t device, unsigned int *current, unsigned int *pending) {
    NVML_REQUIRE_INIT();
    if (!device) return NVML_ERROR_INVALID_ARGUMENT;
    (void)current;
    (void)pending;
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlDeviceGetC2cModeInfoV(nvmlDevice_t device, nvmlC2cModeInfo_v1_t *info) {
    NVML_REQUIRE_INIT();
    if (!device || !info) return NVML_ERROR_INVALID_ARGUMENT;
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlDeviceGetHostVgpuMode(nvmlDevice_t device, unsigned int *mode) {
    NVML_REQUIRE_INIT();
    if (!device || !mode) return NVML_ERROR_INVALID_ARGUMENT;
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlDeviceGetVgpuHeterogeneousMode(nvmlDevice_t device, nvmlVgpuHeterogeneousMode_v1_t *mode) {
    NVML_REQUIRE_INIT();
    if (!device || !mode) return NVML_ERROR_INVALID_ARGUMENT;
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlDeviceGetGspFirmwareVersion(nvmlDevice_t device, char *version) {
    NVML_REQUIRE_INIT();
    if (!device || !version) return NVML_ERROR_INVALID_ARGUMENT;
    snprintf(version, NVML_GSP_FIRMWARE_VERSION_BUF_SIZE, "%s", kDriverVersion);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetCurrentClocksThrottleReasons(nvmlDevice_t device, unsigned long long *reasons) {
    NVML_REQUIRE_INIT();
    if (nvml_bridge::enabled()) return NVML_ERROR_NOT_SUPPORTED;
    if (!device || !reasons) return NVML_ERROR_INVALID_ARGUMENT;
    *reasons = 0x1ULL;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetSupportedClocksThrottleReasons(nvmlDevice_t device, unsigned long long *supportedReasons) {
    NVML_REQUIRE_INIT();
    if (!device || !supportedReasons) return NVML_ERROR_INVALID_ARGUMENT;
    *supportedReasons = 0x1ffULL;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetGpuMaxPcieLinkGeneration(nvmlDevice_t device, unsigned int *maxLinkGen) {
    NVML_REQUIRE_INIT();
    if (!device || !maxLinkGen) return NVML_ERROR_INVALID_ARGUMENT;
    ModeledPcieLink current, gpu_max;
    const nvmlReturn_t result = modeled_pcie(device, current, gpu_max);
    if (result == NVML_SUCCESS) *maxLinkGen = gpu_max.generation;
    return result;
}

nvmlReturn_t nvmlDeviceGetVirtualizationMode(nvmlDevice_t device, nvmlGpuVirtualizationMode_t *pVirtualMode) {
    NVML_REQUIRE_INIT();
    if (!device || !pVirtualMode) return NVML_ERROR_INVALID_ARGUMENT;
    *pVirtualMode = NVML_GPU_VIRTUALIZATION_MODE_NONE;
    return NVML_SUCCESS;
}

// Compute processes. Without the state bridge the only process NVML can see
// is this one: it is listed on a device while it holds simulated device
// memory there, with that memory as usedGpuMemory. A real driver also lists
// a process whose context holds no allocations; a context alone is not
// modeled here. MIG instance ids are reported as 0xFFFFFFFF (not MIG),
// as measured on a non-MIG GPU.
struct NativeProcessV1 {
    unsigned int pid;
    unsigned long long usedGpuMemory;
};

struct NativeProcessV2 {
    unsigned int pid;
    unsigned long long usedGpuMemory;
    unsigned int gpuInstanceId;
    unsigned int computeInstanceId;
};

static nvmlReturn_t native_compute_processes(nvmlDevice_t device, unsigned int* infoCount, void* infos, bool legacy) {
    const Device* dev = reinterpret_cast<const Device*>(device);
    const unsigned long long used = static_cast<unsigned long long>(dev->used_memory);
    const unsigned int required = used > 0 ? 1u : 0u;
    const unsigned int capacity = *infoCount;
    *infoCount = required;
    if (required == 0) return NVML_SUCCESS;
    if (!infos || capacity < required) return NVML_ERROR_INSUFFICIENT_SIZE;
    const unsigned int pid = static_cast<unsigned int>(getpid());
    if (legacy) {
        static_cast<NativeProcessV1*>(infos)[0] = {pid, used};
    } else {
        static_cast<NativeProcessV2*>(infos)[0] = {pid, used, 0xFFFFFFFFu, 0xFFFFFFFFu};
    }
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetComputeRunningProcesses(nvmlDevice_t device, unsigned int *infoCount, void *infos) {
    NVML_REQUIRE_INIT();
    if (!device || !infoCount) return NVML_ERROR_INVALID_ARGUMENT;
    if (nvml_bridge::enabled()) return nvml_bridge::processes(device, infoCount, infos, true);
    return native_compute_processes(device, infoCount, infos, true);
}

nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v2(nvmlDevice_t device, unsigned int *infoCount, void *infos) {
    NVML_REQUIRE_INIT();
    if (!device || !infoCount) return NVML_ERROR_INVALID_ARGUMENT;
    if (nvml_bridge::enabled()) return nvml_bridge::processes(device, infoCount, infos, false);
    return native_compute_processes(device, infoCount, infos, false);
}

nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v3(nvmlDevice_t device, unsigned int *infoCount, void *infos) {
    NVML_REQUIRE_INIT();
    return nvmlDeviceGetComputeRunningProcesses_v2(device, infoCount, infos);
}

nvmlReturn_t nvmlDeviceGetRunningProcessDetailList(nvmlDevice_t device, void* details) {
    NVML_REQUIRE_INIT();
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlDeviceGetProcessUtilization(nvmlDevice_t device, void* samples,
                                           unsigned int* count, unsigned long long lastSeen) {
    NVML_REQUIRE_INIT();
    if (nvml_bridge::enabled()) return nvml_bridge::process_utilization(device, samples, count, lastSeen);
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses(nvmlDevice_t device, unsigned int *infoCount, void *infos) {
    NVML_REQUIRE_INIT();
    if (!device || !infoCount) return NVML_ERROR_INVALID_ARGUMENT;
    *infoCount = 0;  // No running graphics processes
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses_v2(nvmlDevice_t device, unsigned int *infoCount, void *infos) {
    NVML_REQUIRE_INIT();
    return nvmlDeviceGetGraphicsRunningProcesses(device, infoCount, infos);
}

nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses_v3(nvmlDevice_t device, unsigned int *infoCount, void *infos) {
    NVML_REQUIRE_INIT();
    return nvmlDeviceGetGraphicsRunningProcesses(device, infoCount, infos);
}

nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses(nvmlDevice_t device, unsigned int *infoCount, void *infos) {
    NVML_REQUIRE_INIT();
    if (!device || !infoCount) return NVML_ERROR_INVALID_ARGUMENT;
    *infoCount = 0;  // No MPS processes
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses_v2(nvmlDevice_t device, unsigned int *infoCount, void *infos) {
    NVML_REQUIRE_INIT();
    return nvmlDeviceGetMPSComputeRunningProcesses(device, infoCount, infos);
}

nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses_v3(nvmlDevice_t device, unsigned int *infoCount, void *infos) {
    NVML_REQUIRE_INIT();
    return nvmlDeviceGetMPSComputeRunningProcesses(device, infoCount, infos);
}

// Clock-related functions
nvmlReturn_t nvmlDeviceGetMaxCustomerBoostClock(nvmlDevice_t device, unsigned int clockType, unsigned int *clockMHz) {
    NVML_REQUIRE_INIT();
    if (!device || !clockMHz) return NVML_ERROR_INVALID_ARGUMENT;
    *clockMHz = profile_clock_mhz((Device*)device, clockType);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMinMaxClockOfPState(nvmlDevice_t device, unsigned int clockType, unsigned int pState, unsigned int *minClockMHz, unsigned int *maxClockMHz) {
    NVML_REQUIRE_INIT();
    if (!device) return NVML_ERROR_INVALID_ARGUMENT;
    if (minClockMHz) *minClockMHz = 210;
    if (maxClockMHz) *maxClockMHz = profile_clock_mhz((Device*)device, clockType);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetApplicationsClock(nvmlDevice_t device, unsigned int clockType, unsigned int *clockMHz) {
    NVML_REQUIRE_INIT();
    return nvmlDeviceGetMaxCustomerBoostClock(device, clockType, clockMHz);
}

nvmlReturn_t nvmlDeviceGetDefaultApplicationsClock(nvmlDevice_t device, unsigned int clockType, unsigned int *clockMHz) {
    NVML_REQUIRE_INIT();
    return nvmlDeviceGetMaxCustomerBoostClock(device, clockType, clockMHz);
}

// Power management
nvmlReturn_t nvmlDeviceGetEnforcedPowerLimit(nvmlDevice_t device, unsigned int *limit) {
    NVML_REQUIRE_INIT();
    if (!device || !limit) return NVML_ERROR_INVALID_ARGUMENT;
    trace_power("nvmlDeviceGetEnforcedPowerLimit", device);
    const unsigned int value = default_power_limit_mw((Device*)device);
    if (!value) return NVML_ERROR_NOT_SUPPORTED;
    *limit = value;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPowerManagementLimitConstraints(nvmlDevice_t device, unsigned int *minLimit, unsigned int *maxLimit) {
    NVML_REQUIRE_INIT();
    if (!device) return NVML_ERROR_INVALID_ARGUMENT;
    trace_power("nvmlDeviceGetPowerManagementLimitConstraints", device);
    unsigned int min_limit = 0, max_limit = 0;
    const nvmlReturn_t status = power_limit_constraints_mw((Device*)device, min_limit, max_limit);
    if (status != NVML_SUCCESS) return status;
    if (minLimit) *minLimit = min_limit;
    if (maxLimit) *maxLimit = max_limit;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPowerManagementDefaultLimit(nvmlDevice_t device, unsigned int *defaultLimit) {
    NVML_REQUIRE_INIT();
    if (!device || !defaultLimit) return NVML_ERROR_INVALID_ARGUMENT;
    trace_power("nvmlDeviceGetPowerManagementDefaultLimit", device);
    const unsigned int value = default_power_limit_mw((Device*)device);
    if (!value) return NVML_ERROR_NOT_SUPPORTED;
    *defaultLimit = value;
    return NVML_SUCCESS;
}

// Workload power profiles: not supported in FakeGPU.
nvmlReturn_t nvmlDeviceWorkloadPowerProfileGetProfilesInfo(nvmlDevice_t device, void *profilesInfo) {
    NVML_REQUIRE_INIT();
    (void)device;
    (void)profilesInfo;
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlDeviceWorkloadPowerProfileGetCurrentProfiles(nvmlDevice_t device, void *currentProfiles) {
    NVML_REQUIRE_INIT();
    (void)device;
    (void)currentProfiles;
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlDeviceWorkloadPowerProfileSetRequestedProfiles(nvmlDevice_t device, void *requestedProfiles) {
    NVML_REQUIRE_INIT();
    (void)device;
    (void)requestedProfiles;
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlDeviceWorkloadPowerProfileClearRequestedProfiles(nvmlDevice_t device, void *requestedProfiles) {
    NVML_REQUIRE_INIT();
    (void)device;
    (void)requestedProfiles;
    return NVML_ERROR_NOT_SUPPORTED;
}

// Architecture info. GpuArch values follow compute capability, not NVML ids.
nvmlReturn_t nvmlDeviceGetArchitecture(nvmlDevice_t device, unsigned int *arch) {
    NVML_REQUIRE_INIT();
    if (!device || !arch) return NVML_ERROR_INVALID_ARGUMENT;
    Device* dev = (Device*)device;
    switch (dev->profile.architecture) {
        case GpuArch::Maxwell: *arch = NVML_DEVICE_ARCH_MAXWELL; break;
        case GpuArch::Pascal: *arch = NVML_DEVICE_ARCH_PASCAL; break;
        case GpuArch::Volta: *arch = NVML_DEVICE_ARCH_VOLTA; break;
        case GpuArch::Turing: *arch = NVML_DEVICE_ARCH_TURING; break;
        case GpuArch::Ampere: *arch = NVML_DEVICE_ARCH_AMPERE; break;
        case GpuArch::Ada: *arch = NVML_DEVICE_ARCH_ADA; break;
        case GpuArch::Hopper: *arch = NVML_DEVICE_ARCH_HOPPER; break;
        case GpuArch::Blackwell: *arch = NVML_DEVICE_ARCH_BLACKWELL; break;
        default: *arch = NVML_DEVICE_ARCH_UNKNOWN; break;
    }
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int *major, int *minor) {
    NVML_REQUIRE_INIT();
    if (!device) return NVML_ERROR_INVALID_ARGUMENT;
    Device* dev = (Device*)device;
    if (major) *major = dev->profile.compute_major;
    if (minor) *minor = dev->profile.compute_minor;
    return NVML_SUCCESS;
}

// BAR1 Memory info (for some nvidia-smi compatibility)
nvmlReturn_t nvmlDeviceGetBAR1MemoryInfo(nvmlDevice_t device, void *bar1Memory) {
    NVML_REQUIRE_INIT();
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
    NVML_REQUIRE_INIT();
    if (!device || !boardId) return NVML_ERROR_INVALID_ARGUMENT;
    *boardId = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetSerial(nvmlDevice_t device, char *serial, unsigned int length) {
    NVML_REQUIRE_INIT();
    if (!device || !serial) return NVML_ERROR_INVALID_ARGUMENT;
    return copy_string(serial, length, "N/A");
}

nvmlReturn_t nvmlDeviceGetBrand(nvmlDevice_t device, unsigned int *type) {
    NVML_REQUIRE_INIT();
    if (!device || !type) return NVML_ERROR_INVALID_ARGUMENT;
    *type = 0;  // NVML_BRAND_UNKNOWN
    return NVML_SUCCESS;
}

// VBIOS
nvmlReturn_t nvmlDeviceGetVbiosVersion(nvmlDevice_t device, char *version, unsigned int length) {
    NVML_REQUIRE_INIT();
    if (!device || !version) return NVML_ERROR_INVALID_ARGUMENT;
    return copy_string(version, length, "92.00.00.00.00");
}

// Info ROM
nvmlReturn_t nvmlDeviceGetInforomVersion(nvmlDevice_t device, unsigned int object, char *version, unsigned int length) {
    NVML_REQUIRE_INIT();
    if (!device || !version) return NVML_ERROR_INVALID_ARGUMENT;
    return copy_string(version, length, "N/A");
}

nvmlReturn_t nvmlDeviceGetInforomImageVersion(nvmlDevice_t device, char *version, unsigned int length) {
    NVML_REQUIRE_INIT();
    if (!device || !version) return NVML_ERROR_INVALID_ARGUMENT;
    return copy_string(version, length, "N/A");
}

nvmlReturn_t nvmlDeviceValidateInforom(nvmlDevice_t device) {
    NVML_REQUIRE_INIT();
    if (!device) return NVML_ERROR_INVALID_ARGUMENT;
    return NVML_SUCCESS;
}

// NVLink
nvmlReturn_t nvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link, unsigned int *isActive) {
    NVML_REQUIRE_INIT();
    if (!device || !isActive) return NVML_ERROR_INVALID_ARGUMENT;
    ModeledNvLinkPeer peer;
    *isActive = modeled_nvlink_peer(
        reinterpret_cast<Device*>(device),
        link,
        peer) ? 1U : 0U;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetNvLinkCapability(nvmlDevice_t device, unsigned int link, unsigned int capability, unsigned int *capResult) {
    NVML_REQUIRE_INIT();
    if (!device || !capResult) return NVML_ERROR_INVALID_ARGUMENT;
    (void)capability;
    ModeledNvLinkPeer peer;
    *capResult = modeled_nvlink_peer(
        reinterpret_cast<Device*>(device),
        link,
        peer) ? 1U : 0U;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetNvLinkRemoteDeviceType(
    nvmlDevice_t device,
    unsigned int link,
    nvmlIntNvLinkDeviceType_t *remoteDeviceType) {
    NVML_REQUIRE_INIT();
    if (!device || !remoteDeviceType) return NVML_ERROR_INVALID_ARGUMENT;
    ModeledNvLinkPeer peer;
    if (!modeled_nvlink_peer(reinterpret_cast<Device*>(device), link, peer)) {
        *remoteDeviceType = NVML_NVLINK_DEVICE_TYPE_UNKNOWN;
    } else {
        *remoteDeviceType = peer.peer_index < 0
            ? NVML_NVLINK_DEVICE_TYPE_SWITCH
            : NVML_NVLINK_DEVICE_TYPE_GPU;
    }
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetNvLinkRemotePciInfo_v2(
    nvmlDevice_t device,
    unsigned int link,
    nvmlPciInfo_t *pci) {
    NVML_REQUIRE_INIT();
    if (!device || !pci) return NVML_ERROR_INVALID_ARGUMENT;
    ModeledNvLinkPeer peer;
    if (!modeled_nvlink_peer(
            reinterpret_cast<Device*>(device),
            link,
            peer)) {
        memset(pci, 0, sizeof(*pci));
        return NVML_ERROR_NOT_SUPPORTED;
    }
    if (peer.peer_index < 0) {
        memset(pci, 0, sizeof(*pci));
        NvmlPciAddress address;
        parse_nvml_pci_bus_id(kNvSwitchPciBusId, address);
        format_nvml_pci_bus_id(address, pci->busId, sizeof(pci->busId), pci->busIdLegacy, sizeof(pci->busIdLegacy));
        pci->domain = address.domain;
        pci->bus = address.bus;
        pci->device = address.device;
        return NVML_SUCCESS;
    }
    Device& remote =
        GlobalState::instance().get_physical_device(peer.peer_index);
    fill_pci_info(remote, *pci);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetGpuFabricInfoV(nvmlDevice_t device, void *gpuFabricInfo) {
    NVML_REQUIRE_INIT();
    if (!device || !gpuFabricInfo) return NVML_ERROR_INVALID_ARGUMENT;

    // Versioned structs: v2/v3 begin with a version field.
    unsigned int version = *static_cast<unsigned int*>(gpuFabricInfo);
    if (version != nvmlGpuFabricInfo_v2 && version != nvmlGpuFabricInfo_v3) return NVML_ERROR_INVALID_ARGUMENT;
    // v3 extends the v2 layout.
    nvmlGpuFabricInfo_v2_t *info = static_cast<nvmlGpuFabricInfo_v2_t*>(gpuFabricInfo);
    memset(info->clusterUuid, 0, sizeof(info->clusterUuid));
    info->status = NVML_ERROR_NOT_SUPPORTED;
    info->cliqueId = 0;
    info->state = 0;      // NOT_SUPPORTED
    info->healthMask = 0; // NOT_SUPPORTED
    if (version == nvmlGpuFabricInfo_v3) {
        static_cast<nvmlGpuFabricInfo_v3_t*>(gpuFabricInfo)->healthSummary = 0;
    }
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetBridgeChipInfo(nvmlDevice_t device, nvmlBridgeChipHierarchy_t *bridgeHierarchy) {
    NVML_REQUIRE_INIT();
    if (!device || !bridgeHierarchy) return NVML_ERROR_INVALID_ARGUMENT;
    bridgeHierarchy->bridgeCount = 0;
    return NVML_SUCCESS;
}

// Retired pages
nvmlReturn_t nvmlDeviceGetRetiredPages(nvmlDevice_t device, unsigned int cause, unsigned int *pageCount, unsigned long long *addresses) {
    NVML_REQUIRE_INIT();
    if (!device || !pageCount) return NVML_ERROR_INVALID_ARGUMENT;
    *pageCount = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetRetiredPages_v2(nvmlDevice_t device, unsigned int cause, unsigned int *pageCount, unsigned long long *addresses, unsigned long long *timestamps) {
    NVML_REQUIRE_INIT();
    return nvmlDeviceGetRetiredPages(device, cause, pageCount, addresses);
}

nvmlReturn_t nvmlDeviceGetRetiredPagesPendingStatus(nvmlDevice_t device, unsigned int *isPending) {
    NVML_REQUIRE_INIT();
    if (!device || !isPending) return NVML_ERROR_INVALID_ARGUMENT;
    *isPending = 0;
    return NVML_SUCCESS;
}

// PCIe throughput
// KB/s over the last second of the modeled link schedule (fakegpu.link_traffic).
nvmlReturn_t nvmlDeviceGetPcieThroughput(nvmlDevice_t device, unsigned int counter, unsigned int *value) {
    NVML_REQUIRE_INIT();
    if (!device || !value || counter > 1) return NVML_ERROR_INVALID_ARGUMENT;
    // counter: 0 = TX (sent by the GPU), 1 = RX
    uint64_t bytes_per_second = 0;
    if (nvml_bridge::enabled()) {
        FakeGpuNvmlLinksV1 published{};
        if (nvml_bridge::links(device, &published) != NVML_SUCCESS || !published.pcie_generation) {
            return NVML_ERROR_NOT_SUPPORTED;
        }
        bytes_per_second = counter == 0 ? published.pcie_tx_bytes_per_second : published.pcie_rx_bytes_per_second;
    } else {
        ModeledPcieLink current, gpu_max;
        const nvmlReturn_t result = modeled_pcie(device, current, gpu_max);
        if (result != NVML_SUCCESS) return result;
        DeviceTransferStats transfer;
        if (modeled_transfer(reinterpret_cast<const Device*>(device), transfer)) {
            const LinkCounter& pcie = transfer.links[static_cast<int>(TransferLink::Pcie)];
            bytes_per_second = counter == 0 ? pcie.tx_bytes_per_second : pcie.rx_bytes_per_second;
        }
    }
    *value = static_cast<unsigned int>(std::min<uint64_t>(bytes_per_second / 1024, 0xFFFFFFFFull));
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPcieReplayCounter(nvmlDevice_t device, unsigned int *value) {
    NVML_REQUIRE_INIT();
    if (!device || !value) return NVML_ERROR_INVALID_ARGUMENT;
    *value = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPcieLinkGeneration(nvmlDevice_t device, unsigned int *currLinkGen) {
    NVML_REQUIRE_INIT();
    if (!device || !currLinkGen) return NVML_ERROR_INVALID_ARGUMENT;
    ModeledPcieLink current, gpu_max;
    const nvmlReturn_t result = modeled_pcie(device, current, gpu_max);
    if (result == NVML_SUCCESS) *currLinkGen = current.generation;
    return result;
}

nvmlReturn_t nvmlDeviceGetPcieLinkWidth(nvmlDevice_t device, unsigned int *currLinkWidth) {
    NVML_REQUIRE_INIT();
    if (!device || !currLinkWidth) return NVML_ERROR_INVALID_ARGUMENT;
    ModeledPcieLink current, gpu_max;
    const nvmlReturn_t result = modeled_pcie(device, current, gpu_max);
    if (result == NVML_SUCCESS) *currLinkWidth = current.width;
    return result;
}

nvmlReturn_t nvmlDeviceGetCurrPcieLinkGeneration(nvmlDevice_t device, unsigned int *currLinkGen) {
    NVML_REQUIRE_INIT();
    return nvmlDeviceGetPcieLinkGeneration(device, currLinkGen);
}

nvmlReturn_t nvmlDeviceGetCurrPcieLinkWidth(nvmlDevice_t device, unsigned int *currLinkWidth) {
    NVML_REQUIRE_INIT();
    return nvmlDeviceGetPcieLinkWidth(device, currLinkWidth);
}

// The maximum the device and system allow: the trained link, since the
// model has no power-saving downtraining.
nvmlReturn_t nvmlDeviceGetMaxPcieLinkGeneration(nvmlDevice_t device, unsigned int *maxLinkGen) {
    NVML_REQUIRE_INIT();
    return nvmlDeviceGetPcieLinkGeneration(device, maxLinkGen);
}

nvmlReturn_t nvmlDeviceGetMaxPcieLinkWidth(nvmlDevice_t device, unsigned int *maxLinkWidth) {
    NVML_REQUIRE_INIT();
    return nvmlDeviceGetPcieLinkWidth(device, maxLinkWidth);
}

static nvmlReturn_t modeled_pair(
    nvmlDevice_t device1,
    nvmlDevice_t device2,
    ModeledDeviceTopology& topology,
    int& first,
    int& second) {
    if (!device1 || !device2) return NVML_ERROR_INVALID_ARGUMENT;
    ModeledSlot slot;
    const Device* left = reinterpret_cast<const Device*>(device1);
    const Device* right = reinterpret_cast<const Device*>(device2);
    if (left->is_mig_device || right->is_mig_device) return NVML_ERROR_NOT_SUPPORTED;
    if (!modeled_slot(left, slot, &topology)) return NVML_ERROR_NOT_SUPPORTED;
    first = left->index;
    second = right->index;
    return topology.slot(second) ? NVML_SUCCESS : NVML_ERROR_INVALID_ARGUMENT;
}

nvmlReturn_t nvmlDeviceGetTopologyCommonAncestor(
    nvmlDevice_t device1,
    nvmlDevice_t device2,
    nvmlGpuTopologyLevel_t *pathInfo) {
    NVML_REQUIRE_INIT();
    if (!pathInfo) return NVML_ERROR_INVALID_ARGUMENT;
    ModeledDeviceTopology topology;
    int first = -1, second = -1;
    const nvmlReturn_t result = modeled_pair(device1, device2, topology, first, second);
    if (result != NVML_SUCCESS) return result;
    const std::string path = topology.pcie_relation(first, second);
    *pathInfo = path == "X" ? NVML_TOPOLOGY_INTERNAL
        : path == "PIX" ? NVML_TOPOLOGY_SINGLE
        : path == "PXB" ? NVML_TOPOLOGY_MULTIPLE
        : path == "PHB" ? NVML_TOPOLOGY_HOSTBRIDGE
        : path == "NODE" ? NVML_TOPOLOGY_NODE
        : NVML_TOPOLOGY_SYSTEM;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetP2PStatus(
    nvmlDevice_t device1,
    nvmlDevice_t device2,
    nvmlGpuP2PCapsIndex_t p2pIndex,
    nvmlGpuP2PStatus_t *p2pStatus) {
    NVML_REQUIRE_INIT();
    if (!p2pStatus) return NVML_ERROR_INVALID_ARGUMENT;
    ModeledDeviceTopology topology;
    int first = -1, second = -1;
    const nvmlReturn_t result = modeled_pair(device1, device2, topology, first, second);
    if (result != NVML_SUCCESS) return result;
    const bool nvlink = topology.nvlink_links(first, second) > 0;
    const ModeledSlot& left = *topology.slot(first);
    const ModeledSlot& right = *topology.slot(second);
    nvmlGpuP2PStatus_t pcie_status = NVML_P2P_STATUS_OK;
    if (first == second || !left.pcie.present() || !right.pcie.present()) {
        pcie_status = NVML_P2P_STATUS_NOT_SUPPORTED;
    } else if (!left.p2p || !right.p2p) {
        pcie_status = NVML_P2P_STATUS_GPU_NOT_SUPPORTED;
    } else if (topology.pcie_relation(first, second) == "SYS") {
        pcie_status = NVML_P2P_STATUS_IOH_TOPOLOGY_NOT_SUPPORTED;
    }
    switch (p2pIndex) {
        case NVML_P2P_CAPS_INDEX_READ:
        case NVML_P2P_CAPS_INDEX_WRITE:
            *p2pStatus = nvlink ? NVML_P2P_STATUS_OK : pcie_status;
            break;
        case NVML_P2P_CAPS_INDEX_NVLINK:
        case NVML_P2P_CAPS_INDEX_ATOMICS:
            *p2pStatus = nvlink ? NVML_P2P_STATUS_OK : NVML_P2P_STATUS_NOT_SUPPORTED;
            break;
        case NVML_P2P_CAPS_INDEX_PCI:
            *p2pStatus = pcie_status;
            break;
        default:
            return NVML_ERROR_INVALID_ARGUMENT;
    }
    return NVML_SUCCESS;
}

// Display mode
nvmlReturn_t nvmlDeviceGetDisplayMode(nvmlDevice_t device, unsigned int *display) {
    NVML_REQUIRE_INIT();
    if (!device || !display) return NVML_ERROR_INVALID_ARGUMENT;
    *display = 0;  // Display disabled (server GPU)
    return NVML_SUCCESS;
}

// Handle by UUID and PCI Bus ID
nvmlReturn_t nvmlDeviceGetHandleByUUID(const char *uuid, nvmlDevice_t *device) {
    NVML_REQUIRE_INIT();
    if (!uuid || !device) return NVML_ERROR_INVALID_ARGUMENT;
    if (nvml_bridge::enabled()) return nvml_bridge::lookup(uuid, true, device);
    GlobalState& state = GlobalState::instance();
    const int count = state.get_physical_device_count();
    for (int index = 0; index < count; ++index) {
        Device& parent = state.get_physical_device(index);
        if (parent.uuid == uuid) {
            *device = (nvmlDevice_t)&parent;
            return NVML_SUCCESS;
        }
        const unsigned int mig_count =
            state.get_mig_device_count(index);
        for (unsigned int mig_index = 0;
             mig_index < mig_count;
             ++mig_index) {
            Device* mig_device =
                state.get_mig_device(index, mig_index);
            if (mig_device && mig_device->uuid == uuid) {
                *device = (nvmlDevice_t)mig_device;
                return NVML_SUCCESS;
            }
        }
    }
    *device = nullptr;
    return NVML_ERROR_NOT_FOUND;
}

nvmlReturn_t nvmlDeviceGetHandleByPciBusId(const char *pciBusId, nvmlDevice_t *device) {
    NVML_REQUIRE_INIT();
    if (!pciBusId || !device) return NVML_ERROR_INVALID_ARGUMENT;
    if (nvml_bridge::enabled()) return nvml_bridge::lookup(pciBusId, false, device);
    GlobalState& state = GlobalState::instance();
    const int count = state.get_physical_device_count();
    for (int index = 0; index < count; ++index) {
        Device& dev = state.get_physical_device(index);
        if (nvml_pci_bus_id_matches(dev.pci_bus_id.c_str(), pciBusId)) {
            *device = (nvmlDevice_t)&dev;
            return NVML_SUCCESS;
        }
    }
    *device = nullptr;
    return NVML_ERROR_NOT_FOUND;
}

nvmlReturn_t nvmlDeviceGetHandleByPciBusId_v2(const char *pciBusId, nvmlDevice_t *device) {
    NVML_REQUIRE_INIT();
    return nvmlDeviceGetHandleByPciBusId(pciBusId, device);
}

nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int *index) {
    NVML_REQUIRE_INIT();
    if (!device || !index) return NVML_ERROR_INVALID_ARGUMENT;
    Device* dev = (Device*)device;
    *index = dev->index;
    return NVML_SUCCESS;
}

// Unit queries (for compute clusters)
nvmlReturn_t nvmlUnitGetCount(unsigned int *unitCount) {
    NVML_REQUIRE_INIT();
    if (!unitCount) return NVML_ERROR_INVALID_ARGUMENT;
    *unitCount = 0;
    return NVML_SUCCESS;
}

// Event set (stub implementations)
nvmlReturn_t nvmlEventSetCreate(void **set) {
    NVML_REQUIRE_INIT();
    if (!set) return NVML_ERROR_INVALID_ARGUMENT;
    *set = (void*)0x1;  // Fake handle
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlEventSetFree(void *set) {
    NVML_REQUIRE_INIT();
    return NVML_SUCCESS;
}

// Accounting (stub)
nvmlReturn_t nvmlDeviceGetAccountingMode(nvmlDevice_t device, unsigned int *mode) {
    NVML_REQUIRE_INIT();
    if (!device || !mode) return NVML_ERROR_INVALID_ARGUMENT;
    *mode = 0;  // Disabled
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetAccountingPids(nvmlDevice_t device, unsigned int *count, unsigned int *pids) {
    NVML_REQUIRE_INIT();
    if (!device || !count) return NVML_ERROR_INVALID_ARGUMENT;
    *count = 0;
    return NVML_SUCCESS;
}

// MIG-related functions
nvmlReturn_t nvmlDeviceGetMaxMigDeviceCount(nvmlDevice_t device, unsigned int *count) {
    NVML_REQUIRE_INIT();
    if (nvml_bridge::enabled()) return NVML_ERROR_NOT_SUPPORTED;
    if (!device || !count) return NVML_ERROR_INVALID_ARGUMENT;
    Device* dev = (Device*)device;
    if (dev->is_mig_device) {
        return NVML_ERROR_NOT_SUPPORTED;
    }
    *count = GlobalState::instance().get_mig_device_count(
        dev->index);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMigDeviceHandleByIndex(nvmlDevice_t device, unsigned int index, nvmlDevice_t *migDevice) {
    NVML_REQUIRE_INIT();
    if (nvml_bridge::enabled()) return NVML_ERROR_NOT_SUPPORTED;
    if (!device || !migDevice) return NVML_ERROR_INVALID_ARGUMENT;
    Device* parent = (Device*)device;
    if (parent->is_mig_device) {
        return NVML_ERROR_NOT_SUPPORTED;
    }
    const unsigned int count =
        GlobalState::instance().get_mig_device_count(
            parent->index);
    if (count == 0) {
        *migDevice = nullptr;
        return NVML_ERROR_NOT_SUPPORTED;
    }
    Device* result = GlobalState::instance().get_mig_device(
        parent->index,
        index);
    if (!result) {
        *migDevice = nullptr;
        return NVML_ERROR_NOT_FOUND;
    }
    *migDevice = (nvmlDevice_t)result;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetGpuInstanceById(nvmlDevice_t device, unsigned int id, void *gpuInstance) {
    NVML_REQUIRE_INIT();
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlDeviceGetGpuInstances(nvmlDevice_t device, unsigned int profileId, void *gpuInstances, unsigned int *count) {
    NVML_REQUIRE_INIT();
    if (count) *count = 0;
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlDeviceIsMigDeviceHandle(nvmlDevice_t device, unsigned int *isMigDevice) {
    NVML_REQUIRE_INIT();
    if (!device || !isMigDevice) return NVML_ERROR_INVALID_ARGUMENT;
    Device* dev = (Device*)device;
    *isMigDevice = dev->is_mig_device ? 1U : 0U;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetDeviceHandleFromMigDeviceHandle(nvmlDevice_t migDevice, nvmlDevice_t *device) {
    NVML_REQUIRE_INIT();
    if (nvml_bridge::enabled()) return NVML_ERROR_NOT_SUPPORTED;
    if (!migDevice || !device) return NVML_ERROR_INVALID_ARGUMENT;
    Device* mig = (Device*)migDevice;
    Device* parent =
        GlobalState::instance().get_parent_device_for_mig(mig);
    if (!parent) {
        *device = nullptr;
        return NVML_ERROR_NOT_SUPPORTED;
    }
    *device = (nvmlDevice_t)parent;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetGpuInstanceId(
    nvmlDevice_t device,
    unsigned int *id) {
    NVML_REQUIRE_INIT();
    if (!device || !id) return NVML_ERROR_INVALID_ARGUMENT;
    Device* dev = (Device*)device;
    if (!dev->is_mig_device) {
        return NVML_ERROR_NOT_SUPPORTED;
    }
    *id = dev->gpu_instance_id;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetComputeInstanceId(
    nvmlDevice_t device,
    unsigned int *id) {
    NVML_REQUIRE_INIT();
    if (!device || !id) return NVML_ERROR_INVALID_ARGUMENT;
    Device* dev = (Device*)device;
    if (!dev->is_mig_device) {
        return NVML_ERROR_NOT_SUPPORTED;
    }
    *id = dev->compute_instance_id;
    return NVML_SUCCESS;
}

// Tuple index - used by nvitop for MIG devices
// nvmlDeviceGetTupleIndex doesn't exist in standard NVML, nvitop may construct it
// For non-MIG devices, return NOT_SUPPORTED

// Additional power functions
nvmlReturn_t nvmlDeviceGetPowerState(nvmlDevice_t device, unsigned int *pState) {
    NVML_REQUIRE_INIT();
    return nvmlDeviceGetPerformanceState(device, pState);
}

nvmlReturn_t nvmlDeviceGetPowerSource(nvmlDevice_t device, unsigned int *powerSource) {
    NVML_REQUIRE_INIT();
    if (!device || !powerSource) return NVML_ERROR_INVALID_ARGUMENT;
    *powerSource = 0;  // AC power
    return NVML_SUCCESS;
}

// Supported/enabled clocks
// One supported clock, returned with NVML's size-query protocol: a count below
// one (pynvml's first call passes 0 and NULL) yields INSUFFICIENT_SIZE and 1.
static nvmlReturn_t single_supported_clock(unsigned int clock, unsigned int *count, unsigned int *clocksMHz) {
    const unsigned int capacity = *count;
    *count = 1;
    if (capacity < 1) return NVML_ERROR_INSUFFICIENT_SIZE;
    if (!clocksMHz) return NVML_ERROR_INVALID_ARGUMENT;
    clocksMHz[0] = clock;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetSupportedGraphicsClocks(nvmlDevice_t device, unsigned int memoryClockMHz, unsigned int *count, unsigned int *clocksMHz) {
    NVML_REQUIRE_INIT();
    if (!device || !count) return NVML_ERROR_INVALID_ARGUMENT;
    Device* dev = (Device*)device;
    if (memoryClockMHz != profile_clock_mhz(dev, 2)) return NVML_ERROR_NOT_FOUND;
    return single_supported_clock(profile_clock_mhz(dev, 0), count, clocksMHz);
}

nvmlReturn_t nvmlDeviceGetSupportedMemoryClocks(nvmlDevice_t device, unsigned int *count, unsigned int *clocksMHz) {
    NVML_REQUIRE_INIT();
    if (!device || !count) return NVML_ERROR_INVALID_ARGUMENT;
    return single_supported_clock(profile_clock_mhz((Device*)device, 2), count, clocksMHz);
}

// Lock clocks (no-op)
nvmlReturn_t nvmlDeviceSetGpuLockedClocks(nvmlDevice_t device, unsigned int minGpuClockMHz, unsigned int maxGpuClockMHz) {
    NVML_REQUIRE_INIT();
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceResetGpuLockedClocks(nvmlDevice_t device) {
    NVML_REQUIRE_INIT();
    return NVML_SUCCESS;
}

// Auto-boost
nvmlReturn_t nvmlDeviceGetAutoBoostedClocksEnabled(nvmlDevice_t device, unsigned int *isEnabled, unsigned int *defaultIsEnabled) {
    NVML_REQUIRE_INIT();
    if (!device) return NVML_ERROR_INVALID_ARGUMENT;
    if (isEnabled) *isEnabled = 1;
    if (defaultIsEnabled) *defaultIsEnabled = 1;
    return NVML_SUCCESS;
}

// Remapped rows
nvmlReturn_t nvmlDeviceGetRemappedRows(nvmlDevice_t device, unsigned int *corrRows, unsigned int *uncRows, unsigned int *isPending, unsigned int *failureOccurred) {
    NVML_REQUIRE_INIT();
    if (!device) return NVML_ERROR_INVALID_ARGUMENT;
    if (corrRows) *corrRows = 0;
    if (uncRows) *uncRows = 0;
    if (isPending) *isPending = 0;
    if (failureOccurred) *failureOccurred = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetRowRemapperHistogram(nvmlDevice_t device, nvmlRowRemapperHistogramValues_t *values) {
    NVML_REQUIRE_INIT();
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
    NVML_REQUIRE_INIT();
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
    constexpr uint32_t NVML_FI_DEV_NVLINK_LINK_COUNT = 91;
    constexpr uint32_t NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX = 138;
    constexpr uint32_t NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX = 139;

    Device* dev = (Device*)device;
    const unsigned int default_limit_mw = default_power_limit_mw(dev);
    const bool trace = std::getenv("FAKEGPU_NVML_TRACE_FIELD_VALUES") != nullptr;

    for (int i = 0; i < valuesCount; ++i) {
        nvmlFieldValue_t &entry = values[i];
        entry.timestamp = 0;
        entry.latencyUsec = 0;

        // Default: mark field unsupported.
        entry.valueType = NVML_VALUE_TYPE_UNSIGNED_INT;
        entry.nvmlReturn = NVML_ERROR_NOT_SUPPORTED;
        entry.value.uiVal = 0;

        // NVLink counters take the link index as scope, or UINT_MAX for the
        // sum over all links. The modeled traffic of a GPU is spread evenly
        // over its ports; values are KiB.
        if (entry.fieldId == NVML_FI_DEV_NVLINK_LINK_COUNT ||
            entry.fieldId == NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX ||
            entry.fieldId == NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX) {
            unsigned int ports = 0;
            uint64_t tx = 0, rx = 0;
            if (!modeled_nvlink_traffic(device, ports, tx, rx)) continue;
            entry.timestamp = static_cast<long long>(std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
            if (entry.fieldId == NVML_FI_DEV_NVLINK_LINK_COUNT) {
                entry.nvmlReturn = NVML_SUCCESS;
                entry.value.uiVal = ports;
            } else if (entry.scopeId < ports || (ports && entry.scopeId == 0xFFFFFFFFu)) {
                const uint64_t bytes = entry.fieldId == NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX ? tx : rx;
                entry.valueType = NVML_VALUE_TYPE_UNSIGNED_LONG_LONG;
                entry.value.ullVal = entry.scopeId < ports ? bytes / ports / 1024 : bytes / 1024;
                entry.nvmlReturn = NVML_SUCCESS;
            } else {
                entry.nvmlReturn = NVML_ERROR_INVALID_ARGUMENT;
            }
            continue;
        }

        // Scope handling: nvidia-smi may query non-GPU scopes (e.g., module power).
        // Only scopeId==0 is supported for FakeGPU.
        if (entry.scopeId != 0) {
            continue;
        }

        // Power fields (mW) answer like the matching nvmlDeviceGetPower* calls.
        unsigned int min_limit_mw = 0, max_limit_mw = 0;
        switch (entry.fieldId) {
            case NVML_FI_DEV_POWER_AVERAGE:
            case NVML_FI_DEV_POWER_INSTANT:
                entry.nvmlReturn = nvmlDeviceGetPowerUsage(device, &entry.value.uiVal);
                break;
            case NVML_FI_DEV_POWER_MIN_LIMIT:
            case NVML_FI_DEV_POWER_MAX_LIMIT:
                entry.nvmlReturn = power_limit_constraints_mw(dev, min_limit_mw, max_limit_mw);
                entry.value.uiVal = entry.fieldId == NVML_FI_DEV_POWER_MIN_LIMIT ? min_limit_mw : max_limit_mw;
                break;
            case NVML_FI_DEV_POWER_DEFAULT_LIMIT:
            case NVML_FI_DEV_POWER_CURRENT_LIMIT:
            case NVML_FI_DEV_POWER_REQUESTED_LIMIT:
                entry.nvmlReturn = default_limit_mw ? NVML_SUCCESS : NVML_ERROR_NOT_SUPPORTED;
                entry.value.uiVal = default_limit_mw;
                break;
            default:
                break;
        }

        if (trace) {
            std::fprintf(stderr,
                         "[FakeNVML] nvmlDeviceGetFieldValues fieldId=%u scopeId=%u -> nvmlReturn=%d uiVal=%u\n",
                         entry.fieldId, entry.scopeId, entry.nvmlReturn, entry.value.uiVal);
        }
    }

    return NVML_SUCCESS;
}

// Samples
nvmlReturn_t nvmlDeviceGetSamples(nvmlDevice_t device, unsigned int type, unsigned long long lastSeenTimeStamp, unsigned int *sampleValType, unsigned int *sampleCount, void *samples) {
    NVML_REQUIRE_INIT();
    if (!device || !sampleCount) return NVML_ERROR_INVALID_ARGUMENT;
    *sampleCount = 0;
    return NVML_SUCCESS;
}

} // extern C
