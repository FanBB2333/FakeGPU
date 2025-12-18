#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace fake_gpu {

// Minimal architecture identifier for NVML and device properties.
// Values match the NVML architecture enum for Ampere (8) so existing
// callers keep seeing the same value.
enum class GpuArch : unsigned int {
    Unknown = 0,
    Ampere = 8,
    Hopper = 9,
    Ada = 10,
};

// Data types supported by a GPU model.
enum class GpuDataType {
    FP32,
    FP16,
    BF16,
    TF32,
    INT8,
    INT4,
};

struct GpuProfile {
    std::string name;
    GpuArch architecture;
    int compute_major;
    int compute_minor;
    uint64_t memory_bytes;
    int sm_count;
    int memory_bus_width_bits;
    int core_clock_mhz;
    int memory_clock_mhz;
    int l2_cache_bytes;
    int shared_mem_per_sm;
    int shared_mem_per_block;
    int shared_mem_per_block_optin;
    int regs_per_block;
    int regs_per_multiprocessor;
    int max_threads_per_multiprocessor;
    int max_blocks_per_multiprocessor;
    unsigned int typical_power_usage_mw;
    unsigned int max_power_limit_mw;
    uint32_t pci_device_id;
    std::vector<GpuDataType> supported_types;

    bool supports(GpuDataType type) const;

    // Default preset that mirrors the current Fake A100 shape.
    static GpuProfile A100();
};

const char* to_string(GpuArch arch);
const char* to_string(GpuDataType type);

} // namespace fake_gpu
