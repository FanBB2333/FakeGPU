#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace fake_gpu {

// Minimal architecture identifier surfaced through NVML/device properties.
// Values roughly mirror compute capability/NVML ids; Ampere/Hopper/Ada keep their
// previous values for compatibility.
enum class GpuArch : unsigned int {
    Unknown = 0,
    Maxwell = 5,
    Pascal = 6,
    Volta = 7,
    Turing = 75,
    Ampere = 8,
    Hopper = 9,
    Ada = 10,
    Blackwell = 11,
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

// Parameter pack that can be hydrated from JSON/CSV to define a GPU.
struct GpuProfileParams {
    std::string name;
    int compute_minor = 0;
    uint64_t memory_bytes = 0;
    int sm_count = 0;
    int memory_bus_width_bits = 0;
    int core_clock_mhz = 0;
    int memory_clock_mhz = 0;
    int l2_cache_bytes = 0;
    int shared_mem_per_sm = 0;
    int shared_mem_per_block = 0;
    int shared_mem_per_block_optin = 0;
    int regs_per_block = 0;
    int regs_per_multiprocessor = 0;
    int max_threads_per_multiprocessor = 0;
    int max_blocks_per_multiprocessor = 0;
    unsigned int typical_power_usage_mw = 0;
    unsigned int max_power_limit_mw = 0;
    uint32_t pci_device_id = 0;
    std::vector<GpuDataType> supported_types;
};

struct GpuProfile;

// Base class that stamps architecture + compute capability onto param-driven profiles.
class ArchProfile {
public:
    ArchProfile(GpuArch arch, int compute_major, std::vector<GpuDataType> default_types);
    GpuProfile build(const GpuProfileParams& params) const;

private:
    GpuArch arch;
    int compute_major;
    std::vector<GpuDataType> default_types;
};

class MaxwellProfile : public ArchProfile {
public:
    MaxwellProfile();
};

class PascalProfile : public ArchProfile {
public:
    PascalProfile();
};

class VoltaProfile : public ArchProfile {
public:
    VoltaProfile();
};

class TuringProfile : public ArchProfile {
public:
    TuringProfile();
};

class AmpereProfile : public ArchProfile {
public:
    AmpereProfile();
};

class HopperProfile : public ArchProfile {
public:
    HopperProfile();
};

class AdaProfile : public ArchProfile {
public:
    AdaProfile();
};

class BlackwellProfile : public ArchProfile {
public:
    BlackwellProfile();
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

    // Factory presets for common GPUs.
    static GpuProfile GTX980();
    static GpuProfile P100();
    static GpuProfile V100();
    static GpuProfile T4();
    static GpuProfile A40();
    // Default preset that mirrors the current Fake A100 shape.
    static GpuProfile A100();
    static GpuProfile H100();
    static GpuProfile L40S();
    static GpuProfile B100();
    static GpuProfile B200();
};

const char* to_string(GpuArch arch);
const char* to_string(GpuDataType type);

} // namespace fake_gpu
