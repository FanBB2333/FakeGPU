#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace fake_gpu {

// Architecture identifier. nvmlDeviceGetArchitecture returns the value as-is,
// so each one is the matching NVML_DEVICE_ARCH_* constant from nvml.h.
enum class GpuArch : unsigned int {
    Unknown = 0xffffffff,
    Maxwell = 3,
    Pascal = 4,
    Volta = 5,
    Turing = 6,
    Ampere = 7,
    Ada = 8,
    Hopper = 9,
    Blackwell = 10,
};

// Data types supported by a GPU model.
enum class GpuDataType {
    FP32,
    FP16,
    BF16,
    TF32,
    INT8,
    INT4,
    FP8,
    FP4,
};

// Parameter pack that can be hydrated from JSON/CSV to define a GPU.
struct GpuProfileParams {
    std::string name;
    // Optional per-profile override. This is needed for architecture families
    // that span multiple CUDA compute-capability major versions.
    int compute_major = 0;
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
    // PCIe generation and lane count of the GPU itself; 0 when unknown.
    int pcie_generation = 0;
    int pcie_lanes = 0;
    std::vector<GpuDataType> supported_types;

    // Optional YAML fields used to resolve the values below.
    uint64_t reported_memory_bytes = 0;  // 0: not published
    unsigned int mig_max_instances = 0;  // absent or 0: no MIG
    std::string segment;                 // directory: datacenter, workstation, ...
    std::string ecc;                     // "supported", "unsupported" or empty
    std::string form_factor;
    std::string memory_kind;
    std::string memory_type;
    // Per-SKU device attributes; std::nullopt takes the class default.
    std::optional<int> ecc_enabled;
    std::optional<int> async_engine_count;
    std::optional<int> single_to_double_precision_perf_ratio;
    std::optional<int> max_persisting_l2_cache_bytes;
    std::optional<int> pageable_memory_access;
    std::optional<int> can_flush_remote_writes;
    std::optional<int> max_access_policy_window_bytes;
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
    std::string id;
    std::string name;
    GpuArch architecture = GpuArch::Unknown;
    int compute_major = 0;
    int compute_minor = 0;
    // Device memory total that CUDA and NVML report: the profile's
    // reported_memory_bytes when it has one, else the nominal capacity.
    uint64_t memory_bytes = 0;
    // The YAML memory_bytes: nominal capacity, vendor GB stored as GiB.
    uint64_t nominal_memory_bytes = 0;
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
    int pcie_generation = 0;
    int pcie_lanes = 0;
    std::vector<GpuDataType> supported_types;
    // Maximum MIG instances (YAML mig_max_instances); 0 when the GPU has no MIG.
    unsigned int mig_max_instances = 0;

    // CUDA device attributes that differ between products. A YAML key of the
    // same name sets one for a SKU; otherwise the class defaults documented in
    // profiles/README.md apply (see resolve_device_attributes in the .cpp).
    int ecc_enabled = 0;                            // CU_DEVICE_ATTRIBUTE_ECC_ENABLED
    int async_engine_count = 0;                     // CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT
    int single_to_double_precision_perf_ratio = 0;  // CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO
    int max_persisting_l2_cache_bytes = 0;          // CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE
    int pageable_memory_access = 0;                 // CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS
    int can_flush_remote_writes = 0;                // CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES
    int max_access_policy_window_bytes = 0;         // CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE
    int reserved_shared_mem_per_block = 0;          // CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK

    bool supports(GpuDataType type) const;
    bool mig_supported() const { return mig_max_instances > 0; }

    // The default device (the compiled `a100` YAML profile).
    static GpuProfile A100();
};

// Returns the preset IDs compiled into the binary (from `profiles/**/*.yaml` at build time).
std::vector<std::string> builtin_profile_ids();

// Resolve the NVIDIA hardware architecture for an explicit CUDA compute capability.
GpuArch architecture_from_compute_capability(int major, int minor);

// Resolve a built-in YAML profile (any `id:` from `profiles/**/*.yaml`
// compiled into the binary) by ID, case-insensitive and with an optional
// .yaml/.yml suffix. Returns nullopt if the ID is unknown.
std::optional<GpuProfile> profile_from_preset_id(const std::string& id);

// Map a cudaDataType_t integer value to the corresponding GPU capability type.
// Returns nullopt for types that do not participate in hardware compatibility checks.
std::optional<GpuDataType> cuda_dtype_to_gpu_dtype(int cuda_data_type);

const char* to_string(GpuArch arch);
const char* to_string(GpuDataType type);

} // namespace fake_gpu
