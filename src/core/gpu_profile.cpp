#include "gpu_profile.hpp"
#include <algorithm>

namespace fake_gpu {

bool GpuProfile::supports(GpuDataType type) const {
    return std::find(supported_types.begin(), supported_types.end(), type) != supported_types.end();
}

GpuProfile GpuProfile::A100() {
    GpuProfile profile;
    profile.name = "Fake NVIDIA A100-SXM4-80GB";
    profile.architecture = GpuArch::Ampere;
    profile.compute_major = 8;
    profile.compute_minor = 0;
    profile.memory_bytes = 80ULL * 1024 * 1024 * 1024;
    profile.sm_count = 108;
    profile.memory_bus_width_bits = 5120;
    profile.core_clock_mhz = 1410;
    profile.memory_clock_mhz = 1215;
    profile.l2_cache_bytes = 41943040;
    profile.shared_mem_per_sm = 167936;
    profile.shared_mem_per_block = 49152;
    profile.shared_mem_per_block_optin = 166912;
    profile.regs_per_block = 65536;
    profile.regs_per_multiprocessor = 65536;
    profile.max_threads_per_multiprocessor = 2048;
    profile.max_blocks_per_multiprocessor = 32;
    profile.typical_power_usage_mw = 300000;  // 300W
    profile.max_power_limit_mw = 400000;      // 400W
    profile.pci_device_id = 0x20B010DE;
    profile.supported_types = {
        GpuDataType::FP32,
        GpuDataType::TF32,
        GpuDataType::FP16,
        GpuDataType::BF16,
        GpuDataType::INT8,
    };
    return profile;
}

const char* to_string(GpuArch arch) {
    switch (arch) {
        case GpuArch::Ampere: return "Ampere";
        case GpuArch::Hopper: return "Hopper";
        case GpuArch::Ada: return "Ada";
        default: return "Unknown";
    }
}

const char* to_string(GpuDataType type) {
    switch (type) {
        case GpuDataType::FP32: return "fp32";
        case GpuDataType::FP16: return "fp16";
        case GpuDataType::BF16: return "bf16";
        case GpuDataType::TF32: return "tf32";
        case GpuDataType::INT8: return "int8";
        case GpuDataType::INT4: return "int4";
        default: return "unknown";
    }
}

} // namespace fake_gpu
