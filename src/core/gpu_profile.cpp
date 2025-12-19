#include "gpu_profile.hpp"

#include <algorithm>
#include <utility>

namespace fake_gpu {

ArchProfile::ArchProfile(GpuArch arch, int compute_major, std::vector<GpuDataType> default_types)
    : arch(arch), compute_major(compute_major), default_types(std::move(default_types)) {
}

GpuProfile ArchProfile::build(const GpuProfileParams& params) const {
    GpuProfile profile;
    profile.name = params.name;
    profile.architecture = arch;
    profile.compute_major = compute_major;
    profile.compute_minor = params.compute_minor;
    profile.memory_bytes = params.memory_bytes;
    profile.sm_count = params.sm_count;
    profile.memory_bus_width_bits = params.memory_bus_width_bits;
    profile.core_clock_mhz = params.core_clock_mhz;
    profile.memory_clock_mhz = params.memory_clock_mhz;
    profile.l2_cache_bytes = params.l2_cache_bytes;
    profile.shared_mem_per_sm = params.shared_mem_per_sm;
    profile.shared_mem_per_block = params.shared_mem_per_block;
    profile.shared_mem_per_block_optin = params.shared_mem_per_block_optin;
    profile.regs_per_block = params.regs_per_block;
    profile.regs_per_multiprocessor = params.regs_per_multiprocessor;
    profile.max_threads_per_multiprocessor = params.max_threads_per_multiprocessor;
    profile.max_blocks_per_multiprocessor = params.max_blocks_per_multiprocessor;
    profile.typical_power_usage_mw = params.typical_power_usage_mw;
    profile.max_power_limit_mw = params.max_power_limit_mw;
    profile.pci_device_id = params.pci_device_id;
    profile.supported_types = params.supported_types.empty() ? default_types : params.supported_types;
    return profile;
}

MaxwellProfile::MaxwellProfile()
    : ArchProfile(GpuArch::Maxwell, 5, {GpuDataType::FP32}) {
}

PascalProfile::PascalProfile()
    : ArchProfile(GpuArch::Pascal, 6, {GpuDataType::FP32, GpuDataType::FP16}) {
}

VoltaProfile::VoltaProfile()
    : ArchProfile(GpuArch::Volta, 7, {GpuDataType::FP32, GpuDataType::FP16}) {
}

TuringProfile::TuringProfile()
    : ArchProfile(GpuArch::Turing, 7, {GpuDataType::FP32, GpuDataType::FP16, GpuDataType::INT8, GpuDataType::INT4}) {
}

AmpereProfile::AmpereProfile()
    : ArchProfile(GpuArch::Ampere, 8, {GpuDataType::FP32, GpuDataType::TF32, GpuDataType::FP16, GpuDataType::BF16, GpuDataType::INT8}) {
}

HopperProfile::HopperProfile()
    : ArchProfile(GpuArch::Hopper, 9, {GpuDataType::FP32, GpuDataType::TF32, GpuDataType::FP16, GpuDataType::BF16, GpuDataType::INT8, GpuDataType::INT4}) {
}

AdaProfile::AdaProfile()
    : ArchProfile(GpuArch::Ada, 8, {GpuDataType::FP32, GpuDataType::TF32, GpuDataType::FP16, GpuDataType::BF16, GpuDataType::INT8}) {
}

BlackwellProfile::BlackwellProfile()
    : ArchProfile(GpuArch::Blackwell, 10, {GpuDataType::FP32, GpuDataType::TF32, GpuDataType::FP16, GpuDataType::BF16, GpuDataType::INT8, GpuDataType::INT4}) {
}

bool GpuProfile::supports(GpuDataType type) const {
    return std::find(supported_types.begin(), supported_types.end(), type) != supported_types.end();
}

namespace {
const MaxwellProfile kMaxwell;
const PascalProfile kPascal;
const VoltaProfile kVolta;
const TuringProfile kTuring;
const AmpereProfile kAmpere;
const HopperProfile kHopper;
const AdaProfile kAda;
const BlackwellProfile kBlackwell;
} // namespace

GpuProfile GpuProfile::GTX980() {
    GpuProfileParams params;
    params.name = "Fake NVIDIA GeForce GTX 980";
    params.compute_minor = 2;
    params.memory_bytes = 4ULL * 1024 * 1024 * 1024;
    params.sm_count = 16;
    params.memory_bus_width_bits = 256;
    params.core_clock_mhz = 1216;
    params.memory_clock_mhz = 1750;
    params.l2_cache_bytes = 2097152;
    params.shared_mem_per_sm = 65536;
    params.shared_mem_per_block = 49152;
    params.shared_mem_per_block_optin = 65536;
    params.regs_per_block = 65536;
    params.regs_per_multiprocessor = 65536;
    params.max_threads_per_multiprocessor = 2048;
    params.max_blocks_per_multiprocessor = 32;
    params.typical_power_usage_mw = 165000;
    params.max_power_limit_mw = 180000;
    params.pci_device_id = 0x13C010DE;
    return kMaxwell.build(params);
}

GpuProfile GpuProfile::P100() {
    GpuProfileParams params;
    params.name = "Fake NVIDIA Tesla P100-PCIE-16GB";
    params.compute_minor = 0;
    params.memory_bytes = 16ULL * 1024 * 1024 * 1024;
    params.sm_count = 56;
    params.memory_bus_width_bits = 4096;
    params.core_clock_mhz = 1328;
    params.memory_clock_mhz = 715;
    params.l2_cache_bytes = 4194304;
    params.shared_mem_per_sm = 65536;
    params.shared_mem_per_block = 49152;
    params.shared_mem_per_block_optin = 65536;
    params.regs_per_block = 65536;
    params.regs_per_multiprocessor = 65536;
    params.max_threads_per_multiprocessor = 2048;
    params.max_blocks_per_multiprocessor = 32;
    params.typical_power_usage_mw = 250000;
    params.max_power_limit_mw = 300000;
    params.pci_device_id = 0x15F810DE;
    return kPascal.build(params);
}

GpuProfile GpuProfile::V100() {
    GpuProfileParams params;
    params.name = "Fake NVIDIA Tesla V100-SXM2-32GB";
    params.compute_minor = 0;
    params.memory_bytes = 32ULL * 1024 * 1024 * 1024;
    params.sm_count = 80;
    params.memory_bus_width_bits = 4096;
    params.core_clock_mhz = 1380;
    params.memory_clock_mhz = 877;
    params.l2_cache_bytes = 6291456;
    params.shared_mem_per_sm = 98304;
    params.shared_mem_per_block = 49152;
    params.shared_mem_per_block_optin = 98304;
    params.regs_per_block = 65536;
    params.regs_per_multiprocessor = 65536;
    params.max_threads_per_multiprocessor = 2048;
    params.max_blocks_per_multiprocessor = 32;
    params.typical_power_usage_mw = 250000;
    params.max_power_limit_mw = 300000;
    params.pci_device_id = 0x1DB410DE;
    return kVolta.build(params);
}

GpuProfile GpuProfile::T4() {
    GpuProfileParams params;
    params.name = "Fake NVIDIA T4";
    params.compute_minor = 5;
    params.memory_bytes = 16ULL * 1024 * 1024 * 1024;
    params.sm_count = 40;
    params.memory_bus_width_bits = 256;
    params.core_clock_mhz = 1590;
    params.memory_clock_mhz = 1000;
    params.l2_cache_bytes = 4194304;
    params.shared_mem_per_sm = 65536;
    params.shared_mem_per_block = 49152;
    params.shared_mem_per_block_optin = 65536;
    params.regs_per_block = 65536;
    params.regs_per_multiprocessor = 65536;
    params.max_threads_per_multiprocessor = 1024;
    params.max_blocks_per_multiprocessor = 32;
    params.typical_power_usage_mw = 70000;
    params.max_power_limit_mw = 75000;
    params.pci_device_id = 0x1EB810DE;
    return kTuring.build(params);
}

GpuProfile GpuProfile::A40() {
    GpuProfileParams params;
    params.name = "Fake NVIDIA A40";
    params.compute_minor = 6;
    params.memory_bytes = 48ULL * 1024 * 1024 * 1024;
    params.sm_count = 84;
    params.memory_bus_width_bits = 384;
    params.core_clock_mhz = 1530;
    params.memory_clock_mhz = 1188;
    params.l2_cache_bytes = 6291456;
    params.shared_mem_per_sm = 102400;
    params.shared_mem_per_block = 49152;
    params.shared_mem_per_block_optin = 102400;
    params.regs_per_block = 65536;
    params.regs_per_multiprocessor = 65536;
    params.max_threads_per_multiprocessor = 1536;
    params.max_blocks_per_multiprocessor = 32;
    params.typical_power_usage_mw = 300000;
    params.max_power_limit_mw = 350000;
    params.pci_device_id = 0x223510DE;
    return kAmpere.build(params);
}

GpuProfile GpuProfile::A100() {
    GpuProfileParams params;
    params.name = "Fake NVIDIA A100-SXM4-80GB";
    params.compute_minor = 0;
    params.memory_bytes = 80ULL * 1024 * 1024 * 1024;
    params.sm_count = 108;
    params.memory_bus_width_bits = 5120;
    params.core_clock_mhz = 1410;
    params.memory_clock_mhz = 1215;
    params.l2_cache_bytes = 41943040;
    params.shared_mem_per_sm = 167936;
    params.shared_mem_per_block = 49152;
    params.shared_mem_per_block_optin = 166912;
    params.regs_per_block = 65536;
    params.regs_per_multiprocessor = 65536;
    params.max_threads_per_multiprocessor = 2048;
    params.max_blocks_per_multiprocessor = 32;
    params.typical_power_usage_mw = 300000;  // 300W
    params.max_power_limit_mw = 400000;      // 400W
    params.pci_device_id = 0x20B010DE;
    return kAmpere.build(params);
}

GpuProfile GpuProfile::H100() {
    GpuProfileParams params;
    params.name = "Fake NVIDIA H100-SXM-80GB";
    params.compute_minor = 0;
    params.memory_bytes = 80ULL * 1024 * 1024 * 1024;
    params.sm_count = 132;
    params.memory_bus_width_bits = 5120;
    params.core_clock_mhz = 1800;
    params.memory_clock_mhz = 1593;
    params.l2_cache_bytes = 52428800;
    params.shared_mem_per_sm = 229376;
    params.shared_mem_per_block = 98304;
    params.shared_mem_per_block_optin = 229376;
    params.regs_per_block = 65536;
    params.regs_per_multiprocessor = 65536;
    params.max_threads_per_multiprocessor = 2048;
    params.max_blocks_per_multiprocessor = 32;
    params.typical_power_usage_mw = 500000;
    params.max_power_limit_mw = 700000;
    params.pci_device_id = 0x233010DE;
    return kHopper.build(params);
}

GpuProfile GpuProfile::L40S() {
    GpuProfileParams params;
    params.name = "Fake NVIDIA L40S";
    params.compute_minor = 9;
    params.memory_bytes = 48ULL * 1024 * 1024 * 1024;
    params.sm_count = 142;
    params.memory_bus_width_bits = 384;
    params.core_clock_mhz = 2520;
    params.memory_clock_mhz = 1500;
    params.l2_cache_bytes = 73728000;
    params.shared_mem_per_sm = 102400;
    params.shared_mem_per_block = 65536;
    params.shared_mem_per_block_optin = 102400;
    params.regs_per_block = 65536;
    params.regs_per_multiprocessor = 65536;
    params.max_threads_per_multiprocessor = 1536;
    params.max_blocks_per_multiprocessor = 32;
    params.typical_power_usage_mw = 300000;
    params.max_power_limit_mw = 350000;
    params.pci_device_id = 0x26B010DE;
    return kAda.build(params);
}

GpuProfile GpuProfile::B100() {
    GpuProfileParams params;
    params.name = "Fake NVIDIA B100";
    params.compute_minor = 0;
    params.memory_bytes = 80ULL * 1024 * 1024 * 1024;
    params.sm_count = 144;
    params.memory_bus_width_bits = 5120;
    params.core_clock_mhz = 1950;
    params.memory_clock_mhz = 2300;
    params.l2_cache_bytes = 65536000;
    params.shared_mem_per_sm = 262144;
    params.shared_mem_per_block = 131072;
    params.shared_mem_per_block_optin = 262144;
    params.regs_per_block = 65536;
    params.regs_per_multiprocessor = 131072;
    params.max_threads_per_multiprocessor = 2048;
    params.max_blocks_per_multiprocessor = 32;
    params.typical_power_usage_mw = 600000;
    params.max_power_limit_mw = 700000;
    params.pci_device_id = 0x26B410DE;
    return kBlackwell.build(params);
}

GpuProfile GpuProfile::B200() {
    GpuProfileParams params;
    params.name = "Fake NVIDIA B200";
    params.compute_minor = 0;
    params.memory_bytes = 192ULL * 1024 * 1024 * 1024;
    params.sm_count = 192;
    params.memory_bus_width_bits = 6144;
    params.core_clock_mhz = 2050;
    params.memory_clock_mhz = 2400;
    params.l2_cache_bytes = 90112000;
    params.shared_mem_per_sm = 262144;
    params.shared_mem_per_block = 131072;
    params.shared_mem_per_block_optin = 262144;
    params.regs_per_block = 65536;
    params.regs_per_multiprocessor = 131072;
    params.max_threads_per_multiprocessor = 2048;
    params.max_blocks_per_multiprocessor = 32;
    params.typical_power_usage_mw = 700000;
    params.max_power_limit_mw = 800000;
    params.pci_device_id = 0x26B510DE;
    return kBlackwell.build(params);
}

const char* to_string(GpuArch arch) {
    switch (arch) {
        case GpuArch::Maxwell: return "Maxwell";
        case GpuArch::Pascal: return "Pascal";
        case GpuArch::Volta: return "Volta";
        case GpuArch::Turing: return "Turing";
        case GpuArch::Ampere: return "Ampere";
        case GpuArch::Hopper: return "Hopper";
        case GpuArch::Ada: return "Ada";
        case GpuArch::Blackwell: return "Blackwell";
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
