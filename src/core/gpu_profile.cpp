#include "gpu_profile.hpp"
#include "generated_profiles.hpp"
#include "logging.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <optional>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace fake_gpu {

namespace {

// Ratio of FP32 to FP64 throughput, as CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_
// PRECISION_PERF_RATIO reports it. CUDA Programming Guide: the "Ratio of FP32
// to FP64 Throughput" row of "Device and Streaming Multiprocessor (SM)
// Information per Compute Capability" for 7.5 and newer, and the CUDA 12.4
// guide's "Throughput of Native Arithmetic Instructions" (32- and 64-bit
// floating-point add/multiply results per clock per SM) for older devices.
int fp32_to_fp64_throughput_ratio(int major, int minor) {
    switch (major) {
        case 5: return 32;                       // 128 : 4
        case 6: return minor == 0 ? 2 : 32;      // 6.0 64 : 32; 6.1/6.2 128 : 4
        case 7: return minor == 5 ? 32 : 2;      // 7.0/7.2 64 : 32; 7.5 64 : 2
        case 8: return minor == 0 ? 2 : 64;      // 8.0 2:1; 8.6, 8.7, 8.9 64:1
        case 9: return 2;
        case 10:
            if (minor == 0) return 2;
            if (minor == 7) return 4;
            return 64;                           // 10.3
        case 11:
        case 12: return 64;
        default: return 2;
    }
}

// Largest accessPolicyWindow.num_bytes. Modeled default: the value one Ampere
// (sm_80) GPU reports under driver 580 (128 MiB - 4 KiB). The access policy
// window controls L2 persistence, which the CUDA Programming Guide limits to
// compute capability 8.0 and newer.
constexpr int kModeledAccessPolicyWindowBytes = 134213632;

// Fill the per-SKU CUDA device attributes from the YAML overrides or the
// defaults of the profile's class (segment, form factor, compute capability).
void resolve_device_attributes(GpuProfile& profile, const GpuProfileParams& params) {
    const bool persistence = profile.compute_major >= 8;
    const bool embedded = params.form_factor == "embedded";

    // ECC_ENABLED reports the current ECC mode. Data-center GPUs ship with
    // ECC on (NVIDIA's nvidia-smi and deviceQuery outputs for A100, H100, L4
    // and GH200 show it enabled); HBM workstation boards that support ECC are
    // modeled the same way. GeForce, Jetson and GDDR workstation boards are
    // modeled with ECC off: NVIDIA does not publish the latter's default.
    int ecc_default = 0;
    if (params.ecc != "unsupported") {
        if (params.segment == "datacenter") {
            ecc_default = 1;
        } else if (params.ecc == "supported" && params.memory_type.rfind("HBM", 0) == 0) {
            ecc_default = 1;
        }
    }
    profile.ecc_enabled = params.ecc_enabled.value_or(ecc_default);

    // Modeled: the number of copy engines CUDA reports varies by SKU and driver.
    profile.async_engine_count = params.async_engine_count.value_or(2);

    profile.single_to_double_precision_perf_ratio = params.single_to_double_precision_perf_ratio.value_or(
        fp32_to_fp64_throughput_ratio(profile.compute_major, profile.compute_minor));

    // L2 persistence needs compute capability 8.0 (CUDA Programming Guide,
    // "L2 cache set-aside"). Modeled maximum: 75% of L2, the share NVIDIA's
    // A100 example sets aside (30 MB of 40 MB); real GPUs can allow less.
    profile.max_persisting_l2_cache_bytes = params.max_persisting_l2_cache_bytes.value_or(
        persistence ? static_cast<int>(static_cast<int64_t>(profile.l2_cache_bytes) * 3 / 4) : 0);

    // Pageable memory access is 1 under Linux HMM (compute capability 7.5+,
    // open kernel modules) and with ATS on NVLink-C2C superchips (CUDA
    // Programming Guide, "Understanding Memory"). The modeled host is a
    // current Linux driver with the open kernel modules. Jetson is modeled 0.
    int pageable_default = 0;
    if (!embedded) {
        if (params.form_factor == "superchip" ||
            profile.compute_major > 7 ||
            (profile.compute_major == 7 && profile.compute_minor >= 5)) {
            pageable_default = 1;
        }
    }
    profile.pageable_memory_access = params.pageable_memory_access.value_or(pageable_default);

    // Modeled 0: an x86 Linux host reports 0 (observed on one Ampere GPU).
    profile.can_flush_remote_writes = params.can_flush_remote_writes.value_or(0);

    profile.max_access_policy_window_bytes = params.max_access_policy_window_bytes.value_or(
        persistence ? kModeledAccessPolicyWindowBytes : 0);

    // CUDA Programming Guide (compute capability 8.x and 9.0): "The 1 KB of
    // shared memory not made available to a thread block is reserved for
    // system use", the gap between the per-SM and the per-block opt-in
    // maximum (1 KB in the catalog's limits). Older GPUs reserve none; their
    // per-block limit is an architectural cap, not a reservation.
    profile.reserved_shared_mem_per_block = persistence
        ? std::max(0, profile.shared_mem_per_sm - profile.shared_mem_per_block_optin)
        : 0;
}

} // namespace

ArchProfile::ArchProfile(GpuArch arch, int compute_major, std::vector<GpuDataType> default_types)
    : arch(arch), compute_major(compute_major), default_types(std::move(default_types)) {
}

GpuProfile ArchProfile::build(const GpuProfileParams& params) const {
    GpuProfile profile;
    profile.name = params.name;
    profile.architecture = arch;
    profile.compute_major = params.compute_major > 0 ? params.compute_major : compute_major;
    profile.compute_minor = params.compute_minor;
    profile.nominal_memory_bytes = params.memory_bytes;
    profile.memory_bytes = params.reported_memory_bytes > 0 ? params.reported_memory_bytes : params.memory_bytes;
    profile.mig_max_instances = params.mig_max_instances;
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
    profile.pcie_generation = params.pcie_generation;
    profile.pcie_lanes = params.pcie_lanes;
    profile.supported_types = params.supported_types.empty() ? default_types : params.supported_types;
    resolve_device_attributes(profile, params);
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
    : ArchProfile(GpuArch::Ampere, 8, {GpuDataType::FP32, GpuDataType::TF32, GpuDataType::FP16, GpuDataType::BF16, GpuDataType::INT8, GpuDataType::INT4}) {
}

// Reduced-precision defaults are Tensor Core input types (CUDA Programming
// Guide, "Input Data Types Supported by Tensor Core Acceleration"): INT4 is a
// Tensor Core input on compute capability 7.5-8.9 only.
HopperProfile::HopperProfile()
    : ArchProfile(GpuArch::Hopper, 9, {GpuDataType::FP32, GpuDataType::TF32, GpuDataType::FP16, GpuDataType::BF16, GpuDataType::INT8, GpuDataType::FP8}) {
}

AdaProfile::AdaProfile()
    : ArchProfile(GpuArch::Ada, 8, {GpuDataType::FP32, GpuDataType::TF32, GpuDataType::FP16, GpuDataType::BF16, GpuDataType::INT8, GpuDataType::INT4, GpuDataType::FP8}) {
}

BlackwellProfile::BlackwellProfile()
    : ArchProfile(GpuArch::Blackwell, 10, {GpuDataType::FP32, GpuDataType::TF32, GpuDataType::FP16, GpuDataType::BF16, GpuDataType::INT8, GpuDataType::FP8, GpuDataType::FP4}) {
}

bool GpuProfile::supports(GpuDataType type) const {
    return std::find(supported_types.begin(), supported_types.end(), type) != supported_types.end();
}

GpuArch architecture_from_compute_capability(int major, int minor) {
    if ((major == 10 && (minor == 0 || minor == 3)) ||
        (major == 11 && minor == 0) ||
        (major == 12 && (minor == 0 || minor == 1))) {
        return GpuArch::Blackwell;
    }
    if (major == 9 && minor == 0) return GpuArch::Hopper;
    if (major == 8 && minor == 9) return GpuArch::Ada;
    if (major == 8 && (minor == 0 || minor == 6 || minor == 7)) return GpuArch::Ampere;
    if (major == 7 && minor == 5) return GpuArch::Turing;
    if (major == 7 && (minor == 0 || minor == 2)) return GpuArch::Volta;
    if (major == 6 && (minor == 0 || minor == 1 || minor == 2)) return GpuArch::Pascal;
    if (major == 5 && (minor == 0 || minor == 2 || minor == 3)) return GpuArch::Maxwell;
    return GpuArch::Unknown;
}

namespace {

std::string trim(const std::string& value) {
    const size_t begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return "";
    const size_t end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::string normalize_profile_id(std::string value) {
    value = trim(to_lower(std::move(value)));
    if (value.size() >= 5 && value.compare(value.size() - 5, 5, ".yaml") == 0) {
        value.resize(value.size() - 5);
    } else if (value.size() >= 4 && value.compare(value.size() - 4, 4, ".yml") == 0) {
        value.resize(value.size() - 4);
    }
    return trim(value);
}

struct ParsedYaml {
    std::unordered_map<std::string, std::string> scalars;
    std::unordered_map<std::string, std::vector<std::string>> lists;
};

bool parse_simple_yaml(const std::string& yaml, ParsedYaml& out, std::string& error) {
    std::istringstream stream(yaml);
    std::string line;
    std::string current_list;
    size_t line_no = 0;

    while (std::getline(stream, line)) {
        ++line_no;
        std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '#') continue;

        if (trimmed.rfind("- ", 0) == 0) {
            if (current_list.empty()) {
                error = "List item without a preceding key on line " + std::to_string(line_no);
                return false;
            }
            out.lists[current_list].push_back(trim(trimmed.substr(2)));
            continue;
        }

        size_t colon = trimmed.find(':');
        if (colon == std::string::npos) {
            error = "Invalid line (missing colon) on line " + std::to_string(line_no);
            return false;
        }

        std::string key = trim(trimmed.substr(0, colon));
        std::string value = trim(trimmed.substr(colon + 1));
        if (!value.empty()) {
            out.scalars[key] = value;
            current_list.clear();
        } else {
            current_list = key;
            out.lists[key]; // ensure list exists
        }
    }
    return true;
}

template <typename T>
bool parse_integer(const std::string& text, T& out) {
    try {
        uint64_t raw = std::stoull(text, nullptr, 0);
        if (raw > static_cast<uint64_t>(std::numeric_limits<T>::max())) return false;
        out = static_cast<T>(raw);
        return true;
    } catch (...) {
        return false;
    }
}

std::optional<GpuDataType> parse_data_type(const std::string& text) {
    const std::string lower = to_lower(text);
    if (lower == "fp32") return GpuDataType::FP32;
    if (lower == "fp16") return GpuDataType::FP16;
    if (lower == "bf16") return GpuDataType::BF16;
    if (lower == "tf32") return GpuDataType::TF32;
    if (lower == "int8") return GpuDataType::INT8;
    if (lower == "int4") return GpuDataType::INT4;
    if (lower == "fp8") return GpuDataType::FP8;
    if (lower == "fp4") return GpuDataType::FP4;
    return std::nullopt;
}

std::optional<GpuArch> parse_arch(const std::string& text) {
    const std::string lower = to_lower(text);
    if (lower == "maxwell") return GpuArch::Maxwell;
    if (lower == "pascal") return GpuArch::Pascal;
    if (lower == "volta") return GpuArch::Volta;
    if (lower == "turing") return GpuArch::Turing;
    if (lower == "ampere") return GpuArch::Ampere;
    if (lower == "hopper") return GpuArch::Hopper;
    if (lower == "ada") return GpuArch::Ada;
    if (lower == "blackwell") return GpuArch::Blackwell;
    return std::nullopt;
}

struct ProfileDefinition {
    std::string id;
    GpuArch arch = GpuArch::Unknown;
    GpuProfileParams params;
};

const ArchProfile* get_arch_profile(GpuArch arch) {
    switch (arch) {
        case GpuArch::Maxwell: {
            static const MaxwellProfile kMaxwell;
            return &kMaxwell;
        }
        case GpuArch::Pascal: {
            static const PascalProfile kPascal;
            return &kPascal;
        }
        case GpuArch::Volta: {
            static const VoltaProfile kVolta;
            return &kVolta;
        }
        case GpuArch::Turing: {
            static const TuringProfile kTuring;
            return &kTuring;
        }
        case GpuArch::Ampere: {
            static const AmpereProfile kAmpere;
            return &kAmpere;
        }
        case GpuArch::Hopper: {
            static const HopperProfile kHopper;
            return &kHopper;
        }
        case GpuArch::Ada: {
            static const AdaProfile kAda;
            return &kAda;
        }
        case GpuArch::Blackwell: {
            static const BlackwellProfile kBlackwell;
            return &kBlackwell;
        }
        default: return nullptr;
    }
}

GpuProfile build_profile_from_definition(const ProfileDefinition& def) {
    GpuProfile profile = get_arch_profile(def.arch)->build(def.params);
    profile.id = def.id;
    return profile;
}

bool parse_required_int(const ParsedYaml& parsed, const std::string& key, uint64_t& value, const char* filename) {
    auto it = parsed.scalars.find(key);
    if (it == parsed.scalars.end()) {
        FGPU_LOG("[GpuProfile] Missing required key '%s' in %s\n", key.c_str(), filename);
        return false;
    }
    if (!parse_integer(it->second, value)) {
        FGPU_LOG("[GpuProfile] Failed to parse numeric key '%s' in %s (value=%s)\n", key.c_str(), filename, it->second.c_str());
        return false;
    }
    return true;
}

std::optional<ProfileDefinition> parse_definition(const ProfileYamlBlob& blob) {
    ParsedYaml parsed;
    std::string error;
    if (!parse_simple_yaml(blob.yaml, parsed, error)) {
        FGPU_LOG("[GpuProfile] Failed to parse %s: %s\n", blob.filename, error.c_str());
        return std::nullopt;
    }

    ProfileDefinition def;
    const std::string filename = blob.filename ? std::string(blob.filename) : "";
    const size_t dot = filename.find_last_of('.');
    const std::string filename_id = dot == std::string::npos ? filename : filename.substr(0, dot);

    auto id_it = parsed.scalars.find("id");
    def.id = to_lower(id_it != parsed.scalars.end() ? id_it->second : filename_id);

    auto name_it = parsed.scalars.find("name");
    def.params.name = name_it != parsed.scalars.end() ? name_it->second : def.id;

    auto arch_it = parsed.scalars.find("architecture");
    if (arch_it == parsed.scalars.end()) {
        FGPU_LOG("[GpuProfile] Missing 'architecture' key in %s\n", blob.filename);
        return std::nullopt;
    }
    std::optional<GpuArch> arch = parse_arch(arch_it->second);
    if (!arch.has_value()) {
        FGPU_LOG("[GpuProfile] Unknown architecture '%s' in %s\n", arch_it->second.c_str(), blob.filename);
        return std::nullopt;
    }
    def.arch = arch.value();

    auto compute_major_it = parsed.scalars.find("compute_major");
    if (compute_major_it == parsed.scalars.end()) {
        FGPU_LOG("[GpuProfile] Missing 'compute_major' key in %s\n", blob.filename);
        return std::nullopt;
    }
    if (!parse_integer(compute_major_it->second, def.params.compute_major) ||
        def.params.compute_major <= 0) {
        FGPU_LOG("[GpuProfile] Invalid compute_major in %s: %s\n", blob.filename, compute_major_it->second.c_str());
        return std::nullopt;
    }

    auto compute_minor_it = parsed.scalars.find("compute_minor");
    if (compute_minor_it != parsed.scalars.end()) {
        if (!parse_integer(compute_minor_it->second, def.params.compute_minor)) {
            FGPU_LOG("[GpuProfile] Invalid compute_minor in %s: %s\n", blob.filename, compute_minor_it->second.c_str());
            return std::nullopt;
        }
    }

    auto require = [&](const char* key, auto& target) {
        uint64_t value = 0;
        if (!parse_required_int(parsed, key, value, blob.filename)) return false;
        target = static_cast<std::remove_reference_t<decltype(target)>>(value);
        return true;
    };
    if (!require("memory_bytes", def.params.memory_bytes) ||
        !require("sm_count", def.params.sm_count) ||
        !require("memory_bus_width_bits", def.params.memory_bus_width_bits) ||
        !require("core_clock_mhz", def.params.core_clock_mhz) ||
        !require("memory_clock_mhz", def.params.memory_clock_mhz) ||
        !require("l2_cache_bytes", def.params.l2_cache_bytes) ||
        !require("shared_mem_per_sm", def.params.shared_mem_per_sm) ||
        !require("shared_mem_per_block", def.params.shared_mem_per_block) ||
        !require("shared_mem_per_block_optin", def.params.shared_mem_per_block_optin) ||
        !require("regs_per_block", def.params.regs_per_block) ||
        !require("regs_per_multiprocessor", def.params.regs_per_multiprocessor) ||
        !require("max_threads_per_multiprocessor", def.params.max_threads_per_multiprocessor) ||
        !require("max_blocks_per_multiprocessor", def.params.max_blocks_per_multiprocessor) ||
        !require("typical_power_usage_mw", def.params.typical_power_usage_mw) ||
        !require("max_power_limit_mw", def.params.max_power_limit_mw) ||
        !require("pci_device_id", def.params.pci_device_id)) {
        return std::nullopt;
    }
    for (const auto& [key, target] : {
             std::pair<const char*, int*>{"pcie_generation", &def.params.pcie_generation},
             std::pair<const char*, int*>{"pcie_lanes", &def.params.pcie_lanes}}) {
        auto optional_it = parsed.scalars.find(key);
        if (optional_it == parsed.scalars.end()) continue;
        if (!parse_integer(optional_it->second, *target) || *target < 0) {
            FGPU_LOG("[GpuProfile] Failed to parse numeric key '%s' in %s (value=%s)\n", key, blob.filename, optional_it->second.c_str());
            return std::nullopt;
        }
    }
    if (auto it = parsed.scalars.find("reported_memory_bytes"); it != parsed.scalars.end()) {
        if (!parse_integer(it->second, def.params.reported_memory_bytes)) {
            FGPU_LOG("[GpuProfile] Failed to parse numeric key 'reported_memory_bytes' in %s (value=%s)\n", blob.filename, it->second.c_str());
            return std::nullopt;
        }
    }
    if (auto it = parsed.scalars.find("mig_max_instances"); it != parsed.scalars.end()) {
        if (!parse_integer(it->second, def.params.mig_max_instances)) {
            FGPU_LOG("[GpuProfile] Failed to parse numeric key 'mig_max_instances' in %s (value=%s)\n", blob.filename, it->second.c_str());
            return std::nullopt;
        }
    }
    for (const auto& [key, target] : {
             std::pair<const char*, std::optional<int>*>{"ecc_enabled", &def.params.ecc_enabled},
             std::pair<const char*, std::optional<int>*>{"async_engine_count", &def.params.async_engine_count},
             std::pair<const char*, std::optional<int>*>{"single_to_double_precision_perf_ratio", &def.params.single_to_double_precision_perf_ratio},
             std::pair<const char*, std::optional<int>*>{"max_persisting_l2_cache_bytes", &def.params.max_persisting_l2_cache_bytes},
             std::pair<const char*, std::optional<int>*>{"pageable_memory_access", &def.params.pageable_memory_access},
             std::pair<const char*, std::optional<int>*>{"can_flush_remote_writes", &def.params.can_flush_remote_writes},
             std::pair<const char*, std::optional<int>*>{"max_access_policy_window_bytes", &def.params.max_access_policy_window_bytes}}) {
        auto optional_it = parsed.scalars.find(key);
        if (optional_it == parsed.scalars.end()) continue;
        int value = 0;
        if (!parse_integer(optional_it->second, value)) {
            FGPU_LOG("[GpuProfile] Failed to parse numeric key '%s' in %s (value=%s)\n", key, blob.filename, optional_it->second.c_str());
            return std::nullopt;
        }
        *target = value;
    }
    for (const auto& [key, target] : {
             std::pair<const char*, std::string*>{"ecc", &def.params.ecc},
             std::pair<const char*, std::string*>{"form_factor", &def.params.form_factor},
             std::pair<const char*, std::string*>{"memory_kind", &def.params.memory_kind},
             std::pair<const char*, std::string*>{"memory_type", &def.params.memory_type}}) {
        if (auto it = parsed.scalars.find(key); it != parsed.scalars.end()) {
            *target = it->second;
        }
    }
    // <architecture>/<segment>/<id>.yaml, as the Python catalog reads it.
    if (blob.relative_path) {
        const std::string relative_path = blob.relative_path;
        const size_t first = relative_path.find('/');
        const size_t second = first == std::string::npos ? std::string::npos : relative_path.find('/', first + 1);
        if (second != std::string::npos) {
            def.params.segment = to_lower(relative_path.substr(first + 1, second - first - 1));
        }
    }

    auto types_it = parsed.lists.find("supported_types");
    if (types_it != parsed.lists.end()) {
        for (const std::string& type_str : types_it->second) {
            std::optional<GpuDataType> maybe_type = parse_data_type(type_str);
            if (!maybe_type.has_value()) {
                FGPU_LOG("[GpuProfile] Unknown data type '%s' in %s\n", type_str.c_str(), blob.filename);
                return std::nullopt;
            }
            def.params.supported_types.push_back(maybe_type.value());
        }
    }

    const GpuArch capability_arch = architecture_from_compute_capability(
        def.params.compute_major,
        def.params.compute_minor
    );
    if (capability_arch == GpuArch::Unknown || capability_arch != def.arch) {
        FGPU_LOG(
            "[GpuProfile] Architecture '%s' does not match compute capability %d.%d in %s\n",
            arch_it->second.c_str(),
            def.params.compute_major,
            def.params.compute_minor,
            blob.filename
        );
        return std::nullopt;
    }
    return def;
}

const std::vector<ProfileDefinition>& builtin_profile_definitions() {
    static const std::vector<ProfileDefinition> kDefinitions = []() {
        std::vector<ProfileDefinition> result;
        for (const ProfileYamlBlob& blob : builtin_profile_yamls()) {
            std::optional<ProfileDefinition> def = parse_definition(blob);
            if (def.has_value()) {
                result.push_back(def.value());
            }
        }
        return result;
    }();
    return kDefinitions;
}

std::optional<GpuProfile> profile_from_yaml_id(const std::string& id) {
    const std::string normalized = to_lower(id);
    for (const ProfileDefinition& def : builtin_profile_definitions()) {
        if (def.id == normalized) {
            return build_profile_from_definition(def);
        }
    }
    return std::nullopt;
}

} // namespace

std::vector<std::string> builtin_profile_ids() {
    std::vector<std::string> ids;
    for (const ProfileDefinition& def : builtin_profile_definitions()) {
        ids.push_back(def.id);
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids;
}

std::optional<GpuProfile> profile_from_preset_id(const std::string& id) {
    const std::string normalized = normalize_profile_id(id);
    if (normalized.empty()) return std::nullopt;

    return profile_from_yaml_id(normalized);
}

std::optional<GpuDataType> cuda_dtype_to_gpu_dtype(int cuda_data_type) {
    switch (cuda_data_type) {
        case 0:  return GpuDataType::FP32;   // CUDA_R_32F
        case 2:  return GpuDataType::FP16;   // CUDA_R_16F
        case 14: return GpuDataType::BF16;   // CUDA_R_16BF
        case 3:  return GpuDataType::INT8;   // CUDA_R_8I
        case 16: // CUDA_R_4I
        case 17: // CUDA_R_4U
        case 18: // CUDA_C_4I
        case 19: // CUDA_C_4U
            return GpuDataType::INT4;
        default:
            return std::nullopt;
    }
}

GpuProfile GpuProfile::A100() {
    // Every compiled YAML profile parses (tests/test_native_profile_attributes.py
    // checks the native catalog against the Python one), so a missing a100
    // means a broken build. Log it and return a zeroed profile rather than
    // abort inside the host process.
    if (std::optional<GpuProfile> profile = profile_from_yaml_id("a100"); profile.has_value()) {
        return profile.value();
    }
    FGPU_LOG("[GpuProfile] The compiled a100 profile is missing; the build is broken\n");
    GpuProfile profile;
    profile.id = "a100";
    profile.name = "a100";
    return profile;
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
        case GpuDataType::FP8: return "fp8";
        case GpuDataType::FP4: return "fp4";
        default: return "unknown";
    }
}

} // namespace fake_gpu
