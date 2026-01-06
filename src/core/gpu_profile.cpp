#include "gpu_profile.hpp"
#include "generated_profiles.hpp"
#include "logging.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <optional>
#include <sstream>
#include <unordered_map>
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
    if (const ArchProfile* arch_profile = get_arch_profile(def.arch)) {
        return arch_profile->build(def.params);
    }

    // Fallback for unknown architecture identifiers
    GpuProfile profile;
    profile.name = def.params.name;
    profile.architecture = def.arch;
    profile.compute_major = static_cast<int>(def.arch);
    profile.compute_minor = def.params.compute_minor;
    profile.memory_bytes = def.params.memory_bytes;
    profile.sm_count = def.params.sm_count;
    profile.memory_bus_width_bits = def.params.memory_bus_width_bits;
    profile.core_clock_mhz = def.params.core_clock_mhz;
    profile.memory_clock_mhz = def.params.memory_clock_mhz;
    profile.l2_cache_bytes = def.params.l2_cache_bytes;
    profile.shared_mem_per_sm = def.params.shared_mem_per_sm;
    profile.shared_mem_per_block = def.params.shared_mem_per_block;
    profile.shared_mem_per_block_optin = def.params.shared_mem_per_block_optin;
    profile.regs_per_block = def.params.regs_per_block;
    profile.regs_per_multiprocessor = def.params.regs_per_multiprocessor;
    profile.max_threads_per_multiprocessor = def.params.max_threads_per_multiprocessor;
    profile.max_blocks_per_multiprocessor = def.params.max_blocks_per_multiprocessor;
    profile.typical_power_usage_mw = def.params.typical_power_usage_mw;
    profile.max_power_limit_mw = def.params.max_power_limit_mw;
    profile.pci_device_id = def.params.pci_device_id;
    profile.supported_types = def.params.supported_types.empty() ? std::vector<GpuDataType>{GpuDataType::FP32} : def.params.supported_types;
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

    auto compute_minor_it = parsed.scalars.find("compute_minor");
    if (compute_minor_it != parsed.scalars.end()) {
        if (!parse_integer(compute_minor_it->second, def.params.compute_minor)) {
            FGPU_LOG("[GpuProfile] Invalid compute_minor in %s: %s\n", blob.filename, compute_minor_it->second.c_str());
            return std::nullopt;
        }
    }

    uint64_t numeric_value = 0;
    if (!parse_required_int(parsed, "memory_bytes", numeric_value, blob.filename)) return std::nullopt;
    def.params.memory_bytes = numeric_value;
    if (!parse_required_int(parsed, "sm_count", numeric_value, blob.filename)) return std::nullopt;
    def.params.sm_count = static_cast<int>(numeric_value);
    if (!parse_required_int(parsed, "memory_bus_width_bits", numeric_value, blob.filename)) return std::nullopt;
    def.params.memory_bus_width_bits = static_cast<int>(numeric_value);
    if (!parse_required_int(parsed, "core_clock_mhz", numeric_value, blob.filename)) return std::nullopt;
    def.params.core_clock_mhz = static_cast<int>(numeric_value);
    if (!parse_required_int(parsed, "memory_clock_mhz", numeric_value, blob.filename)) return std::nullopt;
    def.params.memory_clock_mhz = static_cast<int>(numeric_value);
    if (!parse_required_int(parsed, "l2_cache_bytes", numeric_value, blob.filename)) return std::nullopt;
    def.params.l2_cache_bytes = static_cast<int>(numeric_value);
    if (!parse_required_int(parsed, "shared_mem_per_sm", numeric_value, blob.filename)) return std::nullopt;
    def.params.shared_mem_per_sm = static_cast<int>(numeric_value);
    if (!parse_required_int(parsed, "shared_mem_per_block", numeric_value, blob.filename)) return std::nullopt;
    def.params.shared_mem_per_block = static_cast<int>(numeric_value);
    if (!parse_required_int(parsed, "shared_mem_per_block_optin", numeric_value, blob.filename)) return std::nullopt;
    def.params.shared_mem_per_block_optin = static_cast<int>(numeric_value);
    if (!parse_required_int(parsed, "regs_per_block", numeric_value, blob.filename)) return std::nullopt;
    def.params.regs_per_block = static_cast<int>(numeric_value);
    if (!parse_required_int(parsed, "regs_per_multiprocessor", numeric_value, blob.filename)) return std::nullopt;
    def.params.regs_per_multiprocessor = static_cast<int>(numeric_value);
    if (!parse_required_int(parsed, "max_threads_per_multiprocessor", numeric_value, blob.filename)) return std::nullopt;
    def.params.max_threads_per_multiprocessor = static_cast<int>(numeric_value);
    if (!parse_required_int(parsed, "max_blocks_per_multiprocessor", numeric_value, blob.filename)) return std::nullopt;
    def.params.max_blocks_per_multiprocessor = static_cast<int>(numeric_value);
    if (!parse_required_int(parsed, "typical_power_usage_mw", numeric_value, blob.filename)) return std::nullopt;
    def.params.typical_power_usage_mw = static_cast<unsigned int>(numeric_value);
    if (!parse_required_int(parsed, "max_power_limit_mw", numeric_value, blob.filename)) return std::nullopt;
    def.params.max_power_limit_mw = static_cast<unsigned int>(numeric_value);
    if (!parse_required_int(parsed, "pci_device_id", numeric_value, blob.filename)) return std::nullopt;
    def.params.pci_device_id = static_cast<uint32_t>(numeric_value);

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

GpuProfile build_profile_with_fallback(const std::string& id, GpuArch arch, const GpuProfileParams& params) {
    std::optional<GpuProfile> yaml_profile = profile_from_yaml_id(id);
    if (yaml_profile.has_value()) return yaml_profile.value();
    if (const ArchProfile* builder = get_arch_profile(arch)) {
        return builder->build(params);
    }
    ProfileDefinition fallback_def;
    fallback_def.arch = arch;
    fallback_def.params = params;
    return build_profile_from_definition(fallback_def);
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

    if (std::optional<GpuProfile> yaml_profile = profile_from_yaml_id(normalized); yaml_profile.has_value()) {
        return yaml_profile.value();
    }

    if (normalized == "gtx980") return GpuProfile::GTX980();
    if (normalized == "p100") return GpuProfile::P100();
    if (normalized == "v100") return GpuProfile::V100();
    if (normalized == "t4") return GpuProfile::T4();
    if (normalized == "a40") return GpuProfile::A40();
    if (normalized == "a100") return GpuProfile::A100();
    if (normalized == "h100") return GpuProfile::H100();
    if (normalized == "l40s") return GpuProfile::L40S();
    if (normalized == "b100") return GpuProfile::B100();
    if (normalized == "b200") return GpuProfile::B200();

    return std::nullopt;
}

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
    return build_profile_with_fallback("gtx980", GpuArch::Maxwell, params);
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
    return build_profile_with_fallback("p100", GpuArch::Pascal, params);
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
    return build_profile_with_fallback("v100", GpuArch::Volta, params);
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
    return build_profile_with_fallback("t4", GpuArch::Turing, params);
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
    return build_profile_with_fallback("a40", GpuArch::Ampere, params);
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
    return build_profile_with_fallback("a100", GpuArch::Ampere, params);
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
    return build_profile_with_fallback("h100", GpuArch::Hopper, params);
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
    return build_profile_with_fallback("l40s", GpuArch::Ada, params);
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
    return build_profile_with_fallback("b100", GpuArch::Blackwell, params);
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
    return build_profile_with_fallback("b200", GpuArch::Blackwell, params);
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
