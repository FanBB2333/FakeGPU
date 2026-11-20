#include "device.hpp"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <tuple>
#include <utility>

namespace fake_gpu {
namespace {

constexpr double kDefaultNvLinkBandwidthGbps = 900.0;
constexpr double kMaximumNvLinkBandwidthGbps = 1'000'000.0;
constexpr uint64_t kMebibyte = 1024ULL * 1024ULL;
constexpr std::size_t kMaximumMigProfileLength = 64;

std::string trim_copy(const std::string& value) {
    const std::size_t begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const std::size_t end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

bool parse_bandwidth(double& result) {
    const char* value = std::getenv("FAKEGPU_NVLINK_BANDWIDTH_GBPS");
    if (!value || !*value) {
        result = kDefaultNvLinkBandwidthGbps;
        return true;
    }
    const std::string trimmed = trim_copy(value);
    if (trimmed.empty()) {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const double parsed = std::strtod(trimmed.c_str(), &end);
    if (
        errno != 0 ||
        end == trimmed.c_str() ||
        !end ||
        *end != '\0' ||
        !std::isfinite(parsed) ||
        parsed <= 0.0 ||
        parsed > kMaximumNvLinkBandwidthGbps) {
        return false;
    }
    result = parsed;
    return true;
}

ModeledDeviceTopology invalid_topology(
    std::vector<ModeledSlot> slots,
    std::string platform,
    std::string error) {
    ModeledDeviceTopology topology;
    topology.source = "modeled_environment_invalid";
    topology.nvlink_source = "modeled_environment_invalid";
    topology.platform = std::move(platform);
    topology.configured = true;
    topology.valid = false;
    topology.error = std::move(error);
    topology.nvlink_peers.resize(slots.size());
    topology.slots = std::move(slots);
    return topology;
}

std::string lowercase_copy(std::string value);

struct TopologyError {
    std::string message;
};

std::vector<std::string> split_text(const std::string& text, char separator) {
    std::vector<std::string> parts;
    std::string part;
    std::istringstream stream(text);
    while (std::getline(stream, part, separator)) {
        parts.push_back(part);
    }
    if (!text.empty() && text.back() == separator) {
        parts.emplace_back();
    }
    return parts;
}

std::vector<std::string> trimmed_fields(const std::string& text) {
    std::vector<std::string> fields = split_text(text, ':');
    for (std::string& field : fields) {
        field = trim_copy(field);
    }
    return fields;
}

bool has_three_or_four_fields(const std::vector<std::string>& fields) {
    return fields.size() >= 3 && fields.size() <= 4 &&
        std::none_of(fields.begin(), fields.end(), [](const std::string& field) {
            return field.empty();
        });
}

bool all_digits(const std::string& text) {
    return !text.empty() && std::all_of(text.begin(), text.end(), [](unsigned char c) {
        return c >= '0' && c <= '9';
    });
}

// Mirrors the Python model: surrounding whitespace, an optional '+', then
// ASCII digits only.
bool parse_decimal(
    const std::string& value,
    uint64_t minimum,
    uint64_t maximum,
    uint64_t& result) {
    const char* const whitespace = " \t\r\n\v\f";
    const std::size_t begin = value.find_first_not_of(whitespace);
    if (begin == std::string::npos) {
        return false;
    }
    std::string digits =
        value.substr(begin, value.find_last_not_of(whitespace) - begin + 1);
    if (digits.front() == '+') {
        digits.erase(0, 1);
    }
    if (!all_digits(digits)) {
        return false;
    }
    errno = 0;
    const unsigned long long parsed = std::strtoull(digits.c_str(), nullptr, 10);
    if (errno == ERANGE || parsed < minimum || parsed > maximum) {
        return false;
    }
    result = static_cast<uint64_t>(parsed);
    return true;
}

bool parse_device_index(
    const std::string& value,
    std::size_t device_count,
    int& result) {
    uint64_t parsed = 0;
    if (!parse_decimal(value, 0, std::numeric_limits<uint64_t>::max(), parsed) ||
        parsed >= device_count) {
        return false;
    }
    result = static_cast<int>(parsed);
    return true;
}

unsigned int parse_unsigned(const std::string& text, const std::string& what) {
    if (!all_digits(text) || text.size() > 9) {
        throw TopologyError{what + " '" + text + "' is not a non-negative integer"};
    }
    return static_cast<unsigned int>(std::stoul(text));
}

double parse_positive(const std::string& text, const std::string& what) {
    errno = 0;
    char* end = nullptr;
    const double value = std::strtod(text.c_str(), &end);
    if (text.empty() || errno != 0 || !end || *end != '\0' || !std::isfinite(value) || value <= 0.0) {
        throw TopologyError{what + " '" + text + "' must be a positive number"};
    }
    return value;
}

double pcie_lane_gbps(unsigned int generation) {
    switch (generation) {
        case 1: return 0.25;
        case 2: return 0.5;
        case 3: return 0.984615;
        case 4: return 1.969231;
        case 5: return 3.938462;
        case 6: return 7.375;
        default: return 0.0;
    }
}

ModeledPcieLink parse_pcie_link(const std::string& text) {
    const std::string lower = lowercase_copy(trim_copy(text));
    if (lower == "none") {
        return {};
    }
    const std::size_t separator = lower.find('x');
    if (separator == std::string::npos ||
        !all_digits(lower.substr(0, separator)) ||
        !all_digits(lower.substr(separator + 1))) {
        throw TopologyError{"PCIe link '" + text + "' is not GENxWIDTH, e.g. 4x16"};
    }
    ModeledPcieLink link;
    link.generation = parse_unsigned(lower.substr(0, separator), "PCIe generation");
    link.width = parse_unsigned(lower.substr(separator + 1), "PCIe width");
    if (pcie_lane_gbps(link.generation) == 0.0) {
        throw TopologyError{"PCIe generation " + std::to_string(link.generation) + " is not one of [1, 2, 3, 4, 5, 6]"};
    }
    static const std::set<unsigned int> widths{1, 2, 4, 8, 16, 32};
    if (!widths.count(link.width)) {
        throw TopologyError{"PCIe width x" + std::to_string(link.width) + " is not one of x1, x2, x4, x8, x16, x32"};
    }
    return link;
}

ModeledSlot parse_slot(const std::string& text) {
    ModeledSlot slot;
    static const std::set<std::string> types{"sxm", "pcie", "superchip", "laptop", "embedded", "any"};
    static const std::set<std::string> host_links{"pcie", "c2c", "unified"};
    for (const std::string& raw : split_text(text, ',')) {
        const std::string item = trim_copy(raw);
        const std::size_t equals = item.find('=');
        if (equals == std::string::npos) {
            throw TopologyError{"slot field '" + item + "' is not key=value"};
        }
        const std::string key = trim_copy(item.substr(0, equals));
        const std::string value = trim_copy(item.substr(equals + 1));
        if (key == "type") {
            if (!types.count(value)) {
                throw TopologyError{"slot type '" + value + "' is not one of sxm, pcie, superchip, laptop, embedded, any"};
            }
            slot.type = value;
        } else if (key == "pcie") {
            slot.pcie = parse_pcie_link(value);
        } else if (key == "gpu") {
            slot.gpu_pcie = parse_pcie_link(value);
        } else if (key == "host") {
            slot.host_pcie = parse_pcie_link(value);
        } else if (key == "link") {
            if (!host_links.count(value)) {
                throw TopologyError{"slot link '" + value + "' is not one of pcie, c2c, unified"};
            }
            slot.host_link = value;
        } else if (key == "hostbw") {
            slot.host_gbps = parse_positive(value, "slot hostbw");
        } else if (key == "numa") {
            slot.numa = parse_unsigned(value, "slot numa");
        } else if (key == "bridge") {
            slot.bridge = parse_unsigned(value, "slot bridge");
        } else if (key == "switch") {
            for (const std::string& part : split_text(value, '.')) {
                if (!value.empty() && !all_digits(part)) {
                    throw TopologyError{"slot switch '" + value + "' is not a dotted switch path such as 0 or 1.2"};
                }
            }
            slot.pcie_switch = value;
        } else if (key == "p2p") {
            if (value != "0" && value != "1") {
                throw TopologyError{"slot p2p '" + value + "' is not 0 or 1"};
            }
            slot.p2p = value == "1";
        } else {
            throw TopologyError{"unknown slot field '" + key + "'"};
        }
    }
    return slot;
}

std::vector<int> parse_members(const std::string& text) {
    std::vector<int> members;
    for (const std::string& raw : split_text(text, ',')) {
        const std::string item = trim_copy(raw);
        const std::size_t dash = item.find('-');
        const std::string first = item.substr(0, dash);
        const std::string last = dash == std::string::npos ? first : item.substr(dash + 1);
        if (!all_digits(first) || !all_digits(last) || first.size() > 6 || last.size() > 6) {
            throw TopologyError{"NVLink member '" + item + "' is not a GPU index or range"};
        }
        const int start = std::stoi(first);
        const int stop = std::stoi(last);
        if (stop < start) {
            throw TopologyError{"NVLink member range '" + item + "' is reversed"};
        }
        for (int member = start; member <= stop; ++member) {
            members.push_back(member);
        }
    }
    std::vector<int> sorted = members;
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
        throw TopologyError{"NVLink members '" + text + "' repeat a GPU"};
    }
    return members;
}

ModeledNvLinkGroup parse_nvlink_group(const std::string& text) {
    const std::vector<std::string> fields = split_text(trim_copy(text), ':');
    if (fields.size() != 4) {
        throw TopologyError{"NVLink group '" + trim_copy(text) + "' is not KIND:MEMBERS:LINKS:GBPS"};
    }
    const std::string kind = trim_copy(fields[0]);
    if (kind != "mesh" && kind != "switch") {
        throw TopologyError{"NVLink group kind '" + kind + "' is not mesh or switch"};
    }
    ModeledNvLinkGroup group;
    group.nvswitch = kind == "switch";
    group.members = parse_members(fields[1]);
    if (group.members.size() < 2) {
        throw TopologyError{"NVLink group '" + trim_copy(text) + "' needs at least two GPUs"};
    }
    const std::string links = trim_copy(fields[2]);
    if (!all_digits(links) || links.size() > 3 || std::stoul(links) < 1 ||
        std::stoul(links) > kMaximumModeledNvLinksPerDevice) {
        throw TopologyError{"NVLink group '" + trim_copy(text) + "' needs 1 to 18 links"};
    }
    group.links = static_cast<unsigned int>(std::stoul(links));
    group.link_gbps = parse_positive(trim_copy(fields[3]), "NVLink GBPS");
    return group;
}

std::string encode_nvlink_group(const ModeledNvLinkGroup& group) {
    std::string text = group.nvswitch ? "switch:" : "mesh:";
    for (std::size_t position = 0; position < group.members.size(); ++position) {
        text += (position ? "," : "") + std::to_string(group.members[position]);
    }
    char gbps[32];
    std::snprintf(gbps, sizeof(gbps), "%g", group.link_gbps);
    return text + ":" + std::to_string(group.links) + ":" + gbps;
}

std::vector<ModeledNvLinkGroup> legacy_nvlink_groups(
    const std::string& groups_value,
    std::size_t device_count,
    double& bandwidth) {
    if (!parse_bandwidth(bandwidth)) {
        throw TopologyError{
            "FAKEGPU_NVLINK_BANDWIDTH_GBPS must be a finite number "
            "greater than 0 and no greater than 1000000"};
    }
    std::set<std::pair<int, int>> links;
    const std::string groups_text = trim_copy(groups_value);
    if (groups_text.empty() || groups_text.front() == ';' || groups_text.back() == ';') {
        throw TopologyError{"FAKEGPU_NVLINK_GROUPS contains an empty group"};
    }
    std::istringstream groups(groups_text);
    std::string group_text;
    std::size_t group_index = 0;
    while (std::getline(groups, group_text, ';')) {
        ++group_index;
        group_text = trim_copy(group_text);
        if (group_text.empty()) {
            throw TopologyError{"FAKEGPU_NVLINK_GROUPS contains an empty group"};
        }
        const std::string invalid =
            "FAKEGPU_NVLINK_GROUPS group " + std::to_string(group_index) + " contains an invalid device index";
        if (group_text.front() == ',' || group_text.back() == ',') {
            throw TopologyError{invalid};
        }
        std::vector<int> members;
        std::istringstream group(group_text);
        std::string member_text;
        while (std::getline(group, member_text, ',')) {
            int member = -1;
            if (!parse_device_index(member_text, device_count, member)) {
                throw TopologyError{invalid};
            }
            members.push_back(member);
        }
        std::sort(members.begin(), members.end());
        members.erase(std::unique(members.begin(), members.end()), members.end());
        if (members.size() < 2) {
            throw TopologyError{
                "each FAKEGPU_NVLINK_GROUPS group must contain at least two distinct device indices"};
        }
        if (members.size() - 1 > kMaximumModeledNvLinksPerDevice) {
            throw TopologyError{"an FAKEGPU_NVLINK_GROUPS group exceeds the 18-link per-device limit"};
        }
        for (std::size_t left = 0; left < members.size(); ++left) {
            for (std::size_t right = left + 1; right < members.size(); ++right) {
                links.emplace(members[left], members[right]);
            }
        }
    }
    if (links.empty()) {
        throw TopologyError{"FAKEGPU_NVLINK_GROUPS did not define any links"};
    }
    std::vector<ModeledNvLinkGroup> result;
    for (const auto& [left, right] : links) {
        result.push_back({false, {left, right}, 1, bandwidth});
    }
    return result;
}

std::string placement_relation(const ModeledSlot& left, const ModeledSlot& right) {
    const std::string& left_switch = left.pcie_switch;
    const std::string& right_switch = right.pcie_switch;
    if (!left_switch.empty() && !right_switch.empty()) {
        if (left_switch == right_switch) {
            return "PIX";
        }
        const std::size_t dot = left_switch.find('.');
        if (dot != std::string::npos &&
            left_switch.substr(0, dot) == right_switch.substr(0, right_switch.find('.'))) {
            return "PXB";
        }
    }
    if (left.numa != right.numa) {
        return "SYS";
    }
    return left.bridge == right.bridge ? "PHB" : "NODE";
}

double round3(double value) {
    return std::round(value * 1000.0) / 1000.0;
}

std::string lowercase_copy(std::string value) {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
    return value;
}

std::string uppercase_copy(std::string value) {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char character) {
            return static_cast<char>(std::toupper(character));
        });
    return value;
}

bool valid_fault_code(const std::string& value) {
    if (value.empty() || value.size() > 64) {
        return false;
    }
    return std::all_of(
        value.begin(),
        value.end(),
        [](unsigned char character) {
            return std::isalnum(character) ||
                character == '_' ||
                character == '-' ||
                character == '.';
        });
}

ModeledFaultModel invalid_fault_model(std::string error) {
    ModeledFaultModel model;
    model.source = "modeled_environment_invalid";
    model.configured = true;
    model.valid = false;
    model.error = std::move(error);
    return model;
}

ModeledMigLayout invalid_mig_layout(std::string error) {
    ModeledMigLayout layout;
    layout.source = "modeled_environment_invalid";
    layout.configured = true;
    layout.valid = false;
    layout.error = std::move(error);
    return layout;
}

bool parse_mig_profile(
    const std::string& value,
    std::string& profile,
    unsigned int& slice_count) {
    profile = lowercase_copy(trim_copy(value));
    if (
        profile.empty() ||
        profile.size() > kMaximumMigProfileLength ||
        !std::all_of(
            profile.begin(),
            profile.end(),
            [](unsigned char character) {
                return std::isalnum(character) ||
                    character == '.' ||
                    character == '_' ||
                    character == '-' ||
                    character == '+';
            })) {
        return false;
    }
    const std::size_t separator = profile.find("g.");
    if (separator == std::string::npos || separator == 0) {
        return false;
    }
    uint64_t parsed_slices = 0;
    if (
        !parse_decimal(
            profile.substr(0, separator),
            1,
            kMaximumModeledMigSlicesPerDevice,
            parsed_slices)) {
        return false;
    }
    slice_count = static_cast<unsigned int>(parsed_slices);
    return true;
}

std::string modeled_mig_uuid(
    const Device& parent,
    unsigned int gpu_instance_id,
    unsigned int compute_instance_id) {
    return synthetic_mig_uuid(parent.uuid, gpu_instance_id, compute_instance_id);
}

// SHA-1 (FIPS 180-4), only for the RFC 4122 name-based UUIDs below.
std::array<unsigned char, 20> sha1(const std::string& message) {
    uint32_t h[5] = {0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u};
    std::string data = message;
    const uint64_t bit_length = static_cast<uint64_t>(message.size()) * 8;
    data.push_back(static_cast<char>(0x80));
    while (data.size() % 64 != 56) data.push_back('\0');
    for (int shift = 56; shift >= 0; shift -= 8) {
        data.push_back(static_cast<char>((bit_length >> shift) & 0xFF));
    }
    auto rotl = [](uint32_t value, int bits) { return (value << bits) | (value >> (32 - bits)); };
    for (std::size_t chunk = 0; chunk < data.size(); chunk += 64) {
        uint32_t w[80];
        for (int i = 0; i < 16; ++i) {
            const auto* p = reinterpret_cast<const unsigned char*>(data.data() + chunk + 4 * i);
            w[i] = (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | uint32_t(p[3]);
        }
        for (int i = 16; i < 80; ++i) w[i] = rotl(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
        for (int i = 0; i < 80; ++i) {
            uint32_t f, k;
            if (i < 20) { f = (b & c) | (~b & d); k = 0x5A827999u; }
            else if (i < 40) { f = b ^ c ^ d; k = 0x6ED9EBA1u; }
            else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDCu; }
            else { f = b ^ c ^ d; k = 0xCA62C1D6u; }
            const uint32_t temp = rotl(a, 5) + f + e + k + w[i];
            e = d; d = c; c = rotl(b, 30); b = a; a = temp;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e;
    }
    std::array<unsigned char, 20> digest{};
    for (int i = 0; i < 20; ++i) digest[i] = static_cast<unsigned char>((h[i / 4] >> (24 - 8 * (i % 4))) & 0xFF);
    return digest;
}

// Python's uuid.uuid5(uuid.NAMESPACE_DNS, name), formatted 8-4-4-4-12.
std::string uuid5_dns(const std::string& name) {
    static const unsigned char kNamespaceDns[16] = {
        0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1,
        0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8};
    std::array<unsigned char, 20> digest =
        sha1(std::string(reinterpret_cast<const char*>(kNamespaceDns), 16) + name);
    digest[6] = static_cast<unsigned char>((digest[6] & 0x0F) | 0x50);
    digest[8] = static_cast<unsigned char>((digest[8] & 0x3F) | 0x80);
    char text[37];
    std::snprintf(
        text, sizeof(text),
        "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
        digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6], digest[7],
        digest[8], digest[9], digest[10], digest[11], digest[12], digest[13], digest[14], digest[15]);
    return text;
}

} // namespace

std::string synthetic_gpu_uuid(const std::string& host, int index, const std::string& profile_id) {
    return "GPU-" + uuid5_dns("fakegpu:" + host + ":" + std::to_string(index) + ":" + profile_id);
}

std::string synthetic_mig_uuid(
    const std::string& parent_uuid,
    unsigned int gpu_instance_id,
    unsigned int compute_instance_id) {
    return "MIG-" + uuid5_dns(
        "fakegpu:mig:" + parent_uuid + ":" + std::to_string(gpu_instance_id) + ":" +
        std::to_string(compute_instance_id));
}

std::string synthetic_pci_bus_id(int index) {
    const int safe_index = std::max(index, 0);
    char text[32];
    std::snprintf(text, sizeof(text), "%08X:%02X:00.0", safe_index / 255, safe_index % 255 + 1);
    return text;
}

bool uuid_string_bytes(const std::string& uuid, unsigned char bytes[16]) {
    if (uuid.size() != 40 || (uuid.compare(0, 4, "GPU-") != 0 && uuid.compare(0, 4, "MIG-") != 0)) {
        return false;
    }
    std::string hex;
    for (std::size_t i = 4; i < uuid.size(); ++i) {
        if (uuid[i] == '-') continue;
        if (!std::isxdigit(static_cast<unsigned char>(uuid[i]))) return false;
        hex.push_back(uuid[i]);
    }
    if (hex.size() != 32) return false;
    for (int i = 0; i < 16; ++i) {
        bytes[i] = static_cast<unsigned char>(std::stoul(hex.substr(2 * i, 2), nullptr, 16));
    }
    return true;
}

const std::string& local_hostname() {
    static const std::string host = []() {
        char buffer[256] = {};
        if (gethostname(buffer, sizeof(buffer) - 1) == 0 && buffer[0] != '\0') return std::string(buffer);
        return std::string("localhost");
    }();
    return host;
}

bool parse_pci_address(const std::string& text, PciAddress& address) {
    unsigned int a = 0, b = 0, c = 0, d = 0;
    int end = -1;
    const int length = static_cast<int>(text.size());
    if (std::sscanf(text.c_str(), "%x:%x:%x.%x%n", &a, &b, &c, &d, &end) == 4 && end == length) {
        address = {a, b, c, d};
        return true;
    }
    end = -1;
    if (std::sscanf(text.c_str(), "%x:%x:%x%n", &a, &b, &c, &end) == 3 && end == length) {
        address = {a, b, c, 0};
        return true;
    }
    end = -1;
    if (std::sscanf(text.c_str(), "%x:%x.%x%n", &a, &b, &c, &end) == 3 && end == length) {
        address = {0, a, b, c};
        return true;
    }
    return false;
}

std::string cuda_pci_bus_id(const PciAddress& address) {
    char text[48];
    std::snprintf(text, sizeof(text), "%04X:%02X:%02X.%X", address.domain, address.bus, address.device,
                  address.function);
    return text;
}

Device::Device(int idx, const GpuProfile& profile)
    : index(idx), profile(profile), used_memory(0), used_memory_peak(0) {
    name = profile.name;
    uuid = synthetic_gpu_uuid(local_hostname(), idx, profile.id);
    total_memory = profile.memory_bytes;
    pci_bus_id = synthetic_pci_bus_id(idx);
}

double ModeledPcieLink::bandwidth_gbps() const {
    return present() ? round3(pcie_lane_gbps(generation) * width) : 0.0;
}

ModeledPcieLink ModeledSlot::gpu_max_pcie() const {
    return gpu_pcie.present() ? gpu_pcie : pcie;
}

double ModeledSlot::host_bandwidth_gbps() const {
    if (host_gbps > 0.0) {
        return host_gbps;
    }
    if (host_link != "pcie" || !pcie.present()) {
        return 0.0;
    }
    double bandwidth = pcie.bandwidth_gbps();
    if (host_pcie.present()) {
        bandwidth = std::min(bandwidth, host_pcie.bandwidth_gbps());
    }
    return bandwidth;
}

const ModeledSlot* ModeledDeviceTopology::slot(int index) const {
    if (index < 0 || static_cast<std::size_t>(index) >= slots.size()) {
        return nullptr;
    }
    return &slots[static_cast<std::size_t>(index)];
}

std::vector<ModeledNvLinkPeerSummary> ModeledDeviceTopology::peers(int index) const {
    std::map<int, ModeledNvLinkPeerSummary> found;
    auto add = [&found](int peer, unsigned int links, const char* via, double bandwidth, unsigned int first) {
        auto it = found.find(peer);
        if (it == found.end()) {
            found[peer] = {peer, links, bandwidth, first, via};
            return;
        }
        if (it->second.via != via) {
            it->second.via = "direct+nvswitch";
        }
        it->second.links += links;
        it->second.bandwidth_gbps += bandwidth;
        it->second.first_link = std::min(it->second.first_link, first);
    };
    if (index < 0 || static_cast<std::size_t>(index) >= nvlink_peers.size()) {
        return {};
    }
    const auto& ports = nvlink_peers[static_cast<std::size_t>(index)];
    int first_switch_port = -1;
    for (const auto& port : ports) {
        if (port.peer_index >= 0) {
            add(port.peer_index, 1, "direct", port.bandwidth_gbps, port.link);
        } else if (first_switch_port < 0) {
            first_switch_port = static_cast<int>(port.link);
        }
    }
    if (first_switch_port >= 0) {
        auto switch_links = [this](int member) {
            unsigned int total = 0;
            for (const auto& group : nvlink_groups) {
                if (group.nvswitch &&
                    std::find(group.members.begin(), group.members.end(), member) != group.members.end()) {
                    total += group.links;
                }
            }
            return total;
        };
        for (const auto& group : nvlink_groups) {
            if (!group.nvswitch ||
                std::find(group.members.begin(), group.members.end(), index) == group.members.end()) {
                continue;
            }
            for (int member : group.members) {
                if (member == index) continue;
                const unsigned int links = std::min(group.links, switch_links(member));
                add(member, links, "nvswitch", links * group.link_gbps,
                    static_cast<unsigned int>(first_switch_port));
            }
        }
    }
    std::vector<ModeledNvLinkPeerSummary> result;
    for (const auto& [peer, summary] : found) {
        result.push_back(summary);
    }
    return result;
}

unsigned int ModeledDeviceTopology::nvlink_links(int source, int target) const {
    for (const auto& peer : peers(source)) {
        if (peer.index == target) return peer.links;
    }
    return 0;
}

std::string ModeledDeviceTopology::pcie_relation(int source, int target) const {
    if (source == target) return "X";
    const ModeledSlot* left = slot(source);
    const ModeledSlot* right = slot(target);
    if (!left || !right) return "SYS";
    return placement_relation(*left, *right);
}

std::string ModeledDeviceTopology::relation(int source, int target) const {
    if (source == target) return "X";
    const unsigned int links = nvlink_links(source, target);
    return links ? "NV" + std::to_string(links) : pcie_relation(source, target);
}

bool ModeledDeviceTopology::can_access_peer(int source, int target) const {
    if (source == target) return false;
    if (nvlink_links(source, target)) return true;
    const ModeledSlot* left = slot(source);
    const ModeledSlot* right = slot(target);
    if (!left || !right || !left->p2p || !right->p2p || !left->pcie.present() || !right->pcie.present()) {
        return false;
    }
    const std::string path = pcie_relation(source, target);
    return path == "PIX" || path == "PXB" || path == "PHB" || path == "NODE";
}

double ModeledDeviceTopology::peer_bandwidth_gbps(int source, int target) const {
    for (const auto& peer : peers(source)) {
        if (peer.index == target) return peer.bandwidth_gbps;
    }
    const ModeledSlot* left = slot(source);
    const ModeledSlot* right = slot(target);
    if (!left || !right || !left->pcie.present() || !right->pcie.present()) return 0.0;
    if (pcie_relation(source, target) == "PIX") {
        return std::min(left->pcie.bandwidth_gbps(), right->pcie.bandwidth_gbps());
    }
    const double a = left->host_bandwidth_gbps();
    const double b = right->host_bandwidth_gbps();
    if (a > 0.0 && b > 0.0) return std::min(a, b);
    return std::max(a, b);
}

double ModeledDeviceTopology::host_bandwidth_gbps(int index) const {
    const ModeledSlot* found = slot(index);
    return found ? found->host_bandwidth_gbps() : 0.0;
}

ModeledDeviceTopology build_modeled_device_topology(
    const std::vector<Device>& devices) {
    const std::size_t device_count = devices.size();
    std::vector<ModeledSlot> defaults(device_count);
    for (std::size_t index = 0; index < device_count; ++index) {
        const GpuProfile& profile = devices[index].profile;
        defaults[index].pcie.generation = profile.pcie_generation > 0
            ? static_cast<unsigned int>(profile.pcie_generation) : 4U;
        defaults[index].pcie.width = profile.pcie_lanes > 0
            ? static_cast<unsigned int>(profile.pcie_lanes) : 16U;
        if (pcie_lane_gbps(defaults[index].pcie.generation) == 0.0) {
            defaults[index].pcie.generation = 4;
        }
    }

    ModeledDeviceTopology topology;
    topology.slots = defaults;
    topology.nvlink_peers.resize(device_count);
    auto env_text = [](const char* name) {
        const char* value = std::getenv(name);
        return value ? trim_copy(value) : std::string();
    };
    const std::string slots_text = env_text("FAKEGPU_SLOTS");
    const std::string fabric_text = env_text("FAKEGPU_NVLINK_FABRIC");
    const char* groups_value = std::getenv("FAKEGPU_NVLINK_GROUPS");
    const bool legacy = groups_value && *groups_value;
    const std::string platform = env_text("FAKEGPU_PLATFORM");
    if (slots_text.empty() && fabric_text.empty() && !legacy) {
        if (!platform.empty()) {
            return invalid_topology(
                defaults,
                platform,
                "FAKEGPU_PLATFORM needs FAKEGPU_SLOTS in native processes; start them with "
                "'fakegpu run --platform' or export 'fakegpu platform env'");
        }
        return topology;
    }

    topology.platform = platform;
    topology.configured = true;
    topology.source = !platform.empty() && (!slots_text.empty() || !fabric_text.empty())
        ? "platform"
        : "modeled_environment";
    try {
        if (!slots_text.empty()) {
            const std::vector<std::string> entries = split_text(slots_text, ';');
            std::size_t described = 0;
            bool empty_entry = false;
            for (const auto& entry : entries) {
                if (trim_copy(entry).empty()) {
                    empty_entry = true;
                } else {
                    ++described;
                }
            }
            if (entries.size() < device_count || empty_entry) {
                throw TopologyError{
                    "FAKEGPU_SLOTS describes " + std::to_string(described) +
                    " GPUs but the process has " + std::to_string(device_count)};
            }
            for (std::size_t index = 0; index < device_count; ++index) {
                topology.slots[index] = parse_slot(entries[index]);
            }
        }
        if (!fabric_text.empty()) {
            for (const auto& entry : split_text(fabric_text, ';')) {
                topology.nvlink_groups.push_back(parse_nvlink_group(entry));
            }
            topology.nvlink_source = topology.source;
        } else if (legacy) {
            topology.nvlink_groups = legacy_nvlink_groups(
                groups_value, device_count, topology.nvlink_bandwidth_gbps);
            topology.nvlink_source = "modeled_environment";
        }
        for (const auto& group : topology.nvlink_groups) {
            for (int member : group.members) {
                if (member < 0 || static_cast<std::size_t>(member) >= device_count) {
                    throw TopologyError{
                        "NVLink group " + encode_nvlink_group(group) + " names GPU " +
                        std::to_string(member) + ", beyond the " +
                        std::to_string(device_count) + " GPUs"};
                }
            }
        }
        for (const auto& group : topology.nvlink_groups) {
            for (int member : group.members) {
                auto& ports = topology.nvlink_peers[static_cast<std::size_t>(member)];
                if (group.nvswitch) {
                    for (unsigned int link = 0; link < group.links; ++link) {
                        ports.push_back({static_cast<unsigned int>(ports.size()), -1, group.link_gbps});
                    }
                    continue;
                }
                for (int peer : group.members) {
                    if (peer == member) continue;
                    for (unsigned int link = 0; link < group.links; ++link) {
                        ports.push_back({static_cast<unsigned int>(ports.size()), peer, group.link_gbps});
                    }
                }
            }
        }
        for (std::size_t index = 0; index < device_count; ++index) {
            if (topology.nvlink_peers[index].size() > kMaximumModeledNvLinksPerDevice) {
                throw TopologyError{
                    fabric_text.empty()
                        ? std::string("a device belongs to more than 18 modeled NVLink peer relationships")
                        : "GPU " + std::to_string(index) + " has more than 18 NVLinks"};
            }
        }
        if (!fabric_text.empty() && !topology.nvlink_groups.empty()) {
            topology.nvlink_bandwidth_gbps = topology.nvlink_groups.front().link_gbps;
        }
    } catch (const TopologyError& error) {
        return invalid_topology(defaults, platform, error.message);
    }
    return topology;
}

int modeled_fault_severity_rank(const std::string& severity) {
    if (severity == "critical") return 4;
    if (severity == "error") return 3;
    if (severity == "warning") return 2;
    if (severity == "info") return 1;
    return 0;
}

ModeledFaultModel build_modeled_fault_model(
    std::size_t device_count) {
    ModeledFaultModel model;
    const char* configured_events = std::getenv(
        "FAKEGPU_FAULT_EVENTS");
    if (!configured_events || !*configured_events) {
        return model;
    }

    const std::string events_text = trim_copy(configured_events);
    if (
        events_text.empty() ||
        events_text.front() == ';' ||
        events_text.back() == ';') {
        return invalid_fault_model(
            "FAKEGPU_FAULT_EVENTS contains an empty event");
    }

    model.source = "modeled_environment";
    model.configured = true;
    std::map<
        std::tuple<int, std::string, std::string>,
        uint64_t> counts;
    std::size_t event_index = 0;
    for (const std::string& entry : split_text(events_text, ';')) {
        const std::string prefix =
            "FAKEGPU_FAULT_EVENTS event " +
            std::to_string(++event_index);
        const std::string event_text = trim_copy(entry);
        if (
            event_text.empty() ||
            event_text.front() == ':' ||
            event_text.back() == ':') {
            return invalid_fault_model(
                prefix + " must use DEVICE:CODE:SEVERITY[:COUNT]");
        }

        const std::vector<std::string> fields = trimmed_fields(event_text);
        if (!has_three_or_four_fields(fields)) {
            return invalid_fault_model(
                prefix + " must use DEVICE:CODE:SEVERITY[:COUNT]");
        }

        int device_index = -1;
        if (!parse_device_index(
                fields[0],
                device_count,
                device_index)) {
            return invalid_fault_model(
                prefix + " contains an invalid device index");
        }
        const std::string code = uppercase_copy(fields[1]);
        if (!valid_fault_code(code)) {
            return invalid_fault_model(
                prefix + " contains an invalid code");
        }
        const std::string severity = lowercase_copy(fields[2]);
        if (modeled_fault_severity_rank(severity) == 0) {
            return invalid_fault_model(
                prefix +
                " severity must be info, warning, error, or critical");
        }
        uint64_t count = 1;
        if (
            fields.size() == 4 &&
            !parse_decimal(
                fields[3],
                1,
                kMaximumModeledFaultCount,
                count)) {
            return invalid_fault_model(
                prefix + " count must be between 1 and 1000000000");
        }

        const auto key = std::make_tuple(
            device_index,
            code,
            severity);
        uint64_t& aggregate = counts[key];
        aggregate = aggregate >= kMaximumModeledFaultCount - count
            ? kMaximumModeledFaultCount
            : aggregate + count;
        if (counts.size() > kMaximumModeledFaultTypes) {
            return invalid_fault_model(
                "FAKEGPU_FAULT_EVENTS exceeds the 128-event-type "
                "limit");
        }
    }

    for (const auto& [key, count] : counts) {
        const auto& [device_index, code, severity] = key;
        model.events.push_back(
            {device_index, code, severity, count});
    }
    std::sort(
        model.events.begin(),
        model.events.end(),
        [](const ModeledFaultEvent& left,
           const ModeledFaultEvent& right) {
            const int left_rank =
                modeled_fault_severity_rank(left.severity);
            const int right_rank =
                modeled_fault_severity_rank(right.severity);
            if (left_rank != right_rank) {
                return left_rank > right_rank;
            }
            if (left.device_index != right.device_index) {
                return left.device_index < right.device_index;
            }
            return left.code < right.code;
        });
    return model;
}

ModeledMigLayout build_modeled_mig_layout(
    const std::vector<Device>& devices) {
    ModeledMigLayout layout;
    const char* configured_layout =
        std::getenv("FAKEGPU_MIG_LAYOUT");
    if (!configured_layout || !*configured_layout) {
        return layout;
    }

    const std::string layout_text = trim_copy(configured_layout);
    if (
        layout_text.empty() ||
        layout_text.front() == ';' ||
        layout_text.back() == ';') {
        return invalid_mig_layout(
            "FAKEGPU_MIG_LAYOUT contains an empty instance entry");
    }

    layout.source = "modeled_environment";
    layout.configured = true;
    std::vector<uint64_t> allocated_memory(devices.size(), 0);
    std::vector<std::size_t> allocated_slices(devices.size(), 0);
    std::vector<unsigned int> instance_counts(devices.size(), 0);

    std::size_t entry_index = 0;
    for (const std::string& entry : split_text(layout_text, ';')) {
        const std::string prefix =
            "FAKEGPU_MIG_LAYOUT entry " +
            std::to_string(++entry_index);
        const std::string entry_text = trim_copy(entry);
        if (
            entry_text.empty() ||
            entry_text.front() == ':' ||
            entry_text.back() == ':') {
            return invalid_mig_layout(
                prefix +
                " must use DEVICE:PROFILE:MEMORY_MIB[:COUNT]");
        }

        const std::vector<std::string> fields = trimmed_fields(entry_text);
        if (!has_three_or_four_fields(fields)) {
            return invalid_mig_layout(
                prefix +
                " must use DEVICE:PROFILE:MEMORY_MIB[:COUNT]");
        }

        int device_index = -1;
        if (!parse_device_index(
                fields[0],
                devices.size(),
                device_index)) {
            return invalid_mig_layout(
                prefix + " contains an invalid device index");
        }

        std::string profile;
        unsigned int slice_count = 0;
        if (!parse_mig_profile(
                fields[1],
                profile,
                slice_count)) {
            return invalid_mig_layout(
                prefix +
                " contains an invalid MIG profile; expected a name "
                "such as 1g.10gb");
        }

        const Device& parent =
            devices[static_cast<std::size_t>(device_index)];
        const uint64_t maximum_mib =
            parent.total_memory / kMebibyte;
        uint64_t memory_mib = 0;
        if (!parse_decimal(
                fields[2],
                1,
                maximum_mib,
                memory_mib)) {
            return invalid_mig_layout(
                prefix +
                " memory must be a positive MiB value no greater "
                "than the parent device memory");
        }
        const uint64_t memory_bytes = memory_mib * kMebibyte;

        uint64_t count = 1;
        if (
            fields.size() == 4 &&
            !parse_decimal(
                fields[3],
                1,
                kMaximumModeledMigInstancesPerDevice,
                count)) {
            return invalid_mig_layout(
                prefix + " count must be between 1 and 8");
        }

        const std::size_t parent_index =
            static_cast<std::size_t>(device_index);
        if (
            count >
            kMaximumModeledMigInstancesPerDevice -
                instance_counts[parent_index]) {
            return invalid_mig_layout(
                prefix +
                " exceeds the 8-instance per-device limit");
        }
        if (
            count >
            (
                kMaximumModeledMigSlicesPerDevice -
                allocated_slices[parent_index]
            ) / slice_count) {
            return invalid_mig_layout(
                prefix +
                " exceeds the 8-slice per-device limit");
        }
        if (
            count >
            (
                parent.total_memory -
                allocated_memory[parent_index]
            ) / memory_bytes) {
            return invalid_mig_layout(
                prefix +
                " allocates more memory than the parent device");
        }

        for (uint64_t offset = 0; offset < count; ++offset) {
            const unsigned int instance_index =
                instance_counts[parent_index]++;
            layout.instances.push_back(
                {
                    device_index,
                    instance_index,
                    instance_index,
                    instance_index,
                    slice_count,
                    profile,
                    modeled_mig_uuid(
                        parent,
                        instance_index,
                        instance_index),
                    memory_bytes,
                });
        }
        allocated_slices[parent_index] +=
            static_cast<std::size_t>(count) * slice_count;
        allocated_memory[parent_index] += count * memory_bytes;
    }

    if (layout.instances.empty()) {
        return invalid_mig_layout(
            "FAKEGPU_MIG_LAYOUT did not define any MIG instances");
    }
    return layout;
}

} // namespace fake_gpu
