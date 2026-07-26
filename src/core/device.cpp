#include "device.hpp"

#include <algorithm>
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

std::string trim_copy(const std::string& value) {
    const std::size_t begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const std::size_t end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

bool parse_device_index(
    const std::string& value,
    std::size_t device_count,
    int& result) {
    const std::string trimmed = trim_copy(value);
    if (trimmed.empty()) {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(trimmed.c_str(), &end, 10);
    if (
        errno != 0 ||
        end == trimmed.c_str() ||
        !end ||
        *end != '\0' ||
        parsed < 0 ||
        static_cast<unsigned long>(parsed) >= device_count ||
        parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    result = static_cast<int>(parsed);
    return true;
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
    std::size_t device_count,
    std::string error) {
    ModeledDeviceTopology topology;
    topology.source = "modeled_environment_invalid";
    topology.configured = true;
    topology.valid = false;
    topology.error = std::move(error);
    topology.nvlink_peers.resize(device_count);
    return topology;
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

bool parse_fault_count(const std::string& value, uint64_t& result) {
    const std::string trimmed = trim_copy(value);
    if (trimmed.empty()) {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed =
        std::strtoull(trimmed.c_str(), &end, 10);
    if (
        errno != 0 ||
        end == trimmed.c_str() ||
        !end ||
        *end != '\0' ||
        parsed == 0 ||
        parsed > kMaximumModeledFaultCount) {
        return false;
    }
    result = static_cast<uint64_t>(parsed);
    return true;
}

ModeledFaultModel invalid_fault_model(std::string error) {
    ModeledFaultModel model;
    model.source = "modeled_environment_invalid";
    model.configured = true;
    model.valid = false;
    model.error = std::move(error);
    return model;
}

} // namespace

Device::Device(int idx, const GpuProfile& profile)
    : index(idx), profile(profile), used_memory(0), used_memory_peak(0) {
    name = profile.name;
    char uuid_buf[64];
    unsigned long long tail = 0x6789abcdef00ULL + static_cast<unsigned long long>(idx);
    snprintf(uuid_buf, sizeof(uuid_buf), "GPU-%08x-%04x-%04x-%04x-%012llx", 
             idx, 0xabcd, 0xef01, 0x2345, tail);
    uuid = std::string(uuid_buf);
    
    total_memory = profile.memory_bytes;
    
    char pci_buf[32];
    snprintf(pci_buf, sizeof(pci_buf), "00000000:%02x:00.0", idx + 1);
    pci_bus_id = std::string(pci_buf);
}

ModeledDeviceTopology build_modeled_device_topology(
    std::size_t device_count) {
    ModeledDeviceTopology topology;
    topology.nvlink_peers.resize(device_count);

    const char* groups_value = std::getenv("FAKEGPU_NVLINK_GROUPS");
    if (!groups_value || !*groups_value) {
        return topology;
    }

    topology.source = "modeled_environment";
    topology.configured = true;
    if (!parse_bandwidth(topology.nvlink_bandwidth_gbps)) {
        return invalid_topology(
            device_count,
            "FAKEGPU_NVLINK_BANDWIDTH_GBPS must be a finite number "
            "greater than 0 and no greater than 1000000");
    }

    std::set<std::pair<int, int>> links;
    const std::string groups_text = trim_copy(groups_value);
    if (
        groups_text.empty() ||
        groups_text.front() == ';' ||
        groups_text.back() == ';') {
        return invalid_topology(
            device_count,
            "FAKEGPU_NVLINK_GROUPS contains an empty group");
    }
    std::istringstream groups(groups_text);
    std::string group_text;
    std::size_t group_index = 0;
    while (std::getline(groups, group_text, ';')) {
        ++group_index;
        group_text = trim_copy(group_text);
        if (group_text.empty()) {
            return invalid_topology(
                device_count,
                "FAKEGPU_NVLINK_GROUPS contains an empty group");
        }
        if (
            group_text.front() == ',' ||
            group_text.back() == ',') {
            return invalid_topology(
                device_count,
                "FAKEGPU_NVLINK_GROUPS group " +
                    std::to_string(group_index) +
                    " contains an invalid device index");
        }

        std::vector<int> members;
        std::istringstream group(group_text);
        std::string member_text;
        while (std::getline(group, member_text, ',')) {
            int member = -1;
            if (!parse_device_index(
                    member_text,
                    device_count,
                    member)) {
                return invalid_topology(
                    device_count,
                    "FAKEGPU_NVLINK_GROUPS group " +
                        std::to_string(group_index) +
                        " contains an invalid device index");
            }
            members.push_back(member);
        }
        std::sort(members.begin(), members.end());
        members.erase(
            std::unique(members.begin(), members.end()),
            members.end());
        if (members.size() < 2) {
            return invalid_topology(
                device_count,
                "each FAKEGPU_NVLINK_GROUPS group must contain at "
                "least two distinct device indices");
        }
        if (members.size() - 1 > kMaximumModeledNvLinksPerDevice) {
            return invalid_topology(
                device_count,
                "an FAKEGPU_NVLINK_GROUPS group exceeds the "
                "18-link per-device limit");
        }
        for (std::size_t left = 0; left < members.size(); ++left) {
            for (
                std::size_t right = left + 1;
                right < members.size();
                ++right) {
                links.emplace(members[left], members[right]);
            }
        }
    }

    if (links.empty()) {
        return invalid_topology(
            device_count,
            "FAKEGPU_NVLINK_GROUPS did not define any links");
    }

    for (const auto& [left, right] : links) {
        topology.nvlink_peers[static_cast<std::size_t>(left)].push_back(
            {0, right, topology.nvlink_bandwidth_gbps});
        topology.nvlink_peers[static_cast<std::size_t>(right)].push_back(
            {0, left, topology.nvlink_bandwidth_gbps});
    }
    for (auto& peers : topology.nvlink_peers) {
        std::sort(
            peers.begin(),
            peers.end(),
            [](const ModeledNvLinkPeer& left,
               const ModeledNvLinkPeer& right) {
                return left.peer_index < right.peer_index;
            });
        if (peers.size() > kMaximumModeledNvLinksPerDevice) {
            return invalid_topology(
                device_count,
                "a device belongs to more than 18 modeled NVLink "
                "peer relationships");
        }
        for (std::size_t link = 0; link < peers.size(); ++link) {
            peers[link].link = static_cast<unsigned int>(link);
        }
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
    std::istringstream events_stream(events_text);
    std::string event_text;
    std::size_t event_index = 0;
    while (std::getline(events_stream, event_text, ';')) {
        ++event_index;
        event_text = trim_copy(event_text);
        if (
            event_text.empty() ||
            event_text.front() == ':' ||
            event_text.back() == ':') {
            return invalid_fault_model(
                "FAKEGPU_FAULT_EVENTS event " +
                std::to_string(event_index) +
                " must use DEVICE:CODE:SEVERITY[:COUNT]");
        }

        std::vector<std::string> fields;
        std::istringstream field_stream(event_text);
        std::string field;
        while (std::getline(field_stream, field, ':')) {
            fields.push_back(trim_copy(field));
        }
        if (fields.size() < 3 || fields.size() > 4) {
            return invalid_fault_model(
                "FAKEGPU_FAULT_EVENTS event " +
                std::to_string(event_index) +
                " must use DEVICE:CODE:SEVERITY[:COUNT]");
        }

        int device_index = -1;
        if (!parse_device_index(
                fields[0],
                device_count,
                device_index)) {
            return invalid_fault_model(
                "FAKEGPU_FAULT_EVENTS event " +
                std::to_string(event_index) +
                " contains an invalid device index");
        }
        const std::string code = uppercase_copy(fields[1]);
        if (!valid_fault_code(code)) {
            return invalid_fault_model(
                "FAKEGPU_FAULT_EVENTS event " +
                std::to_string(event_index) +
                " contains an invalid code");
        }
        const std::string severity = lowercase_copy(fields[2]);
        if (modeled_fault_severity_rank(severity) == 0) {
            return invalid_fault_model(
                "FAKEGPU_FAULT_EVENTS event " +
                std::to_string(event_index) +
                " severity must be info, warning, error, or critical");
        }
        uint64_t count = 1;
        if (
            fields.size() == 4 &&
            !parse_fault_count(fields[3], count)) {
            return invalid_fault_model(
                "FAKEGPU_FAULT_EVENTS event " +
                std::to_string(event_index) +
                " count must be between 1 and 1000000000");
        }

        const auto key = std::make_tuple(
            device_index,
            code,
            severity);
        uint64_t& aggregate = counts[key];
        aggregate = std::min<uint64_t>(
            kMaximumModeledFaultCount,
            aggregate >= kMaximumModeledFaultCount - count
                ? kMaximumModeledFaultCount
                : aggregate + count);
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

} // namespace fake_gpu
