#include "device.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <set>
#include <sstream>
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

} // namespace fake_gpu
