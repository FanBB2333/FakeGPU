#include "link_traffic.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace fake_gpu {

namespace {

constexpr std::size_t kMaximumSegments = 4096;

TransferLink host_transfer_link(const ModeledDeviceTopology& topology, int index) {
    const ModeledSlot* slot = topology.slot(index);
    if (!slot) return TransferLink::None;
    if (slot->host_link == "pcie") return TransferLink::Pcie;
    if (slot->host_link == "c2c") return TransferLink::C2c;
    return TransferLink::None;
}

double modeled_seconds_for(uint64_t bytes, double bandwidth_gbps) {
    if (bandwidth_gbps <= 0.0 || bytes == 0) return 0.0;
    return static_cast<double>(bytes) / (bandwidth_gbps * 1e9);
}

} // namespace

const char* transfer_link_name(std::size_t link) {
    static const char* const names[kTransferLinkCount] = {"pcie", "nvlink", "c2c"};
    return link < kTransferLinkCount ? names[link] : "none";
}

const char* copy_kind_name(std::size_t kind) {
    static const char* const names[kCopyKindCount] = {
        "host_to_device", "device_to_host", "peer_send", "peer_receive"};
    return kind < kCopyKindCount ? names[kind] : "unknown";
}

double monotonic_seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

double DeviceTransferStats::modeled_seconds() const {
    double total = 0.0;
    for (const auto& copy : copies) total += copy.modeled_seconds;
    return total;
}

bool DeviceTransferStats::any_copies() const {
    for (const auto& copy : copies) {
        if (copy.bytes) return true;
    }
    return false;
}

TransferRoute route_transfer(const ModeledDeviceTopology& topology, int source, int target) {
    TransferRoute route;
    if (source < 0 && target < 0) return route;
    if (source < 0 || target < 0) {
        const int index = source < 0 ? target : source;
        const TransferLink link = host_transfer_link(topology, index);
        route.source = source < 0 ? TransferLink::None : link;
        route.target = target < 0 ? TransferLink::None : link;
        route.bandwidth_gbps = topology.host_bandwidth_gbps(index);
        return route;
    }
    if (source == target) return route;
    if (topology.nvlink_links(source, target) > 0) {
        route.source = route.target = TransferLink::NvLink;
        route.bandwidth_gbps = topology.peer_bandwidth_gbps(source, target);
        return route;
    }
    if (topology.can_access_peer(source, target)) {
        route.source = route.target = TransferLink::Pcie;
        route.bandwidth_gbps = topology.peer_bandwidth_gbps(source, target);
        return route;
    }
    // Staged through host memory over both CPU links.
    route.source = host_transfer_link(topology, source);
    route.target = host_transfer_link(topology, target);
    const double a = topology.host_bandwidth_gbps(source);
    const double b = topology.host_bandwidth_gbps(target);
    route.bandwidth_gbps = (a > 0.0 && b > 0.0) ? std::min(a, b) : std::max(a, b);
    return route;
}

void LinkTrafficModel::Schedule::add(double now, uint64_t amount, double seconds) {
    bytes += amount;
    if (seconds <= 0.0) return;
    const double start = std::max(now, busy_until);
    const double end = start + seconds;
    busy_until = end;
    const double size = static_cast<double>(amount);
    if (!segments.empty()) {
        Segment& tail = segments.back();
        const bool contiguous = tail.end == start;
        const bool same_rate =
            std::fabs(tail.bytes * seconds - size * (tail.end - tail.start)) <= 1e-9 * tail.bytes * seconds;
        if (contiguous && (same_rate || segments.size() >= kMaximumSegments)) {
            tail.end = end;
            tail.bytes += size;
            return;
        }
    }
    segments.push_back({start, end, size});
}

uint64_t LinkTrafficModel::Schedule::delivered(double now) const {
    double pending = 0.0;
    for (auto it = segments.rbegin(); it != segments.rend(); ++it) {
        if (it->end <= now) break;
        pending += it->bytes * (it->end - std::max(it->start, now)) / (it->end - it->start);
    }
    const double value = static_cast<double>(bytes) - pending;
    return value > 0.0 ? static_cast<uint64_t>(std::llround(value)) : 0;
}

uint64_t LinkTrafficModel::Schedule::rate(double now, double window) {
    while (!segments.empty() && segments.front().end <= now - window) {
        segments.pop_front();
    }
    const double low = now - window;
    double total = 0.0;
    for (const auto& segment : segments) {
        if (segment.start >= now) break;
        const double overlap = std::min(segment.end, now) - std::max(segment.start, low);
        if (overlap > 0.0) total += segment.bytes * overlap / (segment.end - segment.start);
    }
    return static_cast<uint64_t>(std::llround(total / window));
}

LinkTrafficModel::DeviceState* LinkTrafficModel::device(std::size_t index) {
    if (index >= devices_.size()) devices_.resize(index + 1);
    return &devices_[index];
}

double LinkTrafficModel::record(
    const ModeledDeviceTopology& topology, int source, int target, uint64_t bytes, double now) {
    const int count = static_cast<int>(topology.slots.size());
    if (bytes == 0 || source == target || source >= count || target >= count) return 0.0;
    const TransferRoute route = route_transfer(topology, source, target);
    const double seconds = modeled_seconds_for(bytes, route.bandwidth_gbps);
    auto count_copy = [&](int index, CopyKind kind) {
        CopyCounter& counter = device(static_cast<std::size_t>(index))->copies[static_cast<int>(kind)];
        counter.calls += 1;
        counter.bytes += bytes;
        counter.modeled_seconds += seconds;
    };
    auto schedule = [&](int index, TransferLink link, bool transmit) {
        if (link == TransferLink::None) return;
        DeviceState* state = device(static_cast<std::size_t>(index));
        Schedule& target_schedule = transmit ? state->tx[static_cast<int>(link)] : state->rx[static_cast<int>(link)];
        target_schedule.add(now, bytes, seconds);
    };
    if (source < 0) {
        count_copy(target, CopyKind::HostToDevice);
        schedule(target, route.target, false);
    } else if (target < 0) {
        count_copy(source, CopyKind::DeviceToHost);
        schedule(source, route.source, true);
    } else {
        count_copy(source, CopyKind::PeerSend);
        count_copy(target, CopyKind::PeerReceive);
        schedule(source, route.source, true);
        schedule(target, route.target, false);
    }
    return seconds;
}

std::vector<DeviceTransferStats> LinkTrafficModel::snapshot(const ModeledDeviceTopology& topology, double now) {
    std::vector<DeviceTransferStats> result(topology.slots.size());
    for (std::size_t index = 0; index < result.size(); ++index) {
        DeviceTransferStats& stats = result[index];
        const ModeledSlot& slot = topology.slots[index];
        stats.host_link = slot.host_link;
        stats.host_bandwidth_gbps = slot.host_bandwidth_gbps();
        if (index >= devices_.size()) continue;
        DeviceState& state = devices_[index];
        for (std::size_t kind = 0; kind < kCopyKindCount; ++kind) stats.copies[kind] = state.copies[kind];
        for (std::size_t link = 0; link < kTransferLinkCount; ++link) {
            stats.links[link].tx_bytes = state.tx[link].delivered(now);
            stats.links[link].rx_bytes = state.rx[link].delivered(now);
            stats.links[link].tx_bytes_per_second = state.tx[link].rate(now, kTransferWindowSeconds);
            stats.links[link].rx_bytes_per_second = state.rx[link].rate(now, kTransferWindowSeconds);
        }
    }
    return result;
}

} // namespace fake_gpu
