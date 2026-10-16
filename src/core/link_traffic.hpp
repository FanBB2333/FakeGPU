#pragma once
#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "device.hpp"

namespace fake_gpu {

// Modeled time and throughput of copies over the GPU links. Mirrors
// fakegpu/link_traffic.py: copies on one link and direction run back to
// back at the bandwidth of their route, and throughput is the data that
// schedule delivered in the last kTransferWindowSeconds. Nothing waits.

enum class TransferLink : int { Pcie = 0, NvLink = 1, C2c = 2, None = -1 };
constexpr std::size_t kTransferLinkCount = 3;

enum class CopyKind : int { HostToDevice = 0, DeviceToHost = 1, PeerSend = 2, PeerReceive = 3 };
constexpr std::size_t kCopyKindCount = 4;

constexpr double kTransferWindowSeconds = 1.0;

const char* transfer_link_name(std::size_t link);
const char* copy_kind_name(std::size_t kind);

struct TransferRoute {
    TransferLink source = TransferLink::None;  // link the sending GPU transmits on
    TransferLink target = TransferLink::None;  // link the receiving GPU receives on
    double bandwidth_gbps = 0.0;               // 0: unknown, no modeled time
};

// Route of a copy between GPUs; -1 stands for host memory.
TransferRoute route_transfer(const ModeledDeviceTopology& topology, int source, int target);

struct CopyCounter {
    uint64_t calls = 0;
    uint64_t bytes = 0;
    double modeled_seconds = 0.0;
};

struct LinkCounter {
    uint64_t tx_bytes = 0;  // delivered so far under the schedule
    uint64_t rx_bytes = 0;
    uint64_t tx_bytes_per_second = 0;  // over the last window
    uint64_t rx_bytes_per_second = 0;
};

struct DeviceTransferStats {
    std::string host_link = "pcie";
    double host_bandwidth_gbps = 0.0;
    CopyCounter copies[kCopyKindCount];
    LinkCounter links[kTransferLinkCount];

    double modeled_seconds() const;
    bool any_copies() const;
};

class LinkTrafficModel {
public:
    // Records one copy at time `now` (monotonic seconds); -1 is host memory.
    // Returns the modeled seconds of the copy.
    double record(const ModeledDeviceTopology& topology, int source, int target, uint64_t bytes, double now);
    std::vector<DeviceTransferStats> snapshot(const ModeledDeviceTopology& topology, double now);

private:
    struct Schedule {
        struct Segment {
            double start = 0.0;
            double end = 0.0;
            double bytes = 0.0;
        };
        std::deque<Segment> segments;
        double busy_until = 0.0;
        uint64_t bytes = 0;

        void add(double now, uint64_t amount, double seconds);
        uint64_t delivered(double now) const;
        uint64_t rate(double now, double window);
    };
    struct DeviceState {
        CopyCounter copies[kCopyKindCount];
        Schedule tx[kTransferLinkCount];
        Schedule rx[kTransferLinkCount];
    };

    DeviceState* device(std::size_t index);
    std::vector<DeviceState> devices_;
};

double monotonic_seconds();

} // namespace fake_gpu
