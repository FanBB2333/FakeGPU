#pragma once

#include "../distributed/cluster_config.hpp"

#include <string>

namespace fake_gpu::nccl {

inline bool validate_direct_init_config(
    const distributed::DistributedConfig& config,
    std::string& error) {
    if (!config.valid()) {
        error = config.validation_error;
        return false;
    }
    if (!config.enabled()) {
        error = "FAKEGPU_DIST_MODE must be set to simulate";
        return false;
    }
    if (config.mode != distributed::DistributedMode::Simulate) {
        error = "only FAKEGPU_DIST_MODE=simulate is implemented for fake NCCL init";
        return false;
    }
    if (config.coordinator_transport != distributed::CoordinatorTransport::Unix) {
        error = "only FAKEGPU_COORDINATOR_TRANSPORT=unix is implemented for fake NCCL init";
        return false;
    }
    if (config.coordinator_address.empty()) {
        error = "FAKEGPU_COORDINATOR_ADDR must be set";
        return false;
    }
    return true;
}

}  // namespace fake_gpu::nccl
