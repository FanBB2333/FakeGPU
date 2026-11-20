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
        error = "FAKEGPU_DIST_MODE must be set to simulate, proxy, or passthrough";
        return false;
    }
    if (config.mode != distributed::DistributedMode::Passthrough &&
        config.coordinator_address.empty()) {
        error = "FAKEGPU_COORDINATOR_ADDR must be set";
        return false;
    }
    return true;
}

}  // namespace fake_gpu::nccl
