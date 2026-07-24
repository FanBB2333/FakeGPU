#pragma once

#include <cstdio>
#include <string>

#include "backend_config.hpp"
#include "global_state.hpp"

namespace fake_gpu {

inline bool record_unsupported_api(
    GlobalState& state,
    const char* operation,
    const char* behavior = "not_executed") {
    state.initialize();
    const UnsupportedApiPolicy policy =
        BackendConfig::instance().unsupported_api_policy();
    const char* operation_name = operation ? operation : "unknown";
    const char* behavior_name = behavior ? behavior : "unsupported";
    const bool first_occurrence = state.record_unsupported_api(
        state.get_current_device(),
        operation_name,
        behavior_name,
        unsupported_api_policy_name(policy));
    if (policy == UnsupportedApiPolicy::Warn && first_occurrence) {
        std::fprintf(
            stderr,
            "[FakeGPU] warning: %s is simulated as %s; "
            "set FAKEGPU_UNSUPPORTED_API=error to reject it.\n",
            operation_name,
            behavior_name);
    }
    return policy == UnsupportedApiPolicy::Error;
}

inline bool record_unsupported_api(
    const char* operation,
    const char* behavior = "not_executed") {
    GlobalState& state = GlobalState::instance();
    return record_unsupported_api(state, operation, behavior);
}

} // namespace fake_gpu
