#include "global_state.hpp"
#include "logging.hpp"
#include "gpu_profile.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>

namespace fake_gpu {

namespace {
constexpr int kDefaultDeviceCount = 8;

std::string trim_copy(const std::string& value) {
    const size_t begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return "";
    const size_t end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

bool parse_positive_int(const std::string& value, int& out) {
    const std::string trimmed = trim_copy(value);
    if (trimmed.empty()) return false;
    char* end = nullptr;
    long parsed = std::strtol(trimmed.c_str(), &end, 10);
    if (end == trimmed.c_str() || (end && *end != '\0')) return false;
    if (parsed <= 0 || parsed > std::numeric_limits<int>::max()) return false;
    out = static_cast<int>(parsed);
    return true;
}

std::vector<GpuProfile> build_profiles_from_spec(const std::string& spec) {
    std::vector<GpuProfile> profiles;
    std::istringstream stream(spec);
    std::string token;

    while (std::getline(stream, token, ',')) {
        token = trim_copy(token);
        if (token.empty()) continue;

        std::string preset_id = token;
        int repeat = 1;

        const size_t colon = token.find(':');
        if (colon != std::string::npos) {
            preset_id = trim_copy(token.substr(0, colon));
            const std::string count_text = trim_copy(token.substr(colon + 1));
            if (!count_text.empty()) {
                int parsed = 0;
                if (parse_positive_int(count_text, parsed)) {
                    repeat = parsed;
                } else {
                    FGPU_LOG("[FakeGPU] Invalid FAKEGPU_PROFILES entry '%s' (bad count '%s')\n", token.c_str(), count_text.c_str());
                }
            }
        }

        std::optional<GpuProfile> maybe_profile = profile_from_preset_id(preset_id);
        if (!maybe_profile.has_value()) {
            FGPU_LOG("[FakeGPU] Unknown GPU preset '%s'; falling back to A100\n", preset_id.c_str());
            maybe_profile = GpuProfile::A100();
        }

        profiles.insert(profiles.end(), repeat, maybe_profile.value());
    }

    return profiles;
}

std::vector<GpuProfile> build_default_profiles() {
    // Default to eight identical A100-class devices to mirror common server setups.
    const char* profiles_env = std::getenv("FAKEGPU_PROFILES");
    if (profiles_env && *profiles_env) {
        std::vector<GpuProfile> profiles = build_profiles_from_spec(profiles_env);
        if (!profiles.empty()) return profiles;
        FGPU_LOG("[FakeGPU] FAKEGPU_PROFILES is set but produced no devices; falling back to defaults\n");
    }

    std::string preset_id = "a100";
    if (const char* preset_env = std::getenv("FAKEGPU_PROFILE"); preset_env && *preset_env) {
        preset_id = trim_copy(preset_env);
    }

    int count = kDefaultDeviceCount;
    if (const char* count_env = std::getenv("FAKEGPU_DEVICE_COUNT"); count_env && *count_env) {
        int parsed = 0;
        if (parse_positive_int(count_env, parsed)) {
            count = parsed;
        } else {
            FGPU_LOG("[FakeGPU] Invalid FAKEGPU_DEVICE_COUNT='%s'; using %d\n", count_env, kDefaultDeviceCount);
        }
    }

    std::optional<GpuProfile> maybe_profile = profile_from_preset_id(preset_id);
    GpuProfile profile = maybe_profile.has_value() ? maybe_profile.value() : GpuProfile::A100();
    return std::vector<GpuProfile>(count, profile);
}
} // namespace

GlobalState& GlobalState::instance() {
    static GlobalState* s_instance = new GlobalState();
    return *s_instance;
}

GlobalState::GlobalState() {
}

void GlobalState::initialize() {
    std::lock_guard<std::mutex> lock(mutex);
    FGPU_LOG("[GlobalState-%p] initialize called. Current devices: %lu\n", this, devices.size());
    if (initialized) return;

    // Pre-allocate to prevent reallocation during emplace_back
    // This is critical because nvitop holds Device* pointers from nvmlDeviceGetHandleByIndex
    // and vector reallocation would invalidate those pointers
    std::vector<GpuProfile> profiles = build_default_profiles();
    devices.reserve(profiles.size());
    
    // Create fake devices from the configured profiles
    for (size_t i = 0; i < profiles.size(); ++i) {
        devices.emplace_back(static_cast<int>(i), profiles[i]);
    }
    initialized = true;
    FGPU_LOG("[GlobalState-%p] Valid devices count after init: %lu\n", this, devices.size());
}

int GlobalState::get_device_count() const {
    return devices.size();
}

Device& GlobalState::get_device(int index) {
    // In a real scenario, check bounds
    if (index < 0 || index >= devices.size()) {
        static Device failure_dev(-1, GpuProfile::A100());
        return failure_dev;
    }
    return devices[index];
}

void GlobalState::set_current_device(int device) {
    std::lock_guard<std::mutex> lock(mutex);
    if (device >= 0 && device < static_cast<int>(devices.size())) {
        current_device = device;
    }
}

int GlobalState::get_current_device() const {
    std::lock_guard<std::mutex> lock(mutex);
    return current_device;
}

bool GlobalState::register_allocation(void* ptr, size_t size, int device) {
    std::lock_guard<std::mutex> lock(mutex);
    if (!ptr) return false;
    if (device < 0 || device >= static_cast<int>(devices.size())) return false;

    Device& dev = devices[device];
    if (dev.used_memory + size > dev.total_memory) {
        return false; // not enough fake memory
    }

    dev.used_memory += size;
    dev.used_memory_peak = std::max(dev.used_memory_peak, dev.used_memory);
    allocations[ptr] = {size, device};
    return true;
}

bool GlobalState::release_allocation(void* ptr, size_t& size, int& device) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = allocations.find(ptr);
    if (it == allocations.end()) return false;

    size = it->second.first;
    device = it->second.second;
    allocations.erase(it);

    if (device >= 0 && device < static_cast<int>(devices.size())) {
        Device& dev = devices[device];
        if (dev.used_memory >= size) {
            dev.used_memory -= size;
        } else {
            dev.used_memory = 0;
        }
    }
    return true;
}

bool GlobalState::get_allocation_info(void* ptr, size_t& size, int& device) const {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = allocations.find(ptr);
    if (it == allocations.end()) return false;

    size = it->second.first;
    device = it->second.second;
    return true;
}

} // namespace fake_gpu

// Ensure initialization when the shared library is preloaded
__attribute__((constructor)) static void fake_gpu_constructor() {
    fake_gpu::GlobalState::instance().initialize();
}
