#include "global_state.hpp"
#include "logging.hpp"
#include "gpu_profile.hpp"

#include <algorithm>

namespace fake_gpu {

namespace {
std::vector<GpuProfile> build_default_profiles() {
    // Keep a single place for default GPU definitions to simplify adding new models.
    constexpr int DEVICE_COUNT = 8;
    std::vector<GpuProfile> profiles;
    profiles.reserve(DEVICE_COUNT);
    for (int i = 0; i < DEVICE_COUNT; ++i) {
        profiles.push_back(GpuProfile::A100());
    }
    return profiles;
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
