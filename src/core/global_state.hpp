#pragma once
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <cstddef>
#include "device.hpp"

namespace fake_gpu {

class GlobalState {
public:
    static GlobalState& instance();

    void initialize();
    int get_device_count() const;
    Device& get_device(int index);
    void set_current_device(int device);
    int get_current_device() const;

    // Allocation tracking
    bool register_allocation(void* ptr, size_t size, int device);
    bool release_allocation(void* ptr, size_t& size, int& device);
    bool get_allocation_info(void* ptr, size_t& size, int& device) const;

private:
    GlobalState();
    ~GlobalState() = default;

    bool initialized = false;
    std::vector<Device> devices;
    mutable std::mutex mutex;

    int current_device = 0;
    std::unordered_map<void*, std::pair<size_t, int>> allocations;
};

} // namespace fake_gpu
