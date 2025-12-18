#include "device.hpp"
#include <cstdio>

namespace fake_gpu {

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

} // namespace fake_gpu
