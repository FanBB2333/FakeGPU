#pragma once

#include <cstddef>
#include <cstdio>
#include <cstring>

// PCI bus ids as NVML writes and reads them (nvml.h).
struct NvmlPciAddress {
    unsigned int domain = 0, bus = 0, device = 0, function = 0;

    bool operator==(const NvmlPciAddress& other) const {
        return domain == other.domain && bus == other.bus && device == other.device && function == other.function;
    }
};

// nvmlDeviceGetHandleByPciBusId formats, all hexadecimal:
// domain:bus:device.function, domain:bus:device and bus:device.function.
inline bool parse_nvml_pci_bus_id(const char* text, NvmlPciAddress& result) {
    if (!text) return false;
    const int length = static_cast<int>(std::strlen(text));
    unsigned int a = 0, b = 0, c = 0, d = 0;
    int end = -1;
    if (std::sscanf(text, "%x:%x:%x.%x%n", &a, &b, &c, &d, &end) == 4 && end == length) {
        result = {a, b, c, d};
        return true;
    }
    end = -1;
    if (std::sscanf(text, "%x:%x:%x%n", &a, &b, &c, &end) == 3 && end == length) {
        result = {a, b, c, 0};
        return true;
    }
    end = -1;
    if (std::sscanf(text, "%x:%x.%x%n", &a, &b, &c, &end) == 3 && end == length) {
        result = {0, a, b, c};
        return true;
    }
    return false;
}

// NVML_DEVICE_PCI_BUS_ID_FMT and NVML_DEVICE_PCI_BUS_ID_LEGACY_FMT.
inline void format_nvml_pci_bus_id(const NvmlPciAddress& address, char* bus_id, std::size_t bus_id_size,
                                   char* legacy, std::size_t legacy_size) {
    std::snprintf(bus_id, bus_id_size, "%08X:%02X:%02X.0", address.domain, address.bus, address.device);
    std::snprintf(legacy, legacy_size, "%04X:%02X:%02X.0", address.domain, address.bus, address.device);
}

// Whether a lookup key names the device's bus id, in any accepted format.
inline bool nvml_pci_bus_id_matches(const char* device_bus_id, const char* key) {
    if (std::strcmp(device_bus_id, key) == 0) return true;
    NvmlPciAddress device, requested;
    return parse_nvml_pci_bus_id(device_bus_id, device) && parse_nvml_pci_bus_id(key, requested) &&
           device == requested;
}
