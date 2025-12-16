#!/usr/bin/env python3
"""Simple test to verify logging can be toggled"""

import ctypes
import os

# Load the fake GPU library
lib_path = os.path.join(os.path.dirname(__file__), "build", "libnvidia-ml.so.1")
nvml = ctypes.CDLL(lib_path)

# Initialize NVML
print("Calling nvmlInit...")
result = nvml.nvmlInit()
print(f"nvmlInit returned: {result}")

# Get device count
device_count = ctypes.c_uint()
print("\nCalling nvmlDeviceGetCount...")
result = nvml.nvmlDeviceGetCount(ctypes.byref(device_count))
print(f"nvmlDeviceGetCount returned: {result}, count: {device_count.value}")

# Shutdown
print("\nCalling nvmlShutdown...")
result = nvml.nvmlShutdown()
print(f"nvmlShutdown returned: {result}")

print("\n=== Test completed ===")
