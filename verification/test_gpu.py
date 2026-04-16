#!/usr/bin/env python3
"""
Simple Python script to test fake GPU detection using pynvml
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fakegpu


def _nvml_lib_name() -> str:
    return 'libnvidia-ml.dylib' if sys.platform == 'darwin' else 'libnvidia-ml.so.1'


result = fakegpu.init(runtime="native", force=True, update_env=True)
print(f"[Python] FakeGPU library dir: {result.lib_dir}")

try:
    import pynvml
except ImportError as e:
    print(f"ERROR: pynvml not installed or failed to import: {e}")
    print("Install with: pip install nvidia-ml-py3")
    sys.exit(1)


def _load_fake_nvml_library():
    pynvml.nvmlLib = result.handles[_nvml_lib_name()]
    return pynvml.nvmlLib


pynvml.nvmlLib = None
pynvml._LoadNvmlLibrary = _load_fake_nvml_library

try:
    print("=== Testing GPU Detection with pynvml ===")

    # Initialize NVML
    print("[Python] Calling pynvml.nvmlInit()...")
    pynvml.nvmlInit()

    # Get device count
    device_count = pynvml.nvmlDeviceGetCount()
    print(f"[Python] Detected {device_count} GPU(s)")

    # Get info for each device
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        uuid = pynvml.nvmlDeviceGetUUID(handle)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        print(f"\n[GPU {i}]")
        print(f"  Name: {name}")
        print(f"  UUID: {uuid}")
        print(f"  Total Memory: {memory_info.total / (1024**3):.2f} GB")
        print(f"  Used Memory:  {memory_info.used / (1024**3):.2f} GB")
        print(f"  Free Memory:  {memory_info.free / (1024**3):.2f} GB")

    # Shutdown
    pynvml.nvmlShutdown()
    print("\n[Python] Test completed successfully!")

except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
