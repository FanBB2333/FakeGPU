"""Send every catalog profile's full-load modeled telemetry through real NVML."""

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path
import sys

from fakegpu.nvml import NvmlStateBridge
from fakegpu.profile_catalog import load_profiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib-dir", required=True, type=Path)
    args = parser.parse_args()
    name = "libnvidia-ml.dylib" if sys.platform == "darwin" else "libnvidia-ml.so.1"
    library = ctypes.CDLL(str((args.lib_dir / name).resolve()))
    setter = library.fakegpuNvmlSetSnapshot
    setter.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p]
    setter.restype = ctypes.c_int
    inventory = {"devices": [], "processes": []}
    for index, profile in enumerate(load_profiles().values()):
        uuid = f"synthetic-catalog-{profile.id}"
        inventory["devices"].append({
            "uuid": uuid, "index": index, "name": profile.name, "profile_id": profile.id,
            "pci_bus_id": f"00000000:{index:04x}:00.0",
            "memory": {"total_bytes": profile.memory_bytes, "used_bytes": 1024},
        })
        inventory["processes"].append({
            "gpu_uuid": uuid, "pid": 12345, "profile_id": profile.id, "timestamp_ns": 10**9,
            "status": "running",
            "memory": {"used_bytes": 1024},
            "activity": {"source": "cpu_dispatch_proxy", "busy_ns": 0, "io_bytes": 0},
        })
    bridge = NvmlStateBridge(state_dir=args.lib_dir, telemetry="reference")
    bridge._snapshot_arrays(inventory)
    for process in inventory["processes"]:
        process["timestamp_ns"] += 10**9
        process["activity"]["busy_ns"] = 10**9
    devices, processes = bridge._snapshot_arrays(inventory)
    assert setter(3, len(devices), devices, len(processes), processes) == 0, "catalog telemetry rejected"
    library.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
    library.nvmlDeviceGetPowerUsage.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
    for index, device in enumerate(devices):
        handle, power = ctypes.c_void_p(), ctypes.c_uint()
        assert library.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)) == 0
        assert library.nvmlDeviceGetPowerUsage(handle, ctypes.byref(power)) == 0
        assert power.value == device.telemetry.power_mw
        assert device.telemetry.valid == 1 and device.telemetry.gpu == 100
    print(f"PASS: full-load modeled telemetry for {len(devices)} catalog profiles via native NVML")


if __name__ == "__main__":
    main()
