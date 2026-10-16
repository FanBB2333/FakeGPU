#!/usr/bin/env python3
"""Check that native NVML/CUDA and the native SMI state follow the platform topology.

Run under ``fakegpu run --platform ...``: the launcher exports FAKEGPU_SLOTS and
FAKEGPU_NVLINK_FABRIC, and every native answer must agree with the Python
model in ``fakegpu.link_topology`` for each GPU and pair of GPUs.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import time
from pathlib import Path

from fakegpu._smi_environment import device_topology_record
from fakegpu.platforms import device_profiles, topology_from_environment

NVML_SUCCESS = 0
NVML_ERROR_NOT_SUPPORTED = 3
NVML_NVLINK_DEVICE_TYPE_GPU = 0
NVML_NVLINK_DEVICE_TYPE_SWITCH = 2
TOPOLOGY_LEVELS = {"X": 0, "PIX": 10, "PXB": 20, "PHB": 30, "NODE": 40, "SYS": 50}
P2P_CAPS_NVLINK = 2


def _die(message: str) -> None:
    print(f"[platform_topology] ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def _library(name: str, *fallbacks: str) -> ctypes.CDLL:
    variable = "DYLD_INSERT_LIBRARIES" if sys.platform == "darwin" else "LD_PRELOAD"
    names = (name, *fallbacks)
    for part in os.environ.get(variable, "").split(":"):
        if Path(part).name in names:
            return ctypes.CDLL(part, mode=ctypes.RTLD_GLOBAL)
    return ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)


def _uint(function, *args) -> tuple[int, int]:
    value = ctypes.c_uint(0xFFFF)
    result = int(function(*args, ctypes.byref(value)))
    return result, int(value.value)


def main() -> int:
    darwin = sys.platform == "darwin"
    nvml = _library("libnvidia-ml.dylib" if darwin else "libnvidia-ml.so.1")
    cudart = _library("libcudart.dylib" if darwin else "libcudart.so.12", "libcudart.so")
    cuda = _library("libcuda.dylib" if darwin else "libcuda.so.1")

    profiles = device_profiles(os.environ)
    model = topology_from_environment(profiles, os.environ)
    if not model.configured or not model.valid:
        _die(f"the Python model is not configured: {model.source} {model.error}")
    count = len(profiles)

    if int(nvml.nvmlInit()) != NVML_SUCCESS or int(cuda.cuInit(0)) != 0:
        _die("nvmlInit or cuInit failed")
    handles = []
    for index in range(count):
        handle = ctypes.c_void_p()
        if int(nvml.nvmlDeviceGetHandleByIndex(ctypes.c_uint(index), ctypes.byref(handle))) != NVML_SUCCESS:
            _die(f"NVML has no GPU {index}")
        handles.append(handle)

    for index, handle in enumerate(handles):
        slot = model.slots[index]
        for function, expected in (
            (nvml.nvmlDeviceGetCurrPcieLinkGeneration, slot.pcie and slot.pcie.generation),
            (nvml.nvmlDeviceGetCurrPcieLinkWidth, slot.pcie and slot.pcie.width),
            (nvml.nvmlDeviceGetMaxPcieLinkGeneration, slot.pcie and slot.pcie.generation),
            (nvml.nvmlDeviceGetGpuMaxPcieLinkGeneration, slot.gpu_max_pcie and slot.gpu_max_pcie.generation),
        ):
            result, value = _uint(function, handle)
            if expected is None:
                if result != NVML_ERROR_NOT_SUPPORTED:
                    _die(f"GPU {index} {function.__name__} should be unsupported, got {result}/{value}")
            elif (result, value) != (NVML_SUCCESS, expected):
                _die(f"GPU {index} {function.__name__}: {result}/{value}, expected {expected}")
        ports = model.ports(index)
        for link in range(len(ports) + 1):
            result, active = _uint(nvml.nvmlDeviceGetNvLinkState, handle, ctypes.c_uint(link))
            if (result, active) != (NVML_SUCCESS, int(link < len(ports))):
                _die(f"GPU {index} NVLink {link} state {result}/{active}, model has {len(ports)} links")
            if link < len(ports):
                result, remote = _uint(nvml.nvmlDeviceGetNvLinkRemoteDeviceType, handle, ctypes.c_uint(link))
                expected = NVML_NVLINK_DEVICE_TYPE_GPU if ports[link].peer is not None else NVML_NVLINK_DEVICE_TYPE_SWITCH
                if (result, remote) != (NVML_SUCCESS, expected):
                    _die(f"GPU {index} NVLink {link} remote type {remote}, expected {expected}")

    for source in range(count):
        for target in range(count):
            relation = model.pcie_relation(source, target)
            result, level = _uint(nvml.nvmlDeviceGetTopologyCommonAncestor, handles[source], handles[target])
            if (result, level) != (NVML_SUCCESS, TOPOLOGY_LEVELS[relation]):
                _die(f"common ancestor {source}->{target}: {result}/{level}, expected {relation}")
            result, status = _uint(
                nvml.nvmlDeviceGetP2PStatus, handles[source], handles[target], ctypes.c_int(P2P_CAPS_NVLINK)
            )
            if (result, status == 0) != (NVML_SUCCESS, model.nvlink_links(source, target) > 0):
                _die(f"NVLink P2P status {source}->{target}: {result}/{status}")
            if source == target:
                continue
            expected = int(model.can_access_peer(source, target))
            runtime = ctypes.c_int(-1)
            driver = ctypes.c_int(-1)
            atomic = ctypes.c_int(-1)
            if int(cudart.cudaDeviceCanAccessPeer(ctypes.byref(runtime), source, target)) != 0:
                _die("cudaDeviceCanAccessPeer failed")
            if int(cuda.cuDeviceCanAccessPeer(ctypes.byref(driver), source, target)) != 0:
                _die("cuDeviceCanAccessPeer failed")
            if int(cuda.cuDeviceGetP2PAttribute(ctypes.byref(atomic), 3, source, target)) != 0:
                _die("cuDeviceGetP2PAttribute failed")
            if (runtime.value, driver.value, atomic.value) != (
                expected, expected, int(model.nvlink_links(source, target) > 0)
            ):
                _die(
                    f"peer access {source}->{target}: runtime {runtime.value}, driver {driver.value}, "
                    f"atomics {atomic.value}; model {expected}"
                )
            if not expected:
                cudart.cudaSetDevice(source)
                if int(cudart.cudaDeviceEnablePeerAccess(target, 0)) != 217:
                    _die(f"cudaDeviceEnablePeerAccess({source}->{target}) should be unsupported")

    directory = os.environ.get("FAKEGPU_SMI_STATE_DIR")
    if directory:
        path = Path(directory) / f"{os.getpid()}.json"
        deadline = time.monotonic() + 3.0
        state: dict = {}
        while time.monotonic() < deadline:
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
                if state.get("devices"):
                    break
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            time.sleep(0.05)
        topology = state.get("topology") or {}
        expected_matrix = [[model.relation(a, b) for b in range(count)] for a in range(count)]
        if topology.get("schema_version") != "fakegpu.device_topology.v2" or topology.get("matrix") != expected_matrix:
            _die(f"native SMI topology differs: {topology.get('matrix')} != {expected_matrix}")
        devices = sorted(state["devices"], key=lambda item: item["index"])
        for position, device in enumerate(devices):
            native = device["topology"]
            python = json.loads(json.dumps(device_topology_record(model, position, devices)))
            for key in ("source", "platform", "slot", "pcie", "host_link", "peer_access"):
                if native.get(key) != python[key]:
                    _die(f"GPU {position} SMI {key}: native {native.get(key)} != Python {python[key]}")
            for key in ("active_links", "peer_count", "link_bandwidth_gbps", "aggregate_bandwidth_gbps", "links", "peers"):
                if native["nvlink"].get(key) != python["nvlink"][key]:
                    _die(f"GPU {position} SMI nvlink.{key}: native {native['nvlink'].get(key)} != Python {python['nvlink'][key]}")
    print(f"[platform_topology] {model.platform}: {count} GPUs match the Python model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
