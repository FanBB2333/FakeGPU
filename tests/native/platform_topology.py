#!/usr/bin/env python3
"""Check that native NVML/CUDA and the native SMI state follow the platform topology.

Run under ``fakegpu run --platform ...``: the launcher exports FAKEGPU_SLOTS and
FAKEGPU_NVLINK_FABRIC, and every native answer must agree with the Python
model in ``fakegpu.link_topology`` for each GPU and pair of GPUs. Copies
between the host and the GPUs and from GPU 0 to GPU 1 must be recorded with
the routes and modeled times of ``fakegpu.link_traffic``.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import time
from pathlib import Path

from fakegpu._smi_environment import device_topology_record
from fakegpu.link_traffic import LINK_KINDS, LinkTraffic, normalize_record
from fakegpu.platforms import device_profiles, topology_from_environment

NVML_SUCCESS = 0
NVML_ERROR_NOT_SUPPORTED = 3
NVML_NVLINK_DEVICE_TYPE_GPU = 0
NVML_NVLINK_DEVICE_TYPE_SWITCH = 2
TOPOLOGY_LEVELS = {"X": 0, "PIX": 10, "PXB": 20, "PHB": 30, "NODE": 40, "SYS": 50}
P2P_CAPS_NVLINK = 2
COPY_BYTES = 1 << 20
FIELD_NVLINK_LINK_COUNT = 91
FIELD_NVLINK_DATA_TX = 138


class _FieldValue(ctypes.Structure):
    _fields_ = [
        ("fieldId", ctypes.c_uint32),
        ("scopeId", ctypes.c_uint32),
        ("timestamp", ctypes.c_int64),
        ("latencyUsec", ctypes.c_int64),
        ("valueType", ctypes.c_int),
        ("nvmlReturn", ctypes.c_int),
        ("value", ctypes.c_uint64),
    ]


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


def _copy_and_check_links(nvml, cudart, handles, model, count: int) -> dict:
    """Copy host->every GPU, GPU 0->GPU 1 and GPU 0->host; check NVML throughput."""

    cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    cudart.cudaMemcpyPeer.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    host = ctypes.create_string_buffer(COPY_BYTES)
    clock = [0.0]
    traffic = LinkTraffic(lambda: model, clock=lambda: clock[0])
    started = time.monotonic()
    buffers = []
    for index in range(count):
        cudart.cudaSetDevice(index)
        pointer = ctypes.c_void_p()
        if int(cudart.cudaMalloc(ctypes.byref(pointer), COPY_BYTES)) != 0:
            _die(f"cudaMalloc on GPU {index} failed")
        buffers.append(pointer)
        if int(cudart.cudaMemcpy(pointer, host, COPY_BYTES, 1)) != 0:
            _die(f"host to GPU {index} copy failed")
        traffic.record(None, index, COPY_BYTES)
    if count > 1:
        if int(cudart.cudaMemcpyPeer(buffers[1], 1, buffers[0], 0, COPY_BYTES)) != 0:
            _die("cudaMemcpyPeer 0->1 failed")
        traffic.record(0, 1, COPY_BYTES)
    cudart.cudaSetDevice(0)
    if int(cudart.cudaMemcpy(host, buffers[0], COPY_BYTES, 2)) != 0:
        _die("GPU 0 to host copy failed")
    traffic.record(0, None, COPY_BYTES)
    # Every modeled copy has finished: the reference schedule ran at t=0.
    clock[0] = 1e-3
    expected = traffic.snapshot()

    nvml.nvmlDeviceGetFieldValues.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(_FieldValue)]
    time.sleep(0.01)  # let the modeled copies (tens of microseconds) finish
    within_window = True
    for index, handle in enumerate(handles):
        pcie = expected[index]["links"]["pcie"]
        for counter, key in ((0, "tx_bytes"), (1, "rx_bytes")):
            result, value = _uint(nvml.nvmlDeviceGetPcieThroughput, handle, ctypes.c_uint(counter))
            if model.slots[index].pcie is None:
                if result != NVML_ERROR_NOT_SUPPORTED:
                    _die(f"GPU {index} has no PCIe link but PCIe throughput returned {result}")
                continue
            within_window = time.monotonic() - started < 0.9
            if result != NVML_SUCCESS or (within_window and value != pcie[key] // 1024):
                _die(f"GPU {index} PCIe throughput {counter}: {result}/{value} KB/s, expected {pcie[key] // 1024}")
        fields = (_FieldValue * 2)(
            _FieldValue(fieldId=FIELD_NVLINK_LINK_COUNT), _FieldValue(fieldId=FIELD_NVLINK_DATA_TX, scopeId=0)
        )
        if int(nvml.nvmlDeviceGetFieldValues(handle, 2, fields)) != NVML_SUCCESS:
            _die(f"GPU {index} nvmlDeviceGetFieldValues failed")
        ports = len(model.ports(index))
        if (fields[0].nvmlReturn, fields[0].value & 0xFFFFFFFF) != (NVML_SUCCESS, ports):
            _die(f"GPU {index} NVLink link count {fields[0].nvmlReturn}/{fields[0].value}, expected {ports}")
        if ports:
            kib = expected[index]["links"]["nvlink"]["tx_bytes"] // ports // 1024
            if (fields[1].nvmlReturn, fields[1].value) != (NVML_SUCCESS, kib):
                _die(f"GPU {index} NVLink TX counter {fields[1].nvmlReturn}/{fields[1].value} KiB, expected {kib}")
    if not within_window:
        print("[platform_topology] note: throughput checked after the 1 s window; values not compared")
    return expected


def _compare_transfer(position: int, native: dict | None, python: dict) -> None:
    if native is None:
        _die(f"GPU {position} SMI state has no transfer record")
    for key in ("host_link", "host_bandwidth_gbps"):
        if key == "host_bandwidth_gbps":
            same = (native[key] is None) == (python[key] is None) and (
                native[key] is None or abs(native[key] - python[key]) <= 1e-4 * python[key]
            )
        else:
            same = native[key] == python[key]
        if not same:
            _die(f"GPU {position} transfer {key}: native {native[key]} != Python {python[key]}")
    for kind, values in python["copies"].items():
        other = native["copies"][kind]
        if (other["calls"], other["bytes"]) != (values["calls"], values["bytes"]) or abs(
            other["modeled_seconds"] - values["modeled_seconds"]
        ) > 1e-9 * max(1.0, values["modeled_seconds"]) + 1e-12:
            _die(f"GPU {position} transfer copies.{kind}: native {other} != Python {values}")
    for link in LINK_KINDS:
        for key in ("tx_bytes", "rx_bytes"):
            if native["links"][link][key] != python["links"][link][key]:
                _die(
                    f"GPU {position} transfer links.{link}.{key}: native {native['links'][link][key]} "
                    f"!= Python {python['links'][link][key]}"
                )


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

    expected = _copy_and_check_links(nvml, cudart, handles, model, count)

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
        while time.monotonic() < deadline:
            # Wait for a publish that includes the copies.
            device = next((item for item in state.get("devices", []) if item.get("index") == 0), {})
            copies = (device.get("transfer") or {}).get("copies", {})
            if copies.get("host_to_device", {}).get("bytes") == COPY_BYTES:
                break
            time.sleep(0.05)
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
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
            _compare_transfer(position, normalize_record(device.get("transfer")), expected[position])
    print(f"[platform_topology] {model.platform}: {count} GPUs match the Python model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
