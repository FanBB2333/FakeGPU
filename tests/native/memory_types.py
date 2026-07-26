#!/usr/bin/env python3
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path


CUDA_SUCCESS = 0

CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2
CU_POINTER_ATTRIBUTE_IS_MANAGED = 8
CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9

CU_MEMORYTYPE_HOST = 1
CU_MEMORYTYPE_DEVICE = 2
CU_MEMORYTYPE_UNIFIED = 4

NVML_SUCCESS = 0
NVML_NVLINK_DEVICE_TYPE_GPU = 0


class NvmlPciInfo(ctypes.Structure):
    _fields_ = [
        ("busIdLegacy", ctypes.c_char * 16),
        ("domain", ctypes.c_uint),
        ("bus", ctypes.c_uint),
        ("device", ctypes.c_uint),
        ("pciDeviceId", ctypes.c_uint),
        ("pciSubSystemId", ctypes.c_uint),
        ("busId", ctypes.c_char * 32),
    ]


def _die(msg: str) -> None:
    print(f"[test_memory_types] ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _preload_env_var() -> str:
    return "DYLD_INSERT_LIBRARIES" if sys.platform == "darwin" else "LD_PRELOAD"


def _libcuda_name() -> str:
    return "libcuda.dylib" if sys.platform == "darwin" else "libcuda.so.1"


def _libnvidia_ml_name() -> str:
    return (
        "libnvidia-ml.dylib"
        if sys.platform == "darwin"
        else "libnvidia-ml.so.1"
    )


def _find_preloaded_path(libname: str) -> str | None:
    preload = os.environ.get(_preload_env_var(), "")
    for part in preload.split(":"):
        part = part.strip()
        if not part:
            continue
        if Path(part).name == libname:
            return part
    return None


def _load_libcuda() -> ctypes.CDLL:
    libname = _libcuda_name()
    path = _find_preloaded_path(libname)
    return ctypes.CDLL(path or libname, mode=ctypes.RTLD_GLOBAL)


def _load_libnvidia_ml() -> ctypes.CDLL:
    libname = _libnvidia_ml_name()
    path = _find_preloaded_path(libname)
    return ctypes.CDLL(path or libname, mode=ctypes.RTLD_GLOBAL)


def _cu_check(result: int, *, what: str) -> None:
    if int(result) != CUDA_SUCCESS:
        _die(f"{what} failed: CUresult={int(result)}")


def _cu_get_ptr_attr(libcuda: ctypes.CDLL, ptr: int, attr: int, ctype: object) -> int:
    out = ctype()
    _cu_check(
        int(libcuda.cuPointerGetAttribute(ctypes.byref(out), ctypes.c_int(attr), ctypes.c_ulonglong(ptr))),
        what=f"cuPointerGetAttribute(ptr=0x{ptr:x}, attr={attr})",
    )
    return int(out.value)


def _configured_smi_state_path() -> Path | None:
    explicit = os.environ.get("FAKEGPU_SMI_STATE_PATH")
    if explicit:
        return Path(explicit)
    directory = os.environ.get("FAKEGPU_SMI_STATE_DIR")
    if directory:
        return Path(directory) / f"{os.getpid()}.json"
    return None


def _validate_native_smi_state() -> None:
    state_path = _configured_smi_state_path()
    if state_path is None:
        return

    deadline = time.monotonic() + 3.0
    state: dict = {}
    while time.monotonic() < deadline:
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.05)
            continue
        devices = state.get("devices") or []
        if (
            state.get("running") is True
            and devices
            and int(devices[0].get("tracked_memory", 0)) >= 8192
        ):
            break
        time.sleep(0.05)
    else:
        _die(f"native SMI state did not become current: {state_path}")

    if state.get("schema_version") != "fakegpu.smi_state.v2":
        _die(f"unexpected native SMI schema: {state.get('schema_version')}")
    if state.get("runtime") != "native":
        _die(f"unexpected native SMI runtime: {state.get('runtime')}")
    fakegpu = state.get("fakegpu") or {}
    if fakegpu.get("backend") != "native_interception":
        _die(f"unexpected native SMI backend: {fakegpu.get('backend')}")
    publisher = state.get("publisher") or {}
    health = publisher.get("health") or {}
    limits = publisher.get("limits") or {}
    if int(health.get("attempted_writes", 0)) < 2:
        _die(f"native SMI publisher attempts are incomplete: {health}")
    if int(health.get("successful_writes", 0)) < 2:
        _die(f"native SMI publisher successes are incomplete: {health}")
    if int(health.get("failed_writes", -1)) != 0:
        _die(f"native SMI publisher reported a failure: {health}")
    if int(health.get("last_serialized_bytes", 0)) <= 0:
        _die(f"native SMI publisher size metric is missing: {health}")
    max_state_bytes = int(limits.get("max_state_bytes", 0))
    if max_state_bytes <= 0 or state_path.stat().st_size > max_state_bytes:
        _die(
            "native SMI state exceeded its configured size: "
            f"{state_path.stat().st_size} > {max_state_bytes}"
        )
    expected_detail_limit = int(
        os.environ.get("FAKEGPU_SMI_DETAIL_LIMIT", "64")
    )
    if int(limits.get("detail_entries", -1)) != expected_detail_limit:
        _die(f"native SMI detail limit mismatch: {limits}")
    temporary_pattern = f".{state_path.name}.*.tmp"
    if list(state_path.parent.glob(temporary_pattern)):
        _die("native SMI left a temporary state file")
    device = state["devices"][0]
    if device.get("profile_id") != "a100":
        _die(f"unexpected native SMI profile: {device.get('profile_id')}")
    if int(device.get("allocation_count", 0)) < 2:
        _die("native SMI did not publish allocation counters")
    if os.environ.get("FAKEGPU_NVLINK_GROUPS"):
        topology = state.get("topology") or {}
        device_topology = device.get("topology") or {}
        nvlink = device_topology.get("nvlink") or {}
        if (
            topology.get("source") != "modeled_environment"
            or topology.get("valid") is not True
            or int(topology.get("link_count", 0)) != 2
            or int(nvlink.get("active_links", 0)) != 1
        ):
            _die(f"native SMI topology mismatch: {topology}")
    if os.environ.get("FAKEGPU_FAULT_EVENTS"):
        faults = state.get("faults") or {}
        health = device.get("health") or {}
        if (
            faults.get("source") != "modeled_environment"
            or faults.get("valid") is not True
            or faults.get("status") != "failed"
            or int(faults.get("event_types_total", 0)) != 2
            or int(faults.get("event_count", 0)) != 4
            or health.get("status") != "failed"
            or health.get("hardware_health") != "unobserved"
            or health.get("max_severity") != "critical"
        ):
            _die(f"native SMI fault model mismatch: {faults}")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fakegpu",
            "nvidia-smi",
            "--state",
            str(state_path),
            "--query-gpu",
            (
                "runtime,runtime.backend,profile.id,memory.tracked,"
                "native.io_calls,native.kernel_launches,state.status,"
                "topology.source,nvlink.active_links,nvlink.bandwidth,"
                "health.status,health.hardware,health.max_severity,"
                "health.event_count"
            ),
            "--format",
            "json",
            "-i",
            "0",
        ],
        text=True,
        capture_output=True,
        env=dict(os.environ),
        timeout=10,
    )
    if completed.returncode != 0:
        _die(
            "native SMI query failed: "
            f"rc={completed.returncode} stderr={completed.stderr}"
        )
    query = json.loads(completed.stdout)
    records = query.get("records") or []
    if len(records) != 1:
        _die(f"native SMI query returned {len(records)} records")
    record = records[0]
    if (
        record.get("runtime") != "native"
        or record.get("runtime.backend") != "native_interception"
        or record.get("profile.id") != "a100"
        or "memory.tracked" not in record
        or record.get("state.status") != "running"
        or (
            os.environ.get("FAKEGPU_NVLINK_GROUPS")
            and (
                record.get("topology.source")
                != "modeled_environment"
                or record.get("nvlink.active_links") != 1
            )
        )
        or (
            os.environ.get("FAKEGPU_FAULT_EVENTS")
            and (
                record.get("health.status") != "failed"
                or record.get("health.hardware") != "unobserved"
                or record.get("health.max_severity") != "critical"
                or record.get("health.event_count") != 1
            )
        )
    ):
        _die(f"native SMI query record mismatch: {record}")

    if os.environ.get("FAKEGPU_NVLINK_GROUPS"):
        for view in (
            ("topo", "-m"),
            ("nvlink", "-s"),
            ("events",),
        ):
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "fakegpu",
                    "nvidia-smi",
                    *view,
                    "--state",
                    str(state_path),
                ],
                text=True,
                capture_output=True,
                env=dict(os.environ),
                timeout=10,
            )
            if completed.returncode != 0:
                _die(
                    f"native SMI {view[0]} view failed: "
                    f"rc={completed.returncode} "
                    f"stderr={completed.stderr}"
                )
            if "modeled" not in completed.stdout:
                _die(
                    f"native SMI {view[0]} view omitted model label"
                )
            if view[0] == "events" and (
                "XID_79" not in completed.stdout
                or "Hardware health is unobserved"
                not in completed.stdout
            ):
                _die(
                    "native SMI events view omitted configured "
                    "fault evidence"
                )


def _validate_modeled_nvlink_nvml() -> None:
    if not os.environ.get("FAKEGPU_NVLINK_GROUPS"):
        return

    nvml = _load_libnvidia_ml()
    nvml.nvmlInit.argtypes = []
    nvml.nvmlInit.restype = ctypes.c_int
    nvml.nvmlShutdown.argtypes = []
    nvml.nvmlShutdown.restype = ctypes.c_int
    nvml.nvmlDeviceGetHandleByIndex.argtypes = [
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    nvml.nvmlDeviceGetHandleByIndex.restype = ctypes.c_int
    nvml.nvmlDeviceGetNvLinkState.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_uint),
    ]
    nvml.nvmlDeviceGetNvLinkState.restype = ctypes.c_int
    nvml.nvmlDeviceGetNvLinkCapability.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_uint),
    ]
    nvml.nvmlDeviceGetNvLinkCapability.restype = ctypes.c_int
    nvml.nvmlDeviceGetNvLinkRemoteDeviceType.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_uint),
    ]
    nvml.nvmlDeviceGetNvLinkRemoteDeviceType.restype = ctypes.c_int
    nvml.nvmlDeviceGetNvLinkRemotePciInfo_v2.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.POINTER(NvmlPciInfo),
    ]
    nvml.nvmlDeviceGetNvLinkRemotePciInfo_v2.restype = ctypes.c_int

    if int(nvml.nvmlInit()) != NVML_SUCCESS:
        _die("nvmlInit failed during modeled NVLink validation")
    try:
        handle = ctypes.c_void_p()
        if (
            int(
                nvml.nvmlDeviceGetHandleByIndex(
                    ctypes.c_uint(0),
                    ctypes.byref(handle),
                )
            )
            != NVML_SUCCESS
        ):
            _die("NVML could not resolve GPU 0")

        active = ctypes.c_uint()
        if (
            int(
                nvml.nvmlDeviceGetNvLinkState(
                    handle,
                    ctypes.c_uint(0),
                    ctypes.byref(active),
                )
            )
            != NVML_SUCCESS
            or int(active.value) != 1
        ):
            _die("modeled NVLink 0 is not active through NVML")

        inactive = ctypes.c_uint(1)
        if (
            int(
                nvml.nvmlDeviceGetNvLinkState(
                    handle,
                    ctypes.c_uint(1),
                    ctypes.byref(inactive),
                )
            )
            != NVML_SUCCESS
            or int(inactive.value) != 0
        ):
            _die("unconfigured NVLink 1 is active through NVML")

        capability = ctypes.c_uint()
        if (
            int(
                nvml.nvmlDeviceGetNvLinkCapability(
                    handle,
                    ctypes.c_uint(0),
                    ctypes.c_uint(0),
                    ctypes.byref(capability),
                )
            )
            != NVML_SUCCESS
            or int(capability.value) != 1
        ):
            _die("modeled NVLink P2P capability is unavailable")

        remote_type = ctypes.c_uint(0xFF)
        if (
            int(
                nvml.nvmlDeviceGetNvLinkRemoteDeviceType(
                    handle,
                    ctypes.c_uint(0),
                    ctypes.byref(remote_type),
                )
            )
            != NVML_SUCCESS
            or int(remote_type.value)
            != NVML_NVLINK_DEVICE_TYPE_GPU
        ):
            _die("modeled NVLink remote endpoint is not a GPU")

        remote_pci = NvmlPciInfo()
        if (
            int(
                nvml.nvmlDeviceGetNvLinkRemotePciInfo_v2(
                    handle,
                    ctypes.c_uint(0),
                    ctypes.byref(remote_pci),
                )
            )
            != NVML_SUCCESS
        ):
            _die("modeled NVLink remote PCI query failed")
        bus_id = bytes(remote_pci.busId).split(b"\0", 1)[0].decode()
        if bus_id != "00000000:02:00.0":
            _die(f"modeled NVLink remote PCI mismatch: {bus_id}")
    finally:
        nvml.nvmlShutdown()


def main() -> int:
    libcuda = _load_libcuda()

    libcuda.cuInit.argtypes = [ctypes.c_uint]
    libcuda.cuInit.restype = ctypes.c_int
    libcuda.cuMemAlloc.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_size_t]
    libcuda.cuMemAlloc.restype = ctypes.c_int
    libcuda.cuMemAllocManaged.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_size_t, ctypes.c_uint]
    libcuda.cuMemAllocManaged.restype = ctypes.c_int
    libcuda.cuMemAllocHost.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    libcuda.cuMemAllocHost.restype = ctypes.c_int
    libcuda.cuMemFree.argtypes = [ctypes.c_ulonglong]
    libcuda.cuMemFree.restype = ctypes.c_int
    libcuda.cuMemFreeHost.argtypes = [ctypes.c_void_p]
    libcuda.cuMemFreeHost.restype = ctypes.c_int
    libcuda.cuPointerGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ulonglong]
    libcuda.cuPointerGetAttribute.restype = ctypes.c_int

    _cu_check(int(libcuda.cuInit(0)), what="cuInit(0)")

    dptr = ctypes.c_ulonglong()
    _cu_check(int(libcuda.cuMemAlloc(ctypes.byref(dptr), ctypes.c_size_t(4096))), what="cuMemAlloc")
    mptr = ctypes.c_ulonglong()
    _cu_check(int(libcuda.cuMemAllocManaged(ctypes.byref(mptr), ctypes.c_size_t(4096), ctypes.c_uint(0))), what="cuMemAllocManaged")
    hptr = ctypes.c_void_p()
    _cu_check(int(libcuda.cuMemAllocHost(ctypes.byref(hptr), ctypes.c_size_t(4096))), what="cuMemAllocHost")

    try:
        device_mem_type = _cu_get_ptr_attr(libcuda, int(dptr.value), CU_POINTER_ATTRIBUTE_MEMORY_TYPE, ctypes.c_int)
        device_is_managed = _cu_get_ptr_attr(libcuda, int(dptr.value), CU_POINTER_ATTRIBUTE_IS_MANAGED, ctypes.c_uint)
        if device_mem_type != CU_MEMORYTYPE_DEVICE or device_is_managed != 0:
            _die(f"device allocation attributes mismatch: mem_type={device_mem_type} is_managed={device_is_managed}")

        managed_mem_type = _cu_get_ptr_attr(libcuda, int(mptr.value), CU_POINTER_ATTRIBUTE_MEMORY_TYPE, ctypes.c_int)
        managed_is_managed = _cu_get_ptr_attr(libcuda, int(mptr.value), CU_POINTER_ATTRIBUTE_IS_MANAGED, ctypes.c_uint)
        if managed_mem_type != CU_MEMORYTYPE_UNIFIED or managed_is_managed != 1:
            _die(f"managed allocation attributes mismatch: mem_type={managed_mem_type} is_managed={managed_is_managed}")

        host_mem_type = _cu_get_ptr_attr(libcuda, int(hptr.value), CU_POINTER_ATTRIBUTE_MEMORY_TYPE, ctypes.c_int)
        host_is_managed = _cu_get_ptr_attr(libcuda, int(hptr.value), CU_POINTER_ATTRIBUTE_IS_MANAGED, ctypes.c_uint)
        if host_mem_type != CU_MEMORYTYPE_HOST or host_is_managed != 0:
            _die(f"host allocation attributes mismatch: mem_type={host_mem_type} is_managed={host_is_managed}")

        for label, ptr in (("device", int(dptr.value)), ("managed", int(mptr.value)), ("host", int(hptr.value))):
            ordinal = _cu_get_ptr_attr(libcuda, ptr, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, ctypes.c_int)
            if ordinal != 0:
                _die(f"{label} allocation device ordinal mismatch: expected 0 got {ordinal}")

        _validate_modeled_nvlink_nvml()
        _validate_native_smi_state()
        print("OK: pointer attribute memory types (device/managed/host) validated")
        return 0
    finally:
        libcuda.cuMemFree(ctypes.c_ulonglong(dptr.value))
        libcuda.cuMemFree(ctypes.c_ulonglong(mptr.value))
        libcuda.cuMemFreeHost(hptr)


if __name__ == "__main__":
    raise SystemExit(main())
