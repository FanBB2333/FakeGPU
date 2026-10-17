"""Opt-in, process-local NVML view of local FakeGPU state publishers.

Use this in a dedicated monitoring process, before importing an NVML client::

    with NvmlStateBridge(state_dir="/tmp/fakegpu-smi"):
        import pynvml
        pynvml.nvmlInit()
        try:
            print(pynvml.nvmlDeviceGetCount())
        finally:
            pynvml.nvmlShutdown()

Closing stops refreshes and clears usage/processes. The selected library and
stable device handles remain valid until process exit; this deliberately does
not switch a running client back to hardware NVML. No system files are changed.
"""

from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Callable, Sequence

from ._api import library_dir
from ._smi_state import (
    DEFAULT_STALE_AFTER_SECONDS,
    _discover_state_paths,
    build_inventory,
    load_live_states,
)


class _Device(ctypes.Structure):
    _fields_ = [
        ("uuid", ctypes.c_char_p),
        ("name", ctypes.c_char_p),
        ("profile_id", ctypes.c_char_p),
        ("pci_bus_id", ctypes.c_char_p),
        ("total_memory", ctypes.c_uint64),
        ("used_memory", ctypes.c_uint64),
    ]


class _Process(ctypes.Structure):
    _fields_ = [
        ("device_index", ctypes.c_uint32),
        ("pid", ctypes.c_uint32),
        ("used_memory", ctypes.c_uint64),
    ]


class _Telemetry(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in (
        "valid", "gpu", "memory", "temperature", "power_mw", "sm_clock", "memory_clock", "pstate",
    )]


class _DeviceV2(ctypes.Structure):
    _fields_ = [("device", _Device), ("telemetry", _Telemetry)]


class _Links(ctypes.Structure):
    _fields_ = [
        *((name, ctypes.c_uint32) for name in (
            "valid", "pcie_generation", "pcie_width", "pcie_max_generation", "pcie_max_width", "nvlink_count",
        )),
        *((name, ctypes.c_uint64) for name in (
            "pcie_tx_bytes_per_second", "pcie_rx_bytes_per_second", "nvlink_tx_bytes", "nvlink_rx_bytes",
        )),
    ]


class _DeviceV3(ctypes.Structure):
    _fields_ = [("device", _Device), ("telemetry", _Telemetry), ("links", _Links)]


class _ProcessV2(ctypes.Structure):
    _fields_ = [("process", _Process), ("timestamp_us", ctypes.c_uint64),
                ("gpu", ctypes.c_uint32), ("memory", ctypes.c_uint32)]


def _uint64(value: int) -> int:
    if not 0 <= value < 2**64:
        raise ValueError("state memory is outside the NVML uint64 range")
    return value


def _text(value: str) -> bytes:
    encoded = value.encode("utf-8")
    if not encoded or b"\0" in encoded or len(encoded) > 512:
        raise ValueError("invalid NVML device identity")
    return encoded


class NvmlStateBridge:
    """Expose fresh local publishers through bundled NVML, on macOS or Linux.

    Must be started before pynvml loads any library. Only one bridge may be
    installed per process. ``errors`` lists unreadable state files; stale,
    exited and remote publishers are left out. A missing directory is empty.
    """

    def __init__(
        self,
        *,
        state_dir: str | os.PathLike[str] | None = None,
        state_paths: Sequence[str | os.PathLike[str]] = (),
        refresh_interval: float = 0.25,
        stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS,
        lib_dir: str | os.PathLike[str] | None = None,
        telemetry: str = "none",
        layout: Callable[[], dict | None] | None = None,
    ):
        if telemetry not in {"none", "reference"}:
            raise ValueError("telemetry must be none or reference")
        self.telemetry = telemetry
        # Optional layout-only state (fakegpu.machine.layout): GPUs listed
        # while no process uses them.
        self._layout = layout
        self._reference = None
        if telemetry == "reference":
            from ._monitor_reference import ReferenceTelemetry

            self._reference = ReferenceTelemetry()
        for name, value in (("refresh_interval", refresh_interval),
                            ("stale_after_seconds", stale_after_seconds)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        directory = state_dir or os.environ.get("FAKEGPU_SMI_STATE_DIR")
        self.state_dir = Path(directory).expanduser() if directory else None
        paths = [Path(path).expanduser() for path in state_paths]
        if not paths and self.state_dir is None:
            fallback = os.environ.get("FAKEGPU_SMI_STATE_PATH")
            if fallback:
                paths.append(Path(fallback).expanduser())
        if not paths and self.state_dir is None:
            raise ValueError("provide --state-dir, --state, or FAKEGPU_SMI_STATE_DIR")
        self.state_paths = paths
        self.refresh_interval = refresh_interval
        self.stale_after_seconds = stale_after_seconds
        self.lib_dir = lib_dir
        self.errors: tuple[str, ...] = ()
        self.last_error: str | None = None
        self.device_count = 0  # Active devices in the most recent snapshot.
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._library: ctypes.CDLL | None = None
        self._set_snapshot = None
        # uuid -> [TX base, RX base, last TX, last RX] of the NVLink counters
        self._nvlink_counters: dict[str, list[int]] = {}

    def _inventory(self) -> dict:
        now = time.time_ns()
        paths = _discover_state_paths(
            explicit_paths=self.state_paths, state_dir=self.state_dir, fallback_state=None,
        )
        states, errors = load_live_states(
            paths, stale_after_seconds=self.stale_after_seconds, now_ns=now,
        )
        self.errors = tuple(errors)
        layout = self._layout() if self._layout is not None else None
        if layout is not None:
            states = [layout, *states]
        return build_inventory(states, now_ns=now)

    @staticmethod
    def _arrays(inventory: dict) -> tuple[ctypes.Array, ctypes.Array]:
        devices = inventory["devices"]
        device_rows = (_Device * len(devices))(*(
            _Device(
                _text(d["uuid"]), _text(d["name"]), _text(d["profile_id"]),
                _text(d["pci_bus_id"]), _uint64(d["memory"]["total_bytes"]),
                _uint64(d["memory"]["used_bytes"]),
            ) for d in devices
        ))
        indices = {d["uuid"]: i for i, d in enumerate(devices)}
        # Deduplicate repeated device entries in a publisher while preserving
        # the total reported allocation for that process and device.
        processes: dict[tuple[int, int], int] = {}
        for p in inventory["processes"]:
            # Publishers list every GPU they can see. Like a CUDA context, a
            # process belongs to a GPU once it has used memory there.
            if not p["memory"]["used_bytes"] and not p["memory"].get("reported_peak_bytes"):
                continue
            key = (indices[p["gpu_uuid"]], p["pid"])
            processes[key] = processes.get(key, 0) + p["memory"]["used_bytes"]
        process_rows = (_Process * len(processes))(*(
            _Process(index, pid, _uint64(used))
            for (index, pid), used in sorted(processes.items())
        ))
        return device_rows, process_rows

    def _links(self, device: dict) -> _Links:
        """PCIe link, NVLink count and modeled link traffic of a device."""
        topology = device.get("topology") or {}
        if not topology.get("slot"):
            return _Links()
        pcie = topology.get("pcie") or {}
        links = _Links(
            valid=1,
            pcie_generation=min(6, int(pcie.get("generation") or 0)),
            pcie_width=min(32, int(pcie.get("width") or 0)),
            pcie_max_generation=min(6, int(pcie.get("max_generation") or 0)),
            pcie_max_width=min(32, int(pcie.get("max_width") or 0)),
            nvlink_count=min(18, int((topology.get("nvlink") or {}).get("active_links") or 0)),
        )
        transfer = device.get("transfer")
        # Keep the last cumulative count even when the last publisher exits
        # and only the idle machine layout remains.
        counters = self._nvlink_counters.setdefault(device["uuid"], [0, 0, 0, 0])
        totals = (0, 0)
        if transfer:
            pcie_traffic, nvlink_traffic = transfer["links"]["pcie"], transfer["links"]["nvlink"]
            links.pcie_tx_bytes_per_second = _uint64(pcie_traffic["tx_bytes_per_second"])
            links.pcie_rx_bytes_per_second = _uint64(pcie_traffic["rx_bytes_per_second"])
            # The sum over live publishers drops when one exits; keep the
            # counters NVML reports from going backwards.
            totals = (nvlink_traffic["tx_bytes"], nvlink_traffic["rx_bytes"])
        for position, value in enumerate(totals):
            if value < counters[2 + position]:
                counters[position] += counters[2 + position] - value
            counters[2 + position] = value
        links.nvlink_tx_bytes = _uint64(counters[0] + counters[2])
        links.nvlink_rx_bytes = _uint64(counters[1] + counters[3])
        return links

    def _snapshot_arrays(self, inventory: dict) -> tuple[ctypes.Array, ctypes.Array]:
        devices, processes = self._arrays(inventory)
        sensors, activity = self._reference.sample(inventory) if self._reference is not None else ({}, {})
        device_rows = (_DeviceV3 * len(devices))()
        for i, device in enumerate(devices):
            values = sensors.get(device.uuid.decode())
            sample = _Telemetry(valid=1, **values) if values else _Telemetry()
            device_rows[i] = _DeviceV3(device, sample, self._links(inventory["devices"][i]))
        process_rows = (_ProcessV2 * len(processes))()
        for i, process in enumerate(processes):
            key = (devices[process.device_index].uuid.decode(), process.pid)
            values = activity.get(key, {"timestamp_us": 0, "gpu": 0, "memory": 0})
            process_rows[i] = _ProcessV2(process, **values)
        return device_rows, process_rows

    def refresh(self) -> None:
        """Synchronously publish a snapshot, also useful for non-TUI clients."""
        with self._lock:
            if self._set_snapshot is None:
                raise RuntimeError("NVML bridge is not started")
            try:
                devices, processes = self._snapshot_arrays(self._inventory())
                status = self._set_snapshot(3, len(devices), devices, len(processes), processes)
                if status:
                    raise RuntimeError(f"NVML snapshot rejected (status {status})")
                self.device_count = len(devices)
                self.last_error = None
            except Exception:
                # Never leave a previous allocation looking live after a reader
                # failure. Keep device handles but clear usage and process rows.
                self._set_snapshot(1, 0, None, 0, None)
                self.device_count = 0
                raise

    def start(self) -> NvmlStateBridge:
        if self._library is not None:
            raise RuntimeError("NVML bridge has already been started")
        if sys.platform not in {"darwin", "linux"}:
            raise RuntimeError("NVML state bridge supports macOS and Linux")
        try:
            import pynvml
        except ImportError as exc:
            raise RuntimeError("install monitoring dependencies: pip install 'fakegpu[monitoring]'") from exc
        with pynvml.libLoadLock:
            if pynvml.nvmlLib is not None:
                raise RuntimeError("start the bridge in a fresh process before loading NVML")
            filename = "libnvidia-ml.dylib" if sys.platform == "darwin" else "libnvidia-ml.so.1"
            library = ctypes.CDLL(str(library_dir(lib_dir=self.lib_dir) / filename))
            try:
                setter = library.fakegpuNvmlSetSnapshot
            except AttributeError as exc:
                raise RuntimeError("rebuild/reinstall FakeGPU: native NVML bridge is missing") from exc
            setter.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p,
                               ctypes.c_uint, ctypes.c_void_p]
            setter.restype = ctypes.c_int
            self._set_snapshot = setter
            self.refresh()
            # nvidia-ml-py otherwise tries libnvidia-ml.so.1 even on macOS.
            # Select the bundled library before nvitop imports/caches functions.
            pynvml._nvmlGetFunctionPointer_cache.clear()
            pynvml.nvmlLib = self._library = library
        self._thread = threading.Thread(target=self._run, name="fakegpu-nvml", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.refresh_interval):
            try:
                self.refresh()
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        with self._lock:
            if self._set_snapshot is not None:
                self._set_snapshot(1, 0, None, 0, None)
                self.device_count = 0

    def __enter__(self) -> NvmlStateBridge:
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.close()
