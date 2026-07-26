from __future__ import annotations

import argparse
import atexit
import csv
import io
import json
import math
import os
import platform
import socket
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


LEGACY_SCHEMA_VERSION = "fakegpu.smi_state.v1"
SCHEMA_VERSION = "fakegpu.smi_state.v2"
REPORT_SCHEMA_VERSION = "fakegpu.smi_report.v1"
QUERY_SCHEMA_VERSION = "fakegpu.smi_query.v1"
SUPPORTED_SCHEMA_VERSIONS = frozenset(
    {LEGACY_SCHEMA_VERSION, SCHEMA_VERSION}
)
DEFAULT_STALE_AFTER_SECONDS = 2.0
DEFAULT_PUBLISHER_DETAIL_LIMIT = 64
MAXIMUM_PUBLISHER_DETAIL_LIMIT = 1024
DEFAULT_MAX_STATE_BYTES = 1024 * 1024
MINIMUM_MAX_STATE_BYTES = 64 * 1024
MAXIMUM_MAX_STATE_BYTES = 64 * 1024 * 1024
DEFAULT_NVLINK_BANDWIDTH_GBPS = 900.0
MAXIMUM_NVLINK_BANDWIDTH_GBPS = 1_000_000.0
MAXIMUM_MODELED_NVLINKS_PER_DEVICE = 18
MAXIMUM_MODELED_FAULT_TYPES = 128
MAXIMUM_MODELED_FAULT_COUNT = 1_000_000_000
FAULT_SEVERITY_RANK = {
    "none": 0,
    "info": 1,
    "warning": 2,
    "error": 3,
    "critical": 4,
}


@dataclass(frozen=True, slots=True)
class QueryField:
    path: str
    unit: str | None = None
    divisor: float = 1.0
    precision: int = 0


GPU_QUERY_FIELDS: dict[str, QueryField] = {
    "timestamp": QueryField("timestamp"),
    "host": QueryField("host"),
    "index": QueryField("index"),
    "name": QueryField("name"),
    "uuid": QueryField("uuid"),
    "pci.bus_id": QueryField("pci_bus_id"),
    "topology.source": QueryField("topology.source"),
    "topology.numa_node": QueryField("topology.numa_node"),
    "topology.pcie_generation": QueryField(
        "topology.pcie_generation"
    ),
    "nvlink.active_links": QueryField(
        "topology.nvlink.active_links"
    ),
    "nvlink.peer_count": QueryField(
        "topology.nvlink.peer_count"
    ),
    "nvlink.bandwidth": QueryField(
        "topology.nvlink.aggregate_bandwidth_gbps",
        unit="Gbps",
        precision=1,
    ),
    "health.status": QueryField("health.status"),
    "health.hardware": QueryField("health.hardware_health"),
    "health.max_severity": QueryField("health.max_severity"),
    "health.event_count": QueryField("health.event_count"),
    "health.event_types": QueryField("health.event_types_total"),
    "driver_version": QueryField("software.driver_version"),
    "cuda_version": QueryField("software.cuda_version"),
    "profile.id": QueryField("profile_id"),
    "profile.status": QueryField("profile.profile_status"),
    "architecture": QueryField("architecture"),
    "compute_cap": QueryField("compute_capability"),
    "compiler_target": QueryField("compiler_target"),
    "memory.total": QueryField(
        "memory.total_bytes",
        unit="MiB",
        divisor=2**20,
    ),
    "memory.used": QueryField(
        "memory.used_bytes",
        unit="MiB",
        divisor=2**20,
    ),
    "memory.free": QueryField(
        "memory.free_bytes",
        unit="MiB",
        divisor=2**20,
    ),
    "memory.tracked": QueryField(
        "memory.tracked_bytes",
        unit="MiB",
        divisor=2**20,
    ),
    "memory.reserved": QueryField(
        "memory.reserved_bytes",
        unit="MiB",
        divisor=2**20,
    ),
    "memory.peak": QueryField(
        "memory.reported_peak_bytes",
        unit="MiB",
        divisor=2**20,
    ),
    "memory.inactive_split": QueryField(
        "memory.inactive_split_bytes",
        unit="MiB",
        divisor=2**20,
    ),
    "memory.utilization": QueryField(
        "memory.utilization_percent",
        unit="%",
        precision=3,
    ),
    "memory.headroom": QueryField(
        "memory.headroom_bytes",
        unit="MiB",
        divisor=2**20,
    ),
    "utilization.gpu": QueryField(
        "telemetry.gpu_utilization_percent",
        unit="%",
        precision=1,
    ),
    "temperature.gpu": QueryField(
        "telemetry.temperature_c",
        unit="C",
        precision=1,
    ),
    "fan.speed": QueryField(
        "telemetry.fan_speed_percent",
        unit="%",
        precision=1,
    ),
    "power.draw": QueryField(
        "telemetry.power_usage_mw",
        unit="W",
        divisor=1000,
        precision=1,
    ),
    "power.default_limit": QueryField(
        "profile.typical_power_usage_mw",
        unit="W",
        divisor=1000,
        precision=1,
    ),
    "power.max_limit": QueryField(
        "profile.max_power_limit_mw",
        unit="W",
        divisor=1000,
        precision=1,
    ),
    "sm_count": QueryField("profile.sm_count"),
    "clocks.sm": QueryField("profile.core_clock_mhz", unit="MHz"),
    "clocks.mem": QueryField(
        "profile.memory_clock_mhz",
        unit="MHz",
    ),
    "memory.bus_width": QueryField(
        "profile.memory_bus_width_bits",
        unit="bit",
    ),
    "memory.kind": QueryField("profile.memory_kind"),
    "supported_types": QueryField("profile.supported_types"),
    "allocator.model": QueryField("allocator.model"),
    "allocator.segments": QueryField("allocator.segment_count"),
    "native.io_calls": QueryField("native_activity.io_calls"),
    "native.io_bytes": QueryField(
        "native_activity.io_bytes",
        unit="MiB",
        divisor=2**20,
    ),
    "native.kernel_launches": QueryField(
        "native_activity.kernel_launches"
    ),
    "native.gemm_calls": QueryField("native_activity.gemm_calls"),
    "native.gemm_flops": QueryField("native_activity.gemm_flops"),
    "native.compatibility_events": QueryField(
        "native_activity.compatibility_events"
    ),
    "native.unsupported_api_calls": QueryField(
        "native_activity.unsupported_api_calls"
    ),
    "processes": QueryField("process_count"),
    "fakegpu.version": QueryField("fakegpu.version"),
    "runtime": QueryField("fakegpu.runtime"),
    "runtime.backend": QueryField("fakegpu.backend"),
    "runtime.mode": QueryField("fakegpu.mode"),
    "runtime.stage": QueryField("stages"),
    "tracking.confidence": QueryField("tracking_confidence"),
    "state.status": QueryField("state.status"),
    "state.age": QueryField(
        "state.max_age_seconds",
        unit="s",
        precision=3,
    ),
}

PROCESS_QUERY_FIELDS: dict[str, QueryField] = {
    "timestamp": QueryField("timestamp"),
    "host": QueryField("host"),
    "gpu_index": QueryField("gpu_index"),
    "gpu_name": QueryField("gpu_name"),
    "gpu_uuid": QueryField("gpu_uuid"),
    "pci.bus_id": QueryField("pci_bus_id"),
    "profile.id": QueryField("profile_id"),
    "pid": QueryField("pid"),
    "process_name": QueryField("process_name"),
    "used_gpu_memory": QueryField(
        "memory.used_bytes",
        unit="MiB",
        divisor=2**20,
    ),
    "peak_gpu_memory": QueryField(
        "memory.reported_peak_bytes",
        unit="MiB",
        divisor=2**20,
    ),
    "tracked_gpu_memory": QueryField(
        "memory.tracked_bytes",
        unit="MiB",
        divisor=2**20,
    ),
    "reserved_gpu_memory": QueryField(
        "memory.reserved_bytes",
        unit="MiB",
        divisor=2**20,
    ),
    "stage": QueryField("stage"),
    "status": QueryField("status"),
    "state.age": QueryField(
        "age_seconds",
        unit="s",
        precision=3,
    ),
    "tracking_confidence": QueryField("tracking_confidence"),
    "allocator.model": QueryField("allocator_model"),
    "dispatch.calls": QueryField("dispatch_tracking.operator_calls"),
    "native.io_calls": QueryField("native_activity.io_calls"),
    "native.io_bytes": QueryField(
        "native_activity.io_bytes",
        unit="MiB",
        divisor=2**20,
    ),
    "native.kernel_launches": QueryField(
        "native_activity.kernel_launches"
    ),
    "native.gemm_calls": QueryField("native_activity.gemm_calls"),
    "native.gemm_flops": QueryField("native_activity.gemm_flops"),
    "native.compatibility_events": QueryField(
        "native_activity.compatibility_events"
    ),
    "native.unsupported_api_calls": QueryField(
        "native_activity.unsupported_api_calls"
    ),
    "fakegpu.version": QueryField("fakegpu.version"),
    "runtime.backend": QueryField("fakegpu.backend"),
}


def configured_state_path() -> Path | None:
    explicit = os.environ.get("FAKEGPU_SMI_STATE_PATH")
    if explicit:
        return Path(explicit).expanduser().resolve()
    directory = os.environ.get("FAKEGPU_SMI_STATE_DIR")
    if directory:
        return Path(directory).expanduser().resolve() / f"{os.getpid()}.json"
    return None


def _configured_size_limit(
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return min(maximum, max(minimum, parsed))


def _modeled_device_topology(
    devices: list[dict[str, Any]],
) -> dict[str, Any]:
    groups_text = os.environ.get("FAKEGPU_NVLINK_GROUPS")
    source = "modeled_none"
    configured = bool(groups_text)
    valid = True
    error = ""
    bandwidth_gbps = DEFAULT_NVLINK_BANDWIDTH_GBPS
    pairs: set[tuple[int, int]] = set()
    device_by_index = {
        _nonnegative_int(device.get("index")): device
        for device in devices
    }

    if configured:
        source = "modeled_environment"
        bandwidth_text = os.environ.get(
            "FAKEGPU_NVLINK_BANDWIDTH_GBPS"
        )
        if bandwidth_text:
            try:
                bandwidth_gbps = float(bandwidth_text)
            except ValueError:
                valid = False
            if not (
                0 < bandwidth_gbps <= MAXIMUM_NVLINK_BANDWIDTH_GBPS
            ):
                valid = False
            if not valid:
                error = (
                    "FAKEGPU_NVLINK_BANDWIDTH_GBPS must be a finite "
                    "number greater than 0 and no greater than 1000000"
                )

    if configured and valid:
        for group_index, group_text in enumerate(
            str(groups_text).split(";"),
            start=1,
        ):
            group_text = group_text.strip()
            if not group_text:
                valid = False
                error = (
                    "FAKEGPU_NVLINK_GROUPS contains an empty group"
                )
                break
            members: list[int] = []
            for member_text in group_text.split(","):
                member_text = member_text.strip()
                unsigned_text = (
                    member_text[1:]
                    if member_text.startswith("+")
                    else member_text
                )
                if not unsigned_text.isdigit():
                    valid = False
                else:
                    member = int(member_text)
                    valid = member in device_by_index
                if not valid:
                    error = (
                        f"FAKEGPU_NVLINK_GROUPS group {group_index} "
                        "contains an invalid device index"
                    )
                    break
                members.append(member)
            if not valid:
                break
            members = sorted(set(members))
            if len(members) < 2:
                valid = False
                error = (
                    "each FAKEGPU_NVLINK_GROUPS group must contain at "
                    "least two distinct device indices"
                )
                break
            if (
                len(members) - 1
                > MAXIMUM_MODELED_NVLINKS_PER_DEVICE
            ):
                valid = False
                error = (
                    "an FAKEGPU_NVLINK_GROUPS group exceeds the "
                    "18-link per-device limit"
                )
                break
            for left_offset, left in enumerate(members):
                for right in members[left_offset + 1 :]:
                    pairs.add((left, right))
        if valid and not pairs:
            valid = False
            error = (
                "FAKEGPU_NVLINK_GROUPS did not define any links"
            )

    peers_by_index: dict[int, list[int]] = {
        index: [] for index in device_by_index
    }
    if valid:
        for left, right in sorted(pairs):
            peers_by_index[left].append(right)
            peers_by_index[right].append(left)
        if any(
            len(peers) > MAXIMUM_MODELED_NVLINKS_PER_DEVICE
            for peers in peers_by_index.values()
        ):
            valid = False
            error = (
                "a device belongs to more than 18 modeled NVLink "
                "peer relationships"
            )
            peers_by_index = {
                index: [] for index in device_by_index
            }

    if configured and not valid:
        source = "modeled_environment_invalid"
        bandwidth_gbps = DEFAULT_NVLINK_BANDWIDTH_GBPS
        pairs.clear()

    for index, device in device_by_index.items():
        peers = []
        for link, peer_index in enumerate(
            sorted(peers_by_index[index])
        ):
            remote = device_by_index[peer_index]
            peers.append(
                {
                    "link": link,
                    "index": peer_index,
                    "uuid": str(remote.get("uuid") or "unknown"),
                    "pci_bus_id": str(
                        remote.get("pci_bus_id") or "unknown"
                    ),
                    "bandwidth_gbps": bandwidth_gbps,
                    "active": True,
                    "source": source,
                }
            )
        device["topology"] = {
            "source": source,
            "configured": configured,
            "valid": valid,
            "error": error,
            "numa_node": None,
            "pcie_generation": None,
            "nvlink": {
                "active_links": len(peers),
                "peer_count": len(peers),
                "aggregate_bandwidth_gbps": (
                    len(peers) * bandwidth_gbps
                ),
                "peers": peers,
            },
        }

    links = []
    if valid:
        for left, right in sorted(pairs):
            links.append(
                {
                    "source_index": left,
                    "target_index": right,
                    "source_uuid": str(
                        device_by_index[left].get("uuid") or "unknown"
                    ),
                    "target_uuid": str(
                        device_by_index[right].get("uuid") or "unknown"
                    ),
                    "kind": "NVLink",
                    "active": True,
                    "bandwidth_gbps": bandwidth_gbps,
                    "source": source,
                }
            )
    return {
        "schema_version": "fakegpu.device_topology.v1",
        "source": source,
        "configured": configured,
        "valid": valid,
        "error": error,
        "nvlink_bandwidth_gbps": bandwidth_gbps,
        "link_count": len(links),
        "links": links,
    }


def _modeled_fault_model(
    devices: list[dict[str, Any]],
    *,
    detail_limit: int,
) -> dict[str, Any]:
    events_text = os.environ.get("FAKEGPU_FAULT_EVENTS")
    configured = bool(events_text)
    valid = True
    error = ""
    source = "modeled_none"
    counts: dict[tuple[int, str, str], int] = {}
    device_by_index = {
        _nonnegative_int(device.get("index")): device
        for device in devices
    }

    if configured:
        source = "modeled_environment"
        normalized_text = str(events_text).strip()
        if (
            not normalized_text
            or normalized_text.startswith(";")
            or normalized_text.endswith(";")
        ):
            valid = False
            error = (
                "FAKEGPU_FAULT_EVENTS contains an empty event"
            )
        else:
            for event_index, event_text in enumerate(
                normalized_text.split(";"),
                start=1,
            ):
                fields = [
                    field.strip() for field in event_text.split(":")
                ]
                prefix = (
                    f"FAKEGPU_FAULT_EVENTS event {event_index}"
                )
                if (
                    len(fields) not in {3, 4}
                    or any(not field for field in fields)
                ):
                    valid = False
                    error = (
                        f"{prefix} must use "
                        "DEVICE:CODE:SEVERITY[:COUNT]"
                    )
                    break
                index_text = fields[0]
                unsigned_index = (
                    index_text[1:]
                    if index_text.startswith("+")
                    else index_text
                )
                if (
                    not unsigned_index.isascii()
                    or not unsigned_index.isdigit()
                ):
                    valid = False
                else:
                    device_index = int(index_text)
                    valid = device_index in device_by_index
                if not valid:
                    error = (
                        f"{prefix} contains an invalid device index"
                    )
                    break

                code = fields[1].upper()
                if (
                    len(code) > 64
                    or any(
                        not character.isascii()
                        or (
                            not character.isalnum()
                            and character not in "_.-"
                        )
                        for character in code
                    )
                ):
                    valid = False
                    error = f"{prefix} contains an invalid code"
                    break

                severity = fields[2].lower()
                if FAULT_SEVERITY_RANK.get(severity, 0) == 0:
                    valid = False
                    error = (
                        f"{prefix} severity must be info, warning, "
                        "error, or critical"
                    )
                    break

                count = 1
                if len(fields) == 4:
                    count_text = fields[3]
                    unsigned_count = (
                        count_text[1:]
                        if count_text.startswith("+")
                        else count_text
                    )
                    if (
                        not unsigned_count.isascii()
                        or not unsigned_count.isdigit()
                    ):
                        valid = False
                    else:
                        count = int(count_text)
                        valid = (
                            1
                            <= count
                            <= MAXIMUM_MODELED_FAULT_COUNT
                        )
                    if not valid:
                        error = (
                            f"{prefix} count must be between 1 and "
                            "1000000000"
                        )
                        break

                key = (device_index, code, severity)
                counts[key] = min(
                    MAXIMUM_MODELED_FAULT_COUNT,
                    counts.get(key, 0) + count,
                )
                if len(counts) > MAXIMUM_MODELED_FAULT_TYPES:
                    valid = False
                    error = (
                        "FAKEGPU_FAULT_EVENTS exceeds the "
                        "128-event-type limit"
                    )
                    break

    if configured and not valid:
        source = "modeled_environment_invalid"
        counts.clear()

    events = [
        {
            "device_index": device_index,
            "code": code,
            "severity": severity,
            "count": count,
            "active": True,
            "source": source,
        }
        for (device_index, code, severity), count in counts.items()
    ]
    events.sort(
        key=lambda event: (
            -FAULT_SEVERITY_RANK[event["severity"]],
            event["device_index"],
            event["code"],
        )
    )

    for device_index, device in device_by_index.items():
        device_events = [
            event
            for event in events
            if event["device_index"] == device_index
        ]
        maximum_severity = (
            "error"
            if not valid
            else _maximum_fault_severity(device_events)
        )
        retained = device_events[:detail_limit]
        device["health"] = {
            "source": source,
            "configured": configured,
            "valid": valid,
            "error": error,
            "hardware_health": "unobserved",
            "status": _modeled_health_status(
                valid=valid,
                maximum_severity=maximum_severity,
            ),
            "max_severity": maximum_severity,
            "event_count": sum(
                int(event["count"]) for event in device_events
            ),
            "event_types_total": len(device_events),
            "event_types_retained": len(retained),
            "events": [dict(event) for event in retained],
        }

    maximum_severity = (
        "error" if not valid else _maximum_fault_severity(events)
    )
    retained = events[:detail_limit]
    return {
        "schema_version": "fakegpu.fault_model.v1",
        "source": source,
        "configured": configured,
        "valid": valid,
        "error": error,
        "hardware_health": "unobserved",
        "status": _modeled_health_status(
            valid=valid,
            maximum_severity=maximum_severity,
        ),
        "max_severity": maximum_severity,
        "event_count": sum(int(event["count"]) for event in events),
        "event_types_total": len(events),
        "event_types_retained": len(retained),
        "events": [dict(event) for event in retained],
    }


def _maximum_fault_severity(
    events: Sequence[Mapping[str, Any]],
) -> str:
    return max(
        (
            str(event.get("severity") or "none")
            for event in events
        ),
        key=lambda severity: FAULT_SEVERITY_RANK.get(severity, 0),
        default="none",
    )


def _modeled_health_status(
    *,
    valid: bool,
    maximum_severity: str,
) -> str:
    if not valid:
        return "configuration_error"
    rank = FAULT_SEVERITY_RANK.get(maximum_severity, 0)
    if rank >= 4:
        return "failed"
    if rank >= 2:
        return "degraded"
    if rank == 1:
        return "modeled_event"
    return "no_modeled_faults"


class SmiStatePublisher:
    """Publish FakeCUDA process and device diagnostics for an external viewer."""

    def __init__(
        self,
        path: str | Path,
        snapshot: Callable[[], dict[str, Any]],
        *,
        interval_seconds: float = 0.25,
        runtime_overhead_bytes: int = 0,
        detail_limit: int | None = None,
        max_state_bytes: int | None = None,
    ):
        self.path = Path(path).expanduser().resolve()
        self.snapshot = snapshot
        self.interval_seconds = max(0.05, float(interval_seconds))
        self.runtime_overhead_bytes = max(0, int(runtime_overhead_bytes))
        self.detail_limit = (
            _configured_size_limit(
                "FAKEGPU_SMI_DETAIL_LIMIT",
                default=DEFAULT_PUBLISHER_DETAIL_LIMIT,
                minimum=0,
                maximum=MAXIMUM_PUBLISHER_DETAIL_LIMIT,
            )
            if detail_limit is None
            else min(
                MAXIMUM_PUBLISHER_DETAIL_LIMIT,
                max(0, int(detail_limit)),
            )
        )
        self.max_state_bytes = (
            _configured_size_limit(
                "FAKEGPU_SMI_MAX_STATE_BYTES",
                default=DEFAULT_MAX_STATE_BYTES,
                minimum=MINIMUM_MAX_STATE_BYTES,
                maximum=MAXIMUM_MAX_STATE_BYTES,
            )
            if max_state_bytes is None
            else min(
                MAXIMUM_MAX_STATE_BYTES,
                max(MINIMUM_MAX_STATE_BYTES, int(max_state_bytes)),
            )
        )
        self._software = _software_metadata()
        self._catalogs = _catalog_metadata()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._registered = False
        self._attempted_writes = 0
        self._successful_writes = 0
        self._failed_writes = 0
        self._last_duration_us = 0
        self._max_duration_us = 0
        self._last_serialized_bytes = 0
        self._last_error = ""

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self.publish_once(running=True)
        self._thread = threading.Thread(
            target=self._run,
            name="fakegpu-smi-publisher",
            daemon=True,
        )
        self._thread.start()
        if not self._registered:
            atexit.register(self.stop)
            self._registered = True

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.2, 2 * self.interval_seconds))
        self._thread = None
        try:
            self.publish_once(running=False)
        except Exception:
            pass

    def publish_once(self, *, running: bool) -> dict[str, Any]:
        started_ns = time.perf_counter_ns()
        self._attempted_writes += 1
        try:
            state, serialized_bytes = self._publish_once(
                running=running
            )
        except Exception as exc:
            self._failed_writes += 1
            self._last_duration_us = max(
                0,
                (time.perf_counter_ns() - started_ns) // 1000,
            )
            self._max_duration_us = max(
                self._max_duration_us,
                self._last_duration_us,
            )
            self._last_error = type(exc).__name__
            raise
        self._successful_writes += 1
        self._last_duration_us = max(
            0,
            (time.perf_counter_ns() - started_ns) // 1000,
        )
        self._max_duration_us = max(
            self._max_duration_us,
            self._last_duration_us,
        )
        self._last_serialized_bytes = serialized_bytes
        self._last_error = ""
        return state

    def _publish_once(
        self,
        *,
        running: bool,
    ) -> tuple[dict[str, Any], int]:
        raw = self.snapshot()
        hostname = socket.gethostname()
        devices: list[dict[str, Any]] = []
        for item in raw.get("devices") or []:
            if not isinstance(item, Mapping):
                continue
            current = _nonnegative_int(item.get("current_memory"))
            peak = max(
                current,
                _nonnegative_int(
                    item.get("peak_memory"),
                    default=current,
                ),
            )
            reserved = max(
                current,
                _nonnegative_int(
                    item.get("current_reserved_memory"),
                    default=current,
                ),
            )
            reserved_peak = max(
                reserved,
                peak,
                _nonnegative_int(
                    item.get("peak_reserved_memory"),
                    default=reserved,
                ),
            )
            total = _nonnegative_int(item.get("total_memory"))
            reported = reserved + self.runtime_overhead_bytes
            reported_peak = reserved_peak + self.runtime_overhead_bytes
            if total:
                reported = min(total, reported)
                reported_peak = min(total, reported_peak)
            index = _nonnegative_int(
                item.get("index"),
                default=len(devices),
            )
            profile = _profile_metadata(item)
            free = max(0, total - reported) if total else None
            headroom = (
                max(0, total - reported_peak) if total else None
            )
            largest_allocations = _mapping_list(
                item.get("largest_allocations")
            )
            largest_allocations_total = len(largest_allocations)
            largest_allocations = largest_allocations[
                : self.detail_limit
            ]
            devices.append(
                {
                    "index": index,
                    "name": str(item.get("name", "Fake NVIDIA GPU")),
                    "profile_id": str(item.get("profile_id", "")),
                    "profile": profile,
                    "uuid": _synthetic_uuid(
                        hostname,
                        index,
                        str(item.get("profile_id", "")),
                    ),
                    "pci_bus_id": _synthetic_pci_bus_id(index),
                    "identity_source": "synthetic",
                    "architecture": profile.get("architecture"),
                    "compute_capability": profile.get(
                        "compute_capability"
                    ),
                    "compiler_target": profile.get("compiler_target"),
                    "total_memory": total,
                    "free_memory": free,
                    "tracked_memory": current,
                    "peak_tracked_memory": peak,
                    "reserved_memory": reserved,
                    "peak_reserved_memory": reserved_peak,
                    "inactive_split_bytes": _nonnegative_int(
                        item.get("inactive_split_bytes")
                    ),
                    "segment_count": _nonnegative_int(
                        item.get("segment_count")
                    ),
                    "reported_memory_source": "reserved",
                    "runtime_overhead_bytes": self.runtime_overhead_bytes,
                    "reported_memory": reported,
                    "reported_peak_memory": reported_peak,
                    "memory_utilization_percent": (
                        round(reported / total * 100, 3)
                        if total
                        else None
                    ),
                    "headroom_bytes": headroom,
                    "headroom_percent": (
                        round(headroom / total * 100, 3)
                        if total and headroom is not None
                        else None
                    ),
                    "allocation_count": _nonnegative_int(
                        item.get("allocation_count")
                    ),
                    "free_count": _nonnegative_int(
                        item.get("free_count")
                    ),
                    "allocator_model": str(
                        item.get("allocator_model")
                        or raw.get("allocator_model")
                        or "unknown"
                    ),
                    "current_bytes_by_category": _integer_mapping(
                        item.get("current_bytes_by_category")
                    ),
                    "peak_by_stage": _integer_mapping(
                        item.get("peak_by_stage")
                    ),
                    "reserved_peak_by_stage": _integer_mapping(
                        item.get("reserved_peak_by_stage")
                    ),
                    "largest_allocations": largest_allocations,
                    "largest_allocations_total": (
                        largest_allocations_total
                    ),
                    "largest_allocations_retained": len(
                        largest_allocations
                    ),
                    "telemetry": {
                        "gpu_utilization_percent": None,
                        "temperature_c": None,
                        "fan_speed_percent": None,
                        "power_usage_mw": None,
                        "source": "hardware_telemetry_unavailable",
                    },
                }
            )
        dispatch_tracking = (
            dict(raw.get("dispatch_tracking") or {})
            if isinstance(raw.get("dispatch_tracking"), Mapping)
            else {}
        )
        topology = _modeled_device_topology(devices)
        faults = _modeled_fault_model(
            devices,
            detail_limit=self.detail_limit,
        )
        state = {
            "schema_version": SCHEMA_VERSION,
            "timestamp_ns": time.time_ns(),
            "hostname": hostname,
            "pid": os.getpid(),
            "process_name": _process_name(),
            "runtime": "fakecuda",
            "fakegpu": {
                "version": _fakegpu_version(),
                "runtime": "fakecuda",
                "backend": str(
                    raw.get("runtime_backend") or "unknown"
                ),
                "mode": os.environ.get("FAKEGPU_MODE", "simulate"),
                "oom_policy": os.environ.get(
                    "FAKEGPU_OOM_POLICY",
                    "default",
                ),
                "unsupported_api_policy": os.environ.get(
                    "FAKEGPU_UNSUPPORTED_API",
                    "default",
                ),
                "distributed_mode": os.environ.get(
                    "FAKEGPU_DIST_MODE",
                    "disabled",
                ),
                "memory_tracking_enabled": bool(
                    raw.get("memory_tracking_enabled", True)
                ),
                "dispatch_memory_tracking_enabled": bool(
                    dispatch_tracking.get("enabled", False)
                ),
                **self._catalogs,
            },
            "software": dict(self._software),
            "publisher": {
                "interval_seconds": self.interval_seconds,
                "runtime_overhead_bytes": (
                    self.runtime_overhead_bytes
                ),
                "source": "python_runtime",
                "health": {
                    "attempted_writes": self._attempted_writes,
                    "successful_writes": (
                        self._successful_writes + 1
                    ),
                    "failed_writes": self._failed_writes,
                    "last_duration_us": self._last_duration_us,
                    "max_duration_us": self._max_duration_us,
                    "last_serialized_bytes": (
                        self._last_serialized_bytes
                    ),
                    "last_error": self._last_error,
                },
                "limits": {
                    "detail_entries": self.detail_limit,
                    "max_state_bytes": self.max_state_bytes,
                },
            },
            "running": bool(running),
            "tracking_confidence": raw.get(
                "tracking_confidence", "C2_torch_tensor_lifetime"
            ),
            "stage": str(
                raw.get("stage")
                or os.environ.get("FAKEGPU_PREFLIGHT_STAGE")
                or "unknown"
            ),
            "allocator_model": str(
                raw.get("allocator_model") or "unknown"
            ),
            "dispatch_tracking": dispatch_tracking,
            "topology": topology,
            "faults": faults,
            "devices": devices,
        }
        serialized_bytes = _atomic_write_json(
            self.path,
            state,
            max_bytes=self.max_state_bytes,
        )
        return state, serialized_bytes

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.publish_once(running=True)
            except Exception:
                continue


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fakegpu nvidia-smi",
        description=(
            "Inspect FakeCUDA devices, processes, profiles, memory, runtime "
            "configuration, modeled topology, and simulated telemetry."
        ),
    )
    parser.add_argument(
        "view",
        nargs="?",
        choices=("topo", "nvlink", "events"),
        help=(
            "Show an NVIDIA-style modeled topology matrix, NVLink "
            "status, or health and reliability events."
        ),
    )
    parser.add_argument("--state", action="append", default=[])
    parser.add_argument("--state-dir")
    parser.add_argument("--include-exited", action="store_true")
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--json",
        action="store_true",
        help="Emit the complete state and normalized inventory as JSON.",
    )
    output_group.add_argument(
        "-L",
        "--list-gpus",
        action="store_true",
        help="List simulated GPUs with UUID, profile, and compute capability.",
    )
    output_group.add_argument(
        "-q",
        "--detail",
        action="store_true",
        help="Show detailed FakeGPU runtime, profile, allocator, and process data.",
    )
    output_group.add_argument(
        "--query-gpu",
        metavar="FIELDS",
        help="Query comma-separated GPU fields.",
    )
    output_group.add_argument(
        "--query-compute-apps",
        metavar="FIELDS",
        help="Query comma-separated process fields.",
    )
    parser.add_argument(
        "-m",
        "--matrix",
        action="store_true",
        help="Show the topology matrix; valid with the topo view.",
    )
    parser.add_argument(
        "-s",
        "--status",
        action="store_true",
        help="Show link status; valid with the nvlink view.",
    )
    parser.add_argument(
        "--format",
        default=None,
        help=(
            "Query output format: csv, csv,noheader, csv,nounits, or json."
        ),
    )
    parser.add_argument(
        "-i",
        "--id",
        action="append",
        default=[],
        metavar="ID",
        help=(
            "Select GPU index, UUID, PCI bus ID, or profile ID; "
            "comma-separated values are accepted."
        ),
    )
    parser.add_argument(
        "--stale-after-seconds",
        type=_positive_float,
        default=DEFAULT_STALE_AFTER_SECONDS,
        help="Mark running state files older than this threshold as stale.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"FakeGPU-SMI {_fakegpu_version()}",
    )
    parser.add_argument(
        "--help-query-gpu",
        action="store_true",
        help="List supported --query-gpu fields and exit.",
    )
    parser.add_argument(
        "--help-query-compute-apps",
        action="store_true",
        help="List supported --query-compute-apps fields and exit.",
    )
    parser.add_argument(
        "-l",
        "--loop",
        type=_positive_float,
        metavar="SECONDS",
        help="Refresh repeatedly at the given interval; JSON output becomes NDJSON.",
    )
    parser.add_argument(
        "--count",
        type=_positive_int,
        help="Stop after this many refreshes; requires --loop.",
    )
    args = parser.parse_args(argv)

    if args.help_query_gpu:
        print(_render_query_help("gpu", GPU_QUERY_FIELDS))
        return 0
    if args.help_query_compute_apps:
        print(_render_query_help("compute-apps", PROCESS_QUERY_FIELDS))
        return 0
    if args.count is not None and args.loop is None:
        parser.error("--count requires --loop")
    if args.matrix and args.view != "topo":
        parser.error("-m/--matrix requires the topo view")
    if args.status and args.view != "nvlink":
        parser.error("-s/--status requires the nvlink view")
    if args.view and (
        args.json
        or args.list_gpus
        or args.detail
        or args.query_gpu
        or args.query_compute_apps
    ):
        parser.error(
            "topo, nvlink, and events views cannot be combined with another "
            "output mode"
        )
    if args.format is not None and not (
        args.query_gpu or args.query_compute_apps
    ):
        parser.error("--format requires --query-gpu or --query-compute-apps")
    query_format = _parse_query_format(
        args.format or "csv",
        parser=parser,
    )
    gpu_query_fields = _parse_query_fields(
        args.query_gpu,
        available=GPU_QUERY_FIELDS,
        parser=parser,
    )
    process_query_fields = _parse_query_fields(
        args.query_compute_apps,
        available=PROCESS_QUERY_FIELDS,
        parser=parser,
    )
    selectors = _device_selectors(args.id)

    explicit_paths = [Path(value).expanduser().resolve() for value in args.state]
    state_dir_text = args.state_dir or os.environ.get("FAKEGPU_SMI_STATE_DIR")
    state_dir = Path(state_dir_text).expanduser().resolve() if state_dir_text else None
    fallback_state_text = None
    if not explicit_paths and state_dir is None:
        fallback_state_text = os.environ.get("FAKEGPU_SMI_STATE_PATH")
    fallback_state = (
        Path(fallback_state_text).expanduser().resolve()
        if fallback_state_text
        else None
    )
    if not explicit_paths and state_dir is None and fallback_state is None:
        parser.error("provide --state, --state-dir, or FAKEGPU_SMI_STATE_PATH")

    refresh = 0
    saw_states = False
    try:
        while True:
            paths = _discover_state_paths(
                explicit_paths=explicit_paths,
                state_dir=state_dir,
                fallback_state=fallback_state,
            )
            states, errors = _load_states(
                paths,
                include_exited=bool(args.include_exited),
            )
            saw_states = saw_states or bool(states)
            inventory = build_inventory(
                states,
                device_selectors=selectors,
                stale_after_seconds=args.stale_after_seconds,
            )

            if refresh and not args.json and query_format["kind"] != "json":
                if sys.stdout.isatty():
                    sys.stdout.write("\x1b[2J\x1b[H")
                else:
                    print()
            if args.json:
                payload = {
                    "schema_version": REPORT_SCHEMA_VERSION,
                    "generated_at_ns": time.time_ns(),
                    "inventory": inventory,
                    "states": states,
                    "errors": errors,
                }
                if args.loop is None:
                    print(json.dumps(payload, indent=2, sort_keys=True))
                else:
                    print(json.dumps(payload, sort_keys=True))
            elif args.list_gpus:
                print(render_gpu_list(inventory, errors=errors))
            elif args.detail:
                print(render_detail(inventory, errors=errors))
            elif args.view == "topo":
                print(render_topology_matrix(inventory, errors=errors))
            elif args.view == "nvlink":
                print(render_nvlink_status(inventory, errors=errors))
            elif args.view == "events":
                print(render_health_events(inventory, errors=errors))
            elif gpu_query_fields:
                print(
                    render_query(
                        inventory["devices"],
                        fields=gpu_query_fields,
                        available=GPU_QUERY_FIELDS,
                        query_kind="gpu",
                        output_format=query_format,
                    )
                )
            elif process_query_fields:
                print(
                    render_query(
                        inventory["processes"],
                        fields=process_query_fields,
                        available=PROCESS_QUERY_FIELDS,
                        query_kind="compute-apps",
                        output_format=query_format,
                    )
                )
            else:
                print(
                    render_table(
                        states,
                        errors=errors,
                        device_selectors=selectors,
                        stale_after_seconds=args.stale_after_seconds,
                        inventory=inventory,
                    )
                )
            sys.stdout.flush()

            refresh += 1
            if args.loop is None:
                break
            if args.count is not None and refresh >= args.count:
                break
            time.sleep(args.loop)
    except KeyboardInterrupt:
        pass
    return 0 if saw_states else 1


def _discover_state_paths(
    *,
    explicit_paths: Sequence[Path],
    state_dir: Path | None,
    fallback_state: Path | None,
) -> list[Path]:
    paths = list(explicit_paths)
    if state_dir is not None:
        paths.extend(sorted(state_dir.glob("*.json")))
    if fallback_state is not None:
        paths.append(fallback_state)
    return list(dict.fromkeys(paths))


def _load_states(
    paths: Sequence[Path],
    *,
    include_exited: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    states: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in paths:
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                raise ValueError("state root must be an object")
            if state.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
                raise ValueError("unsupported schema")
            devices = state.get("devices")
            if not isinstance(devices, list) or any(
                not isinstance(item, Mapping) for item in devices
            ):
                raise ValueError("devices must be an array of objects")
            if include_exited or bool(state.get("running")):
                states.append(state)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
    return states, errors


def build_inventory(
    states: Sequence[Mapping[str, Any]],
    *,
    device_selectors: Sequence[str] = (),
    stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS,
    now_ns: int | None = None,
) -> dict[str, Any]:
    """Normalize and aggregate process state into device and runtime records."""

    current_ns = int(now_ns if now_ns is not None else time.time_ns())
    selectors = tuple(str(value).casefold() for value in device_selectors)
    device_rows: dict[tuple[str, str], dict[str, Any]] = {}
    process_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []

    for state in states:
        host = str(state.get("hostname") or "localhost")
        pid = _nonnegative_int(state.get("pid"))
        process_name = str(state.get("process_name") or "python")
        stage = str(state.get("stage") or "unknown")
        confidence = str(
            state.get("tracking_confidence") or "unknown"
        )
        age_seconds = _state_age_seconds(state, now_ns=current_ns)
        status = _state_status(
            state,
            age_seconds=age_seconds,
            stale_after_seconds=stale_after_seconds,
        )
        fakegpu = _state_fakegpu_metadata(state)
        software = _state_software_metadata(state)
        dispatch_tracking = (
            dict(state.get("dispatch_tracking") or {})
            if isinstance(state.get("dispatch_tracking"), Mapping)
            else {}
        )
        selected_device = False
        timestamp_ns = _nonnegative_int(state.get("timestamp_ns"))
        timestamp = _format_timestamp_ns(timestamp_ns)
        raw_devices = state.get("devices") or []
        for raw_device in raw_devices:
            if not isinstance(raw_device, Mapping):
                continue
            device = _normalize_device(raw_device, host=host)
            if selectors and not _matches_device(device, selectors):
                continue
            selected_device = True
            process = {
                "timestamp_ns": timestamp_ns,
                "timestamp": timestamp,
                "host": host,
                "gpu_index": device["index"],
                "gpu_name": device["name"],
                "gpu_uuid": device["uuid"],
                "pci_bus_id": device["pci_bus_id"],
                "profile_id": device["profile_id"],
                "pid": pid,
                "process_name": process_name,
                "stage": stage,
                "status": status,
                "age_seconds": age_seconds,
                "running": bool(state.get("running")),
                "tracking_confidence": confidence,
                "allocator_model": device["allocator_model"],
                "memory": dict(device["memory"]),
                "fakegpu": dict(fakegpu),
                "software": dict(software),
                "dispatch_tracking": dispatch_tracking,
                "native_activity": dict(device["native_activity"]),
            }
            process_rows.append(process)

            key = (host, str(device["uuid"]))
            aggregate = device_rows.setdefault(
                key,
                _empty_device_aggregate(
                    host=host,
                    device=device,
                    fakegpu=fakegpu,
                    software=software,
                ),
            )
            aggregate["_names"].add(device["name"])
            aggregate["_profile_ids"].add(device["profile_id"])
            aggregate["_statuses"].add(status)
            aggregate["_stages"].add(stage)
            aggregate["_tracking"].add(confidence)
            if age_seconds is not None:
                aggregate["_ages"].append(age_seconds)
            aggregate["timestamp_ns"] = max(
                int(aggregate["timestamp_ns"]),
                timestamp_ns,
            )
            aggregate["process_count"] += 1
            if status == "stale":
                aggregate["stale_process_count"] += 1
            if status == "exited":
                aggregate["exited_process_count"] += 1
            _accumulate_memory(
                aggregate["memory"],
                device["memory"],
            )
            _accumulate_allocator(
                aggregate["allocator"],
                device,
                pid=pid,
                process_name=process_name,
            )
            _accumulate_native_activity(
                aggregate["native_activity"],
                device["native_activity"],
                pid=pid,
                process_name=process_name,
            )

        if selected_device or (not selectors and not raw_devices):
            runtime_rows.append(
                {
                    "schema_version": state.get("schema_version"),
                    "timestamp_ns": timestamp_ns,
                    "timestamp": timestamp,
                    "host": host,
                    "pid": pid,
                    "process_name": process_name,
                    "stage": stage,
                    "status": status,
                    "age_seconds": age_seconds,
                    "tracking_confidence": confidence,
                    "allocator_model": str(
                        state.get("allocator_model") or "unknown"
                    ),
                    "fakegpu": fakegpu,
                    "software": software,
                    "publisher": (
                        dict(state.get("publisher") or {})
                        if isinstance(
                            state.get("publisher"),
                            Mapping,
                        )
                        else {}
                    ),
                    "dispatch_tracking": dispatch_tracking,
                }
            )

    devices = [
        _finalize_device_aggregate(item)
        for item in device_rows.values()
    ]
    devices.sort(
        key=lambda item: (
            str(item["host"]),
            int(item["index"]),
            str(item["uuid"]),
        )
    )
    process_rows.sort(
        key=lambda item: (
            str(item["host"]),
            int(item["gpu_index"]),
            int(item["pid"]),
        )
    )
    runtime_rows.sort(
        key=lambda item: (str(item["host"]), int(item["pid"]))
    )
    return {
        "device_count": len(devices),
        "process_count": len(process_rows),
        "runtime_count": len(runtime_rows),
        "devices": devices,
        "processes": process_rows,
        "runtimes": runtime_rows,
    }


def render_table(
    states: Sequence[dict[str, Any]],
    *,
    errors: Sequence[str] = (),
    device_selectors: Sequence[str] = (),
    stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS,
    inventory: Mapping[str, Any] | None = None,
) -> str:
    if inventory is None:
        inventory = build_inventory(
            states,
            device_selectors=device_selectors,
            stale_after_seconds=stale_after_seconds,
        )
    versions = sorted(
        {
            str(item["fakegpu"].get("version") or "unknown")
            for item in inventory["runtimes"]
        }
    )
    cuda_versions = sorted(
        {
            str(item["software"].get("cuda_version") or "N/A")
            for item in inventory["runtimes"]
        }
    )
    lines = [
        "FakeGPU-SMI (simulated CUDA telemetry)",
        (
            f"FakeGPU Version: {', '.join(versions) if versions else 'unknown'}"
            f" | CUDA Version: "
            f"{', '.join(cuda_versions) if cuda_versions else 'N/A'}"
            " | Hardware-only telemetry: N/A"
        ),
    ]
    if not inventory["devices"]:
        lines.append("No published FakeCUDA processes found.")
    else:
        lines.extend(
            [
                "Devices:",
                "| Host | GPU | Profile | CC | Name | UUID | Current / Total | Memory Util | Processes |",
                "|---|---:|---|---|---|---|---:|---:|---:|",
            ]
        )
        for item in inventory["devices"]:
            memory = item["memory"]
            utilization = memory["utilization_percent"]
            utilization_text = (
                f"{utilization:.1f}%"
                if utilization is not None
                else "N/A"
            )
            lines.append(
                f"| {_table_cell(item['host'])} | {item['index']} | "
                f"{_table_cell(item['profile_id'])} | "
                f"{_table_cell(item['compute_capability'] or 'N/A')} | "
                f"{_table_cell(item['name'])} | "
                f"{_table_cell(item['uuid'])} | "
                f"{_mib(memory['used_bytes'])} MiB / "
                f"{_mib(memory['total_bytes'])} MiB | "
                f"{utilization_text} | {item['process_count']} |"
            )
        lines.extend(
            [
                "Processes:",
                "| Host | GPU | Profile | PID | Process | Stage | Simulated current | Simulated peak | Allocated current | Allocated peak | Reserved current | Reserved peak | Confidence | Status | Age |",
                "|---|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|",
            ]
        )
        for item in inventory["processes"]:
            memory = item["memory"]
            suffix = " (exited)" if item["status"] == "exited" else ""
            lines.append(
                f"| {_table_cell(item['host'])} | "
                f"{item['gpu_index']} | "
                f"{_table_cell(item['profile_id'])} | {item['pid']} | "
                f"{_table_cell(item['process_name'] + suffix)} | "
                f"{_table_cell(item['stage'])} | "
                f"{_mib(memory['used_bytes'])} MiB | "
                f"{_mib(memory['reported_peak_bytes'])} MiB | "
                f"{_mib(memory['tracked_bytes'])} MiB | "
                f"{_mib(memory['tracked_peak_bytes'])} MiB | "
                f"{_mib(memory['reserved_bytes'])} MiB | "
                f"{_mib(memory['reserved_peak_bytes'])} MiB | "
                f"{_table_cell(item['tracking_confidence'])} | "
                f"{item['status']} | {_format_age(item['age_seconds'])} |"
            )
    for error in errors:
        lines.append(f"warning: {error}")
    return "\n".join(lines)


def render_gpu_list(
    inventory: Mapping[str, Any],
    *,
    errors: Sequence[str] = (),
) -> str:
    lines = []
    for item in inventory.get("devices") or []:
        lines.append(
            f"GPU {item['index']} @ {item['host']}: {item['name']} "
            f"(UUID: {item['uuid']}, Profile: {item['profile_id']}, "
            f"CC: {item['compute_capability'] or 'N/A'}, "
            f"PCI: {item['pci_bus_id']})"
        )
    if not lines:
        lines.append("No published FakeCUDA GPUs found.")
    for error in errors:
        lines.append(f"warning: {error}")
    return "\n".join(lines)


def render_topology_matrix(
    inventory: Mapping[str, Any],
    *,
    errors: Sequence[str] = (),
) -> str:
    devices = list(inventory.get("devices") or [])
    lines = ["FakeGPU modeled topology matrix"]
    if not devices:
        lines.append("No published FakeCUDA GPUs found.")
    else:
        devices_by_host: dict[str, list[Mapping[str, Any]]] = {}
        for device in devices:
            devices_by_host.setdefault(
                str(device.get("host") or "localhost"),
                [],
            ).append(device)
        for host, host_devices in sorted(devices_by_host.items()):
            host_devices.sort(
                key=lambda item: (
                    _nonnegative_int(item.get("index")),
                    str(item.get("uuid") or ""),
                )
            )
            if len(devices_by_host) > 1 or lines[-1] != lines[0]:
                lines.append("")
            lines.append(f"Host: {host}")
            labels = _topology_labels(host_devices)
            rows = [
                [
                    "",
                    *labels,
                    "CPU Affinity",
                    "NUMA Affinity",
                ]
            ]
            for row_index, device in enumerate(host_devices):
                topology = device["topology"]
                rows.append(
                    [
                        labels[row_index],
                        *[
                            _topology_relation(device, target)
                            for target in host_devices
                        ],
                        "N/A",
                        _display_value(topology.get("numa_node")),
                    ]
                )
            widths = [
                max(len(str(row[column])) for row in rows)
                for column in range(len(rows[0]))
            ]
            for row in rows:
                lines.append(
                    "  ".join(
                        str(value).ljust(widths[column])
                        for column, value in enumerate(row)
                    ).rstrip()
                )
            source_values = sorted(
                {
                    str(
                        device["topology"].get("source")
                        or "modeled_none"
                    )
                    for device in host_devices
                }
            )
            lines.append(
                "Topology source: "
                + ", ".join(source_values)
                + " (modeled, not measured)"
            )
            config_errors = sorted(
                {
                    str(device["topology"].get("error") or "")
                    for device in host_devices
                    if device["topology"].get("error")
                }
            )
            for config_error in config_errors:
                lines.append(f"Configuration error: {config_error}")
        lines.extend(
            [
                "",
                "Legend:",
                "  X    Self",
                "  NV1  One active modeled NVLink peer relationship",
                (
                    "  PIX  Same host; PCIe hierarchy is not "
                    "measured"
                ),
            ]
        )
    for error in errors:
        lines.append(f"warning: {error}")
    return "\n".join(lines)


def render_nvlink_status(
    inventory: Mapping[str, Any],
    *,
    errors: Sequence[str] = (),
) -> str:
    devices = list(inventory.get("devices") or [])
    lines = [
        "FakeGPU modeled NVLink status",
        "Bandwidth values are configuration inputs, not measurements.",
    ]
    if not devices:
        lines.append("No published FakeCUDA GPUs found.")
    for device in devices:
        topology = device["topology"]
        nvlink = topology["nvlink"]
        lines.append("")
        lines.append(
            f"GPU {device['index']} @ {device['host']}: "
            f"{device['name']} ({device['uuid']})"
        )
        lines.append(
            "  Source: "
            f"{topology.get('source') or 'modeled_none'}"
        )
        if topology.get("error"):
            lines.append(
                f"  Configuration error: {topology['error']}"
            )
        active_peers = [
            peer
            for peer in nvlink.get("peers") or []
            if peer.get("active")
        ]
        if not active_peers:
            lines.append("  No active modeled NVLink peers.")
            continue
        for peer in sorted(
            active_peers,
            key=lambda item: _nonnegative_int(item.get("link")),
        ):
            lines.append(
                f"  Link {peer['link']}: Active -> "
                f"GPU {peer['index']} ({peer['uuid']}, "
                f"{peer['pci_bus_id']}), "
                f"{_format_gbps(peer.get('bandwidth_gbps'))} Gbps"
            )
    for error in errors:
        lines.append(f"warning: {error}")
    return "\n".join(lines)


def render_health_events(
    inventory: Mapping[str, Any],
    *,
    errors: Sequence[str] = (),
) -> str:
    records = _health_event_records(inventory)
    maximum_severity = _maximum_fault_severity(records)
    rank = FAULT_SEVERITY_RANK.get(maximum_severity, 0)
    reliability_status = (
        "failed"
        if rank >= 4
        else "degraded"
        if rank >= 2
        else "informational"
        if rank == 1
        else "no_events"
    )
    lines = [
        "FakeGPU health and reliability events",
        (
            "Hardware health is unobserved; modeled faults and runtime "
            "evidence are reported separately."
        ),
        (
            f"Reliability status: {reliability_status} | "
            f"maximum severity: {maximum_severity} | "
            f"{len(records)} event types / "
            f"{sum(_nonnegative_int(item.get('count')) for item in records)} "
            "occurrences"
        ),
    ]
    if records:
        lines.extend(
            [
                "| Severity | Kind | Host | GPU | Code | Count | Source | Detail |",
                "|---|---|---|---:|---|---:|---|---|",
            ]
        )
        for record in records:
            lines.append(
                f"| {_table_cell(record['severity'])} | "
                f"{_table_cell(record['kind'])} | "
                f"{_table_cell(record['host'])} | "
                f"{_table_cell(record['gpu'])} | "
                f"{_table_cell(record['code'])} | "
                f"{record['count']} | "
                f"{_table_cell(record['source'])} | "
                f"{_table_cell(record['detail'])} |"
            )
    else:
        lines.append(
            "No modeled fault or runtime reliability events found."
        )
    if any(
        _nonnegative_int(device["health"].get("event_types_total"))
        > _nonnegative_int(
            device["health"].get("event_types_retained")
        )
        for device in inventory.get("devices") or []
    ):
        lines.append(
            "Some modeled fault details were omitted by "
            "FAKEGPU_SMI_DETAIL_LIMIT."
        )
    for error in errors:
        lines.append(f"warning: {error}")
    return "\n".join(lines)


def _health_event_records(
    inventory: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    configuration_keys: set[tuple[str, str, str]] = set()
    for device in inventory.get("devices") or []:
        host = str(device.get("host") or "localhost")
        gpu = _nonnegative_int(device.get("index"))
        health = device["health"]
        if health.get("error"):
            detail = str(health["error"])
            key = (host, "FAULT_CONFIG", detail)
            if key not in configuration_keys:
                configuration_keys.add(key)
                records.append(
                    {
                        "severity": "error",
                        "kind": "configuration",
                        "host": host,
                        "gpu": "N/A",
                        "code": "FAULT_CONFIG",
                        "count": 1,
                        "source": health.get("source")
                        or "modeled_environment_invalid",
                        "detail": detail,
                    }
                )
        for event in health.get("events") or []:
            records.append(
                {
                    "severity": str(
                        event.get("severity") or "info"
                    ),
                    "kind": "modeled_fault",
                    "host": host,
                    "gpu": gpu,
                    "code": str(event.get("code") or "UNKNOWN"),
                    "count": _nonnegative_int(event.get("count")),
                    "source": str(
                        event.get("source") or health.get("source")
                    ),
                    "detail": (
                        "configured through FAKEGPU_FAULT_EVENTS"
                    ),
                }
            )
        topology = device["topology"]
        if topology.get("error"):
            detail = str(topology["error"])
            key = (host, "NVLINK_CONFIG", detail)
            if key not in configuration_keys:
                configuration_keys.add(key)
                records.append(
                    {
                        "severity": "error",
                        "kind": "configuration",
                        "host": host,
                        "gpu": "N/A",
                        "code": "NVLINK_CONFIG",
                        "count": 1,
                        "source": topology.get("source")
                        or "modeled_environment_invalid",
                        "detail": detail,
                    }
                )
        for event in device["native_activity"].get(
            "unsupported_apis"
        ) or []:
            records.append(
                {
                    "severity": "warning",
                    "kind": "compatibility",
                    "host": host,
                    "gpu": gpu,
                    "code": str(
                        event.get("operation") or "UNKNOWN_API"
                    ),
                    "count": _nonnegative_int(event.get("count")),
                    "source": "native_activity",
                    "detail": (
                        f"{event.get('behavior') or 'unknown'}; "
                        f"policy {event.get('policy') or 'unknown'}"
                    ),
                }
            )
    for runtime in inventory.get("runtimes") or []:
        publisher = runtime.get("publisher") or {}
        health = (
            publisher.get("health") or {}
            if isinstance(publisher, Mapping)
            else {}
        )
        failures = _nonnegative_int(health.get("failed_writes"))
        if failures:
            records.append(
                {
                    "severity": "error",
                    "kind": "publisher",
                    "host": str(
                        runtime.get("host") or "localhost"
                    ),
                    "gpu": "N/A",
                    "code": str(
                        health.get("last_error")
                        or "PUBLISH_FAILURE"
                    ),
                    "count": failures,
                    "source": str(
                        publisher.get("source") or "publisher"
                    ),
                    "detail": (
                        f"pid {runtime.get('pid', 0)}; "
                        f"{health.get('successful_writes', 0)} "
                        "successful writes"
                    ),
                }
            )
        if runtime.get("status") == "stale":
            records.append(
                {
                    "severity": "warning",
                    "kind": "state",
                    "host": str(
                        runtime.get("host") or "localhost"
                    ),
                    "gpu": "N/A",
                    "code": "STALE_STATE",
                    "count": 1,
                    "source": "state_freshness",
                    "detail": (
                        f"pid {runtime.get('pid', 0)}; age "
                        f"{_format_age(runtime.get('age_seconds'))}"
                    ),
                }
            )
    records.sort(
        key=lambda record: (
            -FAULT_SEVERITY_RANK.get(
                str(record.get("severity") or "none"),
                0,
            ),
            str(record.get("host") or ""),
            str(record.get("gpu") or ""),
            str(record.get("kind") or ""),
            str(record.get("code") or ""),
        )
    )
    return records


def render_detail(
    inventory: Mapping[str, Any],
    *,
    errors: Sequence[str] = (),
) -> str:
    lines = ["FakeGPU-SMI detailed report"]
    runtimes = list(inventory.get("runtimes") or [])
    if runtimes:
        lines.append("Runtime processes:")
    for item in runtimes:
        fakegpu = item["fakegpu"]
        software = item["software"]
        dispatch = item["dispatch_tracking"]
        publisher = item["publisher"]
        publisher_health = (
            dict(publisher.get("health") or {})
            if isinstance(publisher.get("health"), Mapping)
            else {}
        )
        publisher_limits = (
            dict(publisher.get("limits") or {})
            if isinstance(publisher.get("limits"), Mapping)
            else {}
        )
        publisher_last_size = _format_bytes(
            publisher_health.get("last_serialized_bytes")
        )
        publisher_max_size = _format_bytes(
            publisher_limits.get("max_state_bytes")
        )
        capability = fakegpu.get("native_capabilities") or {}
        profile_catalog = fakegpu.get("profile_catalog") or {}
        lines.extend(
            [
                (
                    f"  [{item['host']} pid={item['pid']}] "
                    f"{item['process_name']}"
                ),
                (
                    "    State: "
                    f"{item['status']}, age "
                    f"{_format_age(item['age_seconds'])}, "
                    f"schema {item['schema_version'] or 'unknown'}"
                ),
                (
                    "    FakeGPU: "
                    f"{fakegpu.get('version') or 'unknown'}, "
                    f"runtime {fakegpu.get('runtime') or 'unknown'}, "
                    f"backend {fakegpu.get('backend') or 'unknown'}, "
                    f"mode {fakegpu.get('mode') or 'unknown'}"
                ),
                (
                    "    Policies: "
                    f"OOM {fakegpu.get('oom_policy') or 'unknown'}, "
                    "unsupported API "
                    f"{fakegpu.get('unsupported_api_policy') or 'unknown'}, "
                    "distributed "
                    f"{fakegpu.get('distributed_mode') or 'unknown'}"
                ),
                (
                    "    Software: "
                    f"Python {software.get('python_version') or 'N/A'}, "
                    f"PyTorch {software.get('torch_version') or 'N/A'}, "
                    f"CUDA {software.get('cuda_version') or 'N/A'}, "
                    f"driver {software.get('driver_version') or 'N/A'}, "
                    f"platform {software.get('platform') or 'N/A'}"
                ),
                (
                    "    Tracking: "
                    f"{item['tracking_confidence']}, allocator "
                    f"{item['allocator_model']}, "
                    "memory "
                    f"{_enabled_text(fakegpu.get('memory_tracking_enabled'))}, "
                    "dispatch "
                    f"{_enabled_text(dispatch.get('enabled'))}"
                ),
                (
                    "    Dispatch: "
                    f"{dispatch.get('operator_calls', 0)} calls, "
                    f"{dispatch.get('output_tensors', 0)} outputs, "
                    f"{dispatch.get('new_allocations', 0)} allocations, "
                    f"{dispatch.get('alias_outputs', 0)} aliases, "
                    f"{dispatch.get('inaccessible_outputs', 0)} inaccessible"
                ),
                (
                    "    Publisher: "
                    f"{publisher.get('interval_seconds', 'N/A')}s interval, "
                    f"{_format_bytes(publisher.get('runtime_overhead_bytes'))} "
                    "runtime overhead"
                ),
                (
                    "    Publisher health: "
                    f"{publisher_health.get('successful_writes', 'N/A')} / "
                    f"{publisher_health.get('attempted_writes', 'N/A')} "
                    "writes, "
                    f"{publisher_health.get('failed_writes', 'N/A')} "
                    "failures, last "
                    f"{publisher_health.get('last_duration_us', 'N/A')} us / "
                    f"{publisher_last_size}"
                ),
                (
                    "    Publisher limits: "
                    f"{publisher_limits.get('detail_entries', 'N/A')} "
                    "detail entries, "
                    f"{publisher_max_size} state size"
                ),
                (
                    "    Catalogs: "
                    f"{profile_catalog.get('profile_count', 'N/A')} "
                    "profiles; native capabilities "
                    f"{capability.get('group_count', 'N/A')} groups / "
                    f"{capability.get('explicit_api_count', 'N/A')} APIs / "
                    f"{capability.get('policy_enforced_api_count', 'N/A')} "
                    "policy-enforced"
                ),
            ]
        )
        if publisher_health.get("last_error"):
            lines.append(
                "    Publisher last error: "
                f"{publisher_health['last_error']}"
            )
        operators = dispatch.get("operators")
        if isinstance(operators, Mapping) and operators:
            ranked_operators = sorted(
                operators.items(),
                key=lambda entry: _nonnegative_int(
                    entry[1].get("calls")
                    if isinstance(entry[1], Mapping)
                    else 0
                ),
                reverse=True,
            )
            lines.append("    Most-called operators:")
            for operator, values in ranked_operators[:5]:
                calls = (
                    _nonnegative_int(values.get("calls"))
                    if isinstance(values, Mapping)
                    else 0
                )
                lines.append(f"      - {operator}: {calls} calls")

    devices = list(inventory.get("devices") or [])
    if not devices:
        lines.append("No published FakeCUDA GPUs found.")
    for item in devices:
        profile = item["profile"]
        memory = item["memory"]
        allocator = item["allocator"]
        telemetry = item["telemetry"]
        native_activity = item["native_activity"]
        topology = item["topology"]
        nvlink = topology["nvlink"]
        health = item["health"]
        lines.extend(
            [
                "",
                f"GPU {item['index']} @ {item['host']}",
                f"  Product Name: {item['name']}",
                f"  UUID: {item['uuid']} (synthetic)",
                f"  PCI Bus ID: {item['pci_bus_id']} (synthetic)",
                (
                    "  Topology: "
                    f"{topology.get('source') or 'modeled_none'}, "
                    "NUMA "
                    f"{_display_value(topology.get('numa_node'))}, "
                    "PCIe generation "
                    f"{_display_value(topology.get('pcie_generation'))}, "
                    f"{nvlink.get('active_links', 0)} active NVLinks / "
                    f"{_format_gbps(nvlink.get('aggregate_bandwidth_gbps'))} "
                    "Gbps modeled aggregate"
                ),
                (
                    "  Health model: "
                    f"{health.get('status') or 'unknown'}, "
                    "hardware "
                    f"{health.get('hardware_health') or 'unobserved'}, "
                    "maximum severity "
                    f"{health.get('max_severity') or 'none'}, "
                    f"{health.get('event_types_total', 0)} event types / "
                    f"{health.get('event_count', 0)} occurrences"
                ),
                (
                    f"  Profile: {item['profile_id']} "
                    f"[{profile.get('profile_status') or 'unknown'}]"
                ),
                (
                    "  Architecture: "
                    f"{item['architecture'] or 'N/A'}, "
                    f"compute capability "
                    f"{item['compute_capability'] or 'N/A'}, "
                    f"target {item['compiler_target'] or 'N/A'}"
                ),
                (
                    "  Compute resources: "
                    f"{profile.get('sm_count', 'N/A')} SMs, "
                    f"{profile.get('core_clock_mhz', 'N/A')} MHz core, "
                    f"{profile.get('memory_clock_mhz', 'N/A')} MHz memory"
                ),
                (
                    "  Memory profile: "
                    f"{profile.get('memory_kind') or 'N/A'}, "
                    f"{profile.get('memory_bus_width_bits', 'N/A')} bit bus, "
                    f"L2 {_format_bytes(profile.get('l2_cache_bytes'))}"
                ),
                (
                    "  Supported types: "
                    f"{_format_list(profile.get('supported_types'))}"
                ),
                (
                    "  Profile power: "
                    f"{_format_watts(profile.get('typical_power_usage_mw'))} "
                    "typical / "
                    f"{_format_watts(profile.get('max_power_limit_mw'))} max"
                ),
                (
                    "  Memory current: "
                    f"{_format_bytes(memory['used_bytes'])} simulated, "
                    f"{_format_bytes(memory['tracked_bytes'])} allocated, "
                    f"{_format_bytes(memory['reserved_bytes'])} reserved, "
                    f"{_format_bytes(memory['free_bytes'])} free"
                ),
                (
                    "  Memory peaks: "
                    f"{_format_bytes(memory['reported_peak_bytes'])} "
                    "simulated, "
                    f"{_format_bytes(memory['tracked_peak_bytes'])} "
                    "allocated, "
                    f"{_format_bytes(memory['reserved_peak_bytes'])} "
                    "reserved"
                ),
                (
                    "  Allocator: "
                    f"{allocator['model']}, "
                    f"{allocator['segment_count']} segments, "
                    f"{_format_bytes(memory['inactive_split_bytes'])} "
                    "inactive split, "
                    f"{allocator['allocation_count']} allocations / "
                    f"{allocator['free_count']} frees"
                ),
                (
                    "  Memory categories: "
                    f"{_format_byte_mapping(allocator['categories'])}"
                ),
                (
                    "  Stage peaks: "
                    f"{_format_byte_mapping(allocator['stage_peaks'])}"
                ),
                (
                    "  Hardware telemetry: "
                    f"GPU utilization "
                    f"{_format_percent(telemetry.get('gpu_utilization_percent'))}, "
                    f"temperature {_format_temperature(telemetry.get('temperature_c'))}, "
                    f"fan {_format_percent(telemetry.get('fan_speed_percent'))}, "
                    f"power {_format_watts(telemetry.get('power_usage_mw'))}"
                ),
                (
                    "  Processes: "
                    f"{item['process_count']} total, "
                    f"{item['stale_process_count']} stale, "
                    f"{item['exited_process_count']} exited"
                ),
            ]
        )
        if health.get("error"):
            lines.append(
                f"  Health configuration error: {health['error']}"
            )
        if health.get("events"):
            lines.append("  Modeled fault events:")
            for event in health["events"][:5]:
                lines.append(
                    "    - "
                    f"[{event.get('severity') or 'info'}] "
                    f"{event.get('code') or 'UNKNOWN'}: "
                    f"{_nonnegative_int(event.get('count'))} "
                    "occurrences"
                )
        if (
            str(item["fakegpu"].get("runtime") or "") == "native"
            or _has_native_activity(native_activity)
        ):
            lines.extend(
                [
                    (
                        "  Native activity: "
                        f"{native_activity['kernel_launches']} kernel launches, "
                        f"{native_activity['gemm_calls']} GEMM calls / "
                        f"{native_activity['gemm_flops']} FLOP, "
                        f"{native_activity['io_calls']} IO calls / "
                        f"{_format_bytes(native_activity['io_bytes'])}"
                    ),
                    (
                        "  Native compatibility: "
                        f"{native_activity['compatibility_events']} events, "
                        f"{native_activity['unsupported_api_calls']} "
                        "unsupported API calls"
                    ),
                ]
            )
            kernels = native_activity["kernels"]
            if kernels:
                lines.append(
                    "  Native kernels: "
                    + ", ".join(
                        f"{name}={count}"
                        for name, count in sorted(
                            kernels.items(),
                            key=lambda entry: (
                                -_nonnegative_int(entry[1]),
                                str(entry[0]),
                            ),
                        )[:5]
                    )
                )
            unsupported_apis = native_activity["unsupported_apis"]
            if unsupported_apis:
                lines.append("  Unsupported native APIs:")
                for event in unsupported_apis[:5]:
                    lines.append(
                        "    - "
                        f"{event.get('operation') or 'unknown'}: "
                        f"{_nonnegative_int(event.get('count'))} calls "
                        f"[{event.get('behavior') or 'unknown'}, "
                        f"{event.get('policy') or 'unknown'}]"
                    )
        allocations = allocator["largest_allocations"]
        if allocations:
            lines.append("  Largest allocations:")
            for allocation in allocations[:5]:
                lines.append(
                    "    - "
                    f"{_format_bytes(allocation.get('bytes'))} "
                    f"[{allocation.get('category') or 'unknown'}] "
                    f"pid={allocation.get('pid', 0)} "
                    f"{allocation.get('source') or allocation.get('operator') or 'unknown'}"
                )
    for error in errors:
        lines.append(f"warning: {error}")
    return "\n".join(lines)


def render_query(
    records: Sequence[Mapping[str, Any]],
    *,
    fields: Sequence[str],
    available: Mapping[str, QueryField],
    query_kind: str,
    output_format: Mapping[str, Any],
) -> str:
    values = [
        {
            field: _query_value(record, available[field])
            for field in fields
        }
        for record in records
    ]
    if output_format["kind"] == "json":
        return json.dumps(
            {
                "schema_version": QUERY_SCHEMA_VERSION,
                "query": query_kind,
                "fields": list(fields),
                "units": {
                    field: available[field].unit for field in fields
                },
                "records": values,
            },
            sort_keys=True,
        )

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    if not output_format["noheader"]:
        writer.writerow(
            [
                _query_csv_header(field, available[field])
                for field in fields
            ]
        )
    for record in values:
        writer.writerow(
            [
                _query_csv_value(
                    record[field],
                    unit=available[field].unit,
                    nounits=bool(output_format["nounits"]),
                )
                for field in fields
            ]
        )
    return buffer.getvalue().rstrip("\n")


def _empty_device_aggregate(
    *,
    host: str,
    device: Mapping[str, Any],
    fakegpu: Mapping[str, Any],
    software: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "timestamp_ns": 0,
        "timestamp": None,
        "host": host,
        "index": int(device["index"]),
        "name": str(device["name"]),
        "profile_id": str(device["profile_id"]),
        "profile": dict(device["profile"]),
        "uuid": str(device["uuid"]),
        "pci_bus_id": str(device["pci_bus_id"]),
        "identity_source": str(device["identity_source"]),
        "architecture": device.get("architecture"),
        "compute_capability": device.get("compute_capability"),
        "compiler_target": device.get("compiler_target"),
        "process_count": 0,
        "stale_process_count": 0,
        "exited_process_count": 0,
        "memory": {
            "total_bytes": 0,
            "used_bytes": 0,
            "free_bytes": 0,
            "reported_peak_bytes": 0,
            "tracked_bytes": 0,
            "tracked_peak_bytes": 0,
            "reserved_bytes": 0,
            "reserved_peak_bytes": 0,
            "inactive_split_bytes": 0,
            "utilization_percent": None,
            "headroom_bytes": 0,
            "headroom_percent": None,
        },
        "allocator": {
            "model": "unknown",
            "segment_count": 0,
            "allocation_count": 0,
            "free_count": 0,
            "categories": {},
            "stage_peaks": {},
            "reserved_stage_peaks": {},
            "largest_allocations": [],
            "_models": set(),
        },
        "native_activity": {
            "io_calls": 0,
            "io_bytes": 0,
            "kernel_launches": 0,
            "gemm_calls": 0,
            "gemm_flops": 0,
            "compatibility_events": 0,
            "unsupported_api_calls": 0,
            "kernels": {},
            "unsupported_apis": [],
        },
        "topology": dict(device["topology"]),
        "health": dict(device["health"]),
        "telemetry": dict(device["telemetry"]),
        "fakegpu": dict(fakegpu),
        "software": dict(software),
        "tracking_confidence": "unknown",
        "stages": "unknown",
        "state": {
            "status": "unknown",
            "max_age_seconds": None,
        },
        "_names": set(),
        "_profile_ids": set(),
        "_statuses": set(),
        "_stages": set(),
        "_tracking": set(),
        "_ages": [],
    }


def _accumulate_memory(
    aggregate: dict[str, Any],
    memory: Mapping[str, Any],
) -> None:
    aggregate["total_bytes"] = max(
        int(aggregate["total_bytes"]),
        int(memory["total_bytes"]),
    )
    for key in (
        "used_bytes",
        "reported_peak_bytes",
        "tracked_bytes",
        "tracked_peak_bytes",
        "reserved_bytes",
        "reserved_peak_bytes",
        "inactive_split_bytes",
    ):
        aggregate[key] = int(aggregate[key]) + int(memory[key])


def _accumulate_allocator(
    aggregate: dict[str, Any],
    device: Mapping[str, Any],
    *,
    pid: int,
    process_name: str,
) -> None:
    aggregate["_models"].add(str(device["allocator_model"]))
    aggregate["segment_count"] += int(device["segment_count"])
    aggregate["allocation_count"] += int(device["allocation_count"])
    aggregate["free_count"] += int(device["free_count"])
    _sum_integer_mapping(
        aggregate["categories"],
        device["current_bytes_by_category"],
    )
    _sum_integer_mapping(
        aggregate["stage_peaks"],
        device["peak_by_stage"],
    )
    _sum_integer_mapping(
        aggregate["reserved_stage_peaks"],
        device["reserved_peak_by_stage"],
    )
    for raw in device["largest_allocations"]:
        item = dict(raw)
        item["pid"] = pid
        item["process_name"] = process_name
        aggregate["largest_allocations"].append(item)


def _accumulate_native_activity(
    aggregate: dict[str, Any],
    activity: Mapping[str, Any],
    *,
    pid: int,
    process_name: str,
) -> None:
    for key in (
        "io_calls",
        "io_bytes",
        "kernel_launches",
        "gemm_calls",
        "gemm_flops",
        "compatibility_events",
        "unsupported_api_calls",
    ):
        aggregate[key] = _nonnegative_int(
            aggregate.get(key)
        ) + _nonnegative_int(activity.get(key))
    _sum_integer_mapping(
        aggregate["kernels"],
        activity.get("kernels") or {},
    )
    for raw_event in activity.get("unsupported_apis") or []:
        if not isinstance(raw_event, Mapping):
            continue
        event = dict(raw_event)
        event["pid"] = pid
        event["process_name"] = process_name
        aggregate["unsupported_apis"].append(event)


def _finalize_device_aggregate(item: dict[str, Any]) -> dict[str, Any]:
    memory = item["memory"]
    total = int(memory["total_bytes"])
    used = int(memory["used_bytes"])
    peak = int(memory["reported_peak_bytes"])
    memory["free_bytes"] = max(0, total - used) if total else 0
    memory["headroom_bytes"] = max(0, total - peak) if total else 0
    memory["utilization_percent"] = (
        round(used / total * 100, 3) if total else None
    )
    memory["headroom_percent"] = (
        round(memory["headroom_bytes"] / total * 100, 3)
        if total
        else None
    )
    item["timestamp"] = _format_timestamp_ns(
        _nonnegative_int(item["timestamp_ns"])
    )

    names = sorted(
        value for value in item.pop("_names") if value
    )
    profiles = sorted(
        value for value in item.pop("_profile_ids") if value
    )
    statuses = set(item.pop("_statuses"))
    stages = sorted(value for value in item.pop("_stages") if value)
    tracking = sorted(value for value in item.pop("_tracking") if value)
    ages = list(item.pop("_ages"))
    item["name"] = ", ".join(names) or "Fake NVIDIA GPU"
    item["profile_id"] = ", ".join(profiles) or "unknown"
    item["stages"] = ", ".join(stages) or "unknown"
    item["tracking_confidence"] = (
        ", ".join(tracking) or "unknown"
    )
    item["state"] = {
        "status": _aggregate_status(statuses),
        "max_age_seconds": max(ages) if ages else None,
    }
    allocator = item["allocator"]
    models = sorted(
        value for value in allocator.pop("_models") if value
    )
    allocator["model"] = ", ".join(models) or "unknown"
    allocator["largest_allocations"].sort(
        key=lambda value: _nonnegative_int(value.get("bytes")),
        reverse=True,
    )
    allocator["largest_allocations"] = allocator[
        "largest_allocations"
    ][:10]
    native_activity = item["native_activity"]
    native_activity["unsupported_apis"].sort(
        key=lambda event: (
            -_nonnegative_int(event.get("count")),
            str(event.get("operation") or ""),
        )
    )
    native_activity["unsupported_apis"] = native_activity[
        "unsupported_apis"
    ][:10]
    return item


def _normalize_device(
    raw: Mapping[str, Any],
    *,
    host: str,
) -> dict[str, Any]:
    index = _nonnegative_int(raw.get("index"))
    profile_id = str(raw.get("profile_id") or "unknown")
    profile = _profile_metadata(raw)
    total = _nonnegative_int(raw.get("total_memory"))
    tracked = _nonnegative_int(raw.get("tracked_memory"))
    tracked_peak = max(
        tracked,
        _nonnegative_int(raw.get("peak_tracked_memory")),
    )
    reserved = max(
        tracked,
        _nonnegative_int(raw.get("reserved_memory"), default=tracked),
    )
    reserved_peak = max(
        reserved,
        tracked_peak,
        _nonnegative_int(
            raw.get("peak_reserved_memory"),
            default=reserved,
        ),
    )
    used = _nonnegative_int(
        raw.get("reported_memory"),
        default=tracked,
    )
    reported_peak = raw.get("reported_peak_memory")
    if reported_peak is None:
        reported_peak = reserved_peak + _nonnegative_int(
            raw.get("runtime_overhead_bytes")
        )
    reported_peak = max(
        used,
        _nonnegative_int(reported_peak, default=used),
    )
    telemetry = (
        dict(raw.get("telemetry") or {})
        if isinstance(raw.get("telemetry"), Mapping)
        else {}
    )
    telemetry.setdefault("gpu_utilization_percent", None)
    telemetry.setdefault("temperature_c", None)
    telemetry.setdefault("fan_speed_percent", None)
    telemetry.setdefault("power_usage_mw", None)
    telemetry.setdefault(
        "source",
        "hardware_telemetry_unavailable",
    )
    topology = _normalize_device_topology(raw.get("topology"))
    health = _normalize_device_health(raw.get("health"))
    return {
        "index": index,
        "name": str(raw.get("name") or "Fake NVIDIA GPU"),
        "profile_id": profile_id,
        "profile": profile,
        "uuid": str(
            raw.get("uuid")
            or _synthetic_uuid(host, index, profile_id)
        ),
        "pci_bus_id": str(
            raw.get("pci_bus_id") or _synthetic_pci_bus_id(index)
        ),
        "identity_source": str(
            raw.get("identity_source") or "synthetic"
        ),
        "architecture": (
            raw.get("architecture") or profile.get("architecture")
        ),
        "compute_capability": (
            raw.get("compute_capability")
            or profile.get("compute_capability")
        ),
        "compiler_target": (
            raw.get("compiler_target")
            or profile.get("compiler_target")
        ),
        "memory": {
            "total_bytes": total,
            "used_bytes": used,
            "free_bytes": (
                max(0, total - used) if total else 0
            ),
            "reported_peak_bytes": reported_peak,
            "tracked_bytes": tracked,
            "tracked_peak_bytes": tracked_peak,
            "reserved_bytes": reserved,
            "reserved_peak_bytes": reserved_peak,
            "inactive_split_bytes": _nonnegative_int(
                raw.get("inactive_split_bytes")
            ),
            "utilization_percent": (
                round(used / total * 100, 3) if total else None
            ),
            "headroom_bytes": (
                max(0, total - reported_peak) if total else 0
            ),
            "headroom_percent": (
                round(
                    max(0, total - reported_peak) / total * 100,
                    3,
                )
                if total
                else None
            ),
        },
        "allocator_model": str(
            raw.get("allocator_model") or "unknown"
        ),
        "segment_count": _nonnegative_int(raw.get("segment_count")),
        "allocation_count": _nonnegative_int(
            raw.get("allocation_count")
        ),
        "free_count": _nonnegative_int(raw.get("free_count")),
        "current_bytes_by_category": _integer_mapping(
            raw.get("current_bytes_by_category")
        ),
        "peak_by_stage": _integer_mapping(raw.get("peak_by_stage")),
        "reserved_peak_by_stage": _integer_mapping(
            raw.get("reserved_peak_by_stage")
        ),
        "largest_allocations": _mapping_list(
            raw.get("largest_allocations")
        ),
        "native_activity": _normalize_native_activity(
            raw.get("native_activity")
        ),
        "topology": topology,
        "health": health,
        "telemetry": telemetry,
    }


def _state_fakegpu_metadata(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    catalogs = _catalog_metadata()
    defaults = {
        "version": "unknown",
        "runtime": str(state.get("runtime") or "fakecuda"),
        "backend": "unknown",
        "mode": "unknown",
        "oom_policy": "unknown",
        "unsupported_api_policy": "unknown",
        "distributed_mode": "unknown",
        "memory_tracking_enabled": True,
        "dispatch_memory_tracking_enabled": False,
        **catalogs,
    }
    raw = state.get("fakegpu")
    if isinstance(raw, Mapping):
        result = dict(defaults)
        result.update(raw)
        for key in ("profile_catalog", "native_capabilities"):
            default_catalog = defaults.get(key)
            raw_catalog = raw.get(key)
            if isinstance(default_catalog, Mapping):
                merged_catalog = dict(default_catalog)
                if isinstance(raw_catalog, Mapping):
                    merged_catalog.update(raw_catalog)
                result[key] = merged_catalog
        return result
    return defaults


def _state_software_metadata(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    raw = state.get("software")
    if isinstance(raw, Mapping):
        return dict(raw)
    return {
        "python_version": None,
        "python_implementation": None,
        "python_executable": None,
        "platform": None,
        "torch_version": None,
        "torch_cuda_build": None,
        "cuda_version": None,
        "cuda_version_source": None,
        "driver_version": "simulated",
    }


def _state_age_seconds(
    state: Mapping[str, Any],
    *,
    now_ns: int,
) -> float | None:
    timestamp = state.get("timestamp_ns")
    if (
        not isinstance(timestamp, int)
        or isinstance(timestamp, bool)
        or timestamp <= 0
    ):
        return None
    return round(max(0, now_ns - timestamp) / 1e9, 6)


def _state_status(
    state: Mapping[str, Any],
    *,
    age_seconds: float | None,
    stale_after_seconds: float,
) -> str:
    if not bool(state.get("running")):
        return "exited"
    if (
        age_seconds is not None
        and age_seconds > stale_after_seconds
    ):
        return "stale"
    return "running"


def _aggregate_status(statuses: set[str]) -> str:
    if "running" in statuses:
        return "running"
    if "stale" in statuses:
        return "stale"
    if "exited" in statuses:
        return "exited"
    return "unknown"


def _matches_device(
    device: Mapping[str, Any],
    selectors: Sequence[str],
) -> bool:
    candidates = {
        str(device.get("index", "")).casefold(),
        str(device.get("uuid", "")).casefold(),
        str(device.get("pci_bus_id", "")).casefold(),
        str(device.get("profile_id", "")).casefold(),
    }
    return any(selector in candidates for selector in selectors)


def _device_selectors(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        part.strip()
        for value in values
        for part in str(value).split(",")
        if part.strip()
    )


def _sum_integer_mapping(
    target: dict[str, int],
    source: Mapping[str, Any],
) -> None:
    for key, value in source.items():
        target[str(key)] = target.get(str(key), 0) + _nonnegative_int(
            value
        )


def _integer_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): _nonnegative_int(candidate)
        for key, candidate in value.items()
    }


def _mapping_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [
        dict(item) for item in value if isinstance(item, Mapping)
    ]


def _normalize_device_health(value: Any) -> dict[str, Any]:
    raw = dict(value) if isinstance(value, Mapping) else {}
    events = []
    for item in _mapping_list(raw.get("events")):
        severity = str(item.get("severity") or "none").lower()
        if severity not in FAULT_SEVERITY_RANK:
            severity = "none"
        events.append(
            {
                "device_index": _nonnegative_int(
                    item.get("device_index")
                ),
                "code": str(item.get("code") or "UNKNOWN"),
                "severity": severity,
                "count": min(
                    MAXIMUM_MODELED_FAULT_COUNT,
                    _nonnegative_int(item.get("count")),
                ),
                "active": bool(item.get("active", True)),
                "source": str(
                    item.get("source")
                    or raw.get("source")
                    or "modeled_none"
                ),
            }
        )
    maximum_severity = str(
        raw.get("max_severity")
        or _maximum_fault_severity(events)
    ).lower()
    if maximum_severity not in FAULT_SEVERITY_RANK:
        maximum_severity = "none"
    valid = bool(raw.get("valid", True))
    return {
        "source": str(raw.get("source") or "modeled_none"),
        "configured": bool(raw.get("configured", False)),
        "valid": valid,
        "error": str(raw.get("error") or ""),
        "hardware_health": str(
            raw.get("hardware_health") or "unobserved"
        ),
        "status": str(
            raw.get("status")
            or _modeled_health_status(
                valid=valid,
                maximum_severity=maximum_severity,
            )
        ),
        "max_severity": maximum_severity,
        "event_count": _nonnegative_int(
            raw.get("event_count"),
            default=sum(int(event["count"]) for event in events),
        ),
        "event_types_total": _nonnegative_int(
            raw.get("event_types_total"),
            default=len(events),
        ),
        "event_types_retained": _nonnegative_int(
            raw.get("event_types_retained"),
            default=len(events),
        ),
        "events": events,
    }


def _normalize_device_topology(value: Any) -> dict[str, Any]:
    raw = dict(value) if isinstance(value, Mapping) else {}
    nvlink_raw = (
        dict(raw.get("nvlink") or {})
        if isinstance(raw.get("nvlink"), Mapping)
        else {}
    )
    peers = []
    for item in _mapping_list(nvlink_raw.get("peers")):
        try:
            bandwidth = float(item.get("bandwidth_gbps", 0))
        except (TypeError, ValueError):
            bandwidth = 0.0
        if not math.isfinite(bandwidth) or bandwidth < 0:
            bandwidth = 0.0
        peers.append(
            {
                "link": _nonnegative_int(item.get("link")),
                "index": _nonnegative_int(item.get("index")),
                "uuid": str(item.get("uuid") or "unknown"),
                "pci_bus_id": str(
                    item.get("pci_bus_id") or "unknown"
                ),
                "bandwidth_gbps": bandwidth,
                "active": bool(item.get("active", True)),
                "source": str(
                    item.get("source")
                    or raw.get("source")
                    or "modeled_none"
                ),
            }
        )
    try:
        aggregate_bandwidth = float(
            nvlink_raw.get("aggregate_bandwidth_gbps", 0)
        )
    except (TypeError, ValueError):
        aggregate_bandwidth = 0.0
    if (
        not math.isfinite(aggregate_bandwidth)
        or aggregate_bandwidth < 0
    ):
        aggregate_bandwidth = 0.0
    return {
        "source": str(raw.get("source") or "modeled_none"),
        "configured": bool(raw.get("configured", False)),
        "valid": bool(raw.get("valid", True)),
        "error": str(raw.get("error") or ""),
        "numa_node": raw.get("numa_node"),
        "pcie_generation": raw.get("pcie_generation"),
        "nvlink": {
            "active_links": _nonnegative_int(
                nvlink_raw.get("active_links"),
                default=sum(
                    1 for peer in peers if peer["active"]
                ),
            ),
            "peer_count": _nonnegative_int(
                nvlink_raw.get("peer_count"),
                default=len(peers),
            ),
            "aggregate_bandwidth_gbps": aggregate_bandwidth,
            "peers": peers,
        },
    }


def _normalize_native_activity(value: Any) -> dict[str, Any]:
    raw = dict(value) if isinstance(value, Mapping) else {}
    result = {
        key: _nonnegative_int(raw.get(key))
        for key in (
            "io_calls",
            "io_bytes",
            "kernel_launches",
            "gemm_calls",
            "gemm_flops",
            "compatibility_events",
            "unsupported_api_calls",
        )
    }
    result["kernels"] = _integer_mapping(raw.get("kernels"))
    result["unsupported_apis"] = _mapping_list(
        raw.get("unsupported_apis")
    )
    return result


def _has_native_activity(activity: Mapping[str, Any]) -> bool:
    return any(
        _nonnegative_int(activity.get(key))
        for key in (
            "io_calls",
            "io_bytes",
            "kernel_launches",
            "gemm_calls",
            "gemm_flops",
            "compatibility_events",
            "unsupported_api_calls",
        )
    )


def _nonnegative_int(value: Any, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _parse_query_fields(
    value: str | None,
    *,
    available: Mapping[str, QueryField],
    parser: argparse.ArgumentParser,
) -> tuple[str, ...]:
    if value is None:
        return ()
    fields = tuple(
        item.strip() for item in value.split(",") if item.strip()
    )
    if not fields:
        parser.error("query field list must not be empty")
    unknown = [field for field in fields if field not in available]
    if unknown:
        parser.error(
            "unsupported query field(s): "
            + ", ".join(unknown)
            + "; use the matching --help-query option"
        )
    return fields


def _parse_query_format(
    value: str,
    *,
    parser: argparse.ArgumentParser,
) -> dict[str, Any]:
    parts = tuple(
        item.strip().lower()
        for item in str(value).split(",")
        if item.strip()
    )
    if parts == ("json",):
        return {"kind": "json", "noheader": True, "nounits": True}
    if not parts or parts[0] != "csv":
        parser.error("--format must start with csv or equal json")
    options = set(parts[1:])
    unknown = options - {"noheader", "nounits"}
    if unknown:
        parser.error(
            "unsupported --format option(s): "
            + ", ".join(sorted(unknown))
        )
    return {
        "kind": "csv",
        "noheader": "noheader" in options,
        "nounits": "nounits" in options,
    }


def _render_query_help(
    query_kind: str,
    fields: Mapping[str, QueryField],
) -> str:
    lines = [f"Supported {query_kind} query fields:"]
    for name, spec in fields.items():
        suffix = f" [{spec.unit}]" if spec.unit else ""
        lines.append(f"  {name}{suffix}")
    return "\n".join(lines)


def _query_value(
    record: Mapping[str, Any],
    spec: QueryField,
) -> Any:
    value = _nested_value(record, spec.path)
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(item) for item in value)
    if isinstance(value, Mapping):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if (
        spec.divisor != 1.0
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        value = float(value) / spec.divisor
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        if spec.precision:
            return round(float(value), spec.precision)
        if isinstance(value, float):
            return int(round(value))
    return value


def _nested_value(record: Mapping[str, Any], path: str) -> Any:
    current: Any = record
    for part in path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _query_csv_value(
    value: Any,
    *,
    unit: str | None,
    nounits: bool,
) -> str:
    if value is None:
        return "N/A"
    text = str(value)
    if unit and not nounits:
        text += f" {unit}"
    return text


def _query_csv_header(field: str, spec: QueryField) -> str:
    if spec.unit:
        return f"{field} [{spec.unit}]"
    return field


@lru_cache(maxsize=1)
def _fakegpu_version() -> str:
    from ._version import __version__

    return __version__


def _software_metadata() -> dict[str, Any]:
    torch_module = sys.modules.get("torch")
    torch_version = (
        str(getattr(torch_module, "__version__", ""))
        if torch_module is not None
        else None
    )
    torch_version_object = (
        getattr(torch_module, "version", None)
        if torch_module is not None
        else None
    )
    torch_cuda = (
        getattr(torch_version_object, "cuda", None)
        if torch_version_object is not None
        else None
    )
    configured_cuda = os.environ.get("FAKEGPU_CUDA_VERSION")
    cuda_version = str(configured_cuda or torch_cuda or "12.1")
    cuda_source = (
        "FAKEGPU_CUDA_VERSION"
        if configured_cuda
        else "torch.version.cuda"
        if torch_cuda
        else "fakecuda_default"
    )
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": torch_version,
        "torch_cuda_build": (
            str(torch_cuda) if torch_cuda is not None else None
        ),
        "cuda_version": cuda_version,
        "cuda_version_source": cuda_source,
        "driver_version": os.environ.get(
            "FAKEGPU_DRIVER_VERSION",
            "simulated",
        ),
    }


@lru_cache(maxsize=1)
def _catalog_metadata() -> dict[str, Any]:
    profile_summary: dict[str, Any] = {}
    capability_summary: dict[str, Any] = {}
    try:
        from .profile_catalog import catalog_summary

        profile_summary = catalog_summary()
    except (OSError, RuntimeError, ValueError):
        profile_summary = {}
    try:
        from .capabilities import load_native_capabilities

        catalog = load_native_capabilities()
        apis = list(catalog.get("apis") or [])
        groups = list(catalog.get("groups") or [])
        classifications: dict[str, int] = {}
        for api in apis:
            if not isinstance(api, Mapping):
                continue
            classification = str(
                api.get("classification") or "unknown"
            )
            classifications[classification] = (
                classifications.get(classification, 0) + 1
            )
        capability_summary = {
            "schema_version": catalog.get("schema_version"),
            "group_count": len(groups),
            "explicit_api_count": len(apis),
            "policy_enforced_api_count": sum(
                bool(api.get("policy_enforced"))
                for api in apis
                if isinstance(api, Mapping)
            ),
            "classifications": dict(sorted(classifications.items())),
        }
    except (OSError, RuntimeError, ValueError):
        capability_summary = {}
    return {
        "profile_catalog": profile_summary,
        "native_capabilities": capability_summary,
    }


@lru_cache(maxsize=256)
def _catalog_profile_metadata(profile_id: str) -> dict[str, Any]:
    from .profile_catalog import get_profile

    return get_profile(profile_id).to_dict()


def _profile_metadata(item: Mapping[str, Any]) -> dict[str, Any]:
    raw_profile = item.get("profile")
    if isinstance(raw_profile, Mapping):
        profile_id = str(
            raw_profile.get("id")
            or item.get("profile_id")
            or ""
        )
        if profile_id:
            try:
                profile = dict(_catalog_profile_metadata(profile_id))
                profile.update(raw_profile)
                return profile
            except (KeyError, OSError, RuntimeError, ValueError):
                pass
        return dict(raw_profile)
    profile_id = str(item.get("profile_id") or "")
    if profile_id:
        try:
            return dict(_catalog_profile_metadata(profile_id))
        except (KeyError, OSError, RuntimeError, ValueError):
            pass
    major = item.get("compute_major")
    minor = item.get("compute_minor")
    capability = (
        f"{int(major)}.{int(minor)}"
        if isinstance(major, int)
        and not isinstance(major, bool)
        and isinstance(minor, int)
        and not isinstance(minor, bool)
        else None
    )
    return {
        "id": profile_id or "unknown",
        "name": str(item.get("name") or "Fake NVIDIA GPU"),
        "profile_status": "unknown",
        "architecture": item.get("architecture"),
        "compute_capability": capability,
        "compiler_target": (
            f"sm_{int(major)}{int(minor)}"
            if capability is not None
            else None
        ),
        "memory_bytes": _nonnegative_int(item.get("total_memory")),
        "memory_kind": "synthetic",
        "sm_count": item.get("sm_count"),
        "memory_bus_width_bits": item.get("memory_bus_width_bits"),
        "core_clock_mhz": item.get("core_clock_mhz"),
        "memory_clock_mhz": item.get("memory_clock_mhz"),
        "l2_cache_bytes": item.get("l2_cache_bytes"),
        "typical_power_usage_mw": item.get(
            "typical_power_usage_mw"
        ),
        "max_power_limit_mw": item.get("max_power_limit_mw"),
        "supported_types": list(item.get("supported_types") or []),
    }


def _synthetic_uuid(host: str, index: int, profile_id: str) -> str:
    identity = uuid.uuid5(
        uuid.NAMESPACE_DNS,
        f"fakegpu:{host}:{index}:{profile_id}",
    )
    return f"GPU-{identity}"


def _synthetic_pci_bus_id(index: int) -> str:
    domain = max(0, int(index)) // 255
    bus = max(0, int(index)) % 255 + 1
    return f"{domain:08X}:{bus:02X}:00.0"


def _format_age(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.3f}s"


def _topology_labels(
    devices: Sequence[Mapping[str, Any]],
) -> list[str]:
    totals: dict[int, int] = {}
    for device in devices:
        index = _nonnegative_int(device.get("index"))
        totals[index] = totals.get(index, 0) + 1
    offsets: dict[int, int] = {}
    labels = []
    for device in devices:
        index = _nonnegative_int(device.get("index"))
        offsets[index] = offsets.get(index, 0) + 1
        label = f"GPU{index}"
        if totals[index] > 1:
            label += f"#{offsets[index]}"
        labels.append(label)
    return labels


def _topology_relation(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
) -> str:
    if source is target:
        return "X"
    target_uuid = str(target.get("uuid") or "")
    target_index = _nonnegative_int(target.get("index"))
    peers = (
        source.get("topology", {})
        .get("nvlink", {})
        .get("peers", [])
    )
    for peer in peers:
        if not isinstance(peer, Mapping) or not peer.get("active"):
            continue
        peer_uuid = str(peer.get("uuid") or "")
        if (
            peer_uuid
            and peer_uuid != "unknown"
            and peer_uuid == target_uuid
        ):
            return "NV1"
        if (
            peer_uuid in {"", "unknown"}
            and _nonnegative_int(peer.get("index")) == target_index
        ):
            return "NV1"
    if str(source.get("host") or "") == str(target.get("host") or ""):
        return "PIX"
    return "SYS"


def _display_value(value: Any) -> str:
    return "N/A" if value is None or value == "" else str(value)


def _format_gbps(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        bandwidth = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(bandwidth) or bandwidth < 0:
        return "N/A"
    return f"{bandwidth:g}"


def _format_timestamp_ns(value: int) -> str | None:
    if value <= 0:
        return None
    try:
        return time.strftime(
            "%Y/%m/%d %H:%M:%S",
            time.localtime(value / 1e9),
        )
    except (OverflowError, OSError, ValueError):
        return None


def _enabled_text(value: Any) -> str:
    return "enabled" if bool(value) else "disabled"


def _format_bytes(value: Any) -> str:
    if value is None:
        return "N/A"
    amount = float(_nonnegative_int(value))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return "N/A"


def _format_watts(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value) / 1000:.1f} W"
    except (TypeError, ValueError):
        return "N/A"


def _format_percent(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _format_temperature(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.1f} C"
    except (TypeError, ValueError):
        return "N/A"


def _format_list(value: Any) -> str:
    if not isinstance(value, (list, tuple, set)) or not value:
        return "N/A"
    return ", ".join(str(item) for item in value)


def _format_byte_mapping(value: Mapping[str, Any]) -> str:
    if not value:
        return "none"
    return ", ".join(
        f"{key}={_format_bytes(candidate)}"
        for key, candidate in sorted(value.items())
    )


def _atomic_write_json(
    path: Path,
    payload: dict[str, Any],
    *,
    max_bytes: int,
) -> int:
    serialized = (
        json.dumps(payload, sort_keys=True) + "\n"
    ).encode("utf-8")
    if len(serialized) > max_bytes:
        raise ValueError(
            f"serialized state is {len(serialized)} bytes; "
            f"limit is {max_bytes} bytes"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(serialized)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return len(serialized)


def _process_name() -> str:
    if not sys.argv:
        return "python"
    return " ".join(sys.argv[:3])


def _mib(value: int) -> int:
    return int(round(int(value) / 2**20))


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _table_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")
