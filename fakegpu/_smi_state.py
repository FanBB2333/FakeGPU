"""The published SMI state schema and the reader side that normalizes it.

Split out of ``smi`` unchanged. ``smi`` writes state files through its
publisher and renders them; this module owns the other half — the schema
versions those files carry, discovering and loading them, normalizing one
device or process record, and aggregating a set of state files into the
inventory that both ``fakegpu nvidia-smi`` and ``fakegpu metrics`` read.

Keeping it here is what lets ``metrics`` consume the inventory without
reaching into ``smi``'s private renderer- and publisher-side helpers.
"""

from __future__ import annotations

import json
import math
import time
import uuid
import warnings
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from ._smi_environment import (
    FAULT_SEVERITY_RANK,
    MAXIMUM_MODELED_FAULT_COUNT,
    MAXIMUM_MODELED_MIG_INSTANCES_PER_DEVICE,
    _maximum_fault_severity,
    _modeled_health_status,
    _nonnegative_int,
)


SCHEMA_VERSION = "fakegpu.smi_state.v2"


LEGACY_SCHEMA_VERSION = "fakegpu.smi_state.v1"


SUPPORTED_SCHEMA_VERSIONS = frozenset(
    {LEGACY_SCHEMA_VERSION, SCHEMA_VERSION}
)


DEFAULT_STALE_AFTER_SECONDS = 2.0


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
        except (OSError, ValueError) as exc:
            errors.append(f"{path}: {exc}")
    return states, errors


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


def _integer_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): _nonnegative_int(candidate)
        for key, candidate in value.items()
    }


def _sum_integer_mapping(
    target: dict[str, int],
    source: Mapping[str, Any],
) -> None:
    for key, value in source.items():
        target[str(key)] = target.get(str(key), 0) + _nonnegative_int(
            value
        )


def _mapping_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [
        dict(item) for item in value if isinstance(item, Mapping)
    ]


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


@lru_cache(maxsize=256)
def _catalog_profile_metadata(profile_id: str) -> dict[str, Any]:
    from .profile_catalog import get_profile

    return get_profile(profile_id).to_dict()


@lru_cache(maxsize=1)
def _catalog_metadata() -> dict[str, Any]:
    profile_summary: dict[str, Any] = {}
    capability_summary: dict[str, Any] = {}
    try:
        from .profile_catalog import catalog_summary

        profile_summary = catalog_summary()
    except (OSError, RuntimeError, ValueError) as exc:
        warnings.warn(
            f"FakeGPU SMI could not load profile catalog metadata: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
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
    except (OSError, RuntimeError, ValueError) as exc:
        warnings.warn(
            f"FakeGPU SMI could not load native capability metadata: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        capability_summary = {}
    return {
        "profile_catalog": profile_summary,
        "native_capabilities": capability_summary,
    }


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


def _normalize_device_mig(
    value: Any,
    *,
    total_memory: int,
) -> dict[str, Any]:
    raw = dict(value) if isinstance(value, Mapping) else {}
    instances = []
    for item in _mapping_list(raw.get("instances")):
        used = item.get("memory_used_bytes")
        free = item.get("memory_free_bytes")
        instances.append(
            {
                "index": _nonnegative_int(item.get("index")),
                "gpu_instance_id": _nonnegative_int(
                    item.get("gpu_instance_id")
                ),
                "compute_instance_id": _nonnegative_int(
                    item.get("compute_instance_id")
                ),
                "profile": str(
                    item.get("profile") or "unknown"
                ),
                "slice_count": _nonnegative_int(
                    item.get("slice_count")
                ),
                "uuid": str(item.get("uuid") or "unknown"),
                "parent_uuid": str(
                    item.get("parent_uuid") or "unknown"
                ),
                "pci_bus_id": str(
                    item.get("pci_bus_id") or "unknown"
                ),
                "memory_total_bytes": _nonnegative_int(
                    item.get("memory_total_bytes")
                ),
                "memory_used_bytes": (
                    None
                    if used is None
                    else _nonnegative_int(used)
                ),
                "memory_free_bytes": (
                    None
                    if free is None
                    else _nonnegative_int(free)
                ),
                "memory_tracking": str(
                    item.get("memory_tracking") or "unobserved"
                ),
                "source": str(
                    item.get("source")
                    or raw.get("source")
                    or "modeled_none"
                ),
            }
        )
    instances.sort(
        key=lambda item: (
            int(item["gpu_instance_id"]),
            int(item["compute_instance_id"]),
            str(item["uuid"]),
        )
    )
    allocated = _nonnegative_int(
        raw.get("allocated_memory_bytes"),
        default=sum(
            int(instance["memory_total_bytes"])
            for instance in instances
        ),
    )
    valid = bool(raw.get("valid", True))
    mode = str(raw.get("mode") or "")
    if mode not in {
        "enabled",
        "disabled",
        "configuration_error",
    }:
        mode = (
            "configuration_error"
            if not valid
            else "enabled"
            if instances
            else "disabled"
        )
    return {
        "source": str(raw.get("source") or "modeled_none"),
        "configured": bool(raw.get("configured", False)),
        "valid": valid,
        "error": str(raw.get("error") or ""),
        "mode": mode,
        "max_instance_count": _nonnegative_int(
            raw.get("max_instance_count"),
            default=MAXIMUM_MODELED_MIG_INSTANCES_PER_DEVICE,
        ),
        "instance_count": _nonnegative_int(
            raw.get("instance_count"),
            default=len(instances),
        ),
        "allocated_memory_bytes": allocated,
        "unallocated_memory_bytes": _nonnegative_int(
            raw.get("unallocated_memory_bytes"),
            default=max(0, total_memory - allocated),
        ),
        "instances": instances,
    }


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
    topology = _normalize_device_topology(raw.get("topology"))
    health = _normalize_device_health(raw.get("health"))
    mig = _normalize_device_mig(
        raw.get("mig"),
        total_memory=total,
    )
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
        "mig": mig,
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
    mig = device.get("mig")
    if isinstance(mig, Mapping):
        for instance in mig.get("instances") or []:
            if not isinstance(instance, Mapping):
                continue
            candidates.add(
                str(instance.get("uuid", "")).casefold()
            )
            candidates.add(
                str(instance.get("profile", "")).casefold()
            )
    return any(selector in candidates for selector in selectors)


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
        "mig": dict(device["mig"]),
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
        "mig_instance_count": sum(
            _nonnegative_int(
                device.get("mig", {}).get("instance_count")
            )
            for device in devices
        ),
        "process_count": len(process_rows),
        "runtime_count": len(runtime_rows),
        "devices": devices,
        "processes": process_rows,
        "runtimes": runtime_rows,
    }
