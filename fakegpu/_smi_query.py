"""The ``--query-*`` field catalog and how one field is read from a state.

Split out of ``smi`` unchanged. Every queryable GPU, process, and runtime
field is declared here once — its path into the normalized inventory, its
CSV header, and how its value renders — so ``nvidia-smi``-style queries and
the query report stay in step with one schema.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._smi_environment import _nonnegative_int


QUERY_SCHEMA_VERSION = "fakegpu.smi_query.v1"


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
    "mig.mode": QueryField("mig.mode"),
    "mig.instance_count": QueryField("mig.instance_count"),
    "mig.allocated_memory": QueryField(
        "mig.allocated_memory_bytes",
        unit="MiB",
        divisor=2**20,
    ),
    "mig.unallocated_memory": QueryField(
        "mig.unallocated_memory_bytes",
        unit="MiB",
        divisor=2**20,
    ),
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


RUNTIME_QUERY_FIELDS: dict[str, QueryField] = {
    "timestamp": QueryField("timestamp"),
    "host": QueryField("host"),
    "pid": QueryField("pid"),
    "process_name": QueryField("process_name"),
    "stage": QueryField("stage"),
    "status": QueryField("status"),
    "state.age": QueryField(
        "age_seconds",
        unit="s",
        precision=3,
    ),
    "tracking.confidence": QueryField("tracking_confidence"),
    "allocator.model": QueryField("allocator_model"),
    "fakegpu.version": QueryField("fakegpu.version"),
    "runtime": QueryField("fakegpu.runtime"),
    "runtime.backend": QueryField("fakegpu.backend"),
    "runtime.mode": QueryField("fakegpu.mode"),
    "policy.oom": QueryField("fakegpu.oom_policy"),
    "policy.unsupported_api": QueryField(
        "fakegpu.unsupported_api_policy"
    ),
    "distributed.mode": QueryField("fakegpu.distributed_mode"),
    "tracking.memory": QueryField(
        "fakegpu.memory_tracking_enabled"
    ),
    "tracking.dispatch": QueryField(
        "fakegpu.dispatch_memory_tracking_enabled"
    ),
    "catalog.profiles": QueryField(
        "fakegpu.profile_catalog.profile_count"
    ),
    "catalog.native_groups": QueryField(
        "fakegpu.native_capabilities.group_count"
    ),
    "catalog.native_apis": QueryField(
        "fakegpu.native_capabilities.explicit_api_count"
    ),
    "catalog.policy_apis": QueryField(
        "fakegpu.native_capabilities.policy_enforced_api_count"
    ),
    "software.python": QueryField("software.python_version"),
    "software.python_implementation": QueryField(
        "software.python_implementation"
    ),
    "software.pytorch": QueryField("software.torch_version"),
    "software.cuda": QueryField("software.cuda_version"),
    "software.cuda_source": QueryField(
        "software.cuda_version_source"
    ),
    "software.driver": QueryField("software.driver_version"),
    "software.platform": QueryField("software.platform"),
    "dispatch.calls": QueryField("dispatch_tracking.operator_calls"),
    "dispatch.outputs": QueryField(
        "dispatch_tracking.output_tensors"
    ),
    "dispatch.allocations": QueryField(
        "dispatch_tracking.new_allocations"
    ),
    "dispatch.aliases": QueryField(
        "dispatch_tracking.alias_outputs"
    ),
    "dispatch.inaccessible": QueryField(
        "dispatch_tracking.inaccessible_outputs"
    ),
    "publisher.interval": QueryField(
        "publisher.interval_seconds",
        unit="s",
        precision=3,
    ),
    "publisher.runtime_overhead": QueryField(
        "publisher.runtime_overhead_bytes",
        unit="MiB",
        divisor=2**20,
    ),
    "publisher.attempted_writes": QueryField(
        "publisher.health.attempted_writes"
    ),
    "publisher.successful_writes": QueryField(
        "publisher.health.successful_writes"
    ),
    "publisher.failed_writes": QueryField(
        "publisher.health.failed_writes"
    ),
    "publisher.last_duration": QueryField(
        "publisher.health.last_duration_us",
        unit="us",
    ),
    "publisher.max_duration": QueryField(
        "publisher.health.max_duration_us",
        unit="us",
    ),
    "publisher.state_size": QueryField(
        "publisher.health.last_serialized_bytes",
        unit="KiB",
        divisor=2**10,
        precision=3,
    ),
    "publisher.last_error": QueryField(
        "publisher.health.last_error"
    ),
    "publisher.detail_limit": QueryField(
        "publisher.limits.detail_entries"
    ),
    "publisher.state_size_limit": QueryField(
        "publisher.limits.max_state_bytes",
        unit="KiB",
        divisor=2**10,
        precision=3,
    ),
}


def _nested_value(record: Mapping[str, Any], path: str) -> Any:
    current: Any = record
    for part in path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


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


def _query_csv_header(field: str, spec: QueryField) -> str:
    if spec.unit:
        return f"{field} [{spec.unit}]"
    return field


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
