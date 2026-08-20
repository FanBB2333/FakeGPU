"""Terminal and machine-readable renderings of an SMI inventory.

Split out of ``smi`` unchanged: the device table, the topology matrix,
NVLink and MIG views, health events, the per-device detail page, and the
``--query-*`` report, plus the value formatting they share.
"""

from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from ._smi_environment import (
    FAULT_SEVERITY_RANK,
    _maximum_fault_severity,
    _nonnegative_int,
)
from ._smi_query import (
    QUERY_SCHEMA_VERSION,
    QueryField,
    _has_native_activity,
    _query_csv_header,
    _query_csv_value,
    _query_value,
)
from ._smi_state import DEFAULT_STALE_AFTER_SECONDS, build_inventory


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
        for instance in item.get("mig", {}).get("instances") or []:
            lines.append(
                "  MIG "
                f"{instance['profile']} Device "
                f"{instance['index']}: "
                f"(UUID: {instance['uuid']}, "
                f"GI: {instance['gpu_instance_id']}, "
                f"CI: {instance['compute_instance_id']}, "
                "Memory: "
                f"{_format_bytes(instance['memory_total_bytes'])})"
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
                ]
            ]
            for row_index, device in enumerate(host_devices):
                rows.append(
                    [
                        labels[row_index],
                        *[
                            _topology_relation(device, target)
                            for target in host_devices
                        ],
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


def render_mig_view(
    inventory: Mapping[str, Any],
    *,
    errors: Sequence[str] = (),
    instance_kind: str = "all",
) -> str:
    if instance_kind not in {"all", "gpu", "compute"}:
        raise ValueError(
            "instance_kind must be all, gpu, or compute"
        )
    devices = list(inventory.get("devices") or [])
    lines = [
        "FakeGPU modeled MIG instances",
        (
            "Instance capacity is configured; per-instance runtime "
            "memory usage is unobserved."
        ),
    ]
    if not devices:
        lines.append("No published FakeCUDA GPUs found.")
    for device in devices:
        mig = device["mig"]
        lines.extend(
            [
                "",
                (
                    f"GPU {device['index']} @ {device['host']}: "
                    f"{device['name']} ({device['uuid']})"
                ),
                (
                    "  MIG mode: "
                    f"{str(mig.get('mode') or 'disabled').replace('_', ' ').title()} "
                    f"({mig.get('source') or 'modeled_none'})"
                ),
                (
                    "  Capacity: "
                    f"{mig.get('instance_count', 0)} instances, "
                    f"{_format_bytes(mig.get('allocated_memory_bytes'))} "
                    "allocated / "
                    f"{_format_bytes(mig.get('unallocated_memory_bytes'))} "
                    "unallocated"
                ),
            ]
        )
        if mig.get("error"):
            lines.append(
                f"  Configuration error: {mig['error']}"
            )
        instances = list(mig.get("instances") or [])
        if not instances:
            lines.append("  No modeled MIG instances.")
            continue
        for instance in instances:
            if instance_kind in {"all", "gpu"}:
                lines.append(
                    "  GPU instance "
                    f"{instance['gpu_instance_id']}: "
                    f"profile {instance['profile']}, "
                    f"{instance['slice_count']} slice(s), "
                    f"{_format_bytes(instance['memory_total_bytes'])}, "
                    f"UUID {instance['uuid']}"
                )
            if instance_kind in {"all", "compute"}:
                prefix = (
                    "    "
                    if instance_kind == "all"
                    else "  "
                )
                lines.append(
                    f"{prefix}Compute instance "
                    f"{instance['compute_instance_id']} "
                    f"(GI {instance['gpu_instance_id']}): "
                    f"profile {instance['profile']}, "
                    "memory tracking "
                    f"{instance['memory_tracking']}"
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
        native_activity = item["native_activity"]
        topology = item["topology"]
        nvlink = topology["nvlink"]
        health = item["health"]
        mig = item["mig"]
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
                    "  MIG model: "
                    f"{mig.get('mode') or 'disabled'}, "
                    f"{mig.get('instance_count', 0)} instances, "
                    f"{_format_bytes(mig.get('allocated_memory_bytes'))} "
                    "allocated / "
                    f"{_format_bytes(mig.get('unallocated_memory_bytes'))} "
                    "unallocated"
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
        if mig.get("error"):
            lines.append(
                f"  MIG configuration error: {mig['error']}"
            )
        if mig.get("instances"):
            lines.append("  Modeled MIG instances:")
            for instance in mig["instances"]:
                lines.append(
                    "    - "
                    f"GI {instance['gpu_instance_id']} / "
                    f"CI {instance['compute_instance_id']}: "
                    f"{instance['profile']}, "
                    f"{_format_bytes(instance['memory_total_bytes'])}, "
                    f"{instance['uuid']}"
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


def _table_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _mib(value: int) -> int:
    return int(round(int(value) / 2**20))
