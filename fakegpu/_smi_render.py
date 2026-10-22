"""Terminal and machine-readable renderings of an SMI inventory.

Split out of ``smi`` unchanged: the device table, the topology matrix,
NVLink and MIG views, health events, the per-device detail page, and the
``--query-*`` report, plus the value formatting they share.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ._smi_environment import (
    FAULT_SEVERITY_RANK,
    _maximum_fault_severity,
    _nonnegative_int,
)
from .link_topology import placement_relation
from .link_traffic import copies_summary, throughput_summary
from ._smi_query import (
    QUERY_SCHEMA_VERSION,
    QueryField,
    _has_native_activity,
    _query_csv_header,
    _query_csv_value,
    _query_timestamp,
    _query_value,
)
from ._smi_state import (
    DEFAULT_STALE_AFTER_SECONDS,
    MODELED_DRIVER_VERSION,
    build_inventory,
)


def render_table(
    states: Sequence[dict[str, Any]],
    *,
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
    driver_versions = sorted(
        {
            str(item["software"].get("driver_version") or MODELED_DRIVER_VERSION)
            for item in inventory["runtimes"]
        }
    ) or [MODELED_DRIVER_VERSION]
    # nvidia-smi's banner carries "Driver Version: X " and "CUDA Version: Y";
    # torch and vLLM collect_env read them with r"Driver Version: (.*?) ".
    lines = [
        "FakeGPU-SMI (simulated CUDA telemetry)",
        (
            f"FakeGPU Version: {', '.join(versions) if versions else 'unknown'}"
            f" | Driver Version: {', '.join(driver_versions)} (modeled)"
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
    return "\n".join(lines)


def render_gpu_list(inventory: Mapping[str, Any]) -> str:
    """List GPUs in nvidia-smi -L's format.

    nvidia-smi 580's format strings are "GPU %u: %s (%s: %s)" and, for MIG
    devices, "  %-15s Device %2u: (%s: %s)" with "MIG <profile>" as the
    first field. Profile, compute capability and PCI bus ID are in -q and
    --json. Only a view of several hosts names each GPU's host.
    """

    devices = list(inventory.get("devices") or [])
    several_hosts = len({str(item["host"]) for item in devices}) > 1
    lines = []
    for item in devices:
        host = f" @ {item['host']}" if several_hosts else ""
        lines.append(
            f"GPU {item['index']}{host}: {item['name']} (UUID: {item['uuid']})"
        )
        for instance in item["mig"]["instances"]:
            lines.append(
                f"  {'MIG ' + str(instance['profile']):<15} "
                f"Device {instance['index']:>2}: (UUID: {instance['uuid']})"
            )
    if not lines:
        lines.append("No published FakeCUDA GPUs found.")
    return "\n".join(lines)


# nvidia-smi topo -m's legend, as nvidia-smi 580.178.04 prints it.
_TOPOLOGY_LEGEND = (
    "Legend:",
    "",
    "  X    = Self",
    "  SYS  = Connection traversing PCIe as well as the SMP interconnect "
    "between NUMA nodes (e.g., QPI/UPI)",
    "  NODE = Connection traversing PCIe as well as the interconnect between "
    "PCIe Host Bridges within a NUMA node",
    "  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge "
    "(typically the CPU)",
    "  PXB  = Connection traversing multiple PCIe bridges (without "
    "traversing the PCIe Host Bridge)",
    "  PIX  = Connection traversing at most a single PCIe bridge",
    "  NV#  = Connection traversing a bonded set of # NVLinks",
)


def render_topology_matrix(inventory: Mapping[str, Any]) -> str:
    """Render ``nvidia-smi topo -m``'s matrix and legend, then FakeGPU's notes.

    nvidia-smi 580 separates cells with tabs and underlines the header row
    with ANSI escapes even when stdout is not a terminal. The self cell is
    " X ", and an empty cell follows NUMA Affinity. Callers split the output
    on "Legend": torch's inductor looks for "NV" before it to detect NVLink,
    so FakeGPU's host, platform and source notes come after the legend.
    """

    devices = list(inventory.get("devices") or [])
    if not devices:
        return "No published FakeCUDA GPUs found."
    labels = _topology_labels(devices)
    header = [*labels, "CPU Affinity", "NUMA Affinity", "GPU NUMA ID"]
    lines = ["\t\x1b[4m" + "\t".join(header) + "\x1b[0m"]
    for label, device in zip(labels, devices):
        slot = device["topology"].get("slot") or {}
        numa = slot.get("numa_node") if slot else None
        cells = [
            " X " if target is device else _topology_relation(device, target)
            for target in devices
        ]
        # CPU affinity and the GPU's own NUMA ID are not modeled.
        numa_text = "N/A" if numa is None else str(numa)
        lines.append("\t".join([label, *cells, "N/A", numa_text, "", "N/A"]))
    lines.extend(["", *_TOPOLOGY_LEGEND, "", "FakeGPU modeled topology matrix"])
    devices_by_host: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    for label, device in zip(labels, devices):
        devices_by_host.setdefault(str(device["host"]), []).append(
            (label, device)
        )
    for host, entries in devices_by_host.items():
        host_devices = [device for _label, device in entries]
        host_labels = ", ".join(label for label, _device in entries)
        lines.append(f"Host: {host} ({host_labels})")
        platforms = sorted(
            {
                str(device["topology"].get("platform") or "")
                for device in host_devices
            }
            - {""}
        )
        for platform in platforms:
            lines.append(f"Platform: {_platform_label(platform)}")
        source_values = sorted(
            {device["topology"]["source"] for device in host_devices}
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
    if len(devices_by_host) > 1:
        lines.append("GPUs on different hosts show SYS.")
    lines.append("CPU Affinity and GPU NUMA ID are not modeled.")
    return "\n".join(lines)


def render_nvlink_status(inventory: Mapping[str, Any]) -> str:
    devices = list(inventory.get("devices") or [])
    lines = [
        "FakeGPU modeled NVLink status",
        (
            "Bandwidth values are configuration inputs, not measurements "
            "(GB/s per link and direction)."
        ),
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
        lines.append(f"  Source: {nvlink['source']}")
        if topology.get("error"):
            lines.append(
                f"  Configuration error: {topology['error']}"
            )
        active_links = [link for link in nvlink["links"] if link["active"]]
        if not active_links:
            lines.append("  No active modeled NVLinks.")
            continue
        for link in sorted(active_links, key=lambda item: item["link"]):
            if link["remote"] == "gpu":
                remote = (
                    f"GPU {link['index']} ({link['uuid']}, "
                    f"{link['pci_bus_id']})"
                )
            else:
                remote = f"NVSwitch ({link['pci_bus_id']})"
            lines.append(
                f"  Link {link['link']}: Active -> {remote}, "
                f"{_format_gbps(link['bandwidth_gbps'])} GB/s"
            )
    return "\n".join(lines)


def render_mig_view(
    inventory: Mapping[str, Any],
    *,
    instance_kind: str = "all",
) -> str:
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
                    f"{mig['mode'].replace('_', ' ').title()} "
                    f"({mig['source']})"
                ),
                (
                    "  Capacity: "
                    f"{mig['instance_count']} instances, "
                    f"{_format_bytes(mig['allocated_memory_bytes'])} "
                    "allocated / "
                    f"{_format_bytes(mig['unallocated_memory_bytes'])} "
                    "unallocated"
                ),
            ]
        )
        if mig.get("error"):
            lines.append(
                f"  Configuration error: {mig['error']}"
            )
        instances = mig["instances"]
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
    return "\n".join(lines)


def render_health_events(inventory: Mapping[str, Any]) -> str:
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
    return "\n".join(lines)


def _health_event_records(
    inventory: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    configuration_keys: set[tuple[str, str, str]] = set()

    def add_configuration(host: str, code: str, block: Mapping[str, Any]) -> None:
        if not block["error"]:
            return
        detail = block["error"]
        key = (host, code, detail)
        if key in configuration_keys:
            return
        configuration_keys.add(key)
        records.append(
            {
                "severity": "error",
                "kind": "configuration",
                "host": host,
                "gpu": "N/A",
                "code": code,
                "count": 1,
                "source": block["source"],
                "detail": detail,
            }
        )

    for device in inventory.get("devices") or []:
        host = device["host"]
        gpu = device["index"]
        health = device["health"]
        add_configuration(host, "FAULT_CONFIG", health)
        for event in health["events"]:
            records.append(
                {
                    "severity": event["severity"],
                    "kind": "modeled_fault",
                    "host": host,
                    "gpu": gpu,
                    "code": event["code"],
                    "count": event["count"],
                    "source": event["source"],
                    "detail": (
                        "configured through FAKEGPU_FAULT_EVENTS"
                    ),
                }
            )
        add_configuration(host, "NVLINK_CONFIG", device["topology"])
        for event in device["native_activity"]["unsupported_apis"]:
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
        publisher = runtime["publisher"]
        health = publisher.get("health")
        if not isinstance(health, Mapping):
            health = {}
        failures = _nonnegative_int(health.get("failed_writes"))
        if failures:
            records.append(
                {
                    "severity": "error",
                    "kind": "publisher",
                    "host": runtime["host"],
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
                        f"pid {runtime['pid']}; "
                        f"{health.get('successful_writes', 0)} "
                        "successful writes"
                    ),
                }
            )
        if runtime["status"] == "stale":
            records.append(
                {
                    "severity": "warning",
                    "kind": "state",
                    "host": runtime["host"],
                    "gpu": "N/A",
                    "code": "STALE_STATE",
                    "count": 1,
                    "source": "state_freshness",
                    "detail": (
                        f"pid {runtime['pid']}; age "
                        f"{_format_age(runtime['age_seconds'])}"
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


def render_detail(inventory: Mapping[str, Any]) -> str:
    lines = ["FakeGPU-SMI detailed report"]
    runtimes = list(inventory.get("runtimes") or [])
    if runtimes:
        lines.append("Runtime processes:")
    for item in runtimes:
        lines.extend(_runtime_detail_lines(item))

    devices = list(inventory.get("devices") or [])
    if not devices:
        lines.append("No published FakeCUDA GPUs found.")
    for item in devices:
        lines.extend(_device_detail_lines(item))
    return "\n".join(lines)


def _runtime_detail_lines(item: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    fakegpu = item["fakegpu"]
    software = item["software"]
    dispatch = item["dispatch_tracking"]
    publisher = item["publisher"]
    publisher_health = publisher.get("health")
    publisher_health = (
        dict(publisher_health)
        if isinstance(publisher_health, Mapping)
        else {}
    )
    publisher_limits = publisher.get("limits")
    publisher_limits = (
        dict(publisher_limits)
        if isinstance(publisher_limits, Mapping)
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
                f"driver {software.get('driver_version') or 'N/A'}"
                f"{' (modeled)' if software.get('driver_version_source') == 'modeled_default' else ''}, "
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
    return lines


def _device_detail_lines(item: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
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
                f"{topology['source']}, "
                f"{nvlink['active_links']} active NVLinks / "
                f"{_format_gbps(nvlink['aggregate_bandwidth_gbps'])} "
                "GB/s modeled aggregate"
            ),
            f"  Slot: {_slot_text(topology)}",
            f"  Copies (modeled): {_copies_text(item['transfer'])}",
            (
                "  Link throughput (modeled): "
                f"{_throughput_text(item['transfer'], topology)}"
            ),
            (
                "  Health model: "
                f"{health['status']}, "
                "hardware "
                f"{health['hardware_health']}, "
                "maximum severity "
                f"{health['max_severity']}, "
                f"{health['event_types_total']} event types / "
                f"{health['event_count']} occurrences"
            ),
            (
                "  MIG model: "
                f"{mig['mode']}, "
                f"{mig['instance_count']} instances, "
                f"{_format_bytes(mig['allocated_memory_bytes'])} "
                "allocated / "
                f"{_format_bytes(mig['unallocated_memory_bytes'])} "
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
                f"[{event['severity']}] "
                f"{event['code']}: "
                f"{event['count']} "
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
    return lines


def render_query(
    records: Sequence[Mapping[str, Any]],
    *,
    fields: Sequence[str],
    available: Mapping[str, QueryField],
    query_kind: str,
    output_format: Mapping[str, Any],
) -> str:
    # nvidia-smi stamps every row with the time of the query, not of the data.
    stamp = _query_timestamp()
    values = [
        {
            field: (
                stamp
                if available[field].path == "timestamp"
                else _query_value(record, available[field])
            )
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

    rows = []
    if not output_format["noheader"]:
        rows.append(
            _csv_row(
                _query_csv_header(field, available[field])
                for field in fields
            )
        )
    for record in values:
        rows.append(
            _csv_row(
                _query_csv_value(
                    record[field],
                    spec=available[field],
                    nounits=bool(output_format["nounits"]),
                )
                for field in fields
            )
        )
    return "\n".join(rows)


def _csv_row(cells: Iterable[Any]) -> str:
    """Join cells like nvidia-smi, with a comma and a space.

    Scripts often split nvidia-smi output on ``", "``. Only a cell that
    contains a comma, quote or newline is quoted, as in RFC 4180.
    """

    quoted = []
    for cell in cells:
        text = str(cell)
        if any(character in text for character in ',"\n'):
            text = '"' + text.replace('"', '""') + '"'
        quoted.append(text)
    return ", ".join(quoted)


def _format_age(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.3f}s"


def _topology_labels(
    devices: Sequence[Mapping[str, Any]],
) -> list[str]:
    totals: dict[int, int] = {}
    for device in devices:
        index = device["index"]
        totals[index] = totals.get(index, 0) + 1
    offsets: dict[int, int] = {}
    labels = []
    for device in devices:
        index = device["index"]
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
    target_uuid = target["uuid"]
    target_index = target["index"]
    topology = source["topology"]
    for peer in topology["nvlink"]["peers"]:
        if not peer["active"]:
            continue
        peer_uuid = peer["uuid"]
        if (
            peer_uuid != "unknown"
            and peer_uuid == target_uuid
        ) or (
            peer_uuid == "unknown"
            and peer["index"] == target_index
        ):
            return f"NV{peer['links']}"
    if str(source.get("host") or "") != str(target.get("host") or ""):
        return "SYS"
    left = topology.get("slot")
    right = target.get("topology", {}).get("slot")
    if not left or not right:
        return "PIX"
    return placement_relation(
        (left.get("numa_node"), left.get("host_bridge"), left.get("pcie_switch")),
        (right.get("numa_node"), right.get("host_bridge"), right.get("pcie_switch")),
    )


def _platform_label(platform_id: str) -> str:
    from .platforms import load_platforms

    platform = load_platforms().get(platform_id)
    return f"{platform.name} ({platform_id})" if platform else platform_id


def _copies_text(transfer: Mapping[str, Any] | None) -> str:
    if not transfer:
        return "not recorded"
    return copies_summary(transfer, _format_bytes)


def _throughput_text(
    transfer: Mapping[str, Any] | None,
    topology: Mapping[str, Any],
) -> str:
    if not transfer:
        return "not recorded"
    links = []
    if topology.get("pcie") or not topology.get("slot"):
        links.append("pcie")
    if transfer.get("host_link") == "c2c":
        links.append("c2c")
    if topology["nvlink"]["active_links"]:
        links.append("nvlink")
    window = transfer["window_seconds"]
    return f"{throughput_summary(transfer, links)} (last {window:g} s)"


def _slot_text(topology: Mapping[str, Any]) -> str:
    slot = topology.get("slot")
    if not slot:
        return "not modeled"
    parts = [slot["type"]]
    if topology.get("platform"):
        parts[0] += f" on {topology['platform']}"
    pcie = topology.get("pcie")
    if pcie:
        text = f"PCIe Gen{pcie['generation']} x{pcie['width']}"
        if (pcie["gpu_max_generation"], pcie["gpu_max_width"]) != (
            pcie["generation"],
            pcie["width"],
        ):
            text += (
                f" (GPU max Gen{pcie['gpu_max_generation']} "
                f"x{pcie['gpu_max_width']})"
            )
        parts.append(text)
    else:
        parts.append("no PCIe link")
    host = topology["host_link"]
    if host["bandwidth_gbps"] is not None:
        parts.append(
            f"host {host['kind']} {_format_gbps(host['bandwidth_gbps'])} GB/s"
        )
    parts.append(f"NUMA {slot['numa_node']}")
    return ", ".join(parts)


def _format_gbps(value: float | None) -> str:
    return "N/A" if value is None else f"{value:g}"


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
