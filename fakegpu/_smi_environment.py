"""Device topology, fault, and MIG models parsed from FAKEGPU_* variables.

These are pure functions over the environment: they turn the platform and
link variables (``FAKEGPU_PLATFORM``, ``FAKEGPU_SLOTS``,
``FAKEGPU_NVLINK_FABRIC``, ``FAKEGPU_NVLINK_GROUPS``),
``FAKEGPU_FAULT_EVENTS``, and ``FAKEGPU_MIG_LAYOUT`` into the modeled
topology, fault, and MIG structures the publisher writes into its state
file and the readers normalize back out.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping, Sequence
from typing import Any


MAXIMUM_MODELED_FAULT_TYPES = 128


MAXIMUM_MODELED_FAULT_COUNT = 1_000_000_000


MAXIMUM_MODELED_MIG_INSTANCES_PER_DEVICE = 8


MAXIMUM_MODELED_MIG_SLICES_PER_DEVICE = 8


MAXIMUM_MODELED_MIG_PROFILE_LENGTH = 64


FAULT_SEVERITY_RANK = {
    "none": 0,
    "info": 1,
    "warning": 2,
    "error": 3,
    "critical": 4,
}


def _nonnegative_int(value: Any, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return default


def _synthetic_mig_uuid(
    parent_uuid: str,
    gpu_instance_id: int,
    compute_instance_id: int,
) -> str:
    """A MIG instance UUID in the modern ``MIG-`` + 8-4-4-4-12 form.

    nvidia-smi -L prints MIG devices as ``(UUID: MIG-xxxxxxxx-xxxx-...)``; the
    native core derives the same value (src/core/device.cpp
    synthetic_mig_uuid).
    """

    identity = uuid.uuid5(
        uuid.NAMESPACE_DNS,
        f"fakegpu:mig:{parent_uuid}:{gpu_instance_id}:{compute_instance_id}",
    )
    return f"MIG-{identity}"


def _modeled_device_topology(
    devices: list[dict[str, Any]],
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Attach each device's slot, PCIe link and NVLinks; return the summary.

    The model comes from ``FAKEGPU_PLATFORM``/``FAKEGPU_SLOTS``/
    ``FAKEGPU_NVLINK_FABRIC`` (or the older ``FAKEGPU_NVLINK_GROUPS``) in
    ``environ`` (default: the process environment); see
    :mod:`fakegpu.link_topology`.
    """

    from .platforms import topology_from_environment

    ordered = sorted(devices, key=lambda item: _nonnegative_int(item.get("index")))
    topology = topology_from_environment(
        [str(device.get("profile_id") or "") for device in ordered],
        os.environ if environ is None else environ,
    )
    for position, device in enumerate(ordered):
        device["topology"] = device_topology_record(topology, position, ordered)
    count = len(ordered)
    links = []
    for left in range(count):
        for peer in topology.peers(left):
            if peer.index <= left:
                continue
            links.append(
                {
                    "source_index": _nonnegative_int(ordered[left].get("index")),
                    "target_index": _nonnegative_int(ordered[peer.index].get("index")),
                    "source_uuid": str(ordered[left].get("uuid") or "unknown"),
                    "target_uuid": str(ordered[peer.index].get("uuid") or "unknown"),
                    "kind": "NVLink",
                    "via": peer.via,
                    "links": peer.links,
                    "active": True,
                    "bandwidth_gbps": peer.bandwidth_gbps,
                    "source": topology.nvlink_source,
                }
            )
    link_rates = [group.link_gbps for group in topology.nvlink]
    return {
        "schema_version": "fakegpu.device_topology.v2",
        "source": topology.source,
        "platform": topology.platform,
        "configured": topology.configured,
        "valid": topology.valid,
        "error": topology.error,
        "nvlink_bandwidth_gbps": (
            link_rates[0] if link_rates else topology.legacy_bandwidth_gbps
        ),
        "link_count": len(links),
        "links": links,
        "matrix": [
            [topology.relation(row, column) for column in range(count)]
            for row in range(count)
        ],
    }


NVSWITCH_PCI_BUS_ID = "00000000:C0:00.0"


def device_topology_record(
    topology: Any,
    position: int,
    devices: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """The ``topology`` block of one device in a published SMI state."""

    slot = topology.slots[position]
    ports = topology.ports(position)
    peers = topology.peers(position)

    def identity(index: int | None) -> tuple[str, str]:
        if index is None:
            return "", NVSWITCH_PCI_BUS_ID
        remote = devices[index]
        return (
            str(remote.get("uuid") or "unknown"),
            str(remote.get("pci_bus_id") or "unknown"),
        )

    pcie = None
    if slot.pcie is not None:
        gpu_max = slot.gpu_max_pcie or slot.pcie
        pcie = {
            "generation": slot.pcie.generation,
            "width": slot.pcie.width,
            "max_generation": slot.pcie.generation,
            "max_width": slot.pcie.width,
            "gpu_max_generation": gpu_max.generation,
            "gpu_max_width": gpu_max.width,
            "bandwidth_gbps": slot.pcie.bandwidth_gbps,
        }
    links = []
    for port in ports:
        uuid, bus_id = identity(port.peer)
        links.append(
            {
                "link": port.link,
                "remote": port.remote,
                "index": (
                    None
                    if port.peer is None
                    else _nonnegative_int(devices[port.peer].get("index"))
                ),
                "uuid": uuid,
                "pci_bus_id": bus_id,
                "bandwidth_gbps": port.bandwidth_gbps,
                "active": True,
            }
        )
    peer_rows = []
    for peer in peers:
        uuid, bus_id = identity(peer.index)
        peer_rows.append(
            {
                "link": peer.first_link,
                "index": _nonnegative_int(devices[peer.index].get("index")),
                "uuid": uuid,
                "pci_bus_id": bus_id,
                "links": peer.links,
                "via": peer.via,
                "bandwidth_gbps": peer.bandwidth_gbps,
                "active": True,
                "source": topology.nvlink_source,
            }
        )
    return {
        "source": topology.source,
        "platform": topology.platform,
        "configured": topology.configured,
        "valid": topology.valid,
        "error": topology.error,
        "slot": {
            "type": slot.type,
            "numa_node": slot.numa,
            "host_bridge": slot.bridge,
            "pcie_switch": slot.switch,
        },
        "pcie": pcie,
        "host_link": {
            "kind": slot.host_link,
            "bandwidth_gbps": slot.host_bandwidth_gbps,
        },
        "peer_access": [
            _nonnegative_int(devices[index].get("index"))
            for index in range(len(devices))
            if topology.can_access_peer(position, index)
        ],
        "nvlink": {
            "source": topology.nvlink_source,
            "active_links": len(ports),
            "peer_count": len(peers),
            "link_bandwidth_gbps": ports[0].bandwidth_gbps if ports else 0.0,
            "aggregate_bandwidth_gbps": sum(port.bandwidth_gbps for port in ports),
            "links": links,
            "peers": peer_rows,
        },
    }


class _ModelError(ValueError):
    """A FAKEGPU_FAULT_EVENTS or FAKEGPU_MIG_LAYOUT configuration error."""


def _unsigned_int(text: str) -> int | None:
    unsigned = text[1:] if text.startswith("+") else text
    if not unsigned.isascii() or not unsigned.isdigit():
        return None
    return int(unsigned)


def _parse_fault_events(
    text: str,
    device_by_index: Mapping[int, Any],
) -> dict[tuple[int, str, str], int]:
    normalized_text = text.strip()
    if (
        not normalized_text
        or normalized_text.startswith(";")
        or normalized_text.endswith(";")
    ):
        raise _ModelError("FAKEGPU_FAULT_EVENTS contains an empty event")
    counts: dict[tuple[int, str, str], int] = {}
    for event_index, event_text in enumerate(
        normalized_text.split(";"),
        start=1,
    ):
        fields = [field.strip() for field in event_text.split(":")]
        prefix = f"FAKEGPU_FAULT_EVENTS event {event_index}"
        if len(fields) not in {3, 4} or any(not field for field in fields):
            raise _ModelError(
                f"{prefix} must use DEVICE:CODE:SEVERITY[:COUNT]"
            )
        device_index = _unsigned_int(fields[0])
        if device_index not in device_by_index:
            raise _ModelError(f"{prefix} contains an invalid device index")

        code = fields[1].upper()
        if len(code) > 64 or any(
            not character.isascii()
            or (not character.isalnum() and character not in "_.-")
            for character in code
        ):
            raise _ModelError(f"{prefix} contains an invalid code")

        severity = fields[2].lower()
        if FAULT_SEVERITY_RANK.get(severity, 0) == 0:
            raise _ModelError(
                f"{prefix} severity must be info, warning, "
                "error, or critical"
            )

        count = 1
        if len(fields) == 4:
            count = _unsigned_int(fields[3])
            if count is None or not 1 <= count <= MAXIMUM_MODELED_FAULT_COUNT:
                raise _ModelError(
                    f"{prefix} count must be between 1 and 1000000000"
                )

        key = (device_index, code, severity)
        counts[key] = min(
            MAXIMUM_MODELED_FAULT_COUNT,
            counts.get(key, 0) + count,
        )
        if len(counts) > MAXIMUM_MODELED_FAULT_TYPES:
            raise _ModelError(
                "FAKEGPU_FAULT_EVENTS exceeds the 128-event-type limit"
            )
    return counts


def _health_summary(
    events: list[dict[str, Any]],
    *,
    source: str,
    configured: bool,
    valid: bool,
    error: str,
    detail_limit: int,
) -> dict[str, Any]:
    maximum_severity = (
        "error" if not valid else _maximum_fault_severity(events)
    )
    retained = events[:detail_limit]
    return {
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
        try:
            counts = _parse_fault_events(events_text, device_by_index)
            source = "modeled_environment"
        except _ModelError as exc:
            valid, error, source = False, str(exc), "modeled_environment_invalid"

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

    summary = {
        "source": source,
        "configured": configured,
        "valid": valid,
        "error": error,
        "detail_limit": detail_limit,
    }
    for device_index, device in device_by_index.items():
        device["health"] = _health_summary(
            [
                event
                for event in events
                if event["device_index"] == device_index
            ],
            **summary,
        )
    return {
        "schema_version": "fakegpu.fault_model.v1",
        **_health_summary(events, **summary),
    }


def _parse_mig_layout(
    text: str,
    device_by_index: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, int]]:
    instances_by_index: dict[int, list[dict[str, Any]]] = {
        index: [] for index in device_by_index
    }
    allocated_memory = {index: 0 for index in device_by_index}
    allocated_slices = {index: 0 for index in device_by_index}
    normalized_text = text.strip()
    if (
        not normalized_text
        or normalized_text.startswith(";")
        or normalized_text.endswith(";")
    ):
        raise _ModelError(
            "FAKEGPU_MIG_LAYOUT contains an empty instance entry"
        )
    for entry_index, entry_text in enumerate(
        normalized_text.split(";"),
        start=1,
    ):
        fields = [field.strip() for field in entry_text.split(":")]
        prefix = f"FAKEGPU_MIG_LAYOUT entry {entry_index}"
        if len(fields) not in {3, 4} or any(not field for field in fields):
            raise _ModelError(
                f"{prefix} must use DEVICE:PROFILE:MEMORY_MIB[:COUNT]"
            )

        device_index = _unsigned_int(fields[0])
        if device_index not in device_by_index:
            raise _ModelError(f"{prefix} contains an invalid device index")

        profile = fields[1].lower()
        separator = profile.find("g.")
        profile_valid = (
            0 < len(profile) <= MAXIMUM_MODELED_MIG_PROFILE_LENGTH
            and all(
                character.isascii()
                and (character.isalnum() or character in "._-+")
                for character in profile
            )
            and separator > 0
            and profile[:separator].isdigit()
        )
        slice_count = int(profile[:separator]) if profile_valid else 0
        if not (
            profile_valid
            and 1 <= slice_count <= MAXIMUM_MODELED_MIG_SLICES_PER_DEVICE
        ):
            raise _ModelError(
                f"{prefix} contains an invalid MIG profile; "
                "expected a name such as 1g.10gb"
            )

        memory_mib = _unsigned_int(fields[2]) or 0
        parent = device_by_index[device_index]
        parent_memory = _nonnegative_int(parent.get("total_memory"))
        memory_bytes = memory_mib * 2**20
        if memory_mib <= 0 or memory_bytes > parent_memory:
            raise _ModelError(
                f"{prefix} memory must be a positive MiB "
                "value no greater than the parent device memory"
            )

        count = 1
        if len(fields) == 4:
            count = _unsigned_int(fields[3]) or 0
        if not 1 <= count <= MAXIMUM_MODELED_MIG_INSTANCES_PER_DEVICE:
            raise _ModelError(f"{prefix} count must be between 1 and 8")

        current_instances = instances_by_index[device_index]
        if (
            len(current_instances) + count
            > MAXIMUM_MODELED_MIG_INSTANCES_PER_DEVICE
        ):
            raise _ModelError(
                f"{prefix} exceeds the 8-instance per-device limit"
            )
        if (
            allocated_slices[device_index] + slice_count * count
            > MAXIMUM_MODELED_MIG_SLICES_PER_DEVICE
        ):
            raise _ModelError(
                f"{prefix} exceeds the 8-slice per-device limit"
            )
        if (
            allocated_memory[device_index] + memory_bytes * count
            > parent_memory
        ):
            raise _ModelError(
                f"{prefix} allocates more memory than the parent device"
            )

        parent_uuid = str(parent.get("uuid") or "unknown")
        for _ in range(count):
            instance_index = len(current_instances)
            current_instances.append(
                {
                    "index": instance_index,
                    "gpu_instance_id": instance_index,
                    "compute_instance_id": instance_index,
                    "profile": profile,
                    "slice_count": slice_count,
                    "uuid": _synthetic_mig_uuid(
                        parent_uuid,
                        instance_index,
                        instance_index,
                    ),
                    "parent_uuid": parent_uuid,
                    "pci_bus_id": str(parent.get("pci_bus_id") or "unknown"),
                    "memory_total_bytes": memory_bytes,
                    "memory_used_bytes": None,
                    "memory_free_bytes": None,
                    "memory_tracking": "unobserved",
                    "source": "modeled_environment",
                }
            )
        allocated_slices[device_index] += slice_count * count
        allocated_memory[device_index] += memory_bytes * count
    return instances_by_index, allocated_memory


def _modeled_mig_layout(
    devices: list[dict[str, Any]],
) -> dict[str, Any]:
    layout_text = os.environ.get("FAKEGPU_MIG_LAYOUT")
    configured = bool(layout_text)
    valid = True
    error = ""
    source = "modeled_none"
    device_by_index = {
        _nonnegative_int(device.get("index")): device
        for device in devices
    }
    instances_by_index: dict[int, list[dict[str, Any]]] = {
        index: [] for index in device_by_index
    }
    allocated_memory = {index: 0 for index in device_by_index}

    if configured:
        try:
            instances_by_index, allocated_memory = _parse_mig_layout(
                layout_text,
                device_by_index,
            )
            source = "modeled_environment"
        except _ModelError as exc:
            valid, error, source = False, str(exc), "modeled_environment_invalid"

    enabled_device_count = 0
    instance_count = 0
    allocated_memory_total = 0
    for index, device in device_by_index.items():
        instances = instances_by_index[index]
        if instances:
            enabled_device_count += 1
        instance_count += len(instances)
        allocated = allocated_memory[index]
        allocated_memory_total += allocated
        total = _nonnegative_int(device.get("total_memory"))
        device["mig"] = {
            "source": source,
            "configured": configured,
            "valid": valid,
            "error": error,
            "mode": _modeled_mig_mode(
                valid=valid,
                has_instances=bool(instances),
            ),
            "max_instance_count": (
                MAXIMUM_MODELED_MIG_INSTANCES_PER_DEVICE
            ),
            "instance_count": len(instances),
            "allocated_memory_bytes": allocated,
            "unallocated_memory_bytes": max(0, total - allocated),
            "instances": instances,
        }

    return {
        "schema_version": "fakegpu.mig_layout.v1",
        "source": source,
        "configured": configured,
        "valid": valid,
        "error": error,
        "enabled_device_count": enabled_device_count,
        "instance_count": instance_count,
        "allocated_memory_bytes": allocated_memory_total,
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


def _modeled_mig_mode(*, valid: bool, has_instances: bool) -> str:
    if not valid:
        return "configuration_error"
    return "enabled" if has_instances else "disabled"
