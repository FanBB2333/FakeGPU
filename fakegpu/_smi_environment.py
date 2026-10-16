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
    except (TypeError, ValueError):
        return default


def _synthetic_mig_uuid(
    parent_uuid: str,
    gpu_instance_id: int,
    compute_instance_id: int,
) -> str:
    suffix = (
        parent_uuid[4:]
        if parent_uuid.startswith("GPU-")
        else parent_uuid
    )
    return (
        f"MIG-{suffix}/{gpu_instance_id}/{compute_instance_id}"
    )


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


NVSWITCH_PCI_BUS_ID = "00000000:c0:00.0"


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
    allocated_memory = {
        index: 0 for index in device_by_index
    }
    allocated_slices = {
        index: 0 for index in device_by_index
    }

    if configured:
        source = "modeled_environment"
        normalized_text = str(layout_text).strip()
        if (
            not normalized_text
            or normalized_text.startswith(";")
            or normalized_text.endswith(";")
        ):
            valid = False
            error = (
                "FAKEGPU_MIG_LAYOUT contains an empty instance entry"
            )
        else:
            for entry_index, entry_text in enumerate(
                normalized_text.split(";"),
                start=1,
            ):
                fields = [
                    field.strip() for field in entry_text.split(":")
                ]
                prefix = f"FAKEGPU_MIG_LAYOUT entry {entry_index}"
                if (
                    len(fields) not in {3, 4}
                    or any(not field for field in fields)
                ):
                    valid = False
                    error = (
                        f"{prefix} must use "
                        "DEVICE:PROFILE:MEMORY_MIB[:COUNT]"
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

                profile = fields[1].lower()
                separator = profile.find("g.")
                profile_valid = (
                    0 < len(profile)
                    <= MAXIMUM_MODELED_MIG_PROFILE_LENGTH
                    and all(
                        character.isascii()
                        and (
                            character.isalnum()
                            or character in "._-+"
                        )
                        for character in profile
                    )
                    and separator > 0
                    and profile[:separator].isascii()
                    and profile[:separator].isdigit()
                )
                slice_count = (
                    int(profile[:separator])
                    if profile_valid
                    else 0
                )
                if not (
                    profile_valid
                    and 1
                    <= slice_count
                    <= MAXIMUM_MODELED_MIG_SLICES_PER_DEVICE
                ):
                    valid = False
                    error = (
                        f"{prefix} contains an invalid MIG profile; "
                        "expected a name such as 1g.10gb"
                    )
                    break

                memory_text = fields[2]
                unsigned_memory = (
                    memory_text[1:]
                    if memory_text.startswith("+")
                    else memory_text
                )
                if (
                    not unsigned_memory.isascii()
                    or not unsigned_memory.isdigit()
                ):
                    memory_mib = 0
                else:
                    memory_mib = int(memory_text)
                parent = device_by_index[device_index]
                parent_memory = _nonnegative_int(
                    parent.get("total_memory")
                )
                memory_bytes = memory_mib * 2**20
                if (
                    memory_mib <= 0
                    or memory_bytes > parent_memory
                ):
                    valid = False
                    error = (
                        f"{prefix} memory must be a positive MiB "
                        "value no greater than the parent device memory"
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
                        count = 0
                    else:
                        count = int(count_text)
                if not (
                    1
                    <= count
                    <= MAXIMUM_MODELED_MIG_INSTANCES_PER_DEVICE
                ):
                    valid = False
                    error = f"{prefix} count must be between 1 and 8"
                    break

                current_instances = instances_by_index[device_index]
                if (
                    len(current_instances) + count
                    > MAXIMUM_MODELED_MIG_INSTANCES_PER_DEVICE
                ):
                    valid = False
                    error = (
                        f"{prefix} exceeds the 8-instance "
                        "per-device limit"
                    )
                    break
                if (
                    allocated_slices[device_index]
                    + slice_count * count
                    > MAXIMUM_MODELED_MIG_SLICES_PER_DEVICE
                ):
                    valid = False
                    error = (
                        f"{prefix} exceeds the 8-slice "
                        "per-device limit"
                    )
                    break
                if (
                    allocated_memory[device_index]
                    + memory_bytes * count
                    > parent_memory
                ):
                    valid = False
                    error = (
                        f"{prefix} allocates more memory than the "
                        "parent device"
                    )
                    break

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
                                str(parent.get("uuid") or "unknown"),
                                instance_index,
                                instance_index,
                            ),
                            "parent_uuid": str(
                                parent.get("uuid") or "unknown"
                            ),
                            "pci_bus_id": str(
                                parent.get("pci_bus_id") or "unknown"
                            ),
                            "memory_total_bytes": memory_bytes,
                            "memory_used_bytes": None,
                            "memory_free_bytes": None,
                            "memory_tracking": "unobserved",
                            "source": source,
                        }
                    )
                allocated_slices[device_index] += (
                    slice_count * count
                )
                allocated_memory[device_index] += (
                    memory_bytes * count
                )

    if configured and valid and not any(instances_by_index.values()):
        valid = False
        error = "FAKEGPU_MIG_LAYOUT did not define any MIG instances"

    if configured and not valid:
        source = "modeled_environment_invalid"
        instances_by_index = {
            index: [] for index in device_by_index
        }
        allocated_memory = {
            index: 0 for index in device_by_index
        }

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
            "mode": (
                "configuration_error"
                if not valid
                else "enabled"
                if instances
                else "disabled"
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
