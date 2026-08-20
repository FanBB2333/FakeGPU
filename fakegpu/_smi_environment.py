"""Device topology, fault, and MIG models parsed from FAKEGPU_* variables.

Split out of ``smi`` unchanged. These are pure functions over the
environment: they turn the ``FAKEGPU_NVLINK_GROUPS``,
``FAKEGPU_FAULT_EVENTS``, and ``FAKEGPU_MIG_LAYOUT`` specs into the modeled
topology, fault, and MIG structures the publisher writes into its state
file and the readers normalize back out.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any


DEFAULT_NVLINK_BANDWIDTH_GBPS = 900.0


MAXIMUM_NVLINK_BANDWIDTH_GBPS = 1_000_000.0


MAXIMUM_MODELED_NVLINKS_PER_DEVICE = 18


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
