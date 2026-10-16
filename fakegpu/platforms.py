"""Host platforms: which slots a machine has and how its GPUs are wired.

SXM modules sit on a baseboard (HGX/DGX) built for one GPU generation, with
NVSwitch or direct NVLink wiring; superchips pair GPUs with Grace CPUs over
NVLink-C2C; PCIe cards go into PCIe slots and may share NVLink bridges.
Templates live in ``data/platforms.json``. :func:`resolve_platform` places a
device list into a platform's slots and returns the resulting
:class:`~fakegpu.link_topology.LinkTopology`, or raises
:class:`PlatformError` with a hint when the GPUs do not fit.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .link_topology import (
    ENVIRONMENT_VARIABLES,
    LEGACY_NVLINK_VARIABLES,
    GpuSlot,
    LinkTopology,
    NvlinkGroup,
    PcieLink,
    nvlink_link_gbps,
    platform_data,
)
from .profile_catalog import GpuProfile, get_profile

AUTO = "auto"


class PlatformError(ValueError):
    pass


@dataclass(frozen=True)
class SlotTemplate:
    type: str
    pcie: str | None
    host_pcie: str | None = None
    numa: int = 0
    bridge: int = 0
    switch: str = ""
    host_link: str = "pcie"
    host_gbps: float | None = None
    p2p: bool = True
    shares_lanes_with: tuple[int, ...] = ()
    lanes_when_shared: int | None = None


@dataclass(frozen=True)
class Platform:
    id: str
    name: str
    kind: str
    summary: str
    sources: tuple[str, ...]
    accepts: tuple[str, ...]
    homogeneous: bool
    slots: tuple[SlotTemplate, ...]
    nvlink: Mapping[str, Any] | None
    notes: str = ""

    @property
    def capacity(self) -> int:
        return len(self.slots)

    @property
    def slot_types(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(slot.type for slot in self.slots))

    def describe_slots(self) -> str:
        counts: dict[str, int] = {}
        for slot in self.slots:
            label = _SLOT_LABELS.get(slot.type, slot.type)
            if slot.pcie and slot.pcie != "card" and slot.type in ("pcie", "laptop"):
                label += f" Gen{PcieLink.parse(slot.pcie).generation} x{PcieLink.parse(slot.pcie).width}"
            counts[label] = counts.get(label, 0) + 1
        return ", ".join(f"{count}x {label}" for label, count in counts.items())


@dataclass(frozen=True)
class ResolvedPlatform:
    platform: Platform
    profiles: tuple[str, ...]
    slot_indices: tuple[int, ...]
    topology: LinkTopology
    notes: tuple[str, ...] = ()

    def environment(self) -> dict[str, str]:
        return self.topology.environment()

    def to_dict(self) -> dict[str, Any]:
        topology = self.topology
        devices = []
        for index, profile_id in enumerate(self.profiles):
            slot = topology.slots[index]
            devices.append(
                {
                    "index": index,
                    "profile_id": profile_id,
                    "slot": self.slot_indices[index],
                    "slot_type": slot.type,
                    "pcie": slot.pcie.label if slot.pcie else None,
                    "gpu_max_pcie": slot.gpu_max_pcie.label if slot.gpu_max_pcie else None,
                    "host_link": slot.host_link,
                    "host_bandwidth_gbps": slot.host_bandwidth_gbps,
                    "numa_node": slot.numa,
                    "nvlink_links": len(topology.ports(index)),
                    "nvlink_bandwidth_gbps": sum(port.bandwidth_gbps for port in topology.ports(index)),
                }
            )
        count = len(self.profiles)
        return {
            "id": self.platform.id,
            "name": self.platform.name,
            "kind": self.platform.kind,
            "summary": self.platform.summary,
            "slots": self.platform.capacity,
            "slot_layout": self.platform.describe_slots(),
            "devices": devices,
            "matrix": [[topology.relation(row, column) for column in range(count)] for row in range(count)],
            "notes": list(self.notes),
        }


_SLOT_LABELS = {
    "sxm": "SXM",
    "pcie": "PCIe",
    "superchip": "superchip",
    "laptop": "laptop",
    "embedded": "embedded",
    "any": "any",
}


@lru_cache(maxsize=1)
def load_platforms() -> dict[str, Platform]:
    platforms: dict[str, Platform] = {}
    for raw in platform_data()["platforms"]:
        slots: list[SlotTemplate] = []
        for group in raw["slots"]:
            for _ in range(int(group.get("count", 1))):
                slots.append(
                    SlotTemplate(
                        type=group["type"],
                        pcie=group.get("pcie"),
                        host_pcie=group.get("host_pcie"),
                        numa=int(group.get("numa", 0)),
                        bridge=int(group.get("bridge", 0)),
                        switch=str(group.get("switch", "")),
                        host_link=group.get("host_link", "pcie"),
                        host_gbps=group.get("host_gbps"),
                        p2p=bool(group.get("p2p", True)),
                        shares_lanes_with=tuple(group.get("shares_lanes_with", ())),
                        lanes_when_shared=group.get("lanes_when_shared"),
                    )
                )
        platform = Platform(
            id=raw["id"],
            name=raw["name"],
            kind=raw["kind"],
            summary=raw["summary"],
            sources=tuple(raw.get("sources", ())),
            accepts=tuple(raw.get("accepts", ())),
            homogeneous=bool(raw.get("homogeneous", False)),
            slots=tuple(slots),
            nvlink=raw.get("nvlink"),
            notes=raw.get("notes", ""),
        )
        platforms[platform.id] = platform
    return platforms


def platform_ids() -> list[str]:
    return list(load_platforms())


def get_platform(platform_id: str) -> Platform:
    platforms = load_platforms()
    key = platform_id.strip().lower()
    if key not in platforms:
        raise PlatformError(f"unknown platform {platform_id!r}; choose auto or one of: {', '.join(platforms)}")
    return platforms[key]


def form_class(profile: GpuProfile) -> str:
    """Slot family a GPU needs; an unknown form factor is treated as a PCIe card."""

    return profile.form_factor if profile.form_factor in ("sxm", "superchip", "laptop", "embedded") else "pcie"


def select_platform(profile_ids: Sequence[str]) -> Platform:
    """The platform ``auto`` picks for this device list."""

    profiles = [get_profile(profile_id) for profile_id in profile_ids]
    if not profiles:
        raise PlatformError("the machine needs at least one device")
    platforms = load_platforms()
    classes = {form_class(profile) for profile in profiles}
    count = len(profiles)
    distinct = list(dict.fromkeys(profile.id for profile in profiles))

    if "sxm" in classes or "superchip" in classes:
        station = [profile for profile in profiles if "dgx-station-gb300" in _accepting(profile.id)]
        if station and len(station) == 1 and classes <= {"superchip", "pcie"}:
            return platforms["dgx-station-gb300"]
        if len(classes) > 1:
            kinds = " and ".join(sorted(_CLASS_NAMES[name] for name in classes))
            raise PlatformError(
                f"{kinds} need different slots and no platform template holds both; "
                "run them as separate machines or pass --platform generic"
            )
        if len(distinct) > 1:
            raise PlatformError(
                f"{_CLASS_NAMES[classes.pop()]} share a board that holds one GPU model, "
                f"but the layout mixes {', '.join(distinct)}; run them as separate machines "
                "or pass --platform generic"
            )
        candidates = [platforms[pid] for pid in _accepting(distinct[0]) if platforms[pid].homogeneous]
        if not candidates:
            raise PlatformError(
                f"no platform template holds {distinct[0]}; pass --platform generic"
            )
        fitting = [platform for platform in candidates if platform.capacity >= count]
        if not fitting:
            largest = max(candidates, key=lambda platform: platform.capacity)
            raise PlatformError(
                f"{count}x {distinct[0]} exceed the {largest.name} ({largest.capacity} GPUs); "
                "split them across distributed nodes or pass --platform generic"
            )
        return min(fitting, key=lambda platform: platform.capacity)

    if classes == {"embedded"}:
        return _single(platforms["jetson"], count, distinct)
    if classes == {"laptop"}:
        return _single(platforms["laptop"], count, distinct)
    if classes != {"pcie"}:
        kinds = " and ".join(sorted(_CLASS_NAMES[name] for name in classes))
        raise PlatformError(f"{kinds} need different hosts; pass --platform generic")
    segments = {profile.segment for profile in profiles}
    if "datacenter" in segments or count > 4:
        choice = platforms["pcie-server"]
    elif count > 2 or "workstation" in segments:
        choice = platforms["workstation"]
    else:
        choice = platforms["desktop"]
    if count > choice.capacity:
        raise PlatformError(
            f"{count} PCIe GPUs exceed the {choice.capacity} slots of the {choice.name}; "
            "split them across distributed nodes or pass --platform generic"
        )
    return choice


def resolve_platform(
    profile_ids: Sequence[str],
    platform: str = AUTO,
    *,
    pcie_generation: int = 0,
    nvlink_bridges: str = "",
) -> ResolvedPlatform:
    """Place the devices into a platform's slots and model their links."""

    profiles = [get_profile(profile_id) for profile_id in profile_ids]
    chosen = select_platform(profile_ids) if platform.strip().lower() == AUTO else get_platform(platform)
    if pcie_generation and pcie_generation not in _pcie_generations():
        raise PlatformError(f"pcie_generation must be 0 (template default) or one of {_pcie_generations()}")
    if len(profiles) > chosen.capacity:
        raise PlatformError(
            f"{len(profiles)} GPUs exceed the {chosen.capacity} slots of the {chosen.name} "
            f"({chosen.describe_slots()})"
        )
    if chosen.homogeneous and len({profile.id for profile in profiles}) > 1:
        raise PlatformError(
            f"the {chosen.name} holds one GPU model, but the layout mixes "
            f"{', '.join(dict.fromkeys(profile.id for profile in profiles))}"
        )

    slot_indices: list[int] = []
    taken: set[int] = set()
    for index, profile in enumerate(profiles):
        slot_index = next(
            (
                position
                for position, slot in enumerate(chosen.slots)
                if position not in taken and _fits(chosen, slot, profile)
            ),
            None,
        )
        if slot_index is None:
            raise PlatformError(_misfit(chosen, profile, index))
        taken.add(slot_index)
        slot_indices.append(slot_index)

    notes: list[str] = []
    unsupported = {item["profile"] for item in platform_data()["pcie_p2p_unsupported"]}
    slots = []
    for index, profile in enumerate(profiles):
        template = chosen.slots[slot_indices[index]]
        slots.append(_slot(template, profile, chosen, taken, pcie_generation, unsupported))
    groups = list(_baseboard_nvlink(chosen, profiles, slot_indices, notes))
    if nvlink_bridges.strip():
        if chosen.kind not in ("pcie", "generic"):
            raise PlatformError(
                f"NVLink bridges connect PCIe cards; the {chosen.name} wires its GPUs itself"
            )
        groups.extend(_bridges(nvlink_bridges, profiles))
    if chosen.notes:
        notes.append(chosen.notes)
    topology = LinkTopology(tuple(slots), tuple(groups), platform=chosen.id, source="platform")
    for index in range(len(profiles)):
        if len(topology.ports(index)) > 18:
            raise PlatformError(f"GPU {index} would have more than 18 NVLinks")
    return ResolvedPlatform(chosen, tuple(profile.id for profile in profiles), tuple(slot_indices), topology, tuple(notes))


def apply_platform_environment(
    environ: MutableMapping[str, str],
    profile_ids: Sequence[str],
    platform: str,
    *,
    pcie_generation: int = 0,
    nvlink_bridges: str = "",
) -> ResolvedPlatform | None:
    """Export a platform's topology unless the environment already sets one.

    Explicit ``FAKEGPU_SLOTS``/``FAKEGPU_NVLINK_FABRIC``/``FAKEGPU_NVLINK_GROUPS``
    win over the platform.
    """

    if any(environ.get(name) for name in (*ENVIRONMENT_VARIABLES[1:], LEGACY_NVLINK_VARIABLES[0])):
        return None
    resolved = resolve_platform(
        profile_ids, platform, pcie_generation=pcie_generation, nvlink_bridges=nvlink_bridges
    )
    environ.update(resolved.environment())
    return resolved


def platform_options(environ: Mapping[str, str]) -> dict[str, Any]:
    """``resolve_platform`` keyword arguments from ``FAKEGPU_PCIE_GEN``/``FAKEGPU_NVLINK_BRIDGES``."""

    generation = (environ.get("FAKEGPU_PCIE_GEN") or "0").strip()
    if not generation.isdigit():
        raise PlatformError(f"FAKEGPU_PCIE_GEN {generation!r} is not a PCIe generation")
    return {
        "pcie_generation": int(generation),
        "nvlink_bridges": environ.get("FAKEGPU_NVLINK_BRIDGES") or "",
    }


def device_profiles(environ: Mapping[str, str]) -> list[str]:
    """Profile IDs of the devices a FakeCUDA process exposes, as the native runtime reads them."""

    from .machine.config import parse_devices

    spec = (environ.get("FAKEGPU_PROFILES") or "").strip()
    if spec:
        return parse_devices(spec)
    profile = (environ.get("FAKEGPU_PROFILE") or "a100").strip()
    count = (environ.get("FAKEGPU_DEVICE_COUNT") or "8").strip()
    return [get_profile(profile).id] * (int(count) if count.isdigit() and int(count) > 0 else 8)


def export_platform(environ: MutableMapping[str, str], platform: str | None = None) -> ResolvedPlatform | None:
    """Resolve ``platform`` (or ``FAKEGPU_PLATFORM``) for the process's devices into ``environ``.

    Raises :class:`PlatformError` when the devices do not fit. Explicit
    ``FAKEGPU_SLOTS``/``FAKEGPU_NVLINK_FABRIC``/``FAKEGPU_NVLINK_GROUPS``
    are left alone.
    """

    platform = (platform or environ.get("FAKEGPU_PLATFORM") or "").strip()
    if not platform:
        return None
    environ["FAKEGPU_PLATFORM"] = platform
    return apply_platform_environment(environ, device_profiles(environ), platform, **platform_options(environ))


def topology_from_environment(profile_ids: Sequence[str], environ: Mapping[str, str]) -> LinkTopology:
    """The topology a FakeCUDA process models for its devices.

    ``FAKEGPU_PLATFORM`` alone (without ``FAKEGPU_SLOTS``) is resolved here,
    so a script started with only the platform name still gets its slots.
    """

    profiles = [_profile_or_none(profile_id) for profile_id in profile_ids]
    platform = (environ.get("FAKEGPU_PLATFORM") or "").strip()
    explicit = any(environ.get(name) for name in (*ENVIRONMENT_VARIABLES[1:], LEGACY_NVLINK_VARIABLES[0]))
    if platform and not explicit:
        try:
            return resolve_platform(profile_ids, platform, **platform_options(environ)).topology
        except (PlatformError, KeyError, ValueError) as exc:
            return LinkTopology(
                LinkTopology.default(profiles).slots,
                platform=platform,
                source="modeled_environment_invalid",
                error=f"FAKEGPU_PLATFORM: {exc}",
                nvlink_source="modeled_environment_invalid",
            )
    return LinkTopology.from_environment(profiles, environ)


# -- helpers ---------------------------------------------------------------

_CLASS_NAMES = {
    "sxm": "SXM modules",
    "superchip": "superchips",
    "pcie": "PCIe cards",
    "laptop": "laptop GPUs",
    "embedded": "embedded GPUs",
}


def _profile_or_none(profile_id: str) -> GpuProfile | None:
    try:
        return get_profile(profile_id)
    except (KeyError, ValueError):
        return None


def _pcie_generations() -> list[int]:
    return [int(row["generation"]) for row in platform_data()["pcie_generations"]]


def _accepting(profile_id: str) -> list[str]:
    return [platform.id for platform in load_platforms().values() if profile_id in platform.accepts]


def _single(platform: Platform, count: int, distinct: list[str]) -> Platform:
    if count > 1:
        raise PlatformError(
            f"a {platform.name.lower()} has one GPU, but the layout has {count} ({', '.join(distinct)}); "
            "pass --platform generic"
        )
    return platform


def _fits(platform: Platform, slot: SlotTemplate, profile: GpuProfile) -> bool:
    if slot.type == "any":
        return True
    if slot.type in ("sxm", "superchip"):
        return form_class(profile) == slot.type and profile.id in platform.accepts
    return form_class(profile) == slot.type


def _misfit(platform: Platform, profile: GpuProfile, index: int) -> str:
    needed = _CLASS_NAMES[form_class(profile)]
    if form_class(profile) in platform.slot_types and platform.accepts and profile.id not in platform.accepts:
        others = _accepting(profile.id)
        hint = f"; it fits {', '.join(others)}" if others else ""
        return (
            f"GPU {index} ({profile.id}) does not fit the {platform.name}, which takes "
            f"{', '.join(platform.accepts)}{hint}"
        )
    if form_class(profile) in platform.slot_types:
        return f"GPU {index} ({profile.id}): all {platform.capacity} slots of the {platform.name} are taken"
    return (
        f"GPU {index} ({profile.id}) is one of the {needed}, but the {platform.name} has "
        f"{platform.describe_slots()}; use auto or a platform with matching slots"
    )


def _slot(
    template: SlotTemplate,
    profile: GpuProfile,
    platform: Platform,
    taken: set[int],
    pcie_generation: int,
    unsupported_p2p: set[str],
) -> GpuSlot:
    card = None
    if profile.pcie_generation or profile.pcie_lanes:
        card = PcieLink(profile.pcie_generation or 4, profile.pcie_lanes or 16)
    pcie = None
    if template.pcie == "card":
        pcie = card or PcieLink(4, 16)
    elif template.pcie:
        offered = PcieLink.parse(template.pcie)
        if pcie_generation and template.type in ("pcie", "any"):
            offered = PcieLink(pcie_generation, offered.width)
        if template.lanes_when_shared and taken & set(template.shares_lanes_with):
            offered = PcieLink(offered.generation, min(offered.width, template.lanes_when_shared))
        pcie = card.limited_by(offered) if card else offered
    host_pcie = PcieLink.parse(template.host_pcie) if template.host_pcie else None
    host_gbps = template.host_gbps
    if template.host_link == "unified" and host_gbps is None and profile.memory_bandwidth_gbps:
        # A copy within shared memory reads and writes every byte.
        host_gbps = round(profile.memory_bandwidth_gbps / 2, 3)
    return GpuSlot(
        type=template.type,
        pcie=pcie,
        gpu_pcie=card or pcie,
        host_pcie=host_pcie,
        host_link=template.host_link,
        host_gbps=host_gbps,
        numa=template.numa,
        bridge=template.bridge,
        switch=template.switch,
        p2p=template.p2p and profile.id not in unsupported_p2p,
    )


def _profile_link_rate(profile: GpuProfile) -> tuple[int | None, float | None]:
    """(links, GB/s per link and direction) of a GPU's NVLink ports, where known."""

    generation_rate = nvlink_link_gbps().get(profile.nvlink_generation or 0)
    total = profile.nvlink_bandwidth_gbps
    if total and profile.nvlink_links:
        return profile.nvlink_links, total / 2 / profile.nvlink_links
    if total and generation_rate:
        return round(total / 2 / generation_rate), generation_rate
    return profile.nvlink_links, generation_rate


def _baseboard_nvlink(
    platform: Platform,
    profiles: Sequence[GpuProfile],
    slot_indices: Sequence[int],
    notes: list[str],
):
    spec = platform.nvlink
    if not spec or len(profiles) < 2:
        return
    board_rate = nvlink_link_gbps()[int(spec["generation"])]
    links, rate = _profile_link_rate(profiles[0])
    if spec["fabric"] == "nvswitch":
        board_links = int(spec["links_per_gpu"])
        if links is None or rate is None:
            notes.append(
                f"{profiles[0].id} does not record its NVLink ports; the board's {board_links} links "
                f"at {board_rate:g} GB/s are assumed"
            )
        yield NvlinkGroup(
            "switch",
            tuple(range(len(profiles))),
            links or board_links,
            rate or board_rate,
        )
        return
    device_by_slot = {slot: index for index, slot in enumerate(slot_indices)}
    for left, right, count in spec["pairs"]:
        if left in device_by_slot and right in device_by_slot:
            members = tuple(sorted((device_by_slot[left], device_by_slot[right])))
            yield NvlinkGroup("mesh", members, int(count), rate or board_rate)


def _bridges(text: str, profiles: Sequence[GpuProfile]):
    used: set[int] = set()
    ways_table = platform_data()["nvlink_bridge_ways"]
    for item in text.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            members = tuple(int(part) for part in item.split("-"))
        except ValueError:
            raise PlatformError(f"NVLink bridge {item!r} is not GPU indices joined by '-', e.g. 0-1") from None
        if len(set(members)) != len(members) or any(not 0 <= member < len(profiles) for member in members):
            raise PlatformError(f"NVLink bridge {item!r} names a GPU outside 0-{len(profiles) - 1} or repeats one")
        if used & set(members):
            raise PlatformError(f"NVLink bridge {item!r} reuses a GPU that already has a bridge")
        used.update(members)
        models = {profiles[member].id for member in members}
        if len(models) > 1:
            raise PlatformError(f"NVLink bridge {item!r} joins different models ({', '.join(sorted(models))})")
        profile = profiles[members[0]]
        links, rate = _profile_link_rate(profile)
        if form_class(profile) != "pcie" or not profile.nvlink_generation or not links or not rate:
            raise PlatformError(f"{profile.id} has no NVLink bridge connector")
        ways = ways_table.get(profile.id, ways_table["default"])
        if len(members) not in ways:
            raise PlatformError(
                f"{profile.id} bridges join {' or '.join(str(way) for way in ways)} GPUs, not {len(members)}"
            )
        per_pair = links // (len(members) - 1)
        yield NvlinkGroup("mesh", tuple(sorted(members)), per_pair, rate)
