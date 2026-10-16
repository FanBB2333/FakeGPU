"""PCIe placement, CPU links and NVLink fabric of the modeled GPUs.

A platform (:mod:`fakegpu.platforms`) resolves into a :class:`LinkTopology`,
which reaches FakeCUDA processes through environment variables. The native
runtime (``src/core/device.cpp``) parses the same variables.

``FAKEGPU_PLATFORM``
    Platform ID, shown by ``fakegpu-smi``.
``FAKEGPU_SLOTS``
    One ``;``-separated entry per GPU, each a ``,``-separated list of
    ``key=value`` pairs:

    ``type``     slot type: ``sxm``, ``pcie``, ``superchip``, ``laptop``,
                 ``embedded`` or ``any``
    ``pcie``     link the GPU trained at, ``GENxWIDTH``; ``none`` for no link
    ``gpu``      the GPU's own maximum link (default: ``pcie``)
    ``host``     narrower upstream link between the slot and the CPU
    ``link``     CPU-GPU path: ``pcie`` (default), ``c2c`` or ``unified``
    ``hostbw``   per-direction GB/s of the CPU-GPU path (default: from the
                 PCIe links)
    ``numa``, ``bridge``, ``switch``
                 NUMA node, PCIe host bridge and PCIe switch path (``a`` or
                 ``a.b`` for two switch levels; empty for a root port)
    ``p2p``      ``0`` when PCIe peer access is not allowed (default ``1``)
``FAKEGPU_NVLINK_FABRIC``
    ``;``-separated groups ``KIND:MEMBERS:LINKS:GBPS``. ``mesh`` gives every
    pair of members LINKS direct links; ``switch`` gives each member LINKS
    links into an NVSwitch fabric. MEMBERS is a comma list that may contain
    ranges (``0-7``); GBPS is per link and direction.

Without ``FAKEGPU_NVLINK_FABRIC``, the older ``FAKEGPU_NVLINK_GROUPS`` and
``FAKEGPU_NVLINK_BANDWIDTH_GBPS`` variables describe one link per peer.
Without ``FAKEGPU_SLOTS``, every GPU sits behind one PCIe switch at its own
link generation and width.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).resolve().parent / "data" / "platforms.json"

ENVIRONMENT_VARIABLES = ("FAKEGPU_PLATFORM", "FAKEGPU_SLOTS", "FAKEGPU_NVLINK_FABRIC")
LEGACY_NVLINK_VARIABLES = ("FAKEGPU_NVLINK_GROUPS", "FAKEGPU_NVLINK_BANDWIDTH_GBPS")

DEFAULT_NVLINK_BANDWIDTH_GBPS = 900.0
MAXIMUM_NVLINK_BANDWIDTH_GBPS = 1_000_000.0
MAXIMUM_NVLINKS_PER_DEVICE = 18
DEFAULT_PCIE = (4, 16)

SLOT_TYPES = ("sxm", "pcie", "superchip", "laptop", "embedded", "any")
HOST_LINKS = ("pcie", "c2c", "unified")
PCIE_PEER_RELATIONS = ("PIX", "PXB", "PHB", "NODE")


class TopologyError(ValueError):
    pass


@lru_cache(maxsize=1)
def platform_data() -> dict[str, Any]:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    if payload.get("schema") != "fakegpu.platforms.v1":
        raise ValueError(f"{DATA_PATH}: expected schema fakegpu.platforms.v1")
    return payload


@lru_cache(maxsize=1)
def pcie_lane_gbps() -> dict[int, float]:
    """Usable GB/s per lane and direction for each PCIe generation."""

    return {int(row["generation"]): float(row["lane_gbps"]) for row in platform_data()["pcie_generations"]}


@lru_cache(maxsize=1)
def nvlink_link_gbps() -> dict[int, float]:
    """GB/s per link and direction for each NVLink generation."""

    return {int(row["generation"]): float(row["link_gbps"]) for row in platform_data()["nvlink_generations"]}


@dataclass(frozen=True)
class PcieLink:
    generation: int
    width: int

    @classmethod
    def parse(cls, text: str) -> PcieLink:
        generation, separator, width = text.strip().lower().partition("x")
        if not separator or not generation.isdigit() or not width.isdigit():
            raise TopologyError(f"PCIe link {text!r} is not GENxWIDTH, e.g. 4x16")
        link = cls(int(generation), int(width))
        if link.generation not in pcie_lane_gbps():
            raise TopologyError(f"PCIe generation {link.generation} is not one of {sorted(pcie_lane_gbps())}")
        if link.width not in (1, 2, 4, 8, 16, 32):
            raise TopologyError(f"PCIe width x{link.width} is not one of x1, x2, x4, x8, x16, x32")
        return link

    def __str__(self) -> str:
        return f"{self.generation}x{self.width}"

    @property
    def label(self) -> str:
        return f"Gen{self.generation} x{self.width}"

    @property
    def bandwidth_gbps(self) -> float:
        return round(pcie_lane_gbps()[self.generation] * self.width, 3)

    def limited_by(self, other: PcieLink | None) -> PcieLink:
        if other is None:
            return self
        return PcieLink(min(self.generation, other.generation), min(self.width, other.width))


@dataclass(frozen=True)
class GpuSlot:
    """Where one GPU sits and how it reaches the CPU."""

    type: str = "any"
    pcie: PcieLink | None = PcieLink(*DEFAULT_PCIE)
    gpu_pcie: PcieLink | None = None
    host_pcie: PcieLink | None = None
    host_link: str = "pcie"
    host_gbps: float | None = None
    numa: int = 0
    bridge: int = 0
    switch: str = "0"
    p2p: bool = True

    @property
    def gpu_max_pcie(self) -> PcieLink | None:
        return self.gpu_pcie or self.pcie

    @property
    def host_bandwidth_gbps(self) -> float | None:
        """Per-direction GB/s between this GPU and host memory."""

        if self.host_gbps is not None:
            return self.host_gbps
        if self.host_link != "pcie" or self.pcie is None:
            return None
        links = [link for link in (self.pcie, self.host_pcie) if link is not None]
        return min(link.bandwidth_gbps for link in links)

    def encode(self) -> str:
        parts = [f"type={self.type}", f"pcie={self.pcie or 'none'}"]
        if self.gpu_pcie and self.gpu_pcie != self.pcie:
            parts.append(f"gpu={self.gpu_pcie}")
        if self.host_pcie:
            parts.append(f"host={self.host_pcie}")
        if self.host_link != "pcie":
            parts.append(f"link={self.host_link}")
        bandwidth = self.host_bandwidth_gbps
        if bandwidth is not None:
            parts.append(f"hostbw={bandwidth:g}")
        parts.extend([f"numa={self.numa}", f"bridge={self.bridge}", f"switch={self.switch}"])
        if not self.p2p:
            parts.append("p2p=0")
        return ",".join(parts)

    @classmethod
    def decode(cls, text: str) -> GpuSlot:
        values: dict[str, Any] = {}
        for item in text.split(","):
            key, separator, value = item.strip().partition("=")
            if not separator:
                raise TopologyError(f"slot field {item.strip()!r} is not key=value")
            value = value.strip()
            if key == "type":
                if value not in SLOT_TYPES:
                    raise TopologyError(f"slot type {value!r} is not one of {', '.join(SLOT_TYPES)}")
                values["type"] = value
            elif key in ("pcie", "gpu", "host"):
                name = {"pcie": "pcie", "gpu": "gpu_pcie", "host": "host_pcie"}[key]
                values[name] = None if value == "none" else PcieLink.parse(value)
            elif key == "link":
                if value not in HOST_LINKS:
                    raise TopologyError(f"slot link {value!r} is not one of {', '.join(HOST_LINKS)}")
                values["host_link"] = value
            elif key == "hostbw":
                values["host_gbps"] = _positive_float(value, "slot hostbw")
            elif key in ("numa", "bridge"):
                if not value.isdigit():
                    raise TopologyError(f"slot {key} {value!r} is not a non-negative integer")
                values[key] = int(value)
            elif key == "switch":
                if value and not all(part.isdigit() for part in value.split(".")):
                    raise TopologyError(f"slot switch {value!r} is not a dotted switch path such as 0 or 1.2")
                values["switch"] = value
            elif key == "p2p":
                if value not in ("0", "1"):
                    raise TopologyError(f"slot p2p {value!r} is not 0 or 1")
                values["p2p"] = value == "1"
            else:
                raise TopologyError(f"unknown slot field {key!r}")
        return cls(**values)


@dataclass(frozen=True)
class NvlinkGroup:
    kind: str
    members: tuple[int, ...]
    links: int
    link_gbps: float

    def encode(self) -> str:
        return f"{self.kind}:{','.join(str(member) for member in self.members)}:{self.links}:{self.link_gbps:g}"

    @classmethod
    def decode(cls, text: str) -> NvlinkGroup:
        fields = [part.strip() for part in text.split(":")]
        if len(fields) != 4:
            raise TopologyError(f"NVLink group {text.strip()!r} is not KIND:MEMBERS:LINKS:GBPS")
        kind, members_text, links_text, gbps_text = fields
        if kind not in ("mesh", "switch"):
            raise TopologyError(f"NVLink group kind {kind!r} is not mesh or switch")
        members = _parse_members(members_text)
        if len(members) < 2:
            raise TopologyError(f"NVLink group {text.strip()!r} needs at least two GPUs")
        if not links_text.isdigit() or not 1 <= int(links_text) <= MAXIMUM_NVLINKS_PER_DEVICE:
            raise TopologyError(f"NVLink group {text.strip()!r} needs 1 to {MAXIMUM_NVLINKS_PER_DEVICE} links")
        return cls(kind, members, int(links_text), _positive_float(gbps_text, "NVLink GBPS"))


@dataclass(frozen=True)
class NvlinkPort:
    """One NVLink of a GPU, in NVML link-index order."""

    link: int
    remote: str  # "gpu" or "nvswitch"
    peer: int | None
    bandwidth_gbps: float


@dataclass(frozen=True)
class NvlinkPeer:
    index: int
    links: int
    via: str  # "direct", "nvswitch" or "direct+nvswitch"
    bandwidth_gbps: float
    first_link: int


@dataclass(frozen=True)
class LinkTopology:
    slots: tuple[GpuSlot, ...]
    nvlink: tuple[NvlinkGroup, ...] = ()
    platform: str = ""
    source: str = "modeled_none"
    error: str = ""
    nvlink_source: str = "modeled_none"
    legacy_bandwidth_gbps: float = DEFAULT_NVLINK_BANDWIDTH_GBPS
    _ports: tuple[tuple[NvlinkPort, ...], ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        ports = tuple(self._build_ports(index) for index in range(len(self.slots)))
        object.__setattr__(self, "_ports", ports)

    # -- construction -----------------------------------------------------

    @classmethod
    def default(cls, profiles: Sequence[Any]) -> LinkTopology:
        """Each GPU at its own PCIe link behind one switch, without NVLink."""

        return cls(tuple(default_slot(profile) for profile in profiles))

    @classmethod
    def from_environment(
        cls,
        profiles: Sequence[Any],
        environ: Mapping[str, str],
    ) -> LinkTopology:
        """Parse the topology variables for GPUs with these catalog profiles.

        An invalid configuration keeps the default placement, drops every
        NVLink, and records the problem in ``error``.
        """

        base = cls.default(profiles)
        count = len(profiles)
        slots_text = (environ.get("FAKEGPU_SLOTS") or "").strip()
        fabric_text = (environ.get("FAKEGPU_NVLINK_FABRIC") or "").strip()
        groups_text = environ.get("FAKEGPU_NVLINK_GROUPS")
        platform = (environ.get("FAKEGPU_PLATFORM") or "").strip()
        if not (slots_text or fabric_text or groups_text):
            return base
        source = "platform" if platform and (slots_text or fabric_text) else "modeled_environment"
        try:
            slots = base.slots
            if slots_text:
                entries = [entry for entry in slots_text.split(";")]
                if len(entries) < count or any(not entry.strip() for entry in entries):
                    raise TopologyError(
                        f"FAKEGPU_SLOTS describes {len([e for e in entries if e.strip()])} GPUs "
                        f"but the process has {count}"
                    )
                slots = tuple(GpuSlot.decode(entry) for entry in entries[:count])
            nvlink_source = "modeled_none"
            legacy_bandwidth = DEFAULT_NVLINK_BANDWIDTH_GBPS
            if fabric_text:
                groups = tuple(NvlinkGroup.decode(entry) for entry in fabric_text.split(";"))
                nvlink_source = source
            elif groups_text:
                groups, legacy_bandwidth = _legacy_groups(groups_text, environ, count)
                nvlink_source = "modeled_environment"
            else:
                groups = ()
            for group in groups:
                outside = [member for member in group.members if member >= count]
                if outside:
                    raise TopologyError(f"NVLink group {group.encode()} names GPU {outside[0]}, beyond the {count} GPUs")
            topology = cls(
                slots,
                groups,
                platform=platform,
                source=source,
                nvlink_source=nvlink_source,
                legacy_bandwidth_gbps=legacy_bandwidth,
            )
            for index in range(count):
                if len(topology.ports(index)) > MAXIMUM_NVLINKS_PER_DEVICE:
                    raise TopologyError(
                        "a device belongs to more than 18 modeled NVLink peer relationships"
                        if not fabric_text
                        else f"GPU {index} has more than {MAXIMUM_NVLINKS_PER_DEVICE} NVLinks"
                    )
            return topology
        except TopologyError as exc:
            return replace(
                base,
                platform=platform,
                source="modeled_environment_invalid",
                error=str(exc),
                nvlink_source="modeled_environment_invalid",
            )

    def environment(self) -> dict[str, str]:
        variables = {"FAKEGPU_SLOTS": ";".join(slot.encode() for slot in self.slots)}
        if self.platform:
            variables["FAKEGPU_PLATFORM"] = self.platform
        if self.nvlink:
            variables["FAKEGPU_NVLINK_FABRIC"] = ";".join(group.encode() for group in self.nvlink)
        return variables

    # -- queries ----------------------------------------------------------

    @property
    def configured(self) -> bool:
        return self.source != "modeled_none"

    @property
    def valid(self) -> bool:
        return not self.error

    def ports(self, index: int) -> tuple[NvlinkPort, ...]:
        return self._ports[index] if 0 <= index < len(self._ports) else ()

    def peers(self, index: int) -> list[NvlinkPeer]:
        found: dict[int, NvlinkPeer] = {}
        for port in self.ports(index):
            if port.peer is not None:
                self._add_peer(found, port.peer, 1, "direct", port.bandwidth_gbps, port.link)
        switch_ports = [port for port in self.ports(index) if port.remote == "nvswitch"]
        if switch_ports:
            for group in self.nvlink:
                if group.kind != "switch" or index not in group.members:
                    continue
                first = next(port.link for port in switch_ports)
                for member in group.members:
                    if member == index:
                        continue
                    links = min(group.links, self._switch_links(member))
                    self._add_peer(found, member, links, "nvswitch", links * group.link_gbps, first)
        return [found[key] for key in sorted(found)]

    def nvlink_links(self, source: int, target: int) -> int:
        return next((peer.links for peer in self.peers(source) if peer.index == target), 0)

    def pcie_relation(self, source: int, target: int) -> str:
        if source == target:
            return "X"
        left, right = self.slots[source], self.slots[target]
        return placement_relation((left.numa, left.bridge, left.switch), (right.numa, right.bridge, right.switch))

    def relation(self, source: int, target: int) -> str:
        """``nvidia-smi topo -m`` cell for this pair of GPUs."""

        if source == target:
            return "X"
        links = self.nvlink_links(source, target)
        return f"NV{links}" if links else self.pcie_relation(source, target)

    def can_access_peer(self, source: int, target: int) -> bool:
        if source == target:
            return False
        if self.nvlink_links(source, target):
            return True
        left, right = self.slots[source], self.slots[target]
        if not (left.p2p and right.p2p and left.pcie and right.pcie):
            return False
        return self.pcie_relation(source, target) in PCIE_PEER_RELATIONS

    def peer_bandwidth_gbps(self, source: int, target: int) -> float | None:
        """Per-direction GB/s of a copy between two GPUs."""

        peer = next((peer for peer in self.peers(source) if peer.index == target), None)
        if peer is not None:
            return peer.bandwidth_gbps
        left, right = self.slots[source], self.slots[target]
        if left.pcie is None or right.pcie is None:
            return None
        if self.pcie_relation(source, target) == "PIX":
            return min(left.pcie.bandwidth_gbps, right.pcie.bandwidth_gbps)
        values = [value for value in (left.host_bandwidth_gbps, right.host_bandwidth_gbps) if value]
        return min(values) if values else None

    def host_bandwidth_gbps(self, index: int) -> float | None:
        return self.slots[index].host_bandwidth_gbps

    # -- internals --------------------------------------------------------

    def _switch_links(self, index: int) -> int:
        return sum(group.links for group in self.nvlink if group.kind == "switch" and index in group.members)

    @staticmethod
    def _add_peer(
        found: dict[int, NvlinkPeer], index: int, links: int, via: str, bandwidth: float, first_link: int
    ) -> None:
        current = found.get(index)
        if current is None:
            found[index] = NvlinkPeer(index, links, via, bandwidth, first_link)
            return
        combined = current.via if current.via == via else "direct+nvswitch"
        found[index] = NvlinkPeer(
            index,
            current.links + links,
            combined,
            current.bandwidth_gbps + bandwidth,
            min(current.first_link, first_link),
        )

    def _build_ports(self, index: int) -> tuple[NvlinkPort, ...]:
        ports: list[NvlinkPort] = []
        for group in self.nvlink:
            if index not in group.members:
                continue
            if group.kind == "switch":
                remotes: list[int | None] = [None] * group.links
            else:
                remotes = [member for member in group.members if member != index for _ in range(group.links)]
            for peer in remotes:
                ports.append(
                    NvlinkPort(len(ports), "nvswitch" if peer is None else "gpu", peer, group.link_gbps)
                )
        return tuple(ports)


def placement_relation(left: Sequence[Any], right: Sequence[Any]) -> str:
    """PCIe path class between two GPUs placed at (NUMA node, host bridge, switch path)."""

    left_numa, left_bridge, left_switch = left
    right_numa, right_bridge, right_switch = right
    left_switch, right_switch = str(left_switch or ""), str(right_switch or "")
    if left_switch and right_switch:
        if left_switch == right_switch:
            return "PIX"
        if "." in left_switch and left_switch.split(".")[0] == right_switch.split(".")[0]:
            return "PXB"
    if left_numa != right_numa:
        return "SYS"
    return "PHB" if left_bridge == right_bridge else "NODE"


def default_slot(profile: Any) -> GpuSlot:
    """Slot for a GPU when no platform is configured: its own link, one PCIe switch."""

    generation = getattr(profile, "pcie_generation", None) or DEFAULT_PCIE[0]
    width = getattr(profile, "pcie_lanes", None) or DEFAULT_PCIE[1]
    return GpuSlot(pcie=PcieLink(generation, width))


def _positive_float(text: str, name: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise TopologyError(f"{name} {text!r} is not a number") from None
    if not math.isfinite(value) or value <= 0:
        raise TopologyError(f"{name} {text!r} must be a positive number")
    return value


def _parse_members(text: str) -> tuple[int, ...]:
    members: list[int] = []
    for item in text.split(","):
        item = item.strip()
        first, separator, last = item.partition("-")
        if not first.isdigit() or (separator and not last.isdigit()):
            raise TopologyError(f"NVLink member {item!r} is not a GPU index or range")
        start, stop = int(first), int(last) if separator else int(first)
        if stop < start:
            raise TopologyError(f"NVLink member range {item!r} is reversed")
        members.extend(range(start, stop + 1))
    if len(set(members)) != len(members):
        raise TopologyError(f"NVLink members {text!r} repeat a GPU")
    return tuple(members)


def _legacy_groups(
    groups_text: str,
    environ: Mapping[str, str],
    count: int,
) -> tuple[tuple[NvlinkGroup, ...], float]:
    """Read ``FAKEGPU_NVLINK_GROUPS`` as one-link pairs, as before platforms existed."""

    bandwidth = DEFAULT_NVLINK_BANDWIDTH_GBPS
    bandwidth_text = environ.get("FAKEGPU_NVLINK_BANDWIDTH_GBPS")
    if bandwidth_text:
        try:
            bandwidth = float(bandwidth_text)
        except ValueError:
            bandwidth = math.nan
        if not 0 < bandwidth <= MAXIMUM_NVLINK_BANDWIDTH_GBPS:
            raise TopologyError(
                "FAKEGPU_NVLINK_BANDWIDTH_GBPS must be a finite number greater than 0 and no greater than 1000000"
            )
    pairs: set[tuple[int, int]] = set()
    for group_index, group_text in enumerate(str(groups_text).split(";"), start=1):
        group_text = group_text.strip()
        if not group_text:
            raise TopologyError("FAKEGPU_NVLINK_GROUPS contains an empty group")
        members: list[int] = []
        for member_text in group_text.split(","):
            member_text = member_text.strip()
            unsigned = member_text[1:] if member_text.startswith("+") else member_text
            if not unsigned.isascii() or not unsigned.isdigit() or int(unsigned) >= count:
                raise TopologyError(f"FAKEGPU_NVLINK_GROUPS group {group_index} contains an invalid device index")
            members.append(int(unsigned))
        members = sorted(set(members))
        if len(members) < 2:
            raise TopologyError("each FAKEGPU_NVLINK_GROUPS group must contain at least two distinct device indices")
        if len(members) - 1 > MAXIMUM_NVLINKS_PER_DEVICE:
            raise TopologyError("an FAKEGPU_NVLINK_GROUPS group exceeds the 18-link per-device limit")
        for offset, left in enumerate(members):
            for right in members[offset + 1 :]:
                pairs.add((left, right))
    if not pairs:
        raise TopologyError("FAKEGPU_NVLINK_GROUPS did not define any links")
    return tuple(NvlinkGroup("mesh", pair, 1, bandwidth) for pair in sorted(pairs)), bandwidth
