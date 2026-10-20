"""Modeled time and throughput of copies over the GPU links.

FakeGPU copies finish at host-memory speed. This module records how long
each copy would occupy its link on the modeled platform (PCIe, NVLink-C2C
or NVLink, at the bandwidth :mod:`fakegpu.link_topology` gives) and turns
that schedule into the throughput NVML and ``fakegpu-smi`` report. Nothing
waits: a program runs as fast as before.

Copies on one link and direction run one after another, each at the
bandwidth of its route, so a burst of copies keeps the link busy for the
sum of their modeled times. Throughput is the data the schedule delivers
in the last :data:`WINDOW_SECONDS`. The native runtime keeps the same
schedule in ``src/core/link_traffic.cpp``.

Routes:

* host to GPU and back: the GPU's CPU link (``pcie`` or ``c2c``); copies
  within unified memory (DGX Spark, Jetson) take time but use no link.
* GPU to GPU: NVLink when the pair has NVLinks, PCIe peer-to-peer when the
  pair has PCIe peer access, otherwise staged through host memory over
  both GPUs' CPU links at the slower of the two.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .link_topology import LinkTopology

SCHEMA_VERSION = "fakegpu.transfer.v1"
WINDOW_SECONDS = 1.0
COPY_KINDS = ("host_to_device", "device_to_host", "peer_send", "peer_receive")
LINK_KINDS = ("pcie", "nvlink", "c2c")
MAXIMUM_SEGMENTS = 4096


@dataclass(frozen=True)
class Route:
    """Links a copy uses and the bandwidth it runs at.

    ``source_link`` is the link the sending GPU transmits on and
    ``target_link`` the link the receiving GPU receives on; ``None`` for the
    host side or a copy that uses no link.
    """

    source_link: str | None
    target_link: str | None
    bandwidth_gbps: float | None


def _host_link(topology: LinkTopology, index: int) -> str | None:
    link = topology.slots[index].host_link
    return link if link in LINK_KINDS else None


def route(topology: LinkTopology, source: int | None, target: int | None) -> Route:
    """The route of a copy; ``None`` stands for host memory."""

    if source is None and target is None:
        return Route(None, None, None)
    if source is None or target is None:
        index = target if source is None else source
        link = _host_link(topology, index)
        bandwidth = topology.host_bandwidth_gbps(index)
        return Route(None if source is None else link, None if target is None else link, bandwidth)
    if source == target:
        return Route(None, None, None)
    if topology.nvlink_links(source, target):
        return Route("nvlink", "nvlink", topology.peer_bandwidth_gbps(source, target))
    if topology.can_access_peer(source, target):
        return Route("pcie", "pcie", topology.peer_bandwidth_gbps(source, target))
    bandwidths = [
        value
        for value in (topology.host_bandwidth_gbps(source), topology.host_bandwidth_gbps(target))
        if value
    ]
    return Route(
        _host_link(topology, source),
        _host_link(topology, target),
        min(bandwidths) if bandwidths else None,
    )


def modeled_seconds(nbytes: int, bandwidth_gbps: float | None) -> float:
    if not bandwidth_gbps or bandwidth_gbps <= 0 or nbytes <= 0:
        return 0.0
    return nbytes / (bandwidth_gbps * 1e9)


class _LinkSchedule:
    """Copies on one link and direction, back to back."""

    __slots__ = ("segments", "busy_until", "bytes")

    def __init__(self) -> None:
        self.segments: deque[list[float]] = deque()  # [start, end, bytes]
        self.busy_until = 0.0
        self.bytes = 0

    def add(self, now: float, nbytes: int, seconds: float, window: float) -> None:
        self._prune(now, window)
        self.bytes += nbytes
        if seconds <= 0:
            return
        start = max(now, self.busy_until)
        end = start + seconds
        self.busy_until = end
        if self.segments:
            tail = self.segments[-1]
            contiguous = tail[1] == start
            same_rate = abs(tail[2] * seconds - nbytes * (tail[1] - tail[0])) <= 1e-9 * tail[2] * seconds
            if contiguous and (same_rate or len(self.segments) >= MAXIMUM_SEGMENTS):
                tail[1] = end
                tail[2] += nbytes
                return
        self.segments.append([start, end, float(nbytes)])

    def _prune(self, now: float, window: float) -> None:
        while self.segments and self.segments[0][1] <= now - window:
            self.segments.popleft()

    def pending(self, now: float) -> float:
        """Bytes the schedule has not delivered by ``now``."""

        remaining = 0.0
        for start, end, nbytes in reversed(self.segments):
            if end <= now:
                break
            remaining += nbytes * (end - max(start, now)) / (end - start)
        return remaining

    def delivered(self, now: float) -> int:
        return max(0, int(round(self.bytes - self.pending(now))))

    def rate(self, now: float, window: float) -> int:
        """Average bytes per second delivered over the last ``window``."""

        self._prune(now, window)
        low = now - window
        total = 0.0
        for start, end, nbytes in self.segments:
            if start >= now:
                break
            overlap = min(end, now) - max(start, low)
            if overlap > 0:
                total += nbytes * overlap / (end - start)
        return int(round(total / window))


class LinkTraffic:
    """Per-process record of modeled copies and link use, thread-safe.

    ``topology`` is called once, on the first copy, so the record can be
    created before the device layout is final.
    """

    def __init__(
        self,
        topology: Callable[[], LinkTopology],
        *,
        clock: Callable[[], float] = time.monotonic,
        window_seconds: float = WINDOW_SECONDS,
    ) -> None:
        self._topology_factory = topology
        self._topology: LinkTopology | None = None
        self._clock = clock
        self._window = window_seconds
        self._lock = threading.Lock()
        self._routes: dict[tuple[int | None, int | None], Route] = {}
        self._copies: dict[int, dict[str, list[float]]] = {}
        self._links: dict[tuple[int, str, str], _LinkSchedule] = {}

    @property
    def topology(self) -> LinkTopology:
        if self._topology is None:
            self._topology = self._topology_factory()
        return self._topology

    def _device_count(self) -> int:
        return len(self.topology.slots)

    def _route(self, source: int | None, target: int | None) -> Route:
        key = (source, target)
        found = self._routes.get(key)
        if found is None:
            found = self._routes[key] = route(self.topology, source, target)
        return found

    def _count(self, device: int, kind: str, nbytes: int, seconds: float) -> None:
        row = self._copies.setdefault(device, {name: [0, 0, 0.0] for name in COPY_KINDS})[kind]
        row[0] += 1
        row[1] += nbytes
        row[2] += seconds

    def _schedule(self, device: int, link: str | None, direction: str, now: float, nbytes: int, seconds: float) -> None:
        if link is None:
            return
        schedule = self._links.get((device, link, direction))
        if schedule is None:
            schedule = self._links[(device, link, direction)] = _LinkSchedule()
        schedule.add(now, nbytes, seconds, self._window)

    def record(self, source: int | None, target: int | None, nbytes: int) -> float:
        """Record one copy; ``None`` stands for host memory.

        Returns the modeled seconds of the copy. Copies within one GPU and
        copies between host buffers are not link traffic and are ignored.
        """

        nbytes = int(nbytes)
        if nbytes <= 0 or source == target:
            return 0.0
        with self._lock:
            count = self._device_count()
            if any(index is not None and not 0 <= index < count for index in (source, target)):
                return 0.0
            path = self._route(source, target)
            seconds = modeled_seconds(nbytes, path.bandwidth_gbps)
            now = self._clock()
            if source is None:
                self._count(target, "host_to_device", nbytes, seconds)
                self._schedule(target, path.target_link, "rx", now, nbytes, seconds)
            elif target is None:
                self._count(source, "device_to_host", nbytes, seconds)
                self._schedule(source, path.source_link, "tx", now, nbytes, seconds)
            else:
                self._count(source, "peer_send", nbytes, seconds)
                self._count(target, "peer_receive", nbytes, seconds)
                self._schedule(source, path.source_link, "tx", now, nbytes, seconds)
                self._schedule(target, path.target_link, "rx", now, nbytes, seconds)
            return seconds

    def snapshot(self) -> dict[int, dict[str, Any]]:
        """The ``transfer`` record of every GPU (see :func:`empty_record`)."""

        with self._lock:
            topology = self.topology
            now = self._clock()
            result: dict[int, dict[str, Any]] = {}
            for index, slot in enumerate(topology.slots):
                record = empty_record(slot.host_link, slot.host_bandwidth_gbps, self._window)
                for kind, (calls, nbytes, seconds) in self._copies.get(index, {}).items():
                    record["copies"][kind] = {
                        "calls": int(calls),
                        "bytes": int(nbytes),
                        "modeled_seconds": float(seconds),
                    }
                for link in LINK_KINDS:
                    entry = record["links"][link]
                    for direction in ("tx", "rx"):
                        schedule = self._links.get((index, link, direction))
                        if schedule is None:
                            continue
                        entry[f"{direction}_bytes"] = schedule.delivered(now)
                        entry[f"{direction}_bytes_per_second"] = schedule.rate(now, self._window)
                result[index] = _with_total(record)
            return result


def empty_record(host_link: str = "pcie", host_bandwidth_gbps: float | None = None, window: float = WINDOW_SECONDS) -> dict[str, Any]:
    """A ``transfer`` record with no copies.

    ``copies`` counts what the program asked for, with the modeled seconds
    of each kind, and ``modeled_seconds`` adds them up; ``links`` counts the
    bytes each link has delivered so far under the modeled schedule and
    their rate over ``window_seconds``. TX is data the GPU sends, RX data it
    receives.
    """

    return {
        "schema_version": SCHEMA_VERSION,
        "host_link": host_link,
        "host_bandwidth_gbps": host_bandwidth_gbps,
        "window_seconds": window,
        "modeled_seconds": 0.0,
        "copies": {kind: {"calls": 0, "bytes": 0, "modeled_seconds": 0.0} for kind in COPY_KINDS},
        "links": {
            link: {"tx_bytes": 0, "rx_bytes": 0, "tx_bytes_per_second": 0, "rx_bytes_per_second": 0}
            for link in LINK_KINDS
        },
    }


def normalize_record(value: Any) -> dict[str, Any] | None:
    """A ``transfer`` record from an SMI state, or ``None`` when absent."""

    if not isinstance(value, Mapping):
        return None
    host_link = value.get("host_link")
    record = empty_record(
        host_link if host_link in ("pcie", "c2c", "unified") else "pcie",
        _positive_float(value.get("host_bandwidth_gbps")),
        _positive_float(value.get("window_seconds")) or WINDOW_SECONDS,
    )
    copies = value.get("copies")
    if isinstance(copies, Mapping):
        for kind in COPY_KINDS:
            raw = copies.get(kind)
            if isinstance(raw, Mapping):
                record["copies"][kind] = {
                    "calls": _count(raw.get("calls")),
                    "bytes": _count(raw.get("bytes")),
                    "modeled_seconds": _positive_float(raw.get("modeled_seconds")) or 0.0,
                }
    links = value.get("links")
    if isinstance(links, Mapping):
        for link in LINK_KINDS:
            raw = links.get(link)
            if isinstance(raw, Mapping):
                record["links"][link] = {key: _count(raw.get(key)) for key in record["links"][link]}
    return _with_total(record)


def merge_records(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Sum the records several processes publish for one GPU."""

    if not records:
        return None
    merged = empty_record(records[0]["host_link"], records[0]["host_bandwidth_gbps"], records[0]["window_seconds"])
    for record in records:
        for kind in COPY_KINDS:
            for key in ("calls", "bytes", "modeled_seconds"):
                merged["copies"][kind][key] += record["copies"][kind][key]
        for link in LINK_KINDS:
            for key in merged["links"][link]:
                merged["links"][link][key] += record["links"][link][key]
    return _with_total(merged)


COPY_LABELS = {
    "host_to_device": "H2D",
    "device_to_host": "D2H",
    "peer_send": "peer out",
    "peer_receive": "peer in",
}
LINK_LABELS = {"pcie": "PCIe", "nvlink": "NVLink", "c2c": "C2C"}


def copies_summary(record: Mapping[str, Any], format_bytes: Callable[[int], str]) -> str:
    """``H2D 2.0 GiB / 68.2 ms, peer out ...`` for the kinds with copies."""

    parts = [
        f"{label} {format_bytes(record['copies'][kind]['bytes'])} / "
        f"{format_seconds(record['copies'][kind]['modeled_seconds'])}"
        for kind, label in COPY_LABELS.items()
        if record["copies"][kind]["bytes"]
    ]
    return ", ".join(parts) or "none"


def throughput_summary(record: Mapping[str, Any], links: Sequence[str]) -> str:
    """``PCIe TX 0 B/s RX 2.0 GiB/s, ...`` for ``links``."""

    parts = [
        f"{LINK_LABELS[link]} TX {format_rate(record['links'][link]['tx_bytes_per_second'])} "
        f"RX {format_rate(record['links'][link]['rx_bytes_per_second'])}"
        for link in links
    ]
    return ", ".join(parts) or "no links"


def format_seconds(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def format_rate(bytes_per_second: int) -> str:
    amount = float(bytes_per_second)
    for unit in ("B/s", "KiB/s", "MiB/s", "GiB/s"):
        if amount < 1024 or unit == "GiB/s":
            return f"{amount:.1f} {unit}" if unit != "B/s" else f"{int(amount)} B/s"
        amount /= 1024
    return f"{amount:.1f} GiB/s"


def _with_total(record: dict[str, Any]) -> dict[str, Any]:
    record["modeled_seconds"] = sum(entry["modeled_seconds"] for entry in record["copies"].values())
    return record


def _count(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, number)


def _positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number > 0 and number == number and number != float("inf") else None
