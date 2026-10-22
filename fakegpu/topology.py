"""Route and time communication over a configured cluster topology.

Units: every ``bandwidth_gbps`` in a topology config (``nics[].bandwidth_gbps``,
``links[].bandwidth_gbps``, ``endpoint_bandwidth_gbps``,
``nvlink_bandwidth_gbps``) is in gigabits per second (Gbit/s) per direction.
A bidirectional link carries that rate in each direction independently.
``latency_us`` values are microseconds.

NVIDIA quotes NVLink bandwidth in GB/s summed over both directions, so convert
before using it here: H100's 900 GB/s is 450 GB/s, or 3600 Gbit/s, per
direction.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import itertools
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from ._cli import (
    add_json_path_argument,
    command_prog,
    emit_json_report,
    usage_error,
)
from .structured_io import load_mapping


SCHEMA_VERSION = "fakegpu.topology_simulation.v1"

# Default per-GPU NVLink rate inside an ``nvlink_domain``, in Gbit/s per
# direction: NVLink 4 (Hopper, 18 links), from NVIDIA's 900 GB/s
# bidirectional figure for H100 (https://developer.nvidia.com/blog/
# introducing-nvidia-hgx-h100-an-accelerated-server-platform-for-ai-and-high-performance-computing/).
DEFAULT_NVLINK_BANDWIDTH_GBPS = 900 / 2 * 8


class TopologyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Link:
    """One link; ``bandwidth_gbps`` is Gbit/s per direction."""

    src: str
    dst: str
    bandwidth_gbps: float
    latency_us: float
    scope: str
    bidirectional: bool = True
    link_id: str | None = None

    @property
    def id(self) -> str:
        return self.link_id or f"{self.src}->{self.dst}"

    def transfer_time_us(self, payload_bytes: int) -> float:
        return self.latency_us + payload_bytes * 8 / (
            self.bandwidth_gbps * 1_000
        )


def _reverse_link_id(link: Link) -> str:
    base = f"{link.src}->{link.dst}"
    suffix = link.id[len(base) :] if link.id.startswith(base) else ""
    return f"{link.dst}->{link.src}{suffix}"


@dataclass(frozen=True, slots=True)
class Flow:
    src_rank: int
    dst_rank: int
    payload_bytes: int
    round_index: int
    phase: str


class Topology:
    def __init__(
        self,
        *,
        links: Sequence[Link],
        ranks: Mapping[int, Sequence[str]],
        rank_nodes: Mapping[int, str],
        node_racks: Mapping[str, str],
        ecmp: bool,
    ) -> None:
        if not links:
            raise TopologyError("topology must contain at least one link")
        if not ranks:
            raise TopologyError("topology must contain at least one rank")
        link_counts: dict[str, int] = defaultdict(int)
        normalized_links: list[Link] = []
        for link in links:
            base_id = link.id
            count = link_counts[base_id]
            link_counts[base_id] += 1
            normalized_id = (
                base_id if count == 0 else f"{base_id}#{count + 1}"
            )
            normalized_links.append(replace(link, link_id=normalized_id))
        self.links = normalized_links
        self.ranks = {
            int(rank): tuple(sorted(set(endpoints)))
            for rank, endpoints in ranks.items()
        }
        self.rank_nodes = {int(rank): str(node) for rank, node in rank_nodes.items()}
        self.node_racks = {str(node): str(rack) for node, rack in node_racks.items()}
        self.ecmp = bool(ecmp)
        self._adjacency: dict[str, list[tuple[str, Link]]] = defaultdict(list)
        for link in self.links:
            if link.bandwidth_gbps <= 0:
                raise TopologyError(f"link {link.id} bandwidth must be positive")
            if link.latency_us < 0:
                raise TopologyError(f"link {link.id} latency must be non-negative")
            self._adjacency[link.src].append((link.dst, link))
            if link.bidirectional:
                reverse = replace(
                    link,
                    src=link.dst,
                    dst=link.src,
                    link_id=_reverse_link_id(link),
                )
                self._adjacency[link.dst].append((link.src, reverse))
        for edges in self._adjacency.values():
            edges.sort(key=lambda item: (item[0], item[1].id))

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "Topology":
        nodes_raw = config.get("nodes")
        ranks_raw = config.get("ranks")
        if not isinstance(nodes_raw, list) or not nodes_raw:
            raise TopologyError("topology config requires a non-empty nodes list")
        if not isinstance(ranks_raw, list) or not ranks_raw:
            raise TopologyError("topology config requires a non-empty ranks list")

        links: list[Link] = []
        node_racks: dict[str, str] = {}
        node_nics: dict[str, list[str]] = defaultdict(list)
        seen_nic_ids: set[str] = set()
        for item in nodes_raw:
            if not isinstance(item, Mapping):
                raise TopologyError("each node must be an object")
            node_id = _required_id(item, "node")
            node_racks[node_id] = str(item.get("rack") or "default")
            nics = item.get("nics")
            if not isinstance(nics, list) or not nics:
                raise TopologyError(f"node {node_id!r} requires at least one NIC")
            for nic_index, nic_raw in enumerate(nics):
                if not isinstance(nic_raw, Mapping):
                    raise TopologyError(f"node {node_id!r} has an invalid NIC")
                nic_id = str(nic_raw.get("id") or f"{node_id}.nic{nic_index}")
                if nic_id in seen_nic_ids:
                    raise TopologyError(f"duplicate NIC id {nic_id!r}")
                seen_nic_ids.add(nic_id)
                switch = str(nic_raw.get("switch") or "")
                if not switch:
                    raise TopologyError(f"NIC {nic_id!r} has no switch")
                node_nics[node_id].append(nic_id)
                links.append(
                    Link(
                        src=nic_id,
                        dst=switch,
                        bandwidth_gbps=_positive_float(
                            nic_raw.get("bandwidth_gbps", 100),
                            f"NIC {nic_id} bandwidth_gbps",
                        ),
                        latency_us=_nonnegative_float(
                            nic_raw.get("latency_us", 1),
                            f"NIC {nic_id} latency_us",
                        ),
                        scope="host_to_leaf",
                    )
                )

        links_raw = config.get("links") or []
        if not isinstance(links_raw, list):
            raise TopologyError("topology links must be a list")
        for item in links_raw:
            if not isinstance(item, Mapping):
                raise TopologyError("each topology link must be an object")
            links.append(
                Link(
                    src=_required_string(item, "src"),
                    dst=_required_string(item, "dst"),
                    bandwidth_gbps=_positive_float(
                        item.get("bandwidth_gbps"),
                        "link bandwidth_gbps",
                    ),
                    latency_us=_nonnegative_float(
                        item.get("latency_us", 0),
                        "link latency_us",
                    ),
                    scope=str(item.get("scope") or "fabric"),
                    bidirectional=bool(item.get("bidirectional", True)),
                )
            )

        rank_endpoints: dict[int, list[str]] = {}
        rank_nodes: dict[int, str] = {}
        nvlink_domains: dict[str, list[str]] = defaultdict(list)
        endpoint_bandwidth = _positive_float(
            config.get("endpoint_bandwidth_gbps", 256),
            "endpoint_bandwidth_gbps",
        )
        endpoint_latency = _nonnegative_float(
            config.get("endpoint_latency_us", 0.5),
            "endpoint_latency_us",
        )
        for item in ranks_raw:
            if not isinstance(item, Mapping):
                raise TopologyError("each rank must be an object")
            rank = _nonnegative_int(item.get("rank"), "rank")
            if rank in rank_endpoints:
                raise TopologyError(f"duplicate rank {rank}")
            node = _required_string(item, "node")
            if node not in node_nics:
                raise TopologyError(f"rank {rank} references unknown node {node!r}")
            endpoint = f"rank:{rank}"
            configured_nic = item.get("nic")
            nics = (
                [str(configured_nic)]
                if configured_nic is not None
                else node_nics[node]
            )
            unknown_nics = sorted(set(nics) - set(node_nics[node]))
            if unknown_nics:
                raise TopologyError(
                    f"rank {rank} references NICs outside node {node!r}: "
                    + ", ".join(unknown_nics)
                )
            rank_endpoints[rank] = [endpoint]
            rank_nodes[rank] = node
            for nic in nics:
                links.append(
                    Link(
                        src=endpoint,
                        dst=nic,
                        bandwidth_gbps=endpoint_bandwidth,
                        latency_us=endpoint_latency,
                        scope="gpu_to_nic",
                    )
                )
            domain = item.get("nvlink_domain")
            if domain is not None:
                nvlink_domains[str(domain)].append(endpoint)

        nvlink_bandwidth = _positive_float(
            config.get("nvlink_bandwidth_gbps", DEFAULT_NVLINK_BANDWIDTH_GBPS),
            "nvlink_bandwidth_gbps",
        )
        nvlink_latency = _nonnegative_float(
            config.get("nvlink_latency_us", 0.25),
            "nvlink_latency_us",
        )
        for endpoints in nvlink_domains.values():
            for src, dst in itertools.combinations(sorted(endpoints), 2):
                links.append(
                    Link(
                        src=src,
                        dst=dst,
                        bandwidth_gbps=nvlink_bandwidth,
                        latency_us=nvlink_latency,
                        scope="nvlink_domain",
                    )
                )
        return cls(
            links=links,
            ranks=rank_endpoints,
            rank_nodes=rank_nodes,
            node_racks=node_racks,
            ecmp=bool(config.get("ecmp", True)),
        )

    def route(
        self,
        src_rank: int,
        dst_rank: int,
        payload_bytes: int,
        *,
        flow_id: str = "",
    ) -> list[Link]:
        if src_rank not in self.ranks or dst_rank not in self.ranks:
            raise TopologyError(
                f"route references unknown rank {src_rank}->{dst_rank}"
            )
        if src_rank == dst_rank:
            return []
        if payload_bytes < 0:
            raise TopologyError("payload_bytes must be non-negative")
        candidates = []
        for src in self.ranks[src_rank]:
            for dst in self.ranks[dst_rank]:
                candidates.extend(
                    self._shortest_paths(
                        src,
                        dst,
                        payload_bytes,
                        limit=32 if self.ecmp else 1,
                    )
                )
        if not candidates:
            raise TopologyError(
                f"no path between ranks {src_rank} and {dst_rank}"
            )
        best_cost = min(cost for cost, _path in candidates)
        tolerance = max(1e-9, abs(best_cost) * 1e-9)
        equal = [
            path
            for cost, path in candidates
            if abs(cost - best_cost) <= tolerance
        ]
        equal.sort(key=lambda path: tuple(link.id for link in path))
        if len(equal) == 1 or not self.ecmp:
            return equal[0]
        digest = hashlib.sha256(
            f"{src_rank}|{dst_rank}|{flow_id}".encode("utf-8")
        ).digest()
        return equal[int.from_bytes(digest[:8], "big") % len(equal)]

    def _shortest_paths(
        self,
        src: str,
        dst: str,
        payload_bytes: int,
        *,
        limit: int,
    ) -> list[tuple[float, list[Link]]]:
        sequence = itertools.count()
        heap: list[
            tuple[float, tuple[str, ...], str, int, tuple[Link, ...]]
        ] = [
            (0.0, (src,), src, next(sequence), ())
        ]
        best_destination = math.inf
        results: list[tuple[float, list[Link]]] = []
        best_node_cost: dict[str, float] = {src: 0.0}
        while heap and len(results) < limit:
            cost, visited, node, _serial, path = heapq.heappop(heap)
            if cost > best_destination + 1e-9:
                break
            if node == dst:
                best_destination = min(best_destination, cost)
                results.append((cost, list(path)))
                continue
            for neighbor, link in self._adjacency.get(node, ()):
                if neighbor in visited:
                    continue
                next_cost = cost + link.transfer_time_us(payload_bytes)
                known = best_node_cost.get(neighbor, math.inf)
                if next_cost > known + max(1e-9, abs(known) * 1e-9):
                    continue
                best_node_cost[neighbor] = min(known, next_cost)
                heapq.heappush(
                    heap,
                    (
                        next_cost,
                        (*visited, neighbor),
                        neighbor,
                        next(sequence),
                        (*path, link),
                    ),
                )
        return results


def simulate_collective(
    topology: Topology,
    *,
    collective: str,
    payload_bytes_per_rank: int,
    algorithm: str = "ring",
) -> dict[str, Any]:
    if payload_bytes_per_rank <= 0:
        raise TopologyError("payload_bytes_per_rank must be greater than zero")
    ranks = sorted(topology.ranks)
    if len(ranks) < 2:
        raise TopologyError("collective simulation requires at least two ranks")
    normalized_collective = collective.lower().replace("-", "_")
    normalized_algorithm = algorithm.lower().replace("-", "_")
    if normalized_collective not in {
        "all_reduce",
        "all_gather",
        "reduce_scatter",
        "broadcast",
    }:
        raise TopologyError(f"unsupported collective {collective!r}")
    if normalized_algorithm not in {"ring", "tree", "hierarchical"}:
        raise TopologyError(f"unsupported algorithm {algorithm!r}")

    if normalized_algorithm == "ring":
        flows = _ring_flows(
            ranks,
            collective=normalized_collective,
            payload_bytes=payload_bytes_per_rank,
        )
    elif normalized_algorithm == "tree":
        flows = _tree_flows(
            ranks,
            collective=normalized_collective,
            payload_bytes=payload_bytes_per_rank,
        )
    else:
        flows = _hierarchical_flows(
            topology,
            ranks,
            collective=normalized_collective,
            payload_bytes=payload_bytes_per_rank,
        )
    return _simulate_flows(
        topology,
        flows,
        operation={
            "kind": "collective",
            "collective": normalized_collective,
            "algorithm": normalized_algorithm,
            "payload_bytes_per_rank": payload_bytes_per_rank,
        },
    )


def simulate_point_to_point(
    topology: Topology,
    *,
    src_rank: int,
    dst_rank: int,
    payload_bytes: int,
) -> dict[str, Any]:
    if payload_bytes <= 0:
        raise TopologyError("payload_bytes must be greater than zero")
    return _simulate_flows(
        topology,
        [
            Flow(
                src_rank=src_rank,
                dst_rank=dst_rank,
                payload_bytes=payload_bytes,
                round_index=0,
                phase="point_to_point",
            )
        ],
        operation={
            "kind": "point_to_point",
            "src_rank": src_rank,
            "dst_rank": dst_rank,
            "payload_bytes": payload_bytes,
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=command_prog(__name__),
        description=(
            "Simulate routed communication across racks, switches, multi-NIC "
            "hosts, NVLink domains, ECMP paths, and contended links."
        ),
    )
    parser.add_argument("config")
    parser.add_argument(
        "--collective",
        choices=["all-reduce", "all-gather", "reduce-scatter", "broadcast"],
        default="all-reduce",
    )
    parser.add_argument(
        "--algorithm",
        choices=["ring", "tree", "hierarchical"],
        default="ring",
    )
    parser.add_argument("--bytes-per-rank", type=int)
    parser.add_argument("--src-rank", type=int)
    parser.add_argument("--dst-rank", type=int)
    parser.add_argument("--bytes", type=int)
    add_json_path_argument(parser)
    args = parser.parse_args(argv)
    try:
        topology = Topology.from_mapping(load_mapping(args.config))
        if args.src_rank is not None or args.dst_rank is not None:
            if (
                args.src_rank is None
                or args.dst_rank is None
                or args.bytes is None
            ):
                raise TopologyError(
                    "point-to-point mode requires --src-rank, --dst-rank, and --bytes"
                )
            report = simulate_point_to_point(
                topology,
                src_rank=args.src_rank,
                dst_rank=args.dst_rank,
                payload_bytes=args.bytes,
            )
        else:
            if args.bytes_per_rank is None:
                raise TopologyError(
                    "collective mode requires --bytes-per-rank"
                )
            report = simulate_collective(
                topology,
                collective=args.collective,
                payload_bytes_per_rank=args.bytes_per_rank,
                algorithm=args.algorithm,
            )
        if args.json_path:
            emit_json_report(args.json_path, report, "Topology simulation")
        else:
            _print_report(report)
        return 0
    except (
            OSError,
            ValueError,
           ) as exc:
        usage_error(parser, exc)


def _simulate_flows(
    topology: Topology,
    flows: Sequence[Flow],
    *,
    operation: Mapping[str, Any],
) -> dict[str, Any]:
    rounds: dict[int, list[Flow]] = defaultdict(list)
    for flow in flows:
        rounds[flow.round_index].append(flow)

    link_totals: dict[str, dict[str, Any]] = {}
    rank_totals: dict[int, dict[str, Any]] = {
        rank: {
            "rank": rank,
            "node": topology.rank_nodes[rank],
            "rack": topology.node_racks.get(
                topology.rank_nodes[rank],
                "default",
            ),
            "sent_bytes": 0,
            "received_bytes": 0,
            "communication_time_us": 0.0,
            "peak_effective_bandwidth_gbps": 0.0,
        }
        for rank in sorted(topology.ranks)
    }
    pair_totals: dict[tuple[int, int], dict[str, Any]] = {}
    transfer_entries = []
    round_entries = []
    total_time_us = 0.0

    for round_index in sorted(rounds):
        routed = []
        round_link_bytes: dict[str, int] = defaultdict(int)
        round_link_flow_counts: dict[str, int] = defaultdict(int)
        round_link_max_payload: dict[str, int] = defaultdict(int)
        link_objects: dict[str, Link] = {}
        for flow_index, flow in enumerate(rounds[round_index]):
            path = topology.route(
                flow.src_rank,
                flow.dst_rank,
                flow.payload_bytes,
                flow_id=f"{round_index}:{flow_index}:{flow.phase}",
            )
            ideal_time = sum(
                link.transfer_time_us(flow.payload_bytes) for link in path
            )
            routed.append((flow, path, ideal_time))
            for link in path:
                round_link_bytes[link.id] += flow.payload_bytes
                round_link_flow_counts[link.id] += 1
                round_link_max_payload[link.id] = max(
                    round_link_max_payload[link.id], flow.payload_bytes
                )
                link_objects[link.id] = link

        link_service_times = {
            link_id: link_objects[link_id].transfer_time_us(payload)
            for link_id, payload in round_link_bytes.items()
        }
        round_duration = max(link_service_times.values(), default=0.0)
        total_time_us += round_duration
        round_entries.append(
            {
                "round": round_index,
                "phase": sorted({flow.phase for flow, _path, _ideal in routed}),
                "flow_count": len(routed),
                "duration_us": round_duration,
                "contended_links": sorted(
                    link_id
                    for link_id, payload in round_link_bytes.items()
                    if round_link_flow_counts[link_id] > 1
                    and payload > 0
                ),
            }
        )
        participants = set()
        for flow, path, ideal_time in routed:
            participants.update((flow.src_rank, flow.dst_rank))
            effective_bandwidth = (
                flow.payload_bytes * 8 / (round_duration * 1_000)
                if round_duration > 0
                else 0.0
            )
            transfer_entries.append(
                {
                    "round": round_index,
                    "phase": flow.phase,
                    "src_rank": flow.src_rank,
                    "dst_rank": flow.dst_rank,
                    "bytes": flow.payload_bytes,
                    "path": [link.id for link in path],
                    "path_scopes": [link.scope for link in path],
                    "ideal_time_us": ideal_time,
                    "scheduled_time_us": round_duration,
                    "contention_penalty_us": max(
                        0.0,
                        round_duration - ideal_time,
                    ),
                    "effective_bandwidth_gbps": effective_bandwidth,
                }
            )
            src_stats = rank_totals[flow.src_rank]
            dst_stats = rank_totals[flow.dst_rank]
            src_stats["sent_bytes"] += flow.payload_bytes
            dst_stats["received_bytes"] += flow.payload_bytes
            src_stats["peak_effective_bandwidth_gbps"] = max(
                src_stats["peak_effective_bandwidth_gbps"],
                effective_bandwidth,
            )
            dst_stats["peak_effective_bandwidth_gbps"] = max(
                dst_stats["peak_effective_bandwidth_gbps"],
                effective_bandwidth,
            )
            pair_key = tuple(sorted((flow.src_rank, flow.dst_rank)))
            pair = pair_totals.setdefault(
                pair_key,
                {
                    "rank_a": pair_key[0],
                    "rank_b": pair_key[1],
                    "total_bytes": 0,
                    "peak_bytes_per_round": 0,
                    "transfer_count": 0,
                    "peak_effective_bandwidth_gbps": 0.0,
                },
            )
            pair["total_bytes"] += flow.payload_bytes
            pair["peak_bytes_per_round"] = max(
                pair["peak_bytes_per_round"],
                flow.payload_bytes,
            )
            pair["transfer_count"] += 1
            pair["peak_effective_bandwidth_gbps"] = max(
                pair["peak_effective_bandwidth_gbps"],
                effective_bandwidth,
            )
        for rank in participants:
            rank_totals[rank]["communication_time_us"] += round_duration

        for link_id, payload in round_link_bytes.items():
            link = link_objects[link_id]
            stats = link_totals.setdefault(
                link_id,
                {
                    "id": link_id,
                    "src": link.src,
                    "dst": link.dst,
                    "scope": link.scope,
                    "bandwidth_gbps": link.bandwidth_gbps,
                    "latency_us": link.latency_us,
                    "total_bytes": 0,
                    "peak_bytes_per_round": 0,
                    "active_rounds": 0,
                    "service_time_us": 0.0,
                    "contention_penalty_us": 0.0,
                },
            )
            stats["total_bytes"] += payload
            stats["peak_bytes_per_round"] = max(
                stats["peak_bytes_per_round"],
                payload,
            )
            stats["active_rounds"] += 1
            stats["service_time_us"] += link_service_times[link_id]
            ideal_single = link.transfer_time_us(
                round_link_max_payload[link_id]
            )
            stats["contention_penalty_us"] += max(
                0.0,
                link_service_times[link_id] - ideal_single,
            )

    links = sorted(
        link_totals.values(),
        key=lambda item: (-float(item["service_time_us"]), item["id"]),
    )
    for item in links:
        item["time_share"] = (
            float(item["service_time_us"]) / total_time_us
            if total_time_us > 0
            else 0.0
        )
        item["average_effective_bandwidth_gbps"] = (
            int(item["total_bytes"])
            * 8
            / (float(item["service_time_us"]) * 1_000)
            if float(item["service_time_us"]) > 0
            else 0.0
        )
    critical_links = [
        {
            "id": item["id"],
            "scope": item["scope"],
            "service_time_us": item["service_time_us"],
            "time_share": item["time_share"],
            "total_bytes": item["total_bytes"],
        }
        for item in links[: min(5, len(links))]
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "method": "routed_round_simulation_with_link_serialization",
        "operation": dict(operation),
        "topology": {
            "rank_count": len(topology.ranks),
            "node_count": len(set(topology.rank_nodes.values())),
            "rack_count": len(set(topology.node_racks.values())),
            "configured_link_count": len(topology.links),
            "ecmp": topology.ecmp,
        },
        "summary": {
            "round_count": len(rounds),
            "flow_count": len(flows),
            "total_payload_bytes": sum(flow.payload_bytes for flow in flows),
            "estimated_time_us": total_time_us,
            "critical_links": critical_links,
        },
        "rounds": round_entries,
        "transfers": transfer_entries,
        "links": links,
        "rank_summary": list(rank_totals.values()),
        "rank_pairs": sorted(pair_totals.values(), key=lambda item: (item["rank_a"], item["rank_b"])),
        "limitations": [
            "The model serializes simultaneous bytes on each route but does not reproduce NCCL packet protocols.",
            "ECMP selection is deterministic per logical flow; adaptive routing and switch buffering are not modeled.",
            "Configured link rates and latencies are analytical inputs, not measured network bandwidth.",
        ],
    }


def _ring_flows(
    ranks: Sequence[int],
    *,
    collective: str,
    payload_bytes: int,
) -> list[Flow]:
    if collective == "broadcast":
        return _tree_flows(
            ranks,
            collective=collective,
            payload_bytes=payload_bytes,
        )
    rank_count = len(ranks)
    if collective == "all_reduce":
        round_count = 2 * (rank_count - 1)
        chunk = _ceil_div(payload_bytes, rank_count)
    elif collective == "all_gather":
        round_count = rank_count - 1
        chunk = payload_bytes
    else:
        round_count = rank_count - 1
        chunk = _ceil_div(payload_bytes, rank_count)
    flows = []
    for round_index in range(round_count):
        if collective == "all_reduce":
            phase = (
                "reduce_scatter" if round_index < rank_count - 1 else "all_gather"
            )
        else:
            phase = collective
        flows.extend(
            Flow(
                src_rank=src,
                dst_rank=ranks[(index + 1) % rank_count],
                payload_bytes=chunk,
                round_index=round_index,
                phase=phase,
            )
            for index, src in enumerate(ranks)
        )
    return flows


def _tree_flows(
    ranks: Sequence[int],
    *,
    collective: str,
    payload_bytes: int,
) -> list[Flow]:
    levels: list[list[tuple[int, int]]] = []
    stride = 1
    while stride < len(ranks):
        levels.append(
            [
                (ranks[base + stride], ranks[base])
                for base in range(0, len(ranks) - stride, stride * 2)
            ]
        )
        stride *= 2
    flows = []
    round_index = 0
    if collective in {"all_reduce", "reduce_scatter"}:
        bytes_for_reduce = (
            _ceil_div(payload_bytes, len(ranks))
            if collective == "reduce_scatter"
            else payload_bytes
        )
        for edges in levels:
            for src, dst in edges:
                flows.append(
                    Flow(src, dst, bytes_for_reduce, round_index, "reduce")
                )
            round_index += 1
    if collective in {"all_reduce", "all_gather", "broadcast"}:
        phase = "broadcast" if collective == "broadcast" else "gather"
        for edges in reversed(levels):
            for child, parent in edges:
                flows.append(
                    Flow(parent, child, payload_bytes, round_index, phase)
                )
            round_index += 1
    return flows


def _hierarchical_flows(
    topology: Topology,
    ranks: Sequence[int],
    *,
    collective: str,
    payload_bytes: int,
) -> list[Flow]:
    if collective != "all_reduce":
        return _tree_flows(
            ranks,
            collective=collective,
            payload_bytes=payload_bytes,
        )
    # ``ranks`` is sorted, so each node's first rank is its leader.
    by_node: dict[str, list[int]] = defaultdict(list)
    for rank in ranks:
        by_node[topology.rank_nodes[rank]].append(rank)
    leaders = [by_node[node][0] for node in sorted(by_node)]
    flows = [
        Flow(rank, members[0], payload_bytes, 0, "intra_node_reduce")
        for members in by_node.values()
        for rank in members[1:]
    ]
    flows.extend(
        replace(
            flow,
            round_index=flow.round_index + 1,
            phase=f"inter_node_{flow.phase}",
        )
        for flow in _tree_flows(
            leaders,
            collective="all_reduce",
            payload_bytes=payload_bytes,
        )
    )
    final_round = max((flow.round_index for flow in flows), default=0) + 1
    flows.extend(
        Flow(members[0], rank, payload_bytes, final_round, "intra_node_broadcast")
        for members in by_node.values()
        for rank in members[1:]
    )
    return flows


def _required_id(item: Mapping[str, Any], kind: str) -> str:
    value = item.get("id")
    if not isinstance(value, str) or not value:
        raise TopologyError(f"{kind} requires a non-empty id")
    return value


def _required_string(item: Mapping[str, Any], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value:
        raise TopologyError(f"{key} must be a non-empty string")
    return value


def _positive_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise TopologyError(f"{name} must be positive") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise TopologyError(f"{name} must be positive")
    return parsed


def _nonnegative_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise TopologyError(f"{name} must be non-negative") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise TopologyError(f"{name} must be non-negative")
    return parsed


def _nonnegative_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise TopologyError(f"{name} must be a non-negative integer") from exc
    if parsed < 0:
        raise TopologyError(f"{name} must be a non-negative integer")
    return parsed


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _print_report(report: Mapping[str, Any]) -> None:
    summary = report["summary"]
    operation = report["operation"]
    print("FakeGPU topology simulation")
    print(f"  operation: {operation['kind']}")
    if operation["kind"] == "collective":
        print(
            f"  collective: {operation['collective']} "
            f"({operation['algorithm']})"
        )
    print(f"  rounds: {summary['round_count']}")
    print(f"  flows: {summary['flow_count']}")
    print(f"  estimated time: {summary['estimated_time_us']:.3f} us")
    if summary["critical_links"]:
        print(f"  critical link: {summary['critical_links'][0]['id']}")


__all__ = [
    "Flow",
    "Link",
    "SCHEMA_VERSION",
    "Topology",
    "TopologyError",
    "main",
    "simulate_collective",
    "simulate_point_to_point",
]
