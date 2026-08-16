from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .operator_profiles import (
    evaluate_operator_profile,
    load_operator_profiles,
    match_operator_profile,
)
from .structured_io import emit_json, load_mapping
from .topology import Topology, simulate_point_to_point


SCHEMA_VERSION = "fakegpu.trace_replay.v1"


class TraceReplayError(ValueError):
    pass


def replay_trace(
    trace: Mapping[str, Any],
    *,
    topology: Topology | None = None,
    operator_profiles: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Replay Chrome/PyTorch, NCCL-style, or FakeGPU cluster trace events."""

    events = _normalize_trace(trace)
    if not events:
        raise TraceReplayError("trace contains no complete replayable events")
    enriched, profile_summary = _apply_profiles(events, operator_profiles)
    events = _remove_fusion_double_counts(enriched)
    topology_cache: dict[tuple[int, int, int], Mapping[str, Any]] = {}
    if topology is not None:
        for event in events:
            if event["category"] != "communication":
                continue
            src = event.get("src_rank")
            dst = event.get("dst_rank")
            payload = int(event.get("bytes", 0) or 0)
            if (
                isinstance(src, int)
                and isinstance(dst, int)
                and src != dst
                and payload > 0
            ):
                key = (src, dst, payload)
                if key not in topology_cache:
                    topology_cache[key] = simulate_point_to_point(
                        topology,
                        src_rank=src,
                        dst_rank=dst,
                        payload_bytes=payload,
                    )
                simulation = topology_cache[key]
                predicted = float(simulation["summary"]["estimated_time_us"])
                event["topology_estimated_duration_us"] = predicted
                event["topology_path"] = list(
                    simulation["transfers"][0]["path"]
                )
                if event["duration_us"] <= 0:
                    event["duration_us"] = predicted

    ranks = sorted({int(event["rank"]) for event in events})
    rank_summaries = [
        _rank_summary(rank, [event for event in events if event["rank"] == rank])
        for rank in ranks
    ]
    pairs = _pair_summary(events)
    links = _link_summary(events)
    resilience = _resilience_summary(events)
    memory_timeline = _memory_timeline(events)
    global_start = min(float(event["start_us"]) for event in events)
    global_end = max(
        float(event["start_us"]) + float(event["duration_us"])
        for event in events
    )
    critical_rank = max(
        rank_summaries,
        key=lambda item: float(item["elapsed_us"]),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "source_schema_version": trace.get("schema_version"),
        "method": "normalized_event_timeline_replay",
        "summary": {
            "event_count": len(events),
            "rank_count": len(ranks),
            "elapsed_us": global_end - global_start,
            "compute_time_us": sum(
                float(item["duration_us"])
                for item in events
                if item["category"] == "compute"
            ),
            "communication_time_us": sum(
                float(item["duration_us"])
                for item in events
                if item["category"] == "communication"
            ),
            "communication_bytes": sum(
                int(item.get("bytes", 0) or 0)
                for item in events
                if item["category"] == "communication"
            ),
            "critical_rank": critical_rank["rank"],
            "critical_rank_elapsed_us": critical_rank["elapsed_us"],
            "critical_links": links[:5],
        },
        "rank_summary": rank_summaries,
        "rank_pairs": pairs,
        "link_summary": links,
        "operator_profile_summary": profile_summary,
        "resilience": resilience,
        "memory_timeline": memory_timeline,
        "events": events,
        "limitations": [
            "Replay preserves recorded dependency-free timestamps; it does not reconstruct missing CUDA stream dependencies.",
            "Communication timing is measured trace duration unless a routed topology estimate is explicitly attached.",
            "Compute duration from a CPU or simulated trace is not a real-GPU kernel benchmark.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fakegpu replay-trace",
        description=(
            "Replay PyTorch Profiler, NCCL-style, or FakeGPU communication "
            "events and report compute/communication overlap, waits, rank "
            "pairs, critical links, memory, and recovery."
        ),
    )
    parser.add_argument("trace")
    parser.add_argument("--topology")
    parser.add_argument(
        "--operator-profiles",
        action="append",
        default=[],
        help="JSON/YAML operator profile catalog; may be repeated.",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        nargs="?",
        const="-",
        help="Write JSON to PATH, or stdout when PATH is omitted.",
    )
    args = parser.parse_args(argv)
    try:
        trace = load_mapping(args.trace)
        topology = (
            Topology.from_mapping(load_mapping(args.topology))
            if args.topology
            else None
        )
        profiles = load_operator_profiles(args.operator_profiles)
        report = replay_trace(
            trace,
            topology=topology,
            operator_profiles=profiles,
        )
        if args.json_path:
            output = emit_json(args.json_path, report)
            if output is not None:
                print(f"Trace replay: {output}")
        else:
            _print_report(report)
        return 0
    except (
            OSError,
            ValueError,
           ) as exc:
        parser.exit(2, f"fakegpu replay-trace: {exc}\n")


def _normalize_trace(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    trace_events = trace.get("traceEvents")
    if isinstance(trace_events, list):
        return _normalize_chrome_events(trace_events)
    events = trace.get("events")
    if isinstance(events, list):
        return _normalize_generic_events(events)
    timeline = trace.get("operation_timeline")
    if isinstance(timeline, Mapping) and isinstance(
        timeline.get("entries"),
        list,
    ):
        return _normalize_cluster_events(timeline["entries"])
    raise TraceReplayError(
        "supported trace roots are traceEvents, events, or operation_timeline.entries"
    )


def _normalize_chrome_events(
    raw_events: Sequence[Any],
) -> list[dict[str, Any]]:
    events = []
    for index, raw in enumerate(raw_events):
        if not isinstance(raw, Mapping) or raw.get("ph") not in {"X", None}:
            continue
        duration = _number(raw.get("dur", raw.get("duration_us", 0)), 0.0)
        if duration < 0:
            continue
        args = raw.get("args")
        args = dict(args) if isinstance(args, Mapping) else {}
        name = str(raw.get("name") or f"event_{index}")
        rank = _event_rank(raw, args)
        event = {
            "index": index,
            "name": name,
            "category": _event_category(name, str(raw.get("cat") or ""), args),
            "rank": rank,
            "thread": str(raw.get("tid", "")),
            "start_us": _number(raw.get("ts", 0), 0.0),
            "duration_us": duration,
            "bytes": _event_bytes(args),
            "src_rank": _optional_int(
                args.get("src_rank", args.get("src"))
            ),
            "dst_rank": _optional_int(
                args.get("dst_rank", args.get("dst"))
            ),
            "dtype": args.get("dtype"),
            "shape": _shape(args.get("shape", args.get("Input Dims"))),
            "numel": _optional_int(args.get("numel")),
            "fusion_id": args.get("fusion_id"),
            "allocated_bytes": _optional_int(
                args.get(
                    "allocated_bytes",
                    args.get("Total Allocated"),
                )
            ),
            "reserved_bytes": _optional_int(
                args.get("reserved_bytes", args.get("Total Reserved"))
            ),
            "generation": _optional_int(args.get("generation")),
        }
        event.update(_event_links(args))
        events.append(event)
    return events


def _normalize_generic_events(
    raw_events: Sequence[Any],
) -> list[dict[str, Any]]:
    events = []
    for index, raw in enumerate(raw_events):
        if not isinstance(raw, Mapping):
            continue
        name = str(raw.get("name") or raw.get("operation") or f"event_{index}")
        category = str(raw.get("category") or _event_category(name, "", raw))
        events.append(
            {
                "index": index,
                "name": name,
                "category": category,
                "rank": int(raw.get("rank", 0) or 0),
                "thread": str(raw.get("thread", "")),
                "start_us": _number(
                    raw.get("start_us", raw.get("timestamp_us", index)),
                    float(index),
                ),
                "duration_us": _number(
                    raw.get("duration_us", raw.get("estimated_time_us", 0)),
                    0.0,
                ),
                "bytes": int(raw.get("bytes", raw.get("payload_bytes", 0)) or 0),
                "src_rank": _optional_int(raw.get("src_rank")),
                "dst_rank": _optional_int(raw.get("dst_rank")),
                "dtype": raw.get("dtype"),
                "shape": _shape(raw.get("shape")),
                "numel": _optional_int(raw.get("numel")),
                "fusion_id": raw.get("fusion_id"),
                "allocated_bytes": _optional_int(raw.get("allocated_bytes")),
                "reserved_bytes": _optional_int(raw.get("reserved_bytes")),
                "generation": _optional_int(raw.get("generation")),
                "links": [str(value) for value in raw.get("links", [])],
            }
        )
    return events


def _normalize_cluster_events(
    raw_events: Sequence[Any],
) -> list[dict[str, Any]]:
    events = []
    cursor_by_rank: dict[int, float] = defaultdict(float)
    for index, raw in enumerate(raw_events):
        if not isinstance(raw, Mapping):
            continue
        raw_ranks = raw.get("ranks")
        if isinstance(raw_ranks, list):
            ranks = [
                int(rank)
                for rank in raw_ranks
                if _optional_int(rank) is not None
            ]
        else:
            ranks = [
                int(raw.get("rank", raw.get("src_rank", 0)) or 0)
            ]
        ranks = sorted(set(ranks)) or [0]
        rendezvous_wait = _number(raw.get("rendezvous_wait_us", 0), 0.0)
        duration = _number(
            raw.get(
                "execution_time_us",
                raw.get(
                    "modeled_time_us",
                    raw.get(
                        "coordinator_duration_us",
                        raw.get(
                            "estimated_time_us",
                            raw.get("duration_us", 0),
                        ),
                    ),
                ),
            ),
            0.0,
        )
        name = str(raw.get("operation") or raw.get("kind") or "collective")
        payload = int(
            raw.get(
                "logical_payload_bytes",
                raw.get("bytes", 0),
            )
            or 0
        )
        for rank in ranks:
            start = _number(
                raw.get("start_us", cursor_by_rank[rank]),
                cursor_by_rank[rank],
            )
            if rendezvous_wait > 0:
                events.append(
                    {
                        "index": len(events),
                        "name": f"{name}.rendezvous_wait",
                        "category": "wait",
                        "rank": rank,
                        "thread": "",
                        "start_us": start,
                        "duration_us": rendezvous_wait,
                        "bytes": 0,
                        "src_rank": None,
                        "dst_rank": None,
                        "dtype": raw.get("data_type", raw.get("dtype")),
                        "shape": None,
                        "numel": None,
                        "fusion_id": None,
                        "allocated_bytes": None,
                        "reserved_bytes": None,
                        "generation": _optional_int(raw.get("generation")),
                        "links": [],
                    }
                )
            communication_start = start + rendezvous_wait
            cursor_by_rank[rank] = max(
                cursor_by_rank[rank],
                communication_start + duration,
            )
            src_rank = _optional_int(raw.get("src_rank"))
            dst_rank = _optional_int(raw.get("dst_rank"))
            if (
                src_rank is None
                and dst_rank is None
                and len(ranks) == 2
                and str(raw.get("kind") or "").lower()
                in {"point_to_point", "p2p", "send", "recv"}
            ):
                src_rank, dst_rank = ranks
            events.append(
                {
                    "index": len(events),
                    "source_index": index,
                    "name": name,
                    "category": "communication",
                    "rank": rank,
                    "thread": "",
                    "start_us": communication_start,
                    "duration_us": duration,
                    "bytes": payload,
                    "src_rank": src_rank,
                    "dst_rank": dst_rank,
                    "dtype": raw.get("data_type", raw.get("dtype")),
                    "shape": None,
                    "numel": None,
                    "fusion_id": None,
                    "allocated_bytes": None,
                    "reserved_bytes": None,
                    "generation": _optional_int(raw.get("generation")),
                    "links": [
                        str(value) for value in raw.get("links", [])
                    ],
                }
            )
    return events


def _apply_profiles(
    events: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = []
    matched_counts: dict[str, int] = defaultdict(int)
    modeled_totals = {
        "workspace_bytes": 0,
        "flops": 0,
        "memory_bytes": 0,
        "latency_us": 0.0,
        "launch_count": 0,
    }
    for raw in events:
        event = dict(raw)
        profile = match_operator_profile(
            profiles,
            operator=str(event["name"]),
            dtype=(
                str(event["dtype"])
                if event.get("dtype") is not None
                else None
            ),
            shape=event.get("shape"),
        )
        if profile is not None:
            numel = event.get("numel")
            if numel is None and event.get("shape"):
                numel = math.prod(event["shape"])
            evaluated = evaluate_operator_profile(profile, numel=numel)
            event["operator_profile"] = evaluated
            matched_counts[str(evaluated["profile_id"])] += 1
            for field in modeled_totals:
                if field in evaluated:
                    modeled_totals[field] += evaluated[field]
            if event["duration_us"] <= 0 and "latency_us" in evaluated:
                event["duration_us"] = float(evaluated["latency_us"])
        output.append(event)
    return output, {
        "catalog_profile_count": len(profiles),
        "matched_event_count": sum(matched_counts.values()),
        "matched_profiles": dict(sorted(matched_counts.items())),
        "modeled_totals": modeled_totals,
    }


def _remove_fusion_double_counts(
    events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    fused_ids: dict[str, set[str]] = defaultdict(set)
    for item in events:
        profile = item.get("operator_profile")
        if item.get("fusion_id") is None or not isinstance(profile, Mapping):
            continue
        fused_ids[str(item["fusion_id"])].update(
            str(name) for name in profile.get("fused_operators", [])
        )
    output = []
    for raw in events:
        fusion_id = raw.get("fusion_id")
        if (
            fusion_id is not None
            and str(fusion_id) in fused_ids
            and str(raw["name"]) in fused_ids[str(fusion_id)]
        ):
            # The event's output is already accounted for by the fused
            # operator profile covering it — drop it from the aggregate.
            profile = raw.get("operator_profile")
            if not (
                isinstance(profile, Mapping)
                and profile.get("fused_operators")
            ):
                continue
        output.append(dict(raw))
    return output


def _rank_summary(
    rank: int,
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    start = min(float(item["start_us"]) for item in events)
    end = max(
        float(item["start_us"]) + float(item["duration_us"])
        for item in events
    )
    compute_intervals = _intervals(events, "compute")
    communication_intervals = _intervals(events, "communication")
    wait_intervals = _intervals(events, "wait")
    compute_time = _interval_union_length(compute_intervals)
    communication_time = _interval_union_length(communication_intervals)
    overlap = _interval_intersection_length(
        compute_intervals,
        communication_intervals,
    )
    explicit_wait = _interval_union_length(wait_intervals)
    exposed_communication = max(0.0, communication_time - overlap)
    elapsed = end - start
    busy = _interval_union_length(
        [*compute_intervals, *communication_intervals, *wait_intervals]
    )
    communication_bytes = sum(
        int(item.get("bytes", 0) or 0)
        for item in events
        if item["category"] == "communication"
    )
    peak_bandwidth = max(
        [
            int(item.get("bytes", 0) or 0)
            * 8
            / (float(item["duration_us"]) * 1_000)
            for item in events
            if item["category"] == "communication"
            and float(item["duration_us"]) > 0
        ],
        default=0.0,
    )
    return {
        "rank": rank,
        "event_count": len(events),
        "elapsed_us": elapsed,
        "compute_time_us": compute_time,
        "communication_time_us": communication_time,
        "compute_communication_overlap_us": overlap,
        "communication_wait_us": exposed_communication,
        "explicit_wait_us": explicit_wait,
        "idle_or_untracked_us": max(0.0, elapsed - busy),
        "communication_bytes": communication_bytes,
        "peak_effective_bandwidth_gbps": peak_bandwidth,
        "compute_utilization": compute_time / elapsed if elapsed > 0 else 0.0,
        "communication_utilization": (
            communication_time / elapsed if elapsed > 0 else 0.0
        ),
    }


def _pair_summary(
    events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    pairs: dict[tuple[int, int], dict[str, Any]] = {}
    for event in events:
        src = event.get("src_rank")
        dst = event.get("dst_rank")
        if not isinstance(src, int) or not isinstance(dst, int) or src == dst:
            continue
        key = tuple(sorted((src, dst)))
        item = pairs.setdefault(
            key,
            {
                "rank_a": key[0],
                "rank_b": key[1],
                "total_bytes": 0,
                "operation_count": 0,
                "peak_bytes_per_operation": 0,
                "peak_effective_bandwidth_gbps": 0.0,
            },
        )
        payload = int(event.get("bytes", 0) or 0)
        duration = float(event["duration_us"])
        bandwidth = payload * 8 / (duration * 1_000) if duration > 0 else 0.0
        item["total_bytes"] += payload
        item["operation_count"] += 1
        item["peak_bytes_per_operation"] = max(
            item["peak_bytes_per_operation"],
            payload,
        )
        item["peak_effective_bandwidth_gbps"] = max(
            item["peak_effective_bandwidth_gbps"],
            bandwidth,
        )
    return sorted(pairs.values(), key=lambda item: (item["rank_a"], item["rank_b"]))


def _link_summary(
    events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    links: dict[str, dict[str, Any]] = {}
    for event in events:
        for link in event.get("topology_path", event.get("links", [])):
            item = links.setdefault(
                str(link),
                {
                    "link": str(link),
                    "total_bytes": 0,
                    "operation_count": 0,
                    "total_communication_time_us": 0.0,
                },
            )
            item["total_bytes"] += int(event.get("bytes", 0) or 0)
            item["operation_count"] += 1
            item["total_communication_time_us"] += float(
                event["duration_us"]
            )
    for item in links.values():
        duration = float(item["total_communication_time_us"])
        item["average_effective_bandwidth_gbps"] = (
            int(item["total_bytes"]) * 8 / (duration * 1_000)
            if duration > 0
            else 0.0
        )
    return sorted(
        links.values(),
        key=lambda item: (
            -float(item["total_communication_time_us"]),
            item["link"],
        ),
    )


def _resilience_summary(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    patterns = {
        "failure": re.compile(r"fail|abort|exit|timeout", re.IGNORECASE),
        "retry": re.compile(r"retry|requeue", re.IGNORECASE),
        "restart": re.compile(r"restart|recover|rejoin", re.IGNORECASE),
        "communicator_change": re.compile(
            r"communicator|shrink|split|destroy|init",
            re.IGNORECASE,
        ),
    }
    state_events = []
    counts = {key: 0 for key in patterns}
    generations = set()
    for event in events:
        name = str(event["name"])
        generation = event.get("generation")
        if isinstance(generation, int):
            generations.add(generation)
        kinds = [
            kind for kind, pattern in patterns.items() if pattern.search(name)
        ]
        if kinds:
            for kind in kinds:
                counts[kind] += 1
            state_events.append(
                {
                    "start_us": event["start_us"],
                    "rank": event["rank"],
                    "name": name,
                    "kinds": kinds,
                    "generation": generation,
                }
            )
    state_events.sort(key=lambda item: (float(item["start_us"]), item["rank"]))
    return {
        **{f"{key}_count": value for key, value in counts.items()},
        "generations": sorted(generations),
        "events": state_events,
        "elastic_restart_observed": counts["restart"] > 0 or len(generations) > 1,
    }


def _memory_timeline(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    samples = [
        {
            "start_us": event["start_us"],
            "rank": event["rank"],
            "name": event["name"],
            "allocated_bytes": event.get("allocated_bytes"),
            "reserved_bytes": event.get("reserved_bytes"),
            "virtual_smi_process_memory_bytes": (
                event.get("reserved_bytes")
                if event.get("reserved_bytes") is not None
                else event.get("allocated_bytes")
            ),
        }
        for event in events
        if event.get("allocated_bytes") is not None
        or event.get("reserved_bytes") is not None
    ]
    return {
        "unit": "bytes",
        "sample_count": len(samples),
        "peak_allocated_bytes": max(
            [
                int(item["allocated_bytes"])
                for item in samples
                if item["allocated_bytes"] is not None
            ],
            default=0,
        ),
        "peak_reserved_bytes": max(
            [
                int(item["reserved_bytes"])
                for item in samples
                if item["reserved_bytes"] is not None
            ],
            default=0,
        ),
        "samples": sorted(
            samples,
            key=lambda item: (float(item["start_us"]), item["rank"]),
        ),
    }


def _intervals(
    events: Sequence[Mapping[str, Any]],
    category: str,
) -> list[tuple[float, float]]:
    return [
        (
            float(item["start_us"]),
            float(item["start_us"]) + float(item["duration_us"]),
        )
        for item in events
        if item["category"] == category and float(item["duration_us"]) > 0
    ]


def _interval_union_length(
    intervals: Sequence[tuple[float, float]],
) -> float:
    if not intervals:
        return 0.0
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def _interval_intersection_length(
    left: Sequence[tuple[float, float]],
    right: Sequence[tuple[float, float]],
) -> float:
    left_merged = _merge_intervals(left)
    right_merged = _merge_intervals(right)
    i = j = 0
    total = 0.0
    while i < len(left_merged) and j < len(right_merged):
        start = max(left_merged[i][0], right_merged[j][0])
        end = min(left_merged[i][1], right_merged[j][1])
        if end > start:
            total += end - start
        if left_merged[i][1] <= right_merged[j][1]:
            i += 1
        else:
            j += 1
    return total


def _merge_intervals(
    intervals: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _event_category(
    name: str,
    category: str,
    args: Mapping[str, Any],
) -> str:
    haystack = f"{name} {category}".lower()
    if any(
        token in haystack
        for token in (
            "nccl",
            "allreduce",
            "all_reduce",
            "allgather",
            "reduce_scatter",
            "broadcast",
            "send",
            "recv",
            "communication",
        )
    ):
        return "communication"
    if any(token in haystack for token in ("wait", "barrier", "idle")):
        return "wait"
    if any(token in haystack for token in ("memory", "alloc", "free")):
        return "memory"
    if args.get("bytes") and (
        args.get("src_rank") is not None or args.get("dst_rank") is not None
    ):
        return "communication"
    return "compute"


def _event_rank(
    event: Mapping[str, Any],
    args: Mapping[str, Any],
) -> int:
    for value in (
        args.get("rank"),
        event.get("rank"),
        event.get("pid"),
    ):
        parsed = _optional_int(value)
        if parsed is not None:
            return parsed
    return 0


def _event_bytes(args: Mapping[str, Any]) -> int:
    for key in (
        "bytes",
        "payload_bytes",
        "message_size",
        "Input size",
    ):
        value = args.get(key)
        parsed = _optional_int(value)
        if parsed is not None:
            return max(0, parsed)
    return 0


def _event_links(args: Mapping[str, Any]) -> dict[str, Any]:
    links = args.get("links")
    return {
        "links": (
            [str(value) for value in links]
            if isinstance(links, list)
            else []
        )
    }


def _shape(value: Any) -> list[int] | None:
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, int) and not isinstance(item, bool) and item >= 0
        for item in value
    ):
        return [int(item) for item in value]
    return None


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _number(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _print_report(report: Mapping[str, Any]) -> None:
    summary = report["summary"]
    print("FakeGPU trace replay")
    print(f"  events: {summary['event_count']}")
    print(f"  ranks: {summary['rank_count']}")
    print(f"  elapsed: {summary['elapsed_us']:.3f} us")
    print(f"  compute: {summary['compute_time_us']:.3f} us")
    print(f"  communication: {summary['communication_time_us']:.3f} us")
    print(f"  critical rank: {summary['critical_rank']}")


__all__ = [
    "SCHEMA_VERSION",
    "TraceReplayError",
    "main",
    "replay_trace",
]
