#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _die(msg: str) -> None:
    print(f"[check_cluster_report] ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _require(obj: dict, key: str, *, ctx: str) -> object:
    if key not in obj:
        _die(f"missing key '{key}' in {ctx}")
    return obj[key]


def _require_counter(obj: dict, key: str, *, ctx: str) -> dict:
    value = _require(obj, key, ctx=ctx)
    if not isinstance(value, dict):
        _die(f"{ctx}.{key} must be an object")
    calls = _require(value, "calls", ctx=f"{ctx}.{key}")
    bytes_value = _require(value, "bytes", ctx=f"{ctx}.{key}")
    if not isinstance(calls, int) or calls < 0:
        _die(f"{ctx}.{key}.calls must be a non-negative integer")
    if not isinstance(bytes_value, int) or bytes_value < 0:
        _die(f"{ctx}.{key}.bytes must be a non-negative integer")
    return value


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate FakeGPU cluster report schema")
    ap.add_argument("--path", default="fake_gpu_cluster_report.json", help="Path to fake_gpu_cluster_report.json")
    ap.add_argument("--expect-world-size", type=int, default=None)
    ap.add_argument("--expect-node-count", type=int, default=None)
    ap.add_argument(
        "--expect-collective",
        choices=["all_reduce", "broadcast", "all_gather", "reduce_scatter", "barrier"],
        action="append",
        default=[],
        help="Require the named collective to have calls > 0 (repeatable)",
    )
    ap.add_argument("--min-ranks", type=int, default=1, help="Minimum number of rank entries expected")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.is_file():
        _die(f"report file not found: {path}")

    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        _die("report root must be a JSON object")

    version = report.get("report_version")
    if version != 4:
        _die(f"unexpected report_version={version!r} (expected 4)")

    schema = _require(report, "schema", ctx="root")
    if schema != "experimental":
        _die(f"unexpected schema={schema!r} (expected 'experimental')")

    cluster = _require(report, "cluster", ctx="root")
    if not isinstance(cluster, dict):
        _die("cluster must be an object")
    world_size = _require(cluster, "world_size", ctx="cluster")
    node_count = _require(cluster, "node_count", ctx="cluster")
    communicators = _require(cluster, "communicators", ctx="cluster")
    transport = _require(cluster, "coordinator_transport", ctx="cluster")
    if not isinstance(world_size, int) or world_size <= 0:
        _die("cluster.world_size must be a positive integer")
    if not isinstance(node_count, int) or node_count <= 0:
        _die("cluster.node_count must be a positive integer")
    if not isinstance(communicators, int) or communicators <= 0:
        _die("cluster.communicators must be a positive integer")
    if not isinstance(transport, str) or not transport:
        _die("cluster.coordinator_transport must be a non-empty string")
    if args.expect_world_size is not None and world_size != args.expect_world_size:
        _die(f"expected world_size={args.expect_world_size}, got {world_size}")
    if args.expect_node_count is not None and node_count != args.expect_node_count:
        _die(f"expected node_count={args.expect_node_count}, got {node_count}")

    collectives = _require(report, "collectives", ctx="root")
    if not isinstance(collectives, dict):
        _die("collectives must be an object")
    non_zero_collectives = 0
    for key in ("all_reduce", "broadcast", "all_gather", "reduce_scatter", "barrier"):
        counter = _require_counter(collectives, key, ctx="collectives")
        if int(counter["calls"]) > 0:
            non_zero_collectives += 1
    if non_zero_collectives == 0:
        _die("expected at least one collective counter to be non-zero")
    for expected_name in args.expect_collective:
        expected = collectives[expected_name]
        if int(expected["calls"]) <= 0:
            _die(f"expected collectives.{expected_name}.calls > 0")

    ranks = _require(report, "ranks", ctx="root")
    if not isinstance(ranks, list) or len(ranks) < args.min_ranks:
        _die(f"ranks must contain at least {args.min_ranks} entries")

    seen_ranks: set[int] = set()
    for index, rank_stats in enumerate(ranks):
        ctx = f"ranks[{index}]"
        if not isinstance(rank_stats, dict):
            _die(f"{ctx} must be an object")
        rank = _require(rank_stats, "rank", ctx=ctx)
        node = _require(rank_stats, "node", ctx=ctx)
        wait_time_ms = _require(rank_stats, "wait_time_ms", ctx=ctx)
        timeouts = _require(rank_stats, "timeouts", ctx=ctx)
        communicator_inits = _require(rank_stats, "communicator_inits", ctx=ctx)
        collective_calls = _require(rank_stats, "collective_calls", ctx=ctx)
        barrier_calls = _require(rank_stats, "barrier_calls", ctx=ctx)
        group_prepares = _require(rank_stats, "group_prepares", ctx=ctx)

        if not isinstance(rank, int) or rank < 0:
            _die(f"{ctx}.rank must be a non-negative integer")
        if rank in seen_ranks:
            _die(f"duplicate rank entry: {rank}")
        seen_ranks.add(rank)
        if not isinstance(node, str) or not node:
            _die(f"{ctx}.node must be a non-empty string")
        if not isinstance(wait_time_ms, (int, float)) or wait_time_ms < 0:
            _die(f"{ctx}.wait_time_ms must be a non-negative number")
        for field_name, field_value in (
            ("timeouts", timeouts),
            ("communicator_inits", communicator_inits),
            ("collective_calls", collective_calls),
            ("barrier_calls", barrier_calls),
            ("group_prepares", group_prepares),
        ):
            if not isinstance(field_value, int) or field_value < 0:
                _die(f"{ctx}.{field_name} must be a non-negative integer")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
