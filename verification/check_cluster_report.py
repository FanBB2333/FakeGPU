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
    estimated_time = value.get("estimated_time_us_total")
    if estimated_time is not None and (
        not isinstance(estimated_time, (int, float)) or estimated_time < 0
    ):
        _die(f"{ctx}.{key}.estimated_time_us_total must be a non-negative number")
    contention_penalty = value.get("contention_penalty_us_total")
    if contention_penalty is not None and (
        not isinstance(contention_penalty, (int, float)) or contention_penalty < 0
    ):
        _die(f"{ctx}.{key}.contention_penalty_us_total must be a non-negative number")
    return value


def _require_non_negative_number(obj: dict, key: str, *, ctx: str) -> float:
    value = _require(obj, key, ctx=ctx)
    if not isinstance(value, (int, float)) or value < 0:
        _die(f"{ctx}.{key} must be a non-negative number")
    return float(value)


def _require_non_negative_integer(obj: dict, key: str, *, ctx: str) -> int:
    value = _require(obj, key, ctx=ctx)
    if not isinstance(value, int) or value < 0:
        _die(f"{ctx}.{key} must be a non-negative integer")
    return value


def _validate_pair_direction(obj: object, *, ctx: str) -> dict:
    if not isinstance(obj, dict):
        _die(f"{ctx} must be an object")
    _require_non_negative_integer(obj, "transfers", ctx=ctx)
    _require_non_negative_integer(obj, "total_bytes", ctx=ctx)
    _require_non_negative_integer(obj, "peak_bytes_per_operation", ctx=ctx)
    for key in (
        "model_bandwidth_gbps",
        "avg_latency_us",
        "estimated_time_us_total",
        "contention_penalty_us_total",
        "average_estimated_throughput_gbps",
        "peak_estimated_throughput_gbps",
    ):
        _require_non_negative_number(obj, key, ctx=ctx)
    return obj


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate FakeGPU cluster report schema")
    ap.add_argument("--path", default="fake_gpu_cluster_report.json", help="Path to fake_gpu_cluster_report.json")
    ap.add_argument("--expect-world-size", type=int, default=None)
    ap.add_argument("--expect-node-count", type=int, default=None)
    ap.add_argument(
        "--expect-collective",
        choices=["all_reduce", "reduce", "broadcast", "all_gather", "reduce_scatter", "all_to_all", "barrier"],
        action="append",
        default=[],
        help="Require the named collective to have calls > 0 (repeatable)",
    )
    ap.add_argument("--expect-links", action="store_true", help="Require non-empty link statistics")
    ap.add_argument(
        "--expect-markdown",
        action="store_true",
        help="Require the companion Markdown report and node-pair table",
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
    if not isinstance(version, str):
        _die(f"unexpected report_version={version!r} (expected a version string)")

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
    for key in ("all_reduce", "reduce", "broadcast", "all_gather", "reduce_scatter", "all_to_all", "barrier"):
        counter = _require_counter(collectives, key, ctx="collectives")
        if int(counter["calls"]) > 0:
            non_zero_collectives += 1
    if non_zero_collectives == 0:
        _die("expected at least one collective counter to be non-zero")
    for expected_name in args.expect_collective:
        expected = collectives[expected_name]
        if int(expected["calls"]) <= 0:
            _die(f"expected collectives.{expected_name}.calls > 0")

    links = report.get("links", [])
    if args.expect_links:
        if not isinstance(links, list) or not links:
            _die("expected non-empty links array")
        for index, link in enumerate(links):
            ctx = f"links[{index}]"
            if not isinstance(link, dict):
                _die(f"{ctx} must be an object")
            src = _require(link, "src", ctx=ctx)
            dst = _require(link, "dst", ctx=ctx)
            scope = _require(link, "scope", ctx=ctx)
            samples = _require(link, "samples", ctx=ctx)
            bytes_value = _require(link, "bytes", ctx=ctx)
            peak_bytes = _require(link, "peak_bytes_per_operation", ctx=ctx)
            bandwidth = _require(link, "bandwidth_gbps", ctx=ctx)
            avg_latency = _require(link, "avg_latency_us", ctx=ctx)
            estimated_time = _require(link, "estimated_time_us_total", ctx=ctx)
            contention_penalty = _require(link, "contention_penalty_us_total", ctx=ctx)
            average_throughput = _require(
                link,
                "average_estimated_throughput_gbps",
                ctx=ctx,
            )
            peak_throughput = _require(
                link,
                "peak_estimated_throughput_gbps",
                ctx=ctx,
            )

            if not isinstance(src, str) or not src:
                _die(f"{ctx}.src must be a non-empty string")
            if not isinstance(dst, str) or not dst:
                _die(f"{ctx}.dst must be a non-empty string")
            if scope not in ("intra_node", "inter_node"):
                _die(f"{ctx}.scope must be intra_node or inter_node")
            if not isinstance(samples, int) or samples <= 0:
                _die(f"{ctx}.samples must be a positive integer")
            if not isinstance(bytes_value, int) or bytes_value < 0:
                _die(f"{ctx}.bytes must be a non-negative integer")
            if not isinstance(peak_bytes, int) or peak_bytes < 0:
                _die(f"{ctx}.peak_bytes_per_operation must be a non-negative integer")
            for field_name, field_value in (
                ("bandwidth_gbps", bandwidth),
                ("avg_latency_us", avg_latency),
                ("estimated_time_us_total", estimated_time),
                ("contention_penalty_us_total", contention_penalty),
                (
                    "average_estimated_throughput_gbps",
                    average_throughput,
                ),
                ("peak_estimated_throughput_gbps", peak_throughput),
            ):
                if not isinstance(field_value, (int, float)) or field_value < 0:
                    _die(f"{ctx}.{field_name} must be a non-negative number")

    node_pairs = _require(report, "node_pairs", ctx="root")
    if not isinstance(node_pairs, list):
        _die("node_pairs must be an array")
    expected_pair_count = node_count * (node_count - 1) // 2
    if len(node_pairs) != expected_pair_count:
        _die(
            f"node_pairs must contain every distinct node pair: "
            f"expected {expected_pair_count}, got {len(node_pairs)}"
        )

    seen_pairs: set[tuple[str, str]] = set()
    for index, pair in enumerate(node_pairs):
        ctx = f"node_pairs[{index}]"
        if not isinstance(pair, dict):
            _die(f"{ctx} must be an object")
        node_a = _require(pair, "node_a", ctx=ctx)
        node_b = _require(pair, "node_b", ctx=ctx)
        scope = _require(pair, "scope", ctx=ctx)
        if not isinstance(node_a, str) or not node_a:
            _die(f"{ctx}.node_a must be a non-empty string")
        if not isinstance(node_b, str) or not node_b:
            _die(f"{ctx}.node_b must be a non-empty string")
        if node_a >= node_b:
            _die(f"{ctx} node names must be ordered and distinct")
        if scope != "inter_node":
            _die(f"{ctx}.scope must be inter_node")
        pair_key = (node_a, node_b)
        if pair_key in seen_pairs:
            _die(f"duplicate node pair: {node_a}, {node_b}")
        seen_pairs.add(pair_key)

        _require_non_negative_integer(pair, "operations", ctx=ctx)
        a_to_b = _validate_pair_direction(
            _require(pair, "a_to_b", ctx=ctx),
            ctx=f"{ctx}.a_to_b",
        )
        b_to_a = _validate_pair_direction(
            _require(pair, "b_to_a", ctx=ctx),
            ctx=f"{ctx}.b_to_a",
        )
        total_bytes = _require_non_negative_integer(pair, "total_bytes", ctx=ctx)
        peak_combined = _require_non_negative_integer(
            pair,
            "peak_combined_bytes_per_operation",
            ctx=ctx,
        )
        estimated_time = _require_non_negative_number(
            pair,
            "estimated_time_us_total",
            ctx=ctx,
        )
        contention_penalty = _require_non_negative_number(
            pair,
            "contention_penalty_us_total",
            ctx=ctx,
        )
        _require_non_negative_number(
            pair,
            "average_estimated_throughput_gbps",
            ctx=ctx,
        )
        _require_non_negative_number(
            pair,
            "peak_estimated_throughput_gbps",
            ctx=ctx,
        )

        expected_total = int(a_to_b["total_bytes"]) + int(b_to_a["total_bytes"])
        if total_bytes != expected_total:
            _die(f"{ctx}.total_bytes must equal both directional totals")
        if peak_combined < max(
            int(a_to_b["peak_bytes_per_operation"]),
            int(b_to_a["peak_bytes_per_operation"]),
        ):
            _die(f"{ctx}.peak_combined_bytes_per_operation is inconsistent")
        expected_time = float(a_to_b["estimated_time_us_total"]) + float(
            b_to_a["estimated_time_us_total"]
        )
        if abs(estimated_time - expected_time) > 0.002:
            _die(f"{ctx}.estimated_time_us_total is inconsistent")
        expected_penalty = float(a_to_b["contention_penalty_us_total"]) + float(
            b_to_a["contention_penalty_us_total"]
        )
        if abs(contention_penalty - expected_penalty) > 0.002:
            _die(f"{ctx}.contention_penalty_us_total is inconsistent")

    if args.expect_markdown:
        markdown_value = _require(
            cluster,
            "markdown_report_path",
            ctx="cluster",
        )
        if not isinstance(markdown_value, str) or not markdown_value:
            _die("cluster.markdown_report_path must be a non-empty string")
        markdown_path = Path(markdown_value)
        if not markdown_path.is_absolute() and not markdown_path.is_file():
            markdown_path = path.parent / markdown_path
        if not markdown_path.is_file():
            _die(f"Markdown report file not found: {markdown_path}")
        markdown = markdown_path.read_text(encoding="utf-8")
        for required_text in (
            "# FakeGPU Cluster Communication Report",
            "## Node-Pair Communication",
            "| Node A | Node B |",
        ):
            if required_text not in markdown:
                _die(
                    f"Markdown report is missing required content: "
                    f"{required_text!r}"
                )

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
