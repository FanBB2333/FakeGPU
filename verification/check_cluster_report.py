#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _die(msg: str) -> None:
    print(f"[check_cluster_report] ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _require(obj: dict, key: str, *, ctx: str) -> object:
    if key not in obj:
        _die(f"missing key '{key}' in {ctx}")
    return obj[key]


def _validate_schema_file(report: dict[str, Any], schema_path: Path) -> None:
    if not schema_path.is_file():
        _die(f"schema file not found: {schema_path}")
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except Exception as exc:
        _die(f"failed to read schema {schema_path}: {exc}")
    if not isinstance(schema, dict):
        _die(f"schema root must be an object: {schema_path}")
    _validate_against_schema(report, schema, "$", schema)


def _resolve_local_ref(ref: str, root_schema: dict[str, Any]) -> dict[str, Any]:
    if not ref.startswith("#/"):
        _die(f"unsupported non-local schema reference: {ref}")
    current: Any = root_schema
    for component in ref[2:].split("/"):
        key = component.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or key not in current:
            _die(f"invalid schema reference: {ref}")
        current = current[key]
    if not isinstance(current, dict):
        _die(f"schema reference does not resolve to an object: {ref}")
    return current


def _validate_against_schema(
    value: Any,
    schema: dict[str, Any],
    path: str,
    root_schema: dict[str, Any],
) -> None:
    ref = schema.get("$ref")
    if isinstance(ref, str):
        _validate_against_schema(
            value,
            _resolve_local_ref(ref, root_schema),
            path,
            root_schema,
        )
        return

    if "type" in schema:
        _validate_type(value, schema["type"], path)
    if "enum" in schema and value not in schema["enum"]:
        _die(f"{path} must be one of {schema['enum']!r}, got {value!r}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            _die(f"{path} must be >= {schema['minimum']!r}")

    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                _die(f"{path}.{key} is missing")
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for key, child_schema in properties.items():
                if key in value and isinstance(child_schema, dict):
                    _validate_against_schema(
                        value[key],
                        child_schema,
                        f"{path}.{key}",
                        root_schema,
                    )

    if isinstance(value, list):
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            _die(f"{path} must contain at least {min_items} item(s)")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate_against_schema(
                    item,
                    item_schema,
                    f"{path}[{index}]",
                    root_schema,
                )


def _validate_type(value: Any, expected: Any, path: str) -> None:
    expected_types = expected if isinstance(expected, list) else [expected]
    if any(_matches_json_type(value, item) for item in expected_types):
        return
    _die(f"{path} must be {expected!r}, got {type(value).__name__}")


def _matches_json_type(value: Any, expected: Any) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


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
    ap.add_argument(
        "--schema",
        default=str(
            Path(__file__).resolve().parent.parent
            / "cluster_report.schema.json"
        ),
        help="Path to cluster_report.schema.json",
    )
    ap.add_argument(
        "--no-schema",
        action="store_true",
        help="Skip JSON Schema validation",
    )
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
        "--expect-point-to-point",
        action="store_true",
        help="Require successful point-to-point sends",
    )
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
    if not args.no_schema:
        _validate_schema_file(report, Path(args.schema))

    version = report.get("report_version")
    if not isinstance(version, str):
        _die(f"unexpected report_version={version!r} (expected a version string)")

    schema_version = _require(report, "schema_version", ctx="root")
    if schema_version != "cluster_report.v1":
        _die(
            f"unexpected schema_version={schema_version!r} "
            "(expected 'cluster_report.v1')"
        )
    legacy_schema = _require(report, "schema", ctx="root")
    if legacy_schema != "experimental":
        _die(
            f"unexpected legacy schema={legacy_schema!r} "
            "(expected 'experimental')"
        )

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
    for expected_name in args.expect_collective:
        expected = collectives[expected_name]
        if int(expected["calls"]) <= 0:
            _die(f"expected collectives.{expected_name}.calls > 0")

    point_to_point = _require(report, "point_to_point", ctx="root")
    if not isinstance(point_to_point, dict):
        _die("point_to_point must be an object")
    p2p_operations = _require_non_negative_integer(
        point_to_point,
        "operations",
        ctx="point_to_point",
    )
    p2p_sends = _require_non_negative_integer(
        point_to_point,
        "sends",
        ctx="point_to_point",
    )
    _require_non_negative_integer(
        point_to_point,
        "bytes",
        ctx="point_to_point",
    )
    _require_non_negative_number(
        point_to_point,
        "estimated_time_us_total",
        ctx="point_to_point",
    )
    _require_non_negative_number(
        point_to_point,
        "contention_penalty_us_total",
        ctx="point_to_point",
    )
    if non_zero_collectives == 0 and p2p_operations == 0:
        _die("expected at least one completed communication operation")
    if args.expect_point_to_point and (p2p_operations <= 0 or p2p_sends <= 0):
        _die("expected successful point-to-point sends")

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
            operations = _require(link, "operations", ctx=ctx)
            collective_operations = _require(
                link,
                "collective_operations",
                ctx=ctx,
            )
            point_to_point_operations = _require(
                link,
                "point_to_point_operations",
                ctx=ctx,
            )
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
            for field_name, field_value in (
                ("operations", operations),
                ("collective_operations", collective_operations),
                ("point_to_point_operations", point_to_point_operations),
            ):
                if not isinstance(field_value, int) or field_value < 0:
                    _die(f"{ctx}.{field_name} must be a non-negative integer")
            if operations != collective_operations + point_to_point_operations:
                _die(f"{ctx}.operations must equal its operation breakdown")
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

        operations = _require_non_negative_integer(pair, "operations", ctx=ctx)
        collective_operations = _require_non_negative_integer(
            pair,
            "collective_operations",
            ctx=ctx,
        )
        point_to_point_operations = _require_non_negative_integer(
            pair,
            "point_to_point_operations",
            ctx=ctx,
        )
        if operations != collective_operations + point_to_point_operations:
            _die(f"{ctx}.operations must equal its operation breakdown")
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
            "## Point-to-Point Summary",
            "## Node-Pair Communication",
            "## Recent Operation Timeline",
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
        point_to_point_calls = _require(
            rank_stats,
            "point_to_point_calls",
            ctx=ctx,
        )
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
            ("point_to_point_calls", point_to_point_calls),
            ("barrier_calls", barrier_calls),
            ("group_prepares", group_prepares),
        ):
            if not isinstance(field_value, int) or field_value < 0:
                _die(f"{ctx}.{field_name} must be a non-negative integer")

    timeline = _require(report, "operation_timeline", ctx="root")
    if not isinstance(timeline, dict):
        _die("operation_timeline must be an object")
    retained_entries = _require_non_negative_integer(
        timeline,
        "retained_entries",
        ctx="operation_timeline",
    )
    _require_non_negative_integer(
        timeline,
        "dropped_entries",
        ctx="operation_timeline",
    )
    entries = _require(timeline, "entries", ctx="operation_timeline")
    if not isinstance(entries, list):
        _die("operation_timeline.entries must be an array")
    if retained_entries != len(entries):
        _die(
            "operation_timeline.retained_entries must equal the entries "
            "array length"
        )

    previous_index = 0
    for index, entry in enumerate(entries):
        ctx = f"operation_timeline.entries[{index}]"
        if not isinstance(entry, dict):
            _die(f"{ctx} must be an object")
        entry_index = _require_non_negative_integer(entry, "index", ctx=ctx)
        if entry_index <= previous_index:
            _die(f"{ctx}.index must be strictly increasing")
        previous_index = entry_index
        ranks_value = _require(entry, "ranks", ctx=ctx)
        if not isinstance(ranks_value, list) or not ranks_value:
            _die(f"{ctx}.ranks must be a non-empty array")
        if any(
            not isinstance(rank, int) or rank < 0
            for rank in ranks_value
        ):
            _die(f"{ctx}.ranks must contain non-negative integers")

        rendezvous = _require_non_negative_number(
            entry,
            "rendezvous_wait_us",
            ctx=ctx,
        )
        execution = _require_non_negative_number(
            entry,
            "execution_time_us",
            ctx=ctx,
        )
        coordinator_duration = _require_non_negative_number(
            entry,
            "coordinator_duration_us",
            ctx=ctx,
        )
        if abs(coordinator_duration - rendezvous - execution) > 0.003:
            _die(f"{ctx}.coordinator_duration_us is inconsistent")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
