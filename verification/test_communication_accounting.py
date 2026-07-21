#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER = REPO_ROOT / "verification" / "check_cluster_report.py"


def _write_four_node_config(path: Path) -> None:
    path.write_text(
        """\
version: 1
cluster:
  name: communication-accounting-test
  default_backend: nccl

nodes:
  - id: node0
    host: host0
    ranks: [0]
    gpus:
      - profile: a100
  - id: node1
    host: host1
    ranks: [1]
    gpus:
      - profile: a100
  - id: node2
    host: host2
    ranks: [2]
    gpus:
      - profile: a100
  - id: node3
    host: host3
    ranks: [3]
    gpus:
      - profile: a100

fabric:
  intra_node:
    type: pcie
    bandwidth_gbps: 64
    latency_us: 4
  inter_node:
    type: tcp
    bandwidth_gbps: 25
    latency_us: 50
    oversubscription: 1.25
""",
        encoding="utf-8",
    )


def _require_pair(
    pairs: dict[tuple[str, str], dict],
    key: tuple[str, str],
    *,
    collective_operations: int,
    point_to_point_operations: int,
) -> dict:
    pair = pairs[key]
    assert pair["collective_operations"] == collective_operations
    assert pair["point_to_point_operations"] == point_to_point_operations
    assert pair["operations"] == (
        collective_operations + point_to_point_operations
    )
    return pair


def main() -> int:
    build_dir = Path(
        os.environ.get("BUILD_DIR", REPO_ROOT / "build")
    ).resolve()
    probe = build_dir / "fakegpu_nccl_direct_test"
    if not probe.is_file():
        print(f"missing NCCL direct probe: {probe}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(
        prefix="fakegpu-communication-accounting-"
    ) as temp_dir_text:
        temp_dir = Path(temp_dir_text)
        config_path = temp_dir / "cluster.yaml"
        json_path = temp_dir / "cluster.json"
        markdown_path = temp_dir / "cluster.md"
        _write_four_node_config(config_path)

        env = dict(os.environ)
        env.update(
            {
                "FAKEGPU_CLUSTER_CONFIG": str(config_path),
                "FAKEGPU_CLUSTER_REPORT_PATH": str(json_path),
                "FAKEGPU_CLUSTER_REPORT_MARKDOWN_PATH": str(markdown_path),
            }
        )
        try:
            completed = subprocess.run(
                [str(probe), "--scenario", "communication-report"],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            print(
                "NCCL communication report probe exceeded 60 seconds; "
                "rebuild the probe and shared libraries together",
                file=sys.stderr,
            )
            return 124
        if completed.returncode != 0:
            print(completed.stdout, end="", file=sys.stderr)
            print(completed.stderr, end="", file=sys.stderr)
            return completed.returncode

        checked = subprocess.run(
            [
                sys.executable,
                str(CHECKER),
                "--path",
                str(json_path),
                "--expect-world-size",
                "4",
                "--expect-node-count",
                "4",
                "--expect-collective",
                "all_reduce",
                "--expect-point-to-point",
                "--expect-links",
                "--expect-markdown",
                "--min-ranks",
                "4",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if checked.returncode != 0:
            print(checked.stdout, end="", file=sys.stderr)
            print(checked.stderr, end="", file=sys.stderr)
            return checked.returncode

        report = json.loads(json_path.read_text(encoding="utf-8"))
        assert report["schema_version"] == "cluster_report.v1"
        assert report["point_to_point"]["operations"] == 1
        assert report["point_to_point"]["sends"] == 2
        assert report["point_to_point"]["bytes"] == 24

        pairs = {
            (pair["node_a"], pair["node_b"]): pair
            for pair in report["node_pairs"]
        }
        assert len(pairs) == 6

        subgroup_pair = _require_pair(
            pairs,
            ("node0", "node2"),
            collective_operations=1,
            point_to_point_operations=0,
        )
        assert subgroup_pair["a_to_b"]["total_bytes"] == 8
        assert subgroup_pair["b_to_a"]["total_bytes"] == 8

        forward_pair = _require_pair(
            pairs,
            ("node0", "node1"),
            collective_operations=0,
            point_to_point_operations=1,
        )
        assert forward_pair["a_to_b"]["total_bytes"] == 12
        assert forward_pair["b_to_a"]["total_bytes"] == 0

        reverse_pair = _require_pair(
            pairs,
            ("node2", "node3"),
            collective_operations=0,
            point_to_point_operations=1,
        )
        assert reverse_pair["a_to_b"]["total_bytes"] == 0
        assert reverse_pair["b_to_a"]["total_bytes"] == 12

        for key in (
            ("node0", "node3"),
            ("node1", "node2"),
            ("node1", "node3"),
        ):
            pair = _require_pair(
                pairs,
                key,
                collective_operations=0,
                point_to_point_operations=0,
            )
            assert pair["total_bytes"] == 0

        ranks = {item["rank"]: item for item in report["ranks"]}
        assert set(ranks) == {0, 1, 2, 3}
        assert [ranks[rank]["point_to_point_calls"] for rank in range(4)] == [
            1,
            1,
            1,
            1,
        ]
        assert [ranks[rank]["collective_calls"] for rank in range(4)] == [
            1,
            1,
            1,
            0,
        ]

        timeline = report["operation_timeline"]
        assert timeline["retained_entries"] == 3
        assert timeline["dropped_entries"] == 0
        entries = timeline["entries"]
        subgroup_entry = next(
            entry for entry in entries
            if entry["kind"] == "collective"
            and entry["ranks"] == [2, 0]
        )
        assert subgroup_entry["operation"] == "allreduce"
        assert subgroup_entry["data_type"] == "float32"
        assert subgroup_entry["reduce_op"] == "sum"
        assert subgroup_entry["logical_payload_bytes"] == 8
        p2p_entry = next(
            entry for entry in entries
            if entry["kind"] == "point_to_point"
        )
        assert p2p_entry["ranks"] == [0, 1, 2, 3]
        assert p2p_entry["data_type"] == "int32"
        assert p2p_entry["reduce_op"] == "none"
        assert p2p_entry["logical_payload_bytes"] == 24
        assert p2p_entry["socket_request_payload_bytes"] == 0
        assert p2p_entry["socket_response_payload_bytes"] == 0
        for entry in entries:
            assert entry["coordinator_duration_us"] >= 0
            assert entry["modeled_time_us"] >= 0

        markdown = markdown_path.read_text(encoding="utf-8")
        assert "## Point-to-Point Summary" in markdown
        assert "## Recent Operation Timeline" in markdown
        assert "| Data type | Reduce op |" in markdown
        assert "| 1 | 2 | 24 B |" in markdown

        print(
            "subgroup/global-rank and point-to-point accounting validation "
            "passed"
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
