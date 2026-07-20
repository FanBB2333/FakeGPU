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
  name: four-node-report-test
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


def main() -> int:
    build_dir = Path(
        os.environ.get("BUILD_DIR", REPO_ROOT / "build")
    ).resolve()
    probe = build_dir / "fakegpu_collective_direct_test"
    if not probe.is_file():
        print(f"missing collective probe: {probe}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(
        prefix="fakegpu-node-pair-report-"
    ) as temp_dir_text:
        temp_dir = Path(temp_dir_text)
        config_path = temp_dir / "cluster.yaml"
        json_path = temp_dir / "cluster.json"
        markdown_path = temp_dir / "project-communication.md"
        raw_report_path = temp_dir / "fakegpu.json"
        _write_four_node_config(config_path)

        env = dict(os.environ)
        env.update(
            {
                "FAKEGPU_CLUSTER_CONFIG": str(config_path),
                "FAKEGPU_CLUSTER_REPORT_PATH": str(json_path),
                "FAKEGPU_CLUSTER_REPORT_MARKDOWN_PATH": str(markdown_path),
                "FAKEGPU_REPORT_PATH": str(raw_report_path),
            }
        )
        completed = subprocess.run(
            [str(probe), "--scenario", "allreduce"],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
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
        assert report["cluster"]["markdown_report_path"] == str(markdown_path)
        pairs = {
            (pair["node_a"], pair["node_b"]): pair
            for pair in report["node_pairs"]
        }
        assert len(pairs) == 6

        active_pairs = {
            ("node0", "node1"),
            ("node0", "node3"),
            ("node1", "node2"),
            ("node2", "node3"),
        }
        inactive_pairs = {
            ("node0", "node2"),
            ("node1", "node3"),
        }
        for key in active_pairs:
            pair = pairs[key]
            assert pair["operations"] > 0
            assert pair["collective_operations"] == pair["operations"]
            assert pair["point_to_point_operations"] == 0
            assert pair["total_bytes"] > 0
            assert pair["peak_combined_bytes_per_operation"] > 0
            assert pair["average_estimated_throughput_gbps"] > 0
            assert pair["peak_estimated_throughput_gbps"] > 0
            assert pair["total_bytes"] == (
                pair["a_to_b"]["total_bytes"]
                + pair["b_to_a"]["total_bytes"]
            )

        for key in inactive_pairs:
            pair = pairs[key]
            assert pair["operations"] == 0
            assert pair["collective_operations"] == 0
            assert pair["point_to_point_operations"] == 0
            assert pair["total_bytes"] == 0
            assert pair["peak_combined_bytes_per_operation"] == 0
            assert pair["a_to_b"]["model_bandwidth_gbps"] == 25
            assert pair["b_to_a"]["model_bandwidth_gbps"] == 25

        markdown = markdown_path.read_text(encoding="utf-8")
        assert "## Node-Pair Communication" in markdown
        for node_a, node_b in pairs:
            assert f"| `{node_a}` | `{node_b}` |" in markdown

        print("complete node-pair JSON and Markdown report validation passed")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
