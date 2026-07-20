#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate local logical multi-node traffic over TCP payloads."
    )
    parser.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build")
    args = parser.parse_args()

    build_dir = args.build_dir.resolve()
    coordinator = build_dir / "fakegpu-coordinator"
    if not coordinator.is_file():
        print(f"missing coordinator binary: {coordinator}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(
        prefix="fakegpu-tcp-bandwidth-validation-"
    ) as temp_dir_text:
        temp_dir = Path(temp_dir_text)
        report_path = temp_dir / "bandwidth.json"
        cluster_report_path = temp_dir / "cluster.json"
        endpoint = f"127.0.0.1:{find_free_port()}"
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "fakegpu",
                "bandwidth",
                "--listen",
                endpoint,
                "--nodes",
                "2",
                "--ranks-per-node",
                "1",
                "--size",
                "64KiB",
                "--warmup",
                "1",
                "--iterations",
                "3",
                "--timeout",
                "10",
                "--build-dir",
                str(build_dir),
                "--json",
                str(report_path),
                "--cluster-report",
                str(cluster_report_path),
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            print(completed.stdout, end="", file=sys.stderr)
            print(completed.stderr, end="", file=sys.stderr)
            return completed.returncode

        report = json.loads(report_path.read_text(encoding="utf-8"))
        cluster_report = json.loads(
            cluster_report_path.read_text(encoding="utf-8")
        )

        assert report["status"] == "success"
        assert report["transport"] == "tcp_socket_payload"
        assert report["world_size"] == 2
        assert report["logical_node_count"] == 2
        assert report["complete_world_measured_locally"] is True
        assert report["algorithmic_bandwidth_gbps"] > 0
        assert report["aggregate_socket_payload_throughput_gbps"] > 0
        assert len(report["rank_reports"]) == 2
        assert all(
            rank_report["sample_values"] == [3.0, 3.0, 3.0]
            for rank_report in report["rank_reports"]
        )

        assert cluster_report["cluster"]["coordinator_transport"] == "tcp"
        assert cluster_report["cluster"]["world_size"] == 2
        assert cluster_report["cluster"]["node_count"] == 2
        assert cluster_report["collectives"]["all_reduce"]["calls"] == 4
        assert any(
            link["scope"] == "inter_node"
            for link in cluster_report["links"]
        )

        print(completed.stdout, end="")
        print("TCP multi-node bandwidth validation passed")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
