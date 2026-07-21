#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_case(
    *,
    build_dir: Path,
    root: Path,
    name: str,
    limit: int,
    iterations: int,
) -> dict[str, Any]:
    case_dir = root / name
    report_path = case_dir / "cluster-report.json"
    benchmark_path = case_dir / "bandwidth.json"
    endpoint = f"127.0.0.1:{_free_port()}"
    env = dict(os.environ)
    env["FAKEGPU_CLUSTER_REPORT_MAX_OPERATIONS"] = str(limit)
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
            "4",
            "--warmup",
            "0",
            "--iterations",
            str(iterations),
            "--timeout",
            "5",
            "--build-dir",
            str(build_dir),
            "--cluster-report",
            str(report_path),
            "--json",
            str(benchmark_path),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=max(30.0, iterations * 0.25),
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"retention case {name} exited with {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    timeline = report["operation_timeline"]
    expected_retained = min(limit, iterations) if limit > 0 else 0
    expected_dropped = iterations - expected_retained
    assert report["schema_version"] == "cluster_report.v1"
    assert report["collectives"]["all_reduce"]["calls"] == iterations
    assert timeline["retained_entries"] == expected_retained
    assert timeline["dropped_entries"] == expected_dropped
    assert len(timeline["entries"]) == expected_retained
    if expected_retained:
        expected_indices = list(
            range(iterations - expected_retained + 1, iterations + 1)
        )
        assert [entry["index"] for entry in timeline["entries"]] == expected_indices
        assert all(entry["operation"] == "allreduce" for entry in timeline["entries"])
    maximum_report_bytes = 200_000 + expected_retained * 4_096
    assert report_path.stat().st_size < maximum_report_bytes
    return {
        "name": name,
        "iterations": iterations,
        "limit": limit,
        "retained_entries": timeline["retained_entries"],
        "dropped_entries": timeline["dropped_entries"],
        "report_bytes": report_path.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate bounded cluster-report operation retention."
    )
    parser.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build")
    parser.add_argument(
        "--stress-iterations",
        type=int,
        default=256,
        help="Number of small TCP collectives used by the bounded stress case.",
    )
    args = parser.parse_args()
    if args.stress_iterations < 65:
        parser.error("--stress-iterations must be at least 65")

    build_dir = args.build_dir.resolve()
    if not (build_dir / "fakegpu-coordinator").is_file():
        print(
            f"missing coordinator binary: {build_dir / 'fakegpu-coordinator'}",
            file=sys.stderr,
        )
        return 2

    with tempfile.TemporaryDirectory(
        prefix="fakegpu-cluster-retention-"
    ) as raw:
        root = Path(raw)
        results = [
            _run_case(
                build_dir=build_dir,
                root=root,
                name="disabled",
                limit=0,
                iterations=4,
            ),
            _run_case(
                build_dir=build_dir,
                root=root,
                name="rolling-window",
                limit=3,
                iterations=8,
            ),
            _run_case(
                build_dir=build_dir,
                root=root,
                name="stress",
                limit=64,
                iterations=args.stress_iterations,
            ),
        ]
    print(
        json.dumps(
            {
                "schema_version": "fakegpu.cluster_retention_validation.v1",
                "status": "success",
                "cases": results,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
