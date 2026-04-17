#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_BIN = REPO_ROOT / "build" / "fakegpu_collective_direct_test"
CHECKER = REPO_ROOT / "verification" / "check_cluster_report.py"
FAST_CONFIG = REPO_ROOT / "verification" / "data" / "cluster_valid.yaml"
SLOW_CONFIG = REPO_ROOT / "verification" / "data" / "cluster_slow_interconnect.yaml"


def run_report(config_path: Path, report_path: Path, raw_report_path: Path) -> dict:
    env = os.environ.copy()
    env.setdefault("XONSH_HISTORY_BACKEND", "dummy")
    env["FAKEGPU_CLUSTER_CONFIG"] = str(config_path)
    env["FAKEGPU_CLUSTER_REPORT_PATH"] = str(report_path)
    env["FAKEGPU_REPORT_PATH"] = str(raw_report_path)

    completed = subprocess.run(
        [str(PROBE_BIN), "--scenario", "allreduce"],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise AssertionError(f"collective probe failed for {config_path}")

    checked = subprocess.run(
        [
            "python3",
            str(CHECKER),
            "--path",
            str(report_path),
            "--expect-world-size",
            "4",
            "--expect-node-count",
            "2",
            "--expect-collective",
            "all_reduce",
            "--expect-links",
            "--min-ranks",
            "4",
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if checked.returncode != 0:
        sys.stderr.write(checked.stdout)
        sys.stderr.write(checked.stderr)
        raise AssertionError(f"cluster report check failed for {config_path}")

    return json.loads(report_path.read_text(encoding="utf-8"))


def sum_scope_time(report: dict, scope: str) -> float:
    return sum(
        float(link["estimated_time_us_total"])
        for link in report["links"]
        if link["scope"] == scope
    )


def sum_scope_penalty(report: dict, scope: str) -> float:
    return sum(
        float(link["contention_penalty_us_total"])
        for link in report["links"]
        if link["scope"] == scope
    )


def main() -> int:
    if not PROBE_BIN.exists():
        print(f"missing collective probe: {PROBE_BIN}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="fakegpu-time-model-") as tmpdir:
        tmp = Path(tmpdir)
        fast_report = tmp / "fast_cluster_report.json"
        fast_raw = tmp / "fast_fakegpu_report.json"
        slow_report = tmp / "slow_cluster_report.json"
        slow_raw = tmp / "slow_fakegpu_report.json"

        fast = run_report(FAST_CONFIG, fast_report, fast_raw)
        slow = run_report(SLOW_CONFIG, slow_report, slow_raw)

        fast_allreduce_time = float(fast["collectives"]["all_reduce"]["estimated_time_us_total"])
        slow_allreduce_time = float(slow["collectives"]["all_reduce"]["estimated_time_us_total"])
        if slow_allreduce_time <= fast_allreduce_time:
            raise AssertionError(
                "expected slow link allreduce estimate to exceed fast config: "
                f"fast={fast_allreduce_time}, slow={slow_allreduce_time}"
            )

        fast_inter_time = sum_scope_time(fast, "inter_node")
        slow_inter_time = sum_scope_time(slow, "inter_node")
        if slow_inter_time <= fast_inter_time:
            raise AssertionError(
                "expected slow inter-node link time to exceed fast config: "
                f"fast={fast_inter_time}, slow={slow_inter_time}"
            )

        fast_penalty = sum_scope_penalty(fast, "inter_node")
        slow_penalty = sum_scope_penalty(slow, "inter_node")
        if slow_penalty <= fast_penalty:
            raise AssertionError(
                "expected slow inter-node contention penalty to exceed fast config: "
                f"fast={fast_penalty}, slow={slow_penalty}"
            )

        print("time model test passed")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
