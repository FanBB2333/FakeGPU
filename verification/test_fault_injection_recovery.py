#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate deterministic NCCL rank-failure injection, async-error "
            "propagation, communicator shrink, and post-recovery All-Reduce."
        )
    )
    parser.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build")
    args = parser.parse_args()

    probe = args.build_dir.resolve() / "fakegpu_nccl_direct_test"
    if not probe.is_file():
        print(f"missing NCCL direct test binary: {probe}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="fakegpu-fault-shrink-") as raw:
        temp_dir = Path(raw)
        report_path = temp_dir / "cluster-report.json"
        markdown_path = temp_dir / "cluster-report.md"
        env = dict(os.environ)
        env.pop("LD_PRELOAD", None)
        env.pop("DYLD_INSERT_LIBRARIES", None)
        env.update(
            {
                "FAKEGPU_MODE": "simulate",
                "FAKEGPU_DIST_MODE": "simulate",
                "FAKEGPU_NCCL_FAULT_RANK": "2",
                "FAKEGPU_NCCL_FAULT_SEQNO": "1",
                "FAKEGPU_NCCL_FAULT_OPERATION": "all_reduce",
                "FAKEGPU_COORDINATOR_TIMEOUT_MS": "2000",
                "FAKEGPU_CLUSTER_REPORT_PATH": str(report_path),
                "FAKEGPU_CLUSTER_REPORT_MARKDOWN_PATH": str(markdown_path),
            }
        )
        completed = subprocess.run(
            [str(probe), "--scenario", "fault-shrink"],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if completed.returncode != 0:
            raise AssertionError(
                f"fault-shrink probe exited with {completed.returncode}\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )

        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["schema_version"] == "cluster_report.v1"
        assert report["collectives"]["all_reduce"]["calls"] == 1

        resilience = report["resilience"]
        assert resilience["failure_count"] == 1
        assert resilience["recovery_count"] == 1

        failure = resilience["failure_events"][0]
        assert failure["global_rank"] == 2
        assert failure["operation"] == "all_reduce"
        assert failure["error_code"] == "injected_rank_failure"
        assert failure["source"] == "injected"
        assert 2 in failure["observed_ranks"]
        assert failure["attempted_payload_bytes"] >= 4

        recovery = resilience["recovery_events"][0]
        assert recovery["abort_parent"] is True
        assert recovery["excluded_ranks"] == [2]
        assert recovery["surviving_ranks"] == [0, 1, 3]
        assert recovery["recovery_time_us"] >= 0

        entries = report["operation_timeline"]["entries"]
        assert len(entries) == 1
        assert entries[0]["operation"] == "allreduce"
        assert entries[0]["ranks"] == [0, 1, 3]

        checker = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "verification" / "check_cluster_report.py"),
                "--path",
                str(report_path),
                "--expect-world-size",
                "4",
                "--expect-collective",
                "all_reduce",
                "--expect-failure",
                "--expect-recovery",
                "--expect-markdown",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if checker.returncode != 0:
            raise AssertionError(
                f"cluster report checker exited with {checker.returncode}\n"
                f"stdout:\n{checker.stdout}\nstderr:\n{checker.stderr}"
            )

        markdown = markdown_path.read_text(encoding="utf-8")
        assert "## Resilience Events" in markdown
        assert "injected_rank_failure" in markdown
        assert "`0,1,3`" in markdown

        print(
            json.dumps(
                {
                    "status": "success",
                    "failure_rank": failure["global_rank"],
                    "failure_seqno": failure["seqno"],
                    "surviving_ranks": recovery["surviving_ranks"],
                    "recovery_time_us": recovery["recovery_time_us"],
                    "post_recovery_all_reduce_calls": report["collectives"][
                        "all_reduce"
                    ]["calls"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
