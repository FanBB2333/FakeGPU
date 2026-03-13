#!/usr/bin/env python3

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE = REPO_ROOT / "build" / "fakegpu_config_probe"
FIXTURE_DIR = REPO_ROOT / "verification" / "data"


def run_case(filename: str, expected_error: str) -> None:
    env = os.environ.copy()
    env["FAKEGPU_DIST_MODE"] = "simulate"
    env["FAKEGPU_COORDINATOR_ADDR"] = "127.0.0.1:29591"
    env["FAKEGPU_CLUSTER_CONFIG"] = str(FIXTURE_DIR / filename)

    completed = subprocess.run(
        [str(PROBE)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode == 0:
        raise AssertionError(f"{filename}: expected failure, got success\n{completed.stdout}")
    if expected_error not in completed.stderr:
        raise AssertionError(
            f"{filename}: missing expected error {expected_error!r}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    print(f"[PASS] {filename}")


def main() -> int:
    if not PROBE.exists():
        print(f"missing probe binary: {PROBE}", file=sys.stderr)
        return 2

    run_case("cluster_invalid_duplicate_rank.yaml", "duplicate rank detected")
    run_case("cluster_invalid_gap_rank.yaml", "ranks must be contiguous")
    print("invalid cluster config checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
