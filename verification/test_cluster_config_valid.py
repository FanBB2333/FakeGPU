#!/usr/bin/env python3

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE = REPO_ROOT / "build" / "fakegpu_config_probe"
VALID_CLUSTER = REPO_ROOT / "verification" / "data" / "cluster_valid.yaml"


def main() -> int:
    if not PROBE.exists():
        print(f"missing probe binary: {PROBE}", file=sys.stderr)
        return 2

    env = os.environ.copy()
    env["FAKEGPU_DIST_MODE"] = "simulate"
    env["FAKEGPU_COORDINATOR_ADDR"] = "127.0.0.1:29591"
    env["FAKEGPU_CLUSTER_CONFIG"] = str(VALID_CLUSTER)

    completed = subprocess.run(
        [str(PROBE)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        print(completed.stdout, file=sys.stderr, end="")
        print(completed.stderr, file=sys.stderr, end="")
        return completed.returncode

    expected_tokens = [
        "dist_mode=simulate",
        f"cluster_config={VALID_CLUSTER}",
        "cluster_name=dev-cluster",
        "world_size=4",
        "node_count=2",
    ]
    for token in expected_tokens:
        if token not in completed.stdout:
            print(f"missing expected token: {token}", file=sys.stderr)
            print(completed.stdout, file=sys.stderr, end="")
            return 1

    print("valid cluster config parsed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
