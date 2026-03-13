#!/usr/bin/env python3

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = REPO_ROOT / "build"
PROBE = BUILD_DIR / "fakegpu_config_probe"


def run_case(name: str, env_updates: dict[str, str], expected_rc: int, expect_text: str, stream: str = "stdout") -> None:
    env = os.environ.copy()
    env.pop("FAKEGPU_DIST_MODE", None)
    env.pop("FAKEGPU_CLUSTER_CONFIG", None)
    env.pop("FAKEGPU_COORDINATOR_ADDR", None)
    env.pop("FAKEGPU_COORDINATOR_TRANSPORT", None)
    env.update(env_updates)

    completed = subprocess.run(
        [str(PROBE)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    output = completed.stdout if stream == "stdout" else completed.stderr
    if completed.returncode != expected_rc:
        raise AssertionError(
            f"{name}: expected rc={expected_rc}, got rc={completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    if expect_text not in output:
        raise AssertionError(
            f"{name}: expected to find {expect_text!r} in {stream}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    print(f"[PASS] {name}")


def main() -> int:
    if not PROBE.exists():
        print(f"missing probe binary: {PROBE}", file=sys.stderr)
        print("build it first with: cmake --build build --target fakegpu_config_probe", file=sys.stderr)
        return 2

    run_case(
        "disabled_defaults",
        {},
        expected_rc=0,
        expect_text="dist_mode=disabled",
    )
    run_case(
        "simulate_valid",
        {
            "FAKEGPU_DIST_MODE": "simulate",
            "FAKEGPU_COORDINATOR_ADDR": "127.0.0.1:29591",
        },
        expected_rc=0,
        expect_text="coordinator_addr=127.0.0.1:29591",
    )
    run_case(
        "invalid_mode",
        {
            "FAKEGPU_DIST_MODE": "broken",
            "FAKEGPU_COORDINATOR_ADDR": "127.0.0.1:29591",
        },
        expected_rc=2,
        expect_text="Invalid FAKEGPU_DIST_MODE",
        stream="stderr",
    )
    run_case(
        "missing_addr",
        {
            "FAKEGPU_DIST_MODE": "simulate",
        },
        expected_rc=2,
        expect_text="FAKEGPU_COORDINATOR_ADDR must be set",
        stream="stderr",
    )
    print("all cluster config checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
