#!/usr/bin/env python3

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COLLECTIVE_BIN = REPO_ROOT / "build" / "fakegpu_collective_direct_test"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run_checked(command: list[str], env: dict[str, str]) -> None:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def main() -> int:
    require(COLLECTIVE_BIN.is_file(), f"missing probe binary: {COLLECTIVE_BIN}")
    with tempfile.TemporaryDirectory(prefix="fakegpu-socket-fallback-") as tmpdir:
        env = dict(os.environ)
        env.update(
            {
                "FAKEGPU_STAGING_FORCE_SOCKET": "1",
                "FAKEGPU_CLUSTER_REPORT_PATH": str(Path(tmpdir) / "cluster_report.json"),
            }
        )

        run_checked([str(COLLECTIVE_BIN), "--scenario", "allreduce"], env)
        run_checked([str(COLLECTIVE_BIN), "--scenario", "allgather"], env)
        run_checked([str(COLLECTIVE_BIN), "--scenario", "alltoall"], env)

    print("socket staging fallback test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
