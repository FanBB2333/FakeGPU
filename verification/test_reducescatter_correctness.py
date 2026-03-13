#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_BIN = REPO_ROOT / "build" / "fakegpu_collective_direct_test"


def main() -> int:
    if not PROBE_BIN.exists():
        print(f"missing collective probe: {PROBE_BIN}", file=sys.stderr)
        return 2

    for _ in range(3):
        completed = subprocess.run(
            [str(PROBE_BIN), "--scenario", "reducescatter"],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            sys.stderr.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            raise AssertionError("reducescatter correctness scenario failed")

    print("reducescatter correctness test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
