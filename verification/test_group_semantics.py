#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE = REPO_ROOT / "build" / "fakegpu_group_semantics_test"


def main() -> int:
    if not PROBE.exists():
        print(f"missing group semantics probe: {PROBE}", file=sys.stderr)
        return 2

    for transport in ("unix", "tcp"):
        completed = subprocess.run(
            [str(PROBE), "--transport", transport],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
        )
        if completed.stdout:
            sys.stdout.write(completed.stdout)
        if completed.stderr:
            sys.stderr.write(completed.stderr)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
