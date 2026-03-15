#!/usr/bin/env python3

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_BIN = REPO_ROOT / "build" / "fakegpu_collective_direct_test"
STAGING_LIMIT_BYTES = 65536
CHUNK_THRESHOLD_BYTES = 32768


def run_probe(*, chunk_bytes: int | None, expect_success: bool) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("FAKEGPU_STAGING_CHUNK_BYTES", None)
    env["FAKEGPU_STAGING_MAX_BYTES"] = str(STAGING_LIMIT_BYTES)
    if chunk_bytes is not None:
        env["FAKEGPU_STAGING_CHUNK_BYTES"] = str(chunk_bytes)

    completed = subprocess.run(
        [str(PROBE_BIN), "--scenario", "chunked"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    if expect_success:
        if completed.returncode != 0:
            raise AssertionError(
                "chunked probe was expected to pass\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        if "chunked scenario passed" not in completed.stdout:
            raise AssertionError(
                "missing success marker from chunked probe\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
    else:
        if completed.returncode == 0:
            raise AssertionError(
                "non-chunked probe was expected to fail because the staging cap is below the tensor size\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        error_output = completed.stderr + completed.stdout
        if "system error" not in error_output and "FAKEGPU_STAGING_MAX_BYTES" not in error_output:
            raise AssertionError(
                "missing expected failure marker for the non-chunked large-tensor path\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

    return completed


def main() -> int:
    if not PROBE_BIN.exists():
        print(f"missing collective direct probe: {PROBE_BIN}", file=sys.stderr)
        print(
            "build it first with: cmake --build build --target fakegpu_collective_direct_test",
            file=sys.stderr,
        )
        return 2

    run_probe(chunk_bytes=None, expect_success=False)
    run_probe(chunk_bytes=CHUNK_THRESHOLD_BYTES, expect_success=True)
    print("chunked collective test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
