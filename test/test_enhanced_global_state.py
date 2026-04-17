"""Build-and-run test for enhanced GlobalState report stats."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT / "build"
PROBE_BIN = BUILD_DIR / "test_enhanced_global_state"


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"command failed: {' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def main() -> None:
    compiler = os.environ.get("CXX", "c++")
    run(
        [
            compiler,
            "-std=c++17",
            "-I",
            str(ROOT / "src" / "core"),
            "-I",
            str(BUILD_DIR),
            str(ROOT / "src" / "core" / "global_state.cpp"),
            str(ROOT / "src" / "core" / "device.cpp"),
            str(ROOT / "src" / "core" / "gpu_profile.cpp"),
            str(ROOT / "verification" / "test_enhanced_global_state.cpp"),
            "-o",
            str(PROBE_BIN),
        ]
    )
    completed = run([str(PROBE_BIN)])
    assert "passed" in completed.stdout, completed.stdout
    print("enhanced GlobalState smoke passed")


if __name__ == "__main__":
    main()
