"""Build-and-run test for hardware compatibility enforcement in cuBLAS stubs."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT / "build"
PROBE_BIN = BUILD_DIR / "test_hw_compat_cublas"


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
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


def build_probe() -> None:
    compiler = os.environ.get("CXX", "c++")
    run(
        [
            compiler,
            "-std=c++17",
            "-I",
            str(ROOT / "src" / "core"),
            "-I",
            str(ROOT / "src" / "cublas"),
            "-I",
            str(BUILD_DIR),
            str(ROOT / "src" / "core" / "global_state.cpp"),
            str(ROOT / "src" / "core" / "device.cpp"),
            str(ROOT / "src" / "core" / "gpu_profile.cpp"),
            str(ROOT / "src" / "distributed" / "cluster_config.cpp"),
            str(ROOT / "src" / "cublas" / "cublas_stubs.cpp"),
            str(ROOT / "verification" / "test_hw_compat_cublas.cpp"),
            "-o",
            str(PROBE_BIN),
        ]
    )


def main() -> None:
    build_probe()

    strict_env = dict(os.environ)
    strict_env["FAKEGPU_STRICT_COMPAT"] = "1"
    strict_run = run([str(PROBE_BIN)], env=strict_env)
    assert "passed" in strict_run.stdout, strict_run.stdout

    relaxed_env = dict(os.environ)
    relaxed_env["FAKEGPU_STRICT_COMPAT"] = "0"
    relaxed_run = run([str(PROBE_BIN)], env=relaxed_env)
    assert "passed" in relaxed_run.stdout, relaxed_run.stdout

    print("hardware compatibility cuBLAS smoke passed")


if __name__ == "__main__":
    main()
