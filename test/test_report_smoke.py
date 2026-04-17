"""Integration smoke test for enhanced native FakeGPU report output."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT / "build"


def run(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
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


def ensure_build() -> None:
    if sys.platform == "darwin":
        run(["cmake", "-S", ".", "-B", "build", "-DCMAKE_C_COMPILER=clang", "-DCMAKE_CXX_COMPILER=clang++"])
    else:
        run(["cmake", "-S", ".", "-B", "build"])
    run(["cmake", "--build", "build"])


def main() -> None:
    ensure_build()

    fd, report_path = tempfile.mkstemp(prefix="fakegpu-report-", suffix=".json")
    os.close(fd)
    os.unlink(report_path)

    env = dict(os.environ)
    env["FAKEGPU_REPORT_PATH"] = report_path
    env["FAKEGPU_TERMINAL_REPORT"] = "1"

    completed = run([sys.executable, "test/test_cuda_direct.py"], env=env)

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    assert report.get("report_version") == "1.5.0", report

    devices = report.get("devices")
    assert isinstance(devices, list) and devices, report
    dev0 = devices[0]

    gpu_profile = dev0.get("gpu_profile")
    assert isinstance(gpu_profile, dict), dev0
    assert gpu_profile.get("architecture"), gpu_profile
    assert gpu_profile.get("compute_capability"), gpu_profile
    assert isinstance(gpu_profile.get("supported_types"), list), gpu_profile

    kernel_launches = dev0.get("kernel_launches")
    assert isinstance(kernel_launches, dict), dev0
    assert "total" in kernel_launches, kernel_launches

    gemm_by_dtype = dev0.get("gemm_by_dtype")
    assert isinstance(gemm_by_dtype, dict), dev0

    stderr = completed.stderr
    assert "FakeGPU Report Summary" in stderr, stderr
    assert "Device 0:" in stderr, stderr

    print("enhanced report smoke passed")


if __name__ == "__main__":
    main()
