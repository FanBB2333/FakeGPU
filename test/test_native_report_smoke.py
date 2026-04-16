"""Regression test for native FakeGPU report generation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    fd, report_path = tempfile.mkstemp(prefix="fakegpu-report-", suffix=".json")
    os.close(fd)
    os.unlink(report_path)

    env = dict(os.environ)
    env["FAKEGPU_REPORT_PATH"] = report_path

    completed = subprocess.run(
        [sys.executable, "test/test_cuda_direct.py"],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"native report smoke command failed with rc={completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    devices = report.get("devices")
    assert isinstance(devices, list) and devices, report
    assert any(int(dev.get("used_memory_peak", 0)) > 0 for dev in devices), report

    print("native report smoke passed")


if __name__ == "__main__":
    main()
