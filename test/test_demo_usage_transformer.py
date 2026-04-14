"""Regression test for the demo_usage transformer training option."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    demo_script = project_root / "demo_usage.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(demo_script),
            "--test",
            "transformer",
            "--quiet",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        raise AssertionError(
            "demo_usage transformer option failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    output = completed.stdout + completed.stderr
    assert "TRANSFORMER" in output.upper(), output
    assert "training simulation finished" in output.lower(), output


if __name__ == "__main__":
    main()
