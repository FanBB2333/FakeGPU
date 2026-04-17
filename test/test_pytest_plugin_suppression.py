"""Regression checks for noisy third-party pytest plugin auto-loading."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "test" / "run_error_simulation_suite.py"

spec = importlib.util.spec_from_file_location("error_sim_runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(runner)


def test_pytest_subprocess_does_not_emit_xonsh_history_errors(tmp_path: Path) -> None:
    test_file = tmp_path / "test_dummy_pytest.py"
    test_file.write_text("def test_dummy():\n    assert True\n", encoding="utf-8")

    env = dict(os.environ)
    env.pop("XONSH_HISTORY_BACKEND", None)
    env.update(runner._pytest_subprocess_env())

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(test_file)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    stderr = completed.stderr
    assert "xonsh" not in stderr.lower(), stderr
    assert "permissionerror" not in stderr.lower(), stderr
