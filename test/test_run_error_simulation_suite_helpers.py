"""Regression checks for run_error_simulation_suite.py helper behavior."""

from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "test" / "run_error_simulation_suite.py"

spec = importlib.util.spec_from_file_location("error_sim_runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(runner)


class _Completed:
    def __init__(self) -> None:
        self.returncode = 0
        self.stdout = ""
        self.stderr = ""


def test_run_test_file_sets_dummy_xonsh_history(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _Completed()

    monkeypatch.setattr(runner.subprocess, "run", _fake_run)
    path = tmp_path / "test_error_dummy.py"
    path.write_text("def test_dummy():\n    pass\n", encoding="utf-8")

    results = runner.run_test_file(path)

    assert len(results) == 1
    env = captured["kwargs"]["env"]
    assert env["XONSH_HISTORY_BACKEND"] == "dummy"
