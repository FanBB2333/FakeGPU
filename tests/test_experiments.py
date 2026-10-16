from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from fakegpu._process import process_marker
from fakegpu.experiments import (
    compare_experiments,
    plan_experiments,
    plot_experiments,
    run_experiments,
)
from fakegpu.structured_io import load_mapping


SCRIPT = """import json, pathlib, sys
size, output = int(sys.argv[1]), pathlib.Path(sys.argv[2])
marker = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else None
if marker is not None and not marker.exists():
    marker.touch()
    raise SystemExit(7)
output.write_text(json.dumps({"value": size * 2, "nested": [size]}))
"""


def fixture(tmp_path, **changes):
    script = tmp_path / "worker.py"
    script.write_text(SCRIPT)
    spec = {
        "schema_version": "fakegpu.experiment_matrix.v1",
        "command": ["{python}", "{script}", "{size}", "{run_dir}/result.json"],
        "inputs": {"script": "worker.py"},
        "parameters": {"size": [2, 4]},
        "repetitions": 2,
        "max_runs": 4,
        "timeout_seconds": 10,
        "artifacts": ["result.json"],
        "metrics": {
            "value": {"artifact": "result.json", "path": ["value"], "unit": "items"},
            "nested": {"artifact": "result.json", "path": ["nested", 0]},
        },
        **changes,
    }
    config = tmp_path / "spec.json"
    config.write_text(json.dumps(spec))
    return config, tmp_path / "workspace"


def test_grid_resume_compare_and_tamper(tmp_path):
    config, root = fixture(tmp_path)
    plan = plan_experiments(config, root)
    assert not root.exists()
    assert [j["parameters"]["size"] for j in plan["jobs"]] == [2, 2, 4, 4]
    assert len({j["run_id"] for j in plan["jobs"]}) == 4
    report = run_experiments(config, root)
    assert report["complete"]
    assert report["workspaces"][0]["groups"][0]["metrics"]["value"]["mean"] == 4
    assert run_experiments(config, root, resume=True) == report
    assert len(list(root.glob("runs/*/attempt-*"))) == 4
    with pytest.raises(ValueError, match="exists"):
        run_experiments(config, root)
    result = next(root.glob("runs/*/attempt-*/result.json"))
    result.write_text('{"value": 999, "nested": [2]}')
    with pytest.raises(ValueError, match="changed"):
        compare_experiments([root])


def test_explicit_retry_preserves_failure_and_uses_new_attempt(tmp_path):
    marker = tmp_path / "marker"
    config, root = fixture(
        tmp_path,
        parameters={"size": [2]},
        repetitions=1,
        command=[
            "{python}",
            "{script}",
            "{size}",
            "{run_dir}/result.json",
            str(marker),
        ],
    )
    first = run_experiments(config, root)
    assert not first["complete"]
    run = first["workspaces"][0]["runs"][0]
    original = root / "runs" / run["run_id"] / "attempt-0001" / "execution.json"
    old = original.read_bytes()
    assert load_mapping(original)["returncode"] == 7
    assert run_experiments(config, root, resume=True) == first
    final = run_experiments(config, root, resume=True, retry_failed=True)
    assert final["complete"]
    assert original.read_bytes() == old
    assert final["workspaces"][0]["runs"][0]["attempts"] == [
        "attempt-0001",
        "attempt-0002",
    ]
    assert "RuntimeError" in load_mapping(original.parent / "outcome.json")["error"]
    (original.parent / "run.log").write_text("changed failure log")
    with pytest.raises(ValueError, match="record changed"):
        compare_experiments([root])


def test_changed_input_blocks_resume(tmp_path):
    config, root = fixture(tmp_path)
    run_experiments(config, root)
    (tmp_path / "worker.py").write_text(SCRIPT + "\n# changed\n")
    with pytest.raises(ValueError, match="changed"):
        run_experiments(config, root, resume=True)


def test_last_worker_mutation_is_a_preserved_failure(tmp_path):
    config, root = fixture(tmp_path, parameters={"size": [1]}, repetitions=1)
    script = tmp_path / "worker.py"
    script.write_text(SCRIPT + '\npathlib.Path(__file__).write_text("# mutation")\n')
    report = run_experiments(config, root)
    assert not report["complete"]
    assert "input changed" in report["workspaces"][0]["runs"][0]["error"]


def test_live_interrupted_process_refuses_overlap(tmp_path, monkeypatch):
    import fakegpu.experiments as module

    config, root = fixture(tmp_path, parameters={"size": [1]}, repetitions=1)
    run_experiments(config, root)
    state_path = next(root.glob("runs/*/state.json"))
    state = load_mapping(state_path)
    state["status"] = "running"
    state_path.write_text(json.dumps(state))
    monkeypatch.setattr(module, "_live", lambda execution: True)
    with pytest.raises(RuntimeError, match="still alive"):
        run_experiments(config, root, resume=True, retry_failed=True)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"max_runs": 3}, "exceeds"),
        ({"parameters": {"size": [1, 1]}}, "duplicate"),
        ({"parameters": {"run_dir": [1]}}, "reserved"),
        ({"command": ["{python}", "{script.__class__}"]}, "field"),
        ({"command": ["{python}", "{size:02d}"]}, "field"),
        ({"artifacts": ["../result.json"]}, "traversal"),
        ({"timeout_seconds": float("nan")}, "timeout"),
        ({"unknown": True}, "unknown"),
        ({"parameters": {"size": [None]}}, "scalars"),
    ],
)
def test_invalid_requests(tmp_path, changes, match):
    config, root = fixture(tmp_path, **changes)
    with pytest.raises(ValueError, match=match):
        plan_experiments(config, root)


def test_missing_artifact_and_nonfinite_metric_fail(tmp_path):
    config, root = fixture(tmp_path, parameters={"size": [1]}, repetitions=1)
    script = tmp_path / "worker.py"
    script.write_text("pass\n")
    assert not run_experiments(config, root)["complete"]
    script.write_text(
        'import pathlib,sys\npathlib.Path(sys.argv[2]).write_text(\'{"value":NaN,"nested":[1]}\')\n'
    )
    report = run_experiments(config, tmp_path / "other")
    assert not report["complete"]
    assert "finite" in report["workspaces"][0]["runs"][0]["error"]


def test_interrupted_state_is_not_inferred_from_artifact(tmp_path):
    config, root = fixture(tmp_path, parameters={"size": [1]}, repetitions=1)
    run_experiments(config, root)
    path = next(root.glob("runs/*/state.json"))
    state = load_mapping(path)
    state["status"] = "running"
    path.write_text(json.dumps(state))
    report = run_experiments(config, root, resume=True)
    assert not report["complete"]
    assert report["workspaces"][0]["runs"][0]["status"] == "interrupted"
    assert run_experiments(config, root, resume=True, retry_failed=True)["complete"]


def test_timeout_kills_owned_process_group(tmp_path):
    config, root = fixture(
        tmp_path, parameters={"size": [1]}, repetitions=1, timeout_seconds=0.5
    )
    (tmp_path / "worker.py").write_text("""import pathlib, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"])
pathlib.Path(sys.argv[2]).write_text(str(child.pid))
time.sleep(60)
""")
    report = run_experiments(config, root)
    assert not report["complete"]
    attempt = next(root.glob("runs/*/attempt-*"))
    assert load_mapping(attempt / "execution.json")["status"] == "timed_out"
    pid = int((attempt / "result.json").read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        marker = process_marker(pid)
        if marker is None or marker["state"] == "Z":
            break
        time.sleep(0.01)
    else:
        pytest.fail("timed-out experiment left a live owned child")


def test_separate_workspaces_and_plot(tmp_path):
    pytest.importorskip("matplotlib")
    config, root = fixture(tmp_path)
    run_experiments(config, root)
    second = tmp_path / "second"
    run_experiments(config, second)
    report = compare_experiments([root, second])
    assert len(report["workspaces"]) == 2
    target = plot_experiments(report, tmp_path / "plot.svg", x="size", metric="value")
    assert "<svg" in target.read_text()
    assert all(
        g["metrics"]["value"]["count"] == 2
        for w in report["workspaces"]
        for g in w["groups"]
    )


def test_cli_standard_library_and_offline_compare(tmp_path):
    config, root = fixture(tmp_path)
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))

    def cli(*args):
        return subprocess.run(
            [sys.executable, "-S", "-m", "fakegpu", *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

    assert (
        cli(
            "plan-experiments", "--config", str(config), "--workspace", str(root)
        ).returncode
        == 0
    )
    out = tmp_path / "report.json.gz"
    assert (
        cli(
            "run-experiments",
            "--config",
            str(config),
            "--workspace",
            str(root),
            "--json",
            str(out),
        ).returncode
        == 0
    )
    config.unlink()
    (tmp_path / "worker.py").unlink()
    comparison = cli("compare-experiments", "--workspace", str(root))
    assert comparison.returncode == 0, comparison.stderr
    assert json.loads(comparison.stdout) == load_mapping(out)
