"""Optional real Trainer regression for evaluation/checkpoint accounting."""

from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from packaging.version import Version
import pytest


def test_observed_native_events_match_real_cpu_trainer(tmp_path):
    versions = {}
    for dependency in ("transformers", "torch", "accelerate"):
        try:
            versions[dependency] = metadata.version(dependency)
        except metadata.PackageNotFoundError:
            pytest.skip(f"optional native Trainer regression requires {dependency}")
    if Version(versions["transformers"]) < Version("4.57"):
        pytest.skip("optional native Trainer regression requires Transformers >= 4.57")
    root = Path(__file__).resolve().parents[1]
    cache_dir = tmp_path / "triton-cache"
    cache_dir.mkdir()
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TRITON_CACHE_DIR": str(cache_dir),
            "PYTHONPATH": str(root) + os.pathsep + env.get("PYTHONPATH", ""),
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            str(root / "tests/support/finetune_native_event_counts.py"),
            str(tmp_path),
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "native-event-counts.json").read_text())
    assert report["transformers"] == versions["transformers"]
    assert report["updates"] == 7
    assert report["checkpoint_steps"] == [3, 6, 7]
    # Installed 5.x Trainers also evaluate at the final optimizer step; a final
    # explicit framework evaluate remains a separate call at the same step.
    expected = (
        [0, 2, 4, 6, 7, 7]
        if Version(versions["transformers"]) >= Version("5")
        else [0, 2, 4, 6, 7]
    )
    assert report["eval_steps"] == expected
    assert report["timeline_passed"] is True
    assert report["passed"] is True
