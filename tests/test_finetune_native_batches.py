"""Optional real Trainer regression for native observer batch accounting."""

from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from packaging.version import Version
import pytest


def test_observed_native_batches_match_real_cpu_trainer(tmp_path):
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
            str(root / "tests/support/finetune_native_batch_counts.py"),
            str(tmp_path),
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "native-batch-counts.json").read_text())
    assert report["transformers"] == versions["transformers"]
    assert [
        (case["optimizer_steps"], case["microsteps"], case["samples"])
        for case in report["cases"]
    ] == [(6, 10, 26), (4, 7, 19)]
    complete = report["cases"][0]
    assert complete["expected_nonpadding_tokens"] == 208
    assert complete["expected_supervised_tokens"] == 184
    assert complete["expected_causal_supervised_tokens"] == 174
    for case in report["cases"]:
        assert case["expected_nonpadding_tokens"] > case["expected_supervised_tokens"]
        assert (
            case["expected_supervised_tokens"]
            > case["expected_causal_supervised_tokens"]
        )
