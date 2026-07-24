from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from fakegpu.repository_analyzer import (
    RepositoryAnalysisError,
    analyze_repository,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_basic_repository(root: Path) -> None:
    root.mkdir()
    (root / "pyproject.toml").write_text(
        """
[project]
name = "tiny-trainer"
version = "0.1.0"
dependencies = ["torch>=2", "transformers"]
""".strip(),
        encoding="utf-8",
    )
    (root / "train.py").write_text(
        """
import torch
from transformers import AutoModel

if __name__ == "__main__":
    AutoModel
    torch.ones(2)
""".strip(),
        encoding="utf-8",
    )


def test_repository_analyzer_discovers_frameworks_and_entrypoint(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "project"
    _write_basic_repository(repository)

    report = analyze_repository(repository)

    assert report["frameworks"] == ["pytorch", "transformers"]
    assert report["entrypoints"] == ["train.py"]
    assert report["readiness"]["verdict"] == "preflight_candidate"
    assert report["readiness"]["fakecuda_candidate"] is True
    assert report["repository"]["git_commit"] is None


def test_repository_analyzer_flags_compiled_acceleration_paths(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "project"
    _write_basic_repository(repository)
    (repository / "kernel.cu").write_text(
        'extern "C" __global__ void kernel() {}',
        encoding="utf-8",
    )
    (repository / "train.py").write_text(
        """
import bitsandbytes
import torch
import triton

compiled = torch.compile(lambda value: value)
""".strip(),
        encoding="utf-8",
    )
    (repository / "deepspeed_config.json").write_text(
        json.dumps({"zero_optimization": {"stage": 2}}),
        encoding="utf-8",
    )

    report = analyze_repository(repository, entrypoints=["train.py"])
    codes = {item["code"] for item in report["findings"]}

    assert report["readiness"]["verdict"] == "requires_real_gpu_or_hybrid"
    assert report["readiness"]["requires_targeted_validation"] is True
    assert {
        "native_cuda_sources",
        "framework_bitsandbytes",
        "framework_triton",
        "torch_compile",
        "deepspeed",
    } <= codes


def test_repository_analyzer_rejects_entrypoint_outside_root(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "project"
    _write_basic_repository(repository)
    outside = tmp_path / "outside.py"
    outside.write_text("pass\n", encoding="utf-8")

    with pytest.raises(RepositoryAnalysisError, match="outside repository"):
        analyze_repository(repository, entrypoints=[outside])


def test_repository_without_entrypoint_is_incomplete(tmp_path: Path) -> None:
    repository = tmp_path / "library"
    repository.mkdir()
    (repository / "module.py").write_text("VALUE = 1\n", encoding="utf-8")

    report = analyze_repository(repository)

    assert report["readiness"]["verdict"] == "analysis_incomplete"
    assert report["readiness"]["static_analysis_complete"] is False
    assert report["readiness"]["fakecuda_candidate"] is False


def test_repository_analyzer_cli_emits_machine_readable_report(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "project"
    _write_basic_repository(repository)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fakegpu",
            "analyze-repo",
            str(repository),
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "fakegpu.repository_analysis.v1"
    assert payload["readiness"]["verdict"] == "preflight_candidate"
