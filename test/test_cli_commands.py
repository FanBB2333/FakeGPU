from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from fakegpu.profile_catalog import (
    architecture_for_compute_capability,
    catalog_summary,
    load_profiles,
    official_compute_capabilities,
    validate_catalog,
)


ROOT = Path(__file__).resolve().parents[1]


def _run_fakegpu(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(ROOT)
        if not env.get("PYTHONPATH")
        else str(ROOT) + os.pathsep + env["PYTHONPATH"]
    )
    env.setdefault("XONSH_HISTORY_BACKEND", "dummy")
    return subprocess.run(
        [sys.executable, "-m", "fakegpu", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )


@pytest.mark.parametrize(
    ("major", "minor", "expected"),
    [
        (5, 2, "maxwell"),
        (6, 0, "pascal"),
        (6, 1, "pascal"),
        (7, 0, "volta"),
        (7, 5, "turing"),
        (8, 0, "ampere"),
        (8, 6, "ampere"),
        (8, 7, "ampere"),
        (8, 9, "ada"),
        (9, 0, "hopper"),
        (10, 0, "blackwell"),
        (10, 3, "blackwell"),
        (11, 0, "blackwell"),
        (12, 0, "blackwell"),
        (12, 1, "blackwell"),
        (8, 8, "unknown"),
        (10, 1, "unknown"),
        (13, 0, "unknown"),
    ],
)
def test_compute_capability_architecture_mapping(
    major: int,
    minor: int,
    expected: str,
) -> None:
    assert architecture_for_compute_capability(major, minor) == expected


def test_profile_catalog_matches_nvidia_snapshot() -> None:
    profiles = load_profiles()
    validation = validate_catalog(profiles)
    summary = catalog_summary(profiles)

    assert validation.errors == ()
    assert summary["profile_count"] == 24
    assert set(summary["architectures"]) == {
        "maxwell",
        "pascal",
        "volta",
        "turing",
        "ampere",
        "ada",
        "hopper",
        "blackwell",
    }
    assert summary["segments"] == {
        "consumer": 2,
        "datacenter": 16,
        "embedded": 2,
        "test": 1,
        "workstation": 3,
    }
    assert summary["compute_capabilities"] == [
        "5.2",
        "6.0",
        "6.1",
        "7.0",
        "7.5",
        "8.0",
        "8.6",
        "8.7",
        "8.9",
        "9.0",
        "10.0",
        "10.3",
        "11.0",
        "12.0",
        "12.1",
    ]
    for profile in profiles.values():
        path_parts = Path(profile.profile_path).parts
        assert path_parts == (
            profile.architecture,
            profile.segment,
            f"{profile.id}.yaml",
        )

    official = official_compute_capabilities()
    expected_models = {
        "Tesla P4": "6.1",
        "NVIDIA A30": "8.0",
        "NVIDIA A10": "8.6",
        "Jetson AGX Orin": "8.7",
        "NVIDIA L4": "8.9",
        "NVIDIA H200": "9.0",
        "NVIDIA B200": "10.0",
        "NVIDIA B300": "10.3",
        "Jetson T5000": "11.0",
        "NVIDIA RTX PRO 5000 Blackwell": "12.0",
        "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": "12.0",
        "NVIDIA GB10 (DGX Spark)": "12.1",
    }
    for model, capability in expected_models.items():
        assert official[model] == capability


def test_torch_patch_registries_are_generated_from_yaml() -> None:
    from fakegpu import torch_patch

    profiles = load_profiles()
    assert set(torch_patch._PROFILE_CC) == set(profiles)
    for profile_id, profile in profiles.items():
        assert torch_patch._PROFILE_CC[profile_id] == profile.compute_capability
        assert torch_patch._PROFILE_NAMES[profile_id] == profile.torch_name
        assert torch_patch._PROFILE_TOTAL_MEMORY[profile_id] == profile.memory_bytes
        assert (
            torch_patch._PROFILE_SUPPORTED_TYPES[profile_id]
            == profile.supported_types
        )
        assert (
            torch_patch._arch_name(profile.compute_major, profile.compute_minor)
            == profile.architecture.title()
        )


def test_doctor_reports_selected_blackwell_profile_as_json() -> None:
    result = _run_fakegpu("doctor", "--profile", "b300", "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)

    assert payload["ok"] is True
    assert payload["profile_summary"]["profile_count"] == 24
    assert payload["selected_profile"]["id"] == "b300"
    assert payload["selected_profile"]["architecture"] == "blackwell"
    assert payload["selected_profile"]["segment"] == "datacenter"
    assert (
        payload["selected_profile"]["profile_path"]
        == "blackwell/datacenter/b300.yaml"
    )
    assert payload["selected_profile"]["compute_capability"] == "10.3"
    assert payload["selected_profile"]["compiler_target"] == "sm_103"


def test_doctor_rejects_unknown_profile() -> None:
    result = _run_fakegpu("doctor", "--profile", "does-not-exist", "--json")
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert any(
        item["status"] == "fail" and "unknown GPU profile" in item["detail"]
        for item in payload["checks"]
    )


def test_demo_runs_tiny_training_with_ada_profile() -> None:
    result = _run_fakegpu("demo", "--profile", "l4", "--steps", "1", "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)

    assert payload["ok"] is True
    assert payload["profile_id"] == "l4"
    assert payload["architecture"] == "ada"
    assert payload["compute_capability"] == "8.9"
    assert payload["compiler_target"] == "sm_89"
    assert payload["tensor_device"] == "cuda:0"
    assert payload["tensor_is_cuda"] is True
    assert payload["steps"] == 1


def test_fakecuda_profile_matrix() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(ROOT)
        if not env.get("PYTHONPATH")
        else str(ROOT) + os.pathsep + env["PYTHONPATH"]
    )
    result = subprocess.run(
        [sys.executable, "verification/test_fakecuda_profile_matrix.py"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    assert "validated 15 profiles across 15 compute capabilities" in result.stdout


def test_top_level_help_names_builtin_commands() -> None:
    result = _run_fakegpu("--help")
    assert result.returncode == 0
    assert "fakegpu demo" in result.stdout
    assert "fakegpu doctor" in result.stdout
