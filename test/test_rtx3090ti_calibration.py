from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_rtx3090ti_profile_matches_fakecuda_registry() -> None:
    import fakegpu.torch_patch as torch_patch

    profile = _parse_simple_yaml(ROOT / "profiles" / "rtx3090ti.yaml")
    assert profile["id"] == "rtx3090ti"
    assert profile["name"] == torch_patch._PROFILE_NAMES["rtx3090ti"]
    assert int(profile["memory_bytes"]) == torch_patch._PROFILE_TOTAL_MEMORY["rtx3090ti"]
    assert torch_patch._PROFILE_CC["rtx3090ti"] == (8, 6)


def test_calibration_checker_accepts_explicit_skip_report(tmp_path: Path) -> None:
    report = {
        "schema_version": "rtx3090ti_calibration.v1",
        "status": "SKIP_NO_RTX3090TI",
        "skip_reason": "torch.cuda.is_available() is false",
        "calibration_gpu": None,
        "workloads": [],
        "notes": ["requires real RTX 3090 Ti"],
    }
    path = tmp_path / "calibration_rtx3090ti.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "verification/check_rtx3090ti_calibration.py",
            "--path",
            str(path),
            "--allow-skip",
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "OK: calibration skipped" in completed.stdout


def test_calibration_compare_records_peak_error() -> None:
    from verification.calibration_rtx3090ti import _compare_workload

    item = _compare_workload(
        "probe",
        {
            "peak_memory": 1000,
            "timeline": [
                {"label": "after_forward", "current_memory": 800, "peak_memory": 900},
                {"label": "after_optimizer_step", "current_memory": 1000, "peak_memory": 1000},
            ],
        },
        {
            "peak_memory": 1100,
            "status": "PASS_FIT",
            "timeline": [
                {"label": "after_forward", "current_memory": 700, "peak_memory": 850},
                {"label": "after_optimizer_step", "current_memory": 1100, "peak_memory": 1100},
            ],
        },
    )
    assert item["peak_error_bytes"] == 100
    assert item["peak_error_percent"] == 10.0
    assert item["calibration_factor"] == 0.909
    assert item["gap_analysis"]["available"] is True
    assert item["gap_analysis"]["largest_current_gap"]["label"] == "after_forward"
    assert item["likely_gap_reason"] == "within_lightweight_calibration_tolerance"


def test_calibration_compare_identifies_optimizer_gap() -> None:
    from verification.calibration_rtx3090ti import _compare_workload

    item = _compare_workload(
        "probe",
        {
            "peak_memory": 3000,
            "timeline": [{"label": "after_optimizer_step", "current_memory": 3000, "peak_memory": 3000}],
        },
        {
            "peak_memory": 1000,
            "status": "PASS_FIT",
            "timeline": [{"label": "after_optimizer_step", "current_memory": 900, "peak_memory": 1000}],
        },
    )
    assert item["calibration_factor"] == 3.0
    assert item["gap_analysis"]["largest_current_gap"]["current_gap_bytes"] == 2100
    assert item["likely_gap_reason"] == "cuda_optimizer_backend_hidden_allocation"


def test_calibration_includes_tiny_transformer_workload() -> None:
    from verification.calibration_rtx3090ti import _workloads

    assert "tiny_transformer_step" in _workloads()


def test_calibration_includes_hf_and_lora_researcher_workloads() -> None:
    from verification.calibration_rtx3090ti import _workloads

    workloads = _workloads()
    assert "hf_tiny_gpt2_step" in workloads
    assert "peft_lora_tiny_step" in workloads


def test_lora_workload_reports_trainable_parameter_bytes() -> None:
    from verification.calibration_rtx3090ti import _workload_peft_lora_tiny_step

    payload = _workload_peft_lora_tiny_step("cpu")
    assert payload["parameter_bytes"] > payload["trainable_parameter_bytes"] > 0
    assert payload["lora_rank"] == 4


def _parse_simple_yaml(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("- "):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if value.strip():
            values[key.strip()] = value.strip()
    return values
