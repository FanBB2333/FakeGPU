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
        {"peak_memory": 1000},
        {"peak_memory": 1100, "status": "PASS_FIT"},
    )
    assert item["peak_error_bytes"] == 100
    assert item["peak_error_percent"] == 10.0


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
