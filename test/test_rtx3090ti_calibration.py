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


def test_rtx_pro_5000_blackwell_profile_matches_measured_hardware() -> None:
    import fakegpu.torch_patch as torch_patch

    profile_id = "rtx-pro-5000-blackwell"
    profile = _parse_simple_yaml(ROOT / "profiles" / f"{profile_id}.yaml")
    assert profile["id"] == profile_id
    assert profile["name"] == torch_patch._PROFILE_NAMES[profile_id]
    assert int(profile["memory_bytes"]) == 76374540288
    assert int(profile["sm_count"]) == 110
    assert int(profile["compute_major"]) == 12
    assert torch_patch._PROFILE_TOTAL_MEMORY[profile_id] == 76374540288
    assert torch_patch._PROFILE_CC[profile_id] == (12, 0)


def test_calibration_checker_accepts_explicit_skip_report(tmp_path: Path) -> None:
    report = {
        "schema_version": "real_gpu_calibration.v1",
        "status": "SKIP_NO_REAL_GPU",
        "skip_reason": "torch.cuda.is_available() is false",
        "calibration_gpu": None,
        "fakecuda_profile": None,
        "workloads": [],
        "notes": ["requires a real CUDA GPU"],
    }
    path = tmp_path / "calibration_real_gpu.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "verification/check_real_gpu_calibration.py",
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


def test_calibration_auto_selects_rtx_pro_5000_profile(tmp_path: Path, monkeypatch) -> None:
    from verification import calibration_rtx3090ti as calibration

    gpu = {
        "name": "NVIDIA RTX PRO 5000 72GB Blackwell",
        "total_memory": 76374540288,
        "compute_capability": "12.0",
        "sm_count": 110,
    }
    monkeypatch.setattr(calibration, "_cuda_probe", lambda: {"status": "available", "gpu": gpu})
    monkeypatch.setattr(calibration, "_workloads", lambda: {"probe": lambda *_args: {}})
    monkeypatch.setattr(
        calibration,
        "_run_real_worker",
        lambda _name, **_kwargs: {"peak_memory": 1000, "result_signature": 0.25},
    )

    selected: list[str] = []

    def fake_worker(_name: str, *, out_dir: Path, profile: str):
        assert out_dir == tmp_path
        selected.append(profile)
        return {"peak_memory": 900, "status": "PASS_FIT", "result_signature": 0.25}

    monkeypatch.setattr(calibration, "_run_fakecuda_preflight", fake_worker)

    native_modes: list[str] = []

    def native_worker(
        _name: str,
        *,
        mode: str,
        out_dir: Path,
        profile: str,
        build_dir: Path,
        **_kwargs,
    ):
        assert out_dir == tmp_path
        assert profile == "rtx-pro-5000-blackwell"
        assert build_dir == ROOT / "build"
        native_modes.append(mode)
        return {
            "peak_memory": 1000,
            "result_signature": 0.25,
            "status": "PASS_EXECUTED",
            "driver_peak_memory": 800 if mode == "hybrid" else None,
        }

    monkeypatch.setattr(calibration, "_run_native_worker", native_worker)
    monkeypatch.setattr(
        calibration,
        "_run_hybrid_oom_probe",
        lambda **_kwargs: {
            "status": "PASS_OOM",
            "error_type": "OutOfMemoryError",
            "requested_bytes": 1001,
            "total_memory": 1000,
        },
    )

    report = calibration.run_calibration(out_dir=tmp_path, build_dir=ROOT / "build")
    assert report["schema_version"] == "real_gpu_calibration.v1"
    assert report["calibration_gpu"] == gpu
    assert report["fakecuda_profile"] == "rtx-pro-5000-blackwell"
    assert selected == ["rtx-pro-5000-blackwell"]
    assert native_modes == ["passthrough", "hybrid"]
    assert report["workloads"][0]["native_modes"]["hybrid_clamp"]["result_signature_match"] is True
    assert report["hybrid_clamp_oom_probe"]["status"] == "PASS_OOM"


def test_repeated_worker_measurements_use_observed_upper_bound() -> None:
    from verification.calibration_rtx3090ti import _aggregate_worker_trials

    trials = [
        {
            "peak_memory": 100,
            "peak_reserved_memory": 130,
            "requested_peak_memory": 95,
            "result_signature": 0.5,
            "parameter_bytes": 32,
            "timeline": [{"label": "done", "peak_memory": 100}],
            "trial_index": 0,
            "nvml": {
                "status": "available",
                "sample_count": 4,
                "peak_process_memory": 500,
                "peak_process_delta_memory": 110,
                "peak_device_used_memory": 800,
                "peak_device_used_delta_memory": 120,
            },
        },
        {
            "peak_memory": 120,
            "peak_reserved_memory": 140,
            "requested_peak_memory": 118,
            "result_signature": 0.50000001,
            "parameter_bytes": 32,
            "timeline": [{"label": "done", "peak_memory": 120}],
            "trial_index": 1,
            "nvml": {
                "status": "available",
                "sample_count": 5,
                "peak_process_memory": 530,
                "peak_process_delta_memory": 125,
                "peak_device_used_memory": 830,
                "peak_device_used_delta_memory": 135,
            },
        },
        {
            "peak_memory": 110,
            "peak_reserved_memory": 135,
            "requested_peak_memory": 105,
            "result_signature": 0.5,
            "parameter_bytes": 32,
            "timeline": [{"label": "done", "peak_memory": 110}],
            "trial_index": 2,
            "nvml": {"status": "unavailable", "reason": "test"},
        },
    ]

    result = _aggregate_worker_trials(
        worker="real",
        workload_name="probe",
        trials=trials,
        warmup_runs=1,
    )
    assert result["peak_memory"] == 120
    assert result["peak_reserved_memory"] == 140
    assert result["trial_count"] == 3
    assert result["warmup_runs"] == 1
    assert result["measurement_summary"]["peak_memory"] == {
        "count": 3,
        "min": 100,
        "median": 110,
        "p95": 119,
        "max": 120,
        "range": 20,
        "relative_range_percent": 18.181818,
    }
    assert result["measurement_summary"]["nvml_process_memory"]["max"] == 530
    assert result["result_signature_stable"] is True
    assert result["workload_descriptor"]["parameters"]["parameter_bytes"] == 32
    assert len(result["workload_signature"]) == 64


def test_nvml_sampler_marks_missing_wsl_process_mapping_as_unavailable(monkeypatch) -> None:
    from verification.calibration_rtx3090ti import _NvmlProcessMemorySampler

    sampler = _NvmlProcessMemorySampler(device_index=0, interval_ms=2.0)
    sampler._status = "available"
    sampler._sample_count = 1
    sampler._baseline_device_used_memory = 100
    sampler._peak_device_used_memory = 120
    monkeypatch.setattr(sampler, "_read_sample", lambda: (None, 140))

    result = sampler.stop()
    assert result["status"] == "available"
    assert result["process_memory_status"] == "unavailable"
    assert "peak_process_memory" not in result
    assert result["peak_device_used_memory"] == 140
    assert "WSL" in result["process_memory_reason"]


def test_calibration_bundle_ignores_unavailable_zero_process_memory() -> None:
    from verification.aggregate_real_gpu_calibrations import _nvml_trial_values

    payload = {
        "trials": [
            {
                "nvml": {
                    "status": "available",
                    "process_memory_status": "unavailable",
                    "peak_process_memory": 0,
                }
            }
        ]
    }
    assert _nvml_trial_values(payload, "peak_process_memory") == []


def test_compare_uses_each_real_trial_for_empirical_gap_summary() -> None:
    from verification.calibration_rtx3090ti import _compare_workload

    item = _compare_workload(
        "probe",
        {
            "peak_memory": 120,
            "trials": [
                {"peak_memory": 100},
                {"peak_memory": 120},
                {"peak_memory": 110},
            ],
            "workload_signature": "abc",
            "workload_descriptor": {"name": "probe", "parameters": {}},
        },
        {"peak_memory": 90, "status": "PASS_FIT"},
    )
    assert item["memory_estimation_method"] == "empirical_repeated_upper_bound"
    assert item["empirical_real_peak_upper_bound_bytes"] == 120
    assert item["empirical_physical_peak_upper_bound_bytes"] == 120
    assert item["empirical_physical_peak_upper_bound_source"] == "torch_allocator_peak"
    assert item["empirical_real_peak_summary"]["median"] == 110
    assert item["empirical_missing_peak_summary"]["min"] == 10
    assert item["empirical_missing_peak_summary"]["max"] == 30


def test_aggregate_calibrations_keeps_gpu_specific_empirical_observations(tmp_path: Path) -> None:
    from verification.aggregate_real_gpu_calibrations import aggregate_reports

    descriptor = {"name": "probe", "parameters": {"batch_size": 2}}
    reports = []
    for profile, gpu_name, peaks, fake_peak in (
        ("rtx3090ti", "NVIDIA GeForce RTX 3090 Ti", [100, 110, 105], 90),
        ("rtx-pro-5000-blackwell", "NVIDIA RTX PRO 5000 72GB Blackwell", [120, 125, 123], 92),
    ):
        path = tmp_path / f"{profile}.json"
        reports.append(
            (
                path,
                {
                    "schema_version": "real_gpu_calibration.v1",
                    "status": "PASS_CALIBRATED",
                    "created_at_unix": 1,
                    "calibration_gpu": {"name": gpu_name, "compute_capability": "8.6"},
                    "fakecuda_profile": profile,
                    "software": {"torch_version": "test"},
                    "workloads": [
                        {
                            "name": "probe",
                            "workload_descriptor": descriptor,
                            "workload_signature": "same-signature",
                            "real_cuda": {
                                "peak_memory": max(peaks),
                                "trials": [
                                    {
                                        "peak_memory": peak,
                                        "peak_reserved_memory": peak + 10,
                                        "nvml": {
                                            "status": "available",
                                            "peak_process_memory": peak + 200,
                                            "peak_process_delta_memory": peak + 20,
                                            "peak_device_used_memory": peak + 300,
                                            "peak_device_used_delta_memory": peak + 30,
                                        },
                                    }
                                    for peak in peaks
                                ],
                            },
                            "fakecuda_preflight": {"peak_memory": fake_peak},
                        }
                    ],
                },
            )
        )

    bundle = aggregate_reports(reports)
    assert bundle["schema_version"] == "real_gpu_calibration_bundle.v1"
    assert len(bundle["workloads"]) == 1
    workload = bundle["workloads"][0]
    assert workload["profiles"] == ["rtx-pro-5000-blackwell", "rtx3090ti"]
    assert workload["total_real_trials"] == 6
    observations = {item["profile"]: item for item in workload["observations"]}
    assert observations["rtx3090ti"]["empirical_real_peak_upper_bound_bytes"] == 110
    assert observations["rtx3090ti"]["empirical_missing_peak_upper_bound_bytes"] == 20
    assert observations["rtx3090ti"]["empirical_physical_peak_upper_bound_bytes"] == 310
    assert observations["rtx3090ti"]["empirical_physical_peak_upper_bound_source"] == "nvml_process_peak"
    assert observations["rtx3090ti"]["nvml_device_used_delta_memory"]["max"] == 140
    assert observations["rtx-pro-5000-blackwell"]["real_cuda_peak_memory"]["median"] == 123


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
    assert item["missing_peak_bytes"] == 0
    assert item["recommended_memory_safety_margin_bytes"] == 0
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
    assert item["missing_peak_bytes"] == 2000
    assert item["recommended_memory_safety_margin_bytes"] == 2000
    assert item["gap_analysis"]["largest_current_gap"]["current_gap_bytes"] == 2100
    assert item["likely_gap_reason"] == "cuda_optimizer_backend_hidden_allocation"


def test_native_mode_comparison_checks_peak_and_result_signature() -> None:
    from verification.calibration_rtx3090ti import _compare_native_mode

    comparison = _compare_native_mode(
        {"peak_memory": 1024, "result_signature": 0.5},
        {
            "peak_memory": 1056,
            "result_signature": 0.5000001,
            "status": "PASS_EXECUTED",
        },
    )
    assert comparison["status"] == "PASS_EXECUTED"
    assert comparison["peak_error_bytes"] == 32
    assert comparison["peak_error_percent"] == 3.125
    assert comparison["result_signature_match"] is True


def test_calibration_includes_tiny_transformer_workload() -> None:
    from verification.calibration_rtx3090ti import _workloads

    assert "tiny_transformer_step" in _workloads()


def test_calibration_includes_hf_and_lora_researcher_workloads() -> None:
    from verification.calibration_rtx3090ti import _workloads

    workloads = _workloads()
    assert "hf_tiny_gpt2_step" in workloads
    assert "peft_lora_tiny_step" in workloads


def test_calibration_includes_gradient_memory_workloads() -> None:
    from verification.calibration_rtx3090ti import _workloads

    workloads = _workloads()
    assert "gradient_accumulation_step" in workloads
    assert "gradient_checkpointing_step" in workloads


def test_gradient_accumulation_workload_reports_microbatches() -> None:
    from verification.calibration_rtx3090ti import _workload_gradient_accumulation_step

    payload = _workload_gradient_accumulation_step("cpu")
    assert payload["microbatches"] == 4
    assert payload["parameter_bytes"] > 0
    assert payload["result_signature"] > 0


def test_gradient_checkpointing_workload_reports_checkpoint_usage() -> None:
    from verification.calibration_rtx3090ti import _workload_gradient_checkpointing_step

    payload = _workload_gradient_checkpointing_step("cpu")
    assert payload["checkpointed_layers"] == 3
    assert payload["uses_gradient_checkpointing"] is True
    assert payload["parameter_bytes"] > 0


def test_lora_workload_reports_trainable_parameter_bytes() -> None:
    from verification.calibration_rtx3090ti import _workload_peft_lora_tiny_step

    payload = _workload_peft_lora_tiny_step("cpu")
    assert payload["parameter_bytes"] > payload["trainable_parameter_bytes"] > 0
    assert payload["lora_rank"] == 4
    assert payload["uses_attention_mask"] is True


def test_hf_workload_uses_attention_mask_for_fakecuda_compat() -> None:
    from verification.calibration_rtx3090ti import _workload_hf_tiny_gpt2_step

    payload = _workload_hf_tiny_gpt2_step("cpu")
    assert payload["framework"] == "transformers"
    assert payload["uses_attention_mask"] is True


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
