#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a real-GPU calibration report.")
    parser.add_argument("--path", required=True)
    parser.add_argument("--allow-skip", action="store_true")
    parser.add_argument("--expected-gpu-name")
    ns = parser.parse_args(argv)

    path = Path(ns.path)
    if not path.is_file():
        _die(f"report not found: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))

    if report.get("schema_version") not in {
        "real_gpu_calibration.v1",
        "rtx3090ti_calibration.v1",
    }:
        _die(f"unexpected schema_version: {report.get('schema_version')!r}")

    status = str(report.get("status", ""))
    if status.startswith("SKIP"):
        if not ns.allow_skip:
            _die(f"calibration skipped: {report.get('skip_reason')}")
        if not report.get("skip_reason"):
            _die("skip report must include skip_reason")
        print(f"OK: calibration skipped with reason: {report.get('skip_reason')}")
        return 0

    if status != "PASS_CALIBRATED":
        _die(f"unexpected status: {status}")

    gpu = _require_dict(report, "calibration_gpu")
    if not gpu.get("name"):
        _die("calibration_gpu.name is required")
    if ns.expected_gpu_name and gpu.get("name") != ns.expected_gpu_name:
        _die(f"unexpected calibration GPU: {gpu.get('name')!r}")
    if int(gpu.get("total_memory", 0) or 0) <= 0:
        _die("calibration_gpu.total_memory must be positive")

    workloads = report.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        _die("workloads must be a non-empty list")

    sampling = report.get("sampling")
    if sampling is not None:
        if not isinstance(sampling, dict):
            _die("sampling must be an object")
        if int(sampling.get("measured_runs_per_workload", 0) or 0) <= 0:
            _die("sampling.measured_runs_per_workload must be positive")
        if int(sampling.get("warmup_runs_per_workload", -1)) < 0:
            _die("sampling.warmup_runs_per_workload must be non-negative")
        if float(sampling.get("nvml_sample_interval_ms", 0) or 0) <= 0:
            _die("sampling.nvml_sample_interval_ms must be positive")

    if report.get("native_modes_included"):
        oom_probe = _require_dict(report, "hybrid_clamp_oom_probe")
        if oom_probe.get("status") != "PASS_OOM":
            _die(f"hybrid_clamp_oom_probe must be PASS_OOM, got {oom_probe.get('status')!r}")
        if oom_probe.get("error_type") != "OutOfMemoryError":
            _die(
                "hybrid_clamp_oom_probe.error_type must be OutOfMemoryError, "
                f"got {oom_probe.get('error_type')!r}"
            )
        if int(oom_probe.get("requested_bytes", 0) or 0) <= int(oom_probe.get("total_memory", 0) or 0):
            _die("hybrid_clamp_oom_probe must request more than total_memory")

    for index, workload in enumerate(workloads):
        if not isinstance(workload, dict):
            _die(f"workloads[{index}] must be an object")
        name = workload.get("name")
        if not name:
            _die(f"workloads[{index}].name is required")
        real = _require_dict(workload, "real_cuda", ctx=f"workloads[{index}]")
        fake = _require_dict(workload, "fakecuda_preflight", ctx=f"workloads[{index}]")
        if int(real.get("peak_memory", 0) or 0) <= 0:
            _die(f"{name}: real peak_memory must be positive")
        _validate_repeated_measurement(real, ctx=f"{name}/real_cuda")
        if int(fake.get("peak_memory", 0) or 0) <= 0:
            _die(f"{name}: fakecuda peak_memory must be positive")
        if fake.get("status") != "PASS_FIT":
            _die(f"{name}: fakecuda preflight status must be PASS_FIT, got {fake.get('status')!r}")
        if "peak_error_bytes" not in workload or "peak_error_percent" not in workload:
            _die(f"{name}: missing peak error fields")
        if "missing_peak_bytes" not in workload:
            _die(f"{name}: missing missing_peak_bytes")
        if "recommended_memory_safety_margin_bytes" not in workload:
            _die(f"{name}: missing recommended_memory_safety_margin_bytes")
        if "calibration_factor" not in workload:
            _die(f"{name}: missing calibration_factor")
        if workload.get("memory_estimation_method") is not None:
            if workload.get("memory_estimation_method") != "empirical_repeated_upper_bound":
                _die(f"{name}: unexpected memory_estimation_method")
            if int(workload.get("empirical_real_peak_upper_bound_bytes", 0) or 0) != int(
                real.get("peak_memory", 0) or 0
            ):
                _die(f"{name}: empirical upper bound must equal the maximum real trial peak")
            if not workload.get("workload_signature"):
                _die(f"{name}: workload_signature is required for empirical calibration")
        if "likely_gap_reason" not in workload:
            _die(f"{name}: missing likely_gap_reason")
        gap = workload.get("gap_analysis")
        if not isinstance(gap, dict) or "available" not in gap:
            _die(f"{name}: missing gap_analysis")

        native_modes = workload.get("native_modes")
        if report.get("native_modes_included"):
            if not isinstance(native_modes, dict):
                _die(f"{name}: native_modes must be an object")
            for mode_name in ("passthrough", "hybrid_clamp"):
                mode = native_modes.get(mode_name)
                if not isinstance(mode, dict):
                    _die(f"{name}: missing native mode {mode_name}")
                if mode.get("status") != "PASS_EXECUTED":
                    _die(f"{name}/{mode_name}: expected PASS_EXECUTED, got {mode.get('status')!r}")
                measurement = _require_dict(mode, "measurement", ctx=f"{name}/{mode_name}")
                if int(measurement.get("peak_memory", 0) or 0) <= 0:
                    _die(f"{name}/{mode_name}: peak_memory must be positive")
                _validate_repeated_measurement(measurement, ctx=f"{name}/{mode_name}")
                if mode.get("result_signature_match") is not True:
                    _die(f"{name}/{mode_name}: result signature does not match real CUDA")
                if mode_name == "hybrid_clamp":
                    driver_peak = measurement.get("driver_peak_memory")
                    if driver_peak is None or int(driver_peak) <= 0:
                        _die(f"{name}/{mode_name}: driver_peak_memory must be positive")

    if report.get("schema_version") == "real_gpu_calibration.v1" and not report.get("fakecuda_profile"):
        _die("fakecuda_profile is required for generic calibration reports")

    print(f"OK: calibrated {len(workloads)} workload(s) on {gpu.get('name')}")
    return 0


def _validate_repeated_measurement(measurement: dict[str, Any], *, ctx: str) -> None:
    trials = measurement.get("trials")
    if trials is None:
        return
    if not isinstance(trials, list) or not trials:
        _die(f"{ctx}.trials must be a non-empty list")
    if int(measurement.get("trial_count", 0) or 0) != len(trials):
        _die(f"{ctx}.trial_count must match trials length")
    peaks = []
    for index, trial in enumerate(trials):
        if not isinstance(trial, dict):
            _die(f"{ctx}.trials[{index}] must be an object")
        peak = int(trial.get("peak_memory", 0) or 0)
        if peak <= 0:
            _die(f"{ctx}.trials[{index}].peak_memory must be positive")
        peaks.append(peak)
        nvml = trial.get("nvml")
        if isinstance(nvml, dict) and nvml.get("status") == "available":
            if int(nvml.get("sample_count", 0) or 0) <= 0:
                _die(f"{ctx}.trials[{index}].nvml.sample_count must be positive")
    if max(peaks) != int(measurement.get("peak_memory", 0) or 0):
        _die(f"{ctx}.peak_memory must equal the maximum trial peak")
    summary = measurement.get("measurement_summary")
    if not isinstance(summary, dict):
        _die(f"{ctx}.measurement_summary must be an object")
    peak_summary = summary.get("peak_memory")
    if not isinstance(peak_summary, dict) or int(peak_summary.get("max", 0) or 0) != max(peaks):
        _die(f"{ctx}.measurement_summary.peak_memory.max must equal the maximum trial peak")
    if measurement.get("result_signature_stable") is False:
        _die(f"{ctx}.result_signature is not stable across trials")


def _require_dict(obj: dict[str, Any], key: str, *, ctx: str = "report") -> dict[str, Any]:
    value = obj.get(key)
    if not isinstance(value, dict):
        _die(f"{ctx}.{key} must be an object")
    return value


def _die(message: str) -> None:
    print(f"[check_real_gpu_calibration] ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
