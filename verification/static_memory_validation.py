#!/usr/bin/env python3
"""Validate the ATen storage-liveness estimator against real CUDA peaks."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "build" / "static_memory_validation" / "static_memory_validation.json"
SCHEMA_VERSION = "static_memory_validation.v1"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    family: str
    parameters: dict[str, Any]
    factory: Callable[[], tuple[Any, tuple[Any, ...]]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--workload", action="append", choices=sorted(_workloads()))
    parser.add_argument("--warmup-runs", type=_nonnegative_int, default=1)
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Capture static estimates without requiring or measuring a real CUDA device.",
    )
    parser.add_argument(
        "--max-underestimate-percent",
        type=_nonnegative_float,
        help="Return exit code 2 when any measured workload exceeds this underestimation limit.",
    )
    parser.add_argument(
        "--min-workspace-coverage",
        type=_fraction,
        help=(
            "Return exit code 2 when modeled workspace-candidate call coverage "
            "is below this fraction."
        ),
    )
    parser.add_argument(
        "--reject-extrapolated-workspaces",
        action="store_true",
        help=(
            "Exclude extrapolated workspace profiles from --min-workspace-coverage."
        ),
    )
    ns = parser.parse_args(argv)
    if ns.reject_extrapolated_workspaces and ns.min_workspace_coverage is None:
        parser.error(
            "--reject-extrapolated-workspaces requires --min-workspace-coverage"
        )

    selected_names = ns.workload or list(_workloads())
    report = run_validation(
        workload_names=selected_names,
        warmup_runs=ns.warmup_runs,
        repeats=ns.repeats,
        static_only=bool(ns.static_only),
        max_underestimate_percent=ns.max_underestimate_percent,
        min_workspace_coverage=ns.min_workspace_coverage,
        reject_extrapolated_workspaces=bool(
            ns.reject_extrapolated_workspaces
        ),
    )

    output = ns.output.resolve()
    markdown = (ns.markdown or output.with_suffix(".md")).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown.write_text(render_markdown(report), encoding="utf-8")
    print(f"static memory validation: {output}")
    print(f"static memory validation markdown: {markdown}")
    return 2 if str(report["status"]).startswith("FAIL_") else 0


def run_validation(
    *,
    workload_names: list[str],
    warmup_runs: int,
    repeats: int,
    static_only: bool,
    max_underestimate_percent: float | None,
    min_workspace_coverage: float | None = None,
    reject_extrapolated_workspaces: bool = False,
) -> dict[str, Any]:
    import torch

    from fakegpu.memory_estimator import estimate_module_memory

    available_workloads = _workloads()
    invalid = [name for name in workload_names if name not in available_workloads]
    if invalid:
        raise ValueError(f"unknown workload(s): {', '.join(invalid)}")

    has_real_cuda = bool(torch.cuda.is_available()) and not static_only
    backend_calibration = _measure_backend_resident() if has_real_cuda else None
    results: list[dict[str, Any]] = []
    for name in workload_names:
        spec = available_workloads[name]
        torch.manual_seed(0)
        model, args = spec.factory()
        static_estimate = estimate_module_memory(
            model,
            args,
            mode="training",
            loss_fn=_training_loss,
            optimizer="adamw",
            retain_forward_outputs=True,
            target_device="cuda" if has_real_cuda else "auto",
        )
        measured = (
            _measure_real_cuda(
                model,
                args,
                warmup_runs=warmup_runs,
                repeats=repeats,
            )
            if has_real_cuda
            else None
        )
        comparison = _compare_estimate(
            static_estimate,
            measured,
            backend_calibration=backend_calibration,
        )
        results.append(
            {
                "name": spec.name,
                "family": spec.family,
                "parameters": spec.parameters,
                "static_estimate": static_estimate,
                "real_cuda": measured,
                "comparison": comparison,
            }
        )
        del model, args
        gc.collect()
        if has_real_cuda:
            torch.cuda.empty_cache()

    measured_comparisons = [
        item["comparison"]
        for item in results
        if isinstance(item.get("comparison"), dict)
        and item["comparison"].get("underestimate_percent") is not None
    ]
    summary = _comparison_summary(measured_comparisons)
    workspace_estimates = [
        dict(item["static_estimate"].get("workspace_estimate") or {})
        for item in results
    ]
    coverage_candidates = sum(
        int((workspace.get("coverage") or {}).get("candidate_operator_count", 0) or 0)
        for workspace in workspace_estimates
    )
    coverage_modeled = sum(
        int((workspace.get("coverage") or {}).get("modeled_operator_count", 0) or 0)
        for workspace in workspace_estimates
    )
    coverage_non_extrapolated = sum(
        int(
            (workspace.get("coverage") or {}).get(
                "non_extrapolated_operator_count",
                0,
            )
            or 0
        )
        for workspace in workspace_estimates
    )
    coverage_unprofiled = sum(
        int((workspace.get("coverage") or {}).get("unprofiled_operator_count", 0) or 0)
        for workspace in workspace_estimates
    )
    modeled_fraction = (
        coverage_modeled / coverage_candidates if coverage_candidates else 1.0
    )
    non_extrapolated_fraction = (
        coverage_non_extrapolated / coverage_candidates
        if coverage_candidates
        else 1.0
    )
    aggregate_workspace_coverage = {
        "unit": "workspace_candidate_operator_calls",
        "candidate_operator_count": coverage_candidates,
        "modeled_operator_count": coverage_modeled,
        "non_extrapolated_operator_count": coverage_non_extrapolated,
        "unprofiled_operator_count": coverage_unprofiled,
        "modeled_fraction": round(modeled_fraction, 6),
        "non_extrapolated_fraction": round(non_extrapolated_fraction, 6),
    }
    summary.update(
        {
            "workspace_profile_count": sum(
                int(workspace.get("profiled_operator_count", 0) or 0)
                for workspace in workspace_estimates
            ),
            "extrapolated_workspace_profile_count": sum(
                int(workspace.get("extrapolated_profile_count", 0) or 0)
                for workspace in workspace_estimates
            ),
            "graph_modeled_attention_operator_count": sum(
                sum(
                    int(count)
                    for count in dict(
                        workspace.get("graph_modeled_attention_operators") or {}
                    ).values()
                )
                for workspace in workspace_estimates
            ),
            "unprofiled_attention_operator_count": sum(
                sum(
                    int(count)
                    for count in dict(
                        workspace.get("unprofiled_attention_operators") or {}
                    ).values()
                )
                for workspace in workspace_estimates
            ),
            "workspace_coverage": aggregate_workspace_coverage,
        }
    )
    failed_underestimate = (
        max_underestimate_percent is not None
        and summary.get("max_underestimate_percent") is not None
        and float(summary["max_underestimate_percent"]) > max_underestimate_percent
    )
    selected_workspace_fraction = (
        non_extrapolated_fraction
        if reject_extrapolated_workspaces
        else modeled_fraction
    )
    failed_workspace_coverage = (
        min_workspace_coverage is not None
        and selected_workspace_fraction + 1e-12 < min_workspace_coverage
    )
    if failed_underestimate:
        status = "FAIL_UNDERESTIMATE"
    elif failed_workspace_coverage:
        status = "FAIL_WORKSPACE_COVERAGE"
    elif has_real_cuda:
        status = "PASS_MEASURED"
    else:
        status = "PASS_STATIC_ONLY"

    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "created_at_unix": int(time.time()),
        "gpu": _gpu_fingerprint() if has_real_cuda else None,
        "software": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda) if torch.version.cuda else None,
        },
        "sampling": {
            "warmup_runs": int(warmup_runs) if has_real_cuda else 0,
            "measured_runs": int(repeats) if has_real_cuda else 0,
            "metric": "torch.cuda.max_memory_allocated",
            "phase_resolved": bool(has_real_cuda),
        },
        "backend_calibration": backend_calibration,
        "validation_limit": {
            "max_underestimate_percent": max_underestimate_percent,
            "min_workspace_coverage": min_workspace_coverage,
            "reject_extrapolated_workspaces": reject_extrapolated_workspaces,
        },
        "summary": summary,
        "workloads": results,
        "notes": [
            "Static estimates come from a fake-tensor ATen forward/backward graph and storage liveness.",
            "Real comparisons use allocator allocated peaks; CUDA context memory is outside this metric.",
            "A fixed backend-resident allocator component is measured once with a representative MLP/AdamW probe.",
            "Graph and optimizer phases are compared separately; eager single-tensor AdamW models the current sqrt/divide pair and the previous loop denominator.",
            "The validation models the eager training-step lifetime where the module output remains referenced through backward and optimizer.step().",
            "CUDA-enabled validation traces fake CUDA tensors so device-specific ATen dispatch matches the measured execution path.",
            "CUDA Flash Attention auxiliary storage is estimated from query shape, dtype, and 64-token sequence tiles.",
            "FP32 Efficient Attention backward workspace is evaluated at the operator's graph-liveness point.",
            "Real CUDA peaks are retained separately for forward, backward, and optimizer phases.",
            "Backend operations without a matched profile, allocator fragmentation, and fused/foreach optimizer extras remain unmodeled.",
            "Workspace coverage is based on candidate operator calls and does not bound bytes missing from an unprofiled operator.",
            "Cross-GPU claims require comparing reports from multiple GPU/software profiles.",
        ],
    }


def _measure_real_cuda(
    model: Any,
    args: tuple[Any, ...],
    *,
    warmup_runs: int,
    repeats: int,
) -> dict[str, Any]:
    import torch

    torch.cuda.empty_cache()
    torch.cuda.manual_seed_all(0)
    device = torch.device("cuda", 0)
    model = model.to(device)
    cuda_args = tuple(_move_to_device(value, device) for value in args)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
        foreach=False,
    )

    for _ in range(max(0, int(warmup_runs))):
        _training_step(model, cuda_args, optimizer)
    torch.cuda.synchronize()

    trials: list[dict[str, Any]] = []
    for trial_index in range(max(1, int(repeats))):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        baseline_allocated = int(torch.cuda.memory_allocated())
        baseline_reserved = int(torch.cuda.memory_reserved())
        baseline_requested = int(
            torch.cuda.memory_stats().get("requested_bytes.all.current", 0) or 0
        )

        torch.cuda.reset_peak_memory_stats()
        output = model(*cuda_args)
        loss = _training_loss(output)
        torch.cuda.synchronize()
        stage_peaks = {"forward": _cuda_stage_memory()}

        torch.cuda.reset_peak_memory_stats()
        loss.backward()
        torch.cuda.synchronize()
        stage_peaks["backward"] = _cuda_stage_memory()

        torch.cuda.reset_peak_memory_stats()
        optimizer.step()
        torch.cuda.synchronize()
        stage_peaks["optimizer"] = _cuda_stage_memory()

        peak_stage = max(
            stage_peaks,
            key=lambda name: int(stage_peaks[name]["peak_allocated_bytes"]),
        )
        peak_allocated = max(
            int(stage["peak_allocated_bytes"]) for stage in stage_peaks.values()
        )
        peak_reserved = max(
            int(stage["peak_reserved_bytes"]) for stage in stage_peaks.values()
        )
        requested_stage_values = [
            int(stage["peak_requested_bytes"])
            for stage in stage_peaks.values()
            if stage.get("peak_requested_bytes") is not None
        ]
        peak_requested_stage = (
            max(
                stage_peaks,
                key=lambda name: int(
                    stage_peaks[name].get("peak_requested_bytes", -1) or -1
                ),
            )
            if requested_stage_values
            else None
        )
        loss_value = float(loss.detach().item())
        trials.append(
            {
                "trial_index": trial_index,
                "baseline_allocated_bytes": baseline_allocated,
                "baseline_reserved_bytes": baseline_reserved,
                "baseline_requested_bytes": baseline_requested,
                "peak_stage": peak_stage,
                "peak_requested_stage": peak_requested_stage,
                "stage_peaks": stage_peaks,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "requested_peak_bytes": (
                    max(requested_stage_values) if requested_stage_values else None
                ),
                "loss": loss_value,
            }
        )
        del output, loss

    allocated = [int(item["peak_allocated_bytes"]) for item in trials]
    reserved = [int(item["peak_reserved_bytes"]) for item in trials]
    requested = [
        int(item["requested_peak_bytes"])
        for item in trials
        if item.get("requested_peak_bytes") is not None
    ]
    losses = [float(item["loss"]) for item in trials]
    peak_trial = max(trials, key=lambda item: int(item["peak_allocated_bytes"]))
    requested_peak_trials = [
        item for item in trials if item.get("requested_peak_bytes") is not None
    ]
    peak_requested_trial = (
        max(
            requested_peak_trials,
            key=lambda item: int(item["requested_peak_bytes"]),
        )
        if requested_peak_trials
        else None
    )
    peak_by_stage: dict[str, dict[str, Any]] = {}
    for stage_name in ("forward", "backward", "optimizer"):
        stage_allocated = [
            int(item["stage_peaks"][stage_name]["peak_allocated_bytes"])
            for item in trials
        ]
        stage_requested = [
            int(item["stage_peaks"][stage_name]["peak_requested_bytes"])
            for item in trials
            if item["stage_peaks"][stage_name].get("peak_requested_bytes") is not None
        ]
        peak_by_stage[stage_name] = {
            "peak_allocated_bytes": max(stage_allocated),
            "peak_allocated_summary": _sample_summary(stage_allocated),
            "peak_requested_bytes": (
                max(stage_requested) if stage_requested else None
            ),
            "peak_requested_summary": (
                _sample_summary(stage_requested) if stage_requested else None
            ),
        }
    result = {
        "status": "PASS_MEASURED",
        "trial_count": len(trials),
        "warmup_runs": max(0, int(warmup_runs)),
        "trials": trials,
        "peak_stage": str(peak_trial["peak_stage"]),
        "peak_requested_stage": (
            peak_requested_trial.get("peak_requested_stage")
            if peak_requested_trial is not None
            else None
        ),
        "peak_by_stage": peak_by_stage,
        "peak_allocated_bytes": max(allocated),
        "peak_allocated_summary": _sample_summary(allocated),
        "peak_reserved_bytes": max(reserved),
        "peak_reserved_summary": _sample_summary(reserved),
        "loss_summary": _sample_summary(losses),
    }
    if requested:
        result["requested_peak_bytes"] = max(requested)
        result["requested_peak_summary"] = _sample_summary(requested)

    optimizer.zero_grad(set_to_none=True)
    del optimizer, cuda_args, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _cuda_stage_memory() -> dict[str, int | None]:
    import torch

    stats = torch.cuda.memory_stats()
    requested_peak = stats.get("requested_bytes.all.peak")
    requested_current = stats.get("requested_bytes.all.current")
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "end_allocated_bytes": int(torch.cuda.memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "end_reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_requested_bytes": (
            int(requested_peak) if requested_peak is not None else None
        ),
        "end_requested_bytes": (
            int(requested_current) if requested_current is not None else None
        ),
    }


def _measure_backend_resident() -> dict[str, Any]:
    import torch

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    initial_allocated = int(torch.cuda.memory_allocated())
    initial_reserved = int(torch.cuda.memory_reserved())
    initial_requested = int(
        torch.cuda.memory_stats().get("requested_bytes.all.current", 0) or 0
    )

    spec = _mlp_spec(
        batch_size=8,
        width=256,
        hidden_size=512,
        dtype="float32",
    )
    model, args = spec.factory()
    device = torch.device("cuda", 0)
    model = model.to(device)
    cuda_args = tuple(_move_to_device(value, device) for value in args)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
        foreach=False,
    )
    _training_step(model, cuda_args, optimizer)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    del optimizer, cuda_args, args, model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    resident_allocated = int(torch.cuda.memory_allocated())
    resident_reserved = int(torch.cuda.memory_reserved())
    resident_requested = int(
        torch.cuda.memory_stats().get("requested_bytes.all.current", 0) or 0
    )
    return {
        "method": "representative_mlp_post_release_resident",
        "probe": spec.name,
        "optimizer": "adamw_single_tensor",
        "initial_allocated_bytes": initial_allocated,
        "initial_reserved_bytes": initial_reserved,
        "initial_requested_bytes": initial_requested,
        "resident_allocated_bytes": max(0, resident_allocated - initial_allocated),
        "resident_reserved_bytes": max(0, resident_reserved - initial_reserved),
        "resident_requested_bytes": max(0, resident_requested - initial_requested),
    }


def _training_step(
    model: Any,
    args: tuple[Any, ...],
    optimizer: Any,
    *,
    zero_grad: bool = True,
) -> float:
    if zero_grad:
        optimizer.zero_grad(set_to_none=True)
    output = model(*args)
    loss = _training_loss(output)
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


def _training_loss(output: Any) -> Any:
    return output.float().square().mean()


def _compare_estimate(
    static_estimate: dict[str, Any],
    measured: dict[str, Any] | None,
    *,
    backend_calibration: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if measured is None:
        return None
    static_estimated = int(static_estimate["estimated_peak_bytes"])
    backend_resident_bytes = int(
        (backend_calibration or {}).get("resident_allocated_bytes", 0) or 0
    )
    estimated = static_estimated + max(0, int(backend_resident_bytes))
    actual = int(measured["peak_allocated_bytes"])
    signed_error = estimated - actual
    absolute_error = abs(signed_error)
    raw_signed_error = static_estimated - actual
    raw_absolute_error = abs(raw_signed_error)
    result = {
        "method": "static_graph_plus_backend_resident_calibration",
        "static_estimated_peak_bytes": static_estimated,
        "backend_resident_calibration_bytes": max(0, int(backend_resident_bytes)),
        "calibrated_estimated_peak_bytes": estimated,
        "estimated_peak_bytes": estimated,
        "actual_peak_allocated_bytes": actual,
        "signed_error_bytes": signed_error,
        "absolute_error_bytes": absolute_error,
        "absolute_error_percent": round(100.0 * absolute_error / actual, 6) if actual else None,
        "underestimate_bytes": max(0, actual - estimated),
        "underestimate_percent": (
            round(100.0 * max(0, actual - estimated) / actual, 6)
            if actual
            else None
        ),
        "raw_signed_error_bytes": raw_signed_error,
        "raw_absolute_error_bytes": raw_absolute_error,
        "raw_absolute_error_percent": (
            round(100.0 * raw_absolute_error / actual, 6) if actual else None
        ),
        "raw_underestimate_bytes": max(0, actual - static_estimated),
        "raw_underestimate_percent": (
            round(100.0 * max(0, actual - static_estimated) / actual, 6)
            if actual
            else None
        ),
    }
    requested_peak = measured.get("requested_peak_bytes")
    backend_requested = (backend_calibration or {}).get("resident_requested_bytes")
    if requested_peak is not None and backend_requested is not None:
        requested_actual = int(requested_peak)
        requested_backend = max(0, int(backend_requested))
        requested_estimated = static_estimated + requested_backend
        requested_signed_error = requested_estimated - requested_actual
        result.update(
            {
                "backend_resident_requested_bytes": requested_backend,
                "calibrated_estimated_requested_peak_bytes": requested_estimated,
                "actual_peak_requested_bytes": requested_actual,
                "requested_signed_error_bytes": requested_signed_error,
                "requested_absolute_error_bytes": abs(requested_signed_error),
                "requested_absolute_error_percent": (
                    round(100.0 * abs(requested_signed_error) / requested_actual, 6)
                    if requested_actual
                    else None
                ),
                "requested_underestimate_bytes": max(
                    0,
                    requested_actual - requested_estimated,
                ),
                "requested_underestimate_percent": (
                    round(
                        100.0
                        * max(0, requested_actual - requested_estimated)
                        / requested_actual,
                        6,
                    )
                    if requested_actual
                    else None
                ),
            }
        )
    peak_by_stage = measured.get("peak_by_stage")
    if isinstance(peak_by_stage, dict):
        forward = dict(peak_by_stage.get("forward") or {})
        backward = dict(peak_by_stage.get("backward") or {})
        optimizer = dict(peak_by_stage.get("optimizer") or {})
        graph_stage_name, graph_stage = max(
            (("forward", forward), ("backward", backward)),
            key=lambda item: int(item[1].get("peak_allocated_bytes", 0) or 0),
        )
        result["actual_peak_stage"] = measured.get("peak_stage")
        result["actual_requested_peak_stage"] = measured.get(
            "peak_requested_stage"
        )
        result["estimated_peak_phase"] = static_estimate.get("peak_phase")
        result["phase_comparison"] = {
            "graph": _compare_phase_peak(
                static_peak_bytes=int(static_estimate["graph_phase_peak_bytes"]),
                actual_allocated_bytes=int(
                    graph_stage.get("peak_allocated_bytes", 0) or 0
                ),
                actual_requested_bytes=graph_stage.get("peak_requested_bytes"),
                backend_allocated_bytes=backend_resident_bytes,
                backend_requested_bytes=backend_requested,
                actual_substage=graph_stage_name,
            ),
            "optimizer": _compare_phase_peak(
                static_peak_bytes=int(
                    static_estimate.get("optimizer_phase_peak_bytes", 0) or 0
                ),
                actual_allocated_bytes=int(
                    optimizer.get("peak_allocated_bytes", 0) or 0
                ),
                actual_requested_bytes=optimizer.get("peak_requested_bytes"),
                backend_allocated_bytes=backend_resident_bytes,
                backend_requested_bytes=backend_requested,
                actual_substage="optimizer",
            ),
        }
    return result


def _compare_phase_peak(
    *,
    static_peak_bytes: int,
    actual_allocated_bytes: int,
    actual_requested_bytes: Any,
    backend_allocated_bytes: int,
    backend_requested_bytes: Any,
    actual_substage: str,
) -> dict[str, Any]:
    estimated_allocated = static_peak_bytes + max(0, int(backend_allocated_bytes))
    allocated_signed_error = estimated_allocated - actual_allocated_bytes
    result = {
        "actual_substage": actual_substage,
        "static_peak_bytes": static_peak_bytes,
        "calibrated_estimated_allocated_bytes": estimated_allocated,
        "actual_peak_allocated_bytes": actual_allocated_bytes,
        "allocated_signed_error_bytes": allocated_signed_error,
        "allocated_underestimate_bytes": max(0, -allocated_signed_error),
    }
    if actual_requested_bytes is not None and backend_requested_bytes is not None:
        estimated_requested = static_peak_bytes + max(
            0,
            int(backend_requested_bytes),
        )
        requested_signed_error = estimated_requested - int(actual_requested_bytes)
        result.update(
            {
                "calibrated_estimated_requested_bytes": estimated_requested,
                "actual_peak_requested_bytes": int(actual_requested_bytes),
                "requested_signed_error_bytes": requested_signed_error,
                "requested_underestimate_bytes": max(0, -requested_signed_error),
            }
        )
    return result


def _comparison_summary(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    if not comparisons:
        return {
            "measured_workload_count": 0,
            "underestimated_workload_count": 0,
            "max_underestimate_percent": None,
            "median_absolute_error_percent": None,
            "requested_underestimated_workload_count": 0,
            "max_requested_underestimate_percent": None,
            "median_requested_absolute_error_percent": None,
            "max_requested_absolute_error_percent": None,
        }
    underestimates = [float(item["underestimate_percent"]) for item in comparisons]
    absolute_errors = [float(item["absolute_error_percent"]) for item in comparisons]
    requested_comparisons = [
        item
        for item in comparisons
        if item.get("requested_underestimate_percent") is not None
        and item.get("requested_absolute_error_percent") is not None
    ]
    requested_underestimates = [
        float(item["requested_underestimate_percent"])
        for item in requested_comparisons
    ]
    requested_absolute_errors = [
        float(item["requested_absolute_error_percent"])
        for item in requested_comparisons
    ]
    return {
        "measured_workload_count": len(comparisons),
        "underestimated_workload_count": sum(value > 0 for value in underestimates),
        "max_underestimate_percent": round(max(underestimates), 6),
        "median_absolute_error_percent": round(statistics.median(absolute_errors), 6),
        "max_absolute_error_percent": round(max(absolute_errors), 6),
        "requested_underestimated_workload_count": sum(
            value > 0 for value in requested_underestimates
        ),
        "max_requested_underestimate_percent": (
            round(max(requested_underestimates), 6)
            if requested_underestimates
            else None
        ),
        "median_requested_absolute_error_percent": (
            round(statistics.median(requested_absolute_errors), 6)
            if requested_absolute_errors
            else None
        ),
        "max_requested_absolute_error_percent": (
            round(max(requested_absolute_errors), 6)
            if requested_absolute_errors
            else None
        ),
    }


def _sample_summary(values: list[int | float]) -> dict[str, int | float]:
    if not values:
        raise ValueError("sample summary requires at least one value")
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "max": ordered[-1],
        "range": ordered[-1] - ordered[0],
    }


def _gpu_fingerprint() -> dict[str, Any]:
    import torch

    props = torch.cuda.get_device_properties(0)
    return {
        "name": str(props.name),
        "total_memory": int(props.total_memory),
        "compute_capability": f"{int(props.major)}.{int(props.minor)}",
        "sm_count": int(props.multi_processor_count),
    }


def _move_to_device(value: Any, device: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _workloads() -> dict[str, WorkloadSpec]:
    return {
        spec.name: spec
        for spec in (
            _mlp_spec(batch_size=8, width=256, hidden_size=512, dtype="float32"),
            _mlp_spec(batch_size=32, width=512, hidden_size=2048, dtype="bfloat16"),
            _transformer_spec(
                batch_size=1,
                seq_len=16,
                hidden_size=64,
                layers=1,
                dtype="float32",
            ),
            _transformer_spec(
                batch_size=2,
                seq_len=16,
                hidden_size=64,
                layers=1,
                dtype="float32",
            ),
            _transformer_spec(
                batch_size=2,
                seq_len=32,
                hidden_size=64,
                layers=1,
                dtype="float32",
            ),
            _transformer_spec(
                batch_size=4,
                seq_len=32,
                hidden_size=64,
                layers=1,
                dtype="float32",
            ),
            _transformer_spec(
                batch_size=2,
                seq_len=64,
                hidden_size=64,
                layers=1,
                dtype="float32",
            ),
            _transformer_spec(
                batch_size=2,
                seq_len=32,
                hidden_size=128,
                layers=1,
                dtype="float32",
            ),
            _transformer_spec(
                batch_size=2,
                seq_len=32,
                hidden_size=64,
                layers=2,
                dtype="float32",
            ),
            _transformer_spec(
                batch_size=3,
                seq_len=33,
                hidden_size=96,
                layers=1,
                dtype="float32",
            ),
            _transformer_spec(
                batch_size=4,
                seq_len=64,
                hidden_size=128,
                layers=2,
                dtype="bfloat16",
            ),
            _transformer_spec(
                batch_size=2,
                seq_len=64,
                hidden_size=128,
                layers=1,
                dtype="bfloat16",
            ),
            _transformer_spec(
                batch_size=2,
                seq_len=128,
                hidden_size=128,
                layers=1,
                dtype="bfloat16",
            ),
        )
    }


def _mlp_spec(
    *,
    batch_size: int,
    width: int,
    hidden_size: int,
    dtype: str,
) -> WorkloadSpec:
    name = f"mlp_b{batch_size}_d{width}_h{hidden_size}_{dtype}"

    def factory() -> tuple[Any, tuple[Any, ...]]:
        import torch

        selected_dtype = getattr(torch, dtype)
        model = torch.nn.Sequential(
            torch.nn.Linear(width, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, width),
        ).to(dtype=selected_dtype)
        inputs = torch.randn((batch_size, width), dtype=selected_dtype)
        return model, (inputs,)

    return WorkloadSpec(
        name=name,
        family="mlp",
        parameters={
            "batch_size": batch_size,
            "width": width,
            "hidden_size": hidden_size,
            "dtype": dtype,
        },
        factory=factory,
    )


def _transformer_spec(
    *,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    layers: int,
    dtype: str,
) -> WorkloadSpec:
    name = (
        f"transformer_b{batch_size}_s{seq_len}_h{hidden_size}_l{layers}_{dtype}"
    )

    def factory() -> tuple[Any, tuple[Any, ...]]:
        import torch

        selected_dtype = getattr(torch, dtype)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        model = torch.nn.TransformerEncoder(layer, num_layers=layers).to(dtype=selected_dtype)
        inputs = torch.randn(
            (batch_size, seq_len, hidden_size),
            dtype=selected_dtype,
        )
        return model, (inputs,)

    return WorkloadSpec(
        name=name,
        family="transformer",
        parameters={
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_size": hidden_size,
            "layers": layers,
            "dtype": dtype,
            "attention_heads": 4,
        },
        factory=factory,
    )


def render_markdown(report: dict[str, Any]) -> str:
    workspace_coverage = dict(
        (report.get("summary") or {}).get("workspace_coverage") or {}
    )
    lines = [
        "# Static Memory Estimator Validation",
        "",
        f"**Status:** `{report.get('status')}`",
        (
            "**Workspace coverage:** "
            f"{float(workspace_coverage.get('modeled_fraction', 1.0)):.1%} "
            f"({int(workspace_coverage.get('modeled_operator_count', 0))}/"
            f"{int(workspace_coverage.get('candidate_operator_count', 0))} "
            "candidate calls)"
        ),
        "",
    ]
    gpu = report.get("gpu")
    if isinstance(gpu, dict):
        lines.extend(
            [
                "## GPU",
                "",
                f"- name: `{gpu.get('name')}`",
                f"- compute capability: `{gpu.get('compute_capability')}`",
                f"- total memory: `{_fmt_bytes(int(gpu.get('total_memory', 0) or 0))}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Results",
            "",
            "| workload | family | static peak | calibrated peak | real allocator peak | real phase | signed error | underestimate | requested error | graph |",
            "|---|---|---:|---:|---:|---|---:|---:|---:|---|",
        ]
    )
    for item in report.get("workloads", []):
        static = item.get("static_estimate") or {}
        real = item.get("real_cuda") or {}
        comparison = item.get("comparison") or {}
        graph = static.get("graph") or {}
        lines.append(
            "| `{name}` | `{family}` | {static_peak} | {calibrated_peak} | {real_peak} | `{real_phase}` | {signed} | {underestimate} | {requested_error} | "
            "{nodes} nodes / {aliases} aliases |".format(
                name=item.get("name"),
                family=item.get("family"),
                static_peak=_fmt_bytes(int(static.get("estimated_peak_bytes", 0) or 0)),
                calibrated_peak=(
                    _fmt_bytes(
                        int(comparison.get("calibrated_estimated_peak_bytes", 0) or 0)
                    )
                    if comparison
                    else "n/a"
                ),
                real_peak=(
                    _fmt_bytes(int(real.get("peak_allocated_bytes", 0) or 0))
                    if real
                    else "n/a"
                ),
                real_phase=real.get("peak_stage") if real else "n/a",
                signed=(
                    _fmt_signed_bytes(int(comparison.get("signed_error_bytes", 0) or 0))
                    if comparison
                    else "n/a"
                ),
                underestimate=(
                    f"{float(comparison.get('underestimate_percent', 0.0)):.2f}%"
                    if comparison
                    else "n/a"
                ),
                requested_error=(
                    _fmt_signed_bytes(
                        int(comparison.get("requested_signed_error_bytes", 0) or 0)
                    )
                    if comparison.get("requested_signed_error_bytes") is not None
                    else "n/a"
                ),
                nodes=int(graph.get("node_count", 0) or 0),
                aliases=int(graph.get("alias_node_count", 0) or 0),
            )
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "- Static peak is the larger of the graph phase and optimizer phase, including persistent AdamW state and matched backend operator profiles.",
            "- Operator-local workspace is evaluated against the storage live at that ATen node; persistent auxiliary storage remains graph-wide.",
            "- Eager single-tensor AdamW models two current-parameter intermediates plus the previous loop denominator.",
            "- Calibrated peak adds one measured backend-resident allocation for the current GPU/software stack.",
            "- Real peak uses `torch.cuda.max_memory_allocated()`.",
            "- Reports also retain requested-byte comparisons when the allocator exposes `requested_bytes.all.peak`.",
            "- Real measurements retain separate forward, backward, and optimizer peaks.",
            "- CUDA context, allocator fragmentation, unmatched operation workspaces, and fused/foreach optimizer extras are listed as unmodeled.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    unit = units[0]
    for candidate in units:
        unit = candidate
        if abs(amount) < 1024.0 or candidate == units[-1]:
            break
        amount /= 1024.0
    return f"{amount:.2f} {unit}"


def _fmt_signed_bytes(value: int) -> str:
    sign = "+" if value >= 0 else "-"
    return sign + _fmt_bytes(abs(value))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative finite number")
    return parsed


def _fraction(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be a finite number between 0 and 1")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
