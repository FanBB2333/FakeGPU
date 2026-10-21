"""Exact-workload native phase envelopes, with separate allocated/reserved peaks.

No shape transfer, fitted safety factor or allocator guarantee is implied.
Budgets must be sealed before an independent holdout starts.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import math
from statistics import fmean
from uuid import UUID

from ._finetune_workflow_store import _now
from ._finetune_allocator import validate_memory_allocator
from .structured_io import canonical_sha256 as _digest

SAMPLE_SCHEMA = "fakegpu.phase_memory_sample.v1"
BUDGET_SCHEMA = "fakegpu.phase_memory_budget.v1"
METRICS = ("allocated", "reserved")


def _time(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("phase memory timestamps require an explicit timezone")
    return result


def _sample(sample):
    if sample.get("schema_version") != SAMPLE_SCHEMA:
        raise ValueError("unsupported native phase memory sample")
    UUID(sample["run_id"])
    if _digest(sample["prediction"]) != sample["prediction_sha256"]:
        raise ValueError("sample prediction identity mismatch")
    if _digest(sample["identity"]) != sample["identity_sha256"]:
        raise ValueError("sample native identity mismatch")
    identity = sample["identity"]
    validate_memory_allocator(identity["allocator"])
    environment = identity["environment"]
    if (
        environment.get("simulated") is not False
        or environment.get("backend") != "cuda"
        or not environment.get("gpu_uuid")
    ):
        raise ValueError("phase samples require an identified real CUDA GPU")
    if identity["allocator"]["backend"] != "native":
        raise ValueError(
            "phase envelopes currently require the native caching allocator"
        )
    if not identity.get("model_files") or not identity.get("implementation"):
        raise ValueError(
            "phase samples require content-bound model and implementation identities"
        )
    if (
        type(sample["process"]["pid"]) is not int
        or sample["process"]["pid"] <= 0
        or not sample["process"].get("start_ticks")
    ):
        raise ValueError("phase samples require an independent process identity")
    if _time(sample["finished_at_utc"]) < _time(sample["started_at_utc"]):
        raise ValueError("sample finished before it started")
    expected = [p["phase"] for p in sample["prediction"]["memory_timeline"]["phases"]]
    if (
        not expected
        or len(set(expected)) != len(expected)
        or set(sample["phases"]) != set(expected)
    ):
        raise ValueError("native sample must cover every predicted phase exactly")
    for phase in sample["phases"].values():
        if (
            set(phase) != set(METRICS)
            or any(type(v) is not int or v <= 0 for v in phase.values())
            or phase["reserved"] < phase["allocated"]
        ):
            raise ValueError(
                "phase peaks must be positive bytes with reserved >= allocated"
            )
    return sample


def _process(sample):
    return (sample["process"]["pid"], sample["process"]["start_ticks"])


def estimate_phase_memory(calibration_samples):
    """Build a five-process exact-workload envelope using observed peaks only."""
    samples = [_sample(s) for s in calibration_samples]
    if len(samples) < 5:
        raise ValueError(
            "phase memory calibration requires at least five independent processes"
        )
    if len({s["run_id"] for s in samples}) != len(samples) or len(
        {_process(s) for s in samples}
    ) != len(samples):
        raise ValueError("calibration processes cannot be reused")
    first = samples[0]
    if any(
        s["prediction_sha256"] != first["prediction_sha256"]
        or s["identity_sha256"] != first["identity_sha256"]
        for s in samples
    ):
        raise ValueError(
            "calibration requires the same plan, model content, GPU, software, allocator and implementation"
        )
    created = _now()
    if any(_time(s["finished_at_utc"]) > _time(created) for s in samples):
        raise ValueError("calibration sample has not finished")
    phases = {}
    for phase in first["phases"]:
        phases[phase] = {}
        for metric in METRICS:
            values = [s["phases"][phase][metric] for s in samples]
            phases[phase][metric] = {
                "min": min(values),
                "max": max(values),
                "mean": fmean(values),
                "samples": values,
            }
    report = {
        "schema_version": BUDGET_SCHEMA,
        "created_at_utc": created,
        "prediction": deepcopy(first["prediction"]),
        "prediction_sha256": first["prediction_sha256"],
        "identity": deepcopy(first["identity"]),
        "identity_sha256": first["identity_sha256"],
        "calibration_processes": [
            {
                "run_id": s["run_id"],
                "process": deepcopy(s["process"]),
                "sample_sha256": _digest(s),
            }
            for s in samples
        ],
        "phases": phases,
        "peak_bytes": {
            metric: max(value[metric]["max"] for value in phases.values())
            for metric in METRICS
        },
        "scope": "Observed per-phase allocated/reserved maxima for this exact workload and native identity. No shape transfer, confidence interval, physical OOM guarantee or hidden safety factor.",
    }
    report["budget_sha256"] = _digest(report)
    return report


def validate_phase_memory(
    budget, holdout, *, max_absolute_error_percent=20.0, max_underestimate_percent=10.0
):
    """Check an independent process against its pre-existing sealed budget."""
    if budget.get("schema_version") != BUDGET_SCHEMA or _digest(
        {k: v for k, v in budget.items() if k != "budget_sha256"}
    ) != budget.get("budget_sha256"):
        raise ValueError("invalid phase memory budget seal")
    holdout = _sample(holdout)
    if (
        holdout["prediction_sha256"] != budget["prediction_sha256"]
        or holdout["identity_sha256"] != budget["identity_sha256"]
    ):
        raise ValueError("holdout plan or native identity differs from calibration")
    if _time(budget["created_at_utc"]) >= _time(holdout["started_at_utc"]):
        raise ValueError("budget must be sealed before the holdout starts")
    if any(
        s["run_id"] == holdout["run_id"]
        or tuple(s["process"][k] for k in ("pid", "start_ticks")) == _process(holdout)
        for s in budget["calibration_processes"]
    ):
        raise ValueError("holdout must be independent of calibration processes")
    limits = {
        "max_absolute_error_percent": max_absolute_error_percent,
        "max_underestimate_percent": max_underestimate_percent,
    }
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or v < 0
        for v in limits.values()
    ):
        raise ValueError("memory validation limits must be finite and nonnegative")
    comparisons = {}
    for phase, actuals in holdout["phases"].items():
        comparisons[phase] = {}
        for metric, actual in actuals.items():
            predicted = budget["phases"][phase][metric]["max"]
            comparisons[phase][metric] = {
                "predicted": predicted,
                "actual": actual,
                "absolute_percentage_error": abs(predicted - actual) / actual * 100,
                "underprediction_percent": max(0, actual - predicted) / actual * 100,
                "exceeds_observed_max": actual > predicted,
            }
    passed = all(
        c["absolute_percentage_error"] <= max_absolute_error_percent
        and c["underprediction_percent"] <= max_underestimate_percent
        for p in comparisons.values()
        for c in p.values()
    )
    return {
        "schema_version": "fakegpu.phase_memory_validation.v1",
        "budget_sha256": budget["budget_sha256"],
        "holdout_run_id": holdout["run_id"],
        "comparison": comparisons,
        "limits": limits,
        "passed": passed,
        "oom_safety_verified": False,
    }


__all__ = ["estimate_phase_memory", "validate_phase_memory"]
