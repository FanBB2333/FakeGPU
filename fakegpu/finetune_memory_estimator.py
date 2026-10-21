"""Estimate a covered native workload's empirical CUDA allocator peak envelope.

Only complete independent runs contribute memory measurements. Peaks from
individual windows or events are never added, and fitting inside an observed
envelope does not establish that a future process will avoid CUDA OOM.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from ._finetune_pipeline_plan import (
    describe_pipeline_components,
    pipeline_plan_from_run,
    validate_pipeline_plan,
)
from .finetune_estimator import FinetuneEstimateError
from .finetune_window_estimator import _run_identity
from .structured_io import canonical_sha256 as _digest


SCHEMA_VERSION = "fakegpu.finetune_memory_estimate.v1"
COMPARISON_SCHEMA_VERSION = "fakegpu.finetune_memory_comparison.v1"
_MEMORY_FIELDS = ("peak_allocated_bytes", "peak_reserved_bytes")
_COMPATIBILITY_FIELDS = (
    "signature",
    "hardware",
    "cache_regime",
    "setup_workload",
    "memory_allocator",
    "execution_policy",
)


def _byte_count(value: Any, name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FinetuneEstimateError(f"{name} must be an integer >= {minimum}")
    return value


def _native_sample(run: Mapping[str, Any]) -> dict[str, Any]:
    plan = pipeline_plan_from_run(run)
    observation = run["observation"]
    peaks = {
        field: _byte_count(observation.get(field), field) for field in _MEMORY_FIELDS
    }
    if peaks["peak_reserved_bytes"] < peaks["peak_allocated_bytes"]:
        raise FinetuneEstimateError("reserved allocator peak must cover allocated peak")
    if peaks["peak_reserved_bytes"] > plan["hardware"]["total_memory_bytes"]:
        raise FinetuneEstimateError(
            "allocator peak exceeds recorded physical GPU memory"
        )
    return {
        "identity": _run_identity(run),
        "plan": plan,
        "peaks": peaks,
        "counts": Counter(
            component["group_id"] for component in describe_pipeline_components(plan)
        ),
        # Renaming a copied observation cannot create another independent sample.
        "evidence_sha256": _digest(
            {"observation": observation, "timing": run["timing"]}
        ),
    }


def estimate_finetuning_memory(
    calibration_runs: Sequence[Mapping[str, Any]],
    target_plan: Mapping[str, Any],
    *,
    memory_budget_bytes: int | None = None,
    reserve_bytes: int = 0,
) -> dict[str, Any]:
    """Return full-run allocator peak ranges for a completely covered CPU plan.

    Every semantic component, including warmup, first occurrences, events,
    setup and finalization, must occur in each of at least two independent
    calibration processes. ``reserve_bytes`` is a caller-selected margin, not
    a fitted uncertainty bound. A budget comparison is not an OOM experiment.
    """
    if (
        not isinstance(calibration_runs, Sequence)
        or isinstance(calibration_runs, (str, bytes))
        or len(calibration_runs) < 2
    ):
        raise FinetuneEstimateError("memory calibration requires two independent runs")
    reserve_bytes = _byte_count(reserve_bytes, "reserve_bytes", allow_zero=True)
    plan = validate_pipeline_plan(target_plan)
    physical_bytes = _byte_count(
        plan["hardware"].get("total_memory_bytes"), "hardware.total_memory_bytes"
    )
    allocator_limit = int(
        physical_bytes * plan["memory_allocator"]["per_process_memory_fraction"]
    )
    if memory_budget_bytes is not None:
        memory_budget_bytes = _byte_count(memory_budget_bytes, "memory_budget_bytes")
        if memory_budget_bytes > physical_bytes:
            raise FinetuneEstimateError(
                "memory budget exceeds recorded physical GPU memory"
            )
        if memory_budget_bytes > allocator_limit:
            raise FinetuneEstimateError(
                "memory budget exceeds the recorded allocator cap"
            )
    samples = [_native_sample(run) for run in calibration_runs]
    for key, values in (
        ("UUID", [sample["identity"]["uuid"] for sample in samples]),
        (
            "process identity",
            [
                (sample["identity"]["pid"], sample["identity"]["started_at_utc"])
                for sample in samples
            ],
        ),
        ("observation", [sample["evidence_sha256"] for sample in samples]),
    ):
        if len(set(values)) != len(values):
            raise FinetuneEstimateError(f"memory calibration repeats a {key}")
    for sample in samples:
        for field in _COMPATIBILITY_FIELDS:
            if sample["plan"][field] != plan[field]:
                raise FinetuneEstimateError(
                    f"memory calibration {field} does not match target"
                )

    components = describe_pipeline_components(plan)
    grouped = {}
    for component in components:
        group = grouped.setdefault(
            component["group_id"],
            {
                "group_id": component["group_id"],
                "kind": component["kind"],
                "descriptor": deepcopy(component["descriptor"]),
                "target_components": 0,
            },
        )
        group["target_components"] += 1
    for group_id, group in grouped.items():
        group["calibration_runs"] = []
        for sample in samples:
            count = sample["counts"][group_id]
            if count < 1:
                raise FinetuneEstimateError(
                    "memory target contains an uncovered component "
                    f"({group['kind']}, {group_id}) in calibration "
                    f"{sample['identity']['uuid']}"
                )
            group["calibration_runs"].append(
                {"run_uuid": sample["identity"]["uuid"], "samples": count}
            )
    envelope = {
        field: {
            "min": min(sample["peaks"][field] for sample in samples),
            "max": max(sample["peaks"][field] for sample in samples),
        }
        for field in _MEMORY_FIELDS
    }
    planning_bytes = envelope["peak_reserved_bytes"]["max"] + reserve_bytes
    budget = {
        "memory_budget_bytes": memory_budget_bytes,
        "reserve_bytes": reserve_bytes,
        "recorded_allocator_limit_bytes": allocator_limit,
        "observed_reserved_peak_plus_reserve_bytes": planning_bytes,
        "fits_recorded_allocator_cap": planning_bytes <= allocator_limit,
        "status": "not_requested"
        if memory_budget_bytes is None
        else (
            "observed_envelope_fits_budget"
            if planning_bytes <= memory_budget_bytes
            else "observed_envelope_exceeds_budget"
        ),
        "oom_safety_verified": False,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "covered_native_process_allocator_peak_envelope",
        **{field: plan[field] for field in _COMPATIBILITY_FIELDS},
        "target_plan": {
            "source": plan["source"],
            "sha256": _digest(plan),
            "workload": plan["workload"],
        },
        "calibration_runs": [
            {
                **sample["identity"],
                **sample["peaks"],
                "evidence_sha256": sample["evidence_sha256"],
                "workload": sample["plan"]["workload"],
            }
            for sample in samples
        ],
        "coverage": {
            "target_components": len(components),
            "target_component_groups": len(grouped),
            "minimum_samples_per_group_per_calibration_run": 1,
            "includes_warmup_and_first_occurrences": True,
            "complete": True,
        },
        "component_groups": list(grouped.values()),
        "memory": envelope,
        "budget": budget,
        "limitations": [
            "The min/max range contains observed complete-run CUDA allocator peaks; "
            "it is not a confidence interval or a bound on future allocations.",
            "All target semantic components must be observed in every calibration "
            "run; window or event peaks are neither inferred nor summed.",
            "The allocated and reserved peaks can occur at different times. "
            "Reserved memory includes allocator caching and is not a minimum "
            "memory requirement.",
            "CUDA allocator statistics exclude allocations by other processes, "
            "the driver and libraries outside the PyTorch allocator.",
            "Matching shapes, event and epoch contexts does not prove identical "
            "allocator state, fragmentation or memory growth in longer runs.",
            "A requested budget comparison or caller-selected reserve does not "
            "establish physical-GPU OOM safety; actual OOM requires a separate "
            "failed execution with an observed CUDA allocation error.",
        ],
    }


def compare_finetuning_memory(
    calibration_runs: Sequence[Mapping[str, Any]],
    observed_run: Mapping[str, Any],
    *,
    memory_budget_bytes: int | None = None,
    reserve_bytes: int = 0,
) -> dict[str, Any]:
    """Validate an independent holdout without using its peaks in the estimate."""
    target = _native_sample(observed_run)
    estimate = estimate_finetuning_memory(
        calibration_runs,
        target["plan"],
        memory_budget_bytes=memory_budget_bytes,
        reserve_bytes=reserve_bytes,
    )
    for sample in estimate["calibration_runs"]:
        if (
            sample["uuid"] == target["identity"]["uuid"]
            or (sample["pid"], sample["started_at_utc"])
            == (target["identity"]["pid"], target["identity"]["started_at_utc"])
            or sample["evidence_sha256"] == target["evidence_sha256"]
        ):
            raise FinetuneEstimateError(
                "memory holdout must be an independent native run"
            )
    comparison = {}
    for field in _MEMORY_FIELDS:
        predicted = estimate["memory"][field]["max"]
        actual = target["peaks"][field]
        comparison[field] = {
            "observed_calibration_max": predicted,
            "actual": actual,
            "within_observed_envelope": estimate["memory"][field]["min"]
            <= actual
            <= predicted,
            "exceeds_observed_max": actual > predicted,
            "absolute_percentage_error": abs(predicted - actual) / actual * 100,
            "underprediction_percent": max(actual - predicted, 0) / actual * 100,
        }
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "estimate": estimate,
        "target_run": target["identity"],
        "comparison": comparison,
    }
