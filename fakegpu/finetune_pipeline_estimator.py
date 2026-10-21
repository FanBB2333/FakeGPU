"""Calibrate a complete native execution partition without target timing fits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
import statistics
from typing import Any

from ._finetune_pipeline_plan import (
    _WORK_KINDS,
    _digest,
    describe_pipeline_components,
    pipeline_plan_from_run,
    validate_pipeline_plan,
)
from .finetune_estimator import FinetuneEstimateError
from .finetune_window_estimator import _require_calibration_runs, _run_identity

SCHEMA_VERSION = "fakegpu.finetune_pipeline_estimate.v1"
_FIELDS = ("wall_seconds", "instrumentation_seconds", "seconds")
_MATCH_FIELDS = (
    "signature",
    "hardware",
    "memory_allocator",
    "execution_policy",
    "cache_regime",
    "setup_workload",
)


def _samples(run: Mapping[str, Any]) -> dict[str, Any]:
    plan = pipeline_plan_from_run(run)
    identity = _run_identity(run)
    timeline = run["observation"]["timeline"]
    components = describe_pipeline_components(plan)
    work = {}
    gaps = {}
    for segment in timeline["segments"]:
        if segment["kind"] in _WORK_KINDS:
            field = (
                "optimizer_step"
                if segment["kind"] == "optimizer_update"
                else "event_id"
            )
            work[(segment["kind"], segment[field])] = segment
        else:
            gaps[
                (segment["kind"], segment["start_seconds"], segment["end_seconds"])
            ] = segment

    def bounds(reference):
        kind = reference["kind"]
        boundaries = {
            "native_start": 0.0,
            "native_end": timeline["span"]["end_seconds"],
            "trainer_start": timeline["trainer"]["start_seconds"],
            "trainer_end": timeline["trainer"]["end_seconds"],
        }
        if kind in boundaries:
            return boundaries[kind], boundaries[kind]
        field = "optimizer_step" if kind == "optimizer_update" else "event_id"
        segment = work[(kind, reference[field])]
        return segment["start_seconds"], segment["end_seconds"]

    groups = {}
    for component in components:
        reference, kind = component["reference"], component["kind"]
        if kind in _WORK_KINDS:
            field = "optimizer_step" if kind == "optimizer_update" else "event_id"
            segment = work[(kind, reference[field])]
            values = {field: segment[field] for field in _FIELDS}
        else:
            start, end = bounds(reference["left"])[1], bounds(reference["right"])[0]
            if end < start:
                raise FinetuneEstimateError("pipeline semantic partition overlaps")
            if start == end:
                values = dict.fromkeys(_FIELDS, 0.0)
            else:
                segment = gaps.get((kind, start, end))
                if segment is None:
                    raise FinetuneEstimateError(
                        "pipeline gap differs from the native partition"
                    )
                values = {field: segment[field] for field in _FIELDS}
        component["observed"] = values
        groups.setdefault(component["group_id"], []).append(component)
    for field in _FIELDS:
        total = sum(component["observed"][field] for component in components)
        if not math.isclose(
            total, timeline["totals"][field], rel_tol=1e-9, abs_tol=1e-9
        ):
            raise FinetuneEstimateError(
                "pipeline semantic components do not cover the full timeline"
            )
    return {
        "plan": plan,
        "identity": identity,
        "evidence_sha256": _digest(
            {"observation": run["observation"], "timing": run["timing"]}
        ),
        "groups": groups,
        "components": components,
    }


def _independent(runs):
    if len({run["identity"]["uuid"] for run in runs}) != len(runs):
        raise FinetuneEstimateError("pipeline calibration requires distinct run UUIDs")
    if len({run["evidence_sha256"] for run in runs}) != len(runs):
        raise FinetuneEstimateError(
            "pipeline calibration contains duplicate timing evidence"
        )


def _interval(processes, count):
    per_component = {}
    for field in _FIELDS:
        values = [process["mean"][field] for process in processes]
        per_component[field] = {
            "min": min(values),
            "expected": statistics.mean(values),
            "max": max(values),
        }
    predicted = {
        field: {name: value * count for name, value in interval.items()}
        for field, interval in per_component.items()
    }
    return per_component, predicted


def _totals(groups):
    return {
        field: {
            name: sum(group["predicted"][field][name] for group in groups)
            for name in ("min", "expected", "max")
        }
        for field in _FIELDS
    }


def _timing(groups, components):
    return {
        **_totals(groups),
        "by_kind": {
            kind: _totals([group for group in groups if group["kind"] == kind])
            for kind in sorted({component["kind"] for component in components})
        },
        "interval_kind": "sum_of_component_process_mean_extrema_not_confidence_interval",
    }


def _estimate(calibrations, plan):
    _independent(calibrations)
    for field in _MATCH_FIELDS:
        if any(run["plan"][field] != plan[field] for run in calibrations):
            raise FinetuneEstimateError(
                f"pipeline calibration requires matching {field}"
            )
    components = describe_pipeline_components(plan)
    target_groups = {}
    for component in components:
        target_groups.setdefault(component["group_id"], []).append(component)
    groups, missing = [], []
    for group_id, target in target_groups.items():
        minimum = target[0]["minimum_samples_per_run"]
        per_process = []
        for run in calibrations:
            samples = run["groups"].get(group_id, [])
            if len(samples) < minimum:
                missing.append(
                    f"{target[0]['kind']} {group_id}, run {run['identity']['uuid']}: {len(samples)} samples (requires {minimum})"
                )
                continue
            per_process.append(
                {
                    "run_uuid": run["identity"]["uuid"],
                    "samples": len(samples),
                    "component_ids": [sample["component_id"] for sample in samples],
                    "mean": {
                        field: statistics.mean(
                            sample["observed"][field] for sample in samples
                        )
                        for field in _FIELDS
                    },
                }
            )
        if len(per_process) != len(calibrations):
            continue
        per_component, predicted = _interval(per_process, len(target))
        groups.append(
            {
                "group_id": group_id,
                "kind": target[0]["kind"],
                "descriptor": deepcopy(target[0]["descriptor"]),
                "minimum_samples_per_run": minimum,
                "calibration_runs": per_process,
                "target_components": len(target),
                "target_component_ids": [
                    component["component_id"] for component in target
                ],
                "per_component": per_component,
                "predicted": predicted,
            }
        )
    if missing:
        raise FinetuneEstimateError(
            "insufficient independent pipeline coverage: " + "; ".join(missing)
        )
    costs = {group["group_id"]: group["per_component"] for group in groups}
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "planned_native_execution_span",
        "signature": deepcopy(plan["signature"]),
        "hardware": deepcopy(plan["hardware"]),
        "memory_allocator": deepcopy(plan["memory_allocator"]),
        "execution_policy": deepcopy(plan["execution_policy"]),
        "cache_regime": deepcopy(plan["cache_regime"]),
        "setup_workload": deepcopy(plan["setup_workload"]),
        "calibration_runs": [
            {
                "identity": deepcopy(run["identity"]),
                "evidence_sha256": run["evidence_sha256"],
                "components": len(run["components"]),
            }
            for run in calibrations
        ],
        "target_plan": {
            "sha256": _digest(plan),
            **{
                field: deepcopy(plan[field])
                for field in ("source", "warmup", "workload")
            },
        },
        "sampling_policy": {
            "minimum_calibration_runs": 2,
            "minimum_samples_per_repeated_group_per_run": 2,
            "minimum_samples_per_first_warmup_initial_final_group_per_run": 1,
            "aggregation": "equal_weight_per_run_component_mean",
            "window_context": "complete_batch_event_epoch_with_warmup_index_and_first_occurrence",
            "gap_context": "both_adjacent_complete_work_descriptors_and_trainer_boundaries",
            "cache_context": "exact_recorded_kernel_cache_regime",
        },
        "coverage": {
            "complete": True,
            "target_components": len(components),
            "target_groups": len(groups),
            "excluded_components": 0,
        },
        "component_groups": groups,
        "target_components": [
            {
                field: deepcopy(component[field])
                for field in ("component_id", "kind", "group_id", "reference")
            }
            | {"predicted": deepcopy(costs[component["group_id"]])}
            for component in components
        ],
        "timing": _timing(groups, components),
        "limitations": [
            "The native execution span ends after native execution and final synchronization, before report construction and CLI output; it is not process end-to-end time.",
            "wall_seconds includes observation overhead. seconds removes only measured batch, event and callback metadata; checkpoint hashing and unmeasured instrumentation remain included.",
            "Every planned component must match all calibration processes. First occurrences and warmup are calibrated separately, not dropped or interpreted as cache state.",
            "Cache regimes describe named kernel-cache directories and their recorded file contents. They do not establish OS page-cache state, dataset caches, cache hits, clocks or system load; uncontrolled regimes provide no cache control.",
            "Gap predictions repeat only matching adjacent semantic boundaries. They are empirical conditional costs, not a fixed leftover overhead or a causal attribution of interference.",
            "A timing-free plan does not authenticate its claimed native sampler or event schedule. Future schedule generation needs a separately validated planner and unchanged data and execution semantics.",
            "Process identities and content hashes detect duplicate evidence but do not authenticate reports or independently prove process isolation. Timing bounds are component mean extrema, not statistical confidence intervals.",
        ],
    }


def _calibrations(calibration_runs):
    _require_calibration_runs(calibration_runs, "pipeline")
    return [_samples(run) for run in calibration_runs]


def estimate_finetuning_pipeline(
    calibration_runs: Sequence[Mapping[str, Any]], target_plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Budget all planned native work from independent calibration processes.

    No target durations are accepted. Every work interval and neighboring gap,
    including first occurrences and warmup, must have independent coverage.
    Repeated groups require two samples per process; singleton stages require
    one sample in each of at least two processes. Process means are equally
    weighted. Hardware, execution, data and declared cache regimes must match.
    """
    return _estimate(
        _calibrations(calibration_runs), validate_pipeline_plan(target_plan)
    )


def _comparison(predicted, actual):
    return {
        "predicted": predicted,
        "actual": actual,
        "absolute_percentage_error": abs(predicted - actual) / actual * 100
        if actual
        else None,
        "underprediction_percent": max(actual - predicted, 0) / actual * 100
        if actual
        else None,
    }


def compare_finetuning_pipeline(
    calibration_runs: Sequence[Mapping[str, Any]], target_run: Mapping[str, Any]
) -> dict[str, Any]:
    """Predict a recorded trace first, then compare its complete measured span."""
    calibrations, target = _calibrations(calibration_runs), _samples(target_run)
    _independent([*calibrations, target])
    result = _estimate(calibrations, target["plan"])
    result["scope"] = "recorded_native_execution_span_replay"
    _attach_comparison(result, target)
    return result


def _attach_comparison(result, target):
    result["target_run"] = {
        field: deepcopy(target[field]) for field in ("identity", "evidence_sha256")
    }
    actuals = {
        component["component_id"]: component["observed"]
        for component in target["components"]
    }
    for component in result["target_components"]:
        component["observed"] = deepcopy(actuals[component["component_id"]])
    for group in result["component_groups"]:
        group["comparison"] = {
            field: _comparison(
                group["predicted"][field]["expected"],
                sum(actuals[index][field] for index in group["target_component_ids"]),
            )
            for field in _FIELDS
        }
    result["timing"]["comparison"] = {
        field: _comparison(
            result["timing"][field]["expected"],
            sum(values[field] for values in actuals.values()),
        )
        for field in _FIELDS
    }
    for kind, values in result["timing"]["by_kind"].items():
        values["comparison"] = {
            field: _comparison(
                values[field]["expected"],
                sum(
                    component["observed"][field]
                    for component in target["components"]
                    if component["kind"] == kind
                ),
            )
            for field in _FIELDS
        }
