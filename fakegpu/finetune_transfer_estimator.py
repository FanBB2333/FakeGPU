"""Empirical native budgets with explicit data-specific and transferable evidence.

This protocol pools compatible execution shapes rather than claiming that two
different token or label distributions have identical costs. Startup and
finalization always use independent measurements of the target dataset. Missing
work is reported before a budget is produced; no residual or size ratio fills it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
import statistics
from typing import Any

from ._finetune_batches import _descriptor
from ._finetune_pipeline_plan import _digest
from ._finetune_sampler import _integer as _byte_count
from ._finetune_transfer_plan import (
    _transfer_plan,
    describe_transfer_components,
    transfer_signature,
    validate_transfer_plan,
)
from .finetune_estimator import FinetuneEstimateError
from .finetune_pipeline_estimator import (
    _FIELDS,
    _attach_comparison,
    _interval,
    _samples as _pipeline_samples,
    _timing,
)
from .finetune_pipeline_plan import _validate_native_schedule_options


SCHEMA_VERSION = "fakegpu.finetune_transfer_estimate.v1"
COVERAGE_SCHEMA_VERSION = "fakegpu.finetune_transfer_coverage.v1"
MEMORY_SCHEMA_VERSION = "fakegpu.finetune_transfer_memory.v1"
_DATA_BOUND_KINDS = {"setup", "finalization"}
_MATCH_FIELDS = ("hardware", "memory_allocator", "execution_policy", "cache_regime")


def _requires_target_data(component):
    event = component["descriptor"].get("event")
    return component["kind"] in _DATA_BOUND_KINDS or (
        isinstance(event, Mapping) and event.get("inside_train") is False
    )


def _sample(run):
    recorded = _pipeline_samples(run)
    _validate_native_schedule_options(run)
    plan = _transfer_plan(run, recorded["plan"])
    components = describe_transfer_components(plan)
    groups = {}
    for component, measured in zip(components, recorded["components"]):
        component["observed"] = deepcopy(measured["observed"])
        groups.setdefault(component["group_id"], []).append(component)
    return {
        **recorded,
        "plan": plan,
        "components": components,
        "groups": groups,
        "peaks": {
            field: run["observation"].get(field)
            for field in ("peak_allocated_bytes", "peak_reserved_bytes")
        },
    }


def _pool(values, name):
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise FinetuneEstimateError(f"{name} must be a sequence of native reports")
    result = [_sample(run) for run in values]
    _unique(result)
    return result


def _unique(samples):
    for name, values in (
        ("run UUID", [sample["identity"]["uuid"] for sample in samples]),
        (
            "process identity",
            [
                (sample["identity"]["pid"], sample["identity"]["started_at_utc"])
                for sample in samples
            ],
        ),
        ("timing evidence", [sample["evidence_sha256"] for sample in samples]),
    ):
        if len(set(values)) != len(values):
            raise FinetuneEstimateError(f"transfer calibration repeats a {name}")


def _merge_pools(work, setup):
    """Permit one report to have both roles without counting it twice."""
    by_uuid = {sample["identity"]["uuid"]: sample for sample in work}
    for sample in setup:
        uuid = sample["identity"]["uuid"]
        if uuid in by_uuid and sample != by_uuid[uuid]:
            raise FinetuneEstimateError(
                "one transfer run UUID has different reports in the two evidence pools"
            )
        by_uuid[uuid] = sample
    result = list(by_uuid.values())
    _unique(result)
    return result


def _prepare(calibration_runs, target_plan, setup_calibration_runs):
    plan = validate_transfer_plan(target_plan)
    work = _pool(calibration_runs, "calibration_runs")
    setup = _pool(setup_calibration_runs, "setup_calibration_runs")
    samples = _merge_pools(work, setup)
    signature = transfer_signature(plan)
    for sample in samples:
        if transfer_signature(sample["plan"]) != signature:
            raise FinetuneEstimateError(
                "transfer calibration requires matching execution signature"
            )
        for field in _MATCH_FIELDS:
            if sample["plan"][field] != plan[field]:
                raise FinetuneEstimateError(
                    f"transfer calibration requires matching {field}"
                )
    for sample in setup:
        if (
            sample["plan"]["setup_workload"] != plan["setup_workload"]
            or sample["plan"]["signature"] != plan["signature"]
        ):
            raise FinetuneEstimateError(
                "setup calibration must match the complete target dataset, native input configuration and preprocessing policy"
            )
    components = describe_transfer_components(plan)
    groups = {}
    for component in components:
        groups.setdefault(component["group_id"], []).append(component)
    return plan, samples, setup, components, groups


def _coverage(samples, setup, groups):
    result = []
    for group_id, targets in groups.items():
        first = targets[0]
        minimum = first["minimum_samples_per_run"]
        data_bound = _requires_target_data(first)
        pool = setup if data_bound else samples
        observations = []
        for sample in pool:
            matches = sample["groups"].get(group_id, [])
            observations.append(
                {
                    "run_uuid": sample["identity"]["uuid"],
                    "samples": len(matches),
                    "eligible": len(matches) >= minimum,
                    "component_ids": [match["component_id"] for match in matches],
                }
            )
        eligible = sum(row["eligible"] for row in observations)
        result.append(
            {
                "group_id": group_id,
                "kind": first["kind"],
                "descriptor": deepcopy(first["descriptor"]),
                "evidence_role": "target_dataset"
                if data_bound
                else "compatible_execution",
                "minimum_samples_per_run": minimum,
                "minimum_independent_runs": 2,
                "eligible_independent_runs": eligible,
                "missing_independent_runs": max(0, 2 - eligible),
                "covered": eligible >= 2,
                "target_components": len(targets),
                "target_component_ids": [row["component_id"] for row in targets],
                "calibration_runs": observations,
            }
        )
    return result


def _window_distribution(plan, components):
    """Preserve full workload descriptors omitted by empirical shape pooling."""
    histogram = {}
    for component in components:
        if component["kind"] != "optimizer_update":
            continue
        update = plan["updates"][component["reference"]["optimizer_step"] - 1]
        window = [_descriptor(batch) for batch in update["batches"]]
        key = _digest(window)
        row = histogram.setdefault(
            key, {"descriptor_sha256": key, "batches": window, "optimizer_updates": 0}
        )
        row["optimizer_updates"] += 1
    return [histogram[key] for key in sorted(histogram)]


def inspect_finetuning_transfer_coverage(
    calibration_runs: Sequence[Mapping[str, Any]],
    target_plan: Mapping[str, Any],
    *,
    setup_calibration_runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """List absent empirical coverage without constructing an incomplete budget.

    Empty pools are accepted so a bounded calibration workflow can determine its
    initial requirements. Incompatible or duplicated evidence remains an error.
    """
    plan, samples, setup, components, groups = _prepare(
        calibration_runs, target_plan, setup_calibration_runs
    )
    coverage = _coverage(samples, setup, groups)
    return {
        "schema_version": COVERAGE_SCHEMA_VERSION,
        "target_plan_sha256": _digest(plan),
        "complete": all(group["covered"] for group in coverage),
        "target_components": len(components),
        "target_groups": len(groups),
        "covered_groups": sum(group["covered"] for group in coverage),
        "missing_groups": [
            deepcopy(group) for group in coverage if not group["covered"]
        ],
        "component_groups": coverage,
    }


def _estimate(plan, samples, setup, components, groups):
    coverage = _coverage(samples, setup, groups)
    missing = [group for group in coverage if not group["covered"]]
    if missing:
        details = "; ".join(
            f"{group['kind']} {group['group_id']}: "
            f"{group['eligible_independent_runs']} independent runs (requires 2)"
            for group in missing
        )
        raise FinetuneEstimateError("insufficient transfer coverage: " + details)
    by_uuid = {sample["identity"]["uuid"]: sample for sample in samples}
    for group in coverage:
        processes = []
        for row in group["calibration_runs"]:
            if not row["eligible"]:
                continue
            observed = by_uuid[row["run_uuid"]]["groups"][group["group_id"]]
            processes.append(
                {
                    **row,
                    "batch_descriptor_distribution": _window_distribution(
                        by_uuid[row["run_uuid"]]["plan"], observed
                    ),
                    "mean": {
                        field: statistics.mean(
                            sample["observed"][field] for sample in observed
                        )
                        for field in _FIELDS
                    },
                }
            )
        group["calibration_runs"] = processes
        group["target_batch_descriptor_distribution"] = _window_distribution(
            plan, groups[group["group_id"]]
        )
        group["per_component"], group["predicted"] = _interval(
            processes, group["target_components"]
        )
    costs = {group["group_id"]: group["per_component"] for group in coverage}
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "planned_native_execution_span_with_empirical_shape_transfer",
        "execution_signature": deepcopy(transfer_signature(plan)),
        **{field: deepcopy(plan[field]) for field in _MATCH_FIELDS},
        "setup_workload": deepcopy(plan["setup_workload"]),
        "target_plan": {
            "sha256": _digest(plan),
            **{
                field: deepcopy(plan[field])
                for field in ("source", "warmup", "workload")
            },
        },
        "calibration_runs": [
            {
                "identity": deepcopy(sample["identity"]),
                "evidence_sha256": sample["evidence_sha256"],
                "setup_workload": deepcopy(sample["plan"]["setup_workload"]),
                "target_dataset_role": sample in setup,
            }
            for sample in samples
        ],
        "sampling_policy": {
            "minimum_independent_runs_per_group": 2,
            "minimum_samples_per_repeated_group_per_run": 2,
            "minimum_samples_per_first_warmup_initial_final_group_per_run": 1,
            "aggregation": "equal_weight_per_eligible_run_component_mean",
            "startup_policy": "target_dataset_measurements_without_size_scaling",
            "work_pooling": "execution_shape_empirical_approximation",
        },
        "empirical_approximation": {
            "enabled": True,
            "model": "execution_shape_conditioned_process_mean",
            "exact_token_distribution_match": False,
            "preserved_workload_statistics": "complete_ordered_batch_descriptor_histograms_per_group_and_process",
            "unmodelled_effects": [
                "token_values",
                "within_shape_attention_mask_and_supervision_distribution",
                "model_state_and_training_duration",
                "unobserved_allocator_or_kernel_history",
            ],
        },
        "coverage": {
            "complete": True,
            "target_components": len(components),
            "target_groups": len(groups),
            "excluded_components": 0,
        },
        "component_groups": coverage,
        "target_components": [
            {
                **{
                    field: deepcopy(component[field])
                    for field in ("component_id", "kind", "group_id", "reference")
                },
                "predicted": deepcopy(costs[component["group_id"]]),
            }
            for component in components
        ],
        "timing": _timing(coverage, components),
        "limitations": [
            "Execution-shape pooling is an empirical approximation. Different token, mask and supervision distributions can have different costs despite matching tensor shapes.",
            "Every planned component requires eligible measurements from at least two independent processes. Missing groups cannot be filled by a total residual or dataset-size ratio.",
            "Setup and finalization use the target dataset and preprocessing policy; changing dataset size does not proportionally scale these costs.",
            "The native span excludes process startup, report construction and CLI output. Net seconds remove only measured observation intervals, not every instrumentation cost.",
            "Timing extrema combine per-process component means and are not confidence intervals or guaranteed prediction bounds.",
            "Named cache manifests do not attest cache hits, external cache dependencies, OS page caches, GPU clocks or system load.",
        ],
    }


def estimate_finetuning_transfer(
    calibration_runs: Sequence[Mapping[str, Any]],
    target_plan: Mapping[str, Any],
    *,
    setup_calibration_runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Budget all native components, explicitly separating dataset-bound costs."""
    return _estimate(*_prepare(calibration_runs, target_plan, setup_calibration_runs))


def _holdout(calibration_runs, target_run, setup_calibration_runs):
    target = _sample(target_run)
    prepared = _prepare(calibration_runs, target["plan"], setup_calibration_runs)
    _unique([*prepared[1], target])
    return target, prepared


def compare_finetuning_transfer(
    calibration_runs: Sequence[Mapping[str, Any]],
    target_run: Mapping[str, Any],
    *,
    setup_calibration_runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Predict an independent holdout before consulting its measured timings."""
    target, prepared = _holdout(calibration_runs, target_run, setup_calibration_runs)
    result = _estimate(*prepared)
    result["scope"] = "recorded_native_execution_span_shape_transfer_replay"
    _attach_comparison(result, target)
    return result


def estimate_finetuning_transfer_memory(
    calibration_runs: Sequence[Mapping[str, Any]],
    target_plan: Mapping[str, Any],
    *,
    setup_calibration_runs: Sequence[Mapping[str, Any]],
    memory_budget_bytes: int | None = None,
    reserve_bytes: int = 0,
) -> dict[str, Any]:
    """Report measured full-process peaks for covered transferable components.

    This is an empirical envelope of the contributing short processes. Combining
    their workloads in another order or running longer can change fragmentation
    or growth; the envelope is not an inferred peak of the combined target run.
    """
    return _memory_estimate(
        _prepare(calibration_runs, target_plan, setup_calibration_runs),
        memory_budget_bytes=memory_budget_bytes,
        reserve_bytes=reserve_bytes,
    )


def _memory_estimate(prepared, *, memory_budget_bytes, reserve_bytes):
    plan, samples, setup, components, groups = prepared
    coverage = _coverage(samples, setup, groups)
    if any(not group["covered"] for group in coverage):
        raise FinetuneEstimateError(
            "transfer memory requires complete component coverage"
        )
    used = {
        row["run_uuid"]
        for group in coverage
        for row in group["calibration_runs"]
        if row["eligible"]
    }
    contributing = [sample for sample in samples if sample["identity"]["uuid"] in used]
    physical = _byte_count(
        plan["hardware"].get("total_memory_bytes"), "total_memory_bytes"
    )
    fraction = plan["memory_allocator"]["per_process_memory_fraction"]
    limit = math.floor(physical * fraction)
    reserve = _byte_count(reserve_bytes, "reserve_bytes", minimum=0)
    if memory_budget_bytes is not None:
        _byte_count(memory_budget_bytes, "memory_budget_bytes")
        if memory_budget_bytes > physical:
            raise FinetuneEstimateError(
                "memory budget exceeds recorded physical GPU memory"
            )
        if memory_budget_bytes > limit:
            raise FinetuneEstimateError(
                "memory budget exceeds the recorded allocator cap"
            )
    for sample in contributing:
        for field, value in sample["peaks"].items():
            _byte_count(value, field)
        allocated, reserved = (
            sample["peaks"][field]
            for field in ("peak_allocated_bytes", "peak_reserved_bytes")
        )
        if reserved < allocated or reserved > physical:
            raise FinetuneEstimateError("invalid native allocator peak envelope")
    envelope = {
        field: {
            "min": min(sample["peaks"][field] for sample in contributing),
            "max": max(sample["peaks"][field] for sample in contributing),
        }
        for field in ("peak_allocated_bytes", "peak_reserved_bytes")
    }
    required = envelope["peak_reserved_bytes"]["max"] + reserve
    return {
        "schema_version": MEMORY_SCHEMA_VERSION,
        "scope": "contributing_transfer_process_allocator_peak_envelope",
        "target_plan_sha256": _digest(plan),
        "execution_signature": deepcopy(transfer_signature(plan)),
        **{field: deepcopy(plan[field]) for field in _MATCH_FIELDS},
        "calibration_runs": [
            {
                "identity": deepcopy(sample["identity"]),
                "evidence_sha256": sample["evidence_sha256"],
                **sample["peaks"],
            }
            for sample in contributing
        ],
        "coverage": {
            "complete": True,
            "target_components": len(components),
            "target_groups": len(groups),
        },
        "memory": envelope,
        "budget": {
            "memory_budget_bytes": memory_budget_bytes,
            "reserve_bytes": reserve,
            "recorded_allocator_limit_bytes": limit,
            "observed_reserved_peak_plus_reserve_bytes": required,
            "fits_recorded_allocator_cap": required <= limit,
            "status": "not_requested"
            if memory_budget_bytes is None
            else (
                "observed_envelope_fits_budget"
                if required <= memory_budget_bytes
                else "observed_envelope_exceeds_budget"
            ),
            "oom_safety_verified": False,
        },
        "limitations": [
            "The envelope contains measured complete-process allocator peaks and never sums individual component or process peaks.",
            "Its contributing processes may have different datasets and shorter schedules. Shape coverage does not establish the combined target run's peak, fragmentation or long-run memory growth.",
            "Allocator statistics exclude memory outside the PyTorch allocator. This is not physical-GPU OOM validation or a minimum memory requirement.",
        ],
    }


def compare_finetuning_transfer_memory(
    calibration_runs: Sequence[Mapping[str, Any]],
    target_run: Mapping[str, Any],
    *,
    setup_calibration_runs: Sequence[Mapping[str, Any]],
    memory_budget_bytes: int | None = None,
    reserve_bytes: int = 0,
) -> dict[str, Any]:
    """Check independent target peaks against the pre-existing empirical envelope."""
    target, prepared = _holdout(calibration_runs, target_run, setup_calibration_runs)
    estimate = _memory_estimate(
        prepared, memory_budget_bytes=memory_budget_bytes, reserve_bytes=reserve_bytes
    )
    comparison = {}
    for field, interval in estimate["memory"].items():
        actual = _byte_count(target["peaks"][field], field)
        comparison[field] = {
            "actual": actual,
            "observed_min": interval["min"],
            "observed_max": interval["max"],
            "within_observed_envelope": interval["min"] <= actual <= interval["max"],
            "exceeds_observed_max": actual > interval["max"],
            "underprediction_percent": max(0, actual - interval["max"]) / actual * 100,
        }
    if (
        target["peaks"]["peak_reserved_bytes"] < target["peaks"]["peak_allocated_bytes"]
        or target["peaks"]["peak_reserved_bytes"]
        > target["plan"]["hardware"]["total_memory_bytes"]
    ):
        raise FinetuneEstimateError("invalid holdout allocator peaks")
    return {
        "schema_version": "fakegpu.finetune_transfer_memory_comparison.v1",
        "estimate": estimate,
        "target_run": {
            field: deepcopy(target[field]) for field in ("identity", "evidence_sha256")
        },
        "comparison": comparison,
    }
