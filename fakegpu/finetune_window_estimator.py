"""Calibrate recorded dynamic optimizer windows using independent native runs.

This API compares an already observed target trace with matching calibration
windows. It does not infer a future sampler order or extrapolate training epochs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timedelta
import hashlib
import json
import statistics
from typing import Any
from uuid import UUID

from ._finetune_batches import _descriptor
from .finetune_estimator import FinetuneEstimateError
from .finetune_run_estimator import validate_native_run


SCHEMA_VERSION = "fakegpu.finetune_window_estimate.v1"
_DENSE_MODELS = {"llama", "mistral", "qwen2", "qwen3", "qwen3_5_text"}
_MINIMUM_RUNS = 2
_MINIMUM_SAMPLES = 2


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _run_identity(run: Mapping[str, Any]) -> dict[str, Any]:
    identity = run.get("run_identity")
    if not isinstance(identity, Mapping):
        raise FinetuneEstimateError("window calibration requires run_identity")
    try:
        value = identity["uuid"]
        if not isinstance(value, str) or str(UUID(value)) != value:
            raise ValueError
        pid = identity["pid"]
        if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
            raise ValueError
        timestamp = identity["started_at_utc"]
        if not isinstance(timestamp, str):
            raise ValueError
        if datetime.fromisoformat(
            timestamp.replace("Z", "+00:00")
        ).utcoffset() != timedelta(0):
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise FinetuneEstimateError(
            "run_identity requires a canonical UUID, positive pid and UTC started_at_utc"
        ) from exc
    return deepcopy(dict(identity))


def _samples(run: Mapping[str, Any]) -> dict[str, Any]:
    config, observation, _, _, schedule = validate_native_run(
        run, require_fixed_shape=False
    )
    if "native_config" not in run or "observed_model" not in observation:
        raise FinetuneEstimateError(
            "window calibration requires native_config and observed_model execution evidence"
        )
    if config.get("packing") is not False:
        raise FinetuneEstimateError("window calibration does not support packing")
    if any(schedule.get(field) != "no" for field in ("eval_strategy", "save_strategy")):
        raise FinetuneEstimateError(
            "window calibration does not support periodic evaluation or checkpointing; "
            "their effects on subsequent window state are not modeled"
        )
    model = run["model_config"]
    if model.get("model_type") not in _DENSE_MODELS or any(
        model.get(field, 0)
        for field in ("num_experts", "num_local_experts", "n_routed_experts")
    ):
        raise FinetuneEstimateError(
            "window calibration currently requires a dense model"
        )
    identity = _run_identity(run)
    seen = set()
    groups = {}
    trace = []
    warmup_count = first_count = excluded_count = 0
    for update in observation["updates"]:
        batches = [_descriptor(batch) for batch in update["batches"]]
        if any(
            batch["attention_mask_kind"] != "binary_2d"
            or batch["labels_kind"] != "integer_2d"
            for batch in batches
        ):
            raise FinetuneEstimateError(
                "window calibration requires known binary masks and integer labels"
            )
        key = _canonical(batches)
        first = key not in seen
        seen.add(key)
        warmup_count += update["warmup"]
        first_count += first
        excluded_reasons = []
        if update["warmup"]:
            excluded_reasons.append("global_warmup")
        if first:
            excluded_reasons.append("first_window_occurrence")
        trace.append(
            {
                "optimizer_step": update["optimizer_step"],
                "epoch_index": update["batches"][0].get("epoch_index"),
                "window_id": hashlib.sha256(key.encode()).hexdigest(),
                "observed_seconds": update["seconds"],
                "eligible": not excluded_reasons,
                "excluded_reasons": excluded_reasons,
            }
        )
        if first or update["warmup"]:
            excluded_count += 1
            continue
        group = groups.setdefault(key, {"batches": batches, "updates": []})
        group["updates"].append(
            {"optimizer_step": update["optimizer_step"], "seconds": update["seconds"]}
        )
    return {
        "identity": identity,
        "signature": observation["signature"],
        # Excluding the top-level identity also catches a copied timing record
        # assigned a different UUID. This detects duplication, not forgery.
        "evidence_sha256": _digest(
            {"observation": observation, "timing": run["timing"]}
        ),
        "groups": groups,
        "trace": trace,
        "sampling": {
            "recorded_updates": len(observation["updates"]),
            "global_warmup_updates": warmup_count,
            "first_window_occurrences": first_count,
            "excluded_updates": excluded_count,
            "eligible_updates": len(observation["updates"]) - excluded_count,
        },
    }


def _comparison(predicted: float, actual: float) -> dict[str, float]:
    return {
        "predicted": predicted,
        "actual": actual,
        "absolute_percentage_error": abs(predicted - actual) / actual * 100,
        "underprediction_percent": max(actual - predicted, 0) / actual * 100,
    }


def estimate_finetuning_windows(
    calibration_runs: Sequence[Mapping[str, Any]], target_run: Mapping[str, Any]
) -> dict[str, Any]:
    """Predict steady windows in an independent, already recorded target trace.

    Each target window needs at least two eligible samples in every calibration
    run, with at least two calibration runs. Global warmup and each window's first
    occurrence are excluded in every run. Process means receive equal weight.
    Report identity and content checks detect duplicate evidence; they cannot
    authenticate a JSON report or prove how a process was launched.
    """
    if (
        not isinstance(calibration_runs, Sequence)
        or isinstance(calibration_runs, (str, bytes))
        or len(calibration_runs) < _MINIMUM_RUNS
    ):
        raise FinetuneEstimateError("window calibration requires at least two runs")
    calibrations = [_samples(run) for run in calibration_runs]
    target = _samples(target_run)
    runs = [*calibrations, target]
    if len({run["identity"]["uuid"] for run in runs}) != len(runs):
        raise FinetuneEstimateError("calibration and target require distinct run UUIDs")
    if len({run["evidence_sha256"] for run in runs}) != len(runs):
        raise FinetuneEstimateError(
            "calibration and target contain duplicate timing evidence"
        )
    if any(run["signature"] != target["signature"] for run in calibrations):
        raise FinetuneEstimateError(
            "window calibration signatures must match target model, data and execution"
        )
    if not target["groups"]:
        raise FinetuneEstimateError("target has no eligible steady windows")

    groups = []
    missing = []
    for key, target_group in target["groups"].items():
        window_id = hashlib.sha256(key.encode()).hexdigest()
        process_samples = []
        for calibration in calibrations:
            updates = calibration["groups"].get(key, {}).get("updates", [])
            if len(updates) < _MINIMUM_SAMPLES:
                missing.append(
                    f"window {window_id}, run {calibration['identity']['uuid']}: "
                    f"{len(updates)} eligible samples (requires {_MINIMUM_SAMPLES})"
                )
                continue
            values = [update["seconds"] for update in updates]
            process_samples.append(
                {
                    "run_uuid": calibration["identity"]["uuid"],
                    "samples": len(values),
                    "optimizer_steps": [update["optimizer_step"] for update in updates],
                    "seconds": {
                        "min": min(values),
                        "mean": statistics.mean(values),
                        "max": max(values),
                    },
                }
            )
        if len(process_samples) != len(calibrations):
            continue
        process_means = [sample["seconds"]["mean"] for sample in process_samples]
        seconds = {
            "min": min(process_means),
            "expected": statistics.mean(process_means),
            "max": max(process_means),
        }
        count = len(target_group["updates"])
        actual = sum(update["seconds"] for update in target_group["updates"])
        groups.append(
            {
                "window_id": window_id,
                "batches": target_group["batches"],
                "calibration_runs": process_samples,
                "seconds_per_window": seconds,
                "target_optimizer_steps": [
                    update["optimizer_step"] for update in target_group["updates"]
                ],
                "target_updates": count,
                "predicted_seconds": {
                    name: value * count for name, value in seconds.items()
                },
                "comparison": _comparison(seconds["expected"] * count, actual),
            }
        )
    if missing:
        raise FinetuneEstimateError(
            "insufficient independent window coverage: " + "; ".join(missing)
        )
    totals = {
        field: sum(group["predicted_seconds"][field] for group in groups)
        for field in ("min", "expected", "max")
    }
    actual_total = sum(group["comparison"]["actual"] for group in groups)
    costs = {group["window_id"]: group["seconds_per_window"] for group in groups}
    target_updates = [
        {
            **update,
            "predicted_seconds": deepcopy(costs[update["window_id"]])
            if update["eligible"]
            else None,
        }
        for update in target["trace"]
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "recorded_steady_optimizer_window_replay",
        "signature": deepcopy(target["signature"]),
        "calibration_runs": [
            {
                key: deepcopy(run[key])
                for key in ("identity", "evidence_sha256", "sampling")
            }
            for run in calibrations
        ],
        "target_run": {
            key: deepcopy(target[key])
            for key in ("identity", "evidence_sha256", "sampling")
        },
        "sampling_policy": {
            "minimum_calibration_runs": _MINIMUM_RUNS,
            "minimum_samples_per_window_per_run": _MINIMUM_SAMPLES,
            "exclude_global_warmup": True,
            "exclude_first_window_occurrence": True,
            "aggregation": "equal_weight_per_run_window_mean",
        },
        "coverage": {
            "target_window_groups": len(groups),
            "target_eligible_updates": target["sampling"]["eligible_updates"],
            "covered_updates": sum(group["target_updates"] for group in groups),
            "complete": True,
        },
        "window_groups": groups,
        "target_updates": target_updates,
        "timing": {
            "steady_optimizer_window_seconds": totals,
            "comparison": _comparison(totals["expected"], actual_total),
            "interval_kind": "sum_of_window_process_mean_extrema_not_confidence_interval",
        },
        "limitations": [
            "This predicts only eligible optimizer windows in the recorded target trace; it does not infer future shuffle order, epochs, tails or dataset sizes.",
            "Matching batch statistics are not proof of identical inputs or mask layouts; token IDs and complete mask positions are not recorded.",
            "First-occurrence exclusion does not establish a controlled cache state; later compilation, clocks and system load can still change timing.",
            "Warmup, observer work outside sampled updates, evaluation, saves and other training-loop or pipeline costs are not predicted.",
            "No dynamic-window memory estimate or per-shape allocator peak is derived.",
            "UUIDs and content hashes detect duplicate records, but do not authenticate reports or independently prove process isolation.",
        ],
    }
