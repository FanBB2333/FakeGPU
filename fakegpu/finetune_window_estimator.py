"""Calibrate recorded dynamic optimizer windows using independent native runs.

This API compares an already observed target trace with matching calibration
windows. It does not infer a future sampler order or extrapolate training epochs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timedelta
import hashlib
import statistics
from typing import Any
from uuid import UUID

from ._finetune_batches import _descriptor
from ._finetune_event_context import describe_root_events
from ._finetune_events import SCHEMA_VERSION as EVENT_SCHEMA
from .finetune_estimator import FinetuneEstimateError
from .finetune_run_estimator import validate_native_run
from .structured_io import canonical_json as _canonical, canonical_sha256 as _digest


SCHEMA_VERSION = "fakegpu.finetune_window_estimate.v1"
_DENSE_MODELS = {"llama", "mistral", "qwen2", "qwen3", "qwen3_5_text"}
_MINIMUM_RUNS = 2
_MINIMUM_SAMPLES = 2


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


def _require_calibration_runs(calibration_runs: Any, purpose: str) -> None:
    if (
        not isinstance(calibration_runs, Sequence)
        or isinstance(calibration_runs, (str, bytes))
        or len(calibration_runs) < _MINIMUM_RUNS
    ):
        raise FinetuneEstimateError(f"{purpose} calibration requires at least two runs")


def _require_dense_model(model: Mapping[str, Any], purpose: str) -> None:
    if model.get("model_type") not in _DENSE_MODELS or any(
        model.get(field, 0)
        for field in ("num_experts", "num_local_experts", "n_routed_experts")
    ):
        raise FinetuneEstimateError(
            f"{purpose} calibration currently requires a dense model"
        )


def _require_comparable_runs(
    calibrations: Sequence[Mapping[str, Any]], target: Mapping[str, Any], purpose: str
) -> None:
    runs = [*calibrations, target]
    if len({run["identity"]["uuid"] for run in runs}) != len(runs):
        raise FinetuneEstimateError("calibration and target require distinct run UUIDs")
    if len({run["evidence_sha256"] for run in runs}) != len(runs):
        raise FinetuneEstimateError(
            "calibration and target contain duplicate timing evidence"
        )
    if any(run["signature"] != target["signature"] for run in calibrations):
        raise FinetuneEstimateError(
            f"{purpose} calibration signatures must match target model, data and execution"
        )


def _event_context(
    latest: dict[str, Any], roots: Mapping[str, Any], update: Mapping[str, Any]
) -> dict[str, Any]:
    preceding = [roots[event_id] for event_id in update["preceding_event_ids"]]
    if preceding:
        boundary = {
            "step": preceding[0]["optimizer_step"],
            "events": [event["descriptor"] for event in preceding],
        }
        for event in preceding:
            latest[event["descriptor"]["kind"]] = boundary
    return {
        kind: None
        if boundary is None
        else {
            "events": boundary["events"],
            "optimizer_updates_since_boundary": update["optimizer_step"]
            - boundary["step"]
            - 1,
        }
        for kind, boundary in latest.items()
    }


def _window_contexts(observation: Mapping[str, Any]):
    """Carry the latest event boundary for each kind into subsequent updates."""
    roots = {
        event["event_id"]: event
        for event in describe_root_events(observation["phase_events"])
    }
    latest = {"evaluation": None, "checkpoint": None}
    for update in observation["updates"]:
        yield _event_context(latest, roots, update)


def _epoch_window_context(update: Mapping[str, Any]) -> dict[str, Any]:
    """Keep every recorded position, including resets at observed epoch changes.

    The shared batch validator already checks one-based, contiguous metadata.
    This opt-in window mode additionally requires that optional metadata.
    """
    batches = update["batches"]
    if any(
        "epoch_index" not in batch or "epoch_microstep" not in batch
        for batch in batches
    ):
        raise FinetuneEstimateError(
            "epoch-conditioned windows require recorded epoch metadata for every batch"
        )
    return {"batch_microsteps": [batch["epoch_microstep"] for batch in batches]}


def _samples(
    run: Mapping[str, Any], *, event_context: bool = False, epoch_context: bool = False
) -> dict[str, Any]:
    config, observation, _, _, schedule = validate_native_run(
        run, require_fixed_shape=False
    )
    if "native_config" not in run or "observed_model" not in observation:
        raise FinetuneEstimateError(
            "window calibration requires native_config and observed_model execution evidence"
        )
    if config.get("packing") is not False:
        raise FinetuneEstimateError("window calibration does not support packing")
    if event_context and config.get("event_observation_version") != EVENT_SCHEMA:
        raise FinetuneEstimateError(
            "event-conditioned windows require the event observation protocol"
        )
    if not event_context and any(
        schedule.get(field) != "no" for field in ("eval_strategy", "save_strategy")
    ):
        raise FinetuneEstimateError(
            "window calibration does not support periodic evaluation or checkpointing; "
            "their effects on subsequent window state are not modeled"
        )
    _require_dense_model(run["model_config"], "window")
    identity = _run_identity(run)
    seen = set()
    groups = {}
    trace = []
    contexts = _window_contexts(observation) if event_context else None
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
        context = next(contexts) if event_context else None
        epoch = _epoch_window_context(update) if epoch_context else None
        descriptor = {"batches": batches} if event_context or epoch_context else batches
        if event_context:
            descriptor["context"] = context
        if epoch_context:
            descriptor["epoch_context"] = epoch
        key = _canonical(descriptor)
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
        if event_context:
            trace[-1]["immediately_after_event"] = any(
                boundary is not None
                and boundary["optimizer_updates_since_boundary"] == 0
                for boundary in context.values()
            )
        if epoch_context:
            trace[-1]["epoch_context"] = epoch
        if first or update["warmup"]:
            excluded_count += 1
            continue
        group = groups.setdefault(key, {"batches": batches, "updates": []})
        if event_context:
            group["context"] = context
        if epoch_context:
            group["epoch_context"] = epoch
        group["updates"].append(
            {"optimizer_step": update["optimizer_step"], "seconds": update["seconds"]}
        )
    result = {
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
    if event_context:
        result["hardware"] = observation["hardware"]
    return result


def _comparison(predicted: float, actual: float) -> dict[str, float]:
    return {
        "predicted": predicted,
        "actual": actual,
        "absolute_percentage_error": abs(predicted - actual) / actual * 100,
        "underprediction_percent": max(actual - predicted, 0) / actual * 100,
    }


def estimate_finetuning_windows(
    calibration_runs: Sequence[Mapping[str, Any]],
    target_run: Mapping[str, Any],
    *,
    event_context: bool = False,
    epoch_context: bool = False,
) -> dict[str, Any]:
    """Predict steady windows in an independent, already recorded target trace.

    Each target window needs at least two eligible samples in every calibration
    run, with at least two calibration runs. Global warmup and each window's first
    occurrence are excluded in every run. Process means receive equal weight.
    With ``event_context=True``, windows also match the latest complete boundary
    containing evaluation and checkpoint calls, with elapsed optimizer updates.
    First occurrences then refer to each combined batch/context descriptor.
    Event durations are not part of the predicted optimizer-window time.
    ``epoch_context=True`` also matches every batch's recorded position within
    its epoch, independently of the absolute epoch number. Recorded position
    resets within a window remain part of the descriptor.
    Either context option makes first-occurrence exclusion apply to the complete
    selected descriptor; periodic events still require ``event_context=True``.
    Report identity and content checks detect duplicate evidence; they cannot
    authenticate a JSON report or prove how a process was launched.
    """
    if not isinstance(event_context, bool):
        raise FinetuneEstimateError("event_context must be boolean")
    if not isinstance(epoch_context, bool):
        raise FinetuneEstimateError("epoch_context must be boolean")
    _require_calibration_runs(calibration_runs, "window")
    calibrations = [
        _samples(run, event_context=event_context, epoch_context=epoch_context)
        for run in calibration_runs
    ]
    target = _samples(
        target_run, event_context=event_context, epoch_context=epoch_context
    )
    _require_comparable_runs(calibrations, target, "window")
    if event_context and any(
        run["hardware"] != target["hardware"] for run in calibrations
    ):
        raise FinetuneEstimateError(
            "event-conditioned windows require matching recorded hardware"
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
        if event_context:
            groups[-1]["context"] = deepcopy(target_group["context"])
        if epoch_context:
            groups[-1]["epoch_context"] = deepcopy(target_group["epoch_context"])
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
    sampling_policy = {
        "minimum_calibration_runs": _MINIMUM_RUNS,
        "minimum_samples_per_window_per_run": _MINIMUM_SAMPLES,
        "exclude_global_warmup": True,
    }
    if not (event_context or epoch_context):
        sampling_policy["exclude_first_window_occurrence"] = True
    sampling_policy["aggregation"] = "equal_weight_per_run_window_mean"
    if event_context:
        sampling_policy.update(
            event_context="latest_complete_boundary_per_event_kind",
            exclude_first_context_window_occurrence=True,
        )
    if epoch_context:
        sampling_policy.update(
            epoch_context="recorded_epoch_microsteps_per_batch",
            exclude_first_context_window_occurrence=True,
        )
    scope = {
        (False, False): "recorded_steady_optimizer_window_replay",
        (True, False): "recorded_event_conditioned_optimizer_window_replay",
        (False, True): "recorded_epoch_conditioned_optimizer_window_replay",
        (True, True): "recorded_event_epoch_conditioned_optimizer_window_replay",
    }[event_context, epoch_context]
    result = {
        "schema_version": SCHEMA_VERSION,
        "scope": scope,
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
        "sampling_policy": sampling_policy,
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
    if event_context:
        result["limitations"].extend(
            [
                "Event context retains the latest complete ordered boundary containing each event kind and the number of optimizer updates since it; it does not prove that older events, cache state or system load have no remaining effect.",
                "Context-conditioned samples validate recorded window timing; they do not isolate a causal event overhead or predict event durations, combined pipeline wall time, or future schedules.",
            ]
        )
    if epoch_context:
        result["limitations"].append(
            "Epoch context uses each batch's recorded one-based microstep within its epoch; it does not include absolute epoch number or infer future epoch boundaries. Matching positions does not establish the cause of timing differences or control sampler, cache and system state."
        )
    return result
