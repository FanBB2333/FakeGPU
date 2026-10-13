"""Calibrate the evaluation/save subtree in an independently observed native run."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import statistics
from typing import Any

from ._finetune_event_context import describe_root_events
from ._finetune_events import SCHEMA_VERSION as EVENT_SCHEMA
from .finetune_estimator import FinetuneEstimateError
from .finetune_run_estimator import validate_native_run
from .finetune_window_estimator import _DENSE_MODELS, _comparison, _run_identity

SCHEMA_VERSION = "fakegpu.finetune_event_estimate.v1"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _samples(run: Mapping[str, Any]) -> dict[str, Any]:
    config, observation, _, timing, _ = validate_native_run(
        run, require_fixed_shape=False
    )
    if config.get("event_observation_version") != EVENT_SCHEMA:
        raise FinetuneEstimateError(
            "event calibration requires the event observation protocol"
        )
    if "native_config" not in run or "observed_model" not in observation:
        raise FinetuneEstimateError(
            "event calibration requires native execution evidence"
        )
    if config.get("packing") is not False:
        raise FinetuneEstimateError("event calibration does not support packing")
    model = run["model_config"]
    if model.get("model_type") not in _DENSE_MODELS or any(
        model.get(field, 0)
        for field in ("num_experts", "num_local_experts", "n_routed_experts")
    ):
        raise FinetuneEstimateError(
            "event calibration currently requires a dense model"
        )
    identity = _run_identity(run)
    groups = {}
    trace = []
    for described in describe_root_events(observation["phase_events"]):
        event = observation["phase_events"][described["event_id"] - 1]
        descriptor = described["descriptor"]
        key = _canonical(descriptor)
        group = groups.setdefault(key, {"descriptor": descriptor, "events": []})
        if event["seconds"] <= 0:
            raise FinetuneEstimateError(
                "event calibration requires positive net event durations"
            )
        group["events"].append(
            {"event_id": event["event_id"], "seconds": event["seconds"]}
        )
        trace.append(
            {
                "event_id": event["event_id"],
                "optimizer_step": event["optimizer_step"],
                "event_group_id": hashlib.sha256(key.encode()).hexdigest(),
                "observed_seconds": event["seconds"],
                "phase": descriptor["phase"],
            }
        )
    if not groups:
        raise FinetuneEstimateError("event calibration requires observed root events")
    for group in groups.values():
        if group["descriptor"]["phase"] != "repeated" and len(group["events"]) != 1:
            raise FinetuneEstimateError(
                "initial, first and final event groups must occur once per run"
            )
    return {
        "identity": identity,
        "signature": observation["signature"],
        "hardware": observation["hardware"],
        "evidence_sha256": _digest({"observation": observation, "timing": timing}),
        "groups": groups,
        "trace": trace,
        "sampling": {
            "recorded_events": len(observation["phase_events"]),
            "root_events": len(trace),
        },
    }


def estimate_finetuning_events(
    calibration_runs: Sequence[Mapping[str, Any]], target_run: Mapping[str, Any]
) -> dict[str, Any]:
    """Predict recorded target root events from two or more independent runs.

    Initial, first and final events each require one occurrence per calibration
    run; repeated events require two. Process means receive equal weight. Target
    timings are used only for comparison after predictions have been computed.
    This does not predict future event triggers, training time, wall time or VRAM.
    """
    if (
        not isinstance(calibration_runs, Sequence)
        or isinstance(calibration_runs, (str, bytes))
        or len(calibration_runs) < 2
    ):
        raise FinetuneEstimateError("event calibration requires at least two runs")
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
            "event calibration signatures must match target model, data and execution"
        )
    if any(run["hardware"] != target["hardware"] for run in calibrations):
        raise FinetuneEstimateError(
            "event calibration requires matching recorded hardware"
        )
    groups = []
    missing = []
    for key, target_group in target["groups"].items():
        descriptor = target_group["descriptor"]
        group_id = hashlib.sha256(key.encode()).hexdigest()
        minimum = 2 if descriptor["phase"] == "repeated" else 1
        process_samples = []
        for calibration in calibrations:
            events = calibration["groups"].get(key, {}).get("events", [])
            if len(events) < minimum:
                missing.append(
                    f"event {group_id}, run {calibration['identity']['uuid']}: "
                    f"{len(events)} samples (requires {minimum})"
                )
                continue
            values = [event["seconds"] for event in events]
            process_samples.append(
                {
                    "run_uuid": calibration["identity"]["uuid"],
                    "samples": len(values),
                    "event_ids": [event["event_id"] for event in events],
                    "seconds": {
                        "min": min(values),
                        "mean": statistics.mean(values),
                        "max": max(values),
                    },
                }
            )
        if len(process_samples) != len(calibrations):
            continue
        means = [sample["seconds"]["mean"] for sample in process_samples]
        seconds = {
            "min": min(means),
            "expected": statistics.mean(means),
            "max": max(means),
        }
        count = len(target_group["events"])
        actual = sum(event["seconds"] for event in target_group["events"])
        groups.append(
            {
                "event_group_id": group_id,
                **deepcopy(descriptor),
                "calibration_runs": process_samples,
                "seconds_per_event": seconds,
                "target_event_ids": [
                    event["event_id"] for event in target_group["events"]
                ],
                "target_events": count,
                "predicted_seconds": {
                    field: value * count for field, value in seconds.items()
                },
                "comparison": _comparison(seconds["expected"] * count, actual),
            }
        )
    if missing:
        raise FinetuneEstimateError(
            "insufficient independent event coverage: " + "; ".join(missing)
        )
    totals = {
        field: sum(group["predicted_seconds"][field] for group in groups)
        for field in ("min", "expected", "max")
    }
    actual = sum(group["comparison"]["actual"] for group in groups)
    costs = {group["event_group_id"]: group["seconds_per_event"] for group in groups}
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "recorded_native_root_event_replay",
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
            "minimum_calibration_runs": 2,
            "minimum_samples_per_repeated_group_per_run": 2,
            "minimum_samples_per_initial_first_final_group_per_run": 1,
            "aggregation": "equal_weight_per_run_event_mean",
        },
        "coverage": {
            "target_event_groups": len(groups),
            "target_root_events": len(target["trace"]),
            "complete": True,
        },
        "event_groups": groups,
        "target_events": [
            {**event, "predicted_seconds": deepcopy(costs[event["event_group_id"]])}
            for event in target["trace"]
        ],
        "timing": {
            "root_event_seconds": totals,
            "comparison": _comparison(totals["expected"], actual),
            "interval_kind": "sum_of_event_process_mean_extrema_not_confidence_interval",
        },
        "limitations": [
            "Only actual root events in the recorded target trace are predicted; nested calls are included once. Future schedule triggers and event counts are not inferred.",
            "Event costs do not predict total training or pipeline wall time, event interference with subsequent optimizer updates, or allocator peaks/VRAM.",
            "First, initial and final calls are calibrated separately; their inclusion does not establish controlled cache, compilation, filesystem or device state.",
            "Checkpoint groups match save policy and file layout. Inventories are files present after saves, not bytes written, I/O traffic or durable writes; volatile file sizes are not matched.",
            "UUIDs and content hashes detect duplicate records, but do not authenticate reports or independently prove process isolation.",
        ],
    }
