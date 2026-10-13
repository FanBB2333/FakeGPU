"""Extrapolate a native framework run without losing its measured workload."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from ._finetune_batches import validate_batch_profile
from ._finetune_allocator import validate_memory_allocator
from ._finetune_cache import validate_cache_regime
from ._finetune_events import validate_event_profile
from ._finetune_model import native_config_signature, native_execution_signature
from ._finetune_timeline import validate_native_timeline
from .finetune_data_report import apply_dataset_report
from .finetune_estimator import (
    FinetuneEstimateError,
    _validate_observation,
    estimate_finetuning,
    finetune_workload_signature,
)


def estimate_finetuning_run(
    run: Mapping[str, Any],
    *,
    profile_id: str | None = None,
    num_samples: int | None = None,
    epochs: float | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Project a fixed-shape native run onto another number of training steps.

    ``max_steps=-1`` removes the source run's step limit and uses epochs.
    This pure API checks report consistency; the CLI additionally rehashes local
    weights and verifies the current software environment before reuse.
    """
    config, observation, _, _, _ = validate_native_run(run)
    if max_steps is not None:
        config["max_steps"] = None if max_steps == -1 else max_steps
    report = estimate_finetuning(
        run["model_config"],
        config,
        profile_id=profile_id or run.get("profile_id"),
        num_samples=num_samples,
        epochs=epochs,
        observation=observation,
    )
    report["dataset"] = run["dataset"]
    report["limitations"].extend(config.get("dataset_limitations", []))
    report["native_run"] = {
        "framework": run["framework"],
        "observed_timing": run["timing"],
        "observed_workload": run["workload"],
    }
    report["timing"]["native_pipeline"] = _pipeline_projection(run, report)
    return report


def validate_native_run(
    run: Mapping[str, Any], *, require_fixed_shape: bool = True
) -> tuple[dict[str, Any], Mapping, Mapping, Mapping, Mapping]:
    """Reconcile native evidence; dynamic validation requires the actual batch ledger."""
    if (
        not isinstance(run, Mapping)
        or run.get("schema_version") != "fakegpu.finetune_run.v1"
    ):
        raise FinetuneEstimateError("unsupported native fine-tuning run schema")
    framework = run.get("framework")
    if framework not in {"huggingface", "llamafactory", "axolotl"}:
        raise FinetuneEstimateError("unsupported native framework")
    for key in (
        "training_config",
        "model_config",
        "dataset",
        "observation",
        "workload",
        "timing",
        "schedule",
    ):
        if not isinstance(run.get(key), Mapping):
            raise FinetuneEstimateError(f"native run requires {key}")
    config = apply_dataset_report(run["training_config"], run["dataset"])
    if config.get("source_framework") != framework:
        raise FinetuneEstimateError(
            "native run framework differs from its training configuration"
        )
    observation = run["observation"]
    if observation.get("source") != "fakegpu.native_framework_cuda_run":
        raise FinetuneEstimateError("native run requires its native CUDA observation")
    try:
        validate_batch_profile(observation)
        validate_event_profile(observation, timing=run["timing"])
        validate_native_timeline(observation, timing=run["timing"])
        if "cache_regime" in run:
            validate_cache_regime(run["cache_regime"])
        if "memory_allocator" in observation:
            validate_memory_allocator(observation["memory_allocator"])
    except ValueError as exc:
        raise FinetuneEstimateError(str(exc)) from exc
    dataset_digest = run["dataset"].get("provenance", {}).get("content_sha256")
    if not dataset_digest or dataset_digest != config.get("data_signature"):
        raise FinetuneEstimateError(
            "native run dataset identity differs from its training configuration"
        )
    if require_fixed_shape and run["workload"].get("fixed_shape") is not True:
        raise FinetuneEstimateError(
            "native extrapolation currently requires fixed full batches at sequence_length; "
            "dynamic shapes remain available in the measured run report"
        )
    if (
        not require_fixed_shape
        and config.get("batch_observation_version") != "fakegpu.finetune_batches.v1"
    ):
        raise FinetuneEstimateError(
            "dynamic validation requires the native batch observation protocol"
        )
    _validate_native_measurements(run, config, require_fixed_shape=require_fixed_shape)
    signature = finetune_workload_signature(
        run["model_config"], config, profile_id=run.get("profile_id")
    )
    _validate_observation(observation, signature, require_fixed_shape=require_fixed_shape)
    observed_model = observation.get("observed_model")
    # Early v1 runs predate kernel implementation recording. Preserve their
    # existing signature checks; the batch protocol requires the complete backend.
    has_backend = (
        isinstance(observed_model, Mapping) and "kernel_implementations" in observed_model
    )
    if "native_config" in run and ("batch_observation_version" in config or has_backend):
        raw = run["native_config"]
        if not isinstance(raw, Mapping) or not isinstance(observed_model, Mapping):
            raise FinetuneEstimateError(
                "native execution identity requires native_config and observed_model"
            )
        try:
            execution = native_execution_signature(
                native_config_signature(raw), observed_model
            )
        except KeyError as exc:
            raise FinetuneEstimateError(
                f"native observed_model lacks backend field {exc}"
            ) from exc
        if execution != config.get("native_execution_signature"):
            raise FinetuneEstimateError(
                "native execution signature differs from recorded configuration and backend"
            )
    return config, observation, run["workload"], run["timing"], run["schedule"]


def _nonnegative(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise FinetuneEstimateError(f"{field} must be finite and non-negative")
    return float(value)


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise FinetuneEstimateError(f"{field} must be a positive integer")
    return value


def _matching_seconds(value: Any, expected: float, field: str) -> None:
    measured = _nonnegative(value, field)
    if not math.isclose(measured, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise FinetuneEstimateError(f"{field} differs from recorded optimizer updates")


def _validate_native_measurements(
    run: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    require_fixed_shape: bool = True,
) -> None:
    """Reconcile redundant counters before treating a run as timing evidence."""
    workload, observation = run["workload"], run["observation"]
    # The runner records the effective Trainer schedule as well as the summary.
    # Legacy reports may lack these optional arguments; when present they must agree.
    model = observation.get("observed_model")
    arguments = model.get("training_arguments") if isinstance(model, Mapping) else None
    if isinstance(arguments, Mapping):
        for field in ("eval_strategy", "save_strategy"):
            if field in arguments and run["schedule"].get(field) != arguments[field]:
                raise FinetuneEstimateError(
                    "native schedule differs from recorded training arguments"
                )
    if workload != observation.get("workload"):
        raise FinetuneEstimateError(
            "native run workload differs from observation.workload"
        )
    if workload.get("kind") != "native_framework_processed_text":
        raise FinetuneEstimateError("native workload must describe processed text")
    batch = _positive_integer(config.get("micro_batch_size"), "micro_batch_size")
    accumulation = _positive_integer(
        config.get("gradient_accumulation_steps"), "gradient_accumulation_steps"
    )
    sequence = _positive_integer(
        workload.get("sequence_length"), "workload.sequence_length"
    )
    if sequence != config.get("sequence_length"):
        raise FinetuneEstimateError(
            "native workload sequence_length differs from training configuration"
        )
    counts = {
        field: _positive_integer(workload.get(field), f"workload.{field}")
        for field in (
            "optimizer_steps",
            "microsteps",
            "samples",
            "warmup_optimizer_steps",
            "measured_optimizer_steps",
            "warmup_microsteps",
            "measured_microsteps",
        )
    }
    if counts["measured_optimizer_steps"] < 2:
        raise FinetuneEstimateError(
            "native run requires at least two measured optimizer updates"
        )
    if (
        counts["warmup_optimizer_steps"] + counts["measured_optimizer_steps"]
        != counts["optimizer_steps"]
    ):
        raise FinetuneEstimateError(
            "native warmup and measured optimizer counts are inconsistent"
        )
    updates = observation.get("updates")
    if not isinstance(updates, list) or len(updates) != counts["optimizer_steps"]:
        raise FinetuneEstimateError(
            "native updates must match workload.optimizer_steps"
        )
    microsteps = samples = warmup_microsteps = 0
    update_seconds = warmup_seconds = 0.0
    steady_seconds: list[float] = []
    steady_per_microstep: list[float] = []
    for index, update in enumerate(updates):
        if not isinstance(update, Mapping):
            raise FinetuneEstimateError("native optimizer updates must be mappings")
        step = _positive_integer(update.get("optimizer_step"), "update.optimizer_step")
        if step != index + 1:
            raise FinetuneEstimateError(
                "native optimizer updates must be recorded in consecutive order"
            )
        warmup = update.get("warmup")
        if not isinstance(warmup, bool) or warmup != (
            index < counts["warmup_optimizer_steps"]
        ):
            raise FinetuneEstimateError(
                "native update warmup flags disagree with workload counts"
            )
        update_microsteps = _positive_integer(
            update.get("microsteps"), "update.microsteps"
        )
        update_samples = _positive_integer(update.get("samples"), "update.samples")
        if update_microsteps > accumulation:
            raise FinetuneEstimateError(
                "update.microsteps exceeds gradient_accumulation_steps"
            )
        # A partial accumulation window is legal, including during warmup. The
        # fixed-shape calibration requirement applies to each microbatch only.
        if require_fixed_shape and update_samples != batch * update_microsteps:
            raise FinetuneEstimateError(
                "native fixed-shape update.samples must match its full microbatches"
            )
        seconds = _nonnegative(update.get("seconds"), "update.seconds")
        if seconds == 0:
            raise FinetuneEstimateError("update.seconds must be positive")
        losses = update.get("losses")
        if (
            not isinstance(losses, list)
            or len(losses) != update_microsteps
            or any(
                isinstance(loss, bool)
                or not isinstance(loss, (int, float))
                or not math.isfinite(loss)
                for loss in losses
            )
        ):
            raise FinetuneEstimateError(
                "native update.losses must contain one finite loss per microstep"
            )
        microsteps += update_microsteps
        samples += update_samples
        update_seconds += seconds
        if warmup:
            warmup_microsteps += update_microsteps
            warmup_seconds += seconds
        else:
            steady_seconds.append(seconds)
            steady_per_microstep.append(seconds / update_microsteps)
    if (
        microsteps != counts["microsteps"]
        or samples != counts["samples"]
        or warmup_microsteps != counts["warmup_microsteps"]
        or microsteps - warmup_microsteps != counts["measured_microsteps"]
    ):
        raise FinetuneEstimateError(
            "native update microstep/sample totals disagree with workload counts"
        )
    for field, expected in (
        ("optimizer_update_seconds", steady_seconds),
        ("seconds_per_microstep", steady_per_microstep),
    ):
        values = observation.get(field)
        if not isinstance(values, list) or len(values) != len(expected):
            raise FinetuneEstimateError(
                f"native {field} must match measured update count"
            )
        for value, target in zip(values, expected):
            _matching_seconds(value, target, field)

    shapes = workload.get("shape_counts")
    if not isinstance(shapes, list) or not shapes:
        raise FinetuneEstimateError("native workload requires shape_counts")
    seen_shapes: set[tuple[int, int]] = set()
    shape_microsteps = shape_samples = 0
    for entry in shapes:
        if not isinstance(entry, Mapping):
            raise FinetuneEstimateError("native shape_counts entries must be mappings")
        shape = entry.get("shape")
        if not isinstance(shape, list) or len(shape) != 2:
            raise FinetuneEstimateError(
                "native shape_counts must contain two-dimensional shapes"
            )
        dimensions = tuple(
            _positive_integer(value, "shape dimension") for value in shape
        )
        if dimensions in seen_shapes:
            raise FinetuneEstimateError("native shape_counts contains duplicate shapes")
        if not require_fixed_shape and (
            dimensions[0] > batch or dimensions[1] > sequence
        ):
            raise FinetuneEstimateError(
                "native recorded batch shape exceeds the configured batch or sequence length"
            )
        seen_shapes.add(dimensions)
        count = _positive_integer(entry.get("microsteps"), "shape microsteps")
        shape_microsteps += count
        shape_samples += dimensions[0] * count
    if shape_microsteps != microsteps or shape_samples != samples:
        raise FinetuneEstimateError(
            "native shape_counts totals disagree with recorded updates"
        )
    if require_fixed_shape and seen_shapes != {(batch, sequence)}:
        raise FinetuneEstimateError(
            "native fixed_shape flag disagrees with recorded batch shapes"
        )

    timing = run["timing"]
    wall = _nonnegative(timing.get("total_wall_seconds"), "total_wall_seconds")
    loop = _nonnegative(timing.get("training_loop_seconds"), "training_loop_seconds")
    if wall < loop:
        raise FinetuneEstimateError(
            "native total wall time must cover its training loop"
        )
    if loop + 1e-6 < update_seconds:
        raise FinetuneEstimateError(
            "native training loop time must cover its recorded optimizer updates"
        )
    for field, expected in (
        ("optimizer_update_sample_seconds", update_seconds),
        ("warmup_training_seconds", warmup_seconds),
        ("trainer_other_seconds", max(0.0, loop - update_seconds)),
    ):
        if field in timing:
            _matching_seconds(timing[field], expected, field)


def _pipeline_projection(
    run: Mapping[str, Any], report: Mapping[str, Any]
) -> dict[str, Any]:
    """Preserve measured fixed costs; never scale preprocessing by raw rows."""
    timing = run["timing"]
    observed_wall = _nonnegative(timing.get("total_wall_seconds"), "total_wall_seconds")
    observed_loop = _nonnegative(
        timing.get("training_loop_seconds"), "training_loop_seconds"
    )
    if observed_wall < observed_loop:
        raise FinetuneEstimateError(
            "native total wall time must cover its training loop"
        )
    measured_microsteps = run["workload"].get("microsteps")
    if (
        isinstance(measured_microsteps, bool)
        or not isinstance(measured_microsteps, int)
        or measured_microsteps < 1
    ):
        raise FinetuneEstimateError("native run microsteps must be a positive integer")
    unavailable = []
    source_count = run["dataset"]["training_inputs"]["num_samples"]
    if report["workload"]["num_samples"] != source_count:
        unavailable.append(
            "A different dataset size has unmeasured preprocessing and loading costs."
        )
    schedule = run.get("schedule", {})
    if schedule.get("eval_strategy") != "no" or schedule.get("save_strategy") != "no":
        unavailable.append(
            "Periodic evaluation or checkpoint schedules need their own event-cost model."
        )
    if unavailable:
        return {
            "available": False,
            "reasons": unavailable,
            "observed_total_wall_seconds": observed_wall,
        }
    # The residual retains initialization, final save, warmup premium, and any
    # unmodeled per-update overhead. Only sampled steady microsteps scale here.
    steady = report["timing"]["seconds_per_microstep"]["expected"]
    fixed = observed_wall - steady * measured_microsteps
    if fixed < 0:
        return {
            "available": False,
            "reasons": [
                "Timing noise exceeds the measured fixed-cost residual; collect a longer calibration."
            ],
            "observed_total_wall_seconds": observed_wall,
        }
    target = report["timing"]["training_loop_seconds"]
    return {
        "available": True,
        "seconds": {key: fixed + value for key, value in target.items()},
        "evidence": "measured_fixed_cost_plus_steady_training_scenario",
        "fixed_cost_seconds": fixed,
        "observed_total_wall_seconds": observed_wall,
        "interval_kind": "steady_sample_extrema_with_one_fixed_cost_not_confidence_interval",
        "scope": "Native runner wall time, including instrumentation, preprocessing, model loading, warmup and final save; excludes Python process startup and downloads.",
        "limitations": [
            "One observed fixed cost is reused for the same dataset and local cache state; it is not a measured distribution.",
            "GPU clocks, other processes, cache state and compilation may change subsequent wall time.",
            "Synchronized instrumentation changes execution overhead and does not measure an uninstrumented framework invocation.",
            "Logging and observer work outside sampled updates remain in the fixed residual of this short calibration; their repeated cost may grow in longer runs and requires held-out run validation.",
        ],
    }
