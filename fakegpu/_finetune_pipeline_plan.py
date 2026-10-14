"""Timing-free native pipeline plans and their complete semantic partition.

Plans retain actual batch order and explicit root calls. A plan is not evidence
that a future sampler or framework will execute that order; a public planner
must establish that separately before using this representation.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
from typing import Any

from ._finetune_allocator import validate_memory_allocator
from ._finetune_batches import _descriptor
from ._finetune_cache import validate_cache_regime
from ._finetune_event_context import describe_root_events
from ._finetune_events import _OPERATIONS
from ._finetune_timeline import SCHEMA_VERSION as TIMELINE_SCHEMA
from .finetune_estimator import FinetuneEstimateError
from .finetune_run_estimator import validate_native_run
from .finetune_window_estimator import _DENSE_MODELS

SCHEMA_VERSION = "fakegpu.finetune_pipeline_plan.v1"
_WORK_KINDS = {"optimizer_update", "evaluation", "checkpoint"}
_PREPROCESSING_FIELDS = (
    "overwrite_cache",
    "tokenized_path",
    "dataset_prepared_path",
    "cache_dir",
)
_TIME_FIELDS = {
    "seconds",
    "wall_seconds",
    "instrumentation_seconds",
    "start_seconds",
    "end_seconds",
    "observed_seconds",
    "predicted_seconds",
    "timing",
    "comparison",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FinetuneEstimateError(
            f"pipeline plan {field} must be an integer >= {minimum}"
        )
    return value


def _without_times(value: Any) -> None:
    if isinstance(value, Mapping):
        if _TIME_FIELDS.intersection(value):
            raise FinetuneEstimateError("pipeline plans must not contain timing fields")
        for child in value.values():
            _without_times(child)
    elif isinstance(value, list):
        for child in value:
            _without_times(child)


def _validate_execution_policy(policy: Mapping[str, Any], steps: int) -> None:
    required = {
        "logging_strategy",
        "requested_logging_steps",
        "resolved_logging_steps",
        "logging_first_step",
        "save_total_limit",
    }
    if set(policy) != required:
        raise FinetuneEstimateError("pipeline execution_policy is incomplete")
    strategy = policy["logging_strategy"]
    requested = policy["requested_logging_steps"]
    if strategy not in {"no", "steps"}:
        raise FinetuneEstimateError(
            "pipeline calibration supports logging_strategy=no or steps with logging every update"
        )
    if (
        isinstance(requested, bool)
        or not isinstance(requested, (int, float))
        or not math.isfinite(requested)
        or requested < 0
        or (strategy == "steps" and requested == 0)
        or (requested >= 1 and requested != int(requested))
    ):
        raise FinetuneEstimateError("pipeline requested_logging_steps is invalid")
    resolved = _integer(policy["resolved_logging_steps"], "resolved_logging_steps")
    expected = math.ceil(steps * requested) if 0 < requested < 1 else int(requested)
    if resolved != expected:
        raise FinetuneEstimateError(
            "pipeline resolved logging steps disagree with its workload"
        )
    if strategy == "steps" and resolved != 1:
        raise FinetuneEstimateError(
            "pipeline calibration supports logging only every update; periodic logging requires another context"
        )
    if not isinstance(policy["logging_first_step"], bool):
        raise FinetuneEstimateError("pipeline logging_first_step must be boolean")
    if policy["save_total_limit"] is not None:
        _integer(policy["save_total_limit"], "save_total_limit", minimum=1)


def _execution_policy_from_run(run: Mapping[str, Any]) -> dict[str, Any]:
    arguments = run["observation"]["observed_model"].get("training_arguments")
    if not isinstance(arguments, Mapping):
        raise FinetuneEstimateError(
            "pipeline calibration requires actual TrainingArguments for logging and checkpoint policy"
        )
    if any(
        arguments.get(field) for field in ("enable_torch_profiler", "profile_modules")
    ):
        raise FinetuneEstimateError(
            "pipeline calibration does not model profiler callback phases"
        )
    if (
        "torch_empty_cache_steps" not in arguments
        or arguments["torch_empty_cache_steps"] is not None
    ):
        raise FinetuneEstimateError(
            "pipeline calibration requires explicit torch_empty_cache_steps=None; periodic cache clearing is not modeled"
        )
    aliases = {
        "logging_strategy": "logging_strategy",
        "requested_logging_steps": "logging_steps",
        "logging_first_step": "logging_first_step",
        "save_total_limit": "save_total_limit",
    }
    schedule = run["schedule"]
    policy = {}
    for field, argument in aliases.items():
        if argument not in arguments or field not in schedule:
            raise FinetuneEstimateError(
                f"pipeline calibration requires recorded {field}"
            )
        if arguments[argument] != schedule[field]:
            raise FinetuneEstimateError(
                f"pipeline {field} differs from actual TrainingArguments"
            )
        policy[field] = deepcopy(arguments[argument])
    if "resolved_logging_steps" not in schedule:
        raise FinetuneEstimateError(
            "pipeline calibration requires actual resolved_logging_steps"
        )
    policy["resolved_logging_steps"] = schedule["resolved_logging_steps"]
    _validate_execution_policy(policy, run["workload"]["optimizer_steps"])
    return policy


def validate_pipeline_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate order, counters and references, returning an owned plan copy.

    Event workloads must subsequently match validated calibration descriptors.
    This structural validator does not authenticate plans or prove CPU schedule
    generation agrees with a particular installed native framework.
    """
    if not isinstance(plan, Mapping) or plan.get("schema_version") != SCHEMA_VERSION:
        raise FinetuneEstimateError("unsupported native pipeline plan schema")
    _without_times(plan)
    if plan.get("source") not in {"recorded_native_timeline", "cpu_native_schedule"}:
        raise FinetuneEstimateError("pipeline plan requires its schedule source")
    for field in (
        "signature",
        "hardware",
        "memory_allocator",
        "execution_policy",
        "cache_regime",
        "setup_workload",
        "warmup",
        "workload",
    ):
        if not isinstance(plan.get(field), Mapping) or not plan[field]:
            raise FinetuneEstimateError(f"pipeline plan requires {field}")
    try:
        validate_cache_regime(plan["cache_regime"])
        validate_memory_allocator(plan["memory_allocator"])
    except ValueError as exc:
        raise FinetuneEstimateError(str(exc)) from exc
    signature = plan["signature"]
    if signature.get("source_framework") not in {
        "llamafactory",
        "axolotl",
        "huggingface",
    }:
        raise FinetuneEstimateError(
            "pipeline plans require a supported native framework"
        )
    if (
        signature.get("packing") is not False
        or _integer(signature.get("world_size"), "world_size", minimum=1) != 1
    ):
        raise FinetuneEstimateError(
            "pipeline plans require single-GPU unpacked execution"
        )
    if signature.get("timeline_observation_version") != TIMELINE_SCHEMA:
        raise FinetuneEstimateError(
            "pipeline plans require the native timeline protocol"
        )
    setup = plan["setup_workload"]
    if setup.get("dataset_content_sha256") != signature.get(
        "data_signature"
    ) or setup.get("checkpoint_sha256") != signature.get("checkpoint_sha256"):
        raise FinetuneEstimateError(
            "pipeline plan setup workload differs from its signature"
        )
    _integer(setup.get("dataset_num_samples"), "setup dataset_num_samples", minimum=1)
    preprocessing = setup.get("preprocessing_policy")
    if not isinstance(preprocessing, Mapping) or set(preprocessing) != set(
        _PREPROCESSING_FIELDS
    ):
        raise FinetuneEstimateError(
            "pipeline setup requires the complete preprocessing_policy"
        )
    warmup = _integer(plan["warmup"].get("optimizer_steps"), "warmup.optimizer_steps")
    updates, roots, operations = (
        plan.get(key) for key in ("updates", "root_events", "operations")
    )
    if not isinstance(updates, list) or not updates:
        raise FinetuneEstimateError("pipeline plan requires ordered updates")
    if not isinstance(roots, list) or not isinstance(operations, list):
        raise FinetuneEstimateError(
            "pipeline plan requires root events and ordered operations"
        )
    if warmup > len(updates):
        raise FinetuneEstimateError("pipeline plan warmup exceeds its updates")
    _validate_execution_policy(plan["execution_policy"], len(updates))
    microsteps = samples = previous_epoch = previous_microstep = 0
    accumulation = _integer(
        signature.get("gradient_accumulation_steps"), "accumulation", minimum=1
    )
    batch_size = _integer(
        signature.get("micro_batch_size"), "micro_batch_size", minimum=1
    )
    for index, update in enumerate(updates, 1):
        if (
            not isinstance(update, Mapping)
            or _integer(update.get("optimizer_step"), "optimizer_step", minimum=1)
            != index
        ):
            raise FinetuneEstimateError(
                "pipeline plan optimizer steps must be consecutive"
            )
        if not isinstance(update.get("warmup"), bool) or update["warmup"] != (
            index <= warmup
        ):
            raise FinetuneEstimateError(
                "pipeline plan warmup flags disagree with its count"
            )
        batches = update.get("batches")
        if not isinstance(batches, list) or not 1 <= len(batches) <= accumulation:
            raise FinetuneEstimateError(
                "pipeline plan requires a legal accumulation window"
            )
        for batch in batches:
            try:
                descriptor = _descriptor(batch)
            except ValueError as exc:
                raise FinetuneEstimateError(str(exc)) from exc
            if (
                descriptor["attention_mask_kind"] != "binary_2d"
                or descriptor["labels_kind"] != "integer_2d"
            ):
                raise FinetuneEstimateError(
                    "pipeline plans require known masks and labels"
                )
            if descriptor["shape"][0] > batch_size:
                raise FinetuneEstimateError(
                    "pipeline plan batch exceeds micro_batch_size"
                )
            epoch = _integer(batch.get("epoch_index"), "epoch_index", minimum=1)
            microstep = _integer(
                batch.get("epoch_microstep"), "epoch_microstep", minimum=1
            )
            expected = (
                (previous_epoch, previous_microstep + 1)
                if epoch == previous_epoch
                else (previous_epoch + 1, 1)
            )
            if (epoch, microstep) != expected:
                raise FinetuneEstimateError(
                    "pipeline plan epoch microsteps must be consecutive"
                )
            previous_epoch, previous_microstep = epoch, microstep
            microsteps += 1
            samples += descriptor["shape"][0]
        preceding = update.get("preceding_event_ids")
        if not isinstance(preceding, list):
            raise FinetuneEstimateError(
                "pipeline plan requires preceding root event IDs"
            )
        for value in preceding:
            _integer(value, "preceding_event_id", minimum=1)
    for field, expected in (
        ("optimizer_steps", len(updates)),
        ("microsteps", microsteps),
        ("samples", samples),
    ):
        if _integer(plan["workload"].get(field), field, minimum=1) != expected:
            raise FinetuneEstimateError(
                f"pipeline plan workload.{field} differs from its updates"
            )

    root_by_id = {}
    seen_events = set()
    for root in roots:
        if not isinstance(root, Mapping):
            raise FinetuneEstimateError("pipeline plan root events must be mappings")
        event_id = _integer(root.get("event_id"), "event_id", minimum=1)
        if event_id in root_by_id:
            raise FinetuneEstimateError("pipeline plan root event IDs must be unique")
        step = _integer(root.get("optimizer_step"), "event.optimizer_step")
        descriptor = root.get("descriptor")
        if step > len(updates) or not isinstance(descriptor, Mapping):
            raise FinetuneEstimateError("pipeline plan has an invalid root event")
        kind = descriptor.get("kind")
        if _OPERATIONS.get(descriptor.get("operation")) != kind or kind not in {
            "evaluation",
            "checkpoint",
        }:
            raise FinetuneEstimateError("pipeline plan has an unknown event operation")
        if not isinstance(descriptor.get("inside_train"), bool):
            raise FinetuneEstimateError(
                "pipeline plan event inside_train must be boolean"
            )
        if not isinstance(descriptor.get("workload"), Mapping) or not isinstance(
            descriptor.get("nested_operations"), list
        ):
            raise FinetuneEstimateError(
                "pipeline plan requires complete event descriptors"
            )
        base = {key: value for key, value in descriptor.items() if key != "phase"}
        key = _canonical(base)
        phase = (
            "initial"
            if step == 0
            else "final"
            if not descriptor["inside_train"]
            else "repeated"
            if key in seen_events
            else "first"
        )
        if descriptor.get("phase") != phase:
            raise FinetuneEstimateError(
                "pipeline plan event phase disagrees with call order"
            )
        if phase in {"first", "repeated"}:
            seen_events.add(key)
        root_by_id[event_id] = root

    limit = plan["execution_policy"]["save_total_limit"]
    checkpoints = sum(
        root["descriptor"]["operation"] == "_save_checkpoint" for root in roots
    )
    if limit is not None and checkpoints > limit:
        raise FinetuneEstimateError(
            "pipeline checkpoint rotation is not modeled; planned checkpoints exceed save_total_limit"
        )

    state, next_step, consumed_roots, pending = "setup", 1, [], []
    for operation in operations:
        if not isinstance(operation, Mapping):
            raise FinetuneEstimateError("pipeline plan operations must be mappings")
        kind = operation.get("kind")
        if kind == "trainer_start" and state == "setup":
            state = "trainer"
        elif (
            kind == "trainer_end"
            and state == "trainer"
            and next_step == len(updates) + 1
        ):
            state = "finalization"
        elif kind == "optimizer_update" and state == "trainer":
            if (
                _integer(
                    operation.get("optimizer_step"),
                    "operation.optimizer_step",
                    minimum=1,
                )
                != next_step
            ):
                raise FinetuneEstimateError(
                    "pipeline plan operation update order is inconsistent"
                )
            if updates[next_step - 1]["preceding_event_ids"] != pending:
                raise FinetuneEstimateError(
                    "pipeline plan preceding events disagree with operation order"
                )
            pending = []
            next_step += 1
        elif kind in {"evaluation", "checkpoint"}:
            event_id = _integer(
                operation.get("event_id"), "operation.event_id", minimum=1
            )
            root = root_by_id.get(event_id)
            if (
                root is None
                or event_id in consumed_roots
                or root["descriptor"]["kind"] != kind
            ):
                raise FinetuneEstimateError(
                    "pipeline plan operation references an invalid root event"
                )
            if (
                root["descriptor"]["inside_train"] != (state == "trainer")
                or root["optimizer_step"] != next_step - 1
            ):
                raise FinetuneEstimateError(
                    "pipeline plan event placement disagrees with its step"
                )
            consumed_roots.append(event_id)
            if state == "trainer":
                pending.append(event_id)
        else:
            raise FinetuneEstimateError(
                "pipeline plan has an invalid operation or Trainer boundary"
            )
    if state != "finalization" or consumed_roots != [
        root["event_id"] for root in roots
    ]:
        raise FinetuneEstimateError(
            "pipeline plan must consume all roots and both Trainer boundaries"
        )
    _canonical(plan)
    return deepcopy(dict(plan))


def pipeline_plan_from_run(run: Mapping[str, Any]) -> dict[str, Any]:
    """Strip all target timings from a validated complete native timeline."""
    config, observation, _, _, _ = validate_native_run(run, require_fixed_shape=False)
    if config.get("timeline_observation_version") != TIMELINE_SCHEMA:
        raise FinetuneEstimateError(
            "pipeline calibration requires the native timeline protocol"
        )
    model = run["model_config"]
    if model.get("model_type") not in _DENSE_MODELS or any(
        model.get(field, 0)
        for field in ("num_experts", "num_local_experts", "n_routed_experts")
    ):
        raise FinetuneEstimateError(
            "pipeline calibration currently requires a dense model"
        )
    if "native_config" not in run or "observed_model" not in observation:
        raise FinetuneEstimateError(
            "pipeline calibration requires native execution evidence"
        )
    timeline = observation["timeline"]
    ordered = [
        (timeline["trainer"]["start_seconds"], 0, {"kind": "trainer_start"}),
        (timeline["trainer"]["end_seconds"], 0, {"kind": "trainer_end"}),
    ]
    for segment in timeline["segments"]:
        if segment["kind"] in _WORK_KINDS:
            reference = {"kind": segment["kind"]}
            field = (
                "optimizer_step"
                if segment["kind"] == "optimizer_update"
                else "event_id"
            )
            reference[field] = segment[field]
            priority = 1
            if segment["kind"] != "optimizer_update":
                event = observation["phase_events"][segment["event_id"] - 1]
                # Zero-duration calls can share a Trainer boundary. Their
                # recorded inside_train flag fixes which side owns the call.
                if (
                    not event["inside_train"]
                    and segment["end_seconds"] <= timeline["trainer"]["start_seconds"]
                ) or (
                    event["inside_train"]
                    and segment["start_seconds"] == timeline["trainer"]["end_seconds"]
                ):
                    priority = -1
            ordered.append((segment["start_seconds"], priority, reference))
    signature = observation["signature"]
    plan = {
        "schema_version": SCHEMA_VERSION,
        "source": "recorded_native_timeline",
        "signature": deepcopy(signature),
        "hardware": deepcopy(observation["hardware"]),
        "memory_allocator": deepcopy(observation.get("memory_allocator")),
        "execution_policy": _execution_policy_from_run(run),
        "cache_regime": deepcopy(run.get("cache_regime")),
        "setup_workload": {
            "dataset_content_sha256": signature["data_signature"],
            "dataset_num_samples": run["dataset"]["training_inputs"]["num_samples"],
            "checkpoint_sha256": signature["checkpoint_sha256"],
            "preprocessing_policy": {
                field: deepcopy(run["native_config"].get(field))
                for field in _PREPROCESSING_FIELDS
            },
        },
        "warmup": {"optimizer_steps": run["workload"]["warmup_optimizer_steps"]},
        "workload": {
            field: run["workload"][field]
            for field in ("optimizer_steps", "microsteps", "samples")
        },
        "updates": [
            {
                field: deepcopy(update[field])
                for field in (
                    "optimizer_step",
                    "warmup",
                    "batches",
                    "preceding_event_ids",
                )
            }
            for update in observation["updates"]
        ],
        "root_events": describe_root_events(observation["phase_events"]),
        "operations": [
            operation for _, _, operation in sorted(ordered, key=lambda item: item[:2])
        ],
    }
    return validate_pipeline_plan(plan)


def describe_pipeline_components(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Describe every work interval and intervening gap, including zero gaps.

    Callers first validate the plan. Semantic hashes omit absolute steps, epochs,
    event IDs and all elapsed times. Gap hashes bind both neighboring complete
    work descriptors; the descriptors themselves appear on their work groups.
    """
    return _describe_components(plan, _pipeline_window)


def _pipeline_window(plan, update, context):
    return {
        "batches": [_descriptor(batch) for batch in update["batches"]],
        "context": context,
        "epoch_context": {
            "batch_microsteps": [
                batch["epoch_microstep"] for batch in update["batches"]
            ]
        },
    }


def _describe_components(plan, describe_window):
    """Partition a validated plan using the selected cost grouping protocol."""
    roots = {root["event_id"]: root for root in plan["root_events"]}
    latest = {"evaluation": None, "checkpoint": None}
    seen_windows, seen_gaps = set(), set()
    result = []
    previous = {"kind": "native_start"}
    previous_reference = {"kind": "native_start"}
    region = "setup"

    def append(kind, descriptor, minimum, reference):
        result.append(
            {
                "component_id": len(result) + 1,
                "kind": kind,
                "descriptor": descriptor,
                "group_id": _digest(descriptor),
                "minimum_samples_per_run": minimum,
                "reference": reference,
            }
        )

    for operation in [*plan["operations"], {"kind": "native_end"}]:
        kind = operation["kind"]
        descriptor = {"kind": kind}
        minimum = 1
        if kind == "optimizer_update":
            update = plan["updates"][operation["optimizer_step"] - 1]
            preceding = [roots[event_id] for event_id in update["preceding_event_ids"]]
            if preceding:
                boundary = {
                    "step": preceding[0]["optimizer_step"],
                    "events": [root["descriptor"] for root in preceding],
                }
                for root in preceding:
                    latest[root["descriptor"]["kind"]] = boundary
            context = {
                name: None
                if boundary is None
                else {
                    "events": boundary["events"],
                    "optimizer_updates_since_boundary": update["optimizer_step"]
                    - boundary["step"]
                    - 1,
                }
                for name, boundary in latest.items()
            }
            window = describe_window(plan, update, context)
            key = _canonical(window)
            first = key not in seen_windows
            seen_windows.add(key)
            descriptor.update(
                window=window,
                stage={
                    "warmup_index": update["optimizer_step"]
                    if update["warmup"]
                    else None,
                    "first_descriptor_occurrence": first,
                },
            )
            minimum = 1 if update["warmup"] or first else 2
        elif kind in {"evaluation", "checkpoint"}:
            event = roots[operation["event_id"]]["descriptor"]
            descriptor["event"] = deepcopy(event)
            minimum = 2 if event["phase"] == "repeated" else 1
        semantic = (
            {"kind": kind, "group_id": _digest(descriptor)}
            if kind in _WORK_KINDS
            else descriptor
        )
        gap = {"kind": region, "left": previous, "right": semantic}
        key = _canonical(gap)
        first = key not in seen_gaps
        seen_gaps.add(key)
        gap["phase"] = "first" if first else "repeated"
        append(
            region,
            gap,
            1 if first else 2,
            {"left": previous_reference, "right": deepcopy(operation)},
        )
        if kind in _WORK_KINDS:
            append(kind, descriptor, minimum, deepcopy(operation))
        if kind == "trainer_start":
            region = "trainer_other"
        elif kind == "trainer_end":
            region = "finalization"
        previous, previous_reference = semantic, deepcopy(operation)
    return result
