"""Validate observed native evaluation/save calls without counting nested work twice.

The ledger describes actual successful calls. Its file inventory describes files
present after a save, not bytes written, storage traffic or durability.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import json
import math
from pathlib import PurePosixPath
from typing import Any

from ._finetune_batches import _descriptor

SCHEMA_VERSION = "fakegpu.finetune_events.v1"
_OPERATIONS = {
    "evaluate": "evaluation",
    "predict": "evaluation",
    "_save_checkpoint": "checkpoint",
    "save_model": "checkpoint",
    "save_pretrained": "checkpoint",
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _integer(value: Any, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"native event {field} must be an integer >= {minimum}")
    return value


def _seconds(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"native event {field} must be finite and nonnegative")
    return float(value)


def _matching(value: Any, expected: float, field: str) -> None:
    if not math.isclose(_seconds(value, field), expected, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"native event {field} differs from the event ledger")


def _optional_bool(value: Any, field: str) -> None:
    if value is not None and not isinstance(value, bool):
        raise ValueError(f"native event {field} must be boolean or null")


def _sha256(value: Any, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"native event {field} requires a SHA256 digest")


def _workload(event: Mapping[str, Any]) -> None:
    value = event.get("workload")
    if not isinstance(value, Mapping):
        raise ValueError("native event requires a workload mapping")
    if event["kind"] == "evaluation":
        dataset = value.get("dataset")
        if not isinstance(dataset, Mapping) or dataset.get("status") not in {
            "known",
            "unknown",
        }:
            raise ValueError("native evaluation requires dataset identity status")
        if dataset["status"] == "known":
            _sha256(dataset.get("content_sha256"), "evaluation.dataset")
            _integer(dataset.get("num_samples"), "dataset.num_samples")
        if value.get("consumed_content_status") not in {"known", "unknown"}:
            raise ValueError("native evaluation requires consumed content status")
        if value["consumed_content_status"] == "known":
            _sha256(value.get("consumed_content_sha256"), "evaluation.consumed_content")
        elif value.get("consumed_content_sha256") is not None:
            raise ValueError("unknown evaluation content requires a null digest")
        _integer(value.get("unobserved_batches"), "unobserved_batches")
        batches = value.get("batches")
        modes = value.get("prediction_loss_only")
        if not isinstance(batches, list) or not isinstance(modes, list):
            raise ValueError(
                "native evaluation requires batches and prediction_loss_only"
            )
        if len(batches) + value["unobserved_batches"] != len(modes):
            raise ValueError(
                "native evaluation prediction modes differ from batch count"
            )
        for batch in batches:
            _descriptor(batch)
        for mode in modes:
            _optional_bool(mode, "prediction_loss_only")
        for field in ("predict_with_generate", "dataloader_drop_last"):
            if field not in value:
                raise ValueError(f"native evaluation requires {field}")
            _optional_bool(value.get(field), field)
        if not isinstance(value.get("compute_metrics"), bool):
            raise ValueError("native evaluation compute_metrics must be boolean")
        if "eval_batch_size" not in value:
            raise ValueError("native evaluation requires eval_batch_size")
        if value.get("eval_batch_size") is not None:
            _integer(value["eval_batch_size"], "eval_batch_size", 1)
    else:
        _optional_bool(value.get("save_only_model"), "save_only_model")
        if value.get("artifact_kind") not in {"adapter", "model", "mixed", "unknown"}:
            raise ValueError("native checkpoint requires artifact_kind")
        if value.get("files_status") not in {"known", "unknown", "unavailable"}:
            raise ValueError("native checkpoint requires files_status")
        files = value.get("files")
        directories = value.get("output_directories")
        if (
            not isinstance(files, list)
            or not isinstance(directories, list)
            or any(not isinstance(path, str) for path in directories)
        ):
            raise ValueError(
                "native checkpoint requires file and directory inventories"
            )
        seen = set()
        total = 0
        for file in files:
            if not isinstance(file, Mapping):
                raise ValueError("native checkpoint file must be a mapping")
            path = file.get("path")
            if (
                not isinstance(path, str)
                or not path
                or PurePosixPath(path).is_absolute()
                or ".." in PurePosixPath(path).parts
                or path in seen
            ):
                raise ValueError(
                    "native checkpoint file paths must be unique and relative"
                )
            seen.add(path)
            total += _integer(file.get("bytes"), "file.bytes")
        if value["files_status"] == "known":
            if _integer(value.get("total_file_bytes"), "total_file_bytes") != total:
                raise ValueError(
                    "native checkpoint total_file_bytes differs from inventory"
                )
        elif value.get("total_file_bytes") is not None:
            raise ValueError(
                "unknown checkpoint inventory requires null total_file_bytes"
            )


def summarize_phase_events(events: Any) -> dict[str, Any]:
    """Validate an ordered call tree and summarize only its nonoverlapping roots."""
    if not isinstance(events, list):
        raise ValueError("native phase_events must be a list")
    indexed = {}
    siblings = {}
    roots = []
    child_wall = Counter()
    child_instrumentation = Counter()
    last_step = 0
    last_start = 0.0
    for expected_id, event in enumerate(events, 1):
        if not isinstance(event, Mapping):
            raise ValueError("native phase event must be a mapping")
        event_id = _integer(event.get("event_id"), "event_id", 1)
        if event_id != expected_id:
            raise ValueError("native event IDs must be consecutive in call-start order")
        operation = event.get("operation")
        if operation not in _OPERATIONS or event.get("kind") != _OPERATIONS[operation]:
            raise ValueError("native event kind differs from its operation")
        if not isinstance(event.get("inside_train"), bool):
            raise ValueError("native event inside_train must be boolean")
        if event.get("status") not in {"succeeded", "failed"}:
            raise ValueError("native event status must be succeeded or failed")
        step = _integer(event.get("optimizer_step"), "optimizer_step")
        if step < last_step:
            raise ValueError("native event optimizer steps must not decrease")
        last_step = step
        if event.get("epoch") is not None:
            _seconds(event["epoch"], "epoch")
        start = _seconds(event.get("start_seconds"), "start_seconds")
        if start < last_start:
            raise ValueError("native event IDs must follow chronological call starts")
        last_start = start
        end = _seconds(event.get("end_seconds"), "end_seconds")
        wall = _seconds(event.get("wall_seconds"), "wall_seconds")
        metadata = _seconds(
            event.get("instrumentation_seconds"), "instrumentation_seconds"
        )
        if end < start or metadata > wall + 1e-9:
            raise ValueError("native event timing bounds are inconsistent")
        _matching(wall, end - start, "wall_seconds")
        _matching(event.get("seconds"), wall - metadata, "seconds")
        _workload(event)
        parent_id = event.get("parent_event_id")
        if parent_id is not None:
            _integer(parent_id, "parent_event_id", 1)
            parent = indexed.get(parent_id)
            if parent is None:
                raise ValueError("native event parent must precede its child")
            if (
                start < parent["start_seconds"] - 1e-9
                or end > parent["end_seconds"] + 1e-9
                or event["inside_train"] != parent["inside_train"]
                or step != parent["optimizer_step"]
            ):
                raise ValueError("native child event is inconsistent with its parent")
            child_wall[parent_id] += wall
            child_instrumentation[parent_id] += metadata
        else:
            roots.append(event)
        previous = siblings.get(parent_id)
        if previous and start < previous["end_seconds"] - 1e-9:
            raise ValueError("native sibling events overlap or are out of order")
        siblings[parent_id] = event
        indexed[event_id] = event
    for event_id, event in indexed.items():
        if child_wall[event_id] > event["wall_seconds"] + 1e-9 or (
            child_instrumentation[event_id] > event["instrumentation_seconds"] + 1e-9
        ):
            raise ValueError("native child timing exceeds its enclosing event")
    kinds = {}
    for kind in ("evaluation", "checkpoint"):
        selected = [event for event in roots if event["kind"] == kind]
        inside = [event for event in selected if event["inside_train"]]
        kinds[kind] = {
            "root_events": len(selected),
            "seconds": sum(event["seconds"] for event in selected),
            "wall_seconds": sum(event["wall_seconds"] for event in selected),
            "instrumentation_seconds": sum(
                event["instrumentation_seconds"] for event in selected
            ),
            "during_training_root_events": len(inside),
            "during_training_seconds": sum(event["seconds"] for event in inside),
            "during_training_wall_seconds": sum(
                event["wall_seconds"] for event in inside
            ),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "events": len(events),
        "root_events": len(roots),
        "nested_events": len(events) - len(roots),
        "failed_events": sum(event["status"] == "failed" for event in events),
        "root_event_ids": [event["event_id"] for event in roots],
        "seconds": sum(event["seconds"] for event in roots),
        "wall_seconds": sum(event["wall_seconds"] for event in roots),
        "instrumentation_seconds": sum(
            event["instrumentation_seconds"] for event in roots
        ),
        "kinds": kinds,
        "timing_definition": "Root durations include nested calls exactly once; metadata time is excluded from seconds, but retained in wall_seconds.",
        "file_definition": "File inventories describe files present after calls, not bytes written, storage traffic, or durable writes.",
    }


def validate_event_profile(
    observation: Mapping[str, Any], timing: Mapping[str, Any] | None = None
) -> None:
    """Reconcile event evidence; reports predating the protocol remain valid."""
    signature = observation.get("signature")
    version = (
        signature.get("event_observation_version")
        if isinstance(signature, Mapping)
        else None
    )
    present = [key in observation for key in ("phase_events", "event_profile")]
    if version is None and not any(present):
        return
    if version != SCHEMA_VERSION or not all(present):
        raise ValueError(
            "native event protocol requires its version, ledger and event_profile"
        )
    expected = summarize_phase_events(observation["phase_events"])
    if not isinstance(observation["event_profile"], Mapping) or _canonical(
        observation["event_profile"]
    ) != _canonical(expected):
        raise ValueError("native event_profile differs from recorded phase_events")
    workload = observation.get("workload")
    if isinstance(workload, Mapping) and "optimizer_steps" in workload:
        steps = _integer(workload["optimizer_steps"], "workload.optimizer_steps")
        if any(
            event["optimizer_step"] > steps for event in observation["phase_events"]
        ):
            raise ValueError("native event optimizer step exceeds consumed updates")
    if "updates" in observation:
        preceding = {}
        for event in observation["phase_events"]:
            if event["parent_event_id"] is None and event["inside_train"]:
                preceding.setdefault(event["optimizer_step"] + 1, []).append(
                    event["event_id"]
                )
        for update in observation["updates"]:
            ids = update.get("preceding_event_ids")
            if (
                not isinstance(ids, list)
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in ids
                )
                or ids != preceding.get(update["optimizer_step"], [])
            ):
                raise ValueError(
                    "native update preceding_event_ids differs from root events"
                )
    if timing is None:
        return
    if "event_instrumentation_seconds" in timing:
        _matching(
            timing["event_instrumentation_seconds"],
            expected["instrumentation_seconds"],
            "timing.event_instrumentation_seconds",
        )
    for kind, summary in expected["kinds"].items():
        for field, count in (
            ("eventcounts", summary["root_events"]),
            ("during_training_eventcounts", summary["during_training_root_events"]),
        ):
            recorded = timing.get(field)
            if (
                not isinstance(recorded, Mapping)
                or _integer(recorded.get(kind), field) != count
            ):
                raise ValueError(
                    f"native event timing.{field} differs from root events"
                )
        _matching(
            timing.get(f"{kind}_seconds"),
            summary["wall_seconds"],
            f"timing.{kind}_seconds",
        )
        _matching(
            timing.get(f"during_training_{kind}_seconds"),
            summary["during_training_wall_seconds"],
            f"timing.during_training_{kind}_seconds",
        )
    if observation["phase_events"]:
        last_end = max(event["end_seconds"] for event in observation["phase_events"])
        if (
            last_end
            > _seconds(timing.get("total_wall_seconds"), "total_wall_seconds") + 1e-9
        ):
            raise ValueError("native event timestamps exceed total wall time")
    if "setup_seconds" in timing and "trainer_wall_seconds" in timing:
        train_start = _seconds(timing["setup_seconds"], "setup_seconds")
        train_end = train_start + _seconds(
            timing["trainer_wall_seconds"], "trainer_wall_seconds"
        )
        for event in observation["phase_events"]:
            start, end = event["start_seconds"], event["end_seconds"]
            if event["inside_train"]:
                valid = start >= train_start - 1e-9 and end <= train_end + 1e-9
            else:
                valid = end <= train_start + 1e-9 or start >= train_end - 1e-9
            if not valid:
                raise ValueError(
                    "native event inside_train disagrees with Trainer timing bounds"
                )
