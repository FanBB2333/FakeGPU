"""Partition one observed native execution without overlapping measured work.

All coordinates share the runner's perf_counter origin. Adjusted seconds remove
only explicitly measured observation costs, not all Python instrumentation or
checkpoint identity hashing. Report construction and CLI output are outside the
native execution span.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from typing import Any

from ._finetune_events import validate_event_profile

SCHEMA_VERSION = "fakegpu.finetune_timeline.v1"
_KINDS = (
    "setup",
    "trainer_other",
    "finalization",
    "optimizer_update",
    "evaluation",
    "checkpoint",
)
_METADATA_KINDS = {
    "trainer_inspection",
    "train_begin_inspection",
    "train_end_parameter_check",
    "loss_collection",
}


def _seconds(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"native timeline {field} must be finite and nonnegative")
    return float(value)


def _matching(value: Any, expected: float, field: str) -> None:
    if not math.isclose(_seconds(value, field), expected, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"native timeline {field} differs from recorded intervals")


def _bounds(value: Any, field: str) -> tuple[float, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"native timeline requires {field} bounds")
    start = _seconds(value.get("start_seconds"), f"{field}.start_seconds")
    end = _seconds(value.get("end_seconds"), f"{field}.end_seconds")
    if end < start:
        raise ValueError(f"native timeline {field} ends before it starts")
    return start, end


def _segment(kind, start, end, instrumentation=0.0, *, step=None, event_id=None):
    wall = end - start
    if instrumentation > wall:
        raise ValueError("native timeline instrumentation exceeds its segment")
    return {
        "kind": kind,
        "start_seconds": start,
        "end_seconds": end,
        "wall_seconds": wall,
        "instrumentation_seconds": instrumentation,
        "seconds": wall - instrumentation,
        "optimizer_step": step,
        "event_id": event_id,
    }


def build_native_timeline(
    updates: list,
    phase_events: list,
    instrumentation_intervals: list,
    *,
    trainer_start_seconds: float,
    trainer_end_seconds: float,
    run_end_seconds: float,
) -> dict[str, Any]:
    """Sweep update/root intervals and fill all gaps, split at Trainer bounds.

    The existing event validator owns call-tree and workload validation. This
    builder checks placement and the measured metadata of roots and updates.
    """
    train_start, train_end = _bounds(
        {"start_seconds": trainer_start_seconds, "end_seconds": trainer_end_seconds},
        "trainer",
    )
    run_end = _seconds(run_end_seconds, "span.end_seconds")
    if train_end > run_end:
        raise ValueError("native timeline Trainer bounds exceed the execution span")
    if not isinstance(updates, list) or not updates:
        raise ValueError("native timeline requires recorded optimizer updates")
    if not isinstance(phase_events, list) or not isinstance(
        instrumentation_intervals, list
    ):
        raise ValueError("native timeline requires event and instrumentation ledgers")
    work = []
    for index, update in enumerate(updates, 1):
        start, end = _bounds(update, "update")
        if (
            isinstance(update.get("optimizer_step"), bool)
            or update.get("optimizer_step") != index
        ):
            raise ValueError("native timeline optimizer steps must be consecutive")
        if start < train_start or end > train_end:
            raise ValueError(
                "native timeline optimizer update is outside Trainer bounds"
            )
        if index > 1 and start < work[-1]["end_seconds"]:
            raise ValueError(
                "native timeline optimizer updates overlap or are unordered"
            )
        metadata = _seconds(
            update.get("batch_instrumentation_seconds"), "update metadata"
        )
        segment = _segment("optimizer_update", start, end, metadata, step=index)
        _matching(
            update.get("wall_seconds"), segment["wall_seconds"], "update.wall_seconds"
        )
        _matching(update.get("seconds"), segment["seconds"], "update.seconds")
        work.append(segment)
    for event in phase_events:
        if not isinstance(event, Mapping):
            raise ValueError("native timeline requires event mappings")
        if event.get("parent_event_id") is not None:
            continue
        start, end = _bounds(event, "root event")
        if end > run_end:
            raise ValueError("native timeline root event exceeds the execution span")
        if event.get("inside_train"):
            if start < train_start or end > train_end:
                raise ValueError("native timeline root event is outside Trainer bounds")
            step = event["optimizer_step"]
            if (
                step > len(updates)
                or (step and start < updates[step - 1]["end_seconds"])
                or (step < len(updates) and end > updates[step]["start_seconds"])
            ):
                raise ValueError(
                    "native timeline event step disagrees with update intervals"
                )
        elif end > train_start and start < train_end:
            raise ValueError(
                "native timeline outside-train event crosses Trainer bounds"
            )
        metadata = _seconds(event.get("instrumentation_seconds"), "event metadata")
        segment = _segment(
            event["kind"],
            start,
            end,
            metadata,
            step=event["optimizer_step"],
            event_id=event["event_id"],
        )
        _matching(
            event.get("wall_seconds"), segment["wall_seconds"], "event.wall_seconds"
        )
        _matching(event.get("seconds"), segment["seconds"], "event.seconds")
        work.append(segment)
    work.sort(key=lambda segment: (segment["start_seconds"], segment["end_seconds"]))
    segments = []

    def add_gaps(start, end):
        for left, right, kind in (
            (0.0, train_start, "setup"),
            (train_start, train_end, "trainer_other"),
            (train_end, run_end, "finalization"),
        ):
            gap_start, gap_end = max(start, left), min(end, right)
            if gap_end > gap_start:
                segments.append(_segment(kind, gap_start, gap_end))

    cursor = 0.0
    for segment in work:
        if segment["start_seconds"] < cursor:
            raise ValueError(
                "native timeline root events and optimizer updates overlap"
            )
        add_gaps(cursor, segment["start_seconds"])
        segments.append(segment)
        cursor = segment["end_seconds"]
    add_gaps(cursor, run_end)

    # These callbacks run outside measured updates/events. Keeping their actual
    # coordinates avoids subtracting setup costs from an unrelated Trainer gap.
    gap_segments = [segment for segment in segments if segment["kind"] in _KINDS[:3]]
    previous_end = 0.0
    gap_index = 0
    for interval in instrumentation_intervals:
        start, end = _bounds(interval, "instrumentation")
        if interval.get("kind") not in _METADATA_KINDS:
            raise ValueError("native timeline has an unknown instrumentation kind")
        if start < previous_end or end > run_end:
            raise ValueError(
                "native timeline instrumentation intervals overlap or exceed the span"
            )
        previous_end = end
        _matching(interval.get("seconds"), end - start, "instrumentation.seconds")
        if start == end:
            continue
        while (
            gap_index < len(gap_segments)
            and start >= gap_segments[gap_index]["end_seconds"]
        ):
            gap_index += 1
        if gap_index == len(gap_segments):
            raise ValueError("native timeline instrumentation is outside a gap")
        gap = gap_segments[gap_index]
        if start < gap["start_seconds"] or end > gap["end_seconds"]:
            raise ValueError(
                "native timeline instrumentation overlaps measured work or a boundary"
            )
        gap["instrumentation_seconds"] += end - start
        gap["seconds"] = gap["wall_seconds"] - gap["instrumentation_seconds"]
    fields = ("wall_seconds", "instrumentation_seconds", "seconds")
    by_kind = {
        kind: {
            field: sum(
                segment[field] for segment in segments if segment["kind"] == kind
            )
            for field in fields
        }
        for kind in _KINDS
    }
    totals = {field: sum(segment[field] for segment in segments) for field in fields}
    _matching(totals["wall_seconds"], run_end, "partition.wall_seconds")
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "native_execution_span",
        "clock": "perf_counter_relative_to_run_start",
        "span": {"start_seconds": 0.0, "end_seconds": run_end},
        "trainer": {"start_seconds": train_start, "end_seconds": train_end},
        "segments": segments,
        "totals": {**totals, "by_kind": by_kind},
        "timing_definition": "The partition covers runner entry through native execution and final synchronization, before report construction or CLI output. Adjusted seconds remove measured batch, event and callback metadata once; checkpoint identity hashing and unmeasured instrumentation remain included.",
    }


def validate_native_timeline(
    observation: Mapping[str, Any], timing: Mapping[str, Any] | None = None
) -> None:
    """Rebuild the partition and reconcile it with all existing timing ledgers."""
    signature = observation.get("signature")
    version = (
        signature.get("timeline_observation_version")
        if isinstance(signature, Mapping)
        else None
    )
    present = any(
        key in observation for key in ("timeline", "instrumentation_intervals")
    )
    # Earlier protocols have no coordinates on updates. Keep their behavior.
    has_update_coordinates = any(
        isinstance(update, Mapping)
        and ("start_seconds" in update or "end_seconds" in update)
        for update in observation.get("updates", [])
    )
    if version is None and not present and not has_update_coordinates:
        return
    timeline = observation.get("timeline")
    if version != SCHEMA_VERSION or not isinstance(timeline, Mapping):
        raise ValueError(
            "native timeline requires its version, timeline and instrumentation ledger"
        )
    if signature.get("event_observation_version") is None:
        raise ValueError("native timeline requires the native event protocol")
    validate_event_profile(observation, timing=timing)
    span_start, span_end = _bounds(timeline.get("span"), "span")
    if span_start != 0.0:
        raise ValueError("native timeline must start at the runner origin")
    train_start, train_end = _bounds(timeline.get("trainer"), "trainer")
    expected = build_native_timeline(
        observation.get("updates"),
        observation.get("phase_events"),
        observation.get("instrumentation_intervals"),
        trainer_start_seconds=train_start,
        trainer_end_seconds=train_end,
        run_end_seconds=span_end,
    )
    if json.dumps(timeline, sort_keys=True, allow_nan=False) != json.dumps(
        expected, sort_keys=True, allow_nan=False
    ):
        raise ValueError("native timeline differs from recorded intervals")
    if timing is None:
        return
    updates = observation["updates"]
    totals = expected["totals"]["by_kind"]
    batch_metadata = totals["optimizer_update"]["instrumentation_seconds"]
    callback_metadata = sum(
        item["seconds"] for item in observation["instrumentation_intervals"]
    )
    inside_events = sum(
        event["wall_seconds"]
        for event in observation["phase_events"]
        if event["parent_event_id"] is None and event["inside_train"]
    )
    loop = train_end - train_start - inside_events
    for field, value in {
        "total_wall_seconds": span_end,
        "setup_seconds": train_start,
        "trainer_wall_seconds": train_end - train_start,
        "finalization_seconds": span_end - train_end,
        "training_loop_seconds": loop,
        "optimizer_update_sample_seconds": totals["optimizer_update"]["seconds"],
        "warmup_training_seconds": sum(
            update["seconds"] for update in updates if update["warmup"]
        ),
        "trainer_other_seconds": totals["trainer_other"]["wall_seconds"]
        + batch_metadata,
        "batch_instrumentation_seconds": batch_metadata,
        "instrumentation_seconds": batch_metadata
        + callback_metadata
        + observation["event_profile"]["instrumentation_seconds"],
    }.items():
        _matching(timing.get(field), value, f"timing.{field}")
