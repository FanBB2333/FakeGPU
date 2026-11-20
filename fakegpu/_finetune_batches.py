"""Describe consumed text batches without inferring their packing algorithm.

Only binary two-dimensional attention masks establish nonpadding positions.
Position-ID zeros are observable offsets, not proof of original sample boundaries.
Timing groups preserve complete ordered accumulation windows: an update's elapsed
time cannot be attributed to its individual shapes or forward/backward calls.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
import math
import statistics
from typing import Any

from .structured_io import canonical_json


SCHEMA_VERSION = "fakegpu.finetune_batches.v1"
_BATCH_FIELDS = (
    "shape",
    "token_slots",
    "attention_mask_kind",
    "row_nonpadding_tokens",
    "nonpadding_tokens",
    "padding_tokens",
    "labels_present",
    "labels_kind",
    "row_supervised_tokens",
    "supervised_tokens",
    "row_causal_supervised_tokens",
    "causal_supervised_tokens",
    "position_ids_kind",
    "position_reset_offsets",
)
_TOKEN_FIELDS = (
    "nonpadding_tokens",
    "padding_tokens",
    "supervised_tokens",
    "causal_supervised_tokens",
)


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"native batch {field} must be an integer >= {minimum}")
    return value


def _seconds(value: Any, field: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (value <= 0 if positive else value < 0)
    ):
        raise ValueError(f"native batch {field} must be finite and nonnegative")
    return float(value)


def _shape(value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("native batch shape must have two positive dimensions")
    return tuple(_integer(size, "shape dimension", minimum=1) for size in value)


def _tensor_rows(value: Any, shape: tuple[int, int]) -> list[list[Any]] | None:
    """Read optional tensor metadata; unsupported representations stay unknown.

    Duck typing keeps Torch optional at import time. Actual tensor copy failures
    propagate instead of being disguised as missing metadata.
    """
    if value is None or tuple(getattr(value, "shape", ())) != shape:
        return None
    detach = getattr(value, "detach", None)
    if not callable(detach):
        return None
    rows = detach().cpu().tolist()
    if (
        not isinstance(rows, list)
        or len(rows) != shape[0]
        or any(not isinstance(row, list) or len(row) != shape[1] for row in rows)
    ):
        return None
    return rows


def _integer_rows(value: Any, shape: tuple[int, int], *, labels: bool = False):
    rows = _tensor_rows(value, shape)
    if rows is None or any(
        isinstance(item, bool)
        or not isinstance(item, int)
        or (item < 0 and not (labels and item == -100))
        for row in rows
        for item in row
    ):
        return None
    return rows


def describe_batch(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Read shape, masks, labels and position resets from one consumed batch.

    Token IDs are never copied or retained. Supervised tokens are nonignored
    labels at binary-mask-active positions. Causal supervised tokens apply the
    usual one-column label shift; position resets do not change that definition.
    """
    if not isinstance(inputs, Mapping) or "input_ids" not in inputs:
        raise ValueError("native batch requires a mapping with input_ids")
    shape = _shape(getattr(inputs["input_ids"], "shape", None))
    mask_value = inputs.get("attention_mask")
    mask = _tensor_rows(mask_value, shape)
    if mask is not None and any(
        not isinstance(value, (bool, int, float)) or value not in (0, 1)
        for row in mask
        for value in row
    ):
        mask = None
    labels_value = inputs.get("labels")
    labels = _integer_rows(labels_value, shape, labels=True)
    positions_value = inputs.get("position_ids")
    positions = _integer_rows(positions_value, shape)

    row_nonpadding = (
        [sum(int(value) for value in row) for row in mask] if mask is not None else None
    )
    row_supervised = row_causal = None
    if mask is not None and labels is not None:
        supervised = [
            [bool(active) and label != -100 for active, label in zip(row, targets)]
            for row, targets in zip(mask, labels)
        ]
        row_supervised = [sum(row) for row in supervised]
        row_causal = [sum(row[1:]) for row in supervised]
    resets = None
    if mask is not None and positions is not None:
        resets = [
            [
                offset
                for offset, (active, position) in enumerate(zip(row, position_row))
                if active and position == 0
            ]
            for row, position_row in zip(mask, positions)
        ]
    nonpadding = sum(row_nonpadding) if row_nonpadding is not None else None
    slots = math.prod(shape)
    return {
        "shape": list(shape),
        "token_slots": slots,
        "attention_mask_kind": (
            "binary_2d"
            if mask is not None
            else "missing"
            if mask_value is None
            else "unsupported"
        ),
        "row_nonpadding_tokens": row_nonpadding,
        "nonpadding_tokens": nonpadding,
        "padding_tokens": slots - nonpadding if nonpadding is not None else None,
        "labels_present": labels_value is not None,
        "labels_kind": (
            "integer_2d"
            if labels is not None
            else "missing"
            if labels_value is None
            else "unsupported"
        ),
        "row_supervised_tokens": row_supervised,
        "supervised_tokens": sum(row_supervised)
        if row_supervised is not None
        else None,
        "row_causal_supervised_tokens": row_causal,
        "causal_supervised_tokens": sum(row_causal) if row_causal is not None else None,
        "position_ids_kind": (
            "integer_2d"
            if positions is not None
            else "missing"
            if positions_value is None
            else "unsupported"
        ),
        "position_reset_offsets": resets,
    }


def _row_counts(batch, field, total_field, size, limits):
    rows = batch[field]
    if not isinstance(rows, list) or len(rows) != size:
        raise ValueError(f"native batch {field} must contain one count per row")
    for count, limit in zip(rows, limits):
        if _integer(count, field) > limit:
            raise ValueError(f"native batch {field} exceeds its row token count")
    if _integer(batch[total_field], total_field) != sum(rows):
        raise ValueError(f"native batch {total_field} differs from row counts")
    return rows


def _descriptor(batch: Any) -> dict[str, Any]:
    if not isinstance(batch, Mapping) or any(key not in batch for key in _BATCH_FIELDS):
        raise ValueError("native batch descriptor is incomplete")
    size, length = _shape(batch["shape"])
    if _integer(batch["token_slots"], "token_slots") != size * length:
        raise ValueError("native batch token_slots disagrees with shape")
    mask_kind = batch["attention_mask_kind"]
    if mask_kind not in ("binary_2d", "block_causal_4d", "missing", "unsupported"):
        raise ValueError("native batch attention_mask_kind is unsupported")
    labels_kind = batch["labels_kind"]
    position_kind = batch["position_ids_kind"]
    if labels_kind not in ("integer_2d", "missing", "unsupported"):
        raise ValueError("native batch labels_kind is unsupported")
    if position_kind not in ("integer_2d", "missing", "unsupported"):
        raise ValueError("native batch position_ids_kind is unsupported")
    if not isinstance(batch["labels_present"], bool) or batch["labels_present"] != (
        labels_kind != "missing"
    ):
        raise ValueError("native batch labels_present disagrees with labels_kind")
    if mask_kind in {"binary_2d", "block_causal_4d"}:
        nonpadding = _row_counts(
            batch, "row_nonpadding_tokens", "nonpadding_tokens", size, [length] * size
        )
        if _integer(batch["padding_tokens"], "padding_tokens") != size * length - sum(
            nonpadding
        ):
            raise ValueError(
                "native batch padding_tokens disagrees with nonpadding_tokens"
            )
    else:
        nonpadding = None
        for field in ("row_nonpadding_tokens", "nonpadding_tokens", "padding_tokens"):
            if batch[field] is not None:
                raise ValueError(
                    f"native batch {field} must be unknown without a binary mask"
                )
    if nonpadding is not None and labels_kind == "integer_2d":
        supervised = _row_counts(
            batch, "row_supervised_tokens", "supervised_tokens", size, nonpadding
        )
        causal = _row_counts(
            batch,
            "row_causal_supervised_tokens",
            "causal_supervised_tokens",
            size,
            [min(count, length - 1) for count in supervised],
        )
        if any(before - after > 1 for before, after in zip(supervised, causal)):
            raise ValueError(
                "native batch causal shift can remove at most one label per row"
            )
    else:
        for field in (
            "row_supervised_tokens",
            "supervised_tokens",
            "row_causal_supervised_tokens",
            "causal_supervised_tokens",
        ):
            if batch[field] is not None:
                raise ValueError(
                    f"native batch {field} requires a binary mask and integer labels"
                )
    resets = batch["position_reset_offsets"]
    if nonpadding is not None and position_kind == "integer_2d":
        if not isinstance(resets, list) or len(resets) != size:
            raise ValueError(
                "native batch position_reset_offsets must have one list per row"
            )
        for offsets, active in zip(resets, nonpadding):
            if not isinstance(offsets, list) or len(offsets) > active:
                raise ValueError(
                    "native batch position_reset_offsets exceeds active positions"
                )
            previous = -1
            for offset in offsets:
                if not previous < _integer(offset, "position reset offset") < length:
                    raise ValueError(
                        "native batch position reset offsets must be sorted, unique and in bounds"
                    )
                previous = offset
    elif resets is not None:
        raise ValueError(
            "native batch position resets require a binary mask and integer position_ids"
        )
    result = deepcopy({field: batch[field] for field in _BATCH_FIELDS})
    if mask_kind == "block_causal_4d":
        from ._finetune_packed_batches import validate_packed_geometry

        if labels_kind != "integer_2d" or position_kind != "integer_2d":
            raise ValueError("packed batches require integer labels and positions")
        result["packed"] = validate_packed_geometry(
            batch.get("packed"),
            shape=(size, length),
            nonpadding=nonpadding,
            resets=resets,
        )
    elif "packed" in batch:
        raise ValueError("packed geometry requires a validated block-causal mask")
    return result


def execution_batch_shape(batch):
    """Preserve legacy shape keys; packed work also binds segments and dtypes."""
    value = {
        field: deepcopy(batch[field])
        for field in (
            "shape",
            "attention_mask_kind",
            "labels_kind",
            "position_ids_kind",
        )
    }
    if batch["attention_mask_kind"] == "block_causal_4d":
        value["packed"] = deepcopy(batch["packed"])
    return value


def _canonical(value: Any) -> str:
    try:
        return canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "native batch profile must contain finite JSON values"
        ) from exc


def summarize_batch_updates(updates: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Group steady timing only for identical complete ordered update windows."""
    if not isinstance(updates, list) or not updates:
        raise ValueError("native batch profile requires recorded updates")
    shapes = Counter()
    totals = Counter()
    coverage = Counter()
    token_totals = Counter()
    groups = {}
    epoch_metadata = None
    previous_epoch = 0
    previous_microstep = 0
    steady_started = False
    for index, update in enumerate(updates, 1):
        if not isinstance(update, Mapping):
            raise ValueError("native batch update must be a mapping")
        if _integer(update.get("optimizer_step"), "optimizer_step", minimum=1) != index:
            raise ValueError("native batch optimizer steps must be consecutive")
        batches = update.get("batches")
        if not isinstance(batches, list) or not batches:
            raise ValueError("native batch profile requires batches for every update")
        descriptors = [_descriptor(batch) for batch in batches]
        seconds = _seconds(update.get("seconds"), "update.seconds", positive=True)
        warmup = update.get("warmup")
        if not isinstance(warmup, bool):
            raise ValueError("native batch update.warmup must be boolean")
        if warmup and steady_started:
            raise ValueError("native batch warmup updates must precede steady updates")
        steady_started |= not warmup
        if "wall_seconds" in update or "batch_instrumentation_seconds" in update:
            wall = _seconds(update.get("wall_seconds"), "update.wall_seconds")
            instrumentation = _seconds(
                update.get("batch_instrumentation_seconds"),
                "update.batch_instrumentation_seconds",
            )
            if instrumentation > wall or not math.isclose(
                wall - instrumentation, seconds, rel_tol=1e-9, abs_tol=1e-9
            ):
                raise ValueError(
                    "native batch update timing does not reconcile wall and instrumentation seconds"
                )
        samples = sum(batch["shape"][0] for batch in descriptors)
        if _integer(update.get("microsteps"), "update.microsteps", minimum=1) != len(
            batches
        ):
            raise ValueError("native batch count disagrees with update.microsteps")
        if _integer(update.get("samples"), "update.samples", minimum=1) != samples:
            raise ValueError("native batch row count disagrees with update.samples")
        totals["optimizer_steps"] += 1
        totals["warmup_optimizer_steps" if warmup else "measured_optimizer_steps"] += 1
        totals["microsteps"] += len(batches)
        totals["warmup_microsteps" if warmup else "measured_microsteps"] += len(batches)
        totals["samples"] += samples
        for batch, descriptor in zip(batches, descriptors):
            has_epoch = "epoch_index" in batch or "epoch_microstep" in batch
            if epoch_metadata is None:
                epoch_metadata = has_epoch
            if has_epoch != epoch_metadata:
                raise ValueError(
                    "native batch epoch metadata must be present consistently"
                )
            if has_epoch:
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
                    raise ValueError(
                        "native batch epoch microsteps must be consecutive"
                    )
                previous_epoch, previous_microstep = epoch, microstep
            shapes[tuple(descriptor["shape"])] += 1
            totals["token_slots"] += descriptor["token_slots"]
            for field in _TOKEN_FIELDS:
                if descriptor[field] is not None:
                    token_totals[field] += descriptor[field]
                    coverage[field.removesuffix("_tokens") + "_microsteps"] += 1
            if descriptor["position_reset_offsets"] is not None:
                coverage["position_reset_microsteps"] += 1
        key = _canonical(descriptors)
        if key not in groups:
            groups[key] = {
                "batches": descriptors,
                "warmup_updates": 0,
                "measured_updates": 0,
                "optimizer_update_seconds": [],
            }
        group = groups[key]
        group["warmup_updates" if warmup else "measured_updates"] += 1
        if not warmup:
            group["optimizer_update_seconds"].append(seconds)
    for group in groups.values():
        samples = group["optimizer_update_seconds"]
        group["sufficient"] = len(samples) >= 2
        group["seconds"] = (
            {
                "min": min(samples),
                "median": statistics.median(samples),
                "max": max(samples),
            }
            if samples
            else None
        )
    result_totals = {
        field: totals[field]
        for field in (
            "optimizer_steps",
            "warmup_optimizer_steps",
            "measured_optimizer_steps",
            "microsteps",
            "warmup_microsteps",
            "measured_microsteps",
            "samples",
            "token_slots",
        )
    }
    for field in _TOKEN_FIELDS:
        result_totals[field] = (
            token_totals[field]
            if coverage[field.removesuffix("_tokens") + "_microsteps"]
            == totals["microsteps"]
            else None
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "totals": result_totals,
        "coverage": {
            field: coverage[field]
            for field in (
                "nonpadding_microsteps",
                "padding_microsteps",
                "supervised_microsteps",
                "causal_supervised_microsteps",
                "position_reset_microsteps",
            )
        },
        "shape_counts": [
            {"shape": list(shape), "microsteps": count}
            for shape, count in sorted(shapes.items())
        ],
        "window_groups": list(groups.values()),
        "timing_scope": "Complete ordered optimizer-update windows; no per-shape timing or memory attribution.",
        "position_reset_definition": "Offsets where position_ids equals zero at binary-mask-active positions; these do not establish original sample boundaries.",
        "causal_supervision_definition": "Nonignored labels at active positions after dropping column zero; position resets do not change this count.",
    }


def validate_batch_profile(observation: Mapping[str, Any]) -> None:
    """Reconcile optional batch evidence, preserving observations predating it."""
    if not isinstance(observation, Mapping):
        raise ValueError("native batch observation must be a mapping")
    updates = observation.get("updates")
    has_batches = isinstance(updates, list) and any(
        isinstance(update, Mapping) and "batches" in update for update in updates
    )
    has_profile = "batch_profile" in observation
    signature = observation.get("signature", {})
    version = (
        signature.get("batch_observation_version")
        if isinstance(signature, Mapping)
        else None
    )
    if version is not None and version != SCHEMA_VERSION:
        raise ValueError("native batch observation version is unsupported")
    if not has_batches and not has_profile and version is None:
        return
    if not has_batches or not has_profile:
        raise ValueError(
            "native batch evidence requires both batches and batch_profile"
        )
    expected = summarize_batch_updates(updates)
    profile = observation["batch_profile"]
    if not isinstance(profile, Mapping) or _canonical(profile) != _canonical(expected):
        raise ValueError(
            "native batch_profile differs from recorded batches and updates"
        )
    workload = observation.get("workload")
    if not isinstance(workload, Mapping):
        raise ValueError("native batch evidence requires workload counts")
    for field in (
        "optimizer_steps",
        "microsteps",
        "samples",
        "warmup_optimizer_steps",
        "measured_optimizer_steps",
        "warmup_microsteps",
        "measured_microsteps",
    ):
        if (
            _integer(workload.get(field), "workload." + field)
            != expected["totals"][field]
        ):
            raise ValueError(
                f"native batch workload.{field} disagrees with recorded batches"
            )
    shape_counts = workload.get("shape_counts")
    if not isinstance(shape_counts, list):
        raise ValueError("native batch workload requires shape_counts")
    recorded_shapes = {}
    for entry in shape_counts:
        if not isinstance(entry, Mapping):
            raise ValueError("native batch shape_counts entries must be mappings")
        shape = _shape(entry.get("shape"))
        if shape in recorded_shapes:
            raise ValueError("native batch shape_counts contains duplicate shapes")
        recorded_shapes[shape] = _integer(
            entry.get("microsteps"), "shape microsteps", minimum=1
        )
    expected_shapes = {
        tuple(entry["shape"]): entry["microsteps"] for entry in expected["shape_counts"]
    }
    if recorded_shapes != expected_shapes:
        raise ValueError(
            "native batch workload.shape_counts disagrees with recorded batches"
        )
    measured = [update for update in updates if not update["warmup"]]
    for field, values in (
        ("optimizer_update_seconds", [update["seconds"] for update in measured]),
        (
            "seconds_per_microstep",
            [update["seconds"] / update["microsteps"] for update in measured],
        ),
    ):
        if field not in observation:
            continue
        recorded = observation[field]
        if not isinstance(recorded, list) or len(recorded) != len(values):
            raise ValueError(f"native batch {field} must match steady updates")
        for actual, value in zip(recorded, values):
            if not math.isclose(
                _seconds(actual, field, positive=True),
                value,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError(f"native batch {field} differs from recorded updates")
    if isinstance(signature, Mapping) and all(
        key in signature for key in ("micro_batch_size", "sequence_length")
    ):
        configured_shape = (
            _integer(
                signature["micro_batch_size"], "signature.micro_batch_size", minimum=1
            ),
            _integer(
                signature["sequence_length"], "signature.sequence_length", minimum=1
            ),
        )
        if not isinstance(workload.get("fixed_shape"), bool) or workload[
            "fixed_shape"
        ] != (set(recorded_shapes) == {configured_shape}):
            raise ValueError(
                "native batch fixed_shape disagrees with recorded shapes and signature"
            )
