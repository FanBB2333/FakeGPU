"""Canonical native root-event descriptors shared by calibration APIs."""

from __future__ import annotations

from collections.abc import Mapping
import json
import re
from typing import Any

from ._finetune_batches import _descriptor
from .finetune_estimator import FinetuneEstimateError


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _workload_key(event: Mapping[str, Any]) -> dict[str, Any]:
    workload = event["workload"]
    if event["kind"] == "evaluation":
        if (
            workload["dataset"]["status"] != "known"
            or workload.get("consumed_content_status") != "known"
            or workload.get("unobserved_batches", 0)
            or not workload["batches"]
        ):
            raise FinetuneEstimateError(
                "event calibration requires known evaluation data and consumed batches"
            )
        batches = [_descriptor(batch) for batch in workload["batches"]]
        if any(
            batch["attention_mask_kind"] != "binary_2d"
            or batch["labels_kind"] != "integer_2d"
            for batch in batches
        ):
            raise FinetuneEstimateError(
                "event calibration requires known evaluation masks and labels"
            )
        if workload.get("predict_with_generate") is not False:
            raise FinetuneEstimateError(
                "event calibration currently requires nongenerative evaluation"
            )
        if workload["compute_metrics"] or any(
            value is not True for value in workload["prediction_loss_only"]
        ):
            raise FinetuneEstimateError(
                "event calibration requires loss-only evaluation without custom metrics"
            )
        return {
            "dataset": {
                field: workload["dataset"][field]
                for field in ("content_sha256", "num_samples")
            },
            "batches": batches,
            "consumed_content_sha256": workload["consumed_content_sha256"],
            **{
                field: workload[field]
                for field in (
                    "prediction_loss_only",
                    "predict_with_generate",
                    "compute_metrics",
                    "eval_batch_size",
                    "dataloader_drop_last",
                )
            },
        }
    if workload["artifact_kind"] == "unknown" or workload["files_status"] != "known":
        raise FinetuneEstimateError(
            "event calibration requires a known checkpoint inventory"
        )
    if not workload["files"]:
        raise FinetuneEstimateError(
            "event calibration requires nonempty checkpoint files"
        )
    if workload["save_only_model"] is None:
        raise FinetuneEstimateError(
            "event calibration requires the save_only_model policy"
        )
    return {
        "save_only_model": workload["save_only_model"],
        "artifact_kind": workload["artifact_kind"],
        # Step names and volatile JSON lengths change without changing save semantics.
        # Files are evidence of the resulting inventory, never a proxy for I/O traffic.
        "file_layout": sorted(
            re.sub(r"(^|/)checkpoint-\d+(/|$)", r"\1checkpoint-N\2", file["path"])
            for file in workload["files"]
        ),
    }


def describe_root_events(events: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Describe validated calls in order, without timing or run-specific identities.

    Initial/final calls do not consume a training event's first occurrence.
    The event ledger must already have passed the native report validator.
    """
    if any(event["status"] != "succeeded" for event in events):
        raise FinetuneEstimateError("event calibration requires successful event calls")
    described = []
    seen = set()
    for event in events:
        if event["parent_event_id"] is not None:
            continue
        descendants = {event["event_id"]: 0}
        subtree = []
        for child in events:
            if child["parent_event_id"] in descendants:
                descendants[child["event_id"]] = len(descendants)
                subtree.append(
                    {
                        "kind": child["kind"],
                        "operation": child["operation"],
                        "parent_operation_index": descendants[child["parent_event_id"]],
                    }
                )
        descriptor = {
            "kind": event["kind"],
            "operation": event["operation"],
            "inside_train": event["inside_train"],
            "workload": _workload_key(event),
            "nested_operations": subtree,
        }
        key = _canonical(descriptor)
        if event["optimizer_step"] == 0:
            phase = "initial"
        elif not event["inside_train"]:
            phase = "final"
        else:
            phase = "repeated" if key in seen else "first"
        if phase in {"first", "repeated"}:
            seen.add(key)
        descriptor["phase"] = phase
        described.append(
            {
                "event_id": event["event_id"],
                "optimizer_step": event["optimizer_step"],
                "descriptor": descriptor,
            }
        )
    return described
