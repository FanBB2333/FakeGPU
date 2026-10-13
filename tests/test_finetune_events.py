from __future__ import annotations

from copy import deepcopy

import pytest

from fakegpu._finetune_events import (
    SCHEMA_VERSION,
    summarize_phase_events,
    validate_event_profile,
)
from test_finetune_window_estimator import _batch


def evaluation_workload():
    return {
        "dataset": {"status": "known", "content_sha256": "d" * 64, "num_samples": 1},
        "batches": [_batch(4)],
        "prediction_loss_only": [True],
        "predict_with_generate": False,
        "compute_metrics": False,
        "eval_batch_size": 1,
        "dataloader_drop_last": False,
        "consumed_content_status": "known",
        "consumed_content_sha256": "e" * 64,
        "unobserved_batches": 0,
    }


def checkpoint_workload():
    return {
        "save_only_model": False,
        "artifact_kind": "adapter",
        "files_status": "known",
        "files": [
            {"path": "adapter_model.safetensors", "bytes": 1024},
            {"path": "trainer_state.json", "bytes": 20},
        ],
        "total_file_bytes": 1044,
        "output_directories": ["/runs/checkpoint-3"],
    }


def event(
    event_id=1,
    *,
    operation="evaluate",
    start=0.0,
    seconds=1.0,
    parent=None,
    step=1,
    inside=True,
):
    kind = "evaluation" if operation in {"evaluate", "predict"} else "checkpoint"
    return {
        "event_id": event_id,
        "parent_event_id": parent,
        "kind": kind,
        "operation": operation,
        "inside_train": inside,
        "optimizer_step": step,
        "epoch": 1.0,
        "start_seconds": start,
        "end_seconds": start + seconds,
        "wall_seconds": seconds,
        "instrumentation_seconds": 0.0,
        "seconds": seconds,
        "status": "succeeded",
        "workload": evaluation_workload()
        if kind == "evaluation"
        else checkpoint_workload(),
    }


def observation(events):
    return {
        "signature": {"event_observation_version": SCHEMA_VERSION},
        "phase_events": events,
        "event_profile": summarize_phase_events(events),
        "workload": {"optimizer_steps": 3},
    }


def test_nested_calls_are_counted_once_and_keep_their_own_evidence():
    events = [
        event(operation="_save_checkpoint", seconds=3),
        event(2, operation="save_model", start=0.5, parent=1),
        event(3, operation="save_pretrained", start=0.6, seconds=0.5, parent=2),
        event(4, start=4, seconds=2, inside=False),
    ]
    profile = summarize_phase_events(events)
    assert profile["events"] == 4
    assert profile["root_events"] == 2
    assert profile["root_event_ids"] == [1, 4]
    assert profile["seconds"] == profile["wall_seconds"] == 5
    assert profile["kinds"]["checkpoint"]["during_training_seconds"] == 3
    assert profile["kinds"]["evaluation"]["during_training_root_events"] == 0


def test_metadata_time_is_subtracted_once_for_the_whole_root():
    events = [
        event(operation="_save_checkpoint", seconds=3),
        event(2, operation="save_model", start=1, parent=1),
    ]
    for entry, overhead in zip(events, [0.3, 0.1]):
        entry["instrumentation_seconds"] = overhead
        entry["seconds"] -= overhead
    profile = summarize_phase_events(events)
    assert profile["seconds"] == 2.7
    assert profile["instrumentation_seconds"] == 0.3


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda e: e[0].update(event_id=2), "consecutive"),
        (lambda e: e[0].update(optimizer_step=True), "integer"),
        (lambda e: e[0].update(operation="unknown"), "operation"),
        (lambda e: e[0].update(inside_train=1), "boolean"),
        (lambda e: e[0].update(seconds=float("nan")), "finite"),
        (lambda e: e[0].update(wall_seconds=8), "wall_seconds"),
        (lambda e: e[0].update(seconds=0.9), "seconds"),
        (lambda e: e[0].update(parent_event_id=1), "parent"),
        (lambda e: e[1].update(inside_train=False), "parent"),
        (lambda e: e[1].update(optimizer_step=2), "parent"),
        (lambda e: e[1].update(parent_event_id=None), "overlap"),
        (
            lambda e: e[1].update(instrumentation_seconds=0.1, seconds=0.9),
            "child timing",
        ),
    ],
)
def test_invalid_tree_and_timing_are_rejected(mutation, match):
    events = [
        event(operation="_save_checkpoint", seconds=3),
        event(2, operation="save_model", start=1, parent=1),
    ]
    mutation(events)
    with pytest.raises(ValueError, match=match):
        summarize_phase_events(events)


def test_preorder_does_not_allow_a_child_record_after_a_later_root():
    events = [event(seconds=3), event(2, start=4), event(3, start=1, parent=1)]
    with pytest.raises(ValueError, match="chronological"):
        summarize_phase_events(events)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda w: w["dataset"].update(content_sha256="unknown"), "SHA256"),
        (lambda w: w.update(prediction_loss_only=[]), "batch count"),
        (lambda w: w.update(consumed_content_status="unknown"), "null digest"),
        (lambda w: w.update(consumed_content_sha256="bad"), "SHA256"),
        (lambda w: w.update(unobserved_batches=-1), "integer"),
        (lambda w: w.update(eval_batch_size=True), "integer"),
        (lambda w: w.update(compute_metrics=1), "boolean"),
    ],
)
def test_evaluation_evidence_is_validated(mutate, match):
    entry = event()
    mutate(entry["workload"])
    with pytest.raises(ValueError, match=match):
        summarize_phase_events([entry])


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda w: w["files"].append(w["files"][0]), "unique"),
        (lambda w: w["files"][0].update(path="../escape"), "relative"),
        (lambda w: w["files"][0].update(path="/absolute"), "relative"),
        (lambda w: w.update(total_file_bytes=100), "total_file_bytes"),
        (lambda w: w["files"][0].update(bytes=True), "integer"),
        (lambda w: w.update(files_status="unknown"), "null total"),
    ],
)
def test_checkpoint_inventory_is_evidence_not_an_io_counter(mutate, match):
    entry = event(operation="save_model")
    mutate(entry["workload"])
    with pytest.raises(ValueError, match=match):
        summarize_phase_events([entry])


def test_unknown_evaluation_data_can_be_reported_without_inventing_a_digest():
    entry = event()
    entry["workload"].update(
        dataset={"status": "unknown", "reason": "iterable"},
        consumed_content_status="unknown",
        consumed_content_sha256=None,
    )
    assert summarize_phase_events([entry])["root_events"] == 1


def test_empty_protocol_and_legacy_reports_are_valid():
    validate_event_profile({})
    validate_event_profile(observation([]))


@pytest.mark.parametrize("field", ["phase_events", "event_profile"])
def test_protocol_cannot_be_silently_downgraded(field):
    report = observation([event()])
    del report[field]
    with pytest.raises(ValueError, match="version, ledger and event_profile"):
        validate_event_profile(report)


def test_profile_and_optimizer_bounds_must_agree():
    report = observation([event()])
    report["event_profile"]["seconds"] += 1
    with pytest.raises(ValueError, match="event_profile differs"):
        validate_event_profile(report)
    report = observation([event(step=4)])
    with pytest.raises(ValueError, match="exceeds consumed"):
        validate_event_profile(report)


def test_timing_aggregates_and_inside_training_counts_are_reconciled():
    report = observation(
        [event(seconds=2), event(2, operation="save_model", start=3, inside=False)]
    )
    timing = {
        "eventcounts": {"evaluation": 1, "checkpoint": 1},
        "during_training_eventcounts": {"evaluation": 1, "checkpoint": 0},
        "evaluation_seconds": 2,
        "checkpoint_seconds": 1,
        "during_training_evaluation_seconds": 2,
        "during_training_checkpoint_seconds": 0,
        "total_wall_seconds": 5,
    }
    validate_event_profile(report, timing)
    before = deepcopy(timing)
    timing["during_training_eventcounts"]["evaluation"] = 0
    with pytest.raises(ValueError, match="during_training_eventcounts"):
        validate_event_profile(report, timing)
    timing = before
    timing["checkpoint_seconds"] = 2
    with pytest.raises(ValueError, match="checkpoint_seconds"):
        validate_event_profile(report, timing)


def test_unknown_batches_keep_the_successful_prediction_mode_count():
    entry = event()
    entry["workload"]["batches"] = []
    entry["workload"]["unobserved_batches"] = 1
    assert summarize_phase_events([entry])["root_events"] == 1


def test_failed_calls_remain_in_the_measured_tree_and_unknown_status_is_rejected():
    entry = event()
    entry["status"] = "failed"
    assert summarize_phase_events([entry])["failed_events"] == 1
    entry["status"] = "pending"
    with pytest.raises(ValueError, match="status"):
        summarize_phase_events([entry])


def test_unavailable_file_inventory_preserves_an_actual_call():
    entry = event(operation="save_model")
    entry["workload"].update(
        files_status="unavailable", files=[], total_file_bytes=None
    )
    assert summarize_phase_events([entry])["root_events"] == 1


def test_event_instrumentation_and_training_bounds_are_reconciled():
    report = observation([event(start=2, seconds=2)])
    timing = {
        "eventcounts": {"evaluation": 1, "checkpoint": 0},
        "during_training_eventcounts": {"evaluation": 1, "checkpoint": 0},
        "evaluation_seconds": 2,
        "checkpoint_seconds": 0,
        "during_training_evaluation_seconds": 2,
        "during_training_checkpoint_seconds": 0,
        "total_wall_seconds": 6,
        "setup_seconds": 1,
        "trainer_wall_seconds": 4,
        "event_instrumentation_seconds": 0,
    }
    validate_event_profile(report, timing)
    timing["event_instrumentation_seconds"] = 1
    with pytest.raises(ValueError, match="event_instrumentation_seconds"):
        validate_event_profile(report, timing)
    timing["event_instrumentation_seconds"] = 0
    timing["setup_seconds"] = 3
    with pytest.raises(ValueError, match="inside_train"):
        validate_event_profile(report, timing)
