from copy import deepcopy

import pytest

from fakegpu._finetune_events import summarize_phase_events
from fakegpu._finetune_timeline import (
    SCHEMA_VERSION,
    build_native_timeline,
    validate_native_timeline,
)
from test_finetune_events import event


def _fixture():
    events = [
        event(start=11, seconds=2),
        event(2, operation="_save_checkpoint", start=13, seconds=3),
        event(3, operation="save_model", start=13.5, seconds=1.5, parent=2),
        event(4, operation="save_pretrained", start=14, seconds=0.8, parent=3),
        event(5, operation="save_model", start=24, seconds=2, step=2, inside=False),
    ]
    for row, metadata in zip(events, [0.2, 0.5, 0.2, 0.1, 0.1]):
        row["instrumentation_seconds"] = metadata
        row["seconds"] -= metadata
    updates = [
        {
            "optimizer_step": 1,
            "start_seconds": 6.0,
            "end_seconds": 10.0,
            "wall_seconds": 4.0,
            "seconds": 3.0,
            "batch_instrumentation_seconds": 1.0,
            "preceding_event_ids": [],
            "warmup": True,
        },
        {
            "optimizer_step": 2,
            "start_seconds": 17.0,
            "end_seconds": 21.0,
            "wall_seconds": 4.0,
            "seconds": 3.6,
            "batch_instrumentation_seconds": 0.4,
            "preceding_event_ids": [1, 2],
            "warmup": False,
        },
    ]
    intervals = [
        {
            "kind": kind,
            "start_seconds": start,
            "end_seconds": end,
            "seconds": end - start,
        }
        for kind, start, end in (
            ("trainer_inspection", 1.0, 2.0),
            ("train_begin_inspection", 5.0, 5.5),
            ("loss_collection", 10.0, 10.25),
            ("loss_collection", 21.0, 21.25),
            ("train_end_parameter_check", 22.0, 22.5),
        )
    ]
    observation = {
        "signature": {
            "event_observation_version": "fakegpu.finetune_events.v1",
            "timeline_observation_version": SCHEMA_VERSION,
        },
        "updates": updates,
        "phase_events": events,
        "event_profile": summarize_phase_events(events),
        "instrumentation_intervals": intervals,
        "workload": {"optimizer_steps": 2},
    }
    observation["timeline"] = _build(observation)
    timing = {
        "total_wall_seconds": 30.0,
        "setup_seconds": 5.0,
        "trainer_wall_seconds": 18.0,
        "training_loop_seconds": 13.0,
        "finalization_seconds": 7.0,
        "optimizer_update_sample_seconds": 6.6,
        "warmup_training_seconds": 3.0,
        "trainer_other_seconds": 6.4,
        "batch_instrumentation_seconds": 1.4,
        "event_instrumentation_seconds": 0.8,
        "instrumentation_seconds": 4.7,
        "checkpoint_hash_seconds": 0.75,
        "evaluation_seconds": 2.0,
        "checkpoint_seconds": 5.0,
        "during_training_evaluation_seconds": 2.0,
        "during_training_checkpoint_seconds": 3.0,
        "eventcounts": {"evaluation": 1, "checkpoint": 2},
        "during_training_eventcounts": {"evaluation": 1, "checkpoint": 1},
    }
    return observation, timing


def _build(observation):
    return build_native_timeline(
        observation["updates"],
        observation["phase_events"],
        observation["instrumentation_intervals"],
        trainer_start_seconds=5.0,
        trainer_end_seconds=23.0,
        run_end_seconds=30.0,
    )


def test_complete_partition_preserves_gaps_and_counts_nested_metadata_once():
    observation, timing = _fixture()
    validate_native_timeline(observation, timing)
    timeline = observation["timeline"]
    segments = timeline["segments"]
    assert segments[0]["start_seconds"] == 0
    assert segments[-1]["end_seconds"] == 30
    assert all(
        left["end_seconds"] == right["start_seconds"]
        for left, right in zip(segments, segments[1:])
    )
    assert [row["event_id"] for row in segments if row["event_id"]] == [1, 2, 5]
    totals = timeline["totals"]
    assert totals["wall_seconds"] == 30
    assert totals["instrumentation_seconds"] == pytest.approx(4.7)
    assert totals["seconds"] == pytest.approx(25.3)
    assert totals["by_kind"]["trainer_other"] == {
        "wall_seconds": 5.0,
        "instrumentation_seconds": 1.5,
        "seconds": 3.5,
    }
    # Legacy trainer_other includes metadata inside updates. It is not raw gap.
    assert timing["trainer_other_seconds"] == pytest.approx(
        totals["by_kind"]["trainer_other"]["wall_seconds"] + 1.4
    )
    # The hash is already inside setup. It is neither extra wall nor metadata.
    assert totals["by_kind"]["setup"]["seconds"] == 4.0


@pytest.mark.parametrize(
    "target,field",
    [
        ("updates", "start_seconds"),
        ("updates", "end_seconds"),
        ("updates", "wall_seconds"),
        ("updates", "batch_instrumentation_seconds"),
    ],
)
def test_new_protocol_requires_complete_update_coordinates(target, field):
    observation, timing = _fixture()
    del observation[target][0][field]
    with pytest.raises(ValueError, match="native timeline"):
        validate_native_timeline(observation, timing)


@pytest.mark.parametrize("field", ["timeline", "instrumentation_intervals"])
def test_new_protocol_requires_its_ledgers(field):
    observation, timing = _fixture()
    del observation[field]
    with pytest.raises(ValueError, match="native timeline"):
        validate_native_timeline(observation, timing)


def test_legacy_reports_are_unchanged_but_partial_protocol_is_rejected():
    validate_native_timeline({"signature": {}, "updates": [{"seconds": 1.0}]})
    with pytest.raises(ValueError, match="requires its version"):
        validate_native_timeline({"signature": {}, "updates": [{"start_seconds": 0.0}]})


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda o: o["updates"][0].update(start_seconds=4), "outside Trainer"),
        (
            lambda o: o["updates"][0].update(end_seconds=12, wall_seconds=6, seconds=5),
            "event step",
        ),
        (
            lambda o: o["updates"][1].update(
                start_seconds=9, wall_seconds=12, seconds=11.6
            ),
            "overlap",
        ),
        (
            lambda o: o["updates"][1].update(
                end_seconds=24, wall_seconds=7, seconds=6.6
            ),
            "outside Trainer",
        ),
        (lambda o: o["updates"][0].update(seconds=4), "update.seconds"),
        (
            lambda o: o["instrumentation_intervals"][0].update(
                start_seconds=6, end_seconds=7
            ),
            "measured work",
        ),
        (
            lambda o: o["instrumentation_intervals"][1].update(
                start_seconds=1.5, end_seconds=2
            ),
            "overlap",
        ),
        (
            lambda o: o["instrumentation_intervals"][0].update(seconds=2),
            "instrumentation.seconds",
        ),
        (
            lambda o: o["timeline"]["segments"][0].update(seconds=5),
            "differs from recorded",
        ),
    ],
)
def test_overlap_and_inconsistent_costs_are_rejected(mutation, match):
    observation, timing = _fixture()
    mutation(observation)
    with pytest.raises(ValueError, match=match):
        validate_native_timeline(observation, timing)


@pytest.mark.parametrize(
    "field",
    [
        "total_wall_seconds",
        "setup_seconds",
        "trainer_wall_seconds",
        "training_loop_seconds",
        "finalization_seconds",
        "optimizer_update_sample_seconds",
        "warmup_training_seconds",
        "trainer_other_seconds",
        "instrumentation_seconds",
        "batch_instrumentation_seconds",
        "event_instrumentation_seconds",
    ],
)
def test_timing_summary_must_reconcile_with_partition(field):
    observation, timing = _fixture()
    timing[field] += 0.5
    with pytest.raises(ValueError, match="native (event|timeline)"):
        validate_native_timeline(observation, timing)


def test_gaps_split_at_trainer_boundaries_even_without_adjacent_work():
    observation, _ = _fixture()
    observation["phase_events"] = []
    observation["updates"] = [observation["updates"][0]]
    timeline = _build(observation)
    assert [
        (row["kind"], row["start_seconds"], row["end_seconds"])
        for row in timeline["segments"]
    ] == [
        ("setup", 0, 5),
        ("trainer_other", 5, 6),
        ("optimizer_update", 6, 10),
        ("trainer_other", 10, 23),
        ("finalization", 23, 30),
    ]


def test_zero_duration_callback_at_work_boundary_needs_no_artificial_gap():
    observation, _ = _fixture()
    observation["instrumentation_intervals"] = [
        {
            "kind": "loss_collection",
            "start_seconds": 13,
            "end_seconds": 13,
            "seconds": 0,
        }
    ]
    timeline = _build(observation)
    assert all(row["wall_seconds"] > 0 for row in timeline["segments"])
    assert timeline["totals"]["instrumentation_seconds"] == pytest.approx(2.2)


def test_setup_event_is_removed_from_setup_gap_without_changing_setup_boundary():
    observation, timing = _fixture()
    for row in observation["phase_events"]:
        row["event_id"] += 1
        if row["parent_event_id"] is not None:
            row["parent_event_id"] += 1
    observation["phase_events"].insert(
        0, event(start=2.5, seconds=1, step=0, inside=False)
    )
    observation["updates"][1]["preceding_event_ids"] = [2, 3]
    observation["event_profile"] = summarize_phase_events(observation["phase_events"])
    observation["timeline"] = _build(observation)
    timing["eventcounts"]["evaluation"] += 1
    timing["evaluation_seconds"] += 1
    validate_native_timeline(observation, timing)
    assert timing["setup_seconds"] == 5
    assert observation["timeline"]["totals"]["by_kind"]["setup"]["wall_seconds"] == 4


def test_builder_does_not_mutate_its_input_ledgers():
    observation, _ = _fixture()
    before = deepcopy(observation)
    _build(observation)
    assert observation == before
