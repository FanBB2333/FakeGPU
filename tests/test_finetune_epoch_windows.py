from __future__ import annotations

from copy import deepcopy
import json

import pytest

from fakegpu._finetune_events import SCHEMA_VERSION as EVENT_SCHEMA
from fakegpu.finetune_data import profile_tokenized_records
from fakegpu.finetune_window_estimator import estimate_finetuning_windows
from test_finetune_event_estimator import _refresh_events
from test_finetune_events import event
from test_finetune_run_estimator import native_run as native_run
from test_finetune_window_estimator import _batch, _refresh, _run


def _epoch_run(base, number, *, epochs=24, factor=1.0):
    run = _run(base, number, epochs=epochs)
    records = [
        {"input_ids": list(range(length)), "labels": list(range(length))}
        for length in (4, 2, 4, 2, 2)
    ]
    run["dataset"] = profile_tokenized_records(
        records, sequence_length=4, source_framework="llamafactory"
    )
    run["training_config"].update(
        num_samples=5, data_signature=run["dataset"]["provenance"]["content_sha256"]
    )
    updates = []
    for epoch in range(1, epochs + 1):
        for first_microstep, lengths, cost in [
            (1, (4, 2), 1),
            (3, (4, 2), 2),
            (5, (2,), 0.5),
        ]:
            batches = [_batch(length) for length in lengths]
            for offset, batch in enumerate(batches):
                batch.update(
                    epoch_index=epoch, epoch_microstep=first_microstep + offset
                )
            updates.append(
                {
                    "seconds": factor * cost,
                    "warmup": len(updates) < 4,
                    "batches": batches,
                }
            )
    run["observation"]["updates"] = updates
    return _refresh(run)


def _add_events(run):
    run["training_config"]["event_observation_version"] = EVENT_SCHEMA
    run["native_config"].update(eval_steps=2, save_steps=4)
    run["schedule"] = {"eval_strategy": "steps", "save_strategy": "steps"}
    _refresh(run)
    events = []
    start = 1.0
    steps = run["workload"]["optimizer_steps"]
    for step in range(1, steps + 1):
        operations = []
        if step % 2 == 0:
            operations.append("evaluate")
        if step % 4 == 0:
            operations.append("_save_checkpoint")
        for operation in operations:
            entry = event(
                len(events) + 1,
                operation=operation,
                start=start,
                seconds=0.1,
                step=step,
            )
            events.append(entry)
            if operation == "_save_checkpoint":
                events.append(
                    event(
                        len(events) + 1,
                        operation="save_model",
                        start=start + 0.02,
                        seconds=0.05,
                        step=step,
                        parent=entry["event_id"],
                    )
                )
            start += 0.2
    trainer_wall = run["timing"]["training_loop_seconds"] + sum(
        entry["wall_seconds"] for entry in events if entry["parent_event_id"] is None
    )
    events.append(
        event(
            len(events) + 1,
            operation="save_model",
            start=1 + trainer_wall + 0.1,
            seconds=0.1,
            step=steps,
            inside=False,
        )
    )
    return _refresh_events(run, events, trainer_wall)


@pytest.fixture
def runs(native_run):
    return [
        _epoch_run(native_run, i, epochs=24 if i < 3 else 30, factor=factor)
        for i, factor in enumerate((1, 3, 2.2), 1)
    ]


def test_epoch_mode_separates_identical_windows_at_different_recorded_positions(runs):
    before = deepcopy(runs)
    result = estimate_finetuning_windows(runs[:2], runs[2], epoch_context=True)
    assert result["scope"] == "recorded_epoch_conditioned_optimizer_window_replay"
    assert (
        result["sampling_policy"]["epoch_context"]
        == "recorded_epoch_microsteps_per_batch"
    )
    assert result["sampling_policy"]["exclude_first_context_window_occurrence"]
    assert "exclude_first_window_occurrence" not in result["sampling_policy"]
    groups = {
        tuple(group["epoch_context"]["batch_microsteps"]): group
        for group in result["window_groups"]
    }
    assert set(groups) == {(1, 2), (3, 4), (5,)}
    assert groups[(1, 2)]["batches"] == groups[(3, 4)]["batches"]
    assert groups[(1, 2)]["window_id"] != groups[(3, 4)]["window_id"]
    assert groups[(1, 2)]["seconds_per_window"]["expected"] == 2
    assert groups[(3, 4)]["seconds_per_window"]["expected"] == 4
    assert all("context" not in group for group in groups.values())
    assert [
        row["epoch_context"]["batch_microsteps"] for row in result["target_updates"][:3]
    ] == [[1, 2], [3, 4], [5]]
    assert runs == before
    groups[(1, 2)]["epoch_context"]["batch_microsteps"].append(99)
    assert runs == before


def test_absolute_epoch_numbers_are_not_part_of_group_identity(runs):
    result = estimate_finetuning_windows(runs[:2], runs[2], epoch_context=True)
    updates = {row["optimizer_step"]: row for row in result["target_updates"]}
    assert updates[7]["epoch_index"] != updates[76]["epoch_index"]
    assert updates[7]["window_id"] == updates[76]["window_id"]
    assert updates[76]["epoch_context"] == {"batch_microsteps": [1, 2]}


def test_event_and_epoch_contexts_jointly_separate_otherwise_matching_windows(runs):
    runs = [_add_events(run) for run in runs]
    result = estimate_finetuning_windows(
        runs[:2], runs[2], event_context=True, epoch_context=True
    )
    assert result["scope"] == "recorded_event_epoch_conditioned_optimizer_window_replay"
    groups = result["window_groups"]
    matched = [
        (left, right)
        for left in groups
        for right in groups
        if left["context"] == right["context"]
        and left["batches"] == right["batches"]
        and left["epoch_context"] != right["epoch_context"]
    ]
    assert matched
    assert all(left["window_id"] != right["window_id"] for left, right in matched)
    assert result["coverage"]["target_window_groups"] == 12


def test_epoch_context_does_not_open_the_periodic_event_gate(runs):
    runs = [_add_events(run) for run in runs]
    with pytest.raises(ValueError, match="does not support periodic"):
        estimate_finetuning_windows(runs[:2], runs[2], epoch_context=True)


@pytest.mark.parametrize("events", [False, True])
def test_explicit_false_keeps_previous_modes_identical(runs, events):
    if events:
        runs = [_add_events(run) for run in runs]
    before = estimate_finetuning_windows(runs[:2], runs[2], event_context=events)
    after = estimate_finetuning_windows(
        runs[:2], runs[2], event_context=events, epoch_context=False
    )
    assert before == after
    assert all("epoch_context" not in group for group in after["window_groups"])


def test_missing_optional_epoch_metadata_is_rejected_only_when_requested(runs):
    for run in runs:
        for update in run["observation"]["updates"]:
            for batch in update["batches"]:
                del batch["epoch_index"]
                del batch["epoch_microstep"]
        _refresh(run)
    estimate_finetuning_windows(runs[:2], runs[2])
    with pytest.raises(ValueError, match="require recorded epoch metadata"):
        estimate_finetuning_windows(runs[:2], runs[2], epoch_context=True)


def test_inconsistent_epoch_positions_use_the_shared_batch_validation(runs):
    runs[2]["observation"]["updates"][1]["batches"][0]["epoch_microstep"] = 1
    with pytest.raises(ValueError, match="epoch microsteps must be consecutive"):
        estimate_finetuning_windows(runs[:2], runs[2], epoch_context=True)


def test_recorded_epoch_reset_inside_a_window_is_preserved(runs):
    for run in runs:
        absolute_microstep = 0
        for update in run["observation"]["updates"]:
            for batch in update["batches"]:
                # A different valid consumed trace places reset boundaries inside
                # some complete accumulation windows. Do not infer a sampler.
                batch["epoch_index"] = absolute_microstep // 3 + 1
                batch["epoch_microstep"] = absolute_microstep % 3 + 1
                absolute_microstep += 1
        _refresh(run)
    result = estimate_finetuning_windows(runs[:2], runs[2], epoch_context=True)
    descriptors = {
        tuple(group["epoch_context"]["batch_microsteps"])
        for group in result["window_groups"]
    }
    assert (3, 1) in descriptors
    assert (1, 2) in descriptors
    same_batches = [
        group
        for group in result["window_groups"]
        if group["epoch_context"]["batch_microsteps"] in ([3, 1], [1, 2])
        and len(group["batches"]) == 2
    ]
    assert len({group["window_id"] for group in same_batches}) == 2


def test_insufficient_position_coverage_has_no_batch_only_fallback(runs):
    run = runs[0]
    # Fully valid existing epoch metadata, but the calibration run never reaches
    # the target's later within-epoch positions before resetting.
    counter = 0
    for update in run["observation"]["updates"]:
        for batch in update["batches"]:
            batch["epoch_index"] = counter // 3 + 1
            batch["epoch_microstep"] = counter % 3 + 1
            counter += 1
    _refresh(run)
    with pytest.raises(ValueError, match="insufficient independent window coverage"):
        estimate_finetuning_windows(runs[:2], runs[2], epoch_context=True)


def test_target_times_never_enter_epoch_calibration(runs, native_run):
    before = estimate_finetuning_windows(runs[:2], runs[2], epoch_context=True)
    target = _epoch_run(native_run, 3, epochs=30, factor=22)
    after = estimate_finetuning_windows(runs[:2], target, epoch_context=True)
    assert (
        before["timing"]["steady_optimizer_window_seconds"]
        == after["timing"]["steady_optimizer_window_seconds"]
    )
    assert after["timing"]["comparison"]["actual"] == pytest.approx(
        before["timing"]["comparison"]["actual"] * 10
    )


@pytest.mark.parametrize("value", [1, "true", None])
def test_epoch_mode_requires_an_explicit_boolean(runs, value):
    with pytest.raises(ValueError, match="epoch_context must be boolean"):
        estimate_finetuning_windows(runs[:2], runs[2], epoch_context=value)


def test_cli_epoch_context_flag_can_be_used_without_events(runs, tmp_path):
    from fakegpu.finetune_cli import windows_main

    paths = []
    for index, run in enumerate(runs):
        path = tmp_path / f"run-{index}.json"
        path.write_text(json.dumps(run))
        paths.append(path)
    output = tmp_path / "result.json"
    assert (
        windows_main(
            [
                "--calibration-run",
                str(paths[0]),
                "--calibration-run",
                str(paths[1]),
                "--holdout-run",
                str(paths[2]),
                "--epoch-context",
                "--json",
                str(output),
            ]
        )
        == 0
    )
    assert (
        json.loads(output.read_text())["scope"]
        == "recorded_epoch_conditioned_optimizer_window_replay"
    )
