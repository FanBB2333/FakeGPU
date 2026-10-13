from __future__ import annotations

from copy import deepcopy
import json

import pytest

from fakegpu._finetune_events import SCHEMA_VERSION
from fakegpu.finetune_window_estimator import estimate_finetuning_windows
from test_finetune_event_estimator import _refresh_events
from test_finetune_events import event
from test_finetune_run_estimator import native_run as native_run
from test_finetune_window_estimator import _refresh, _run


def _context_run(base, number, *, steps=48, seconds=1.0, overlap="normal"):
    run = _run(base, number, seconds=seconds, epochs=steps // 2)
    run["training_config"]["event_observation_version"] = SCHEMA_VERSION
    run["native_config"].update(eval_steps=4, save_steps=6)
    run["schedule"] = {"eval_strategy": "steps", "save_strategy": "steps"}
    for update in run["observation"]["updates"]:
        update["warmup"] = update["optimizer_step"] <= 4
    _refresh(run)
    events = []
    start = 1.0
    for step in range(1, steps + 1):
        operations = []
        if step % 4 == 0:
            operations.append("evaluate")
        if step % 6 == 0:
            operations.append("_save_checkpoint")
        if len(operations) == 2:
            if overlap == "reverse":
                operations.reverse()
            elif overlap == "duplicate":
                operations.append("evaluate")
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
        _context_run(native_run, 1, seconds=1),
        _context_run(native_run, 2, seconds=3),
        _context_run(native_run, 3, steps=60, seconds=2.2),
    ]


def _target_group(result, step):
    update = next(
        update
        for update in result["target_updates"]
        if update["optimizer_step"] == step
    )
    return next(
        group
        for group in result["window_groups"]
        if group["window_id"] == update["window_id"]
    )


def test_context_mode_preserves_both_event_boundaries_and_their_distances(runs):
    before = deepcopy(runs)
    result = estimate_finetuning_windows(runs[:2], runs[2], event_context=True)
    assert result["scope"] == "recorded_event_conditioned_optimizer_window_replay"
    assert result["coverage"]["complete"]
    assert "exclude_first_window_occurrence" not in result["sampling_policy"]
    assert result["sampling_policy"]["exclude_first_context_window_occurrence"]
    at_boundary = _target_group(result, 25)["context"]
    assert at_boundary["evaluation"] == at_boundary["checkpoint"]
    assert [entry["operation"] for entry in at_boundary["evaluation"]["events"]] == [
        "evaluate",
        "_save_checkpoint",
    ]
    assert at_boundary["evaluation"]["optimizer_updates_since_boundary"] == 0
    assert all(
        entry["phase"] == "repeated" for entry in at_boundary["evaluation"]["events"]
    )
    later = _target_group(result, 26)["context"]
    assert (
        later["evaluation"]["optimizer_updates_since_boundary"]
        == later["checkpoint"]["optimizer_updates_since_boundary"]
        == 1
    )
    after_eval = _target_group(result, 29)["context"]
    assert [entry["operation"] for entry in after_eval["evaluation"]["events"]] == [
        "evaluate"
    ]
    assert after_eval["evaluation"]["optimizer_updates_since_boundary"] == 0
    assert after_eval["checkpoint"]["optimizer_updates_since_boundary"] == 4
    assert after_eval["checkpoint"]["events"] == at_boundary["checkpoint"]["events"]
    assert [
        update["immediately_after_event"]
        for update in result["target_updates"]
        if update["optimizer_step"] in [25, 26]
    ] == [True, False]
    assert runs == before
    _target_group(result, 29)["context"]["evaluation"]["events"][0]["workload"][
        "dataset"
    ]["content_sha256"] = "mutated"
    assert runs == before


def test_equal_recent_evaluation_does_not_forget_the_previous_checkpoint(runs):
    result = estimate_finetuning_windows(runs[:2], runs[2], event_context=True)
    first, second = _target_group(result, 29), _target_group(result, 33)
    assert first["batches"] == second["batches"]
    assert first["context"]["evaluation"] == second["context"]["evaluation"]
    assert first["context"]["checkpoint"]["optimizer_updates_since_boundary"] == 4
    assert second["context"]["checkpoint"]["optimizer_updates_since_boundary"] == 2
    assert first["window_id"] != second["window_id"]


def test_periodic_gate_remains_the_default(runs):
    with pytest.raises(ValueError, match="does not support periodic"):
        estimate_finetuning_windows(runs[:2], runs[2])
    with pytest.raises(ValueError, match="does not support periodic"):
        estimate_finetuning_windows(runs[:2], runs[2], event_context=False)


def test_event_mode_requires_real_protocol_even_with_no_periodic_schedule(native_run):
    legacy = [_run(native_run, i) for i in range(1, 4)]
    with pytest.raises(ValueError, match="event observation protocol"):
        estimate_finetuning_windows(legacy[:2], legacy[2], event_context=True)


@pytest.mark.parametrize("value", [1, "true", None])
def test_mode_must_be_explicit_boolean(runs, value):
    with pytest.raises(ValueError, match="event_context must be boolean"):
        estimate_finetuning_windows(runs[:2], runs[2], event_context=value)


@pytest.mark.parametrize("overlap", ["reverse", "duplicate"])
def test_changed_boundary_order_or_multiplicity_requires_its_own_coverage(
    runs, native_run, overlap
):
    target = _context_run(native_run, 3, steps=60, seconds=2.2, overlap=overlap)
    with pytest.raises(ValueError, match="insufficient independent window coverage"):
        estimate_finetuning_windows(runs[:2], target, event_context=True)


def test_repeated_calls_in_a_boundary_are_retained_in_order(native_run):
    values = [
        _context_run(native_run, i, seconds=i, overlap="duplicate") for i in range(1, 4)
    ]
    result = estimate_finetuning_windows(values[:2], values[2], event_context=True)
    context = _target_group(result, 25)["context"]
    assert context["evaluation"] == context["checkpoint"]
    assert [entry["operation"] for entry in context["evaluation"]["events"]] == [
        "evaluate",
        "_save_checkpoint",
        "evaluate",
    ]


def test_context_coverage_needs_two_samples_in_every_calibration_run(runs, native_run):
    runs[0] = _context_run(native_run, 1, steps=36)
    with pytest.raises(
        ValueError,
        match=r"insufficient independent window coverage:.*1 eligible samples",
    ):
        estimate_finetuning_windows(runs[:2], runs[2], event_context=True)


def test_target_update_seconds_do_not_enter_the_prediction(runs, native_run):
    before = estimate_finetuning_windows(runs[:2], runs[2], event_context=True)
    target = _context_run(native_run, 3, steps=60, seconds=22)
    after = estimate_finetuning_windows(runs[:2], target, event_context=True)
    assert (
        before["timing"]["steady_optimizer_window_seconds"]
        == after["timing"]["steady_optimizer_window_seconds"]
    )
    assert after["timing"]["comparison"]["actual"] == pytest.approx(
        before["timing"]["comparison"]["actual"] * 10
    )


def test_event_durations_are_not_part_of_the_window_context(runs):
    before = estimate_finetuning_windows(runs[:2], runs[2], event_context=True)
    for entry in runs[2]["observation"]["phase_events"]:
        entry["instrumentation_seconds"] = entry["wall_seconds"] / 2
        entry["seconds"] = entry["wall_seconds"] / 2
    _refresh_events(runs[2])
    after = estimate_finetuning_windows(runs[:2], runs[2], event_context=True)
    assert before["window_groups"] == after["window_groups"]
    assert before["timing"] == after["timing"]


def test_future_events_do_not_change_prior_window_groups(runs):
    before = estimate_finetuning_windows(runs[:2], runs[2], event_context=True)
    # The final event cannot have appeared in any consumed window's preceding IDs.
    workload = runs[2]["observation"]["phase_events"][-1]["workload"]
    workload["save_only_model"] = not workload["save_only_model"]
    _refresh_events(runs[2])
    after = estimate_finetuning_windows(runs[:2], runs[2], event_context=True)
    assert before["window_groups"] == after["window_groups"]
    assert before["target_updates"] == after["target_updates"]


def test_loss_only_and_successful_event_scope_is_shared(runs):
    runs[2]["observation"]["phase_events"][0]["workload"]["compute_metrics"] = True
    _refresh_events(runs[2])
    with pytest.raises(ValueError, match="loss-only evaluation"):
        estimate_finetuning_windows(runs[:2], runs[2], event_context=True)


def test_cli_opt_in_is_required_for_periodic_event_windows(runs, tmp_path):
    from fakegpu.finetune_cli import windows_main

    paths = []
    for index, run in enumerate(runs):
        path = tmp_path / f"run-{index}.json"
        path.write_text(json.dumps(run))
        paths.append(path)
    output = tmp_path / "result.json"
    arguments = [
        "--calibration-run",
        str(paths[0]),
        "--calibration-run",
        str(paths[1]),
        "--holdout-run",
        str(paths[2]),
        "--json",
        str(output),
    ]
    with pytest.raises(SystemExit) as error:
        windows_main(arguments)
    assert error.value.code == 2
    assert not output.exists()
    assert windows_main([*arguments, "--event-context"]) == 0
    assert (
        json.loads(output.read_text())["scope"]
        == "recorded_event_conditioned_optimizer_window_replay"
    )


def test_event_context_requires_matching_actual_hardware_metadata(runs):
    runs[2]["observation"]["hardware"]["device_uuid"] = "different-recorded-device"
    with pytest.raises(ValueError, match="matching recorded hardware"):
        estimate_finetuning_windows(runs[:2], runs[2], event_context=True)
