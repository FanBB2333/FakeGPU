from __future__ import annotations

from copy import deepcopy

import pytest

from fakegpu._finetune_cache import capture_cache_regime
from fakegpu._finetune_events import SCHEMA_VERSION as EVENT_SCHEMA
from fakegpu._finetune_events import summarize_phase_events
from fakegpu._finetune_pipeline_plan import (
    describe_pipeline_components,
    pipeline_plan_from_run,
    validate_pipeline_plan,
)
from fakegpu._finetune_timeline import SCHEMA_VERSION as TIMELINE_SCHEMA
from fakegpu._finetune_timeline import build_native_timeline
from fakegpu.finetune_estimator import FinetuneEstimateError
from fakegpu.finetune_pipeline_estimator import (
    compare_finetuning_pipeline,
    estimate_finetuning_pipeline,
)
from test_finetune_epoch_windows import _epoch_run
from test_finetune_events import event
from test_finetune_run_estimator import native_run as native_run
from test_finetune_window_estimator import _refresh


def _pipeline_run(base, number, *, epochs=24, factor=1.0):
    run = _epoch_run(base, number, epochs=epochs, factor=factor)
    run["training_config"].update(
        event_observation_version=EVENT_SCHEMA,
        timeline_observation_version=TIMELINE_SCHEMA,
    )
    run["native_config"].update(eval_steps=2, save_steps=4)
    run["schedule"] = {
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "logging_strategy": "no",
        "requested_logging_steps": 500,
        "resolved_logging_steps": 500,
        "logging_first_step": False,
        "save_total_limit": None,
    }
    _refresh(run)
    run["cache_regime"] = capture_cache_regime()
    events, metadata = [], []
    start = factor * 2
    cursor = start + factor * 0.2
    metadata.append(
        {
            "kind": "trainer_inspection",
            "start_seconds": factor * 0.2,
            "end_seconds": factor * 0.3,
            "seconds": factor * 0.1,
        }
    )

    def add_event(step, operation, *, inside=True):
        nonlocal cursor
        root = event(
            len(events) + 1,
            operation=operation,
            start=cursor,
            seconds=factor * 0.4,
            step=step,
            inside=inside,
        )
        root["instrumentation_seconds"] = factor * 0.01
        root["seconds"] -= root["instrumentation_seconds"]
        events.append(root)
        if operation == "_save_checkpoint":
            events.append(
                event(
                    len(events) + 1,
                    operation="save_model",
                    start=cursor + factor * 0.1,
                    seconds=factor * 0.2,
                    step=step,
                    parent=root["event_id"],
                )
            )
        cursor = root["end_seconds"] + factor * 0.04

    add_event(0, "evaluate")
    for update in run["observation"]["updates"]:
        step = update["optimizer_step"]
        update["preceding_event_ids"] = [
            entry["event_id"]
            for entry in events
            if entry["parent_event_id"] is None
            and entry["inside_train"]
            and entry["optimizer_step"] == step - 1
        ]
        update["batch_instrumentation_seconds"] = factor * 0.02
        update["wall_seconds"] = (
            update["seconds"] + update["batch_instrumentation_seconds"]
        )
        update["start_seconds"] = cursor
        update["end_seconds"] = cursor + update["wall_seconds"]
        cursor = update["end_seconds"]
        metadata.append(
            {
                "kind": "loss_collection",
                "start_seconds": cursor,
                "end_seconds": cursor + factor * 0.01,
                "seconds": factor * 0.01,
            }
        )
        cursor += factor * 0.03
        if step % 2 == 0:
            add_event(step, "evaluate")
        if step % 4 == 0:
            add_event(step, "_save_checkpoint")
    trainer_end = cursor + factor * 0.2
    metadata.append(
        {
            "kind": "train_end_parameter_check",
            "start_seconds": cursor,
            "end_seconds": cursor + factor * 0.05,
            "seconds": factor * 0.05,
        }
    )
    cursor = trainer_end + factor * 0.2
    add_event(len(run["observation"]["updates"]), "save_model", inside=False)
    end = cursor + factor * 0.3
    observation = run["observation"]
    observation["observed_model"]["training_arguments"] = {
        "logging_strategy": "no",
        "logging_steps": 500,
        "logging_first_step": False,
        "save_total_limit": None,
        "torch_empty_cache_steps": None,
    }
    observation["memory_allocator"] = {
        "schema_version": "fakegpu.finetune_allocator.v1",
        "backend": "native",
        "per_process_memory_fraction": 1.0,
        "environment": {
            "PYTORCH_ALLOC_CONF": None,
            "PYTORCH_CUDA_ALLOC_CONF": None,
            "PYTORCH_NO_CUDA_MEMORY_CACHING": None,
        },
    }
    observation["phase_events"] = events
    observation["event_profile"] = summarize_phase_events(events)
    observation["instrumentation_intervals"] = metadata
    timeline = build_native_timeline(
        observation["updates"],
        events,
        metadata,
        trainer_start_seconds=start,
        trainer_end_seconds=trainer_end,
        run_end_seconds=end,
    )
    observation["timeline"] = timeline
    totals = timeline["totals"]["by_kind"]
    profile = observation["event_profile"]
    inside_events = sum(
        entry["wall_seconds"]
        for entry in events
        if entry["parent_event_id"] is None and entry["inside_train"]
    )
    run["timing"].update(
        total_wall_seconds=end,
        setup_seconds=start,
        trainer_wall_seconds=trainer_end - start,
        training_loop_seconds=trainer_end - start - inside_events,
        finalization_seconds=end - trainer_end,
        trainer_other_seconds=totals["trainer_other"]["wall_seconds"]
        + totals["optimizer_update"]["instrumentation_seconds"],
        batch_instrumentation_seconds=totals["optimizer_update"][
            "instrumentation_seconds"
        ],
        event_instrumentation_seconds=profile["instrumentation_seconds"],
        instrumentation_seconds=timeline["totals"]["instrumentation_seconds"],
        eventcounts={
            kind: group["root_events"] for kind, group in profile["kinds"].items()
        },
        during_training_eventcounts={
            kind: group["during_training_root_events"]
            for kind, group in profile["kinds"].items()
        },
    )
    for kind, group in profile["kinds"].items():
        run["timing"][f"{kind}_seconds"] = group["wall_seconds"]
        run["timing"][f"during_training_{kind}_seconds"] = group[
            "during_training_wall_seconds"
        ]
    return run


@pytest.fixture
def runs(native_run):
    return [
        _pipeline_run(native_run, index, factor=factor, epochs=epochs)
        for index, factor, epochs in ((1, 1, 24), (2, 3, 28), (3, 2.2, 32))
    ]


def test_full_pipeline_includes_every_stage_and_counts_nested_events_once(runs):
    before = deepcopy(runs)
    result = compare_finetuning_pipeline(runs[:2], runs[2])
    assert runs == before
    assert result["coverage"]["complete"]
    assert result["coverage"]["excluded_components"] == 0
    assert set(result["timing"]["by_kind"]) == {
        "setup",
        "trainer_other",
        "finalization",
        "optimizer_update",
        "evaluation",
        "checkpoint",
    }
    for field in ("wall_seconds", "instrumentation_seconds", "seconds"):
        comparison = result["timing"]["comparison"][field]
        assert comparison["actual"] == pytest.approx(
            runs[2]["observation"]["timeline"]["totals"][field]
        )
        assert comparison["predicted"] == pytest.approx(comparison["actual"] / 2.2 * 2)
    assert (
        sum(
            component["kind"] in {"evaluation", "checkpoint"}
            for component in result["target_components"]
        )
        == runs[2]["observation"]["event_profile"]["root_events"]
    )
    windows = [
        group
        for group in result["component_groups"]
        if group["kind"] == "optimizer_update"
    ]
    assert {
        group["descriptor"]["stage"]["warmup_index"]
        for group in windows
        if group["descriptor"]["stage"]["warmup_index"]
    } == {1, 2, 3, 4}
    assert any(
        group["descriptor"]["stage"]["first_descriptor_occurrence"]
        and group["descriptor"]["stage"]["warmup_index"] is None
        for group in windows
    )
    assert any(group["minimum_samples_per_run"] == 2 for group in windows)


def test_future_estimate_has_no_target_timings_and_matches_replay(runs):
    plan = pipeline_plan_from_run(runs[2])
    plan["source"] = "cpu_native_schedule"
    plan["projection"] = {
        "epochs": 32,
        "reference_uuid": runs[0]["run_identity"]["uuid"],
    }
    result = estimate_finetuning_pipeline(runs[:2], plan)
    replay = compare_finetuning_pipeline(runs[:2], runs[2])
    assert result["scope"] == "planned_native_execution_span"
    assert "comparison" not in result["timing"]
    assert "target_run" not in result
    for field in ("wall_seconds", "instrumentation_seconds", "seconds"):
        assert result["timing"][field] == replay["timing"][field]


def test_target_timings_are_only_used_after_predictions(runs, native_run):
    original = compare_finetuning_pipeline(runs[:2], runs[2])
    changed = compare_finetuning_pipeline(
        runs[:2], _pipeline_run(native_run, 4, epochs=32, factor=17)
    )
    for field in ("wall_seconds", "instrumentation_seconds", "seconds"):
        assert changed["timing"][field] == original["timing"][field]
        assert (
            changed["timing"]["comparison"][field]["actual"]
            != original["timing"]["comparison"][field]["actual"]
        )


def test_process_means_receive_equal_weight_with_unequal_lengths(runs):
    result = compare_finetuning_pipeline(runs[:2], runs[2])
    groups = [
        group
        for group in result["component_groups"]
        if group["minimum_samples_per_run"] == 2
    ]
    assert any(
        group["calibration_runs"][0]["samples"]
        != group["calibration_runs"][1]["samples"]
        for group in groups
    )
    for group in groups:
        means = [
            process["mean"]["wall_seconds"] for process in group["calibration_runs"]
        ]
        assert group["per_component"]["wall_seconds"]["expected"] == pytest.approx(
            sum(means) / 2
        )


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda plan: plan["signature"].update(optimizer="different"), "signature"),
        (lambda plan: plan["hardware"].update(cuda_version="other"), "hardware"),
        (
            lambda plan: plan["memory_allocator"].update(backend="cudaMallocAsync"),
            "memory_allocator",
        ),
        (
            lambda plan: plan["cache_regime"].update(mode="empty_kernel_cache"),
            "cache_regime",
        ),
        (
            lambda plan: plan["setup_workload"].update(dataset_num_samples=6),
            "setup_workload",
        ),
    ],
)
def test_execution_data_hardware_and_cache_cohorts_must_match(runs, mutation, match):
    plan = pipeline_plan_from_run(runs[2])
    mutation(plan)
    with pytest.raises(FinetuneEstimateError, match=match):
        estimate_finetuning_pipeline(runs[:2], plan)


def test_missing_tail_coverage_is_not_replaced_by_a_fixed_gap(runs, native_run):
    target = _pipeline_run(native_run, 4, epochs=33, factor=2)
    with pytest.raises(
        FinetuneEstimateError, match="insufficient independent pipeline coverage"
    ):
        compare_finetuning_pipeline(runs[:2], target)


def test_repeated_groups_require_two_samples_in_each_process(runs, native_run):
    short = _pipeline_run(native_run, 4, epochs=8, factor=2)
    with pytest.raises(FinetuneEstimateError, match="requires 2"):
        estimate_finetuning_pipeline([runs[0], short], pipeline_plan_from_run(runs[2]))


def test_independence_and_minimum_processes_are_required(runs):
    plan = pipeline_plan_from_run(runs[2])
    with pytest.raises(FinetuneEstimateError, match="at least two"):
        estimate_finetuning_pipeline(runs[:1], plan)
    with pytest.raises(FinetuneEstimateError, match="distinct run UUIDs"):
        estimate_finetuning_pipeline([runs[0], runs[0]], plan)
    copied = deepcopy(runs[0])
    copied["run_identity"] = deepcopy(runs[1]["run_identity"])
    with pytest.raises(FinetuneEstimateError, match="duplicate timing evidence"):
        estimate_finetuning_pipeline([runs[0], copied], plan)
    with pytest.raises(FinetuneEstimateError, match="distinct run UUIDs"):
        compare_finetuning_pipeline(runs[:2], runs[0])


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda plan: plan.update(timing={}), "timing fields"),
        (lambda plan: plan["updates"][0].update(seconds=1), "timing fields"),
        (
            lambda plan: plan["updates"][0]["batches"][0].pop("epoch_microstep"),
            "epoch_microstep",
        ),
        (
            lambda plan: plan["updates"][0].update(preceding_event_ids=[]),
            "preceding events",
        ),
        (lambda plan: plan["operations"].pop(0), "placement"),
        (
            lambda plan: plan["root_events"][0]["descriptor"].update(phase="repeated"),
            "phase",
        ),
        (lambda plan: plan["workload"].update(samples=1), "workload.samples"),
        (lambda plan: plan["updates"][0].update(warmup=False), "warmup"),
    ],
)
def test_timing_free_plan_rejects_inconsistent_trace(runs, mutate, match):
    plan = pipeline_plan_from_run(runs[2])
    mutate(plan)
    with pytest.raises(FinetuneEstimateError, match=match):
        validate_pipeline_plan(plan)


def test_component_keys_ignore_absolute_epoch_and_step_but_keep_epoch_position(runs):
    components = describe_pipeline_components(pipeline_plan_from_run(runs[2]))
    repeated = [
        component
        for component in components
        if component["kind"] == "optimizer_update"
        and component["minimum_samples_per_run"] == 2
    ]
    assert len({component["group_id"] for component in repeated}) < len(repeated)
    windows = [component["descriptor"]["window"] for component in repeated]
    assert {
        tuple(window["epoch_context"]["batch_microsteps"]) for window in windows
    } == {(1, 2), (3, 4), (5,)}
    assert all(
        "epoch_index" not in batch for window in windows for batch in window["batches"]
    )


def test_invalid_or_missing_timeline_cannot_downgrade(runs):
    bad = deepcopy(runs[0])
    bad["observation"].pop("timeline")
    with pytest.raises(FinetuneEstimateError, match="timeline"):
        pipeline_plan_from_run(bad)


def test_zero_final_gap_remains_a_calibrated_component(runs):
    for run in runs:
        observation = run["observation"]
        old = observation["timeline"]
        end = observation["phase_events"][-1]["end_seconds"]
        observation["timeline"] = build_native_timeline(
            observation["updates"],
            observation["phase_events"],
            observation["instrumentation_intervals"],
            trainer_start_seconds=old["trainer"]["start_seconds"],
            trainer_end_seconds=old["trainer"]["end_seconds"],
            run_end_seconds=end,
        )
        run["timing"].update(
            total_wall_seconds=end,
            finalization_seconds=end - old["trainer"]["end_seconds"],
        )
    result = compare_finetuning_pipeline(runs[:2], runs[2])
    last = result["target_components"][-1]
    assert last["kind"] == "finalization"
    assert last["reference"]["right"] == {"kind": "native_end"}
    assert last["observed"]["wall_seconds"] == 0
    assert last["predicted"]["wall_seconds"] == {"min": 0, "expected": 0, "max": 0}
    group = next(
        group
        for group in result["component_groups"]
        if group["group_id"] == last["group_id"]
    )
    assert group["comparison"]["wall_seconds"]["absolute_percentage_error"] is None


def test_same_step_call_order_changes_gap_and_subsequent_window_coverage(runs):
    plan = pipeline_plan_from_run(runs[2])
    indices = [
        index
        for index, root in enumerate(plan["root_events"])
        if root["optimizer_step"] == 4
    ]
    first, second = indices
    roots = plan["root_events"]
    event_ids = [roots[index]["event_id"] for index in indices]
    roots[first], roots[second] = roots[second], roots[first]
    op_indices = [
        index
        for index, operation in enumerate(plan["operations"])
        if operation.get("event_id") in event_ids
    ]
    first, second = op_indices
    operations = plan["operations"]
    operations[first], operations[second] = operations[second], operations[first]
    plan["updates"][4]["preceding_event_ids"].reverse()
    validate_pipeline_plan(plan)
    with pytest.raises(
        FinetuneEstimateError, match="insufficient independent pipeline coverage"
    ):
        estimate_finetuning_pipeline(runs[:2], plan)


@pytest.mark.parametrize("field", ["optimizer_step", "event_id"])
def test_boolean_operation_references_are_not_integer_ids(runs, field):
    plan = pipeline_plan_from_run(runs[2])
    next(operation for operation in plan["operations"] if field in operation)[field] = (
        True
    )
    with pytest.raises(FinetuneEstimateError, match="integer"):
        validate_pipeline_plan(plan)


def test_zero_duration_outside_event_at_trainer_start_keeps_its_side(runs):
    run = runs[0]
    observation = run["observation"]
    previous = observation["timeline"]
    start = previous["trainer"]["start_seconds"]
    observation["phase_events"][0].update(
        inside_train=False,
        start_seconds=start,
        end_seconds=start,
        wall_seconds=0.0,
        instrumentation_seconds=0.0,
        seconds=0.0,
    )
    observation["updates"][0]["preceding_event_ids"] = []
    profile = summarize_phase_events(observation["phase_events"])
    observation["event_profile"] = profile
    timeline = build_native_timeline(
        observation["updates"],
        observation["phase_events"],
        observation["instrumentation_intervals"],
        trainer_start_seconds=start,
        trainer_end_seconds=previous["trainer"]["end_seconds"],
        run_end_seconds=previous["span"]["end_seconds"],
    )
    observation["timeline"] = timeline
    totals = timeline["totals"]["by_kind"]
    timing = run["timing"]
    timing["event_instrumentation_seconds"] = profile["instrumentation_seconds"]
    timing["instrumentation_seconds"] = timeline["totals"]["instrumentation_seconds"]
    for kind, group in profile["kinds"].items():
        timing[f"{kind}_seconds"] = group["wall_seconds"]
        timing[f"during_training_{kind}_seconds"] = group[
            "during_training_wall_seconds"
        ]
        timing["eventcounts"][kind] = group["root_events"]
        timing["during_training_eventcounts"][kind] = group[
            "during_training_root_events"
        ]
    timing["training_loop_seconds"] = timing["trainer_wall_seconds"] - sum(
        group["during_training_wall_seconds"] for group in profile["kinds"].values()
    )
    timing["trainer_other_seconds"] = (
        totals["trainer_other"]["wall_seconds"]
        + totals["optimizer_update"]["instrumentation_seconds"]
    )
    plan = pipeline_plan_from_run(run)
    assert plan["operations"][:2] == [
        {"kind": "evaluation", "event_id": 1},
        {"kind": "trainer_start"},
    ]


def _set_execution_policy(run, **changes):
    run["schedule"].update(changes)
    arguments = run["observation"]["observed_model"]["training_arguments"]
    for field, value in changes.items():
        if field != "resolved_logging_steps":
            arguments[
                "logging_steps" if field == "requested_logging_steps" else field
            ] = value


@pytest.mark.parametrize(
    "field,value",
    [
        ("save_total_limit", 100),
        ("logging_first_step", True),
    ],
)
def test_policy_omitted_from_old_execution_signature_is_still_matched(
    runs, field, value
):
    plan = pipeline_plan_from_run(runs[2])
    plan["execution_policy"][field] = value
    with pytest.raises(FinetuneEstimateError, match="matching execution_policy"):
        estimate_finetuning_pipeline(runs[:2], plan)


def test_periodic_logging_cannot_be_hidden_in_ordinary_gap_groups(runs):
    _set_execution_policy(
        runs[0],
        logging_strategy="steps",
        requested_logging_steps=80,
        resolved_logging_steps=80,
    )
    with pytest.raises(FinetuneEstimateError, match="periodic logging"):
        pipeline_plan_from_run(runs[0])
    plan = pipeline_plan_from_run(runs[2])
    plan["execution_policy"].update(
        logging_strategy="steps", requested_logging_steps=80, resolved_logging_steps=80
    )
    with pytest.raises(FinetuneEstimateError, match="periodic logging"):
        estimate_finetuning_pipeline([runs[1], runs[2]], plan)


def test_logging_every_update_is_calibrated_without_mixing_no_logging(runs):
    original = estimate_finetuning_pipeline(runs[:2], pipeline_plan_from_run(runs[2]))
    for run in runs:
        _set_execution_policy(
            run,
            logging_strategy="steps",
            requested_logging_steps=1,
            resolved_logging_steps=1,
        )
    result = compare_finetuning_pipeline(runs[:2], runs[2])
    assert result["timing"]["wall_seconds"] == original["timing"]["wall_seconds"]
    plan = pipeline_plan_from_run(runs[2])
    plan["execution_policy"]["logging_strategy"] = "no"
    with pytest.raises(FinetuneEstimateError, match="matching execution_policy"):
        estimate_finetuning_pipeline(runs[:2], plan)


def test_source_and_target_checkpoint_rotation_are_both_rejected(runs):
    _set_execution_policy(runs[0], save_total_limit=2)
    with pytest.raises(FinetuneEstimateError, match="rotation"):
        pipeline_plan_from_run(runs[0])
    # 24 checkpoints in the target would exceed the source's 18-checkpoint limit.
    plan = pipeline_plan_from_run(runs[2])
    plan["execution_policy"]["save_total_limit"] = 18
    with pytest.raises(FinetuneEstimateError, match="rotation"):
        validate_pipeline_plan(plan)


def test_logging_ratio_resolution_is_checked_against_workload_and_actual_state(runs):
    _set_execution_policy(
        runs[0],
        logging_strategy="steps",
        requested_logging_steps=0.01,
        resolved_logging_steps=1,
    )
    plan = pipeline_plan_from_run(runs[0])
    assert plan["execution_policy"]["resolved_logging_steps"] == 1
    runs[0]["schedule"]["resolved_logging_steps"] = 2
    with pytest.raises(FinetuneEstimateError, match="resolved logging steps"):
        pipeline_plan_from_run(runs[0])


def test_actual_arguments_and_schedule_cannot_disagree_on_execution_policy(runs):
    runs[0]["schedule"]["save_total_limit"] = 100
    with pytest.raises(FinetuneEstimateError, match="actual TrainingArguments"):
        pipeline_plan_from_run(runs[0])


@pytest.mark.parametrize("value", [1, 10, "missing"])
def test_periodic_cache_clearing_requires_its_own_window_context(runs, value):
    arguments = runs[0]["observation"]["observed_model"]["training_arguments"]
    if value == "missing":
        arguments.pop("torch_empty_cache_steps")
    else:
        arguments["torch_empty_cache_steps"] = value
    with pytest.raises(FinetuneEstimateError, match="cache clearing"):
        pipeline_plan_from_run(runs[0])


@pytest.mark.parametrize("field", ["peak_allocated_bytes", "peak_reserved_bytes"])
def test_future_plan_does_not_accept_target_memory_measurements(runs, field):
    plan = pipeline_plan_from_run(runs[2])
    plan[field] = 1
    with pytest.raises(FinetuneEstimateError, match="timing fields"):
        validate_pipeline_plan(plan)


@pytest.mark.parametrize(
    "field,value",
    [
        ("overwrite_cache", True),
        ("tokenized_path", "/different/tokenized"),
        ("dataset_prepared_path", "/different/prepared"),
        ("cache_dir", "/different/cache"),
    ],
)
def test_setup_preprocessing_options_cannot_hide_behind_legacy_signature(
    runs, field, value
):
    plan = pipeline_plan_from_run(runs[2])
    signature = deepcopy(runs[1]["observation"]["signature"])
    runs[1]["native_config"][field] = value
    changed = pipeline_plan_from_run(runs[1])
    assert changed["signature"] == signature
    assert changed["setup_workload"]["preprocessing_policy"][field] == value
    with pytest.raises(FinetuneEstimateError, match="matching setup_workload"):
        estimate_finetuning_pipeline(runs[:2], plan)


def test_setup_preprocessing_missing_field_is_not_a_default(runs):
    plan = pipeline_plan_from_run(runs[2])
    assert plan["setup_workload"]["preprocessing_policy"] == {
        "overwrite_cache": None,
        "tokenized_path": None,
        "dataset_prepared_path": None,
        "cache_dir": None,
    }
    plan["setup_workload"]["preprocessing_policy"].pop("overwrite_cache")
    with pytest.raises(FinetuneEstimateError, match="complete preprocessing_policy"):
        validate_pipeline_plan(plan)
