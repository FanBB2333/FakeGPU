from copy import deepcopy

import pytest

from fakegpu._finetune_model import native_config_signature, native_execution_signature
from fakegpu.finetune_estimator import finetune_workload_signature
from fakegpu.finetune_pipeline_estimator import (
    compare_finetuning_pipeline,
    estimate_finetuning_pipeline,
)
from fakegpu.finetune_pipeline_plan import _schedule_calls, plan_finetuning_pipeline
from test_finetune_pipeline_estimator import _pipeline_run
from test_finetune_run_estimator import native_run as native_run


def sequential_run(base, number, *, epochs=24, factor=1):
    run = _pipeline_run(base, number, epochs=epochs, factor=factor)
    run["native_config"]["disable_shuffling"] = True
    config = run["training_config"]
    config["software_versions"] = {"llamafactory": "0.9.5", "transformers": "5.6.0"}
    run["observation"]["software_versions"] = deepcopy(config["software_versions"])
    run["schedule"].update(
        eval_on_start=True,
        eval_delay=0,
        load_best_model_at_end=False,
        requested_eval_steps=2,
        requested_save_steps=4,
        resolved_eval_steps=2,
        resolved_save_steps=4,
    )
    refresh_signature(run)
    return run


def refresh_signature(run):
    config = run["training_config"]
    config["native_execution_signature"] = native_execution_signature(
        native_config_signature(run["native_config"]),
        run["observation"]["observed_model"],
    )
    run["observation"]["signature"] = finetune_workload_signature(
        run["model_config"], config, profile_id=run["profile_id"]
    )


def test_future_duration_budget_matches_independent_longer_run(native_run):
    calibrations = [
        sequential_run(native_run, 1),
        sequential_run(native_run, 2, factor=3),
    ]
    before = deepcopy(calibrations)
    plan = plan_finetuning_pipeline(calibrations[0], epochs=32)
    prediction = estimate_finetuning_pipeline(calibrations, plan)
    # The holdout is constructed only after the future estimate is available.
    target = sequential_run(native_run, 3, epochs=32, factor=2.2)
    comparison = compare_finetuning_pipeline(calibrations, target)
    for field in ("wall_seconds", "instrumentation_seconds", "seconds"):
        assert prediction["timing"][field] == comparison["timing"][field]
    assert calibrations == before
    assert plan["source"] == "cpu_native_schedule"
    assert plan["workload"] == {
        "optimizer_steps": 96,
        "microsteps": 160,
        "samples": 160,
    }
    assert plan["updates"][-1]["batches"][-1]["epoch_index"] == 32


@pytest.mark.parametrize("steps", [72, 73, 74, 77, 96])
def test_future_partial_epochs_and_forced_terminal_events(native_run, steps):
    plan = plan_finetuning_pipeline(sequential_run(native_run, 1), max_steps=steps)
    full, remainder = divmod(steps, 3)
    assert plan["workload"]["microsteps"] == full * 5 + remainder * 2
    assert plan["workload"]["samples"] == plan["workload"]["microsteps"]
    checkpoints = [
        row["optimizer_step"]
        for row in plan["root_events"]
        if row["descriptor"]["operation"] == "_save_checkpoint"
    ]
    assert checkpoints == sorted(set([*range(4, steps + 1, 4), steps]))
    evaluations = [
        row["optimizer_step"]
        for row in plan["root_events"]
        if row["descriptor"]["operation"] == "evaluate"
    ]
    assert evaluations == sorted(set([0, *range(2, steps + 1, 2), steps]))
    assert plan["root_events"][-1]["descriptor"]["phase"] == "final"
    assert plan["root_events"][-1]["optimizer_step"] == steps


def test_fractional_epochs_count_complete_optimizer_windows(native_run):
    plan = plan_finetuning_pipeline(sequential_run(native_run, 1), epochs=2.5)
    assert plan["workload"] == {"optimizer_steps": 8, "microsteps": 14, "samples": 14}
    assert plan["projection"]["epoch_step_rule"] == "trainer_epoch_windows"
    assert [batch["epoch_microstep"] for batch in plan["updates"][-1]["batches"]] == [
        3,
        4,
    ]


def axolotl_sequential_run(base, number, **kwargs):
    run = sequential_run(base, number, **kwargs)
    run["framework"] = "axolotl"
    config = run["training_config"]
    config["source_framework"] = "axolotl"
    run["dataset"]["source_framework"] = "axolotl"
    run["observation"]["native_framework"] = "axolotl"
    run["native_config"].pop("disable_shuffling")
    run["native_config"]["curriculum_sampling"] = True
    config["software_versions"] = {"axolotl": "0.18.0", "transformers": "5.14.1"}
    run["observation"]["software_versions"] = deepcopy(config["software_versions"])
    refresh_signature(run)
    return run


@pytest.mark.parametrize(("epochs", "steps"), [(2.5, 7), (32, 80)])
def test_axolotl_epochs_follow_its_max_steps_conversion(native_run, epochs, steps):
    # Axolotl 0.18.0 hands Trainer max_steps = ceil(rows * num_epochs /
    # batch_size) for unpacked data (utils/trainer.py
    # calculate_total_num_steps; core/builders/base.py). With 5 rows, micro
    # batch 1 and accumulation 2, Trainer's per-epoch windows would give 8 and 96.
    run = axolotl_sequential_run(native_run, 1)
    config = run["training_config"]
    assert (config["num_samples"], config["micro_batch_size"]) == (5, 1)
    assert config["gradient_accumulation_steps"] == 2
    plan = plan_finetuning_pipeline(run, epochs=epochs)
    assert plan["workload"]["optimizer_steps"] == steps
    assert plan["projection"]["epoch_step_rule"] == "axolotl_max_steps"
    assert plan["projection"]["updates_per_epoch"] == 3
    # Trainer still walks each epoch's windows, the short last one included.
    assert plan["updates"][2]["batches"][-1]["epoch_microstep"] == 5


def test_future_checkpoint_rotation_requires_separate_cost_coverage(native_run):
    run = sequential_run(native_run, 1)
    run["schedule"]["save_total_limit"] = 18
    run["observation"]["observed_model"]["training_arguments"]["save_total_limit"] = 18
    run["native_config"]["save_total_limit"] = 18
    refresh_signature(run)
    assert (
        plan_finetuning_pipeline(run, max_steps=72)["workload"]["optimizer_steps"] == 72
    )
    with pytest.raises(ValueError, match="rotation"):
        plan_finetuning_pipeline(run, max_steps=96)


def test_future_logging_ratio_cannot_silently_change_log_frequency(native_run):
    run = sequential_run(native_run, 1)
    run["native_config"]["logging_steps"] = 0.01
    run["schedule"].update(
        logging_strategy="steps",
        requested_logging_steps=0.01,
        resolved_logging_steps=1,
        logging_first_step=False,
    )
    run["observation"]["observed_model"]["training_arguments"].update(
        logging_strategy="steps", logging_steps=0.01, logging_first_step=False
    )
    refresh_signature(run)
    assert (
        plan_finetuning_pipeline(run, max_steps=96)["execution_policy"][
            "resolved_logging_steps"
        ]
        == 1
    )
    with pytest.raises(ValueError, match="logging"):
        plan_finetuning_pipeline(run, max_steps=200)


def test_disabled_logging_ratio_resolves_against_future_duration(native_run):
    run = sequential_run(native_run, 1)
    run["native_config"]["logging_steps"] = 0.01
    run["schedule"].update(
        logging_strategy="no",
        requested_logging_steps=0.01,
        resolved_logging_steps=1,
        logging_first_step=False,
    )
    run["observation"]["observed_model"]["training_arguments"].update(
        logging_strategy="no", logging_steps=0.01, logging_first_step=False
    )
    refresh_signature(run)
    plan = plan_finetuning_pipeline(run, max_steps=200)
    assert plan["execution_policy"]["resolved_logging_steps"] == 2


def test_mixed_schedule_processes_step_events_before_epoch_events():
    run = {
        "schedule": {
            "eval_on_start": False,
            "eval_delay": 0,
            "load_best_model_at_end": False,
            "eval_strategy": "epoch",
            "save_strategy": "steps",
            "requested_save_steps": 3,
        }
    }
    calls, resolved = _schedule_calls(run, steps=5, updates_per_epoch=3)
    assert calls == [
        (3, "_save_checkpoint"),
        (3, "evaluate"),
        (5, "_save_checkpoint"),
        (5, "evaluate"),
    ]
    assert resolved == {"save": 3}


@pytest.mark.parametrize("field", ["early_stopping_steps", "early_stopping_patience"])
def test_future_early_stopping_cannot_assume_the_requested_duration(native_run, field):
    run = sequential_run(native_run, 1)
    run["native_config"][field] = 3
    refresh_signature(run)
    with pytest.raises(ValueError, match="early stopping"):
        plan_finetuning_pipeline(run, max_steps=96)


@pytest.mark.parametrize("field", ["enable_torch_profiler", "profile_modules"])
def test_pipeline_profiler_cycles_need_explicit_phase_coverage(native_run, field):
    run = sequential_run(native_run, 1)
    run["observation"]["observed_model"]["training_arguments"][field] = True
    with pytest.raises(ValueError, match="profiler"):
        plan_finetuning_pipeline(run, max_steps=96)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"epochs": 0}, "epochs"),
        ({"epochs": float("nan")}, "epochs"),
        ({"max_steps": True}, "max_steps"),
        ({"epochs": 2, "max_steps": 5}, "exactly one"),
        ({}, "exactly one"),
        ({"epochs": 2, "num_samples": 100}, "dataset size"),
    ],
)
def test_ambiguous_or_invalid_workload_is_rejected(native_run, changes, match):
    with pytest.raises(ValueError, match=match):
        plan_finetuning_pipeline(sequential_run(native_run, 1), **changes)


@pytest.mark.parametrize(
    "change,match",
    [
        ("random", "disable_shuffling"),
        ("version", "validated framework"),
        ("delay", "eval_delay"),
        ("schedule", "recorded events"),
    ],
)
def test_unverified_execution_rules_do_not_silently_generate_a_plan(
    native_run, change, match
):
    run = sequential_run(native_run, 1)
    if change == "random":
        run["native_config"]["disable_shuffling"] = False
    elif change == "version":
        run["training_config"]["software_versions"]["transformers"] = "99.0.0"
        run["observation"]["software_versions"] = deepcopy(
            run["training_config"]["software_versions"]
        )
    elif change == "delay":
        run["schedule"]["eval_delay"] = 1
    else:
        run["schedule"]["requested_eval_steps"] = 3
    refresh_signature(run)
    with pytest.raises(ValueError, match=match):
        plan_finetuning_pipeline(run, epochs=32)
