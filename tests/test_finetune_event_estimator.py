from __future__ import annotations

from copy import deepcopy
import json

import pytest

from fakegpu._finetune_events import SCHEMA_VERSION, summarize_phase_events
from fakegpu.finetune_estimator import FinetuneEstimateError
from fakegpu.finetune_event_estimator import estimate_finetuning_events
from fakegpu.finetune_run_estimator import validate_native_run
from test_finetune_events import event
from test_finetune_run_estimator import native_run as native_run
from test_finetune_window_estimator import _refresh, _run


def _events_run(base, number, *, seconds=1.0, repetitions=2):
    run = _run(base, number, seconds=number / 10, epochs=5)
    run["training_config"]["event_observation_version"] = SCHEMA_VERSION
    run["native_config"].update(eval_steps=2, save_steps=2)
    run["schedule"] = {"eval_strategy": "steps", "save_strategy": "steps"}
    _refresh(run)
    events = []
    start = 1.0
    calls = [(0, "evaluate")]
    for index in range(repetitions + 1):
        calls.extend([(2 * index + 2, "evaluate"), (2 * index + 3, "_save_checkpoint")])
    for step, operation in calls:
        entry = event(
            len(events) + 1,
            operation=operation,
            start=start,
            seconds=seconds,
            step=step,
        )
        events.append(entry)
        if operation == "_save_checkpoint":
            events.append(
                event(
                    len(events) + 1,
                    operation="save_model",
                    start=start + seconds / 4,
                    seconds=seconds / 2,
                    parent=entry["event_id"],
                    step=step,
                )
            )
        start += seconds + 0.1
    training = run["timing"]["training_loop_seconds"]
    inside_wall = sum(
        entry["wall_seconds"] for entry in events if entry["parent_event_id"] is None
    )
    trainer_wall = inside_wall + training
    events.append(
        event(
            len(events) + 1,
            operation="save_model",
            start=1 + trainer_wall + 0.1,
            seconds=seconds,
            step=10,
            inside=False,
        )
    )
    return _refresh_events(run, events, trainer_wall)


def _refresh_events(run, events=None, trainer_wall=None):
    observation = run["observation"]
    if events is not None:
        observation["phase_events"] = events
    events = observation["phase_events"]
    profile = summarize_phase_events(events)
    observation["event_profile"] = profile
    for update in observation["updates"]:
        update["preceding_event_ids"] = [
            entry["event_id"]
            for entry in events
            if entry["parent_event_id"] is None
            and entry["inside_train"]
            and entry["optimizer_step"] + 1 == update["optimizer_step"]
        ]
    timing = run["timing"]
    timing.update(
        total_wall_seconds=max(entry["end_seconds"] for entry in events) + 2,
        setup_seconds=1.0,
        trainer_wall_seconds=trainer_wall or timing["trainer_wall_seconds"],
        eventcounts={
            kind: group["root_events"] for kind, group in profile["kinds"].items()
        },
        during_training_eventcounts={
            kind: group["during_training_root_events"]
            for kind, group in profile["kinds"].items()
        },
        event_instrumentation_seconds=profile["instrumentation_seconds"],
    )
    for kind, group in profile["kinds"].items():
        timing[f"{kind}_seconds"] = group["wall_seconds"]
        timing[f"during_training_{kind}_seconds"] = group[
            "during_training_wall_seconds"
        ]
    return run


@pytest.fixture
def runs(native_run):
    return [
        _events_run(native_run, number, seconds=seconds)
        for number, seconds in enumerate([1, 3, 2.2], 1)
    ]


def test_calibrates_first_repeated_initial_final_and_never_double_counts_nested(runs):
    before = deepcopy(runs)
    result = estimate_finetuning_events(runs[:2], runs[2])
    assert result["timing"]["root_event_seconds"] == {
        "min": 8,
        "expected": 16,
        "max": 24,
    }
    assert result["timing"]["comparison"]["actual"] == pytest.approx(17.6)
    assert result["timing"]["comparison"]["absolute_percentage_error"] == pytest.approx(
        100 / 11
    )
    assert result["coverage"] == {
        "target_event_groups": 6,
        "target_root_events": 8,
        "complete": True,
    }
    assert [group["phase"] for group in result["event_groups"]] == [
        "initial",
        "first",
        "first",
        "repeated",
        "repeated",
        "final",
    ]
    assert [group["target_events"] for group in result["event_groups"]] == [
        1,
        1,
        1,
        2,
        2,
        1,
    ]
    assert result["target_run"]["sampling"] == {"recorded_events": 11, "root_events": 8}
    assert runs == before
    result["signature"]["profile"]["id"] = "mutated"
    assert runs == before


def test_run_means_are_equal_weight_with_unequal_event_counts(native_run):
    result = estimate_finetuning_events(
        [
            _events_run(native_run, 1, seconds=1, repetitions=3),
            _events_run(native_run, 2, seconds=3),
        ],
        _events_run(native_run, 3, seconds=2.2),
    )
    assert result["timing"]["root_event_seconds"]["expected"] == 16
    repeated = [
        group for group in result["event_groups"] if group["phase"] == "repeated"
    ]
    assert [sample["samples"] for sample in repeated[0]["calibration_runs"]] == [3, 2]


def test_target_durations_do_not_enter_prediction(runs):
    first = estimate_finetuning_events(runs[:2], runs[2])
    for entry in runs[2]["observation"]["phase_events"]:
        entry["instrumentation_seconds"] = entry["wall_seconds"] / 2
        entry["seconds"] = entry["wall_seconds"] / 2
    _refresh_events(runs[2])
    second = estimate_finetuning_events(runs[:2], runs[2])
    assert (
        first["timing"]["root_event_seconds"] == second["timing"]["root_event_seconds"]
    )
    assert second["timing"]["comparison"]["actual"] == pytest.approx(
        first["timing"]["comparison"]["actual"] / 2
    )


def test_variable_state_json_sizes_and_output_directories_do_not_create_new_groups(
    runs,
):
    for entry in runs[2]["observation"]["phase_events"]:
        if entry["kind"] == "checkpoint":
            entry["workload"]["files"][1]["bytes"] += entry["event_id"]
            entry["workload"]["total_file_bytes"] += entry["event_id"]
            entry["workload"]["output_directories"] = ["/different/run"]
    _refresh_events(runs[2])
    assert estimate_finetuning_events(runs[:2], runs[2])["coverage"]["complete"]


def test_repeated_group_needs_two_samples_in_each_calibration_run(native_run):
    runs = [
        _events_run(native_run, 1, repetitions=1),
        _events_run(native_run, 2),
        _events_run(native_run, 3),
    ]
    with pytest.raises(
        FinetuneEstimateError, match="insufficient independent event coverage"
    ):
        estimate_finetuning_events(runs[:2], runs[2])


@pytest.mark.parametrize("field", ["phase_events", "event_profile"])
def test_public_api_refuses_protocol_downgrade(runs, field):
    del runs[2]["observation"][field]
    with pytest.raises(
        FinetuneEstimateError, match="version, ledger and event_profile"
    ):
        estimate_finetuning_events(runs[:2], runs[2])


def test_public_api_reconciles_phase_aggregates(runs):
    runs[2]["timing"]["checkpoint_seconds"] += 1
    with pytest.raises(FinetuneEstimateError, match="checkpoint_seconds"):
        estimate_finetuning_events(runs[:2], runs[2])


def test_public_api_reconciles_events_preceding_updates(runs):
    runs[2]["observation"]["updates"][0]["preceding_event_ids"] = []
    with pytest.raises(FinetuneEstimateError, match="preceding_event_ids"):
        estimate_finetuning_events(runs[:2], runs[2])


def test_legacy_native_runs_without_event_protocol_still_validate(native_run):
    validate_native_run(native_run)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda r: r[0]["run_identity"].update(uuid=r[2]["run_identity"]["uuid"]),
            "distinct run UUIDs",
        ),
        (
            lambda r: r[2]["observation"]["phase_events"][2].update(status="failed"),
            "successful event calls",
        ),
        (
            lambda r: r[2]["observation"]["phase_events"][0]["workload"].update(
                predict_with_generate=True
            ),
            "nongenerative",
        ),
        (
            lambda r: r[2]["observation"]["phase_events"][0]["workload"][
                "dataset"
            ].update(content_sha256="f" * 64),
            "coverage",
        ),
        (
            lambda r: r[2]["observation"]["phase_events"][0]["workload"].update(
                consumed_content_sha256="f" * 64
            ),
            "coverage",
        ),
        (
            lambda r: r[2]["observation"]["phase_events"][-1]["workload"].update(
                save_only_model=True
            ),
            "coverage",
        ),
    ],
)
def test_unmatched_or_unverified_evidence_never_falls_back(runs, mutation, match):
    mutation(runs)
    _refresh_events(runs[2])
    with pytest.raises(FinetuneEstimateError, match=match):
        estimate_finetuning_events(runs[:2], runs[2])


def test_duplicate_timing_evidence_is_rejected(runs):
    runs[1] = deepcopy(runs[0])
    runs[1]["run_identity"] = deepcopy(runs[2]["run_identity"])
    runs[1]["run_identity"]["uuid"] = "00000000-0000-0000-0000-000000000009"
    with pytest.raises(FinetuneEstimateError, match="duplicate timing evidence"):
        estimate_finetuning_events(runs[:2], runs[2])


def test_cli_writes_only_a_validated_event_report(runs, tmp_path):
    from fakegpu.finetune_cli import events_main

    paths = []
    for index, run in enumerate(runs):
        path = tmp_path / f"run-{index}.json"
        path.write_text(json.dumps(run))
        paths.append(path)
    output = tmp_path / "events.json"
    assert (
        events_main(
            [
                "--calibration-run",
                str(paths[0]),
                "--calibration-run",
                str(paths[1]),
                "--holdout-run",
                str(paths[2]),
                "--json",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text())
    assert report["timing"]["root_event_seconds"]["expected"] == 16


@pytest.mark.parametrize(
    "field,value",
    [
        ("compute_metrics", True),
        ("prediction_loss_only", [False]),
        ("prediction_loss_only", [None]),
    ],
)
def test_unbound_prediction_and_metric_costs_are_rejected(runs, field, value):
    runs[2]["observation"]["phase_events"][0]["workload"][field] = value
    _refresh_events(runs[2])
    with pytest.raises(FinetuneEstimateError, match="loss-only evaluation"):
        estimate_finetuning_events(runs[:2], runs[2])


def test_sparse_routing_models_are_not_silently_treated_as_dense(runs):
    run = runs[2]
    run["model_config"]["num_experts"] = 2
    _refresh(run)
    _refresh_events(run, trainer_wall=20)
    # Keep final-call bounds separate from the regenerated training timing.
    run["timing"].pop("setup_seconds")
    run["timing"].pop("trainer_wall_seconds")
    with pytest.raises(FinetuneEstimateError, match="dense model"):
        estimate_finetuning_events(runs[:2], run)


def test_unknown_evaluation_batches_are_reportable_but_not_calibration_samples(runs):
    workload = runs[2]["observation"]["phase_events"][0]["workload"]
    workload["batches"] = []
    workload["unobserved_batches"] = 1
    workload["consumed_content_status"] = "unknown"
    workload["consumed_content_sha256"] = None
    _refresh_events(runs[2])
    with pytest.raises(FinetuneEstimateError, match="known evaluation data"):
        estimate_finetuning_events(runs[:2], runs[2])


def test_only_target_event_counts_enter_replay_and_unseen_events_fail(runs):
    entry = runs[2]["observation"]["phase_events"][-1]
    entry["workload"]["files"].append({"path": "optimizer.pt", "bytes": 64})
    entry["workload"]["total_file_bytes"] += 64
    _refresh_events(runs[2])
    with pytest.raises(
        FinetuneEstimateError, match="insufficient independent event coverage"
    ):
        estimate_finetuning_events(runs[:2], runs[2])
