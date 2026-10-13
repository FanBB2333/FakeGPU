from copy import deepcopy
import json
from pathlib import Path

import pytest

from fakegpu._finetune_events import SCHEMA_VERSION
from fakegpu.finetune_window_estimator import estimate_finetuning_windows
from native import finetune_event_window_matrix as matrix
from test_finetune_event_estimator import _refresh_events
from test_finetune_events import event
from test_finetune_run_estimator import native_run as native_run
from test_finetune_window_estimator import _refresh, _run


def _periodic_run(base, number, *, seconds=1.0, epochs=36):
    run = _run(base, number, seconds=seconds, epochs=epochs)
    run["training_config"]["event_observation_version"] = SCHEMA_VERSION
    run["native_config"].update(eval_steps=4, save_steps=6)
    run["schedule"] = {"eval_strategy": "steps", "save_strategy": "steps"}
    for update in run["observation"]["updates"]:
        update["warmup"] = update["optimizer_step"] <= 4
    _refresh(run)
    events = []
    clock = 1.0
    for update in run["observation"]["updates"]:
        clock += update["seconds"]
        step = update["optimizer_step"]
        terminal = step == epochs * 2
        for operation, active in (
            ("evaluate", step % 4 == 0 or terminal),
            ("_save_checkpoint", step % 6 == 0 or terminal),
        ):
            if active:
                events.append(
                    event(
                        len(events) + 1,
                        operation=operation,
                        step=step,
                        start=clock,
                        seconds=0.1,
                    )
                )
                clock += 0.1
    trainer_wall = run["timing"]["training_loop_seconds"] + sum(
        row["wall_seconds"] for row in events
    )
    return _refresh_events(run, events, trainer_wall)


def test_reference_keeps_each_kind_boundary_bundle_and_distance(native_run):
    run = _periodic_run(native_run, 1)
    trace = matrix._reference_windows(run)
    at13 = json.loads(trace[12]["key"])["context"]
    assert at13["evaluation"] == at13["checkpoint"]
    assert [row["operation"] for row in at13["evaluation"]["events"]] == [
        "evaluate",
        "_save_checkpoint",
    ]
    assert at13["evaluation"]["optimizer_updates_since_boundary"] == 0
    at17 = json.loads(trace[16]["key"])["context"]
    assert [row["operation"] for row in at17["evaluation"]["events"]] == ["evaluate"]
    assert at17["evaluation"]["optimizer_updates_since_boundary"] == 0
    assert at17["checkpoint"]["events"] == at13["checkpoint"]["events"]
    assert at17["checkpoint"]["optimizer_updates_since_boundary"] == 4


def test_independent_reference_matches_api_and_checks_longer_postevent_suffix(
    native_run,
):
    calibrations = [
        _periodic_run(native_run, 1),
        _periodic_run(native_run, 2, seconds=3),
    ]
    target = _periodic_run(native_run, 3, seconds=2.2, epochs=50)
    projection = estimate_finetuning_windows(calibrations, target, event_context=True)
    reference = matrix._reference_comparison(
        calibrations, target, projection, calibration_steps=72
    )
    assert reference["minimum_calibration_samples_per_group_per_run"] >= 2
    assert reference["longer_holdout_suffix"]["optimizer_updates"] == 28
    assert reference["suffix_immediately_after_event"]["optimizer_updates"] > 0
    assert reference["whole_trace"]["absolute_percentage_error"] == pytest.approx(
        100 / 11
    )


@pytest.mark.parametrize(
    "field",
    [
        "eligible",
        "window_id",
        "immediately_after_event",
        "predicted_seconds",
        "observed_seconds",
    ],
)
def test_reference_rejects_projection_context_or_timing_drift(native_run, field):
    calibrations = [
        _periodic_run(native_run, 1),
        _periodic_run(native_run, 2, seconds=3),
    ]
    target = _periodic_run(native_run, 3, epochs=50)
    projection = estimate_finetuning_windows(calibrations, target, event_context=True)
    projection = deepcopy(projection)
    row = next(row for row in projection["target_updates"] if row["eligible"])
    if field in {"eligible", "immediately_after_event"}:
        row[field] = not row[field]
    elif field == "window_id":
        row[field] = "different"
    elif field == "predicted_seconds":
        row[field]["expected"] *= 2
    else:
        row[field] *= 2
    with pytest.raises(ValueError, match="production event-window"):
        matrix._reference_comparison(
            calibrations, target, projection, calibration_steps=72
        )


def test_prepare_only_delegates_native_audits_and_preserves_longer_plan(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "eval_strategy": "steps",
                "eval_steps": 4,
                "save_strategy": "steps",
                "save_steps": 6,
                "save_total_limit": 100,
                "save_only_model": False,
                "do_eval": True,
                "dataset_dir": "./data",
            }
        )
    )
    output = tmp_path / "prepared"
    assert (
        matrix.main(
            [
                "--framework",
                "llamafactory",
                "--config",
                str(source),
                "--output-dir",
                str(output),
                "--prepare-only",
            ]
        )
        == 0
    )
    summary = json.loads((output / "event-window-summary.json").read_text())
    events = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "prepared"
    assert summary["sampling_plan"]["warmup_steps"] == 4
    assert len(events["runs"]) == 4
    for name, entry in events["runs"].items():
        config = json.loads(Path(entry["config"]).read_text())
        assert config["max_steps"] == (72 if name.startswith("calibration") else 101)
