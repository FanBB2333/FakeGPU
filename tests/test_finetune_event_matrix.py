import json
from copy import deepcopy
import hashlib

import pytest

from native import finetune_event_matrix as matrix


@pytest.mark.parametrize("framework", ["llamafactory", "axolotl"])
@pytest.mark.parametrize("steps", [20, 29])
def test_step_schedule_keeps_terminal_and_framework_final_calls(framework, steps):
    calls = matrix._expected_calls(
        framework, steps, strategy="steps", updates_per_epoch=3
    )
    evaluations = [
        call["optimizer_step"]
        for call in calls
        if call["operation"] == "evaluate" and call["inside_train"]
    ]
    checkpoints = [
        call["optimizer_step"]
        for call in calls
        if call["operation"] == "_save_checkpoint"
    ]
    assert evaluations == ([0] if framework == "axolotl" else []) + list(
        range(4, steps + 1, 4)
    ) + ([] if steps % 4 == 0 else [steps])
    assert checkpoints == list(range(6, steps + 1, 6)) + [steps]
    if framework == "llamafactory":
        assert calls[-2:] == [
            {"operation": "save_model", "optimizer_step": steps, "inside_train": False},
            {"operation": "evaluate", "optimizer_step": steps, "inside_train": False},
        ]
        # The periodic terminal eval and post-train eval are separate real calls.
        assert (
            sum(
                call["operation"] == "evaluate" and call["optimizer_step"] == steps
                for call in calls
            )
            == 2
        )
    else:
        assert calls[-1] == {
            "operation": "save_pretrained",
            "optimizer_step": steps,
            "inside_train": False,
        }


@pytest.mark.parametrize("framework", ["llamafactory", "axolotl"])
def test_epoch_schedule_retains_truncated_final_epoch(framework):
    calls = matrix._expected_calls(framework, 14, strategy="epoch", updates_per_epoch=3)
    assert [
        call["optimizer_step"]
        for call in calls
        if call["operation"] == "_save_checkpoint"
    ] == [3, 6, 9, 12, 14]
    assert [
        call["optimizer_step"]
        for call in calls
        if call["operation"] == "evaluate" and call["inside_train"]
    ] == ([0] if framework == "axolotl" else []) + [3, 6, 9, 12, 14]


def test_call_reference_excludes_nested_calls_but_keeps_same_step_roots():
    expected = matrix._expected_calls(
        "llamafactory", 8, strategy="steps", updates_per_epoch=3
    )
    events = [
        dict(call, event_id=index, parent_event_id=None, status="succeeded")
        for index, call in enumerate(expected, 1)
    ]
    events.insert(
        3,
        dict(events[2], parent_event_id=events[2]["event_id"], operation="save_model"),
    )
    report = {"observation": {"phase_events": events}}
    matrix._check_calls(report, expected)
    events.pop()
    with pytest.raises(ValueError, match="independent reference"):
        matrix._check_calls(report, expected)


def test_checkpoint_inventory_checks_full_state_and_no_rotation(tmp_path):
    expected = matrix._expected_calls(
        "llamafactory", 8, strategy="steps", updates_per_epoch=3
    )
    for step in [6, 8]:
        checkpoint = tmp_path / f"checkpoint-{step}"
        checkpoint.mkdir()
        for name in [
            "adapter_model.safetensors",
            "adapter_config.json",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
        ]:
            (checkpoint / name).write_bytes(b"fixture")
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": step})
        )
    (tmp_path / "adapter_model.safetensors").write_bytes(b"fixture")
    inventory = matrix._checkpoint_inventory(tmp_path, expected)
    assert [row["global_step"] for row in inventory] == [6, 8]
    assert all(row["file_count"] == 6 and row["bytes"] > 0 for row in inventory)
    (tmp_path / "checkpoint-6" / "optimizer.pt").unlink()
    with pytest.raises(ValueError, match="resumable"):
        matrix._checkpoint_inventory(tmp_path, expected)


def test_prepare_only_retains_native_schedules_and_separates_output(tmp_path):
    source = tmp_path / "source.json"
    config = {
        "eval_strategy": "steps",
        "eval_steps": 4,
        "save_strategy": "steps",
        "save_steps": 6,
        "save_total_limit": 100,
        "save_only_model": False,
        "do_eval": True,
        "dataset_dir": "./data",
        "model_name_or_path": "./model",
        "num_train_epochs": 100,
    }
    source.write_text(json.dumps(config))
    destination = tmp_path / "prepared"
    assert (
        matrix.main(
            [
                "--framework",
                "llamafactory",
                "--config",
                str(source),
                "--output-dir",
                str(destination),
                "--prepare-only",
            ]
        )
        == 0
    )
    summary = json.loads((destination / "summary.json").read_text())
    assert summary["status"] == "prepared"
    assert list(summary["runs"]) == [
        "calibration-1",
        "calibration-2",
        "holdout-1",
        "holdout-2",
    ]
    for name, entry in summary["runs"].items():
        generated = json.loads(open(entry["config"]).read())
        assert generated["max_steps"] == (20 if name.startswith("calibration") else 29)
        assert generated["eval_steps"] == 4 and generated["save_steps"] == 6
        assert (
            generated["save_total_limit"] == 100
            and generated["save_only_model"] is False
        )
        assert generated["output_dir"] == str(destination / name / "model")
    with pytest.raises(SystemExit):
        matrix.main(
            [
                "--framework",
                "llamafactory",
                "--config",
                str(source),
                "--output-dir",
                str(destination),
                "--prepare-only",
            ]
        )


def _checkpoint_report(scale, *, repeated=2):
    events = []
    for index in range(1 + repeated):
        events.append(
            {
                "event_id": index + 1,
                "parent_event_id": None,
                "kind": "checkpoint",
                "operation": "_save_checkpoint",
                "optimizer_step": (index + 1) * 6,
                "inside_train": True,
                "seconds": scale * (3 if index == 0 else 1),
                "workload": {
                    "save_only_model": False,
                    "artifact_kind": "training_checkpoint",
                    "files": [
                        {
                            "path": f"checkpoint-{(index + 1) * 6}/optimizer.pt",
                            "bytes": 100 + index,
                        }
                    ],
                },
            }
        )
    return {"observation": {"phase_events": events}}


def _checkpoint_projection(target):
    groups = []
    # Calibrations have first costs 3 and 9, repeated costs 1 and 3:
    # independent per-run averages predict first=6 and repeated=2.
    for key, events in matrix._reference_groups(target).items():
        first = json.loads(key)["phase"] == "first"
        expected = (6 if first else 2) * len(events)
        groups.append(
            {
                "event_group_id": hashlib.sha256(key.encode()).hexdigest(),
                "target_event_ids": [event["event_id"] for event in events],
                "predicted_seconds": {"expected": expected},
                "comparison": {"actual": sum(event["seconds"] for event in events)},
            }
        )
    return {
        "event_groups": groups,
        "timing": {
            "root_event_seconds": {
                "expected": sum(
                    group["predicted_seconds"]["expected"] for group in groups
                )
            }
        },
    }


def test_independent_event_reference_weights_processes_equally_and_keeps_first():
    calibrations = [_checkpoint_report(1), _checkpoint_report(3, repeated=4)]
    target = _checkpoint_report(2, repeated=3)
    reference = matrix._reference_comparison(
        calibrations, target, _checkpoint_projection(target)
    )
    assert reference == {
        "root_events": 4,
        "event_groups": 2,
        "predicted_seconds": 12,
        "observed_seconds": 12,
        "absolute_percentage_error": 0,
    }
    with pytest.raises(ValueError, match="coverage"):
        matrix._reference_comparison(
            [_checkpoint_report(1, repeated=1), calibrations[1]],
            target,
            _checkpoint_projection(target),
        )


@pytest.mark.parametrize(
    "field", ["predicted_seconds", "comparison", "target_event_ids"]
)
def test_independent_event_reference_rejects_estimator_drift(field):
    target = _checkpoint_report(2, repeated=3)
    projection = deepcopy(_checkpoint_projection(target))
    group = projection["event_groups"][0]
    if field == "target_event_ids":
        group[field] = []
    else:
        group[field]["expected" if field == "predicted_seconds" else "actual"] *= 2
    with pytest.raises(ValueError, match="production event"):
        matrix._reference_comparison(
            [_checkpoint_report(1), _checkpoint_report(3)], target, projection
        )


@pytest.mark.parametrize("interval", [0, -1, True, 0.25, 2.0, None])
def test_template_rejects_nonpositive_or_noninteger_intervals(interval):
    config = {
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "eval_steps": interval,
        "save_steps": 4,
        "save_total_limit": 100,
        "save_only_model": False,
        "do_eval": True,
    }
    with pytest.raises(ValueError, match="positive integer"):
        matrix._validate_template(config, "llamafactory")


def _schedule_report(eval_steps, save_steps):
    calls = matrix._expected_calls(
        "llamafactory",
        9,
        strategy="steps",
        updates_per_epoch=3,
        eval_steps=eval_steps,
        save_steps=save_steps,
    )
    return {
        "schema_version": "fakegpu.finetune_run.v1",
        "training_config": {
            "num_samples": 13,
            "micro_batch_size": 3,
            "gradient_accumulation_steps": 2,
        },
        "native_config": {"eval_steps": eval_steps, "save_steps": save_steps},
        "workload": {"optimizer_steps": 9},
        "schedule": {
            "eval_strategy": "steps",
            "save_strategy": "steps",
            "requested_eval_steps": eval_steps,
            "resolved_eval_steps": eval_steps,
            "requested_save_steps": save_steps,
            "resolved_save_steps": save_steps,
        },
        "observation": {
            "phase_events": [
                dict(call, parent_event_id=None, status="succeeded") for call in calls
            ]
        },
    }


@pytest.mark.parametrize("intervals", [(2, 4), (4, 6), (3, 7)])
def test_actual_integer_schedule_drives_call_reference(tmp_path, intervals):
    report = _schedule_report(*intervals)
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    _, calls = matrix._read_report(path, "llamafactory", 9, "steps")
    assert [
        call["optimizer_step"]
        for call in calls
        if call["operation"] == "_save_checkpoint"
    ] == list(range(intervals[1], 10, intervals[1])) + [9]


@pytest.mark.parametrize(
    "field",
    [
        "requested_eval_steps",
        "resolved_eval_steps",
        "requested_save_steps",
        "resolved_save_steps",
    ],
)
def test_native_schedule_drift_is_not_reinterpreted_as_another_fixture(tmp_path, field):
    report = _schedule_report(2, 4)
    report["schedule"][field] += 1
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="requested/resolved"):
        matrix._read_report(path, "llamafactory", 9, "steps")
