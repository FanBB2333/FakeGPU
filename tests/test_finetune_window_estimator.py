from __future__ import annotations

from copy import deepcopy
import json
from uuid import UUID

import pytest

from fakegpu._finetune_batches import SCHEMA_VERSION, summarize_batch_updates
from fakegpu._finetune_model import native_config_signature, native_execution_signature
from fakegpu.finetune_data import profile_tokenized_records
from fakegpu.finetune_estimator import finetune_workload_signature
from fakegpu.finetune_window_estimator import estimate_finetuning_windows
from test_finetune_run_estimator import native_run as native_run


def _batch(length):
    return {
        "shape": [1, length],
        "token_slots": length,
        "attention_mask_kind": "binary_2d",
        "row_nonpadding_tokens": [length],
        "nonpadding_tokens": length,
        "padding_tokens": 0,
        "labels_present": True,
        "labels_kind": "integer_2d",
        "row_supervised_tokens": [length],
        "supervised_tokens": length,
        "row_causal_supervised_tokens": [length - 1],
        "causal_supervised_tokens": length - 1,
        "position_ids_kind": "missing",
        "position_reset_offsets": None,
    }


def _refresh(run):
    observation = run["observation"]
    updates = observation["updates"]
    for step, update in enumerate(updates, 1):
        update.update(
            optimizer_step=step,
            microsteps=len(update["batches"]),
            samples=sum(batch["shape"][0] for batch in update["batches"]),
            losses=[1.0] * len(update["batches"]),
        )
    profile = summarize_batch_updates(updates)
    observation["batch_profile"] = profile
    workload = {
        field: profile["totals"][field]
        for field in (
            "optimizer_steps",
            "microsteps",
            "samples",
            "warmup_optimizer_steps",
            "measured_optimizer_steps",
            "warmup_microsteps",
            "measured_microsteps",
        )
    }
    workload.update(
        kind="native_framework_processed_text",
        fixed_shape=False,
        sequence_length=4,
        sha256=run["training_config"]["data_signature"],
        shape_counts=profile["shape_counts"],
    )
    observation["workload"] = deepcopy(workload)
    run["workload"] = workload
    measured = [update for update in updates if not update["warmup"]]
    observation["optimizer_update_seconds"] = [update["seconds"] for update in measured]
    observation["seconds_per_microstep"] = [
        update["seconds"] / update["microsteps"] for update in measured
    ]
    total = sum(update["seconds"] for update in updates)
    run["timing"] = {
        "total_wall_seconds": total + 5.0,
        "training_loop_seconds": total + 1.0,
        "optimizer_update_sample_seconds": total,
        "warmup_training_seconds": sum(
            update["seconds"] for update in updates if update["warmup"]
        ),
        "trainer_other_seconds": 1.0,
    }
    run["training_config"]["native_execution_signature"] = native_execution_signature(
        native_config_signature(run["native_config"]), observation["observed_model"]
    )
    observation["signature"] = finetune_workload_signature(
        run["model_config"], run["training_config"], profile_id=run["profile_id"]
    )
    return run


def _run(base, number, *, seconds=1.0, epochs=3):
    run = deepcopy(base)
    run["run_identity"] = {
        "uuid": str(UUID(int=number)),
        "pid": 1000 + number,
        "started_at_utc": f"2026-10-13T00:00:{number:02d}+00:00",
    }
    run["native_config"] = {"seed": 42}
    run["observation"]["observed_model"] = {
        "kernel_implementations": {},
        "collator_class": "fixture.StandardTextCollator",
        "tf32_matmul": False,
        "tf32_cudnn": False,
    }
    run["dataset"] = profile_tokenized_records(
        [
            {"input_ids": [1, 2, 3, 4], "labels": [1, 2, 3, 4]},
            {"input_ids": [1, 2], "labels": [1, 2]},
            {"input_ids": [2, 3], "labels": [2, 3]},
        ],
        sequence_length=4,
        source_framework="llamafactory",
    )
    run["training_config"].update(
        batch_observation_version=SCHEMA_VERSION,
        data_signature=run["dataset"]["provenance"]["content_sha256"],
        num_samples=3,
        max_steps=None,
        epochs=epochs,
    )
    updates = []
    for epoch in range(1, epochs + 1):
        for index, lengths in enumerate(([4, 2], [2])):
            batches = [_batch(length) for length in lengths]
            for offset, batch in enumerate(batches, 1 if index == 0 else 3):
                batch.update(epoch_index=epoch, epoch_microstep=offset)
            updates.append(
                {
                    "seconds": seconds if index == 0 else seconds / 2,
                    "warmup": epoch == 1,
                    "batches": batches,
                }
            )
    run["observation"]["updates"] = updates
    return _refresh(run)


@pytest.fixture
def runs(native_run):
    return [
        _run(native_run, 1, seconds=1),
        _run(native_run, 2, seconds=3),
        _run(native_run, 3, seconds=2.2),
    ]


def test_dynamic_windows_preserve_order_and_partial_accumulation(runs):
    before = deepcopy(runs)
    result = estimate_finetuning_windows(runs[:2], runs[2])
    assert result["scope"] == "recorded_steady_optimizer_window_replay"
    assert result["coverage"] == {
        "target_window_groups": 2,
        "target_eligible_updates": 4,
        "covered_updates": 4,
        "complete": True,
    }
    assert result["timing"]["steady_optimizer_window_seconds"] == {
        "min": 3,
        "expected": 6,
        "max": 9,
    }
    assert result["timing"]["comparison"]["actual"] == pytest.approx(6.6)
    assert result["timing"]["comparison"]["absolute_percentage_error"] == pytest.approx(
        100 / 11
    )
    assert [len(group["batches"]) for group in result["window_groups"]] == [2, 1]
    assert result["target_run"]["sampling"] == {
        "recorded_updates": 6,
        "global_warmup_updates": 2,
        "first_window_occurrences": 2,
        "excluded_updates": 2,
        "eligible_updates": 4,
    }
    assert runs == before
    result["signature"]["profile"]["id"] = "mutated"
    assert runs == before


def test_process_means_receive_equal_weight_despite_different_lengths(native_run):
    result = estimate_finetuning_windows(
        [_run(native_run, 1, seconds=1, epochs=5), _run(native_run, 2, seconds=3)],
        _run(native_run, 3, seconds=2.2),
    )
    assert result["timing"]["steady_optimizer_window_seconds"]["expected"] == 6
    assert [
        sample["samples"] for sample in result["window_groups"][0]["calibration_runs"]
    ] == [4, 2]


def test_target_timings_never_enter_calibration(runs):
    before = estimate_finetuning_windows(runs[:2], runs[2])
    for update in runs[2]["observation"]["updates"]:
        update["seconds"] *= 100
    _refresh(runs[2])
    after = estimate_finetuning_windows(runs[:2], runs[2])
    assert (
        after["timing"]["steady_optimizer_window_seconds"]
        == before["timing"]["steady_optimizer_window_seconds"]
    )
    assert after["timing"]["comparison"]["actual"] == pytest.approx(660)


def test_per_update_projection_can_validate_new_epoch_suffix(runs):
    report = estimate_finetuning_windows(runs[:2], runs[2])
    updates = report["target_updates"]
    assert updates[0]["eligible"] is False
    assert updates[0]["excluded_reasons"] == [
        "global_warmup",
        "first_window_occurrence",
    ]
    assert updates[0]["predicted_seconds"] is None
    suffix = [item for item in updates if item["epoch_index"] >= 3 and item["eligible"]]
    assert [item["optimizer_step"] for item in suffix] == [5, 6]
    assert sum(item["predicted_seconds"]["expected"] for item in suffix) == 3
    assert sum(item["observed_seconds"] for item in suffix) == pytest.approx(3.3)


def test_first_occurrence_after_global_warmup_is_excluded(native_run):
    reports = [
        _run(native_run, number, seconds=number, epochs=4) for number in (1, 2, 3)
    ]
    for report in reports:
        # Make the original B occurrence another A during global warmup. B now
        # first occurs at update 4, outside global warmup, and is discarded.
        updates = report["observation"]["updates"]
        updates[1]["batches"] = deepcopy(updates[0]["batches"])
        for update in updates:
            for batch in update["batches"]:
                batch.pop("epoch_index")
                batch.pop("epoch_microstep")
        _refresh(report)
    result = estimate_finetuning_windows(reports[:2], reports[2])
    assert result["target_run"]["sampling"]["excluded_updates"] == 3
    assert result["target_updates"][3]["excluded_reasons"] == [
        "first_window_occurrence"
    ]
    assert result["target_updates"][3]["predicted_seconds"] is None


@pytest.mark.parametrize("count", [0, 1])
def test_requires_two_independent_calibration_runs(runs, count):
    with pytest.raises(ValueError, match="at least two runs"):
        estimate_finetuning_windows(runs[:count], runs[2])


def test_rejects_repeated_run_uuid(runs):
    runs[2]["run_identity"]["uuid"] = runs[0]["run_identity"]["uuid"]
    with pytest.raises(ValueError, match="distinct run UUIDs"):
        estimate_finetuning_windows(runs[:2], runs[2])


def test_rejects_copied_timing_evidence_with_fresh_uuid(runs):
    runs[1] = deepcopy(runs[0])
    runs[1]["run_identity"]["uuid"] = str(UUID(int=99))
    with pytest.raises(ValueError, match="duplicate timing evidence"):
        estimate_finetuning_windows(runs[:2], runs[2])


@pytest.mark.parametrize(
    "field,value",
    [
        ("uuid", "not-a-uuid"),
        ("pid", True),
        ("pid", 0),
        ("started_at_utc", "2026-10-13T00:00:00"),
        ("started_at_utc", "2026-10-13T00:00:00+08:00"),
    ],
)
def test_run_identity_requires_valid_process_metadata(runs, field, value):
    runs[0]["run_identity"][field] = value
    with pytest.raises(ValueError, match="canonical UUID"):
        estimate_finetuning_windows(runs[:2], runs[2])


def test_requires_original_execution_evidence(runs):
    runs[0].pop("native_config")
    with pytest.raises(ValueError, match="requires native_config"):
        estimate_finetuning_windows(runs[:2], runs[2])


def test_does_not_allow_matching_signatures_to_hide_changed_native_config(runs):
    runs[0]["native_config"]["seed"] = 100
    with pytest.raises(ValueError, match="native execution signature"):
        estimate_finetuning_windows(runs[:2], runs[2])


def test_changed_execution_signature_is_not_pooled(runs):
    runs[0]["native_config"]["seed"] = 100
    _refresh(runs[0])
    with pytest.raises(ValueError, match="signatures must match"):
        estimate_finetuning_windows(runs[:2], runs[2])


def test_changed_dataset_identity_is_not_pooled(runs):
    run = runs[0]
    dataset = profile_tokenized_records(
        [{"input_ids": [8, 9], "labels": [8, 9]}] * 3,
        sequence_length=4,
        source_framework="llamafactory",
    )
    run["dataset"] = dataset
    run["training_config"]["data_signature"] = dataset["provenance"]["content_sha256"]
    _refresh(run)
    with pytest.raises(ValueError, match="signatures must match"):
        estimate_finetuning_windows(runs[:2], runs[2])


def test_every_calibration_run_needs_two_samples_per_target_window(native_run):
    calibrations = [_run(native_run, 1, epochs=2), _run(native_run, 2, seconds=3)]
    with pytest.raises(
        ValueError,
        match=r"insufficient independent window coverage:.*1 eligible samples",
    ):
        estimate_finetuning_windows(calibrations, _run(native_run, 3, seconds=2))


@pytest.mark.parametrize("change", ["order", "labels", "tail_shape"])
def test_missing_window_coverage_has_no_shape_or_token_fallback(runs, change):
    target = runs[2]
    for index, update in enumerate(target["observation"]["updates"]):
        if change == "order" and index % 2 == 0:
            update["batches"].reverse()
        elif change == "labels" and index % 2 == 1:
            update["batches"][0].update(
                row_supervised_tokens=[1],
                supervised_tokens=1,
                row_causal_supervised_tokens=[0],
                causal_supervised_tokens=0,
            )
        elif change == "tail_shape" and index % 2 == 1:
            update["batches"] = [_batch(3)]
    # Group matching excludes epoch location, while ledger order stays truthful.
    for epoch in range(3):
        batches = sum(
            (
                item["batches"]
                for item in target["observation"]["updates"][2 * epoch : 2 * epoch + 2]
            ),
            [],
        )
        for offset, batch in enumerate(batches, 1):
            batch.update(epoch_index=epoch + 1, epoch_microstep=offset)
    _refresh(target)
    with pytest.raises(ValueError, match="insufficient independent window coverage"):
        estimate_finetuning_windows(runs[:2], target)


@pytest.mark.parametrize("kind", ["mask", "labels"])
def test_unknown_token_metadata_cannot_calibrate_windows(runs, kind):
    for update in runs[0]["observation"]["updates"]:
        for batch in update["batches"]:
            if kind == "mask":
                batch.update(
                    attention_mask_kind="missing",
                    row_nonpadding_tokens=None,
                    nonpadding_tokens=None,
                    padding_tokens=None,
                )
            else:
                batch.update(labels_kind="missing", labels_present=False)
            batch.update(
                row_supervised_tokens=None,
                supervised_tokens=None,
                row_causal_supervised_tokens=None,
                causal_supervised_tokens=None,
            )
    _refresh(runs[0])
    with pytest.raises(ValueError, match="known binary masks and integer labels"):
        estimate_finetuning_windows(runs[:2], runs[2])


@pytest.mark.parametrize("model_type", ["qwen3_moe", "qwen3_5_moe_text"])
def test_moe_cost_is_not_inferred_from_shape(runs, model_type):
    runs[0]["model_config"]["model_type"] = model_type
    _refresh(runs[0])
    with pytest.raises(ValueError, match="requires a dense model"):
        estimate_finetuning_windows(runs[:2], runs[2])


def test_damaged_batch_profile_is_recomputed(runs):
    runs[0]["observation"]["batch_profile"]["totals"]["samples"] += 1
    with pytest.raises(ValueError, match="batch_profile differs"):
        estimate_finetuning_windows(runs[:2], runs[2])


def test_packing_remains_unsupported(runs):
    runs[0]["training_config"]["packing"] = True
    _refresh(runs[0])
    with pytest.raises(ValueError, match="packing"):
        estimate_finetuning_windows(runs[:2], runs[2])


@pytest.mark.parametrize("run_index", [0, 2], ids=["calibration", "target"])
@pytest.mark.parametrize("field", ["eval_strategy", "save_strategy"])
def test_periodic_events_cannot_reuse_window_calibration(runs, run_index, field):
    runs[run_index]["schedule"][field] = "steps"
    with pytest.raises(ValueError, match="periodic evaluation or checkpointing"):
        estimate_finetuning_windows(runs[:2], runs[2])


def test_target_with_only_warmup_or_first_occurrences_has_no_prediction(runs):
    target = runs[2]
    target["observation"]["updates"] = target["observation"]["updates"][:4]
    for index, length in ((2, 3), (3, 1)):
        target["observation"]["updates"][index]["batches"] = [_batch(length)]
    for update in target["observation"]["updates"]:
        for batch in update["batches"]:
            batch.pop("epoch_index", None)
            batch.pop("epoch_microstep", None)
    _refresh(target)
    with pytest.raises(ValueError, match="no eligible steady windows"):
        estimate_finetuning_windows(runs[:2], target)


def test_legacy_reports_do_not_gain_dynamic_calibration(runs):
    runs[0]["training_config"].pop("batch_observation_version")
    _refresh(runs[0])
    with pytest.raises(ValueError, match="native batch observation protocol"):
        estimate_finetuning_windows(runs[:2], runs[2])


def test_cli_reads_reports_without_training_dependencies(runs, tmp_path):
    from fakegpu.finetune_cli import windows_main

    paths = []
    for index, run in enumerate(runs):
        path = tmp_path / f"run-{index}.json"
        path.write_text(json.dumps(run))
        paths.append(str(path))
    output = tmp_path / "comparison.json"
    assert (
        windows_main(
            [
                "--calibration-run",
                paths[0],
                "--calibration-run",
                paths[1],
                "--holdout-run",
                paths[2],
                "--json",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text())["timing"]["comparison"]["predicted"] == 6
