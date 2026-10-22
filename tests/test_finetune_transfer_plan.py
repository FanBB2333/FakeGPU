"""Synthetic protocol fixtures exercise the complete sampler/ledger boundary.

Input digests below identify fixture rows; they are not presented as native
tensor measurements. Real tensor attestation is tested by the sampler suite.
"""

from copy import deepcopy
import math

import pytest

from fakegpu._finetune_pipeline_plan import (
    _digest,
    describe_pipeline_components,
    pipeline_plan_from_run,
)
from fakegpu._finetune_sampler import (
    SCHEMA_VERSION as SAMPLER_SCHEMA,
    VERIFICATION_SCHEMA_VERSION,
    validate_sampler_plan,
)
from fakegpu._finetune_transfer_plan import (
    describe_transfer_components,
    transfer_plan_from_run,
    transfer_signature,
    validate_transfer_plan,
)
from fakegpu.finetune_data import profile_tokenized_records
from fakegpu.finetune_transfer_estimator import (
    compare_finetuning_transfer,
    compare_finetuning_transfer_memory,
    estimate_finetuning_transfer,
    estimate_finetuning_transfer_memory,
    inspect_finetuning_transfer_coverage,
)
from fakegpu.finetune_transfer_plan import plan_finetuning_transfer
from test_finetune_pipeline_plan import refresh_signature, sequential_run
from test_finetune_run_estimator import native_run as native_run
from test_finetune_window_estimator import _batch, _refresh


def transfer_run(base, number, *, epochs=24, factor=1, rows=5, horizon=96):
    """Add a deterministic, independently sealed future schedule to a ledger."""
    run = sequential_run(base, number, epochs=epochs, factor=factor)
    lengths = [(4, 2, 4, 2, 2)[index % 5] for index in range(rows)]
    records = [
        {"input_ids": list(range(length)), "labels": list(range(length))}
        for length in lengths
    ]
    dataset = profile_tokenized_records(
        records, sequence_length=4, source_framework="llamafactory"
    )
    run["dataset"] = dataset
    run["training_config"].update(
        num_samples=rows,
        data_signature=dataset["provenance"]["content_sha256"],
        sampler_observation_version=SAMPLER_SCHEMA,
    )
    versions = {
        "torch": "fixture",
        "llamafactory": "0.9.5",
        "transformers": "5.6.0",
        "accelerate": "1.11.0",
    }
    run["training_config"]["software_versions"] = versions
    run["observation"]["software_versions"] = deepcopy(versions)
    per_epoch = math.ceil(rows / 2)
    planned = []
    for step in range(horizon):
        epoch, position = divmod(step, per_epoch)
        batches = []
        for index in range(position * 2, min(position * 2 + 2, rows)):
            batch = _batch(lengths[index])
            batch.update(
                epoch_index=epoch + 1,
                epoch_microstep=index + 1,
                row_indices=[index],
                input_sha256=_digest({"fixture_row": index, "length": lengths[index]}),
            )
            batches.append(batch)
        planned.append({"optimizer_step": step + 1, "batches": batches})
    observed = run["observation"]["updates"]
    for actual, expected in zip(observed, planned):
        actual["batches"] = deepcopy(expected["batches"])
    # Refresh consumed-token counters without rebuilding the original timeline.
    timing = deepcopy(run["timing"])
    _refresh(run)
    run["timing"] = timing
    sampling = {
        "schema_version": SAMPLER_SCHEMA,
        "framework": "llamafactory",
        "dataset": {
            "num_samples": rows,
            "content_sha256": dataset["provenance"]["content_sha256"],
        },
        "sampler": {
            "kind": "sequential",
            "class": "torch.utils.data.sampler.SequentialSampler",
            "seed": None,
            "epoch_seed_rule": None,
            "epoch_order_sha256": [_digest(list(range(rows)))]
            * math.ceil(horizon / per_epoch),
            "versions": versions,
        },
        "collator_class": run["observation"]["observed_model"]["collator_class"],
        "dataloader": {
            "num_workers": 0,
            "prefetch_factor": None,
            "persistent_workers": False,
            "pin_memory": False,
            "drop_last": False,
        },
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "updates_per_epoch": per_epoch,
        "updates": planned,
        "workload": {
            "optimizer_steps": horizon,
            "microsteps": sum(len(update["batches"]) for update in planned),
            "samples": sum(len(update["batches"]) for update in planned),
        },
    }
    sampling["plan_sha256"] = _digest(sampling)
    run["sampling_plan"] = validate_sampler_plan(sampling)
    batches = [batch for update in observed for batch in update["batches"]]
    run["observation"]["sampling_verification"] = {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "plan_sha256": sampling["plan_sha256"],
        "optimizer_steps": len(observed),
        "microsteps": len(batches),
        "samples": sum(batch["shape"][0] for batch in batches),
        "input_sha256": _digest([batch["input_sha256"] for batch in batches]),
        "matched": True,
    }
    refresh_signature(run)
    return run


def test_future_plan_can_use_a_short_prefix_without_a_complete_observed_epoch(
    native_run,
):
    run = transfer_run(native_run, 1, epochs=4, rows=41, horizon=84)
    assert len(run["observation"]["updates"]) == 12
    assert all(
        batch["epoch_index"] == 1
        for update in run["observation"]["updates"]
        for batch in update["batches"]
    )
    plan = plan_finetuning_transfer(run, epochs=4)
    assert plan["workload"] == {
        "optimizer_steps": 84,
        "microsteps": 164,
        "samples": 164,
    }
    assert plan["updates"][-1]["batches"][0]["epoch_index"] == 4
    assert len(plan["updates"][-1]["batches"]) == 1


def test_future_prediction_and_later_independent_replay_agree(native_run):
    runs = [transfer_run(native_run, 1), transfer_run(native_run, 2, factor=3)]
    plan = plan_finetuning_transfer(runs[0], max_steps=96)
    before = deepcopy(runs)
    prediction = estimate_finetuning_transfer(runs, plan, setup_calibration_runs=runs)
    target = transfer_run(native_run, 3, epochs=32, factor=2.2)
    replay = compare_finetuning_transfer(runs, target, setup_calibration_runs=runs)
    for field in ("wall_seconds", "instrumentation_seconds", "seconds"):
        assert prediction["timing"][field] == replay["timing"][field]
        assert replay["timing"]["comparison"][field]["actual"] == pytest.approx(
            target["observation"]["timeline"]["totals"][field]
        )
    assert runs == before
    assert prediction["empirical_approximation"]["enabled"]
    memory = estimate_finetuning_transfer_memory(
        runs, plan, setup_calibration_runs=runs
    )
    measured_memory = compare_finetuning_transfer_memory(
        runs, target, setup_calibration_runs=runs
    )["estimate"]
    # Recorded and future plans retain different source/projection provenance.
    assert {
        key: value
        for key, value in measured_memory.items()
        if key != "target_plan_sha256"
    } == {key: value for key, value in memory.items() if key != "target_plan_sha256"}


def test_transfer_partition_preserves_every_original_interval_and_reference(native_run):
    run = transfer_run(native_run, 1)
    original = describe_pipeline_components(pipeline_plan_from_run(run))
    transferred = describe_transfer_components(transfer_plan_from_run(run))
    fields = ("component_id", "kind", "reference")
    assert [{field: row[field] for field in fields} for row in transferred] == [
        {field: row[field] for field in fields} for row in original
    ]


def test_epoch_start_interior_and_tail_remain_distinct(native_run):
    plan = transfer_plan_from_run(transfer_run(native_run, 1))
    groups = [
        row
        for row in describe_transfer_components(plan)
        if row["kind"] == "optimizer_update"
    ]
    assert [row["descriptor"]["window"]["epoch_context"] for row in groups[:3]] == [
        {"starts_epoch": True, "ends_epoch": False},
        {"starts_epoch": False, "ends_epoch": False},
        {"starts_epoch": False, "ends_epoch": True},
    ]
    assert groups[0]["descriptor"]["window"]["first_shape_use"] == [True, True]
    assert all(
        not any(row["descriptor"]["window"]["first_shape_use"]) for row in groups[1:]
    )


def test_prior_kernel_shape_use_is_separate_from_first_complete_window(native_run):
    plan = transfer_plan_from_run(transfer_run(native_run, 1))
    changed = deepcopy(plan)
    # Both plans later first encounter [4,2], but only the second has already
    # seen shape 2 in a different ordered accumulation window.
    plan["updates"][0]["batches"] = [_batch(4), _batch(4)]
    changed["updates"][0]["batches"] = [_batch(2), _batch(4)]
    for value in (plan, changed):
        for offset, batch in enumerate(value["updates"][0]["batches"], 1):
            batch.update(epoch_index=1, epoch_microstep=offset)
    first, second = [
        [
            row
            for row in describe_transfer_components(value)
            if row["kind"] == "optimizer_update"
        ][1]
        for value in (plan, changed)
    ]
    assert first["descriptor"]["stage"]["first_descriptor_occurrence"]
    assert second["descriptor"]["stage"]["first_descriptor_occurrence"]
    assert first["descriptor"]["window"]["first_shape_use"] == [False, True]
    assert second["descriptor"]["window"]["first_shape_use"] == [False, False]
    assert first["group_id"] != second["group_id"]


def test_training_data_locators_can_change_but_setup_identity_stays_bound(native_run):
    first = transfer_run(native_run, 1)
    second = transfer_run(native_run, 2)
    first["native_config"].update(dataset="first", dataset_dir="/data/first")
    second["native_config"].update(dataset="second", dataset_dir="/data/second")
    refresh_signature(first)
    refresh_signature(second)
    left, right = transfer_plan_from_run(first), transfer_plan_from_run(second)
    assert (
        left["signature"]["native_execution_signature"]
        != right["signature"]["native_execution_signature"]
    )
    assert transfer_signature(left) == transfer_signature(right)
    assert left["setup_workload"] == right["setup_workload"]
    with pytest.raises(ValueError, match="native input configuration"):
        estimate_finetuning_transfer(
            [first, second], left, setup_calibration_runs=[first, second]
        )


@pytest.mark.parametrize("framework", ["huggingface", "llamafactory", "axolotl"])
def test_data_locator_normalization_preserves_each_frameworks_format_options(
    native_run, framework
):
    runs = [transfer_run(native_run, 1), transfer_run(native_run, 2)]
    for index, run in enumerate(runs):
        run["framework"] = framework
        run["training_config"]["source_framework"] = framework
        run["dataset"]["source_framework"] = framework
        run["observation"]["native_framework"] = framework
        versions = {
            "torch": "fixture",
            "transformers": "5.14.1" if framework == "axolotl" else "5.6.0",
            "accelerate": "1.13.0" if framework == "axolotl" else "1.11.0",
        }
        if framework != "huggingface":
            versions[framework] = "0.18.0" if framework == "axolotl" else "0.9.5"
        run["training_config"]["software_versions"] = versions
        run["observation"]["software_versions"] = deepcopy(versions)
        sampling = run["sampling_plan"]
        sampling["framework"] = framework
        sampling["sampler"]["versions"] = versions
        sampling.pop("plan_sha256")
        sampling["plan_sha256"] = _digest(sampling)
        run["observation"]["sampling_verification"]["plan_sha256"] = sampling[
            "plan_sha256"
        ]
        if framework == "huggingface":
            run["native_config"].update(
                train_dataset=f"/data/{index}.jsonl", pad_to_multiple_of=8
            )
        elif framework == "llamafactory":
            run["native_config"].update(
                dataset=f"dataset-{index}",
                dataset_dir=f"/data/{index}",
                template="qwen3",
            )
        else:
            run["native_config"]["datasets"] = [
                {
                    "path": f"/data/{index}.jsonl",
                    "type": "alpaca",
                    "field_instruction": "instruction",
                }
            ]
        refresh_signature(run)
    assert transfer_signature(transfer_plan_from_run(runs[0])) == transfer_signature(
        transfer_plan_from_run(runs[1])
    )
    if framework == "huggingface":
        runs[1]["native_config"]["pad_to_multiple_of"] = 16
    elif framework == "llamafactory":
        runs[1]["native_config"]["template"] = "other-template"
    else:
        runs[1]["native_config"]["datasets"][0]["field_instruction"] = "other-field"
    refresh_signature(runs[1])
    assert transfer_signature(transfer_plan_from_run(runs[0])) != transfer_signature(
        transfer_plan_from_run(runs[1])
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", 99),
        ("lora_dropout", 0.5),
        ("template", "different"),
        ("eval_dataset", "other-eval"),
    ],
)
def test_non_data_execution_options_are_not_discarded_from_identity(
    native_run, field, value
):
    runs = [transfer_run(native_run, 1), transfer_run(native_run, 2, factor=3)]
    target = plan_finetuning_transfer(runs[0], max_steps=96)
    runs[1]["native_config"][field] = value
    refresh_signature(runs[1])
    with pytest.raises(ValueError, match="execution signature"):
        estimate_finetuning_transfer(runs, target, setup_calibration_runs=runs)


def test_new_horizon_requires_new_target_setup_measurements(native_run):
    runs = [
        transfer_run(native_run, 1),
        transfer_run(native_run, 2, factor=3, horizon=108),
    ]
    target = plan_finetuning_transfer(runs[0], max_steps=96)
    with pytest.raises(ValueError, match="complete target dataset"):
        estimate_finetuning_transfer(runs, target, setup_calibration_runs=runs)


def test_different_dataset_sizes_share_work_but_keep_target_startup(native_run):
    library = [
        transfer_run(native_run, 1, factor=100),
        transfer_run(native_run, 2, factor=200),
    ]
    target_calibrations = [
        transfer_run(native_run, 3, rows=15),
        transfer_run(native_run, 4, rows=15, factor=3),
    ]
    target = plan_finetuning_transfer(target_calibrations[0], max_steps=96)
    prediction = estimate_finetuning_transfer(
        library, target, setup_calibration_runs=target_calibrations
    )
    assert prediction["timing"]["by_kind"]["setup"]["wall_seconds"]["expected"] == 4
    assert prediction["setup_workload"]["dataset_num_samples"] == 15
    assert {
        run["setup_workload"]["dataset_num_samples"]
        for run in prediction["calibration_runs"]
    } == {5, 15}
    assert any(
        row["run_uuid"] == library[0]["run_identity"]["uuid"]
        for group in prediction["component_groups"]
        if group["kind"] == "optimizer_update"
        for row in group["calibration_runs"]
    )
    with pytest.raises(ValueError, match="complete target dataset"):
        estimate_finetuning_transfer(library, target, setup_calibration_runs=library)


def test_missing_shapes_are_reported_without_dropping_target_work(native_run):
    runs = [
        transfer_run(native_run, 1, epochs=4, rows=41, horizon=84),
        transfer_run(native_run, 2, epochs=4, factor=3, rows=41, horizon=84),
    ]
    target = plan_finetuning_transfer(runs[0], max_steps=84)
    coverage = inspect_finetuning_transfer_coverage(
        runs, target, setup_calibration_runs=runs
    )
    assert not coverage["complete"]
    assert any(
        group["kind"] == "optimizer_update"
        and group["descriptor"]["window"]["epoch_context"]["ends_epoch"]
        for group in coverage["missing_groups"]
    )
    with pytest.raises(ValueError, match="insufficient transfer coverage"):
        estimate_finetuning_transfer(runs, target, setup_calibration_runs=runs)


@pytest.mark.parametrize(
    "arguments,match",
    [
        ({}, "exactly one"),
        ({"epochs": 1, "max_steps": 10}, "exactly one"),
        ({"epochs": -1}, "epochs"),
        ({"epochs": True}, "epochs"),
        ({"max_steps": 97}, "horizon"),
        ({"max_steps": True}, "max_steps"),
    ],
)
def test_invalid_or_unrecorded_duration_is_rejected(native_run, arguments, match):
    with pytest.raises(ValueError, match=match):
        plan_finetuning_transfer(transfer_run(native_run, 1), **arguments)


def test_fractional_epoch_retains_whole_optimizer_windows(native_run):
    plan = plan_finetuning_transfer(transfer_run(native_run, 1), epochs=2.5)
    assert plan["workload"] == {"optimizer_steps": 8, "microsteps": 14, "samples": 14}


@pytest.mark.parametrize(("epochs", "steps"), [(2.5, 7), (32, 80)])
def test_axolotl_transfer_epochs_follow_its_max_steps_conversion(
    native_run, epochs, steps
):
    # Axolotl 0.18.0 hands Trainer max_steps = ceil(rows * num_epochs /
    # batch_size) for unpacked data: 5 rows, micro batch 1, accumulation 2.
    run = transfer_run(native_run, 1)
    versions = {
        "torch": "fixture",
        "axolotl": "0.18.0",
        "transformers": "5.14.1",
        "accelerate": "1.13.0",
    }
    run["framework"] = "axolotl"
    run["training_config"]["source_framework"] = "axolotl"
    run["dataset"]["source_framework"] = "axolotl"
    run["observation"]["native_framework"] = "axolotl"
    run["training_config"]["software_versions"] = versions
    run["observation"]["software_versions"] = deepcopy(versions)
    sampling = run["sampling_plan"]
    sampling["framework"] = "axolotl"
    sampling["sampler"]["versions"] = versions
    sampling.pop("plan_sha256")
    sampling["plan_sha256"] = _digest(sampling)
    run["observation"]["sampling_verification"]["plan_sha256"] = sampling[
        "plan_sha256"
    ]
    refresh_signature(run)
    plan = plan_finetuning_transfer(run, epochs=epochs)
    assert plan["workload"]["optimizer_steps"] == steps
    assert plan["projection"]["epoch_step_rule"] == "axolotl_max_steps"


def test_packed_axolotl_epochs_are_not_converted_without_its_estimate():
    from fakegpu.finetune_pipeline_plan import _target_steps

    with pytest.raises(ValueError, match="multipack"):
        _target_steps(
            1.0, None, 3, framework="axolotl", rows=5, batch_size=2, packed=True
        )
    assert _target_steps(None, 4, 3, framework="axolotl", packed=True) == 4


def test_old_reports_remain_available_only_to_the_original_protocol(native_run):
    run = sequential_run(native_run, 1)
    pipeline_plan_from_run(run)
    with pytest.raises(ValueError, match="verified native sampling plan"):
        transfer_plan_from_run(run)


def test_changed_consumed_input_digest_is_not_accepted(native_run):
    run = transfer_run(native_run, 1)
    run["observation"]["updates"][0]["batches"][0]["input_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="differs"):
        transfer_plan_from_run(run)


def test_changed_cpu_plan_or_work_identity_is_not_accepted(native_run):
    plan = transfer_plan_from_run(transfer_run(native_run, 1))
    plan["transfer_identity"]["dataloader"]["num_workers"] = 2
    with pytest.raises(ValueError, match="execution identity"):
        validate_transfer_plan(plan)
