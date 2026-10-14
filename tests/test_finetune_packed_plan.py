"""Packed plan identity, exact source bins and consumed tensor attestation."""

from copy import deepcopy

import pytest

from fakegpu._finetune_packed_sampler import (
    plan_native_packed_batches,
    validate_packed_sampler_plan,
)
from fakegpu._finetune_sampler import (
    NativeBatchMatcher,
    validate_sampling_verification,
    validate_sampler_plan,
    _digest,
)
from fakegpu.finetune_data import profile_tokenized_records
from fakegpu.finetune_estimator import FinetuneEstimateError
from test_finetune_packed_sampler import trainer as trainer


def plan(trainer):
    from fakegpu._finetune_packed_sampler import install_packed_random_sampling

    install_packed_random_sampling(
        trainer, framework="huggingface", seed=123, replace=setattr
    )
    profile = profile_tokenized_records(trainer.train_dataset, sequence_length=8)
    return plan_native_packed_batches(
        trainer, max_steps=5, framework="huggingface", dataset_profile=profile
    )


def reseal(value):
    value["plan_sha256"] = _digest(
        {key: item for key, item in value.items() if key != "plan_sha256"}
    )


def test_plan_native_prefix_and_exact_tensor_attestation(trainer):
    result = plan(trainer)
    assert result["schema_version"] == "fakegpu.finetune_sampler_plan.v3"
    assert result["workload"] == {"optimizer_steps": 5, "microsteps": 8, "samples": 20}
    assert validate_sampler_plan(result) == result
    matcher = NativeBatchMatcher(result)
    actual = []
    for update in result["updates"]:
        batches = []
        for batch in update["batches"]:
            inputs = trainer.data_collator(
                [
                    deepcopy(trainer.train_dataset[index])
                    for index in batch["row_indices"]
                ]
            )
            batches.append(
                matcher.observe(
                    inputs,
                    epoch_index=batch["epoch_index"],
                    epoch_microstep=batch["epoch_microstep"],
                )
            )
        actual.append({"optimizer_step": update["optimizer_step"], "batches": batches})
    verification = matcher.finish()
    assert validate_sampling_verification(result, actual, verification) == verification


@pytest.mark.parametrize(
    "change",
    [
        "duplicate",
        "missing",
        "epoch_length",
        "bin",
        "geometry",
        "dtype",
        "logical",
        "source_index",
        "seed",
        "tail",
        "hash",
        "extra_update",
        "extra_batch",
        "boolean_step",
        "invalid_dtype_type",
        "invalid_epoch_hashes",
        "float_row_index",
        "float_bin_index",
        "float_workload",
    ],
)
def test_invalid_packed_schedules_rejected(trainer, change):
    value = plan(trainer)
    packing = value["packing"]
    batch = value["updates"][0]["batches"][0]
    if change == "duplicate":
        packing["epoch_source_bins"][0][0][1] = deepcopy(
            packing["epoch_source_bins"][0][0][0]
        )
    elif change == "missing":
        packing["epoch_source_bins"][0][-1] = [[1]]
    elif change == "epoch_length":
        packing["microsteps_per_epoch"] += 1
    elif change == "bin":
        batch["source_bins"][0].append(2)
    elif change == "geometry":
        batch["packed"]["segments"][0][0][1] = 3
    elif change == "dtype":
        batch["packed"]["tensor_dtypes"]["attention_mask"] = "torch.int64"
    elif change == "logical":
        batch["packed"]["logical_sequences"] += 1
    elif change == "source_index":
        batch["row_indices"][0] = 8
    elif change == "seed":
        value["sampler"]["seed"] = True
    elif change == "tail":
        value["dataloader"]["drop_last"] = True
    elif change == "hash":
        batch["input_sha256"] = "broken"
    elif change == "extra_batch":
        batch["seconds"] = 1
    elif change == "boolean_step":
        value["updates"][0]["optimizer_step"] = True
    elif change == "invalid_dtype_type":
        batch["packed"]["tensor_dtypes"]["input_ids"] = []
    elif change == "invalid_epoch_hashes":
        value["sampler"]["epoch_order_sha256"] = None
    elif change == "float_row_index":
        batch["row_indices"][0] = float(batch["row_indices"][0])
    elif change == "float_bin_index":
        batch["source_bins"][0][0] = float(batch["source_bins"][0][0])
    elif change == "float_workload":
        value["workload"]["samples"] = float(value["workload"]["samples"])
    else:
        value["updates"][0]["seconds"] = 1
    reseal(value)
    with pytest.raises(ValueError):
        validate_packed_sampler_plan(value)


def test_actual_changed_packed_input_rejected(trainer):
    result = plan(trainer)
    first = result["updates"][0]["batches"][0]
    inputs = trainer.data_collator(
        [deepcopy(trainer.train_dataset[index]) for index in first["row_indices"]]
    )
    inputs["input_ids"][0, 2] += 1
    with pytest.raises(FinetuneEstimateError, match="tensor contents"):
        NativeBatchMatcher(result).observe(inputs, epoch_index=1, epoch_microstep=1)


@pytest.mark.parametrize("seed", [True, -1, 2**63, 1.2, "1"])
def test_packing_seed_rejected_before_file_or_gpu_access(seed):
    from fakegpu.finetune_runner import run_finetuning

    with pytest.raises(ValueError, match="packing_seed"):
        run_finetuning(
            "missing.json",
            framework="huggingface",
            profile_id="a100",
            planning_max_steps=8,
            packing_seed=seed,
        )


def test_packing_requires_a_horizon_and_exclusive_sampler():
    from fakegpu.finetune_runner import run_finetuning

    for options in ({}, {"planning_max_steps": 8, "length_grouping_seed": 0}):
        with pytest.raises(ValueError, match="planning horizon"):
            run_finetuning(
                "missing.json",
                framework="huggingface",
                profile_id="a100",
                packing_seed=0,
                **options,
            )


def test_packing_seed_zero_is_forwarded_without_changing_legacy_commands(tmp_path):
    from fakegpu.finetune_workflow import _native_command

    request = {
        "python_executable": "python",
        "framework": "huggingface",
        "profile_id": "a100",
        "warmup_steps": 2,
        "cache_regime": "uncontrolled",
        "planning_duration": {"epochs": None, "max_steps": 8},
    }
    original = _native_command(request, tmp_path)
    assert "--packing-seed" not in original
    seeded = _native_command(dict(request, packing_seed=0), tmp_path)
    offset = seeded.index("--packing-seed")
    assert seeded[offset + 1] == "0"
    assert seeded[:offset] + seeded[offset + 2 :] == original
