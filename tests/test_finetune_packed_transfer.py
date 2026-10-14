"""Synthetic packed protocol budgets; these fixtures are not GPU measurements."""

from copy import deepcopy
from collections import defaultdict

import pytest

from fakegpu._finetune_sampler import _digest, validate_sampler_plan
from fakegpu._finetune_transfer_plan import (
    transfer_plan_from_run,
    validate_transfer_plan,
    describe_transfer_components,
)
from fakegpu.finetune_transfer_plan import plan_finetuning_transfer
from fakegpu.finetune_transfer_estimator import estimate_finetuning_transfer
from fakegpu.finetune_estimator import FinetuneEstimateError
from test_finetune_transfer_plan import (
    transfer_run,
    refresh_signature,
    native_run as native_run,
)
from test_finetune_window_estimator import _refresh


def packed_run(base, number):
    run = transfer_run(base, number, factor=number)
    sampling = run["sampling_plan"]
    sampling["schema_version"] = "fakegpu.finetune_sampler_plan.v3"
    sampling["sampler"]["kind"] = "seedable_packed_random"
    seed = 123
    sampling["sampler"].update(
        seed=seed,
        **{"class": "accelerate.data_loader.SeedableRandomSampler"},
        epoch_seed_rule="seed + zero_based_epoch",
    )
    sampling["collator_class"] = (
        "fakegpu._finetune_packed_batches.IsolatedPackingCollator"
    )
    epochs = defaultdict(list)
    orders = defaultdict(list)
    for update in sampling["updates"]:
        for batch in update["batches"]:
            batch["attention_mask_kind"] = "block_causal_4d"
            batch["position_ids_kind"] = "integer_2d"
            batch["position_reset_offsets"] = [[0] for _ in range(batch["shape"][0])]
            batch["packed"] = {
                "segments": [[[0, count]] for count in batch["row_nonpadding_tokens"]],
                "logical_sequences": batch["shape"][0],
                "tensor_dtypes": {
                    "input_ids": "torch.int64",
                    "attention_mask": "torch.float32",
                    "labels": "torch.int64",
                    "position_ids": "torch.int64",
                },
            }
            batch["source_bins"] = [[index] for index in batch["row_indices"]]
            epochs[batch["epoch_index"]].append(batch["source_bins"])
            orders[batch["epoch_index"]].extend(batch["row_indices"])
    sampling["sampler"]["epoch_order_sha256"] = [
        _digest(orders[index]) for index in sorted(orders)
    ]
    sampling["packing"] = {
        "mode": "qwen35_reference.v1",
        "batch_sampler": "torch.utils.data.sampler.BatchSampler",
        "native_drop_last": False,
        "microsteps_per_epoch": len(epochs[1]),
        "row_logical_sequences": [1] * sampling["dataset"]["num_samples"],
        "epoch_source_bins": [epochs[index] for index in sorted(epochs)],
    }
    sampling["plan_sha256"] = _digest(
        {key: value for key, value in sampling.items() if key != "plan_sha256"}
    )
    validate_sampler_plan(sampling)
    run["native_config"]["packing"] = True
    run["training_config"].update(
        packing=True,
        packing_seed=seed,
        packing_reference={
            "model_mode": "qwen35_reference.v1",
            "implementation_sha256": "f" * 64,
        },
        sampler_observation_version=sampling["schema_version"],
    )
    run["observation"]["observed_model"]["collator_class"] = sampling["collator_class"]
    for actual, planned in zip(run["observation"]["updates"], sampling["updates"]):
        actual["batches"] = deepcopy(planned["batches"])
    run["observation"]["sampling_verification"]["plan_sha256"] = sampling["plan_sha256"]
    timing = deepcopy(run["timing"])
    _refresh(run)
    run["timing"] = timing
    refresh_signature(run)
    return run


def test_packed_full_budget_preserves_epoch_bins_and_geometry(native_run):
    runs = [packed_run(native_run, 1), packed_run(native_run, 2)]
    target = plan_finetuning_transfer(runs[0], max_steps=96)
    assert target["signature"]["packing"] is True
    assert target["setup_workload"]["native_packing"]["microsteps_per_epoch"] == 5
    budget = estimate_finetuning_transfer(runs, target, setup_calibration_runs=runs)
    assert budget["coverage"]["complete"]
    components = describe_transfer_components(target)
    update = next(row for row in components if row["kind"] == "optimizer_update")
    assert "packed" in update["descriptor"]["window"]["batches"][0]


def test_packed_seed_and_epoch_counts_are_bound_to_signature(native_run):
    run = packed_run(native_run, 1)
    plan = transfer_plan_from_run(run)
    changed = deepcopy(plan)
    changed["signature"]["packing_seed"] += 1
    with pytest.raises(FinetuneEstimateError, match="seed"):
        validate_transfer_plan(changed)
    plan["setup_workload"]["native_packing"]["microsteps_per_epoch"] += 1
    with pytest.raises(FinetuneEstimateError, match="epoch counts"):
        validate_transfer_plan(plan)


def test_packed_budget_cannot_pool_other_reference_implementations(native_run):
    runs = [packed_run(native_run, 1), packed_run(native_run, 2)]
    target = plan_finetuning_transfer(runs[0], max_steps=96)
    runs[1]["training_config"]["packing_reference"]["implementation_sha256"] = "e" * 64
    refresh_signature(runs[1])
    with pytest.raises(FinetuneEstimateError):
        estimate_finetuning_transfer(runs, target, setup_calibration_runs=runs)


def test_packed_reference_requires_an_implementation_digest(native_run):
    plan = transfer_plan_from_run(packed_run(native_run, 1))
    del plan["signature"]["packing_reference"]["implementation_sha256"]
    with pytest.raises(FinetuneEstimateError, match="reference implementation"):
        validate_transfer_plan(plan)


def padded_packed_run(base, width=8):
    run = packed_run(base, 1)
    for update in run["sampling_plan"]["updates"]:
        for batch in update["batches"]:
            batch["shape"][1] = width
            batch["token_slots"] = batch["shape"][0] * width
            batch["padding_tokens"] = batch["token_slots"] - batch["nonpadding_tokens"]
    sampling = run["sampling_plan"]
    sampling["plan_sha256"] = _digest(
        {k: v for k, v in sampling.items() if k != "plan_sha256"}
    )
    for actual, expected in zip(run["observation"]["updates"], sampling["updates"]):
        actual["batches"] = deepcopy(expected["batches"])
    run["observation"]["sampling_verification"]["plan_sha256"] = sampling["plan_sha256"]
    timing = deepcopy(run["timing"])
    _refresh(run)
    run["timing"] = timing
    refresh_signature(run)
    return run


def test_attested_native_packing_uses_collated_width_instead_of_source_row_cap(
    native_run,
):
    from fakegpu.finetune_run_estimator import validate_native_run

    run = padded_packed_run(native_run)
    assert run["training_config"]["sequence_length"] == 4
    assert run["workload"]["shape_counts"][0]["shape"] == [1, 8]
    validate_native_run(run, require_fixed_shape=False)
    assert transfer_plan_from_run(run)["updates"][0]["batches"][0]["shape"] == [1, 8]


def test_packed_source_cap_exception_requires_consumed_input_attestation(native_run):
    from fakegpu.finetune_run_estimator import validate_native_run

    run = padded_packed_run(native_run)
    del run["observation"]["sampling_verification"]
    with pytest.raises(FinetuneEstimateError, match="verification"):
        validate_native_run(run, require_fixed_shape=False)


def test_legacy_fixed_projection_requires_sampler_aware_packing_budget(native_run):
    from fakegpu.finetune_run_estimator import (
        estimate_finetuning_run,
        validate_native_run,
    )

    run = padded_packed_run(native_run, width=4)
    run["workload"]["fixed_shape"] = True
    run["observation"]["workload"]["fixed_shape"] = True
    validate_native_run(run)
    with pytest.raises(FinetuneEstimateError, match="packing.*transfer"):
        estimate_finetuning_run(run, max_steps=96)


def test_packing_signature_seed_cannot_be_a_numerically_equal_float(native_run):
    plan = transfer_plan_from_run(packed_run(native_run, 1))
    plan["signature"]["packing_seed"] = 123.0
    with pytest.raises(FinetuneEstimateError, match="packing_seed"):
        validate_transfer_plan(plan)
