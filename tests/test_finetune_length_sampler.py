"""Seeded native length grouping, RNG isolation and serialized plan identity."""

from copy import deepcopy

import pytest

from fakegpu._finetune_sampler import (
    LENGTH_SCHEMA_VERSION,
    NativeBatchMatcher,
    plan_native_batches,
    validate_sampler_plan,
    validate_length_grouping_evaluation,
)
from fakegpu.structured_io import canonical_sha256 as _digest
from fakegpu.finetune_estimator import FinetuneEstimateError
from fakegpu.finetune_runner import run_finetuning
from fakegpu.finetune_workflow import _native_command
from test_finetune_sampler import trainer as trainer, reseal
from fakegpu._finetune_transfer_plan import transfer_plan_from_run, validate_transfer_plan
from fakegpu.finetune_transfer_plan import plan_finetuning_transfer
from fakegpu.finetune_transfer_estimator import estimate_finetuning_transfer
from test_finetune_transfer_plan import transfer_run, native_run as native_run, refresh_signature


torch = pytest.importorskip("torch")
trainer_utils = pytest.importorskip("transformers.trainer_pt_utils")
native_sampler = pytest.importorskip("fakegpu._finetune_length_sampler")
LengthGroupedSampler = trainer_utils.LengthGroupedSampler
get_length_grouped_indices = trainer_utils.get_length_grouped_indices
SeedableLengthGroupedSampler = native_sampler.SeedableLengthGroupedSampler
install_length_grouping = native_sampler.install_length_grouping
FixedLengthGroupedSampler = native_sampler.FixedLengthGroupedSampler


def configured(trainer):
    trainer.args.train_sampling_strategy = "group_by_length"
    trainer.args.train_batch_size = trainer.args.per_device_train_batch_size
    trainer.args.eval_strategy = "no"
    trainer.args.do_eval = False
    trainer.eval_dataset = None
    trainer._get_train_sampler = lambda dataset=None: LengthGroupedSampler(
        6, dataset=trainer.train_dataset if dataset is None else dataset
    )
    trainer.get_train_dataloader = lambda: torch.utils.data.DataLoader(
        trainer.train_dataset,
        sampler=trainer._get_train_sampler(),
        batch_size=3,
        collate_fn=trainer.data_collator,
    )
    return trainer


def installed(trainer, seed=123):
    trainer = configured(trainer)
    original = trainer._get_train_sampler
    install_length_grouping(trainer, seed=seed, replace=setattr)
    return trainer, original


def planned(trainer, steps=8):
    return plan_native_batches(
        trainer, max_steps=steps, framework="huggingface", sequence_length=64
    )


def test_native_global_rng_grouping_is_not_claimed_predictable(trainer):
    configured(trainer)
    with pytest.raises(FinetuneEstimateError, match="global RNG.*length_grouping_seed"):
        planned(trainer)


def test_epoch_seeded_grouping_matches_native_algorithm_and_preserves_rng(trainer):
    trainer, _ = installed(trainer)
    state = torch.get_rng_state().clone()
    rows = deepcopy(trainer.train_dataset)
    result = planned(trainer)
    assert result["schema_version"] == LENGTH_SCHEMA_VERSION
    assert torch.equal(torch.get_rng_state(), state)
    assert trainer.train_dataset == rows
    assert result["workload"] == {"optimizer_steps": 8, "microsteps": 14, "samples": 38}
    lengths = [len(row["input_ids"]) for row in rows]
    assert result["sampler"]["grouping"] == {
        "batch_size": 6, "lengths": lengths, "lengths_sha256": _digest(lengths)
    }
    native = trainer._get_train_sampler()
    for epoch in range(3):
        # Perturb the model/global RNG between epochs; sampler outputs stay fixed.
        torch.rand(37 + epoch)
        native.set_epoch(epoch)
        actual = list(native)
        expected = get_length_grouped_indices(
            lengths, 6, generator=torch.Generator().manual_seed(123 + epoch)
        )
        assert actual == expected
        assert lengths[actual[0]] == max(lengths)
        prefix = [
            index for update in result["updates"] for batch in update["batches"]
            if batch["epoch_index"] == epoch + 1 for index in batch["row_indices"]
        ]
        assert prefix == actual[:len(prefix)]


def test_attestation_uses_actual_grouped_row_contents(trainer):
    trainer, _ = installed(trainer)
    result = planned(trainer)
    matcher = NativeBatchMatcher(result)
    for update in result["updates"]:
        for batch in update["batches"]:
            inputs = trainer.data_collator([
                deepcopy(trainer.train_dataset[index]) for index in batch["row_indices"]
            ])
            evidence = matcher.observe(
                inputs, epoch_index=batch["epoch_index"],
                epoch_microstep=batch["epoch_microstep"],
            )
            assert evidence["input_sha256"] == batch["input_sha256"]
    assert matcher.finish(expected_updates=8)["matched"] is True


@pytest.mark.parametrize("field,value,match", [
    ("train_sampling_strategy", "random", "train_sampling_strategy"),
    ("eval_strategy", "steps", "evaluation disabled"),
    ("do_eval", True, "evaluation disabled"),
])
def test_install_rejects_unsupported_native_policy_without_mutation(trainer, field, value, match):
    configured(trainer)
    original = trainer._get_train_sampler
    setattr(trainer.args, field, value)
    if field in {"eval_strategy", "do_eval"}:
        match = "one finite evaluation dataset"
    with pytest.raises(FinetuneEstimateError, match=match):
        install_length_grouping(trainer, seed=123, replace=setattr)
    assert trainer._get_train_sampler is original


def test_evaluation_repeats_fixed_native_grouping_without_model_rng_dependency(trainer):
    configured(trainer)
    trainer.args.eval_strategy = "steps"
    trainer.args.do_eval = True
    trainer.eval_dataset = deepcopy(trainer.train_dataset)
    trainer._get_eval_sampler = lambda dataset: LengthGroupedSampler(3, dataset=dataset)
    original = trainer._get_eval_sampler
    policy = install_length_grouping(trainer, seed=123, replace=setattr)
    sampler = trainer._get_eval_sampler(trainer.eval_dataset)
    assert type(sampler) is FixedLengthGroupedSampler
    state = torch.get_rng_state().clone()
    expected = get_length_grouped_indices(
        sampler.lengths, 3, generator=torch.Generator().manual_seed(123)
    )
    for epoch in range(4):
        sampler.set_epoch(epoch)
        assert list(sampler) == expected
    assert torch.equal(torch.get_rng_state(), state)
    assert trainer._get_eval_sampler is not original
    assert policy["iteration_seed_rule"] == "same_seed_each_invocation"
    assert policy["lengths_sha256"] == _digest(sampler.lengths)
    assert validate_length_grouping_evaluation(policy, seed=123) == policy


@pytest.mark.parametrize("field,value", [
    ("schema_version", "wrong"), ("kind", "random"), ("seed", True),
    ("seed", 124), ("iteration_seed_rule", "seed + epoch"),
    ("dataset_rows", 0), ("grouping_batch_size", False), ("lengths_sha256", "bad"),
])
def test_invalid_evaluation_policy_rejected(field, value):
    policy = {
        "schema_version": "fakegpu.finetune_eval_sampler.v1",
        "kind": "fixed_seed_length_grouped", "seed": 123,
        "iteration_seed_rule": "same_seed_each_invocation",
        "grouping_batch_size": 2, "dataset_rows": 7, "lengths_sha256": "a" * 64,
    }
    policy[field] = value
    with pytest.raises(FinetuneEstimateError):
        validate_length_grouping_evaluation(policy, seed=123)


def test_unsupported_evaluation_sampler_does_not_partially_replace_training(trainer):
    configured(trainer)
    trainer.eval_dataset = trainer.train_dataset
    trainer._get_eval_sampler = lambda dataset: torch.utils.data.SequentialSampler(dataset)
    original = trainer._get_train_sampler
    with pytest.raises(FinetuneEstimateError, match="ordinary native LengthGroupedSampler"):
        install_length_grouping(trainer, seed=123, replace=setattr)
    assert trainer._get_train_sampler is original


def test_prepared_evaluation_loader_is_rejected(trainer):
    configured(trainer)
    trainer._eval_dataloaders = {"eval": object()}
    with pytest.raises(FinetuneEstimateError, match="before dataloader preparation"):
        install_length_grouping(trainer, seed=123, replace=setattr)


def test_cached_loader_and_custom_sampler_rejected_before_install(trainer):
    configured(trainer)
    original = trainer._get_train_sampler
    trainer._train_dataloader = object()
    with pytest.raises(FinetuneEstimateError, match="before dataloader"):
        install_length_grouping(trainer, seed=123, replace=setattr)
    assert trainer._get_train_sampler is original
    del trainer._train_dataloader
    trainer._get_train_sampler = lambda dataset=None: torch.utils.data.RandomSampler(trainer.train_dataset)
    original = trainer._get_train_sampler
    with pytest.raises(FinetuneEstimateError, match="ordinary native"):
        install_length_grouping(trainer, seed=123, replace=setattr)
    assert trainer._get_train_sampler is original


def test_original_sampler_is_restorable(trainer):
    configured(trainer)
    original = trainer._get_train_sampler
    records = []

    def replace(obj, name, value):
        records.append((obj, name, obj.__dict__[name]))
        setattr(obj, name, value)

    install_length_grouping(trainer, seed=123, replace=replace)
    assert isinstance(trainer._get_train_sampler(), SeedableLengthGroupedSampler)
    for obj, name, value in reversed(records):
        setattr(obj, name, value)
    assert trainer._get_train_sampler is original


@pytest.mark.parametrize("change,match", [
    ("schema", "v2"),
    ("seed", "epoch seed"),
    ("length_count", "dataset size"),
    ("length_type", "native grouping length"),
    ("length_sha", "SHA256"),
    ("batch_size", "accumulation"),
    ("kind", "v2"),
])
def test_v2_identity_cannot_be_silently_downgraded_or_modified(trainer, change, match):
    trainer, _ = installed(trainer)
    result = planned(trainer)
    grouping = result["sampler"]["grouping"]
    if change == "schema":
        result["schema_version"] = "fakegpu.finetune_sampler_plan.v1"
        with pytest.raises(FinetuneEstimateError):
            reseal(result)
            validate_sampler_plan(result)
        return
    if change == "seed":
        result["sampler"]["seed"] = None
    elif change == "length_count":
        grouping["lengths"].pop()
    elif change == "length_type":
        grouping["lengths"][0] = True
    elif change == "length_sha":
        grouping["lengths"][0] += 1
    elif change == "batch_size":
        grouping["batch_size"] = 3
    else:
        result["sampler"]["kind"] = "seedable_random"
        result["sampler"]["class"] = "accelerate.data_loader.SeedableRandomSampler"
    reseal(result)
    with pytest.raises(FinetuneEstimateError, match=match):
        validate_sampler_plan(result)


@pytest.mark.parametrize("seed", [-1, True, 1.5, 2**63])
def test_bad_seed_rejected_before_config_or_gpu(seed):
    with pytest.raises(ValueError, match="length_grouping_seed"):
        run_finetuning(
            "missing.json", framework="huggingface", profile_id="a100",
            planning_max_steps=8, length_grouping_seed=seed,
        )


def test_seed_requires_horizon_before_config_or_gpu():
    with pytest.raises(ValueError, match="planning horizon"):
        run_finetuning(
            "missing.json", framework="huggingface", profile_id="a100",
            length_grouping_seed=0,
        )


def test_workspace_child_passes_explicit_seed_and_preserves_legacy_command(tmp_path):
    request = {
        "python_executable": "python", "framework": "huggingface",
        "profile_id": "a100", "warmup_steps": 2, "cache_regime": "uncontrolled",
        "planning_duration": {"epochs": None, "max_steps": 8},
    }
    legacy = _native_command(request, tmp_path)
    seeded = _native_command({**request, "length_grouping_seed": 0}, tmp_path)
    assert "--length-grouping-seed" not in legacy
    index = seeded.index("--length-grouping-seed")
    assert seeded[index + 1] == "0"
    assert seeded[:index] + seeded[index + 2:] == legacy


def v2_fixture(base, number):
    """Synthetic protocol conversion, not a native sampler execution claim."""
    run = transfer_run(base, number, factor=number)
    sampling = run["sampling_plan"]
    sampling["schema_version"] = LENGTH_SCHEMA_VERSION
    sampling["sampler"].update(
        kind="seedable_length_grouped",
        seed=123,
        epoch_seed_rule="seed + zero_based_epoch",
        **{"class": "fakegpu._finetune_length_sampler.SeedableLengthGroupedSampler"},
        grouping={"batch_size": 2, "lengths": [4, 2, 4, 2, 2], "lengths_sha256": _digest([4, 2, 4, 2, 2])},
    )
    reseal(sampling)
    run["training_config"].update(
        sampler_observation_version=LENGTH_SCHEMA_VERSION,
        length_grouping_seed=123,
    )
    run["observation"]["sampling_verification"]["plan_sha256"] = sampling["plan_sha256"]
    refresh_signature(run)
    return run


def test_v2_native_signature_and_future_budget_are_bound_to_grouping(native_run):
    runs = [v2_fixture(native_run, 1), v2_fixture(native_run, 2)]
    plan = plan_finetuning_transfer(runs[0], max_steps=96)
    assert plan["transfer_identity"]["sampler"]["grouping"] == runs[0]["sampling_plan"]["sampler"]["grouping"]
    budget = estimate_finetuning_transfer(runs, plan, setup_calibration_runs=runs)
    assert budget["coverage"]["complete"] is True
    changed = deepcopy(plan)
    changed["signature"]["length_grouping_seed"] = 42
    with pytest.raises(FinetuneEstimateError, match="workload identity"):
        validate_transfer_plan(changed)


def test_grouping_lengths_cannot_be_pooled_across_native_workloads(native_run):
    run = v2_fixture(native_run, 1)
    target = plan_finetuning_transfer(run, max_steps=96)
    changed = v2_fixture(native_run, 2)
    grouping = changed["sampling_plan"]["sampler"]["grouping"]
    grouping["lengths"][1] = 3
    grouping["lengths_sha256"] = _digest(grouping["lengths"])
    reseal(changed["sampling_plan"])
    changed["observation"]["sampling_verification"]["plan_sha256"] = changed["sampling_plan"]["plan_sha256"]
    transfer_plan_from_run(changed)
    with pytest.raises(FinetuneEstimateError, match="execution signature"):
        estimate_finetuning_transfer([run, changed], target, setup_calibration_runs=[run])


def test_v2_signature_rejects_boolean_seed_even_when_equal_to_zero(native_run):
    run = v2_fixture(native_run, 1)
    run["sampling_plan"]["sampler"]["seed"] = 0
    reseal(run["sampling_plan"])
    run["observation"]["sampling_verification"]["plan_sha256"] = run["sampling_plan"]["plan_sha256"]
    run["training_config"]["length_grouping_seed"] = 0
    refresh_signature(run)
    plan = transfer_plan_from_run(run)
    plan["signature"]["length_grouping_seed"] = False
    with pytest.raises(FinetuneEstimateError, match="length_grouping_seed"):
        validate_transfer_plan(plan)


def test_v1_sampling_plan_cannot_claim_an_unapplied_length_seed(native_run):
    plan = transfer_plan_from_run(transfer_run(native_run, 1))
    plan["signature"]["length_grouping_seed"] = 123
    with pytest.raises(FinetuneEstimateError, match="workload identity"):
        validate_transfer_plan(plan)


def test_v1_sampling_plan_cannot_claim_grouped_evaluation(native_run):
    plan = transfer_plan_from_run(transfer_run(native_run, 1))
    plan["signature"]["length_grouping_evaluation"] = {}
    with pytest.raises(FinetuneEstimateError, match="v2 length sampler"):
        validate_transfer_plan(plan)


def test_evaluation_policy_is_validated_and_part_of_cost_identity(native_run):
    run = v2_fixture(native_run, 1)
    run["training_config"]["length_grouping_evaluation"] = {
        "schema_version": "fakegpu.finetune_eval_sampler.v1", "kind": "fixed_seed_length_grouped",
        "seed": 123, "iteration_seed_rule": "same_seed_each_invocation",
        "grouping_batch_size": 2, "dataset_rows": 7, "lengths_sha256": "a" * 64,
    }
    refresh_signature(run)
    plan = transfer_plan_from_run(run)
    assert plan["transfer_identity"]["signature"]["length_grouping_evaluation"] == run["training_config"]["length_grouping_evaluation"]
    plan["signature"]["length_grouping_evaluation"]["seed"] = 124
    with pytest.raises(FinetuneEstimateError, match="explicit seed"):
        validate_transfer_plan(plan)
