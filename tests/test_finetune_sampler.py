from copy import deepcopy
from types import SimpleNamespace
import hashlib
import json
import random

import pytest

import fakegpu._finetune_sampler as sampler_module
from fakegpu._finetune_sampler import (
    NativeBatchMatcher,
    batch_input_sha256,
    plan_native_batches,
    validate_native_sampler_versions,
    validate_sampler_plan,
    validate_sampling_verification,
)
from fakegpu.finetune_data import profile_tokenized_records
from fakegpu.finetune_estimator import FinetuneEstimateError


torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
SeedableRandomSampler = pytest.importorskip(
    "accelerate.data_loader"
).SeedableRandomSampler


class Collator:
    def __call__(self, rows):
        width = (max(len(row["input_ids"]) for row in rows) + 7) // 8 * 8
        return {
            name: torch.tensor(
                [row[name] + [padding] * (width - len(row[name])) for row in rows]
            )
            for name, padding in (
                ("input_ids", 0),
                ("attention_mask", 0),
                ("labels", -100),
            )
        }


@pytest.fixture
def trainer(monkeypatch):
    versions = {
        "torch": torch.__version__,
        "transformers": "5.6.0",
        "accelerate": "1.11.0",
    }
    monkeypatch.setattr(
        sampler_module.importlib.metadata, "version", versions.__getitem__
    )
    data = []
    for index in range(13):
        ids = [index + 1] * (2 + index % 11)
        data.append(
            {
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
                "labels": [-100] + ids[1:],
            }
        )
    args = SimpleNamespace(
        world_size=1,
        n_gpu=0,
        dataloader_num_workers=0,
        dataloader_prefetch_factor=None,
        dataloader_persistent_workers=False,
        dataloader_pin_memory=False,
        dataloader_drop_last=False,
        gradient_accumulation_steps=2,
        per_device_train_batch_size=3,
        data_seed=42,
        seed=42,
    )
    result = SimpleNamespace(
        train_dataset=data,
        args=args,
        accelerator=SimpleNamespace(num_processes=1),
        data_collator=Collator(),
    )
    result.native_sampler = SeedableRandomSampler(data, data_seed=args.data_seed)
    result.get_train_dataloader = lambda: torch.utils.data.DataLoader(
        data,
        batch_size=3,
        sampler=result.native_sampler,
        collate_fn=result.data_collator,
        drop_last=args.dataloader_drop_last,
    )
    return result


def plan(trainer, *, steps=8):
    return plan_native_batches(
        trainer, max_steps=steps, framework="huggingface", sequence_length=64
    )


def reseal(value):
    value.pop("plan_sha256")
    value["plan_sha256"] = hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def inputs_for(trainer, batch):
    return trainer.data_collator(
        [deepcopy(trainer.train_dataset[index]) for index in batch["row_indices"]]
    )


def test_random_planning_preserves_streams_and_sampler_and_dataset(trainer):
    source = deepcopy(trainer.train_dataset)
    state = torch.get_rng_state().clone()
    python_state, numpy_state = random.getstate(), np.random.get_state()
    before_sampler = deepcopy(trainer.native_sampler.__dict__)
    result = plan(trainer)
    assert torch.equal(state, torch.get_rng_state())
    assert python_state == random.getstate()
    assert all(np.array_equal(a, b) for a, b in zip(numpy_state, np.random.get_state()))
    assert trainer.native_sampler.__dict__ == before_sampler
    assert trainer.train_dataset == source
    assert result["dataset"] == {
        "num_samples": 13,
        "content_sha256": profile_tokenized_records(source, sequence_length=64)[
            "provenance"
        ]["content_sha256"],
    }
    assert result["workload"] == {"optimizer_steps": 8, "microsteps": 14, "samples": 38}
    assert [len(update["batches"]) for update in result["updates"]] == [
        2,
        2,
        1,
        2,
        2,
        1,
        2,
        2,
    ]
    for epoch in range(3):
        expected = torch.randperm(
            13, generator=torch.Generator().manual_seed(42 + epoch)
        ).tolist()
        actual = [
            index
            for update in result["updates"]
            for batch in update["batches"]
            if batch["epoch_index"] == epoch + 1
            for index in batch["row_indices"]
        ]
        assert actual == expected[: len(actual)]


def test_sequential_tail_and_partial_epoch(trainer):
    trainer.native_sampler = torch.utils.data.SequentialSampler(trainer.train_dataset)
    result = plan(trainer, steps=4)
    assert result["sampler"]["kind"] == "sequential"
    assert result["sampler"]["seed"] is None
    assert result["workload"] == {"optimizer_steps": 4, "microsteps": 7, "samples": 19}
    assert [
        batch["row_indices"]
        for update in result["updates"]
        for batch in update["batches"]
    ] == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
        [12],
        [0, 1, 2],
        [3, 4, 5],
    ]


def test_matcher_attests_actual_prefix_and_serialized_evidence(trainer):
    result = plan(trainer)
    matcher = NativeBatchMatcher(result)
    observed = []
    for update in result["updates"][:4]:
        batches = []
        for batch in update["batches"]:
            inputs = inputs_for(trainer, batch)
            batches.append(
                matcher.observe(
                    inputs,
                    epoch_index=batch["epoch_index"],
                    epoch_microstep=batch["epoch_microstep"],
                )
            )
        observed.append(
            {
                "optimizer_step": update["optimizer_step"],
                "batches": batches,
                "seconds": 1.0,
            }
        )
    verification = matcher.finish(expected_updates=4)
    assert verification["matched"]
    assert verification["samples"] == 19
    assert (
        validate_sampling_verification(result, observed, verification) == verification
    )
    observed[0]["batches"][0]["input_sha256"] = "1" * 64
    with pytest.raises(FinetuneEstimateError, match="batch differs"):
        validate_sampling_verification(result, observed, verification)


def test_matcher_detects_same_shape_wrong_tokens_and_incomplete_epoch(trainer):
    result = plan(trainer)
    matcher = NativeBatchMatcher(result)
    batch = result["updates"][0]["batches"][0]
    inputs = inputs_for(trainer, batch)
    inputs["input_ids"][0, 0] += 1
    with pytest.raises(FinetuneEstimateError, match="tensor contents"):
        matcher.observe(inputs, epoch_index=1, epoch_microstep=1)
    with pytest.raises(FinetuneEstimateError, match="complete planned prefix"):
        matcher.finish(expected_updates=1)


def test_hash_binds_dtype_masks_labels_positions_and_field_presence():
    values = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.tensor([[1, 1]]),
        "labels": torch.tensor([[-100, 2]]),
    }
    original = batch_input_sha256(values)
    changes = [
        {**values, "input_ids": values["input_ids"].to(torch.int32)},
        {**values, "attention_mask": torch.tensor([[1, 0]])},
        {**values, "labels": torch.tensor([[1, 2]])},
        {**values, "position_ids": torch.tensor([[0, 1]])},
        {key: value for key, value in values.items() if key != "labels"},
    ]
    assert all(batch_input_sha256(change) != original for change in changes)
    assert batch_input_sha256({**values, "position_ids": None}) == original
    with pytest.raises(FinetuneEstimateError, match="supports only"):
        batch_input_sha256({**values, "token_type_ids": torch.zeros(1, 2)})


@pytest.mark.parametrize(
    "option,value,message",
    [
        ("world_size", 2, "one device"),
        ("dataloader_num_workers", 1, "deterministic text collator"),
        ("sample_packing", True, "packing"),
        ("auto_find_batch_size", True, "adaptive batch"),
        ("dataloader_drop_last", True, "retain tail"),
        ("data_seed", 99, "differs from configured"),
    ],
)
def test_unsupported_execution_fails_before_claiming_a_plan(
    trainer, option, value, message
):
    setattr(trainer.args, option, value)
    with pytest.raises(FinetuneEstimateError, match=message):
        plan(trainer)


def test_plain_random_and_custom_samplers_are_rejected(trainer):
    trainer.native_sampler = torch.utils.data.RandomSampler(trainer.train_dataset)
    with pytest.raises(FinetuneEstimateError, match="seedable random"):
        plan(trainer)


def test_stochastic_collator_is_rejected_and_rng_restored(trainer):
    base = trainer.data_collator
    state = torch.get_rng_state().clone()

    def stochastic(rows):
        values = base(rows)
        values["input_ids"][0, 0] = torch.randint(0, 2**31, ()).item()
        return values

    trainer.data_collator = stochastic
    with pytest.raises(FinetuneEstimateError, match="deterministic collation"):
        plan(trainer)
    assert torch.equal(state, torch.get_rng_state())


def test_serialized_plan_checks_content_digest_and_semantics(trainer):
    result = plan(trainer)
    result["updates"][0]["batches"][0]["row_indices"][0] = 12
    with pytest.raises(FinetuneEstimateError, match="SHA256 differs"):
        validate_sampler_plan(result)
    reseal(result)
    with pytest.raises(FinetuneEstimateError, match="repeats|SHA256 is inconsistent"):
        validate_sampler_plan(result)


def test_version_gate_checks_accelerate_as_well_as_transformers():
    with pytest.raises(FinetuneEstimateError, match="validated framework"):
        validate_native_sampler_versions(
            "huggingface",
            {"torch": "2.9.1", "transformers": "5.6.0", "accelerate": "1.13.0"},
        )


def test_empty_data_and_invalid_duration(trainer):
    with pytest.raises(FinetuneEstimateError, match="max_steps"):
        plan(trainer, steps=True)
    trainer.train_dataset.clear()
    with pytest.raises(FinetuneEstimateError, match="dataset rows"):
        plan(trainer)
