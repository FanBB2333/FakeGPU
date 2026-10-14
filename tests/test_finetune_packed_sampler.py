"""Packed epochs attest source rows separately from executed tensor rows."""

from types import SimpleNamespace

import pytest

from fakegpu._finetune_packed_batches import IsolatedPackingCollator
from fakegpu._finetune_packed_sampler import (
    install_packed_random_sampling,
    inspect_packed_epochs,
)
from fakegpu.finetune_estimator import FinetuneEstimateError

torch = pytest.importorskip("torch")
pytest.importorskip("accelerate")


@pytest.fixture
def trainer(monkeypatch):
    monkeypatch.setattr(
        "fakegpu._finetune_packed_sampler.validate_native_sampler_versions",
        lambda framework, versions=None: (
            {"test": "synthetic"} if versions is None else versions
        ),
    )
    dataset = [
        {
            "input_ids": [index + 1] * 6,
            "attention_mask": [1, 1, 2, 2, 2, 2],
            "labels": [-100, index + 1, -100, index + 1, index + 1, index + 1],
            "position_ids": [0, 1, 0, 1, 2, 3],
        }
        for index in range(7)
    ]

    def native(features):
        return {
            key: torch.tensor(
                [
                    row[key] + ([-100] * 2 if key == "labels" else [0] * 2)
                    for row in features
                ]
            )
            for key in ("input_ids", "attention_mask", "labels")
        }

    instance = SimpleNamespace(
        train_dataset=dataset,
        accelerator=SimpleNamespace(
            num_processes=1, dataloader_config=SimpleNamespace(data_seed=42)
        ),
        args=SimpleNamespace(
            world_size=1,
            n_gpu=0,
            gradient_accumulation_steps=2,
            per_device_train_batch_size=3,
            dataloader_drop_last=False,
            dataloader_num_workers=0,
            dataloader_prefetch_factor=None,
            dataloader_persistent_workers=False,
            dataloader_pin_memory=False,
        ),
    )
    instance.data_collator = IsolatedPackingCollator(
        native, framework="huggingface", mask_dtype=torch.float32
    )
    instance._get_train_sampler = lambda dataset=None: torch.utils.data.RandomSampler(
        instance.train_dataset if dataset is None else dataset
    )

    def loader():
        from accelerate.data_loader import prepare_data_loader

        if not hasattr(instance, "_train_dataloader"):
            native_loader = torch.utils.data.DataLoader(
                instance.train_dataset,
                batch_sampler=torch.utils.data.BatchSampler(
                    instance._get_train_sampler(), 3, instance.args.dataloader_drop_last
                ),
                collate_fn=instance.data_collator,
            )
            instance._train_dataloader = prepare_data_loader(
                native_loader,
                device=torch.device("cpu"),
                num_processes=1,
                use_seedable_sampler=True,
                data_seed=instance.accelerator.dataloader_config.data_seed,
            )
        return instance._train_dataloader

    instance.get_train_dataloader = loader
    return instance


def planned(trainer, seed=123):
    install_packed_random_sampling(
        trainer, framework="huggingface", seed=seed, replace=setattr
    )
    return inspect_packed_epochs(trainer, framework="huggingface", epochs=3)


def test_epoch_counts_keep_logical_sequences_and_real_tails(trainer):
    state = torch.get_rng_state().clone()
    plan = planned(trainer)
    assert torch.equal(torch.get_rng_state(), state)
    assert trainer._train_dataloader.batch_sampler.sampler.epoch == 0
    assert plan == inspect_packed_epochs(trainer, framework="huggingface", epochs=3)
    assert plan["native_declared_microsteps_per_epoch"] == 3
    for epoch, row in enumerate(plan["epochs"]):
        assert row["microsteps"] == 3 and row["optimizer_updates"] == 2
        assert row["source_dataset_rows"] == row["packed_rows"] == 7
        assert row["logical_sequences"] == 14
        assert row["dropped_dataset_row_indices"] == []
        indices = [
            index
            for batch in row["batches"]
            for group in batch["source_bins"]
            for index in group
        ]
        expected = torch.randperm(
            7, generator=torch.Generator().manual_seed(123 + epoch)
        ).tolist()
        assert indices == expected
        assert row["batches"][-1]["shape"][0] == 1


def test_drop_last_is_explicit_and_not_counted_as_consumed_data(trainer):
    trainer.args.dataloader_drop_last = True
    plan = planned(trainer)
    for row in plan["epochs"]:
        assert row["microsteps"] == 2 and row["optimizer_updates"] == 1
        assert row["source_dataset_rows"] == row["packed_rows"] == 6
        assert row["logical_sequences"] == 12
        assert len(row["dropped_dataset_row_indices"]) == 1


def test_unseeded_random_packing_cannot_claim_future_order(trainer):
    with pytest.raises(FinetuneEstimateError, match="independent packing seed"):
        inspect_packed_epochs(trainer, framework="huggingface", epochs=1)


def test_prepared_loader_cannot_be_reseeded(trainer):
    trainer.get_train_dataloader()
    with pytest.raises(FinetuneEstimateError, match="before loader"):
        install_packed_random_sampling(
            trainer, framework="huggingface", seed=123, replace=setattr
        )


def test_repeated_native_pad_changes_fail_and_restore_rng(trainer):
    original = trainer.data_collator.native_collator
    counter = [0]

    def stochastic(features):
        result = original(features)
        counter[0] += 1
        torch.rand(1)
        result["input_ids"][0, -1] = counter[0]
        return result

    trainer.data_collator.native_collator = stochastic
    state = torch.get_rng_state().clone()
    with pytest.raises(FinetuneEstimateError, match="deterministic"):
        planned(trainer)
    assert torch.equal(torch.get_rng_state(), state)
