"""Packed membership evidence keeps independent samples and tensor rows separate."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from fakegpu._finetune_packed_batches import (
    IsolatedPackingCollator,
    describe_packed_batch,
    packed_input_sha256,
    packed_source_membership,
    install_isolated_packing_collation,
)
from fakegpu.finetune_estimator import FinetuneEstimateError

torch = pytest.importorskip("torch")


def flat():
    return [
        {
            "input_ids": [1, 2, 3, 4, 5, 0],
            "labels": [-100, 2, -100, 4, 5, -100],
            "attention_mask": [1, 1, 2, 2, 2, 0],
            "position_ids": [0, 1, 0, 1, 2, 0],
        }
    ]


def native_flat(features):
    return {key: torch.tensor([row[key] for row in features]) for key in features[0]}


def collator(native=native_flat, framework="huggingface", dtype=torch.float32):
    return IsolatedPackingCollator(native, framework=framework, mask_dtype=dtype)


def test_flat_sources_are_unmodified_and_causal_boundaries_verified():
    source = flat()
    before = deepcopy(source)
    result = collator()(source)
    assert source == before
    assert result["attention_mask"].shape == (1, 1, 6, 6)
    assert result["position_ids"].tolist() == [[0, 1, 0, 1, 2, 0]]
    assert result["attention_mask"][0, 0, 3, 0].item() == torch.finfo(torch.float32).min
    descriptor = describe_packed_batch(result)
    assert descriptor["packed_rows"] == 1
    assert descriptor["logical_sequences"] == 2
    assert descriptor["segments"] == [[[0, 2], [2, 5]]]
    assert descriptor["nonpadding_tokens"] == 5 and descriptor["padding_tokens"] == 1
    assert (
        descriptor["supervised_tokens"] == descriptor["causal_supervised_tokens"] == 3
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_real_mask_bytes_and_dtype_are_hashed(dtype):
    result = collator(dtype=dtype)(flat())
    digest = packed_input_sha256(result)
    assert digest == packed_input_sha256(deepcopy(result))
    changed = deepcopy(result)
    changed["input_ids"][0, 2] += 1
    assert packed_input_sha256(changed) != digest
    if dtype != torch.float32:
        assert digest != packed_input_sha256(collator()(flat()))


@pytest.mark.parametrize(
    "change", ["tokens", "labels", "positions", "mask", "boundary_loss"]
)
def test_native_changes_cannot_be_hidden_by_mask_construction(change):
    def changed(features):
        result = native_flat(features)
        if change == "tokens":
            result["input_ids"][0, 0] += 1
        elif change == "labels":
            result["labels"][0, 1] += 1
        elif change == "positions":
            result["position_ids"] = torch.arange(6)[None]
        elif change == "mask":
            result["attention_mask"][0, 4] = 0
        return result

    source = flat()
    if change == "boundary_loss":
        source[0]["labels"][2] = 3
    with pytest.raises(FinetuneEstimateError):
        collator(changed)(source)


@pytest.mark.parametrize(
    "mask", [[1, 1, 3, 3, 3, 0], [1, 0, 2, 2, 2, 0], [2, 2, 3, 3, 3, 0]]
)
def test_invalid_source_segment_ids_rejected(mask):
    source = flat()
    source[0]["attention_mask"] = mask
    with pytest.raises(FinetuneEstimateError):
        packed_source_membership(source, framework="huggingface")


def test_axolotl_bins_prove_original_sample_membership():
    source = [
        [
            {"input_ids": [1, 2], "labels": [-100, 2], "attention_mask": [1, 1]},
            {
                "input_ids": [3, 4, 5],
                "labels": [-100, 4, 5],
                "attention_mask": [1, 1, 1],
            },
        ]
    ]

    def native(features):
        assert features == source
        result = native_flat(flat())
        result["attention_mask"] = torch.tensor([[1, 1, 1, 1, 1, 0]])
        return result

    result = collator(native, framework="axolotl")(source)
    assert describe_packed_batch(result)["logical_sequences"] == 2


def test_native_4d_mask_is_preserved_and_checked_against_source():
    valid = collator()(flat())

    def native(features):
        return deepcopy(valid)

    result = collator(native, framework="llamafactory")(flat())
    assert torch.equal(result["attention_mask"], valid["attention_mask"])
    valid["attention_mask"][0, 0, 3, 0] = 0
    with pytest.raises(FinetuneEstimateError, match="isolate"):
        collator(native, framework="llamafactory")(flat())


def test_inactive_native_positions_are_canonicalized_without_changing_active_tokens():
    def native(features):
        result = native_flat(features)
        result["position_ids"] = torch.tensor([[0, 1, 0, 1, 2, 17]])
        return result

    result = collator(native)(flat())
    assert result["position_ids"].tolist() == [[0, 1, 0, 1, 2, 0]]


def test_active_native_position_dtype_is_not_silently_coerced():
    def native(features):
        result = native_flat(features)
        result["position_ids"] = torch.tensor([[0.0, 1.0, 0.0, 1.0, 2.0, 0.0]])
        return result

    with pytest.raises(FinetuneEstimateError, match="position IDs"):
        collator(native)(flat())


def test_explicit_axolotl_collation_retains_masks_before_native_loader_and_restores():
    class Dataset:
        def __getitem__(self, index):
            return {"row": index}

        def __getitems__(self, indices):
            return [self[index] for index in indices]

    trainer = SimpleNamespace(
        data_collator=native_flat,
        args=SimpleNamespace(sample_packing_drop_attention_mask=True),
        train_dataset=Dataset(),
    )
    restores = []

    def replace(obj, name, value):
        restores.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    policy = install_isolated_packing_collation(
        trainer, framework="axolotl", mask_dtype=torch.float32, replace=replace
    )
    assert trainer.args.sample_packing_drop_attention_mask is False
    assert isinstance(trainer.data_collator, IsolatedPackingCollator)
    assert policy["source_masks_retained"] is True
    assert trainer.train_dataset.__getitems__([[0, 1], [2]]) == [
        [{"row": 0}, {"row": 1}],
        [{"row": 2}],
    ]
    assert trainer.train_dataset.__getitems__([0, 1]) == [{"row": 0}, {"row": 1}]
    with pytest.raises(FinetuneEstimateError, match="already installed"):
        install_isolated_packing_collation(
            trainer, framework="axolotl", mask_dtype=torch.float32, replace=replace
        )
    for obj, name, value in reversed(restores):
        setattr(obj, name, value)
    assert trainer.data_collator is native_flat
    assert trainer.args.sample_packing_drop_attention_mask is True
