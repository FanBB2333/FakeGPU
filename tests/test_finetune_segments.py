"""Segment boundaries, isolation masks and loss boundaries are explicit contracts."""

import pytest

from fakegpu._finetune_segments import segment_ranges, validate_segment_inputs
from fakegpu.finetune_estimator import FinetuneEstimateError


@pytest.mark.parametrize("positions,active,expected", [
    ([0, 1, 2], [True] * 3, [(0, 3)]),
    ([0, 0, 1, 0], [True] * 4, [(0, 1), (1, 3), (3, 4)]),
    ([0, 1, 0, 0], [True, True, False, False], [(0, 2)]),
])
def test_segment_ranges(positions, active, expected):
    assert segment_ranges(positions, active) == expected


@pytest.mark.parametrize("positions,active", [
    ([], []), ([0], []), ([1, 2], [True, True]),
    ([0, 2], [True, True]), ([False], [True]), ([0], [1]),
    ([0, 0], [False, True]), ([0, 1], [True, False]), ([0], [False]),
])
def test_invalid_ranges(positions, active):
    with pytest.raises(FinetuneEstimateError):
        segment_ranges(positions, active)


@pytest.fixture
def packed_inputs():
    torch = pytest.importorskip("torch")
    allowed = torch.zeros(6, 6, dtype=torch.bool)
    allowed[:2, :2] = torch.ones(2, 2, dtype=torch.bool).tril()
    allowed[2:5, 2:5] = torch.ones(3, 3, dtype=torch.bool).tril()
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 0]]),
        "position_ids": torch.tensor([[0, 1, 0, 1, 2, 0]]),
        "attention_mask": torch.where(allowed, 0., torch.finfo(torch.float32).min)[None, None],
        "labels": torch.tensor([[-100, 2, -100, 4, 5, -100]]),
        "use_cache": False,
    }


def test_valid_mask_and_ignored_segment_starts(packed_inputs):
    assert validate_segment_inputs(packed_inputs) == [[(0, 2), (2, 5)]]


@pytest.mark.parametrize("mutation", [
    "cross_segment", "future", "padding", "nan", "weak_negative",
    "boundary_label", "padding_label", "cache", "past", "positions", "mask_shape",
])
def test_reject_semantic_mismatches(packed_inputs, mutation):
    torch = pytest.importorskip("torch")
    mask = packed_inputs["attention_mask"]
    if mutation == "cross_segment":
        mask[0, 0, 3, 0] = 0
    elif mutation == "future":
        mask[0, 0, 0, 1] = 0
    elif mutation == "padding":
        mask[0, 0, 0, 5] = 0
    elif mutation == "nan":
        mask[0, 0, 0, 1] = float("nan")
    elif mutation == "weak_negative":
        mask[0, 0, 0, 1] = -1
    elif mutation == "boundary_label":
        packed_inputs["labels"][0, 2] = 3
    elif mutation == "padding_label":
        packed_inputs["labels"][0, 5] = 0
    elif mutation == "cache":
        packed_inputs["use_cache"] = True
    elif mutation == "past":
        packed_inputs["past_key_values"] = object()
    elif mutation == "positions":
        packed_inputs["position_ids"] = packed_inputs["position_ids"].float()
    elif mutation == "mask_shape":
        packed_inputs["attention_mask"] = torch.ones(1, 6)
    with pytest.raises(FinetuneEstimateError):
        validate_segment_inputs(packed_inputs)
