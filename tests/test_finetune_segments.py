"""Segment boundaries, isolation masks and loss boundaries are explicit contracts."""

import pytest

from fakegpu._finetune_segments import segment_ranges, validate_segment_inputs
from fakegpu.finetune_estimator import FinetuneEstimateError


@pytest.mark.parametrize(
    "positions,active,expected",
    [
        ([0, 1, 2], [True] * 3, [(0, 3)]),
        ([0, 0, 1, 0], [True] * 4, [(0, 1), (1, 3), (3, 4)]),
        ([0, 1, 0, 0], [True, True, False, False], [(0, 2)]),
    ],
)
def test_segment_ranges(positions, active, expected):
    assert segment_ranges(positions, active) == expected


@pytest.mark.parametrize(
    "positions,active",
    [
        ([], []),
        ([0], []),
        ([1, 2], [True, True]),
        ([0, 2], [True, True]),
        ([False], [True]),
        ([0], [1]),
        ([0, 0], [False, True]),
        ([0, 1], [True, False]),
        ([0], [False]),
    ],
)
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
        "attention_mask": torch.where(allowed, 0.0, torch.finfo(torch.float32).min)[
            None, None
        ],
        "labels": torch.tensor([[-100, 2, -100, 4, 5, -100]]),
        "use_cache": False,
    }


def test_valid_mask_and_ignored_segment_starts(packed_inputs):
    assert validate_segment_inputs(packed_inputs) == [[(0, 2), (2, 5)]]


@pytest.mark.parametrize(
    "mutation",
    [
        "cross_segment",
        "future",
        "padding",
        "nan",
        "weak_negative",
        "boundary_label",
        "padding_label",
        "cache",
        "past",
        "positions",
        "mask_shape",
    ],
)
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


def test_direct_forward_call_validates_and_restores_instance(
    packed_inputs, monkeypatch
):
    """PEFT calls the base .forward() without running Module pre-hooks."""
    from types import SimpleNamespace
    import sys
    import torch
    import fakegpu._finetune_segments as segments

    class Decoder(torch.nn.Module):
        def forward(self, **kwargs):
            return kwargs

    class Native(torch.nn.Module):
        def __init__(self):
            super().__init__()
            decoder = Decoder()
            decoder.linear_attn = Decoder()
            self.model = SimpleNamespace(layers=[decoder])
            self.config = SimpleNamespace(layer_types=["linear_attention"])

        def forward(self, **kwargs):
            return kwargs["fakegpu_segment_active"]

    monkeypatch.setitem(
        sys.modules,
        "transformers.models.qwen3_5.modeling_qwen3_5",
        SimpleNamespace(Qwen3_5DecoderLayer=Decoder, Qwen3_5ForCausalLM=Native),
    )
    monkeypatch.setattr(segments.importlib.metadata, "version", lambda _: "5.6.0")
    model = Native()
    original = model.forward.__func__
    with segments.isolated_qwen35_segments(model):
        assert model.forward(**packed_inputs).tolist() == [[True] * 5 + [False]]
        with pytest.raises(ValueError, match="use_cache=False"):
            model.forward(**{**packed_inputs, "use_cache": True})
    assert "forward" not in model.__dict__ and model.forward.__func__ is original
    assert not model._forward_pre_hooks
    with pytest.raises(RuntimeError, match="test failure"):
        with segments.isolated_qwen35_segments(model):
            raise RuntimeError("test failure")
    assert "forward" not in model.__dict__ and model.forward.__func__ is original


def test_equal_length_segments_share_native_batch_without_padding():
    import torch
    from types import SimpleNamespace
    from fakegpu._finetune_segments import _linear_decoder_forward

    calls = []

    def native(*, hidden_states, cache_params, attention_mask):
        assert cache_params is attention_mask is None
        calls.append(tuple(hidden_states.shape))
        return hidden_states * 2

    layer = SimpleNamespace(
        input_layernorm=lambda x: x,
        linear_attn=native,
        post_attention_layernorm=lambda x: x,
        mlp=torch.zeros_like,
    )
    hidden = torch.arange(16.0).reshape(2, 4, 2).requires_grad_(True)
    positions = torch.tensor([[0, 1, 0, 1], [0, 1, 0, 0]])
    active = torch.tensor([[True] * 4, [True, True, False, False]])
    output = _linear_decoder_forward(
        layer,
        hidden,
        None,
        position_ids=positions,
        use_cache=False,
        fakegpu_segment_active=active,
    )
    assert calls == [(3, 2, 2)]
    expected = torch.where(active[:, :, None], hidden * 3, hidden)
    assert torch.equal(output, expected)
    output.sum().backward()
    assert torch.equal(
        hidden.grad, torch.where(active[:, :, None], 3.0, 1.0).expand_as(hidden)
    )
