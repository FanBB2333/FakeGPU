"""Opt-in Qwen3.5 segment isolation using the installed, unmodified kernels.

Each linear-attention segment starts with no convolution or recurrent cache.
Full attention keeps the caller's validated block-causal mask. The per-instance
overrides live for the whole forward/backward scope, including recomputation;
model parameters and serialized checkpoints keep their native representation.
This reference path does not claim varlen-kernel throughput or packing budgets.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib.metadata
import inspect
from types import MethodType

from .finetune_estimator import FinetuneEstimateError


def segment_ranges(positions, active):
    """Validate reset positions and right padding; return nonempty [start, end)."""
    if not positions or len(positions) != len(active):
        raise FinetuneEstimateError("segment positions and active mask must align")
    spans, start, expected, padded = [], None, 0, False
    for index, (position, enabled) in enumerate(zip(positions, active)):
        if isinstance(position, bool) or not isinstance(position, int) or position < 0:
            raise FinetuneEstimateError("segment positions must be nonnegative integers")
        if not isinstance(enabled, bool):
            raise FinetuneEstimateError("segment active mask must contain booleans")
        if not enabled:
            padded = True
            if position != 0:
                raise FinetuneEstimateError("segment padding positions must be zero")
            if start is not None:
                spans.append((start, index))
                start = None
            continue
        if padded:
            raise FinetuneEstimateError("segment execution requires right padding")
        if position == 0:
            if start is not None:
                spans.append((start, index))
            start, expected = index, 0
        if start is None or position != expected:
            raise FinetuneEstimateError("each segment must start at zero and increment by one")
        expected += 1
    if start is not None:
        spans.append((start, len(positions)))
    if not spans:
        raise FinetuneEstimateError("segment execution rejects empty rows")
    return spans


def validate_segment_inputs(inputs):
    """Validate actual tensors, attention isolation and causal label boundaries."""
    import torch

    tokens = inputs.get("input_ids")
    if tokens is None:
        tokens = inputs.get("inputs_embeds")
    if tokens is None or tokens.ndim not in (2, 3):
        raise FinetuneEstimateError("segment execution requires text inputs")
    shape = tuple(tokens.shape[:2])
    positions, mask = inputs.get("position_ids"), inputs.get("attention_mask")
    if positions is None or tuple(positions.shape) != shape or positions.dtype not in (
        torch.int32, torch.int64
    ):
        raise FinetuneEstimateError("segment execution requires explicit integer 2D positions")
    if mask is None or tuple(mask.shape) != (shape[0], 1, shape[1], shape[1]):
        raise FinetuneEstimateError("segment execution requires an explicit block-causal 4D mask")
    if not mask.dtype.is_floating_point:
        raise FinetuneEstimateError("segment attention mask must be floating point")
    if inputs.get("past_key_values") is not None or inputs.get("use_cache") is not False:
        raise FinetuneEstimateError("segment execution requires use_cache=False and no past cache")
    masks = mask.detach().cpu()
    position_rows = positions.detach().cpu().tolist()
    labels = inputs.get("labels")
    if labels is not None and (tuple(labels.shape) != shape or labels.dtype not in (
        torch.int32, torch.int64
    )):
        raise FinetuneEstimateError("segment labels must be integer 2D tensors")
    label_rows = labels.detach().cpu().tolist() if labels is not None else None
    all_spans = []
    for row, position_row in enumerate(position_rows):
        row_mask = masks[row, 0]
        active = (row_mask.diagonal() == 0).tolist()
        spans = segment_ranges(position_row, active)
        expected = torch.zeros_like(row_mask, dtype=torch.bool)
        for start, end in spans:
            expected[start:end, start:end] = torch.ones(
                end - start, end - start, dtype=torch.bool
            ).tril()
            if start and label_rows is not None and label_rows[row][start] != -100:
                raise FinetuneEstimateError("causal labels must ignore every segment's first token")
        blocked = (row_mask == torch.finfo(mask.dtype).min) | torch.isneginf(row_mask)
        if not torch.equal(row_mask == 0, expected) or not bool(blocked[~expected].all()):
            raise FinetuneEstimateError("segment mask must isolate every segment and padding")
        if label_rows is not None and any(
            label != -100 for label, enabled in zip(label_rows[row], active) if not enabled
        ):
            raise FinetuneEstimateError("segment padding labels must be ignored")
        all_spans.append(spans)
    return all_spans


def _linear_decoder_forward(self, hidden_states, position_embeddings,
                            attention_mask=None, position_ids=None,
                            past_key_values=None, **kwargs):
    import torch

    if past_key_values is not None or kwargs.get("use_cache") is not False:
        raise FinetuneEstimateError("segmented linear attention cannot use a cache")
    if position_ids is None or tuple(position_ids.shape) != tuple(hidden_states.shape[:2]):
        raise FinetuneEstimateError("segmented decoder requires text position IDs")
    residual = hidden_states
    normalized = self.input_layernorm(hidden_states)
    active_rows = kwargs["fakegpu_segment_active"].detach().cpu().tolist()
    rows = []
    for row, positions in enumerate(position_ids.detach().cpu().tolist()):
        # TF 5.14 drops a custom 4D mask from its recurrent mask. Pass the
        # validated active rows explicitly through native decoder kwargs, which
        # also preserves them during gradient checkpoint recomputation.
        spans = segment_ranges(positions, active_rows[row])
        pieces = [self.linear_attn(
            hidden_states=normalized[row:row + 1, start:end],
            cache_params=None, attention_mask=None,
        ) for start, end in spans]
        end = spans[-1][1]
        if end < len(positions):
            pieces.append(torch.zeros_like(normalized[row:row + 1, end:]))
        rows.append(torch.cat(pieces, dim=1))
    mixed = residual + torch.cat(rows, dim=0)
    return mixed + self.mlp(self.post_attention_layernorm(mixed))


@contextmanager
def isolated_qwen35_segments(model):
    """Temporarily isolate a native dense text model, restoring on every exit.

    This explicit training-only reference interface supports Transformers 5.6.0
    and 5.14.1. It rejects global framework packing patches and nested installs.
    A fresh native model needs a fresh installation after loading a checkpoint.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DecoderLayer, Qwen3_5ForCausalLM,
    )

    version = importlib.metadata.version("transformers")
    if version not in {"5.6.0", "5.14.1"} or type(model) is not Qwen3_5ForCausalLM:
        raise FinetuneEstimateError("segment reference requires a supported native Qwen3.5 text model")
    layers = list(model.model.layers)
    linear = []
    for layer, kind in zip(layers, model.config.layer_types):
        if type(layer) is not Qwen3_5DecoderLayer or "forward" in layer.__dict__:
            raise FinetuneEstimateError("segment reference requires unmodified native decoder instances")
        if layer.forward.__func__.__module__ != Qwen3_5DecoderLayer.__module__:
            raise FinetuneEstimateError("segment reference rejects global decoder patches")
        if kind == "linear_attention":
            native = layer.linear_attn.forward.__func__
            if version == "5.14.1" and native.__module__ == "transformers.integrations.accelerate":
                native = inspect.getclosurevars(native).nonlocals.get("forward_func")
            if native is None or native.__module__ != Qwen3_5DecoderLayer.__module__:
                raise FinetuneEstimateError("segment reference rejects global linear-attention patches")
            linear.append(layer)
    if len(layers) != len(model.config.layer_types) or not linear:
        raise FinetuneEstimateError("segment reference requires a complete hybrid decoder")

    def validate(module, args, kwargs):
        if "fakegpu_segment_active" in kwargs:
            raise FinetuneEstimateError("segment metadata is supplied only by the validated interface")
        original_kwargs = kwargs
        if args:
            if len(args) != 1 or "input_ids" in kwargs:
                raise FinetuneEstimateError("segment reference accepts only positional input_ids")
            kwargs = {**kwargs, "input_ids": args[0]}
        validate_segment_inputs(kwargs)
        active = kwargs["attention_mask"][:, 0].diagonal(dim1=-2, dim2=-1) == 0
        return args, {**original_kwargs, "fakegpu_segment_active": active}

    hook = model.register_forward_pre_hook(validate, with_kwargs=True)
    installed = []
    try:
        for layer in linear:
            layer.forward = MethodType(_linear_decoder_forward, layer)
            installed.append(layer)
        yield model
    finally:
        for layer in installed:
            del layer.forward
        hook.remove()
