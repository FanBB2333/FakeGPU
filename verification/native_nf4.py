"""Small PyTorch-only NF4 weight storage used by QLoRA memory validation.

This module is deliberately not a replacement for bitsandbytes.  It stores two
4-bit codes per byte and dequantizes one linear weight at a time for ordinary
PyTorch matmul.  Block scales are FP32, or optionally nested-quantized to 8-bit
with an FP32 scale every 256 values and a mean offset.  A custom autograd
function recomputes the frozen weight during backward instead of retaining
every dequantized matrix.  That makes the allocator behavior useful for
validating FakeGPU's quantized-training memory model when no external 4-bit
backend is installed.
"""

from __future__ import annotations

import math
import time
from typing import Any

import torch
import torch.nn.functional as F


NF4_CODEBOOK = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)


def nf4_tensor_layout(
    numel: int,
    *,
    block_size: int = 64,
    compute_element_size: int = 2,
    double_quantization: bool = False,
    scale_block_size: int = 256,
) -> dict[str, int | float]:
    """Return the exact storage and implementation workspace for one tensor."""

    if numel <= 0:
        raise ValueError("numel must be greater than zero")
    if block_size <= 0 or block_size % 2:
        raise ValueError("block_size must be a positive even integer")
    if compute_element_size <= 0:
        raise ValueError("compute_element_size must be greater than zero")
    if scale_block_size <= 0:
        raise ValueError("scale_block_size must be greater than zero")

    block_count = (numel + block_size - 1) // block_size
    padded_numel = block_count * block_size
    packed_bytes = padded_numel // 2
    second_level_block_count = 0
    second_level_scale_bytes = 0
    scale_offset_bytes = 0
    scale_lookup_bytes = 0
    scale_dequantization_workspace_bytes = 0
    if double_quantization:
        scale_bytes = block_count
        second_level_block_count = (
            block_count + scale_block_size - 1
        ) // scale_block_size
        second_level_scale_bytes = second_level_block_count * 4
        scale_offset_bytes = 4
        scale_lookup_bytes = 256 * 4
        scale_index_bytes = block_count * 4
        scale_value_bytes = block_count * 4
        expanded_second_level_bytes = (
            second_level_block_count * scale_block_size * 4
        )
        scale_dequantization_workspace_bytes = max(
            scale_index_bytes + scale_value_bytes,
            scale_value_bytes + expanded_second_level_bytes,
        )
    else:
        # bitsandbytes emits FP32 absmax values for the first quantization.
        scale_bytes = block_count * 4
    # The PyTorch implementation uses a byte -> two NF4 values lookup table.
    lookup_bytes = 256 * 2 * compute_element_size
    storage_bytes = (
        packed_bytes
        + scale_bytes
        + second_level_scale_bytes
        + scale_offset_bytes
        + scale_lookup_bytes
        + lookup_bytes
    )
    int32_index_bytes = packed_bytes * 4
    unpacked_value_bytes = padded_numel * compute_element_size
    dequantized_weight_bytes = padded_numel * compute_element_size
    reconstructed_scale_bytes = block_count * 4 if double_quantization else 0
    compute_scale_bytes = block_count * compute_element_size
    dequantization_workspace_bytes = max(
        scale_dequantization_workspace_bytes,
        compute_scale_bytes + int32_index_bytes + unpacked_value_bytes,
        compute_scale_bytes + unpacked_value_bytes + dequantized_weight_bytes,
        reconstructed_scale_bytes + compute_scale_bytes,
    )
    return {
        "logical_numel": int(numel),
        "block_size": int(block_size),
        "block_count": int(block_count),
        "padded_numel": int(padded_numel),
        "packed_weight_bytes": int(packed_bytes),
        "scale_bytes": int(scale_bytes),
        "scale_dtype_bytes": 1 if double_quantization else 4,
        "scale_block_size": int(scale_block_size),
        "second_level_block_count": int(second_level_block_count),
        "second_level_scale_bytes": int(second_level_scale_bytes),
        "scale_offset_bytes": int(scale_offset_bytes),
        "scale_lookup_bytes": int(scale_lookup_bytes),
        "lookup_bytes": int(lookup_bytes),
        "storage_bytes": int(storage_bytes),
        "scale_dequantization_workspace_bytes": int(
            scale_dequantization_workspace_bytes
        ),
        "dequantization_workspace_bytes": int(dequantization_workspace_bytes),
        "effective_bits_per_weight": float(8.0 * storage_bytes / numel),
    }


def plan_nf4_lora(
    model: torch.nn.Module,
    *,
    rank: int,
    block_size: int = 64,
    double_quantization: bool = False,
    scale_block_size: int = 256,
    target_modules: str = "all-linear",
) -> dict[str, Any]:
    """Describe all frozen linear weights and LoRA adapters before conversion."""

    if rank <= 0:
        raise ValueError("rank must be greater than zero")
    if target_modules != "all-linear":
        raise ValueError("native NF4 validation currently requires target_modules='all-linear'")

    output_embedding = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    modules: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or module is output_embedding:
            continue
        weight = module.weight
        layout = nf4_tensor_layout(
            int(weight.numel()),
            block_size=block_size,
            compute_element_size=int(weight.element_size()),
            double_quantization=double_quantization,
            scale_block_size=scale_block_size,
        )
        modules.append(
            {
                "name": name,
                "in_features": int(module.in_features),
                "out_features": int(module.out_features),
                "weight_numel": int(weight.numel()),
                "original_weight_bytes": int(weight.numel()) * int(weight.element_size()),
                "bias_numel": int(module.bias.numel()) if module.bias is not None else 0,
                "bias_bytes": (
                    int(module.bias.numel()) * int(module.bias.element_size())
                    if module.bias is not None
                    else 0
                ),
                "lora_parameter_count": rank
                * (int(module.in_features) + int(module.out_features)),
                **layout,
            }
        )

    if not modules:
        raise ValueError("the model does not contain target linear modules")
    original_weight_bytes = sum(int(item["original_weight_bytes"]) for item in modules)
    storage_bytes = sum(int(item["storage_bytes"]) for item in modules)
    logical_numel = sum(int(item["logical_numel"]) for item in modules)
    return {
        "backend": "pytorch_native_nf4",
        "format": "nf4_blockwise",
        "double_quantization": bool(double_quantization),
        "block_size": int(block_size),
        "scale_block_size": int(scale_block_size),
        "scale_storage_dtype": "uint8" if double_quantization else "float32",
        "module_count": len(modules),
        "quantized_weight_count": logical_numel,
        "original_parameter_count": sum(
            int(item["weight_numel"]) + int(item["bias_numel"]) for item in modules
        ),
        "original_parameter_bytes": original_weight_bytes
        + sum(int(item["bias_bytes"]) for item in modules),
        "original_weight_bytes": original_weight_bytes,
        "packed_weight_bytes": sum(int(item["packed_weight_bytes"]) for item in modules),
        "scale_bytes": sum(int(item["scale_bytes"]) for item in modules),
        "second_level_scale_bytes": sum(
            int(item["second_level_scale_bytes"]) for item in modules
        ),
        "scale_offset_bytes": sum(
            int(item["scale_offset_bytes"]) for item in modules
        ),
        "scale_lookup_bytes": sum(
            int(item["scale_lookup_bytes"]) for item in modules
        ),
        "lookup_bytes": sum(int(item["lookup_bytes"]) for item in modules),
        "quantized_storage_bytes": storage_bytes,
        "storage_delta_bytes": storage_bytes - original_weight_bytes,
        "effective_bits_per_weight": float(8.0 * storage_bytes / logical_numel),
        "largest_dequantization_workspace_bytes": max(
            int(item["dequantization_workspace_bytes"]) for item in modules
        ),
        "lora_parameter_count": sum(int(item["lora_parameter_count"]) for item in modules),
        "adapter_dtype": "float32",
        "modules": modules,
        "limitations": [
            "PyTorch reference kernels are used instead of fused bitsandbytes CUDA kernels.",
            *(
                [
                    "Nested scales reproduce the bitsandbytes dynamic 8-bit map with PyTorch searchsorted rather than its fused kernels."
                ]
                if double_quantization
                else ["Double quantization is not applied."]
            ),
        ],
    }


def _byte_lookup(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    codebook = torch.tensor(NF4_CODEBOOK, dtype=dtype, device=device)
    values = torch.arange(256, dtype=torch.int64, device=device)
    low = torch.bitwise_and(values, 0x0F)
    high = torch.bitwise_right_shift(values, 4)
    return torch.stack((codebook[low], codebook[high]), dim=1)


def _dynamic_scale_lookup(device: torch.device) -> torch.Tensor:
    """Reproduce bitsandbytes' default signed 8-bit dynamic map."""

    values: list[float] = []
    max_exponent_bits = 7
    non_sign_bits = 7
    for exponent in range(max_exponent_bits):
        fraction_items = 2 ** (
            exponent + non_sign_bits - max_exponent_bits
        ) + 1
        boundaries = torch.linspace(0.1, 1.0, fraction_items, dtype=torch.float32)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        magnitude = 10 ** (-(max_exponent_bits - 1) + exponent)
        values.extend((magnitude * means).tolist())
        values.extend((-magnitude * means).tolist())
    values.extend((0.0, 1.0))
    if len(values) != 256:
        raise AssertionError("dynamic 8-bit scale map must contain 256 values")
    return torch.tensor(sorted(values), dtype=torch.float32, device=device)


def _quantize_scales(
    scales: torch.Tensor,
    *,
    scale_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    offset = scales.mean().to(torch.float32)
    centered = scales - offset
    block_count = (int(centered.numel()) + scale_block_size - 1) // scale_block_size
    padded_count = block_count * scale_block_size
    if padded_count != int(centered.numel()):
        centered = F.pad(centered, (0, padded_count - int(centered.numel())))
    blocks = centered.view(block_count, scale_block_size)
    absmax = blocks.abs().amax(dim=1).clamp_min(torch.finfo(torch.float32).tiny)
    normalized = blocks / absmax.unsqueeze(1)
    lookup = _dynamic_scale_lookup(scales.device)
    flat_normalized = normalized.reshape(-1)
    upper = torch.searchsorted(lookup, flat_normalized).clamp_(max=255)
    lower = (upper - 1).clamp_(min=0)
    use_upper = (lookup[upper] - flat_normalized).abs() < (
        lookup[lower] - flat_normalized
    ).abs()
    codes = torch.where(use_upper, upper, lower).to(torch.uint8)[: scales.numel()].clone()
    return codes, absmax, offset, lookup.unsqueeze(1)


def _quantize_weight(
    weight: torch.Tensor,
    *,
    block_size: int,
    chunk_blocks: int,
    double_quantization: bool,
    scale_block_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if weight.device.type != "cpu":
        raise ValueError("native NF4 conversion expects CPU checkpoint weights")
    if chunk_blocks <= 0:
        raise ValueError("chunk_blocks must be greater than zero")

    source_dtype = weight.dtype
    flat = weight.detach().to(dtype=torch.float32).contiguous().view(-1)
    block_count = (int(flat.numel()) + block_size - 1) // block_size
    padded_numel = block_count * block_size
    if padded_numel != int(flat.numel()):
        flat = F.pad(flat, (0, padded_numel - int(flat.numel())))
    blocks = flat.view(block_count, block_size)
    codes = torch.empty(padded_numel, dtype=torch.uint8)
    scales = torch.empty(block_count, dtype=torch.float32)
    codebook = torch.tensor(NF4_CODEBOOK, dtype=torch.float32)
    tiny = torch.finfo(torch.float32).tiny
    for first in range(0, block_count, chunk_blocks):
        last = min(block_count, first + chunk_blocks)
        chunk = blocks[first:last]
        absmax = chunk.abs().amax(dim=1).clamp_min(tiny)
        normalized = chunk / absmax.unsqueeze(1)
        distances = (normalized.unsqueeze(-1) - codebook.view(1, 1, -1)).abs()
        chunk_codes = distances.argmin(dim=-1).to(torch.uint8).reshape(-1)
        codes[first * block_size : last * block_size].copy_(chunk_codes)
        scales[first:last].copy_(absmax)
    packed = torch.bitwise_or(codes[0::2], torch.bitwise_left_shift(codes[1::2], 4))
    lookup = _byte_lookup(source_dtype, weight.device)
    if double_quantization:
        scales, scale_absmax, scale_offset, scale_lookup = _quantize_scales(
            scales,
            scale_block_size=scale_block_size,
        )
    else:
        scale_absmax = torch.empty(0, dtype=torch.float32)
        scale_offset = torch.empty(0, dtype=torch.float32)
        scale_lookup = torch.empty(0, dtype=torch.float32)
    return packed, scales, lookup, scale_absmax, scale_offset, scale_lookup


def _dequantize_scales(
    scales: torch.Tensor,
    scale_absmax: torch.Tensor,
    scale_offset: torch.Tensor,
    scale_lookup: torch.Tensor,
    *,
    scale_block_size: int,
) -> torch.Tensor:
    if scale_absmax.numel() == 0:
        return scales
    indices = scales.to(dtype=torch.int32)
    values = F.embedding(indices, scale_lookup).reshape(-1)
    del indices
    group_scales = scale_absmax.repeat_interleave(scale_block_size)[: scales.numel()]
    values.mul_(group_scales).add_(scale_offset)
    return values


def _dequantize_weight(
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
    byte_lookup: torch.Tensor,
    scale_absmax: torch.Tensor,
    scale_offset: torch.Tensor,
    scale_lookup: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
    block_size: int,
    scale_block_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    dequantized_scales = _dequantize_scales(
        scales,
        scale_absmax,
        scale_offset,
        scale_lookup,
        scale_block_size=scale_block_size,
    ).to(dtype=dtype)
    indices = packed_weight.to(dtype=torch.int32)
    values = F.embedding(indices, byte_lookup).reshape(-1)
    del indices
    block_values = values.view(int(scales.numel()), block_size)
    weight = block_values * dequantized_scales.unsqueeze(1)
    del values, block_values, dequantized_scales
    return weight.reshape(-1)[: out_features * in_features].view(out_features, in_features)


class _FrozenNF4LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        packed_weight: torch.Tensor,
        scales: torch.Tensor,
        byte_lookup: torch.Tensor,
        scale_absmax: torch.Tensor,
        scale_offset: torch.Tensor,
        scale_lookup: torch.Tensor,
        bias: torch.Tensor | None,
        out_features: int,
        in_features: int,
        block_size: int,
        scale_block_size: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(
            packed_weight,
            scales,
            byte_lookup,
            scale_absmax,
            scale_offset,
            scale_lookup,
        )
        ctx.out_features = int(out_features)
        ctx.in_features = int(in_features)
        ctx.block_size = int(block_size)
        ctx.scale_block_size = int(scale_block_size)
        weight = _dequantize_weight(
            packed_weight,
            scales,
            byte_lookup,
            scale_absmax,
            scale_offset,
            scale_lookup,
            out_features=ctx.out_features,
            in_features=ctx.in_features,
            block_size=ctx.block_size,
            scale_block_size=ctx.scale_block_size,
            dtype=inputs.dtype,
        )
        return F.linear(inputs, weight, bias)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        (
            packed_weight,
            scales,
            byte_lookup,
            scale_absmax,
            scale_offset,
            scale_lookup,
        ) = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            weight = _dequantize_weight(
                packed_weight,
                scales,
                byte_lookup,
                scale_absmax,
                scale_offset,
                scale_lookup,
                out_features=ctx.out_features,
                in_features=ctx.in_features,
                block_size=ctx.block_size,
                scale_block_size=ctx.scale_block_size,
                dtype=grad_output.dtype,
            )
            grad_input = torch.matmul(grad_output, weight)
        return (
            grad_input,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class NativeNF4LoraLinear(torch.nn.Module):
    """Frozen packed NF4 base linear plus trainable LoRA A/B matrices."""

    def __init__(
        self,
        linear: torch.nn.Linear,
        *,
        rank: int,
        alpha: int,
        dropout: float,
        block_size: int,
        double_quantization: bool,
        scale_block_size: int,
        chunk_blocks: int,
    ) -> None:
        super().__init__()
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        self.block_size = int(block_size)
        self.scale_block_size = int(scale_block_size)
        (
            packed,
            scales,
            lookup,
            scale_absmax,
            scale_offset,
            scale_lookup,
        ) = _quantize_weight(
            linear.weight,
            block_size=block_size,
            chunk_blocks=chunk_blocks,
            double_quantization=double_quantization,
            scale_block_size=scale_block_size,
        )
        self.register_buffer("packed_weight", packed)
        self.register_buffer("scales", scales)
        self.register_buffer("byte_lookup", lookup)
        self.register_buffer("scale_absmax", scale_absmax)
        self.register_buffer("scale_offset", scale_offset)
        self.register_buffer("scale_lookup", scale_lookup)
        self.register_buffer(
            "bias",
            linear.bias.detach().clone() if linear.bias is not None else None,
        )
        self.lora_A = torch.nn.Parameter(
            torch.empty(rank, self.in_features, dtype=torch.float32)
        )
        self.lora_B = torch.nn.Parameter(
            torch.zeros(self.out_features, rank, dtype=torch.float32)
        )
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = torch.nn.Dropout(p=float(dropout))
        self.scaling = float(alpha) / float(rank)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = _FrozenNF4LinearFunction.apply(
            inputs,
            self.packed_weight,
            self.scales,
            self.byte_lookup,
            self.scale_absmax,
            self.scale_offset,
            self.scale_lookup,
            self.bias,
            self.out_features,
            self.in_features,
            self.block_size,
            self.scale_block_size,
        )
        adapter_input = self.lora_dropout(inputs).to(dtype=self.lora_A.dtype)
        adapter = F.linear(F.linear(adapter_input, self.lora_A), self.lora_B)
        return base + adapter.to(dtype=base.dtype) * self.scaling


def convert_model_to_native_nf4_lora(
    model: torch.nn.Module,
    *,
    rank: int,
    alpha: int,
    dropout: float,
    block_size: int = 64,
    double_quantization: bool = False,
    scale_block_size: int = 256,
    target_modules: str = "all-linear",
    chunk_blocks: int = 16_384,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Freeze a model and replace target linears with packed NF4 + LoRA."""

    plan = plan_nf4_lora(
        model,
        rank=rank,
        block_size=block_size,
        double_quantization=double_quantization,
        scale_block_size=scale_block_size,
        target_modules=target_modules,
    )
    started = time.monotonic()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    targets = {str(item["name"]) for item in plan["modules"]}
    for name, module in list(model.named_modules()):
        if name not in targets:
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(
            parent,
            child_name,
            NativeNF4LoraLinear(
                module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                block_size=block_size,
                double_quantization=double_quantization,
                scale_block_size=scale_block_size,
                chunk_blocks=chunk_blocks,
            ),
        )
    plan["conversion_seconds"] = time.monotonic() - started
    return model, plan
