"""Small PyTorch-only NF4 weight storage used by QLoRA memory validation.

This module is deliberately not a replacement for bitsandbytes.  It stores two
4-bit codes per byte, keeps one BF16/FP16 scale per block, and dequantizes one
linear weight at a time for ordinary PyTorch matmul.  A custom autograd function
recomputes the frozen weight during backward instead of retaining every
dequantized matrix.  That makes the allocator behavior useful for validating
FakeGPU's quantized-training memory model when no external 4-bit backend is
installed.
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
) -> dict[str, int | float]:
    """Return the exact storage and implementation workspace for one tensor."""

    if numel <= 0:
        raise ValueError("numel must be greater than zero")
    if block_size <= 0 or block_size % 2:
        raise ValueError("block_size must be a positive even integer")
    if compute_element_size <= 0:
        raise ValueError("compute_element_size must be greater than zero")

    block_count = (numel + block_size - 1) // block_size
    padded_numel = block_count * block_size
    packed_bytes = padded_numel // 2
    scale_bytes = block_count * compute_element_size
    # The PyTorch implementation uses a byte -> two NF4 values lookup table.
    lookup_bytes = 256 * 2 * compute_element_size
    storage_bytes = packed_bytes + scale_bytes + lookup_bytes
    int32_index_bytes = packed_bytes * 4
    unpacked_value_bytes = padded_numel * compute_element_size
    dequantized_weight_bytes = padded_numel * compute_element_size
    dequantization_workspace_bytes = max(
        int32_index_bytes + unpacked_value_bytes,
        unpacked_value_bytes + dequantized_weight_bytes,
    )
    return {
        "logical_numel": int(numel),
        "block_size": int(block_size),
        "block_count": int(block_count),
        "padded_numel": int(padded_numel),
        "packed_weight_bytes": int(packed_bytes),
        "scale_bytes": int(scale_bytes),
        "lookup_bytes": int(lookup_bytes),
        "storage_bytes": int(storage_bytes),
        "dequantization_workspace_bytes": int(dequantization_workspace_bytes),
        "effective_bits_per_weight": float(8.0 * storage_bytes / numel),
    }


def plan_nf4_lora(
    model: torch.nn.Module,
    *,
    rank: int,
    block_size: int = 64,
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
        "double_quantization": False,
        "block_size": int(block_size),
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
            "Block scales are stored directly in the compute dtype; double quantization is not applied.",
        ],
    }


def _byte_lookup(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    codebook = torch.tensor(NF4_CODEBOOK, dtype=dtype, device=device)
    values = torch.arange(256, dtype=torch.int64, device=device)
    low = torch.bitwise_and(values, 0x0F)
    high = torch.bitwise_right_shift(values, 4)
    return torch.stack((codebook[low], codebook[high]), dim=1)


def _quantize_weight(
    weight: torch.Tensor,
    *,
    block_size: int,
    chunk_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    scales = torch.empty(block_count, dtype=source_dtype)
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
        scales[first:last].copy_(absmax.to(source_dtype))
    packed = torch.bitwise_or(codes[0::2], torch.bitwise_left_shift(codes[1::2], 4))
    lookup = _byte_lookup(source_dtype, weight.device)
    return packed, scales, lookup


def _dequantize_weight(
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
    byte_lookup: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
    block_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    indices = packed_weight.to(dtype=torch.int32)
    values = F.embedding(indices, byte_lookup).reshape(-1)
    del indices
    block_values = values.view(int(scales.numel()), block_size)
    weight = block_values * scales.to(dtype=dtype).unsqueeze(1)
    del values, block_values
    return weight.reshape(-1)[: out_features * in_features].view(out_features, in_features)


class _FrozenNF4LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        packed_weight: torch.Tensor,
        scales: torch.Tensor,
        byte_lookup: torch.Tensor,
        bias: torch.Tensor | None,
        out_features: int,
        in_features: int,
        block_size: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(packed_weight, scales, byte_lookup)
        ctx.out_features = int(out_features)
        ctx.in_features = int(in_features)
        ctx.block_size = int(block_size)
        weight = _dequantize_weight(
            packed_weight,
            scales,
            byte_lookup,
            out_features=ctx.out_features,
            in_features=ctx.in_features,
            block_size=ctx.block_size,
            dtype=inputs.dtype,
        )
        return F.linear(inputs, weight, bias)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        packed_weight, scales, byte_lookup = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            weight = _dequantize_weight(
                packed_weight,
                scales,
                byte_lookup,
                out_features=ctx.out_features,
                in_features=ctx.in_features,
                block_size=ctx.block_size,
                dtype=grad_output.dtype,
            )
            grad_input = torch.matmul(grad_output, weight)
        return grad_input, None, None, None, None, None, None, None


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
        chunk_blocks: int,
    ) -> None:
        super().__init__()
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        self.block_size = int(block_size)
        packed, scales, lookup = _quantize_weight(
            linear.weight,
            block_size=block_size,
            chunk_blocks=chunk_blocks,
        )
        self.register_buffer("packed_weight", packed)
        self.register_buffer("scales", scales)
        self.register_buffer("byte_lookup", lookup)
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
            self.bias,
            self.out_features,
            self.in_features,
            self.block_size,
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
    target_modules: str = "all-linear",
    chunk_blocks: int = 16_384,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Freeze a model and replace target linears with packed NF4 + LoRA."""

    plan = plan_nf4_lora(
        model,
        rank=rank,
        block_size=block_size,
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
                chunk_blocks=chunk_blocks,
            ),
        )
    plan["conversion_seconds"] = time.monotonic() - started
    return model, plan
