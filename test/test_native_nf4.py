from __future__ import annotations

import torch

from verification.native_nf4 import (
    NativeNF4LoraLinear,
    convert_model_to_native_nf4_lora,
    nf4_tensor_layout,
    plan_nf4_lora,
)


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(8, 6, bias=True, dtype=torch.bfloat16)
        self.lm_head = torch.nn.Linear(6, 8, bias=False, dtype=torch.bfloat16)

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.lm_head


def test_nf4_tensor_layout_counts_packed_storage_and_workspace() -> None:
    layout = nf4_tensor_layout(128, block_size=64, compute_element_size=2)

    assert layout["packed_weight_bytes"] == 64
    assert layout["scale_bytes"] == 4
    assert layout["lookup_bytes"] == 1024
    assert layout["storage_bytes"] == 1092
    assert layout["dequantization_workspace_bytes"] == 512


def test_plan_excludes_output_embedding_and_counts_lora_parameters() -> None:
    model = _TinyModel()
    plan = plan_nf4_lora(model, rank=2, block_size=8)

    assert plan["module_count"] == 1
    assert plan["modules"][0]["name"] == "proj"
    assert plan["quantized_weight_count"] == 48
    assert plan["original_parameter_count"] == 54
    assert plan["lora_parameter_count"] == 28
    assert plan["adapter_dtype"] == "float32"


def test_native_nf4_lora_forward_backward_and_storage_match_plan() -> None:
    model = _TinyModel()
    model, plan = convert_model_to_native_nf4_lora(
        model,
        rank=2,
        alpha=4,
        dropout=0.0,
        block_size=8,
        chunk_blocks=4,
    )

    assert isinstance(model.proj, NativeNF4LoraLinear)
    quantized_buffers = (
        model.proj.packed_weight.untyped_storage().nbytes()
        + model.proj.scales.untyped_storage().nbytes()
        + model.proj.byte_lookup.untyped_storage().nbytes()
    )
    assert quantized_buffers == plan["quantized_storage_bytes"]
    assert all(
        parameter.requires_grad
        for parameter in (model.proj.lora_A, model.proj.lora_B)
    )
    assert model.proj.lora_A.dtype == torch.float32
    assert model.proj.lora_B.dtype == torch.float32
    assert model.lm_head.weight.requires_grad is False

    inputs = torch.randn(2, 3, 8, dtype=torch.bfloat16, requires_grad=True)
    outputs = model.proj(inputs)
    outputs.float().square().mean().backward()

    assert outputs.shape == (2, 3, 6)
    assert inputs.grad is not None
    assert model.proj.lora_A.grad is not None
    assert model.proj.lora_B.grad is not None
