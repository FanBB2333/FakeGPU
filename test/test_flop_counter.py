from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from fakegpu.flop_counter import MatmulFlopCounterMode


def test_counter_counts_mm_and_bmm() -> None:
    with MatmulFlopCounterMode() as counter:
        torch.ones(2, 3) @ torch.ones(3, 4)
        torch.bmm(torch.ones(5, 2, 3), torch.ones(5, 3, 4))
    assert counter.total_flops == 48 + 240
    assert counter.flops_by_operator["aten::mm"] == 48
    assert counter.flops_by_operator["aten::bmm"] == 240


def test_counter_counts_linear_inside_inference_mode() -> None:
    inputs = torch.ones(2, 3)
    weight = torch.ones(4, 3)
    with torch.inference_mode():
        with MatmulFlopCounterMode() as counter:
            F.linear(inputs, weight)
    assert counter.total_flops == 48
    assert counter.flops_by_operator["aten::mm"] == 48


def test_counter_supports_grouped_query_sdpa() -> None:
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 1, 5, 4)
    value = torch.randn(1, 1, 5, 4)
    try:
        with MatmulFlopCounterMode() as counter:
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                enable_gqa=True,
            )
    except TypeError:
        pytest.skip("this PyTorch version does not expose enable_gqa")
    assert output.shape == (1, 2, 3, 4)
    assert counter.total_flops == 480


def test_counter_handles_distinct_value_dimension() -> None:
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 1, 5, 4)
    value = torch.randn(1, 1, 5, 6)
    try:
        with MatmulFlopCounterMode() as counter:
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                enable_gqa=True,
            )
    except (RuntimeError, TypeError):
        pytest.skip("this PyTorch version does not support this SDPA shape")
    assert output.shape == (1, 2, 3, 6)
    # QK^T: 2*1*2*3*5*4; attention*V: 2*1*2*3*5*6.
    assert counter.total_flops == 600
