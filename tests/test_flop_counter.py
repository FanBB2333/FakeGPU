from __future__ import annotations

import torch

from fakegpu import MatmulFlopCounterMode


def test_matmul_flop_counter_counts_matrix_multiplication() -> None:
    left = torch.randn(2, 3)
    right = torch.randn(3, 4)

    with MatmulFlopCounterMode() as counter:
        left @ right

    assert counter.total_flops == 2 * 2 * 3 * 4
    assert counter.flops_by_operator["aten::mm"] == counter.total_flops
