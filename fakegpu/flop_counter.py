from __future__ import annotations

import math
from collections import Counter
from typing import Any


class MatmulFlopCounterMode:
    """Count matrix-heavy FLOPs, including grouped-query SDPA.

    PyTorch's counter already decomposes composite operators so that linear
    projections are counted even under ``torch.inference_mode``.  This wrapper
    keeps that behavior and replaces the fused-attention formulas that assume
    the query and key/value head counts are equal.
    """

    def __init__(self) -> None:
        import torch
        from torch.utils.flop_counter import FlopCounterMode

        custom_mapping = {}
        for operator_name in (
            "_scaled_dot_product_efficient_attention",
            "_scaled_dot_product_flash_attention",
            "_scaled_dot_product_cudnn_attention",
            "_scaled_dot_product_flash_attention_for_cpu",
        ):
            operator = getattr(torch.ops.aten, operator_name, None)
            if operator is not None:
                custom_mapping[operator] = _grouped_query_sdpa_flops
        self._counter = FlopCounterMode(
            display=False,
            custom_mapping=custom_mapping,
        )

    def __enter__(self) -> MatmulFlopCounterMode:
        self._counter.__enter__()
        return self

    def __exit__(self, *args: Any) -> Any:
        return self._counter.__exit__(*args)

    @property
    def total_flops(self) -> int:
        return int(self._counter.get_total_flops())

    @property
    def flops_by_operator(self) -> Counter[str]:
        counts = self._counter.get_flop_counts().get("Global", {})
        return Counter(
            {
                _operator_name(operator): int(flops)
                for operator, flops in counts.items()
            }
        )


def _grouped_query_sdpa_flops(
    query_shape: tuple[int, ...],
    key_shape: tuple[int, ...],
    value_shape: tuple[int, ...],
    *args: Any,
    out_shape: Any = None,
    **kwargs: Any,
) -> int:
    del args, out_shape, kwargs
    if len(query_shape) < 4 or len(key_shape) < 4 or len(value_shape) < 4:
        return 0
    batch = math.prod(int(value) for value in query_shape[:-3])
    query_heads = int(query_shape[-3])
    query_tokens = int(query_shape[-2])
    query_head_dim = int(query_shape[-1])
    key_tokens = int(key_shape[-2])
    value_head_dim = int(value_shape[-1])
    qk_flops = (
        2 * batch * query_heads * query_tokens * key_tokens * query_head_dim
    )
    attention_value_flops = (
        2 * batch * query_heads * query_tokens * key_tokens * value_head_dim
    )
    return qk_flops + attention_value_flops


def _operator_name(operator: Any) -> str:
    name = str(operator)
    if name.startswith("aten."):
        return "aten::" + name.removeprefix("aten.")
    return name
