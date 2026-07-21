from __future__ import annotations

import math
from collections import Counter
from typing import Any


class MatmulFlopCounterMode:
    """Count matrix-multiply FLOPs with grouped-query SDPA support."""

    def __new__(cls):
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode

        class _Mode(TorchDispatchMode):
            def __init__(self):
                super().__init__()
                self.total_flops = 0
                self.flops_by_operator: Counter[str] = Counter()

            def __torch_dispatch__(
                self,
                func: Any,
                types: tuple[type, ...],
                args: tuple[Any, ...] = (),
                kwargs: dict[str, Any] | None = None,
            ) -> Any:
                keyword_args = kwargs or {}
                output = func(*args, **keyword_args)
                name = str(getattr(getattr(func, "_schema", None), "name", func))
                flops = _operator_flops(name, args, keyword_args, output, torch=torch)
                if flops:
                    self.total_flops += flops
                    self.flops_by_operator[name] += flops
                return output

        return _Mode()


def _operator_flops(
    name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    output: Any,
    *,
    torch: Any,
) -> int:
    if name == "aten::mm" and len(args) >= 2:
        return _matrix_product_flops(args[0], args[1])
    if name == "aten::bmm" and len(args) >= 2:
        return _matrix_product_flops(args[0], args[1])
    if name == "aten::addmm" and len(args) >= 3:
        return _matrix_product_flops(args[1], args[2])
    if name == "aten::matmul" and len(args) >= 2:
        return _matrix_product_flops(args[0], args[1], output=output)
    if "scaled_dot_product" in name and len(args) >= 3:
        return _sdpa_flops(args[0], args[1], args[2])
    return 0


def _matrix_product_flops(left: Any, right: Any, *, output: Any = None) -> int:
    left_shape = tuple(int(value) for value in left.shape)
    right_shape = tuple(int(value) for value in right.shape)
    if len(left_shape) < 2 or len(right_shape) < 2:
        return 0
    reduction = left_shape[-1]
    if reduction != right_shape[-2]:
        return 0
    if output is not None and hasattr(output, "numel"):
        output_elements = int(output.numel())
    else:
        batch = math.prod(left_shape[:-2])
        output_elements = batch * left_shape[-2] * right_shape[-1]
    return 2 * output_elements * reduction


def _sdpa_flops(query: Any, key: Any, value: Any) -> int:
    query_shape = tuple(int(value) for value in query.shape)
    key_shape = tuple(int(value) for value in key.shape)
    value_shape = tuple(int(dimension) for dimension in value.shape)
    if len(query_shape) < 4 or len(key_shape) < 4 or len(value_shape) < 4:
        return 0
    batch = math.prod(query_shape[:-3])
    query_heads = query_shape[-3]
    query_tokens = query_shape[-2]
    query_head_dim = query_shape[-1]
    key_tokens = key_shape[-2]
    value_head_dim = value_shape[-1]
    qk_flops = (
        2 * batch * query_heads * query_tokens * key_tokens * query_head_dim
    )
    attention_value_flops = (
        2 * batch * query_heads * query_tokens * key_tokens * value_head_dim
    )
    return qk_flops + attention_value_flops
