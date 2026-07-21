from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "fakegpu.llm_inference_estimate.v1"

_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

_TORCH_DTYPE_TO_BYTES = {
    "float16": 2,
    "half": 2,
    "bfloat16": 2,
    "float32": 4,
    "float": 4,
    "float64": 8,
    "double": 8,
}


def inspect_safetensors_checkpoint(model_dir: str | Path) -> dict[str, Any]:
    """Read safetensors headers without loading tensor payloads."""

    root = Path(model_dir).expanduser().resolve()
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"invalid safetensors weight map: {index_path}")
        shard_names = sorted({str(value) for value in weight_map.values()})
        indexed_total = int((index.get("metadata") or {}).get("total_size", 0) or 0)
    else:
        shard_names = [path.name for path in sorted(root.glob("*.safetensors"))]
        indexed_total = 0

    if not shard_names:
        raise FileNotFoundError(f"no safetensors checkpoint found under {root}")

    parameter_count = 0
    checkpoint_bytes = 0
    tensor_count = 0
    dtype_bytes: dict[str, int] = {}
    for shard_name in shard_names:
        shard_path = root / shard_name
        header = _read_safetensors_header(shard_path)
        for tensor_name, metadata in header.items():
            if tensor_name == "__metadata__":
                continue
            if not isinstance(metadata, dict):
                raise ValueError(f"invalid tensor metadata for {tensor_name!r} in {shard_path}")
            dtype = str(metadata.get("dtype", ""))
            shape = metadata.get("shape")
            if dtype not in _DTYPE_BYTES:
                raise ValueError(f"unsupported safetensors dtype {dtype!r} in {shard_path}")
            if not isinstance(shape, list) or any(
                not isinstance(dimension, int) or dimension < 0
                for dimension in shape
            ):
                raise ValueError(f"invalid shape for {tensor_name!r} in {shard_path}")
            numel = math.prod(shape)
            nbytes = numel * _DTYPE_BYTES[dtype]
            offsets = metadata.get("data_offsets")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(not isinstance(value, int) for value in offsets)
                or offsets[1] - offsets[0] != nbytes
            ):
                raise ValueError(f"invalid data offsets for {tensor_name!r} in {shard_path}")
            parameter_count += numel
            checkpoint_bytes += nbytes
            tensor_count += 1
            dtype_bytes[dtype] = dtype_bytes.get(dtype, 0) + nbytes

    if indexed_total and indexed_total != checkpoint_bytes:
        raise ValueError(
            f"checkpoint index reports {indexed_total} bytes, headers report {checkpoint_bytes}"
        )

    return {
        "format": "safetensors",
        "shard_count": len(shard_names),
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "checkpoint_bytes": checkpoint_bytes,
        "dtype_bytes": dict(sorted(dtype_bytes.items())),
        "source": "safetensors_headers",
    }


def estimate_decoder_inference(
    model_dir: str | Path,
    *,
    batch_size: int = 1,
    prompt_tokens: int,
    generated_tokens: int = 1,
    dtype: str = "auto",
    use_cache: bool = True,
    attention_implementation: str = "eager",
    runtime_overhead_bytes: int = 0,
) -> dict[str, Any]:
    """Estimate dense decoder-only inference memory and matrix FLOPs.

    The checkpoint is inspected through safetensors headers only. No model
    tensors are materialized and no CUDA context is created.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if prompt_tokens <= 0:
        raise ValueError("prompt_tokens must be greater than zero")
    if generated_tokens <= 0:
        raise ValueError("generated_tokens must be greater than zero")
    if runtime_overhead_bytes < 0:
        raise ValueError("runtime_overhead_bytes must be non-negative")
    if attention_implementation not in {"eager", "sdpa"}:
        raise ValueError("attention_implementation must be 'eager' or 'sdpa'")

    root = Path(model_dir).expanduser().resolve()
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"model config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint = inspect_safetensors_checkpoint(root)
    dimensions = _decoder_dimensions(config)
    selected_dtype = _normalize_dtype(dtype, config=config)
    element_bytes = _TORCH_DTYPE_TO_BYTES[selected_dtype]
    parameter_bytes = int(checkpoint["parameter_count"]) * element_bytes

    prefill_flops = _forward_matmul_flops(
        dimensions,
        batch_size=batch_size,
        query_tokens=prompt_tokens,
        key_tokens=prompt_tokens,
    )
    decode_steps: list[dict[str, int]] = []
    for step in range(max(0, generated_tokens - 1)):
        key_tokens = prompt_tokens + step + 1
        flops = _forward_matmul_flops(
            dimensions,
            batch_size=batch_size,
            query_tokens=1,
            key_tokens=key_tokens,
        )
        decode_steps.append(
            {
                "step": step + 1,
                "query_tokens": 1,
                "key_tokens": key_tokens,
                "matmul_flops": flops,
            }
        )

    cache_tokens_after_generation = prompt_tokens + max(0, generated_tokens - 1)
    kv_cache_prefill = (
        _kv_cache_bytes(
            dimensions,
            batch_size=batch_size,
            tokens=prompt_tokens,
            element_bytes=element_bytes,
        )
        if use_cache
        else 0
    )
    kv_cache_final = (
        _kv_cache_bytes(
            dimensions,
            batch_size=batch_size,
            tokens=cache_tokens_after_generation,
            element_bytes=element_bytes,
        )
        if use_cache
        else 0
    )
    prefill_transient = _forward_transient_bytes(
        dimensions,
        batch_size=batch_size,
        query_tokens=prompt_tokens,
        key_tokens=prompt_tokens,
        element_bytes=element_bytes,
        attention_implementation=attention_implementation,
    )
    decode_transient = _forward_transient_bytes(
        dimensions,
        batch_size=batch_size,
        query_tokens=1,
        key_tokens=cache_tokens_after_generation,
        element_bytes=element_bytes,
        attention_implementation=attention_implementation,
    )
    input_bytes = batch_size * prompt_tokens * 8
    prefill_peak = parameter_bytes + input_bytes + kv_cache_prefill + prefill_transient["peak_bytes"]
    decode_peak = parameter_bytes + 8 * batch_size + kv_cache_final + decode_transient["peak_bytes"]
    tensor_peak = max(prefill_peak, decode_peak)

    total_decode_flops = sum(step["matmul_flops"] for step in decode_steps)
    return {
        "schema_version": SCHEMA_VERSION,
        "method": "safetensors_headers_plus_dense_decoder_shape_model",
        "model": {
            "path": str(root),
            "model_type": str(config.get("model_type", "unknown")),
            "architectures": list(config.get("architectures") or []),
            **dimensions,
        },
        "inputs": {
            "batch_size": batch_size,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "decode_forward_steps": len(decode_steps),
            "use_cache": use_cache,
            "attention_implementation": attention_implementation,
            "dtype": selected_dtype,
            "element_bytes": element_bytes,
        },
        "checkpoint": checkpoint,
        "memory": {
            "parameter_bytes": parameter_bytes,
            "input_bytes": input_bytes,
            "kv_cache_bytes_after_prefill": kv_cache_prefill,
            "kv_cache_bytes_after_generation": kv_cache_final,
            "prefill_transient": prefill_transient,
            "decode_transient": decode_transient,
            "estimated_prefill_tensor_peak_bytes": prefill_peak,
            "estimated_decode_tensor_peak_bytes": decode_peak,
            "estimated_tensor_peak_bytes": tensor_peak,
            "runtime_overhead_bytes": runtime_overhead_bytes,
            "estimated_process_peak_bytes": tensor_peak + runtime_overhead_bytes,
        },
        "compute": {
            "metric": "matrix_multiply_flops",
            "convention": "one multiply-add is two FLOPs",
            "prefill_flops": prefill_flops,
            "decode_steps": decode_steps,
            "decode_flops_total": total_decode_flops,
            "total_flops": prefill_flops + total_decode_flops,
        },
        "tracking_confidence": "L2_dense_decoder_shape_model",
        "unmodeled_components": [
            "cuda_context_and_loaded_modules",
            "caching_allocator_fragmentation",
            "backend_fused_attention_workspace" if attention_implementation == "sdpa" else "backend_kernel_workspace",
            "quantization_or_adapter_runtime_state",
        ],
        "notes": [
            "Checkpoint storage and parameter count come from safetensors headers without loading payloads.",
            "KV-cache bytes use layer, KV-head, head-dimension, sequence, batch, and dtype dimensions.",
            "FLOPs cover dense projection, attention matrix multiplication, MLP, and LM-head matrix multiplication.",
            "Elementwise normalization, activation, softmax, rotary embedding, and sampling operations are not included in matrix FLOPs.",
            "The eager-attention transient estimate is conservative; SDPA workspace remains backend-specific.",
        ],
    }


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"truncated safetensors file: {path}")
        header_length = struct.unpack("<Q", length_bytes)[0]
        if header_length <= 0 or header_length > 128 * 1024 * 1024:
            raise ValueError(f"invalid safetensors header length in {path}")
        header_bytes = handle.read(header_length)
    if len(header_bytes) != header_length:
        raise ValueError(f"truncated safetensors header: {path}")
    header = json.loads(header_bytes.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header must be an object: {path}")
    return header


def _decoder_dimensions(config: dict[str, Any]) -> dict[str, int]:
    required = {
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "num_attention_heads": config.get("num_attention_heads"),
        "intermediate_size": config.get("intermediate_size"),
        "vocab_size": config.get("vocab_size"),
    }
    missing = [name for name, value in required.items() if not isinstance(value, int) or value <= 0]
    if missing:
        raise ValueError("missing positive decoder dimensions: " + ", ".join(missing))
    if any(key in config for key in ("num_experts", "num_local_experts", "n_routed_experts")):
        raise ValueError("mixture-of-experts decoder configs require a dedicated compute model")
    hidden_size = int(required["hidden_size"])
    attention_heads = int(required["num_attention_heads"])
    head_dim = int(config.get("head_dim") or (hidden_size // attention_heads))
    kv_heads = int(config.get("num_key_value_heads") or attention_heads)
    return {
        "hidden_size": hidden_size,
        "num_hidden_layers": int(required["num_hidden_layers"]),
        "num_attention_heads": attention_heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "intermediate_size": int(required["intermediate_size"]),
        "vocab_size": int(required["vocab_size"]),
    }


def _normalize_dtype(dtype: str, *, config: dict[str, Any]) -> str:
    value = str(config.get("torch_dtype", "float32")) if dtype == "auto" else str(dtype)
    value = value.lower().replace("torch.", "")
    aliases = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32", "fp64": "float64"}
    value = aliases.get(value, value)
    if value not in _TORCH_DTYPE_TO_BYTES:
        choices = ", ".join(sorted(_TORCH_DTYPE_TO_BYTES))
        raise ValueError(f"unsupported inference dtype {value!r}; expected one of: {choices}")
    return value


def _forward_matmul_flops(
    dimensions: dict[str, int],
    *,
    batch_size: int,
    query_tokens: int,
    key_tokens: int,
) -> int:
    hidden = dimensions["hidden_size"]
    layers = dimensions["num_hidden_layers"]
    heads = dimensions["num_attention_heads"]
    kv_width = dimensions["num_key_value_heads"] * dimensions["head_dim"]
    intermediate = dimensions["intermediate_size"]
    vocab = dimensions["vocab_size"]
    projection_per_layer = (
        4 * batch_size * query_tokens * hidden * hidden
        + 4 * batch_size * query_tokens * hidden * kv_width
        + 6 * batch_size * query_tokens * hidden * intermediate
    )
    attention_per_layer = (
        4
        * batch_size
        * heads
        * query_tokens
        * key_tokens
        * dimensions["head_dim"]
    )
    lm_head = 2 * batch_size * query_tokens * hidden * vocab
    return layers * (projection_per_layer + attention_per_layer) + lm_head


def _kv_cache_bytes(
    dimensions: dict[str, int],
    *,
    batch_size: int,
    tokens: int,
    element_bytes: int,
) -> int:
    return (
        2
        * dimensions["num_hidden_layers"]
        * batch_size
        * dimensions["num_key_value_heads"]
        * tokens
        * dimensions["head_dim"]
        * element_bytes
    )


def _forward_transient_bytes(
    dimensions: dict[str, int],
    *,
    batch_size: int,
    query_tokens: int,
    key_tokens: int,
    element_bytes: int,
    attention_implementation: str,
) -> dict[str, int]:
    hidden = dimensions["hidden_size"]
    heads = dimensions["num_attention_heads"]
    kv_heads = dimensions["num_key_value_heads"]
    head_dim = dimensions["head_dim"]
    intermediate = dimensions["intermediate_size"]
    vocab = dimensions["vocab_size"]
    hidden_bytes = batch_size * query_tokens * hidden * element_bytes
    q_bytes = batch_size * query_tokens * heads * head_dim * element_bytes
    kv_bytes = 2 * batch_size * key_tokens * kv_heads * head_dim * element_bytes
    score_bytes = batch_size * heads * query_tokens * key_tokens * element_bytes
    attention_bytes = hidden_bytes + q_bytes + kv_bytes + hidden_bytes
    if attention_implementation == "eager":
        attention_bytes += 2 * score_bytes
    mlp_bytes = hidden_bytes + 4 * batch_size * query_tokens * intermediate * element_bytes
    logits_bytes = hidden_bytes + batch_size * query_tokens * vocab * element_bytes
    return {
        "attention_bytes": attention_bytes,
        "mlp_bytes": mlp_bytes,
        "logits_bytes": logits_bytes,
        "peak_bytes": max(attention_bytes, mlp_bytes, logits_bytes),
    }
