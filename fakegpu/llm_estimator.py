from __future__ import annotations

import json
import math
import struct
from collections.abc import Mapping, Sequence
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
    adapter_dirs: Sequence[str | Path] | None = None,
    expert_parallel_size: int = 1,
    target_profile: str | None = None,
    compute_acceleration_factor: float = 1.0,
) -> dict[str, Any]:
    """Estimate decoder-only inference memory, communication, and FLOPs.

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
    if expert_parallel_size <= 0:
        raise ValueError("expert_parallel_size must be greater than zero")
    if (
        not math.isfinite(compute_acceleration_factor)
        or compute_acceleration_factor <= 0
    ):
        raise ValueError(
            "compute_acceleration_factor must be finite and positive"
        )
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
    quantization = _quantization_summary(config, checkpoint)
    base_parameter_bytes = (
        int(checkpoint["checkpoint_bytes"])
        if quantization["enabled"]
        else int(checkpoint["parameter_count"]) * element_bytes
    )
    adapters = _adapter_summary(
        adapter_dirs or (),
        element_bytes=element_bytes,
    )
    adapter_parameter_bytes = sum(
        int(adapter["runtime_parameter_bytes"]) for adapter in adapters
    )
    parameter_bytes = base_parameter_bytes + adapter_parameter_bytes

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
    prefill_peak = (
        parameter_bytes
        + input_bytes
        + kv_cache_prefill
        + prefill_transient["peak_bytes"]
    )
    decode_peak = (
        parameter_bytes
        + 8 * batch_size
        + kv_cache_final
        + decode_transient["peak_bytes"]
    )
    tensor_peak = max(prefill_peak, decode_peak)

    total_decode_flops = sum(step["matmul_flops"] for step in decode_steps)
    total_flops = prefill_flops + total_decode_flops
    communication = _expert_parallel_communication(
        dimensions,
        batch_size=batch_size,
        prompt_tokens=prompt_tokens,
        decode_forward_steps=len(decode_steps),
        element_bytes=element_bytes,
        expert_parallel_size=expert_parallel_size,
    )
    memory_traffic = _inference_memory_traffic(
        parameter_bytes=parameter_bytes,
        prefill_cache_bytes=kv_cache_prefill,
        final_cache_bytes=kv_cache_final,
        prefill_transient=prefill_transient,
        decode_transient=decode_transient,
        decode_forward_steps=len(decode_steps),
    )
    performance = None
    if target_profile:
        from .performance_model import estimate_roofline

        performance = estimate_roofline(
            profile_id=target_profile,
            flops=total_flops,
            memory_bytes=memory_traffic["expected_bytes"],
            launch_count=_estimated_matmul_launches(
                dimensions,
                forward_steps=1 + len(decode_steps),
            ),
            compute_acceleration_factor=compute_acceleration_factor,
        )

    model_kind = str(dimensions["model_kind"])
    unmodeled_components = [
        "cuda_context_and_loaded_modules",
        "caching_allocator_fragmentation",
        (
            "backend_fused_attention_workspace"
            if attention_implementation == "sdpa"
            else "backend_kernel_workspace"
        ),
        "quantized_kernel_workspace" if quantization["enabled"] else None,
        "adapter_kernel_fusion_and_merge_state" if adapters else None,
        "expert_load_imbalance" if model_kind == "mixture_of_experts" else None,
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "method": "safetensors_headers_plus_decoder_shape_and_roofline_model",
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
            "expert_parallel_size": expert_parallel_size,
            "target_profile": target_profile,
            "compute_acceleration_factor": compute_acceleration_factor,
        },
        "checkpoint": checkpoint,
        "weight_storage": {
            "base_model_runtime_bytes": base_parameter_bytes,
            "adapter_runtime_bytes": adapter_parameter_bytes,
            "total_runtime_parameter_bytes": parameter_bytes,
            "compute_dtype": selected_dtype,
            "quantization": quantization,
            "adapters": adapters,
        },
        "memory": {
            "base_parameter_bytes": base_parameter_bytes,
            "adapter_parameter_bytes": adapter_parameter_bytes,
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
            "total_flops": total_flops,
            "model_kind": model_kind,
        },
        "communication": communication,
        "memory_traffic": memory_traffic,
        "performance": performance,
        "tracking_confidence": (
            "L3_checkpoint_aware_decoder_shape_model"
            if quantization["enabled"] or adapters or model_kind == "mixture_of_experts"
            else "L2_dense_decoder_shape_model"
        ),
        "unmodeled_components": [
            item for item in unmodeled_components if item is not None
        ],
        "notes": [
            "Checkpoint storage and parameter count come from safetensors headers without loading payloads.",
            "KV-cache bytes use layer, KV-head, head-dimension, sequence, batch, and dtype dimensions.",
            "Quantized base-weight memory uses exact safetensors payload bytes, including scale and metadata tensors.",
            "Adapter runtime bytes use adapter parameter count and the selected compute dtype.",
            "FLOPs cover projection, attention matrix multiplication, active dense or routed experts, and LM-head matrix multiplication.",
            "Elementwise normalization, activation, softmax, rotary embedding, and sampling operations are not included in matrix FLOPs.",
            "The eager-attention transient estimate is conservative; SDPA workspace remains backend-specific.",
            "Expert-parallel traffic assumes a uniform token distribution; hot experts can increase per-rank peaks.",
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


def _decoder_dimensions(config: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "num_attention_heads": config.get("num_attention_heads"),
        "intermediate_size": config.get("intermediate_size"),
        "vocab_size": config.get("vocab_size"),
    }
    missing = [
        name
        for name, value in required.items()
        if not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ]
    if missing:
        raise ValueError(
            "missing positive decoder dimensions: " + ", ".join(missing)
        )
    hidden_size = int(required["hidden_size"])
    attention_heads = int(required["num_attention_heads"])
    if hidden_size % attention_heads and "head_dim" not in config:
        raise ValueError(
            "hidden_size must be divisible by num_attention_heads when "
            "head_dim is not configured"
        )
    head_dim = int(config.get("head_dim") or (hidden_size // attention_heads))
    kv_heads = int(config.get("num_key_value_heads") or attention_heads)
    if head_dim <= 0:
        raise ValueError("head_dim must be greater than zero")
    if kv_heads <= 0 or kv_heads > attention_heads:
        raise ValueError(
            "num_key_value_heads must be between one and "
            "num_attention_heads"
        )

    expert_count = _first_positive_integer(
        config,
        ("num_experts", "num_local_experts", "n_routed_experts"),
    )
    if expert_count:
        experts_per_token = _first_positive_integer(
            config,
            (
                "num_experts_per_tok",
                "num_experts_per_token",
                "experts_per_token",
                "top_k",
            ),
        ) or 1
        if experts_per_token > expert_count:
            raise ValueError(
                "experts per token must not exceed the routed expert count"
            )
        moe_intermediate_size = _first_positive_integer(
            config,
            ("moe_intermediate_size", "expert_intermediate_size"),
        ) or int(required["intermediate_size"])
        shared_expert_intermediate_size = _first_positive_integer(
            config,
            ("shared_expert_intermediate_size",),
        ) or 0
        if shared_expert_intermediate_size == 0:
            shared_experts = _first_positive_integer(
                config,
                ("n_shared_experts", "num_shared_experts"),
            ) or 0
            shared_expert_intermediate_size = (
                shared_experts * moe_intermediate_size
            )
        moe_layer_count = _moe_layer_count(
            config,
            layer_count=int(required["num_hidden_layers"]),
        )
    else:
        experts_per_token = 0
        moe_intermediate_size = 0
        shared_expert_intermediate_size = 0
        moe_layer_count = 0

    dimensions: dict[str, Any] = {
        "hidden_size": hidden_size,
        "num_hidden_layers": int(required["num_hidden_layers"]),
        "num_attention_heads": attention_heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "intermediate_size": int(required["intermediate_size"]),
        "vocab_size": int(required["vocab_size"]),
        "model_kind": (
            "mixture_of_experts" if expert_count else "dense_decoder"
        ),
        "num_routed_experts": expert_count,
        "num_experts_per_token": experts_per_token,
        "moe_intermediate_size": moe_intermediate_size,
        "shared_expert_intermediate_size": (
            shared_expert_intermediate_size
        ),
        "moe_layer_count": moe_layer_count,
        "dense_layer_count": (
            int(required["num_hidden_layers"]) - moe_layer_count
        ),
    }
    return dimensions


def _first_positive_integer(
    config: Mapping[str, Any],
    keys: Sequence[str],
) -> int:
    for key in keys:
        if key not in config or config[key] is None:
            continue
        value = config[key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
        return int(value)
    return 0


def _moe_layer_count(
    config: Mapping[str, Any],
    *,
    layer_count: int,
) -> int:
    mlp_only_layers = config.get("mlp_only_layers")
    if isinstance(mlp_only_layers, list):
        dense_layers = {
            int(value)
            for value in mlp_only_layers
            if isinstance(value, int)
            and not isinstance(value, bool)
            and 0 <= value < layer_count
        }
        return layer_count - len(dense_layers)

    sparse_step = config.get("decoder_sparse_step")
    if sparse_step is None:
        return layer_count
    if (
        not isinstance(sparse_step, int)
        or isinstance(sparse_step, bool)
        or sparse_step <= 0
    ):
        raise ValueError("decoder_sparse_step must be a positive integer")
    return layer_count // sparse_step


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
    dimensions: Mapping[str, Any],
    *,
    batch_size: int,
    query_tokens: int,
    key_tokens: int,
) -> int:
    hidden = dimensions["hidden_size"]
    heads = dimensions["num_attention_heads"]
    kv_width = dimensions["num_key_value_heads"] * dimensions["head_dim"]
    intermediate = dimensions["intermediate_size"]
    vocab = dimensions["vocab_size"]
    attention_projection_per_layer = (
        4 * batch_size * query_tokens * hidden * hidden
        + 4 * batch_size * query_tokens * hidden * kv_width
    )
    attention_per_layer = (
        4
        * batch_size
        * heads
        * query_tokens
        * key_tokens
        * dimensions["head_dim"]
    )
    attention_flops = dimensions["num_hidden_layers"] * (
        attention_projection_per_layer + attention_per_layer
    )
    dense_mlp_flops = (
        dimensions["dense_layer_count"]
        * 6
        * batch_size
        * query_tokens
        * hidden
        * intermediate
    )
    moe_mlp_flops = (
        dimensions["moe_layer_count"]
        * 6
        * batch_size
        * query_tokens
        * hidden
        * dimensions["moe_intermediate_size"]
        * dimensions["num_experts_per_token"]
    )
    shared_expert_flops = (
        dimensions["moe_layer_count"]
        * 6
        * batch_size
        * query_tokens
        * hidden
        * dimensions["shared_expert_intermediate_size"]
    )
    router_flops = (
        dimensions["moe_layer_count"]
        * 2
        * batch_size
        * query_tokens
        * hidden
        * dimensions["num_routed_experts"]
    )
    lm_head = 2 * batch_size * query_tokens * hidden * vocab
    return (
        attention_flops
        + dense_mlp_flops
        + moe_mlp_flops
        + shared_expert_flops
        + router_flops
        + lm_head
    )


def _kv_cache_bytes(
    dimensions: Mapping[str, Any],
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
    dimensions: Mapping[str, Any],
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
    dense_mlp_bytes = (
        hidden_bytes
        + 4
        * batch_size
        * query_tokens
        * intermediate
        * element_bytes
        if dimensions["dense_layer_count"]
        else 0
    )
    routed_mlp_bytes = 0
    router_bytes = 0
    if dimensions["moe_layer_count"]:
        routed_mlp_bytes = (
            hidden_bytes
            + 4
            * batch_size
            * query_tokens
            * dimensions["moe_intermediate_size"]
            * dimensions["num_experts_per_token"]
            * element_bytes
            + 4
            * batch_size
            * query_tokens
            * dimensions["shared_expert_intermediate_size"]
            * element_bytes
        )
        router_bytes = (
            batch_size
            * query_tokens
            * dimensions["num_routed_experts"]
            * 4
        )
        routed_mlp_bytes += router_bytes
    mlp_bytes = max(dense_mlp_bytes, routed_mlp_bytes)
    logits_bytes = hidden_bytes + batch_size * query_tokens * vocab * element_bytes
    return {
        "attention_bytes": attention_bytes,
        "mlp_bytes": mlp_bytes,
        "dense_mlp_bytes": dense_mlp_bytes,
        "routed_mlp_bytes": routed_mlp_bytes,
        "router_bytes": router_bytes,
        "logits_bytes": logits_bytes,
        "peak_bytes": max(attention_bytes, mlp_bytes, logits_bytes),
    }


def _quantization_summary(
    config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    raw = config.get("quantization_config")
    quantization_config = dict(raw) if isinstance(raw, Mapping) else {}
    method = next(
        (
            str(quantization_config[key])
            for key in ("quant_method", "method", "format")
            if quantization_config.get(key)
        ),
        "",
    )
    if not method:
        method = next(
            (
                name
                for name, enabled in (
                    ("bitsandbytes_4bit", config.get("load_in_4bit")),
                    ("bitsandbytes_8bit", config.get("load_in_8bit")),
                )
                if enabled
            ),
            "",
        )
    dtype_bytes = dict(checkpoint.get("dtype_bytes") or {})
    quantized_storage_dtypes = sorted(
        dtype
        for dtype in dtype_bytes
        if dtype in {"U8", "I8", "F8_E4M3", "F8_E5M2"}
    )
    enabled = bool(
        quantization_config
        or method
        or quantized_storage_dtypes
    )
    if enabled and not method:
        method = "checkpoint_low_precision_storage"

    bits: int | float | None = None
    for key in ("bits", "weight_bits", "w_bit"):
        value = quantization_config.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            bits = value
            break
    if bits is None:
        if config.get("load_in_4bit"):
            bits = 4
        elif config.get("load_in_8bit") or quantized_storage_dtypes:
            bits = 8

    return {
        "enabled": enabled,
        "method": method or None,
        "declared_weight_bits": bits,
        "checkpoint_storage_bytes": int(checkpoint["checkpoint_bytes"]),
        "checkpoint_dtype_bytes": dtype_bytes,
        "quantized_storage_dtypes": quantized_storage_dtypes,
        "storage_accounting": (
            "exact_safetensors_payload"
            if enabled
            else "parameter_count_times_compute_dtype"
        ),
    }


def _adapter_summary(
    adapter_dirs: Sequence[str | Path],
    *,
    element_bytes: int,
) -> list[dict[str, Any]]:
    adapters: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for raw in adapter_dirs:
        root = Path(raw).expanduser().resolve()
        if root in seen:
            continue
        seen.add(root)
        checkpoint = inspect_safetensors_checkpoint(root)
        adapter_config_path = root / "adapter_config.json"
        adapter_config: Mapping[str, Any] = {}
        if adapter_config_path.is_file():
            loaded = json.loads(
                adapter_config_path.read_text(encoding="utf-8")
            )
            if isinstance(loaded, Mapping):
                adapter_config = loaded
        adapters.append(
            {
                "path": str(root),
                "adapter_type": str(
                    adapter_config.get("peft_type", "unspecified")
                ),
                "rank": adapter_config.get("r"),
                "parameter_count": int(checkpoint["parameter_count"]),
                "checkpoint_bytes": int(checkpoint["checkpoint_bytes"]),
                "runtime_parameter_bytes": (
                    int(checkpoint["parameter_count"]) * element_bytes
                ),
                "dtype_bytes": checkpoint["dtype_bytes"],
            }
        )
    return adapters


def _expert_parallel_communication(
    dimensions: Mapping[str, Any],
    *,
    batch_size: int,
    prompt_tokens: int,
    decode_forward_steps: int,
    element_bytes: int,
    expert_parallel_size: int,
) -> dict[str, Any]:
    if (
        dimensions["model_kind"] != "mixture_of_experts"
        or expert_parallel_size == 1
    ):
        return {
            "enabled": False,
            "expert_parallel_size": expert_parallel_size,
            "prefill_dispatch_bytes": 0,
            "prefill_combine_bytes": 0,
            "decode_dispatch_bytes_total": 0,
            "decode_combine_bytes_total": 0,
            "total_bytes": 0,
            "assumption": "no_cross_rank_expert_dispatch",
        }

    def remote_bytes(tokens: int) -> int:
        logical = (
            batch_size
            * tokens
            * dimensions["num_experts_per_token"]
            * dimensions["hidden_size"]
            * element_bytes
            * dimensions["moe_layer_count"]
        )
        return math.ceil(
            logical * (expert_parallel_size - 1) / expert_parallel_size
        )

    prefill_dispatch = remote_bytes(prompt_tokens)
    decode_dispatch = remote_bytes(1) * decode_forward_steps
    return {
        "enabled": True,
        "expert_parallel_size": expert_parallel_size,
        "prefill_dispatch_bytes": prefill_dispatch,
        "prefill_combine_bytes": prefill_dispatch,
        "decode_dispatch_bytes_total": decode_dispatch,
        "decode_combine_bytes_total": decode_dispatch,
        "total_bytes": 2 * (prefill_dispatch + decode_dispatch),
        "assumption": "uniform_expert_placement_and_token_routing",
    }


def _inference_memory_traffic(
    *,
    parameter_bytes: int,
    prefill_cache_bytes: int,
    final_cache_bytes: int,
    prefill_transient: Mapping[str, int],
    decode_transient: Mapping[str, int],
    decode_forward_steps: int,
) -> dict[str, Any]:
    forward_steps = 1 + decode_forward_steps
    weight_reads = parameter_bytes * forward_steps
    kv_traffic = prefill_cache_bytes + final_cache_bytes * decode_forward_steps
    transient_peak_sum = int(prefill_transient["peak_bytes"]) + (
        int(decode_transient["peak_bytes"]) * decode_forward_steps
    )
    lower = weight_reads + kv_traffic
    expected = lower + 2 * transient_peak_sum
    upper = lower + 4 * (
        sum(
            int(prefill_transient[key])
            for key in ("attention_bytes", "mlp_bytes", "logits_bytes")
        )
        + decode_forward_steps
        * sum(
            int(decode_transient[key])
            for key in ("attention_bytes", "mlp_bytes", "logits_bytes")
        )
    )
    return {
        "lower_bytes": lower,
        "expected_bytes": expected,
        "upper_bytes": max(expected, upper),
        "weight_read_bytes": weight_reads,
        "kv_cache_traffic_bytes": kv_traffic,
        "method": "analytical_weight_kv_and_transient_interval",
    }


def _estimated_matmul_launches(
    dimensions: Mapping[str, Any],
    *,
    forward_steps: int,
) -> int:
    attention_launches = dimensions["num_hidden_layers"] * 6
    dense_mlp_launches = dimensions["dense_layer_count"] * 3
    moe_launches = dimensions["moe_layer_count"] * (
        4 + (3 if dimensions["shared_expert_intermediate_size"] else 0)
    )
    return forward_steps * (
        attention_launches + dense_mlp_launches + moe_launches + 1
    )
