from __future__ import annotations

import json
import math
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "fakegpu.llm_inference_estimate.v1"
VISION_UNMODELED_COMPONENTS = (
    "vision_encoder_activations",
    "vision_encoder_flops",
    "multimodal_projector_activations_and_flops",
)
VISION_ESTIMATE_WARNING = (
    "Vision encoder and multimodal projector activations and FLOPs are not modeled. "
    "Checkpoint weights are included, but transient memory and compute cover only "
    "the decoder; fit/admission and latency estimates require multimodal validation."
)
KV_CACHE_STRATEGIES = frozenset(
    {"dynamic", "static", "quantized", "paged"}
)
# Transformers QuantizedCache default q_group_size (cache_utils.py, 4.57).
DEFAULT_KV_QUANTIZATION_GROUP_SIZE = 64
MLA_KV_CACHE_LAYOUTS = frozenset({"auto", "latent", "decompressed"})
# Transformers config classes that derive layer_types from a fixed pattern
# when a checkpoint config omits it: a layer is full attention when
# (index + 1) is a multiple of the pattern (gemma3, cohere2, olmo3 every
# 6th/4th/4th layer; gemma2 and gpt_oss alternate).
_SLIDING_WINDOW_PATTERN_DEFAULTS = {
    "gemma3_text": 6,
    "gemma3": 6,
    "cohere2": 4,
    "olmo3": 4,
    "gemma2": 2,
    "gpt_oss": 2,
}

# Bits per element for every dtype a safetensors 0.8 header accepts. F4 and
# the F6 types are packed: the header shape counts logical elements, and the
# payload is numel * bits / 8 bytes, which must end on a byte boundary.
_DTYPE_BITS = {
    "BOOL": 8,
    "U8": 8,
    "I8": 8,
    "F8_E4M3": 8,
    "F8_E5M2": 8,
    "F8_E4M3FNUZ": 8,
    "F8_E5M2FNUZ": 8,
    "F8_E8M0": 8,
    "F4": 4,
    "F6_E2M3": 6,
    "F6_E3M2": 6,
    "I16": 16,
    "U16": 16,
    "F16": 16,
    "BF16": 16,
    "I32": 32,
    "U32": 32,
    "F32": 32,
    "C64": 64,
    "I64": 64,
    "U64": 64,
    "F64": 64,
}
# Byte width of the whole-byte dtypes; packed dtypes have no per-element width.
_DTYPE_BYTES = {
    dtype: bits // 8 for dtype, bits in _DTYPE_BITS.items() if bits % 8 == 0
}
# Low-precision storage that marks a checkpoint as quantized.
_LOW_PRECISION_STORAGE_DTYPES = frozenset(
    dtype for dtype, bits in _DTYPE_BITS.items() if bits <= 8 and dtype != "BOOL"
)

_TORCH_DTYPE_TO_BYTES = {
    "float16": 2,
    "half": 2,
    "bfloat16": 2,
    "float32": 4,
    "float": 4,
    "float64": 8,
    "double": 8,
}

# These Transformers decoders tie the two canonical weight names below when
# tie_word_embeddings is explicit. Other checkpoint layouts remain unchanged.
_CANONICAL_TIED_EMBEDDING_MODELS = frozenset(
    {"llama", "mistral", "mixtral", "qwen2", "qwen2_moe", "qwen3", "qwen3_moe"}
)


def estimate_kv_cache_memory(
    *,
    num_hidden_layers: int,
    num_key_value_heads: int,
    head_dim: int,
    batch_size: int,
    prompt_tokens: int,
    generated_tokens: int = 1,
    element_bytes: int = 2,
    strategy: str = "dynamic",
    quantized_bits: int = 4,
    quantized_residual_tokens: int = 128,
    block_tokens: int = 16,
    max_cache_tokens: int | None = None,
    window_tokens: int | None = None,
    use_cache: bool = True,
    elements_per_token_per_layer: int | None = None,
    cache_layout: str = "separate_key_value",
    sliding_window_layers: int = 0,
    sliding_window_tokens: int | None = None,
    prefill_chunk_tokens: int | None = None,
    quantized_group_size: int = DEFAULT_KV_QUANTIZATION_GROUP_SIZE,
) -> dict[str, Any]:
    """Estimate allocated KV-cache storage for common serving strategies.

    ``sliding_window_layers`` of the ``num_hidden_layers`` layers attend over
    ``sliding_window_tokens`` (from the model config); the rest keep their
    whole context. ``window_tokens`` is an explicit override that caps every
    layer instead, as before per-layer accounting existed.
    """

    positive_integers = {
        "num_hidden_layers": num_hidden_layers,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "batch_size": batch_size,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "element_bytes": element_bytes,
        "block_tokens": block_tokens,
        "quantized_group_size": quantized_group_size,
    }
    for name, value in positive_integers.items():
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(use_cache, bool):
        raise ValueError("use_cache must be a boolean")
    if elements_per_token_per_layer is not None and (
        not isinstance(elements_per_token_per_layer, int)
        or isinstance(elements_per_token_per_layer, bool)
        or elements_per_token_per_layer <= 0
    ):
        raise ValueError(
            "elements_per_token_per_layer must be a positive integer"
        )
    if not isinstance(cache_layout, str) or not cache_layout.strip():
        raise ValueError("cache_layout must be a non-empty string")
    if strategy not in KV_CACHE_STRATEGIES:
        choices = ", ".join(sorted(KV_CACHE_STRATEGIES))
        raise ValueError(
            f"unsupported KV-cache strategy {strategy!r}; "
            f"expected one of: {choices}"
        )
    if (
        not isinstance(quantized_bits, int)
        or isinstance(quantized_bits, bool)
        or quantized_bits not in {2, 4, 8}
    ):
        raise ValueError("quantized_bits must be one of: 2, 4, 8")
    if (
        not isinstance(quantized_residual_tokens, int)
        or isinstance(quantized_residual_tokens, bool)
        or quantized_residual_tokens < 0
    ):
        raise ValueError(
            "quantized_residual_tokens must be a non-negative integer"
        )
    for name, value in (
        ("max_cache_tokens", max_cache_tokens),
        ("window_tokens", window_tokens),
        ("sliding_window_tokens", sliding_window_tokens),
        ("prefill_chunk_tokens", prefill_chunk_tokens),
    ):
        if value is not None and (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")
    if (
        not isinstance(sliding_window_layers, int)
        or isinstance(sliding_window_layers, bool)
        or not 0 <= sliding_window_layers <= num_hidden_layers
    ):
        raise ValueError(
            "sliding_window_layers must be between zero and "
            "num_hidden_layers"
        )
    if sliding_window_layers and sliding_window_tokens is None:
        raise ValueError(
            "sliding_window_layers require sliding_window_tokens"
        )
    if strategy != "static" and max_cache_tokens is not None:
        raise ValueError(
            "max_cache_tokens is only valid for the static KV-cache strategy"
        )

    # The explicit window caps every layer (the pre-existing override); a
    # quantized Transformers cache builds one QuantizedLayer per layer and
    # keeps the whole history even on sliding-window layers.
    if not sliding_window_layers:
        sliding_reason = "no_sliding_window_layers"
    elif window_tokens is not None:
        sliding_reason = "window_tokens_override_applies_to_every_layer"
    elif strategy == "quantized":
        sliding_reason = "quantized_cache_keeps_full_history_per_layer"
    else:
        sliding_reason = "config_layer_types"
    sliding_layers = (
        sliding_window_layers
        if sliding_reason == "config_layer_types"
        else 0
    )
    full_layers = num_hidden_layers - sliding_layers

    final_logical_tokens = prompt_tokens + generated_tokens - 1
    storage_bits = (
        quantized_bits if strategy == "quantized" else element_bytes * 8
    )
    static_capacity_tokens = None
    if strategy == "static":
        static_capacity_tokens = max_cache_tokens or final_logical_tokens
        if window_tokens is not None:
            static_capacity_tokens = min(
                static_capacity_tokens,
                window_tokens,
            )
        final_effective_tokens = _effective_context_tokens(
            final_logical_tokens,
            window_tokens=window_tokens,
        )
        if full_layers and static_capacity_tokens < final_effective_tokens:
            raise ValueError(
                "max_cache_tokens must cover the requested generation "
                "after applying window_tokens"
            )

    resolved_elements_per_token_per_layer = (
        2 * num_key_value_heads * head_dim
        if elements_per_token_per_layer is None
        else elements_per_token_per_layer
    )
    elements_per_token_per_sequence = (
        num_hidden_layers * resolved_elements_per_token_per_layer
    )

    def full_tokens(logical_tokens: int) -> tuple[int, int]:
        effective = _effective_context_tokens(
            logical_tokens,
            window_tokens=window_tokens,
        )
        if strategy == "static":
            allocated = static_capacity_tokens
        elif strategy == "paged":
            allocated = math.ceil(effective / block_tokens) * block_tokens
        else:
            allocated = effective
        return effective, allocated

    def sliding_tokens(logical_tokens: int, *, prefill: bool) -> tuple[int, int]:
        window = int(sliding_window_tokens)
        if strategy == "dynamic":
            # Transformers 4.57 DynamicSlidingWindowLayer keeps a view of the
            # last window-1 tokens of the concatenated states, so the
            # prefill keeps the whole prompt's storage; each decode step
            # re-concatenates and leaves min(tokens, window) behind.
            if prefill or generated_tokens == 1:
                effective = logical_tokens
            else:
                effective = min(logical_tokens, window)
            return effective, effective
        if strategy == "static":
            # StaticSlidingWindowLayer preallocates min(window, capacity).
            capacity = min(window, int(static_capacity_tokens))
            return min(logical_tokens, capacity), capacity
        # Paged: vLLM frees blocks that fall wholly outside the window, so a
        # sequence holds the last window-1 computed tokens plus the
        # scheduled ones, and one extra block when the window does not
        # start on a block boundary.
        scheduled = (
            min(prefill_chunk_tokens or logical_tokens, logical_tokens)
            if prefill
            else 1
        )
        effective = min(logical_tokens, window - 1 + scheduled)
        all_blocks = math.ceil(logical_tokens / block_tokens)
        if effective >= logical_tokens:
            blocks = all_blocks
        else:
            blocks = min(
                all_blocks,
                math.ceil(effective / block_tokens) + 1,
            )
        return effective, blocks * block_tokens

    def layer_bytes(layers: int, tokens: int, bits: int) -> int:
        return _kv_storage_bytes(
            layers * resolved_elements_per_token_per_layer,
            batch_size=batch_size,
            tokens=tokens,
            storage_bits=bits,
        )

    def phase_allocation(logical_tokens: int, *, prefill: bool) -> dict[str, Any]:
        compute_bits = element_bytes * 8
        if not use_cache:
            full_effective = full_allocated = 0
            sliding_effective = sliding_allocated = 0
        else:
            full_effective, full_allocated = full_tokens(logical_tokens)
            sliding_effective, sliding_allocated = (
                sliding_tokens(logical_tokens, prefill=prefill)
                if sliding_layers
                else (0, 0)
            )
        base_dtype_bytes = layer_bytes(
            full_layers, full_effective, compute_bits
        ) + layer_bytes(sliding_layers, sliding_effective, compute_bits)
        metadata_bytes = 0
        if strategy == "quantized":
            # Transformers 4.57 QuantizedLayer quantizes the whole prefill
            # at the first update and then appends generated tokens to a
            # full-precision residual, requantizing everything once it
            # reaches residual_length-1 tokens (at least one token sits in
            # the residual right after each requantization).
            decode_steps = 0 if prefill else generated_tokens - 1
            full_precision_tokens = min(
                full_effective,
                decode_steps,
                max(1, quantized_residual_tokens - 1),
            )
            quantized_tokens = full_effective - full_precision_tokens
            quantized_elements = (
                full_layers
                * resolved_elements_per_token_per_layer
                * batch_size
                * quantized_tokens
            )
            # optimum-quanto and HQQ keep one scale and one zero-point per
            # q_group_size elements in the compute dtype.
            metadata_bytes = (
                2
                * element_bytes
                * math.ceil(quantized_elements / quantized_group_size)
            )
            storage_bytes = (
                layer_bytes(full_layers, full_precision_tokens, compute_bits)
                + layer_bytes(full_layers, quantized_tokens, quantized_bits)
                + metadata_bytes
            )
            allocated_bytes = storage_bytes
            full_allocated_bytes = allocated_bytes
            sliding_allocated_bytes = 0
            sliding_storage_bytes = 0
        else:
            full_precision_tokens = quantized_tokens = None
            sliding_storage_bytes = layer_bytes(
                sliding_layers, sliding_effective, storage_bits
            )
            storage_bytes = base_dtype_bytes
            full_allocated_bytes = layer_bytes(
                full_layers, full_allocated, storage_bits
            )
            sliding_allocated_bytes = layer_bytes(
                sliding_layers, sliding_allocated, storage_bits
            )
            allocated_bytes = full_allocated_bytes + sliding_allocated_bytes
        primary_effective = full_effective if full_layers else sliding_effective
        primary_allocated = full_allocated if full_layers else sliding_allocated
        return {
            "logical_tokens_per_sequence": logical_tokens,
            "effective_tokens_per_sequence": primary_effective,
            "allocated_tokens_per_sequence": primary_allocated,
            "full_precision_residual_tokens_per_sequence": (
                full_precision_tokens
            ),
            "quantized_tokens_per_sequence": quantized_tokens,
            "quantization_metadata_bytes": (
                metadata_bytes if strategy == "quantized" else None
            ),
            "base_dtype_logical_bytes": base_dtype_bytes,
            "storage_logical_bytes": storage_bytes,
            "allocated_bytes": allocated_bytes,
            "quantization_savings_bytes": (
                base_dtype_bytes - storage_bytes
            ),
            "reservation_overhead_bytes": (
                allocated_bytes - storage_bytes
            ),
            "allocation_utilization_percent": (
                storage_bytes / allocated_bytes * 100
                if allocated_bytes
                else None
            ),
            "full_attention": {
                "layers": full_layers,
                "effective_tokens_per_sequence": full_effective,
                "allocated_tokens_per_sequence": full_allocated,
                "allocated_bytes": full_allocated_bytes,
            },
            "sliding_attention": (
                {
                    "layers": sliding_layers,
                    "window_tokens": sliding_window_tokens,
                    "effective_tokens_per_sequence": sliding_effective,
                    "allocated_tokens_per_sequence": sliding_allocated,
                    "storage_logical_bytes": sliding_storage_bytes,
                    "allocated_bytes": sliding_allocated_bytes,
                }
                if sliding_layers
                else None
            ),
        }

    return {
        "enabled": use_cache,
        "strategy": strategy,
        "storage_bits_per_element": storage_bits,
        "quantized_residual_tokens": (
            quantized_residual_tokens
            if strategy == "quantized"
            else None
        ),
        "quantized_group_size": (
            quantized_group_size if strategy == "quantized" else None
        ),
        "compute_element_bytes": element_bytes,
        "batch_size": batch_size,
        "cache_layout": cache_layout.strip(),
        "elements_per_token_per_layer": (
            resolved_elements_per_token_per_layer
        ),
        "elements_per_token_per_sequence": (
            elements_per_token_per_sequence
        ),
        "bytes_per_token_per_sequence_at_compute_dtype": (
            elements_per_token_per_sequence * element_bytes
        ),
        "full_attention_bytes_per_token_per_sequence": (
            full_layers * resolved_elements_per_token_per_layer * element_bytes
        ),
        "attention_layers": {
            "full": full_layers,
            "sliding": sliding_layers,
            "configured_sliding": sliding_window_layers,
            "sliding_window_tokens": sliding_window_tokens,
            "sliding_window_accounting": sliding_reason,
        },
        "block_tokens": block_tokens if strategy == "paged" else None,
        "max_cache_tokens": (
            static_capacity_tokens if strategy == "static" else None
        ),
        "window_tokens": window_tokens,
        "prefill": phase_allocation(prompt_tokens, prefill=True),
        "generation": phase_allocation(final_logical_tokens, prefill=False),
        "formula": (
            "full_precision_residual_bytes + quantized_history_bytes + "
            "group_scale_and_zero_point_bytes"
            if strategy == "quantized"
            else (
                "ceil(layers * batch * tokens * cache_elements_per_layer "
                "* storage_bits / 8), with sliding-window layers holding "
                "their window"
                if sliding_layers
                else (
                    "ceil(layers * batch * tokens * cache_elements_per_layer "
                    "* storage_bits / 8)"
                )
            )
        ),
        "modeled_overheads": [
            {
                "static": "static_reservation",
                "paged": "paged_block_rounding",
                "quantized": "quantized_storage",
            }.get(strategy, "exact_dynamic_growth")
        ],
    }


def inspect_safetensors_checkpoint(
    model_dir: str | Path,
    *,
    files: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Read safetensors headers without loading tensor payloads."""

    root = Path(model_dir).expanduser().resolve()
    index_path = root / "model.safetensors.index.json"
    if files is not None:
        shard_names = sorted({str(value) for value in files})
        if not shard_names:
            raise ValueError("files must contain at least one safetensors file")
        for shard_name in shard_names:
            candidate = Path(shard_name)
            if candidate.is_absolute() or candidate.name != shard_name:
                raise ValueError(
                    "safetensors file names must be direct children "
                    f"of {root}"
                )
            if candidate.suffix != ".safetensors":
                raise ValueError(
                    f"not a safetensors file: {shard_name}"
                )
        indexed_total = 0
    elif index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"invalid safetensors weight map: {index_path}")
        shard_names = sorted({str(value) for value in weight_map.values()})
        metadata = index.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError(f"invalid safetensors index metadata: {index_path}")
        try:
            indexed_total = int(metadata.get("total_size", 0) or 0)
        except (TypeError, OverflowError) as exc:
            raise ValueError(
                f"invalid safetensors index total_size: {index_path}"
            ) from exc
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
            if dtype not in _DTYPE_BITS:
                raise ValueError(f"unsupported safetensors dtype {dtype!r} in {shard_path}")
            if not isinstance(shape, list) or any(
                not isinstance(dimension, int) or dimension < 0
                for dimension in shape
            ):
                raise ValueError(f"invalid shape for {tensor_name!r} in {shard_path}")
            numel = math.prod(shape)
            nbits = numel * _DTYPE_BITS[dtype]
            if nbits % 8:
                raise ValueError(
                    f"packed {dtype} tensor {tensor_name!r} in {shard_path} "
                    "does not end on a byte boundary"
                )
            nbytes = nbits // 8
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
    compute_precision: str = "fp32",
    kv_cache_strategy: str = "dynamic",
    kv_cache_bits: int = 4,
    kv_cache_residual_tokens: int = 128,
    kv_cache_block_tokens: int = 16,
    kv_cache_max_tokens: int | None = None,
    kv_cache_window_tokens: int | None = None,
    include_decode_steps: bool = True,
    logits_to_keep: int = 0,
    mla_kv_cache_layout: str = "auto",
    kv_cache_group_size: int = DEFAULT_KV_QUANTIZATION_GROUP_SIZE,
) -> dict[str, Any]:
    """Estimate decoder-only inference memory, communication, and FLOPs.

    The checkpoint is inspected through safetensors headers only. No model
    tensors are materialized and no CUDA context is created.
    logits_to_keep=0 projects every query token; a positive value projects
    only that many trailing tokens, as in last-token-only generation.
    Sliding-window layers from the model config keep only their window;
    kv_cache_window_tokens overrides that and caps every layer.
    mla_kv_cache_layout picks the MLA cache: decompressed per-head keys and
    values as Transformers caches them, or the latent plus RoPE key of paged
    backends; auto chooses latent only for the paged strategy.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if prompt_tokens <= 0:
        raise ValueError("prompt_tokens must be greater than zero")
    if generated_tokens <= 0:
        raise ValueError("generated_tokens must be greater than zero")
    if not isinstance(include_decode_steps, bool):
        raise ValueError("include_decode_steps must be a boolean")
    if (
        not isinstance(logits_to_keep, int)
        or isinstance(logits_to_keep, bool)
        or logits_to_keep < 0
    ):
        raise ValueError("logits_to_keep must be a non-negative integer")
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
    if not isinstance(config, dict):
        raise ValueError("model config must be an object")
    decoder_config = config.get("text_config", config)
    if not isinstance(decoder_config, dict):
        raise ValueError("text_config must be an object")
    _require_attention_only_decoder(decoder_config)
    checkpoint = inspect_safetensors_checkpoint(root)
    dimensions = _decoder_dimensions(decoder_config)
    sliding_layout = _sliding_window_layout(
        decoder_config
        if "model_type" in decoder_config
        else {**decoder_config, "model_type": config.get("model_type")}
    )
    if sliding_layout["sliding_attention_layers"]:
        dimensions["sliding_attention_layers"] = int(
            sliding_layout["sliding_attention_layers"]
        )
        dimensions["sliding_window_tokens"] = int(
            sliding_layout["sliding_window_tokens"]
        )
    cache_layout = _kv_cache_layout(
        dimensions,
        mla_layout=mla_kv_cache_layout,
        kv_cache_strategy=kv_cache_strategy,
    )
    has_vision_encoder = any(
        config.get(key) is not None
        for key in ("vision_config", "vision_tower", "mm_vision_tower")
    )
    if has_vision_encoder:
        dimensions["has_vision_encoder"] = True
    dtype_config = (
        decoder_config
        if any(decoder_config.get(key) is not None for key in ("dtype", "torch_dtype"))
        else config
    )
    auto_fallback = "float32"
    if dtype == "auto" and not any(
        dtype_config.get(key) for key in ("dtype", "torch_dtype")
    ):
        # transformers' dtype="auto" falls back to the checkpoint's weights.
        auto_fallback = _first_floating_checkpoint_dtype(root) or auto_fallback
    selected_dtype = _normalize_dtype(
        dtype,
        config=dtype_config,
        auto_fallback=auto_fallback,
    )
    element_bytes = _TORCH_DTYPE_TO_BYTES[selected_dtype]
    quantization = _quantization_summary(config, checkpoint)
    base_parameter_bytes = (
        int(checkpoint["checkpoint_bytes"])
        if quantization["enabled"]
        else int(checkpoint["parameter_count"]) * element_bytes
    )
    tied_embedding_storage = (
        None
        if quantization["enabled"]
        else _tied_embedding_storage(
            root,
            config=config,
            decoder_config=decoder_config,
            element_bytes=element_bytes,
        )
    )
    if tied_embedding_storage is not None:
        base_parameter_bytes -= tied_embedding_storage["removed_runtime_bytes"]
        quantization["storage_accounting"] = (
            "unique_runtime_parameters_times_compute_dtype"
        )
    adapters = _adapter_summary(
        adapter_dirs or (),
        element_bytes=element_bytes,
    )
    adapter_parameter_bytes = sum(
        int(adapter["runtime_parameter_bytes"]) for adapter in adapters
    )
    parameter_bytes = base_parameter_bytes + adapter_parameter_bytes

    kv_cache = estimate_kv_cache_memory(
        num_hidden_layers=int(dimensions["num_hidden_layers"]),
        num_key_value_heads=int(dimensions["num_key_value_heads"]),
        head_dim=int(dimensions["head_dim"]),
        batch_size=batch_size,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        element_bytes=element_bytes,
        strategy=kv_cache_strategy,
        quantized_bits=kv_cache_bits,
        quantized_residual_tokens=kv_cache_residual_tokens,
        block_tokens=kv_cache_block_tokens,
        max_cache_tokens=kv_cache_max_tokens,
        window_tokens=kv_cache_window_tokens,
        use_cache=use_cache,
        elements_per_token_per_layer=int(
            cache_layout["elements_per_token_per_layer"]
        ),
        cache_layout=str(cache_layout["layout"]),
        sliding_window_layers=int(
            dimensions.get("sliding_attention_layers", 0)
        ),
        sliding_window_tokens=dimensions.get("sliding_window_tokens"),
        quantized_group_size=kv_cache_group_size,
    )
    prefill_key_tokens = _effective_context_tokens(
        prompt_tokens,
        window_tokens=kv_cache_window_tokens,
    )
    prefill_flops = _forward_matmul_flops(
        dimensions,
        batch_size=batch_size,
        query_tokens=prompt_tokens,
        key_tokens=prefill_key_tokens,
        logits_to_keep=logits_to_keep,
    )
    decode_step_count = generated_tokens - 1
    decode_steps: list[dict[str, int]] = []
    decode_step_specs: list[tuple[int, int, int, int]] = []
    if include_decode_steps or not use_cache:
        for step in range(decode_step_count):
            logical_key_tokens = prompt_tokens + step + 1
            key_tokens = _effective_context_tokens(
                logical_key_tokens,
                window_tokens=kv_cache_window_tokens,
            )
            query_tokens = key_tokens if not use_cache else 1
            flops = _forward_matmul_flops(
                dimensions,
                batch_size=batch_size,
                query_tokens=query_tokens,
                key_tokens=key_tokens,
                logits_to_keep=logits_to_keep,
            )
            decode_step_specs.append(
                (logical_key_tokens, key_tokens, query_tokens, flops)
            )
            if include_decode_steps:
                decode_step = {
                    "step": step + 1,
                    "query_tokens": query_tokens,
                    "key_tokens": key_tokens,
                    "matmul_flops": flops,
                }
                if logical_key_tokens != key_tokens:
                    decode_step["logical_key_tokens"] = logical_key_tokens
                decode_steps.append(decode_step)

    decode_key_token_sum = (
        sum(spec[1] for spec in decode_step_specs)
        if not use_cache
        else _sum_decode_context_tokens(
            prompt_tokens,
            decode_step_count,
            window_tokens=kv_cache_window_tokens,
        )
    )
    if use_cache:
        decode_flops_at_zero_key = _forward_matmul_flops(
            dimensions,
            batch_size=batch_size,
            query_tokens=1,
            key_tokens=0,
        )
        decode_flops_per_key_token = (
            _forward_matmul_flops(
                dimensions,
                batch_size=batch_size,
                query_tokens=1,
                key_tokens=1,
            )
            - decode_flops_at_zero_key
        )
        total_decode_flops = (
            decode_step_count * decode_flops_at_zero_key
            + decode_key_token_sum * decode_flops_per_key_token
        )
    else:
        total_decode_flops = sum(spec[3] for spec in decode_step_specs)
    final_key_tokens = _effective_context_tokens(
        prompt_tokens + decode_step_count,
        window_tokens=kv_cache_window_tokens,
    )
    decode_query_tokens = 1 if use_cache else final_key_tokens
    kv_cache_prefill = int(kv_cache["prefill"]["allocated_bytes"])
    kv_cache_final = int(kv_cache["generation"]["allocated_bytes"])
    prefill_transient = _forward_transient_bytes(
        dimensions,
        batch_size=batch_size,
        query_tokens=prompt_tokens,
        key_tokens=prefill_key_tokens,
        element_bytes=element_bytes,
        attention_implementation=attention_implementation,
        logits_to_keep=logits_to_keep,
    )
    decode_transient = _forward_transient_bytes(
        dimensions,
        batch_size=batch_size,
        query_tokens=decode_query_tokens,
        key_tokens=final_key_tokens,
        element_bytes=element_bytes,
        attention_implementation=attention_implementation,
        logits_to_keep=logits_to_keep,
    )
    decode_input_bytes = batch_size * decode_query_tokens * 8
    if use_cache:
        decode_transient_sum = {
            key: value * decode_step_count
            for key, value in decode_transient.items()
        }
    else:
        decode_transient_sum = {
            key: 0
            for key in (
                "attention_bytes",
                "mlp_bytes",
                "dense_mlp_bytes",
                "routed_mlp_bytes",
                "router_bytes",
                "logits_bytes",
                "peak_bytes",
            )
        }
        for _, key_tokens, query_tokens, _ in decode_step_specs:
            transient = _forward_transient_bytes(
                dimensions,
                batch_size=batch_size,
                query_tokens=query_tokens,
                key_tokens=key_tokens,
                element_bytes=element_bytes,
                attention_implementation=attention_implementation,
                logits_to_keep=logits_to_keep,
            )
            for key, value in transient.items():
                decode_transient_sum[key] += value
    input_bytes = batch_size * prompt_tokens * 8
    prefill_peak = (
        parameter_bytes
        + input_bytes
        + kv_cache_prefill
        + prefill_transient["peak_bytes"]
    )
    decode_peak = (
        parameter_bytes
        + decode_input_bytes
        + kv_cache_final
        + decode_transient["peak_bytes"]
    )
    tensor_peak = max(prefill_peak, decode_peak)

    total_flops = prefill_flops + total_decode_flops
    communication = _expert_parallel_communication(
        dimensions,
        batch_size=batch_size,
        prompt_tokens=prompt_tokens,
        decode_forward_steps=decode_step_count,
        element_bytes=element_bytes,
        expert_parallel_size=expert_parallel_size,
    )
    memory_traffic = _inference_memory_traffic(
        parameter_bytes=parameter_bytes,
        prefill_cache_bytes=kv_cache_prefill,
        final_cache_bytes=kv_cache_final,
        prefill_transient=prefill_transient,
        decode_transient_sum=decode_transient_sum,
        decode_forward_steps=decode_step_count,
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
                forward_steps=1 + decode_step_count,
            ),
            compute_acceleration_factor=compute_acceleration_factor,
            compute_precision=compute_precision,
        )

    model_kind = str(dimensions["model_kind"])
    attention_kind = str(dimensions["attention_kind"])
    unmodeled_components = [
        "cuda_context_and_loaded_modules",
        "caching_allocator_fragmentation",
        *(VISION_UNMODELED_COMPONENTS if has_vision_encoder else ()),
        (
            "backend_fused_attention_workspace"
            if attention_implementation == "sdpa"
            else "backend_kernel_workspace"
        ),
        "quantized_kernel_workspace" if quantization["enabled"] else None,
        (
            "quantized_kv_cache_dequantization_workspace"
            if use_cache and kv_cache_strategy == "quantized"
            else None
        ),
        (
            "paged_kv_cache_block_table_and_scheduler"
            if use_cache and kv_cache_strategy == "paged"
            else None
        ),
        "adapter_kernel_fusion_and_merge_state" if adapters else None,
        "expert_load_imbalance" if model_kind == "mixture_of_experts" else None,
        (
            "mla_backend_projection_absorption_and_workspace"
            if attention_kind == "multi_head_latent"
            else None
        ),
    ]
    compute_report: dict[str, Any] = {
        "metric": "matrix_multiply_flops",
        "convention": "one multiply-add is two FLOPs",
        "prefill_flops": prefill_flops,
        "decode_steps": decode_steps,
        "decode_flops_total": total_decode_flops,
        "total_flops": total_flops,
        "model_kind": model_kind,
    }
    if not include_decode_steps:
        compute_report["decode_steps_included"] = False
        compute_report["decode_step_summary"] = {
            "count": decode_step_count,
            "first_key_tokens": (
                _effective_context_tokens(
                    prompt_tokens + 1,
                    window_tokens=kv_cache_window_tokens,
                )
                if decode_step_count
                else None
            ),
            "last_key_tokens": final_key_tokens if decode_step_count else None,
            "key_tokens_sum": decode_key_token_sum,
            "matmul_flops": total_decode_flops,
        }

    report = {
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
            "decode_forward_steps": decode_step_count,
            "use_cache": use_cache,
            "logits_to_keep": logits_to_keep,
            "attention_implementation": attention_implementation,
            "dtype": selected_dtype,
            "element_bytes": element_bytes,
            "expert_parallel_size": expert_parallel_size,
            "target_profile": target_profile,
            "compute_acceleration_factor": compute_acceleration_factor,
            "compute_precision": compute_precision,
            "kv_cache_strategy": kv_cache_strategy,
            "kv_cache_bits": (
                kv_cache_bits
                if kv_cache_strategy == "quantized"
                else None
            ),
            "kv_cache_residual_tokens": (
                kv_cache_residual_tokens
                if kv_cache_strategy == "quantized"
                else None
            ),
            "kv_cache_block_tokens": (
                kv_cache_block_tokens
                if kv_cache_strategy == "paged"
                else None
            ),
            "kv_cache_max_tokens": kv_cache["max_cache_tokens"],
            "kv_cache_window_tokens": kv_cache_window_tokens,
            "kv_cache_group_size": (
                kv_cache_group_size
                if kv_cache_strategy == "quantized"
                else None
            ),
            "mla_kv_cache_layout": cache_layout["mla_layout"],
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
        "kv_cache": kv_cache,
        "memory": {
            "base_parameter_bytes": base_parameter_bytes,
            "adapter_parameter_bytes": adapter_parameter_bytes,
            "parameter_bytes": parameter_bytes,
            "input_bytes": input_bytes,
            "decode_input_bytes": decode_input_bytes,
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
        "memory_timeline": {
            "unit": "bytes",
            "source": "checkpoint_and_decoder_shape_model",
            "phases": [
                {
                    "phase": "prefill",
                    "peak_bytes": prefill_peak + runtime_overhead_bytes,
                    "tensor_peak_bytes": prefill_peak,
                    "process_peak_bytes": prefill_peak + runtime_overhead_bytes,
                    "components": {
                        "parameters": parameter_bytes,
                        "inputs": input_bytes,
                        "kv_cache": kv_cache_prefill,
                        "transient": prefill_transient["peak_bytes"],
                        "runtime_overhead": runtime_overhead_bytes,
                    },
                },
                {
                    "phase": "decode",
                    "peak_bytes": decode_peak + runtime_overhead_bytes,
                    "tensor_peak_bytes": decode_peak,
                    "process_peak_bytes": decode_peak + runtime_overhead_bytes,
                    "components": {
                        "parameters": parameter_bytes,
                        "inputs": decode_input_bytes,
                        "kv_cache": kv_cache_final,
                        "transient": decode_transient["peak_bytes"],
                        "runtime_overhead": runtime_overhead_bytes,
                    },
                },
            ],
            "peak_bytes": tensor_peak + runtime_overhead_bytes,
            "peak_phase": (
                "prefill" if prefill_peak >= decode_peak else "decode"
            ),
            "virtual_smi_projection": {
                "tracked_process_memory_bytes": (
                    tensor_peak + runtime_overhead_bytes
                ),
                "runtime_overhead_calibrated": runtime_overhead_bytes > 0,
            },
        },
        "compute": compute_report,
        "communication": communication,
        "memory_traffic": memory_traffic,
        "performance": performance,
        "tracking_confidence": (
            "L3_checkpoint_aware_decoder_shape_model"
            if (
                quantization["enabled"]
                or adapters
                or model_kind == "mixture_of_experts"
                or attention_kind == "multi_head_latent"
            )
            else "L2_dense_decoder_shape_model"
        ),
        "unmodeled_components": [
            item for item in unmodeled_components if item is not None
        ],
        "notes": [
            "Checkpoint storage and parameter count come from safetensors headers without loading payloads.",
            "KV-cache bytes use the architecture-specific per-layer cache layout, sequence, batch, storage bit width, and allocation-strategy dimensions.",
            (
                (
                    "MLA cache storage keeps one compressed KV latent plus the decoupled RoPE key component per layer and token, as paged MLA backends do; backend-specific absorbed projections remain unmodeled."
                    if cache_layout["mla_layout"] == "latent"
                    else "MLA cache storage keeps decompressed per-head keys (qk_nope + qk_rope) and values (v_head_dim) per layer and token, as Transformers' DeepseekV3 attention caches them."
                )
                if attention_kind == "multi_head_latent"
                else "Standard attention cache storage keeps separate key and value tensors per KV head."
            ),
            "Quantized KV cache follows Transformers QuantizedCache: the prefill is quantized at once, at most residual_length-1 generated tokens stay at the compute dtype, and each q_group_size group keeps a compute-dtype scale and zero-point.",
            "Sliding-window layers from the model config hold only their window in the cache; a Transformers dynamic cache still keeps the whole prompt on those layers until the first decode step.",
            "Static reservation and paged block rounding are included; backend block-table and scheduler metadata remain implementation-specific.",
            "Quantized base-weight memory uses exact safetensors payload bytes, including scale and metadata tensors.",
            "Adapter runtime bytes use adapter parameter count and the selected compute dtype.",
            "FLOPs cover projection, attention matrix multiplication, active dense or routed experts, and LM-head matrix multiplication.",
            "LM-head FLOPs and logits storage use logits_to_keep trailing tokens per forward (0 means all); Transformer layers still process all query tokens.",
            "Elementwise normalization, activation, softmax, rotary embedding, and sampling operations are not included in matrix FLOPs.",
            "The eager-attention transient estimate is conservative; SDPA workspace remains backend-specific.",
            "Expert-parallel traffic assumes a uniform token distribution; hot experts can increase per-rank peaks.",
        ],
    }
    if not include_decode_steps:
        report["inputs"]["include_decode_steps"] = False
    if tied_embedding_storage is not None:
        report["weight_storage"]["tied_embedding_storage"] = tied_embedding_storage
        report["notes"].append(
            "Runtime storage counts the declared tied input/output embedding once; "
            "checkpoint storage still includes both serialized tensors."
        )
    if has_vision_encoder:
        report["notes"].append(VISION_ESTIMATE_WARNING)
    return report


def _tied_embedding_storage(
    root: Path,
    *,
    config: Mapping[str, Any],
    decoder_config: Mapping[str, Any],
    element_bytes: int,
) -> dict[str, Any] | None:
    """Account for an explicitly tied canonical pair without loading tensors."""
    if (
        config.get("model_type") not in _CANONICAL_TIED_EMBEDDING_MODELS
        or decoder_config.get("tie_word_embeddings") is not True
        or config.get("torchscript")
        or decoder_config.get("torchscript")
    ):
        return None
    input_name, output_name = "model.embed_tokens.weight", "lm_head.weight"
    names = (input_name, output_name)
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        if any(name not in weight_map for name in names):
            return None
        paths = [root / name for name in sorted({weight_map[n] for n in names})]
    else:
        paths = sorted(root.glob("*.safetensors"))
    tensors = {}
    for path in paths:
        header = _read_safetensors_header(path)
        for name in names:
            if name in header:
                if name in tensors:
                    return None
                tensors[name] = header[name]
    if len(tensors) != 2:
        return None
    input_tensor, output_tensor = (tensors[name] for name in names)
    expected_shape = [decoder_config["vocab_size"], decoder_config["hidden_size"]]
    if (
        input_tensor["shape"] != expected_shape
        or output_tensor["shape"] != expected_shape
        or input_tensor["dtype"] != output_tensor["dtype"]
        or input_tensor["dtype"] not in {"F16", "BF16", "F32", "F64"}
    ):
        return None
    removed_parameters = math.prod(expected_shape)
    return {
        "input_name": input_name,
        "output_name": output_name,
        "removed_parameter_count": removed_parameters,
        "removed_runtime_bytes": removed_parameters * element_bytes,
        "source": "explicit_tie_word_embeddings_and_matching_checkpoint_headers",
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


def _require_attention_only_decoder(config: Mapping[str, Any]) -> None:
    """Keep inference-only assumptions out of shared fine-tuning dimensions."""
    recurrent_fields = (
        "linear_num_key_heads", "linear_num_value_heads",
        "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim",
    )
    layers = config.get("layer_types")
    if any(field in config for field in recurrent_fields) or (
        layers is not None and (
            not isinstance(layers, list)
            or len(layers) != config.get("num_hidden_layers")
            or any(
                not isinstance(kind, str)
                or kind not in {"full_attention", "sliding_attention"}
                for kind in layers
            )
        )
    ):
        raise ValueError(
            "inference estimates do not support recurrent or linear attention; "
            "KV/state memory and FLOPs require an architecture-specific model"
        )


def _sliding_window_layout(config: Mapping[str, Any]) -> dict[str, Any]:
    """Count the sliding-window layers the way Transformers configs resolve them.

    Follows the 4.57 config classes: explicit ``layer_types`` win; Qwen-style
    configs keep full attention unless ``use_sliding_window`` is set and then
    slide from ``max_window_layers`` on; ``sliding_window_pattern`` (or the
    model-type default) marks every pattern-th layer as full attention; any
    other config with a ``sliding_window`` slides on every layer, as
    ``DynamicCache``/``StaticCache`` do when ``layer_types`` is absent.
    """

    layer_count = int(config["num_hidden_layers"])
    raw_window = config.get("sliding_window")
    window = (
        int(raw_window)
        if isinstance(raw_window, int)
        and not isinstance(raw_window, bool)
        and raw_window > 0
        else None
    )
    layer_types = config.get("layer_types")
    if isinstance(layer_types, list):
        sliding = sum(kind == "sliding_attention" for kind in layer_types)
        source = "layer_types"
    elif (
        window is None
        or config.get("use_sliding_window") is False
        # Qwen-style configs (with max_window_layers) default
        # use_sliding_window to False and then drop sliding_window.
        or (
            config.get("use_sliding_window") is None
            and "max_window_layers" in config
        )
    ):
        sliding = 0
        source = "none"
    elif config.get("use_sliding_window") is True and isinstance(
        config.get("max_window_layers"), int
    ):
        start = max(0, int(config["max_window_layers"]))
        sliding = max(0, layer_count - start)
        source = "max_window_layers"
    else:
        pattern = config.get("sliding_window_pattern")
        source = "sliding_window_pattern"
        if not isinstance(pattern, int) or isinstance(pattern, bool):
            pattern = _SLIDING_WINDOW_PATTERN_DEFAULTS.get(
                str(config.get("model_type"))
            )
            source = "model_type_pattern_default"
        if isinstance(pattern, int) and pattern > 0:
            sliding = sum(
                1 for index in range(layer_count) if (index + 1) % pattern
            )
        else:
            sliding = layer_count
            source = "sliding_window_on_every_layer"
    if sliding and window is None:
        raise ValueError(
            "sliding_attention layers require a positive sliding_window"
        )
    return {
        "sliding_attention_layers": sliding,
        "sliding_window_tokens": window if sliding else None,
        "source": source,
    }


def _kv_cache_layout(
    dimensions: Mapping[str, Any],
    *,
    mla_layout: str,
    kv_cache_strategy: str,
) -> dict[str, Any]:
    """Pick the cache layout a runtime stores for these attention dimensions.

    Transformers' DeepseekV3 attention caches decompressed per-head keys
    (qk_nope + qk_rope) and values (v_head_dim); paged MLA backends such as
    vLLM's cache one latent plus the RoPE key. ``auto`` follows the
    strategy: paged means latent, the Transformers caches mean decompressed.
    """

    if mla_layout not in MLA_KV_CACHE_LAYOUTS:
        choices = ", ".join(sorted(MLA_KV_CACHE_LAYOUTS))
        raise ValueError(f"mla_kv_cache_layout must be one of: {choices}")
    if dimensions.get("attention_kind") != "multi_head_latent":
        return {
            "layout": str(dimensions["kv_cache_layout"]),
            "elements_per_token_per_layer": int(
                dimensions["kv_cache_elements_per_token_per_layer"]
            ),
            "mla_layout": None,
        }
    resolved = mla_layout
    if resolved == "auto":
        resolved = "latent" if kv_cache_strategy == "paged" else "decompressed"
    if resolved == "latent":
        return {
            "layout": str(dimensions["kv_cache_layout"]),
            "elements_per_token_per_layer": int(
                dimensions["kv_cache_elements_per_token_per_layer"]
            ),
            "mla_layout": "latent",
        }
    return {
        "layout": "decompressed_per_head_key_value",
        "elements_per_token_per_layer": int(dimensions["num_attention_heads"])
        * (
            int(dimensions["qk_nope_head_dim"])
            + int(dimensions["qk_rope_head_dim"])
            + int(dimensions["v_head_dim"])
        ),
        "mla_layout": "decompressed",
    }


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
    mla_keys = (
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
    )
    has_mla_configuration = any(
        config.get(key) is not None for key in mla_keys
    )
    if has_mla_configuration:
        mla_values = {
            key: _first_positive_integer(config, (key,))
            for key in mla_keys
        }
        missing_mla = [
            key for key, value in mla_values.items() if value <= 0
        ]
        if missing_mla:
            raise ValueError(
                "multi-head latent attention requires positive: "
                + ", ".join(missing_mla)
            )
        kv_lora_rank = mla_values["kv_lora_rank"]
        qk_nope_head_dim = mla_values["qk_nope_head_dim"]
        qk_rope_head_dim = mla_values["qk_rope_head_dim"]
        v_head_dim = mla_values["v_head_dim"]
        q_lora_rank = _first_positive_integer(
            config,
            ("q_lora_rank",),
        ) or None
        head_dim = qk_nope_head_dim + qk_rope_head_dim
        kv_heads = 1
        attention_kind = "multi_head_latent"
        kv_cache_layout = "compressed_latent_and_rope"
        kv_cache_elements_per_token_per_layer = (
            kv_lora_rank + qk_rope_head_dim
        )
    else:
        if hidden_size % attention_heads and "head_dim" not in config:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads when "
                "head_dim is not configured"
            )
        head_dim = int(
            config.get("head_dim") or (hidden_size // attention_heads)
        )
        kv_heads = int(
            config.get("num_key_value_heads") or attention_heads
        )
        kv_lora_rank = 0
        q_lora_rank = None
        qk_nope_head_dim = head_dim
        qk_rope_head_dim = 0
        v_head_dim = head_dim
        attention_kind = (
            "multi_head"
            if kv_heads == attention_heads
            else "multi_query"
            if kv_heads == 1
            else "grouped_query"
        )
        kv_cache_layout = "separate_key_value"
        kv_cache_elements_per_token_per_layer = (
            2 * kv_heads * head_dim
        )
    if head_dim <= 0:
        raise ValueError("head_dim must be greater than zero")
    if kv_heads <= 0 or kv_heads > attention_heads:
        raise ValueError(
            "num_key_value_heads must be between one and "
            "num_attention_heads"
        )

    # Llama 4 dense layers use intermediate_size_mlp, while its experts and
    # always-present shared expert use intermediate_size.
    llama4_text = config.get("model_type") == "llama4_text"
    dense_intermediate_size = (
        _first_positive_integer(config, ("intermediate_size_mlp",))
        if llama4_text
        else 0
    ) or int(required["intermediate_size"])
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
            ) or int(llama4_text)
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
        "attention_kind": attention_kind,
        "kv_cache_layout": kv_cache_layout,
        "kv_cache_elements_per_token_per_layer": (
            kv_cache_elements_per_token_per_layer
        ),
        "q_lora_rank": q_lora_rank,
        "kv_lora_rank": kv_lora_rank,
        "qk_nope_head_dim": qk_nope_head_dim,
        "qk_rope_head_dim": qk_rope_head_dim,
        "v_head_dim": v_head_dim,
        "intermediate_size": dense_intermediate_size,
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
    # Llama 4 lists its sparse layers explicitly; transformers derives the
    # list from interleave_moe_layer_step when it is absent.
    moe_layers = config.get("moe_layers")
    if isinstance(moe_layers, list):
        return len(
            {
                value
                for value in moe_layers
                if isinstance(value, int)
                and not isinstance(value, bool)
                and 0 <= value < layer_count
            }
        )
    interleave_step = _first_positive_integer(
        config,
        ("interleave_moe_layer_step",),
    )
    if interleave_step:
        return len(range(interleave_step - 1, layer_count, interleave_step))
    mlp_only_layers = config.get("mlp_only_layers")
    dense_layers = (
        {
            value
            for value in mlp_only_layers
            if isinstance(value, int)
            and not isinstance(value, bool)
            and 0 <= value < layer_count
        }
        if isinstance(mlp_only_layers, list)
        else set()
    )
    sparse_step = config.get("decoder_sparse_step")
    if sparse_step is None:
        sparse_step = 1
    elif (
        not isinstance(sparse_step, int)
        or isinstance(sparse_step, bool)
        or sparse_step <= 0
    ):
        raise ValueError("decoder_sparse_step must be a positive integer")
    first_dense = config.get("first_k_dense_replace")
    if first_dense is None:
        first_dense = 0
    elif (
        not isinstance(first_dense, int)
        or isinstance(first_dense, bool)
        or first_dense < 0
    ):
        raise ValueError("first_k_dense_replace must be a non-negative integer")
    return sum(
        1
        for index in range(first_dense, layer_count)
        if index not in dense_layers and (index + 1) % sparse_step == 0
    )


def _first_floating_checkpoint_dtype(root: Path) -> str | None:
    """Mirror transformers' dtype="auto" fallback for configs without dtype.

    transformers reads the first shard and returns the dtype of the first
    floating, non-FP8 tensor in safetensors' sorted key order.
    """
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        first_shard = root / sorted({str(name) for name in weight_map.values()})[0]
    else:
        first_shard = sorted(root.glob("*.safetensors"))[0]
    floating = {"F16": "float16", "BF16": "bfloat16", "F32": "float32", "F64": "float64"}
    header = _read_safetensors_header(first_shard)
    for name in sorted(header):
        metadata = header[name]
        if name != "__metadata__" and metadata.get("dtype") in floating:
            return floating[metadata["dtype"]]
    return None


def _normalize_dtype(
    dtype: str,
    *,
    config: dict[str, Any],
    auto_fallback: str = "float32",
) -> str:
    value = (
        str(config.get("dtype") or config.get("torch_dtype") or auto_fallback)
        if dtype == "auto"
        else str(dtype)
    )
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
    logits_to_keep: int = 0,
) -> int:
    hidden = dimensions["hidden_size"]
    heads = dimensions["num_attention_heads"]
    kv_width = dimensions["num_key_value_heads"] * dimensions["head_dim"]
    intermediate = dimensions["intermediate_size"]
    vocab = dimensions["vocab_size"]
    token_count = batch_size * query_tokens
    if dimensions["attention_kind"] == "multi_head_latent":
        qk_head_dim = dimensions["head_dim"]
        q_lora_rank = dimensions["q_lora_rank"]
        kv_lora_rank = dimensions["kv_lora_rank"]
        q_projection = (
            2 * token_count * hidden * heads * qk_head_dim
            if q_lora_rank is None
            else (
                2 * token_count * hidden * q_lora_rank
                + 2 * token_count * q_lora_rank * heads * qk_head_dim
            )
        )
        kv_a_projection = (
            2
            * token_count
            * hidden
            * (
                kv_lora_rank + dimensions["qk_rope_head_dim"]
            )
        )
        kv_b_projection = (
            2
            * token_count
            * kv_lora_rank
            * heads
            * (
                dimensions["qk_nope_head_dim"]
                + dimensions["v_head_dim"]
            )
        )
        output_projection = (
            2
            * token_count
            * heads
            * dimensions["v_head_dim"]
            * hidden
        )
        attention_projection_per_layer = (
            q_projection
            + kv_a_projection
            + kv_b_projection
            + output_projection
        )
        attention_per_layer = (
            2
            * batch_size
            * heads
            * query_tokens
            * key_tokens
            * (qk_head_dim + dimensions["v_head_dim"])
        )
    else:
        # q_proj and o_proj span heads * head_dim, which differs from
        # hidden_size when the config overrides head_dim.
        query_width = heads * dimensions["head_dim"]
        attention_projection_per_layer = (
            4 * token_count * hidden * query_width
            + 4 * token_count * hidden * kv_width
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
    logits_tokens = min(query_tokens, logits_to_keep) if logits_to_keep else query_tokens
    lm_head = 2 * batch_size * logits_tokens * hidden * vocab
    return (
        attention_flops
        + dense_mlp_flops
        + moe_mlp_flops
        + shared_expert_flops
        + router_flops
        + lm_head
    )


def _effective_context_tokens(
    tokens: int,
    *,
    window_tokens: int | None,
) -> int:
    return (
        min(tokens, window_tokens)
        if window_tokens is not None
        else tokens
    )


def _sum_decode_context_tokens(
    prompt_tokens: int,
    decode_step_count: int,
    *,
    window_tokens: int | None,
) -> int:
    """Sum effective KV context lengths without materializing each step."""

    count = decode_step_count
    first = prompt_tokens + 1
    if window_tokens is None:
        return count * first + count * (count - 1) // 2
    uncapped = min(count, max(0, window_tokens - first + 1))
    uncapped_sum = uncapped * first + uncapped * (uncapped - 1) // 2
    return uncapped_sum + (count - uncapped) * window_tokens


def _kv_storage_bytes(
    elements_per_token_per_sequence: int,
    *,
    batch_size: int,
    tokens: int,
    storage_bits: int,
) -> int:
    total_bits = (
        elements_per_token_per_sequence
        * batch_size
        * tokens
        * storage_bits
    )
    return (total_bits + 7) // 8


def _forward_transient_bytes(
    dimensions: Mapping[str, Any],
    *,
    batch_size: int,
    query_tokens: int,
    key_tokens: int,
    element_bytes: int,
    attention_implementation: str,
    logits_to_keep: int = 0,
) -> dict[str, int]:
    hidden = dimensions["hidden_size"]
    heads = dimensions["num_attention_heads"]
    head_dim = dimensions["head_dim"]
    intermediate = dimensions["intermediate_size"]
    vocab = dimensions["vocab_size"]
    hidden_bytes = batch_size * query_tokens * hidden * element_bytes
    q_bytes = batch_size * query_tokens * heads * head_dim * element_bytes
    kv_bytes = (
        batch_size
        * key_tokens
        * dimensions["kv_cache_elements_per_token_per_layer"]
        * element_bytes
    )
    attention_output_bytes = (
        batch_size
        * query_tokens
        * heads
        * dimensions["v_head_dim"]
        * element_bytes
    )
    score_bytes = batch_size * heads * query_tokens * key_tokens * element_bytes
    attention_bytes = (
        hidden_bytes + q_bytes + kv_bytes + attention_output_bytes
    )
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
    logits_tokens = min(query_tokens, logits_to_keep) if logits_to_keep else query_tokens
    logits_bytes = hidden_bytes + batch_size * logits_tokens * vocab * element_bytes
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
        dtype for dtype in dtype_bytes if dtype in _LOW_PRECISION_STORAGE_DTYPES
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
        elif config.get("load_in_8bit"):
            bits = 8
        elif quantized_storage_dtypes:
            bits = min(_DTYPE_BITS[dtype] for dtype in quantized_storage_dtypes)

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
    decode_transient_sum: Mapping[str, int],
    decode_forward_steps: int,
) -> dict[str, Any]:
    forward_steps = 1 + decode_forward_steps
    weight_reads = parameter_bytes * forward_steps
    kv_traffic = prefill_cache_bytes + final_cache_bytes * decode_forward_steps
    transient_peak_sum = int(prefill_transient["peak_bytes"]) + int(
        decode_transient_sum["peak_bytes"]
    )
    lower = weight_reads + kv_traffic
    expected = lower + 2 * transient_peak_sum
    upper = lower + 4 * sum(
        int(prefill_transient[key]) + int(decode_transient_sum[key])
        for key in ("attention_bytes", "mlp_bytes", "logits_bytes")
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
