"""Per-layer KV accounting: sliding windows, MLA layouts, quantized caches.

Expected values follow Transformers 4.57.3 (``cache_utils.py`` and the
config classes) and optimum-quanto / HQQ quantizers.
"""

from __future__ import annotations

import pytest

from fakegpu.llm_estimator import (
    _sliding_window_layout,
    estimate_decoder_inference,
    estimate_kv_cache_memory,
)
from test_llm_estimator import _write_model


# 34 layers with every sixth layer full attention, like Gemma 3's default
# sliding_window_pattern: 5 full and 29 sliding-window layers.
GEMMA_LIKE_LAYER_TYPES = [
    "full_attention" if (index + 1) % 6 == 0 else "sliding_attention"
    for index in range(34)
]
# Bytes per token per layer: 2 (K and V) * 1 KV head * head_dim 4 * bf16.
TOKEN_LAYER_BYTES = 2 * 1 * 4 * 2


def _write_gemma_like(root, **overrides):
    _write_model(
        root,
        config_overrides={
            "model_type": "gemma3_text",
            "num_hidden_layers": 34,
            "layer_types": GEMMA_LIKE_LAYER_TYPES,
            "sliding_window": 64,
            **overrides,
        },
    )


@pytest.mark.parametrize(
    ("config", "sliding", "source"),
    [
        (
            {
                "num_hidden_layers": 4,
                "sliding_window": 8,
                "layer_types": ["sliding_attention", "full_attention"] * 2,
            },
            2,
            "layer_types",
        ),
        # Qwen2/Qwen3 configs drop sliding_window unless use_sliding_window,
        # and use_sliding_window defaults to False.
        (
            {
                "num_hidden_layers": 4,
                "sliding_window": 8,
                "use_sliding_window": False,
                "max_window_layers": 2,
            },
            0,
            "none",
        ),
        (
            {
                "num_hidden_layers": 4,
                "sliding_window": 8,
                "max_window_layers": 2,
            },
            0,
            "none",
        ),
        # Layers from max_window_layers on slide.
        (
            {
                "num_hidden_layers": 4,
                "sliding_window": 8,
                "use_sliding_window": True,
                "max_window_layers": 1,
            },
            3,
            "max_window_layers",
        ),
        # Gemma3TextConfig: sliding_window_pattern defaults to 6.
        (
            {
                "model_type": "gemma3_text",
                "num_hidden_layers": 12,
                "sliding_window": 8,
            },
            10,
            "model_type_pattern_default",
        ),
        (
            {
                "model_type": "custom",
                "num_hidden_layers": 8,
                "sliding_window": 8,
                "sliding_window_pattern": 4,
            },
            6,
            "sliding_window_pattern",
        ),
        # DynamicCache without layer_types slides on every layer.
        (
            {
                "model_type": "mistral",
                "num_hidden_layers": 4,
                "sliding_window": 8,
            },
            4,
            "sliding_window_on_every_layer",
        ),
        (
            {
                "model_type": "mistral",
                "num_hidden_layers": 4,
                "sliding_window": None,
            },
            0,
            "none",
        ),
    ],
)
def test_sliding_window_layers_follow_transformers_configs(
    config,
    sliding,
    source,
):
    layout = _sliding_window_layout(config)

    assert layout["sliding_attention_layers"] == sliding
    assert layout["source"] == source
    assert layout["sliding_window_tokens"] == (8 if sliding else None)


def test_dynamic_sliding_layers_keep_prompt_then_window(tmp_path):
    # DynamicSlidingWindowLayer keeps a view of the concatenated states, so
    # the prefill retains the whole prompt and each decode step leaves
    # min(tokens, window) behind.
    model_dir = tmp_path / "gemma-like"
    _write_gemma_like(model_dir)

    report = estimate_decoder_inference(
        model_dir,
        prompt_tokens=100,
        generated_tokens=3,
        kv_cache_strategy="dynamic",
    )

    kv_cache = report["kv_cache"]
    assert kv_cache["attention_layers"]["full"] == 5
    assert kv_cache["attention_layers"]["sliding"] == 29
    assert kv_cache["prefill"]["allocated_bytes"] == (
        34 * 100 * TOKEN_LAYER_BYTES
    )
    assert kv_cache["generation"]["sliding_attention"][
        "effective_tokens_per_sequence"
    ] == 64
    assert kv_cache["generation"]["allocated_bytes"] == (
        5 * 102 * TOKEN_LAYER_BYTES + 29 * 64 * TOKEN_LAYER_BYTES
    )


def test_window_override_and_quantized_cache_apply_to_every_layer(tmp_path):
    model_dir = tmp_path / "gemma-like"
    _write_gemma_like(model_dir)

    override = estimate_decoder_inference(
        model_dir,
        prompt_tokens=100,
        generated_tokens=3,
        kv_cache_window_tokens=32,
    )
    quantized = estimate_decoder_inference(
        model_dir,
        prompt_tokens=100,
        generated_tokens=3,
        kv_cache_strategy="quantized",
    )

    assert override["kv_cache"]["attention_layers"]["sliding"] == 0
    assert override["kv_cache"]["generation"]["allocated_bytes"] == (
        34 * 32 * TOKEN_LAYER_BYTES
    )
    # QuantizedCache builds a QuantizedLayer for every layer.
    assert quantized["kv_cache"]["attention_layers"]["sliding"] == 0
    assert quantized["kv_cache"]["attention_layers"][
        "sliding_window_accounting"
    ] == "quantized_cache_keeps_full_history_per_layer"


def test_paged_sliding_layers_hold_window_plus_scheduled_tokens():
    # vLLM frees blocks outside the window: decode keeps window-1 tokens plus
    # the scheduled one, and one extra block for an unaligned window start.
    report = estimate_kv_cache_memory(
        num_hidden_layers=3,
        num_key_value_heads=1,
        head_dim=4,
        batch_size=1,
        prompt_tokens=200,
        generated_tokens=11,
        element_bytes=2,
        strategy="paged",
        block_tokens=16,
        sliding_window_layers=2,
        sliding_window_tokens=64,
        prefill_chunk_tokens=32,
    )

    prefill = report["prefill"]["sliding_attention"]
    generation = report["generation"]["sliding_attention"]
    assert prefill["effective_tokens_per_sequence"] == 63 + 32
    assert prefill["allocated_tokens_per_sequence"] == (6 + 1) * 16
    assert generation["effective_tokens_per_sequence"] == 64
    assert generation["allocated_tokens_per_sequence"] == (4 + 1) * 16
    assert report["generation"]["full_attention"][
        "allocated_tokens_per_sequence"
    ] == 224


def test_quantized_residual_is_bounded_by_residual_length():
    # QuantizedLayer requantizes once the residual reaches
    # residual_length - 1 tokens, so at most that many stay unquantized.
    report = estimate_kv_cache_memory(
        num_hidden_layers=2,
        num_key_value_heads=1,
        head_dim=32,
        batch_size=1,
        prompt_tokens=4,
        generated_tokens=10,
        element_bytes=2,
        strategy="quantized",
        quantized_bits=2,
        quantized_residual_tokens=4,
        quantized_group_size=32,
    )

    generation = report["generation"]
    assert generation["full_precision_residual_tokens_per_sequence"] == 3
    assert generation["quantized_tokens_per_sequence"] == 10
    quantized_elements = 2 * 64 * 10
    assert generation["quantization_metadata_bytes"] == (
        2 * 2 * quantized_elements // 32
    )
    assert generation["storage_logical_bytes"] == (
        quantized_elements * 2 // 8
        + 2 * 2 * quantized_elements // 32
        + 2 * 64 * 3 * 2
    )


def test_mla_layout_option_is_validated(tmp_path):
    model_dir = tmp_path / "model"
    _write_model(model_dir)

    with pytest.raises(ValueError, match="mla_kv_cache_layout"):
        estimate_decoder_inference(
            model_dir,
            prompt_tokens=4,
            mla_kv_cache_layout="compressed",
        )
