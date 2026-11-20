"""vLLM startup model: batch defaults, profile run, KV groups and dtypes.

Expected values follow the vLLM 0.19.1 sources: ``engine/arg_utils.py``
(get_batch_defaults), ``v1/worker/gpu_model_runner.py`` (profile_run,
_dummy_sampler_run), ``v1/sample/ops/topk_topp_triton.py``,
``v1/core/kv_cache_utils.py`` (KV cache groups, logged size, concurrency)
and ``v1/kv_cache_interface.py`` (SlidingWindowSpec). Budgets are
synthetic.
"""

from __future__ import annotations

import json

import pytest

from fakegpu._serving_vllm import (
    VLLM_MODELED_NON_TORCH_MEMORY_BYTES,
    _triton_top_k_top_p_rows,
    resolve_vllm_batch_limits,
    vllm_default_block_tokens,
)
from fakegpu.serving_plan import (
    ServingPlanError,
    estimate_serving_plan,
    main as serving_main,
)
from test_kv_cache_layers import TOKEN_LAYER_BYTES, _write_gemma_like
from test_llm_estimator import _write_model


GIB = 2**30


@pytest.mark.parametrize(
    ("memory", "name", "context", "tokens", "seqs"),
    [
        (32 * GIB, None, "llm_class", 8192, 256),
        (32 * GIB, None, "api_server", 2048, 256),
        (80 * GIB, "Example Large GPU", "llm_class", 16384, 1024),
        (80 * GIB, "Example Large GPU", "api_server", 8192, 1024),
        # Devices named A100 keep the small table at any memory size.
        (80 * GIB, "NVIDIA A100-SXM4-80GB", "llm_class", 8192, 256),
        (None, None, "llm_class", 8192, 256),
    ],
)
def test_vllm_batch_defaults_follow_device_memory_and_usage_context(
    memory,
    name,
    context,
    tokens,
    seqs,
):
    limits = resolve_vllm_batch_limits(
        max_num_batched_tokens=None,
        max_num_seqs=None,
        max_model_len_tokens=32768,
        device_memory_bytes=memory,
        device_name=name,
        usage_context=context,
    )

    assert limits["max_num_batched_tokens"] == tokens
    assert limits["max_num_seqs"] == seqs
    assert limits["max_num_batched_tokens_source"] == "vllm_default"


def test_vllm_plan_defaults_to_api_server_batch_limits(tmp_path):
    model_dir = tmp_path / "model"
    _write_model(model_dir)

    report = estimate_serving_plan(
        model_dir,
        active_sequences=1,
        prompt_tokens=4000,
        generated_tokens=96,
        runtime="vllm",
        device_capacity_bytes=GIB,
    )

    configuration = report["runtime"]["configuration"]
    assert configuration["usage_context"] == "api_server"
    assert configuration["max_num_seqs"] == 256
    assert configuration["max_num_batched_tokens"] == 2048
    assert report["scheduler"]["configured_max_batch_size"] == 256


def test_vllm_batch_defaults_cap_tokens_and_sequences():
    def limits(**kwargs):
        return resolve_vllm_batch_limits(
            device_memory_bytes=32 * GIB,
            device_name=None,
            usage_context="llm_class",
            **kwargs,
        )

    # Chunked prefill caps the token budget at max_num_seqs * max_model_len.
    short = limits(
        max_num_batched_tokens=None,
        max_num_seqs=None,
        max_model_len_tokens=16,
    )
    assert short["max_num_batched_tokens"] == 256 * 16
    # A default sequence limit never exceeds the token budget.
    small_budget = limits(
        max_num_batched_tokens=100,
        max_num_seqs=None,
        max_model_len_tokens=4096,
    )
    assert small_budget["max_num_seqs"] == 100
    with pytest.raises(ServingPlanError, match="max_num_seqs"):
        limits(
            max_num_batched_tokens=32,
            max_num_seqs=64,
            max_model_len_tokens=4096,
        )


@pytest.mark.parametrize(
    ("rows", "sm_count", "expected"),
    [
        (100, 108, 108),
        (100, 96, 96),
        (40, 108, 64),
        (16, None, 16),
        (5, None, 8),
    ],
)
def test_triton_top_k_top_p_buffer_rows(rows, sm_count, expected):
    assert _triton_top_k_top_p_rows(rows, sm_count) == expected


@pytest.mark.parametrize(
    ("dimensions", "capability", "expected"),
    [
        ({"attention_kind": "grouped_query"}, "10.0", 16),
        (
            {"attention_kind": "multi_head_latent", "qk_nope_head_dim": 128},
            "8.0",
            16,
        ),
        (
            {"attention_kind": "multi_head_latent", "qk_nope_head_dim": 128},
            "9.0",
            16,
        ),
        # Compute capability 10.x: FlashInfer MLA, else CUTLASS MLA.
        (
            {"attention_kind": "multi_head_latent", "qk_nope_head_dim": 128},
            "10.0",
            32,
        ),
        (
            {"attention_kind": "multi_head_latent", "qk_nope_head_dim": 96},
            "10.0",
            128,
        ),
        (
            {"attention_kind": "multi_head_latent", "qk_nope_head_dim": 128},
            None,
            16,
        ),
    ],
)
def test_vllm_default_block_size_follows_mla_backend(
    dimensions,
    capability,
    expected,
):
    assert vllm_default_block_tokens(
        dimensions,
        compute_capability=capability,
    ) == expected


def test_vllm_profile_run_models_forward_and_sampler(tmp_path):
    model_dir = tmp_path / "model"
    _write_model(model_dir)

    report = estimate_serving_plan(
        model_dir,
        active_sequences=1,
        max_batch_size=16,
        prompt_tokens=60,
        generated_tokens=4,
        runtime="vllm",
        device_capacity_bytes=GIB,
        vllm_max_num_batched_tokens=64,
    )

    runtime = report["runtime"]
    profile = runtime["memory"]["profile"]
    tokens, rows, hidden, vocab = 64, 16, 8, 32
    assert runtime["configuration"]["max_num_batched_tokens"] == tokens
    assert runtime["configuration"]["profiled_sequence_count"] == rows
    assert profile["sampler_components"] == {
        "hidden_states": tokens * hidden * 2,
        "sampled_hidden_states": 2 * rows * hidden * 2,
        "logits": rows * vocab * 2,
        "float32_logits_probs_and_noise": 3 * rows * vocab * 4,
        # No SM count without a target profile: next_pow2(rows) rows.
        "top_k_top_p_workspace": 16 * vocab * 4,
    }
    assert profile["non_torch_memory_bytes"] == (
        VLLM_MODELED_NON_TORCH_MEMORY_BYTES
    )
    assert runtime["memory"]["non_kv_cache_memory_bytes"] == (
        report["memory"]["parameter_bytes"]
        + VLLM_MODELED_NON_TORCH_MEMORY_BYTES
        + max(profile["forward_activation_bytes"], profile["sampler_bytes"])
    )
    assert "vllm_cuda_graph_capture_memory" in report[
        "unmodeled_components"
    ]


def test_vllm_profile_run_uses_sort_path_below_eight_requests(tmp_path):
    model_dir = tmp_path / "model"
    _write_model(model_dir)

    report = estimate_serving_plan(
        model_dir,
        active_sequences=1,
        max_batch_size=4,
        prompt_tokens=60,
        generated_tokens=4,
        runtime="vllm",
        device_capacity_bytes=GIB,
        vllm_max_num_batched_tokens=64,
    )

    components = report["runtime"]["memory"]["profile"]["sampler_components"]
    assert components["float32_logits_probs_and_noise"] == 4 * 32 * 4
    assert components["top_k_top_p_workspace"] == 4 * 32 * (4 + 8 + 4 + 2)


def test_vllm_hybrid_sliding_model_uses_padded_kv_cache_groups(tmp_path):
    # 5 full and 29 sliding layers: group size 5, one full group, six
    # sliding groups and one padding layer; every block id spends one page
    # in five layers of each group.
    model_dir = tmp_path / "gemma-like"
    _write_gemma_like(model_dir)
    page_bytes = 16 * TOKEN_LAYER_BYTES
    block_bytes = 5 * page_bytes

    report = estimate_serving_plan(
        model_dir,
        active_sequences=1,
        max_batch_size=4,
        prompt_tokens=1000,
        generated_tokens=24,
        runtime="vllm",
        device_capacity_bytes=GIB,
        vllm_max_num_batched_tokens=256,
        vllm_kv_cache_memory_bytes=1000 * block_bytes + 7,
        vllm_non_kv_cache_memory_bytes=0,
    )

    cache = report["runtime"]["kv_cache"]
    assert cache["groups"]["group_size"] == 5
    assert cache["groups"]["full_groups"] == 1
    assert cache["groups"]["sliding_groups"] == 6
    assert cache["groups"]["padding_layers"] == 1
    assert cache["page_bytes_per_layer"] == page_bytes
    assert cache["block_bytes"] == block_bytes
    assert cache["num_gpu_blocks"] == 1000
    # "GPU KV cache size" divides the blocks among the seven groups.
    assert cache["logged_size_tokens"] == 1000 // 7 * 16
    # One max_model_len request: cdiv(1024, 16) full blocks plus, per
    # sliding group, cdiv(min(63 + 256, 1024), 16) + 1 blocks.
    assert cache["max_model_len_blocks"] == 64 + 6 * 21
    assert cache["max_model_len_concurrency"] == pytest.approx(1000 / 190)
    # Prefill is chunked at max_num_batched_tokens, so the last chunk holds
    # 63 full blocks and, per sliding group, the window-1 tokens plus the
    # 256-token chunk: cdiv(319, 16) + 1 blocks.
    assert report["inputs"]["prefill_chunk_tokens"] == 256
    assert cache["logical_requested_peak_bytes"] == (63 + 6 * 21) * block_bytes
    # Decode: 64 full blocks and 5 blocks per sliding group.
    decode = report["memory_timeline"]["phases"][1]["components"]
    assert decode["logical_kv_cache"] == (64 + 6 * 5) * block_bytes


def test_vllm_kv_cache_dtype_sets_page_size(tmp_path):
    model_dir = tmp_path / "model"
    _write_model(model_dir)
    common = dict(
        active_sequences=1,
        max_batch_size=4,
        prompt_tokens=60,
        generated_tokens=4,
        runtime="vllm",
        device_capacity_bytes=GIB,
    )

    auto = estimate_serving_plan(model_dir, **common)
    fp8 = estimate_serving_plan(
        model_dir,
        vllm_kv_cache_dtype="fp8",
        **common,
    )

    assert auto["runtime"]["kv_cache"]["block_bytes"] == 16 * 2 * 8 * 2
    assert fp8["runtime"]["kv_cache"]["block_bytes"] == 16 * 2 * 8 * 1
    assert fp8["inputs"]["vllm"]["kv_cache_dtype"] == "fp8"
    auto_blocks = auto["runtime"]["kv_cache"]["num_gpu_blocks"]
    assert fp8["runtime"]["kv_cache"]["num_gpu_blocks"] in {
        2 * auto_blocks,
        2 * auto_blocks + 1,
    }
    with pytest.raises(ServingPlanError, match="vllm_kv_cache_dtype"):
        estimate_serving_plan(
            model_dir,
            vllm_kv_cache_dtype="int8",
            **common,
        )
    with pytest.raises(ServingPlanError, match="require runtime='vllm'"):
        estimate_serving_plan(
            model_dir,
            active_sequences=1,
            prompt_tokens=60,
            vllm_max_num_batched_tokens=64,
        )


def test_paged_prefix_hit_skips_whole_blocks_only(tmp_path):
    # get_computed_blocks caps the hit at prompt_tokens - 1 and counts
    # full blocks only.
    model_dir = tmp_path / "model"
    _write_model(model_dir)

    report = estimate_serving_plan(
        model_dir,
        active_sequences=2,
        prompt_tokens=40,
        generated_tokens=2,
        shared_prefix_tokens=40,
    )

    prefix_cache = report["optimizations"]["prefix_cache"]
    assert prefix_cache["shared_prefix_tokens"] == 40
    assert prefix_cache["hit_tokens"] == 32
    assert report["optimizations"]["chunked_prefill"][
        "uncached_prompt_tokens"
    ] == 8


def test_serving_cli_exposes_vllm_scheduler_and_cache_flags(
    tmp_path,
    capsys,
):
    model_dir = tmp_path / "model"
    output_path = tmp_path / "plan.json"
    _write_model(model_dir)

    assert serving_main(
        [
            "--model-dir",
            str(model_dir),
            "--active-sequences",
            "1",
            "--prompt-tokens",
            "60",
            "--runtime",
            "vllm",
            "--device-memory-gib",
            "1",
            "--vllm-max-num-batched-tokens",
            "128",
            "--vllm-kv-cache-dtype",
            "fp8_e5m2",
            "--vllm-usage-context",
            "api_server",
            "--json",
            str(output_path),
        ]
    ) == 0

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["inputs"]["vllm"]["max_num_batched_tokens"] == 128
    assert report["inputs"]["vllm"]["max_num_seqs"] == 128
    assert report["inputs"]["vllm"]["kv_cache_dtype"] == "fp8_e5m2"
    assert report["inputs"]["vllm"]["usage_context"] == "api_server"
    assert report["inputs"]["max_batch_size"] == 128
    assert report["inputs"]["kv_cache_block_tokens"] == 16
    capsys.readouterr()
