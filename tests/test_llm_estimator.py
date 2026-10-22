from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import pytest

from fakegpu.calibration import (
    compare_memory_reports,
    verify_calibration_reports,
)
from fakegpu.llm_estimator import (
    SCHEMA_VERSION as LLM_SCHEMA_VERSION,
    estimate_decoder_inference,
    estimate_kv_cache_memory,
    inspect_safetensors_checkpoint,
)
from fakegpu.llm_cli import main as llm_main
from fakegpu.serving_plan import (
    REQUEST_MANIFEST_SCHEMA_VERSION,
    REQUEST_SET_SCHEMA_VERSION,
    SCHEMA_VERSION as SERVING_SCHEMA_VERSION,
    ServingPlanError,
    estimate_serving_kv_pool,
    estimate_serving_plan,
    estimate_serving_request_kv_pool,
    estimate_serving_request_set,
    load_serving_requests,
    main as serving_main,
)


def _write_safetensors(
    root: Path,
    *,
    filename: str,
    header: dict[str, object],
) -> None:
    offset = 0
    normalized: dict[str, object] = {}
    dtype_bytes = {
        "BF16": 2, "F16": 2, "F32": 4, "I8": 1, "U8": 1, "F8_E4M3": 1,
    }
    for name, raw in header.items():
        metadata = dict(raw)
        shape = metadata["shape"]
        nbytes = math.prod(shape) * dtype_bytes[str(metadata["dtype"])]
        metadata["data_offsets"] = [offset, offset + nbytes]
        offset += nbytes
        normalized[name] = metadata
    encoded = json.dumps(normalized, separators=(",", ":")).encode("utf-8")
    padding = (-len(encoded)) % 8
    encoded += b" " * padding
    (root / filename).write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + b"\0" * offset
    )


def _write_model(
    root: Path,
    *,
    config_overrides: dict[str, object] | None = None,
    checkpoint_dtype: str = "BF16",
) -> None:
    root.mkdir()
    config = {
        "architectures": ["TinyForCausalLM"],
        "model_type": "tiny",
        "torch_dtype": "bfloat16",
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "intermediate_size": 16,
        "vocab_size": 32,
    }
    config.update(config_overrides or {})
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    hidden_size = int(config["hidden_size"])
    vocab_size = int(config["vocab_size"])
    header = {
        "model.embed.weight": {
            "dtype": checkpoint_dtype,
            "shape": [vocab_size, hidden_size],
        },
        "model.proj.weight": {
            "dtype": checkpoint_dtype,
            "shape": [hidden_size, hidden_size],
        },
    }
    _write_safetensors(
        root,
        filename="model.safetensors",
        header=header,
    )


def test_checkpoint_headers_are_read_without_tensor_materialization(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_model(model_dir)
    report = inspect_safetensors_checkpoint(model_dir)
    assert report["tensor_count"] == 2
    assert report["parameter_count"] == 320
    assert report["checkpoint_bytes"] == 640
    assert report["dtype_bytes"] == {"BF16": 640}


@pytest.mark.parametrize(
    ("index", "message"),
    [
        ([], "invalid safetensors weight map"),
        (
            {"weight_map": {"a": "model.safetensors"}, "metadata": "oops"},
            "invalid safetensors index metadata",
        ),
        (
            {
                "weight_map": {"a": "model.safetensors"},
                "metadata": {"total_size": [1]},
            },
            "invalid safetensors index total_size",
        ),
    ],
)
def test_checkpoint_inspection_rejects_malformed_index(
    tmp_path: Path,
    index: object,
    message: str,
) -> None:
    model_dir = tmp_path / "model"
    _write_model(model_dir)
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(index),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        inspect_safetensors_checkpoint(model_dir)


def test_checkpoint_inspection_can_select_an_explicit_file_family(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "variants"
    model_dir.mkdir()
    _write_safetensors(
        model_dir,
        filename="model.safetensors",
        header={
            "base.weight": {
                "dtype": "BF16",
                "shape": [2, 2],
            }
        },
    )
    _write_safetensors(
        model_dir,
        filename="model.fp16.safetensors",
        header={
            "variant.weight": {
                "dtype": "F16",
                "shape": [4, 4],
            }
        },
    )

    report = inspect_safetensors_checkpoint(
        model_dir,
        files=["model.fp16.safetensors"],
    )
    assert report["parameter_count"] == 16
    assert report["checkpoint_bytes"] == 32
    assert report["dtype_bytes"] == {"F16": 32}


def test_dense_decoder_memory_and_flops_are_shape_aware(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_model(model_dir)
    report = estimate_decoder_inference(
        model_dir,
        batch_size=1,
        prompt_tokens=4,
        generated_tokens=2,
        runtime_overhead_bytes=1024,
    )
    assert report["memory"]["parameter_bytes"] == 640
    assert report["memory"]["kv_cache_bytes_after_prefill"] == 128
    assert report["memory"]["kv_cache_bytes_after_generation"] == 160
    assert report["kv_cache"]["strategy"] == "dynamic"
    assert report["kv_cache"]["prefill"][
        "allocation_utilization_percent"
    ] == 100
    assert report["compute"]["prefill_flops"] == 12288
    assert report["compute"]["decode_steps"] == [
        {
            "step": 1,
            "query_tokens": 1,
            "key_tokens": 5,
            "matmul_flops": 3136,
        }
    ]
    assert report["memory"]["estimated_process_peak_bytes"] == (
        report["memory"]["estimated_tensor_peak_bytes"] + 1024
    )
    assert report["memory_timeline"]["peak_bytes"] == report["memory"][
        "estimated_process_peak_bytes"
    ]
    assert {
        phase["phase"] for phase in report["memory_timeline"]["phases"]
    } == {"prefill", "decode"}
    assert report["model"]["model_kind"] == "dense_decoder"
    assert report["communication"]["enabled"] is False
    assert report["memory_traffic"]["lower_bytes"] > 0


@pytest.mark.parametrize("use_cache", [True, False])
def test_trailing_logits_preserve_forward_work_and_reduce_projection(
    tmp_path: Path, use_cache: bool,
) -> None:
    model_dir = tmp_path / "model"
    _write_model(model_dir, config_overrides={"vocab_size": 1024})
    common = dict(batch_size=2, prompt_tokens=4, generated_tokens=4, use_cache=use_cache)
    full = estimate_decoder_inference(model_dir, **common)
    trailing = estimate_decoder_inference(model_dir, logits_to_keep=2, **common)
    aggregated = estimate_decoder_inference(
        model_dir, logits_to_keep=2, include_decode_steps=False, **common,
    )

    assert full["compute"]["prefill_flops"] - trailing["compute"]["prefill_flops"] == (
        2 * 2 * (4 - 2) * 8 * 1024
    )
    assert trailing["memory"]["prefill_transient"]["logits_bytes"] == (
        2 * 4 * 8 * 2 + 2 * 2 * 1024 * 2
    )
    assert trailing["memory"]["estimated_tensor_peak_bytes"] < full["memory"]["estimated_tensor_peak_bytes"]
    assert trailing["memory_traffic"]["expected_bytes"] < full["memory_traffic"]["expected_bytes"]
    for phase in ("prefill_transient", "decode_transient"):
        for component in ("attention_bytes", "mlp_bytes"):
            assert trailing["memory"][phase][component] == full["memory"][phase][component]
    for full_step, step in zip(full["compute"]["decode_steps"], trailing["compute"]["decode_steps"]):
        queries = step["query_tokens"]
        assert queries == full_step["query_tokens"]
        assert full_step["matmul_flops"] - step["matmul_flops"] == (
            2 * 2 * max(0, queries - 2) * 8 * 1024
        )
    assert trailing["kv_cache"] == full["kv_cache"]
    assert aggregated["compute"]["total_flops"] == trailing["compute"]["total_flops"]
    assert aggregated["memory_traffic"] == trailing["memory_traffic"]


def test_c11_qwen2vl_last_token_logits_match_recorded_matrix_work(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_model(model_dir)
    # Only dimensions are needed to reproduce job 1723's decoder computation.
    config = {
        "hidden_size": 1536, "num_hidden_layers": 28,
        "num_attention_heads": 12, "num_key_value_heads": 2,
        "intermediate_size": 8960, "vocab_size": 151936,
        "torch_dtype": "bfloat16",
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    common = dict(prompt_tokens=296, generated_tokens=14, attention_implementation="sdpa")
    full = estimate_decoder_inference(model_dir, **common)
    trailing = estimate_decoder_inference(model_dir, logits_to_keep=1, **common)
    assert full["compute"]["total_flops"] == 969_676_283_904
    # Profiler mm/addmm + analytical dense-equivalent QK/AV. Excludes RoPE bmm.
    assert trailing["compute"]["total_flops"] == 816_235_413_504 + 15_750_389_760
    assert trailing["compute"]["decode_flops_total"] == full["compute"]["decode_flops_total"]


@pytest.mark.parametrize(
    ("tensors", "expected"),
    [
        ({"model.embed.weight": "F16", "model.proj.weight": "F16"}, "float16"),
        # Sorted key order decides; FP8 and integer tensors are skipped.
        (
            {
                "a.weight": "F8_E4M3",
                "b.weight_scale": "I8",
                "c.norm.weight": "F32",
                "d.weight": "BF16",
            },
            "float32",
        ),
    ],
)
def test_auto_dtype_without_config_dtype_follows_checkpoint(
    tmp_path: Path,
    tensors: dict[str, str],
    expected: str,
) -> None:
    model_dir = tmp_path / "model"
    _write_model(model_dir, config_overrides={"torch_dtype": None})
    (model_dir / "model.safetensors").unlink()
    _write_safetensors(
        model_dir,
        filename="model.safetensors",
        header={
            name: {"dtype": dtype, "shape": [8, 8]}
            for name, dtype in tensors.items()
        },
    )
    # transformers 5 from_pretrained defaults to dtype="auto"; without a
    # config dtype it loads in the first floating, non-FP8 tensor's dtype
    # of the first shard (safetensors keys iterate in sorted order).
    report = estimate_decoder_inference(model_dir, prompt_tokens=1)

    assert report["inputs"]["dtype"] == expected


def test_query_and_output_projection_flops_follow_head_dim_override(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    _write_model(model_dir)
    # Qwen3-0.6B decoder dimensions: 16 heads x head_dim 128 = 2048, which is
    # twice hidden_size, so q_proj is 1024->2048 and o_proj is 2048->1024.
    config = {
        "hidden_size": 1024, "num_hidden_layers": 28,
        "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
        "intermediate_size": 3072, "vocab_size": 151936,
        "torch_dtype": "bfloat16",
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    report = estimate_decoder_inference(model_dir, prompt_tokens=1)
    # transformers Qwen3ForCausalLM on the meta device has per-layer weights
    # q 2048x1024, k 1024x1024, v 1024x1024, o 1024x2048 and gate/up/down
    # 3x3072x1024; one token costs two FLOPs per weight, plus the LM head and
    # QK/AV over one key token.
    attention_projections = 2 * 28 * (2 * 2048 * 1024 + 2 * 1024 * 1024)
    mlp = 2 * 28 * 3 * 3072 * 1024
    lm_head = 2 * 1024 * 151936
    attention_scores = 4 * 16 * 1 * 1 * 128 * 28
    assert attention_projections == 352_321_536
    assert report["compute"]["prefill_flops"] == (
        attention_projections + mlp + lm_head + attention_scores
    )


@pytest.mark.parametrize("value", [-1, True, 1.5, "1", None])
def test_logits_to_keep_rejects_invalid_values(tmp_path: Path, value: object) -> None:
    with pytest.raises(ValueError, match="logits_to_keep must be a non-negative integer"):
        estimate_decoder_inference(tmp_path, prompt_tokens=4, logits_to_keep=value)


def test_llm_cli_limits_logits_and_clamps_to_query_length(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_model(model_dir)
    output = tmp_path / "estimate.json"
    assert llm_main([
        "--model-dir", str(model_dir), "--prompt-tokens", "4",
        "--logits-to-keep", "1", "--json", str(output),
    ]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["inputs"]["logits_to_keep"] == 1
    assert report["compute"]["prefill_flops"] == 12288 - 2 * 3 * 8 * 32
    full = estimate_decoder_inference(model_dir, prompt_tokens=4)
    clamped = estimate_decoder_inference(model_dir, prompt_tokens=4, logits_to_keep=100)
    assert clamped["compute"] == full["compute"]
    assert clamped["memory"] == full["memory"]


def test_nested_text_config_uses_decoder_dimensions_and_dtype(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_model(model_dir)
    original = estimate_decoder_inference(model_dir, prompt_tokens=4)
    config_path = model_dir / "config.json"
    text_config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config["dtype"] = text_config.pop("torch_dtype")
    config_path.write_text(json.dumps({
        "model_type": "qwen2_vl",
        "architectures": ["Qwen2VLForConditionalGeneration"],
        "text_config": text_config,
        "hidden_size": 1024,
        "torch_dtype": "float32",
        "vision_config": {"hidden_size": 1024},
    }), encoding="utf-8")
    report = estimate_decoder_inference(model_dir, prompt_tokens=4)
    assert report["model"]["hidden_size"] == 8
    assert report["model"]["model_type"] == "qwen2_vl"
    assert report["inputs"]["dtype"] == "bfloat16"
    assert report["memory"] == original["memory"]
    assert report["compute"] == original["compute"]
    assert report["model"]["has_vision_encoder"] is True
    assert "vision_encoder_activations" in report["unmodeled_components"]
    assert "vision_encoder_flops" in report["unmodeled_components"]
    assert "multimodal_projector_activations_and_flops" in report["unmodeled_components"]
    assert any("cover only the decoder" in note for note in report["notes"])
    assert "vision_encoder_flops" not in original["unmodeled_components"]


@pytest.mark.parametrize("config", [[], {"text_config": []}, {"text_config": None}])
def test_malformed_nested_decoder_config_raises_value_error(tmp_path: Path, config: object) -> None:
    model_dir = tmp_path / "model"
    _write_model(model_dir)
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="must be an object"):
        estimate_decoder_inference(model_dir, prompt_tokens=4)


@pytest.mark.parametrize("heterogeneous", [False, True])
@pytest.mark.parametrize("vision_role", ["target", "draft"])
def test_serving_propagates_multimodal_coverage(
    tmp_path: Path, heterogeneous: bool, vision_role: str,
) -> None:
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    _write_model(target)
    _write_model(draft)
    common = dict(max_batch_size=2, draft_model_dir=draft, device_capacity_bytes=2**20)

    def plan():
        if heterogeneous:
            return estimate_serving_request_set(
                target, [{"id": "a", "prompt_tokens": 4, "generated_tokens": 3}], **common,
            )
        return estimate_serving_plan(target, active_sequences=1, prompt_tokens=4, **common)

    before = plan()
    selected = target if vision_role == "target" else draft
    path = selected / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["vision_config"] = {}
    path.write_text(json.dumps(config), encoding="utf-8")
    after = plan()
    prefix = "" if vision_role == "target" else "draft_"
    assert prefix + "vision_encoder_activations" in after["unmodeled_components"]
    assert prefix + "vision_encoder_flops" in after["unmodeled_components"]
    assert after["workload_signature"] != before["workload_signature"]
    assert after["memory"] == before["memory"]
    assert any("multimodal validation" in note for note in after["notes"])


def test_multimodal_cli_reports_incomplete_coverage(tmp_path: Path, capsys) -> None:
    model_dir = tmp_path / "model"
    _write_model(model_dir, config_overrides={"mm_vision_tower": "owner/vision-model"})
    assert llm_main(["--model-dir", str(model_dir), "--prompt-tokens", "4"]) == 0
    assert "warning: Vision encoder" in capsys.readouterr().out
    assert serving_main([
        "--model-dir", str(model_dir), "--prompt-tokens", "4",
        "--active-sequences", "1", "--max-batch-size", "2",
    ]) == 0
    assert "warning: Vision encoder" in capsys.readouterr().out


def test_decoder_without_kv_cache_recomputes_full_decode_context(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    _write_model(model_dir)
    uncached = estimate_decoder_inference(
        model_dir,
        prompt_tokens=4,
        generated_tokens=3,
        use_cache=False,
    )
    cached = estimate_decoder_inference(
        model_dir,
        prompt_tokens=4,
        generated_tokens=3,
        use_cache=True,
    )

    assert uncached["kv_cache"]["enabled"] is False
    assert uncached["memory"]["kv_cache_bytes_after_generation"] == 0
    assert [
        (step["query_tokens"], step["key_tokens"])
        for step in uncached["compute"]["decode_steps"]
    ] == [(5, 5), (6, 6)]
    assert uncached["memory"]["decode_input_bytes"] == 48
    assert uncached["memory_timeline"]["phases"][1]["components"][
        "inputs"
    ] == 48
    assert uncached["compute"]["decode_flops_total"] > cached[
        "compute"
    ]["decode_flops_total"]
    assert uncached["memory"]["decode_transient"]["peak_bytes"] > cached[
        "memory"
    ]["decode_transient"]["peak_bytes"]


def test_decode_step_aggregation_preserves_flops_without_per_token_records(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    _write_model(model_dir)
    expanded = estimate_decoder_inference(
        model_dir,
        prompt_tokens=8,
        generated_tokens=20,
        kv_cache_window_tokens=12,
    )
    aggregated = estimate_decoder_inference(
        model_dir,
        prompt_tokens=8,
        generated_tokens=20,
        kv_cache_window_tokens=12,
        include_decode_steps=False,
    )

    assert expanded["compute"]["decode_flops_total"] == aggregated[
        "compute"
    ]["decode_flops_total"]
    assert aggregated["compute"]["decode_steps"] == []
    assert aggregated["compute"]["decode_steps_included"] is False
    assert aggregated["compute"]["decode_step_summary"] == {
        "count": 19,
        "first_key_tokens": 9,
        "last_key_tokens": 12,
        "key_tokens_sum": 222,
        "matmul_flops": expanded["compute"]["decode_flops_total"],
    }
    assert aggregated["inputs"]["include_decode_steps"] is False


def test_mla_cache_layout_follows_runtime_and_architecture_flops(
    tmp_path: Path,
) -> None:
    mla_dir = tmp_path / "mla"
    standard_dir = tmp_path / "standard"
    common_config = {
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 32,
        "vocab_size": 64,
    }
    _write_model(
        mla_dir,
        config_overrides={
            **common_config,
            "q_lora_rank": 8,
            "kv_lora_rank": 6,
            "qk_nope_head_dim": 4,
            "qk_rope_head_dim": 2,
            "v_head_dim": 4,
        },
    )
    _write_model(
        standard_dir,
        config_overrides={
            **common_config,
            "num_key_value_heads": 4,
            "head_dim": 6,
        },
    )

    mla = estimate_decoder_inference(
        mla_dir,
        batch_size=2,
        prompt_tokens=4,
        generated_tokens=2,
        attention_implementation="sdpa",
    )
    standard = estimate_decoder_inference(
        standard_dir,
        batch_size=2,
        prompt_tokens=4,
        generated_tokens=2,
        attention_implementation="sdpa",
    )

    assert mla["model"]["attention_kind"] == "multi_head_latent"
    assert mla["model"]["kv_cache_layout"] == (
        "compressed_latent_and_rope"
    )
    assert mla["model"]["head_dim"] == 6
    assert mla["model"]["kv_cache_elements_per_token_per_layer"] == 8
    # Transformers' DeepseekV3 attention caches decompressed per-head keys
    # (qk_nope + qk_rope) and values (v_head_dim): 4 * (4 + 2 + 4).
    assert mla["inputs"]["mla_kv_cache_layout"] == "decompressed"
    assert mla["kv_cache"]["cache_layout"] == (
        "decompressed_per_head_key_value"
    )
    assert mla["kv_cache"]["elements_per_token_per_layer"] == 40
    assert mla["kv_cache"]["prefill"]["allocated_bytes"] == 2 * 40 * 8 * 2
    latent = estimate_decoder_inference(
        mla_dir,
        batch_size=2,
        prompt_tokens=4,
        generated_tokens=2,
        attention_implementation="sdpa",
        mla_kv_cache_layout="latent",
    )
    assert latent["inputs"]["mla_kv_cache_layout"] == "latent"
    assert latent["kv_cache"]["elements_per_token_per_layer"] == 8
    assert latent["kv_cache"]["prefill"]["allocated_bytes"] == 256
    assert standard["inputs"]["mla_kv_cache_layout"] is None
    assert standard["kv_cache"]["prefill"]["allocated_bytes"] == 1536
    assert mla["compute"]["prefill_flops"] > 0
    assert mla["compute"]["prefill_flops"] != standard["compute"][
        "prefill_flops"
    ]
    assert (
        mla["memory"]["prefill_transient"]["attention_bytes"]
        < standard["memory"]["prefill_transient"]["attention_bytes"]
    )
    assert "mla_backend_projection_absorption_and_workspace" in mla[
        "unmodeled_components"
    ]

    serving = estimate_serving_plan(
        mla_dir,
        active_sequences=2,
        max_batch_size=8,
        prompt_tokens=4,
        generated_tokens=2,
        kv_cache_strategy="paged",
        kv_cache_block_tokens=4,
        runtime="vllm",
        device_capacity_bytes=2**20,
        memory_utilization=1,
    )
    assert serving["kv_cache"]["cache_layout"] == (
        "compressed_latent_and_rope"
    )
    assert serving["runtime"]["kv_cache"]["block_bytes"] == 128


def test_mla_requires_a_complete_positive_cache_configuration(
    tmp_path: Path,
) -> None:
    incomplete_dir = tmp_path / "incomplete-mla"
    _write_model(
        incomplete_dir,
        config_overrides={"kv_lora_rank": 512},
    )

    with pytest.raises(
        ValueError,
        match="multi-head latent attention requires positive",
    ):
        estimate_decoder_inference(
            incomplete_dir,
            prompt_tokens=4,
        )


def test_serving_plan_report_envelopes_have_golden_contract(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    _write_model(model_dir)
    homogeneous = estimate_serving_plan(
        model_dir,
        active_sequences=2,
        max_batch_size=4,
        prompt_tokens=8,
        generated_tokens=3,
        device_capacity_bytes=2**20,
        memory_utilization=1,
    )
    heterogeneous = estimate_serving_request_set(
        model_dir,
        [
            {"id": "a", "prompt_tokens": 8, "generated_tokens": 3},
            {"id": "b", "prompt_tokens": 8, "generated_tokens": 3},
        ],
        max_batch_size=4,
        device_capacity_bytes=2**20,
        memory_utilization=1,
    )

    assert {
        "schema_version": homogeneous["schema_version"],
        "method": homogeneous["method"],
        "peak_bytes": homogeneous["memory_timeline"]["peak_bytes"],
        "peak_phase": homogeneous["memory_timeline"]["peak_phase"],
        "requested_fits": homogeneous["scheduler"]["requested_fits"],
        "limiting_factor": homogeneous["scheduler"]["limiting_factor"],
        "kv_decode_bytes": homogeneous["kv_cache"]["decode"][
            "allocated_bytes"
        ],
    } == {
        "schema_version": "fakegpu.llm_serving_plan.v1",
        "method": (
            "safetensors_headers_plus_decoder_shape_and_serving_pool_model"
        ),
        "peak_bytes": 4096,
        "peak_phase": "prefill",
        "requested_fits": True,
        "limiting_factor": "configured_max_batch_size",
        "kv_decode_bytes": 1024,
    }
    assert {
        "schema_version": heterogeneous["schema_version"],
        "method": heterogeneous["method"],
        "peak_bytes": heterogeneous["memory_timeline"]["peak_bytes"],
        "peak_phase": heterogeneous["memory_timeline"]["peak_phase"],
        "requested_fits": heterogeneous["scheduler"]["requested_fits"],
        "limiting_factor": heterogeneous["scheduler"]["limiting_factor"],
        "admitted_ids": heterogeneous["scheduler"]["admitted_request_ids"],
        "kv_decode_bytes": heterogeneous["kv_cache"]["decode"][
            "private_allocated_bytes"
        ],
    } == {
        "schema_version": "fakegpu.llm_serving_request_set_plan.v1",
        "method": (
            "safetensors_headers_plus_decoder_shape_and_"
            "heterogeneous_serving_request_model"
        ),
        "peak_bytes": 2880,
        "peak_phase": "prefill",
        "requested_fits": True,
        "limiting_factor": "request_manifest_exhausted",
        "admitted_ids": ["a", "b"],
        "kv_decode_bytes": 1024,
    }


def test_kv_cache_strategies_account_for_storage_and_reservation() -> None:
    common = {
        "num_hidden_layers": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "batch_size": 1,
        "prompt_tokens": 4,
        "generated_tokens": 2,
        "element_bytes": 2,
    }

    dynamic = estimate_kv_cache_memory(**common, strategy="dynamic")
    static = estimate_kv_cache_memory(
        **common,
        strategy="static",
        max_cache_tokens=16,
    )
    quantized = estimate_kv_cache_memory(
        **common,
        strategy="quantized",
        quantized_bits=4,
        quantized_residual_tokens=2,
    )
    paged = estimate_kv_cache_memory(
        **common,
        strategy="paged",
        block_tokens=4,
    )

    assert dynamic["prefill"]["allocated_bytes"] == 128
    assert dynamic["generation"]["allocated_bytes"] == 160
    assert static["prefill"]["allocated_bytes"] == 512
    assert static["generation"]["allocated_bytes"] == 512
    assert static["prefill"]["reservation_overhead_bytes"] == 384
    assert quantized["storage_bits_per_element"] == 4
    assert quantized["quantized_group_size"] == 64
    # Transformers' QuantizedLayer quantizes the whole prefill: 2 layers *
    # 8 elements * 4 tokens at 4 bits, plus one bfloat16 scale and zero
    # point for the single 64-element group.
    assert quantized["prefill"]["quantization_metadata_bytes"] == 4
    assert quantized["prefill"]["allocated_bytes"] == 32 + 4
    assert quantized["prefill"][
        "full_precision_residual_tokens_per_sequence"
    ] == 0
    # The one decode step appends its token to the full-precision residual.
    assert quantized["generation"]["allocated_bytes"] == 32 + 4 + 32
    assert quantized["generation"][
        "full_precision_residual_tokens_per_sequence"
    ] == 1
    assert quantized["generation"]["quantized_tokens_per_sequence"] == 4
    assert quantized["generation"]["quantization_savings_bytes"] == 92
    assert paged["prefill"]["allocated_bytes"] == 128
    assert paged["generation"]["allocated_bytes"] == 256
    assert paged["generation"]["reservation_overhead_bytes"] == 96


def test_sliding_window_caps_cache_memory_and_attention_context(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    _write_model(model_dir)

    report = estimate_decoder_inference(
        model_dir,
        prompt_tokens=8,
        generated_tokens=4,
        kv_cache_window_tokens=6,
    )

    assert report["memory"]["kv_cache_bytes_after_prefill"] == 192
    assert report["memory"]["kv_cache_bytes_after_generation"] == 192
    assert report["kv_cache"]["generation"][
        "logical_tokens_per_sequence"
    ] == 11
    assert report["kv_cache"]["generation"][
        "effective_tokens_per_sequence"
    ] == 6
    assert all(
        step["key_tokens"] == 6
        for step in report["compute"]["decode_steps"]
    )


def test_kv_cache_configuration_rejects_invalid_capacity() -> None:
    with pytest.raises(ValueError, match="must cover"):
        estimate_kv_cache_memory(
            num_hidden_layers=2,
            num_key_value_heads=1,
            head_dim=4,
            batch_size=1,
            prompt_tokens=4,
            generated_tokens=2,
            strategy="static",
            max_cache_tokens=4,
        )

    with pytest.raises(ValueError, match="quantized_bits"):
        estimate_kv_cache_memory(
            num_hidden_layers=2,
            num_key_value_heads=1,
            head_dim=4,
            batch_size=1,
            prompt_tokens=4,
            strategy="quantized",
            quantized_bits=3,
        )


def test_llm_cli_exposes_paged_cache_controls(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_dir = tmp_path / "model"
    output_path = tmp_path / "estimate.json"
    _write_model(model_dir)

    assert (
        llm_main(
            [
                "--model-dir",
                str(model_dir),
                "--prompt-tokens",
                "4",
                "--generated-tokens",
                "2",
                "--kv-cache-strategy",
                "paged",
                "--kv-cache-block-tokens",
                "4",
                "--json",
                str(output_path),
            ]
        )
        == 0
    )
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["kv_cache"]["strategy"] == "paged"
    assert report["kv_cache"]["generation"][
        "allocated_tokens_per_sequence"
    ] == 8
    assert "KV cache: paged" in capsys.readouterr().out


def test_llm_cli_json_without_a_path_writes_the_report_to_stdout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_dir = tmp_path / "model"
    _write_model(model_dir)

    assert (
        llm_main(
            [
                "--model-dir",
                str(model_dir),
                "--prompt-tokens",
                "4",
                "--generated-tokens",
                "2",
                "--json",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    report, end = json.JSONDecoder().raw_decode(stdout)
    assert report["schema_version"] == LLM_SCHEMA_VERSION
    assert "FakeGPU LLM inference estimate" in stdout[end:]


def test_moe_flops_memory_and_expert_parallel_traffic_are_modeled(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "moe"
    _write_model(
        model_dir,
        config_overrides={
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 12,
            "shared_expert_intermediate_size": 8,
        },
    )

    report = estimate_decoder_inference(
        model_dir,
        batch_size=2,
        prompt_tokens=4,
        generated_tokens=2,
        expert_parallel_size=2,
    )

    assert report["model"]["model_kind"] == "mixture_of_experts"
    assert report["model"]["num_routed_experts"] == 4
    assert report["model"]["num_experts_per_token"] == 2
    assert report["compute"]["total_flops"] > 0
    assert report["memory"]["prefill_transient"]["router_bytes"] > 0
    assert report["communication"]["enabled"] is True
    assert report["communication"]["total_bytes"] > 0


@pytest.mark.parametrize(
    ("mlp_only_layers", "moe_layer_count"),
    [([], 2), ([1], 1), ([0, 3], 1)],
)
def test_moe_layer_count_applies_mlp_only_layers_and_sparse_step(
    tmp_path: Path,
    mlp_only_layers: list[int],
    moe_layer_count: int,
) -> None:
    model_dir = tmp_path / "moe"
    _write_model(
        model_dir,
        config_overrides={
            "num_hidden_layers": 4,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "decoder_sparse_step": 2,
            "mlp_only_layers": mlp_only_layers,
        },
    )

    report = estimate_decoder_inference(model_dir, prompt_tokens=4)

    assert report["model"]["moe_layer_count"] == moe_layer_count
    assert report["model"]["dense_layer_count"] == 4 - moe_layer_count


@pytest.mark.parametrize(
    ("first_k_dense_replace", "moe_layer_count"),
    [(None, 4), (0, 4), (3, 1), (6, 0)],
)
def test_moe_layer_count_skips_first_k_dense_layers(
    tmp_path: Path,
    first_k_dense_replace: int | None,
    moe_layer_count: int,
) -> None:
    model_dir = tmp_path / "moe"
    _write_model(
        model_dir,
        config_overrides={
            "num_hidden_layers": 4,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "first_k_dense_replace": first_k_dense_replace,
        },
    )

    report = estimate_decoder_inference(model_dir, prompt_tokens=4)

    assert report["model"]["moe_layer_count"] == moe_layer_count
    assert report["model"]["dense_layer_count"] == 4 - moe_layer_count


@pytest.mark.parametrize(
    ("layer_selection", "moe_layer_count"),
    [
        ({"moe_layers": list(range(1, 48, 2))}, 24),
        ({"interleave_moe_layer_step": 2}, 24),
        ({"interleave_moe_layer_step": 1}, 48),
    ],
)
def test_llama4_moe_layers_dense_width_and_shared_expert(
    tmp_path: Path,
    layer_selection: dict[str, object],
    moe_layer_count: int,
) -> None:
    model_dir = tmp_path / "llama4"
    _write_model(model_dir)
    # Llama-4-Maverick text_config. transformers' Llama4TextDecoderLayer makes
    # layer i sparse iff i is in config.moe_layers (default
    # range(step - 1, layers, step)); dense layers use intermediate_size_mlp and
    # every Llama4TextMoe adds one intermediate_size shared expert.
    text_config = {
        "model_type": "llama4_text",
        "hidden_size": 5120, "num_hidden_layers": 48,
        "num_attention_heads": 40, "num_key_value_heads": 8, "head_dim": 128,
        "intermediate_size": 8192, "intermediate_size_mlp": 16384,
        "num_local_experts": 128, "num_experts_per_tok": 1,
        "vocab_size": 202048, "torch_dtype": "bfloat16",
        **layer_selection,
    }
    config = {"model_type": "llama4", "text_config": text_config}
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    report = estimate_decoder_inference(model_dir, prompt_tokens=1)
    model = report["model"]

    dense_layer_count = 48 - moe_layer_count
    assert model["moe_layer_count"] == moe_layer_count
    assert model["dense_layer_count"] == dense_layer_count
    assert model["intermediate_size"] == 16384
    assert model["moe_intermediate_size"] == 8192
    assert model["shared_expert_intermediate_size"] == 8192
    # Two FLOPs per weight of the meta-device module tree for one token:
    # q/o 5120x5120, k/v 1024x5120, gate/up/down per dense layer, one routed
    # and one shared expert plus the 5120x128 router per sparse layer.
    attention = 2 * 48 * (2 * 5120 * 5120 + 2 * 5120 * 1024)
    dense = 2 * dense_layer_count * 3 * 5120 * 16384
    sparse = 2 * moe_layer_count * (2 * 3 * 5120 * 8192 + 5120 * 128)
    lm_head = 2 * 5120 * 202048
    scores = 4 * 40 * 128 * 48
    if moe_layer_count == 24:
        assert dense + sparse == 24_190_648_320
    assert report["compute"]["prefill_flops"] == (
        attention + dense + sparse + lm_head + scores
    )


def test_moe_layer_count_rejects_invalid_first_k_dense_replace(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "moe"
    _write_model(
        model_dir,
        config_overrides={
            "num_hidden_layers": 4,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "first_k_dense_replace": -1,
        },
    )

    with pytest.raises(ValueError, match="first_k_dense_replace"):
        estimate_decoder_inference(model_dir, prompt_tokens=4)


def test_quantized_checkpoint_uses_exact_payload_storage(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "quantized"
    _write_model(
        model_dir,
        config_overrides={
            "quantization_config": {
                "quant_method": "gptq",
                "bits": 8,
            }
        },
        checkpoint_dtype="I8",
    )

    report = estimate_decoder_inference(
        model_dir,
        prompt_tokens=4,
        dtype="bfloat16",
    )

    assert report["checkpoint"]["checkpoint_bytes"] == 320
    assert report["memory"]["base_parameter_bytes"] == 320
    assert report["weight_storage"]["quantization"]["enabled"] is True
    assert report["weight_storage"]["quantization"]["method"] == "gptq"
    assert report["weight_storage"]["quantization"][
        "storage_accounting"
    ] == "exact_safetensors_payload"


def test_adapter_parameters_and_target_profile_are_included(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    _write_model(model_dir)
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 4}),
        encoding="utf-8",
    )
    _write_safetensors(
        adapter_dir,
        filename="adapter_model.safetensors",
        header={
            "base.q_proj.lora_A.weight": {
                "dtype": "F32",
                "shape": [4, 8],
            },
            "base.q_proj.lora_B.weight": {
                "dtype": "F32",
                "shape": [8, 4],
            },
        },
    )

    report = estimate_decoder_inference(
        model_dir,
        prompt_tokens=4,
        adapter_dirs=[adapter_dir],
        target_profile="rtx2080ti",
        compute_acceleration_factor=2,
    )

    assert report["memory"]["adapter_parameter_bytes"] == 128
    assert report["memory"]["parameter_bytes"] == 768
    adapter = report["weight_storage"]["adapters"][0]
    assert adapter["adapter_type"] == "LORA"
    assert adapter["checkpoint_bytes"] == 256
    assert report["performance"]["profile"]["architecture"] == "turing"
    interval = report["performance"]["latency_interval_seconds"]
    assert interval["lower"] < interval["expected"] < interval["upper"]


@pytest.mark.parametrize("field", ["batch_size", "prompt_tokens", "generated_tokens"])
def test_estimator_rejects_non_positive_shapes(tmp_path: Path, field: str) -> None:
    model_dir = tmp_path / "model"
    _write_model(model_dir)
    kwargs = {"batch_size": 1, "prompt_tokens": 4, "generated_tokens": 1}
    kwargs[field] = 0
    with pytest.raises(ValueError):
        estimate_decoder_inference(model_dir, **kwargs)


def test_serving_kv_pool_shares_prefix_across_active_sequences() -> None:
    common = {
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "active_sequences": 8,
        "prompt_tokens": 4096,
        "generated_tokens": 257,
        "element_bytes": 2,
        "strategy": "paged",
        "block_tokens": 16,
    }

    baseline = estimate_serving_kv_pool(**common)
    shared = estimate_serving_kv_pool(
        **common,
        shared_prefix_tokens=1024,
    )

    assert baseline["decode"]["allocated_bytes"] == 4_563_402_752
    assert shared["decode"]["allocated_bytes"] == 3_623_878_656
    assert shared["decode"]["prefix_cache_savings_bytes"] == 939_524_096
    assert shared["decode"]["shared_segment"][
        "allocated_tokens_per_sequence"
    ] == 1024
    assert shared["decode"]["private_segments"][
        "sequence_count"
    ] == 8
    assert shared["decode"]["allocation_utilization_percent"] == 100


@pytest.mark.parametrize(
    ("prompt_tokens", "shared_prefix_tokens", "shared_blocks", "private_blocks"),
    [
        # A 1-token prefix fills no block, so nothing is shared.
        (17, 1, 0, 2),
        # 20 prefix tokens share one full block; each 40-token sequence
        # still holds cdiv(40, 16) - 1 = 2 blocks of its own.
        (40, 20, 1, 2),
    ],
)
def test_serving_paged_prefix_shares_only_full_blocks(
    prompt_tokens: int,
    shared_prefix_tokens: int,
    shared_blocks: int,
    private_blocks: int,
) -> None:
    # vLLM v0.32.0 vllm/v1/core/kv_cache_utils.py:980-982 hashes, and so
    # shares, only full blocks of a prompt prefix.
    report = estimate_serving_kv_pool(
        num_hidden_layers=2,
        num_key_value_heads=1,
        head_dim=4,
        active_sequences=2,
        prompt_tokens=prompt_tokens,
        element_bytes=2,
        strategy="paged",
        shared_prefix_tokens=shared_prefix_tokens,
        block_tokens=16,
    )

    block_bytes = 16 * 32
    prefill = report["prefill"]
    assert prefill["retained_shared_prefix_tokens"] == 16 * shared_blocks
    assert prefill["shared_segment"]["allocated_tokens_per_sequence"] == (
        16 * shared_blocks
    )
    assert prefill["private_segments"][
        "allocated_tokens_per_sequence"
    ] == 16 * private_blocks
    assert prefill["allocated_bytes"] == block_bytes * (
        shared_blocks + 2 * private_blocks
    )


def test_serving_prefix_cache_rejects_unsupported_storage() -> None:
    common = {
        "num_hidden_layers": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "active_sequences": 2,
        "prompt_tokens": 16,
        "shared_prefix_tokens": 8,
    }

    with pytest.raises(ServingPlanError, match="prefix-shareable"):
        estimate_serving_kv_pool(**common, strategy="quantized")
    with pytest.raises(ServingPlanError, match="must not exceed"):
        estimate_serving_kv_pool(
            **{**common, "shared_prefix_tokens": 17},
            strategy="paged",
        )


def test_serving_plan_models_chunked_prefill_and_prefix_hits(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    _write_model(model_dir)

    baseline = estimate_serving_plan(
        model_dir,
        active_sequences=8,
        max_batch_size=64,
        prompt_tokens=64,
        generated_tokens=17,
        device_capacity_bytes=2**20,
    )
    optimized = estimate_serving_plan(
        model_dir,
        active_sequences=8,
        max_batch_size=64,
        prompt_tokens=64,
        generated_tokens=17,
        prefill_chunk_tokens=16,
        shared_prefix_tokens=32,
        device_capacity_bytes=2**20,
    )

    assert optimized["schema_version"] == SERVING_SCHEMA_VERSION
    assert optimized["validation_status"] == "Modeled"
    assert optimized["accuracy"]["status"] == "uncalibrated"
    assert optimized["optimizations"]["chunked_prefill"]["enabled"] is True
    assert optimized["optimizations"]["chunked_prefill"][
        "effective_query_tokens"
    ] == 16
    assert optimized["optimizations"]["prefix_cache"][
        "prompt_token_hit_percent"
    ] == 50
    assert optimized["kv_cache"]["decode"][
        "prefix_cache_savings_bytes"
    ] > 0
    assert optimized["memory_timeline"]["peak_bytes"] < baseline[
        "memory_timeline"
    ]["peak_bytes"]
    assert optimized["scheduler"]["requested_fits"] is True
    assert optimized["scheduler"]["admissible_active_sequences"] == 64


def test_serving_plan_reports_memory_limited_admission(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    _write_model(model_dir)

    report = estimate_serving_plan(
        model_dir,
        active_sequences=4,
        max_batch_size=32,
        prompt_tokens=64,
        generated_tokens=17,
        device_capacity_bytes=1024,
        memory_utilization=1,
    )

    assert report["scheduler"]["memory_limited_active_sequences"] == 0
    assert report["scheduler"]["admissible_active_sequences"] == 0
    assert report["scheduler"]["requested_fits"] is False
    assert report["scheduler"]["limiting_factor"] == "usable_device_memory"
    assert report["memory_timeline"][
        "usable_capacity_headroom_bytes"
    ] < 0


def test_serving_plan_can_report_memory_without_a_capacity(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    _write_model(model_dir)

    report = estimate_serving_plan(
        model_dir,
        active_sequences=2,
        max_batch_size=8,
        prompt_tokens=16,
    )

    assert report["target"]["capacity_source"] == "unavailable"
    assert report["scheduler"]["fits_memory"] is None
    assert report["scheduler"]["requested_fits"] is None
    assert report["scheduler"]["admissible_active_sequences"] is None


def test_serving_plan_uses_profile_capacity(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_model(model_dir)

    report = estimate_serving_plan(
        model_dir,
        active_sequences=2,
        max_batch_size=8,
        prompt_tokens=16,
        target_profile="rtx2080ti",
        memory_utilization=0.8,
    )

    profile = report["target"]["profile"]
    assert profile["id"] == "rtx2080ti"
    assert report["target"]["capacity_source"] == "gpu_profile"
    assert report["target"]["usable_capacity_bytes"] == math.floor(
        profile["memory_bytes"] * 0.8
    )


def test_serving_cli_writes_a_machine_readable_plan(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_dir = tmp_path / "model"
    output_path = tmp_path / "serving.json"
    _write_model(model_dir)

    assert (
        serving_main(
            [
                "--model-dir",
                str(model_dir),
                "--active-sequences",
                "4",
                "--max-batch-size",
                "16",
                "--prompt-tokens",
                "64",
                "--generated-tokens",
                "17",
                "--prefill-chunk-tokens",
                "16",
                "--shared-prefix-tokens",
                "32",
                "--device-memory-gib",
                "1",
                "--json",
                str(output_path),
            ]
        )
        == 0
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == SERVING_SCHEMA_VERSION
    assert report["scheduler"]["requested_fits"] is True
    output = capsys.readouterr().out
    assert "FakeGPU LLM serving plan" in output
    assert "accuracy: uncalibrated" in output


def test_vllm_runtime_models_executor_budget_and_reserved_kv_blocks(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    _write_model(model_dir)
    # Synthetic budget large enough for the modeled non-torch allowance.
    capacity_bytes = 2**30

    report = estimate_serving_plan(
        model_dir,
        active_sequences=2,
        max_batch_size=8,
        prompt_tokens=64,
        generated_tokens=17,
        runtime="vllm",
        device_capacity_bytes=capacity_bytes,
        memory_utilization=0.75,
    )

    runtime = report["runtime"]
    runtime_memory = runtime["memory"]
    runtime_cache = runtime["kv_cache"]
    requested_memory = math.ceil(capacity_bytes * 0.75)
    available = (
        requested_memory
        - runtime_memory["non_kv_cache_memory_bytes"]
    )
    expected_blocks = available // runtime_cache["block_bytes"]
    expected_cache_bytes = expected_blocks * runtime_cache["block_bytes"]

    assert runtime["engine"] == "vllm"
    assert report["inputs"]["runtime"] == "vllm"
    assert report["target"]["usable_capacity_bytes"] == requested_memory
    assert runtime_memory[
        "requested_model_executor_memory_bytes"
    ] == requested_memory
    assert runtime_memory["non_kv_cache_memory_source"] == (
        "fakegpu_vllm_profile_run_model"
    )
    # The default token budget is capped at max_num_seqs * max_model_len,
    # and the profile run spreads it over min(tokens, max_num_seqs) requests.
    assert runtime["configuration"]["max_num_seqs"] == 8
    assert runtime["configuration"]["max_num_batched_tokens"] == 8 * 81
    assert runtime["configuration"]["profiled_sequence_count"] == 8
    assert runtime_cache["num_gpu_blocks"] == expected_blocks
    assert runtime_cache["allocatable_bytes"] == expected_cache_bytes
    assert runtime_cache["size_tokens"] == expected_blocks * 16
    assert runtime_cache["logical_requested_fits"] is True
    assert report["scheduler"]["requested_fits"] is True
    for phase in report["memory_timeline"]["phases"]:
        components = phase["components"]
        assert components["kv_cache"] == expected_cache_bytes
        assert components["vllm_reserved_kv_cache"] == expected_cache_bytes
        assert components["logical_kv_cache"] < expected_cache_bytes


def test_vllm_runtime_uses_current_default_and_explicit_profile_inputs(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    _write_model(model_dir)
    block_bytes = 16 * 32
    configured_cache_bytes = 20 * block_bytes + 7

    report = estimate_serving_plan(
        model_dir,
        active_sequences=1,
        max_batch_size=4,
        prompt_tokens=16,
        generated_tokens=4,
        runtime="vllm",
        device_capacity_bytes=2**20,
        vllm_kv_cache_memory_bytes=configured_cache_bytes,
        vllm_non_kv_cache_memory_bytes=100_000,
    )

    runtime = report["runtime"]
    runtime_cache = runtime["kv_cache"]
    assert report["target"]["memory_utilization"] == pytest.approx(0.92)
    assert runtime["configuration"][
        "gpu_memory_utilization_ignored"
    ] is True
    assert runtime["memory"]["non_kv_cache_memory_source"] == (
        "user_supplied_vllm_profile_result"
    )
    assert runtime["memory"]["non_kv_cache_memory_bytes"] == 100_000
    assert runtime_cache["source"] == "kv_cache_memory_bytes_override"
    assert runtime_cache["num_gpu_blocks"] == 20
    assert runtime_cache["allocatable_bytes"] == 20 * block_bytes
    assert runtime_cache["block_rounding_tail_bytes"] == 7


def test_vllm_runtime_admission_is_limited_by_reserved_kv_pool(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    _write_model(model_dir)
    common = {
        "max_batch_size": 8,
        "prompt_tokens": 64,
        "generated_tokens": 17,
    }
    two_sequences = estimate_serving_plan(
        model_dir,
        active_sequences=2,
        **common,
    )
    # vLLM holds one block back as the null block.
    cache_bytes = max(
        phase["components"]["kv_cache"]
        for phase in two_sequences["memory_timeline"]["phases"]
    ) + 16 * 32

    report = estimate_serving_plan(
        model_dir,
        active_sequences=4,
        runtime="vllm",
        device_capacity_bytes=2**20,
        vllm_kv_cache_memory_bytes=cache_bytes,
        vllm_non_kv_cache_memory_bytes=0,
        **common,
    )

    assert report["scheduler"]["memory_limited_active_sequences"] == 2
    assert report["scheduler"]["admissible_active_sequences"] == 2
    assert report["scheduler"]["fits_memory"] is False
    assert report["scheduler"]["requested_fits"] is False
    assert report["scheduler"]["limiting_factor"] == (
        "vllm_kv_cache_capacity"
    )


def test_vllm_runtime_supports_ordered_request_prefix_admission(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    _write_model(model_dir)
    requests = [
        {"id": "a", "prompt_tokens": 32, "generated_tokens": 8},
        {"id": "b", "prompt_tokens": 64, "generated_tokens": 8},
        {"id": "c", "prompt_tokens": 96, "generated_tokens": 8},
    ]
    prefix = estimate_serving_request_set(
        model_dir,
        requests[:2],
        max_batch_size=3,
    )
    # vLLM holds one block back as the null block.
    cache_bytes = max(
        phase["components"]["kv_cache"]
        for phase in prefix["memory_timeline"]["phases"]
    ) + 16 * 32

    report = estimate_serving_request_set(
        model_dir,
        requests,
        max_batch_size=3,
        runtime="vllm",
        device_capacity_bytes=2**20,
        vllm_kv_cache_memory_bytes=cache_bytes,
        vllm_non_kv_cache_memory_bytes=0,
    )

    assert report["runtime"]["configuration"]["profile_scope"] == (
        "vllm_profile_run_max_num_batched_tokens"
    )
    # max_model_len comes from the longest of the three candidates.
    assert report["runtime"]["configuration"]["max_model_len_tokens"] == 104
    assert report["runtime"]["configuration"][
        "max_num_batched_tokens"
    ] == 3 * 104
    assert report["scheduler"]["admitted_request_ids"] == ["a", "b"]
    assert report["scheduler"]["rejected_request_ids"] == ["c"]
    assert report["scheduler"]["limiting_factor"] == (
        "vllm_kv_cache_capacity"
    )


def test_vllm_runtime_holds_back_the_null_block(tmp_path: Path) -> None:
    # vLLM v0.32.0 vllm/v1/core/block_pool.py:184-188 pops block 0 as the
    # null block, so ten blocks hold four 31-token requests (2 blocks each).
    model_dir = tmp_path / "model"
    _write_model(model_dir)
    block_bytes = 16 * 32

    report = estimate_serving_plan(
        model_dir,
        active_sequences=5,
        max_batch_size=8,
        prompt_tokens=16,
        generated_tokens=16,
        runtime="vllm",
        device_capacity_bytes=2**20,
        vllm_kv_cache_memory_bytes=10 * block_bytes,
        vllm_non_kv_cache_memory_bytes=0,
    )

    runtime_cache = report["runtime"]["kv_cache"]
    assert runtime_cache["num_gpu_blocks"] == 10
    assert runtime_cache["allocatable_bytes"] == 10 * block_bytes
    assert runtime_cache["usable_bytes"] == 9 * block_bytes
    assert runtime_cache["logical_requested_fits"] is False
    assert runtime_cache["headroom_bytes"] == -block_bytes
    assert report["scheduler"]["admissible_active_sequences"] == 4
    assert report["scheduler"]["limiting_factor"] == (
        "vllm_kv_cache_capacity"
    )


@pytest.mark.parametrize(
    ("cache_blocks", "failure"),
    [
        # check_memory = available - one null block must be positive.
        (1, "no_available_kv_cache_memory"),
        # max_model_len 33 needs cdiv(33, 16) = 3 blocks beyond the null one.
        (2, "max_model_len_exceeds_kv_cache"),
        (3, "max_model_len_exceeds_kv_cache"),
        (4, None),
    ],
)
def test_vllm_runtime_startup_requires_one_max_model_len_request(
    tmp_path: Path,
    cache_blocks: int,
    failure: str | None,
) -> None:
    # vLLM v0.32.0 vllm/v1/core/kv_cache_utils.py:1003-1040 and 2617-2637
    # raise at startup instead of serving fewer requests.
    model_dir = tmp_path / "model"
    _write_model(model_dir)

    report = estimate_serving_plan(
        model_dir,
        active_sequences=1,
        max_batch_size=1,
        prompt_tokens=16,
        generated_tokens=17,
        runtime="vllm",
        device_capacity_bytes=2**20,
        vllm_kv_cache_memory_bytes=cache_blocks * 16 * 32,
        vllm_non_kv_cache_memory_bytes=0,
    )

    memory = report["runtime"]["memory"]
    assert report["runtime"]["kv_cache"]["max_model_len_blocks"] == 3
    assert memory["initialization_failure"] == failure
    assert memory["initialization_fits"] is (failure is None)
    assert report["scheduler"]["requested_fits"] is (failure is None)
    if failure is not None:
        assert report["scheduler"]["admissible_active_sequences"] == 0
        assert report["scheduler"]["limiting_factor"] == (
            "vllm_model_executor_budget"
        )


def test_vllm_runtime_concurrency_counts_whole_blocks_per_request(
    tmp_path: Path,
) -> None:
    # vLLM v0.32.0 vllm/v1/core/kv_cache_utils.py:1195-1227 divides
    # num_blocks by cdiv(max_model_len, block_size): 60 / cdiv(81, 16).
    model_dir = tmp_path / "model"
    _write_model(model_dir)

    report = estimate_serving_plan(
        model_dir,
        active_sequences=1,
        max_batch_size=1,
        prompt_tokens=64,
        generated_tokens=17,
        runtime="vllm",
        device_capacity_bytes=2**20,
        vllm_kv_cache_memory_bytes=60 * 16 * 32,
        vllm_non_kv_cache_memory_bytes=0,
    )

    runtime_cache = report["runtime"]["kv_cache"]
    assert runtime_cache["max_model_len_blocks"] == 6
    assert runtime_cache["max_model_len_concurrency"] == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("runtime", "target_profile", "expected_dtype"),
    [
        # vLLM v0.32.0 vllm/config/model.py:2405-2426 downcasts float32 to
        # the platform's first dtype; vllm/platforms/cuda.py:255-261 puts
        # bfloat16 first from compute capability 8.0 and float16 before.
        ("vllm", "a100", "bfloat16"),
        ("vllm", "t4", "float16"),
        ("generic", "a100", "float32"),
    ],
)
def test_vllm_runtime_auto_dtype_downcasts_float32_checkpoints(
    tmp_path: Path,
    runtime: str,
    target_profile: str,
    expected_dtype: str,
) -> None:
    model_dir = tmp_path / "model"
    _write_model(
        model_dir,
        config_overrides={"torch_dtype": "float32"},
        checkpoint_dtype="F32",
    )

    report = estimate_serving_plan(
        model_dir,
        active_sequences=1,
        max_batch_size=1,
        prompt_tokens=16,
        runtime=runtime,
        target_profile=target_profile,
    )

    element_bytes = 4 if expected_dtype == "float32" else 2
    assert report["inputs"]["dtype"] == expected_dtype
    assert report["inputs"]["element_bytes"] == element_bytes
    assert report["memory"]["target_parameter_bytes"] == 320 * element_bytes
    assert report["kv_cache"][
        "bytes_per_token_per_sequence_at_compute_dtype"
    ] == 2 * 2 * 1 * 4 * element_bytes


def test_vllm_runtime_rejects_unsupported_or_misplaced_options(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    draft_dir = tmp_path / "draft"
    _write_model(model_dir)
    _write_model(draft_dir)
    common = {
        "active_sequences": 1,
        "max_batch_size": 4,
        "prompt_tokens": 16,
    }

    with pytest.raises(ServingPlanError, match="requires kv_cache_strategy"):
        estimate_serving_plan(
            model_dir,
            **common,
            runtime="vllm",
            kv_cache_strategy="dynamic",
        )
    with pytest.raises(ServingPlanError, match="require runtime='vllm'"):
        estimate_serving_plan(
            model_dir,
            **common,
            vllm_kv_cache_memory_bytes=1024,
        )
    with pytest.raises(ServingPlanError, match="speculative decoding"):
        estimate_serving_plan(
            model_dir,
            **common,
            runtime="vllm",
            draft_model_dir=draft_dir,
        )


def test_vllm_runtime_cli_reports_reserved_cache(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_dir = tmp_path / "model"
    output_path = tmp_path / "vllm-serving.json"
    _write_model(model_dir)

    assert (
        serving_main(
            [
                "--model-dir",
                str(model_dir),
                "--active-sequences",
                "2",
                "--max-batch-size",
                "8",
                "--prompt-tokens",
                "64",
                "--runtime",
                "vllm",
                "--device-memory-gib",
                "1",
                "--json",
                str(output_path),
            ]
        )
        == 0
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["runtime"]["engine"] == "vllm"
    assert report["runtime"]["kv_cache"]["allocatable_bytes"] > 0
    output = capsys.readouterr().out
    assert "vLLM reserved KV cache" in output


def test_speculative_serving_models_dual_weights_caches_and_verification(
    tmp_path: Path,
) -> None:
    target_dir = tmp_path / "target"
    draft_dir = tmp_path / "draft"
    _write_model(target_dir)
    _write_model(
        draft_dir,
        config_overrides={
            "hidden_size": 4,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "intermediate_size": 8,
        },
    )
    common = {
        "active_sequences": 4,
        "max_batch_size": 16,
        "prompt_tokens": 64,
        "generated_tokens": 8,
        "kv_cache_strategy": "dynamic",
    }
    baseline = estimate_serving_plan(target_dir, **common)
    baseline_capacity = baseline["memory_timeline"]["peak_bytes"]
    report = estimate_serving_plan(
        target_dir,
        **common,
        draft_model_dir=draft_dir,
        speculative_tokens=4,
        speculative_acceptance_rate=0.5,
        device_capacity_bytes=baseline_capacity,
        memory_utilization=1,
    )

    speculative = report["speculative_decoding"]
    assert report["method"].endswith(
        "_with_draft_model_speculative_decoding"
    )
    assert report["memory_timeline"]["source"].endswith(
        "_with_draft_model_speculative_decoding"
    )
    assert speculative["enabled"] is True
    assert speculative["proposal_tokens_per_step"] == 4
    assert speculative["effective_proposal_tokens"] == {
        "minimum": 4,
        "maximum": 4,
        "by_request": {"homogeneous_pool": 4},
    }
    assert speculative["acceptance"][
        "expected_accepted_draft_tokens_per_step"
    ] == pytest.approx(0.9375)
    assert speculative["acceptance"][
        "expected_output_tokens_per_target_step"
    ] == pytest.approx(1.9375)
    assert speculative["performance_status"] == (
        "analytical_target_call_reduction_only"
    )
    assert speculative["compatibility"]["vocabulary_size_matches"] is True
    assert speculative["compatibility"]["tokenizer_identity"] == (
        "unverified"
    )
    assert speculative["compatibility"][
        "draft_weight_bytes_smaller_than_target"
    ] is True
    assert speculative["target"]["parameter_bytes"] == 640
    assert speculative["draft"]["parameter_bytes"] == 288
    assert speculative["memory"]["combined_parameter_bytes"] == 928
    assert speculative["target"]["verification_query_tokens"] == 4
    assert speculative["target"]["kv_cache"]["decode"][
        "allocated_bytes"
    ] == 9_600
    assert speculative["draft"]["kv_cache"]["decode"][
        "allocated_bytes"
    ] == 4_800
    assert speculative["memory"][
        "combined_decode_kv_cache_bytes"
    ] == 14_400
    assert report["memory"]["parameter_bytes"] == 928
    assert report["memory_timeline"]["peak_bytes"] > baseline_capacity
    assert report["scheduler"]["requested_fits"] is False
    assert (
        "speculative_draft_model_weights_and_kv_cache"
        not in report["unmodeled_components"]
    )

    rejected = estimate_serving_plan(
        target_dir,
        **common,
        draft_model_dir=draft_dir,
        speculative_tokens=4,
        speculative_acceptance_rate=0,
    )
    accepted = estimate_serving_plan(
        target_dir,
        **common,
        draft_model_dir=draft_dir,
        speculative_tokens=4,
        speculative_acceptance_rate=1,
    )
    assert rejected["memory_timeline"]["peak_bytes"] == accepted[
        "memory_timeline"
    ]["peak_bytes"]
    assert rejected["speculative_decoding"]["acceptance"][
        "expected_output_tokens_per_target_step"
    ] == 1
    assert accepted["speculative_decoding"]["acceptance"][
        "expected_output_tokens_per_target_step"
    ] == 5

    readme_example = estimate_serving_plan(
        target_dir,
        **common,
        draft_model_dir=draft_dir,
        speculative_tokens=5,
        speculative_acceptance_rate=0.7,
    )
    assert round(
        readme_example["speculative_decoding"]["acceptance"][
            "expected_output_tokens_per_target_step"
        ],
        2,
    ) == 2.94


def test_speculative_serving_signature_is_path_stable_and_calibration_scoped(
    tmp_path: Path,
) -> None:
    target_left = tmp_path / "target-left"
    target_right = tmp_path / "target-right"
    draft_left = tmp_path / "draft-left"
    draft_right = tmp_path / "draft-right"
    _write_model(target_left)
    _write_model(target_right)
    draft_config = {
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "intermediate_size": 8,
    }
    _write_model(draft_left, config_overrides=draft_config)
    _write_model(draft_right, config_overrides=draft_config)
    common = {
        "active_sequences": 2,
        "max_batch_size": 8,
        "prompt_tokens": 64,
        "generated_tokens": 16,
        "kv_cache_strategy": "paged",
        "draft_dtype": "bfloat16",
        "speculative_tokens": 3,
        "speculative_acceptance_rate": 0.5,
        "memory_utilization": 1,
    }
    left = estimate_serving_plan(
        target_left,
        draft_model_dir=draft_left,
        target_profile="a100",
        device_capacity_bytes=2**30,
        **common,
    )
    relocated = estimate_serving_plan(
        target_right,
        draft_model_dir=draft_right,
        target_profile="a100",
        device_capacity_bytes=2**31,
        **common,
    )

    signature = left["workload_signature"]
    assert signature.startswith(
        "fakegpu.serving_workload.v1:sha256:"
    )
    assert len(signature.rsplit(":", 1)[-1]) == 64
    assert relocated["workload_signature"] == signature

    changed = estimate_serving_plan(
        target_left,
        draft_model_dir=draft_left,
        target_profile="a100",
        device_capacity_bytes=2**30,
        **{**common, "speculative_tokens": 4},
    )
    assert changed["workload_signature"] != signature

    def observation(plan: dict[str, object]) -> dict[str, object]:
        inputs = plan["inputs"]
        timeline = plan["memory_timeline"]
        target = plan["target"]
        assert isinstance(inputs, dict)
        assert isinstance(timeline, dict)
        assert isinstance(target, dict)
        profile = target["profile"]
        assert isinstance(profile, dict)
        return {
            "schema_version": "serving_observation.v1",
            "workload_signature": plan["workload_signature"],
            "profile": profile["id"],
            "compute_capability": profile["compute_capability"],
            "inputs": {
                "dtype": inputs["dtype"],
                "prompt_tokens": inputs["prompt_tokens"],
            },
            "memory_timeline": {
                "phases": [
                    {
                        "phase": phase["phase"],
                        "peak_bytes": phase["peak_bytes"],
                    }
                    for phase in timeline["phases"]
                ]
            },
        }

    matching = compare_memory_reports(
        left,
        observation(left),
        workload="speculative-serving",
    )
    matching_gate = verify_calibration_reports(
        [matching],
        max_underestimate_percent=0,
        max_absolute_percentage_error_percent=0,
    )
    assert matching_gate["status"] == "passed"
    assert matching_gate["metrics"]["dimension_mismatch_count"] == 0
    assert matching["dimensions"]["prediction"]["gpu_profile"] == (
        "a100"
    )

    drifted = compare_memory_reports(
        left,
        observation(changed),
        workload="speculative-serving",
    )
    drifted_gate = verify_calibration_reports(
        [drifted],
        max_underestimate_percent=100,
    )
    assert drifted_gate["status"] == "failed"
    assert drifted_gate["metrics"]["dimension_mismatch_count"] == 1
    assert drifted_gate["failures"] == [
        {
            "gate": "matching_workload_dimensions",
            "actual": 1,
            "maximum": 0,
        }
    ]

    gpu_drift_observation = observation(left)
    gpu_drift_observation["profile"] = "h100"
    gpu_drifted = compare_memory_reports(
        left,
        gpu_drift_observation,
        workload="speculative-serving",
    )
    gpu_drifted_gate = verify_calibration_reports(
        [gpu_drifted],
        max_underestimate_percent=0,
    )
    assert gpu_drifted_gate["status"] == "failed"
    assert gpu_drifted_gate["metrics"]["dimension_mismatch_count"] == 1
    assert gpu_drifted["dimensions"]["observation"]["gpu_profile"] == (
        "h100"
    )


def test_speculative_serving_rejects_invalid_or_incompatible_drafts(
    tmp_path: Path,
) -> None:
    target_dir = tmp_path / "target"
    draft_dir = tmp_path / "draft"
    incompatible_dir = tmp_path / "incompatible"
    _write_model(target_dir)
    _write_model(draft_dir)
    _write_model(
        incompatible_dir,
        config_overrides={"vocab_size": 64},
    )
    common = {
        "active_sequences": 1,
        "max_batch_size": 4,
        "prompt_tokens": 16,
    }

    with pytest.raises(ServingPlanError, match="vocab_size must match"):
        estimate_serving_plan(
            target_dir,
            **common,
            draft_model_dir=incompatible_dir,
        )
    with pytest.raises(ServingPlanError, match="positive integer"):
        estimate_serving_plan(
            target_dir,
            **common,
            draft_model_dir=draft_dir,
            speculative_tokens=0,
        )
    with pytest.raises(ServingPlanError, match=r"interval \[0, 1\]"):
        estimate_serving_plan(
            target_dir,
            **common,
            draft_model_dir=draft_dir,
            speculative_acceptance_rate=1.01,
        )


@pytest.mark.parametrize(
    "strategy",
    ["dynamic", "paged", "quantized", "static"],
)
def test_speculative_serving_supports_each_kv_cache_strategy(
    tmp_path: Path,
    strategy: str,
) -> None:
    target_dir = tmp_path / "target"
    draft_dir = tmp_path / "draft"
    _write_model(target_dir)
    _write_model(
        draft_dir,
        config_overrides={
            "hidden_size": 4,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "intermediate_size": 8,
        },
    )

    report = estimate_serving_plan(
        target_dir,
        active_sequences=2,
        max_batch_size=8,
        prompt_tokens=16,
        generated_tokens=8,
        kv_cache_strategy=strategy,
        kv_cache_bits=4,
        kv_cache_residual_tokens=4,
        kv_cache_block_tokens=8,
        kv_cache_max_tokens=32 if strategy == "static" else None,
        kv_cache_window_tokens=24,
        draft_model_dir=draft_dir,
        speculative_tokens=3,
    )

    speculative = report["speculative_decoding"]
    assert speculative["acceptance"]["status"] == "not_provided"
    assert speculative["acceptance"][
        "expected_output_tokens_per_target_step"
    ] is None
    assert speculative["performance_status"] == (
        "not_estimated_acceptance_not_provided"
    )
    target_cache = speculative["target"]["kv_cache"]
    draft_cache = speculative["draft"]["kv_cache"]
    assert target_cache["strategy"] == strategy
    assert draft_cache["strategy"] == strategy
    assert target_cache["decode"]["effective_tokens_per_sequence"] == 24
    assert draft_cache["decode"]["effective_tokens_per_sequence"] == 24
    assert speculative["memory"]["combined_decode_kv_cache_bytes"] == (
        target_cache["decode"]["allocated_bytes"]
        + draft_cache["decode"]["allocated_bytes"]
    )
    assert report["memory"]["estimated_process_peak_bytes"] == report[
        "memory_timeline"
    ]["peak_bytes"]
    if strategy == "quantized":
        assert target_cache["storage_bits_per_element"] == 4
        assert draft_cache["storage_bits_per_element"] == 4
        assert target_cache["decode"]["allocated_bytes"] < (
            target_cache[
                "bytes_per_token_per_sequence_at_compute_dtype"
            ]
            * 2
            * 24
        )
        assert draft_cache["decode"]["allocated_bytes"] < (
            draft_cache[
                "bytes_per_token_per_sequence_at_compute_dtype"
            ]
            * 2
            * 24
        )
    if strategy == "static":
        assert target_cache["decode"]["allocated_bytes"] == target_cache[
            "prefill"
        ]["allocated_bytes"]
        assert draft_cache["decode"]["allocated_bytes"] == draft_cache[
            "prefill"
        ]["allocated_bytes"]


def test_speculative_serving_handles_heterogeneous_generation_bounds(
    tmp_path: Path,
) -> None:
    target_dir = tmp_path / "target"
    draft_dir = tmp_path / "draft"
    _write_model(target_dir)
    _write_model(
        draft_dir,
        config_overrides={
            "hidden_size": 4,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "intermediate_size": 8,
        },
    )
    requests = [
        {
            "id": "short-chat",
            "prompt_tokens": 32,
            "generated_tokens": 2,
            "prefix_group": "system",
            "shared_prefix_tokens": 16,
        },
        {
            "id": "long-chat",
            "prompt_tokens": 64,
            "generated_tokens": 8,
            "prefix_group": "system",
            "shared_prefix_tokens": 16,
        },
        {
            "id": "completion",
            "prompt_tokens": 16,
            "generated_tokens": 4,
        },
    ]

    report = estimate_serving_request_set(
        target_dir,
        requests,
        max_batch_size=8,
        prefill_chunk_tokens=8,
        prefill_concurrency=2,
        kv_cache_strategy="dynamic",
        draft_model_dir=draft_dir,
        speculative_tokens=4,
        speculative_acceptance_rate=0.75,
        device_capacity_bytes=2**20,
        memory_utilization=1,
    )
    reordered = estimate_serving_request_set(
        target_dir,
        list(reversed(requests)),
        max_batch_size=8,
        prefill_chunk_tokens=8,
        prefill_concurrency=2,
        kv_cache_strategy="dynamic",
        draft_model_dir=draft_dir,
        speculative_tokens=4,
        speculative_acceptance_rate=0.75,
        device_capacity_bytes=2**20,
        memory_utilization=1,
    )

    speculative = report["speculative_decoding"]
    assert report["workload_signature"].startswith(
        "fakegpu.serving_workload.v1:sha256:"
    )
    assert reordered["workload_signature"] != report[
        "workload_signature"
    ]
    assert speculative["effective_proposal_tokens"] == {
        "minimum": 2,
        "maximum": 4,
        "by_request": {
            "short-chat": 2,
            "long-chat": 4,
            "completion": 4,
        },
    }
    assert speculative["target"]["verification_query_tokens"] == 4
    assert speculative["memory"]["combined_decode_kv_cache_bytes"] == (
        speculative["target"]["kv_cache"]["decode"][
            "allocated_bytes"
        ]
        + speculative["draft"]["kv_cache"]["decode"][
            "allocated_bytes"
        ]
    )
    assert speculative["target"]["kv_cache"]["decode"][
        "prefix_cache_savings_bytes"
    ] > 0
    assert speculative["draft"]["kv_cache"]["decode"][
        "prefix_cache_savings_bytes"
    ] > 0
    assert report["scheduler"]["requested_fits"] is True
    assert report["memory"]["draft_parameter_bytes"] == 288
    request_details = report["memory"]["request_details"]
    assert [
        detail["effective_speculative_tokens"]
        for detail in request_details
    ] == [2, 4, 4]


def test_serving_cli_exposes_speculative_memory_without_claiming_speedup(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target_dir = tmp_path / "target"
    draft_dir = tmp_path / "draft"
    output_path = tmp_path / "speculative-serving.json"
    _write_model(target_dir)
    _write_model(draft_dir)

    assert (
        serving_main(
            [
                "--model-dir",
                str(target_dir),
                "--draft-model-dir",
                str(draft_dir),
                "--active-sequences",
                "2",
                "--max-batch-size",
                "8",
                "--prompt-tokens",
                "32",
                "--generated-tokens",
                "8",
                "--speculative-tokens",
                "3",
                "--speculative-acceptance-rate",
                "0.5",
                "--device-memory-gib",
                "1",
                "--json",
                str(output_path),
            ]
        )
        == 0
    )
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["speculative_decoding"]["enabled"] is True
    assert report["speculative_decoding"]["performance_status"] == (
        "analytical_target_call_reduction_only"
    )
    output = capsys.readouterr().out
    assert "speculative draft:" in output
    assert "expected tokens per target step: 1.875" in output

    assert (
        serving_main(
            [
                "--model-dir",
                str(target_dir),
                "--draft-model-dir",
                str(draft_dir),
                "--active-sequences",
                "1",
                "--prompt-tokens",
                "16",
            ]
        )
        == 0
    )
    assert (
        "expected tokens per target step: acceptance unavailable"
        in capsys.readouterr().out
    )

    with pytest.raises(SystemExit):
        serving_main(
            [
                "--model-dir",
                str(target_dir),
                "--active-sequences",
                "1",
                "--prompt-tokens",
                "16",
                "--speculative-tokens",
                "3",
            ]
        )
    assert "require --draft-model-dir" in capsys.readouterr().err


def test_serving_request_manifest_models_mixed_kv_segments(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "requests.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": REQUEST_MANIFEST_SCHEMA_VERSION,
                "requests": [
                    {
                        "id": "chat-a",
                        "prompt_tokens": 32,
                        "generated_tokens": 5,
                        "prefix_group": "system",
                        "shared_prefix_tokens": 16,
                    },
                    {
                        "id": "chat-b",
                        "prompt_tokens": 48,
                        "generated_tokens": 9,
                        "prefix_group": "system",
                        "shared_prefix_tokens": 16,
                    },
                    {
                        "id": "completion",
                        "prompt_tokens": 8,
                        "generated_tokens": 2,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_serving_requests(manifest_path)
    report = estimate_serving_request_kv_pool(
        manifest["requests"],
        num_hidden_layers=2,
        num_key_value_heads=1,
        head_dim=4,
        element_bytes=2,
        strategy="paged",
        block_tokens=16,
    )

    assert manifest["source"] == str(manifest_path.resolve())
    assert report["request_ids"] == [
        "chat-a",
        "chat-b",
        "completion",
    ]
    assert report["prefill"]["allocated_bytes"] == 2560
    assert report["prefill"]["without_prefix_cache_bytes"] == 3072
    assert report["prefill"]["prefix_cache_savings_bytes"] == 512
    assert report["prefill"][
        "logical_prefix_sharing_savings_bytes"
    ] == 512
    assert report["prefill"]["quantization_savings_bytes"] == 0
    assert report["decode"]["allocated_bytes"] == 3584
    assert report["decode"]["prefix_cache_savings_bytes"] == 512
    assert report["decode"]["prefix_groups"][0]["member_ids"] == [
        "chat-a",
        "chat-b",
    ]

    quantized_requests = [
        {
            "id": "short",
            "prompt_tokens": 4,
            "generated_tokens": 2,
        },
        {
            "id": "long",
            "prompt_tokens": 8,
            "generated_tokens": 2,
        },
    ]
    quantized = estimate_serving_request_kv_pool(
        quantized_requests,
        num_hidden_layers=2,
        num_key_value_heads=1,
        head_dim=4,
        element_bytes=2,
        strategy="quantized",
        quantized_bits=4,
        quantized_residual_tokens=2,
    )
    individual_quantized_bytes = sum(
        estimate_kv_cache_memory(
            num_hidden_layers=2,
            num_key_value_heads=1,
            head_dim=4,
            batch_size=1,
            prompt_tokens=request["prompt_tokens"],
            generated_tokens=request["generated_tokens"],
            element_bytes=2,
            strategy="quantized",
            quantized_bits=4,
            quantized_residual_tokens=2,
        )["generation"]["allocated_bytes"]
        for request in quantized_requests
    )
    assert quantized["decode"]["allocated_bytes"] == (
        individual_quantized_bytes
    )
    assert quantized["decode"]["quantization_savings_bytes"] > 0
    assert quantized["quantized_residual_tokens"] == 2

    static = estimate_serving_request_kv_pool(
        quantized_requests,
        num_hidden_layers=2,
        num_key_value_heads=1,
        head_dim=4,
        element_bytes=2,
        strategy="static",
    )
    assert static["max_cache_tokens"] is None
    assert static["max_cache_tokens_by_request"] == {
        "short": 5,
        "long": 9,
    }

    portable_expected = [
        {
            "id": "portable",
            "prompt_tokens": 16,
            "generated_tokens": 4,
            "prefix_group": None,
            "shared_prefix_tokens": 0,
        }
    ]
    toml_path = tmp_path / "requests.toml"
    toml_path.write_text(
        'schema_version = "fakegpu.serving_requests.v1"\n\n'
        "[[requests]]\n"
        'id = "portable"\n'
        "prompt_tokens = 16\n"
        "generated_tokens = 4\n",
        encoding="utf-8",
    )
    yaml_path = tmp_path / "requests.yaml"
    yaml_path.write_text(
        "schema_version: fakegpu.serving_requests.v1\n"
        "requests:\n"
        "  - id: portable\n"
        "    prompt_tokens: 16\n"
        "    generated_tokens: 4\n",
        encoding="utf-8",
    )
    assert load_serving_requests(toml_path)["requests"] == (
        portable_expected
    )
    assert load_serving_requests(yaml_path)["requests"] == (
        portable_expected
    )

    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "fakegpu.serving_requests.v0",
                "requests": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ServingPlanError, match="unsupported.*schema"):
        load_serving_requests(manifest_path)


def test_serving_request_manifest_rejects_ambiguous_groups() -> None:
    common = {
        "num_hidden_layers": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
    }
    with pytest.raises(ServingPlanError, match="duplicate"):
        estimate_serving_request_kv_pool(
            [
                {
                    "id": "same",
                    "prompt_tokens": 16,
                },
                {
                    "id": "same",
                    "prompt_tokens": 32,
                },
            ],
            **common,
        )
    with pytest.raises(ServingPlanError, match="one shared_prefix"):
        estimate_serving_request_kv_pool(
            [
                {
                    "id": "left",
                    "prompt_tokens": 16,
                    "prefix_group": "system",
                    "shared_prefix_tokens": 8,
                },
                {
                    "id": "right",
                    "prompt_tokens": 32,
                    "prefix_group": "system",
                    "shared_prefix_tokens": 16,
                },
            ],
            **common,
        )
    with pytest.raises(ServingPlanError, match="prefix-shareable"):
        estimate_serving_request_kv_pool(
            [
                {
                    "id": "left",
                    "prompt_tokens": 16,
                    "prefix_group": "system",
                    "shared_prefix_tokens": 8,
                },
                {
                    "id": "right",
                    "prompt_tokens": 16,
                    "prefix_group": "system",
                    "shared_prefix_tokens": 8,
                },
            ],
            strategy="quantized",
            **common,
        )


def test_serving_request_kv_pool_matches_individual_cache_strategies() -> None:
    requests = [
        {
            "id": "short",
            "prompt_tokens": 17,
            "generated_tokens": 3,
        },
        {
            "id": "medium",
            "prompt_tokens": 65,
            "generated_tokens": 9,
        },
        {
            "id": "long",
            "prompt_tokens": 129,
            "generated_tokens": 17,
        },
    ]
    common = {
        "num_hidden_layers": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "element_bytes": 2,
        "block_tokens": 16,
        "quantized_bits": 4,
        "quantized_residual_tokens": 8,
    }

    for strategy in ("dynamic", "paged", "quantized", "static"):
        aggregate = estimate_serving_request_kv_pool(
            requests,
            strategy=strategy,
            **common,
        )
        for output_phase, individual_phase in (
            ("prefill", "prefill"),
            ("decode", "generation"),
        ):
            individuals = [
                estimate_kv_cache_memory(
                    batch_size=1,
                    prompt_tokens=request["prompt_tokens"],
                    generated_tokens=request["generated_tokens"],
                    strategy=strategy,
                    **common,
                )[individual_phase]
                for request in requests
            ]
            assert aggregate[output_phase]["allocated_bytes"] == sum(
                int(item["allocated_bytes"]) for item in individuals
            )
            assert aggregate[output_phase][
                "storage_logical_bytes"
            ] == sum(
                int(item["storage_logical_bytes"])
                for item in individuals
            )


def test_serving_request_prefix_groups_follow_sliding_window() -> None:
    requests = [
        {
            "id": "short",
            "prompt_tokens": 8,
            "generated_tokens": 5,
            "prefix_group": "system",
            "shared_prefix_tokens": 4,
        },
        {
            "id": "long",
            "prompt_tokens": 20,
            "generated_tokens": 9,
            "prefix_group": "system",
            "shared_prefix_tokens": 4,
        },
        {
            "id": "completion",
            "prompt_tokens": 13,
            "generated_tokens": 3,
        },
    ]
    expected = {
        "dynamic": {
            "prefill": 1184,
            "decode": 1632,
        },
        "paged": {
            "prefill": 1280,
            "decode": 1664,
        },
    }

    for strategy in ("dynamic", "paged"):
        report = estimate_serving_request_kv_pool(
            requests,
            num_hidden_layers=2,
            num_key_value_heads=1,
            head_dim=4,
            element_bytes=2,
            strategy=strategy,
            block_tokens=4,
            window_tokens=24,
        )
        assert report["prefill"]["allocated_bytes"] == expected[
            strategy
        ]["prefill"]
        assert report["decode"]["allocated_bytes"] == expected[
            strategy
        ]["decode"]
        assert report["prefill"]["prefix_cache_savings_bytes"] == 128
        assert report["decode"]["prefix_cache_savings_bytes"] == 0
        assert report["prefill"]["requests"][1][
            "retained_shared_prefix_tokens"
        ] == 4
        assert report["decode"]["requests"][1][
            "retained_shared_prefix_tokens"
        ] == 0


def test_serving_request_kv_pool_scales_with_kv_head_architecture() -> None:
    requests = [
        {
            "id": "chat-a",
            "prompt_tokens": 4096,
            "generated_tokens": 257,
            "prefix_group": "system",
            "shared_prefix_tokens": 1024,
        },
        {
            "id": "chat-b",
            "prompt_tokens": 8192,
            "generated_tokens": 513,
            "prefix_group": "system",
            "shared_prefix_tokens": 1024,
        },
        {
            "id": "rag",
            "prompt_tokens": 12288,
            "generated_tokens": 129,
        },
    ]
    allocated_bytes = {}
    for architecture, kv_heads in (
        ("mha", 32),
        ("gqa", 8),
        ("mqa", 1),
    ):
        report = estimate_serving_request_kv_pool(
            requests,
            num_hidden_layers=32,
            num_key_value_heads=kv_heads,
            head_dim=128,
            element_bytes=2,
            strategy="paged",
            block_tokens=16,
        )
        allocated_bytes[architecture] = report["decode"][
            "allocated_bytes"
        ]

    assert allocated_bytes["mha"] == 4 * allocated_bytes["gqa"]
    assert allocated_bytes["gqa"] == 8 * allocated_bytes["mqa"]


def test_serving_request_set_admits_a_manifest_prefix(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    _write_model(model_dir)
    requests = [
        {
            "id": "chat-a",
            "prompt_tokens": 64,
            "generated_tokens": 17,
            "prefix_group": "system",
            "shared_prefix_tokens": 32,
        },
        {
            "id": "chat-b",
            "prompt_tokens": 96,
            "generated_tokens": 9,
            "prefix_group": "system",
            "shared_prefix_tokens": 32,
        },
        {
            "id": "completion",
            "prompt_tokens": 128,
            "generated_tokens": 33,
        },
    ]
    prefix_plan = estimate_serving_request_set(
        model_dir,
        requests[:2],
        max_batch_size=3,
        prefill_chunk_tokens=16,
        prefill_concurrency=2,
    )
    capacity_bytes = prefix_plan["memory_timeline"]["peak_bytes"]

    report = estimate_serving_request_set(
        model_dir,
        requests,
        max_batch_size=3,
        prefill_chunk_tokens=16,
        prefill_concurrency=2,
        device_capacity_bytes=capacity_bytes,
        memory_utilization=1,
    )

    assert report["schema_version"] == REQUEST_SET_SCHEMA_VERSION
    assert report["inputs"]["mode"] == "heterogeneous_request_set"
    assert report["accuracy"]["status"] == "uncalibrated"
    assert report["scheduler"]["admissible_active_sequences"] == 2
    assert report["scheduler"]["admitted_request_count"] == 2
    assert report["scheduler"]["rejected_request_count"] == 1
    assert report["scheduler"]["available_slots"] is None
    assert report["scheduler"]["admitted_request_ids"] == [
        "chat-a",
        "chat-b",
    ]
    assert report["scheduler"]["rejected_request_ids"] == [
        "completion"
    ]
    assert report["scheduler"]["requested_fits"] is False
    shapes = report["optimizations"]["continuous_batching"][
        "request_shapes"
    ]
    assert shapes["prompt_tokens"] == {
        "minimum": 64,
        "maximum": 128,
        "total": 288,
    }
    assert report["optimizations"]["chunked_prefill"][
        "effective_concurrency"
    ] == 2
    assert report["optimizations"]["chunked_prefill"][
        "transient_savings_bytes"
    ] > 0
    assert report["optimizations"]["prefix_cache"]["group_count"] == 1


def test_serving_request_set_handles_a_large_ordered_manifest(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    _write_model(model_dir)
    requests = []
    for index in range(128):
        request = {
            "id": f"request-{index:03d}",
            "prompt_tokens": (32, 64, 96, 128)[index % 4],
            "generated_tokens": (5, 9, 17)[index % 3],
        }
        if index % 4 in (0, 1):
            request.update(
                prefix_group=f"system-{index // 4}",
                shared_prefix_tokens=16,
            )
        requests.append(request)

    prefix_plan = estimate_serving_request_set(
        model_dir,
        requests[:64],
        max_batch_size=128,
        prefill_chunk_tokens=16,
        prefill_concurrency=8,
    )
    capacity_bytes = prefix_plan["memory_timeline"]["peak_bytes"]
    memory_limited = estimate_serving_request_set(
        model_dir,
        requests,
        max_batch_size=128,
        prefill_chunk_tokens=16,
        prefill_concurrency=8,
        device_capacity_bytes=capacity_bytes,
        memory_utilization=1,
    )

    assert memory_limited["scheduler"][
        "admissible_active_sequences"
    ] == 64
    assert memory_limited["scheduler"]["admitted_request_ids"] == [
        request["id"] for request in requests[:64]
    ]
    assert memory_limited["scheduler"]["rejected_request_ids"] == [
        request["id"] for request in requests[64:]
    ]
    assert memory_limited["scheduler"]["limiting_factor"] == (
        "usable_device_memory"
    )
    assert memory_limited["optimizations"]["chunked_prefill"][
        "effective_concurrency"
    ] == 8

    batch_limited = estimate_serving_request_set(
        model_dir,
        requests,
        max_batch_size=32,
        prefill_chunk_tokens=16,
        prefill_concurrency=8,
        device_capacity_bytes=2**30,
        memory_utilization=1,
    )
    assert batch_limited["scheduler"][
        "admissible_active_sequences"
    ] == 32
    assert batch_limited["scheduler"]["fits_configured_limit"] is False
    assert batch_limited["scheduler"]["fits_memory"] is True
    assert batch_limited["scheduler"]["limiting_factor"] == (
        "configured_max_batch_size"
    )
    assert batch_limited["scheduler"][
        "configured_candidate_request_ids"
    ] == [request["id"] for request in requests[:32]]


def test_serving_cli_accepts_a_request_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_dir = tmp_path / "model"
    manifest_path = tmp_path / "requests.json"
    output_path = tmp_path / "mixed-serving.json"
    _write_model(model_dir)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": REQUEST_MANIFEST_SCHEMA_VERSION,
                "requests": [
                    {
                        "id": "short",
                        "prompt_tokens": 16,
                        "generated_tokens": 4,
                    },
                    {
                        "id": "long",
                        "prompt_tokens": 64,
                        "generated_tokens": 16,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    assert (
        serving_main(
            [
                "--model-dir",
                str(model_dir),
                "--requests",
                str(manifest_path),
                "--max-batch-size",
                "8",
                "--prefill-chunk-tokens",
                "8",
                "--prefill-concurrency",
                "2",
                "--device-memory-gib",
                "1",
                "--json",
                str(output_path),
            ]
        )
        == 0
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == REQUEST_SET_SCHEMA_VERSION
    assert report["inputs"]["request_manifest"] == str(
        manifest_path.resolve()
    )
    assert report["scheduler"]["requested_active_sequences"] == 2
    assert report["scheduler"]["requested_fits"] is True
    output = capsys.readouterr().out
    assert "active sequences: 2" in output
    assert "accuracy: uncalibrated" in output

    with pytest.raises(SystemExit):
        serving_main(
            [
                "--model-dir",
                str(model_dir),
                "--requests",
                str(manifest_path),
                "--prompt-tokens",
                "16",
            ]
        )
    assert "--requests cannot be combined" in capsys.readouterr().err
