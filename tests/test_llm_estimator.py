from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import pytest

from fakegpu.llm_estimator import (
    estimate_decoder_inference,
    estimate_kv_cache_memory,
    inspect_safetensors_checkpoint,
)
from fakegpu.llm_cli import main as llm_main
from fakegpu.serving_plan import (
    SCHEMA_VERSION as SERVING_SCHEMA_VERSION,
    ServingPlanError,
    estimate_serving_kv_pool,
    estimate_serving_plan,
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
    dtype_bytes = {"BF16": 2, "F16": 2, "F32": 4, "I8": 1, "U8": 1}
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
    header = {
        "model.embed.weight": {
            "dtype": checkpoint_dtype,
            "shape": [32, 8],
        },
        "model.proj.weight": {
            "dtype": checkpoint_dtype,
            "shape": [8, 8],
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
    assert quantized["prefill"]["allocated_bytes"] == 80
    assert quantized["generation"]["allocated_bytes"] == 88
    assert quantized["generation"][
        "full_precision_residual_tokens_per_sequence"
    ] == 2
    assert quantized["generation"]["quantized_tokens_per_sequence"] == 3
    assert quantized["generation"]["quantization_savings_bytes"] == 72
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


def test_serving_prefix_segments_are_rounded_independently() -> None:
    report = estimate_serving_kv_pool(
        num_hidden_layers=2,
        num_key_value_heads=1,
        head_dim=4,
        active_sequences=2,
        prompt_tokens=17,
        element_bytes=2,
        strategy="paged",
        shared_prefix_tokens=1,
        block_tokens=16,
    )

    prefill = report["prefill"]
    assert prefill["shared_segment"]["allocated_tokens_per_sequence"] == 16
    assert prefill["private_segments"][
        "allocated_tokens_per_sequence"
    ] == 16
    assert prefill["without_prefix_cache_bytes"] == 2048
    assert prefill["allocated_bytes"] == 1536
    assert prefill["reservation_overhead_bytes"] == 480


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
