from __future__ import annotations

from copy import deepcopy
import math

import pytest

from fakegpu.finetune_config import normalize_finetune_config
from fakegpu.finetune_estimator import (
    FinetuneEstimateError,
    estimate_finetuning,
    finetune_workload_signature,
)
from fakegpu.profile_catalog import get_profile


def _model() -> dict:
    return {
        "model_type": "llama",
        "hidden_size": 64,
        "intermediate_size": 192,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 256,
        "tie_word_embeddings": False,
    }


def _training(**overrides) -> dict:
    config = normalize_finetune_config(
        {
            "framework": "huggingface",
            "model_name_or_path": "test/model",
            "bf16": True,
            "torch_dtype": "bfloat16",
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "num_train_epochs": 2,
            "num_samples": 100,
            "sequence_length": 32,
        }
    )
    config.update(overrides)
    return config


def _estimate(training=None, model=None, **kwargs) -> dict:
    return estimate_finetuning(
        model or _model(), training or _training(), profile_id="a100", **kwargs
    )


def _lora(**overrides) -> dict:
    return _training(
        method="lora",
        lora_rank=8,
        lora_target_modules=["q_proj", "v_proj"],
        **overrides,
    )


def _hybrid(moe=False) -> dict:
    model = _model()
    model.update(
        {
            "model_type": "qwen3_5_moe_text" if moe else "qwen3_5_text",
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
        }
    )
    if moe:
        model.pop("intermediate_size")
        model.update(
            num_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
            shared_expert_intermediate_size=32,
        )
    return model


def _observation(config=None) -> dict:
    return {
        "schema_version": "fakegpu.finetune_observation.v1",
        "signature": finetune_workload_signature(
            _model(), config or _training(), profile_id="a100"
        ),
        "seconds_per_microstep": [0.2, 0.3, 0.4],
        "peak_allocated_bytes": 1_000_000,
        "peak_reserved_bytes": 2_000_000,
    }


def test_steps_keep_partial_batches_and_accumulation_windows() -> None:
    report = _estimate(_training(num_samples=19, micro_batch_size=2, epochs=2))
    work = report["workload"]
    assert work["microsteps_per_epoch"] == 10
    assert work["optimizer_steps_per_epoch"] == 3
    assert work["microsteps"] == 20
    assert work["optimizer_steps"] == 6
    assert work["processed_samples"] == 38
    assert work["processed_tokens"] == 38 * 32
    assert work["padded_token_slots"] == 40 * 32


def test_max_steps_overrides_epochs_and_handles_short_last_update() -> None:
    report = _estimate(
        _training(num_samples=19, micro_batch_size=2, epochs=100, max_steps=4)
    )
    assert report["workload"]["optimizer_steps"] == 4
    assert report["workload"]["microsteps"] == 14
    assert report["workload"]["processed_samples"] == 27
    assert report["workload"]["stop_rule"] == "max_steps"


def test_fractional_epoch_rounds_to_complete_optimizer_update() -> None:
    report = _estimate(_training(num_samples=19, epochs=0.5))
    assert report["workload"]["optimizer_steps"] == 2
    assert report["workload"]["microsteps"] == 8


def test_drop_last_and_dataset_cap_are_not_confused() -> None:
    report = _estimate(
        _training(num_samples=19, max_samples=3, dataloader_drop_last=True)
    )
    assert report["workload"]["num_samples"] == 19
    assert report["workload"]["microsteps_per_epoch"] == 9
    assert report["workload"]["processed_samples"] == 36
    with pytest.raises(FinetuneEstimateError, match="every sample"):
        _estimate(_training(num_samples=1, dataloader_drop_last=True))


def test_lora_only_allocates_trainable_gradients_and_optimizer() -> None:
    full = _estimate()["memory"]
    lora = _estimate(_lora())["memory"]
    # q: (64,64), v: (64,32), four layers, rank 8.
    trainable = 4 * 8 * ((64 + 64) + (64 + 32))
    assert lora["trainable_parameters"] == trainable
    assert lora["adapter_weight_bytes"] == trainable * 4
    assert lora["gradient_bytes"] == trainable * 4
    assert lora["optimizer_state_bytes"] == trainable * 8
    assert lora["master_weight_bytes"] == 0
    assert lora["base_weight_bytes"] == full["base_weight_bytes"]
    assert full["optimizer_state_bytes"] == full["trainable_parameters"] * 4
    assert lora["persistent_bytes"] < full["persistent_bytes"]


def test_qlora_keeps_embeddings_dense_and_accounts_for_scale_metadata() -> None:
    lora = _estimate(_lora())["memory"]
    qlora = _estimate(_lora(quantization_bits=4) | {"method": "qlora"})["memory"]
    dense_parameters = qlora["total_base_parameters"] - qlora["quantized_parameters"]
    assert (
        qlora["quantization"]["scale_bytes"]
        == math.ceil(qlora["quantized_parameters"] / 64) * 4
    )
    assert qlora["quantization"]["codebook_bytes"] == 4 * 7 * 16 * 4
    assert (
        qlora["base_weight_bytes"]
        == math.ceil(qlora["quantized_parameters"] / 2)
        + qlora["quantization_metadata_bytes"]
        + dense_parameters * 4
    )
    assert qlora["optimizer_state_bytes"] == lora["optimizer_state_bytes"]
    assert qlora["base_weight_bytes"] < lora["base_weight_bytes"]


def test_double_quantization_changes_scales_not_four_bit_weight_payload() -> None:
    model = _model() | {"hidden_size": 256, "intermediate_size": 768}
    config = _lora(
        quantization_bits=4, quantization_type="nf4", double_quantization=False
    ) | {"method": "qlora"}
    plain = _estimate(config, model)["memory"]["quantization"]
    nested = _estimate(config | {"double_quantization": True}, model)["memory"][
        "quantization"
    ]
    assert nested["packed_weight_bytes"] == plain["packed_weight_bytes"]
    assert nested["scale_bytes"] * 4 == plain["scale_bytes"]
    assert nested["nested_scale_bytes"] > 0
    assert nested["offset_bytes"] == 4 * 7 * 4
    assert nested["metadata_bytes"] < plain["metadata_bytes"]
    fp4 = _estimate(config | {"quantization_type": "fp4"}, model)["memory"][
        "quantization"
    ]
    assert fp4["packed_weight_bytes"] == plain["packed_weight_bytes"]
    assert fp4["metadata_bytes"] == plain["metadata_bytes"]
    assert fp4["type"] == "fp4"


def test_quantized_storage_container_pads_each_tensor() -> None:
    model = _model() | {
        "hidden_size": 6,
        "intermediate_size": 9,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "num_hidden_layers": 1,
    }
    config = _lora(quantization_bits=4, quantization_storage="uint8") | {
        "method": "qlora"
    }
    byte = _estimate(config, model)["memory"]["quantization"]
    float32 = _estimate(config | {"quantization_storage": "fp32"}, model)["memory"][
        "quantization"
    ]
    assert byte["packed_weight_bytes"] == 4 * 18 + 3 * 27
    assert float32["packed_weight_bytes"] == 4 * 20 + 3 * 28
    assert byte["metadata_bytes"] == float32["metadata_bytes"]


def test_lora_dropout_adds_retained_inputs_and_mask_without_parameter_growth() -> None:
    off = _estimate(_lora(lora_dropout=0.0))["memory"]
    on = _estimate(_lora(lora_dropout=0.2))["memory"]
    assert on["persistent_bytes"] == off["persistent_bytes"]
    assert off["adapter_activation_bytes"] > 0
    assert off["dropout_activation_bytes"] == 0
    assert (
        on["activation_bytes"] - off["activation_bytes"]
        == on["dropout_activation_bytes"]
    )
    checkpointed = _estimate(_lora(lora_dropout=0.2, gradient_checkpointing=True))[
        "memory"
    ]
    assert checkpointed["dropout_activation_bytes"] < on["dropout_activation_bytes"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("learning_rate", 0.01),
        ("lora_alpha", 32),
        ("lora_dropout", 0.2),
        ("lora_bias", "all"),
        ("quantization_type", "nf4"),
        ("double_quantization", True),
        ("quantization_compute_dtype", "bf16"),
        ("quantization_storage", "bf16"),
    ],
)
def test_optimizer_adapter_and_quantization_options_enter_signature(
    field, value
) -> None:
    config = _lora()
    before = finetune_workload_signature(_model(), config, profile_id="a100")
    after = finetune_workload_signature(
        _model(), config | {field: value}, profile_id="a100"
    )
    assert before != after
    assert after[field] == value


def test_gradient_accumulation_does_not_multiply_persistent_memory() -> None:
    one = _estimate(_lora(gradient_accumulation_steps=1))
    eight = _estimate(_lora(gradient_accumulation_steps=8))
    assert one["memory"] == eight["memory"]
    assert one["workload"]["microsteps"] == eight["workload"]["microsteps"]
    assert eight["workload"]["optimizer_steps"] < one["workload"]["optimizer_steps"]


def test_checkpointing_trades_activation_memory_for_recomputation() -> None:
    off = _estimate()
    on = _estimate(_training(gradient_checkpointing=True))
    assert on["memory"]["persistent_bytes"] == off["memory"]["persistent_bytes"]
    assert on["memory"]["activation_bytes"] < off["memory"]["activation_bytes"]
    assert (
        on["timing"]["forward_backward_flops_per_microstep"]
        > off["timing"]["forward_backward_flops_per_microstep"]
    )


def test_qwen_hybrid_counts_gated_projection_only_in_full_attention_layers() -> None:
    report = _estimate(_lora(), _hybrid())
    assert report["model"]["full_attention_layers"] == 1
    assert report["model"]["linear_attention_layers"] == 3
    assert report["memory"]["trainable_parameters"] == 8 * ((64 + 128) + (64 + 32))
    assert report["memory"]["recurrent_state_bytes"] > 0
    assert report["coverage"] == "partial"
    assert report["memory"]["fit_status"] == "uncertain"


def test_official_qwen35_08b_dimensions_match_real_text_parameter_counts() -> None:
    # Qwen/Qwen3.5-0.8B config.json; counts independently measured with the
    # pretrained AutoModelForCausalLM + PEFT q_proj/v_proj, rank 8 on CUDA.
    model = {
        "model_type": "qwen3_5_text",
        "hidden_size": 1024,
        "intermediate_size": 3584,
        "num_hidden_layers": 24,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "vocab_size": 248320,
        "tie_word_embeddings": True,
        "full_attention_interval": 4,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
    }
    report = _estimate(_lora(), model)
    assert report["model"]["text_parameters"] == 752_393_024
    assert report["memory"]["trainable_parameters"] == 319_488


def test_moe_storage_and_active_compute_scale_separately() -> None:
    small_model = _hybrid(moe=True)
    larger_model = deepcopy(small_model)
    larger_model["num_experts"] = 16
    small = _estimate(model=small_model)
    larger = _estimate(model=larger_model)
    assert larger["memory"]["persistent_bytes"] > small["memory"]["persistent_bytes"]
    # Optimizer work scales with resident trainable expert parameters, while
    # forward/backward matmuls scale with top-k rather than expert count.
    optimizer_difference = (
        12
        * (
            larger["memory"]["trainable_parameters"]
            - small["memory"]["trainable_parameters"]
        )
        / 4
    )
    assert (
        larger["timing"]["forward_backward_flops_per_microstep"]
        - small["timing"]["forward_backward_flops_per_microstep"]
        == optimizer_difference
    )
    with pytest.raises(FinetuneEstimateError, match="target-parameter"):
        _estimate(_lora() | {"lora_target_modules": ["all-linear"]}, small_model)


def test_composite_vision_weights_remain_resident_but_inactive() -> None:
    text = _hybrid()
    composite = {
        "text_config": text,
        "vision_config": {
            "hidden_size": 32,
            "intermediate_size": 64,
            "depth": 2,
            "in_channels": 3,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 16,
            "spatial_merge_size": 2,
            "out_hidden_size": 64,
        },
    }
    text_report = _estimate(model=text)
    report = _estimate(model=composite)
    assert (
        report["memory"]["trainable_parameters"]
        == text_report["memory"]["trainable_parameters"]
    )
    assert (
        report["memory"]["base_weight_bytes"]
        > text_report["memory"]["base_weight_bytes"]
    )
    assert (
        report["timing"]["forward_backward_flops_per_microstep"]
        == text_report["timing"]["forward_backward_flops_per_microstep"]
    )
    with pytest.raises(FinetuneEstimateError, match="vision adapters"):
        _estimate(_lora() | {"lora_target_modules": ["all-linear"]}, composite)


def test_observation_extrapolates_data_volume_but_not_execution_shape() -> None:
    observation = _observation()
    report = _estimate(observation=observation, num_samples=200, epochs=3)
    assert report["timing"]["training_loop_seconds"]["expected"] == pytest.approx(90)
    assert report["timing"]["measurement_count"] == 3
    assert report["memory"]["selected_peak_bytes"] == 2_000_000 + 512 * 1024**2
    assert report["memory"]["fit_status"] == "supplied_observation_fits"
    assert report["timing"]["evidence"] == "supplied_observation_extrapolation"
    with pytest.raises(FinetuneEstimateError, match="signature mismatch"):
        _estimate(observation=observation, sequence_length=64)
    with pytest.raises(FinetuneEstimateError, match="signature mismatch"):
        estimate_finetuning(
            _model(), _training(), profile_id="rtx3090ti-reference", observation=observation
        )


@pytest.mark.parametrize(
    "change",
    [
        {"dtype": "fp32"},
        {"micro_batch_size": 1},
        {"gradient_accumulation_steps": 2},
        {"gradient_checkpointing": True},
        {"optimizer": "sgd"},
        {"software_versions": {"torch": "different"}},
        {"source_framework": "axolotl"},
        {"checkpoint_sha256": "different"},
    ],
)
def test_calibration_rejects_changed_execution_contract(change) -> None:
    with pytest.raises(FinetuneEstimateError, match="signature mismatch"):
        _estimate(_training(**change), observation=_observation())


def test_signature_is_owned_and_strict_about_model_metadata() -> None:
    training = _lora()
    signature = finetune_workload_signature(_model(), training, profile_id="a100")
    training["lora_target_modules"].append("k_proj")
    assert signature["lora_target_modules"] == ["q_proj", "v_proj"]
    changed_model = _model() | {"tie_word_embeddings": True}
    with pytest.raises(FinetuneEstimateError, match="signature mismatch"):
        _estimate(model=changed_model, observation=_observation())


def test_real_probe_requires_matching_provenance_and_uses_physical_capacity() -> None:
    training = _training(
        software_versions={"torch": "test", "transformers": "test"},
        checkpoint_sha256="a" * 64,
    )
    profile = get_profile("a100")
    observation = _observation(training) | {
        "source": "fakegpu.huggingface_trainer_cuda_probe",
        "checkpoint_identity": {
            "algorithm": "sha256_content_manifest_v1",
            "model_dir": training["model_name_or_path"],
            "sha256": training["checkpoint_sha256"],
        },
        "hardware": {
            "name": profile.name,
            "compute_capability": profile.compute_capability_text,
            "total_memory_bytes": profile.memory_bytes - 1_000_000,
        },
        "software_versions": training["software_versions"],
        "workload": {"sequence_length": 32, "measured_optimizer_steps": 3},
    }
    report = _estimate(training, observation=observation)
    assert report["timing"]["evidence"] == "matched_real_measurement_extrapolation"
    assert report["memory"]["fit_status"] == "observed_probe_fits"
    assert report["memory"]["device_capacity_bytes"] == profile.memory_bytes - 1_000_000
    with pytest.raises(FinetuneEstimateError, match="hardware"):
        _estimate(training, observation=observation | {"hardware": {}})
    with pytest.raises(FinetuneEstimateError, match="software_versions"):
        _estimate(training, observation=observation | {"software_versions": {}})
    with pytest.raises(FinetuneEstimateError, match="workload"):
        _estimate(training, observation=observation | {"workload": {}})
    with pytest.raises(FinetuneEstimateError, match="checkpoint content identity"):
        _estimate(training, observation=observation | {"checkpoint_identity": {}})


def test_actual_parameter_counts_override_shape_counts_and_enter_signature() -> None:
    report = _estimate(
        _lora(), parameter_count=1_000_000, trainable_parameter_count=8_000
    )
    assert report["memory"]["base_weight_bytes"] == 2_000_000
    assert report["memory"]["adapter_weight_bytes"] == 32_000
    assert report["signature"]["parameter_count"] == 1_000_000
    assert report["signature"]["trainable_parameter_count"] == 8_000
    assert report["model"]["parameter_source"] == "provided_parameter_count"


def test_amp_does_not_silently_downcast_base_weight_storage() -> None:
    report = _estimate(_training(parameter_dtype=None))
    assert report["memory"]["parameter_dtype"] == "fp32"
    assert (
        report["memory"]["base_weight_bytes"]
        == report["memory"]["total_base_parameters"] * 4
    )
    assert report["memory"]["master_weight_bytes"] == 0


@pytest.mark.parametrize("dtype_name", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize(
    "optimizer_name", ["adamw_torch", "adamw_torch_fused", "sgd", "sgd_momentum"]
)
def test_torch_optimizer_storage_matches_real_cpu_step(
    dtype_name, optimizer_name
) -> None:
    torch = pytest.importorskip("torch")
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
        dtype_name
    ]
    parameter = torch.nn.Parameter(torch.ones(8, dtype=dtype, device="cpu"))
    if optimizer_name.startswith("adamw"):
        optimizer = torch.optim.AdamW(
            [parameter], lr=0.01, eps=1e-4, fused=optimizer_name.endswith("fused")
        )
        state_names = {"exp_avg", "exp_avg_sq"}
    else:
        optimizer = torch.optim.SGD(
            [parameter],
            lr=0.01,
            momentum=0.9 if optimizer_name == "sgd_momentum" else 0.0,
        )
        state_names = {"momentum_buffer"} if optimizer_name == "sgd_momentum" else set()
    parameter.grad = torch.full_like(parameter, 0.125)
    optimizer.step()
    state = optimizer.state[parameter]
    assert set(state) - {"step"} == state_names
    observed_bytes = sum(
        state[name].numel() * state[name].element_size() for name in state_names
    )
    assert all(state[name].dtype == dtype for name in state_names)
    memory = _estimate(
        _training(parameter_dtype=dtype_name, optimizer=optimizer_name),
        parameter_count=parameter.numel(),
    )["memory"]
    assert memory["optimizer_state_bytes"] == observed_bytes
    assert memory["master_weight_bytes"] == 0
    assert memory["optimizer_state_basis"] == "pytorch_parameter_dtype_no_master_copy"


def test_other_optimizer_implementations_keep_explicit_conservative_budget() -> None:
    memory = _estimate(_training(optimizer="adamw_apex_fused"))["memory"]
    assert memory["optimizer_state_bytes"] == memory["trainable_parameters"] * 8
    assert memory["master_weight_bytes"] == memory["trainable_parameters"] * 4
    assert (
        memory["optimizer_state_basis"] == "conservative_fp32_state_and_master_budget"
    )


def test_oom_status_distinguishes_analytical_estimate_from_measured_budget() -> None:
    capacity = get_profile("a100").memory_bytes
    analytical = _estimate(runtime_overhead_bytes=capacity)["memory"]
    supplied = _estimate(observation=_observation(), runtime_overhead_bytes=capacity)[
        "memory"
    ]
    assert analytical["fit_status"] == "estimated_oom"
    assert supplied["fit_status"] == "budget_exceeds_capacity"


@pytest.mark.parametrize(
    "change,match",
    [
        ({"world_size": 2}, "one GPU"),
        ({"distributed_config": {"zero_stage": 2}}, "sharding/offload"),
        ({"optimizer": "paged_adamw_8bit"}, "unsupported optimizer"),
        ({"num_samples": 0}, "num_samples"),
        ({"num_samples": True}, "num_samples"),
        ({"epochs": float("nan")}, "epochs"),
        ({"sequence_length": 0}, "sequence_length"),
    ],
)
def test_invalid_or_unsupported_configurations_fail_explicitly(change, match) -> None:
    with pytest.raises(FinetuneEstimateError, match=match):
        _estimate(_training(**change))


@pytest.mark.parametrize(
    "change,match",
    [
        ({"seconds_per_microstep": [float("nan")]}, "finite positive"),
        ({"seconds_per_microstep": []}, "at least two"),
        ({"seconds_per_microstep": [0.1]}, "at least two"),
        ({"peak_allocated_bytes": 0}, "positive integer"),
        ({"peak_reserved_bytes": 1}, "cannot be less"),
        ({"schema_version": "unknown"}, "schema"),
    ],
)
def test_invalid_observations_are_never_used(change, match) -> None:
    with pytest.raises(FinetuneEstimateError, match=match):
        _estimate(observation=_observation() | change)


def test_unknown_architecture_cannot_silently_use_dense_decoder_formula() -> None:
    with pytest.raises(FinetuneEstimateError, match="unsupported text model_type"):
        _estimate(model=_model() | {"model_type": "new_custom_architecture"})


def test_runtime_budget_and_analytical_intervals_are_exposed() -> None:
    report = _estimate(runtime_overhead_bytes=1234)
    assert report["memory"]["runtime_overhead_bytes"] == 1234
    peak = report["memory"]["peak_bytes"]
    time = report["timing"]["training_loop_seconds"]
    assert peak["lower"] < peak["expected"] < peak["upper"]
    assert time["lower"] < time["expected"] < time["upper"]
    assert report["timing"]["roofline"]["compute_model"]["basis"] == "scalar_fp32_fma"
    assert report["timing"]["evidence"] == "analytical_scenario_range"


def test_optimizer_step_counts_against_huggingface_trainer(tmp_path) -> None:
    """Real CPU Trainer checks must not inherit Torch/CUDA caches from pytest."""
    from importlib import metadata
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys

    from packaging.version import Version

    versions = {}
    for dependency in ("transformers", "torch", "accelerate"):
        try:
            versions[dependency] = metadata.version(dependency)
        except metadata.PackageNotFoundError:
            pytest.skip(f"optional Trainer regression requires {dependency}")
    if Version(versions["transformers"]) < Version("4.57"):
        pytest.skip("optional Trainer regression requires Transformers >= 4.57")
    # Check installed versions without importing the frameworks in this process.
    # A broken installed dependency fails in the child and must not be skipped.
    root = Path(__file__).resolve().parents[1]
    cache_dir = tmp_path / "triton-cache"
    cache_dir.mkdir()
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TRITON_CACHE_DIR": str(cache_dir),
            "PYTHONPATH": str(root) + os.pathsep + env.get("PYTHONPATH", ""),
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            str(root / "tests/support/finetune_trainer_counts.py"),
            str(tmp_path),
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "trainer-counts.json").read_text())
    assert len(report["cases"]) == 12
    assert report["transformers"] == versions["transformers"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("actual_model_signature", "a" * 64),
        ("gradient_checkpointing_kwargs", {"use_reentrant": False}),
    ],
)
def test_native_execution_details_bind_calibration_without_changing_old_signatures(
    field, value
):
    config = _lora()
    before = finetune_workload_signature(_model(), config, profile_id="a100")
    assert field not in before
    after = finetune_workload_signature(
        _model(), config | {field: value}, profile_id="a100"
    )
    assert after[field] == value
    assert {key: item for key, item in after.items() if key != field} == before


def _native_qlora_storage_config():
    config = _lora()
    config.update(
        method="qlora",
        quantization_bits=4,
        quantization_type="nf4",
        quantization_storage="bf16",
        quantization_compute_dtype="bf16",
        double_quantization=True,
        parameter_count=1_000_000,
        actual_model_signature="a" * 64,
        actual_quantized_storage={
            "quantized_parameters": 800_000,
            "packed_weight_bytes": 400_000,
            "metadata_bytes": 14_000,
            "unquantized_weight_bytes": 400_000,
        },
    )
    return config


def test_native_qlora_uses_inspected_mixed_precision_weights_and_quantized_subset():
    report = _estimate(_native_qlora_storage_config())
    memory = report["memory"]
    assert memory["base_weight_bytes"] == 814_000
    assert memory["quantized_parameters"] == 800_000
    assert memory["quantization_metadata_bytes"] == 14_000
    assert (
        memory["quantization"]["basis"]
        == "inspected_native_bitsandbytes_tensor_storage"
    )
    assert (
        report["signature"]["actual_quantized_storage"]["unquantized_weight_bytes"]
        == 400_000
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("quantized_parameters", 1_000_001),
        ("quantized_parameters", 0),
        ("packed_weight_bytes", 1),
        ("metadata_bytes", -1),
        ("metadata_bytes", True),
        ("metadata_bytes", 1.5),
        ("unquantized_weight_bytes", 800_001),
        ("unquantized_weight_bytes", 399_999),
    ],
)
def test_invalid_native_quantized_inventory_is_rejected(field, value):
    config = _native_qlora_storage_config()
    config["actual_quantized_storage"][field] = value
    with pytest.raises(FinetuneEstimateError, match="actual_quantized_storage"):
        _estimate(config)


def test_native_quantized_inventory_must_bind_a_model_and_complete_schema():
    config = _native_qlora_storage_config()
    del config["actual_model_signature"]
    with pytest.raises(FinetuneEstimateError, match="actual_model_signature"):
        _estimate(config)
    config = _native_qlora_storage_config()
    del config["actual_quantized_storage"]["metadata_bytes"]
    with pytest.raises(FinetuneEstimateError, match="invalid fields"):
        _estimate(config)
