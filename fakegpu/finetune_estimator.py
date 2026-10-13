"""Single-GPU text fine-tuning estimates with explicit coverage and evidence.

No framework, checkpoint tensors, or CUDA runtime are imported. Shape formulas
are deliberately separate from measured observations: an analytical interval is
not a confidence interval, and an unmodelled kernel cannot establish OOM safety.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping
from typing import Any

from .llm_estimator import _decoder_dimensions
from .performance_model import estimate_roofline
from .profile_catalog import get_profile


SCHEMA_VERSION = "fakegpu.finetune_estimate.v1"
OBSERVATION_SCHEMA_VERSION = "fakegpu.finetune_observation.v1"
_SUPPORTED_MODELS = {
    "llama",
    "mistral",
    "qwen2",
    "qwen3",
    "qwen3_moe",
    "qwen3_5_text",
    "qwen3_5_moe_text",
}
_DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2}


class FinetuneEstimateError(ValueError):
    """A configuration cannot produce a supported, meaningful estimate."""


def finetune_workload_signature(
    model_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    *,
    profile_id: str,
    sequence_length: int | None = None,
) -> dict[str, Any]:
    """Identify an execution shape; dataset size and epochs can be extrapolated.

    ``software_versions`` and ``execution_backend`` may be supplied by a probe
    to distinguish kernel/runtime stacks. Missing values remain explicit in the
    signature, rather than accepting observations from a different stack.
    """
    keys = (
        "source_framework",
        "model_name_or_path",
        "method",
        "dtype",
        "tf32",
        "parameter_dtype",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "lora_rank",
        "lora_target_modules",
        "lora_alpha",
        "lora_dropout",
        "lora_bias",
        "gradient_checkpointing",
        "quantization_bits",
        "quantization_type",
        "double_quantization",
        "quantization_compute_dtype",
        "quantization_storage",
        "optimizer",
        "learning_rate",
        "lr_scheduler_type",
        "warmup_steps",
        "warmup_ratio",
        "weight_decay",
        "max_grad_norm",
        "packing",
        "dataloader_drop_last",
        "world_size",
        "distributed_config",
        "attention_implementation",
        "software_versions",
        "execution_backend",
        "modules_to_save",
        "parameter_count",
        "trainable_parameter_count",
        "model_revision",
        "checkpoint_sha256",
    )
    profile = get_profile(profile_id)
    model_json = json.dumps(dict(model_config), sort_keys=True, separators=(",", ":"))
    fields = {key: training_config.get(key) for key in keys}
    # Native runners bind preprocessing and framework-specific execution choices.
    # Keep absent fields absent so existing v1 fixed-shape probe records still match.
    for key in (
        "data_signature",
        "native_execution_signature",
        "actual_model_signature",
        "actual_quantized_storage",
        "gradient_checkpointing_kwargs",
        "batch_observation_version",
        "event_observation_version",
    ):
        if key in training_config:
            fields[key] = training_config[key]
    fields["sequence_length"] = (
        sequence_length
        if sequence_length is not None
        else training_config.get("sequence_length")
    )
    fields["model_config_sha256"] = hashlib.sha256(model_json.encode()).hexdigest()
    fields["profile"] = {
        "id": profile.id,
        "memory_bytes": profile.memory_bytes,
        "compute_capability": profile.compute_capability_text,
        "sm_count": profile.sm_count,
        "core_clock_mhz": profile.core_clock_mhz,
        "memory_clock_mhz": profile.memory_clock_mhz,
        "memory_bus_width_bits": profile.memory_bus_width_bits,
    }
    # JSON roundtrip owns all nested values; callers cannot mutate a signature
    # by later changing the normalized configuration.
    return json.loads(json.dumps(fields, sort_keys=True))


def estimate_finetuning(
    model_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    *,
    profile_id: str,
    num_samples: int | None = None,
    epochs: float | None = None,
    sequence_length: int | None = None,
    observation: Mapping[str, Any] | None = None,
    compute_acceleration_factor: float = 1.0,
    runtime_overhead_bytes: int = 512 * 1024**2,
    parameter_count: int | None = None,
    trainable_parameter_count: int | None = None,
) -> dict[str, Any]:
    """Estimate text SFT training-loop time and peak single-GPU VRAM.

    Input is the normalized output of ``normalize_finetune_config``. Samples
    mean tokenized sequences after packing/filtering; sequence_length is the
    padded execution length. Download, preprocessing, evaluation, checkpoint
    saves and initialization are excluded from training-loop time.
    """
    config = dict(training_config)
    _check_supported_training(config)
    for key, override in (
        ("num_samples", num_samples),
        ("epochs", epochs),
        ("sequence_length", sequence_length),
        ("parameter_count", parameter_count),
        ("trainable_parameter_count", trainable_parameter_count),
    ):
        if override is not None:
            config[key] = override
    workload = _workload(config)
    profile = get_profile(profile_id)
    if (
        not isinstance(runtime_overhead_bytes, int)
        or isinstance(runtime_overhead_bytes, bool)
        or runtime_overhead_bytes < 0
    ):
        raise FinetuneEstimateError(
            "runtime_overhead_bytes must be a non-negative integer"
        )
    model = _model_shapes(model_config)
    if config.get("parameter_count") is not None:
        measured_count = _positive_int(config["parameter_count"], "parameter_count")
        if measured_count <= model["vision_parameters"]:
            raise FinetuneEstimateError(
                "parameter_count must exceed resident vision parameters"
            )
        model["total_parameters"] = measured_count
        model["text_parameters"] = measured_count - model["vision_parameters"]
        model["parameter_source"] = "provided_parameter_count"
    signature = finetune_workload_signature(model_config, config, profile_id=profile_id)
    memory = _memory(model, config, workload, runtime_overhead_bytes)
    flops = _training_flops(model, config, workload, memory["trainable_parameters"])
    roofline = estimate_roofline(
        profile_id=profile_id,
        flops=flops,
        memory_bytes=memory["persistent_bytes"] * 2 + memory["activation_bytes"] * 3,
        launch_count=model["dimensions"]["num_hidden_layers"] * 30,
        compute_acceleration_factor=compute_acceleration_factor,
    )
    step_interval = roofline["latency_interval_seconds"]
    timing = {
        "evidence": "analytical_scenario_range",
        "seconds_per_microstep": step_interval,
        "training_loop_seconds": {
            key: value * workload["microsteps"] for key, value in step_interval.items()
        },
        "forward_backward_flops_per_microstep": flops,
        "roofline": roofline,
        "includes": ["forward", "backward", "amortized_optimizer_step"],
        "excludes": [
            "download",
            "tokenization",
            "initialization",
            "evaluation",
            "checkpoint_io",
        ],
    }
    limitations = [
        "Time covers the training loop, not download, preprocessing, initialization, evaluation or checkpoint I/O.",
        "Analytical ranges are scenarios, not statistical confidence intervals or guaranteed memory bounds.",
        "Samples must count tokenized sequences after packing/filtering; all sequences are assumed padded to sequence_length.",
        "Optimizer step counts flush partial accumulation windows at epoch boundaries, as verified with Transformers 4.57.3 and 5.12.1; older or custom trainer schedules may differ.",
        "Time extrapolation uses full-microbatch measurements for tail batches; short accumulation windows may have different amortized optimizer overhead.",
        "Activation/workspace storage depends on fused kernels, allocator behavior and framework versions; calibrate on the target stack.",
        "Explicit PyTorch AdamW/SGD uses parameter-dtype optimizer states without a master copy; other implementations use a conservative FP32 state/master-weight budget.",
        "Compute precision does not determine parameter storage: absent parameter_dtype, base weights are budgeted as FP32.",
        "Automatic attention dispatch budgets eager score storage; explicit fused backends and calibration can refine it.",
    ] + model["limitations"]
    if config.get("packing"):
        limitations.append(
            "Packing is enabled: raw dataset row counts must be converted to packed sequence counts before estimation."
        )
    if config["method"] == "qlora":
        limitations.append(
            "QLoRA uses inspected packed weights, quantization metadata and unquantized tensor bytes when provided by a native run; otherwise it budgets default bitsandbytes blocks and FP32 unquantized tensors as in PEFT k-bit preparation. Dequantization/fusion performance remains analytical."
        )
    if config.get("lora_dropout"):
        limitations.append(
            "LoRA dropout includes FP32 adapter inputs, retained dropout outputs and byte masks; fused dropout and random-number generation costs are not benchmarked."
        )
    if observation is not None:
        measured = _validate_observation(observation, signature)
        real_probe = measured["verified_probe_metadata"]
        times = measured["seconds_per_microstep"]
        interval = {
            "lower": min(times),
            "expected": statistics.mean(times),
            "upper": max(times),
        }
        timing.update(
            {
                "evidence": (
                    "matched_real_measurement_extrapolation"
                    if real_probe
                    else "supplied_observation_extrapolation"
                ),
                "seconds_per_microstep": interval,
                "training_loop_seconds": {
                    key: value * workload["microsteps"]
                    for key, value in interval.items()
                },
                "measurement_count": len(times),
                "sample_standard_deviation_seconds": statistics.stdev(times)
                if len(times) > 1
                else None,
                "interval_kind": "observed_min_max_not_confidence_interval",
            }
        )
        memory["observed_peak_allocated_bytes"] = measured["peak_allocated_bytes"]
        memory["observed_peak_reserved_bytes"] = measured.get("peak_reserved_bytes")
        measured_peak = measured.get(
            "peak_reserved_bytes", measured["peak_allocated_bytes"]
        )
        memory["selected_peak_bytes"] = measured_peak + runtime_overhead_bytes
        memory["evidence"] = (
            "matched_real_measurement_plus_runtime_budget"
            if real_probe
            else "supplied_observation_plus_runtime_budget"
        )
        limitations.append(
            "Measured extrema describe the probe only; long-run allocator growth and workload drift remain unmeasured."
        )
    else:
        memory["selected_peak_bytes"] = memory["peak_bytes"]["expected"]
        memory["evidence"] = "analytical_scenario_range"
    capacity = profile.memory_bytes
    if observation is not None and measured["verified_probe_metadata"]:
        capacity = min(capacity, measured["hardware"]["total_memory_bytes"])
    memory["device_capacity_bytes"] = capacity
    memory["estimated_headroom_bytes"] = capacity - memory["selected_peak_bytes"]
    if observation is not None:
        memory["fit_status"] = (
            (
                "observed_probe_fits"
                if measured["verified_probe_metadata"]
                else "supplied_observation_fits"
            )
            if memory["selected_peak_bytes"] <= capacity
            else "budget_exceeds_capacity"
        )
    elif memory["peak_bytes"]["lower"] > profile.memory_bytes:
        memory["fit_status"] = "estimated_oom"
    elif (
        model["coverage"] == "partial"
        or memory["peak_bytes"]["upper"] > profile.memory_bytes
    ):
        memory["fit_status"] = "uncertain"
    else:
        memory["fit_status"] = "estimated_fit"
    return {
        "schema_version": SCHEMA_VERSION,
        "source_framework": config.get("source_framework"),
        "method": config["method"],
        "profile_id": profile_id,
        "coverage": model["coverage"],
        "model": {
            key: value for key, value in model.items() if key != "linear_modules"
        },
        "workload": workload,
        "memory": memory,
        "timing": timing,
        "signature": signature,
        "assumptions": list(config.get("assumptions", [])),
        "limitations": limitations,
    }


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise FinetuneEstimateError(f"{name} must be a positive integer")
    return value


def _check_supported_training(config: Mapping[str, Any]) -> None:
    if config.get("method") not in {"full", "lora", "qlora"}:
        raise FinetuneEstimateError("method must be full, lora, or qlora")
    if config.get("dtype") not in _DTYPE_BYTES:
        raise FinetuneEstimateError("dtype must be fp32, fp16, or bf16")
    distributed = config.get("distributed_config") or {}
    unsupported = (
        config.get("world_size", 1) != 1
        or distributed.get("zero_stage", 0) != 0
        or distributed.get("sharding_strategy", "replicated") != "replicated"
        or any(
            distributed.get(key, "none") != "none"
            for key in ("parameter_offload_device", "optimizer_offload_device")
        )
    )
    if unsupported:
        raise FinetuneEstimateError(
            "fine-tuning time estimation currently requires one GPU without sharding/offload; use training-plan for distributed memory"
        )
    optimizer = str(config.get("optimizer", "adamw_torch")).lower()
    if not (
        optimizer in {"adam", "sgd", "sgd_momentum"} or optimizer.startswith("adamw")
    ) or any(token in optimizer for token in ("8bit", "paged", "bnb")):
        raise FinetuneEstimateError(f"unsupported optimizer memory model: {optimizer}")
    if config.get("method") == "qlora" and config.get("quantization_bits") != 4:
        raise FinetuneEstimateError("QLoRA currently requires 4-bit base weights")
    if config.get("method") != "qlora" and config.get("quantization_bits") is not None:
        raise FinetuneEstimateError("quantized base weights require method=qlora")
    if config.get("method") != "full":
        dropout = config.get("lora_dropout", 0.0)
        if dropout is None:
            dropout = 0.0
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (int, float))
            or not math.isfinite(dropout)
            or not 0 <= dropout < 1
        ):
            raise FinetuneEstimateError("lora_dropout must be finite and in [0, 1)")
        if config.get("lora_bias") not in {None, "none"}:
            raise FinetuneEstimateError("LoRA bias training is not covered")
    if config.get("modules_to_save"):
        raise FinetuneEstimateError(
            "modules_to_save requires explicit trainable tensor inspection and is not covered"
        )


def _workload(config: Mapping[str, Any]) -> dict[str, Any]:
    samples = _positive_int(config.get("num_samples"), "num_samples")
    batch = _positive_int(config.get("micro_batch_size"), "micro_batch_size")
    accumulation = _positive_int(
        config.get("gradient_accumulation_steps"), "gradient_accumulation_steps"
    )
    sequence = _positive_int(config.get("sequence_length"), "sequence_length")
    epochs = config.get("epochs", 1.0)
    if (
        isinstance(epochs, bool)
        or not isinstance(epochs, (int, float))
        or not math.isfinite(epochs)
        or epochs <= 0
    ):
        raise FinetuneEstimateError("epochs must be finite and positive")
    per_epoch = (
        samples // batch
        if config.get("dataloader_drop_last")
        else math.ceil(samples / batch)
    )
    if per_epoch == 0:
        raise FinetuneEstimateError("dataloader_drop_last removes every sample")
    updates_per_epoch = math.ceil(per_epoch / accumulation)
    max_steps = config.get("max_steps")
    if max_steps is not None:
        optimizer_steps = _positive_int(max_steps, "max_steps")
        complete_epochs, remainder = divmod(optimizer_steps, updates_per_epoch)
        microsteps = complete_epochs * per_epoch + min(
            remainder * accumulation, per_epoch
        )
        stop_rule = "max_steps"
    else:
        # Finish optimizer updates rather than stopping halfway through an
        # accumulation window; the last update of each epoch can be shorter.
        optimizer_steps = math.ceil(float(epochs) * updates_per_epoch)
        complete_epochs, remainder = divmod(optimizer_steps, updates_per_epoch)
        microsteps = complete_epochs * per_epoch + min(
            remainder * accumulation, per_epoch
        )
        stop_rule = "epochs"
    full_epochs, partial_microsteps = divmod(microsteps, per_epoch)
    samples_per_epoch = min(samples, per_epoch * batch)
    processed_samples = full_epochs * samples_per_epoch + min(
        partial_microsteps * batch, samples_per_epoch
    )
    return {
        "num_samples": samples,
        "epochs": float(epochs),
        "stop_rule": stop_rule,
        "micro_batch_size": batch,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": batch * accumulation,
        "microsteps_per_epoch": per_epoch,
        "optimizer_steps_per_epoch": updates_per_epoch,
        "microsteps": microsteps,
        "optimizer_steps": optimizer_steps,
        "sequence_length": sequence,
        "processed_samples": processed_samples,
        "processed_tokens": processed_samples * sequence,
        "padded_token_slots": microsteps * batch * sequence,
        "effective_epochs": processed_samples / samples,
    }


def _model_shapes(config: Mapping[str, Any]) -> dict[str, Any]:
    text = dict(config.get("text_config", config))
    kind = str(text.get("model_type", ""))
    if kind not in _SUPPORTED_MODELS:
        raise FinetuneEstimateError(
            f"unsupported text model_type {kind!r}; supported: {', '.join(sorted(_SUPPORTED_MODELS))}"
        )
    if text.get("kv_lora_rank"):
        raise FinetuneEstimateError(
            "multi-head latent attention is not covered by the fine-tuning model"
        )
    if "intermediate_size" not in text and "moe_intermediate_size" in text:
        text["intermediate_size"] = text["moe_intermediate_size"]
    dimensions = _decoder_dimensions(text)
    hidden = dimensions["hidden_size"]
    layers = dimensions["num_hidden_layers"]
    heads = dimensions["num_attention_heads"]
    head_dim = dimensions["head_dim"]
    kv_width = dimensions["num_key_value_heads"] * head_dim
    hybrid = kind.startswith("qwen3_5")
    layer_types = text.get("layer_types")
    if layer_types is None:
        interval = _positive_int(
            text.get("full_attention_interval", 4), "full_attention_interval"
        )
        layer_types = [
            "full_attention"
            if not hybrid or (i + 1) % interval == 0
            else "linear_attention"
            for i in range(layers)
        ]
    if (
        not isinstance(layer_types, list)
        or len(layer_types) != layers
        or set(layer_types) - {"full_attention", "linear_attention"}
    ):
        raise FinetuneEstimateError(
            "layer_types must describe every full_attention/linear_attention layer"
        )
    if "linear_attention" in layer_types and not hybrid:
        raise FinetuneEstimateError(
            "linear attention is currently modelled only for Qwen3.5"
        )
    full_layers = layer_types.count("full_attention")
    linear_layers = layers - full_layers
    modules: list[tuple[str, int, int, int]] = []

    def linear(name: str, width_in: int, width_out: int, count: int) -> None:
        if count:
            modules.append((name, width_in, width_out, count))

    q_width = heads * head_dim * (2 if hybrid else 1)
    for name, width_in, width_out in (
        ("q_proj", hidden, q_width),
        ("k_proj", hidden, kv_width),
        ("v_proj", hidden, kv_width),
        ("o_proj", heads * head_dim, hidden),
    ):
        linear(name, width_in, width_out, full_layers)
    nonmatrix = (2 * layers + 1) * hidden
    if hybrid or kind.startswith("qwen3"):
        nonmatrix += 2 * head_dim * full_layers
    if text.get("attention_bias", kind == "qwen2"):
        nonmatrix += full_layers * (
            q_width + 2 * kv_width + (hidden if kind != "qwen2" else 0)
        )
    recurrent = None
    if linear_layers:
        recurrent = {
            key: _positive_int(text.get(key), key)
            for key in (
                "linear_num_key_heads",
                "linear_num_value_heads",
                "linear_key_head_dim",
                "linear_value_head_dim",
                "linear_conv_kernel_dim",
            )
        }
        key_width = recurrent["linear_num_key_heads"] * recurrent["linear_key_head_dim"]
        value_width = (
            recurrent["linear_num_value_heads"] * recurrent["linear_value_head_dim"]
        )
        conv_width = 2 * key_width + value_width
        for name, width_in, width_out in (
            ("in_proj_qkv", hidden, conv_width),
            ("in_proj_z", hidden, value_width),
            ("in_proj_b", hidden, recurrent["linear_num_value_heads"]),
            ("in_proj_a", hidden, recurrent["linear_num_value_heads"]),
            ("out_proj", value_width, hidden),
        ):
            linear(name, width_in, width_out, linear_layers)
        nonmatrix += linear_layers * (
            conv_width * recurrent["linear_conv_kernel_dim"]
            + 2 * recurrent["linear_num_value_heads"]
            + recurrent["linear_value_head_dim"]
        )
    dense_layers = dimensions["dense_layer_count"]
    moe_layers = dimensions["moe_layer_count"]
    for width, count in (
        (dimensions["intermediate_size"], dense_layers),
        (
            dimensions["moe_intermediate_size"],
            moe_layers * dimensions["num_routed_experts"],
        ),
        (dimensions["shared_expert_intermediate_size"], moe_layers),
    ):
        if width:
            for name, width_in, width_out in (
                ("gate_proj", hidden, width),
                ("up_proj", hidden, width),
                ("down_proj", width, hidden),
            ):
                linear(name, width_in, width_out, count)
    nonmatrix += (
        moe_layers
        * hidden
        * (
            dimensions["num_routed_experts"]
            + bool(dimensions["shared_expert_intermediate_size"])
        )
    )
    embedding = dimensions["vocab_size"] * hidden
    tied = config.get("tie_word_embeddings", text.get("tie_word_embeddings", False))
    text_parameters = (
        sum(i * o * n for _, i, o, n in modules)
        + nonmatrix
        + embedding * (1 if tied else 2)
    )
    vision_parameters = 0
    limitations: list[str] = []
    if config.get("vision_config") is not None:
        if not hybrid:
            raise FinetuneEstimateError(
                "vision residency is currently modelled only for Qwen3.5"
            )
        vision_parameters = _qwen_vision_parameters(config["vision_config"])
        limitations.append(
            "Text-only SFT: the Qwen3.5 vision tower is resident but inactive; image/video training and vision adapters are excluded."
        )
    if linear_layers:
        limitations.append(
            "Qwen3.5 linear-attention projections and recurrent state are modelled separately; chunk-kernel backward workspaces and unfused fallback costs have partial analytical coverage."
        )
    if moe_layers:
        limitations.append(
            "MoE stores all experts but computes selected experts; routing imbalance, expert dispatch workspaces and auxiliary losses have partial analytical coverage."
        )
    if text.get("mtp_num_hidden_layers"):
        limitations.append(
            "Multi-token prediction heads/training are excluded; this models standard next-token SFT."
        )
    return {
        "model_type": kind,
        "dimensions": dimensions,
        "full_attention_layers": full_layers,
        "linear_attention_layers": linear_layers,
        "text_parameters": text_parameters,
        "vision_parameters": vision_parameters,
        "total_parameters": text_parameters + vision_parameters,
        "embedding_parameters": embedding,
        "tied_embeddings": bool(tied),
        "linear_modules": modules,
        "recurrent_dimensions": recurrent,
        "coverage": "partial" if linear_layers or moe_layers else "analytical",
        "parameter_source": "architecture_config_formulas",
        "limitations": limitations,
    }


def _qwen_vision_parameters(config: Mapping[str, Any]) -> int:
    required = (
        "hidden_size",
        "intermediate_size",
        "depth",
        "in_channels",
        "patch_size",
        "temporal_patch_size",
        "num_position_embeddings",
        "spatial_merge_size",
        "out_hidden_size",
    )
    values = {
        key: _positive_int(config.get(key), "vision_config." + key) for key in required
    }
    h, intermediate = values["hidden_size"], values["intermediate_size"]
    patch = (
        h
        * values["in_channels"]
        * values["patch_size"] ** 2
        * values["temporal_patch_size"]
        + h
    )
    blocks = values["depth"] * (4 * h * h + 2 * h * intermediate + 9 * h + intermediate)
    merged = h * values["spatial_merge_size"] ** 2
    merger = (
        2 * h
        + merged * merged
        + merged
        + merged * values["out_hidden_size"]
        + values["out_hidden_size"]
    )
    if config.get("deepstack_visual_indexes"):
        raise FinetuneEstimateError("deepstack visual merger residency is not covered")
    return patch + values["num_position_embeddings"] * h + blocks + merger


def _quantized_storage(
    modules: list[tuple[str, int, int, int]], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Budget bitsandbytes' default block layout, including per-tensor padding.

    NF4 and FP4 both store two four-bit values per byte. Double quantization
    changes scale storage, not the weight packing. Lookup tables are budgeted
    per tensor; a backend that shares them may use slightly less memory.
    """
    kind = config.get("quantization_type") or "fp4"
    storage = config.get("quantization_storage") or "uint8"
    compute_dtype = config.get("quantization_compute_dtype") or "fp32"
    nested = config.get("double_quantization")
    nested = False if nested is None else nested
    if kind not in {"nf4", "fp4"}:
        raise FinetuneEstimateError("quantization_type must be nf4 or fp4")
    storage_bytes = {"uint8": 1, **_DTYPE_BYTES}
    if storage not in storage_bytes:
        raise FinetuneEstimateError(
            "quantization_storage must be uint8, fp16, bf16, or fp32"
        )
    if compute_dtype not in _DTYPE_BYTES:
        raise FinetuneEstimateError(
            "quantization_compute_dtype must be fp32, fp16, or bf16"
        )
    if not isinstance(nested, bool):
        raise FinetuneEstimateError("double_quantization must be a boolean")
    packed = scales = nested_scales = offsets = tables = 0
    for _, width_in, width_out, count in modules:
        parameters = width_in * width_out
        blocks = math.ceil(parameters / 64)
        unit = storage_bytes[storage]
        packed += math.ceil(parameters / (unit * 2)) * unit * count
        scales += blocks * (1 if nested else 4) * count
        tables += 16 * 4 * count
        if nested:
            nested_scales += math.ceil(blocks / 256) * 4 * count
            offsets += 4 * count
            tables += 256 * 4 * count
    return {
        "type": kind,
        "storage_dtype": storage,
        "compute_dtype": compute_dtype,
        "double_quantization": nested,
        "block_size": 64,
        "nested_block_size": 256 if nested else None,
        "packed_weight_bytes": packed,
        "scale_bytes": scales,
        "nested_scale_bytes": nested_scales,
        "offset_bytes": offsets,
        "codebook_bytes": tables,
        "metadata_bytes": scales + nested_scales + offsets + tables,
        "basis": "bitsandbytes_default_blocks_with_per_tensor_codebook_budget",
    }


def _observed_quantized_storage(
    config: Mapping[str, Any], total_parameters: int
) -> dict[str, int] | None:
    storage = config.get("actual_quantized_storage")
    if storage is None:
        return None
    fields = {
        "quantized_parameters",
        "packed_weight_bytes",
        "metadata_bytes",
        "unquantized_weight_bytes",
    }
    if not isinstance(storage, Mapping) or set(storage) != fields:
        raise FinetuneEstimateError("actual_quantized_storage has invalid fields")
    if not config.get("actual_model_signature"):
        raise FinetuneEstimateError(
            "actual_quantized_storage requires actual_model_signature"
        )
    for key, value in storage.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise FinetuneEstimateError(
                f"actual_quantized_storage.{key} must be a nonnegative integer"
            )
    count = storage["quantized_parameters"]
    dense = total_parameters - count
    if count <= 0 or dense < 0 or storage["packed_weight_bytes"] < math.ceil(count / 2):
        raise FinetuneEstimateError(
            "actual_quantized_storage disagrees with logical parameter counts"
        )
    if not dense * 2 <= storage["unquantized_weight_bytes"] <= dense * 4:
        raise FinetuneEstimateError(
            "actual_quantized_storage has inconsistent unquantized weight bytes"
        )
    return dict(storage)


def _memory(
    model: Mapping[str, Any],
    config: Mapping[str, Any],
    workload: Mapping[str, Any],
    overhead: int,
) -> dict[str, Any]:
    method = config["method"]
    element = _DTYPE_BYTES[config["dtype"]]
    parameter_dtype = config.get("parameter_dtype") or "fp32"
    if parameter_dtype not in _DTYPE_BYTES:
        raise FinetuneEstimateError("parameter_dtype must be fp32, fp16, or bf16")
    parameter_element = _DTYPE_BYTES[parameter_dtype]
    adapter_count = 0
    adapter_input_elements = 0
    adapter_rank_elements = 0
    if method != "full":
        rank = _positive_int(config.get("lora_rank"), "lora_rank")
        target_names = config.get("lora_target_modules")
        if (
            not isinstance(target_names, list)
            or not target_names
            or any(not isinstance(name, str) for name in target_names)
        ):
            raise FinetuneEstimateError(
                "lora_target_modules must be an explicit non-empty list"
            )
        targets = set(target_names)
        available = {name for name, _, _, _ in model["linear_modules"]}
        if targets & {"all", "all-linear"}:
            if len(targets) != 1:
                raise FinetuneEstimateError(
                    "all-linear cannot be combined with named LoRA targets"
                )
            targets = available
            if model["vision_parameters"]:
                raise FinetuneEstimateError(
                    "all-linear on a composite vision model includes vision adapters; use a text-only config"
                )
        if model["dimensions"]["num_routed_experts"] and targets & {
            "gate_proj",
            "up_proj",
            "down_proj",
        }:
            raise FinetuneEstimateError(
                "MoE expert LoRA needs actual target-parameter inspection; select explicit attention targets"
            )
        if targets - available:
            raise FinetuneEstimateError(
                "unmodelled LoRA target modules: "
                + ", ".join(sorted(targets - available))
            )
        adapter_count = sum(
            rank * (i + o) * count
            for name, i, o, count in model["linear_modules"]
            if name in targets
        )
        adapter_input_elements = sum(
            i * (1 if config.get("gradient_checkpointing") else count)
            for name, i, _, count in model["linear_modules"]
            if name in targets
        )
        adapter_rank_elements = sum(
            rank * (1 if config.get("gradient_checkpointing") else count)
            for name, _, _, count in model["linear_modules"]
            if name in targets
        )
    trainable = model["text_parameters"] if method == "full" else adapter_count
    if config.get("trainable_parameter_count") is not None:
        trainable = _positive_int(
            config["trainable_parameter_count"], "trainable_parameter_count"
        )
        if method == "full" and trainable > model["text_parameters"]:
            raise FinetuneEstimateError(
                "trainable_parameter_count cannot exceed text parameters for text-only full tuning"
            )
        if method != "full":
            adapter_count = trainable
    trainable_element = parameter_element if method == "full" else 4
    quantized = 0
    quantization_metadata = 0
    quantization = None
    if method == "qlora":
        if model["dimensions"]["num_routed_experts"]:
            raise FinetuneEstimateError(
                "MoE QLoRA requires inspecting which fused expert tensors are actually quantized"
            )
        observed_storage = _observed_quantized_storage(
            config, model["total_parameters"]
        )
        if observed_storage is not None:
            quantized = observed_storage["quantized_parameters"]
            quantization = {
                **observed_storage,
                "type": config["quantization_type"],
                "storage_dtype": config["quantization_storage"],
                "compute_dtype": config["quantization_compute_dtype"],
                "double_quantization": config["double_quantization"],
                "basis": "inspected_native_bitsandbytes_tensor_storage",
            }
            unquantized_bytes = observed_storage["unquantized_weight_bytes"]
        else:
            quantized = sum(i * o * count for _, i, o, count in model["linear_modules"])
            if quantized > model["total_parameters"]:
                raise FinetuneEstimateError(
                    "parameter_count must describe logical parameters before 4-bit packing"
                )
            quantization = _quantized_storage(model["linear_modules"], config)
            unquantized_bytes = (model["total_parameters"] - quantized) * 4
        quantization_metadata = quantization["metadata_bytes"]
        base_bytes = (
            quantization["packed_weight_bytes"]
            + quantization_metadata
            + unquantized_bytes
        )
    else:
        base_bytes = model["total_parameters"] * parameter_element
    adapter_bytes = adapter_count * 4
    gradient = trainable * trainable_element
    optimizer = str(config.get("optimizer", "adamw_torch")).lower()
    torch_optimizer = optimizer in {
        "adamw_torch",
        "adamw_torch_fused",
        "sgd",
        "sgd_momentum",
    }
    optimizer_element = trainable_element if torch_optimizer else 4
    optimizer_bytes = (
        trainable
        * optimizer_element
        * (
            2
            if optimizer.startswith("adam")
            else 1
            if optimizer == "sgd_momentum"
            else 0
        )
    )
    master = (
        trainable * 4
        if not torch_optimizer and method == "full" and parameter_element < 4
        else 0
    )
    persistent = base_bytes + adapter_bytes + gradient + optimizer_bytes + master
    dimensions = model["dimensions"]
    h = dimensions["hidden_size"]
    tokens = workload["micro_batch_size"] * workload["sequence_length"]
    sequence = workload["sequence_length"]
    layers = dimensions["num_hidden_layers"]
    mlp_width = max(
        dimensions["intermediate_size"],
        dimensions["moe_intermediate_size"] * dimensions["num_experts_per_token"]
        + dimensions["shared_expert_intermediate_size"],
    )
    layer_activations = tokens * element * (8 * h + 3 * mlp_width)
    attention = config.get("attention_implementation") or "auto"
    if attention == "auto":
        # Unknown SDPA dispatch can select the quadratic math implementation;
        # budget eager score storage rather than silently assuming Flash.
        attention = "eager"
    if attention not in {"sdpa", "eager", "flash_attention_2", "flash_attention_3"}:
        raise FinetuneEstimateError(
            f"unsupported attention implementation {attention!r}"
        )
    attention_scores = 0
    if attention == "eager":
        attention_scores = (
            workload["micro_batch_size"]
            * dimensions["num_attention_heads"]
            * sequence**2
            * 4
            * model["full_attention_layers"]
        )
    recurrent_bytes = 0
    recurrent = model["recurrent_dimensions"]
    if recurrent:
        recurrent_bytes = (
            4
            * workload["micro_batch_size"]
            * recurrent["linear_num_value_heads"]
            * recurrent["linear_key_head_dim"]
            * recurrent["linear_value_head_dim"]
            * math.ceil(sequence / 64)
            * model["linear_attention_layers"]
        )
    retained = layer_activations * layers + attention_scores + recurrent_bytes
    if config.get("gradient_checkpointing"):
        retained = (
            tokens * h * element * layers
            + layer_activations
            + math.ceil((attention_scores + recurrent_bytes) / layers)
        )
    logits = tokens * dimensions["vocab_size"] * 4
    adapter_activation = tokens * (adapter_input_elements + adapter_rank_elements) * 4
    dropout_activation = (
        tokens * adapter_input_elements * 5 if config.get("lora_dropout") else 0
    )
    activation = retained + logits + adapter_activation + dropout_activation
    largest_matrix = max(i * o for _, i, o, _ in model["linear_modules"])
    quantization_element = _DTYPE_BYTES[
        config.get("quantization_compute_dtype") or "fp32"
    ]
    workspace = max(
        tokens * h * element * 4,
        largest_matrix * max(element, quantization_element) if method == "qlora" else 0,
    )
    return {
        "total_base_parameters": model["total_parameters"],
        "trainable_parameters": trainable,
        "frozen_parameters": model["total_parameters"]
        - (trainable if method == "full" else 0),
        "adapter_parameters": adapter_count,
        "parameter_dtype": parameter_dtype,
        "base_weight_bytes": base_bytes,
        "adapter_weight_bytes": adapter_bytes,
        "gradient_bytes": gradient,
        "optimizer_state_bytes": optimizer_bytes,
        "optimizer_state_dtype": (
            parameter_dtype if torch_optimizer and method == "full" else "fp32"
        ),
        "optimizer_state_basis": (
            "pytorch_parameter_dtype_no_master_copy"
            if torch_optimizer
            else "conservative_fp32_state_and_master_budget"
        ),
        "master_weight_bytes": master,
        "persistent_bytes": persistent,
        "activation_bytes": activation,
        "adapter_activation_bytes": adapter_activation,
        "dropout_activation_bytes": dropout_activation,
        "logits_bytes": logits,
        "attention_score_bytes": attention_scores,
        "recurrent_state_bytes": recurrent_bytes,
        "workspace_bytes": workspace,
        "runtime_overhead_bytes": overhead,
        "quantized_parameters": quantized,
        "quantization_metadata_bytes": quantization_metadata,
        "quantization": quantization,
        "peak_bytes": {
            "lower": persistent + math.ceil(activation * 0.5) + overhead,
            "expected": persistent + activation + workspace + overhead,
            "upper": persistent + activation * 2 + workspace * 2 + overhead,
        },
    }


def _training_flops(
    model: Mapping[str, Any],
    config: Mapping[str, Any],
    workload: Mapping[str, Any],
    trainable: int,
) -> int:
    dimensions = model["dimensions"]
    tokens = workload["micro_batch_size"] * workload["sequence_length"]
    matrix_parameters = sum(i * o * n for _, i, o, n in model["linear_modules"])
    if dimensions["num_routed_experts"]:
        inactive = (
            dimensions["num_routed_experts"] - dimensions["num_experts_per_token"]
        )
        matrix_parameters -= (
            dimensions["moe_layer_count"]
            * inactive
            * 3
            * dimensions["hidden_size"]
            * dimensions["moe_intermediate_size"]
        )
    forward = 2 * tokens * (matrix_parameters + model["embedding_parameters"])
    forward += (
        4
        * tokens
        * workload["sequence_length"]
        * dimensions["num_attention_heads"]
        * dimensions["head_dim"]
        * model["full_attention_layers"]
    )
    recurrent = model["recurrent_dimensions"]
    if recurrent:
        forward += (
            8
            * tokens
            * recurrent["linear_num_value_heads"]
            * recurrent["linear_key_head_dim"]
            * recurrent["linear_value_head_dim"]
            * model["linear_attention_layers"]
        )
    # Frozen weights still participate in forward and input-gradient matmuls.
    multiplier = 3 if config["method"] == "full" else 2
    if config.get("gradient_checkpointing"):
        multiplier += 1
    adapter_flops = 0 if config["method"] == "full" else 6 * tokens * trainable
    return int(
        forward * multiplier
        + adapter_flops
        + 12 * trainable / workload["gradient_accumulation_steps"]
    )


def _validate_observation(
    observation: Mapping[str, Any],
    signature: Mapping[str, Any],
    *,
    require_fixed_shape: bool = True,
) -> dict[str, Any]:
    if observation.get("schema_version") != OBSERVATION_SCHEMA_VERSION:
        raise FinetuneEstimateError("unsupported fine-tuning observation schema")
    if observation.get("signature") != signature:
        raise FinetuneEstimateError(
            "calibration signature mismatch: profile, model, execution shape and software must match"
        )
    samples = observation.get("seconds_per_microstep")
    if (
        not isinstance(samples, list)
        or len(samples) < 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in samples
        )
    ):
        raise FinetuneEstimateError(
            "seconds_per_microstep must contain at least two finite positive observations"
        )
    allocated = _positive_int(
        observation.get("peak_allocated_bytes"), "peak_allocated_bytes"
    )
    if "peak_reserved_bytes" in observation:
        reserved = _positive_int(
            observation["peak_reserved_bytes"], "peak_reserved_bytes"
        )
        if reserved < allocated:
            raise FinetuneEstimateError(
                "peak_reserved_bytes cannot be less than peak_allocated_bytes"
            )
    result = dict(observation)
    result["verified_probe_metadata"] = False
    source = observation.get("source")
    if source in {
        "fakegpu.huggingface_trainer_cuda_probe",
        "fakegpu.native_framework_cuda_run",
    }:
        identity = observation.get("checkpoint_identity", {})
        digest = signature.get("checkpoint_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(identity, Mapping)
            or identity.get("algorithm") != "sha256_content_manifest_v1"
            or identity.get("sha256") != digest
            or identity.get("model_dir") != signature.get("model_name_or_path")
        ):
            raise FinetuneEstimateError(
                "real probe requires a matching local checkpoint content identity"
            )
        profile = get_profile(signature["profile"]["id"])
        hardware = observation.get("hardware", {})
        if (
            not isinstance(hardware, Mapping)
            or hardware.get("name")
            not in {profile.name, profile.torch_name, profile.official_model}
            or hardware.get("compute_capability") != profile.compute_capability_text
        ):
            raise FinetuneEstimateError(
                "probe hardware does not match the target profile"
            )
        _positive_int(hardware.get("total_memory_bytes"), "hardware.total_memory_bytes")
        software = observation.get("software_versions")
        if not software or software != signature.get("software_versions"):
            raise FinetuneEstimateError(
                "probe software_versions must match the calibration signature"
            )
        workload = observation.get("workload", {})
        if (
            not isinstance(workload, Mapping)
            or workload.get("sequence_length") != signature["sequence_length"]
            or workload.get("measured_optimizer_steps") != len(samples)
        ):
            raise FinetuneEstimateError(
                "probe workload does not match sequence length and measurement count"
            )
        if (
            source == "fakegpu.huggingface_trainer_cuda_probe"
            and signature.get("source_framework") != "huggingface"
        ):
            raise FinetuneEstimateError(
                "the Hugging Face probe cannot calibrate another framework"
            )
        if source == "fakegpu.native_framework_cuda_run":
            from ._finetune_batches import validate_batch_profile
            from ._finetune_events import validate_event_profile

            try:
                validate_batch_profile(observation)
                validate_event_profile(observation)
            except ValueError as exc:
                raise FinetuneEstimateError(str(exc)) from exc
            framework = observation.get("native_framework")
            if framework not in {
                "llamafactory",
                "axolotl",
            } or framework != signature.get("source_framework"):
                raise FinetuneEstimateError(
                    "native framework must match the calibration signature"
                )
            for key in ("data_signature", "native_execution_signature"):
                digest = signature.get(key)
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                ):
                    raise FinetuneEstimateError(f"native observation requires {key}")
            if workload.get("sha256") != signature["data_signature"]:
                raise FinetuneEstimateError(
                    "native processed data must match the calibration signature"
                )
            if require_fixed_shape and workload.get("fixed_shape") is not True:
                raise FinetuneEstimateError(
                    "native calibration requires fixed full batches at sequence_length"
                )
        result["verified_probe_metadata"] = True
    return result
