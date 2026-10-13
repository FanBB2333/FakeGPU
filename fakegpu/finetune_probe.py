"""Real-GPU evidence for fixed-shape text fine-tuning.

Optional ML dependencies are imported only when a probe is requested. The
probe runs the actual Hugging Face Trainer; other framework configurations
remain estimates until measured by their own runners.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
from pathlib import Path
import tempfile
import time
from collections.abc import Mapping
from typing import Any

from ._calibration_cuda import _reject_simulated_cuda_environment
from ._local_checkpoint import bind_local_checkpoint
from .finetune_estimator import finetune_workload_signature
from .profile_catalog import get_profile


def installed_training_versions() -> dict[str, str]:
    versions = {}
    for name in (
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "bitsandbytes",
        "flash-linear-attention",
        "causal-conv1d",
        "flash-attn",
        "triton",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def measure_finetuning(
    model_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    *,
    model_dir: str | Path,
    profile_id: str,
    warmup_steps: int = 2,
    measure_steps: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Measure complete optimizer updates, including accumulation and data loading.

    The generated arithmetic text is repeated to exactly the requested sequence
    length. Evidence describes this fixed-shape workload, not convergence or a
    user's arbitrary data pipeline. Initialization is recorded separately.
    """
    for name, value, minimum in (
        ("warmup_steps", warmup_steps, 1),
        ("measure_steps", measure_steps, 2),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    config = dict(training_config)
    if config["source_framework"] != "huggingface":
        raise ValueError(
            "the built-in probe executes huggingface Trainer; use a framework-specific runner for other frameworks"
        )
    distributed = config.get("distributed_config", {})
    if (
        config["world_size"] != 1
        or distributed.get("sharding_strategy", "replicated") != "replicated"
        or distributed.get("zero_stage", 0) != 0
        or any(
            distributed.get(k, "none") != "none"
            for k in ("parameter_offload_device", "optimizer_offload_device")
        )
    ):
        raise ValueError(
            "the built-in probe currently measures single-GPU, unsharded training"
        )
    if model_config.get("vision_config"):
        raise ValueError(
            "the built-in text probe requires explicit --text-only for multimodal checkpoints"
        )
    if config.get("packing"):
        raise ValueError(
            "the fixed-shape probe requires packing=false; packed datasets need their actual runner"
        )
    if config.get("optimizer") not in {"adamw_torch", "adamw_torch_fused"}:
        raise ValueError(
            "the built-in probe requires optim=adamw_torch or adamw_torch_fused"
        )
    if config.get("attention_implementation") not in {"sdpa", "eager"}:
        raise ValueError(
            "the built-in probe requires explicit attn_implementation=sdpa or eager"
        )
    length = config.get("sequence_length")
    if isinstance(length, bool) or not isinstance(length, int) or length < 2:
        raise ValueError("the probe requires sequence_length >= 2")
    root = Path(model_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("model_dir must be a local pretrained model directory")

    import torch
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    _reject_simulated_cuda_environment(torch)
    if not torch.cuda.is_available() or getattr(
        torch.cuda, "_fakegpu_simulated", False
    ):
        raise RuntimeError(
            "a physical CUDA device is required for fine-tuning observations"
        )
    if torch.cuda.device_count() != 1:
        raise ValueError(
            "select exactly one physical GPU with CUDA_VISIBLE_DEVICES before starting the probe"
        )
    device = torch.cuda.get_device_properties(0)
    profile = get_profile(profile_id)
    if device.name not in {profile.name, profile.torch_name, profile.official_model}:
        raise ValueError(
            f"selected GPU {device.name!r} does not match profile {profile_id!r} ({profile.name})"
        )
    if (device.major, device.minor) != profile.compute_capability:
        raise ValueError(
            "selected GPU compute capability does not match the target profile"
        )
    if config["dtype"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("bf16 was requested but is unsupported by this CUDA device")
    if config.get("gradient_checkpointing") and config["method"] != "full":
        checkpointing_kwargs = {"use_reentrant": False}
    else:
        checkpointing_kwargs = None

    checkpoint_hash_started = time.perf_counter()
    config, checkpoint_identity = bind_local_checkpoint(config, root)
    checkpoint_hash_seconds = time.perf_counter() - checkpoint_hash_started
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    config["software_versions"] = installed_training_versions()
    accumulation = config["gradient_accumulation_steps"]
    micro_batch = config["micro_batch_size"]
    samples = (warmup_steps + measure_steps) * accumulation * micro_batch
    initialization_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(
        root, local_files_only=True, trust_remote_code=False
    )
    records = []
    for index in range(samples):
        x, y = index % 13, index % 7
        tokens = tokenizer.encode(
            f"Question: What is {x} + {y}? Answer: {x + y}.\n", add_special_tokens=False
        )
        if not tokens:
            raise ValueError("tokenizer produced no tokens for the synthetic data")
        ids = (tokens * math.ceil(length / len(tokens)))[:length]
        records.append(
            {"input_ids": ids, "attention_mask": [1] * length, "labels": ids.copy()}
        )
    data_digest = hashlib.sha256(repr(records).encode()).hexdigest()
    torch_dtypes = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtype = torch_dtypes[config.get("parameter_dtype") or "fp32"]
    architecture = AutoConfig.for_model(
        model_config["model_type"],
        **{key: value for key, value in model_config.items() if key != "model_type"},
    )
    model_options: dict[str, Any] = {
        "config": architecture,
        "local_files_only": True,
        "trust_remote_code": False,
        "dtype": dtype,
        "attn_implementation": config["attention_implementation"],
    }
    if config["method"] == "qlora":
        from transformers import BitsAndBytesConfig

        if config["software_versions"]["bitsandbytes"] == "not-installed":
            raise ImportError(
                "QLoRA probing requires bitsandbytes in the probe environment"
            )
        model_options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config["quantization_type"],
            bnb_4bit_use_double_quant=config["double_quantization"],
            bnb_4bit_compute_dtype=torch_dtypes[config["quantization_compute_dtype"]],
            bnb_4bit_quant_storage=(
                torch.uint8
                if config["quantization_storage"] == "uint8"
                else torch_dtypes[config["quantization_storage"]]
            ),
        )
        model_options["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(root, **model_options)
    model.config.use_cache = False
    base_parameter_count = (
        model.num_parameters()
        if config["method"] == "qlora"
        else sum(p.numel() for p in model.parameters())
    )
    if config["method"] in {"lora", "qlora"}:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if config["method"] == "qlora":
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=config["gradient_checkpointing"]
            )
        model = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=config["lora_rank"],
                lora_alpha=config["lora_alpha"],
                target_modules=config["lora_target_modules"],
                lora_dropout=config["lora_dropout"],
                bias=config["lora_bias"],
            ),
        )
    if config["method"] != "qlora":
        model.to("cuda")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    torch.cuda.synchronize()
    initialization_seconds = time.perf_counter() - initialization_start
    update_seconds: list[float] = []
    data_loading_seconds = [0.0]
    startup_peaks = {"allocated": 0, "reserved": 0}

    class MeasuredTrainer(Trainer):
        def get_batch_samples(self, *args, **kwargs):
            started = time.perf_counter()
            result = super().get_batch_samples(*args, **kwargs)
            torch.cuda.synchronize()
            data_loading_seconds[0] += time.perf_counter() - started
            return result

    class TimingCallback(TrainerCallback):
        started = 0.0

        def on_step_begin(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            if state.global_step == warmup_steps:
                startup_peaks["allocated"] = torch.cuda.max_memory_allocated()
                startup_peaks["reserved"] = torch.cuda.max_memory_reserved()
                torch.cuda.reset_peak_memory_stats()
            self.started = time.perf_counter()

        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            if state.global_step > warmup_steps:
                update_seconds.append(
                    time.perf_counter() - self.started + data_loading_seconds[0]
                )
            data_loading_seconds[0] = 0.0

    with tempfile.TemporaryDirectory(prefix="fakegpu-finetune-") as output_dir:
        arguments = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=micro_batch,
            gradient_accumulation_steps=accumulation,
            max_steps=warmup_steps + measure_steps,
            bf16=config["dtype"] == "bf16",
            fp16=config["dtype"] == "fp16",
            tf32=config.get("tf32"),
            gradient_checkpointing=config["gradient_checkpointing"],
            gradient_checkpointing_kwargs=checkpointing_kwargs,
            optim=config["optimizer"],
            learning_rate=config["learning_rate"],
            lr_scheduler_type=config["lr_scheduler_type"],
            warmup_steps=config["warmup_steps"],
            warmup_ratio=config["warmup_ratio"],
            weight_decay=config["weight_decay"],
            max_grad_norm=config["max_grad_norm"],
            save_strategy="no",
            logging_strategy="no",
            report_to=[],
            disable_tqdm=True,
            logging_nan_inf_filter=False,
            dataloader_pin_memory=True,
            dataloader_drop_last=False,
            seed=seed,
            data_seed=seed,
        )
        trainer = MeasuredTrainer(
            model=model,
            args=arguments,
            train_dataset=records,
            callbacks=[TimingCallback()],
        )
        outcome = trainer.train()
    if len(update_seconds) != measure_steps or not math.isfinite(outcome.training_loss):
        raise RuntimeError(
            "probe did not produce the requested finite training measurements"
        )
    return {
        "schema_version": "fakegpu.finetune_observation.v1",
        "source": "fakegpu.huggingface_trainer_cuda_probe",
        "signature": finetune_workload_signature(
            model_config, config, profile_id=profile_id
        ),
        "seconds_per_microstep": [seconds / accumulation for seconds in update_seconds],
        "optimizer_update_seconds": update_seconds,
        "peak_allocated_bytes": max(
            startup_peaks["allocated"], torch.cuda.max_memory_allocated()
        ),
        "peak_reserved_bytes": max(
            startup_peaks["reserved"], torch.cuda.max_memory_reserved()
        ),
        "steady_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "steady_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "startup_peaks": startup_peaks,
        "initialization_seconds": initialization_seconds,
        "checkpoint_identity": checkpoint_identity,
        "checkpoint_hash_seconds": checkpoint_hash_seconds,
        "training_loss": outcome.training_loss,
        "observed_model": {
            "parameter_count": base_parameter_count,
            "trainable_parameter_count": trainable,
        },
        "hardware": {
            "name": device.name,
            "total_memory_bytes": device.total_memory,
            "compute_capability": f"{device.major}.{device.minor}",
            "cuda_version": torch.version.cuda,
        },
        "software_versions": config["software_versions"],
        "workload": {
            "kind": "synthetic_fixed_length_arithmetic_text",
            "sha256": data_digest,
            "seed": seed,
            "sequence_length": length,
            "samples": samples,
            "warmup_optimizer_steps": warmup_steps,
            "measured_optimizer_steps": measure_steps,
        },
        "limitations": [
            "Synthetic text measures fixed-shape training, not convergence or arbitrary input pipelines.",
            "Timing excludes model initialization, checkpoint saving, and evaluation.",
            "Checkpoint identity streams the selected weight files before model loading; its measured read cost is excluded from training-loop time.",
            "Timing samples are complete optimizer updates divided by gradient accumulation.",
            "Peak allocated/reserved memory is PyTorch allocator memory, not total device usage.",
            "This probe runs Hugging Face Trainer and cannot validate another framework's execution.",
        ],
    }
