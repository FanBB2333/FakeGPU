"""Execute supported native framework training and observe the actual Trainer.

Framework packages remain optional. This module imports no Torch or training
framework until run_finetuning has validated the requested local configuration.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any

from ._calibration_cuda import _reject_simulated_cuda_environment
from ._finetune_observer import NativeTrainerObserver, capture_native_trainer
from ._local_checkpoint import bind_local_checkpoint
from .finetune_config import normalize_finetune_config
from .finetune_data import profile_tokenized_records
from .finetune_estimator import finetune_workload_signature
from .finetune_probe import installed_training_versions
from .profile_catalog import get_profile
from .structured_io import load_mapping


_RUN_LOCK = threading.Lock()
_DTYPES = {"torch.float32": "fp32", "torch.float16": "fp16", "torch.bfloat16": "bf16"}


def native_training_versions(framework: str) -> dict[str, str]:
    """Read package metadata without importing an optional training framework."""
    if framework not in {"llamafactory", "axolotl"}:
        raise ValueError("native framework must be llamafactory or axolotl")
    versions = installed_training_versions()
    for package in (framework, "datasets", "trl"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _prepare_config(path: Path, framework: str, warmup_steps: int):
    if framework not in {"llamafactory", "axolotl"}:
        raise ValueError("native framework must be llamafactory or axolotl")
    if (
        isinstance(warmup_steps, bool)
        or not isinstance(warmup_steps, int)
        or warmup_steps < 1
    ):
        raise ValueError("warmup_steps must be an integer >= 1")
    raw = load_mapping(path)
    normalization_input = dict(raw)
    if raw.get("disable_gradient_checkpointing") is True:
        # This native option has no effect when checkpointing is disabled.
        normalization_input.pop("use_reentrant_gc", None)
    config = normalize_finetune_config(
        normalization_input, framework=framework, base_dir=path.parent
    )
    if config["method"] not in {"full", "lora"}:
        raise ValueError(
            "native runner currently supports full tuning and unquantized LoRA"
        )
    if config["packing"] or raw.get("neat_packing"):
        raise ValueError("native runner requires unpacked text data")
    distributed = config["distributed_config"]
    if (
        config["world_size"] != 1
        or distributed.get("sharding_strategy", "replicated") != "replicated"
        or distributed.get("zero_stage", 0) != 0
        or any(
            distributed.get(key, "none") != "none"
            for key in ("parameter_offload_device", "optimizer_offload_device")
        )
        or int(os.environ.get("WORLD_SIZE", "1")) != 1
    ):
        raise ValueError("native runner requires one GPU without sharding or offload")
    if config["attention_implementation"] not in {"sdpa", "eager"}:
        raise ValueError("native runner requires explicit sdpa or eager attention")
    if config["optimizer"] not in {"adamw_torch", "adamw_torch_fused"}:
        raise ValueError("native runner supports adamw_torch or adamw_torch_fused")
    if not config["sequence_length"] or config["sequence_length"] < 2:
        raise ValueError("native runner requires sequence_length >= 2")
    if config["max_steps"] is not None and config["max_steps"] < warmup_steps + 2:
        raise ValueError("native run needs at least two measured updates after warmup")
    for key in (
        "trust_remote_code",
        "streaming",
        "auto_resume_from_checkpoints",
        "use_v1_kernels",
        "lora_mlp_kernel",
        "lora_qkv_kernel",
        "lora_o_kernel",
        "lora_embedding_kernel",
        "fused_attn_kernel",
        "train_from_scratch",
        "reinit_weights",
        "overrides_of_model_kwargs",
        "model_kwargs",
        "overrides_of_model_config",
        "model_config",
        "use_kernels",
        "use_ray",
        "use_mca",
        "use_hyper_parallel",
        "use_kt",
    ):
        if raw.get(key):
            raise ValueError(f"native runner does not support {key}")
    model_dir = Path(config["model_name_or_path"]).expanduser().resolve()
    if not model_dir.is_dir():
        raise ValueError("native runner requires an existing local text checkpoint")
    model_config = json.loads((model_dir / "config.json").read_text())
    if model_config.get("vision_config") or model_config.get("audio_config"):
        raise ValueError("native runner requires an exported pure text checkpoint")
    if model_config.get("quantization_config"):
        raise ValueError("native runner requires unquantized checkpoint weights")
    output = raw.get("output_dir")
    if not isinstance(output, str) or not output:
        raise ValueError("native runner requires an explicit output_dir")
    output_dir = Path(output).expanduser()
    if not output_dir.is_absolute():
        output_dir = path.parent / output_dir
    if output_dir.resolve() == model_dir:
        raise ValueError("output_dir must differ from the source checkpoint")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(
            "native runner requires a fresh output_dir to avoid automatic resume"
        )
    return raw, config, model_config, model_dir


def _execution_signature(raw: dict[str, Any]) -> str:
    # Changing run duration or artifact locations does not change a microstep.
    excluded = {
        "output_dir",
        "logging_dir",
        "max_steps",
        "num_epochs",
        "num_train_epochs",
        "epochs",
        "save_steps",
        "eval_steps",
        "logging_steps",
        "save_strategy",
        "eval_strategy",
        "evaluation_strategy",
        "save_total_limit",
        "dataset_prepared_path",
        "cache_dir",
        "tokenized_path",
        "overwrite_cache",
        "overwrite_output_dir",
        "run_name",
    }
    payload = {key: value for key, value in raw.items() if key not in excluded}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@contextmanager
def _native_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _invoke_native(framework, config_path, raw):
    if framework == "llamafactory":
        from llamafactory.train.tuner import run_exp

        run_exp(args=raw)
    else:
        from axolotl.cli.args import TrainerCliArgs
        from axolotl.cli.config import load_cfg
        from axolotl.cli.train import do_train

        do_train(load_cfg(str(config_path)), TrainerCliArgs())


def _actual_training_config(trainer, normalized):
    import torch

    args, model = trainer.args, trainer.model
    if (
        args.world_size != 1
        or getattr(trainer, "is_deepspeed_enabled", False)
        or getattr(trainer, "is_fsdp_enabled", False)
    ):
        raise ValueError("actual Trainer enabled unsupported distributed training")
    parameters = list(model.named_parameters())
    if any(parameter.device.type != "cuda" for _, parameter in parameters):
        raise ValueError(
            "actual model parameters must all reside on the selected CUDA GPU"
        )
    if getattr(model, "is_quantized", False):
        raise ValueError("actual model is quantized")
    dtype_counts = Counter()
    trainable_dtypes = Counter()
    frozen_dtypes = Counter()
    for _, parameter in parameters:
        dtype_counts[str(parameter.dtype)] += parameter.numel()
        (trainable_dtypes if parameter.requires_grad else frozen_dtypes)[
            str(parameter.dtype)
        ] += parameter.numel()
    base_dtypes = frozen_dtypes or dtype_counts
    storage_dtype = max(base_dtypes, key=base_dtypes.get)
    if storage_dtype not in _DTYPES:
        raise ValueError("actual base parameters have an unsupported storage dtype")
    result = dict(normalized)
    result.update(
        {
            "parameter_dtype": _DTYPES[storage_dtype],
            "dtype": "bf16" if args.bf16 else "fp16" if args.fp16 else "fp32",
            "micro_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "epochs": args.num_train_epochs,
            "max_steps": args.max_steps if args.max_steps > 0 else None,
            "world_size": args.world_size,
            "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "gradient_checkpointing": bool(
                getattr(model, "is_gradient_checkpointing", False)
            ),
            "optimizer": getattr(args.optim, "value", args.optim),
            "learning_rate": args.learning_rate,
            "lr_scheduler_type": getattr(
                args.lr_scheduler_type, "value", args.lr_scheduler_type
            ),
            "warmup_steps": args.warmup_steps,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "dataloader_drop_last": args.dataloader_drop_last,
            "trainable_parameter_count": sum(trainable_dtypes.values()),
        }
    )
    total = sum(dtype_counts.values())
    result["parameter_count"] = (
        total - result["trainable_parameter_count"]
        if result["method"] == "lora"
        else total
    )
    result["distributed_config"] = dict(
        result["distributed_config"],
        activation_checkpointing=result["gradient_checkpointing"],
    )
    if result["method"] == "lora":
        adapters = getattr(model, "peft_config", {})
        if len(adapters) != 1:
            raise ValueError("actual LoRA model must contain exactly one adapter")
        adapter = next(iter(adapters.values()))
        result.update(
            {
                "lora_rank": adapter.r,
                "lora_alpha": adapter.lora_alpha,
                "lora_dropout": adapter.lora_dropout,
                "lora_bias": adapter.bias,
            }
        )
        targets = adapter.target_modules
        result["lora_target_modules"] = (
            [targets] if isinstance(targets, str) else sorted(targets)
        )
    attention = getattr(model.config, "_attn_implementation", None)
    if attention not in {"eager", "sdpa"}:
        raise ValueError(f"actual attention backend {attention!r} is unsupported")
    result["attention_implementation"] = attention
    result["execution_backend"] = "native_framework_trainer"
    result["assumptions"] = [
        "Configuration values are reconciled with the actual native Trainer and model."
    ]
    kernel_choices = {}
    for name in (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
    ):
        choices = set()
        for module in model.modules():
            if hasattr(module, name):
                implementation = getattr(module, name)
                choices.add(
                    None
                    if implementation is None
                    else (
                        getattr(
                            implementation,
                            "__module__",
                            type(implementation).__module__,
                        )
                        + "."
                        + getattr(
                            implementation,
                            "__qualname__",
                            type(implementation).__qualname__,
                        )
                    )
                )
        if choices:
            kernel_choices[name] = sorted(choices, key=str)
    native_backend = {
        "kernel_implementations": kernel_choices,
        "collator_class": type(trainer.data_collator).__module__
        + "."
        + type(trainer.data_collator).__name__,
        "tf32_matmul": result["tf32"],
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
    }
    result["native_execution_signature"] = hashlib.sha256(
        json.dumps(
            {
                "native_config_signature": normalized["native_execution_signature"],
                "backend": native_backend,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    details = {
        "model_class": type(model).__module__ + "." + type(model).__name__,
        "trainer_class": type(trainer).__module__ + "." + type(trainer).__name__,
        "collator_class": type(trainer.data_collator).__module__
        + "."
        + type(trainer.data_collator).__name__,
        "parameter_count": result["parameter_count"],
        "total_parameter_count": total,
        "trainable_parameter_count": result["trainable_parameter_count"],
        "parameter_dtype_counts": dict(dtype_counts),
        "trainable_dtype_counts": dict(trainable_dtypes),
        "gradient_checkpointing": result["gradient_checkpointing"],
        "attention_implementation": attention,
        "training_arguments": args.to_dict(),
        **native_backend,
    }
    return result, details


def run_finetuning(
    config_path: str | Path, *, framework: str, profile_id: str, warmup_steps: int = 2
) -> dict[str, Any]:
    """Run native SFT; return evidence only after successful finite GPU updates.

    Use a fresh process/environment for each framework. No config field changes
    the configured training duration; warmup_steps excludes initial updates only
    from steady timing samples, independently of the native LR warmup setting.
    """
    started = time.perf_counter()
    path = Path(config_path).expanduser().resolve()
    raw, config, model_config, model_dir = _prepare_config(
        path, framework, warmup_steps
    )
    _reject_simulated_cuda_environment()
    if not _RUN_LOCK.acquire(blocking=False):
        raise RuntimeError(
            "native Trainer instrumentation cannot run concurrently in one process"
        )
    try:
        import torch
        from transformers import Trainer, TrainerCallback

        _reject_simulated_cuda_environment(torch)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError(
                "select exactly one physical CUDA GPU with CUDA_VISIBLE_DEVICES"
            )
        device = torch.cuda.get_device_properties(0)
        profile = get_profile(profile_id)
        if (
            device.name
            not in {profile.name, profile.torch_name, profile.official_model}
            or (device.major, device.minor) != profile.compute_capability
        ):
            raise ValueError("selected CUDA GPU does not match the requested profile")
        if config["dtype"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("selected CUDA GPU does not support BF16")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        hash_started = time.perf_counter()
        config, checkpoint_identity = bind_local_checkpoint(config, model_dir)
        hash_seconds = time.perf_counter() - hash_started
        config["software_versions"] = native_training_versions(framework)
        config["native_execution_signature"] = _execution_signature(raw)
        state = {}

        def inspect_trainer(trainer):
            dataset = trainer.train_dataset
            if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
                raise ValueError(
                    "native runner requires a finite map-style training dataset"
                )
            state["dataset"] = profile_tokenized_records(
                dataset,
                sequence_length=config["sequence_length"],
                source_framework=framework,
            )
            state["dataset"]["native_training_split_verified"] = True
            config["num_samples"] = state["dataset"]["training_inputs"]["num_samples"]
            config["data_signature"] = state["dataset"]["provenance"]["content_sha256"]

        def inspect_started_trainer(trainer):
            state["config"], state["model"] = _actual_training_config(trainer, config)
            if trainer.state.max_steps < warmup_steps + 2:
                raise ValueError(
                    "actual Trainer schedules fewer than two updates after warmup"
                )

        observer = NativeTrainerObserver(
            torch,
            warmup_steps=warmup_steps,
            on_trainer=inspect_trainer,
            on_train_begin=inspect_started_trainer,
        )
        with (
            capture_native_trainer(Trainer, TrainerCallback, observer),
            _native_directory(path.parent),
        ):
            _invoke_native(framework, path, raw)
        torch.cuda.synchronize()
        observer.record_peaks()
        finished = time.perf_counter()
        if observer.trainer is None or observer.outcome is None:
            raise RuntimeError(
                "native framework did not complete an observed Trainer.train call"
            )
        measured = [update for update in observer.updates if not update["warmup"]]
        if len(measured) < 2 or not math.isfinite(
            float(observer.outcome.training_loss)
        ):
            raise RuntimeError(
                "native training did not produce two finite measured optimizer updates"
            )
        if (
            not observer.parameter_changes
            or not observer.parameter_changes["has_changed_trainable_parameters"]
        ):
            raise RuntimeError(
                "native training produced no observed change in sampled trainable weights"
            )
        config = state["config"]
        shape_counts = [
            {"shape": list(shape), "microsteps": count}
            for shape, count in sorted(observer.shapes.items())
        ]
        expected_shape = (config["micro_batch_size"], config["sequence_length"])
        fixed_shape = set(observer.shapes) == {expected_shape}
        trainer_wall = observer.trainer_finished - observer.trainer_started
        sampled_updates = sum(update["seconds"] for update in observer.updates)
        training_loop = trainer_wall - sum(observer.during_train_phases.values())
        timing = {
            "total_wall_seconds": finished - started,
            "setup_seconds": observer.trainer_started - started,
            "trainer_wall_seconds": trainer_wall,
            "training_loop_seconds": training_loop,
            "evaluation_seconds": observer.phases["evaluation"],
            "checkpoint_seconds": observer.phases["checkpoint"],
            "finalization_seconds": finished - observer.trainer_finished,
            "optimizer_update_sample_seconds": sampled_updates,
            "trainer_other_seconds": max(0.0, training_loop - sampled_updates),
            "instrumentation_seconds": observer.instrumentation_seconds,
            "checkpoint_hash_seconds": hash_seconds,
            "warmup_training_seconds": sum(
                update["seconds"] for update in observer.updates if update["warmup"]
            ),
            "eventcounts": {
                key: observer.events[key] for key in ("evaluation", "checkpoint")
            },
            "during_training_eventcounts": {
                key: observer.during_train_events[key]
                for key in ("evaluation", "checkpoint")
            },
            "during_training_evaluation_seconds": observer.during_train_phases[
                "evaluation"
            ],
            "during_training_checkpoint_seconds": observer.during_train_phases[
                "checkpoint"
            ],
        }
        observation = {
            "schema_version": "fakegpu.finetune_observation.v1",
            "source": "fakegpu.native_framework_cuda_run",
            "native_framework": framework,
            "signature": finetune_workload_signature(
                model_config, config, profile_id=profile_id
            ),
            "optimizer_update_seconds": [update["seconds"] for update in measured],
            "seconds_per_microstep": [
                update["seconds"] / update["microsteps"] for update in measured
            ],
            "peak_allocated_bytes": observer.peak_allocated,
            "peak_reserved_bytes": observer.peak_reserved,
            "initialization_seconds": timing["setup_seconds"],
            "checkpoint_hash_seconds": hash_seconds,
            "checkpoint_identity": checkpoint_identity,
            "training_loss": float(observer.outcome.training_loss),
            "observed_model": state["model"],
            "parameter_changes": observer.parameter_changes,
            "software_versions": config["software_versions"],
            "hardware": {
                "name": device.name,
                "total_memory_bytes": device.total_memory,
                "compute_capability": f"{device.major}.{device.minor}",
                "cuda_version": torch.version.cuda,
            },
            "workload": {
                "kind": "native_framework_processed_text",
                "sha256": config["data_signature"],
                "sequence_length": config["sequence_length"],
                "samples": observer.samples,
                "optimizer_steps": len(observer.updates),
                "microsteps": observer.microsteps,
                "warmup_optimizer_steps": warmup_steps,
                "measured_optimizer_steps": len(measured),
                "warmup_microsteps": sum(
                    update["microsteps"]
                    for update in observer.updates
                    if update["warmup"]
                ),
                "measured_microsteps": sum(update["microsteps"] for update in measured),
                "shape_counts": shape_counts,
                "fixed_shape": fixed_shape,
            },
            "updates": observer.updates,
            "limitations": [
                "Update timing includes accumulation-window prefetch and forward/backward/optimizer work; evaluation and checkpoint events are timed separately.",
                "Allocator peaks include initialization, training and final saves; they are not total device usage.",
                "Wall time includes observer instrumentation and checkpoint hashing, reported separately.",
                "Changed weights are verified using sampled values per trainable tensor, not an exhaustive tensor comparison.",
                "Dynamic batch shapes are recorded but are unsuitable for fixed-shape timing extrapolation.",
            ],
        }
        args = observer.trainer.args
        schedule = {
            key: getattr(getattr(args, key, "no"), "value", getattr(args, key, "no"))
            for key in ("eval_strategy", "save_strategy")
        }
        return {
            "schema_version": "fakegpu.finetune_run.v1",
            "profile_id": profile_id,
            "framework": framework,
            "model_config": model_config,
            "training_config": config,
            "dataset": state["dataset"],
            "observation": observation,
            "timing": timing,
            "checkpoint_identity": checkpoint_identity,
            "native_config": raw,
            "config_path": str(path),
            "schedule": schedule,
            "workload": dict(observation["workload"]),
        }
    finally:
        _RUN_LOCK.release()
