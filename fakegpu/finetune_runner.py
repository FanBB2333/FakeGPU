"""Execute supported native framework training and observe the actual Trainer.

Framework packages remain optional. This module imports no Torch or training
framework until run_finetuning has validated the requested local configuration.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any
from uuid import uuid4

from ._calibration_cuda import _reject_simulated_cuda_environment
from ._finetune_batches import summarize_batch_updates, validate_batch_profile
from ._finetune_events import summarize_phase_events, validate_event_profile
from ._finetune_model import (
    actual_training_config as _actual_training_config,
    native_config_signature as _execution_signature,
)
from ._finetune_observer import NativeTrainerObserver, capture_native_trainer
from ._local_checkpoint import bind_local_checkpoint
from .finetune_config import normalize_finetune_config
from .finetune_data import profile_tokenized_records
from .finetune_estimator import finetune_workload_signature
from .finetune_probe import installed_training_versions
from .profile_catalog import get_profile
from .structured_io import load_mapping


_RUN_LOCK = threading.Lock()


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
    config = normalize_finetune_config(raw, framework=framework, base_dir=path.parent)
    if config["quantization_bits"] not in {None, 4}:
        raise ValueError("native runner supports only 4-bit bitsandbytes quantization")
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
        "unfrozen_parameters",
        "pure_bf16",
        "export_quantization_bit",
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


def run_finetuning(
    config_path: str | Path, *, framework: str, profile_id: str, warmup_steps: int = 2
) -> dict[str, Any]:
    """Run native SFT; return evidence only after successful finite GPU updates.

    Use a fresh process/environment for each framework. No config field changes
    the configured training duration; warmup_steps excludes initial updates only
    from steady timing samples, independently of the native LR warmup setting.
    """
    started = time.perf_counter()
    run_identity = {
        "uuid": str(uuid4()),
        "pid": os.getpid(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
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
        config["batch_observation_version"] = "fakegpu.finetune_batches.v1"
        config["event_observation_version"] = "fakegpu.finetune_events.v1"
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
            run_started=started,
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
        if config["method"] == "qlora":
            # Linear4bit can select its effective compute dtype on first forward.
            # Reconcile after execution rather than recording only its load-time default.
            state["config"], state["model"] = _actual_training_config(
                observer.trainer, config
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
            "batch_instrumentation_seconds": observer.batch_instrumentation_seconds,
            "event_instrumentation_seconds": observer.event_instrumentation_seconds,
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
            "batch_profile": summarize_batch_updates(observer.updates),
            "phase_events": observer.phase_events,
            "event_profile": summarize_phase_events(observer.phase_events),
            "limitations": [
                "Update timing includes accumulation-window prefetch and forward/backward/optimizer work; evaluation and checkpoint events are timed separately.",
                "Allocator peaks include initialization, training and final saves; they are not total device usage.",
                "Wall time includes observer instrumentation and checkpoint hashing, reported separately.",
                "Changed weights are verified using sampled values per trainable tensor, not an exhaustive tensor comparison.",
                "Dynamic batch shapes are recorded but are unsuitable for fixed-shape timing extrapolation.",
                "Batch metadata collection synchronizes each microstep; its measured cost is excluded from update samples and included in total wall time.",
                "Timing groups describe complete ordered accumulation windows, not separate per-shape kernel timings or memory peaks.",
                "Token counts require a binary two-dimensional attention mask. Position-ID resets do not establish original packed-example boundaries.",
                "Phase event timings synchronize CUDA and count only root intervals in phase totals; nested calls remain in the event ledger.",
                "Event metadata costs remain in raw wall time and are excluded from net event seconds. Save manifests describe present files, not physical disk writes or durability.",
            ],
        }
        validate_batch_profile(observation)
        validate_event_profile(observation, timing=timing)
        args = observer.trainer.args
        schedule = {
            key: getattr(getattr(args, key, "no"), "value", getattr(args, key, "no"))
            for key in ("eval_strategy", "save_strategy")
        }
        schedule.update(
            requested_eval_steps=args.eval_steps,
            requested_save_steps=args.save_steps,
            resolved_eval_steps=observer.trainer.state.eval_steps,
            resolved_save_steps=observer.trainer.state.save_steps,
            eval_on_start=args.eval_on_start,
            eval_delay=args.eval_delay,
            save_only_model=args.save_only_model,
            save_total_limit=args.save_total_limit,
            load_best_model_at_end=args.load_best_model_at_end,
            max_steps=observer.trainer.state.max_steps,
            num_train_epochs=args.num_train_epochs,
        )
        return {
            "schema_version": "fakegpu.finetune_run.v1",
            "run_identity": run_identity,
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
