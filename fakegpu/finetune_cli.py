"""Command-line entry points for framework-aware fine-tuning estimates."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import time
from typing import Any

from ._cli import add_json_path_argument, command_prog, usage_error
from ._local_checkpoint import bind_local_checkpoint
from .finetune_config import load_finetune_config
from .finetune_estimator import estimate_finetuning
from .structured_io import emit_json, load_mapping


def _input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", required=True, help="Framework JSON/YAML/TOML configuration."
    )
    parser.add_argument(
        "--framework",
        default="auto",
        help="auto, huggingface, llamafactory, or axolotl.",
    )
    parser.add_argument(
        "--model",
        help="Local model directory or config.json; overrides the framework model path.",
    )
    parser.add_argument(
        "--profile",
        required=True,
        help="Target GPU profile ID from the supported catalog.",
    )
    parser.add_argument(
        "--sequence-length", type=int, help="Processed tokens per training sequence."
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Explicitly select a multimodal model's text backbone.",
    )


def _load_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    training = load_finetune_config(args.config, framework=args.framework)
    model_path = args.model or training.get("model_name_or_path")
    if not model_path:
        raise ValueError(
            "provide --model or a model path in the framework configuration"
        )
    path = Path(model_path).expanduser().resolve()
    config_path = path / "config.json" if path.is_dir() else path
    if not config_path.is_file():
        raise ValueError(
            "estimation reads local model metadata; provide a local model directory or config.json"
        )
    model = load_mapping(config_path)
    if args.text_only:
        model = dict(model.get("text_config", model))
    training["model_name_or_path"] = str(config_path.parent)
    if args.sequence_length is not None:
        training["sequence_length"] = args.sequence_length
    return model, training, config_path.parent


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=command_prog(__name__),
        description="Estimate fine-tuning optimizer steps, total training time, and peak VRAM.",
    )
    _input_arguments(parser)
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Number of processed training sequences per epoch.",
    )
    parser.add_argument(
        "--epochs", type=float, help="Training epochs, unless max_steps overrides them."
    )
    parser.add_argument(
        "--max-steps", type=int, help="Override the number of optimizer updates."
    )
    parser.add_argument(
        "--observation",
        help="Matching real-GPU probe JSON for calibrated extrapolation.",
    )
    parser.add_argument(
        "--compute-acceleration-factor",
        type=float,
        default=1.0,
        help="Explicit matrix/scalar throughput factor for the uncalibrated roofline model.",
    )
    add_json_path_argument(parser)
    args = parser.parse_args(argv)
    try:
        model, training, model_dir = _load_inputs(args)
        if args.max_steps is not None:
            training["max_steps"] = args.max_steps
        observation = load_mapping(args.observation) if args.observation else None
        if observation is not None:
            from .finetune_probe import installed_training_versions

            training["software_versions"] = installed_training_versions()
            checkpoint_hash_started = time.perf_counter()
            training, checkpoint_identity = bind_local_checkpoint(training, model_dir)
            checkpoint_hash_seconds = time.perf_counter() - checkpoint_hash_started
        report = estimate_finetuning(
            model,
            training,
            profile_id=args.profile,
            num_samples=args.num_samples,
            epochs=args.epochs,
            observation=observation,
            compute_acceleration_factor=args.compute_acceleration_factor,
        )
        if observation is not None:
            report["checkpoint_identity"] = checkpoint_identity
            report["checkpoint_hash_seconds"] = checkpoint_hash_seconds
    except (OSError, ValueError, TypeError) as exc:
        usage_error(parser, exc)
    if args.json_path is not None:
        emit_json(args.json_path, report)
    else:
        _print_report(report)
    return 0


def _print_report(report: dict[str, Any]) -> None:
    workload, memory, timing = report["workload"], report["memory"], report["timing"]
    seconds = timing["training_loop_seconds"]
    print(
        f"Fine-tuning: {report['source_framework']} / {report['method']} / {report['profile_id']}"
    )
    print(
        f"Training sequences: {workload['num_samples']:,}; tokens per sequence: {workload['sequence_length']:,}"
    )
    print(
        f"Optimizer updates: {workload['optimizer_steps']:,}; microsteps: {workload['microsteps']:,}"
    )
    print(
        f"Training-loop time: {seconds['expected'] / 3600:.3f} hours "
        f"(range {seconds['lower'] / 3600:.3f}–{seconds['upper'] / 3600:.3f})"
    )
    print(
        f"Peak VRAM budget: {memory['selected_peak_bytes'] / 1024**3:.2f} GiB / "
        f"{memory['device_capacity_bytes'] / 1024**3:.2f} GiB; {memory['fit_status']}"
    )
    print(
        f"Evidence: {timing['evidence']}; architecture coverage: {report['coverage']}"
    )
    print(
        "Scope: training loop; download, tokenization, loading, evaluation and checkpoint I/O are excluded."
    )
    for limitation in report["limitations"]:
        print(f"  - {limitation}")


def probe_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=command_prog(__name__, "probe_main"),
        description="Measure a local model with Hugging Face Trainer on a physical CUDA device.",
    )
    _input_arguments(parser)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=2,
        help="Optimizer updates excluded from timing.",
    )
    parser.add_argument(
        "--measure-steps",
        type=int,
        default=5,
        help="Optimizer updates included in timing.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, help="Observation JSON path.")
    args = parser.parse_args(argv)
    try:
        from .finetune_probe import measure_finetuning

        model, training, model_dir = _load_inputs(args)
        observation = measure_finetuning(
            model,
            training,
            model_dir=model_dir,
            profile_id=args.profile,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
            seed=args.seed,
        )
        emit_json(args.output, observation)
    except (OSError, ValueError, TypeError, ImportError, RuntimeError) as exc:
        usage_error(parser, exc)
    print(f"Fine-tuning observation written to {Path(args.output).resolve()}")
    return 0
