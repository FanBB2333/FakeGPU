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


def _input_arguments(
    parser: argparse.ArgumentParser, *, native_report: bool = False
) -> None:
    inputs = (
        parser.add_mutually_exclusive_group(required=True) if native_report else parser
    )
    inputs.add_argument(
        "--config",
        required=not native_report,
        help="Framework JSON/YAML/TOML configuration.",
    )
    if native_report:
        inputs.add_argument(
            "--run-report",
            help="Measured native framework run JSON; verifies local weights and software before reuse.",
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
        required=not native_report,
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
    _input_arguments(parser, native_report=True)
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Number of processed training sequences per epoch.",
    )
    parser.add_argument(
        "--epochs", type=float, help="Training epochs, unless max_steps overrides them."
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Override optimizer updates; -1 removes the step limit and uses epochs.",
    )
    parser.add_argument(
        "--observation",
        help="Matching real-GPU probe JSON for calibrated extrapolation.",
    )
    parser.add_argument(
        "--dataset-report",
        help="Full-scan profile-finetune-data JSON; supplies processed counts and sequence length.",
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
        if args.run_report:
            report = _estimate_native_report(args)
            if args.json_path is not None:
                emit_json(args.json_path, report)
            else:
                _print_report(report)
            return 0
        if not args.profile:
            raise ValueError("--profile is required with --config")
        model, training, model_dir = _load_inputs(args)
        dataset = load_mapping(args.dataset_report) if args.dataset_report else None
        if dataset is not None:
            from .finetune_data_report import apply_dataset_report

            training = apply_dataset_report(
                training,
                dataset,
                num_samples=args.num_samples,
                sequence_length=args.sequence_length,
            )
        if args.max_steps is not None:
            training["max_steps"] = None if args.max_steps == -1 else args.max_steps
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
        if dataset is not None:
            report["dataset"] = dataset
            report["limitations"].extend(training["dataset_limitations"])
    except (OSError, ValueError, TypeError) as exc:
        usage_error(parser, exc)
    if args.json_path is not None:
        emit_json(args.json_path, report)
    else:
        _print_report(report)
    return 0


def _estimate_native_report(args: argparse.Namespace) -> dict[str, Any]:
    from .finetune_run_estimator import estimate_finetuning_run
    from .finetune_runner import native_training_versions

    if any(
        (
            args.model,
            args.text_only,
            args.sequence_length is not None,
            args.observation,
            args.dataset_report,
            args.framework != "auto",
            args.compute_acceleration_factor != 1.0,
        )
    ):
        raise ValueError(
            "--run-report uses its recorded model, framework, data and execution shape; only profile, samples, epochs and max-steps may be overridden"
        )
    run = load_mapping(args.run_report)
    if run.get("schema_version") != "fakegpu.finetune_run.v1":
        raise ValueError("unsupported native fine-tuning run schema")
    training = run.get("training_config")
    if not isinstance(training, dict) or not training.get("model_name_or_path"):
        raise ValueError("native run requires a local model in training_config")
    model_dir = Path(training["model_name_or_path"])
    hash_started = time.perf_counter()
    training, identity = bind_local_checkpoint(training, model_dir)
    hash_seconds = time.perf_counter() - hash_started
    training["software_versions"] = native_training_versions(run.get("framework"))
    run["training_config"] = training
    # A changed config.json also invalidates reuse, even when weights match.
    run["model_config"] = load_mapping(model_dir / "config.json")
    report = estimate_finetuning_run(
        run,
        profile_id=args.profile,
        num_samples=args.num_samples,
        epochs=args.epochs,
        max_steps=args.max_steps,
    )
    report["checkpoint_identity"] = identity
    report["checkpoint_hash_seconds"] = hash_seconds
    return report


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
    pipeline = timing.get("native_pipeline")
    if pipeline is not None:
        if pipeline["available"]:
            print(
                f"Native pipeline scenario: {pipeline['seconds']['expected'] / 3600:.3f} hours (same dataset and cache state; measured setup and final save included)"
            )
        else:
            print(
                "Native pipeline scenario unavailable: " + " ".join(pipeline["reasons"])
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


def run_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=command_prog(__name__, "run_main"),
        description="Run native LLaMA-Factory or Axolotl training and record data, timing and CUDA memory.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Native framework config; relative paths resolve from its directory.",
    )
    parser.add_argument(
        "--framework", required=True, choices=("llamafactory", "axolotl")
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=2,
        help="Initial optimizer updates excluded from steady timing.",
    )
    parser.add_argument("--output", required=True, help="Run report JSON path.")
    args = parser.parse_args(argv)
    try:
        from .finetune_runner import run_finetuning

        report = run_finetuning(
            args.config,
            framework=args.framework,
            profile_id=args.profile,
            warmup_steps=args.warmup_steps,
        )
        emit_json(args.output, report)
    except (OSError, ValueError, TypeError, ImportError, RuntimeError) as exc:
        usage_error(parser, exc)
    print(f"Native fine-tuning run written to {Path(args.output).resolve()}")
    return 0


def _validation_main(
    argv, *, entrypoint, description, estimate, allow_event_context=False
):
    parser = argparse.ArgumentParser(
        prog=command_prog(__name__, entrypoint), description=description
    )
    parser.add_argument(
        "--calibration-run", action="append", required=True,
        help="Native run JSON; repeat for at least two independent calibration processes.",
    )
    parser.add_argument(
        "--holdout-run", required=True,
        help="Independent native run JSON to predict and compare.",
    )
    if allow_event_context:
        parser.add_argument(
            "--event-context", action="store_true",
            help="Match steady windows using their recorded evaluation/checkpoint history.",
        )
    add_json_path_argument(parser)
    args = parser.parse_args(argv)
    try:
        options = (
            {"event_context": True}
            if allow_event_context and args.event_context else {}
        )
        report = estimate(
            [load_mapping(path) for path in args.calibration_run],
            load_mapping(args.holdout_run),
            **options,
        )
        emit_json(args.json_path or "-", report)
    except (OSError, ValueError, TypeError) as exc:
        usage_error(parser, exc)
    return 0


def windows_main(argv: Sequence[str] | None = None) -> int:
    from .finetune_window_estimator import estimate_finetuning_windows

    return _validation_main(
        argv, entrypoint="windows_main",
        description="Validate dynamic accumulation-window timing against an independent recorded native run.",
        estimate=estimate_finetuning_windows,
        allow_event_context=True,
    )


def events_main(argv: Sequence[str] | None = None) -> int:
    from .finetune_event_estimator import estimate_finetuning_events

    return _validation_main(
        argv, entrypoint="events_main",
        description="Validate native evaluation and save costs against independent recorded events.",
        estimate=estimate_finetuning_events,
    )
