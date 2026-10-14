"""Commands for explicit cached-feature diffusion training plans and evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from ._cli import add_json_path_argument, command_prog, usage_error
from .diffusion_training import (
    _run,
    _finite,
    compare_diffusion_training,
    estimate_diffusion_training,
    plan_diffusion_training,
    project_diffusion_training,
)
from .structured_io import emit_json, load_mapping
from ._finetune_cache import MODES as CACHE_MODES


def _parser(function, description):
    parser = argparse.ArgumentParser(
        prog=command_prog(__name__, function), description=description
    )
    add_json_path_argument(parser)
    return parser


def plan_main(argv=None):
    parser = _parser(
        "plan_main",
        "Inspect cached latent/text tensors and exact UNet training storage; no activation fallback.",
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument(
        "--dataset",
        required=True,
        help="FP32 cached-feature safetensors with explicit timesteps.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--method", choices=("full", "lora"), default="lora")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--ema-decay", type=float)
    args = parser.parse_args(argv)
    try:
        report = plan_diffusion_training(
            args.model_dir,
            args.dataset,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            epochs=args.epochs,
            method=args.method,
            lora_rank=args.lora_rank,
            gradient_checkpointing=args.gradient_checkpointing,
            ema_decay=args.ema_decay,
        )
        emit_json(args.json_path or "-", report)
    except (OSError, ValueError, TypeError) as exc:
        usage_error(parser, exc)
    return 0


def _plan(args):
    value = load_mapping(args.plan)
    return (
        value
        if args.epochs is None
        else project_diffusion_training(value, epochs=args.epochs)
    )


def run_main(argv=None):
    parser = _parser(
        "run_main",
        "Run all cached-feature epochs on exactly one visible CUDA GPU and save native weights.",
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-regime", choices=CACHE_MODES, default="uncontrolled")
    args = parser.parse_args(argv)
    try:
        if (
            args.json_path
            and args.json_path != "-"
            and Path(args.json_path).expanduser().exists()
        ):
            raise ValueError("native report already exists; choose a new attempt path")
        from .diffusion_training_runner import run_diffusion_training

        report = run_diffusion_training(
            _plan(args), output_dir=args.output_dir, cache_regime=args.cache_regime
        )
        # Preserve completed native evidence before its independent CPU validator.
        emit_json(args.json_path or "-", report)
        _run(report)
    except (OSError, ValueError, TypeError) as exc:
        usage_error(parser, exc)
    return 0


def estimate_main(argv=None):
    parser = _parser(
        "estimate_main",
        "Budget cached UNet training from two compatible independent CUDA processes; status 1 means missing coverage.",
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--calibration-run", action="append", required=True)
    args = parser.parse_args(argv)
    try:
        report = estimate_diffusion_training(
            _plan(args), [load_mapping(p) for p in args.calibration_run]
        )
        emit_json(args.json_path or "-", report)
    except (OSError, ValueError, TypeError) as exc:
        usage_error(parser, exc)
    return 0 if report["coverage"]["complete"] else 1


def validate_main(argv=None):
    parser = _parser(
        "validate_main",
        "Compare an independent cached-feature training holdout with prior native calibration.",
    )
    parser.add_argument("--holdout-run", required=True)
    parser.add_argument("--calibration-run", action="append", required=True)
    parser.add_argument("--max-total-time-error-percent", type=float)
    parser.add_argument("--max-memory-underestimate-percent", type=float)
    args = parser.parse_args(argv)
    try:
        report = compare_diffusion_training(
            [load_mapping(p) for p in args.calibration_run],
            load_mapping(args.holdout_run),
        )
        limits = {
            "total_time_error_percent": args.max_total_time_error_percent,
            "memory_underestimate_percent": args.max_memory_underestimate_percent,
        }
        for name, value in limits.items():
            if value is not None:
                _finite(value, name)
        passed = (
            args.max_total_time_error_percent is None
            or report["time_comparison"]["total_seconds"]["absolute_percentage_error"]
            <= args.max_total_time_error_percent
        ) and (
            args.max_memory_underestimate_percent is None
            or max(
                v["underprediction_percent"]
                for v in report["memory_comparison"].values()
            )
            <= args.max_memory_underestimate_percent
        )
        report["validation"] = {"limits": limits, "passed": passed}
        emit_json(args.json_path or "-", report)
    except (OSError, ValueError, TypeError) as exc:
        usage_error(parser, exc)
    return 0 if passed else 1
