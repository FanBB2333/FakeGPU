"""Native calibration workspaces and CPU-only budget reuse."""

from __future__ import annotations

import argparse
import math
import subprocess

from ._cli import add_json_path_argument, command_prog, usage_error
from .structured_io import emit_json


def _arguments(parser):
    parser.add_argument(
        "--workspace", required=True, help="Calibration evidence directory."
    )
    duration = parser.add_mutually_exclusive_group(required=True)
    duration.add_argument("--epochs", type=float, help="Future training epochs.")
    duration.add_argument("--max-steps", type=int, help="Future optimizer updates.")
    parser.add_argument("--memory-budget-gib", type=float)
    parser.add_argument("--reserve-gib", type=float, default=0)
    parser.add_argument(
        "--reuse-calibration-run",
        action="append",
        help="Import an existing native report into the work-cost pool; setup remains tied to this workspace.",
    )
    add_json_path_argument(parser)


def _bytes(value, field):
    if value is None:
        return None
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field} must be finite and nonnegative")
    return math.ceil(value * 1024**3)


def _budget_options(args):
    return {
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "memory_budget_bytes": _bytes(args.memory_budget_gib, "memory budget"),
        "reserve_bytes": _bytes(args.reserve_gib, "reserve"),
        "reuse_calibration_runs": args.reuse_calibration_run,
    }


def calibrate_main(argv=None):
    from .finetune_workflow import calibrate_finetuning

    parser = argparse.ArgumentParser(
        prog=command_prog(__name__, "calibrate_main"),
        description="Run two fixed native calibrations, preserve attempts and seal future timing/memory budgets.",
    )
    _arguments(parser)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--framework", required=True, choices=("huggingface", "llamafactory", "axolotl")
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--python",
        dest="python_executable",
        help="Python interpreter with the requested native framework installed.",
    )
    parser.add_argument(
        "--cuda-device",
        required=True,
        help="One CUDA_VISIBLE_DEVICES index or GPU UUID.",
    )
    parser.add_argument(
        "--calibration-steps",
        required=True,
        type=int,
        help="Fixed optimizer updates in each of the two fresh calibration processes.",
    )
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument(
        "--length-grouping-seed",
        type=int,
        help="Separate length grouping from model RNG; evaluation uses a fixed independent ordering.",
    )
    parser.add_argument(
        "--packing-seed",
        type=int,
        help="Use isolated Qwen3.5 text LoRA packing with an independent source seed",
    )
    parser.add_argument(
        "--cache-regime",
        choices=("uncontrolled", "empty_kernel_cache"),
        default="empty_kernel_cache",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        help="Maximum wall time for each newly launched child process.",
    )
    parser.add_argument(
        "--calibration-budget-seconds",
        type=float,
        help="Total wall-time limit across new calibration attempts in this invocation.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse verified successful attempts and preserve failed attempts before retrying their slots.",
    )
    args = parser.parse_args(argv)
    try:
        result = calibrate_finetuning(
            args.config,
            framework=args.framework,
            profile_id=args.profile,
            workspace=args.workspace,
            cuda_visible_devices=args.cuda_device,
            calibration_steps=args.calibration_steps,
            python_executable=args.python_executable,
            warmup_steps=args.warmup_steps,
            cache_regime=args.cache_regime,
            timeout_seconds=args.timeout_seconds,
            calibration_budget_seconds=args.calibration_budget_seconds,
            resume=args.resume,
            length_grouping_seed=args.length_grouping_seed,
            **(
                {"packing_seed": args.packing_seed}
                if args.packing_seed is not None
                else {}
            ),
            **_budget_options(args),
        )
        emit_json(args.json_path or "-", result)
    except (
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as exc:
        usage_error(parser, exc)
    return 0 if result["status"] == "complete" else 1


def budget_main(argv=None):
    from .finetune_workflow import budget_finetuning_workspace

    parser = argparse.ArgumentParser(
        prog=command_prog(__name__, "budget_main"),
        description="Recompute a budget from verified calibration evidence without launching training.",
    )
    _arguments(parser)
    args = parser.parse_args(argv)
    try:
        result = budget_finetuning_workspace(args.workspace, **_budget_options(args))
        emit_json(args.json_path or "-", result)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        usage_error(parser, exc)
    return 0 if result["status"] == "complete" else 1
