"""Future native training plans and independently calibrated complete budgets."""

from __future__ import annotations

import argparse
import math

from ._cli import add_json_path_argument, command_prog, usage_error
from .structured_io import emit_json, load_mapping


def _duration_arguments(parser):
    duration = parser.add_mutually_exclusive_group(required=True)
    duration.add_argument(
        "--epochs",
        type=float,
        help=(
            "Target training epochs on the calibrated dataset, counted in updates "
            "as the source framework does (Axolotl: its max_steps conversion)."
        ),
    )
    duration.add_argument("--max-steps", type=int, help="Target optimizer updates.")
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Verify the processed dataset row count; different data require new calibration.",
    )


def plan_main(argv=None):
    from .finetune_pipeline_plan import plan_finetuning_pipeline

    parser = argparse.ArgumentParser(
        prog=command_prog(__name__, "plan_main"),
        description="Plan future sequential native training without reading any target timing.",
    )
    parser.add_argument(
        "--run-report",
        required=True,
        help="Calibration report containing an observed complete epoch.",
    )
    _duration_arguments(parser)
    add_json_path_argument(parser)
    args = parser.parse_args(argv)
    try:
        plan = plan_finetuning_pipeline(
            load_mapping(args.run_report),
            epochs=args.epochs,
            max_steps=args.max_steps,
            num_samples=args.num_samples,
        )
        emit_json(args.json_path or "-", plan)
    except (OSError, ValueError, TypeError) as exc:
        usage_error(parser, exc)
    return 0


def _budget_arguments(parser):
    parser.add_argument(
        "--calibration-run",
        action="append",
        required=True,
        help="Native run report; at least two independent runs with matching cache state.",
    )
    parser.add_argument(
        "--memory-budget-gib",
        type=float,
        help="Explicit allocator planning budget, bounded by the recorded device capacity.",
    )
    parser.add_argument(
        "--reserve-gib",
        type=float,
        default=0,
        help="Extra memory headroom beyond observed reserved peaks.",
    )
    add_json_path_argument(parser)


def _bytes(value, field):
    if value is None:
        return None
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field} must be finite and nonnegative")
    return math.ceil(value * 1024**3)


def estimate_main(argv=None):
    from .finetune_memory_estimator import estimate_finetuning_memory
    from .finetune_pipeline_estimator import estimate_finetuning_pipeline

    parser = argparse.ArgumentParser(
        prog=command_prog(__name__, "estimate_main"),
        description="Estimate complete native execution time and a measured memory envelope for a future plan.",
    )
    parser.add_argument(
        "--plan", required=True, help="Timing-free plan-finetune-pipeline JSON."
    )
    _budget_arguments(parser)
    args = parser.parse_args(argv)
    try:
        calibrations = [load_mapping(path) for path in args.calibration_run]
        plan = load_mapping(args.plan)
        report = {
            "schema_version": "fakegpu.finetune_pipeline_budget.v1",
            "pipeline": estimate_finetuning_pipeline(calibrations, plan),
            "memory": estimate_finetuning_memory(
                calibrations,
                plan,
                memory_budget_bytes=_bytes(args.memory_budget_gib, "memory budget"),
                reserve_bytes=_bytes(args.reserve_gib, "reserve"),
            ),
        }
        emit_json(args.json_path or "-", report)
    except (OSError, ValueError, TypeError) as exc:
        usage_error(parser, exc)
    return 0


def validate_main(argv=None):
    from .finetune_memory_estimator import compare_finetuning_memory
    from .finetune_pipeline_estimator import compare_finetuning_pipeline

    parser = argparse.ArgumentParser(
        prog=command_prog(__name__, "validate_main"),
        description="Compare full-span timing and memory predictions against an independent native run.",
    )
    parser.add_argument("--holdout-run", required=True)
    _budget_arguments(parser)
    args = parser.parse_args(argv)
    try:
        calibrations = [load_mapping(path) for path in args.calibration_run]
        target = load_mapping(args.holdout_run)
        report = {
            "schema_version": "fakegpu.finetune_pipeline_validation.v1",
            "pipeline": compare_finetuning_pipeline(calibrations, target),
            "memory": compare_finetuning_memory(
                calibrations,
                target,
                memory_budget_bytes=_bytes(args.memory_budget_gib, "memory budget"),
                reserve_bytes=_bytes(args.reserve_gib, "reserve"),
            ),
        }
        emit_json(args.json_path or "-", report)
    except (OSError, ValueError, TypeError) as exc:
        usage_error(parser, exc)
    return 0
