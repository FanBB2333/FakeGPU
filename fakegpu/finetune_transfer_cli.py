"""Sampled future plans and explicit empirical transfer budgets."""

from __future__ import annotations

import argparse

from ._cli import add_json_path_argument, command_prog, usage_error
from .finetune_pipeline_cli import _budget_arguments, _bytes
from .structured_io import emit_json, load_mapping


def plan_main(argv=None):
    from .finetune_transfer_plan import plan_finetuning_transfer

    parser = argparse.ArgumentParser(
        prog=command_prog(__name__, "plan_main"),
        description="Plan future native training from its verified CPU sampling horizon.",
    )
    parser.add_argument("--run-report", required=True)
    duration = parser.add_mutually_exclusive_group(required=True)
    duration.add_argument("--epochs", type=float)
    duration.add_argument("--max-steps", type=int)
    add_json_path_argument(parser)
    args = parser.parse_args(argv)
    try:
        emit_json(
            args.json_path or "-",
            plan_finetuning_transfer(
                load_mapping(args.run_report),
                epochs=args.epochs,
                max_steps=args.max_steps,
            ),
        )
    except (OSError, ValueError, TypeError) as exc:
        usage_error(parser, exc)
    return 0


def _parser(function, description):
    parser = argparse.ArgumentParser(
        prog=command_prog(__name__, function), description=description
    )
    _budget_arguments(parser)
    parser.add_argument(
        "--setup-calibration-run",
        action="append",
        required=True,
        help="Two independent native reports for this exact target dataset and planning horizon.",
    )
    return parser


def _inputs(args):
    return (
        [load_mapping(path) for path in args.calibration_run],
        [load_mapping(path) for path in args.setup_calibration_run],
    )


def estimate_main(argv=None):
    from .finetune_transfer_estimator import (
        estimate_finetuning_transfer,
        estimate_finetuning_transfer_memory,
        inspect_finetuning_transfer_coverage,
    )

    parser = _parser(
        "estimate_main",
        "Budget sampled native training with explicit empirical shape transfer.",
    )
    parser.add_argument("--plan", required=True)
    args = parser.parse_args(argv)
    try:
        work, setup = _inputs(args)
        plan = load_mapping(args.plan)
        coverage = inspect_finetuning_transfer_coverage(
            work, plan, setup_calibration_runs=setup
        )
        result = {
            "schema_version": "fakegpu.finetune_transfer_budget.v1",
            "coverage": coverage,
        }
        if coverage["complete"]:
            result.update(
                pipeline=estimate_finetuning_transfer(
                    work, plan, setup_calibration_runs=setup
                ),
                memory=estimate_finetuning_transfer_memory(
                    work,
                    plan,
                    setup_calibration_runs=setup,
                    memory_budget_bytes=_bytes(args.memory_budget_gib, "memory budget"),
                    reserve_bytes=_bytes(args.reserve_gib, "reserve"),
                ),
            )
        emit_json(args.json_path or "-", result)
    except (OSError, ValueError, TypeError) as exc:
        usage_error(parser, exc)
    return 0 if coverage["complete"] else 1


def validate_main(argv=None):
    from .finetune_transfer_estimator import (
        compare_finetuning_transfer,
        compare_finetuning_transfer_memory,
    )

    parser = _parser(
        "validate_main",
        "Compare a transfer budget against independent native timing and memory evidence.",
    )
    parser.add_argument("--holdout-run", required=True)
    args = parser.parse_args(argv)
    try:
        work, setup = _inputs(args)
        target = load_mapping(args.holdout_run)
        emit_json(
            args.json_path or "-",
            {
                "schema_version": "fakegpu.finetune_transfer_validation.v1",
                "pipeline": compare_finetuning_transfer(
                    work, target, setup_calibration_runs=setup
                ),
                "memory": compare_finetuning_transfer_memory(
                    work,
                    target,
                    setup_calibration_runs=setup,
                    memory_budget_bytes=_bytes(args.memory_budget_gib, "memory budget"),
                    reserve_bytes=_bytes(args.reserve_gib, "reserve"),
                ),
            },
        )
    except (OSError, ValueError, TypeError) as exc:
        usage_error(parser, exc)
    return 0
