"""Plan, run, resume and compare finite foreground experiment sweeps."""

from __future__ import annotations

import argparse

from ._cli import add_json_path_argument, command_prog, usage_error
from .experiments import (
    compare_experiments,
    plan_experiments,
    plot_experiments,
    run_experiments,
)
from .structured_io import emit_json


def _parser(function, description):
    parser = argparse.ArgumentParser(
        prog=command_prog(__name__, function), description=description
    )
    add_json_path_argument(parser)
    return parser


def plan_main(argv=None):
    parser = _parser(
        "plan_main",
        "Hash declared inputs and expand a bounded sweep without execution.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args(argv)
    try:
        emit_json(args.json_path or "-", plan_experiments(args.config, args.workspace))
    except (OSError, ValueError, TypeError) as exc:
        usage_error(parser, exc)
    return 0


def run_main(argv=None):
    parser = _parser(
        "run_main",
        "Run a sweep sequentially; preserve attempts, logs and artifact hashes. Commands must stay in the foreground.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Explicitly retry failed/interrupted attempts; requires --resume.",
    )
    args = parser.parse_args(argv)
    try:
        report = run_experiments(
            args.config,
            args.workspace,
            resume=args.resume,
            retry_failed=args.retry_failed,
        )
        emit_json(args.json_path or "-", report)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        usage_error(parser, exc)
    return 0 if report["complete"] else 1


def compare_main(argv=None):
    parser = _parser(
        "compare_main",
        "Verify artifacts and compare repetitions per workspace; optional Matplotlib mean/range figure.",
    )
    parser.add_argument("--workspace", action="append", required=True)
    parser.add_argument("--plot")
    parser.add_argument("--x", help="Numeric parameter for --plot.")
    parser.add_argument("--metric", help="Recorded metric for --plot.")
    args = parser.parse_args(argv)
    if bool(args.plot) != bool(args.x and args.metric) or (
        not args.plot and (args.x or args.metric)
    ):
        parser.error("--plot, --x and --metric must be provided together")
    try:
        report = compare_experiments(args.workspace)
        if args.plot:
            plot_experiments(report, args.plot, x=args.x, metric=args.metric)
        emit_json(args.json_path or "-", report)
    except (OSError, ValueError, TypeError, ImportError) as exc:
        usage_error(parser, exc)
    return 0 if report["complete"] else 1
