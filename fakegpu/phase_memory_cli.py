"""Additional ``calibrate phase-*`` actions; optional CUDA imports stay lazy."""

from pathlib import Path

from ._cli import add_json_path_argument
from .phase_memory import estimate_phase_memory, validate_phase_memory
from .structured_io import emit_json, load_mapping


ACTIONS = ("phase-sample", "phase-estimate", "phase-validate")


def configure(subparsers):
    parser = subparsers.add_parser(
        "phase-sample",
        help="Measure both allocated/reserved peaks in one fresh real-CUDA process (local serving or SD/SDXL).",
    )
    parser.add_argument("prediction")
    add_json_path_argument(parser)
    parser = subparsers.add_parser(
        "phase-estimate",
        help="Seal exact-workload phase envelopes from at least five independent native processes.",
    )
    parser.add_argument("--calibration-sample", action="append", required=True)
    add_json_path_argument(parser)
    parser = subparsers.add_parser(
        "phase-validate",
        help="Validate an independent holdout against an already sealed phase budget.",
    )
    parser.add_argument("--budget", required=True)
    parser.add_argument("--holdout-sample", required=True)
    parser.add_argument("--max-absolute-error-percent", type=float, default=20.0)
    parser.add_argument("--max-underestimate-percent", type=float, default=10.0)
    add_json_path_argument(parser)


def dispatch(args):
    if (
        args.action in {"phase-sample", "phase-estimate"}
        and args.json_path
        and args.json_path != "-"
        and Path(args.json_path).expanduser().exists()
    ):
        raise ValueError(
            "sample or sealed budget already exists; choose a fresh output path"
        )
    if args.action == "phase-sample":
        from .phase_memory_runner import sample_phase_memory

        report = sample_phase_memory(load_mapping(args.prediction))
    elif args.action == "phase-estimate":
        report = estimate_phase_memory(
            [load_mapping(p) for p in args.calibration_sample]
        )
    else:
        report = validate_phase_memory(
            load_mapping(args.budget),
            load_mapping(args.holdout_sample),
            max_absolute_error_percent=args.max_absolute_error_percent,
            max_underestimate_percent=args.max_underestimate_percent,
        )
    emit_json(args.json_path or "-", report)
    return 0 if args.action != "phase-validate" or report["passed"] else 1
