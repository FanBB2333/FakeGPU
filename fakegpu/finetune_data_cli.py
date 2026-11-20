"""Dataset profiling CLI and consistency checks for estimator report inputs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

from ._cli import add_json_path_argument, command_prog, usage_error
from .finetune_data import profile_finetune_dataset
from .structured_io import emit_json


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=command_prog(__name__),
        description="Measure tokens, truncation and padding from a full local JSONL dataset scan.",
    )
    parser.add_argument(
        "--data",
        action="append",
        required=True,
        help="Local JSONL file; repeat to include multiple files in order.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        required=True,
        help="Maximum processed sequence length and fixed padding budget.",
    )
    parser.add_argument(
        "--tokenizer",
        help="Local tokenizer directory; downloads and custom code are disabled.",
    )
    parser.add_argument(
        "--format", choices=("auto", "text", "messages", "input_ids"), default="auto"
    )
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--messages-field", default="messages")
    parser.add_argument(
        "--packing",
        choices=("none", "concatenate"),
        default="none",
        help="Generic token concatenation is distinct from framework packing.",
    )
    parser.add_argument(
        "--drop-remainder",
        action="store_true",
        help="Discard the final incomplete sequence with concatenate packing.",
    )
    parser.add_argument(
        "--no-special-tokens",
        action="store_true",
        help="Do not add tokenizer special tokens to raw text.",
    )
    parser.add_argument(
        "--separator-token-id",
        type=int,
        help="Explicit separator inserted between rows during concatenate packing.",
    )
    parser.add_argument(
        "--processed",
        action="store_true",
        help="Input rows are already processed training sequences; prohibit further packing or truncation.",
    )
    parser.add_argument(
        "--source-framework",
        help="Descriptive provenance only; does not verify framework execution.",
    )
    add_json_path_argument(parser)
    parser.add_argument("--json-path", dest="json_path", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        report = profile_finetune_dataset(
            args.data,
            sequence_length=args.sequence_length,
            tokenizer_path=args.tokenizer,
            record_format=args.format,
            text_field=args.text_field,
            messages_field=args.messages_field,
            packing=args.packing,
            drop_remainder=args.drop_remainder,
            add_special_tokens=not args.no_special_tokens,
            separator_token_id=args.separator_token_id,
            processed=args.processed,
            source_framework=args.source_framework,
        )
        if args.json_path is not None:
            emit_json(args.json_path, report)
        else:
            _print_report(report)
    except (OSError, ValueError, TypeError, ImportError, RuntimeError) as exc:
        usage_error(parser, exc)
    return 0


def _print_report(report: dict[str, Any]) -> None:
    inputs, tokens, padding = (
        report["training_inputs"],
        report["tokens"],
        report["padding"],
    )
    print(
        f"Dataset rows: {report['source_records']:,}; output sequences: {inputs['num_samples']:,}; sequence length: {inputs['sequence_length']:,}"
    )
    print(
        f"Tokens: {tokens['raw_tokens']:,}; truncated: {tokens['truncated_tokens']:,}; padding: {padding['padding_tokens']:,} ({padding['padding_fraction']:.1%})"
    )
    print(
        f"Count basis: {inputs['count_basis']}; packing: {report['packing']['strategy']}"
    )
    for limitation in report["limitations"]:
        print(f"  - {limitation}")


__all__ = ["main"]
