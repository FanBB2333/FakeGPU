#!/usr/bin/env python3
"""Summarize Qwen SFT real/static/FakeCUDA worker reports as one matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "fakegpu.qwen_sft_matrix.v1"
MATCHING_FIELDS = (
    "model_dir",
    "dtype",
    "attention_implementation",
    "optimizer",
    "training_method",
    "gradient_checkpointing",
    "gradient_accumulation_steps",
    "lora",
    "batch_size",
    "sequence_length",
    "data_seed",
)


def summarize_cases(
    cases: dict[str, dict[str, dict[str, Any]]],
    *,
    max_static_error_percent: float = 2.0,
    max_fakecuda_error_percent: float = 3.0,
    max_upper_bound_overestimate_percent: float = 25.0,
) -> dict[str, Any]:
    results = []
    for name, reports in sorted(cases.items()):
        real = _require_report(reports, "real", name)
        static = _require_report(reports, "static", name)
        fake = reports.get("fakecuda")
        if fake is not None:
            _require_success(fake, "fakecuda", name)
        for report in (static, *([fake] if fake is not None else [])):
            for field in MATCHING_FIELDS:
                if report.get(field) != real.get(field):
                    raise ValueError(f"case {name!r} has mismatched {field}")
            if report["batch"]["fingerprint_sha256"] != real["batch"]["fingerprint_sha256"]:
                raise ValueError(f"case {name!r} has mismatched random batches")

        observed = int(real["memory_phases"]["overall_peak_bytes"])
        static_peak = int(static["memory_phases"]["overall_peak_bytes"])
        static_comparison = _comparison(static_peak, observed)
        static_kind = _prediction_kind(static)
        if static_kind == "upper_bound":
            static_passed = (
                static_comparison["signed_error"]
                >= -(observed * max_static_error_percent / 100.0)
                and static_comparison["signed_error"]
                <= observed * max_upper_bound_overestimate_percent / 100.0
            )
        else:
            static_passed = (
                static_comparison["absolute_error_percent"]
                <= max_static_error_percent
            )

        fake_comparison = None
        fake_passed = True
        if fake is not None:
            fake_comparison = _comparison(
                int(fake["memory_phases"]["overall_peak_bytes"]),
                observed,
            )
            fake_passed = (
                fake_comparison["absolute_error_percent"]
                <= max_fakecuda_error_percent
            )

        results.append(
            {
                "name": name,
                "status": "success" if static_passed and fake_passed else "failed",
                "model_dir": real["model_dir"],
                "gpu_name": real["gpu_name"],
                "training_method": real.get("training_method", "full"),
                "gradient_checkpointing": bool(real.get("gradient_checkpointing", False)),
                "gradient_accumulation_steps": int(real.get("gradient_accumulation_steps", 1)),
                "batch_size": int(real["batch_size"]),
                "sequence_length": int(real["sequence_length"]),
                "parameter_count": int(real["parameters"]["parameter_count"]),
                "trainable_parameter_count": int(
                    real["parameters"]["trainable_parameter_count"]
                ),
                "real_peak_bytes": observed,
                "static_peak_bytes": static_peak,
                "static_prediction_kind": static_kind,
                "static_comparison": static_comparison,
                "static_passed": static_passed,
                "fakecuda_peak_bytes": (
                    int(fake["memory_phases"]["overall_peak_bytes"])
                    if fake is not None
                    else None
                ),
                "fakecuda_comparison": fake_comparison,
                "fakecuda_passed": fake_passed if fake is not None else None,
                "real_phase_peaks": {
                    "forward": int(real["memory_phases"]["forward_peak_bytes"]),
                    "backward": int(real["memory_phases"]["backward_peak_bytes"]),
                    "optimizer": int(real["memory_phases"]["optimizer_peak_bytes"]),
                },
                "elapsed_seconds": {
                    "real": float(real["elapsed_seconds"]),
                    "static": float(static["elapsed_seconds"]),
                    "fakecuda": float(fake["elapsed_seconds"]) if fake is not None else None,
                },
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "success" if results and all(item["status"] == "success" for item in results) else "failed",
        "limits": {
            "max_static_error_percent": max_static_error_percent,
            "max_fakecuda_error_percent": max_fakecuda_error_percent,
            "max_upper_bound_overestimate_percent": max_upper_bound_overestimate_percent,
        },
        "case_count": len(results),
        "cases": results,
    }


def render_markdown(report: dict[str, Any]) -> str:
    rows = []
    for item in report["cases"]:
        fake_peak = (
            f"{_gib(item['fakecuda_peak_bytes']):.3f}"
            if item["fakecuda_peak_bytes"] is not None
            else "—"
        )
        fake_error = (
            f"{item['fakecuda_comparison']['absolute_error_percent']:.3f}%"
            if item["fakecuda_comparison"] is not None
            else "—"
        )
        rows.append(
            f"| {item['name']} | {item['gpu_name']} | {item['training_method']} | "
            f"{item['gradient_checkpointing']} | {item['gradient_accumulation_steps']} | "
            f"{item['sequence_length']} | {item['trainable_parameter_count']:,} | "
            f"{_gib(item['real_peak_bytes']):.3f} | {_gib(item['static_peak_bytes']):.3f} | "
            f"{item['static_comparison']['absolute_error_percent']:.3f}% | "
            f"{item['static_prediction_kind']} | {fake_peak} | {fake_error} | {item['status']} |"
        )
    return "\n".join(
        [
            "# Qwen SFT Memory Matrix",
            "",
            f"**Status:** `{report['status']}`",
            "",
            "| Case | GPU | Method | Checkpoint | Accum. | Seq. | Trainable params | Real GiB | Static GiB | Static error | Static kind | FakeCUDA GiB | FakeCUDA error | Status |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|",
            *rows,
            "",
        ]
    )


def load_cases(input_dir: Path, selected: list[str] | None = None) -> dict[str, dict[str, dict[str, Any]]]:
    selected_names = set(selected or [])
    cases: dict[str, dict[str, dict[str, Any]]] = {}
    for path in sorted(input_dir.glob("*.json")):
        stem = path.stem
        matched_mode = next(
            (mode for mode in ("real", "static", "fake") if stem.endswith(f"-{mode}")),
            None,
        )
        if matched_mode is None:
            continue
        name = stem[: -(len(matched_mode) + 1)]
        if selected_names and name not in selected_names:
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        mode = "fakecuda" if matched_mode == "fake" else matched_mode
        if report.get("mode") != mode:
            raise ValueError(f"{path} contains mode {report.get('mode')!r}, expected {mode!r}")
        cases.setdefault(name, {})[mode] = report
    if selected_names:
        missing = selected_names - cases.keys()
        if missing:
            raise ValueError(f"matrix cases not found: {', '.join(sorted(missing))}")
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--case", action="append")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown")
    parser.add_argument("--max-static-error-percent", type=float, default=2.0)
    parser.add_argument("--max-fakecuda-error-percent", type=float, default=3.0)
    parser.add_argument("--max-upper-bound-overestimate-percent", type=float, default=25.0)
    args = parser.parse_args(argv)
    try:
        cases = load_cases(Path(args.input_dir), args.case)
        report = summarize_cases(
            cases,
            max_static_error_percent=args.max_static_error_percent,
            max_fakecuda_error_percent=args.max_fakecuda_error_percent,
            max_upper_bound_overestimate_percent=args.max_upper_bound_overestimate_percent,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"Qwen SFT matrix summary failed: {exc}", file=sys.stderr)
        return 2
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = Path(args.markdown).expanduser().resolve() if args.markdown else output.with_suffix(".md")
    markdown.write_text(render_markdown(report), encoding="utf-8")
    print(f"Qwen SFT matrix: {report['status']} ({report['case_count']} cases)")
    print(f"JSON: {output}")
    print(f"Markdown: {markdown}")
    return 0 if report["status"] == "success" else 1


def _require_report(reports: dict[str, dict[str, Any]], mode: str, name: str) -> dict[str, Any]:
    report = reports.get(mode)
    if report is None:
        raise ValueError(f"case {name!r} is missing a {mode} report")
    _require_success(report, mode, name)
    return report


def _require_success(report: dict[str, Any], mode: str, name: str) -> None:
    if report.get("status") != "success" or report.get("mode") != mode:
        raise ValueError(f"case {name!r} does not contain a successful {mode} report")


def _prediction_kind(static: dict[str, Any]) -> str:
    analysis = static.get("static_analysis") or {}
    checkpointing = str(analysis.get("checkpointing", "disabled"))
    accumulation = str(analysis.get("gradient_accumulation", "single_microbatch_exact"))
    if checkpointing == "disabled" and accumulation == "single_microbatch_exact":
        return "exact"
    if checkpointing.startswith("analytical_") or accumulation == "in_place_largest_gradient_temporary":
        return "analytical"
    return "upper_bound"


def _comparison(predicted: int, observed: int) -> dict[str, int | float]:
    signed = predicted - observed
    return {
        "predicted": predicted,
        "observed": observed,
        "signed_error": signed,
        "absolute_error_bytes": abs(signed),
        "absolute_error_percent": round(100.0 * abs(signed) / observed, 9) if observed else 0.0,
    }


def _gib(value: int) -> float:
    return int(value) / (1024**3)


if __name__ == "__main__":
    raise SystemExit(main())
