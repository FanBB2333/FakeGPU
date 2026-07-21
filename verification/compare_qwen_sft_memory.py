#!/usr/bin/env python3
"""Compare matching real-CUDA, FakeCUDA, and static Qwen SFT reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "fakegpu.qwen_sft_memory_comparison.v1"


def compare_reports(
    real: dict[str, Any],
    fake: dict[str, Any],
    static: dict[str, Any],
    *,
    max_overall_error_percent: float = 2.0,
    max_phase_error_percent: float = 5.0,
) -> dict[str, Any]:
    _require_success(real, "real")
    _require_success(fake, "fakecuda")
    _require_success(static, "static")
    matching_fields = (
        "model_dir",
        "dtype",
        "attention_implementation",
        "optimizer",
        "batch_size",
        "sequence_length",
        "data_seed",
    )
    for field in matching_fields:
        if real.get(field) != fake.get(field) or real.get(field) != static.get(field):
            raise ValueError(f"real/fake/static {field} mismatch")
    fingerprints = {
        str(report["batch"]["fingerprint_sha256"])
        for report in (real, fake, static)
    }
    if len(fingerprints) != 1:
        raise ValueError("real/fake/static random SFT batches do not match")

    comparisons = {
        "model_load_allocator": _comparison(
            int(fake["memory_phases"]["model_load_current_bytes"]),
            int(real["memory_phases"]["model_load_current_bytes"]),
        ),
        "fakecuda_forward_peak": _comparison(
            int(fake["memory_phases"]["forward_peak_bytes"]),
            int(real["memory_phases"]["forward_peak_bytes"]),
        ),
        "fakecuda_backward_peak": _comparison(
            int(fake["memory_phases"]["backward_peak_bytes"]),
            int(real["memory_phases"]["backward_peak_bytes"]),
        ),
        "fakecuda_optimizer_peak": _comparison(
            int(fake["memory_phases"]["optimizer_peak_bytes"]),
            int(real["memory_phases"]["optimizer_peak_bytes"]),
        ),
        "fakecuda_overall_peak": _comparison(
            int(fake["memory_phases"]["overall_peak_bytes"]),
            int(real["memory_phases"]["overall_peak_bytes"]),
        ),
        "static_overall_peak": _comparison(
            int(static["memory_phases"]["overall_peak_bytes"]),
            int(real["memory_phases"]["overall_peak_bytes"]),
        ),
        "fakecuda_loss": _float_comparison(float(fake["loss"]), float(real["loss"])),
    }
    real_parameter_bytes = int(real["parameters"]["parameter_bytes"])
    checks = {
        "random_batch_exact": len(fingerprints) == 1,
        "parameter_count_exact": len(
            {int(report["parameters"]["parameter_count"]) for report in (real, fake, static)}
        )
        == 1,
        "parameter_bytes_exact": (
            real_parameter_bytes
            == int(fake["parameters"]["parameter_bytes"])
            == int(static["parameters"]["parameter_bytes"])
            == int(static["memory_phases"]["parameter_bytes"])
        ),
        "model_load_error_within_limit": (
            comparisons["model_load_allocator"]["absolute_error_percent"]
            <= max_phase_error_percent
        ),
        "forward_peak_error_within_limit": (
            comparisons["fakecuda_forward_peak"]["absolute_error_percent"]
            <= max_phase_error_percent
        ),
        "backward_peak_error_within_limit": (
            comparisons["fakecuda_backward_peak"]["absolute_error_percent"]
            <= max_phase_error_percent
        ),
        "optimizer_peak_error_within_limit": (
            comparisons["fakecuda_optimizer_peak"]["absolute_error_percent"]
            <= max_phase_error_percent
        ),
        "fakecuda_overall_error_within_limit": (
            comparisons["fakecuda_overall_peak"]["absolute_error_percent"]
            <= max_overall_error_percent
        ),
        "static_overall_error_within_limit": (
            comparisons["static_overall_peak"]["absolute_error_percent"]
            <= max_overall_error_percent
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "success" if all(checks.values()) else "failed",
        "model": {
            "model_dir": real["model_dir"],
            "parameter_count": int(real["parameters"]["parameter_count"]),
            "parameter_bytes": real_parameter_bytes,
            "dtype": real["dtype"],
        },
        "workload": {
            "batch_size": int(real["batch_size"]),
            "sequence_length": int(real["sequence_length"]),
            "masked_prefix_tokens": int(real["batch"]["masked_prefix_tokens"]),
            "optimizer": real["optimizer"],
            "attention_implementation": real["attention_implementation"],
            "fingerprint_sha256": next(iter(fingerprints)),
        },
        "limits": {
            "max_overall_error_percent": float(max_overall_error_percent),
            "max_phase_error_percent": float(max_phase_error_percent),
        },
        "comparisons": comparisons,
        "checks": checks,
        "real": {
            "gpu_name": real["gpu_name"],
            "loss": float(real["loss"]),
            "memory_phases": real["memory_phases"],
            "phase_seconds": real["phase_seconds"],
        },
        "fakecuda": {
            "gpu_name": fake["gpu_name"],
            "loss": float(fake["loss"]),
            "memory_phases": fake["memory_phases"],
            "phase_seconds": fake["phase_seconds"],
        },
        "static": {
            "gpu_name": static["gpu_name"],
            "memory_phases": static["memory_phases"],
            "elapsed_seconds": static["elapsed_seconds"],
            "tracking_confidence": static["static_estimate"]["tracking_confidence"],
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    labels = {
        "model_load_allocator": "Model load: FakeCUDA vs real allocator",
        "fakecuda_forward_peak": "Forward peak: FakeCUDA vs real allocator",
        "fakecuda_backward_peak": "Backward peak: FakeCUDA vs real allocator",
        "fakecuda_optimizer_peak": "Optimizer peak: FakeCUDA vs real allocator",
        "fakecuda_overall_peak": "Overall peak: FakeCUDA vs real allocator",
        "static_overall_peak": "Overall peak: static graph vs real allocator",
        "fakecuda_loss": "SFT loss: FakeCUDA CPU execution vs real CUDA",
    }
    rows = []
    for key, label in labels.items():
        item = report["comparisons"][key]
        if key == "fakecuda_loss":
            rows.append(
                f"| {label} | {item['predicted']:.8f} | {item['observed']:.8f} | "
                f"{item['signed_error']:+.8f} | {item['absolute_error_percent']:.6f}% |"
            )
        else:
            rows.append(
                f"| {label} | {item['predicted']:,} | {item['observed']:,} | "
                f"{item['signed_error']:+,} | {item['absolute_error_percent']:.6f}% |"
            )
    checks = [
        f"- [{'x' if passed else ' '}] `{name}`"
        for name, passed in report["checks"].items()
    ]
    model = report["model"]
    workload = report["workload"]
    return "\n".join(
        [
            "# Qwen SFT Memory Validation",
            "",
            f"**Status:** `{report['status']}`",
            "",
            f"- Model: `{model['model_dir']}`",
            f"- Parameters: `{model['parameter_count']:,}` (`{model['parameter_bytes']:,}` bytes)",
            f"- Workload: batch `{workload['batch_size']}`, sequence `{workload['sequence_length']}`, "
            f"masked prefix `{workload['masked_prefix_tokens']}`, `{model['dtype']}`",
            f"- Optimizer: `{workload['optimizer']}`; attention: `{workload['attention_implementation']}`",
            "",
            "| Comparison | Predicted | Observed | Signed error | Absolute error |",
            "|---|---:|---:|---:|---:|",
            *rows,
            "",
            "## Checks",
            "",
            *checks,
            "",
            "The memory reference is PyTorch's peak allocated bytes for this process. "
            "It excludes CUDA context and allocator-reserved-but-unused memory.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", required=True)
    parser.add_argument("--fakecuda", required=True)
    parser.add_argument("--static", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown")
    parser.add_argument("--max-overall-error-percent", type=float, default=2.0)
    parser.add_argument("--max-phase-error-percent", type=float, default=5.0)
    args = parser.parse_args(argv)
    try:
        reports = [
            json.loads(Path(path).read_text(encoding="utf-8"))
            for path in (args.real, args.fakecuda, args.static)
        ]
        report = compare_reports(
            *reports,
            max_overall_error_percent=args.max_overall_error_percent,
            max_phase_error_percent=args.max_phase_error_percent,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"Qwen SFT memory comparison failed: {exc}", file=sys.stderr)
        return 2
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = Path(args.markdown).expanduser().resolve() if args.markdown else output.with_suffix(".md")
    markdown.write_text(render_markdown(report), encoding="utf-8")
    print(f"Qwen SFT memory comparison: {report['status']}")
    print(f"JSON: {output}")
    print(f"Markdown: {markdown}")
    return 0 if report["status"] == "success" else 1


def _comparison(predicted: int, observed: int) -> dict[str, int | float]:
    signed = int(predicted) - int(observed)
    return {
        "predicted": int(predicted),
        "observed": int(observed),
        "signed_error": signed,
        "absolute_error_bytes": abs(signed),
        "absolute_error_percent": round(100.0 * abs(signed) / observed, 9) if observed else 0.0,
    }


def _float_comparison(predicted: float, observed: float) -> dict[str, float]:
    signed = float(predicted) - float(observed)
    return {
        "predicted": float(predicted),
        "observed": float(observed),
        "signed_error": signed,
        "absolute_error": abs(signed),
        "absolute_error_percent": round(100.0 * abs(signed) / abs(observed), 9) if observed else 0.0,
    }


def _require_success(report: dict[str, Any], mode: str) -> None:
    if report.get("status") != "success" or report.get("mode") != mode:
        raise ValueError(f"expected a successful {mode} worker report")


if __name__ == "__main__":
    raise SystemExit(main())
