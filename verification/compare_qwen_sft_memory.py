#!/usr/bin/env python3
"""Compare matching real-CUDA, FakeCUDA, and static Qwen SFT reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "fakegpu.qwen_sft_memory_comparison.v2"


def compare_reports(
    real: dict[str, Any],
    fake: dict[str, Any],
    static: dict[str, Any],
    *,
    max_overall_error_percent: float = 2.0,
    max_phase_error_percent: float = 5.0,
    max_upper_bound_overestimate_percent: float = 25.0,
) -> dict[str, Any]:
    _require_success(real, "real")
    _require_success(fake, "fakecuda")
    _require_success(static, "static")
    matching_fields = (
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
    for field in matching_fields:
        if real.get(field) != fake.get(field) or real.get(field) != static.get(field):
            raise ValueError(f"real/fake/static {field} mismatch")
    fingerprints = {
        str(report["batch"]["fingerprint_sha256"])
        for report in (real, fake, static)
    }
    if len(fingerprints) != 1:
        raise ValueError("real/fake/static random SFT batches do not match")

    static_analysis = static.get("static_analysis") or {
        "checkpointing": "disabled",
        "gradient_accumulation": "single_microbatch_exact",
    }
    static_is_exact = (
        static_analysis.get("checkpointing") == "disabled"
        and static_analysis.get("gradient_accumulation") == "single_microbatch_exact"
    )
    checkpointing_analysis = str(static_analysis.get("checkpointing", ""))
    accumulation_analysis = str(static_analysis.get("gradient_accumulation", ""))
    static_is_analytical = (
        checkpointing_analysis.startswith("analytical_")
        and accumulation_analysis
        in {"single_microbatch_exact", "in_place_largest_gradient_temporary"}
    ) or (
        checkpointing_analysis == "disabled"
        and accumulation_analysis == "in_place_largest_gradient_temporary"
    )
    static_graph_peak = int(
        static["memory_phases"].get(
            "accumulation_graph_estimated_peak_bytes",
            static["memory_phases"]["first_step_graph_phase_peak_bytes"],
        )
    )
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
        "static_first_step_graph_peak": _comparison(
            static_graph_peak,
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
    static_overall_comparison = comparisons["static_overall_peak"]
    static_graph_comparison = comparisons["static_first_step_graph_peak"]
    if static_is_exact or static_is_analytical:
        static_overall_passed = (
            static_overall_comparison["absolute_error_percent"]
            <= max_overall_error_percent
        )
        static_backward_passed = (
            static_graph_comparison["absolute_error_percent"]
            <= max_phase_error_percent
        )
    else:
        maximum_underestimate_bytes = (
            int(real["memory_phases"]["overall_peak_bytes"])
            * max_overall_error_percent
            / 100.0
        )
        static_overall_passed = (
            static_overall_comparison["signed_error"] >= -maximum_underestimate_bytes
            and static_overall_comparison["signed_error"]
            <= int(real["memory_phases"]["overall_peak_bytes"])
            * max_upper_bound_overestimate_percent
            / 100.0
        )
        static_backward_passed = (
            static_graph_comparison["signed_error"]
            >= -int(real["memory_phases"]["backward_peak_bytes"])
            * max_phase_error_percent
            / 100.0
            and static_graph_comparison["signed_error"]
            <= int(real["memory_phases"]["backward_peak_bytes"])
            * max_upper_bound_overestimate_percent
            / 100.0
        )
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
        "static_backward_peak_within_limit": static_backward_passed,
        "optimizer_peak_error_within_limit": (
            comparisons["fakecuda_optimizer_peak"]["absolute_error_percent"]
            <= max_phase_error_percent
        ),
        "fakecuda_overall_error_within_limit": (
            comparisons["fakecuda_overall_peak"]["absolute_error_percent"]
            <= max_overall_error_percent
        ),
        "static_overall_within_limit": static_overall_passed,
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
            "training_method": real.get("training_method", "full"),
            "gradient_checkpointing": bool(real.get("gradient_checkpointing", False)),
            "gradient_accumulation_steps": int(real.get("gradient_accumulation_steps", 1)),
            "lora": real.get("lora"),
            "fingerprint_sha256": next(iter(fingerprints)),
        },
        "limits": {
            "max_overall_error_percent": float(max_overall_error_percent),
            "max_phase_error_percent": float(max_phase_error_percent),
            "max_upper_bound_overestimate_percent": float(
                max_upper_bound_overestimate_percent
            ),
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
            "analysis": static_analysis,
            "prediction_kind": (
                "exact"
                if static_is_exact
                else "analytical"
                if static_is_analytical
                else "upper_bound"
            ),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    labels = {
        "model_load_allocator": "Model load: FakeCUDA vs real allocator",
        "fakecuda_forward_peak": "Forward peak: FakeCUDA vs real allocator",
        "fakecuda_backward_peak": "Backward peak: FakeCUDA vs real allocator",
        "static_first_step_graph_peak": "Backward peak: static first-step graph vs real allocator",
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
            f"- Training: `{workload['training_method']}`; gradient checkpointing: "
            f"`{workload['gradient_checkpointing']}`; accumulation steps: "
            f"`{workload['gradient_accumulation_steps']}`",
            f"- Static prediction: `{report['static']['prediction_kind']}`",
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
            "The FakeCUDA execution-only backward value is diagnostic: short-lived CPU-kernel "
            "temporaries are not all visible to its runtime tracker. The gating backward prediction "
            "comes from the captured ATen storage-liveness graph.",
            "Gradient-checkpointing graphs use an analytical retained-loss/optimizer floor when the "
            "loss operators are recognized; otherwise they remain explicit uncheckpointed upper bounds.",
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
    parser.add_argument("--max-upper-bound-overestimate-percent", type=float, default=25.0)
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
            max_upper_bound_overestimate_percent=args.max_upper_bound_overestimate_percent,
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
