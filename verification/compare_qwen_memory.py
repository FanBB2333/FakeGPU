#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "fakegpu.qwen_memory_comparison.v1"


def compare_reports(real: dict[str, Any], fake: dict[str, Any]) -> dict[str, Any]:
    _require_success(real, "real")
    _require_success(fake, "fakecuda")
    for field in ("model_dir", "dtype", "attention_implementation", "prompt_tokens"):
        if real.get(field) != fake.get(field):
            raise ValueError(f"real/fake {field} mismatch")

    real_load = int(real["memory_phases"]["model_load_current_bytes"])
    fake_load = int(fake["memory_phases"]["model_load_current_bytes"])
    real_inference_peak = int(real["memory_phases"]["inference_peak_bytes"])
    fake_inference_peak = int(fake["memory_phases"]["inference_peak_bytes"])
    static_peak = int(fake["static_estimate"]["memory"]["estimated_tensor_peak_bytes"])
    real_flops = sum(int(value) for value in real["measured_matmul_flops_by_step"])
    fake_flops = sum(int(value) for value in fake["measured_matmul_flops_by_step"])
    estimated_flops = int(fake["static_estimate"]["compute"]["total_flops"])

    real_inference = _timeline_entry(real, "after_inference")
    fake_inference = _timeline_entry(fake, "after_inference")
    nvml_process = int((real_inference.get("nvml") or {}).get("process_memory") or 0)
    if nvml_process <= 0:
        raise ValueError("real report does not contain NVML process memory")
    real_inference_reserved = int(
        real_inference.get("reserved_bytes", real_inference["allocated_bytes"])
    )
    fake_inference_reserved = int(
        fake_inference.get("reserved_bytes", fake_inference["allocated_bytes"])
    )
    runtime_overhead = max(0, nvml_process - real_inference_reserved)
    simulated_process = fake_inference_reserved + runtime_overhead

    comparisons = {
        "model_load_allocator": _comparison(fake_load, real_load),
        "inference_allocator_peak": _comparison(
            fake_inference_peak, real_inference_peak
        ),
        "static_inference_peak": _comparison(static_peak, real_inference_peak),
        "virtual_smi_process": _comparison(simulated_process, nvml_process),
        "fakecuda_flops": _comparison(fake_flops, real_flops),
        "static_flops": _comparison(estimated_flops, real_flops),
    }
    checks = {
        "parameter_count_exact": (
            int(real["parameters"]["parameter_count"])
            == int(fake["parameters"]["parameter_count"])
            == int(fake["static_estimate"]["checkpoint"]["parameter_count"])
        ),
        "parameter_bytes_exact": (
            int(real["parameters"]["parameter_bytes"])
            == int(fake["parameters"]["parameter_bytes"])
            == int(fake["static_estimate"]["memory"]["parameter_bytes"])
        ),
        "generated_tokens_match": real.get("generated_tokens")
        == fake.get("generated_tokens"),
        "model_load_error_below_1_percent": comparisons["model_load_allocator"][
            "absolute_error_percent"
        ]
        < 1.0,
        "inference_peak_error_below_1_percent": comparisons["inference_allocator_peak"][
            "absolute_error_percent"
        ]
        < 1.0,
        "static_peak_error_below_1_percent": comparisons["static_inference_peak"][
            "absolute_error_percent"
        ]
        < 1.0,
        "virtual_smi_error_below_1_percent": comparisons["virtual_smi_process"][
            "absolute_error_percent"
        ]
        < 1.0,
        "static_flop_error_below_0_01_percent": comparisons["static_flops"][
            "absolute_error_percent"
        ]
        < 0.01,
        "fakecuda_flops_match_real": fake_flops == real_flops,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "success" if all(checks.values()) else "failed",
        "model": {
            "model_dir": real["model_dir"],
            "parameter_count": int(real["parameters"]["parameter_count"]),
            "parameter_bytes": int(real["parameters"]["parameter_bytes"]),
            "dtype": real["dtype"],
            "prompt_tokens": int(real["prompt_tokens"]),
            "generated_tokens": list(real.get("generated_tokens") or []),
        },
        "calibration": {
            "nvml_runtime_overhead_bytes": runtime_overhead,
            "method": "real_nvml_process_minus_real_allocator_reserved_after_inference",
        },
        "comparisons": comparisons,
        "checks": checks,
        "real": {
            "model_load_allocator_bytes": real_load,
            "inference_allocator_peak_bytes": real_inference_peak,
            "inference_allocator_reserved_bytes": real_inference_reserved,
            "inference_nvml_process_bytes": nvml_process,
            "matrix_flops": real_flops,
        },
        "fakecuda": {
            "model_load_tracked_bytes": fake_load,
            "inference_tracked_peak_bytes": fake_inference_peak,
            "inference_reserved_bytes": fake_inference_reserved,
            "virtual_smi_process_bytes": simulated_process,
            "matrix_flops": fake_flops,
        },
        "static": {
            "inference_peak_bytes": static_peak,
            "matrix_flops": estimated_flops,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    rows = []
    labels = {
        "model_load_allocator": "Model load: FakeCUDA tracked vs real allocator",
        "inference_allocator_peak": "Inference peak: FakeCUDA tracked vs real allocator",
        "static_inference_peak": "Inference peak: static estimate vs real allocator",
        "virtual_smi_process": "Process memory: virtual SMI vs real NVML",
        "fakecuda_flops": "Matrix FLOPs: FakeCUDA execution vs real CUDA",
        "static_flops": "Matrix FLOPs: static estimate vs real CUDA",
    }
    for key, label in labels.items():
        item = report["comparisons"][key]
        rows.append(
            f"| {label} | {item['predicted']:,} | {item['observed']:,} | "
            f"{item['signed_error']:+,} | {item['absolute_error_percent']:.6f}% |"
        )
    checks = [
        f"- [{'x' if passed else ' '}] `{name}`"
        for name, passed in report["checks"].items()
    ]
    model = report["model"]
    return "\n".join(
        [
            "# Qwen Inference Memory Validation",
            "",
            f"**Status:** `{report['status']}`",
            "",
            f"- Model: `{model['model_dir']}`",
            f"- Parameters: `{model['parameter_count']:,}`",
            f"- Parameter bytes: `{model['parameter_bytes']:,}`",
            f"- Prompt tokens: `{model['prompt_tokens']}`",
            f"- Generated token IDs: `{model['generated_tokens']}`",
            "",
            "| Comparison | Predicted | Observed | Signed error | Absolute error |",
            "|---|---:|---:|---:|---:|",
            *rows,
            "",
            "## Checks",
            "",
            *checks,
            "",
            "The virtual-SMI comparison adds the runtime overhead measured in the same real-CUDA run. It is an empirical calibration for this GPU/software/model path, not a portable constant.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True)
    parser.add_argument("--fakecuda", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown")
    args = parser.parse_args(argv)
    try:
        real = json.loads(Path(args.real).read_text(encoding="utf-8"))
        fake = json.loads(Path(args.fakecuda).read_text(encoding="utf-8"))
        report = compare_reports(real, fake)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"Qwen memory comparison failed: {exc}", file=sys.stderr)
        return 2
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown = (
        Path(args.markdown).expanduser().resolve()
        if args.markdown
        else output.with_suffix(".md")
    )
    markdown.write_text(render_markdown(report), encoding="utf-8")
    print(f"Qwen memory comparison: {report['status']}")
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
        "absolute_error_percent": round(100.0 * abs(signed) / observed, 9)
        if observed
        else 0.0,
    }


def _timeline_entry(report: dict[str, Any], label: str) -> dict[str, Any]:
    for item in report.get("timeline") or []:
        if item.get("label") == label:
            return item
    raise ValueError(f"timeline entry not found: {label}")


def _require_success(report: dict[str, Any], mode: str) -> None:
    if report.get("status") != "success" or report.get("mode") != mode:
        raise ValueError(f"expected a successful {mode} worker report")


if __name__ == "__main__":
    raise SystemExit(main())
