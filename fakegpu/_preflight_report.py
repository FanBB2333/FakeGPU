"""Markdown rendering for FakeGPU preflight reports.

This module is deliberately independent from the subprocess runner so the report
format can evolve and be tested without importing runtime setup code.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any


STATUS_PASS_FIT = "PASS_FIT"
STATUS_FAIL_OOM = "FAIL_OOM"
STATUS_FAIL_RUNTIME = "FAIL_RUNTIME"
STATUS_WARN_INCOMPLETE = "WARN_INCOMPLETE_TRACKING"


def _fmt_bytes(value: int) -> str:
    sign = "-" if value < 0 else ""
    b = abs(int(value))
    if b >= 1024**3:
        return f"{sign}{b / 1024**3:.2f} GiB"
    if b >= 1024**2:
        return f"{sign}{b / 1024**2:.2f} MiB"
    if b >= 1024:
        return f"{sign}{b / 1024:.2f} KiB"
    return f"{sign}{b} B"


def render_markdown_report(report: dict[str, Any]) -> str:
    status = report["status"]
    stage = report["stage"]
    runtime = report["runtime"]
    devices = report["devices"]
    has_safety_factor = any("tracked_peak_memory" in dev for dev in devices)
    device_header = "| GPU | Name | Peak | Total | Headroom | Allocations |"
    device_rule = "|---:|---|---:|---:|---:|---:|"
    if has_safety_factor:
        device_header = "| GPU | Name | Tracked Peak | Estimated Peak | Total | Headroom | Allocations |"
        device_rule = "|---:|---|---:|---:|---:|---:|---:|"

    lines = [
        "# FakeGPU Preflight Report",
        "",
        f"**Status:** `{status}`",
        f"**Runtime:** `{runtime}`",
        f"**Stage:** `{stage}`",
        f"**Tracking confidence:** `{report['tracking_confidence']}`",
        "",
        "## Summary",
        "",
        _summary_sentence(report),
        "",
        "## Command",
        "",
        "```bash",
        shlex.join(report["command"]),
        "```",
        "",
        "## Device Memory",
        "",
        device_header,
        device_rule,
    ]
    for dev in devices:
        cells = [str(dev["index"]), dev["name"]]
        if has_safety_factor:
            cells.append(_fmt_bytes(dev.get("tracked_peak_memory", dev["peak_memory"])))
        cells.extend(
            [
                _fmt_bytes(dev["peak_memory"]),
                _fmt_bytes(dev["total_memory"]),
                _fmt_bytes(dev["headroom_bytes"]),
                str(dev["allocation_count"]),
            ]
        )
        lines.append(f"| {' | '.join(cells)} |")

    memory_estimation = report["memory_estimation"]
    if memory_estimation["method"] == "empirical_repeated_upper_bound":
        lines.extend(
            [
                "",
                "## Empirical Memory Calibration",
                "",
                f"- source: `{memory_estimation['source']}`",
                f"- workload: `{memory_estimation['workload']}`",
                f"- workload signature: `{memory_estimation['workload_signature']}`",
                f"- metric: `{memory_estimation['metric']}`",
                "- metric sources: `{}`".format(
                    ", ".join(
                        sorted(
                            {
                                str(dev.get("memory_calibration_metric_source"))
                                for dev in devices
                                if dev.get("memory_calibration_metric_source")
                            }
                        )
                    )
                ),
                f"- matched profiles: `{', '.join(memory_estimation['matched_profiles'])}`",
                f"- matched devices: `{memory_estimation['matched_device_count']}`",
            ]
        )

    stage_rows: list[str] = []
    allocation_rows: list[str] = []
    allocation_has_stack = any(
        alloc.get("stack")
        for dev in devices
        for alloc in dev["largest_allocations"]
    )
    for dev in devices:
        dev_index = dev["index"]
        for stage, peak in sorted(dev["peak_by_stage"].items()):
            stage_rows.append(f"| {dev_index} | `{stage}` | {_fmt_bytes(peak)} |")
        for alloc in dev["largest_allocations"]:
            origin = _format_stack_origin(alloc.get("stack")) if allocation_has_stack else ""
            row = "| {device} | {size} | `{dtype}` | `{shape}` | `{stage}` | `{category}` |".format(
                device=int(alloc.get("device", dev_index)),
                size=_fmt_bytes(int(alloc.get("bytes", 0))),
                dtype=alloc.get("dtype"),
                shape=alloc.get("shape"),
                stage=alloc.get("stage"),
                category=alloc.get("category"),
            )
            if allocation_has_stack:
                row = row[:-1] + f" `{origin}` |"
            allocation_rows.append(
                row
            )

    if stage_rows:
        lines.extend(
            [
                "",
                "## Stage Peaks",
                "",
                "| GPU | Stage | Peak |",
                "|---:|---|---:|",
                *stage_rows,
            ]
        )

    category_rows: list[str] = []
    for dev in devices:
        for category, size in sorted(dev["current_bytes_by_category"].items()):
            category_rows.append(f"| {dev['index']} | `{category}` | {_fmt_bytes(int(size))} |")

    if devices:
        if not category_rows:
            category_rows.append("| 0 | `_none_live` | 0 B |")
        lines.extend(
            [
                "",
                "## Current Memory By Category",
                "",
                "| GPU | Category | Current |",
                "|---:|---|---:|",
                *category_rows,
            ]
        )

    if allocation_rows:
        allocation_header = "| GPU | Size | Dtype | Shape | Stage | Category |"
        allocation_rule = "|---:|---:|---|---|---|---|"
        if allocation_has_stack:
            allocation_header = "| GPU | Size | Dtype | Shape | Stage | Category | Origin |"
            allocation_rule = "|---:|---:|---|---|---|---|---|"
        lines.extend(
            [
                "",
                "## Largest Allocations",
                "",
                allocation_header,
                allocation_rule,
                *allocation_rows,
            ]
        )

    errors = report["errors"]
    if errors:
        lines.extend(["", "## Failure Reason", ""])
        for error in errors:
            lines.append(f"- `{error['type']}`: {error['message']}")

    warnings = report["warnings"]
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")

    lines.extend(
        [
            "",
            "## Confidence",
            "",
            _confidence_sentence(report["tracking_confidence"]),
            "",
            "## Suggested Next Steps",
            "",
            *_next_steps(report),
        ]
    )

    lines.extend(
        [
            "",
            "## Logs",
            "",
            f"- stdout: `{report['logs']['stdout']}`",
            f"- stderr: `{report['logs']['stderr']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _summary_sentence(report: dict[str, Any]) -> str:
    status = report["status"]
    stage = report["stage"]
    confidence = report["tracking_confidence"]
    peak = sum(dev["peak_memory"] for dev in report["devices"])
    headroom = min((dev["headroom_bytes"] for dev in report["devices"]), default=0)
    target = (
        ", ".join(
            f"{item['profile_id']} x {item['count']}"
            for item in report["target_profiles"]
        )
        or "the selected target profile"
    )
    factor = report["memory_safety_factor"]
    margin = report["memory_safety_margin_bytes"]
    empirical = report["memory_estimation"]["method"] == "empirical_repeated_upper_bound"
    peak_label = "estimated peak memory" if empirical or factor > 1.0 or margin > 0 else "peak tracked memory"

    if status == STATUS_PASS_FIT:
        return (
            f"The command completed `{stage}` without tracked OOM on {target}; "
            f"{peak_label} was {_fmt_bytes(peak)} with minimum headroom {_fmt_bytes(headroom)} "
            f"at `{confidence}` confidence."
        )
    if status == STATUS_FAIL_OOM:
        return (
            f"The command reached `{stage}` and failed with tracked OOM on {target}; "
            f"{peak_label} was {_fmt_bytes(peak)} at `{confidence}` confidence."
        )
    if status == STATUS_WARN_INCOMPLETE:
        return (
            f"The command completed, but tracking was incomplete at `{confidence}` confidence; "
            "treat fit/no-fit as unresolved."
        )
    return f"The command failed before a reliable fit/no-fit result was produced at `{confidence}` confidence."


def _confidence_sentence(confidence: str) -> str:
    descriptions = {
        "C0_incomplete": "C0 means no usable runtime memory report was produced.",
        "C1_weight_storage": "C1 mainly covers weights and explicit storage.",
        "C2_torch_tensor_lifetime": "C2 tracks torch-level tensor lifetimes and is suitable for fakecuda preflight decisions.",
        "C3_torch_dispatch_lifetime": "C3 tracks operator outputs and storage aliases at PyTorch dispatch boundaries.",
        "C3_native_cuda_allocations": "C3 tracks native CUDA allocation events.",
        "C4_real_gpu_calibrated": "C4 means the result has been calibrated against a real GPU run.",
    }
    return descriptions.get(confidence, f"`{confidence}` is not a recognized confidence level.")


def _next_steps(report: dict[str, Any]) -> list[str]:
    status = report["status"]
    confidence = report["tracking_confidence"]
    steps: list[str] = []

    if status == STATUS_PASS_FIT:
        steps.append("- Repeat with the target production profile if this run used a small profile.")
        steps.append("- Attach `preflight_report.json` and `preflight_report.md` to the Slurm submission notes.")
        if confidence != "C4_real_gpu_calibrated":
            steps.append("- For high-risk jobs, calibrate a reduced workload on the available real GPU before cluster submission.")
    elif status == STATUS_FAIL_OOM:
        steps.append("- Reduce batch size, sequence length, activation checkpoint scope, or optimizer state footprint.")
        steps.append("- Re-run with `--allocation-stacks` to locate the largest allocations in user code.")
        steps.append("- Repeat the same command with the intended cluster GPU profile after memory changes.")
    elif status == STATUS_WARN_INCOMPLETE:
        steps.append("- Re-run under `--runtime fakecuda` or enable a runtime that produces a memory report.")
        steps.append("- Treat this report as control-flow evidence, not a fit/no-fit decision.")
    else:
        steps.append("- Inspect `preflight_stderr.log` and fix runtime or dependency errors before memory tuning.")
        steps.append("- Re-run preflight after the command reaches the target stage.")

    return steps


def _format_stack_origin(stack: Any) -> str:
    if not isinstance(stack, list) or not stack:
        return ""
    frame = stack[-1]
    if not isinstance(frame, dict):
        return ""
    file_name = Path(str(frame.get("file", ""))).name
    line = frame.get("line", "")
    function = frame.get("function", "")
    origin = f"{file_name}:{line} {function}".strip()
    return origin.replace("|", "\\|").replace("`", "'")
