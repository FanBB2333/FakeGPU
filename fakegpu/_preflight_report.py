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
    status = report.get("status", "UNKNOWN")
    stage = report.get("stage", "unknown")
    runtime = report.get("runtime", "unknown")
    has_safety_factor = any("tracked_peak_memory" in dev for dev in report.get("devices", []) if isinstance(dev, dict))
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
        f"**Tracking confidence:** `{report.get('tracking_confidence', 'unknown')}`",
        "",
        "## Summary",
        "",
        _summary_sentence(report),
        "",
        "## Command",
        "",
        "```bash",
        shlex.join([str(part) for part in report.get("command", [])]),
        "```",
        "",
        "## Device Memory",
        "",
        device_header,
        device_rule,
    ]
    for dev in report.get("devices", []):
        if has_safety_factor:
            lines.append(
                "| {index} | {name} | {tracked} | {estimated} | {total} | {headroom} | {allocs} |".format(
                    index=dev.get("index", 0),
                    name=dev.get("name", ""),
                    tracked=_fmt_bytes(int(dev.get("tracked_peak_memory", dev.get("peak_memory", 0)))),
                    estimated=_fmt_bytes(int(dev.get("peak_memory", 0))),
                    total=_fmt_bytes(int(dev.get("total_memory", 0))),
                    headroom=_fmt_bytes(int(dev.get("headroom_bytes", 0))),
                    allocs=int(dev.get("allocation_count", 0)),
                )
            )
        else:
            lines.append(
                "| {index} | {name} | {peak} | {total} | {headroom} | {allocs} |".format(
                    index=dev.get("index", 0),
                    name=dev.get("name", ""),
                    peak=_fmt_bytes(int(dev.get("peak_memory", 0))),
                    total=_fmt_bytes(int(dev.get("total_memory", 0))),
                    headroom=_fmt_bytes(int(dev.get("headroom_bytes", 0))),
                    allocs=int(dev.get("allocation_count", 0)),
                )
            )

    memory_estimation = report.get("memory_estimation")
    if isinstance(memory_estimation, dict) and memory_estimation.get("method") == "empirical_repeated_upper_bound":
        lines.extend(
            [
                "",
                "## Empirical Memory Calibration",
                "",
                f"- source: `{memory_estimation.get('source')}`",
                f"- workload: `{memory_estimation.get('workload')}`",
                f"- workload signature: `{memory_estimation.get('workload_signature')}`",
                f"- metric: `{memory_estimation.get('metric')}`",
                "- metric sources: `{}`".format(
                    ", ".join(
                        sorted(
                            {
                                str(dev.get("memory_calibration_metric_source"))
                                for dev in report.get("devices", [])
                                if isinstance(dev, dict) and dev.get("memory_calibration_metric_source")
                            }
                        )
                    )
                ),
                f"- matched profiles: `{', '.join(memory_estimation.get('matched_profiles', []))}`",
                f"- matched devices: `{memory_estimation.get('matched_device_count')}`",
            ]
        )

    stage_rows: list[str] = []
    allocation_rows: list[str] = []
    allocation_has_stack = any(
        alloc.get("stack")
        for dev in report.get("devices", [])
        for alloc in (dev.get("largest_allocations", []) or [])
    )
    for dev in report.get("devices", []):
        dev_index = int(dev.get("index", 0))
        for stage, peak in sorted((dev.get("peak_by_stage") or {}).items()):
            stage_rows.append(f"| {dev_index} | `{stage}` | {_fmt_bytes(int(peak))} |")
        for alloc in dev.get("largest_allocations", []) or []:
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
    for dev in report.get("devices", []):
        dev_index = int(dev.get("index", 0))
        categories = dev.get("current_bytes_by_category") or {}
        for category, size in sorted(categories.items()):
            category_rows.append(f"| {dev_index} | `{category}` | {_fmt_bytes(int(size))} |")

    if report.get("devices"):
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

    errors = report.get("errors", [])
    if errors:
        lines.extend(["", "## Failure Reason", ""])
        for error in errors:
            lines.append(f"- `{error.get('type', 'Error')}`: {error.get('message', '')}")

    warnings = report.get("warnings", [])
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")

    lines.extend(
        [
            "",
            "## Confidence",
            "",
            _confidence_sentence(str(report.get("tracking_confidence", "unknown"))),
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
            f"- stdout: `{report.get('logs', {}).get('stdout')}`",
            f"- stderr: `{report.get('logs', {}).get('stderr')}`",
            "",
        ]
    )
    return "\n".join(lines)


def _summary_sentence(report: dict[str, Any]) -> str:
    status = str(report.get("status", "UNKNOWN"))
    stage = str(report.get("stage", "unknown"))
    confidence = str(report.get("tracking_confidence", "unknown"))
    peak = _total_peak_memory(report)
    headroom = _min_headroom(report)
    target = _target_profile_text(report)
    factor = float(report.get("memory_safety_factor", 1.0) or 1.0)
    margin = int(report.get("memory_safety_margin_bytes", 0) or 0)
    estimation = report.get("memory_estimation")
    empirical = isinstance(estimation, dict) and estimation.get("method") == "empirical_repeated_upper_bound"
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
    status = str(report.get("status", "UNKNOWN"))
    confidence = str(report.get("tracking_confidence", "unknown"))
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


def _target_profile_text(report: dict[str, Any]) -> str:
    profiles = report.get("target_profiles")
    if not isinstance(profiles, list) or not profiles:
        return "the selected target profile"
    parts: list[str] = []
    for item in profiles:
        if not isinstance(item, dict):
            continue
        profile_id = item.get("profile_id", "unknown")
        count = item.get("count", 1)
        parts.append(f"{profile_id} x {count}")
    return ", ".join(parts) or "the selected target profile"


def _total_peak_memory(report: dict[str, Any]) -> int:
    total = 0
    for dev in report.get("devices", []) or []:
        if isinstance(dev, dict):
            total += int(dev.get("peak_memory", 0) or 0)
    return total


def _min_headroom(report: dict[str, Any]) -> int:
    headrooms: list[int] = []
    for dev in report.get("devices", []) or []:
        if isinstance(dev, dict):
            headrooms.append(int(dev.get("headroom_bytes", 0) or 0))
    return min(headrooms) if headrooms else 0


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
