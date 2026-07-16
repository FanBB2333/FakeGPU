from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ._api import env as fakegpu_env
from ._api import _warn_if_macos_injection_may_be_blocked
from ._version import __version__


STATUS_PASS_FIT = "PASS_FIT"
STATUS_FAIL_OOM = "FAIL_OOM"
STATUS_FAIL_RUNTIME = "FAIL_RUNTIME"
STATUS_WARN_INCOMPLETE = "WARN_INCOMPLETE_TRACKING"


@dataclass(frozen=True)
class PreflightPaths:
    report_dir: Path
    json_report: Path
    markdown_report: Path
    stdout_log: Path
    stderr_log: Path
    runtime_report: Path
    cluster_report: Path
    child_report: Path
    stage_log: Path


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    cmd = list(ns.command)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        parser.error("missing command to run after --")
    return run_preflight(ns, cmd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fakegpu preflight",
        description="Run an AI workload preflight and write fit/OOM reports.",
    )
    parser.add_argument(
        "--runtime",
        choices=["fakecuda", "native", "hybrid", "passthrough"],
        default="fakecuda",
        help="Runtime used for the preflight command.",
    )
    parser.add_argument("--devices", help="Device preset spec, e.g. 'a100:4,h100:4'.")
    parser.add_argument("--profile", help="GPU preset ID for all devices, e.g. a100 or h100.")
    parser.add_argument("--device-count", type=int, help="Number of logical devices to expose.")
    parser.add_argument(
        "--stage",
        default="completed",
        help="Target stage name, e.g. import, model_load, forward, backward, optimizer_step, n_steps.",
    )
    parser.add_argument("--steps", type=int, default=None, help="Target number of steps for n_steps preflights.")
    parser.add_argument("--report-dir", default="preflight-report", help="Directory for preflight outputs.")
    parser.add_argument("--strict", action="store_true", help="Treat incomplete tracking as a failing result.")
    parser.add_argument(
        "--allocation-stacks",
        action="store_true",
        help="Record short Python stack traces for largest fakecuda allocations.",
    )
    parser.add_argument(
        "--memory-safety-factor",
        type=float,
        default=None,
        help=(
            "Multiply tracked peak memory before fit/OOM classification. "
            "Use this with a factor from real-GPU calibration when fakecuda undercounts a workload family."
        ),
    )
    parser.add_argument(
        "--memory-safety-margin",
        default=None,
        help=(
            "Add a fixed byte margin before fit/OOM classification, e.g. 18089460, 18MiB, or 0.02GiB. "
            "Prefer this when calibration shows a mostly fixed backend workspace gap."
        ),
    )
    parser.add_argument(
        "--memory-calibration",
        help=(
            "Path to a repeated real-GPU calibration report or aggregated bundle. "
            "For an exact matching workload/profile, use its observed physical-memory upper bound."
        ),
    )
    parser.add_argument(
        "--calibration-workload",
        help="Exact workload name or workload signature to select from --memory-calibration.",
    )
    parser.add_argument("--build-dir", help="FakeGPU CMake build directory for native/hybrid/passthrough runtimes.")
    parser.add_argument("--lib-dir", help="Directory containing FakeGPU shared libraries.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run after --")
    return parser


def run_preflight(ns: argparse.Namespace, command: Sequence[str]) -> int:
    paths = _make_paths(Path(ns.report_dir))
    paths.report_dir.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    warnings: list[str] = []
    setup_error: Exception | None = None
    completed: subprocess.CompletedProcess[str] | None = None
    child_command = list(command)

    try:
        child_env = _build_child_env(ns, paths)
        child_command = _prepare_child_command(ns.runtime, command, warnings)
        if child_command:
            _warn_if_macos_injection_may_be_blocked(child_command[0])
        completed = subprocess.run(
            child_command,
            cwd=os.getcwd(),
            env=child_env,
            text=True,
            capture_output=True,
        )
        paths.stdout_log.write_text(completed.stdout or "", encoding="utf-8")
        paths.stderr_log.write_text(completed.stderr or "", encoding="utf-8")
    except Exception as exc:
        setup_error = exc
        paths.stdout_log.write_text("", encoding="utf-8")
        paths.stderr_log.write_text(str(exc) + "\n", encoding="utf-8")

    duration = time.monotonic() - start
    report = build_report(
        ns=ns,
        command=list(command),
        executed_command=child_command,
        paths=paths,
        completed=completed,
        setup_error=setup_error,
        warnings=warnings,
        duration_seconds=duration,
    )
    paths.json_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    paths.markdown_report.write_text(render_markdown_report(report), encoding="utf-8")
    print(f"preflight report: {paths.json_report}")
    print(f"preflight markdown: {paths.markdown_report}")
    return _exit_code_for_status(str(report["status"]), strict=bool(ns.strict))


def build_report(
    *,
    ns: argparse.Namespace,
    command: list[str],
    executed_command: list[str],
    paths: PreflightPaths,
    completed: subprocess.CompletedProcess[str] | None,
    setup_error: Exception | None,
    warnings: list[str],
    duration_seconds: float,
) -> dict[str, Any]:
    stdout = paths.stdout_log.read_text(encoding="utf-8") if paths.stdout_log.exists() else ""
    stderr = paths.stderr_log.read_text(encoding="utf-8") if paths.stderr_log.exists() else ""
    raw_report, raw_report_kind = _load_raw_runtime_report(paths)
    devices, tracking_confidence = _normalize_devices(raw_report, raw_report_kind)
    memory_safety_factor = _resolve_memory_safety_factor(ns)
    memory_safety_margin = _resolve_memory_safety_margin(ns)
    if memory_safety_factor > 1.0 or memory_safety_margin > 0:
        devices = _apply_memory_safety_adjustment(devices, memory_safety_factor, memory_safety_margin)
        if memory_safety_factor > 1.0:
            warnings.append(f"Applied memory safety factor {memory_safety_factor:.3f} to tracked peaks.")
        if memory_safety_margin > 0:
            warnings.append(f"Applied memory safety margin {_fmt_bytes(memory_safety_margin)} to tracked peaks.")

    calibration_gpu: dict[str, Any] | None = None
    memory_estimation: dict[str, Any] = {
        "method": (
            "manual_factor_or_margin"
            if memory_safety_factor > 1.0 or memory_safety_margin > 0
            else "tracked_peak"
        )
    }
    calibration_path = getattr(ns, "memory_calibration", None)
    calibration_workload = getattr(ns, "calibration_workload", None)
    if calibration_path:
        try:
            if not calibration_workload:
                raise ValueError("--memory-calibration requires --calibration-workload")
            devices, empirical = _apply_empirical_memory_calibration(
                devices,
                path=Path(calibration_path),
                workload_selector=str(calibration_workload),
            )
            memory_estimation = empirical
            calibration_gpu = empirical.get("calibration_gpu")
            matched = int(empirical.get("matched_device_count", 0) or 0)
            if devices and matched == len(devices):
                tracking_confidence = "C4_real_gpu_calibrated"
            warnings.extend(str(value) for value in empirical.get("warnings", []))
        except Exception as exc:
            if setup_error is None:
                setup_error = ValueError(f"empirical memory calibration failed: {exc}")
            warnings.append(f"Empirical memory calibration was not applied: {exc}")
    elif calibration_workload:
        warnings.append("--calibration-workload was ignored because --memory-calibration was not provided.")

    exit_code = completed.returncode if completed is not None else 1
    oom_detected = _looks_like_oom(stdout, stderr, raw_report)
    strict_skip_detected = bool(ns.strict) and _looks_like_skip(stdout, stderr)
    status = _classify_status(
        exit_code=exit_code,
        setup_error=setup_error,
        oom_detected=oom_detected,
        strict_skip_detected=strict_skip_detected,
        devices=devices,
        tracking_confidence=tracking_confidence,
    )
    if status == STATUS_WARN_INCOMPLETE:
        warnings.append("No runtime memory report was produced; fit/no-fit confidence is incomplete.")
    if strict_skip_detected:
        warnings.append("Strict mode detected skipped tests in the child command output.")

    errors = _collect_errors(
        status=status,
        setup_error=setup_error,
        stdout=stdout,
        stderr=stderr,
        raw_report=raw_report,
        strict_skip_detected=strict_skip_detected,
    )
    stage = _resolve_stage(
        raw_report=raw_report,
        stage_log=paths.stage_log,
        exit_code=exit_code,
        default_stage=str(ns.stage),
    )

    return {
        "schema_version": "preflight.v1",
        "fakegpu_version": __version__,
        "git_commit": _git_commit(Path(os.getcwd())),
        "command": command,
        "executed_command": executed_command,
        "cwd": os.getcwd(),
        "runtime": ns.runtime,
        "target": {
            "devices": ns.devices,
            "profile": ns.profile,
            "device_count": ns.device_count,
            "stage": ns.stage,
            "steps": ns.steps,
            "memory_safety_factor": memory_safety_factor,
            "memory_safety_margin_bytes": memory_safety_margin,
            "memory_calibration": str(calibration_path) if calibration_path else None,
            "calibration_workload": str(calibration_workload) if calibration_workload else None,
        },
        "target_profiles": _target_profiles(ns),
        "calibration_gpu": calibration_gpu,
        "status": status,
        "stage": stage,
        "exit_code": exit_code,
        "duration_seconds": round(duration_seconds, 3),
        "tracking_confidence": tracking_confidence,
        "memory_safety_factor": memory_safety_factor,
        "memory_safety_margin_bytes": memory_safety_margin,
        "memory_estimation": memory_estimation,
        "devices": devices,
        "warnings": warnings,
        "errors": errors,
        "logs": {
            "stdout": str(paths.stdout_log),
            "stderr": str(paths.stderr_log),
        },
        "raw_reports": {
            "runtime": str(paths.runtime_report) if paths.runtime_report.exists() else None,
            "cluster": str(paths.cluster_report) if paths.cluster_report.exists() else None,
            "fakecuda_child": str(paths.child_report) if paths.child_report.exists() else None,
            "stage_log": str(paths.stage_log) if paths.stage_log.exists() else None,
        },
    }


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


def _make_paths(report_dir: Path) -> PreflightPaths:
    resolved = report_dir.resolve()
    return PreflightPaths(
        report_dir=resolved,
        json_report=resolved / "preflight_report.json",
        markdown_report=resolved / "preflight_report.md",
        stdout_log=resolved / "preflight_stdout.log",
        stderr_log=resolved / "preflight_stderr.log",
        runtime_report=resolved / "fakegpu_runtime_report.json",
        cluster_report=resolved / "fakegpu_cluster_report.json",
        child_report=resolved / "fakegpu_child_report.json",
        stage_log=resolved / "preflight_stage.jsonl",
    )


def _build_child_env(ns: argparse.Namespace, paths: PreflightPaths) -> dict[str, str]:
    base = dict(os.environ)
    base["FAKEGPU_TERMINAL_REPORT"] = "1"
    base["FAKEGPU_REPORT_PATH"] = str(paths.runtime_report)
    base["FAKEGPU_CLUSTER_REPORT_PATH"] = str(paths.cluster_report)
    base["FAKEGPU_PREFLIGHT_CHILD_REPORT"] = str(paths.child_report)
    base["FAKEGPU_PREFLIGHT_STAGE_LOG"] = str(paths.stage_log)
    base["FAKEGPU_PREFLIGHT_TARGET_STAGE"] = str(ns.stage)
    if ns.allocation_stacks:
        base["FAKEGPU_ALLOCATION_STACKS"] = "1"
    if ns.steps is not None:
        base["FAKEGPU_PREFLIGHT_STEPS"] = str(int(ns.steps))
    effective_device_count = ns.device_count
    if effective_device_count is None and ns.devices:
        effective_device_count = _infer_device_count_from_devices(str(ns.devices))
    if effective_device_count is not None:
        base["FAKEGPU_DEVICE_COUNT"] = str(int(effective_device_count))
    if ns.profile:
        base["FAKEGPU_PROFILE"] = str(ns.profile)
        if not ns.devices:
            base.pop("FAKEGPU_PROFILES", None)
    if ns.devices:
        base["FAKEGPU_PROFILES"] = str(ns.devices)

    if ns.runtime == "fakecuda":
        return base

    mode = "simulate" if ns.runtime == "native" else str(ns.runtime)
    return fakegpu_env(
        build_dir=ns.build_dir,
        lib_dir=ns.lib_dir,
        mode=mode,
        profile=ns.profile,
        device_count=ns.device_count,
        devices=ns.devices,
        base_env=base,
    )


def _prepare_child_command(runtime: str, command: Sequence[str], warnings: list[str]) -> list[str]:
    cmd = list(command)
    if runtime != "fakecuda":
        return cmd
    if not cmd or not _looks_like_python(cmd[0]):
        warnings.append("fakecuda runtime can only auto-initialize Python commands; command was run without bootstrap.")
        return cmd
    python_options, target_args = _split_python_options(cmd[1:])
    return [cmd[0], *python_options, "-m", "fakegpu._preflight_bootstrap", "--", *target_args]


def _looks_like_python(executable: str) -> bool:
    name = Path(executable).name.lower()
    return name.startswith("python") or Path(executable).resolve() == Path(sys.executable).resolve()


def _split_python_options(args: Sequence[str]) -> tuple[list[str], list[str]]:
    passthrough: list[str] = []
    remaining = list(args)
    index = 0
    no_value_options = {"-B", "-E", "-I", "-O", "-OO", "-q", "-s", "-S", "-u"}
    value_options = {"-W", "-X"}
    while index < len(remaining):
        item = remaining[index]
        if item in no_value_options:
            passthrough.append(item)
            index += 1
            continue
        if item in value_options:
            if index + 1 >= len(remaining):
                break
            passthrough.extend([item, remaining[index + 1]])
            index += 2
            continue
        if any(item.startswith(prefix) and item != prefix for prefix in value_options):
            passthrough.append(item)
            index += 1
            continue
        break
    return passthrough, remaining[index:]


def _infer_device_count_from_devices(devices: str) -> int:
    total = 0
    for spec in devices.split(","):
        spec = spec.strip()
        if not spec:
            continue
        parts = spec.split(":", 1)
        if len(parts) == 2 and parts[1].strip().isdigit():
            total += int(parts[1].strip())
        else:
            total += 1
    return max(total, 1)


def _target_profiles(ns: argparse.Namespace) -> list[dict[str, int | str]]:
    if ns.devices:
        profiles: list[dict[str, int | str]] = []
        for spec in str(ns.devices).split(","):
            spec = spec.strip()
            if not spec:
                continue
            profile, _, count_text = spec.partition(":")
            count = int(count_text) if count_text.strip().isdigit() else 1
            profiles.append({"profile_id": profile.strip(), "count": count})
        return profiles

    if ns.profile:
        return [{"profile_id": str(ns.profile), "count": int(ns.device_count or 1)}]

    return []


def _load_raw_runtime_report(paths: PreflightPaths) -> tuple[dict[str, Any] | None, str | None]:
    for path, kind in (
        (paths.child_report, "fakecuda_child"),
        (paths.runtime_report, "native_runtime"),
    ):
        if not path.exists():
            continue
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(loaded, dict):
            return loaded, kind
    return None, None


def _normalize_devices(raw_report: dict[str, Any] | None, raw_report_kind: str | None) -> tuple[list[dict[str, Any]], str]:
    if raw_report is None:
        return [], "C0_incomplete"

    confidence = "C2_torch_tensor_lifetime" if raw_report_kind == "fakecuda_child" else "C3_native_cuda_allocations"
    devices: list[dict[str, Any]] = []
    for raw in raw_report.get("devices", []) or []:
        total = int(raw.get("total_memory", 0) or 0)
        peak = int(raw.get("peak_memory", raw.get("used_memory_peak", 0)) or 0)
        current = int(raw.get("current_memory", raw.get("used_memory_current", 0)) or 0)
        alloc = raw.get("alloc", {})
        allocation_count = int(raw.get("allocation_count", alloc.get("calls", 0) if isinstance(alloc, dict) else 0) or 0)
        headroom = total - peak
        headroom_percent = (100.0 * headroom / total) if total > 0 else None
        devices.append(
            {
                "index": int(raw.get("index", len(devices))),
                "name": str(raw.get("name", "")),
                "profile_id": raw.get("profile_id"),
                "total_memory": total,
                "peak_memory": peak,
                "current_memory": current,
                "headroom_bytes": headroom,
                "headroom_percent": round(headroom_percent, 3) if headroom_percent is not None else None,
                "allocation_count": allocation_count,
                "current_bytes_by_category": dict(raw.get("current_bytes_by_category", {}) or {}),
                "peak_by_stage": dict(raw.get("peak_by_stage", {}) or {}),
                "largest_allocations": list(raw.get("largest_allocations", []) or []),
                "tracking_confidence": confidence,
            }
        )
    return devices, confidence if devices else "C0_incomplete"


def _resolve_memory_safety_factor(ns: argparse.Namespace) -> float:
    raw_value = ns.memory_safety_factor
    if raw_value is None:
        env_value = os.environ.get("FAKEGPU_PREFLIGHT_MEMORY_SAFETY_FACTOR")
        if env_value:
            try:
                raw_value = float(env_value)
            except ValueError:
                raw_value = 1.0
    if raw_value is None:
        return 1.0
    return max(1.0, float(raw_value))


def _resolve_memory_safety_margin(ns: argparse.Namespace) -> int:
    raw_value = ns.memory_safety_margin
    if raw_value is None:
        raw_value = os.environ.get("FAKEGPU_PREFLIGHT_MEMORY_SAFETY_MARGIN")
    if raw_value is None or str(raw_value).strip() == "":
        return 0
    return max(0, _parse_byte_quantity(str(raw_value)))


def _parse_byte_quantity(value: str) -> int:
    text = value.strip()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]*)", text)
    if not match:
        raise argparse.ArgumentTypeError(f"invalid byte quantity: {value!r}")
    amount = float(match.group(1))
    unit = match.group(2).lower()
    multipliers = {
        "": 1,
        "b": 1,
        "byte": 1,
        "bytes": 1,
        "k": 1000,
        "kb": 1000,
        "m": 1000**2,
        "mb": 1000**2,
        "g": 1000**3,
        "gb": 1000**3,
        "kib": 1024,
        "ki": 1024,
        "mib": 1024**2,
        "mi": 1024**2,
        "gib": 1024**3,
        "gi": 1024**3,
    }
    if unit not in multipliers:
        raise argparse.ArgumentTypeError(f"invalid byte unit in {value!r}")
    return int(amount * multipliers[unit])


def _apply_memory_safety_adjustment(
    devices: list[dict[str, Any]],
    factor: float,
    margin_bytes: int,
) -> list[dict[str, Any]]:
    import math

    adjusted: list[dict[str, Any]] = []
    for dev in devices:
        item = dict(dev)
        tracked_peak = int(item.get("peak_memory", 0) or 0)
        estimated_peak = int(math.ceil(tracked_peak * factor)) + int(margin_bytes)
        total = int(item.get("total_memory", 0) or 0)
        headroom = total - estimated_peak
        headroom_percent = (100.0 * headroom / total) if total > 0 else None
        item["tracked_peak_memory"] = tracked_peak
        item["peak_memory"] = estimated_peak
        item["estimated_peak_memory"] = estimated_peak
        item["memory_safety_factor"] = round(factor, 6)
        item["memory_safety_margin_bytes"] = int(margin_bytes)
        item["headroom_bytes"] = headroom
        item["headroom_percent"] = round(headroom_percent, 3) if headroom_percent is not None else None
        adjusted.append(item)
    return adjusted


def _apply_empirical_memory_calibration(
    devices: list[dict[str, Any]],
    *,
    path: Path,
    workload_selector: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    calibration = _load_empirical_memory_calibration(path, workload_selector)
    candidates = calibration["candidates"]
    adjusted: list[dict[str, Any]] = []
    matched_profiles: list[str] = []
    warnings: list[str] = []
    calibration_gpus: list[dict[str, Any]] = []

    for dev in devices:
        item = dict(dev)
        profile = str(item.get("profile_id") or "")
        candidate = candidates.get(profile)
        if not isinstance(candidate, dict):
            warnings.append(
                f"No empirical calibration observation matched device {item.get('index', 0)} "
                f"with profile {profile or '<unknown>'}."
            )
            adjusted.append(item)
            continue

        tracked_peak = int(item.get("tracked_peak_memory", item.get("peak_memory", 0)) or 0)
        previous_estimate = int(item.get("peak_memory", 0) or 0)
        allocator_upper_bound = int(candidate["empirical_real_peak_upper_bound_bytes"])
        empirical_upper_bound = int(
            candidate.get("empirical_physical_peak_upper_bound_bytes", allocator_upper_bound)
        )
        empirical_source = str(
            candidate.get("empirical_physical_peak_upper_bound_source") or "torch_allocator_peak"
        )
        estimated_peak = max(previous_estimate, empirical_upper_bound)
        total = int(item.get("total_memory", 0) or 0)
        headroom = total - estimated_peak
        headroom_percent = (100.0 * headroom / total) if total > 0 else None
        item.update(
            {
                "tracked_peak_memory": tracked_peak,
                "peak_memory": estimated_peak,
                "estimated_peak_memory": estimated_peak,
                "memory_estimation_method": "empirical_repeated_upper_bound",
                "empirical_calibration_peak_memory": empirical_upper_bound,
                "empirical_allocator_peak_memory": allocator_upper_bound,
                "memory_calibration_metric": "physical_peak",
                "memory_calibration_metric_source": empirical_source,
                "memory_calibration_sample_count": int(candidate.get("sample_count", 0) or 0),
                "memory_calibration_workload_signature": calibration["workload_signature"],
                "memory_calibration_profile": profile,
                "headroom_bytes": headroom,
                "headroom_percent": round(headroom_percent, 3) if headroom_percent is not None else None,
            }
        )
        adjusted.append(item)
        matched_profiles.append(profile)
        for gpu in candidate.get("gpus", []):
            if isinstance(gpu, dict) and gpu not in calibration_gpus:
                calibration_gpus.append(gpu)
        warnings.append(
            "Applied empirical physical upper bound {peak} ({source}) from {samples} "
            "post-warmup real-GPU sample(s) for workload {workload} on profile {profile}.".format(
                peak=_fmt_bytes(empirical_upper_bound),
                source=empirical_source,
                samples=int(candidate.get("sample_count", 0) or 0),
                workload=calibration["workload_name"],
                profile=profile,
            )
        )

    calibration_gpu = {
        "method": "empirical_repeated_upper_bound",
        "metric": "physical_peak",
        "source": str(path.resolve()),
        "workload": calibration["workload_name"],
        "workload_signature": calibration["workload_signature"],
        "observations": calibration_gpus,
    }
    return adjusted, {
        "method": "empirical_repeated_upper_bound",
        "metric": "physical_peak",
        "source": str(path.resolve()),
        "source_schema_version": calibration["source_schema_version"],
        "workload": calibration["workload_name"],
        "workload_signature": calibration["workload_signature"],
        "matched_device_count": len(matched_profiles),
        "matched_profiles": matched_profiles,
        "calibration_gpu": calibration_gpu,
        "warnings": warnings,
    }


def _load_empirical_memory_calibration(
    path: Path,
    workload_selector: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"calibration file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("calibration file must contain a JSON object")
    schema = str(payload.get("schema_version") or "")
    if schema == "real_gpu_calibration_bundle.v1":
        return _load_empirical_bundle(payload, workload_selector)
    if schema in {"real_gpu_calibration.v1", "rtx3090ti_calibration.v1"}:
        return _load_empirical_report(payload, workload_selector)
    raise ValueError(f"unsupported calibration schema: {schema!r}")


def _load_empirical_bundle(payload: dict[str, Any], workload_selector: str) -> dict[str, Any]:
    workloads = payload.get("workloads")
    if not isinstance(workloads, list):
        raise ValueError("calibration bundle workloads must be a list")
    matches = [
        item
        for item in workloads
        if isinstance(item, dict)
        and (
            str(item.get("name") or "") == workload_selector
            or str(item.get("workload_signature") or "") == workload_selector
        )
    ]
    if not matches:
        raise ValueError(f"workload {workload_selector!r} was not found in the calibration bundle")
    if len(matches) > 1:
        signatures = ", ".join(str(item.get("workload_signature")) for item in matches)
        raise ValueError(
            f"workload name {workload_selector!r} has multiple signatures; select one of: {signatures}"
        )
    workload = matches[0]
    candidates: dict[str, dict[str, Any]] = {}
    observations = workload.get("observations")
    if not isinstance(observations, list):
        raise ValueError("calibration bundle workload observations must be a list")
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        profile = str(observation.get("profile") or "")
        allocator_upper_bound = int(observation.get("empirical_real_peak_upper_bound_bytes", 0) or 0)
        physical_upper_bound = int(
            observation.get("empirical_physical_peak_upper_bound_bytes", allocator_upper_bound) or 0
        )
        physical_source = str(
            observation.get("empirical_physical_peak_upper_bound_source") or "torch_allocator_peak"
        )
        if not profile or allocator_upper_bound <= 0 or physical_upper_bound <= 0:
            continue
        candidate = candidates.setdefault(
            profile,
            {
                "empirical_real_peak_upper_bound_bytes": 0,
                "empirical_physical_peak_upper_bound_bytes": 0,
                "empirical_physical_peak_upper_bound_source": "torch_allocator_peak",
                "sample_count": 0,
                "gpus": [],
            },
        )
        candidate["empirical_real_peak_upper_bound_bytes"] = max(
            int(candidate["empirical_real_peak_upper_bound_bytes"]),
            allocator_upper_bound,
        )
        if physical_upper_bound > int(candidate["empirical_physical_peak_upper_bound_bytes"]):
            candidate["empirical_physical_peak_upper_bound_bytes"] = physical_upper_bound
            candidate["empirical_physical_peak_upper_bound_source"] = physical_source
        candidate["sample_count"] = int(candidate["sample_count"]) + int(
            observation.get("sample_count", 0) or 0
        )
        gpu = observation.get("gpu")
        if isinstance(gpu, dict) and gpu not in candidate["gpus"]:
            candidate["gpus"].append(gpu)
    if not candidates:
        raise ValueError("calibration bundle has no usable profile observations")
    return {
        "source_schema_version": "real_gpu_calibration_bundle.v1",
        "workload_name": str(workload.get("name") or workload_selector),
        "workload_signature": str(workload.get("workload_signature") or ""),
        "candidates": candidates,
    }


def _load_empirical_report(payload: dict[str, Any], workload_selector: str) -> dict[str, Any]:
    if payload.get("status") != "PASS_CALIBRATED":
        raise ValueError(f"calibration report status is {payload.get('status')!r}, not PASS_CALIBRATED")
    workloads = payload.get("workloads")
    if not isinstance(workloads, list):
        raise ValueError("calibration report workloads must be a list")
    matches = [
        item
        for item in workloads
        if isinstance(item, dict)
        and (
            str(item.get("name") or "") == workload_selector
            or str(item.get("workload_signature") or "") == workload_selector
        )
    ]
    if not matches:
        raise ValueError(f"workload {workload_selector!r} was not found in the calibration report")
    if len(matches) > 1:
        raise ValueError(f"calibration report contains duplicate workload selector {workload_selector!r}")
    workload = matches[0]
    real = workload.get("real_cuda")
    if not isinstance(real, dict):
        raise ValueError("calibration workload has no real_cuda measurement")
    trials = real.get("trials")
    if isinstance(trials, list):
        peaks = [
            int(item.get("peak_memory", 0) or 0)
            for item in trials
            if isinstance(item, dict) and int(item.get("peak_memory", 0) or 0) > 0
        ]
    else:
        peak = int(real.get("peak_memory", 0) or 0)
        peaks = [peak] if peak > 0 else []
    if not peaks:
        raise ValueError("calibration workload has no positive real-CUDA peak samples")
    allocator_upper_bound = max(peaks)
    physical_upper_bound = int(
        workload.get("empirical_physical_peak_upper_bound_bytes", 0) or 0
    )
    physical_source = str(
        workload.get("empirical_physical_peak_upper_bound_source") or ""
    )
    if physical_upper_bound <= 0:
        process_peaks = _positive_nvml_trial_values(real, "peak_process_memory")
        device_deltas = _positive_nvml_trial_values(real, "peak_device_used_delta_memory")
        if process_peaks:
            physical_upper_bound = max(allocator_upper_bound, max(process_peaks))
            physical_source = "nvml_process_peak"
        elif device_deltas:
            physical_upper_bound = max(allocator_upper_bound, max(device_deltas))
            physical_source = "torch_allocator_peak_with_nvml_device_delta"
        else:
            physical_upper_bound = allocator_upper_bound
            physical_source = "torch_allocator_peak"
    profile = str(payload.get("fakecuda_profile") or "")
    if not profile:
        raise ValueError("calibration report has no fakecuda_profile")
    signature = str(workload.get("workload_signature") or real.get("workload_signature") or "")
    if not signature:
        raise ValueError("calibration workload has no workload_signature")
    gpu = payload.get("calibration_gpu")
    return {
        "source_schema_version": str(payload.get("schema_version")),
        "workload_name": str(workload.get("name") or workload_selector),
        "workload_signature": signature,
        "candidates": {
            profile: {
                "empirical_real_peak_upper_bound_bytes": allocator_upper_bound,
                "empirical_physical_peak_upper_bound_bytes": physical_upper_bound,
                "empirical_physical_peak_upper_bound_source": physical_source,
                "sample_count": len(peaks),
                "gpus": [gpu] if isinstance(gpu, dict) else [],
            }
        },
    }


def _positive_nvml_trial_values(measurement: dict[str, Any], metric: str) -> list[int]:
    trials = measurement.get("trials")
    if isinstance(trials, list):
        return [
            int(nvml.get(metric, 0) or 0)
            for item in trials
            if isinstance(item, dict)
            and isinstance((nvml := item.get("nvml")), dict)
            and nvml.get("status") == "available"
            and int(nvml.get(metric, 0) or 0) > 0
        ]
    nvml = measurement.get("nvml")
    if isinstance(nvml, dict) and int(nvml.get(metric, 0) or 0) > 0:
        return [int(nvml[metric])]
    return []


def _classify_status(
    *,
    exit_code: int,
    setup_error: Exception | None,
    oom_detected: bool,
    strict_skip_detected: bool,
    devices: list[dict[str, Any]],
    tracking_confidence: str,
) -> str:
    if setup_error is not None:
        return STATUS_FAIL_RUNTIME
    if strict_skip_detected:
        return STATUS_FAIL_RUNTIME
    if exit_code != 0:
        return STATUS_FAIL_OOM if oom_detected else STATUS_FAIL_RUNTIME
    if any(int(dev.get("headroom_bytes", 0)) < 0 for dev in devices):
        return STATUS_FAIL_OOM
    if tracking_confidence == "C0_incomplete":
        return STATUS_WARN_INCOMPLETE
    return STATUS_PASS_FIT


def _looks_like_oom(stdout: str, stderr: str, raw_report: dict[str, Any] | None) -> bool:
    haystack = "\n".join([stdout, stderr]).lower()
    if _contains_oom_marker(haystack):
        return True
    exc = raw_report.get("exception") if isinstance(raw_report, dict) else None
    if isinstance(exc, dict):
        exc_text = f"{exc.get('type', '')}\n{exc.get('message', '')}".lower()
        return _contains_oom_marker(exc_text)
    return False


def _contains_oom_marker(text: str) -> bool:
    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in (
            r"\bout\s+of\s+memory\b",
            r"\b(?:cuda)?outofmemory(?:error)?\b",
            r"\bcuda\s+oom\b",
            r"\boom\b",
        )
    )


def _looks_like_skip(stdout: str, stderr: str) -> bool:
    haystack = "\n".join([stdout, stderr])
    patterns = [
        r"\b\d+\s+skipped\b",
        r"\bskipped\s+in\s+\d",
        r"\bSKIPPED\b",
        r"\bSkipped:\s",
    ]
    return any(re.search(pattern, haystack) for pattern in patterns)


def _collect_errors(
    *,
    status: str,
    setup_error: Exception | None,
    stdout: str,
    stderr: str,
    raw_report: dict[str, Any] | None,
    strict_skip_detected: bool,
) -> list[dict[str, str]]:
    if setup_error is not None:
        return [{"type": type(setup_error).__name__, "message": str(setup_error)}]
    if strict_skip_detected:
        return [
            {
                "type": "SkippedTest",
                "message": "Strict preflight treats skipped child tests as runtime failure.",
            }
        ]
    if status not in {STATUS_FAIL_OOM, STATUS_FAIL_RUNTIME}:
        return []

    exc = raw_report.get("exception") if isinstance(raw_report, dict) else None
    if isinstance(exc, dict):
        return [
            {
                "type": str(exc.get("type", "RuntimeError")),
                "message": str(exc.get("message", "")),
            }
        ]

    message = _extract_relevant_error_line(stderr) or _extract_relevant_error_line(stdout)
    return [{"type": "RuntimeError", "message": message or f"process exited with status {status}"}]


def _extract_relevant_error_line(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if _contains_oom_marker(line):
            return line
    return lines[-1] if lines else None


def _resolve_stage(
    *,
    raw_report: dict[str, Any] | None,
    stage_log: Path,
    exit_code: int,
    default_stage: str,
) -> str:
    if isinstance(raw_report, dict) and raw_report.get("stage"):
        return str(raw_report["stage"])

    last_stage = _last_stage_from_log(stage_log)
    if last_stage:
        return last_stage

    return default_stage if exit_code == 0 else "unknown_or_last_seen"


def _last_stage_from_log(path: Path) -> str | None:
    if not path.exists():
        return None
    last: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except Exception:
            continue
        if isinstance(event, dict) and event.get("stage"):
            last = str(event["stage"])
    return last


def _exit_code_for_status(status: str, *, strict: bool) -> int:
    if status == STATUS_PASS_FIT:
        return 0
    if status == STATUS_WARN_INCOMPLETE:
        return 3 if strict else 0
    if status == STATUS_FAIL_OOM:
        return 2
    return 1


def _git_commit(cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd),
            text=True,
            capture_output=True,
            check=False,
            timeout=2,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


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
