from __future__ import annotations

import argparse
import json
import os
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

    exit_code = completed.returncode if completed is not None else 1
    oom_detected = _looks_like_oom(stdout, stderr, raw_report)
    status = _classify_status(
        exit_code=exit_code,
        setup_error=setup_error,
        oom_detected=oom_detected,
        devices=devices,
        tracking_confidence=tracking_confidence,
    )
    if status == STATUS_WARN_INCOMPLETE:
        warnings.append("No runtime memory report was produced; fit/no-fit confidence is incomplete.")

    errors = _collect_errors(
        status=status,
        setup_error=setup_error,
        stdout=stdout,
        stderr=stderr,
        raw_report=raw_report,
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
        },
        "status": status,
        "stage": stage,
        "exit_code": exit_code,
        "duration_seconds": round(duration_seconds, 3),
        "tracking_confidence": tracking_confidence,
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
    lines = [
        "# FakeGPU Preflight Report",
        "",
        f"**Status:** `{status}`",
        f"**Runtime:** `{runtime}`",
        f"**Stage:** `{stage}`",
        f"**Tracking confidence:** `{report.get('tracking_confidence', 'unknown')}`",
        "",
        "## Command",
        "",
        "```bash",
        shlex.join([str(part) for part in report.get("command", [])]),
        "```",
        "",
        "## Device Memory",
        "",
        "| GPU | Name | Peak | Total | Headroom | Allocations |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for dev in report.get("devices", []):
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
        lines.extend(["", "## Errors", ""])
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
            "## Logs",
            "",
            f"- stdout: `{report.get('logs', {}).get('stdout')}`",
            f"- stderr: `{report.get('logs', {}).get('stderr')}`",
            "",
        ]
    )
    return "\n".join(lines)


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


def _classify_status(
    *,
    exit_code: int,
    setup_error: Exception | None,
    oom_detected: bool,
    devices: list[dict[str, Any]],
    tracking_confidence: str,
) -> str:
    if setup_error is not None:
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
    if "outofmemory" in haystack or "out of memory" in haystack or "cuda oom" in haystack:
        return True
    exc = raw_report.get("exception") if isinstance(raw_report, dict) else None
    if isinstance(exc, dict):
        exc_text = f"{exc.get('type', '')}\n{exc.get('message', '')}".lower()
        return "outofmemory" in exc_text or "out of memory" in exc_text or "oom" in exc_text
    return False


def _collect_errors(
    *,
    status: str,
    setup_error: Exception | None,
    stdout: str,
    stderr: str,
    raw_report: dict[str, Any] | None,
) -> list[dict[str, str]]:
    if setup_error is not None:
        return [{"type": type(setup_error).__name__, "message": str(setup_error)}]
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
        lower = line.lower()
        if "out of memory" in lower or "outofmemory" in lower or "oom" in lower:
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
