"""Combine the memory records of a ``fakegpu run`` into a verdict and print it.

Each process of the run that used a simulated GPU leaves one record (see
``fakegpu._run_bootstrap``). Processes on the same GPU add up, as they would
on real hardware. Every number in the summary names where it comes from: a
tracking confidence level for measured memory, the profile catalog for
capacities.
"""

from __future__ import annotations

import json
from pathlib import Path
import shlex
import textwrap
from typing import Any, Sequence

from ._preflight_report import _fmt_bytes

SCHEMA_VERSION = "fakegpu.run.v1"

# Weakest first: a GPU used by several processes reports the weakest level.
CONFIDENCE_ORDER = (
    "C0_incomplete",
    "C1_weight_storage",
    "C2_torch_tensor_lifetime",
    "C3_native_cuda_allocations",
    "C3_torch_dispatch_lifetime",
    "C4_real_gpu_calibrated",
)
CONFIDENCE_TEXT = {
    "C0_incomplete": "no usable memory tracking",
    "C1_weight_storage": "weights and explicit storage only",
    "C2_torch_tensor_lifetime": "tensor lifetimes tracked in PyTorch on the CPU",
    "C3_native_cuda_allocations": "native CUDA allocation calls tracked",
    "C3_torch_dispatch_lifetime": "operator outputs tracked at PyTorch dispatch on the CPU",
    "C4_real_gpu_calibrated": "calibrated against a real GPU",
}
NOT_COUNTED = "the CUDA context, library workspaces and NCCL buffers a real GPU adds."
MAX_LISTED_GPUS = 16
_BYPASSED = "Python started with -I, -E or -S skips the hook that gives it simulated GPUs."


def load_records(run_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(run_dir.glob("process-*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def build_report(
    *,
    command: Sequence[str],
    exit_code: int,
    duration_seconds: float,
    layout: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """``layout`` lists the run's GPUs: ``index``, ``profile_id``, ``name``, ``total_memory``."""

    devices = {
        int(gpu["index"]): {
            "index": int(gpu["index"]),
            "profile_id": gpu.get("profile_id"),
            "name": gpu.get("name"),
            "total_memory": int(gpu.get("total_memory") or 0),
            "peak_memory": 0,
            "process_count": 0,
            "oom_events": 0,
            "last_oom": None,
            "tracking_confidence": None,
        }
        for gpu in layout
    }
    processes = []
    overridden = False
    for record in records:
        process = {key: record.get(key) for key in ("pid", "rank", "local_rank", "argv", "exception", "start_error")}
        processes.append(process)
        snapshot = record.get("snapshot") or {}
        for used in snapshot.get("devices", []):
            if not (used.get("allocation_count") or used.get("oom_events")):
                continue
            index = int(used.get("index", 0))
            device = devices.setdefault(index, {
                "index": index, "profile_id": used.get("profile_id"), "name": used.get("name"),
                "total_memory": int(used.get("total_memory") or 0), "peak_memory": 0, "process_count": 0,
                "oom_events": 0, "last_oom": None, "tracking_confidence": None,
            })
            # The process's own GPU wins: a script may pick its GPUs with fakegpu.init().
            actual = (used.get("profile_id") or device["profile_id"], int(used.get("total_memory") or 0))
            if actual != (device["profile_id"], device["total_memory"]):
                overridden = True
                device["profile_id"], device["total_memory"] = actual
                device["name"] = used.get("name") or device["name"]
            device["peak_memory"] += max(int(used.get("peak_memory") or 0), int(used.get("peak_reserved_memory") or 0))
            device["process_count"] += 1
            device["oom_events"] += int(used.get("oom_events") or 0)
            if used.get("last_oom"):
                device["last_oom"] = dict(used["last_oom"])
            device["tracking_confidence"] = _weakest(
                device["tracking_confidence"], used.get("tracking_confidence") or snapshot.get("tracking_confidence")
            )
    for device in devices.values():
        device["headroom_bytes"] = device["total_memory"] - device["peak_memory"]
    ordered = [devices[index] for index in sorted(devices)]
    used_devices = [device for device in ordered if device["process_count"]]
    confidence = None
    for device in used_devices:
        confidence = _weakest(confidence, device["tracking_confidence"])
    verdict, text = _verdict(exit_code, ordered, used_devices, processes)
    return {
        "schema_version": SCHEMA_VERSION,
        "command": list(command),
        "exit_code": exit_code,
        "duration_seconds": round(duration_seconds, 3),
        "verdict": verdict,
        "verdict_text": text,
        "tracking_confidence": confidence or "C0_incomplete",
        "devices": ordered,
        "processes": processes,
        "notes": (
            ["The command chose its own GPUs, e.g. with fakegpu.init(); the rows show the GPUs it used."]
            if overridden else []
        ),
    }


def _weakest(current: str | None, candidate: str | None) -> str | None:
    if candidate is None:
        return current
    if current is None:
        return candidate
    rank = {name: position for position, name in enumerate(CONFIDENCE_ORDER)}
    return min(current, candidate, key=lambda name: rank.get(name, -1))


def _verdict(
    exit_code: int, devices: list[dict[str, Any]], used: list[dict[str, Any]], processes: list[dict[str, Any]]
) -> tuple[str, str]:
    raised = [p for p in processes if "OutOfMemory" in str((p.get("exception") or {}).get("type", ""))]
    oom_devices = [device for device in used if device["last_oom"]]
    over = [device for device in used if device["headroom_bytes"] < 0]
    if raised or over:
        if oom_devices:
            device = oom_devices[0]
            oom = device["last_oom"]
            return "does_not_fit", (
                f"Does not fit. GPU {device['index']} ran out of memory: tried to allocate "
                f"{_fmt_bytes(oom['requested_bytes'])} with {_fmt_bytes(oom['reserved_bytes'])} reserved "
                f"of {_fmt_bytes(device['total_memory'])}."
            )
        device = (over or used or devices)[0]
        return "does_not_fit", (
            f"Does not fit. GPU {device['index']} peaked at {_fmt_bytes(device['peak_memory'])} "
            f"of {_fmt_bytes(device['total_memory'])}."
        )
    if exit_code != 0:
        failure = next((p["exception"] for p in processes if p.get("exception")), None)
        reason = "was interrupted" if exit_code in (130, 143) else f"failed (exit {exit_code})"
        text = f"Unresolved. The command {reason} before finishing; the peaks cover the run until then."
        if failure:
            text += f" First error: {failure['type']}: {_shorten(failure['message'])}"
        elif not processes:
            text += f" No process used a simulated GPU. {_BYPASSED}"
        return "unresolved", text
    if not used:
        errors = [p["start_error"] for p in processes if p.get("start_error")]
        if errors:
            return "no_gpu_use", f"No simulated GPU was used: FakeCUDA did not start ({_shorten(errors[0])})."
        return "no_gpu_use", f"No simulated GPU memory was used. Does the command put tensors on 'cuda'? {_BYPASSED}"
    largest = max(used, key=lambda device: device["peak_memory"] / max(device["total_memory"], 1))
    share = 100.0 * largest["peak_memory"] / largest["total_memory"] if largest["total_memory"] else 0.0
    share_text = "under 0.1" if 0 < share < 0.1 else f"{share:.1f}"
    text = (
        f"Fits. Largest peak: {_fmt_bytes(largest['peak_memory'])} on GPU {largest['index']}, "
        f"{share_text}% of its capacity."
    )
    caught = sum(device["oom_events"] for device in used)
    if caught:
        text += f" The command caught {caught} simulated out-of-memory error(s) and continued."
    return "fits", text


def _shorten(text: Any, limit: int = 160) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _label(confidence: str | None) -> str:
    return (confidence or "C0_incomplete").split("_", 1)[0]


def render(report: dict[str, Any], *, report_path: Path | None = None) -> str:
    devices = report["devices"]
    used = [device for device in devices if device["process_count"]]
    process_count = sum(1 for process in report["processes"] if not process.get("start_error"))
    lines = ["", "── FakeGPU run summary " + "─" * 48]
    lines.append(f" Command   {shlex.join(report['command'])}")
    lines.append(
        f" Result    exit {report['exit_code']} after {report['duration_seconds']:.1f} s; "
        f"{process_count} process{'es' if process_count != 1 else ''} used simulated GPUs"
    )
    lines.append(f" GPUs      {_layout_text(devices)}  [catalog]")
    lines.append("")
    lines.append("  GPU  Peak memory          Capacity     Headroom     Processes")
    listed = devices if len(devices) <= MAX_LISTED_GPUS else used
    for device in listed:
        peak = (
            f"{_fmt_bytes(device['peak_memory']):>11}  [{_label(device['tracking_confidence'])}]"
            if device["process_count"] else f"{'—':>11}      "
        )
        lines.append(
            f"  {device['index']:>3}  {peak}    {_fmt_bytes(device['total_memory']):>10}   "
            f"{_fmt_bytes(device['headroom_bytes']):>10}   {device['process_count']:>9}"
        )
    if len(listed) < len(devices):
        lines.append(f"       {len(devices) - len(listed)} more GPUs were not used.")
    lines.append("")
    lines.extend(_wrapped("Verdict", report["verdict_text"]))
    for note in report.get("notes", []):
        lines.extend(_wrapped("Note", note))
    labels = sorted({device["tracking_confidence"] for device in used if device["tracking_confidence"]})
    for confidence in labels:
        text = CONFIDENCE_TEXT.get(confidence, confidence)
        suffix = "" if confidence == "C4_real_gpu_calibrated" else "; not calibrated against a real GPU (C4)"
        lines.extend(_wrapped(f"[{_label(confidence)}]", f"Peak reserved memory: {text}{suffix}."))
    lines.extend(_wrapped("[catalog]", "Capacity from the FakeGPU profile catalog."))
    lines.extend(_wrapped("Excludes", NOT_COUNTED))
    if report_path is not None:
        lines.append(f" {'Report':<9} {report_path}")
    lines.append("─" * 71)
    return "\n".join(lines) + "\n"


def _wrapped(key: str, text: str, width: int = 71) -> list[str]:
    return textwrap.wrap(text, width, initial_indent=f" {key:<9} ", subsequent_indent=" " * 11)


def _layout_text(devices: list[dict[str, Any]]) -> str:
    groups: list[list[Any]] = []
    for device in devices:
        key = (device["profile_id"], device["name"], device["total_memory"])
        if groups and groups[-1][0] == key:
            groups[-1][1] += 1
        else:
            groups.append([key, 1])
    if len(groups) == 1:
        (profile_id, name, total), count = groups[0]
        return f"{count} × {profile_id} ({name}, {_fmt_bytes(total)}{' each' if count > 1 else ''})"
    return ", ".join(f"{count} × {key[0]}" for key, count in groups)
