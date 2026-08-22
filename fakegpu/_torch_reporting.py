"""Formatting helpers for the torch-layer FakeGPU report."""

from __future__ import annotations

import sys
from typing import Any

from .profile_catalog import architecture_for_compute_capability


def _arch_name(major: int, minor: int) -> str:
    """Return the architecture name for a compute capability."""
    architecture = architecture_for_compute_capability(major, minor)
    return architecture.title() if architecture != "unknown" else "Unknown"


def _fmt_bytes(b: int) -> str:
    if b >= 1024**3:
        return f"{b / 1024**3:.1f} GB"
    elif b >= 1024**2:
        return f"{b / 1024**2:.1f} MB"
    elif b >= 1024:
        return f"{b / 1024:.1f} KB"
    return f"{b} B"


def render_terminal_summary(
    tracker: Any,
    device_profiles: list[dict[str, Any]],
    *,
    arch_name: Any = _arch_name,
    fmt_bytes: Any = _fmt_bytes,
) -> None:
    """Write the torch-layer memory summary to stderr."""
    if tracker is None:
        return

    lines: list[str] = []
    lines.append("")
    lines.append("======================================================")
    lines.append("             FakeGPU Report Summary")
    lines.append("======================================================")

    for i, prof in enumerate(device_profiles):
        if i >= len(tracker._total):
            break
        name = prof.get("name", "NVIDIA A100-SXM4-80GB")
        cc_major = prof.get("compute_major", 8)
        cc_minor = prof.get("compute_minor", 0)
        arch = arch_name(cc_major, cc_minor)

        total = tracker._total[i]
        peak = tracker._peak[i]
        reserved_peak = tracker._reserved_peak[i]
        peak_pct = (100.0 * reserved_peak / total) if total > 0 else 0.0

        alloc = tracker._alloc_calls[i]
        free = tracker._free_calls[i]

        lines.append(f" Device {i}: {name} ({arch}, cc {cc_major}.{cc_minor})")
        lines.append(
            f"   Memory: {fmt_bytes(peak)} allocated | "
            f"{fmt_bytes(reserved_peak)} reserved / {fmt_bytes(total)} "
            f"({peak_pct:.1f}%)"
        )
        lines.append(f"   Alloc: {alloc} calls | Free: {free} calls")
        lines.append("------------------------------------------------------")

    lines.append(" Peak VRAM by GPU:")
    for i, peak in enumerate(tracker._peak[: len(device_profiles)]):
        lines.append(
            f"   GPU {i}: {fmt_bytes(peak)} allocated | "
            f"{fmt_bytes(tracker._reserved_peak[i])} reserved"
        )
    lines.append("------------------------------------------------------")

    lines.append("======================================================")
    lines.append("")

    sys.stderr.write("\n".join(lines))
    sys.stderr.flush()
