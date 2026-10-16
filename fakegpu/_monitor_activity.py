"""Opt-in CPU dispatch observations used only as a simulated load proxy."""

from __future__ import annotations

import os
import threading


_lock = threading.Lock()
_counters: dict[int, dict[str, int]] = {}


def enabled() -> bool:
    return os.environ.get("FAKEGPU_TELEMETRY") == "reference"


def record(device: int, elapsed_ns: int, io_bytes: int) -> None:
    with _lock:
        row = _counters.setdefault(device, {"busy_ns": 0, "io_bytes": 0, "operations": 0})
        row["busy_ns"] += max(0, elapsed_ns)
        row["io_bytes"] += max(0, io_bytes)
        row["operations"] += 1


def snapshot() -> dict[int, dict]:
    if not enabled():
        return {}
    with _lock:
        return {index: {"source": "cpu_dispatch_proxy", **row} for index, row in _counters.items()}


def empty() -> dict:
    return {"source": "cpu_dispatch_proxy", "busy_ns": 0, "io_bytes": 0, "operations": 0}
