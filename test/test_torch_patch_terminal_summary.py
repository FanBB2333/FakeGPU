"""Regression checks for torch_patch terminal summary formatting."""

from __future__ import annotations

import contextlib
import io

import fakegpu.torch_patch as tp


def test_dump_terminal_summary_includes_peak_vram_by_gpu(monkeypatch) -> None:
    tracker = tp._DeviceMemoryTracker([80 * 1024**3, 80 * 1024**3])
    tracker._peak = [int(9.2 * 1024**2), 512 * 1024]
    tracker._alloc_calls = [4, 1]
    tracker._free_calls = [4, 1]

    monkeypatch.setenv("FAKEGPU_TERMINAL_REPORT", "1")
    monkeypatch.setattr(tp, "_memory_tracker", tracker)
    monkeypatch.setattr(
        tp,
        "_DEVICE_PROFILES",
        [
            {"name": "NVIDIA A100-SXM4-80GB", "compute_major": 8, "compute_minor": 0},
            {"name": "NVIDIA A100-SXM4-80GB", "compute_major": 8, "compute_minor": 0},
        ],
    )

    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        tp._dump_terminal_summary()

    output = stderr.getvalue()
    assert "FakeGPU Report Summary" in output
    assert "Peak VRAM by GPU:" in output
    assert "GPU 0: 9.2 MB" in output
    assert "GPU 1: 512.0 KB" in output
