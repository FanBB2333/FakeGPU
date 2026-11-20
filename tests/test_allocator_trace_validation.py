from __future__ import annotations

from scripts.validation.allocator_trace_validation import (
    COMPARISON_SCHEMA_VERSION,
    SCHEMA_VERSION,
    compare,
    render_markdown,
)


def _capture(mode: str, reserved_offset: int = 0) -> dict:
    stages = []
    for index, (operation, allocated, reserved) in enumerate(
        (
            ("allocate", 1024, 2 * 1024**2),
            ("free", 0, 2 * 1024**2),
            ("empty_cache", 0, 0),
        )
    ):
        stages.append(
            {
                "label": f"{index:02d}_{operation}",
                "operation": operation,
                "requested_bytes": 1024 if operation == "allocate" else 0,
                "allocated_delta_bytes": allocated,
                "reserved_delta_bytes": reserved + reserved_offset,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "hostname": "host",
        "gpu_name": "GPU",
        "compute_capability": "8.6",
        "torch_version": "2.12",
        "torch_cuda_version": "13.0",
        "profile": "rtx3090ti-reference" if mode == "fakecuda" else None,
        "stages": stages,
    }


def test_allocator_trace_comparison_accepts_segment_tolerance() -> None:
    report = compare(
        _capture("real"),
        _capture("fakecuda", reserved_offset=512),
        reserved_tolerance_bytes=1024,
    )
    assert report["schema_version"] == COMPARISON_SCHEMA_VERSION
    # The fixture adds an offset to empty_cache too, which must fail the
    # explicit baseline check even though it is inside the byte tolerance.
    assert report["status"] == "failed"
    assert report["checks"]["reserved_trace_within_tolerance"] is True
    assert report["checks"]["empty_cache_returns_to_baseline"] is False


def test_allocator_trace_comparison_and_markdown_success() -> None:
    report = compare(_capture("real"), _capture("fakecuda"))
    assert report["status"] == "success"
    assert report["max_absolute_allocated_error_bytes"] == 0
    assert "Allocator Trace Validation" in render_markdown(report)


def test_allocator_trace_capture_takes_the_profile_from_the_user(
    tmp_path,
) -> None:
    # The FakeCUDA side compares against a GPU the user names; there is no
    # built-in default device.
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "validation" / "allocator_trace_validation.py"
    env = dict(os.environ)
    env.pop("TARGET_GPU_PROFILE", None)
    env["PYTHONPATH"] = str(root)
    env["FAKEGPU_TERMINAL_REPORT"] = "0"
    output = tmp_path / "fake.json"
    command = [
        sys.executable,
        str(script),
        "capture",
        "--mode",
        "fakecuda",
        "--output",
        str(output),
    ]
    missing = subprocess.run(
        command, env=env, capture_output=True, text=True, timeout=120
    )
    assert missing.returncode == 2
    assert "TARGET_GPU_PROFILE" in missing.stderr

    env["TARGET_GPU_PROFILE"] = "test-512m"
    completed = subprocess.run(
        command, env=env, capture_output=True, text=True, timeout=120
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text())["profile"] == "test-512m"
