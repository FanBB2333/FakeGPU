#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate RTX 3090 Ti calibration report.")
    parser.add_argument("--path", required=True)
    parser.add_argument("--allow-skip", action="store_true")
    ns = parser.parse_args(argv)

    path = Path(ns.path)
    if not path.is_file():
        _die(f"report not found: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))

    if report.get("schema_version") != "rtx3090ti_calibration.v1":
        _die(f"unexpected schema_version: {report.get('schema_version')!r}")

    status = str(report.get("status", ""))
    if status.startswith("SKIP"):
        if not ns.allow_skip:
            _die(f"calibration skipped: {report.get('skip_reason')}")
        if not report.get("skip_reason"):
            _die("skip report must include skip_reason")
        print(f"OK: calibration skipped with reason: {report.get('skip_reason')}")
        return 0

    if status != "PASS_CALIBRATED":
        _die(f"unexpected status: {status}")

    gpu = _require_dict(report, "calibration_gpu")
    if gpu.get("name") != "NVIDIA GeForce RTX 3090 Ti":
        _die(f"unexpected calibration GPU: {gpu.get('name')!r}")
    if int(gpu.get("total_memory", 0) or 0) <= 0:
        _die("calibration_gpu.total_memory must be positive")

    workloads = report.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        _die("workloads must be a non-empty list")

    for index, workload in enumerate(workloads):
        if not isinstance(workload, dict):
            _die(f"workloads[{index}] must be an object")
        name = workload.get("name")
        if not name:
            _die(f"workloads[{index}].name is required")
        real = _require_dict(workload, "real_cuda", ctx=f"workloads[{index}]")
        fake = _require_dict(workload, "fakecuda_preflight", ctx=f"workloads[{index}]")
        if int(real.get("peak_memory", 0) or 0) <= 0:
            _die(f"{name}: real peak_memory must be positive")
        if int(fake.get("peak_memory", 0) or 0) <= 0:
            _die(f"{name}: fakecuda peak_memory must be positive")
        if fake.get("status") != "PASS_FIT":
            _die(f"{name}: fakecuda preflight status must be PASS_FIT, got {fake.get('status')!r}")
        if "peak_error_bytes" not in workload or "peak_error_percent" not in workload:
            _die(f"{name}: missing peak error fields")
        if "calibration_factor" not in workload:
            _die(f"{name}: missing calibration_factor")
        if "likely_gap_reason" not in workload:
            _die(f"{name}: missing likely_gap_reason")
        gap = workload.get("gap_analysis")
        if not isinstance(gap, dict) or "available" not in gap:
            _die(f"{name}: missing gap_analysis")

    print(f"OK: calibrated {len(workloads)} workload(s) on RTX 3090 Ti")
    return 0


def _require_dict(obj: dict[str, Any], key: str, *, ctx: str = "report") -> dict[str, Any]:
    value = obj.get(key)
    if not isinstance(value, dict):
        _die(f"{ctx}.{key} must be an object")
    return value


def _die(message: str) -> None:
    print(f"[check_rtx3090ti_calibration] ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
