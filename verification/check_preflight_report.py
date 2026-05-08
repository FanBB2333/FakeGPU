#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate FakeGPU preflight_report.json")
    parser.add_argument("--path", default="preflight_report.json", help="Path to preflight_report.json")
    parser.add_argument("--expect-status", help="Expected report status")
    parser.add_argument("--expect-runtime", help="Expected runtime name")
    parser.add_argument("--expect-device", action="store_true", help="Require at least one device entry")
    ns = parser.parse_args()

    path = Path(ns.path)
    if not path.is_file():
        _die(f"report file not found: {path}")

    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        _die("report root must be a JSON object")

    _require(report, "schema_version", "preflight.v1")
    _require(report, "status")
    _require(report, "runtime")
    _require(report, "command")
    _require(report, "stage")
    _require(report, "tracking_confidence")
    _require(report, "logs")

    if ns.expect_status and report.get("status") != ns.expect_status:
        _die(f"status={report.get('status')!r}, expected {ns.expect_status!r}")
    if ns.expect_runtime and report.get("runtime") != ns.expect_runtime:
        _die(f"runtime={report.get('runtime')!r}, expected {ns.expect_runtime!r}")

    logs = report["logs"]
    if not isinstance(logs, dict):
        _die("logs must be an object")
    for key in ("stdout", "stderr"):
        value = logs.get(key)
        if not isinstance(value, str) or not value:
            _die(f"logs.{key} must be a non-empty string")
        if not Path(value).is_file():
            _die(f"logs.{key} does not exist: {value}")

    devices = report.get("devices")
    if ns.expect_device and (not isinstance(devices, list) or not devices):
        _die("expected at least one device entry")
    if isinstance(devices, list):
        for index, dev in enumerate(devices):
            _validate_device(index, dev)

    print(f"OK: preflight report {path} status={report.get('status')}")
    return 0


def _validate_device(index: int, dev: Any) -> None:
    if not isinstance(dev, dict):
        _die(f"devices[{index}] must be an object")
    for field in (
        "index",
        "total_memory",
        "peak_memory",
        "headroom_bytes",
        "allocation_count",
        "tracking_confidence",
    ):
        if field not in dev:
            _die(f"devices[{index}].{field} is missing")
    if int(dev["total_memory"]) < 0:
        _die(f"devices[{index}].total_memory must be >= 0")
    if int(dev["peak_memory"]) < 0:
        _die(f"devices[{index}].peak_memory must be >= 0")


def _require(report: dict[str, Any], key: str, expected: Any | None = None) -> Any:
    if key not in report:
        _die(f"{key} is missing")
    value = report[key]
    if expected is not None and value != expected:
        _die(f"{key}={value!r}, expected {expected!r}")
    return value


def _die(message: str) -> None:
    print(f"[check_preflight_report] ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit(main())
