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
    parser.add_argument(
        "--schema",
        default=str(Path(__file__).resolve().parent.parent / "preflight_report.schema.json"),
        help="Path to preflight_report.schema.json",
    )
    parser.add_argument("--no-schema", action="store_true", help="Skip JSON schema validation")
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

    if not ns.no_schema:
        _validate_schema_file(report, Path(ns.schema))

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


def _validate_schema_file(report: dict[str, Any], schema_path: Path) -> None:
    if not schema_path.is_file():
        _die(f"schema file not found: {schema_path}")
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except Exception as exc:
        _die(f"failed to read schema {schema_path}: {exc}")
    _validate_against_schema(report, schema, "$")


def _validate_against_schema(value: Any, schema: dict[str, Any], path: str) -> None:
    if "type" in schema:
        _validate_type(value, schema["type"], path)

    if "enum" in schema and value not in schema["enum"]:
        _die(f"{path} must be one of {schema['enum']!r}, got {value!r}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            _die(f"{path} must be >= {schema['minimum']!r}")

    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                _die(f"{path}.{key} is missing")
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for key, child_schema in properties.items():
                if key in value and isinstance(child_schema, dict):
                    _validate_against_schema(value[key], child_schema, f"{path}.{key}")
        additional = schema.get("additionalProperties")
        if isinstance(additional, dict):
            for key, item in value.items():
                if key not in properties:
                    _validate_against_schema(item, additional, f"{path}.{key}")
        if additional is False:
            extra = sorted(key for key in value if key not in properties)
            if extra:
                _die(f"{path} has unexpected field(s): {', '.join(extra)}")

    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        item_schema = schema["items"]
        for index, item in enumerate(value):
            _validate_against_schema(item, item_schema, f"{path}[{index}]")


def _validate_type(value: Any, expected: Any, path: str) -> None:
    expected_types = expected if isinstance(expected, list) else [expected]
    if any(_matches_json_type(value, item) for item in expected_types):
        return
    _die(f"{path} must be {expected!r}, got {type(value).__name__}")


def _matches_json_type(value: Any, expected: Any) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _validate_device(index: int, dev: Any) -> None:
    if not isinstance(dev, dict):
        _die(f"devices[{index}] must be an object")
    for field in (
        "index",
        "total_memory",
        "peak_memory",
        "headroom_bytes",
        "allocation_count",
        "current_bytes_by_category",
        "peak_by_stage",
        "largest_allocations",
        "tracking_confidence",
    ):
        if field not in dev:
            _die(f"devices[{index}].{field} is missing")
    if int(dev["total_memory"]) < 0:
        _die(f"devices[{index}].total_memory must be >= 0")
    if int(dev["peak_memory"]) < 0:
        _die(f"devices[{index}].peak_memory must be >= 0")
    if not isinstance(dev["current_bytes_by_category"], dict):
        _die(f"devices[{index}].current_bytes_by_category must be an object")
    if not isinstance(dev["peak_by_stage"], dict):
        _die(f"devices[{index}].peak_by_stage must be an object")
    if not isinstance(dev["largest_allocations"], list):
        _die(f"devices[{index}].largest_allocations must be a list")
    for alloc_index, alloc in enumerate(dev["largest_allocations"]):
        if not isinstance(alloc, dict):
            _die(f"devices[{index}].largest_allocations[{alloc_index}] must be an object")
        for field in ("bytes", "device", "stage", "category"):
            if field not in alloc:
                _die(f"devices[{index}].largest_allocations[{alloc_index}].{field} is missing")
        stack = alloc.get("stack")
        if stack is not None:
            if not isinstance(stack, list):
                _die(f"devices[{index}].largest_allocations[{alloc_index}].stack must be a list")
            for frame_index, frame in enumerate(stack):
                if not isinstance(frame, dict):
                    _die(
                        f"devices[{index}].largest_allocations[{alloc_index}]."
                        f"stack[{frame_index}] must be an object"
                    )
                for field in ("file", "line", "function"):
                    if field not in frame:
                        _die(
                            f"devices[{index}].largest_allocations[{alloc_index}]."
                            f"stack[{frame_index}].{field} is missing"
                        )


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
