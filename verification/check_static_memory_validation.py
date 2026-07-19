#!/usr/bin/env python3
"""Validate a static-memory estimator report."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--allow-static-only", action="store_true")
    parser.add_argument("--max-underestimate-percent", type=float)
    ns = parser.parse_args(argv)

    report = _load_report(ns.path)
    _require(report.get("schema_version") == "static_memory_validation.v1", "unexpected schema")
    status = str(report.get("status") or "")
    allowed_statuses = {"PASS_MEASURED"}
    if ns.allow_static_only:
        allowed_statuses.add("PASS_STATIC_ONLY")
    _require(status in allowed_statuses, f"unexpected status: {status!r}")

    workloads = report.get("workloads")
    _require(isinstance(workloads, list) and workloads, "workloads must be a non-empty list")
    if status == "PASS_MEASURED":
        backend = report.get("backend_calibration")
        _require(isinstance(backend, dict), "backend_calibration is required")
        _require(
            "resident_allocated_bytes" in backend
            and int(backend.get("resident_allocated_bytes", 0) or 0) >= 0,
            "backend_calibration.resident_allocated_bytes must be non-negative",
        )
        _require(
            "resident_requested_bytes" in backend
            and int(backend.get("resident_requested_bytes", 0) or 0) >= 0,
            "backend_calibration.resident_requested_bytes must be non-negative",
        )
    for index, item in enumerate(workloads):
        _require(isinstance(item, dict), f"workloads[{index}] must be an object")
        context = f"workloads[{index}]"
        _require(bool(item.get("name")), f"{context}.name is required")
        static = item.get("static_estimate")
        _require(isinstance(static, dict), f"{context}.static_estimate is required")
        _require(
            int(static.get("estimated_peak_bytes", 0) or 0) > 0,
            f"{context}.static_estimate.estimated_peak_bytes must be positive",
        )
        workspace = static.get("workspace_estimate")
        _require(
            isinstance(workspace, dict),
            f"{context}.static_estimate.workspace_estimate is required",
        )
        workspace_bytes = int(static.get("workspace_estimate_bytes", 0) or 0)
        _require(workspace_bytes >= 0, f"{context} workspace bytes must be non-negative")
        _require(
            workspace_bytes == int(workspace.get("total_bytes", -1)),
            f"{context} workspace byte fields disagree",
        )
        profiles = workspace.get("profiles")
        _require(isinstance(profiles, list), f"{context} workspace profiles must be a list")
        for profile_index, profile in enumerate(profiles):
            _require(
                isinstance(profile, dict)
                and int(profile.get("bytes", 0) or 0) > 0,
                f"{context} workspace profile {profile_index} is invalid",
            )
        graph = static.get("graph")
        _require(isinstance(graph, dict), f"{context}.static_estimate.graph is required")
        _require(int(graph.get("node_count", 0) or 0) > 0, f"{context} graph is empty")
        _require(
            isinstance(graph.get("operator_histogram"), dict),
            f"{context} graph operator histogram is required",
        )
        _require(
            int(graph.get("unique_storage_count", 0) or 0) > 0,
            f"{context} graph has no storages",
        )
        _require(not graph.get("warnings"), f"{context} graph contains warnings")

        if status == "PASS_MEASURED":
            real = item.get("real_cuda")
            comparison = item.get("comparison")
            _require(isinstance(real, dict), f"{context}.real_cuda is required")
            _require(isinstance(comparison, dict), f"{context}.comparison is required")
            _require(
                comparison.get("method") == "static_graph_plus_backend_resident_calibration",
                f"{context}.comparison.method is unexpected",
            )
            _require(
                int(real.get("peak_allocated_bytes", 0) or 0) > 0,
                f"{context}.real_cuda.peak_allocated_bytes must be positive",
            )
            underestimate = comparison.get("underestimate_percent")
            _require(
                isinstance(underestimate, (int, float)) and math.isfinite(float(underestimate)),
                f"{context}.comparison.underestimate_percent must be finite",
            )
            if real.get("requested_peak_bytes") is not None:
                requested_underestimate = comparison.get("requested_underestimate_percent")
                _require(
                    isinstance(requested_underestimate, (int, float))
                    and math.isfinite(float(requested_underestimate)),
                    f"{context}.comparison.requested_underestimate_percent must be finite",
                )
            if ns.max_underestimate_percent is not None:
                _require(
                    float(underestimate) <= ns.max_underestimate_percent,
                    f"{context} underestimate {float(underestimate):.3f}% exceeds "
                    f"{ns.max_underestimate_percent:.3f}%",
                )

    print(f"OK: static memory validation {ns.path} status={status} workloads={len(workloads)}")
    return 0


def _load_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        _die(f"report not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _die(f"unable to load report {path}: {exc}")
    if not isinstance(payload, dict):
        _die("report must contain a JSON object")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        _die(message)


def _die(message: str) -> None:
    print(f"[check_static_memory_validation] ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit(main())
