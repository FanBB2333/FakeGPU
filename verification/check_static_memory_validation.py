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
    parser.add_argument("--min-workspace-coverage", type=float)
    parser.add_argument("--reject-extrapolated-workspaces", action="store_true")
    ns = parser.parse_args(argv)
    if ns.min_workspace_coverage is not None:
        _require(
            math.isfinite(ns.min_workspace_coverage)
            and 0.0 <= ns.min_workspace_coverage <= 1.0,
            "--min-workspace-coverage must be between 0 and 1",
        )
    if ns.reject_extrapolated_workspaces:
        _require(
            ns.min_workspace_coverage is not None,
            "--reject-extrapolated-workspaces requires --min-workspace-coverage",
        )

    report = _load_report(ns.path)
    _require(report.get("schema_version") == "static_memory_validation.v1", "unexpected schema")
    status = str(report.get("status") or "")
    allowed_statuses = {"PASS_MEASURED"}
    if ns.allow_static_only:
        allowed_statuses.add("PASS_STATIC_ONLY")
    _require(status in allowed_statuses, f"unexpected status: {status!r}")

    workloads = report.get("workloads")
    _require(isinstance(workloads, list) and workloads, "workloads must be a non-empty list")
    phase_resolved = bool((report.get("sampling") or {}).get("phase_resolved"))
    summary = report.get("summary")
    _require(isinstance(summary, dict), "summary is required")
    for field in (
        "workspace_profile_count",
        "extrapolated_workspace_profile_count",
        "graph_modeled_attention_operator_count",
        "unprofiled_attention_operator_count",
    ):
        if field in summary:
            _require(
                int(summary.get(field, -1)) >= 0,
                f"summary.{field} must be non-negative",
            )
    coverage = summary.get("workspace_coverage")
    if coverage is None:
        _require(
            ns.min_workspace_coverage is None,
            "summary.workspace_coverage is required for a coverage threshold",
        )
    elif isinstance(coverage, dict):
        for field in (
            "candidate_operator_count",
            "modeled_operator_count",
            "non_extrapolated_operator_count",
            "unprofiled_operator_count",
        ):
            _require(
                int(coverage.get(field, -1)) >= 0,
                f"summary.workspace_coverage.{field} must be non-negative",
            )
        candidate_count = int(coverage["candidate_operator_count"])
        modeled_count = int(coverage["modeled_operator_count"])
        non_extrapolated_count = int(coverage["non_extrapolated_operator_count"])
        unprofiled_count = int(coverage["unprofiled_operator_count"])
        _require(
            modeled_count + unprofiled_count == candidate_count,
            "summary.workspace_coverage counts are inconsistent",
        )
        _require(
            0 <= non_extrapolated_count <= modeled_count,
            "summary.workspace_coverage non-extrapolated count is inconsistent",
        )
        expected_modeled_fraction = (
            modeled_count / candidate_count if candidate_count else 1.0
        )
        expected_non_extrapolated_fraction = (
            non_extrapolated_count / candidate_count if candidate_count else 1.0
        )
        _require(
            math.isclose(
                float(coverage.get("modeled_fraction", -1.0)),
                expected_modeled_fraction,
                abs_tol=1e-6,
            ),
            "summary.workspace_coverage.modeled_fraction is inconsistent",
        )
        _require(
            math.isclose(
                float(coverage.get("non_extrapolated_fraction", -1.0)),
                expected_non_extrapolated_fraction,
                abs_tol=1e-6,
            ),
            "summary.workspace_coverage.non_extrapolated_fraction is inconsistent",
        )
        if ns.min_workspace_coverage is not None:
            selected_fraction = (
                expected_non_extrapolated_fraction
                if ns.reject_extrapolated_workspaces
                else expected_modeled_fraction
            )
            _require(
                selected_fraction + 1e-12 >= ns.min_workspace_coverage,
                f"workspace coverage {selected_fraction:.6f} is below "
                f"{ns.min_workspace_coverage:.6f}",
            )
    else:
        _require(False, "summary.workspace_coverage must be an object")
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
        item_coverage = workspace.get("coverage")
        if item_coverage is not None:
            _require(
                isinstance(item_coverage, dict),
                f"{context}.static_estimate.workspace_estimate.coverage must be an object",
            )
        workspace_bytes = int(static.get("workspace_estimate_bytes", 0) or 0)
        _require(workspace_bytes >= 0, f"{context} workspace bytes must be non-negative")
        _require(
            workspace_bytes == int(workspace.get("total_bytes", -1)),
            f"{context} workspace byte fields disagree",
        )
        profiles = workspace.get("profiles")
        _require(isinstance(profiles, list), f"{context} workspace profiles must be a list")
        profile_bytes = sum(
            int(profile.get("bytes", 0) or 0) for profile in profiles
        )
        profiled_bytes_sum = int(
            workspace.get("profiled_bytes_sum", profile_bytes)
        )
        _require(
            profiled_bytes_sum == profile_bytes,
            f"{context} workspace profiled byte sum is inconsistent",
        )
        _require(
            workspace_bytes == profiled_bytes_sum,
            f"{context} workspace total does not match profiled bytes",
        )
        if "profiled_operator_count" in workspace:
            _require(
                int(workspace["profiled_operator_count"]) == len(profiles),
                f"{context} workspace profile count is inconsistent",
            )
        if "extrapolated_profile_count" in workspace:
            _require(
                0
                <= int(workspace["extrapolated_profile_count"])
                <= len(profiles),
                f"{context} extrapolated workspace profile count is invalid",
            )
        lifetime_model = workspace.get("lifetime_model")
        _require(
            lifetime_model in {None, "node_liveness.v1"},
            f"{context} workspace lifetime model is unsupported",
        )
        for profile_index, profile in enumerate(profiles):
            _require(
                isinstance(profile, dict)
                and int(profile.get("bytes", 0) or 0) > 0,
                f"{context} workspace profile {profile_index} is invalid",
            )
            if lifetime_model == "node_liveness.v1":
                _require(
                    profile.get("lifetime")
                    in {"graph_phase_persistent", "operator_local"},
                    f"{context} workspace profile {profile_index} has invalid lifetime",
                )
        if lifetime_model == "node_liveness.v1":
            _require(
                isinstance(workspace.get("graph_modeled_attention_operators"), dict),
                f"{context} graph-modeled attention operators are required",
            )
            peak_contribution = int(
                static.get("workspace_peak_contribution_bytes", -1)
            )
            _require(
                0 <= peak_contribution <= profiled_bytes_sum,
                f"{context} workspace peak contribution is invalid",
            )
            _require(
                peak_contribution
                == int(workspace.get("effective_peak_contribution_bytes", -1)),
                f"{context} workspace peak contribution fields disagree",
            )
            peak_candidate = workspace.get("peak_candidate")
            _require(
                isinstance(peak_candidate, dict)
                and int(peak_candidate.get("combined_live_bytes", -1)) >= 0,
                f"{context} workspace peak candidate is required",
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
            if phase_resolved:
                peak_by_stage = real.get("peak_by_stage")
                _require(
                    isinstance(peak_by_stage, dict),
                    f"{context}.real_cuda.peak_by_stage is required",
                )
                for stage_name in ("forward", "backward", "optimizer"):
                    stage = peak_by_stage.get(stage_name)
                    _require(
                        isinstance(stage, dict)
                        and int(stage.get("peak_allocated_bytes", 0) or 0) > 0,
                        f"{context}.real_cuda.peak_by_stage.{stage_name} is invalid",
                    )
                _require(
                    real.get("peak_stage") in {"forward", "backward", "optimizer"},
                    f"{context}.real_cuda.peak_stage is invalid",
                )
                if real.get("requested_peak_bytes") is not None:
                    _require(
                        real.get("peak_requested_stage")
                        in {"forward", "backward", "optimizer"},
                        f"{context}.real_cuda.peak_requested_stage is invalid",
                    )
                _require(
                    int(real.get("peak_allocated_bytes", 0) or 0)
                    == max(
                        int(peak_by_stage[name]["peak_allocated_bytes"])
                        for name in ("forward", "backward", "optimizer")
                    ),
                    f"{context}.real_cuda peak does not match stage peaks",
                )
                phase_comparison = comparison.get("phase_comparison")
                _require(
                    isinstance(phase_comparison, dict)
                    and isinstance(phase_comparison.get("graph"), dict)
                    and isinstance(phase_comparison.get("optimizer"), dict),
                    f"{context}.comparison.phase_comparison is required",
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
                if real.get("requested_peak_bytes") is not None:
                    _require(
                        float(requested_underestimate)
                        <= ns.max_underestimate_percent,
                        f"{context} requested-byte underestimate "
                        f"{float(requested_underestimate):.3f}% exceeds "
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
