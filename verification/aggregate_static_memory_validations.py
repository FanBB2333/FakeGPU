#!/usr/bin/env python3
"""Aggregate static-memory validation reports across GPU/software profiles."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "static_memory_validation_bundle.v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    ns = parser.parse_args(argv)

    reports = [(path, _load_report(path)) for path in ns.reports]
    bundle = aggregate_reports(reports)
    output = ns.output.resolve()
    markdown = (ns.markdown or output.with_suffix(".md")).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    markdown.write_text(render_markdown(bundle), encoding="utf-8")
    print(f"static memory validation bundle: {output}")
    print(f"static memory validation bundle markdown: {markdown}")
    return 0


def aggregate_reports(
    reports: list[tuple[Path, dict[str, Any]]],
) -> dict[str, Any]:
    if not reports:
        raise ValueError("at least one validation report is required")

    source_reports: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    all_comparisons: list[dict[str, Any]] = []
    gpu_keys: set[tuple[str, str]] = set()

    for path, report in reports:
        _validate_report(path, report)
        gpu = dict(report.get("gpu") or {})
        software = dict(report.get("software") or {})
        backend_calibration = dict(report.get("backend_calibration") or {})
        gpu_key = (str(gpu.get("name") or ""), str(gpu.get("compute_capability") or ""))
        gpu_keys.add(gpu_key)
        source_reports.append(
            {
                "path": str(path),
                "gpu": gpu,
                "software": software,
                "backend_calibration": backend_calibration,
                "sampling": dict(report.get("sampling") or {}),
                "summary": dict(report.get("summary") or {}),
            }
        )

        for item in report["workloads"]:
            name = str(item.get("name") or "")
            parameters = dict(item.get("parameters") or {})
            parameter_signature = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
            key = (name, parameter_signature)
            workload = grouped.setdefault(
                key,
                {
                    "name": name,
                    "family": item.get("family"),
                    "parameters": parameters,
                    "observations": [],
                },
            )
            static = dict(item.get("static_estimate") or {})
            graph = dict(static.get("graph") or {})
            comparison = dict(item.get("comparison") or {})
            observation = {
                "source_report": str(path),
                "gpu": gpu,
                "software": software,
                "backend_calibration": backend_calibration,
                "static_estimated_peak_bytes": int(static.get("estimated_peak_bytes", 0) or 0),
                "profiled_workspace_bytes": int(
                    static.get("workspace_estimate_bytes", 0) or 0
                ),
                "graph_fingerprint": str(graph.get("graph_fingerprint") or ""),
                "graph_node_count": int(graph.get("node_count", 0) or 0),
                "graph_alias_node_count": int(graph.get("alias_node_count", 0) or 0),
                "real_peak_allocated_bytes": int(
                    (item.get("real_cuda") or {}).get("peak_allocated_bytes", 0) or 0
                ),
                "comparison": comparison,
            }
            workload["observations"].append(observation)
            all_comparisons.append(comparison)

    workloads = []
    for workload in grouped.values():
        observations = workload["observations"]
        present_in_all_reports = len(observations) == len(source_reports)
        static_peaks = {
            int(item.get("static_estimated_peak_bytes", 0) or 0)
            for item in observations
        }
        fingerprints = {
            str(item.get("graph_fingerprint") or "")
            for item in observations
            if item.get("graph_fingerprint")
        }
        workload["present_in_all_reports"] = present_in_all_reports
        workload["static_peak_consistent"] = (
            present_in_all_reports and len(static_peaks) == 1
        )
        workload["graph_fingerprint_consistent"] = (
            present_in_all_reports and len(fingerprints) == 1
        )
        workload["gpu_observation_count"] = len(observations)
        workload["max_underestimate_percent"] = max(
            float((item.get("comparison") or {}).get("underestimate_percent", 0.0) or 0.0)
            for item in observations
        )
        workload["max_absolute_error_percent"] = max(
            float((item.get("comparison") or {}).get("absolute_error_percent", 0.0) or 0.0)
            for item in observations
        )
        workloads.append(workload)

    workloads.sort(key=lambda item: (str(item.get("family")), str(item.get("name"))))
    absolute_errors = [
        float(item.get("absolute_error_percent", 0.0) or 0.0)
        for item in all_comparisons
    ]
    underestimates = [
        float(item.get("underestimate_percent", 0.0) or 0.0)
        for item in all_comparisons
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_unix": int(time.time()),
        "source_reports": source_reports,
        "summary": {
            "source_report_count": len(source_reports),
            "gpu_profile_count": len(gpu_keys),
            "workload_count": len(workloads),
            "observation_count": len(all_comparisons),
            "all_static_peaks_consistent": all(
                bool(item["static_peak_consistent"]) for item in workloads
            ),
            "all_graph_fingerprints_consistent": all(
                bool(item["graph_fingerprint_consistent"]) for item in workloads
            ),
            "underestimated_observation_count": sum(value > 0 for value in underestimates),
            "max_underestimate_percent": max(underestimates) if underestimates else None,
            "median_absolute_error_percent": (
                statistics.median(absolute_errors) if absolute_errors else None
            ),
            "max_absolute_error_percent": max(absolute_errors) if absolute_errors else None,
        },
        "workloads": workloads,
        "notes": [
            "Static peak consistency checks the device-independent ATen storage estimate across reports.",
            "Backend-resident calibration remains scoped to each GPU and software stack.",
            "Validation covers only the recorded workload parameters, dtypes, optimizers, and graph fingerprints.",
        ],
    }


def render_markdown(bundle: dict[str, Any]) -> str:
    summary = bundle.get("summary") or {}
    lines = [
        "# Cross-GPU Static Memory Validation",
        "",
        f"**Source reports:** `{summary.get('source_report_count')}`",
        f"**GPU profiles:** `{summary.get('gpu_profile_count')}`",
        f"**Workloads:** `{summary.get('workload_count')}`",
        f"**Static peaks consistent:** `{summary.get('all_static_peaks_consistent')}`",
        f"**Graph fingerprints consistent:** `{summary.get('all_graph_fingerprints_consistent')}`",
        f"**Maximum underestimate:** `{_fmt_percent(summary.get('max_underestimate_percent'))}`",
        f"**Maximum absolute error:** `{_fmt_percent(summary.get('max_absolute_error_percent'))}`",
        "",
        "| workload | GPU | static peak | profiled workspace | backend resident | calibrated peak | real peak | error | underestimate | static consistent | graph consistent |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for workload in bundle.get("workloads", []):
        static_consistent = bool(workload.get("static_peak_consistent"))
        graph_consistent = bool(workload.get("graph_fingerprint_consistent"))
        for observation in workload.get("observations", []):
            gpu = observation.get("gpu") or {}
            backend = observation.get("backend_calibration") or {}
            comparison = observation.get("comparison") or {}
            lines.append(
                "| `{name}` | `{gpu}` | {static} | {workspace} | {backend} | {calibrated} | {real} | {error} | {underestimate} | `{static_consistent}` | `{graph_consistent}` |".format(
                    name=workload.get("name"),
                    gpu=gpu.get("name"),
                    static=_fmt_bytes(
                        int(observation.get("static_estimated_peak_bytes", 0) or 0)
                    ),
                    workspace=_fmt_bytes(
                        int(observation.get("profiled_workspace_bytes", 0) or 0)
                    ),
                    backend=_fmt_bytes(
                        int(backend.get("resident_allocated_bytes", 0) or 0)
                    ),
                    calibrated=_fmt_bytes(
                        int(comparison.get("calibrated_estimated_peak_bytes", 0) or 0)
                    ),
                    real=_fmt_bytes(
                        int(observation.get("real_peak_allocated_bytes", 0) or 0)
                    ),
                    error=_fmt_percent(comparison.get("absolute_error_percent")),
                    underestimate=_fmt_percent(comparison.get("underestimate_percent")),
                    static_consistent="yes" if static_consistent else "no",
                    graph_consistent="yes" if graph_consistent else "no",
                )
            )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "- Static values are compared only for identical workload parameter sets.",
            "- Backend-resident values are retained per GPU/software profile.",
            "- Results do not extrapolate to untested operators, shapes, optimizer implementations, or distributed strategies.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_report(path: Path, report: dict[str, Any]) -> None:
    if report.get("schema_version") != "static_memory_validation.v1":
        raise ValueError(f"unsupported schema in {path}: {report.get('schema_version')!r}")
    if report.get("status") != "PASS_MEASURED":
        raise ValueError(f"validation report did not pass: {path} ({report.get('status')!r})")
    if not isinstance(report.get("gpu"), dict):
        raise ValueError(f"validation report has no GPU fingerprint: {path}")
    workloads = report.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        raise ValueError(f"validation report has no workloads: {path}")
    for item in workloads:
        if not isinstance(item, dict):
            raise ValueError(f"validation report contains a non-object workload: {path}")
        if not isinstance(item.get("static_estimate"), dict):
            raise ValueError(f"workload has no static estimate in {path}")
        if not isinstance(item.get("real_cuda"), dict):
            raise ValueError(f"workload has no real CUDA measurement in {path}")
        if not isinstance(item.get("comparison"), dict):
            raise ValueError(f"workload has no comparison in {path}")


def _load_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"validation report not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"validation report must contain an object: {path}")
    return payload


def _fmt_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    unit = units[0]
    for candidate in units:
        unit = candidate
        if abs(amount) < 1024.0 or candidate == units[-1]:
            break
        amount /= 1024.0
    return f"{amount:.2f} {unit}"


def _fmt_percent(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}%"


if __name__ == "__main__":
    raise SystemExit(main())
