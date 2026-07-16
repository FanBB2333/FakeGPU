#!/usr/bin/env python3
"""Combine repeated real-GPU calibration reports into an empirical lookup bundle."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

try:
    from .calibration_rtx3090ti import (
        _measurement_values,
        _summarize_numeric_samples,
        _workload_descriptor,
        _workload_signature,
    )
except ImportError:
    from calibration_rtx3090ti import (  # type: ignore[no-redef]
        _measurement_values,
        _summarize_numeric_samples,
        _workload_descriptor,
        _workload_signature,
    )


SCHEMA_VERSION = "real_gpu_calibration_bundle.v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", help="calibration_real_gpu.json report paths")
    parser.add_argument("--output", required=True, help="output JSON bundle path")
    parser.add_argument("--markdown", help="optional Markdown summary path")
    ns = parser.parse_args(argv)

    sources = [Path(value).resolve() for value in ns.reports]
    reports = [(path, _load_report(path)) for path in sources]
    bundle = aggregate_reports(reports)
    output = Path(ns.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    if ns.markdown:
        markdown = Path(ns.markdown)
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(render_markdown(bundle), encoding="utf-8")
        print(f"calibration bundle markdown: {markdown}")
    print(f"calibration bundle: {output}")
    return 0


def aggregate_reports(
    reports: list[tuple[Path, dict[str, Any]]],
) -> dict[str, Any]:
    if not reports:
        raise ValueError("at least one calibration report is required")

    grouped: dict[str, dict[str, Any]] = {}
    source_summaries: list[dict[str, Any]] = []
    for path, report in reports:
        _validate_source_report(path, report)
        gpu = dict(report.get("calibration_gpu") or {})
        profile = str(report.get("fakecuda_profile") or "")
        source_summaries.append(
            {
                "path": str(path),
                "gpu": gpu,
                "profile": profile,
                "software": dict(report.get("software") or {}),
                "sampling": dict(report.get("sampling") or {}),
            }
        )
        for item in report.get("workloads", []):
            if not isinstance(item, dict):
                continue
            real = item.get("real_cuda")
            fake = item.get("fakecuda_preflight")
            if not isinstance(real, dict) or not isinstance(fake, dict):
                continue
            name = str(item.get("name") or real.get("workload") or "")
            descriptor = item.get("workload_descriptor")
            if not isinstance(descriptor, dict):
                descriptor = _workload_descriptor(name, real)
            signature = str(item.get("workload_signature") or _workload_signature(descriptor))
            workload = grouped.setdefault(
                signature,
                {
                    "name": name,
                    "workload_signature": signature,
                    "workload_descriptor": descriptor,
                    "observations": [],
                },
            )
            workload["observations"].append(
                _build_observation(
                    path=path,
                    report=report,
                    item=item,
                    gpu=gpu,
                    profile=profile,
                    real=real,
                    fake=fake,
                )
            )

    workloads = sorted(
        grouped.values(),
        key=lambda item: (str(item.get("name")), str(item.get("workload_signature"))),
    )
    for workload in workloads:
        observations = workload.get("observations", [])
        workload["profiles"] = sorted(
            {
                str(item.get("profile"))
                for item in observations
                if isinstance(item, dict) and item.get("profile")
            }
        )
        workload["total_real_trials"] = sum(
            int(item.get("sample_count", 0) or 0)
            for item in observations
            if isinstance(item, dict)
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_unix": int(time.time()),
        "source_reports": source_summaries,
        "workloads": workloads,
        "notes": [
            "Each observation is scoped to one exact workload signature and one GPU profile.",
            "The physical upper bound prefers post-warmup NVML process peaks and records its fallback source; it is not a fitted universal factor.",
            "Use a matching profile and workload signature; do not extrapolate these values to different shapes or software stacks.",
        ],
    }


def _build_observation(
    *,
    path: Path,
    report: dict[str, Any],
    item: dict[str, Any],
    gpu: dict[str, Any],
    profile: str,
    real: dict[str, Any],
    fake: dict[str, Any],
) -> dict[str, Any]:
    real_peaks = _measurement_values(real, "peak_memory")
    reserved_peaks = _measurement_values(real, "peak_reserved_memory")
    requested_peaks = _measurement_values(real, "requested_peak_memory")
    fake_peak = int(fake.get("peak_memory", 0) or 0)
    missing_peaks = [max(0, peak - fake_peak) for peak in real_peaks]
    nvml_process_peaks = _nvml_trial_values(real, "peak_process_memory")
    nvml_process_deltas = _nvml_trial_values(real, "peak_process_delta_memory")
    nvml_device_peaks = _nvml_trial_values(real, "peak_device_used_memory")
    nvml_device_deltas = _nvml_trial_values(real, "peak_device_used_delta_memory")
    allocator_upper_bound = max(real_peaks)
    if nvml_process_peaks:
        physical_upper_bound = max(allocator_upper_bound, max(nvml_process_peaks))
        physical_upper_bound_source = "nvml_process_peak"
    elif nvml_device_deltas:
        physical_upper_bound = max(allocator_upper_bound, max(nvml_device_deltas))
        physical_upper_bound_source = "torch_allocator_peak_with_nvml_device_delta"
    else:
        physical_upper_bound = allocator_upper_bound
        physical_upper_bound_source = "torch_allocator_peak"
    observation: dict[str, Any] = {
        "source_report": str(path),
        "profile": profile,
        "gpu": gpu,
        "software": dict(report.get("software") or {}),
        "sample_count": len(real_peaks),
        "fakecuda_tracked_peak_memory": fake_peak,
        "real_cuda_peak_memory": _summary(real_peaks),
        "real_cuda_reserved_peak_memory": _summary(reserved_peaks),
        "real_cuda_requested_peak_memory": _summary(requested_peaks),
        "missing_peak_memory": _summary(missing_peaks),
        "empirical_real_peak_upper_bound_bytes": allocator_upper_bound,
        "empirical_missing_peak_upper_bound_bytes": max(missing_peaks),
        "empirical_physical_peak_upper_bound_bytes": physical_upper_bound,
        "empirical_physical_peak_upper_bound_source": physical_upper_bound_source,
        "empirical_physical_missing_peak_upper_bound_bytes": max(
            0,
            physical_upper_bound - fake_peak,
        ),
        "calibration_created_at_unix": report.get("created_at_unix"),
    }
    if nvml_process_peaks:
        observation["nvml_process_peak_memory"] = _summary(nvml_process_peaks)
    if nvml_process_deltas:
        observation["nvml_process_delta_memory"] = _summary(nvml_process_deltas)
    if nvml_device_peaks:
        observation["nvml_device_used_peak_memory"] = _summary(nvml_device_peaks)
    if nvml_device_deltas:
        observation["nvml_device_used_delta_memory"] = _summary(nvml_device_deltas)
    return observation


def _summary(values: list[int]) -> dict[str, int | float]:
    if not values:
        return {}
    return _summarize_numeric_samples(values, integer=True)


def _nvml_trial_values(payload: dict[str, Any], metric: str) -> list[int]:
    trials = payload.get("trials")
    if not isinstance(trials, list):
        nvml = payload.get("nvml")
        if isinstance(nvml, dict) and int(nvml.get(metric, 0) or 0) > 0:
            return [int(nvml[metric])]
        return []
    return [
        int(nvml[metric])
        for item in trials
        if isinstance(item, dict)
        and isinstance((nvml := item.get("nvml")), dict)
        and nvml.get("status") == "available"
        and int(nvml.get(metric, 0) or 0) > 0
    ]


def _load_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"calibration report not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"calibration report must be an object: {path}")
    return payload


def _validate_source_report(path: Path, report: dict[str, Any]) -> None:
    if report.get("schema_version") not in {
        "real_gpu_calibration.v1",
        "rtx3090ti_calibration.v1",
    }:
        raise ValueError(f"unsupported calibration schema in {path}: {report.get('schema_version')!r}")
    if report.get("status") != "PASS_CALIBRATED":
        raise ValueError(f"calibration report did not pass: {path} ({report.get('status')!r})")
    if not report.get("fakecuda_profile"):
        raise ValueError(f"calibration report has no fakecuda_profile: {path}")
    workloads = report.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        raise ValueError(f"calibration report has no workloads: {path}")


def render_markdown(bundle: dict[str, Any]) -> str:
    lines = [
        "# Empirical Real-GPU Memory Calibration Bundle",
        "",
        f"**Sources:** {len(bundle.get('source_reports', []))}",
        "",
        "| workload | profile | GPU | trials | allocator median | allocator upper | physical upper | physical source | fakecuda tracked | physical missing | NVML device delta |",
        "|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for workload in bundle.get("workloads", []):
        if not isinstance(workload, dict):
            continue
        for observation in workload.get("observations", []):
            if not isinstance(observation, dict):
                continue
            peak = observation.get("real_cuda_peak_memory") or {}
            nvml_device_delta = observation.get("nvml_device_used_delta_memory") or {}
            nvml_device_delta_text = (
                _fmt_bytes(int(nvml_device_delta.get("max", 0)))
                if nvml_device_delta
                else "n/a"
            )
            gpu = observation.get("gpu") or {}
            lines.append(
                "| `{name}` | `{profile}` | `{gpu}` | {trials} | {median} | {allocator_upper} | {physical_upper} | `{physical_source}` | {fake} | {physical_missing} | {nvml_device_delta} |".format(
                    name=workload.get("name"),
                    profile=observation.get("profile"),
                    gpu=gpu.get("name"),
                    trials=observation.get("sample_count"),
                    median=_fmt_bytes(int(peak.get("median", 0))),
                    allocator_upper=_fmt_bytes(
                        int(observation.get("empirical_real_peak_upper_bound_bytes", 0))
                    ),
                    physical_upper=_fmt_bytes(
                        int(observation.get("empirical_physical_peak_upper_bound_bytes", 0))
                    ),
                    physical_source=observation.get("empirical_physical_peak_upper_bound_source"),
                    fake=_fmt_bytes(int(observation.get("fakecuda_tracked_peak_memory", 0))),
                    physical_missing=_fmt_bytes(
                        int(observation.get("empirical_physical_missing_peak_upper_bound_bytes", 0))
                    ),
                    nvml_device_delta=nvml_device_delta_text,
                )
            )
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in bundle.get("notes", []))
    lines.append("")
    return "\n".join(lines)


def _fmt_bytes(value: int) -> str:
    amount = float(max(0, int(value)))
    for unit in ("B", "KiB", "MiB", "GiB"):
        if amount < 1024.0 or unit == "GiB":
            return f"{int(amount)} B" if unit == "B" else f"{amount:.2f} {unit}"
        amount /= 1024.0
    return f"{amount:.2f} GiB"


if __name__ == "__main__":
    raise SystemExit(main())
