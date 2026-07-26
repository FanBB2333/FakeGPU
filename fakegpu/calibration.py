from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .structured_io import StructuredDataError, load_mapping, write_json


COMPARISON_SCHEMA_VERSION = "fakegpu.calibration_comparison.v1"
BUNDLE_SCHEMA_VERSION = "fakegpu.workload_calibration_bundle.v1"
VERIFICATION_SCHEMA_VERSION = "fakegpu.calibration_verification.v1"


class CalibrationError(ValueError):
    pass


def compare_memory_reports(
    prediction: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    workload: str | None = None,
) -> dict[str, Any]:
    """Compare phase-aware memory predictions with observed peaks.

    The extractor accepts FakeGPU static/LLM/preflight reports, real-GPU
    calibration reports and bundles, and a small generic ``memory_timeline``
    contract. Matching is exact by phase first and falls back to each report's
    canonical peak when the schemas expose different phase names.
    """

    predicted = _memory_points(prediction, role="prediction", workload=workload)
    observed = _memory_points(observation, role="observation", workload=workload)
    pairs = _match_points(predicted, observed)
    if not pairs:
        raise CalibrationError(
            "prediction and observation contain no comparable memory peaks"
        )

    comparisons = []
    for phase, predicted_point, observed_point in pairs:
        predicted_bytes = int(predicted_point["bytes"])
        observed_bytes = int(observed_point["bytes"])
        signed_error = predicted_bytes - observed_bytes
        absolute_error = abs(signed_error)
        interval = predicted_point.get("interval")
        within_interval = None
        if isinstance(interval, Mapping):
            lower = interval.get("lower")
            upper = interval.get("upper")
            if lower is not None and upper is not None:
                within_interval = int(lower) <= observed_bytes <= int(upper)
        comparisons.append(
            {
                "phase": phase,
                "predicted_bytes": predicted_bytes,
                "observed_bytes": observed_bytes,
                "signed_error_bytes": signed_error,
                "absolute_error_bytes": absolute_error,
                "absolute_percentage_error": (
                    absolute_error / observed_bytes
                    if observed_bytes > 0
                    else None
                ),
                "prediction_to_observation_ratio": (
                    predicted_bytes / observed_bytes
                    if observed_bytes > 0
                    else None
                ),
                "prediction_source": predicted_point["source"],
                "observation_source": observed_point["source"],
                "prediction_interval_bytes": (
                    dict(interval) if isinstance(interval, Mapping) else None
                ),
                "observation_within_prediction_interval": within_interval,
            }
        )

    absolute_errors = [
        int(item["absolute_error_bytes"]) for item in comparisons
    ]
    observed_values = [int(item["observed_bytes"]) for item in comparisons]
    signed_errors = [int(item["signed_error_bytes"]) for item in comparisons]
    mape_values = [
        float(item["absolute_percentage_error"])
        for item in comparisons
        if item["absolute_percentage_error"] is not None
    ]
    underestimates_percent = [
        max(
            0.0,
            (
                int(item["observed_bytes"]) - int(item["predicted_bytes"])
            )
            / int(item["observed_bytes"])
            * 100,
        )
        for item in comparisons
        if int(item["observed_bytes"]) > 0
    ]
    interval_results = [
        bool(item["observation_within_prediction_interval"])
        for item in comparisons
        if item["observation_within_prediction_interval"] is not None
    ]
    recommended_margin = max(0, max(-value for value in signed_errors))
    ratios = [
        observed_value / int(item["predicted_bytes"])
        for item, observed_value in zip(comparisons, observed_values)
        if int(item["predicted_bytes"]) > 0
    ]
    recommended_factor = max([1.0, *ratios])
    attribution = _attribute_error(
        prediction=prediction,
        observation=observation,
        comparisons=comparisons,
    )
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "workload": workload,
        "prediction_schema_version": prediction.get("schema_version"),
        "observation_schema_version": observation.get("schema_version"),
        "dimensions": {
            "prediction": _report_dimensions(prediction),
            "observation": _report_dimensions(observation),
        },
        "comparisons": comparisons,
        "summary": {
            "phase_count": len(comparisons),
            "mean_absolute_error_bytes": int(
                round(statistics.fmean(absolute_errors))
            ),
            "max_absolute_error_bytes": max(absolute_errors),
            "mean_absolute_percentage_error": (
                statistics.fmean(mape_values) if mape_values else None
            ),
            "median_absolute_percentage_error_percent": (
                statistics.median(mape_values) * 100
                if mape_values
                else None
            ),
            "p95_absolute_percentage_error_percent": (
                _percentile(mape_values, 0.95) * 100
                if mape_values
                else None
            ),
            "max_absolute_percentage_error_percent": (
                max(mape_values) * 100 if mape_values else None
            ),
            "max_underestimate_percent": max(
                underestimates_percent,
                default=0.0,
            ),
            "prediction_interval_coverage_percent": (
                sum(interval_results) / len(interval_results) * 100
                if interval_results
                else None
            ),
            "worst_phase": max(
                comparisons,
                key=lambda item: int(item["absolute_error_bytes"]),
            )["phase"],
            "underprediction_phase_count": sum(
                value < 0 for value in signed_errors
            ),
            "overprediction_phase_count": sum(
                value > 0 for value in signed_errors
            ),
            "recommended_memory_safety_margin_bytes": recommended_margin,
            "recommended_memory_safety_factor": recommended_factor,
        },
        "error_attribution": attribution,
        "calibration_application": {
            "preflight_flags": [
                "--memory-safety-margin",
                str(recommended_margin),
                "--memory-safety-factor",
                f"{recommended_factor:.9g}",
            ],
            "scope": (
                "Apply only to the same workload signature, software stack, "
                "dtype, shapes, and GPU profile."
            ),
        },
    }


def verify_calibration_reports(
    reports: Sequence[Mapping[str, Any]],
    *,
    labels: Sequence[str] | None = None,
    max_underestimate_percent: float = 5.0,
    max_absolute_percentage_error_percent: float | None = None,
    min_interval_coverage_percent: float | None = None,
    capacity_bytes: int | None = None,
    max_false_safe_count: int = 0,
    min_comparisons: int = 1,
    require_matching_dimensions: bool = True,
) -> dict[str, Any]:
    """Evaluate comparison reports against publication and OOM-risk gates."""

    if not reports:
        raise CalibrationError("at least one comparison report is required")
    if labels is not None and len(labels) != len(reports):
        raise CalibrationError("labels and reports must have the same length")
    _validate_percentage(
        "max_underestimate_percent",
        max_underestimate_percent,
    )
    if max_absolute_percentage_error_percent is not None:
        _validate_percentage(
            "max_absolute_percentage_error_percent",
            max_absolute_percentage_error_percent,
        )
    if min_interval_coverage_percent is not None:
        _validate_percentage(
            "min_interval_coverage_percent",
            min_interval_coverage_percent,
            bounded=True,
        )
    if (
        capacity_bytes is not None
        and (
            not isinstance(capacity_bytes, int)
            or isinstance(capacity_bytes, bool)
            or capacity_bytes <= 0
        )
    ):
        raise CalibrationError("capacity_bytes must be a positive integer")
    if (
        not isinstance(max_false_safe_count, int)
        or isinstance(max_false_safe_count, bool)
        or max_false_safe_count < 0
    ):
        raise CalibrationError(
            "max_false_safe_count must be a non-negative integer"
        )
    if (
        not isinstance(min_comparisons, int)
        or isinstance(min_comparisons, bool)
        or min_comparisons <= 0
    ):
        raise CalibrationError("min_comparisons must be a positive integer")

    absolute_percentage_errors: list[float] = []
    underestimates: list[float] = []
    interval_results: list[bool] = []
    false_safe_events: list[dict[str, Any]] = []
    report_results: list[dict[str, Any]] = []
    dimension_mismatch_count = 0

    for index, report in enumerate(reports):
        label = str(labels[index]) if labels is not None else f"report_{index}"
        if report.get("schema_version") != COMPARISON_SCHEMA_VERSION:
            raise CalibrationError(
                f"{label}: expected schema_version "
                f"{COMPARISON_SCHEMA_VERSION!r}"
            )
        raw_comparisons = report.get("comparisons")
        if not isinstance(raw_comparisons, list) or not raw_comparisons:
            raise CalibrationError(f"{label}: comparison report is empty")

        dimension_mismatches = _dimension_mismatches(
            report.get("dimensions")
        )
        dimension_mismatch_count += len(dimension_mismatches)
        valid_comparison_count = 0
        for item in raw_comparisons:
            if not isinstance(item, Mapping):
                raise CalibrationError(
                    f"{label}: comparison entries must be objects"
                )
            predicted_bytes = _required_non_negative_integer(
                item,
                "predicted_bytes",
                label=label,
            )
            observed_bytes = _required_non_negative_integer(
                item,
                "observed_bytes",
                label=label,
            )
            valid_comparison_count += 1
            if observed_bytes > 0:
                absolute_percentage_errors.append(
                    abs(predicted_bytes - observed_bytes)
                    / observed_bytes
                    * 100
                )
                underestimates.append(
                    max(
                        0.0,
                        (observed_bytes - predicted_bytes)
                        / observed_bytes
                        * 100,
                    )
                )
            within_interval = item.get(
                "observation_within_prediction_interval"
            )
            if isinstance(within_interval, bool):
                interval_results.append(within_interval)
            if (
                capacity_bytes is not None
                and predicted_bytes <= capacity_bytes < observed_bytes
            ):
                false_safe_events.append(
                    {
                        "report": label,
                        "phase": str(item.get("phase") or "unknown"),
                        "predicted_bytes": predicted_bytes,
                        "observed_bytes": observed_bytes,
                        "capacity_bytes": capacity_bytes,
                    }
                )
        report_results.append(
            {
                "id": label,
                "workload": report.get("workload"),
                "comparison_count": valid_comparison_count,
                "dimension_mismatches": dimension_mismatches,
            }
        )

    comparison_count = sum(
        int(item["comparison_count"]) for item in report_results
    )
    max_underestimate = max(underestimates, default=0.0)
    maximum_ape = max(absolute_percentage_errors, default=None)
    interval_coverage = (
        sum(interval_results) / len(interval_results) * 100
        if interval_results
        else None
    )
    failures: list[dict[str, Any]] = []
    if comparison_count < min_comparisons:
        failures.append(
            {
                "gate": "minimum_comparisons",
                "actual": comparison_count,
                "required": min_comparisons,
            }
        )
    if not absolute_percentage_errors:
        failures.append(
            {
                "gate": "positive_observation_count",
                "actual": 0,
                "minimum": 1,
            }
        )
    elif max_underestimate > max_underestimate_percent:
        failures.append(
            {
                "gate": "maximum_underestimate_percent",
                "actual": max_underestimate,
                "maximum": max_underestimate_percent,
            }
        )
    if (
        max_absolute_percentage_error_percent is not None
        and maximum_ape is not None
        and maximum_ape > max_absolute_percentage_error_percent
    ):
        failures.append(
            {
                "gate": "maximum_absolute_percentage_error_percent",
                "actual": maximum_ape,
                "maximum": max_absolute_percentage_error_percent,
            }
        )
    if min_interval_coverage_percent is not None:
        if interval_coverage is None:
            failures.append(
                {
                    "gate": "minimum_prediction_interval_coverage_percent",
                    "actual": None,
                    "minimum": min_interval_coverage_percent,
                    "reason": "no prediction intervals were reported",
                }
            )
        elif interval_coverage < min_interval_coverage_percent:
            failures.append(
                {
                    "gate": "minimum_prediction_interval_coverage_percent",
                    "actual": interval_coverage,
                    "minimum": min_interval_coverage_percent,
                }
            )
    if (
        capacity_bytes is not None
        and len(false_safe_events) > max_false_safe_count
    ):
        failures.append(
            {
                "gate": "maximum_false_safe_count",
                "actual": len(false_safe_events),
                "maximum": max_false_safe_count,
            }
        )
    if require_matching_dimensions and dimension_mismatch_count:
        failures.append(
            {
                "gate": "matching_workload_dimensions",
                "actual": dimension_mismatch_count,
                "maximum": 0,
            }
        )

    return {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "passed" if not failures else "failed",
        "report_count": len(reports),
        "comparison_count": comparison_count,
        "thresholds": {
            "max_underestimate_percent": max_underestimate_percent,
            "max_absolute_percentage_error_percent": (
                max_absolute_percentage_error_percent
            ),
            "min_interval_coverage_percent": (
                min_interval_coverage_percent
            ),
            "capacity_bytes": capacity_bytes,
            "max_false_safe_count": max_false_safe_count,
            "min_comparisons": min_comparisons,
            "require_matching_dimensions": require_matching_dimensions,
        },
        "metrics": {
            "positive_observation_count": len(
                absolute_percentage_errors
            ),
            "maximum_underestimate_percent": max_underestimate,
            "median_absolute_percentage_error_percent": (
                statistics.median(absolute_percentage_errors)
                if absolute_percentage_errors
                else None
            ),
            "p95_absolute_percentage_error_percent": (
                _percentile(absolute_percentage_errors, 0.95)
                if absolute_percentage_errors
                else None
            ),
            "maximum_absolute_percentage_error_percent": maximum_ape,
            "prediction_interval_count": len(interval_results),
            "prediction_interval_coverage_percent": interval_coverage,
            "false_safe_count": (
                len(false_safe_events)
                if capacity_bytes is not None
                else None
            ),
            "dimension_mismatch_count": dimension_mismatch_count,
        },
        "false_safe_events": false_safe_events,
        "reports": report_results,
        "failures": failures,
    }


def build_workload_calibration_bundle(
    reports: Sequence[Mapping[str, Any]],
    *,
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Create a reusable bundle from comparison or raw calibration reports."""

    if not reports:
        raise CalibrationError("at least one calibration report is required")
    if labels is not None and len(labels) != len(reports):
        raise CalibrationError("labels and reports must have the same length")

    entries: list[dict[str, Any]] = []
    for index, report in enumerate(reports):
        label = (
            str(labels[index])
            if labels is not None
            else f"report_{index}"
        )
        schema = str(report.get("schema_version") or "")
        if schema == COMPARISON_SCHEMA_VERSION:
            comparisons = report.get("comparisons")
            if not isinstance(comparisons, list):
                raise CalibrationError(
                    f"{label}: comparison report has no comparisons"
                )
            entries.append(
                {
                    "id": label,
                    "workload": report.get("workload"),
                    "prediction_schema_version": report.get(
                        "prediction_schema_version"
                    ),
                    "observation_schema_version": report.get(
                        "observation_schema_version"
                    ),
                    "dimensions": dict(report.get("dimensions") or {}),
                    "comparisons": [
                        dict(item)
                        for item in comparisons
                        if isinstance(item, Mapping)
                    ],
                    "recommendation": dict(report.get("summary") or {}),
                }
            )
            continue

        points = _memory_points(report, role="observation", workload=None)
        entries.append(
            {
                "id": label,
                "workload": report.get("workload"),
                "source_schema_version": report.get("schema_version"),
                "dimensions": _report_dimensions(report),
                "observations": [
                    {
                        "phase": phase,
                        "bytes": int(point["bytes"]),
                        "source": point["source"],
                    }
                    for phase, point in sorted(points.items())
                ],
            }
        )

    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "created_at_unix": int(time.time()),
        "entries": entries,
        "index_dimensions": [
            "workload",
            "gpu_profile",
            "compute_capability",
            "torch_version",
            "cuda_version",
            "dtype",
            "shape",
        ],
        "notes": [
            "Entries are exact-scope evidence, not universal correction factors.",
            "A consumer should reject mismatched workload, stack, dtype, shape, or profile dimensions.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fakegpu calibrate",
        description="Compare memory predictions with observations or bundle calibration evidence.",
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare one predicted report with one observed report.",
    )
    compare_parser.add_argument("prediction")
    compare_parser.add_argument("observation")
    compare_parser.add_argument("--workload")
    compare_parser.add_argument(
        "--json",
        dest="json_path",
        nargs="?",
        const="-",
        help="Write JSON to PATH, or stdout when PATH is omitted.",
    )

    bundle_parser = subparsers.add_parser(
        "bundle",
        help="Build an indexed workload calibration bundle.",
    )
    bundle_parser.add_argument("reports", nargs="+")
    bundle_parser.add_argument("--output", required=True)

    verify_parser = subparsers.add_parser(
        "verify",
        help="Apply reliability and OOM-risk gates to comparison reports.",
    )
    verify_parser.add_argument("reports", nargs="+")
    verify_parser.add_argument(
        "--max-underestimate-percent",
        type=float,
        default=5.0,
    )
    verify_parser.add_argument(
        "--max-absolute-percentage-error-percent",
        type=float,
    )
    verify_parser.add_argument(
        "--min-interval-coverage-percent",
        type=float,
    )
    verify_parser.add_argument("--capacity-bytes", type=int)
    verify_parser.add_argument(
        "--max-false-safe-count",
        type=int,
        default=0,
    )
    verify_parser.add_argument(
        "--min-comparisons",
        type=int,
        default=1,
    )
    verify_parser.add_argument(
        "--allow-dimension-mismatch",
        action="store_true",
    )
    verify_parser.add_argument(
        "--json",
        dest="json_path",
        nargs="?",
        const="-",
        help="Write JSON to PATH, or stdout when PATH is omitted.",
    )
    args = parser.parse_args(argv)

    try:
        if args.action == "compare":
            prediction = load_mapping(args.prediction)
            observation = load_mapping(args.observation)
            report = compare_memory_reports(
                prediction,
                observation,
                workload=args.workload,
            )
            if args.json_path:
                payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
                if args.json_path == "-":
                    print(payload, end="")
                else:
                    output = write_json(args.json_path, report)
                    print(f"Calibration comparison: {output}")
            else:
                _print_comparison(report)
            return 0

        if args.action == "verify":
            paths = [
                Path(value).expanduser().resolve()
                for value in args.reports
            ]
            reports = [load_mapping(path) for path in paths]
            report = verify_calibration_reports(
                reports,
                labels=[path.stem for path in paths],
                max_underestimate_percent=args.max_underestimate_percent,
                max_absolute_percentage_error_percent=(
                    args.max_absolute_percentage_error_percent
                ),
                min_interval_coverage_percent=(
                    args.min_interval_coverage_percent
                ),
                capacity_bytes=args.capacity_bytes,
                max_false_safe_count=args.max_false_safe_count,
                min_comparisons=args.min_comparisons,
                require_matching_dimensions=(
                    not args.allow_dimension_mismatch
                ),
            )
            if args.json_path:
                payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
                if args.json_path == "-":
                    print(payload, end="")
                else:
                    output = write_json(args.json_path, report)
                    print(f"Calibration verification: {output}")
            else:
                _print_verification(report)
            return 0 if report["status"] == "passed" else 1

        paths = [Path(value).expanduser().resolve() for value in args.reports]
        reports = [load_mapping(path) for path in paths]
        bundle = build_workload_calibration_bundle(
            reports,
            labels=[path.stem for path in paths],
        )
        output = write_json(args.output, bundle)
        print(f"Calibration bundle: {output}")
        return 0
    except (
        CalibrationError,
        FileNotFoundError,
        OSError,
        StructuredDataError,
        ValueError,
    ) as exc:
        parser.exit(2, f"fakegpu calibrate: {exc}\n")


def _memory_points(
    report: Mapping[str, Any],
    *,
    role: str,
    workload: str | None,
) -> dict[str, dict[str, Any]]:
    schema = str(report.get("schema_version") or "")
    if schema == "real_gpu_calibration_bundle.v1":
        return _bundle_points(report, workload=workload)
    if schema in {"real_gpu_calibration.v1", "rtx3090ti_calibration.v1"}:
        return _real_calibration_points(report, workload=workload)

    timeline = report.get("memory_timeline")
    if isinstance(timeline, Mapping):
        phases = timeline.get("phases")
        if isinstance(phases, list):
            points = {}
            for item in phases:
                if not isinstance(item, Mapping):
                    continue
                phase = str(item.get("phase") or item.get("name") or "")
                value = _first_integer(
                    item,
                    (
                        "process_peak_bytes",
                        "peak_bytes",
                        "allocated_bytes",
                        "bytes",
                    ),
                )
                if phase and value is not None:
                    point: dict[str, Any] = {
                        "bytes": value,
                        "source": f"memory_timeline.{phase}",
                    }
                    interval = item.get("interval_bytes")
                    if isinstance(interval, Mapping):
                        point["interval"] = dict(interval)
                    points[phase] = point
            if points:
                return points

    if schema == "static_memory_estimate.v1":
        return _static_memory_points(report)
    if schema == "fakegpu.llm_inference_estimate.v1":
        return _llm_memory_points(report)

    devices = report.get("devices")
    if isinstance(devices, list):
        points = {}
        for index, item in enumerate(devices):
            if not isinstance(item, Mapping):
                continue
            device = item.get("index", index)
            value = _first_integer(
                item,
                (
                    "peak_memory",
                    "peak_memory_bytes",
                    "empirical_calibration_peak_memory",
                ),
            )
            if value is not None:
                points[f"device_{device}"] = {
                    "bytes": value,
                    "source": f"devices[{index}]",
                }
        if points:
            return points

    value = _first_integer(
        report,
        (
            "estimated_process_peak_bytes",
            "estimated_peak_bytes",
            "peak_memory",
            "peak_memory_bytes",
            "observed_peak_bytes",
        ),
    )
    if value is not None:
        return {
            "peak": {
                "bytes": value,
                "source": f"{role}.canonical_peak",
            }
        }
    raise CalibrationError(
        f"unsupported {role} memory report schema {schema!r}"
    )


def _static_memory_points(
    report: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    fields = {
        "first_step": "first_step_estimated_peak_bytes",
        "steady_state": "steady_state_estimated_peak_bytes",
        "graph": "graph_phase_peak_bytes",
        "optimizer": "optimizer_phase_peak_bytes",
    }
    points: dict[str, dict[str, Any]] = {}
    for phase, field in fields.items():
        value = report.get(field)
        if isinstance(value, int) and not isinstance(value, bool):
            point: dict[str, Any] = {
                "bytes": value,
                "source": field,
            }
            interval = report.get(
                {
                    "first_step": "first_step_estimated_peak_interval_bytes",
                    "steady_state": "estimated_peak_interval_bytes",
                    "graph": "graph_phase_peak_interval_bytes",
                }.get(phase, "")
            )
            if isinstance(interval, Mapping):
                point["interval"] = dict(interval)
            points[phase] = point
    return points


def _llm_memory_points(
    report: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    memory = report.get("memory")
    if not isinstance(memory, Mapping):
        raise CalibrationError("LLM estimate has no memory section")
    overhead = int(memory.get("runtime_overhead_bytes", 0) or 0)
    result = {}
    for phase, field in (
        ("prefill", "estimated_prefill_tensor_peak_bytes"),
        ("decode", "estimated_decode_tensor_peak_bytes"),
    ):
        value = memory.get(field)
        if isinstance(value, int) and not isinstance(value, bool):
            result[phase] = {
                "bytes": value + overhead,
                "source": f"memory.{field}+runtime_overhead_bytes",
            }
    process_peak = memory.get("estimated_process_peak_bytes")
    if isinstance(process_peak, int) and not isinstance(process_peak, bool):
        result["peak"] = {
            "bytes": process_peak,
            "source": "memory.estimated_process_peak_bytes",
        }
    return result


def _real_calibration_points(
    report: Mapping[str, Any],
    *,
    workload: str | None,
) -> dict[str, dict[str, Any]]:
    workloads = report.get("workloads")
    if not isinstance(workloads, list):
        raise CalibrationError("real-GPU calibration report has no workloads")
    selected = [
        item
        for item in workloads
        if isinstance(item, Mapping)
        and (
            workload is None
            or workload
            in {
                str(item.get("name") or ""),
                str(item.get("workload_signature") or ""),
            }
        )
    ]
    if workload is None and len(selected) != 1:
        raise CalibrationError(
            "calibration report contains multiple workloads; pass --workload"
        )
    if len(selected) != 1:
        raise CalibrationError(f"workload {workload!r} was not found uniquely")
    real = selected[0].get("real_cuda")
    if not isinstance(real, Mapping):
        raise CalibrationError("selected workload has no real_cuda section")

    points: dict[str, dict[str, Any]] = {}
    for phase, field in (
        ("forward", "forward_peak_memory"),
        ("backward", "backward_peak_memory"),
        ("optimizer", "optimizer_peak_memory"),
        ("peak", "peak_memory"),
    ):
        values = _numeric_samples(real, field)
        if values:
            points[phase] = {
                "bytes": max(values),
                "source": f"real_cuda.{field}.max",
            }
    if not points:
        raise CalibrationError("selected workload has no positive peak samples")
    return points


def _bundle_points(
    report: Mapping[str, Any],
    *,
    workload: str | None,
) -> dict[str, dict[str, Any]]:
    workloads = report.get("workloads")
    if not isinstance(workloads, list):
        raise CalibrationError("calibration bundle has no workloads")
    selected = [
        item
        for item in workloads
        if isinstance(item, Mapping)
        and (
            workload is None
            or workload
            in {
                str(item.get("name") or ""),
                str(item.get("workload_signature") or ""),
            }
        )
    ]
    if workload is None and len(selected) != 1:
        raise CalibrationError(
            "calibration bundle contains multiple workloads; pass --workload"
        )
    if len(selected) != 1:
        raise CalibrationError(f"workload {workload!r} was not found uniquely")
    observations = selected[0].get("observations")
    if not isinstance(observations, list):
        raise CalibrationError("bundle workload has no observations")
    values = [
        int(item["empirical_physical_peak_upper_bound_bytes"])
        for item in observations
        if isinstance(item, Mapping)
        and isinstance(
            item.get("empirical_physical_peak_upper_bound_bytes"),
            int,
        )
        and not isinstance(
            item.get("empirical_physical_peak_upper_bound_bytes"),
            bool,
        )
    ]
    if not values:
        raise CalibrationError("bundle workload has no physical peak")
    return {
        "peak": {
            "bytes": max(values),
            "source": "observations.empirical_physical_peak_upper_bound_bytes.max",
        }
    }


def _match_points(
    predicted: Mapping[str, Mapping[str, Any]],
    observed: Mapping[str, Mapping[str, Any]],
) -> list[tuple[str, Mapping[str, Any], Mapping[str, Any]]]:
    common = sorted(set(predicted) & set(observed))
    if common:
        return [(phase, predicted[phase], observed[phase]) for phase in common]
    predicted_peak = _canonical_point(predicted)
    observed_peak = _canonical_point(observed)
    if predicted_peak is None or observed_peak is None:
        return []
    return [("peak", predicted_peak, observed_peak)]


def _canonical_point(
    points: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    for name in ("peak", "steady_state", "first_step"):
        if name in points:
            return points[name]
    return (
        max(points.values(), key=lambda item: int(item["bytes"]))
        if points
        else None
    )


def _attribute_error(
    *,
    prediction: Mapping[str, Any],
    observation: Mapping[str, Any],
    comparisons: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    causes: list[dict[str, Any]] = []
    underprediction = any(
        int(item["signed_error_bytes"]) < 0 for item in comparisons
    )
    unmodeled = prediction.get("unmodeled_components")
    if underprediction and isinstance(unmodeled, list):
        for component in unmodeled:
            causes.append(
                {
                    "component": str(component),
                    "evidence": "declared_unmodeled_component",
                    "direction": "possible_underprediction",
                }
            )
    workspace = prediction.get("workspace_estimate")
    if isinstance(workspace, Mapping):
        coverage = workspace.get("coverage")
        if isinstance(coverage, Mapping) and not bool(
            coverage.get("upper_bound_complete", True)
        ):
            causes.append(
                {
                    "component": "unprofiled_operator_workspace",
                    "evidence": "workspace_upper_bound_incomplete",
                    "direction": "possible_underprediction",
                }
            )

    reserved = _find_numeric_key(observation, "peak_reserved_memory")
    allocated = _find_numeric_key(observation, "peak_memory")
    if reserved is not None and allocated is not None and reserved > allocated:
        causes.append(
            {
                "component": "allocator_reservation_and_fragmentation",
                "evidence": {
                    "reserved_minus_allocated_bytes": reserved - allocated,
                },
                "direction": "physical_process_peak_above_allocated_peak",
            }
        )
    if not causes:
        causes.append(
            {
                "component": "unattributed",
                "evidence": "reports_do_not_expose_component_breakdown",
                "direction": "unknown",
            }
        )
    return causes


def _report_dimensions(report: Mapping[str, Any]) -> dict[str, Any]:
    dimensions: dict[str, Any] = {}
    for source in (
        report,
        report.get("inputs"),
        report.get("model"),
        report.get("calibration_gpu"),
        report.get("software"),
    ):
        if not isinstance(source, Mapping):
            continue
        for key in (
            "workload_signature",
            "profile",
            "target_profile",
            "compute_capability",
            "torch_version",
            "cuda_version",
            "dtype",
            "shape",
            "batch_size",
            "prompt_tokens",
        ):
            if key in source and source[key] is not None:
                normalized_key = (
                    "gpu_profile"
                    if key in {"profile", "target_profile"}
                    else key
                )
                dimensions[normalized_key] = source[key]
    return dimensions


def _numeric_samples(payload: Mapping[str, Any], field: str) -> list[int]:
    value = payload.get(field)
    if isinstance(value, int) and not isinstance(value, bool):
        return [value]
    if isinstance(value, Mapping):
        return [
            int(candidate)
            for key in ("max", "median", "expected", "value")
            if isinstance((candidate := value.get(key)), int)
            and not isinstance(candidate, bool)
        ][:1]
    trials = payload.get("trials")
    if isinstance(trials, list):
        return [
            int(item[field])
            for item in trials
            if isinstance(item, Mapping)
            and isinstance(item.get(field), int)
            and not isinstance(item.get(field), bool)
        ]
    return []


def _first_integer(
    payload: Mapping[str, Any],
    keys: Sequence[str],
) -> int | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _find_numeric_key(payload: Any, key: str) -> int | None:
    if isinstance(payload, Mapping):
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, Mapping):
            for candidate_key in ("max", "median", "value"):
                candidate = value.get(candidate_key)
                if isinstance(candidate, int) and not isinstance(candidate, bool):
                    return candidate
        for nested in payload.values():
            result = _find_numeric_key(nested, key)
            if result is not None:
                return result
    elif isinstance(payload, list):
        for nested in payload:
            result = _find_numeric_key(nested, key)
            if result is not None:
                return result
    return None


def _validate_percentage(
    name: str,
    value: float,
    *,
    bounded: bool = False,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (bounded and value > 100)
    ):
        suffix = " between 0 and 100" if bounded else " non-negative"
        raise CalibrationError(f"{name} must be finite and{suffix}")


def _required_non_negative_integer(
    payload: Mapping[str, Any],
    key: str,
    *,
    label: str,
) -> int:
    value = payload.get(key)
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise CalibrationError(
            f"{label}: {key} must be a non-negative integer"
        )
    return value


def _dimension_mismatches(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping):
        return [
            {
                "dimension": "dimensions",
                "prediction": None,
                "observation": None,
                "reason": "comparison report has no dimensions object",
            }
        ]
    prediction = payload.get("prediction")
    observation = payload.get("observation")
    if not isinstance(prediction, Mapping) or not isinstance(
        observation,
        Mapping,
    ):
        return [
            {
                "dimension": "dimensions",
                "prediction": None,
                "observation": None,
                "reason": "prediction and observation dimensions are required",
            }
        ]
    return [
        {
            "dimension": key,
            "prediction": prediction.get(key),
            "observation": observation.get(key),
        }
        for key in sorted(set(prediction) | set(observation))
        if prediction.get(key) != observation.get(key)
    ]


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * quantile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return (
        ordered[lower_index] * (1 - fraction)
        + ordered[upper_index] * fraction
    )


def _print_comparison(report: Mapping[str, Any]) -> None:
    summary = report["summary"]
    print("FakeGPU memory calibration comparison")
    print(f"  phases: {summary['phase_count']}")
    print(
        "  mean absolute error: "
        f"{int(summary['mean_absolute_error_bytes']) / 2**20:.2f} MiB"
    )
    mape = summary["mean_absolute_percentage_error"]
    print(f"  mean absolute percentage error: {mape * 100:.2f}%" if mape is not None else "  mean absolute percentage error: n/a")
    print(
        "  recommended safety margin: "
        f"{int(summary['recommended_memory_safety_margin_bytes']) / 2**20:.2f} MiB"
    )
    print(
        "  recommended safety factor: "
        f"{float(summary['recommended_memory_safety_factor']):.6g}"
    )


def _print_verification(report: Mapping[str, Any]) -> None:
    metrics = report["metrics"]
    print(f"FakeGPU calibration verification: {report['status']}")
    print(f"  comparisons: {report['comparison_count']}")
    print(
        "  maximum underestimate: "
        f"{float(metrics['maximum_underestimate_percent']):.3f}%"
    )
    maximum_ape = metrics["maximum_absolute_percentage_error_percent"]
    print(
        f"  maximum absolute percentage error: {float(maximum_ape):.3f}%"
        if maximum_ape is not None
        else "  maximum absolute percentage error: n/a"
    )
    interval_coverage = metrics["prediction_interval_coverage_percent"]
    print(
        f"  prediction interval coverage: {float(interval_coverage):.3f}%"
        if interval_coverage is not None
        else "  prediction interval coverage: n/a"
    )
    for failure in report["failures"]:
        print(f"  failed gate: {failure['gate']}")


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "COMPARISON_SCHEMA_VERSION",
    "VERIFICATION_SCHEMA_VERSION",
    "CalibrationError",
    "build_workload_calibration_bundle",
    "compare_memory_reports",
    "main",
    "verify_calibration_reports",
]
