from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import statistics
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .structured_io import emit_json, load_mapping, write_json


COMPARISON_SCHEMA_VERSION = "fakegpu.calibration_comparison.v1"
BUNDLE_SCHEMA_VERSION = "fakegpu.workload_calibration_bundle.v1"
VERIFICATION_SCHEMA_VERSION = "fakegpu.calibration_verification.v1"
SERVING_OBSERVATION_SCHEMA_VERSION = (
    "fakegpu.serving_memory_observation.v1"
)
SERVING_SAMPLE_SCHEMA_VERSION = "fakegpu.serving_peak_sample.v1"
_SERVING_SAMPLE_MARKER = "FAKEGPU_SERVING_SAMPLE="
_SERVING_TARGET_MEMORY_TOLERANCE_PERCENT = 2.0
_SERVING_PLAN_SCHEMA_VERSIONS = frozenset(
    {
        "fakegpu.llm_serving_plan.v1",
        "fakegpu.llm_serving_request_set_plan.v1",
    }
)


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


def build_serving_memory_observation(
    plan: Mapping[str, Any],
    *,
    phase_samples: Mapping[str, Sequence[int]],
    minimum_samples_per_phase: int = 5,
    source: str = "real_cuda_process_peak",
) -> dict[str, Any]:
    """Build phase-aware serving evidence from repeated real-GPU peaks."""

    context = _serving_plan_observation_context(plan)
    schema = str(context["schema_version"])
    workload_signature = str(context["workload_signature"])
    inputs = context["inputs"]
    profile = context["profile"]
    compute_capability = str(context["compute_capability"])
    planned_phases = list(context["planned_phases"])
    if (
        not isinstance(minimum_samples_per_phase, int)
        or isinstance(minimum_samples_per_phase, bool)
        or minimum_samples_per_phase <= 0
    ):
        raise CalibrationError(
            "minimum_samples_per_phase must be a positive integer"
        )
    source = str(source).strip()
    if not source:
        raise CalibrationError("source must be a non-empty string")
    normalized_phase_samples: dict[str, Sequence[int]] = {}
    for name, samples in phase_samples.items():
        normalized_name = str(name)
        if normalized_name in normalized_phase_samples:
            raise CalibrationError(
                f"duplicate observation phase {normalized_name!r}"
            )
        normalized_phase_samples[normalized_name] = samples
    supplied_phases = set(normalized_phase_samples)
    expected_phases = set(planned_phases)
    if supplied_phases != expected_phases:
        missing = sorted(expected_phases - supplied_phases)
        unexpected = sorted(supplied_phases - expected_phases)
        details = []
        if missing:
            details.append(f"missing phases: {', '.join(missing)}")
        if unexpected:
            details.append(
                f"unexpected phases: {', '.join(unexpected)}"
            )
        raise CalibrationError("; ".join(details))

    observed_phases: list[dict[str, Any]] = []
    phase_sample_counts: dict[str, int] = {}
    insufficient_phases: list[str] = []
    for phase in planned_phases:
        raw_samples = normalized_phase_samples[phase]
        if isinstance(raw_samples, (str, bytes)) or not isinstance(
            raw_samples,
            Sequence,
        ):
            raise CalibrationError(
                f"{phase} samples must be a non-empty sequence"
            )
        samples = []
        for index, value in enumerate(raw_samples):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise CalibrationError(
                    f"{phase} samples[{index}] must be a positive integer"
                )
            samples.append(int(value))
        if not samples:
            raise CalibrationError(
                f"{phase} samples must be a non-empty sequence"
            )
        sample_count = len(samples)
        phase_sample_counts[phase] = sample_count
        if sample_count < minimum_samples_per_phase:
            insufficient_phases.append(phase)
        observed_phases.append(
            {
                "phase": phase,
                "peak_bytes": max(samples),
                "sample_count": sample_count,
                "samples_bytes": samples,
                "minimum_bytes": min(samples),
                "median_bytes": statistics.median(samples),
                "maximum_bytes": max(samples),
                "aggregation": "maximum",
            }
        )

    peak_phase_report = max(
        observed_phases,
        key=lambda item: int(item["peak_bytes"]),
    )
    evidence_status = (
        "ready_for_comparison"
        if not insufficient_phases
        else "insufficient_samples"
    )
    observation_inputs = {"dtype": inputs["dtype"]}
    for key in ("batch_size", "prompt_tokens", "shape"):
        if inputs.get(key) is not None:
            observation_inputs[key] = inputs[key]
    return {
        "schema_version": SERVING_OBSERVATION_SCHEMA_VERSION,
        "evidence_status": evidence_status,
        "source_plan_schema_version": schema,
        "source_plan_method": plan.get("method"),
        "workload_signature": workload_signature,
        "profile": str(profile["id"]),
        "compute_capability": str(compute_capability),
        "inputs": observation_inputs,
        "measurement": {
            "source": source,
            "metric": "process_peak_memory_bytes",
            "aggregation": "maximum_of_isolated_runs",
            "minimum_samples_per_phase": minimum_samples_per_phase,
            "phase_sample_counts": phase_sample_counts,
            "insufficient_phases": insufficient_phases,
            "complete_phase_coverage": True,
        },
        "memory_timeline": {
            "unit": "bytes",
            "source": source,
            "phases": observed_phases,
            "peak_bytes": int(peak_phase_report["peak_bytes"]),
            "peak_phase": peak_phase_report["phase"],
        },
        "notes": [
            (
                "Each phase uses the maximum repeated peak as a "
                "conservative observation."
            ),
            (
                "ready_for_comparison records sample sufficiency only; "
                "GPU-validated status still requires calibration gates."
            ),
        ],
    }


def build_cuda_serving_sample(
    plan: Mapping[str, Any],
    *,
    phase_peaks: Mapping[str, int],
    metric: str,
    framework: str,
    framework_version: str,
    run_index: int | None = None,
    software: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a serving-runner protocol sample from a real CUDA runtime.

    This is the compatibility entry point for vLLM and custom runners that
    already have trustworthy phase measurements. CUDA device and software
    identity are collected here so each runner does not need to duplicate the
    versioned protocol.
    """

    context = _serving_plan_observation_context(plan)
    normalized_peaks = _normalize_serving_phase_peaks(
        phase_peaks,
        planned_phases=context["planned_phases"],
    )
    normalized_metric = _nonempty_string(metric, "metric")
    normalized_framework = _nonempty_string(framework, "framework")
    normalized_framework_version = _nonempty_string(
        framework_version,
        "framework_version",
    )
    normalized_run_index = _serving_sample_run_index(run_index)
    _validate_collector_environment_contract(
        context,
        run_index=normalized_run_index,
    )

    torch_module = _load_optional_module(
        "torch",
        purpose="building a real-CUDA serving sample",
    )
    environment = _real_cuda_environment(
        torch_module,
        framework=normalized_framework,
        framework_version=normalized_framework_version,
        software=software,
    )
    sample = {
        "schema_version": SERVING_SAMPLE_SCHEMA_VERSION,
        "workload_signature": context["workload_signature"],
        "run_index": normalized_run_index,
        "metric": normalized_metric,
        "phases": {
            phase: {"peak_bytes": peak}
            for phase, peak in normalized_peaks.items()
        },
        "environment": environment,
    }
    _validate_serving_runner_sample(
        sample,
        context=context,
        run_index=normalized_run_index,
    )
    return sample


def measure_transformers_serving_sample(
    plan: Mapping[str, Any],
    *,
    model_dir: str | os.PathLike[str] | None = None,
    metric: str = "reserved",
    trust_remote_code: bool = False,
    run_index: int | None = None,
) -> dict[str, Any]:
    """Measure one homogeneous Transformers serving sample on real CUDA.

    The adapter executes a full prefill followed by token-at-a-time decoding
    with the returned dynamic KV cache. Optional framework dependencies are
    imported only when this function is called.
    """

    workload = _transformers_serving_workload(plan, model_dir=model_dir)
    metric_key = str(metric).strip().lower()
    metric_name = {
        "allocated": "torch.cuda.max_memory_allocated",
        "reserved": "torch.cuda.max_memory_reserved",
    }.get(metric_key)
    if metric_name is None:
        raise CalibrationError("metric must be 'allocated' or 'reserved'")

    torch_module = _load_optional_module(
        "torch",
        purpose="measuring Transformers serving memory",
    )
    transformers_module = _load_optional_module(
        "transformers",
        purpose="measuring Transformers serving memory",
    )
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        raise CalibrationError(
            "Transformers serving measurement requires an available real "
            "CUDA device"
        )
    _reject_simulated_cuda_environment()
    device_index = int(cuda.current_device())
    cuda.set_device(device_index)
    device = f"cuda:{device_index}"
    dtype = _transformers_torch_dtype(torch_module, workload["dtype"])

    load_kwargs: dict[str, Any] = {
        "attn_implementation": workload["attention_implementation"],
        "dtype": dtype,
        "local_files_only": True,
        "trust_remote_code": bool(trust_remote_code),
        "use_safetensors": True,
    }
    model_factory = transformers_module.AutoModelForCausalLM
    try:
        model = model_factory.from_pretrained(
            str(workload["model_dir"]),
            **load_kwargs,
        )
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")
        model = model_factory.from_pretrained(
            str(workload["model_dir"]),
            **load_kwargs,
        )
    model = model.to(device)
    model.eval()

    token_id = _synthetic_token_id(model.config)
    input_ids = torch_module.full(
        (
            workload["active_sequences"],
            workload["prompt_tokens"],
        ),
        token_id,
        dtype=torch_module.long,
        device=device,
    )
    peak_reader = getattr(cuda, f"max_memory_{metric_key}")
    with torch_module.inference_mode():
        _reset_cuda_phase_peak(cuda, device_index)
        prefill_output = model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )
        cuda.synchronize(device_index)
        prefill_peak = int(peak_reader(device_index))
        past_key_values, next_token = _next_transformers_token(
            prefill_output,
            phase="prefill",
        )
        del prefill_output
        del input_ids

        cuda.synchronize(device_index)
        cuda.empty_cache()
        _reset_cuda_phase_peak(cuda, device_index)
        for _ in range(max(0, workload["generated_tokens"] - 1)):
            decode_output = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values, next_token = _next_transformers_token(
                decode_output,
                phase="decode",
            )
            del decode_output
        cuda.synchronize(device_index)
        decode_peak = int(peak_reader(device_index))

    return build_cuda_serving_sample(
        plan,
        phase_peaks={
            "prefill": prefill_peak,
            "decode": decode_peak,
        },
        metric=metric_name,
        framework="transformers",
        framework_version=str(transformers_module.__version__),
        run_index=run_index,
        software={
            "attention_implementation": workload[
                "attention_implementation"
            ],
            "cache_implementation": "dynamic",
            "dtype": workload["dtype"],
            "runner": "fakegpu.calibrate.sample-transformers",
        },
    )


def collect_serving_memory_observation(
    plan: Mapping[str, Any],
    *,
    command: Sequence[str],
    repetitions: int = 5,
    minimum_samples_per_phase: int = 5,
    timeout_seconds: float = 600.0,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Collect serving peaks by repeatedly invoking a JSON sample runner."""

    context = _serving_plan_observation_context(plan)
    normalized_command = _normalize_runner_command(command)
    if (
        not isinstance(repetitions, int)
        or isinstance(repetitions, bool)
        or repetitions <= 0
    ):
        raise CalibrationError("repetitions must be a positive integer")
    if (
        not isinstance(minimum_samples_per_phase, int)
        or isinstance(minimum_samples_per_phase, bool)
        or minimum_samples_per_phase <= 0
    ):
        raise CalibrationError(
            "minimum_samples_per_phase must be a positive integer"
        )
    if (
        not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0
    ):
        raise CalibrationError("timeout_seconds must be finite and positive")
    if source is not None and (
        not isinstance(source, str) or not source.strip()
    ):
        raise CalibrationError("source must be a non-empty string")
    target_memory_bytes = context["profile"].get("memory_bytes")
    if (
        not isinstance(target_memory_bytes, int)
        or isinstance(target_memory_bytes, bool)
        or target_memory_bytes <= 0
    ):
        raise CalibrationError(
            "serving plan target profile has no valid memory_bytes"
        )

    working_directory = Path(cwd or os.getcwd()).expanduser().resolve()
    if not working_directory.is_dir():
        raise CalibrationError(
            f"runner working directory does not exist: {working_directory}"
        )
    child_environment = dict(os.environ)
    if env is not None:
        if not isinstance(env, Mapping):
            raise CalibrationError("env must be a string mapping")
        for key, value in env.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise CalibrationError("env must be a string mapping")
            child_environment[key] = value

    planned_phases = list(context["planned_phases"])
    phase_samples: dict[str, list[int]] = {
        phase: [] for phase in planned_phases
    }
    common_environment: dict[str, Any] | None = None
    reported_metric: str | None = None
    run_reports: list[dict[str, Any]] = []
    for run_index in range(1, repetitions + 1):
        run_environment = dict(child_environment)
        run_environment.update(
            {
                "FAKEGPU_SERVING_RUN_INDEX": str(run_index),
                "FAKEGPU_SERVING_RUN_COUNT": str(repetitions),
                "FAKEGPU_SERVING_SAMPLE_SCHEMA": (
                    SERVING_SAMPLE_SCHEMA_VERSION
                ),
                "FAKEGPU_SERVING_PLAN_SCHEMA": str(
                    context["schema_version"]
                ),
                "FAKEGPU_SERVING_WORKLOAD_SIGNATURE": str(
                    context["workload_signature"]
                ),
                "FAKEGPU_SERVING_TARGET_PROFILE": str(
                    context["profile"]["id"]
                ),
                "FAKEGPU_SERVING_COMPUTE_CAPABILITY": str(
                    context["compute_capability"]
                ),
            }
        )
        started = time.monotonic()
        try:
            completed = subprocess.run(
                normalized_command,
                cwd=working_directory,
                env=run_environment,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=float(timeout_seconds),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise CalibrationError(
                f"serving runner timed out on repetition {run_index} "
                f"after {float(timeout_seconds):.3f}s"
            ) from exc
        except (OSError, ValueError) as exc:
            raise CalibrationError(
                f"serving runner could not start on repetition "
                f"{run_index}: {exc}"
            ) from exc
        duration_seconds = time.monotonic() - started
        if completed.returncode != 0:
            diagnostic = _runner_diagnostic(
                completed.stderr or completed.stdout
            )
            suffix = f": {diagnostic}" if diagnostic else ""
            raise CalibrationError(
                f"serving runner failed on repetition {run_index} with "
                f"status {completed.returncode}{suffix}"
            )

        sample = _parse_serving_runner_sample(
            completed.stdout,
            run_index=run_index,
        )
        peaks, sample_environment, metric = _validate_serving_runner_sample(
            sample,
            context=context,
            run_index=run_index,
        )
        if reported_metric is None:
            reported_metric = metric
        elif metric != reported_metric:
            raise CalibrationError(
                f"serving runner repetition {run_index} changed metric "
                f"from {reported_metric!r} to {metric!r}"
            )
        if common_environment is None:
            common_environment = sample_environment
        elif sample_environment != common_environment:
            raise CalibrationError(
                f"serving runner repetition {run_index} changed its "
                "GPU or software environment"
            )
        for phase in planned_phases:
            phase_samples[phase].append(peaks[phase])
        run_reports.append(
            {
                "run_index": run_index,
                "duration_seconds": round(duration_seconds, 6),
                "phase_peaks_bytes": peaks,
            }
        )

    if reported_metric is None or common_environment is None:
        # Unreachable in practice: repetitions > 0 is validated above and
        # the first loop iteration always sets both before any use. Guard
        # explicitly rather than with assert, which python -O strips,
        # so a future change to the loop can't silently embed None here.
        raise RuntimeError(
            "serving runner produced no successful repetitions"
        )
    observation = build_serving_memory_observation(
        plan,
        phase_samples=phase_samples,
        minimum_samples_per_phase=minimum_samples_per_phase,
        source=source or reported_metric,
    )
    command_digest = hashlib.sha256(
        b"\0".join(
            item.encode("utf-8", errors="surrogatepass")
            for item in normalized_command
        )
    ).hexdigest()
    observation["measurement"].update(
        {
            "reported_metric": reported_metric,
            "environment": common_environment,
            "runner": {
                "protocol": SERVING_SAMPLE_SCHEMA_VERSION,
                "repetitions": repetitions,
                "successful_runs": repetitions,
                "timeout_seconds": float(timeout_seconds),
                "executable": Path(normalized_command[0]).name,
                "command_fingerprint": f"sha256:{command_digest}",
                "runs": run_reports,
            },
        }
    )
    observation["notes"].append(
        "Runner GPU and software metadata matched across all repetitions."
    )
    return observation


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
        description=(
            "Collect serving observations, compare memory predictions, "
            "or bundle calibration evidence."
        ),
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

    observe_parser = subparsers.add_parser(
        "observe-serving",
        help=(
            "Build a phase-aware observation from repeated serving peaks."
        ),
    )
    observe_parser.add_argument("plan")
    observe_parser.add_argument(
        "--prefill-peak-bytes",
        action="append",
        required=True,
        metavar="BYTES[,BYTES...]",
        help=(
            "One or more isolated prefill process peaks in bytes; repeat "
            "the flag or provide comma-separated values."
        ),
    )
    observe_parser.add_argument(
        "--decode-peak-bytes",
        action="append",
        required=True,
        metavar="BYTES[,BYTES...]",
        help=(
            "One or more isolated decode process peaks in bytes; repeat "
            "the flag or provide comma-separated values."
        ),
    )
    observe_parser.add_argument(
        "--minimum-samples-per-phase",
        type=int,
        default=5,
    )
    observe_parser.add_argument(
        "--source",
        default="real_cuda_process_peak",
        help="Measurement source recorded in the observation report.",
    )
    observe_parser.add_argument(
        "--strict",
        action="store_true",
        help="Return status 1 when either phase has too few samples.",
    )
    observe_parser.add_argument(
        "--json",
        dest="json_path",
        nargs="?",
        const="-",
        help="Write JSON to PATH, or stdout when PATH is omitted.",
    )

    sample_transformers_parser = subparsers.add_parser(
        "sample-transformers",
        help=(
            "Measure one homogeneous Transformers serving sample on CUDA."
        ),
    )
    sample_transformers_parser.add_argument(
        "plan",
        nargs="?",
        help=(
            "Serving plan path; defaults to "
            "$FAKEGPU_SERVING_PLAN_PATH under collect-serving."
        ),
    )
    sample_transformers_parser.add_argument(
        "--model-dir",
        help="Relocated local checkpoint directory (default: plan model path).",
    )
    sample_transformers_parser.add_argument(
        "--metric",
        choices=("allocated", "reserved"),
        default="reserved",
    )
    sample_transformers_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow code shipped by the local model directory.",
    )
    sample_transformers_parser.add_argument("--run-index", type=int)

    emit_sample_parser = subparsers.add_parser(
        "emit-serving-sample",
        help=(
            "Wrap vLLM or custom phase peaks in the serving sample protocol."
        ),
    )
    emit_sample_parser.add_argument(
        "plan",
        nargs="?",
        help=(
            "Serving plan path; defaults to "
            "$FAKEGPU_SERVING_PLAN_PATH under collect-serving."
        ),
    )
    emit_sample_parser.add_argument(
        "--prefill-peak-bytes",
        type=int,
        required=True,
    )
    emit_sample_parser.add_argument(
        "--decode-peak-bytes",
        type=int,
        required=True,
    )
    emit_sample_parser.add_argument("--metric", required=True)
    emit_sample_parser.add_argument("--framework", required=True)
    emit_sample_parser.add_argument("--framework-version", required=True)
    emit_sample_parser.add_argument("--run-index", type=int)

    collect_parser = subparsers.add_parser(
        "collect-serving",
        help="Repeat a JSON sample runner and build a serving observation.",
    )
    collect_parser.add_argument("plan")
    collect_parser.add_argument("--repetitions", type=int, default=5)
    collect_parser.add_argument(
        "--minimum-samples-per-phase",
        type=int,
        default=5,
    )
    collect_parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=600.0,
    )
    collect_parser.add_argument("--cwd")
    collect_parser.add_argument(
        "--source",
        help=(
            "Override the metric name reported by the sample runner in "
            "the observation source field."
        ),
    )
    collect_parser.add_argument(
        "--strict",
        action="store_true",
        help="Return status 1 when either phase has too few samples.",
    )
    collect_parser.add_argument(
        "--json",
        dest="json_path",
        nargs="?",
        const="-",
        help="Write JSON to PATH, or stdout when PATH is omitted.",
    )
    collect_parser.add_argument(
        "command",
        nargs="+",
        help="Sample runner command after --.",
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
        if args.action == "sample-transformers":
            plan_path = _serving_sample_plan_path(args.plan)
            plan = load_mapping(plan_path)
            sample = measure_transformers_serving_sample(
                plan,
                model_dir=args.model_dir,
                metric=args.metric,
                trust_remote_code=args.trust_remote_code,
                run_index=args.run_index,
            )
            print(
                _SERVING_SAMPLE_MARKER
                + json.dumps(sample, sort_keys=True)
            )
            return 0

        if args.action == "emit-serving-sample":
            plan_path = _serving_sample_plan_path(args.plan)
            plan = load_mapping(plan_path)
            sample = build_cuda_serving_sample(
                plan,
                phase_peaks={
                    "prefill": args.prefill_peak_bytes,
                    "decode": args.decode_peak_bytes,
                },
                metric=args.metric,
                framework=args.framework,
                framework_version=args.framework_version,
                run_index=args.run_index,
                software={
                    "runner": "fakegpu.calibrate.emit-serving-sample",
                },
            )
            print(
                _SERVING_SAMPLE_MARKER
                + json.dumps(sample, sort_keys=True)
            )
            return 0

        if args.action == "collect-serving":
            plan_path = Path(args.plan).expanduser().resolve()
            plan = load_mapping(plan_path)
            command = list(args.command)
            if command and command[0] == "--":
                command = command[1:]
            report = collect_serving_memory_observation(
                plan,
                command=command,
                repetitions=args.repetitions,
                minimum_samples_per_phase=(
                    args.minimum_samples_per_phase
                ),
                timeout_seconds=args.timeout_seconds,
                cwd=args.cwd,
                env={"FAKEGPU_SERVING_PLAN_PATH": str(plan_path)},
                source=args.source,
            )
            if args.json_path:
                output = emit_json(args.json_path, report)
                if output is not None:
                    print(f"Serving observation: {output}")
            else:
                _print_serving_observation(report)
            if (
                args.strict
                and report["evidence_status"] != "ready_for_comparison"
            ):
                return 1
            return 0

        if args.action == "observe-serving":
            plan = load_mapping(args.plan)
            report = build_serving_memory_observation(
                plan,
                phase_samples={
                    "prefill": _parse_peak_samples(
                        args.prefill_peak_bytes,
                        label="prefill",
                    ),
                    "decode": _parse_peak_samples(
                        args.decode_peak_bytes,
                        label="decode",
                    ),
                },
                minimum_samples_per_phase=(
                    args.minimum_samples_per_phase
                ),
                source=args.source,
            )
            if args.json_path:
                output = emit_json(args.json_path, report)
                if output is not None:
                    print(f"Serving observation: {output}")
            else:
                _print_serving_observation(report)
            if (
                args.strict
                and report["evidence_status"] != "ready_for_comparison"
            ):
                return 1
            return 0

        if args.action == "compare":
            prediction = load_mapping(args.prediction)
            observation = load_mapping(args.observation)
            report = compare_memory_reports(
                prediction,
                observation,
                workload=args.workload,
            )
            if args.json_path:
                output = emit_json(args.json_path, report)
                if output is not None:
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
                output = emit_json(args.json_path, report)
                if output is not None:
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
            OSError,
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
    target = report.get("target")
    if isinstance(target, Mapping):
        profile = target.get("profile")
        if isinstance(profile, Mapping):
            if profile.get("id") is not None:
                dimensions["gpu_profile"] = profile["id"]
            if profile.get("compute_capability") is not None:
                dimensions["compute_capability"] = profile[
                    "compute_capability"
                ]
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


def _serving_sample_plan_path(value: str | None) -> Path:
    selected = value or os.environ.get("FAKEGPU_SERVING_PLAN_PATH")
    if not selected:
        raise CalibrationError(
            "serving sample requires PLAN or $FAKEGPU_SERVING_PLAN_PATH"
        )
    path = Path(selected).expanduser().resolve()
    if not path.is_file():
        raise CalibrationError(f"serving plan does not exist: {path}")
    return path


def _transformers_serving_workload(
    plan: Mapping[str, Any],
    *,
    model_dir: str | os.PathLike[str] | None,
) -> dict[str, Any]:
    context = _serving_plan_observation_context(plan)
    if context["schema_version"] != "fakegpu.llm_serving_plan.v1":
        raise CalibrationError(
            "sample-transformers supports homogeneous serving plans only"
        )
    if context["planned_phases"] != ["prefill", "decode"]:
        raise CalibrationError(
            "sample-transformers requires prefill and decode phases"
        )
    inputs = context["inputs"]
    if str(inputs.get("runtime") or "generic") != "generic":
        raise CalibrationError(
            "sample-transformers requires a generic serving runtime plan"
        )
    speculative = inputs.get("speculative_decoding")
    if isinstance(speculative, Mapping) and speculative.get("enabled"):
        raise CalibrationError(
            "sample-transformers does not yet execute speculative decoding"
        )
    if inputs.get("kv_cache_strategy") != "dynamic":
        raise CalibrationError(
            "sample-transformers requires --kv-cache-strategy dynamic"
        )
    if int(inputs.get("shared_prefix_tokens") or 0) != 0:
        raise CalibrationError(
            "sample-transformers does not emulate shared-prefix caching"
        )
    prompt_tokens = _positive_plan_integer(inputs, "prompt_tokens")
    prefill_chunk_tokens = inputs.get("prefill_chunk_tokens")
    if (
        prefill_chunk_tokens is not None
        and int(prefill_chunk_tokens) < prompt_tokens
    ):
        raise CalibrationError(
            "sample-transformers does not emulate chunked prefill"
        )
    for key in ("kv_cache_max_tokens", "kv_cache_window_tokens"):
        if inputs.get(key) is not None:
            raise CalibrationError(
                f"sample-transformers does not emulate {key}"
            )

    weight_storage = plan.get("weight_storage")
    if not isinstance(weight_storage, Mapping):
        raise CalibrationError("serving plan has no weight_storage")
    quantization = weight_storage.get("quantization")
    if isinstance(quantization, Mapping) and quantization.get("enabled"):
        raise CalibrationError(
            "sample-transformers does not load quantized checkpoints"
        )
    if weight_storage.get("adapters"):
        raise CalibrationError(
            "sample-transformers does not load adapter checkpoints"
        )

    model = plan.get("model")
    if not isinstance(model, Mapping):
        raise CalibrationError("serving plan has no model identity")
    selected_model_dir = model_dir or model.get("path")
    if selected_model_dir is None:
        raise CalibrationError(
            "sample-transformers requires a local model directory"
        )
    root = Path(selected_model_dir).expanduser().resolve()
    if not root.is_dir():
        raise CalibrationError(f"model directory does not exist: {root}")
    config_path = root / "config.json"
    if not config_path.is_file():
        raise CalibrationError(f"model config does not exist: {config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CalibrationError(
            f"model config is not valid JSON: {config_path}"
        ) from exc
    if not isinstance(config, Mapping):
        raise CalibrationError(f"model config must be an object: {config_path}")

    from .llm_estimator import (
        _decoder_dimensions,
        inspect_safetensors_checkpoint,
    )

    observed_dimensions = _decoder_dimensions(config)
    mismatched_dimensions = [
        key
        for key, value in observed_dimensions.items()
        if model.get(key) != value
    ]
    if mismatched_dimensions:
        raise CalibrationError(
            "model config no longer matches the serving plan: "
            + ", ".join(mismatched_dimensions)
        )
    expected_checkpoint = plan.get("checkpoint")
    observed_checkpoint = inspect_safetensors_checkpoint(root)
    if not isinstance(expected_checkpoint, Mapping) or dict(
        expected_checkpoint
    ) != observed_checkpoint:
        raise CalibrationError(
            "model checkpoint no longer matches the serving plan"
        )

    attention_implementation = str(
        inputs.get("attention_implementation") or ""
    )
    if attention_implementation not in {"eager", "sdpa"}:
        raise CalibrationError(
            "sample-transformers supports eager or sdpa attention only"
        )
    dtype = str(inputs.get("dtype") or "")
    if dtype not in {"float16", "bfloat16", "float32", "float64"}:
        raise CalibrationError(
            f"sample-transformers does not support dtype {dtype!r}"
        )
    return {
        "model_dir": root,
        "active_sequences": _positive_plan_integer(
            inputs,
            "active_sequences",
        ),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": _positive_plan_integer(
            inputs,
            "generated_tokens",
        ),
        "attention_implementation": attention_implementation,
        "dtype": dtype,
    }


def _positive_plan_integer(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise CalibrationError(f"serving plan {key} must be positive")
    return int(value)


def _normalize_serving_phase_peaks(
    phase_peaks: Mapping[str, int],
    *,
    planned_phases: Sequence[str],
) -> dict[str, int]:
    if not isinstance(phase_peaks, Mapping):
        raise CalibrationError("phase_peaks must be a mapping")
    normalized = {str(key): value for key, value in phase_peaks.items()}
    expected = set(planned_phases)
    supplied = set(normalized)
    if supplied != expected:
        missing = sorted(expected - supplied)
        unexpected = sorted(supplied - expected)
        details = []
        if missing:
            details.append(f"missing phases: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected phases: {', '.join(unexpected)}")
        raise CalibrationError("; ".join(details))
    result = {}
    for phase in planned_phases:
        peak = normalized[phase]
        if (
            not isinstance(peak, int)
            or isinstance(peak, bool)
            or peak <= 0
        ):
            raise CalibrationError(
                f"{phase} peak must be a positive integer"
            )
        result[phase] = int(peak)
    return result


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CalibrationError(f"{label} must be a non-empty string")
    return value.strip()


def _serving_sample_run_index(value: int | None) -> int:
    raw_value: Any = (
        os.environ.get("FAKEGPU_SERVING_RUN_INDEX", "1")
        if value is None
        else value
    )
    try:
        run_index = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise CalibrationError("run_index must be a positive integer") from exc
    if isinstance(raw_value, bool) or run_index <= 0:
        raise CalibrationError("run_index must be a positive integer")
    return run_index


def _validate_collector_environment_contract(
    context: Mapping[str, Any],
    *,
    run_index: int,
) -> None:
    expected = {
        "FAKEGPU_SERVING_SAMPLE_SCHEMA": SERVING_SAMPLE_SCHEMA_VERSION,
        "FAKEGPU_SERVING_PLAN_SCHEMA": str(context["schema_version"]),
        "FAKEGPU_SERVING_WORKLOAD_SIGNATURE": str(
            context["workload_signature"]
        ),
        "FAKEGPU_SERVING_TARGET_PROFILE": str(context["profile"]["id"]),
        "FAKEGPU_SERVING_COMPUTE_CAPABILITY": str(
            context["compute_capability"]
        ),
        "FAKEGPU_SERVING_RUN_INDEX": str(run_index),
    }
    for key, expected_value in expected.items():
        actual = os.environ.get(key)
        if actual is not None and actual != expected_value:
            raise CalibrationError(
                f"{key} does not match the loaded serving plan"
            )


def _load_optional_module(name: str, *, purpose: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise CalibrationError(
            f"{purpose} requires the optional {name!r} package"
        ) from exc


def _reject_simulated_cuda_environment() -> None:
    fakegpu_mode = os.environ.get("FAKEGPU_MODE", "").strip().lower()
    if fakegpu_mode and fakegpu_mode != "passthrough":
        raise CalibrationError(
            "serving samples require real CUDA; FAKEGPU_MODE must be unset "
            "or 'passthrough'"
        )


def _real_cuda_environment(
    torch_module: Any,
    *,
    framework: str,
    framework_version: str,
    software: Mapping[str, Any] | None,
) -> dict[str, Any]:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        raise CalibrationError(
            "building a CUDA serving sample requires an available real "
            "CUDA device"
        )
    _reject_simulated_cuda_environment()
    cuda_version = str(getattr(torch_module.version, "cuda", "") or "")
    if not cuda_version:
        raise CalibrationError("PyTorch does not report a CUDA runtime version")
    device_index = int(cuda.current_device())
    cuda.set_device(device_index)
    properties = cuda.get_device_properties(device_index)
    total_memory = getattr(properties, "total_memory", None)
    if (
        not isinstance(total_memory, int)
        or isinstance(total_memory, bool)
        or total_memory <= 0
    ):
        raise CalibrationError("CUDA device does not report total memory")
    capability = cuda.get_device_capability(device_index)
    if (
        not isinstance(capability, Sequence)
        or len(capability) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in capability
        )
    ):
        raise CalibrationError("CUDA device does not report compute capability")
    gpu_name = str(
        getattr(properties, "name", None)
        or cuda.get_device_name(device_index)
        or ""
    ).strip()
    if not gpu_name:
        raise CalibrationError("CUDA device does not report a name")

    software_report: dict[str, Any] = {
        "framework": framework,
        "framework_version": framework_version,
        "cuda_version": cuda_version,
        "torch_version": str(torch_module.__version__),
    }
    if software is not None:
        if not isinstance(software, Mapping):
            raise CalibrationError("software must be a mapping")
        reserved_keys = set(software_report)
        collisions = sorted(reserved_keys.intersection(map(str, software)))
        if collisions:
            raise CalibrationError(
                "software metadata cannot replace: "
                + ", ".join(collisions)
            )
        for key, item in software.items():
            normalized_key = _nonempty_string(str(key), "software key")
            try:
                json.dumps(item, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise CalibrationError(
                    f"software.{normalized_key} must be JSON-compatible"
                ) from exc
            software_report[normalized_key] = item

    environment: dict[str, Any] = {
        "backend": "cuda",
        "simulated": False,
        "gpu_name": gpu_name,
        "compute_capability": [int(capability[0]), int(capability[1])],
        "total_memory_bytes": int(total_memory),
        "device_index": device_index,
        "software": software_report,
    }
    gpu_uuid = getattr(properties, "uuid", None)
    if gpu_uuid:
        if isinstance(gpu_uuid, bytes):
            gpu_uuid = gpu_uuid.decode("utf-8", errors="replace")
        environment["gpu_uuid"] = str(gpu_uuid)
    allocator_reader = getattr(cuda, "get_allocator_backend", None)
    if callable(allocator_reader):
        environment["allocator"] = str(allocator_reader())
    return environment


def _transformers_torch_dtype(torch_module: Any, dtype: str) -> Any:
    try:
        return getattr(torch_module, dtype)
    except AttributeError as exc:
        raise CalibrationError(
            f"installed PyTorch does not expose dtype {dtype!r}"
        ) from exc


def _synthetic_token_id(config: Any) -> int:
    vocab_size = getattr(config, "vocab_size", None)
    if (
        not isinstance(vocab_size, int)
        or isinstance(vocab_size, bool)
        or vocab_size <= 0
    ):
        raise CalibrationError("Transformers model has no positive vocab_size")
    for name in ("bos_token_id", "pad_token_id", "eos_token_id"):
        value = getattr(config, name, None)
        if isinstance(value, list):
            value = value[0] if value else None
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and 0 <= value < vocab_size
        ):
            return int(value)
    return 0


def _reset_cuda_phase_peak(cuda: Any, device_index: int) -> None:
    cuda.synchronize(device_index)
    cuda.reset_peak_memory_stats(device_index)


def _next_transformers_token(
    output: Any,
    *,
    phase: str,
) -> tuple[Any, Any]:
    past_key_values = getattr(output, "past_key_values", None)
    logits = getattr(output, "logits", None)
    if past_key_values is None:
        raise CalibrationError(
            f"Transformers {phase} output did not return past_key_values"
        )
    if logits is None:
        raise CalibrationError(
            f"Transformers {phase} output did not return logits"
        )
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return past_key_values, next_token


def _serving_plan_observation_context(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(plan, Mapping):
        raise CalibrationError("serving plan must be a mapping")
    schema = str(plan.get("schema_version") or "")
    if schema not in _SERVING_PLAN_SCHEMA_VERSIONS:
        choices = ", ".join(sorted(_SERVING_PLAN_SCHEMA_VERSIONS))
        raise CalibrationError(
            f"serving plan schema must be one of: {choices}"
        )
    workload_signature = str(plan.get("workload_signature") or "")
    if not workload_signature:
        raise CalibrationError("serving plan has no workload_signature")
    inputs = plan.get("inputs")
    if not isinstance(inputs, Mapping) or not inputs.get("dtype"):
        raise CalibrationError("serving plan has no input dtype")
    target = plan.get("target")
    profile = target.get("profile") if isinstance(target, Mapping) else None
    if not isinstance(profile, Mapping) or not profile.get("id"):
        raise CalibrationError(
            "serving plan must select a target profile before observation"
        )
    compute_capability = profile.get("compute_capability")
    if compute_capability is None:
        raise CalibrationError(
            "serving plan target profile has no compute_capability"
        )

    timeline = plan.get("memory_timeline")
    raw_phases = (
        timeline.get("phases")
        if isinstance(timeline, Mapping)
        else None
    )
    if not isinstance(raw_phases, list) or not raw_phases:
        raise CalibrationError("serving plan has no memory timeline phases")
    planned_phases = [
        str(item.get("phase") or "")
        for item in raw_phases
        if isinstance(item, Mapping) and item.get("phase")
    ]
    if len(planned_phases) != len(raw_phases):
        raise CalibrationError(
            "serving plan memory timeline contains an invalid phase"
        )
    if len(set(planned_phases)) != len(planned_phases):
        raise CalibrationError(
            "serving plan memory timeline contains duplicate phases"
        )
    return {
        "schema_version": schema,
        "workload_signature": workload_signature,
        "inputs": inputs,
        "profile": profile,
        "compute_capability": str(compute_capability),
        "planned_phases": planned_phases,
    }


def _normalize_runner_command(command: Sequence[str]) -> list[str]:
    if isinstance(command, (str, bytes)) or not isinstance(
        command,
        Sequence,
    ):
        raise CalibrationError("runner command must be a non-empty sequence")
    normalized = []
    for index, item in enumerate(command):
        if not isinstance(item, str):
            raise CalibrationError(
                f"runner command item {index} must be a string"
            )
        if "\0" in item:
            raise CalibrationError(
                f"runner command item {index} contains a null byte"
            )
        normalized.append(item)
    if not normalized or not normalized[0].strip():
        raise CalibrationError("runner command must name an executable")
    return normalized


def _parse_serving_runner_sample(
    stdout: str,
    *,
    run_index: int,
) -> Mapping[str, Any]:
    stripped = stdout.strip()
    if not stripped:
        raise CalibrationError(
            f"serving runner repetition {run_index} produced no stdout"
        )
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    marker_lines = [
        line[len(_SERVING_SAMPLE_MARKER) :]
        for line in lines
        if line.startswith(_SERVING_SAMPLE_MARKER)
    ]
    if marker_lines:
        return _decode_serving_sample_json(
            marker_lines[-1],
            run_index=run_index,
        )
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, Mapping):
        return payload

    fallback: Mapping[str, Any] | None = None
    for line in reversed(lines):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(candidate, Mapping):
            continue
        if candidate.get("schema_version") == SERVING_SAMPLE_SCHEMA_VERSION:
            return candidate
        if fallback is None:
            fallback = candidate
    if fallback is not None:
        return fallback
    raise CalibrationError(
        f"serving runner repetition {run_index} did not emit a JSON object; "
        f"use the {_SERVING_SAMPLE_MARKER!r} prefix when stdout has logs"
    )


def _decode_serving_sample_json(
    value: str,
    *,
    run_index: int,
) -> Mapping[str, Any]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise CalibrationError(
            f"serving runner repetition {run_index} emitted invalid "
            f"sample JSON: {exc.msg}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise CalibrationError(
            f"serving runner repetition {run_index} sample must be a "
            "JSON object"
        )
    return payload


def _validate_serving_runner_sample(
    sample: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    run_index: int,
) -> tuple[dict[str, int], dict[str, Any], str]:
    label = f"serving runner repetition {run_index}"
    schema = str(sample.get("schema_version") or "")
    if schema != SERVING_SAMPLE_SCHEMA_VERSION:
        raise CalibrationError(
            f"{label} schema must be {SERVING_SAMPLE_SCHEMA_VERSION!r}"
        )
    signature = str(sample.get("workload_signature") or "")
    if signature != context["workload_signature"]:
        raise CalibrationError(f"{label} workload_signature does not match")
    sample_run_index = sample.get("run_index")
    if sample_run_index is not None and (
        not isinstance(sample_run_index, int)
        or isinstance(sample_run_index, bool)
        or sample_run_index != run_index
    ):
        raise CalibrationError(
            f"{label} reported run_index {sample_run_index!r}"
        )
    metric = str(sample.get("metric") or "").strip()
    if not metric:
        raise CalibrationError(f"{label} metric must be a non-empty string")

    raw_phases = sample.get("phases")
    if not isinstance(raw_phases, Mapping):
        raise CalibrationError(f"{label} phases must be an object")
    expected_phases = set(context["planned_phases"])
    supplied_phases = {str(name) for name in raw_phases}
    if supplied_phases != expected_phases:
        missing = sorted(expected_phases - supplied_phases)
        unexpected = sorted(supplied_phases - expected_phases)
        details = []
        if missing:
            details.append(f"missing phases: {', '.join(missing)}")
        if unexpected:
            details.append(
                f"unexpected phases: {', '.join(unexpected)}"
            )
        raise CalibrationError(f"{label} {'; '.join(details)}")
    peaks: dict[str, int] = {}
    for phase in context["planned_phases"]:
        raw_phase = raw_phases.get(phase)
        if not isinstance(raw_phase, Mapping):
            raise CalibrationError(f"{label} {phase} phase must be an object")
        peak = raw_phase.get("peak_bytes")
        if (
            not isinstance(peak, int)
            or isinstance(peak, bool)
            or peak <= 0
        ):
            raise CalibrationError(
                f"{label} {phase} peak_bytes must be a positive integer"
            )
        peaks[phase] = peak

    environment = _normalize_serving_runner_environment(
        sample.get("environment"),
        context=context,
        label=label,
    )
    return peaks, environment, metric


def _normalize_serving_runner_environment(
    payload: Any,
    *,
    context: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise CalibrationError(f"{label} environment must be an object")
    backend = str(payload.get("backend") or "").strip().lower()
    if backend != "cuda":
        raise CalibrationError(f"{label} environment backend must be 'cuda'")
    if payload.get("simulated") is not False:
        raise CalibrationError(
            f"{label} environment simulated must be false"
        )
    gpu_name = str(payload.get("gpu_name") or "").strip()
    if not gpu_name:
        raise CalibrationError(
            f"{label} environment gpu_name must be a non-empty string"
        )
    compute_capability = _normalize_compute_capability(
        payload.get("compute_capability"),
        label=f"{label} environment compute_capability",
    )
    expected_capability = str(context["compute_capability"])
    if compute_capability != expected_capability:
        raise CalibrationError(
            f"{label} compute_capability {compute_capability!r} does not "
            f"match target profile {expected_capability!r}"
        )
    total_memory_bytes = payload.get("total_memory_bytes")
    if (
        not isinstance(total_memory_bytes, int)
        or isinstance(total_memory_bytes, bool)
        or total_memory_bytes <= 0
    ):
        raise CalibrationError(
            f"{label} environment total_memory_bytes must be a positive "
            "integer"
        )
    expected_memory_bytes = int(context["profile"]["memory_bytes"])
    capacity_difference_bytes = total_memory_bytes - expected_memory_bytes
    capacity_difference_percent = (
        abs(capacity_difference_bytes) / expected_memory_bytes * 100
    )
    if (
        capacity_difference_percent
        > _SERVING_TARGET_MEMORY_TOLERANCE_PERCENT
    ):
        raise CalibrationError(
            f"{label} total_memory_bytes {total_memory_bytes} does not "
            f"match target profile {expected_memory_bytes} within "
            f"{_SERVING_TARGET_MEMORY_TOLERANCE_PERCENT:.1f}%"
        )

    software = payload.get("software")
    if not isinstance(software, Mapping):
        raise CalibrationError(f"{label} environment software must be an object")
    normalized_software = {str(key): value for key, value in software.items()}
    for key in ("framework", "framework_version", "cuda_version"):
        value = normalized_software.get(key)
        if not isinstance(value, str) or not value.strip():
            raise CalibrationError(
                f"{label} environment software.{key} must be a "
                "non-empty string"
            )

    normalized: dict[str, Any] = {
        "backend": backend,
        "simulated": False,
        "gpu_name": gpu_name,
        "compute_capability": compute_capability,
        "total_memory_bytes": total_memory_bytes,
        "target_profile_memory_bytes": expected_memory_bytes,
        "target_capacity_difference_bytes": capacity_difference_bytes,
        "target_capacity_difference_percent": round(
            capacity_difference_percent,
            6,
        ),
        "target_capacity_tolerance_percent": (
            _SERVING_TARGET_MEMORY_TOLERANCE_PERCENT
        ),
        "software": normalized_software,
    }
    for key in (
        "gpu_uuid",
        "driver_version",
        "device_index",
        "mig_profile",
        "allocator",
    ):
        if key in payload and payload[key] is not None:
            normalized[key] = payload[key]
    return normalized


def _normalize_compute_capability(value: Any, *, label: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
        and all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in value
        )
    ):
        return f"{value[0]}.{value[1]}"
    raise CalibrationError(f"{label} must be a string or [major, minor]")


def _runner_diagnostic(value: str, *, limit: int = 500) -> str:
    normalized = " ".join(value.strip().split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def _parse_peak_samples(
    values: Sequence[str],
    *,
    label: str,
) -> list[int]:
    samples: list[int] = []
    for group in values:
        for raw_value in str(group).split(","):
            normalized = raw_value.strip().replace("_", "")
            if not normalized:
                raise CalibrationError(
                    f"{label} peak samples contain an empty value"
                )
            try:
                value = int(normalized)
            except ValueError as exc:
                raise CalibrationError(
                    f"{label} peak sample must be an integer: "
                    f"{raw_value!r}"
                ) from exc
            if value <= 0:
                raise CalibrationError(
                    f"{label} peak sample must be positive: {value}"
                )
            samples.append(value)
    return samples


def _print_serving_observation(report: Mapping[str, Any]) -> None:
    measurement = report["measurement"]
    print("FakeGPU serving memory observation")
    print(f"  evidence status: {report['evidence_status']}")
    print(f"  workload signature: {report['workload_signature']}")
    print(f"  profile: {report['profile']}")
    for phase in report["memory_timeline"]["phases"]:
        print(
            f"  {phase['phase']}: {phase['sample_count']} samples, "
            f"maximum {int(phase['peak_bytes']) / 2**20:.2f} MiB"
        )
    print(
        "  minimum samples per phase: "
        f"{measurement['minimum_samples_per_phase']}"
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
