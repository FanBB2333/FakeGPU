"""The serving-sample protocol shared with real-CUDA measurement runners.

Split out of ``calibration`` unchanged. A runner — FakeGPU's own
Transformers adapter, vLLM, or a custom script — reports one measured
serving peak as a ``fakegpu.serving_peak_sample.v1`` document, optionally
prefixed with ``FAKEGPU_SERVING_SAMPLE=`` when its stdout also carries logs.
This module owns that document: how it is built, validated, parsed back out
of runner output, and how the environment a runner reports is checked
against the plan it claims to measure.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class CalibrationError(ValueError):
    pass


SERVING_SAMPLE_SCHEMA_VERSION = "fakegpu.serving_peak_sample.v1"


_SERVING_SAMPLE_MARKER = "FAKEGPU_SERVING_SAMPLE="


_SERVING_PLAN_SCHEMA_VERSIONS = frozenset(
    {
        "fakegpu.llm_serving_plan.v1",
        "fakegpu.llm_serving_request_set_plan.v1",
    }
)


_SERVING_TARGET_MEMORY_TOLERANCE_PERCENT = 2.0


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CalibrationError(f"{label} must be a non-empty string")
    return value.strip()


def _positive_plan_integer(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise CalibrationError(f"serving plan {key} must be positive")
    return int(value)


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

    for line in reversed(lines):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(candidate, Mapping):
            continue
        if candidate.get("schema_version") == SERVING_SAMPLE_SCHEMA_VERSION:
            return candidate
    raise CalibrationError(
        f"serving runner repetition {run_index} did not emit a "
        f"{SERVING_SAMPLE_SCHEMA_VERSION!r} JSON object; use the "
        f"{_SERVING_SAMPLE_MARKER!r} prefix when stdout has logs"
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


def _runner_diagnostic(value: str, *, limit: int = 500) -> str:
    normalized = " ".join(value.strip().split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."
