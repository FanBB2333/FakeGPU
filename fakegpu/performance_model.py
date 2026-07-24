from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .profile_catalog import get_profile


SCHEMA_VERSION = "fakegpu.roofline_estimate.v1"

# Scalar FP32 CUDA-core issue width. Tensor-core throughput is deliberately not
# inferred from marketing names; callers can provide an explicit acceleration
# factor for a measured or documented matrix path.
_FP32_CORES_PER_SM = {
    (5, 0): 128,
    (5, 2): 128,
    (5, 3): 128,
    (6, 0): 64,
    (6, 1): 128,
    (6, 2): 128,
    (7, 0): 64,
    (7, 2): 64,
    (7, 5): 64,
    (8, 0): 64,
    (8, 6): 128,
    (8, 7): 128,
    (8, 9): 128,
    (9, 0): 128,
    (10, 0): 128,
    (10, 3): 128,
    (11, 0): 128,
    (12, 0): 128,
    (12, 1): 128,
}


class PerformanceModelError(ValueError):
    pass


def profile_roofline(profile_id: str) -> dict[str, Any]:
    """Return a profile-derived scalar compute and memory-bandwidth ceiling."""

    profile = get_profile(profile_id)
    try:
        fp32_cores_per_sm = _FP32_CORES_PER_SM[profile.compute_capability]
    except KeyError as exc:
        raise PerformanceModelError(
            "no scalar issue-width model for compute capability "
            f"{profile.compute_capability_text}"
        ) from exc

    scalar_fp32_peak = (
        profile.sm_count
        * fp32_cores_per_sm
        * 2
        * profile.core_clock_mhz
        * 1_000_000
    )
    memory_bandwidth = (
        profile.memory_bus_width_bits
        / 8
        * profile.memory_clock_mhz
        * 1_000_000
        * 2
    )
    return {
        "profile_id": profile.id,
        "profile_name": profile.name,
        "architecture": profile.architecture,
        "segment": profile.segment,
        "compute_capability": profile.compute_capability_text,
        "compiler_target": profile.compiler_target,
        "sm_count": profile.sm_count,
        "core_clock_mhz": profile.core_clock_mhz,
        "memory_clock_mhz": profile.memory_clock_mhz,
        "memory_bus_width_bits": profile.memory_bus_width_bits,
        "fp32_cores_per_sm": fp32_cores_per_sm,
        "scalar_fp32_peak_flops_per_second": int(scalar_fp32_peak),
        "profile_memory_bandwidth_bytes_per_second": int(memory_bandwidth),
        "ridge_point_flops_per_byte": scalar_fp32_peak / memory_bandwidth,
        "profile_status": profile.profile_status,
        "spec_source": profile.spec_source,
    }


def estimate_roofline(
    *,
    profile_id: str,
    flops: int,
    memory_bytes: int,
    launch_count: int = 1,
    compute_acceleration_factor: float = 1.0,
    efficiency: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Estimate an analytical latency interval from a GPU profile.

    ``compute_acceleration_factor`` is explicit because tensor-core rates vary
    by dtype, sparsity, instruction shape, and product segment. A value of one
    uses only the scalar FP32 FMA ceiling derived from the profile.
    """

    if not isinstance(flops, int) or isinstance(flops, bool) or flops < 0:
        raise PerformanceModelError("flops must be a non-negative integer")
    if (
        not isinstance(memory_bytes, int)
        or isinstance(memory_bytes, bool)
        or memory_bytes < 0
    ):
        raise PerformanceModelError(
            "memory_bytes must be a non-negative integer"
        )
    if flops == 0 and memory_bytes == 0:
        raise PerformanceModelError(
            "at least one of flops or memory_bytes must be positive"
        )
    if (
        not isinstance(launch_count, int)
        or isinstance(launch_count, bool)
        or launch_count < 0
    ):
        raise PerformanceModelError(
            "launch_count must be a non-negative integer"
        )
    if (
        not math.isfinite(compute_acceleration_factor)
        or compute_acceleration_factor <= 0
    ):
        raise PerformanceModelError(
            "compute_acceleration_factor must be finite and positive"
        )

    assumptions = _efficiency_assumptions(efficiency)
    hardware = profile_roofline(profile_id)
    compute_ceiling = (
        hardware["scalar_fp32_peak_flops_per_second"]
        * compute_acceleration_factor
    )
    bandwidth_ceiling = hardware[
        "profile_memory_bandwidth_bytes_per_second"
    ]
    intervals: dict[str, dict[str, Any]] = {}
    for name in ("optimistic", "expected", "conservative"):
        selected = assumptions[name]
        compute_seconds = (
            flops / (compute_ceiling * selected["compute_efficiency"])
            if flops
            else 0.0
        )
        memory_seconds = (
            memory_bytes
            / (bandwidth_ceiling * selected["memory_efficiency"])
            if memory_bytes
            else 0.0
        )
        launch_seconds = (
            launch_count * selected["launch_overhead_microseconds"] / 1_000_000
        )
        intervals[name] = {
            "seconds": max(compute_seconds, memory_seconds) + launch_seconds,
            "compute_seconds": compute_seconds,
            "memory_seconds": memory_seconds,
            "launch_seconds": launch_seconds,
            **selected,
        }

    ideal_compute_seconds = flops / compute_ceiling if flops else 0.0
    ideal_memory_seconds = (
        memory_bytes / bandwidth_ceiling if memory_bytes else 0.0
    )
    arithmetic_intensity = flops / memory_bytes if memory_bytes else None
    if ideal_compute_seconds > ideal_memory_seconds:
        bottleneck = "compute"
    elif ideal_memory_seconds > ideal_compute_seconds:
        bottleneck = "memory"
    else:
        bottleneck = "balanced"

    return {
        "schema_version": SCHEMA_VERSION,
        "method": "profile_scalar_roofline_with_efficiency_interval",
        "profile": hardware,
        "workload": {
            "flops": flops,
            "memory_bytes": memory_bytes,
            "launch_count": launch_count,
            "arithmetic_intensity_flops_per_byte": arithmetic_intensity,
        },
        "compute_model": {
            "basis": "scalar_fp32_fma",
            "compute_acceleration_factor": compute_acceleration_factor,
            "effective_compute_ceiling_flops_per_second": int(
                compute_ceiling
            ),
        },
        "bottleneck": bottleneck,
        "latency_interval_seconds": {
            "lower": intervals["optimistic"]["seconds"],
            "expected": intervals["expected"]["seconds"],
            "upper": intervals["conservative"]["seconds"],
        },
        "scenarios": intervals,
        "confidence": "P1_profile_analytical_range",
        "limitations": [
            "This is a roofline interval, not a kernel benchmark or scheduler simulation.",
            "Profile clocks and memory clocks are ceilings; power, thermals, occupancy, and contention are not simulated.",
            "Tensor-core acceleration is included only when the caller supplies an explicit factor.",
            "Kernel fusion changes both launch count and memory traffic and must be reflected in the workload inputs.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fakegpu estimate-roofline",
        description=(
            "Estimate a profile-aware analytical latency interval from FLOPs, "
            "memory traffic, and launch count."
        ),
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--flops", type=int, required=True)
    parser.add_argument("--memory-bytes", type=int, required=True)
    parser.add_argument("--launch-count", type=int, default=1)
    parser.add_argument(
        "--compute-acceleration-factor",
        type=float,
        default=1.0,
        help="Explicit matrix/tensor throughput factor over scalar FP32.",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        nargs="?",
        const="-",
        help="Write JSON to PATH, or stdout when PATH is omitted.",
    )
    args = parser.parse_args(argv)
    try:
        report = estimate_roofline(
            profile_id=args.profile,
            flops=args.flops,
            memory_bytes=args.memory_bytes,
            launch_count=args.launch_count,
            compute_acceleration_factor=args.compute_acceleration_factor,
        )
    except (OSError, PerformanceModelError, ValueError) as exc:
        parser.exit(2, f"fakegpu estimate-roofline: {exc}\n")

    if args.json_path:
        payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.json_path == "-":
            print(payload, end="")
        else:
            output = Path(args.json_path).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(payload, encoding="utf-8")
            print(f"Roofline estimate: {output}")
    else:
        interval = report["latency_interval_seconds"]
        print("FakeGPU analytical roofline estimate")
        print(f"  profile: {report['profile']['profile_id']}")
        print(f"  bottleneck: {report['bottleneck']}")
        print(
            "  latency: "
            f"{interval['lower'] * 1_000:.3f} / "
            f"{interval['expected'] * 1_000:.3f} / "
            f"{interval['upper'] * 1_000:.3f} ms "
            "(lower / expected / upper)"
        )
    return 0


def _efficiency_assumptions(
    override: Mapping[str, float] | None,
) -> dict[str, dict[str, float]]:
    assumptions = {
        "optimistic": {
            "compute_efficiency": 0.85,
            "memory_efficiency": 0.85,
            "launch_overhead_microseconds": 2.0,
        },
        "expected": {
            "compute_efficiency": 0.55,
            "memory_efficiency": 0.65,
            "launch_overhead_microseconds": 8.0,
        },
        "conservative": {
            "compute_efficiency": 0.20,
            "memory_efficiency": 0.35,
            "launch_overhead_microseconds": 25.0,
        },
    }
    if override is None:
        return assumptions
    unknown = sorted(set(override) - {"compute", "memory", "launch_us"})
    if unknown:
        raise PerformanceModelError(
            "unknown efficiency overrides: " + ", ".join(unknown)
        )
    expected = assumptions["expected"]
    for key, destination in (
        ("compute", "compute_efficiency"),
        ("memory", "memory_efficiency"),
        ("launch_us", "launch_overhead_microseconds"),
    ):
        if key not in override:
            continue
        value = float(override[key])
        if not math.isfinite(value) or value <= 0:
            raise PerformanceModelError(
                f"efficiency override {key!r} must be finite and positive"
            )
        if key != "launch_us" and value > 1:
            raise PerformanceModelError(
                f"efficiency override {key!r} must not exceed one"
            )
        lower_limit, upper_limit = {
            "compute": (0.20, 0.85),
            "memory": (0.35, 0.85),
            "launch_us": (2.0, 25.0),
        }[key]
        if not lower_limit <= value <= upper_limit:
            raise PerformanceModelError(
                f"efficiency override {key!r} must be between "
                f"{lower_limit} and {upper_limit} to preserve the interval"
            )
        expected[destination] = value
    return assumptions


__all__ = [
    "PerformanceModelError",
    "SCHEMA_VERSION",
    "estimate_roofline",
    "profile_roofline",
]
