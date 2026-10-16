from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from typing import Any

from ._cli import (
    add_json_path_argument,
    command_prog,
    usage_error,
)
from .profile_catalog import THROUGHPUT_FIELDS, get_profile
from .structured_io import emit_json


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


# ``--compute-precision`` names: throughput fields without their unit suffix.
COMPUTE_PRECISIONS = tuple(name.rsplit("_", 1)[0] for name in THROUGHPUT_FIELDS)


class PerformanceModelError(ValueError):
    pass


def profile_roofline(profile_id: str) -> dict[str, Any]:
    """Return a profile's peak compute rates and memory-bandwidth ceiling.

    Peaks come from the profile's published or derived throughput; profiles
    without throughput data fall back to the scalar FP32 FMA rate of the SM
    count and boost clock. Bandwidth prefers NVIDIA's published figure over
    bus width times memory clock.
    """

    profile = get_profile(profile_id)
    peaks = {
        name.rsplit("_", 1)[0]: {
            "operations_per_second": int(value * 1e12),
            "basis": profile.throughput_basis[name],
        }
        for name, value in profile.throughput.items()
        if value > 0
    }
    fp32_cores_per_sm = _FP32_CORES_PER_SM.get(profile.compute_capability)
    if "fp32" not in peaks and fp32_cores_per_sm and profile.sm_count > 0 and profile.core_clock_mhz > 0:
        peaks["fp32"] = {
            "operations_per_second": profile.sm_count * fp32_cores_per_sm * 2 * profile.core_clock_mhz * 1_000_000,
            "basis": "scalar_fma_formula",
        }
    if profile.memory_bandwidth_gbps:
        memory_bandwidth = profile.memory_bandwidth_gbps * 1e9
        bandwidth_basis = "published"
    elif profile.memory_bus_width_bits > 0 and profile.memory_clock_mhz > 0:
        memory_bandwidth = profile.memory_bus_width_bits / 8 * profile.memory_clock_mhz * 1_000_000 * 2
        bandwidth_basis = "bus_width_and_memory_clock"
    else:
        memory_bandwidth = 0.0
        bandwidth_basis = ""
    unavailable = []
    if "fp32" not in peaks:
        unavailable += ["fp32_tflops"] + [key for key in ("sm_count", "core_clock_mhz") if getattr(profile, key) <= 0]
    if not memory_bandwidth:
        unavailable += ["memory_bandwidth_gbps"] + [
            key for key in ("memory_bus_width_bits", "memory_clock_mhz") if getattr(profile, key) <= 0
        ]
    if unavailable:
        raise PerformanceModelError(
            f"profile {profile.id!r} has no verified throughput specifications for "
            f"{', '.join(unavailable)}; use a profile with documented clocks and bandwidth"
        )
    scalar_fp32_peak = peaks["fp32"]["operations_per_second"]
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
        "memory_bandwidth_basis": bandwidth_basis,
        "peak_throughput": peaks,
        "ridge_point_flops_per_byte": scalar_fp32_peak / memory_bandwidth,
        "profile_status": profile.profile_status,
        "spec_source": profile.spec_source,
        "throughput_source": profile.throughput_source,
    }


def estimate_roofline(
    *,
    profile_id: str,
    flops: int,
    memory_bytes: int,
    launch_count: int = 1,
    compute_acceleration_factor: float = 1.0,
    compute_precision: str = "fp32",
) -> dict[str, Any]:
    """Estimate an analytical latency interval from a GPU profile.

    ``compute_precision`` selects the profile's dense peak for that path, for
    example ``tensor_bf16``; the default is the scalar FP32 FMA ceiling.
    ``compute_acceleration_factor`` further scales the selected peak.
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

    if compute_precision not in COMPUTE_PRECISIONS:
        raise PerformanceModelError(
            f"compute_precision must be one of {', '.join(COMPUTE_PRECISIONS)}"
        )

    assumptions = EFFICIENCY_ASSUMPTIONS
    hardware = profile_roofline(profile_id)
    peak = hardware["peak_throughput"].get(compute_precision)
    if peak is None:
        raise PerformanceModelError(
            f"profile {profile_id!r} has no known {compute_precision} throughput"
        )
    compute_ceiling = peak["operations_per_second"] * compute_acceleration_factor
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
            "basis": "scalar_fp32_fma" if compute_precision == "fp32" else f"{compute_precision}_peak",
            "precision": compute_precision,
            "peak_basis": peak["basis"],
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
            "Tensor-core rates apply only when a tensor compute precision is selected; derived peaks are architecture rates times SM count and boost clock.",
            "Kernel fusion changes both launch count and memory traffic and must be reflected in the workload inputs.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=command_prog(__name__),
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
        help="Explicit factor applied to the selected compute peak.",
    )
    parser.add_argument(
        "--compute-precision",
        choices=COMPUTE_PRECISIONS,
        default="fp32",
        help="Peak to use: fp32 (scalar FMA, default) or a tensor path such as tensor_bf16.",
    )
    add_json_path_argument(parser)
    args = parser.parse_args(argv)
    try:
        report = estimate_roofline(
            profile_id=args.profile,
            flops=args.flops,
            memory_bytes=args.memory_bytes,
            launch_count=args.launch_count,
            compute_acceleration_factor=args.compute_acceleration_factor,
            compute_precision=args.compute_precision,
        )
    except (OSError, ValueError) as exc:
        usage_error(parser, exc)

    if args.json_path:
        output = emit_json(args.json_path, report)
        if output is not None:
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


EFFICIENCY_ASSUMPTIONS = {
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


__all__ = [
    "PerformanceModelError",
    "SCHEMA_VERSION",
    "estimate_roofline",
    "profile_roofline",
]
