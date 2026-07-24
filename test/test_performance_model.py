from __future__ import annotations

import math

import pytest

from fakegpu.performance_model import (
    PerformanceModelError,
    estimate_roofline,
    profile_roofline,
)
from fakegpu.profile_catalog import load_profiles


def test_roofline_uses_profile_architecture_and_clock_data() -> None:
    turing = profile_roofline("rtx2080ti")
    ampere = profile_roofline("rtx3090ti")
    blackwell = profile_roofline("rtx-pro-5000-blackwell")

    assert turing["architecture"] == "turing"
    assert turing["compiler_target"] == "sm_75"
    assert ampere["architecture"] == "ampere"
    assert ampere["scalar_fp32_peak_flops_per_second"] > turing[
        "scalar_fp32_peak_flops_per_second"
    ]
    assert blackwell["architecture"] == "blackwell"
    assert blackwell["compiler_target"] == "sm_120"


def test_roofline_supports_every_catalog_profile() -> None:
    profiles = load_profiles()
    for profile_id, profile in profiles.items():
        report = profile_roofline(profile_id)
        assert report["architecture"] == profile.architecture
        assert report["compute_capability"] == (
            profile.compute_capability_text
        )
        assert report["scalar_fp32_peak_flops_per_second"] > 0
        assert report["profile_memory_bandwidth_bytes_per_second"] > 0


def test_roofline_returns_ordered_interval_and_bottleneck() -> None:
    report = estimate_roofline(
        profile_id="rtx3090ti",
        flops=2_000_000_000_000,
        memory_bytes=50_000_000,
        launch_count=10,
    )
    interval = report["latency_interval_seconds"]

    assert 0 < interval["lower"] < interval["expected"] < interval["upper"]
    assert report["bottleneck"] == "compute"
    assert report["compute_model"]["basis"] == "scalar_fp32_fma"

    memory_bound = estimate_roofline(
        profile_id="rtx3090ti",
        flops=1_000,
        memory_bytes=10_000_000_000,
    )
    assert memory_bound["bottleneck"] == "memory"


def test_explicit_acceleration_reduces_compute_bound_latency() -> None:
    scalar = estimate_roofline(
        profile_id="a100",
        flops=1_000_000_000_000,
        memory_bytes=1,
    )
    accelerated = estimate_roofline(
        profile_id="a100",
        flops=1_000_000_000_000,
        memory_bytes=1,
        compute_acceleration_factor=8,
    )
    assert accelerated["latency_interval_seconds"]["expected"] < scalar[
        "latency_interval_seconds"
    ]["expected"]

    zero_traffic = estimate_roofline(
        profile_id="a100",
        flops=1,
        memory_bytes=0,
    )
    assert zero_traffic["workload"][
        "arithmetic_intensity_flops_per_byte"
    ] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"flops": -1, "memory_bytes": 1},
        {"flops": 1, "memory_bytes": -1},
        {"flops": 0, "memory_bytes": 0},
        {"flops": 1, "memory_bytes": 1, "launch_count": -1},
        {
            "flops": 1,
            "memory_bytes": 1,
            "compute_acceleration_factor": math.inf,
        },
    ],
)
def test_roofline_rejects_invalid_workload(kwargs: dict[str, object]) -> None:
    with pytest.raises(PerformanceModelError):
        estimate_roofline(profile_id="a100", **kwargs)
