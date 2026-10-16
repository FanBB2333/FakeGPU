"""Peak throughput derivation and the checked-in rate table."""

from __future__ import annotations

from dataclasses import replace

import pytest

from fakegpu.performance_model import PerformanceModelError, estimate_roofline, profile_roofline
from fakegpu.profile_catalog import get_profile, load_profiles
from fakegpu.throughput import derive_throughput, per_sm_rates

RATES = {
    "schema": "fakegpu.throughput_rates.v1",
    "rates": [
        {"compute_capability": "5.2", "class": "all", "per_sm_per_clock": {"fp32_tflops": 256, "fp64_tflops": 8}},
        {"compute_capability": "5.2", "class": "geforce", "per_sm_per_clock": {"fp64_tflops": 4}},
        {"compute_capability": "5.2", "class": "professional", "per_sm_per_clock": {"fp64_tflops": 16}},
    ],
    "overrides": [{"profile_id": "gtx980", "per_sm_per_clock": {"tensor_fp16_tflops": 0}}],
}


def test_rates_merge_class_entries_and_profile_overrides() -> None:
    consumer = get_profile("gtx980")
    assert per_sm_rates(consumer, RATES) == {"fp32_tflops": 256, "fp64_tflops": 4, "tensor_fp16_tflops": 0}
    professional = replace(consumer, id="quadro", segment="workstation")
    assert per_sm_rates(professional, RATES) == {"fp32_tflops": 256, "fp64_tflops": 16}


def test_derived_peak_is_rate_times_sms_times_boost_clock() -> None:
    profile = replace(get_profile("gtx980"), sm_count=16, core_clock_mhz=1216)
    derived = derive_throughput(profile, RATES)
    assert derived["fp32_tflops"] == pytest.approx(256 * 16 * 1216e6 / 1e12, rel=1e-3)
    assert derive_throughput(replace(profile, core_clock_mhz=0), RATES) == {}


def test_profiles_keep_derived_throughput_in_sync_with_rate_table() -> None:
    for profile in load_profiles().values():
        derived = derive_throughput(profile)
        for key, basis in profile.throughput_basis.items():
            if basis == "derived":
                assert profile.throughput[key] == derived.get(key), (profile.id, key)
        missing = set(derived) - set(profile.throughput)
        assert not missing, (profile.id, sorted(missing))


def test_roofline_selects_the_requested_precision_peak(monkeypatch) -> None:
    profile = replace(
        get_profile("gtx980"),
        throughput={"fp32_tflops": 5.0, "tensor_bf16_tflops": 40.0},
        throughput_basis={"fp32_tflops": "published", "tensor_bf16_tflops": "derived"},
        throughput_source="https://www.nvidia.com/example",
        memory_bandwidth_gbps=100.0,
    )
    monkeypatch.setattr("fakegpu.performance_model.get_profile", lambda _profile_id: profile)

    hardware = profile_roofline(profile.id)
    assert hardware["memory_bandwidth_basis"] == "published"
    assert hardware["profile_memory_bandwidth_bytes_per_second"] == 100_000_000_000
    scalar = estimate_roofline(profile_id=profile.id, flops=10**13, memory_bytes=1)
    tensor = estimate_roofline(profile_id=profile.id, flops=10**13, memory_bytes=1, compute_precision="tensor_bf16")
    assert tensor["compute_model"]["peak_basis"] == "derived"
    assert tensor["compute_model"]["effective_compute_ceiling_flops_per_second"] == 40 * 10**12
    assert tensor["latency_interval_seconds"]["expected"] < scalar["latency_interval_seconds"]["expected"]
    with pytest.raises(PerformanceModelError, match="no known tensor_fp8 throughput"):
        estimate_roofline(profile_id=profile.id, flops=1, memory_bytes=1, compute_precision="tensor_fp8")
