from __future__ import annotations

import copy
import ctypes

import pytest

from fakegpu._monitor_reference import ReferenceTelemetry
from fakegpu.nvml import _DeviceV2, _ProcessV2
from fakegpu.profile_catalog import get_profile


def inventory(*, profile="a100", busy=0, io_bytes=0, timestamp=10**9, used=2**30):
    return {
        "devices": [{"uuid": "synthetic-test", "profile_id": profile, "index": 0,
                     "memory": {"used_bytes": used}}],
        "processes": [{"gpu_uuid": "synthetic-test", "pid": 123, "profile_id": profile,
                       "timestamp_ns": timestamp, "activity": {
                           "source": "cpu_dispatch_proxy", "busy_ns": busy, "io_bytes": io_bytes,
                       }}],
    }


def test_resident_memory_is_not_engine_utilization():
    model = ReferenceTelemetry()
    devices, _ = model.sample(inventory(used=70 * 2**30), now=0)
    sample = devices["synthetic-test"]
    assert sample["gpu"] == sample["memory"] == 0
    assert sample["temperature"] == 29 and sample["power_mw"] == 56000
    assert sample["pstate"] == 0


def test_activity_changes_sensors_and_cools_after_idle():
    model = ReferenceTelemetry()
    model.sample(inventory(), now=0)
    active = inventory(busy=10**9, io_bytes=2039 * 10**9, timestamp=2 * 10**9)
    devices, processes = model.sample(active, now=10)
    assert devices["synthetic-test"]["gpu"] == 100
    assert devices["synthetic-test"]["memory"] == 100
    assert devices["synthetic-test"]["power_mw"] == 400000
    assert devices["synthetic-test"]["sm_clock"] == 1410
    hot = devices["synthetic-test"]["temperature"]
    assert 29 < hot < 74
    assert processes[("synthetic-test", 123)]["gpu"] == 100
    # Polling one publisher sample twice must not turn utilization into zero.
    again, _ = model.sample(active, now=10)
    assert again == devices
    cool, _ = model.sample(inventory(busy=10**9, io_bytes=2039 * 10**9, timestamp=3 * 10**9), now=20)
    assert cool["synthetic-test"]["gpu"] == 0
    assert cool["synthetic-test"]["power_mw"] == 56000
    assert 29 < cool["synthetic-test"]["temperature"] < hot


@pytest.mark.parametrize("profile", ["a100-pcie-80gb", "a100-sxm4-40gb", "h100", "rtx4090"])
def test_other_profiles_follow_the_default_curve(profile):
    model = ReferenceTelemetry()
    assert model.source(profile) == "default_model" and model.source("a100") == "a100_reference_model"
    devices, _ = model.sample(inventory(profile=profile), now=0)
    idle = devices["synthetic-test"]
    # A100 observations are never applied to another profile.
    assert (idle["temperature"], idle["power_mw"], idle["gpu"], idle["sm_clock"]) == (30, 1000, 0, 210)
    catalog = get_profile(profile)
    devices, _ = model.sample(inventory(profile=profile, busy=10**9, timestamp=2 * 10**9), now=10)
    busy = devices["synthetic-test"]
    assert busy["power_mw"] == catalog.max_power_limit_mw
    assert (busy["sm_clock"], busy["memory_clock"]) == (catalog.core_clock_mhz, catalog.memory_clock_mhz)
    assert 30 < busy["temperature"] < 75


def test_default_curve_falls_back_where_the_catalog_has_no_figure():
    model = ReferenceTelemetry()
    model.sample(inventory(profile="b100"), now=0)
    devices, _ = model.sample(inventory(profile="b100", busy=10**9, timestamp=2 * 10**9), now=10)
    busy = devices["synthetic-test"]
    assert (busy["power_mw"], busy["sm_clock"], busy["memory_clock"]) == (300000, 1500, 1500)


def test_profiles_outside_the_catalog_remain_unknown():
    assert ReferenceTelemetry().sample(inventory(profile="unknown")) == ({}, {})


def test_missing_observations_remain_unknown():
    data = inventory()
    data["processes"][0]["activity"] = {}
    assert ReferenceTelemetry().sample(data) == ({}, {})


def test_multiple_processes_cap_activity_and_reset_on_exit():
    data = inventory()
    data["processes"].append(copy.deepcopy(data["processes"][0]))
    data["processes"][1]["pid"] = 456
    model = ReferenceTelemetry()
    model.sample(data)
    for process in data["processes"]:
        process["timestamp_ns"] += 10**9
        process["activity"]["busy_ns"] = 900_000_000
    devices, processes = model.sample(data)
    assert devices["synthetic-test"]["gpu"] == 100
    assert all(p["gpu"] == 90 for p in processes.values())
    assert model.sample({"devices": [], "processes": []}) == ({}, {})
    assert not model._previous and not model._temperatures


def test_counter_restart_does_not_create_negative_utilization():
    model = ReferenceTelemetry()
    model.sample(inventory(busy=10**9))
    devices, _ = model.sample(inventory(busy=0, timestamp=2 * 10**9))
    assert devices["synthetic-test"]["gpu"] == 0


def test_version_two_snapshot_layout_matches_native_abi():
    assert ctypes.sizeof(_DeviceV2) == 80
    assert _DeviceV2.telemetry.offset == 48
    assert ctypes.sizeof(_ProcessV2) == 32
    assert _ProcessV2.timestamp_us.offset == 16
