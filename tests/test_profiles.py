from __future__ import annotations

import pytest

from fakegpu._profiles import _PROFILE_CATALOG, _resolve_per_device_profiles


@pytest.fixture(autouse=True)
def clear_profile_environment(monkeypatch):
    for key in ("FAKEGPU_PROFILES", "FAKEGPU_PROFILE", "FAKEGPU_DEVICE_NAME"):
        monkeypatch.delenv(key, raising=False)


def test_every_catalog_profile_resolves_consistently(monkeypatch):
    for profile_id, profile in _PROFILE_CATALOG.items():
        monkeypatch.setenv("FAKEGPU_PROFILE", profile_id)
        uniform = _resolve_per_device_profiles(2)
        monkeypatch.setenv("FAKEGPU_PROFILES", f"{profile_id}:2")
        assert _resolve_per_device_profiles(2) == uniform
        assert uniform[0]["total_memory"] == profile.memory_bytes
        assert uniform[0]["name"] == profile.torch_name
        assert (uniform[0]["compute_major"], uniform[0]["compute_minor"]) == profile.compute_capability
        monkeypatch.delenv("FAKEGPU_PROFILES")


def test_profile_padding_and_truncation_keep_independent_devices(monkeypatch):
    monkeypatch.setenv("FAKEGPU_PROFILES", "a100:2, h100:1")
    devices = _resolve_per_device_profiles(5)
    assert [device["profile_id"] for device in devices] == ["a100", "a100", "h100", "h100", "h100"]
    assert _resolve_per_device_profiles(1) == devices[:1]
    assert _resolve_per_device_profiles(0) == []
    devices[-1]["name"] = "changed"
    assert devices[-2]["name"] != "changed"


@pytest.mark.parametrize("spec", ["", ",,", "a100:0", "unknown:2"])
def test_empty_and_unknown_profile_specs_keep_default_capacity(monkeypatch, spec):
    monkeypatch.setenv("FAKEGPU_PROFILES", spec)
    devices = _resolve_per_device_profiles(2)
    assert len(devices) == 2
    assert all(device["total_memory"] == 80 * 1024**3 for device in devices)
    devices[0]["total_memory"] = 1
    assert devices[1]["total_memory"] == 80 * 1024**3
