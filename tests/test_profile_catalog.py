from __future__ import annotations

from dataclasses import replace

import pytest
import yaml

from fakegpu.profile_catalog import get_profile, load_profiles, profile_directory, validate_catalog
from scripts.render_gpu_catalog import OUTPUT, WEBSITE, render, render_website


def test_public_inventory_matches_runtime_profiles() -> None:
    assert OUTPUT.read_text(encoding="utf-8") == render()


def test_website_inventory_matches_runtime_profiles() -> None:
    source = WEBSITE.read_text(encoding="utf-8")
    assert render_website(source) == source


def test_profiles_are_valid_yaml_for_external_configuration_tools() -> None:
    for profile in load_profiles().values():
        path = profile_directory() / profile.profile_path
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert raw["id"] == profile.id, path
        assert raw["architecture"] == profile.architecture, path
        assert raw["memory_bytes"] == profile.memory_bytes, path


@pytest.mark.parametrize(
    ("profile_id", "capability", "memory_gib", "sm_count"),
    [
        ("gtx980ti", "5.2", 6, 22),
        ("gtx1660-super", "7.5", 6, 22),
        ("rtx2050-laptop", "8.6", 4, 16),
        ("rtx3050a-laptop", "8.9", 4, 14),
        ("rtx4060ti-8gb", "8.9", 8, 34),
        ("rtx5090d-v2", "12.0", 24, 170),
        ("m10", "5.0", 8, 5),
        ("a16", "8.6", 16, 10),
        ("a10g", "8.6", 24, 80),
        ("h100-nvl-94gb", "9.0", 94, 132),
        ("quadro-gp100", "6.0", 16, 56),
        ("quadro-gv100", "7.0", 32, 80),
    ],
)
def test_researched_variants_keep_product_specific_resources(
    profile_id: str, capability: str, memory_gib: int, sm_count: int
) -> None:
    profile = get_profile(profile_id)
    assert profile.compute_capability_text == capability
    assert profile.memory_bytes == memory_gib * 2**30
    assert profile.sm_count == sm_count


def test_reference_profile_can_report_unknown_specs_without_claiming_measurements() -> None:
    profile = replace(get_profile("gtx980"), sm_count=0, core_clock_mhz=0)
    validation = validate_catalog({profile.id: profile})
    assert validation.ok
    assert "sm_count" in profile.to_dict()["unavailable_fields"]
    assert "core_clock_mhz" in profile.to_dict()["unavailable_fields"]


@pytest.mark.parametrize("status", ["measured", "synthetic"])
def test_non_reference_profile_requires_known_sm_count(status: str) -> None:
    profile = replace(get_profile("gtx980"), profile_status=status, sm_count=0)
    assert not validate_catalog({profile.id: profile}).ok


@pytest.mark.parametrize("field", ["sm_count", "memory_clock_mhz", "max_power_limit_mw", "pci_device_id"])
def test_catalog_rejects_negative_device_properties(field: str) -> None:
    profile = replace(get_profile("gtx980"), **{field: -1})
    validation = validate_catalog({profile.id: profile})
    assert any(f"{field} must not be negative" in error for error in validation.errors)
