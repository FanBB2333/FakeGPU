from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

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


def _write_profile(tmp_path, extra: str) -> None:
    """Write gtx980 with only its required fields plus ``extra``."""

    source = (profile_directory() / get_profile("gtx980").profile_path).read_text(encoding="utf-8")
    head, _, rest = source.partition("pci_device_id:")
    pci_line, _, _ = rest.partition("\n")
    types = source[source.index("supported_types:"):].split("throughput_published:")[0]
    target = tmp_path / "maxwell" / "consumer" / "gtx980.yaml"
    target.parent.mkdir(parents=True)
    target.write_text(f"{head}pci_device_id:{pci_line}\n{extra}{types}", encoding="utf-8")


def test_optional_specifications_and_throughput_provenance_are_parsed(tmp_path) -> None:
    _write_profile(
        tmp_path,
        "memory_type: GDDR5\nmemory_bandwidth_gbps: 224\nform_factor: pcie\npcie_generation: 3\n"
        "pcie_lanes: 16\nnvlink_generation: 0\nmig_max_instances: 0\nfp32_tflops: 4.981\n"
        "tensor_fp16_tflops: 0\nthroughput_source: https://www.nvidia.com/example\n"
        "throughput_published:\n  - fp32_tflops\nthroughput_derived:\n  - tensor_fp16_tflops\n",
    )
    profile = load_profiles(tmp_path)["gtx980"]

    assert (profile.memory_type, profile.memory_bandwidth_gbps, profile.form_factor) == ("GDDR5", 224.0, "pcie")
    assert (profile.pcie_generation, profile.nvlink_generation, profile.mig_max_instances) == (3, 0, 0)
    assert profile.base_clock_mhz is None and profile.ecc == ""
    assert profile.throughput == {"fp32_tflops": 4.981, "tensor_fp16_tflops": 0.0}
    assert profile.throughput_basis == {"fp32_tflops": "published", "tensor_fp16_tflops": "derived"}
    assert validate_catalog({profile.id: profile}).ok


def test_throughput_values_require_provenance(tmp_path) -> None:
    _write_profile(tmp_path, "fp32_tflops: 4.981\n")
    with pytest.raises(ValueError, match="throughput_published or throughput_derived"):
        load_profiles(tmp_path)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"memory_type": "GDDR9"}, "unknown memory_type"),
        ({"form_factor": "oam"}, "unknown form_factor"),
        ({"ecc": "maybe"}, "ecc must be"),
        ({"nvlink_generation": 0, "nvlink_bandwidth_gbps": 300.0}, "contradicts NVLink"),
        ({"base_clock_mhz": 5000}, "base_clock_mhz exceeds"),
        ({"min_power_limit_mw": 999_000_000}, "min_power_limit_mw exceeds"),
        (
            {"throughput": {"fp32_tflops": 1.0}, "throughput_basis": {"fp32_tflops": "published"}, "throughput_source": ""},
            "throughput_source",
        ),
    ],
)
def test_catalog_rejects_inconsistent_optional_specifications(changes: dict, message: str) -> None:
    profile = replace(get_profile("gtx980"), **changes)
    assert any(message in error for error in validate_catalog({profile.id: profile}).errors)


ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = Path(os.environ.get("FAKEGPU_BUILD_DIR", ROOT / "build"))
NVML_LIBRARY = BUILD_DIR / ("libnvidia-ml.dylib" if sys.platform == "darwin" else "libnvidia-ml.so.1")


@pytest.mark.skipif(not NVML_LIBRARY.exists(), reason="native libraries are not built")
def test_nvml_reports_architecture_as_nvml_device_arch_constants(tmp_path) -> None:
    # NVML_DEVICE_ARCH_* in nvml.h: MAXWELL 3, PASCAL 4, VOLTA 5, TURING 6,
    # AMPERE 7, ADA 8, HOPPER 9, BLACKWELL 10.
    expected = {"gtx980": 3, "p100": 4, "v100": 5, "t4": 6, "a100": 7, "l40s": 8, "h100": 9, "b200": 10}
    code = textwrap.dedent(
        f"""
        import ctypes, json
        nvml = ctypes.CDLL({str(NVML_LIBRARY)!r})
        nvml.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        nvml.nvmlDeviceGetArchitecture.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
        assert nvml.nvmlInit_v2() == 0
        values = []
        for index in range({len(expected)}):
            handle, arch = ctypes.c_void_p(), ctypes.c_uint()
            assert nvml.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)) == 0
            assert nvml.nvmlDeviceGetArchitecture(handle, ctypes.byref(arch)) == 0
            values.append(arch.value)
        nvml.nvmlShutdown()
        print(json.dumps(values))
        """
    )
    env = dict(
        os.environ,
        FAKEGPU_PROFILES=",".join(expected),
        FAKEGPU_HOME=str(tmp_path),
        FAKEGPU_MACHINE="off",
        FAKEGPU_TERMINAL_REPORT="0",
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env, text=True, capture_output=True, timeout=60
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == list(expected.values())
