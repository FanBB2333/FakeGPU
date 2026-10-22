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

from fakegpu.profile_catalog import (
    get_profile,
    load_profiles,
    profile_directory,
    profile_id_for_device_name,
    validate_catalog,
)
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


# Device ID and name rows of the NVIDIA driver supported-chips list
# (https://download.nvidia.com/XFree86/Linux-x86_64/595.99.02/README/supportedchips.html);
# torch.cuda.get_device_name and nvidia-smi report these names.
@pytest.mark.parametrize(
    ("profile_id", "device_id", "driver_name"),
    [
        ("a100", 0x20B2, "NVIDIA A100-SXM4-80GB"),
        ("a100-pcie-80gb", 0x20B5, "NVIDIA A100 80GB PCIe"),
        ("h100", 0x2330, "NVIDIA H100 80GB HBM3"),
        ("h100-pcie-80gb", 0x2331, "NVIDIA H100 PCIe"),
        ("h100-nvl-94gb", 0x2321, "NVIDIA H100 NVL"),
        ("h200", 0x2335, "NVIDIA H200"),
        ("h200-nvl", 0x233B, "NVIDIA H200 NVL"),
        ("h20-96gb", 0x2329, "NVIDIA H20"),
        ("h20-141gb", 0x232C, "NVIDIA H20-3e"),
        ("h800-sxm5-80gb", 0x2324, "NVIDIA H800"),
        ("h800-pcie-80gb", 0x2322, "NVIDIA H800 PCIe"),
        ("h800-nvl-94gb", 0x233A, "NVIDIA H800 NVL"),
        ("gh200-144gb", 0x2348, "NVIDIA GH200 144G HBM3e"),
        # 2342 is GH200 120GB or 480GB by subsystem; NVIDIA's GH200 MGX quick
        # start guide shows a 96 GB GH200 reporting GH200 120GB.
        ("gh200-96gb", 0x2342, "NVIDIA GH200 120GB"),
        ("l20-lc", 0x26BA, "NVIDIA L20"),
        ("l40s", 0x26B9, "NVIDIA L40S"),
        ("t4", 0x1EB8, "Tesla T4"),
        ("rtx3090ti-reference", 0x2203, "NVIDIA GeForce RTX 3090 Ti"),
        ("rtx3050-oem", 0x2508, "NVIDIA GeForce RTX 3050 OEM"),
        # Every device ID the list gives these products carries this one name.
        ("rtx3050-laptop-6gb", None, "NVIDIA GeForce RTX 3050 6GB Laptop GPU"),
        ("rtx2050-laptop", None, "NVIDIA GeForce RTX 2050"),
        # Turing notebook parts: the list names the full-power (non-Max-Q)
        # subsystems after the product alone (1F11/1F15, 1F10/1F14, 1E90,
        # 1E91/1ED1 "Super", 1E93/1ED3 "Super", 1F99/1F9D, 1F95, 2191); only
        # specific OEM subsystems add "with Max-Q Design".
        ("rtx2060-laptop", None, "NVIDIA GeForce RTX 2060"),
        ("rtx2070-laptop", None, "NVIDIA GeForce RTX 2070"),
        ("rtx2070-super-laptop", None, "NVIDIA GeForce RTX 2070 Super"),
        ("rtx2080-laptop", None, "NVIDIA GeForce RTX 2080"),
        ("rtx2080-super-laptop", None, "NVIDIA GeForce RTX 2080 Super"),
        ("gtx1650-laptop", None, "NVIDIA GeForce GTX 1650"),
        ("gtx1650ti-laptop", None, "NVIDIA GeForce GTX 1650 Ti"),
        ("gtx1660ti-laptop", None, "NVIDIA GeForce GTX 1660 Ti"),
        # Memory variants: the list has no "12 GB", "8 GB", "GDDR6X" or
        # "GDDR6" names, so these report the product name.
        ("rtx3080-12gb", None, "NVIDIA GeForce RTX 3080"),
        ("rtx3060-8gb", None, "NVIDIA GeForce RTX 3060"),
        ("rtx3060ti-gddr6x", None, "NVIDIA GeForce RTX 3060 Ti"),
        ("rtx3050-6gb", None, "NVIDIA GeForce RTX 3050"),
        ("rtx3080-laptop-8gb", None, "NVIDIA GeForce RTX 3080 Laptop GPU"),
        ("rtx2060-12gb", None, "NVIDIA GeForce RTX 2060"),
        ("gtx1650-gddr6", None, "NVIDIA GeForce GTX 1650"),
    ],
)
def test_device_names_follow_the_driver_supported_chips_list(
    profile_id: str, device_id: int | None, driver_name: str
) -> None:
    profile = get_profile(profile_id)
    if device_id is not None:
        assert profile.pci_device_id >> 16 == device_id
    assert profile.torch_name == driver_name


def test_jetson_thor_follows_the_developer_kit_device_query() -> None:
    # deviceQuery on the Jetson AGX Thor Developer Kit (CUDA 13.0), NVIDIA's
    # setup guide: Device 0 "NVIDIA Thor", L2 Cache Size 33554432 bytes,
    # 131881811968 bytes of global memory.
    profile = get_profile("jetson-t5000")
    assert profile.torch_name == "NVIDIA Thor"
    assert profile.l2_cache_bytes == 33554432
    assert profile.device_memory_bytes == 131881811968


@pytest.mark.parametrize(
    ("device_name", "profile_id"),
    [
        # Base products, not the laptop or memory variant that shares the name.
        ("NVIDIA GeForce GTX 1080", "gtx1080"),
        ("Quadro P4000", "quadro-p4000"),
        ("NVIDIA GeForce RTX 4070", "rtx4070"),
        ("NVIDIA L20", "l20"),
        ("NVIDIA GeForce RTX 3060", "rtx3060"),
        ("nvidia geforce rtx 3060", "rtx3060"),
        ("NVIDIA GeForce RTX 2060", "rtx2060"),
        # The driver spells the notebook part "Super" and the desktop card "SUPER".
        ("NVIDIA GeForce RTX 2070 Super", "rtx2070-super-laptop"),
        ("NVIDIA GeForce RTX 2070 SUPER", "rtx2070-super"),
        ("NVIDIA H100 80GB HBM3", "h100"),
        ("NVIDIA Unknown GPU", None),
    ],
)
def test_device_name_lookup_prefers_the_canonical_profile(device_name: str, profile_id: str | None) -> None:
    assert profile_id_for_device_name(device_name) == profile_id


def test_int4_is_listed_only_where_tensor_cores_accept_it() -> None:
    # CUDA Programming Guide, "Input Data Types Supported by Tensor Core
    # Acceleration": INT4 on compute capability 7.5-8.9, none on 9.0 and newer.
    catalog = load_profiles()
    for profile in catalog.values():
        if profile.compute_capability >= (9, 0):
            assert "int4" not in profile.supported_types, profile.id
            assert "int8" in profile.supported_types, profile.id
    h100 = replace(catalog["h100"], supported_types=catalog["h100"].supported_types + ("int4",))
    assert any("int4 is not a Tensor Core input" in error for error in validate_catalog({"h100": h100}).errors)


def test_shared_device_names_need_exactly_one_canonical_profile() -> None:
    catalog = load_profiles()
    assert validate_catalog(catalog).ok
    variant = replace(catalog["rtx3060-8gb"], canonical=True)
    errors = validate_catalog({**catalog, variant.id: variant}).errors
    assert any("exactly one needs canonical" in error and "rtx3060-8gb" in error for error in errors)
    base = replace(catalog["rtx3060"], canonical=False)
    errors = validate_catalog({**catalog, base.id: base}).errors
    assert any("found 0" in error for error in errors)
    unique = replace(catalog["a100"], canonical=True)
    errors = validate_catalog({**catalog, unique.id: unique}).errors
    assert any(error.startswith("a100: canonical is set") for error in errors)


# Resident blocks, resident threads, shared memory per SM and per block (opt-in)
# from the CUDA Programming Guide compute-capability tables: the current guide
# (https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
# for 7.5 and newer, the CUDA 12.4 guide for older devices.
_CUDA_SM_LIMITS = {
    (5, 0): (32, 2048, 64, 48),
    (5, 2): (32, 2048, 96, 48),
    (6, 0): (32, 2048, 64, 48),
    (6, 1): (32, 2048, 96, 48),
    (7, 0): (32, 2048, 96, 96),
    (7, 5): (16, 1024, 64, 64),
    (8, 0): (32, 2048, 164, 163),
    (8, 6): (16, 1536, 100, 99),
    (8, 7): (16, 1536, 164, 163),
    (8, 9): (24, 1536, 100, 99),
    (9, 0): (32, 2048, 228, 227),
    (10, 0): (32, 2048, 228, 227),
    (10, 3): (32, 2048, 228, 227),
    (11, 0): (24, 1536, 228, 227),
    (12, 0): (24, 1536, 100, 99),
    (12, 1): (24, 1536, 100, 99),
}


def test_per_sm_limits_follow_the_cuda_programming_guide() -> None:
    for profile in load_profiles().values():
        blocks, threads, shared_kib, optin_kib = _CUDA_SM_LIMITS[profile.compute_capability]
        assert (
            profile.max_blocks_per_multiprocessor,
            profile.max_threads_per_multiprocessor,
            profile.shared_mem_per_sm,
            profile.shared_mem_per_block_optin,
            profile.regs_per_multiprocessor,
        ) == (blocks, threads, shared_kib * 1024, optin_kib * 1024, 65536), profile.id


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
