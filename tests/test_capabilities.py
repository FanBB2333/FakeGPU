from __future__ import annotations

import json
from pathlib import Path

import pytest

from fakegpu.capabilities import (
    _load_native_capabilities_cached,
    audit_native_capability_sources,
    load_native_capabilities,
    main,
    native_capability_report,
)


ROOT = Path(__file__).resolve().parents[1]


def test_capability_catalog_classifies_high_risk_native_apis() -> None:
    catalog = load_native_capabilities()
    entries = {item["api"]: item for item in catalog["apis"]}
    for api in ("cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel", "cuLaunchKernelEx"):
        assert entries[api]["classification"] == "cpu_ptx_execution"
        assert entries[api]["policy_enforced"] is False
    assert entries["cudaStreamAddCallback"]["classification"] == (
        "callback_not_registered"
    )
    assert entries["nvmlDeviceGetAccountingMode"]["classification"] == (
        "synthetic_telemetry"
    )


def test_capability_catalog_cache_normalizes_path_argument_types() -> None:
    catalog_path = ROOT / "fakegpu" / "data" / "native_api_capabilities.json"
    _load_native_capabilities_cached.cache_clear()
    try:
        load_native_capabilities(str(catalog_path))
        load_native_capabilities(catalog_path)
        info = _load_native_capabilities_cached.cache_info()
        assert info.misses == 1
        assert info.hits == 1
    finally:
        _load_native_capabilities_cached.cache_clear()


@pytest.mark.parametrize("api", [
    "cublasStrsm_v2", "cublasDsymm_v2", "cublasCdotc_v2", "cublasZdotc_v2",
    "cublasSgetrfBatched", "cublasCgetrsBatched", "cublasZgeqrfBatched", "cublasDgelsBatched",
])
def test_cublas_synthetic_algorithms_are_explicitly_classified(api: str) -> None:
    report = native_capability_report(library="cublas", classification="not_executed")
    entries = {entry["api"]: entry for entry in report["apis"]}
    assert entries[api]["policy_enforced"] is True


@pytest.mark.parametrize("api", [
    "cublasSdot_v2", "cublasHgemm", "cublasSgemm_v2", "cublasGemmEx",
    "cublasGemmStridedBatchedEx", "cublasGemmBatchedEx", "cublasLtMatmul",
])
def test_cublas_numeric_paths_only_apply_policy_to_fallbacks(api: str) -> None:
    report = native_capability_report(library="cublas")
    entries = {entry["api"]: entry for entry in report["apis"]}
    assert entries[api]["classification"] == "mixed_numeric_and_compatibility"
    assert entries[api]["policy_enforced"] is True


def test_native_source_stubs_and_policy_calls_are_classified() -> None:
    audit = audit_native_capability_sources(ROOT)
    assert audit["status"] == "passed", audit
    assert audit["policy_api_count"] >= 20
    assert audit["stub_api_count"] >= 15
    assert audit["unclassified_policy_apis"] == []
    assert audit["unclassified_stub_apis"] == []
    assert audit["declared_policy_apis_not_found"] == []


def test_capability_cli_filters_and_writes_json(capsys) -> None:
    assert (
        main(
            [
                "--source-root",
                str(ROOT),
                "--library",
                "cuda_driver",
                "--classification",
                "not_executed",
                "--strict",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["source_audit"]["status"] == "passed"
    assert {item["api"] for item in payload["apis"]} == {
        "cuGraphLaunch", "cuDevicePrimaryCtxReset",
    }


def test_capability_report_summarizes_catalog() -> None:
    report = native_capability_report()
    assert report["summary"]["group_count"] == 5
    assert report["summary"]["explicit_api_count"] >= 25
    assert report["summary"]["policy_enforced_api_count"] >= 20
