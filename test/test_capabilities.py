from __future__ import annotations

import json
from pathlib import Path

from fakegpu.capabilities import (
    audit_native_capability_sources,
    load_native_capabilities,
    main,
    native_capability_report,
)


ROOT = Path(__file__).resolve().parents[1]


def test_capability_catalog_classifies_high_risk_native_apis() -> None:
    catalog = load_native_capabilities()
    entries = {item["api"]: item for item in catalog["apis"]}
    assert entries["cudaLaunchKernel"]["classification"] == "not_executed"
    assert entries["cudaLaunchKernel"]["policy_enforced"] is True
    assert entries["cudaStreamAddCallback"]["classification"] == (
        "callback_not_registered"
    )
    assert entries["nvmlDeviceGetAccountingMode"]["classification"] == (
        "synthetic_telemetry"
    )


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
        "cuGraphLaunch",
        "cuLaunchKernel",
        "cuLaunchKernelEx",
    }


def test_capability_report_summarizes_catalog() -> None:
    report = native_capability_report()
    assert report["summary"]["group_count"] == 5
    assert report["summary"]["explicit_api_count"] >= 25
    assert report["summary"]["policy_enforced_api_count"] >= 20
