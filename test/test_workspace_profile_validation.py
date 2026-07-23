from __future__ import annotations

from verification.workspace_profile_validation import (
    CAPTURE_SCHEMA_VERSION,
    CATALOG_SCHEMA_VERSION,
    build_catalog,
    render_summary_markdown,
    summarize_reports,
)


def _report(profile: str, workspace_bytes: int) -> dict:
    return {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "profile_id": profile,
        "gpu_name": profile,
        "compute_capability": "8.6" if profile == "rtx3090ti" else "12.0",
        "torch_version": "2.12.1+cu130",
        "torch_cuda_version": "13.0",
        "cudnn_version": 91000,
        "warmups": 3,
        "measurements": [
            {
                "name": "mm-fp32-1024",
                "operator": "aten::mm",
                "input_dtypes": ["torch.float32", "torch.float32"],
                "input_shapes": [[1024, 1024], [1024, 1024]],
                "allocated_workspace_peak_bytes": workspace_bytes,
                "reserved_workspace_peak_bytes": workspace_bytes,
            }
        ],
    }


def test_workspace_capture_builds_stack_specific_catalog() -> None:
    catalog = build_catalog(
        [
            _report("rtx3090ti", 0),
            _report("rtx-pro-5000-blackwell", 4096),
        ]
    )
    assert catalog["schema_version"] == CATALOG_SCHEMA_VERSION
    assert len(catalog["profiles"]) == 2
    first = catalog["profiles"][0]
    assert first["confidence"] == "measured_physical_gpu_exact_stack_shape"
    assert first["match"]["profile_ids"]
    assert first["match"]["torch_versions"] == ["2.12.1+cu130"]


def test_workspace_capture_summary_reports_cross_stack_range() -> None:
    summary = summarize_reports(
        [
            _report("rtx3090ti", 0),
            _report("rtx-pro-5000-blackwell", 4096),
        ]
    )
    workload = summary["workloads"][0]
    assert workload["allocated_workspace_range_bytes"] == 4096
    assert workload["portable_across_observed_stacks"] is False
    assert "Backend Workspace Profile Validation" in render_summary_markdown(summary)
