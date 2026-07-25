from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fakegpu.workspace_cli import main as workspace_main
from fakegpu.workspace_profiles import (
    SCHEMA_VERSION,
    WorkspaceProfileError,
    load_workspace_profiles,
    match_workspace_profile,
)


def _write_catalog(path: Path, profiles: list[dict]) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "profiles": profiles,
            }
        ),
        encoding="utf-8",
    )
    return path


def _node(torch):
    left = torch.empty((2, 4), dtype=torch.float32)
    right = torch.empty((4, 8), dtype=torch.float32)
    output = torch.empty((2, 8), dtype=torch.float32)
    return SimpleNamespace(
        name="mm",
        args=(
            SimpleNamespace(meta={"val": left}),
            SimpleNamespace(meta={"val": right}),
        ),
        kwargs={},
        meta={"val": output},
    )


def test_workspace_profile_matches_stack_dtype_shape_and_formula(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    catalog = _write_catalog(
        tmp_path / "profiles.json",
        [
            {
                "id": "mm-linear",
                "operator": "aten.mm.default",
                "lifetime": "operator_local",
                "priority": 10,
                "confidence": "measured_test",
                "match": {
                    "device_types": ["cuda"],
                    "profile_ids": ["rtx3090ti"],
                    "architectures": ["ampere"],
                    "compute_capabilities": ["8.6"],
                    "torch_versions": ["*"],
                    "cuda_versions": ["*"],
                    "input_dtypes": ["torch.float32", "torch.float32"],
                    "input_shapes": [[2, 4], [4, {"min": 8, "max": 16}]],
                },
                "formula": {
                    "kind": "linear_io",
                    "fixed_bytes": 100,
                    "input_bytes_multiplier": 1,
                    "output_bytes_multiplier": 2,
                    "alignment_bytes": 256,
                },
                "bounds": {
                    "lower_multiplier": 0.5,
                    "upper_multiplier": 1.5,
                },
            }
        ],
    )

    profile = match_workspace_profile(
        _node(torch),
        "aten.mm.default",
        target_device=torch.device("cuda"),
        target_profile="rtx3090ti",
        profile_paths=[catalog],
    )
    assert profile is not None
    assert profile["profile"] == "mm-linear"
    assert profile["bytes"] == 512
    assert profile["signature"]["compute_capability"] == "8.6"
    assert profile["calculation"]["input_bytes"] == 160
    assert profile["calculation"]["output_bytes"] == 64
    assert profile["lower_bytes"] == 256
    assert profile["upper_bytes"] == 768
    assert profile["declared_interval"] is True


def test_highest_priority_profile_shadows_generic_match(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    catalog = _write_catalog(
        tmp_path / "profiles.json",
        [
            {
                "id": "generic",
                "operator": "aten.mm.default",
                "lifetime": "operator_local",
                "priority": 0,
                "match": {"device_types": ["cpu"]},
                "bytes": 100,
            },
            {
                "id": "specific",
                "operator": "aten.mm.default",
                "lifetime": "operator_local",
                "priority": 20,
                "match": {
                    "device_types": ["cpu"],
                    "input_shapes": [[2, 4], [4, 8]],
                },
                "bytes": 200,
            },
        ],
    )
    profile = match_workspace_profile(
        _node(torch),
        "aten.mm.default",
        target_device=torch.device("cpu"),
        profile_paths=[catalog],
    )
    assert profile is not None
    assert profile["profile"] == "specific"
    assert profile["bytes"] == 200
    assert profile["shadowed_profiles"] == ["generic"]


def test_workspace_profile_catalog_rejects_invalid_formula(tmp_path: Path) -> None:
    catalog = _write_catalog(
        tmp_path / "invalid.json",
        [
            {
                "id": "invalid",
                "operator": "aten.mm.default",
                "lifetime": "operator_local",
                "match": {},
                "formula": {"kind": "unknown"},
            }
        ],
    )
    with pytest.raises(WorkspaceProfileError, match="formula.kind"):
        load_workspace_profiles([catalog])


def test_workspace_profile_cli_reports_validated_catalog(
    tmp_path: Path,
    capsys,
) -> None:
    catalog = _write_catalog(
        tmp_path / "profiles.json",
        [
            {
                "id": "fixed",
                "operator_regex": "aten[.]mm",
                "lifetime": "operator_local",
                "match": {},
                "bytes": 4096,
            }
        ],
    )
    assert workspace_main(["--path", str(catalog), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile_count"] == len(payload["profile_ids"])
    assert "fixed" in payload["profile_ids"]


def test_static_estimator_applies_external_workspace_profile(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    from fakegpu.memory_estimator import estimate_module_memory

    catalog = _write_catalog(
        tmp_path / "profiles.json",
        [
            {
                "id": "cpu-mm-test",
                "operator": "aten::mm(Tensor self, Tensor mat2) -> Tensor",
                "lifetime": "operator_local",
                "match": {
                    "device_types": ["cpu"],
                    "input_dtypes": ["torch.float32", "torch.float32"],
                    "input_shapes": [[2, 4], [4, 8]],
                },
                "bytes": 4096,
            }
        ],
    )

    class Matmul(torch.nn.Module):
        def forward(self, left, right):
            return torch.mm(left, right)

    report = estimate_module_memory(
        Matmul(),
        (torch.randn(2, 4), torch.randn(4, 8)),
        mode="forward",
        optimizer="none",
        target_device="cpu",
        workspace_profile_paths=[catalog],
    )
    assert report["workspace_estimate"]["profiled_operator_count"] == 1
    assert report["workspace_estimate"]["profiles"][0]["profile"] == "cpu-mm-test"
    assert report["workspace_peak_contribution_bytes"] == 4096
    assert report["workspace_estimate"]["unprofiled_workspace_candidates"] == {}
    coverage = report["workspace_estimate"]["coverage"]
    assert coverage["status"] == "complete"
    assert coverage["candidate_operator_count"] == 1
    assert coverage["modeled_fraction"] == 1.0

    from fakegpu import require_workspace_coverage

    evaluation = require_workspace_coverage(
        report,
        minimum_fraction=1.0,
        allow_extrapolated=False,
    )
    assert evaluation["passed"] is True
    assert evaluation["selected_fraction"] == 1.0


def test_workspace_coverage_gate_rejects_unprofiled_operator() -> None:
    torch = pytest.importorskip("torch")
    from fakegpu import WorkspaceCoverageError, require_workspace_coverage
    from fakegpu.memory_estimator import estimate_module_memory

    class Matmul(torch.nn.Module):
        def forward(self, left, right):
            return torch.mm(left, right)

    report = estimate_module_memory(
        Matmul(),
        (torch.randn(2, 4), torch.randn(4, 8)),
        mode="forward",
        optimizer="none",
        target_device="cpu",
    )
    coverage = report["workspace_estimate"]["coverage"]
    assert coverage["status"] == "incomplete"
    assert coverage["candidate_operator_count"] == 1
    assert coverage["unprofiled_operator_count"] == 1
    assert coverage["modeled_fraction"] == 0.0
    assert coverage["all_candidate_calls_modeled"] is False
    assert coverage["upper_bound_complete"] is False
    assert report["estimated_peak_interval_bytes"]["upper"] is None

    with pytest.raises(WorkspaceCoverageError, match="below the required") as exc:
        require_workspace_coverage(report, minimum_fraction=1.0)
    assert exc.value.evaluation["passed"] is False
    assert exc.value.evaluation["unprofiled_operator_count"] == 1

    with pytest.raises(
        WorkspaceCoverageError, match="upper bound is incomplete"
    ):
        require_workspace_coverage(
            report,
            minimum_fraction=0.0,
            require_upper_bound=True,
        )


def test_explicit_unknown_workspace_upper_bound_produces_memory_interval() -> None:
    torch = pytest.importorskip("torch")
    from fakegpu import require_workspace_coverage
    from fakegpu.memory_estimator import estimate_module_memory

    class Matmul(torch.nn.Module):
        def forward(self, left, right):
            return torch.mm(left, right)

    report = estimate_module_memory(
        Matmul(),
        (torch.randn(2, 4), torch.randn(4, 8)),
        mode="forward",
        optimizer="none",
        target_device="cpu",
        unknown_workspace_upper_bound_bytes=8192,
    )
    interval = report["estimated_peak_interval_bytes"]
    assert interval["lower"] == interval["expected"]
    assert interval["upper"] == interval["expected"] + 8192
    assert interval["upper_bound_complete"] is True
    call = report["workspace_estimate"]["unprofiled_workspace_calls"][0]
    assert call["upper_bytes"] == 8192
    assert call["input_bytes"] == 160
    assert call["output_bytes"] == 64
    evaluation = require_workspace_coverage(
        report,
        minimum_fraction=0.0,
        require_upper_bound=True,
    )
    assert evaluation["upper_bound_passed"] is True


@pytest.mark.parametrize("value", [True, 1.5, "1024"])
def test_unknown_workspace_upper_bound_requires_integer(
    value: object,
) -> None:
    torch = pytest.importorskip("torch")
    from fakegpu.memory_estimator import estimate_module_memory

    with pytest.raises(
        ValueError,
        match="unknown_workspace_upper_bound_bytes",
    ):
        estimate_module_memory(
            torch.nn.Linear(4, 4),
            (torch.ones(1, 4),),
            mode="forward",
            optimizer="none",
            unknown_workspace_upper_bound_bytes=value,
        )


def test_workspace_coverage_without_candidates_is_complete() -> None:
    torch = pytest.importorskip("torch")
    from fakegpu import require_workspace_coverage
    from fakegpu.memory_estimator import estimate_module_memory

    report = estimate_module_memory(
        torch.nn.ReLU(),
        (torch.randn(2, 4),),
        mode="forward",
        optimizer="none",
        target_device="cpu",
    )
    coverage = report["workspace_estimate"]["coverage"]
    assert coverage["status"] == "no_workspace_candidates"
    assert coverage["candidate_operator_count"] == 0
    assert coverage["modeled_fraction"] == 1.0
    assert require_workspace_coverage(report)["passed"] is True
