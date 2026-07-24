from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parent.parent


def test_graph_liveness_deduplicates_view_storage() -> None:
    from torch.fx.experimental.proxy_tensor import make_fx

    from fakegpu.memory_estimator import analyze_graph_memory

    def view_and_clone(value):
        flattened = value.view(-1)
        prefix = flattened[:4]
        return prefix, value.clone()

    graph_module = make_fx(view_and_clone, tracing_mode="fake")(torch.randn(2, 4))
    report = analyze_graph_memory(
        graph_module,
        placeholder_categories=["input"],
        retain_placeholder_categories={"input"},
    )

    assert report["unique_storage_count"] == 2
    assert report["alias_node_count"] >= 2
    assert report["total_unique_storage_bytes"] == 64
    assert report["peak_live_bytes"] == 64
    assert report["live_bytes_by_node"][report["peak_node"]["name"]] == 64
    assert report["retained_placeholder_storage_count"] == 1
    assert report["warnings"] == []


def test_forward_estimate_uses_fx_storage_liveness() -> None:
    from fakegpu.memory_estimator import estimate_module_memory

    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.GELU(),
        torch.nn.Linear(8, 2),
    )
    inputs = torch.randn(3, 4)

    report = estimate_module_memory(
        model,
        (inputs,),
        mode="forward",
        optimizer="adamw",
    )

    expected_parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    assert report["schema_version"] == "static_memory_estimate.v1"
    assert report["method"] == "fx_aten_storage_liveness"
    assert report["tracking_confidence"] == "S1_fx_forward_liveness"
    expected_trace_device = "cuda:0" if torch._C._has_cuda else "cpu"
    assert report["trace_device"] == expected_trace_device
    assert report["parameter_bytes"] == expected_parameter_bytes
    assert report["input_bytes"] == inputs.numel() * inputs.element_size()
    assert report["optimizer_state_bytes"] == 0
    assert report["estimated_peak_bytes"] == report["graph_peak_live_bytes"]
    assert report["graph"]["operator_count"] > 0
    assert len(report["graph"]["graph_fingerprint"]) == 64
    json.dumps(report)


def test_training_estimate_captures_backward_and_adamw_state() -> None:
    from fakegpu.memory_estimator import estimate_module_memory

    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.GELU(),
        torch.nn.Linear(8, 2),
    )
    inputs = torch.randn(3, 4)
    report = estimate_module_memory(
        model,
        (inputs,),
        mode="training",
        optimizer="adamw",
    )

    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    expected_optimizer_state = 2 * parameter_bytes
    parameter_storage_sizes = [
        parameter.untyped_storage().nbytes() for parameter in model.parameters()
    ]
    expected_optimizer_temporary = max(
        2 * current + previous
        for previous, current in zip(
            [0, *parameter_storage_sizes[:-1]],
            parameter_storage_sizes,
        )
    )

    assert report["method"] == "aot_autograd_aten_storage_liveness"
    assert report["tracking_confidence"] == "S2_aot_training_liveness"
    assert report["trainable_parameter_bytes"] == parameter_bytes
    assert report["optimizer_state_bytes"] == expected_optimizer_state
    assert report["optimizer_temporary_bytes"] == expected_optimizer_temporary
    assert report["optimizer_temporary"]["assumption"] == "single_tensor_eager"
    assert report["optimizer_state"]["state_tensors_per_parameter"] == 2
    assert report["retain_forward_outputs"] is True
    assert report["optimizer_temporary"]["current_parameter_temporary_count"] == 2
    assert report["estimated_peak_bytes"] == max(
        report["graph_peak_live_bytes"] + expected_optimizer_state,
        report["post_graph_live_bytes"]
        + expected_optimizer_state
        + expected_optimizer_temporary,
    )
    assert report["graph_phase_peak_bytes"] == (
        report["graph_peak_live_bytes"] + expected_optimizer_state
    )
    assert report["first_step_graph_phase_peak_bytes"] == report["graph_peak_live_bytes"]
    assert report["optimizer_phase_peak_bytes"] == (
        report["post_graph_live_bytes"]
        + expected_optimizer_state
        + expected_optimizer_temporary
    )
    assert report["first_step_estimated_peak_bytes"] == max(
        report["graph_peak_live_bytes"],
        report["optimizer_phase_peak_bytes"],
    )
    assert report["steady_state_estimated_peak_bytes"] == report["estimated_peak_bytes"]
    assert report["graph"]["peak_bytes_by_category"]["trainable_parameter"] == parameter_bytes
    gradient_phase = report["graph"]["gradient_production_phase"]
    assert gradient_phase is not None
    assert gradient_phase["peak_live_bytes"] >= parameter_bytes
    assert gradient_phase["peak_bytes_by_category"]["gradient"] > 0
    forward_phase = report["graph"]["forward_phase"]
    assert forward_phase is not None
    assert forward_phase["method"] == "explicit_backward_operator_boundary"
    assert forward_phase["peak_node"]["index"] < forward_phase[
        "backward_start_node"
    ]["index"]
    assert report["graph"]["alias_node_count"] > 0
    assert "optimizer_fused_or_foreach_extra_temporaries" in report["unmodeled_components"]


def test_training_estimate_excludes_frozen_parameters_from_gradients_and_optimizer() -> None:
    from fakegpu.memory_estimator import estimate_module_memory

    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.GELU(),
        torch.nn.Linear(8, 2),
    )
    model[0].weight.requires_grad_(False)
    model[0].bias.requires_grad_(False)
    report = estimate_module_memory(
        model,
        (torch.randn(3, 4),),
        mode="training",
        optimizer="sgd_momentum",
    )

    trainable_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    frozen_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
        if not parameter.requires_grad
    )
    assert report["trainable_parameter_bytes"] == trainable_bytes
    assert report["frozen_parameter_bytes"] == frozen_bytes
    assert report["optimizer_state_bytes"] == trainable_bytes
    assert report["optimizer_temporary_bytes"] == 0


def test_single_tensor_adamw_temporary_tracks_parameter_iteration_order() -> None:
    from fakegpu.memory_estimator import estimate_module_memory

    model = torch.nn.ModuleList(
        [
            torch.nn.Linear(16, 64, bias=False),
            torch.nn.Linear(4, 4, bias=False),
            torch.nn.Linear(16, 64, bias=False),
            torch.nn.Linear(4, 4, bias=False),
            torch.nn.Linear(16, 64, bias=False),
        ]
    )

    class Wrapped(torch.nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layers = layers

        def forward(self, value):
            return sum(
                layer(value if layer.in_features == 16 else value[:, :4]).mean()
                for layer in self.layers
            )

    wrapped = Wrapped(model)
    report = estimate_module_memory(
        wrapped,
        (torch.randn(2, 16),),
        mode="training",
        optimizer="adamw",
    )
    storage_sizes = [
        parameter.untyped_storage().nbytes() for parameter in wrapped.parameters()
    ]
    expected_temporary = max(
        2 * current + previous
        for previous, current in zip([0, *storage_sizes[:-1]], storage_sizes)
    )
    assert sum(storage_sizes) > expected_temporary
    assert report["optimizer_temporary_bytes"] == expected_temporary
    assert report["optimizer_temporary"]["largest_parameter_tensor_bytes"] == max(
        storage_sizes
    )


def test_optimizer_state_is_per_parameter_when_parameters_share_storage() -> None:
    from fakegpu.memory_estimator import estimate_module_memory

    class SharedParameters(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            shared = torch.randn(4)
            self.left = torch.nn.Parameter(shared)
            self.right = torch.nn.Parameter(shared)

        def forward(self, value):
            return value * self.left + value * self.right

    model = SharedParameters()
    report = estimate_module_memory(
        model,
        (torch.randn(2, 4),),
        mode="training",
        optimizer="adamw",
    )
    logical_parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    shared_storage_bytes = model.left.untyped_storage().nbytes()

    assert report["trainable_parameter_bytes"] == shared_storage_bytes
    assert report["optimizer_state"]["parameter_tensor_count"] == 2
    assert report["optimizer_state"]["parameter_storage_count"] == 1
    assert report["optimizer_state_bytes"] == 2 * logical_parameter_bytes
    assert report["optimizer_temporary_bytes"] == 3 * (
        model.left.numel() * model.left.element_size()
    )


def test_fake_trace_preserves_aliases_across_input_dtypes() -> None:
    from fakegpu.memory_estimator import estimate_module_memory

    class MixedDtypeAliases(torch.nn.Module):
        def forward(self, byte_values, float_values):
            return byte_values.float().sum() + float_values.sum()

    byte_values = torch.arange(16, dtype=torch.uint8)
    float_values = byte_values.view(torch.float32)
    report = estimate_module_memory(
        MixedDtypeAliases(),
        (byte_values, float_values),
        mode="forward",
        target_device="cpu",
    )

    assert report["input_bytes"] == 16
    assert report["graph"]["retained_placeholder_storage_count"] == 1
    assert report["graph"]["peak_bytes_by_category"]["input"] == 16


def test_flash_attention_workspace_profile_uses_sequence_tiles() -> None:
    from fakegpu.memory_estimator import _flash_attention_workspace_bytes

    query_64 = torch.empty((2, 4, 64, 32), dtype=torch.bfloat16)
    query_128 = torch.empty((2, 4, 128, 32), dtype=torch.bfloat16)

    bytes_64, sequence_64, tiles_64 = _flash_attention_workspace_bytes(
        query_64,
        sequence_dimension=2,
    )
    bytes_128, sequence_128, tiles_128 = _flash_attention_workspace_bytes(
        query_128,
        sequence_dimension=2,
    )

    assert (sequence_64, tiles_64) == (64, 1)
    assert (sequence_128, tiles_128) == (128, 2)
    assert bytes_64 == 2 * query_64.numel() * query_64.element_size()
    assert bytes_128 == 4 * query_128.numel() * query_128.element_size()


@pytest.mark.parametrize(
    ("shape", "expected_bytes"),
    [
        ((1, 4, 16, 16), 65_856),
        ((2, 4, 16, 16), 139_904),
        ((1, 4, 32, 32), 66_112),
        ((2, 4, 32, 16), 148_608),
        ((4, 4, 32, 16), 297_216),
        ((2, 4, 64, 16), 166_016),
        ((2, 4, 64, 32), 198_784),
        ((3, 4, 33, 24), 236_400),
    ],
)
def test_efficient_attention_workspace_profile_matches_calibration_grid(
    shape: tuple[int, ...],
    expected_bytes: int,
) -> None:
    from fakegpu.memory_estimator import (
        _efficient_attention_backward_workspace_bytes,
    )

    query = torch.empty(shape, dtype=torch.float32)
    (
        total_bytes,
        batch_size,
        sequence_length,
        fixed_bytes,
        query_scratch_bytes,
        row_metadata_bytes,
    ) = _efficient_attention_backward_workspace_bytes(query)

    assert total_bytes == expected_bytes
    assert fixed_bytes == batch_size * 65_536
    assert query_scratch_bytes == (query.numel() * query.element_size()) * (
        batch_size > 1
    )
    assert row_metadata_bytes == batch_size * (16 * sequence_length + 64)
    assert total_bytes == fixed_bytes + query_scratch_bytes + row_metadata_bytes


def test_operator_local_workspace_uses_node_liveness() -> None:
    from fakegpu.memory_estimator import _workspace_peak_summary

    graph = {
        "peak_live_bytes": 100,
        "peak_node": {"name": "global_peak"},
        "live_bytes_by_node": {
            "global_peak": 100,
            "efficient_attention_backward": 70,
        },
    }
    profiles = [
        {
            "node": "persistent_source",
            "lifetime": "graph_phase_persistent",
            "bytes": 20,
        },
        {
            "node": "efficient_attention_backward",
            "lifetime": "operator_local",
            "bytes": 40,
        },
    ]

    summary = _workspace_peak_summary(graph, profiles)

    assert summary["profiled_bytes_sum"] == 60
    assert summary["graph_phase_persistent_bytes"] == 20
    assert summary["operator_local_peak_bytes"] == 40
    assert summary["total_bytes"] == 60
    assert summary["effective_peak_contribution_bytes"] == 30
    assert summary["peak_candidate"] == {
        "node": "efficient_attention_backward",
        "graph_live_bytes": 70,
        "graph_phase_persistent_bytes": 20,
        "operator_workspace_bytes": 40,
        "combined_live_bytes": 130,
    }


def test_cuda_workspace_profile_is_not_applied_to_cpu_trace() -> None:
    from types import SimpleNamespace

    from fakegpu.memory_estimator import _backend_workspace_profile

    query = torch.empty((2, 4, 32, 16), dtype=torch.float32)
    node = SimpleNamespace(
        name="efficient_attention_backward",
        args=(None, SimpleNamespace(meta={"val": query})),
    )
    target = "aten::_scaled_dot_product_efficient_attention_backward(Tensor query)"

    assert (
        _backend_workspace_profile(
            node,
            target,
            target_device=torch.device("cpu"),
        )
        is None
    )
    profile = _backend_workspace_profile(
        node,
        target,
        target_device=torch.device("cuda"),
    )
    assert profile is not None
    assert profile["lifetime"] == "operator_local"
    assert profile["confidence"] == "validated_two_gpu_multi_shape"


def test_training_estimate_requires_scalar_loss() -> None:
    from fakegpu.memory_estimator import estimate_module_memory

    model = torch.nn.Linear(4, 2)
    with pytest.raises(ValueError, match="scalar"):
        estimate_module_memory(
            model,
            (torch.randn(3, 4),),
            mode="training",
            loss_fn=lambda output: output,
        )
    with pytest.raises(ValueError, match="scalar"):
        estimate_module_memory(
            model,
            (torch.randn(3, 4),),
            mode="training",
            loss_fn=lambda output: output.mean().reshape(1),
        )


def test_training_estimate_rejects_complex_custom_loss() -> None:
    from fakegpu.memory_estimator import estimate_module_memory

    model = torch.nn.Linear(4, 2)
    with pytest.raises(ValueError, match="real floating-point"):
        estimate_module_memory(
            model,
            (torch.randn(3, 4),),
            mode="training",
            loss_fn=lambda output: output.mean().to(torch.complex64),
        )


def test_default_loss_converts_complex_output_to_real_scalar() -> None:
    from fakegpu.memory_estimator import _default_loss

    loss = _default_loss(torch.tensor([1 + 2j, 2 - 1j]))

    assert loss.dtype == torch.float32
    assert loss.shape == ()
    assert loss.item() == pytest.approx(5.0)


def test_explicit_cuda_trace_requires_cuda_enabled_torch() -> None:
    if torch._C._has_cuda:
        pytest.skip("test applies to CPU-only PyTorch builds")

    from fakegpu.memory_estimator import estimate_module_memory

    with pytest.raises(RuntimeError, match="linked with CUDA"):
        estimate_module_memory(
            torch.nn.Linear(4, 2),
            (torch.randn(3, 4),),
            mode="forward",
            target_device="cuda",
        )


def test_static_validation_cli_writes_checkable_static_only_report(tmp_path: Path) -> None:
    report_path = tmp_path / "static_memory_validation.json"
    completed = subprocess.run(
        [
            sys.executable,
            "verification/static_memory_validation.py",
            "--static-only",
            "--workload",
            "mlp_b8_d256_h512_float32",
            "--output",
            str(report_path),
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "PASS_STATIC_ONLY"
    assert len(report["workloads"]) == 1
    static = report["workloads"][0]["static_estimate"]
    assert static["estimated_peak_bytes"] > 0
    assert static["workspace_estimate"]["profiled_bytes_sum"] == 0
    assert static["workspace_estimate_bytes"] == 0
    assert static["workspace_peak_contribution_bytes"] == 0
    assert static["workspace_estimate"]["peak_candidate"]["combined_live_bytes"] > 0
    assert report["summary"]["extrapolated_workspace_profile_count"] == 0
    coverage = report["summary"]["workspace_coverage"]
    assert coverage["candidate_operator_count"] == 5
    assert coverage["modeled_operator_count"] == 0
    assert coverage["unprofiled_operator_count"] == 5
    assert coverage["modeled_fraction"] == 0.0

    checked = subprocess.run(
        [
            sys.executable,
            "verification/check_static_memory_validation.py",
            "--path",
            str(report_path),
            "--allow-static-only",
            "--min-workspace-coverage",
            "0",
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )
    assert checked.returncode == 0, checked.stderr

    gated_report_path = tmp_path / "static_memory_validation_gated.json"
    gated = subprocess.run(
        [
            sys.executable,
            "verification/static_memory_validation.py",
            "--static-only",
            "--workload",
            "mlp_b8_d256_h512_float32",
            "--min-workspace-coverage",
            "1",
            "--output",
            str(gated_report_path),
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )
    assert gated.returncode == 2, gated.stderr
    gated_report = json.loads(gated_report_path.read_text(encoding="utf-8"))
    assert gated_report["status"] == "FAIL_WORKSPACE_COVERAGE"


def test_static_validation_bundle_separates_peak_and_graph_consistency(tmp_path: Path) -> None:
    from verification.aggregate_static_memory_validations import (
        aggregate_reports,
        render_markdown,
    )

    reports = []
    for index, (gpu_name, fingerprint, backend_bytes, error_percent) in enumerate(
        (
            ("GPU A", "graph-a", 100, 1.0),
            ("GPU B", "graph-b", 120, 2.0),
        )
    ):
        path = tmp_path / f"report-{index}.json"
        reports.append(
            (
                path,
                {
                    "schema_version": "static_memory_validation.v1",
                    "status": "PASS_MEASURED",
                    "gpu": {"name": gpu_name, "compute_capability": str(index)},
                    "software": {"torch_version": f"test-{index}"},
                    "backend_calibration": {
                        "resident_allocated_bytes": backend_bytes,
                        "resident_requested_bytes": backend_bytes,
                    },
                    "sampling": {"measured_runs": 3},
                    "summary": {},
                    "workloads": [
                        {
                            "name": "probe",
                            "family": "test",
                            "parameters": {"batch_size": 2},
                            "static_estimate": {
                                "estimated_peak_bytes": 1000,
                                "workspace_estimate": {
                                    "coverage": {
                                        "modeled_fraction": 1.0 if index == 0 else 0.5,
                                        "non_extrapolated_fraction": (
                                            1.0 if index == 0 else 0.25
                                        ),
                                    }
                                },
                                "graph": {
                                    "graph_fingerprint": fingerprint,
                                    "node_count": 10,
                                    "alias_node_count": 2,
                                },
                            },
                            "real_cuda": {"peak_allocated_bytes": 1100},
                            "comparison": {
                                "calibrated_estimated_peak_bytes": 1100,
                                "underestimate_percent": 0.0,
                                "absolute_error_percent": error_percent,
                            },
                        }
                    ],
                },
            )
        )

    bundle = aggregate_reports(reports)
    assert bundle["schema_version"] == "static_memory_validation_bundle.v1"
    assert bundle["summary"]["gpu_profile_count"] == 2
    assert bundle["summary"]["all_static_peaks_consistent"] is True
    assert bundle["summary"]["all_graph_fingerprints_consistent"] is False
    assert bundle["summary"]["max_absolute_error_percent"] == 2.0
    assert bundle["summary"]["workspace_coverage_observation_count"] == 2
    assert bundle["summary"]["missing_workspace_coverage_observation_count"] == 0
    assert bundle["summary"]["incomplete_workspace_coverage_observation_count"] == 1
    assert bundle["summary"]["minimum_workspace_modeled_fraction"] == 0.5
    assert (
        bundle["summary"]["minimum_workspace_non_extrapolated_fraction"]
        == 0.25
    )
    assert bundle["workloads"][0]["static_peak_consistent"] is True
    assert bundle["workloads"][0]["graph_fingerprint_consistent"] is False
    assert bundle["workloads"][0]["present_in_all_reports"] is True
    markdown = render_markdown(bundle)
    assert "Minimum workspace call coverage" in markdown
    assert "50.0%" in markdown

    reports[1][1]["workloads"][0]["name"] = "different-probe"
    incomplete_bundle = aggregate_reports(reports)
    assert incomplete_bundle["summary"]["all_static_peaks_consistent"] is False
    assert all(
        workload["present_in_all_reports"] is False
        for workload in incomplete_bundle["workloads"]
    )


def test_measured_report_allows_zero_backend_resident_calibration(tmp_path: Path) -> None:
    from verification.check_static_memory_validation import main as check_report

    path = tmp_path / "zero-backend.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "static_memory_validation.v1",
                "status": "PASS_MEASURED",
                "backend_calibration": {
                    "resident_allocated_bytes": 0,
                    "resident_requested_bytes": 0,
                },
                "summary": {
                    "workspace_profile_count": 0,
                    "extrapolated_workspace_profile_count": 0,
                    "graph_modeled_attention_operator_count": 0,
                    "unprofiled_attention_operator_count": 0,
                },
                "workloads": [
                    {
                        "name": "probe",
                        "static_estimate": {
                            "estimated_peak_bytes": 1,
                            "workspace_estimate_bytes": 0,
                            "workspace_estimate": {
                                "total_bytes": 0,
                                "profiled_bytes_sum": 0,
                                "profiled_operator_count": 0,
                                "profiles": [],
                                "graph_modeled_attention_operators": {},
                                "peak_candidate": {
                                    "combined_live_bytes": 1,
                                },
                            },
                            "graph": {
                                "node_count": 1,
                                "operator_histogram": {},
                                "unique_storage_count": 1,
                                "warnings": [],
                            },
                        },
                        "real_cuda": {
                            "peak_allocated_bytes": 1,
                            "peak_stage": "forward",
                            "peak_by_stage": {
                                "forward": {"peak_allocated_bytes": 1},
                                "backward": {"peak_allocated_bytes": 1},
                                "optimizer": {"peak_allocated_bytes": 1},
                            },
                        },
                        "comparison": {
                            "method": "static_graph_plus_backend_resident_calibration",
                            "underestimate_percent": 0.0,
                            "phase_comparison": {
                                "graph": {},
                                "optimizer": {},
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert check_report(["--path", str(path)]) == 0


def test_training_estimator_suspends_and_restores_fakegpu_saved_tensor_hook() -> None:
    code = "\n".join(
        [
            "import fakegpu",
            "fakegpu.patch_torch()",
            "import torch",
            "import fakegpu.torch_patch as torch_patch",
            "from fakegpu.memory_estimator import estimate_module_memory",
            "assert torch_patch._saved_tensors_hooks_cm is not None",
            "model = torch.nn.Linear(4, 2)",
            "report = estimate_module_memory(model, (torch.randn(3, 4),), mode='training')",
            "assert report['estimated_peak_bytes'] > 0",
            "assert torch_patch._saved_tensors_hooks_cm is not None",
        ]
    )
    env = dict(os.environ)
    env["FAKEGPU_TERMINAL_REPORT"] = "0"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
