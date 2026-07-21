from __future__ import annotations

from verification.qwen_sft_memory_worker import (
    _apply_nf4_static_adjustment,
    _checkpoint_graph_estimate,
)


def _estimate(*, trainable_parameter_bytes: int) -> dict:
    return {
        "first_step_graph_phase_peak_bytes": 2_000,
        "parameter_bytes": 1_000,
        "trainable_parameter_bytes": trainable_parameter_bytes,
        "buffer_bytes": 5,
        "input_bytes": 2,
        "workspace_peak_contribution_bytes": 3,
        "post_graph_live_bytes": 1_100,
        "graph": {
            "peak_bytes_by_category": {"output": 100},
            "top_peak_storages": [
                {
                    "bytes": 200,
                    "category": "temporary",
                    "producer_target": "aten::_log_softmax_backward_data",
                },
                {
                    "bytes": 500,
                    "category": "temporary",
                    "producer_target": "aten::mm",
                },
            ],
        },
    }


def test_checkpoint_graph_estimate_removes_checkpointed_lora_activations() -> None:
    report = _checkpoint_graph_estimate(_estimate(trainable_parameter_bytes=10))

    assert report["method"] == "analytical_loss_and_optimizer_floor"
    assert report["estimated_peak_bytes"] == 1_310
    assert report["loss_temporary_bytes"] == 200
    assert report["uncheckpointed_upper_bound_bytes"] == 2_000


def test_checkpoint_graph_estimate_preserves_full_parameter_gradient_floor() -> None:
    report = _checkpoint_graph_estimate(_estimate(trainable_parameter_bytes=600))

    assert report["method"] == "analytical_full_parameter_graph_floor"
    assert report["estimated_peak_bytes"] == 2_000


def test_nf4_static_adjustment_substitutes_storage_and_adds_workspace() -> None:
    estimate = {
        "parameter_bytes": 1_000,
        "trainable_parameter_bytes": 100,
        "frozen_parameter_bytes": 900,
        "buffer_bytes": 20,
        "post_graph_live_bytes": 1_200,
        "first_step_graph_phase_peak_bytes": 2_000,
        "graph_phase_peak_bytes": 2_200,
        "optimizer_phase_peak_bytes": 1_800,
        "first_step_estimated_peak_bytes": 2_000,
        "steady_state_estimated_peak_bytes": 2_200,
        "estimated_peak_bytes": 2_200,
        "peak_phase": "graph",
        "workspace_estimate_bytes": 10,
        "workspace_peak_contribution_bytes": 10,
        "workspace_estimate": {
            "total_bytes": 10,
            "effective_peak_contribution_bytes": 10,
            "profiles": [],
        },
        "tracking_confidence": "S2",
        "unmodeled_components": [],
        "notes": [],
        "graph": {
            "peak_live_bytes": 1_990,
            "final_live_bytes": 1_200,
            "total_unique_storage_bytes": 3_000,
            "peak_bytes_by_category": {"frozen_parameter": 900},
            "final_bytes_by_category": {"frozen_parameter": 900},
        },
    }
    plan = {
        "original_weight_bytes": 800,
        "quantized_storage_bytes": 250,
        "largest_dequantization_workspace_bytes": 100,
        "module_count": 2,
    }

    adjusted = _apply_nf4_static_adjustment(estimate, plan)

    assert adjusted["parameter_bytes"] == 200
    assert adjusted["buffer_bytes"] == 270
    assert adjusted["persistent_model_storage_bytes"] == 470
    assert adjusted["first_step_graph_phase_peak_bytes"] == 1_550
    assert adjusted["optimizer_phase_peak_bytes"] == 1_250
    assert adjusted["workspace_peak_contribution_bytes"] == 110
    assert estimate["parameter_bytes"] == 1_000
