from __future__ import annotations

from verification.qwen_sft_memory_worker import _checkpoint_graph_estimate


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
