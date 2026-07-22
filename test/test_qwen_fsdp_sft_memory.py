from __future__ import annotations

from verification.run_qwen_fsdp_sft_memory import (
    compare_fsdp_reports,
    render_markdown,
)


def _static_report() -> dict:
    return {
        "status": "success",
        "mode": "static",
        "model_dir": "/models/Qwen3.5-0.8B",
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "training_method": "full",
        "gradient_checkpointing": False,
        "gradient_accumulation_steps": 1,
        "batch_size": 1,
        "sequence_length": 16,
        "data_seed": 7,
        "batch": {"fingerprint_sha256": "same-batch"},
        "static_estimate": {
            "parameter_bytes": 1_000,
            "trainable_parameter_bytes": 1_000,
            "frozen_parameter_bytes": 0,
            "first_step_graph_phase_peak_bytes": 2_600,
            "optimizer_phase_peak_bytes": 4_500,
            "optimizer_state_bytes": 2_000,
            "optimizer_temporary_bytes": 500,
            "optimizer_temporary": {
                "current_parameter_temporary_count": 2,
                "retained_previous_temporary_count": 0,
            },
        },
    }


def _rank_report(rank: int, *, overall: int = 2_500) -> dict:
    plan = {
        "world_size": 2,
        "unit_count": 2,
        "units": [],
        "unsharded_parameter_bytes": 1_000,
        "local_shard_parameter_bytes": 500,
        "padding_bytes": 0,
        "largest_unsharded_unit_bytes": 500,
        "largest_local_shard_bytes": 250,
    }
    return {
        "status": "success",
        "rank": rank,
        "world_size": 2,
        "model_dir": "/models/Qwen3.5-0.8B",
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "training_method": "full",
        "gradient_checkpointing": False,
        "gradient_accumulation_steps": 1,
        "batch_size": 1,
        "sequence_length": 16,
        "data_seed": 7,
        "batch": {"fingerprint_sha256": "same-batch"},
        "gpu_name": "Test GPU",
        "compute_capability": [9, 0],
        "torch_version": "2.9.0",
        "torch_cuda_version": "12.8",
        "loss": 10.0,
        "parameters": {
            "logical": {"parameter_bytes": 1_000},
            "local_shard": {"parameter_bytes": 500},
            "local_gradient_bytes": 500,
            "optimizer_state_tensor_bytes": 1_000,
        },
        "fsdp": {"sharding_plan": plan},
        "memory_phases": {
            "graph_peak_bytes": 2_100,
            "optimizer_peak_bytes": 2_500,
            "overall_peak_bytes": overall,
        },
        "phase_seconds": {
            "model_load_and_wrap": 1.0,
            "forward": 1.0,
            "backward": 1.0,
            "optimizer": 1.0,
        },
    }


def _cluster_report() -> dict:
    return {
        "collectives": {
            "all_gather": {"calls": 2, "bytes": 2_000},
            "reduce_scatter": {"calls": 1, "bytes": 1_000},
        },
        "node_pairs": [],
        "operation_timeline": {},
    }


def test_fsdp_comparison_accepts_matching_phase_predictions() -> None:
    report = compare_fsdp_reports(
        [_rank_report(0), _rank_report(1)],
        _static_report(),
        _cluster_report(),
        max_error_percent=1.0,
    )

    assert report["status"] == "success"
    assert report["projection"]["first_step_peak_bytes"] == 2_500
    assert report["ranks"][0]["comparisons"]["overall"][
        "absolute_error_percent"
    ] == 0.0
    assert "Qwen Hybrid FSDP SFT Memory Validation" in render_markdown(report)


def test_fsdp_comparison_rejects_large_phase_error() -> None:
    report = compare_fsdp_reports(
        [_rank_report(0, overall=3_000), _rank_report(1, overall=3_000)],
        _static_report(),
        _cluster_report(),
        max_error_percent=5.0,
    )

    assert report["status"] == "failed"
    assert report["ranks"][0]["status"] == "failed"
