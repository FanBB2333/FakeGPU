from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from fakegpu.fsdp_memory import build_fully_shard_plan
from verification.run_qwen_fsdp2_lora_sft_memory import (
    compare_fsdp2_lora_reports,
    render_markdown,
)


ROOT = Path(__file__).resolve().parents[1]


def _plan() -> dict:
    return build_fully_shard_plan(
        [
            {
                "name": "root",
                "is_root": True,
                "parameters": [
                    {
                        "name": "root.weight",
                        "parameter_index": 0,
                        "shape": [4, 2],
                        "element_size": 2,
                        "trainable": False,
                    }
                ],
            },
            {
                "name": "layer",
                "parameters": [
                    {
                        "name": "layer.weight",
                        "parameter_index": 1,
                        "shape": [4, 2],
                        "element_size": 2,
                        "trainable": False,
                    },
                    {
                        "name": "layer.lora_a",
                        "parameter_index": 2,
                        "shape": [4, 1],
                        "element_size": 4,
                        "trainable": True,
                    },
                    {
                        "name": "layer.lora_b",
                        "parameter_index": 3,
                        "shape": [2, 2],
                        "element_size": 4,
                        "trainable": True,
                    },
                ],
            },
        ],
        world_size=2,
    )


def _static_report() -> dict:
    return {
        "status": "success",
        "mode": "static",
        "model_dir": "/models/Qwen3.5-0.8B",
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "training_method": "lora",
        "lora": {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.0,
            "target_modules": "all-linear",
        },
        "gradient_checkpointing": False,
        "gradient_accumulation_steps": 1,
        "batch_size": 1,
        "sequence_length": 16,
        "data_seed": 7,
        "batch": {"fingerprint_sha256": "same-batch"},
        "static_estimate": {
            "parameter_bytes": 64,
            "trainable_parameter_bytes": 32,
            "frozen_parameter_bytes": 32,
            "first_step_graph_phase_peak_bytes": 164,
            "post_graph_live_bytes": 116,
            "optimizer_phase_peak_bytes": 228,
            "optimizer_state_bytes": 64,
            "optimizer_temporary_bytes": 48,
            "workspace_peak_contribution_bytes": 0,
            "optimizer_temporary": {"optimizer": "adamw"},
            "graph": {
                "peak_bytes_by_category": {
                    "frozen_parameter": 32,
                    "trainable_parameter": 32,
                    "temporary": 100,
                },
                "final_bytes_by_category": {
                    "frozen_parameter": 32,
                    "trainable_parameter": 32,
                    "gradient": 32,
                    "output": 20,
                },
                "gradient_production_phase": {
                    "peak_live_bytes": 146,
                    "peak_bytes_by_category": {
                        "frozen_parameter": 32,
                        "trainable_parameter": 32,
                        "gradient": 32,
                        "temporary": 50,
                    },
                },
            },
        },
    }


def _rank_report(rank: int, *, overall: int = 194) -> dict:
    return {
        "status": "success",
        "rank": rank,
        "world_size": 2,
        "model_dir": "/models/Qwen3.5-0.8B",
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "training_method": "lora",
        "lora": {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.0,
            "target_modules": "all-linear",
        },
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
        "loss_digest_sha256": "same-loss",
        "parameters": {
            "logical": {"parameter_bytes": 64},
            "local_shard": {
                "parameter_storage_bytes": 32,
                "logical_trainable_parameter_bytes": 16,
            },
            "local_gradient_storage_bytes": 16,
            "optimizer_state_tensor_bytes": 32,
        },
        "fsdp2": {"sharding_plan": _plan()},
        "memory_phases": {
            "forward_peak_bytes": 184,
            "backward_peak_bytes": 194,
            "graph_peak_bytes": 194,
            "optimizer_peak_bytes": 172,
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
            "all_gather": {"calls": 2, "bytes": 128},
            "reduce_scatter": {"calls": 1, "bytes": 64},
        },
        "node_pairs": [],
        "operation_timeline": {},
    }


def test_fsdp2_lora_comparison_accepts_matching_phase_predictions() -> None:
    report = compare_fsdp2_lora_reports(
        [_rank_report(0), _rank_report(1)],
        _static_report(),
        _cluster_report(),
        max_error_percent=1.0,
    )

    assert report["status"] == "success"
    assert report["projections"][0]["first_step_peak_bytes"] == 194
    assert report["ranks"][0]["comparisons"]["overall"][
        "absolute_error_percent"
    ] == 0.0
    assert "Qwen Hybrid FSDP2 LoRA SFT" in render_markdown(report)


def test_fsdp2_lora_comparison_rejects_large_phase_error() -> None:
    report = compare_fsdp2_lora_reports(
        [_rank_report(0, overall=300), _rank_report(1, overall=300)],
        _static_report(),
        _cluster_report(),
        max_error_percent=5.0,
    )

    assert report["status"] == "failed"
    assert report["ranks"][0]["status"] == "failed"


def test_fsdp2_lora_controller_imports_package_when_started_as_script() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import run_qwen_fsdp2_lora_sft_memory; "
                "import fakegpu.fsdp_memory; "
                "print('ok')"
            ),
        ],
        cwd=ROOT / "verification",
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"
