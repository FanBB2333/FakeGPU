from __future__ import annotations

import copy

import pytest

from verification.run_qwen_deepspeed_lora_sft import (
    render_markdown,
    validate_qwen_deepspeed_reports,
)


def _rank_report(rank: int) -> dict[str, object]:
    return {
        "status": "success",
        "rank": rank,
        "world_size": 2,
        "zero_stage": 3,
        "engine": {
            "effective_zero_stage": 3,
            "gradient_accumulation_steps": 2,
        },
        "global_steps_after_microsteps": [0, 1],
        "loss": 4.25 + rank,
        "parameters": {"trainable_parameter_count": 1024},
        "optimizer_update_probe": {
            "before_squared_norm": 0.0,
            "after_squared_norm": 0.125,
            "gathered_after_squared_norms": [0.125, 0.125],
        },
        "memory_phases": {"overall_peak_bytes": 1024 * (rank + 1)},
    }


def test_validate_qwen_deepspeed_reports_accepts_update() -> None:
    validate_qwen_deepspeed_reports(
        [_rank_report(0), _rank_report(1)],
        world_size=2,
        zero_stage=3,
        gradient_accumulation_steps=2,
    )


def test_validate_qwen_deepspeed_reports_rejects_missing_update() -> None:
    reports = [_rank_report(0), _rank_report(1)]
    reports[1] = copy.deepcopy(reports[1])
    probe = reports[1]["optimizer_update_probe"]
    assert isinstance(probe, dict)
    probe["after_squared_norm"] = 0.0

    with pytest.raises(AssertionError, match="was not updated"):
        validate_qwen_deepspeed_reports(
            reports,
            world_size=2,
            zero_stage=3,
            gradient_accumulation_steps=2,
        )


def test_render_markdown_includes_memory_and_communication() -> None:
    report = {
        "status": "success",
        "model_dir": "/models/Qwen3.5-0.8B",
        "physical_device_name": "Example GPU",
        "deepspeed_version": "0.19.2",
        "zero_stage": 3,
        "world_size": 2,
        "dtype": "bfloat16",
        "sequence_length": 16,
        "ranks": [
            {
                "rank": 0,
                "loss": 4.25,
                "overall_peak_bytes": 2 * 1024**3,
                "update_probe_after": 0.125,
            }
        ],
        "collective_calls": {"all_gather": 4, "reduce_scatter": 2},
        "node_pair_total_bytes": 4096,
        "node_pair_peak_bytes_per_operation": 2048,
    }

    markdown = render_markdown(report)

    assert "Qwen Hybrid DeepSpeed LoRA SFT" in markdown
    assert "2.000 GiB" in markdown
    assert "`all_gather`" in markdown
