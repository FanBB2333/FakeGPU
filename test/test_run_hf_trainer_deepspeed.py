from __future__ import annotations

import copy

import pytest

from verification.run_hf_trainer_deepspeed import (
    validate_hf_trainer_reports,
)


def _rank_report(rank: int) -> dict[str, object]:
    after_hash = "b" * 64
    return {
        "status": "success",
        "rank": rank,
        "zero_stage": 3,
        "precision": "bf16",
        "dataset": {"fingerprint_sha256": "d" * 64},
        "trainer_identity": {
            "process_index": rank,
            "local_process_index": 0,
            "world_size": 2,
        },
        "engine": {
            "type": "DeepSpeedEngine",
            "effective_zero_stage": 3,
        },
        "training": {"global_step": 1, "training_loss": 2.5},
        "parameter_update_probe": {
            "update_norm": 0.01,
            "before_sha256": "a" * 64,
            "after_sha256": after_hash,
            "gathered_after_sha256": [after_hash, after_hash],
        },
        "resolved_deepspeed_config": {
            "zero_optimization": {"stage": 3}
        },
    }


def test_validate_hf_trainer_reports_accepts_hybrid_mapping() -> None:
    validate_hf_trainer_reports(
        [_rank_report(0), _rank_report(1)],
        world_size=2,
        zero_stage=3,
        precision="bf16",
        max_steps=1,
    )


def test_validate_hf_trainer_reports_rejects_physical_rank_one() -> None:
    reports = [_rank_report(0), _rank_report(1)]
    reports[1] = copy.deepcopy(reports[1])
    reports[1]["trainer_identity"]["local_process_index"] = 1

    with pytest.raises(AssertionError, match="physical cuda:0"):
        validate_hf_trainer_reports(
            reports,
            world_size=2,
            zero_stage=3,
            precision="bf16",
            max_steps=1,
        )


def test_validate_hf_trainer_reports_rejects_rank_divergence() -> None:
    reports = [_rank_report(0), _rank_report(1)]
    reports[0] = copy.deepcopy(reports[0])
    reports[0]["parameter_update_probe"]["gathered_after_sha256"] = [
        "b" * 64,
        "c" * 64,
    ]

    with pytest.raises(AssertionError, match="differ by rank"):
        validate_hf_trainer_reports(
            reports,
            world_size=2,
            zero_stage=3,
            precision="bf16",
            max_steps=1,
        )
