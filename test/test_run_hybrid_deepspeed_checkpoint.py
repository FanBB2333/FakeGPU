from __future__ import annotations

import copy

import pytest

from verification.run_hybrid_deepspeed_checkpoint import (
    validate_checkpoint_rank_reports,
)


def _rank_report(rank: int, zero_stage: int = 3) -> dict[str, object]:
    saved = [[0.7, -0.2]]
    final = [[0.68, -0.18]]
    return {
        "status": "success",
        "rank": rank,
        "zero_stage": zero_stage,
        "effective_zero_stage": zero_stage,
        "precision": "fp32",
        "global_steps_after_resume": 2,
        "client_state": {
            "validation_token": "fakegpu-deepspeed-checkpoint-v1",
            "saved_global_steps": 1,
        },
        "learning_rates": {
            "saved": 0.025,
            "restored": 0.025,
            "uninterrupted_after_second": 0.0125,
            "resumed_after_second": 0.0125,
        },
        "parameters": {
            "saved": saved,
            "restored": saved,
            "uninterrupted_after_second": final,
            "resumed_after_second": final,
        },
        "checkpoint_files": [
            "latest",
            "global_step1/mp_rank_00_model_states.pt",
            "global_step1/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt",
        ],
    }


def test_validate_checkpoint_rank_reports_accepts_resume() -> None:
    validate_checkpoint_rank_reports(
        [_rank_report(0), _rank_report(1)],
        world_size=2,
        zero_stage=3,
        precision="fp32",
    )


def test_validate_checkpoint_rank_reports_rejects_optimizer_mismatch() -> None:
    reports = [_rank_report(0), _rank_report(1)]
    reports[1] = copy.deepcopy(reports[1])
    reports[1]["parameters"]["resumed_after_second"] = [[0.5, 0.5]]

    with pytest.raises(AssertionError, match="optimizer state mismatch"):
        validate_checkpoint_rank_reports(
            reports,
            world_size=2,
            zero_stage=3,
            precision="fp32",
        )


def test_validate_checkpoint_rank_reports_requires_optimizer_file() -> None:
    reports = [_rank_report(0), _rank_report(1)]
    for report in reports:
        report["checkpoint_files"] = [
            "latest",
            "global_step1/mp_rank_00_model_states.pt",
        ]

    with pytest.raises(AssertionError, match="no optimizer state"):
        validate_checkpoint_rank_reports(
            reports,
            world_size=2,
            zero_stage=3,
            precision="fp32",
        )
