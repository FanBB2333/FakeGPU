from __future__ import annotations

import copy

import pytest

from verification.run_hybrid_deepspeed_pipeline import (
    EXPECTED_FINAL,
    _validate_communication,
    _validate_rank_reports,
)


def _rank_report(rank: int) -> dict[str, object]:
    return {
        "status": "success",
        "engine_type": "PipelineEngine",
        "pipe_parallel_size": 2,
        "pipe_stage_id": rank,
        "gradient_accumulation_steps": 1,
        "global_steps": 1,
        "activation_checkpoint_interval": 0,
        "loss": 6.25,
        "all_stage_parameters": copy.deepcopy(EXPECTED_FINAL[1]),
    }


def test_validate_rank_reports_accepts_pipeline_update() -> None:
    _validate_rank_reports(
        [_rank_report(0), _rank_report(1)],
        precision="fp32",
        activation_checkpoint_interval=0,
        gradient_accumulation_steps=1,
    )


def test_validate_rank_reports_accepts_two_micro_batches() -> None:
    reports = [_rank_report(0), _rank_report(1)]
    for report in reports:
        report["gradient_accumulation_steps"] = 2
        report["loss"] = 4.25
        report["all_stage_parameters"] = copy.deepcopy(EXPECTED_FINAL[2])
    _validate_rank_reports(
        reports,
        precision="fp32",
        activation_checkpoint_interval=0,
        gradient_accumulation_steps=2,
    )


def test_validate_rank_reports_rejects_stage_divergence() -> None:
    reports = [_rank_report(0), _rank_report(1)]
    reports[1]["all_stage_parameters"] = [
        EXPECTED_FINAL[1][0],
        [0.4, 0.6],
    ]
    with pytest.raises(AssertionError, match="parameter mismatch"):
        _validate_rank_reports(
            reports,
            precision="fp32",
            activation_checkpoint_interval=0,
            gradient_accumulation_steps=1,
        )


def test_validate_communication_accepts_pipeline_p2p() -> None:
    summary = _validate_communication(
        {
            "point_to_point": {
                "operations": 5,
                "sends": 5,
                "bytes": 128,
            },
            "ranks": [
                {"point_to_point_calls": 5},
                {"point_to_point_calls": 5},
            ],
            "node_pairs": [
                {
                    "point_to_point_operations": 5,
                }
            ],
        }
    )
    assert summary == {"operations": 5, "sends": 5, "bytes": 128}


def test_validate_communication_rejects_missing_p2p() -> None:
    with pytest.raises(AssertionError, match="P2P communication is empty"):
        _validate_communication(
            {
                "point_to_point": {
                    "operations": 0,
                    "sends": 0,
                    "bytes": 0,
                }
            }
        )
