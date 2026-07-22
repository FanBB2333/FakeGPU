from __future__ import annotations

import copy

import pytest

from verification.deepspeed_autotp_worker import (
    EXPECTED_COLUMN,
    EXPECTED_LOSS,
    EXPECTED_OUTPUT,
    EXPECTED_ROW,
    _tensor_close,
)
from verification.run_hybrid_deepspeed_autotp import (
    _validate_communication,
    _validate_rank_reports,
)


def _rank_report(rank: int) -> dict[str, object]:
    return {
        "status": "success",
        "rank": rank,
        "world_size": 2,
        "zero_stage": 0,
        "effective_zero_stage": 0,
        "precision": "fp32",
        "engine_type": "DeepSpeedEngine",
        "autotp_size": 2,
        "tensor_parallel_rank": rank,
        "tensor_parallel_world_size": 2,
        "column_layer_type": "LinearLayer",
        "row_layer_type": "LinearAllreduce",
        "column_local_shape": [2, 4],
        "row_local_shape": [2, 2],
        "global_steps": 1,
        "output": copy.deepcopy(EXPECTED_OUTPUT),
        "loss": EXPECTED_LOSS,
        "full_column_weight_after_step": copy.deepcopy(EXPECTED_COLUMN),
        "full_row_weight_after_step": copy.deepcopy(EXPECTED_ROW),
    }


def test_tensor_close_supports_nested_weights() -> None:
    assert _tensor_close(
        [[1.0, 2.001], [3.0, 4.0]],
        [[1.0, 2.0], [3.0, 4.0]],
        0.01,
    )
    assert not _tensor_close([[1.0]], [[1.1]], 0.01)


def test_validate_rank_reports_accepts_autotp_update() -> None:
    _validate_rank_reports(
        [_rank_report(0), _rank_report(1)],
        zero_stage=0,
        precision="fp32",
    )


def test_validate_rank_reports_rejects_unsharded_column() -> None:
    reports = [_rank_report(0), _rank_report(1)]
    reports[1]["column_local_shape"] = [4, 4]
    with pytest.raises(AssertionError, match="column_local_shape mismatch"):
        _validate_rank_reports(
            reports,
            zero_stage=0,
            precision="fp32",
        )


def test_validate_rank_reports_rejects_bad_update() -> None:
    reports = [_rank_report(0), _rank_report(1)]
    reports[0]["full_row_weight_after_step"] = [[0.0, 0.0], [0.0, 0.0]]
    with pytest.raises(AssertionError, match="row update mismatch"):
        _validate_rank_reports(
            reports,
            zero_stage=0,
            precision="fp32",
        )


def test_validate_communication_requires_tp_collectives() -> None:
    summary = _validate_communication(
        {
            "collectives": {
                "all_reduce": {"calls": 4},
                "all_gather": {"calls": 2},
            },
            "node_pairs": [{"total_bytes": 512}],
        }
    )
    assert summary == {"all_reduce": 4, "all_gather": 2}

    with pytest.raises(AssertionError, match="incomplete"):
        _validate_communication(
            {
                "collectives": {
                    "all_reduce": {"calls": 4},
                    "all_gather": {"calls": 0},
                }
            }
        )
