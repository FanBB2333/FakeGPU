from __future__ import annotations

import copy

import pytest

from verification.deepspeed_autoep_worker import (
    EXPECTED_INITIAL_W1,
    _nested_close,
)
from verification.run_hybrid_deepspeed_autoep import (
    _validate_communication,
    _validate_rank_reports,
)


def _rank_report(rank: int) -> dict[str, object]:
    output = [[[1.0 + rank, 2.0], [3.0, 4.0 + rank]]]
    return {
        "status": "success",
        "rank": rank,
        "world_size": 2,
        "zero_stage": 0,
        "effective_zero_stage": 0,
        "precision": "fp32",
        "engine_type": "DeepSpeedEngine",
        "moe_layer_type": "AutoEPMoELayer",
        "expert_parallel_size": 2,
        "expert_parallel_rank": rank,
        "num_global_experts": 2,
        "num_local_experts": 1,
        "local_expert_shape": [1, 2, 2],
        "global_steps": 1,
        "tokens_per_expert": [2.0, 1.0] if rank == 0 else [1.0, 2.0],
        "initial_full_w1": copy.deepcopy(EXPECTED_INITIAL_W1),
        "final_full_w1": [
            [[0.9, 0.0], [0.0, 0.9]],
            [[0.4, 0.0], [0.0, 0.4]],
        ],
        "output": output,
        "reference_output": copy.deepcopy(output),
        "loss": 2.0 + rank,
        "reference_loss": 2.0 + rank,
        "gradient_norms": {"w1": 1.0, "w2": 1.0, "w3": 1.0},
        "parameter_max_abs_deltas": {"w1": 0.1, "w2": 0.1, "w3": 0.1},
    }


def test_nested_close_supports_autoep_tensors() -> None:
    assert _nested_close([[[1.0, 2.01]]], [[[1.0, 2.0]]], 0.02)
    assert not _nested_close([[[1.0, 2.1]]], [[[1.0, 2.0]]], 0.02)


def test_validate_rank_reports_accepts_autoep_update() -> None:
    _validate_rank_reports(
        [_rank_report(0), _rank_report(1)],
        zero_stage=0,
        precision="fp32",
    )


def test_validate_rank_reports_rejects_missing_expert_update() -> None:
    reports = [_rank_report(0), _rank_report(1)]
    reports[1]["parameter_max_abs_deltas"] = {
        "w1": 0.0,
        "w2": 0.1,
        "w3": 0.1,
    }
    with pytest.raises(AssertionError, match="parameter did not update"):
        _validate_rank_reports(
            reports,
            zero_stage=0,
            precision="fp32",
        )


def test_validate_communication_accepts_all_to_all_or_p2p() -> None:
    collective = {
        "collectives": {"all_to_all": {"calls": 4}},
        "point_to_point": {"sends": 0},
        "node_pairs": [{"total_bytes": 128}],
    }
    assert _validate_communication(collective) == {
        "all_to_all": 4,
        "point_to_point_sends": 0,
    }

    p2p = {
        "collectives": {"all_to_all": {"calls": 0}},
        "point_to_point": {"sends": 4},
        "node_pairs": [{"total_bytes": 128}],
    }
    assert _validate_communication(p2p) == {
        "all_to_all": 0,
        "point_to_point_sends": 4,
    }

    with pytest.raises(AssertionError, match="did not record"):
        _validate_communication(
            {
                "collectives": {"all_to_all": {"calls": 0}},
                "point_to_point": {"sends": 0},
                "node_pairs": [{"total_bytes": 0}],
            }
        )
