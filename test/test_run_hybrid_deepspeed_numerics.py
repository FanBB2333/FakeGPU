from __future__ import annotations

import copy

import pytest

from verification.run_hybrid_deepspeed_numerics import (
    _collective_calls,
    _validate_collective_calls,
    _validate_rank_reports,
)


def _rank_report(rank: int, zero_stage: int = 3) -> dict[str, object]:
    final_parameter = [[0.775, -0.45]]
    return {
        "status": "success",
        "rank": rank,
        "zero_stage": zero_stage,
        "effective_zero_stage": zero_stage,
        "gradient_accumulation_steps": 2,
        "micro_step_1_global_steps": 0,
        "micro_step_2_global_steps": 1,
        "parameters_after_micro_steps": [
            [[1.0, 0.0]],
            final_parameter,
        ],
        "gathered_parameters": [
            final_parameter[0],
            final_parameter[0],
        ],
        "offload": {
            "optimizer_requested": False,
            "parameters_requested": False,
            "optimizer_state_devices": ["cuda:0"],
            "local_parameter_partition_device": "cuda:0",
        },
    }


def test_validate_rank_reports_accepts_two_rank_zero3() -> None:
    _validate_rank_reports(
        [_rank_report(0), _rank_report(1)],
        world_size=2,
        zero_stage=3,
        precision="fp32",
    )


def test_validate_rank_reports_rejects_early_optimizer_step() -> None:
    reports = [_rank_report(0), _rank_report(1)]
    reports[0] = copy.deepcopy(reports[0])
    reports[0]["micro_step_1_global_steps"] = 1

    with pytest.raises(AssertionError, match="accumulation boundary"):
        _validate_rank_reports(
            reports,
            world_size=2,
            zero_stage=3,
            precision="fp32",
        )


def test_collective_calls_fill_missing_operations() -> None:
    calls = _collective_calls(
        {
            "collectives": {
                "broadcast": {"calls": 2},
                "reduce_scatter": {"calls": 3},
            }
        }
    )

    assert calls == {
        "all_reduce": 0,
        "all_gather": 0,
        "reduce_scatter": 3,
        "broadcast": 2,
        "reduce": 0,
        "all_to_all": 0,
    }


def test_zero3_requires_parameter_all_gather() -> None:
    calls = {
        "all_reduce": 0,
        "all_gather": 0,
        "reduce_scatter": 1,
        "broadcast": 1,
        "reduce": 0,
        "all_to_all": 0,
    }

    with pytest.raises(AssertionError, match="did not gather parameters"):
        _validate_collective_calls(calls, zero_stage=3)


def test_validate_rank_reports_accepts_cpu_offload() -> None:
    reports = [_rank_report(0), _rank_report(1)]
    for report in reports:
        report["offload"] = {
            "optimizer_requested": True,
            "parameters_requested": True,
            "optimizer_state_devices": ["cpu"],
            "local_parameter_partition_device": "cpu",
        }

    _validate_rank_reports(
        reports,
        world_size=2,
        zero_stage=3,
        precision="fp32",
        offload="optimizer-and-parameter",
    )
