from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from verification.run_hybrid_deepspeed_pipeline import (
    EXPECTED_FINAL,
    _validate_communication,
    _validate_rank_reports,
)
from verification.deepspeed_pipeline_worker import (
    _enable_batched_deepspeed_p2p,
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
        "p2p_api": "send_recv",
        "p2p_process_group": "pipeline_default",
        "loss": 6.25,
        "all_stage_parameters": copy.deepcopy(EXPECTED_FINAL[1]),
    }


def test_validate_rank_reports_accepts_pipeline_update() -> None:
    _validate_rank_reports(
        [_rank_report(0), _rank_report(1)],
        precision="fp32",
        activation_checkpoint_interval=0,
        gradient_accumulation_steps=1,
        batched_p2p=False,
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
        batched_p2p=False,
    )


def test_validate_rank_reports_accepts_batched_p2p() -> None:
    reports = [_rank_report(0), _rank_report(1)]
    for report in reports:
        report["p2p_api"] = "batch_isend_irecv"
        report["p2p_process_group"] = "dedicated"
    _validate_rank_reports(
        reports,
        precision="fp32",
        activation_checkpoint_interval=0,
        gradient_accumulation_steps=1,
        batched_p2p=True,
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
            batched_p2p=False,
        )


def test_enable_batched_deepspeed_p2p_uses_one_work_item() -> None:
    calls: list[tuple[object, object, int, object]] = []

    class Work:
        def wait(self) -> str:
            return "complete"

    class Dist:
        isend = object()
        irecv = object()

        @staticmethod
        def P2POp(
            operation: object,
            tensor: object,
            peer: int,
            *,
            group: object,
        ) -> tuple[object, object, int, object]:
            return operation, tensor, peer, group

        @staticmethod
        def batch_isend_irecv(
            operations: list[tuple[object, object, int, object]],
        ) -> list[Work]:
            calls.extend(operations)
            return [Work()]

    class Grid:
        @staticmethod
        def get_stage_id() -> int:
            return 0

        @staticmethod
        def stage_to_global(*, stage_id: int) -> int:
            return stage_id

    validated: list[tuple[int, int]] = []
    p2p = SimpleNamespace(
        _grid=Grid(),
        _is_valid_send_recv=lambda src, dst: validated.append((src, dst)),
    )
    process_group = object()
    _enable_batched_deepspeed_p2p(Dist, p2p, process_group)

    assert p2p.send("activation", 1) == "complete"
    assert p2p.recv("gradient", 1) == "complete"
    assert calls == [
        (Dist.isend, "activation", 1, process_group),
        (Dist.irecv, "gradient", 1, process_group),
    ]
    assert validated == [(0, 1), (1, 0)]


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
