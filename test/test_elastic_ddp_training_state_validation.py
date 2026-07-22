from __future__ import annotations

from copy import deepcopy

import pytest

from verification.elastic_ddp_training_state_validation import (
    EXPECTED_GATHERED_PENDING_GRADIENTS,
    EXPECTED_PENDING_GRADIENTS,
    EXPECTED_STEP_ONE_EXP_AVG,
    EXPECTED_STEP_ONE_EXP_AVG_SQ,
    EXPECTED_STEP_ONE_GRADIENT,
    EXPECTED_STEP_ONE_PARAMETERS,
    EXPECTED_STEP_ONE_STATE,
    EXPECTED_STEP_TWO_EXP_AVG,
    EXPECTED_STEP_TWO_EXP_AVG_SQ,
    EXPECTED_STEP_TWO_GRADIENT,
    EXPECTED_STEP_TWO_PARAMETERS,
    EXPECTED_STEP_TWO_STATE,
    validate_elastic_ddp_training_state_reports,
)


def _reports() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    initial: list[dict[str, object]] = []
    restarted: list[dict[str, object]] = []
    random_values = {0: 0.6766379475593567, 1: 0.5308855175971985}
    for rank in (0, 1):
        path = f"/tmp/training-state-rank-{rank}.pt"
        digest = f"sha256-rank-{rank}"
        initial.append(
            {
                "backend": "gloo",
                "status": (
                    "expected_process_exit"
                    if rank == 1
                    else "waiting_for_restart"
                ),
                "rank": rank,
                "world_size": 2,
                "restart_count": 0,
                "restart_arrival_count": 1,
                "local_restart_count": 0,
                "observed_local_restart_counts": [0, 0],
                "max_restarts": 1,
                "completed_optimizer_steps": 1,
                "accumulation_micro_step": 1,
                "optimizer": "adamw",
                "learning_rate": 0.005,
                "scheduler_last_epoch": 1,
                "initial_process_group_destroyed_before_exit": False,
                "selected_for_initial_exit": rank == 1,
                "expected_process_exit_code": 86 if rank == 1 else None,
                "gradient_after_step_one": EXPECTED_STEP_ONE_GRADIENT,
                "parameters_after_step_one": EXPECTED_STEP_ONE_PARAMETERS,
                "exp_avg_after_step_one": EXPECTED_STEP_ONE_EXP_AVG,
                "exp_avg_sq_after_step_one": EXPECTED_STEP_ONE_EXP_AVG_SQ,
                "pending_gradient": EXPECTED_PENDING_GRADIENTS[rank],
                "gathered_pending_gradients": (
                    EXPECTED_GATHERED_PENDING_GRADIENTS
                ),
                "gathered_state_after_step_one": EXPECTED_STEP_ONE_STATE,
                "optimizer_state_step": 1,
                "expected_next_random": random_values[rank],
                "saved_checkpoint": {
                    "path": path,
                    "bytes": 9000,
                    "sha256": digest,
                },
                "store_events": [{"operation": "set"}],
                "pid": 100 + rank,
                "run_id": "training-state-test",
            }
        )
        restarted.append(
            {
                "backend": "gloo",
                "status": "success",
                "rank": rank,
                "world_size": 2,
                "restart_count": 1,
                "restart_arrival_count": 2,
                "local_restart_count": 1,
                "observed_local_restart_counts": [1, 1],
                "max_restarts": 1,
                "optimizer": "adamw",
                "restored_optimizer_steps": 1,
                "restored_accumulation_micro_step": 1,
                "completed_optimizer_steps": 2,
                "accumulation_micro_step": 0,
                "checkpoint_loaded": True,
                "loaded_checkpoint": {
                    "path": path,
                    "bytes": 9000,
                    "sha256": digest,
                },
                "restored_parameters": EXPECTED_STEP_ONE_PARAMETERS,
                "restored_exp_avg": EXPECTED_STEP_ONE_EXP_AVG,
                "restored_exp_avg_sq": EXPECTED_STEP_ONE_EXP_AVG_SQ,
                "restored_pending_gradient": EXPECTED_PENDING_GRADIENTS[rank],
                "gathered_restored_pending_gradients": (
                    EXPECTED_GATHERED_PENDING_GRADIENTS
                ),
                "gathered_restored_state": EXPECTED_STEP_ONE_STATE,
                "gradient_after_step_two": EXPECTED_STEP_TWO_GRADIENT,
                "parameters_after_step_two": EXPECTED_STEP_TWO_PARAMETERS,
                "exp_avg_after_step_two": EXPECTED_STEP_TWO_EXP_AVG,
                "exp_avg_sq_after_step_two": EXPECTED_STEP_TWO_EXP_AVG_SQ,
                "gathered_state_after_step_two": EXPECTED_STEP_TWO_STATE,
                "optimizer_state_step": 2,
                "learning_rate": 0.0025,
                "scheduler_last_epoch": 2,
                "restored_next_random": random_values[rank],
                "store_events": [{"operation": "set"}],
                "pid": 200 + rank,
                "run_id": "training-state-test",
            }
        )
    return initial, restarted


def test_validate_training_state_reports_accepts_complete_resume() -> None:
    initial, restarted = _reports()

    summary = validate_elastic_ddp_training_state_reports(
        initial,
        restarted,
        backend="gloo",
        require_physical_gpu=False,
    )

    assert summary["status"] == "success"
    assert summary["completed_optimizer_steps"] == 2
    assert summary["parameters_after_step_two"] == (
        EXPECTED_STEP_TWO_PARAMETERS
    )
    assert summary["restored_next_random"] == {
        "0": 0.6766379475593567,
        "1": 0.5308855175971985,
    }


def test_validate_training_state_reports_rejects_scheduler_drift() -> None:
    initial, restarted = _reports()
    invalid = deepcopy(restarted)
    invalid[0]["learning_rate"] = 0.005

    with pytest.raises(AssertionError, match="learning rate"):
        validate_elastic_ddp_training_state_reports(
            initial,
            invalid,
            backend="gloo",
            require_physical_gpu=False,
        )


def test_validate_training_state_reports_accepts_dynamic_failed_rank() -> None:
    initial, restarted = _reports()
    initial[0].update(
        {
            "status": "expected_process_exit",
            "selected_for_initial_exit": True,
            "expected_process_exit_code": 86,
        }
    )
    initial[1].update(
        {
            "status": "waiting_for_restart",
            "selected_for_initial_exit": False,
            "expected_process_exit_code": None,
        }
    )
    restarted[1]["local_restart_count"] = 0

    summary = validate_elastic_ddp_training_state_reports(
        initial,
        restarted,
        backend="gloo",
        require_physical_gpu=False,
    )

    assert summary["failed_rank"] == 0
    assert summary["local_restart_counts"] == {"0": 1, "1": 0}
