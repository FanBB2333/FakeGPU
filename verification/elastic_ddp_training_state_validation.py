from __future__ import annotations

from pathlib import Path
from typing import Any


EXPECTED_STEP_ONE_GRADIENT = [[2.5, 5.0]]
EXPECTED_STEP_ONE_PARAMETERS = [[0.989, -0.01]]
EXPECTED_STEP_ONE_EXP_AVG = [[0.25, 0.5]]
EXPECTED_STEP_ONE_EXP_AVG_SQ = [[0.0625, 0.25]]
EXPECTED_PENDING_GRADIENTS = {
    0: [[2.5, 5.0]],
    1: [[3.0, 6.0]],
}
EXPECTED_GATHERED_PENDING_GRADIENTS = [[2.5, 5.0], [3.0, 6.0]]
EXPECTED_STEP_ONE_STATE = [
    [0.989, -0.01, 0.25, 0.5, 0.0625, 0.25],
] * 2
EXPECTED_STEP_TWO_GRADIENT = [[6.5, 13.0]]
EXPECTED_STEP_TWO_PARAMETERS = [[0.98383826, -0.01466224]]
EXPECTED_STEP_TWO_EXP_AVG = [[0.875, 1.75]]
EXPECTED_STEP_TWO_EXP_AVG_SQ = [[0.484375, 1.9375]]
EXPECTED_STEP_TWO_STATE = [
    [
        0.98383826,
        -0.01466224,
        0.875,
        1.75,
        0.484375,
        1.9375,
    ],
] * 2
EXPECTED_NEXT_RANDOM = {
    0: 0.6766379475593567,
    1: 0.5308855175971985,
}
EXPECTED_PROCESS_EXIT_CODE = 86


def _nested_allclose(
    actual: object,
    expected: list[list[float]],
    *,
    tolerance: float = 1e-6,
) -> bool:
    if not isinstance(actual, list) or len(actual) != len(expected):
        return False
    for actual_row, expected_row in zip(actual, expected):
        if not isinstance(actual_row, list) or len(actual_row) != len(expected_row):
            return False
        if any(
            abs(float(actual_value) - expected_value) > tolerance
            for actual_value, expected_value in zip(actual_row, expected_row)
        ):
            return False
    return True


def validate_elastic_ddp_training_state_reports(
    initial: list[dict[str, Any]],
    restarted: list[dict[str, Any]],
    *,
    backend: str,
    require_physical_gpu: bool,
    check_checkpoint_files: bool = False,
) -> dict[str, Any]:
    if len(initial) != 2 or len(restarted) != 2:
        raise AssertionError(
            "elastic training-state recovery requires two initial and two "
            "restarted worker reports"
        )
    initial_by_rank = {int(item.get("rank", -1)): item for item in initial}
    restarted_by_rank = {
        int(item.get("rank", -1)): item for item in restarted
    }
    if set(initial_by_rank) != {0, 1} or set(restarted_by_rank) != {0, 1}:
        raise AssertionError("elastic training-state ranks are invalid")

    for rank, report in initial_by_rank.items():
        expected_status = (
            "expected_process_exit" if rank == 1 else "waiting_for_restart"
        )
        if report.get("backend") != backend:
            raise AssertionError(f"initial backend mismatch: {report}")
        if report.get("status") != expected_status:
            raise AssertionError(
                f"unexpected initial training-state status: {report}"
            )
        if int(report.get("restart_count", -1)) != 0:
            raise AssertionError(f"initial restart count is invalid: {report}")
        if int(report.get("restart_arrival_count", -1)) != 1:
            raise AssertionError(f"initial arrival count is invalid: {report}")
        if int(report.get("local_restart_count", -1)) != 0:
            raise AssertionError(
                f"initial local restart count is invalid: {report}"
            )
        if report.get("observed_local_restart_counts") != [0, 0]:
            raise AssertionError(
                f"initial restart generation is invalid: {report}"
            )
        if int(report.get("max_restarts", -1)) != 1:
            raise AssertionError(f"initial restart limit is invalid: {report}")
        if int(report.get("world_size", 0)) != 2:
            raise AssertionError(f"initial world size is invalid: {report}")
        if int(report.get("completed_optimizer_steps", -1)) != 1:
            raise AssertionError(
                f"initial optimizer step count is invalid: {report}"
            )
        if int(report.get("accumulation_micro_step", -1)) != 1:
            raise AssertionError(
                f"initial accumulation progress is invalid: {report}"
            )
        if report.get("optimizer") != "adamw":
            raise AssertionError(f"unexpected optimizer metadata: {report}")
        if abs(float(report.get("learning_rate", -1.0)) - 0.005) > 1e-12:
            raise AssertionError(f"initial learning rate is invalid: {report}")
        if int(report.get("scheduler_last_epoch", -1)) != 1:
            raise AssertionError(f"initial scheduler state is invalid: {report}")
        if report.get("initial_process_group_destroyed_before_exit") is not False:
            raise AssertionError(
                f"initial communicator was cleaned up before exit: {report}"
            )
        if rank == 1:
            if (
                report.get("selected_for_initial_exit") is not True
                or int(report.get("expected_process_exit_code", -1))
                != EXPECTED_PROCESS_EXIT_CODE
            ):
                raise AssertionError(f"failure marker is invalid: {report}")
        elif report.get("selected_for_initial_exit") is not False:
            raise AssertionError(f"unexpected failure selection: {report}")
        if (
            not _nested_allclose(
                report.get("gradient_after_step_one"),
                EXPECTED_STEP_ONE_GRADIENT,
            )
            or not _nested_allclose(
                report.get("parameters_after_step_one"),
                EXPECTED_STEP_ONE_PARAMETERS,
            )
            or not _nested_allclose(
                report.get("exp_avg_after_step_one"),
                EXPECTED_STEP_ONE_EXP_AVG,
            )
            or not _nested_allclose(
                report.get("exp_avg_sq_after_step_one"),
                EXPECTED_STEP_ONE_EXP_AVG_SQ,
            )
            or not _nested_allclose(
                report.get("pending_gradient"),
                EXPECTED_PENDING_GRADIENTS[rank],
            )
            or not _nested_allclose(
                report.get("gathered_pending_gradients"),
                EXPECTED_GATHERED_PENDING_GRADIENTS,
            )
            or not _nested_allclose(
                report.get("gathered_state_after_step_one"),
                EXPECTED_STEP_ONE_STATE,
            )
        ):
            raise AssertionError(
                f"initial AdamW training state is invalid: {report}"
            )
        if int(report.get("optimizer_state_step", -1)) != 1:
            raise AssertionError(f"initial AdamW step is invalid: {report}")
        expected_random = EXPECTED_NEXT_RANDOM[rank]
        if (
            abs(float(report.get("expected_next_random", -1.0)) - expected_random)
            > 1e-12
        ):
            raise AssertionError(f"initial RNG state is invalid: {report}")
        saved = report.get("saved_checkpoint")
        if (
            not isinstance(saved, dict)
            or int(saved.get("bytes", 0)) <= 0
            or not saved.get("sha256")
        ):
            raise AssertionError(f"checkpoint metadata is invalid: {report}")
        if check_checkpoint_files and not Path(str(saved.get("path", ""))).is_file():
            raise AssertionError(f"checkpoint file is missing: {saved}")
        if not report.get("store_events"):
            raise AssertionError("process-group store tracing is empty")
        if require_physical_gpu and not report.get("physical_device_name"):
            raise AssertionError(f"physical GPU metadata is missing: {report}")

    for rank, report in restarted_by_rank.items():
        if report.get("backend") != backend or report.get("status") != "success":
            raise AssertionError(f"restarted worker failed: {report}")
        if int(report.get("restart_count", -1)) != 1:
            raise AssertionError(f"restart generation is invalid: {report}")
        if int(report.get("restart_arrival_count", -1)) != 2:
            raise AssertionError(f"restart arrival count is invalid: {report}")
        if int(report.get("max_restarts", -1)) != 1:
            raise AssertionError(f"restart limit is invalid: {report}")
        if int(report.get("world_size", 0)) != 2:
            raise AssertionError(f"restarted world size is invalid: {report}")
        if report.get("optimizer") != "adamw":
            raise AssertionError(f"restarted optimizer metadata is invalid: {report}")
        observed_counts = report.get("observed_local_restart_counts")
        if (
            not isinstance(observed_counts, list)
            or len(observed_counts) != 2
            or max(int(value) for value in observed_counts) != 1
        ):
            raise AssertionError(f"restart counts did not converge: {report}")
        if rank == 1 and int(report.get("local_restart_count", -1)) != 1:
            raise AssertionError("failed rank did not report its local restart")
        if int(report.get("restored_optimizer_steps", -1)) != 1:
            raise AssertionError(f"optimizer step was not restored: {report}")
        if int(report.get("restored_accumulation_micro_step", -1)) != 1:
            raise AssertionError(
                f"accumulation progress was not restored: {report}"
            )
        if int(report.get("completed_optimizer_steps", -1)) != 2:
            raise AssertionError(f"final optimizer step is invalid: {report}")
        if int(report.get("accumulation_micro_step", -1)) != 0:
            raise AssertionError(f"final accumulation state is invalid: {report}")
        if report.get("checkpoint_loaded") is not True:
            raise AssertionError(f"checkpoint was not loaded: {report}")
        before = initial_by_rank[rank]
        saved = before["saved_checkpoint"]
        loaded = report.get("loaded_checkpoint")
        if (
            not isinstance(loaded, dict)
            or saved.get("sha256") != loaded.get("sha256")
            or int(saved.get("bytes", 0)) != int(loaded.get("bytes", -1))
            or saved.get("path") != loaded.get("path")
        ):
            raise AssertionError(
                f"loaded checkpoint differs from saved file: {report}"
            )
        if check_checkpoint_files and not Path(str(loaded.get("path", ""))).is_file():
            raise AssertionError(f"loaded checkpoint is missing: {loaded}")
        if (
            not _nested_allclose(
                report.get("restored_parameters"),
                EXPECTED_STEP_ONE_PARAMETERS,
            )
            or not _nested_allclose(
                report.get("restored_exp_avg"), EXPECTED_STEP_ONE_EXP_AVG
            )
            or not _nested_allclose(
                report.get("restored_exp_avg_sq"),
                EXPECTED_STEP_ONE_EXP_AVG_SQ,
            )
            or not _nested_allclose(
                report.get("restored_pending_gradient"),
                EXPECTED_PENDING_GRADIENTS[rank],
            )
            or not _nested_allclose(
                report.get("gathered_restored_pending_gradients"),
                EXPECTED_GATHERED_PENDING_GRADIENTS,
            )
            or not _nested_allclose(
                report.get("gathered_restored_state"),
                EXPECTED_STEP_ONE_STATE,
            )
            or not _nested_allclose(
                report.get("gradient_after_step_two"),
                EXPECTED_STEP_TWO_GRADIENT,
            )
            or not _nested_allclose(
                report.get("parameters_after_step_two"),
                EXPECTED_STEP_TWO_PARAMETERS,
            )
            or not _nested_allclose(
                report.get("exp_avg_after_step_two"),
                EXPECTED_STEP_TWO_EXP_AVG,
            )
            or not _nested_allclose(
                report.get("exp_avg_sq_after_step_two"),
                EXPECTED_STEP_TWO_EXP_AVG_SQ,
            )
            or not _nested_allclose(
                report.get("gathered_state_after_step_two"),
                EXPECTED_STEP_TWO_STATE,
            )
        ):
            raise AssertionError(f"resumed AdamW state is invalid: {report}")
        if int(report.get("optimizer_state_step", -1)) != 2:
            raise AssertionError(f"final AdamW step is invalid: {report}")
        if abs(float(report.get("learning_rate", -1.0)) - 0.0025) > 1e-12:
            raise AssertionError(f"final learning rate is invalid: {report}")
        if int(report.get("scheduler_last_epoch", -1)) != 2:
            raise AssertionError(f"final scheduler state is invalid: {report}")
        if (
            abs(
                float(report.get("restored_next_random", -1.0))
                - EXPECTED_NEXT_RANDOM[rank]
            )
            > 1e-12
        ):
            raise AssertionError(f"restored RNG sequence is invalid: {report}")
        if int(before["pid"]) == int(report["pid"]):
            raise AssertionError("torchrun did not replace every worker")
        if not report.get("store_events"):
            raise AssertionError("restarted process-group store tracing is empty")
        if require_physical_gpu and not report.get("physical_device_name"):
            raise AssertionError(f"physical GPU metadata is missing: {report}")

    run_ids = {str(item.get("run_id", "")) for item in initial + restarted}
    if len(run_ids) != 1 or not next(iter(run_ids)):
        raise AssertionError(f"elastic run IDs are inconsistent: {run_ids}")
    return {
        "schema_version": "fakegpu.elastic_ddp_training_state_validation.v1",
        "status": "success",
        "backend": backend,
        "world_size": 2,
        "failed_rank": 1,
        "failure_exit_code": EXPECTED_PROCESS_EXIT_CODE,
        "restart_count": 1,
        "completed_optimizer_steps": 2,
        "run_id": next(iter(run_ids)),
        "initial_pids": {
            str(rank): int(initial_by_rank[rank]["pid"]) for rank in (0, 1)
        },
        "restarted_pids": {
            str(rank): int(restarted_by_rank[rank]["pid"])
            for rank in (0, 1)
        },
        "local_restart_counts": {
            str(rank): int(restarted_by_rank[rank]["local_restart_count"])
            for rank in (0, 1)
        },
        "restored_parameters": restarted_by_rank[0]["restored_parameters"],
        "restored_exp_avg": restarted_by_rank[0]["restored_exp_avg"],
        "restored_exp_avg_sq": restarted_by_rank[0]["restored_exp_avg_sq"],
        "parameters_after_step_two": restarted_by_rank[0][
            "parameters_after_step_two"
        ],
        "exp_avg_after_step_two": restarted_by_rank[0][
            "exp_avg_after_step_two"
        ],
        "exp_avg_sq_after_step_two": restarted_by_rank[0][
            "exp_avg_sq_after_step_two"
        ],
        "restored_next_random": {
            str(rank): float(restarted_by_rank[rank]["restored_next_random"])
            for rank in (0, 1)
        },
        "initial_workers": initial,
        "restarted_workers": restarted,
    }
