from __future__ import annotations

from copy import deepcopy

import pytest

from verification.elastic_ddp_training_state_validation import (
    EXPECTED_GATHERED_PENDING_GRADIENTS,
    EXPECTED_PENDING_GRADIENTS,
    EXPECTED_SAMPLE_IDS,
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


DATA_SAMPLES = {
    0: [
        {"sample_id": 1, "worker_id": 0, "worker_random": 0.11},
        {"sample_id": 3, "worker_id": 1, "worker_random": 0.31},
        {"sample_id": 5, "worker_id": 0, "worker_random": 0.51},
        {"sample_id": 7, "worker_id": 1, "worker_random": 0.71},
    ],
    1: [
        {"sample_id": 2, "worker_id": 0, "worker_random": 0.21},
        {"sample_id": 4, "worker_id": 1, "worker_random": 0.41},
        {"sample_id": 6, "worker_id": 0, "worker_random": 0.61},
        {"sample_id": 8, "worker_id": 1, "worker_random": 0.81},
    ],
}


def _reports() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    initial: list[dict[str, object]] = []
    restarted: list[dict[str, object]] = []
    random_values = {0: 0.6766379475593567, 1: 0.5308855175971985}
    for rank in (0, 1):
        source_rank = 1 - rank
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
                "node_name": "local",
                "local_rank": rank,
                "world_size": 2,
                "restart_count": 0,
                "restart_arrival_count": 1,
                "local_restart_count": 0,
                "observed_local_restart_counts": [0, 0],
                "max_restarts": 1,
                "completed_optimizer_steps": 1,
                "accumulation_micro_step": 1,
                "optimizer": "adamw",
                "dataloader_workers": 2,
                "dataloader_prefetch_factor": 2,
                "dataloader_persistent_workers": True,
                "dataloader_start_method": "spawn",
                "dataloader_worker_pids": [
                    300 + rank * 10,
                    301 + rank * 10,
                ],
                "checkpoint_replication_mode": "all_rank_states_per_host",
                "replicated_rank_state_ids": [0, 1],
                "rank_state_replica_count": 2,
                "resume_rank_shift": 1,
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
                "sampler": "DistributedSampler",
                "sampler_epoch": 0,
                "samples_consumed": 3,
                "consumed_sample_ids": EXPECTED_SAMPLE_IDS[rank][:3],
                "next_sample_id": EXPECTED_SAMPLE_IDS[rank][3],
                "committed_data_samples": DATA_SAMPLES[rank][:3],
                "staged_prefetched_sample": DATA_SAMPLES[rank][3],
                "loader_batches_yielded": 4,
                "loader_seed": 2026072300 + rank,
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
                "node_name": "local",
                "local_rank": rank,
                "world_size": 2,
                "restart_count": 1,
                "restart_arrival_count": 2,
                "local_restart_count": 1,
                "observed_local_restart_counts": [1, 1],
                "max_restarts": 1,
                "optimizer": "adamw",
                "dataloader_workers": 2,
                "dataloader_prefetch_factor": 2,
                "dataloader_persistent_workers": True,
                "dataloader_start_method": "spawn",
                "dataloader_worker_pids": [
                    400 + rank * 10,
                    401 + rank * 10,
                ],
                "checkpoint_replication_mode": "all_rank_states_per_host",
                "checkpoint_storage_owner_rank": rank,
                "replicated_rank_state_ids": [0, 1],
                "rank_state_replica_count": 2,
                "resume_rank_shift": 1,
                "resume_source_rank": source_rank,
                "rank_remapped": True,
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
                "restored_pending_gradient": (
                    EXPECTED_PENDING_GRADIENTS[source_rank]
                ),
                "gathered_restored_pending_gradients": (
                    [
                        EXPECTED_PENDING_GRADIENTS[1][0],
                        EXPECTED_PENDING_GRADIENTS[0][0],
                    ]
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
                "restored_next_random": random_values[source_rank],
                "restored_sampler": "DistributedSampler",
                "restored_sampler_epoch": 0,
                "restored_samples_consumed": 3,
                "restored_consumed_sample_ids": (
                    EXPECTED_SAMPLE_IDS[source_rank][:3]
                ),
                "restored_committed_data_samples": (
                    DATA_SAMPLES[source_rank][:3]
                ),
                "restored_loader_seed": 2026072300 + source_rank,
                "restored_dataloader_workers": 2,
                "restored_prefetch_factor": 2,
                "restored_persistent_workers": True,
                "restored_multiprocessing_context": "spawn",
                "replayed_prefetched_sample": DATA_SAMPLES[source_rank][3],
                "resumed_sample_id": EXPECTED_SAMPLE_IDS[source_rank][3],
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
        "0": 0.5308855175971985,
        "1": 0.6766379475593567,
    }
    assert summary["resume_rank_map"] == {"0": 1, "1": 0}
    assert summary["resumed_sample_ids"] == {"0": 8, "1": 7}
    assert summary["replayed_prefetched_samples"] == {
        "0": DATA_SAMPLES[1][3],
        "1": DATA_SAMPLES[0][3],
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


def test_validate_training_state_reports_rejects_sampler_cursor_drift() -> None:
    initial, restarted = _reports()
    invalid = deepcopy(restarted)
    invalid[0]["resumed_sample_id"] = 7

    with pytest.raises(AssertionError, match="sampler state"):
        validate_elastic_ddp_training_state_reports(
            initial,
            invalid,
            backend="gloo",
            require_physical_gpu=False,
        )


def test_validate_training_state_reports_rejects_duplicate_resume_source() -> None:
    initial, restarted = _reports()
    invalid = deepcopy(restarted)
    invalid[1]["resume_source_rank"] = 1

    with pytest.raises(AssertionError, match="not a permutation"):
        validate_elastic_ddp_training_state_reports(
            initial,
            invalid,
            backend="gloo",
            require_physical_gpu=False,
        )


def test_validate_training_state_reports_rejects_worker_rng_drift() -> None:
    initial, restarted = _reports()
    invalid = deepcopy(restarted)
    replayed = invalid[0]["replayed_prefetched_sample"]
    assert isinstance(replayed, dict)
    replayed["worker_random"] = 0.99

    with pytest.raises(AssertionError, match="DataLoader state"):
        validate_elastic_ddp_training_state_reports(
            initial,
            invalid,
            backend="gloo",
            require_physical_gpu=False,
        )


def test_validate_training_state_reports_accepts_storage_rank_swap() -> None:
    initial, restarted = _reports()
    swapped = deepcopy(restarted)
    for current_rank, storage_rank in ((0, 1), (1, 0)):
        report = swapped[current_rank]
        report["local_rank"] = storage_rank
        report["checkpoint_storage_owner_rank"] = storage_rank
        report["loaded_checkpoint"] = deepcopy(
            initial[storage_rank]["saved_checkpoint"]
        )
        report["local_restart_count"] = 1 if storage_rank == 1 else 0
        report["observed_local_restart_counts"] = [1, 0]

    summary = validate_elastic_ddp_training_state_reports(
        initial,
        swapped,
        backend="gloo",
        require_physical_gpu=False,
    )

    assert summary["storage_rank_transitions"] == {
        "local/local_0": {"initial_rank": 0, "restarted_rank": 1},
        "local/local_1": {"initial_rank": 1, "restarted_rank": 0},
    }
