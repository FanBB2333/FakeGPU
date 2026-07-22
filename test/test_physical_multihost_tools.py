from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import pytest

from verification.run_physical_multihost import (
    NodeSpec,
    _encoded_remote_command,
    _require_matching_deepspeed_pipeline_stack,
    _shell_command,
    _validate_elastic_ddp_checkpoint_reports,
    _validate_elastic_ddp_restart_reports,
    _validate_cluster_report,
    _validate_alltoallv_reports,
    _validate_deepspeed_pipeline_reports,
    _validate_deepspeed_reports,
    _write_markdown,
    parse_node_spec,
)


def test_parse_node_spec_supports_posix_and_wsl() -> None:
    posix = parse_node_spec(
        "name=gpu-a;ssh=gpu-a;repo=/srv/fakegpu;python=/opt/python; shell=posix"
    )
    assert posix == NodeSpec(
        name="gpu-a",
        ssh="gpu-a",
        repo="/srv/fakegpu",
        python="/opt/python",
        shell="posix",
    )

    wsl = parse_node_spec(
        "ssh=user@windows;repo=/home/user/fakegpu;"
        "python=/opt/torch/bin/python;shell=wsl"
    )
    assert wsl.name == "user@windows"
    assert wsl.shell == "wsl"


@pytest.mark.parametrize(
    "value",
    [
        "ssh=gpu-a;repo=relative;python=/opt/python",
        "ssh=gpu-a;repo=/srv/fakegpu;python=python3",
        "ssh=gpu-a;repo=/srv/fakegpu;python=/opt/python;shell=powershell",
        "ssh=gpu-a;repo=/srv/fakegpu",
        "ssh=gpu-a;ssh=gpu-b;repo=/srv/fakegpu;python=/opt/python",
        "ssh=gpu-a;repo=/srv/fakegpu;python=/opt/python;extra=value",
    ],
)
def test_parse_node_spec_rejects_invalid_fields(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_node_spec(value)


def test_encoded_remote_command_roundtrips_shell_text() -> None:
    command = "printf '%s\\n' 'spaces and $shell characters'"
    decoder = _encoded_remote_command(command)
    encoded = decoder.split()[2]
    assert base64.b64decode(encoded).decode("utf-8") == command


def test_shell_command_quotes_environment_and_paths() -> None:
    node = NodeSpec(
        name="gpu",
        ssh="gpu",
        repo="/srv/fake gpu",
        python="/opt/python",
    )
    command = _shell_command(
        node,
        [node.python, "-c", "print('ok')"],
        env={"VALUE": "with spaces"},
        create_dir="/tmp/report dir",
    )
    assert "cd '/srv/fake gpu'" in command
    assert "mkdir -p '/tmp/report dir'" in command
    assert "'VALUE=with spaces'" in command


def test_validate_elastic_ddp_restart_reports_accepts_physical_restart() -> None:
    initial = [
        {
            "node_name": "pro5000",
            "backend": "nccl",
            "status": "waiting_for_restart",
            "rank": 0,
            "world_size": 2,
            "restart_count": 0,
            "restart_arrival_count": 1,
            "local_restart_count": 0,
            "observed_local_restart_counts": [0, 0],
            "max_restarts": 1,
            "pid": 100,
            "run_id": "elastic-test",
            "initial_all_reduce_value": 3.0,
            "selected_for_initial_exit": False,
            "initial_process_group_destroyed_before_exit": False,
        },
        {
            "node_name": "rtx3090ti",
            "backend": "nccl",
            "status": "expected_process_exit",
            "rank": 1,
            "world_size": 2,
            "restart_count": 0,
            "restart_arrival_count": 1,
            "local_restart_count": 0,
            "observed_local_restart_counts": [0, 0],
            "max_restarts": 1,
            "pid": 200,
            "run_id": "elastic-test",
            "initial_all_reduce_value": 3.0,
            "selected_for_initial_exit": True,
            "expected_process_exit_code": 86,
            "initial_process_group_destroyed_before_exit": False,
        },
    ]
    restarted = [
        {
            "node_name": node_name,
            "backend": "nccl",
            "status": "success",
            "rank": rank,
            "world_size": 2,
            "restart_count": 1,
            "restart_arrival_count": 2,
            "local_restart_count": 1 if node_name == "rtx3090ti" else 0,
            "observed_local_restart_counts": [1, 0],
            "max_restarts": 1,
            "pid": pid,
            "run_id": "elastic-test",
            "gradient": [[1.5, 3.0]],
            "parameters_after_step": [[0.85, -0.3]],
            "gathered_parameter_vectors": [
                [0.85, -0.3],
                [0.85, -0.3],
            ],
            "physical_device_name": gpu,
        }
        for node_name, rank, pid, gpu in (
            ("pro5000", 0, 101, "NVIDIA RTX PRO 5000 Blackwell"),
            ("rtx3090ti", 1, 201, "NVIDIA GeForce RTX 3090 Ti"),
        )
    ]

    summary = _validate_elastic_ddp_restart_reports(
        initial,
        restarted,
        node_names=["pro5000", "rtx3090ti"],
        failed_node="rtx3090ti",
    )

    assert summary["status"] == "success"
    assert summary["initial_pids"] == {"pro5000": 100, "rtx3090ti": 200}
    assert summary["restarted_pids"] == {"pro5000": 101, "rtx3090ti": 201}
    assert summary["local_restart_counts"] == {
        "pro5000": 0,
        "rtx3090ti": 1,
    }


def test_validate_elastic_ddp_restart_reports_rejects_reused_worker() -> None:
    initial = [
        {
            "node_name": node_name,
            "backend": "nccl",
            "status": status,
            "rank": rank,
            "world_size": 2,
            "restart_count": 0,
            "restart_arrival_count": 1,
            "local_restart_count": 0,
            "observed_local_restart_counts": [0, 0],
            "max_restarts": 1,
            "pid": rank + 10,
            "run_id": "elastic-test",
            "initial_all_reduce_value": 3.0,
            "selected_for_initial_exit": rank == 1,
            "expected_process_exit_code": 86 if rank == 1 else None,
            "initial_process_group_destroyed_before_exit": False,
        }
        for node_name, rank, status in (
            ("pro5000", 0, "waiting_for_restart"),
            ("rtx3090ti", 1, "expected_process_exit"),
        )
    ]
    restarted = [
        {
            "node_name": node_name,
            "backend": "nccl",
            "status": "success",
            "rank": rank,
            "world_size": 2,
            "restart_count": 1,
            "restart_arrival_count": 2,
            "local_restart_count": 1 if node_name == "rtx3090ti" else 0,
            "observed_local_restart_counts": [1, 0],
            "max_restarts": 1,
            "pid": rank + 10,
            "run_id": "elastic-test",
            "gradient": [[1.5, 3.0]],
            "parameters_after_step": [[0.85, -0.3]],
            "gathered_parameter_vectors": [[0.85, -0.3], [0.85, -0.3]],
            "physical_device_name": "GPU",
        }
        for node_name, rank in (("pro5000", 0), ("rtx3090ti", 1))
    ]

    with pytest.raises(AssertionError, match="did not replace"):
        _validate_elastic_ddp_restart_reports(
            initial,
            restarted,
            node_names=["pro5000", "rtx3090ti"],
            failed_node="rtx3090ti",
        )


def test_validate_elastic_ddp_checkpoint_reports_accepts_optimizer_resume() -> None:
    node_names = ["pro5000", "rtx3090ti"]
    initial = []
    restarted = []
    for rank, node_name in enumerate(node_names):
        checkpoint_path = f"/tmp/checkpoint-{node_name}.pt"
        digest = f"sha256-{node_name}"
        initial.append(
            {
                "node_name": node_name,
                "backend": "nccl",
                "status": (
                    "expected_process_exit"
                    if node_name == "rtx3090ti"
                    else "waiting_for_restart"
                ),
                "rank": rank,
                "world_size": 2,
                "restart_count": 0,
                "restart_arrival_count": 1,
                "local_restart_count": 0,
                "observed_local_restart_counts": [0, 0],
                "max_restarts": 1,
                "completed_steps": 1,
                "pid": 100 + rank,
                "run_id": "checkpoint-test",
                "selected_for_initial_exit": node_name == "rtx3090ti",
                "expected_process_exit_code": (
                    86 if node_name == "rtx3090ti" else None
                ),
                "initial_process_group_destroyed_before_exit": False,
                "gradient_after_step_one": [[1.5, 3.0]],
                "parameters_after_step_one": [[0.85, -0.3]],
                "momentum_after_step_one": [[1.5, 3.0]],
                "gathered_state_after_step_one": [
                    [0.85, -0.3, 1.5, 3.0],
                    [0.85, -0.3, 1.5, 3.0],
                ],
                "saved_checkpoint": {
                    "path": checkpoint_path,
                    "bytes": 2048,
                    "sha256": digest,
                },
                "physical_device_name": (
                    "NVIDIA RTX PRO 5000 Blackwell"
                    if rank == 0
                    else "NVIDIA GeForce RTX 3090 Ti"
                ),
            }
        )
        restarted.append(
            {
                "node_name": node_name,
                "backend": "nccl",
                "status": "success",
                "rank": rank,
                "world_size": 2,
                "restart_count": 1,
                "restart_arrival_count": 2,
                "local_restart_count": (
                    1 if node_name == "rtx3090ti" else 0
                ),
                "observed_local_restart_counts": [0, 1],
                "max_restarts": 1,
                "completed_steps": 2,
                "restored_completed_steps": 1,
                "checkpoint_loaded": True,
                "pid": 200 + rank,
                "run_id": "checkpoint-test",
                "loaded_checkpoint": {
                    "path": checkpoint_path,
                    "bytes": 2048,
                    "sha256": digest,
                },
                "restored_parameters": [[0.85, -0.3]],
                "restored_momentum": [[1.5, 3.0]],
                "gathered_restored_state": [
                    [0.85, -0.3, 1.5, 3.0],
                    [0.85, -0.3, 1.5, 3.0],
                ],
                "gradient_after_step_two": [[1.5, 3.0]],
                "parameters_after_step_two": [[0.565, -0.87]],
                "momentum_after_step_two": [[2.85, 5.7]],
                "gathered_state_after_step_two": [
                    [0.565, -0.87, 2.85, 5.7],
                    [0.565, -0.87, 2.85, 5.7],
                ],
                "physical_device_name": (
                    "NVIDIA RTX PRO 5000 Blackwell"
                    if rank == 0
                    else "NVIDIA GeForce RTX 3090 Ti"
                ),
            }
        )

    summary = _validate_elastic_ddp_checkpoint_reports(
        initial,
        restarted,
        node_names=node_names,
        failed_node="rtx3090ti",
    )

    assert summary["status"] == "success"
    assert summary["completed_steps"] == 2
    assert summary["restored_momentum"] == [[1.5, 3.0]]
    assert summary["momentum_after_step_two"] == [[2.85, 5.7]]
    assert summary["local_restart_counts"] == {
        "pro5000": 0,
        "rtx3090ti": 1,
    }


def test_validate_deepspeed_reports_accepts_two_physical_ranks() -> None:
    reports = [
        {
            "status": "success",
            "effective_zero_stage": 3,
            "micro_step_1_global_steps": 0,
            "micro_step_2_global_steps": 1,
            "parameters_after_micro_steps": [
                [[1.0, 0.0]],
                [[0.775, -0.45]],
            ],
            "gathered_parameters": [
                [0.775, -0.45],
                [0.775, -0.45],
            ],
        }
        for _ in range(2)
    ]

    _validate_deepspeed_reports(
        reports,
        zero_stage=3,
        precision="fp32",
    )


def test_validate_deepspeed_reports_rejects_cross_rank_divergence() -> None:
    reports = [
        {
            "status": "success",
            "effective_zero_stage": 2,
            "micro_step_1_global_steps": 0,
            "micro_step_2_global_steps": 1,
            "parameters_after_micro_steps": [
                [[1.0, 0.0]],
                [[0.775, -0.45]],
            ],
            "gathered_parameters": [
                [0.775, -0.45],
                [0.7, -0.4],
            ],
        }
        for _ in range(2)
    ]

    with pytest.raises(AssertionError, match="inconsistent"):
        _validate_deepspeed_reports(
            reports,
            zero_stage=2,
            precision="fp32",
        )


def test_validate_deepspeed_pipeline_reports_accepts_two_stages() -> None:
    expected_parameters = [
        [0.5, -1.0, -0.5, 0.0],
        [0.5, 0.0],
    ]
    reports = [
        {
            "status": "success",
            "rank": rank,
            "world_size": 2,
            "engine_type": "PipelineEngine",
            "pipe_parallel_size": 2,
            "pipe_stage_id": rank,
            "gradient_accumulation_steps": 1,
            "activation_checkpoint_interval": 0,
            "global_steps": 1,
            "precision": "fp32",
            "p2p_api": "batch_isend_irecv",
            "p2p_process_group": "dedicated",
            "loss": 6.25,
            "all_stage_parameters": expected_parameters,
        }
        for rank in range(2)
    ]

    _validate_deepspeed_pipeline_reports(reports)


def test_validate_deepspeed_pipeline_reports_rejects_stage_divergence() -> None:
    reports = [
        {
            "status": "success",
            "rank": rank,
            "world_size": 2,
            "engine_type": "PipelineEngine",
            "pipe_parallel_size": 2,
            "pipe_stage_id": rank,
            "gradient_accumulation_steps": 1,
            "activation_checkpoint_interval": 0,
            "global_steps": 1,
            "precision": "fp32",
            "p2p_api": "batch_isend_irecv",
            "p2p_process_group": "dedicated",
            "loss": 6.25,
            "all_stage_parameters": [
                [0.5, -1.0, -0.5, 0.0],
                [0.4, 0.1],
            ],
        }
        for rank in range(2)
    ]

    with pytest.raises(AssertionError, match="parameter mismatch"):
        _validate_deepspeed_pipeline_reports(reports)


def test_require_matching_deepspeed_pipeline_stack_accepts_same_versions() -> None:
    nodes = [
        NodeSpec("gpu-a", "gpu-a", "/repo", "/python"),
        NodeSpec("gpu-b", "gpu-b", "/repo", "/python"),
    ]
    payload = {
        "torch_version": "2.8.0+cu128",
        "torch_cuda_version": "12.8",
        "deepspeed_version": "0.15.3",
    }
    _require_matching_deepspeed_pipeline_stack(nodes, [payload, payload])


def test_require_matching_deepspeed_pipeline_stack_rejects_mismatch() -> None:
    nodes = [
        NodeSpec("gpu-a", "gpu-a", "/repo", "/python"),
        NodeSpec("gpu-b", "gpu-b", "/repo", "/python"),
    ]
    preflight = [
        {
            "torch_version": "2.8.0+cu128",
            "torch_cuda_version": "12.8",
            "deepspeed_version": "0.15.3",
        },
        {
            "torch_version": "2.12.1+cu130",
            "torch_cuda_version": "13.0",
            "deepspeed_version": "0.19.2",
        },
    ]
    with pytest.raises(RuntimeError, match="requires matching PyTorch"):
        _require_matching_deepspeed_pipeline_stack(nodes, preflight)


def test_validate_cluster_report_requires_physical_pipeline_p2p(
    tmp_path: Path,
) -> None:
    payload = {
        "schema_version": "cluster_report.v1",
        "cluster": {
            "world_size": 2,
            "node_count": 2,
            "coordinator_transport": "tcp",
        },
        "collectives": {"all_gather": {"calls": 2}},
        "point_to_point": {"operations": 4, "sends": 4, "bytes": 128},
        "ranks": [
            {"point_to_point_calls": 4, "timeouts": 0},
            {"point_to_point_calls": 4, "timeouts": 0},
        ],
        "node_pairs": [
            {
                "total_bytes": 160,
                "point_to_point_operations": 4,
            }
        ],
    }
    path = tmp_path / "cluster-report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    _validate_cluster_report(
        path,
        expected_collectives={"all_gather"},
        expect_point_to_point=True,
        expect_timeout=False,
    )

    payload["point_to_point"] = {"operations": 0, "sends": 0, "bytes": 0}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError, match="point-to-point traffic is empty"):
        _validate_cluster_report(
            path,
            expected_collectives={"all_gather"},
            expect_point_to_point=True,
            expect_timeout=False,
        )


def test_validate_cluster_report_requires_new_elastic_communicator(
    tmp_path: Path,
) -> None:
    payload = {
        "schema_version": "cluster_report.v1",
        "cluster": {
            "world_size": 2,
            "node_count": 2,
            "coordinator_transport": "tcp",
            "communicators": 2,
        },
        "collectives": {},
        "point_to_point": {"operations": 0, "sends": 0, "bytes": 0},
        "ranks": [{"timeouts": 0}, {"timeouts": 0}],
        "node_pairs": [],
    }
    path = tmp_path / "cluster-report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    _validate_cluster_report(
        path,
        expected_collectives=set(),
        expect_point_to_point=False,
        expect_timeout=False,
        expect_elastic_restart=True,
    )

    payload["cluster"]["communicators"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError, match="new communicator"):
        _validate_cluster_report(
            path,
            expected_collectives=set(),
            expect_point_to_point=False,
            expect_timeout=False,
            expect_elastic_restart=True,
        )

    payload["cluster"]["communicators"] = 3
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError, match=r"communicators \(4\)"):
        _validate_cluster_report(
            path,
            expected_collectives=set(),
            expect_point_to_point=False,
            expect_timeout=False,
            expected_minimum_communicators=4,
        )


def test_validate_cluster_report_accepts_rank_failure_recovery(
    tmp_path: Path,
) -> None:
    payload = {
        "schema_version": "cluster_report.v1",
        "cluster": {
            "world_size": 4,
            "node_count": 2,
            "coordinator_transport": "tcp",
        },
        "collectives": {"all_reduce": {"calls": 1}},
        "point_to_point": {"operations": 0, "sends": 0, "bytes": 0},
        "ranks": [{"timeouts": 0} for _ in range(4)],
        "node_pairs": [{"total_bytes": 24}],
        "resilience": {
            "failure_count": 1,
            "recovery_count": 1,
            "failure_events": [
                {
                    "source": "injected",
                    "global_rank": 2,
                    "observed_ranks": [0, 1, 2, 3],
                    "attempted_payload_bytes": 16,
                }
            ],
            "recovery_events": [
                {
                    "excluded_ranks": [2],
                    "surviving_ranks": [0, 1, 3],
                }
            ],
        },
    }
    path = tmp_path / "cluster-report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    _validate_cluster_report(
        path,
        expected_collectives={"all_reduce"},
        expect_point_to_point=False,
        expect_timeout=False,
        expected_world_size=4,
        expect_recovery=True,
    )

    payload["resilience"]["recovery_events"][0]["surviving_ranks"] = [0, 1]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError, match="unexpected surviving ranks"):
        _validate_cluster_report(
            path,
            expected_collectives={"all_reduce"},
            expect_point_to_point=False,
            expect_timeout=False,
            expected_world_size=4,
            expect_recovery=True,
        )


def test_validate_cluster_report_accepts_process_exit_recovery(
    tmp_path: Path,
) -> None:
    payload = {
        "schema_version": "cluster_report.v1",
        "cluster": {
            "world_size": 4,
            "node_count": 2,
            "coordinator_transport": "tcp",
        },
        "collectives": {"all_reduce": {"calls": 1}},
        "point_to_point": {"operations": 0, "sends": 0, "bytes": 0},
        "ranks": [{"timeouts": 1} for _ in range(4)],
        "node_pairs": [{"total_bytes": 24}],
        "resilience": {
            "failure_count": 1,
            "recovery_count": 1,
            "failure_events": [
                {
                    "source": "collective_timeout",
                    "global_rank": 2,
                    "observed_ranks": [0, 1, 3],
                    "attempted_payload_bytes": 12,
                    "error_code": "timeout_waiting_for_collective",
                }
            ],
            "recovery_events": [
                {
                    "excluded_ranks": [2],
                    "surviving_ranks": [0, 1, 3],
                }
            ],
        },
    }
    path = tmp_path / "cluster-report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    _validate_cluster_report(
        path,
        expected_collectives={"all_reduce"},
        expect_point_to_point=False,
        expect_timeout=True,
        expected_world_size=4,
        expect_process_exit=True,
    )

    payload["resilience"]["failure_events"].append(
        dict(payload["resilience"]["failure_events"][0])
    )
    payload["resilience"]["failure_count"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError, match="unexpected timeout failures"):
        _validate_cluster_report(
            path,
            expected_collectives={"all_reduce"},
            expect_point_to_point=False,
            expect_timeout=True,
            expected_world_size=4,
            expect_process_exit=True,
        )


def test_validate_alltoallv_reports_accepts_nonuniform_and_sparse() -> None:
    plans = [
        [
            ("nonuniform", [1, 2], [1, 3], [0.0, 1000.0, 1001.0, 1002.0]),
            ("sparse", [2, 0], [2, 1], [0.0, 1.0, 1000.0]),
        ],
        [
            ("nonuniform", [3, 1], [2, 1], [100.0, 101.0, 1100.0]),
            ("sparse", [1, 2], [0, 2], [1100.0, 1101.0]),
        ],
    ]
    reports = []
    for rank in range(2):
        variants = [
            {
                "name": name,
                "send_splits": sends,
                "recv_splits": receives,
                "received_values": values,
                "expected_values": values,
                "payload_validated": True,
                "operation_seconds": 0.01,
            }
            for name, sends, receives, values in plans[rank]
        ]
        reports.append(
            {
                "status": "success",
                "rank": rank,
                "world_size": 2,
                "elements_per_unit": 1,
                "variants": variants,
            }
        )

    _validate_alltoallv_reports(reports)


def test_physical_report_markdown_contains_node_pair_table(tmp_path: Path) -> None:
    report = {
        "status": "success",
        "session": "test",
        "git_commit": "abc123",
        "coordinator_endpoint": "10.0.0.1:29591",
        "nodes": [
            {
                "name": "gpu-a",
                "gpu_name": "GPU A",
                "compute_capability": [8, 6],
                "torch_version": "2.0",
                "torch_cuda_version": "12.0",
            },
            {
                "name": "gpu-b",
                "gpu_name": "GPU B",
                "compute_capability": [12, 0],
                "torch_version": "2.1",
                "torch_cuda_version": "13.0",
            },
        ],
        "cases": {
            "ddp": [{}, {}],
            "elastic_ddp_restart": {"failure_exit_code": 86},
            "elastic_ddp_checkpoint": {"failure_exit_code": 86},
            "elastic_ddp_training_state": {"failure_exit_code": 86},
            "ddp_options": {},
            "fsdp": [{}, {}],
            "fsdp2": [{}, {}],
            "fsdp2_mixed": {"fp16": [{}, {}], "bf16": [{}, {}]},
            "fsdp2_low_reduce": {
                "fp16": [{}, {}],
                "bf16": [{}, {}],
            },
            "deepspeed_zero2": [{}, {}],
            "deepspeed_zero3": [{}, {}],
            "deepspeed_pipeline": [{}, {}],
            "alltoallv": [{}, {}],
            "collective_mismatch": [
                {"mismatch_result": 5},
                {"mismatch_result": 5},
            ],
            "fault_shrink": [
                {"participated_in_shrink": True, "child_rank": 0},
                {"participated_in_shrink": True, "child_rank": 1},
                {"participated_in_shrink": False},
                {"participated_in_shrink": True, "child_rank": 2},
            ],
            "process_exit_shrink": [
                {"participated_in_shrink": True, "child_rank": 0},
                {"participated_in_shrink": True, "child_rank": 1},
                {
                    "participated_in_shrink": False,
                    "process_exit_code": 86,
                },
                {"participated_in_shrink": True, "child_rank": 2},
            ],
            "missing_peer": {},
        },
        "cluster_summary": {
            "collectives": {
                "all_reduce": {"calls": 2, "bytes": 32},
                "broadcast": {"calls": 0, "bytes": 0},
            },
            "node_pairs": [
                {
                    "node_a": "gpu-a",
                    "node_b": "gpu-b",
                    "total_bytes": 64,
                    "peak_combined_bytes_per_operation": 32,
                    "operations": 2,
                    "collective_operations": 2,
                    "point_to_point_operations": 0,
                }
            ],
        },
    }
    path = tmp_path / "report.md"
    _write_markdown(path, report)
    markdown = path.read_text(encoding="utf-8")
    assert "| `gpu-a` | `gpu-b` | 64 | 32 | 2 | 2 | 0 |" in markdown
    assert "DDP options" in markdown
    assert "Elastic DDP restart" in markdown
    assert "Elastic DDP checkpoint recovery" in markdown
    assert "Elastic DDP training-state recovery" in markdown
    assert "Hybrid FSDP" in markdown
    assert "Hybrid FSDP2" in markdown
    assert "FSDP2 mixed precision" in markdown
    assert "FSDP2 low-precision reduction" in markdown
    assert "Hybrid DeepSpeed" in markdown
    assert "DeepSpeed Pipeline" in markdown
    assert "Physical all-to-all-v" in markdown
    assert "Collective mismatch" in markdown
    assert "Injected rank failure and recovery" in markdown
    assert "Process exit and recovery" in markdown
