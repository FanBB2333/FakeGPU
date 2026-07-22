#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import secrets
import shlex
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
CLUSTER_CONFIG_RELATIVE = Path("verification/data/cluster_tcp_2r.yaml")
DDP_WORKER_RELATIVE = Path("test/test_ddp_hybrid_numerics.py")
FSDP_WORKER_RELATIVE = Path("test/test_fsdp_hybrid_numerics.py")
FSDP2_WORKER_RELATIVE = Path("test/test_fsdp2_hybrid_numerics.py")
DEEPSPEED_WORKER_RELATIVE = Path("test/test_deepspeed_hybrid_numerics.py")
DEEPSPEED_PIPELINE_WORKER_RELATIVE = Path("verification/deepspeed_pipeline_worker.py")
FAULT_WORKER_RELATIVE = Path("verification/remote_nccl_fault_worker.py")
ALLTOALLV_WORKER_RELATIVE = Path(
    "verification/remote_nccl_alltoallv_worker.py"
)
SESSION_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
DDP_OPTION_VARIANTS = ("no-sync", "find-unused", "static-graph")
DDP_EXPECTED_RESULTS: dict[str, dict[str, list[list[float]]]] = {
    "basic": {
        "gradient": [[1.5, 3.0]],
        "parameters_after_step": [[0.85, -0.3]],
    },
    "no-sync": {
        "gradient": [[4.5, 9.0]],
        "parameters_after_step": [[0.55, -0.9]],
    },
    "find-unused": {
        "gradient": [[0.5, 1.0], [1.0, 2.0]],
        "parameters_after_step": [[0.95, -0.1], [-0.1, 0.8]],
    },
    "static-graph": {
        "gradient": [[1.5, 3.0]],
        "parameters_after_step": [[0.7, -0.6]],
    },
}
DEEPSPEED_PIPELINE_EXPECTED_FINAL = [
    [0.5, -1.0, -0.5, 0.0],
    [0.5, 0.0],
]


@dataclass(frozen=True)
class NodeSpec:
    name: str
    ssh: str
    repo: str
    python: str
    shell: str = "posix"


def parse_node_spec(value: str) -> NodeSpec:
    fields: dict[str, str] = {}
    for raw_item in value.split(";"):
        item = raw_item.strip()
        if not item:
            continue
        key, separator, raw_value = item.partition("=")
        if separator != "=" or not key.strip() or not raw_value.strip():
            raise argparse.ArgumentTypeError(
                "node fields must use key=value separated by semicolons"
            )
        normalized = key.strip().lower()
        if normalized in fields:
            raise argparse.ArgumentTypeError(f"duplicate node field: {normalized}")
        fields[normalized] = raw_value.strip()

    allowed = {"name", "ssh", "repo", "python", "shell"}
    unknown = sorted(set(fields) - allowed)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown node fields: {', '.join(unknown)}")
    missing = [key for key in ("ssh", "repo", "python") if not fields.get(key)]
    if missing:
        raise argparse.ArgumentTypeError(
            f"node is missing required fields: {', '.join(missing)}"
        )
    shell = fields.get("shell", "posix").lower()
    if shell not in {"posix", "wsl"}:
        raise argparse.ArgumentTypeError("node shell must be posix or wsl")
    if not fields["repo"].startswith("/"):
        raise argparse.ArgumentTypeError("node repo must be an absolute Linux path")
    if not fields["python"].startswith("/"):
        raise argparse.ArgumentTypeError("node python must be an absolute Linux path")
    return NodeSpec(
        name=fields.get("name", fields["ssh"]),
        ssh=fields["ssh"],
        repo=fields["repo"].rstrip("/"),
        python=fields["python"],
        shell=shell,
    )


def _encoded_remote_command(command: str) -> str:
    encoded = base64.b64encode(command.encode("utf-8")).decode("ascii")
    return f"printf %s {encoded} | base64 -d | bash"


def _ssh_argv(node: NodeSpec, command: str) -> list[str]:
    decoder = _encoded_remote_command(command)
    if node.shell == "wsl":
        remote_command = f'wsl.exe -e bash -lc "{decoder}"'
    else:
        remote_command = decoder
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        node.ssh,
        remote_command,
    ]


def _run_remote(
    node: NodeSpec,
    command: str,
    *,
    timeout: float = 30.0,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        _ssh_argv(node, command),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"remote command failed on {node.name} with {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def _start_remote(node: NodeSpec, command: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        _ssh_argv(node, command),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _shell_command(
    node: NodeSpec,
    argv: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    create_dir: str | None = None,
) -> str:
    components = [f"cd {shlex.quote(node.repo)}"]
    if create_dir is not None:
        components.append(f"mkdir -p {shlex.quote(create_dir)}")
    invocation: list[str] = []
    if env:
        invocation.append("env")
        invocation.extend(f"{key}={value}" for key, value in sorted(env.items()))
    invocation.extend(argv)
    components.append("exec " + shlex.join(invocation))
    return " && ".join(components)


def _request(endpoint: str, payload: str, timeout: float = 2.0) -> str:
    host, port_text = endpoint.rsplit(":", 1)
    with socket.create_connection((host, int(port_text)), timeout=timeout) as sock:
        sock.sendall((payload + "\n").encode("utf-8"))
        response = bytearray()
        while not response.endswith(b"\n"):
            chunk = sock.recv(4096)
            if not chunk:
                break
            response.extend(chunk)
    return response.decode("utf-8", errors="replace").strip()


def _wait_for_coordinator(
    endpoint: str,
    process: subprocess.Popen[str],
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(
                f"remote coordinator exited with {process.returncode}\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}"
            )
        try:
            response = _request(endpoint, "PING", timeout=0.5)
            if response.startswith("OK ") and "status=ready" in response:
                return
        except OSError as exc:
            last_error = exc
        time.sleep(0.1)
    raise TimeoutError(f"coordinator did not become ready at {endpoint}: {last_error}")


def _local_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def _remote_preflight(node: NodeSpec) -> dict[str, Any]:
    code = """
import importlib.metadata
import json
import pathlib
import subprocess
import torch

head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
tracked_status = subprocess.check_output(
    ["git", "status", "--porcelain", "--untracked-files=no"], text=True
).strip()
payload = {
    "head": head,
    "tracked_status": tracked_status,
    "python": __import__("sys").executable,
    "torch_version": str(torch.__version__),
    "torch_cuda_version": str(torch.version.cuda),
    "cuda_available": bool(torch.cuda.is_available()),
    "coordinator_exists": pathlib.Path("build/fakegpu-coordinator").is_file(),
    "nccl_exists": pathlib.Path("build/libnccl.so.2").is_file(),
    "ddp_worker_exists": pathlib.Path("test/test_ddp_hybrid_numerics.py").is_file(),
    "fsdp_worker_exists": pathlib.Path("test/test_fsdp_hybrid_numerics.py").is_file(),
    "fsdp2_worker_exists": pathlib.Path("test/test_fsdp2_hybrid_numerics.py").is_file(),
    "deepspeed_worker_exists": pathlib.Path("test/test_deepspeed_hybrid_numerics.py").is_file(),
    "deepspeed_pipeline_worker_exists": pathlib.Path("verification/deepspeed_pipeline_worker.py").is_file(),
    "fault_worker_exists": pathlib.Path("verification/remote_nccl_fault_worker.py").is_file(),
    "alltoallv_worker_exists": pathlib.Path("verification/remote_nccl_alltoallv_worker.py").is_file(),
}
try:
    payload["deepspeed_version"] = importlib.metadata.version("deepspeed")
except importlib.metadata.PackageNotFoundError:
    payload["deepspeed_version"] = None
if payload["cuda_available"]:
    payload["gpu_name"] = torch.cuda.get_device_name(0)
    payload["compute_capability"] = list(torch.cuda.get_device_capability(0))
print(json.dumps(payload, sort_keys=True))
""".strip()
    command = _shell_command(node, [node.python, "-c", code])
    completed = _run_remote(node, command, timeout=45.0)
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"preflight on {node.name} produced no JSON")
    payload = json.loads(lines[-1])
    required = (
        "coordinator_exists",
        "nccl_exists",
        "ddp_worker_exists",
        "fsdp_worker_exists",
        "fsdp2_worker_exists",
        "fault_worker_exists",
        "alltoallv_worker_exists",
        "cuda_available",
    )
    missing = [key for key in required if not payload.get(key)]
    if missing:
        raise RuntimeError(
            f"preflight on {node.name} failed checks: {', '.join(missing)}"
        )
    if payload.get("tracked_status"):
        raise RuntimeError(
            f"remote repository on {node.name} has tracked modifications: "
            f"{payload['tracked_status']}"
        )
    return payload


def _collect_processes(
    processes: list[tuple[NodeSpec, subprocess.Popen[str]]],
    *,
    timeout: float,
    label: str,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    outputs: list[dict[str, Any]] = []
    failures: list[str] = []
    try:
        for node, process in processes:
            remaining = max(0.1, deadline - time.monotonic())
            try:
                stdout, stderr = process.communicate(timeout=remaining)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate(timeout=5)
                failures.append(
                    f"{node.name} exceeded the {label} timeout\n"
                    f"stdout:\n{stdout}\nstderr:\n{stderr}"
                )
                continue
            outputs.append(
                {
                    "node": node,
                    "returncode": process.returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                }
            )
            if process.returncode != 0:
                failures.append(
                    f"{label} failed on {node.name} with {process.returncode}\n"
                    f"stdout:\n{stdout}\nstderr:\n{stderr}"
                )
        if failures:
            raise RuntimeError("\n\n".join(failures))
        return outputs
    finally:
        for _, process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


def _read_remote_text(node: NodeSpec, path: str) -> str:
    completed = _run_remote(
        node,
        f"test -f {shlex.quote(path)} && cat {shlex.quote(path)}",
        timeout=20.0,
    )
    return completed.stdout


def _read_remote_json(node: NodeSpec, path: str) -> dict[str, Any]:
    return json.loads(_read_remote_text(node, path))


def _distributed_env(
    *,
    node: NodeSpec,
    endpoint: str,
    timeout_ms: int,
    hybrid: bool,
) -> dict[str, str]:
    return {
        "FAKEGPU_MODE": "hybrid" if hybrid else "simulate",
        "FAKEGPU_DEVICE_COUNT": "1",
        "FAKEGPU_DIST_MODE": "simulate",
        "FAKEGPU_CLUSTER_CONFIG": f"{node.repo}/{CLUSTER_CONFIG_RELATIVE}",
        "FAKEGPU_COORDINATOR_TRANSPORT": "tcp",
        "FAKEGPU_COORDINATOR_ADDR": endpoint,
        "FAKEGPU_COORDINATOR_TIMEOUT_MS": str(timeout_ms),
        "FAKEGPU_STAGING_FORCE_SOCKET": "1",
        "OMP_NUM_THREADS": "1",
    }


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


def _validate_deepspeed_reports(
    reports: list[dict[str, Any]],
    *,
    zero_stage: int,
    precision: str,
) -> None:
    expected_initial = [[1.0, 0.0]]
    expected_final = [[0.775, -0.45]]
    tolerance = 1e-6 if precision == "fp32" else 1e-2
    for rank, report in enumerate(reports):
        if report.get("status") != "success":
            raise AssertionError(
                f"DeepSpeed ZeRO-{zero_stage} rank {rank} failed: {report}"
            )
        if report.get("effective_zero_stage") != zero_stage:
            raise AssertionError(f"DeepSpeed rank {rank} stage mismatch: {report}")
        if report.get("micro_step_1_global_steps") != 0:
            raise AssertionError(
                f"DeepSpeed rank {rank} stepped before accumulation boundary"
            )
        if report.get("micro_step_2_global_steps") != 1:
            raise AssertionError(f"DeepSpeed rank {rank} missed accumulation boundary")
        parameters = report.get("parameters_after_micro_steps")
        if not isinstance(parameters, list) or len(parameters) != 2:
            raise AssertionError(
                f"DeepSpeed rank {rank} report is incomplete: {report}"
            )
        if not _nested_allclose(
            parameters[0],
            expected_initial,
            tolerance=tolerance,
        ):
            raise AssertionError(
                f"DeepSpeed rank {rank} changed parameters early: {parameters}"
            )
        if not _nested_allclose(
            parameters[1],
            expected_final,
            tolerance=tolerance,
        ):
            raise AssertionError(f"DeepSpeed rank {rank} update mismatch: {parameters}")
        gathered = report.get("gathered_parameters")
        if not isinstance(gathered, list) or len(gathered) != 2:
            raise AssertionError(
                f"DeepSpeed rank {rank} cross-rank result is incomplete"
            )
        if any(
            not _nested_allclose(
                [value],
                expected_final,
                tolerance=tolerance,
            )
            for value in gathered
        ):
            raise AssertionError(
                f"DeepSpeed rank {rank} observed inconsistent parameters"
            )


def _validate_deepspeed_pipeline_reports(
    reports: list[dict[str, Any]],
) -> None:
    if len(reports) != 2:
        raise AssertionError(
            f"expected two DeepSpeed Pipeline reports, got {len(reports)}"
        )
    for rank, report in enumerate(reports):
        if report.get("status") != "success":
            raise AssertionError(f"DeepSpeed Pipeline rank {rank} failed: {report}")
        expected_fields = {
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
        }
        mismatches = {
            field: (report.get(field), expected)
            for field, expected in expected_fields.items()
            if report.get(field) != expected
        }
        if mismatches:
            raise AssertionError(
                f"DeepSpeed Pipeline rank {rank} metadata mismatch: {mismatches}"
            )
        if abs(float(report.get("loss", float("nan"))) - 6.25) > 1e-6:
            raise AssertionError(
                f"DeepSpeed Pipeline rank {rank} loss mismatch: {report.get('loss')}"
            )
        parameters = report.get("all_stage_parameters")
        if not _nested_allclose(
            parameters,
            DEEPSPEED_PIPELINE_EXPECTED_FINAL,
        ):
            raise AssertionError(
                f"DeepSpeed Pipeline rank {rank} parameter mismatch: {parameters}"
            )


def _require_matching_deepspeed_pipeline_stack(
    nodes: list[NodeSpec],
    preflight: list[dict[str, Any]],
) -> None:
    fields = ("torch_version", "torch_cuda_version", "deepspeed_version")
    stacks = {
        tuple(str(payload.get(field)) for field in fields)
        for payload in preflight
    }
    if len(stacks) == 1:
        return
    details = ", ".join(
        f"{node.name}=(torch={payload.get('torch_version')}, "
        f"cuda={payload.get('torch_cuda_version')}, "
        f"deepspeed={payload.get('deepspeed_version')})"
        for node, payload in zip(nodes, preflight)
    )
    raise RuntimeError(
        "DeepSpeed Pipeline requires matching PyTorch, CUDA runtime, and "
        f"DeepSpeed versions on every physical stage; found {details}"
    )


def _run_ddp_case(
    nodes: list[NodeSpec],
    *,
    endpoint: str,
    master_host: str,
    master_port: int,
    remote_root: str,
    timeout: float,
    variant: str = "basic",
    report_name: str = "ddp",
) -> list[dict[str, Any]]:
    if variant not in DDP_EXPECTED_RESULTS:
        raise ValueError(f"unknown DDP variant: {variant}")
    processes: list[tuple[NodeSpec, subprocess.Popen[str]]] = []
    timeout_ms = max(1, round(timeout * 1000))
    for rank, node in enumerate(nodes):
        report_dir = f"{remote_root}/{report_name}"
        env = _distributed_env(
            node=node,
            endpoint=endpoint,
            timeout_ms=timeout_ms,
            hybrid=True,
        )
        env["LD_PRELOAD"] = f"{node.repo}/build/libnccl.so.2"
        argv = [
            node.python,
            "-m",
            "torch.distributed.run",
            "--nnodes=2",
            "--nproc-per-node=1",
            f"--node-rank={rank}",
            f"--master-addr={master_host}",
            f"--master-port={master_port}",
            "--max-restarts=0",
            str(DDP_WORKER_RELATIVE),
            f"--report-dir={report_dir}",
            f"--variant={variant}",
        ]
        processes.append(
            (
                node,
                _start_remote(
                    node,
                    _shell_command(
                        node,
                        argv,
                        env=env,
                        create_dir=report_dir,
                    ),
                ),
            )
        )
    _collect_processes(
        processes,
        timeout=timeout,
        label=f"Hybrid DDP ({variant})",
    )

    reports = [
        _read_remote_json(
            node,
            f"{remote_root}/{report_name}/rank_{rank}.json",
        )
        for rank, node in enumerate(nodes)
    ]
    expected = DDP_EXPECTED_RESULTS[variant]
    for rank, report in enumerate(reports):
        if report.get("status") != "success":
            raise AssertionError(f"DDP {variant} rank {rank} failed: {report}")
        if report.get("variant") != variant:
            raise AssertionError(
                f"DDP rank {rank} reported variant={report.get('variant')!r}"
            )
        for field, expected_value in expected.items():
            if not _nested_allclose(report.get(field), expected_value):
                raise AssertionError(
                    f"DDP {variant} rank {rank} {field} mismatch: "
                    f"{report.get(field)} != {expected_value}"
                )
    return reports


def _run_fsdp_case(
    nodes: list[NodeSpec],
    *,
    endpoint: str,
    master_host: str,
    master_port: int,
    remote_root: str,
    timeout: float,
) -> list[dict[str, Any]]:
    processes: list[tuple[NodeSpec, subprocess.Popen[str]]] = []
    timeout_ms = max(1, round(timeout * 1000))
    for rank, node in enumerate(nodes):
        report_dir = f"{remote_root}/fsdp"
        env = _distributed_env(
            node=node,
            endpoint=endpoint,
            timeout_ms=timeout_ms,
            hybrid=True,
        )
        env["LD_PRELOAD"] = f"{node.repo}/build/libnccl.so.2"
        argv = [
            node.python,
            "-m",
            "torch.distributed.run",
            "--nnodes=2",
            "--nproc-per-node=1",
            f"--node-rank={rank}",
            f"--master-addr={master_host}",
            f"--master-port={master_port}",
            "--max-restarts=0",
            str(FSDP_WORKER_RELATIVE),
            f"--report-dir={report_dir}",
        ]
        processes.append(
            (
                node,
                _start_remote(
                    node,
                    _shell_command(
                        node,
                        argv,
                        env=env,
                        create_dir=report_dir,
                    ),
                ),
            )
        )
    _collect_processes(processes, timeout=timeout, label="Hybrid FSDP")

    reports = [
        _read_remote_json(node, f"{remote_root}/fsdp/rank_{rank}.json")
        for rank, node in enumerate(nodes)
    ]
    expected_local_gradients = ([1.5], [3.0])
    expected_full = [[0.85, -0.3]]
    for rank, report in enumerate(reports):
        if report.get("status") != "success":
            raise AssertionError(f"FSDP rank {rank} failed: {report}")
        if report.get("local_shard_numel") != 1:
            raise AssertionError(f"FSDP rank {rank} was not sharded: {report}")
        local_gradient = report.get("local_shard_gradient")
        if not (
            isinstance(local_gradient, list)
            and len(local_gradient) == 1
            and abs(float(local_gradient[0]) - expected_local_gradients[rank][0])
            <= 1e-6
        ):
            raise AssertionError(
                f"FSDP rank {rank} gradient mismatch: {local_gradient}"
            )
        for field in (
            "full_parameter_after_step",
            "full_state_dict_weight",
            "restored_full_parameter",
        ):
            if not _nested_allclose(report.get(field), expected_full):
                raise AssertionError(
                    f"FSDP rank {rank} {field} mismatch: {report.get(field)}"
                )
    return reports


def _run_fsdp2_case(
    nodes: list[NodeSpec],
    *,
    endpoint: str,
    master_host: str,
    master_port: int,
    remote_root: str,
    timeout: float,
    precision: str,
    reduce_precision: str = "fp32",
) -> list[dict[str, Any]]:
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"unsupported FSDP2 precision: {precision}")
    if reduce_precision not in {"fp32", "parameter"}:
        raise ValueError(f"unsupported FSDP2 reduce precision: {reduce_precision}")
    if precision == "fp32" and reduce_precision == "parameter":
        raise ValueError("FSDP2 parameter reduction requires fp16 or bf16")

    report_name = (
        f"fsdp2-{precision}"
        if reduce_precision == "fp32"
        else f"fsdp2-{precision}-reduce"
    )
    report_dir = f"{remote_root}/{report_name}"
    processes: list[tuple[NodeSpec, subprocess.Popen[str]]] = []
    timeout_ms = max(1, round(timeout * 1000))
    for rank, node in enumerate(nodes):
        env = _distributed_env(
            node=node,
            endpoint=endpoint,
            timeout_ms=timeout_ms,
            hybrid=True,
        )
        env["LD_PRELOAD"] = f"{node.repo}/build/libnccl.so.2"
        argv = [
            node.python,
            "-m",
            "torch.distributed.run",
            "--nnodes=2",
            "--nproc-per-node=1",
            f"--node-rank={rank}",
            f"--master-addr={master_host}",
            f"--master-port={master_port}",
            "--max-restarts=0",
            str(FSDP2_WORKER_RELATIVE),
            f"--report-dir={report_dir}",
            f"--precision={precision}",
            f"--reduce-precision={reduce_precision}",
        ]
        processes.append(
            (
                node,
                _start_remote(
                    node,
                    _shell_command(
                        node,
                        argv,
                        env=env,
                        create_dir=report_dir,
                    ),
                ),
            )
        )
    _collect_processes(
        processes,
        timeout=timeout,
        label=f"Hybrid FSDP2 ({precision})",
    )

    reports = [
        _read_remote_json(node, f"{report_dir}/rank_{rank}.json")
        for rank, node in enumerate(nodes)
    ]
    expected_gradient = [[1.5, 3.0]]
    expected_full = [[0.85, -0.3], [-0.15, 0.7]]
    tolerance = 1e-6 if precision == "fp32" else 2e-2
    for rank, report in enumerate(reports):
        if report.get("status") != "success":
            raise AssertionError(f"FSDP2 {precision} rank {rank} failed: {report}")
        if report.get("precision") != precision:
            raise AssertionError(f"FSDP2 rank {rank} precision mismatch: {report}")
        if report.get("reduce_precision") != reduce_precision:
            raise AssertionError(
                f"FSDP2 rank {rank} reduce precision mismatch: {report}"
            )
        if report.get("parameter_type") != "DTensor":
            raise AssertionError(
                f"FSDP2 rank {rank} did not expose a DTensor: {report}"
            )
        if report.get("local_shard_shape") != [1, 2]:
            raise AssertionError(f"FSDP2 rank {rank} shard shape mismatch: {report}")
        if not _nested_allclose(
            report.get("local_shard_gradient"),
            expected_gradient,
            tolerance=tolerance,
        ):
            raise AssertionError(
                f"FSDP2 rank {rank} gradient mismatch: "
                f"{report.get('local_shard_gradient')}"
            )
        if not _nested_allclose(
            report.get("local_shard_after_step"),
            [expected_full[rank]],
            tolerance=tolerance,
        ):
            raise AssertionError(
                f"FSDP2 rank {rank} local parameter mismatch: "
                f"{report.get('local_shard_after_step')}"
            )
        if not _nested_allclose(
            report.get("full_parameter_after_step"),
            expected_full,
            tolerance=tolerance,
        ):
            raise AssertionError(
                f"FSDP2 rank {rank} full parameter mismatch: "
                f"{report.get('full_parameter_after_step')}"
            )
    return reports


def _run_deepspeed_case(
    nodes: list[NodeSpec],
    *,
    endpoint: str,
    master_host: str,
    master_port: int,
    remote_root: str,
    timeout: float,
    zero_stage: int,
    precision: str,
) -> list[dict[str, Any]]:
    if zero_stage not in {2, 3}:
        raise ValueError(f"unsupported DeepSpeed ZeRO stage: {zero_stage}")
    if precision not in {"fp32", "bf16"}:
        raise ValueError(f"unsupported DeepSpeed precision: {precision}")

    report_dir = f"{remote_root}/deepspeed-zero{zero_stage}-{precision}"
    processes: list[tuple[NodeSpec, subprocess.Popen[str]]] = []
    timeout_ms = max(1, round(timeout * 1000))
    for rank, node in enumerate(nodes):
        env = _distributed_env(
            node=node,
            endpoint=endpoint,
            timeout_ms=timeout_ms,
            hybrid=True,
        )
        env["LD_PRELOAD"] = f"{node.repo}/build/libnccl.so.2"
        if node.shell == "wsl":
            env["DS_IGNORE_CUDA_DETECTION"] = "1"
        argv = [
            node.python,
            "-m",
            "torch.distributed.run",
            "--nnodes=2",
            "--nproc-per-node=1",
            f"--node-rank={rank}",
            f"--master-addr={master_host}",
            f"--master-port={master_port}",
            "--max-restarts=0",
            str(DEEPSPEED_WORKER_RELATIVE),
            f"--report-dir={report_dir}",
            f"--zero-stage={zero_stage}",
            f"--precision={precision}",
        ]
        processes.append(
            (
                node,
                _start_remote(
                    node,
                    _shell_command(
                        node,
                        argv,
                        env=env,
                        create_dir=report_dir,
                    ),
                ),
            )
        )
    _collect_processes(
        processes,
        timeout=timeout,
        label=f"Hybrid DeepSpeed ZeRO-{zero_stage} ({precision})",
    )

    reports = [
        _read_remote_json(node, f"{report_dir}/rank_{rank}.json")
        for rank, node in enumerate(nodes)
    ]
    _validate_deepspeed_reports(
        reports,
        zero_stage=zero_stage,
        precision=precision,
    )
    return reports


def _run_deepspeed_pipeline_case(
    nodes: list[NodeSpec],
    *,
    endpoint: str,
    master_host: str,
    master_port: int,
    remote_root: str,
    timeout: float,
) -> list[dict[str, Any]]:
    report_dir = f"{remote_root}/deepspeed-pipeline"
    processes: list[tuple[NodeSpec, subprocess.Popen[str]]] = []
    timeout_ms = max(1, round(timeout * 1000))
    for rank, node in enumerate(nodes):
        env = _distributed_env(
            node=node,
            endpoint=endpoint,
            timeout_ms=timeout_ms,
            hybrid=True,
        )
        env["LD_PRELOAD"] = f"{node.repo}/build/libnccl.so.2"
        if node.shell == "wsl":
            env["DS_IGNORE_CUDA_DETECTION"] = "1"
        argv = [
            node.python,
            "-m",
            "torch.distributed.run",
            "--nnodes=2",
            "--nproc-per-node=1",
            f"--node-rank={rank}",
            f"--master-addr={master_host}",
            f"--master-port={master_port}",
            "--max-restarts=0",
            str(DEEPSPEED_PIPELINE_WORKER_RELATIVE),
            f"--report-dir={report_dir}",
            "--precision=fp32",
            "--activation-checkpoint-interval=0",
            "--gradient-accumulation-steps=1",
            "--batched-p2p",
        ]
        processes.append(
            (
                node,
                _start_remote(
                    node,
                    _shell_command(
                        node,
                        argv,
                        env=env,
                        create_dir=report_dir,
                    ),
                ),
            )
        )
    _collect_processes(
        processes,
        timeout=timeout,
        label="Hybrid DeepSpeed Pipeline",
    )

    reports = [
        _read_remote_json(node, f"{report_dir}/rank_{rank}.json")
        for rank, node in enumerate(nodes)
    ]
    _validate_deepspeed_pipeline_reports(reports)
    return reports


def _run_mismatch_case(
    nodes: list[NodeSpec],
    *,
    endpoint: str,
    remote_root: str,
    session: str,
    timeout: float,
) -> list[dict[str, Any]]:
    timeout_ms = min(5_000, max(1_000, round(timeout * 100)))
    processes: list[tuple[NodeSpec, subprocess.Popen[str]]] = []
    for rank, node in enumerate(nodes):
        report_path = f"{remote_root}/faults/mismatch-rank-{rank}.json"
        env = _distributed_env(
            node=node,
            endpoint=endpoint,
            timeout_ms=timeout_ms,
            hybrid=False,
        )
        argv = [
            node.python,
            str(FAULT_WORKER_RELATIVE),
            "--case=collective-mismatch",
            f"--rank={rank}",
            "--world-size=2",
            f"--session={session}-mismatch",
            f"--timeout-ms={timeout_ms}",
            f"--nccl-lib={node.repo}/build/libnccl.so.2",
            f"--report={report_path}",
        ]
        processes.append(
            (
                node,
                _start_remote(
                    node,
                    _shell_command(
                        node,
                        argv,
                        env=env,
                        create_dir=f"{remote_root}/faults",
                    ),
                ),
            )
        )
    _collect_processes(processes, timeout=timeout, label="collective mismatch")
    reports = [
        _read_remote_json(
            node,
            f"{remote_root}/faults/mismatch-rank-{rank}.json",
        )
        for rank, node in enumerate(nodes)
    ]
    if any(report.get("status") != "success" for report in reports):
        raise AssertionError(f"collective mismatch validation failed: {reports}")
    if any(int(report.get("async_error", 0)) == 0 for report in reports):
        raise AssertionError(f"collective mismatch was not persistent: {reports}")
    return reports


def _run_missing_peer_case(
    node: NodeSpec,
    *,
    endpoint: str,
    remote_root: str,
    session: str,
    timeout: float,
) -> dict[str, Any]:
    timeout_ms = 750
    report_path = f"{remote_root}/faults/missing-peer-rank-1.json"
    env = _distributed_env(
        node=node,
        endpoint=endpoint,
        timeout_ms=timeout_ms,
        hybrid=False,
    )
    argv = [
        node.python,
        str(FAULT_WORKER_RELATIVE),
        "--case=missing-peer",
        "--rank=1",
        "--world-size=2",
        f"--session={session}-missing-peer",
        f"--timeout-ms={timeout_ms}",
        f"--nccl-lib={node.repo}/build/libnccl.so.2",
        f"--report={report_path}",
    ]
    process = _start_remote(
        node,
        _shell_command(
            node,
            argv,
            env=env,
            create_dir=f"{remote_root}/faults",
        ),
    )
    _collect_processes(
        [(node, process)],
        timeout=timeout,
        label="missing peer",
    )
    report = _read_remote_json(node, report_path)
    if report.get("status") != "success":
        raise AssertionError(f"missing-peer validation failed: {report}")
    if "timeout" not in str(report.get("comm_init_last_error", "")).lower():
        raise AssertionError(f"missing-peer diagnostic was incomplete: {report}")
    return report


def _validate_alltoallv_reports(reports: list[dict[str, Any]]) -> None:
    expected_splits = {
        "nonuniform": [
            {"send": [1, 2], "recv": [1, 3]},
            {"send": [3, 1], "recv": [2, 1]},
        ],
        "sparse": [
            {"send": [2, 0], "recv": [2, 1]},
            {"send": [1, 2], "recv": [0, 2]},
        ],
    }
    if len(reports) != 2:
        raise AssertionError(f"expected two all-to-all-v reports, got {len(reports)}")
    for rank, report in enumerate(reports):
        if report.get("status") != "success":
            raise AssertionError(f"all-to-all-v rank {rank} failed: {report}")
        if report.get("rank") != rank or report.get("world_size") != 2:
            raise AssertionError(f"all-to-all-v rank metadata mismatch: {report}")
        variants = report.get("variants")
        if not isinstance(variants, list) or len(variants) != 2:
            raise AssertionError(f"all-to-all-v rank {rank} variants are incomplete")
        by_name = {variant.get("name"): variant for variant in variants}
        if set(by_name) != set(expected_splits):
            raise AssertionError(
                f"all-to-all-v rank {rank} variant names mismatch: {by_name}"
            )
        for name, plans in expected_splits.items():
            variant = by_name[name]
            expected = plans[rank]
            if variant.get("send_splits") != expected["send"]:
                raise AssertionError(
                    f"all-to-all-v rank {rank} {name} send split mismatch"
                )
            if variant.get("recv_splits") != expected["recv"]:
                raise AssertionError(
                    f"all-to-all-v rank {rank} {name} receive split mismatch"
                )
            if variant.get("received_values") != variant.get("expected_values"):
                raise AssertionError(
                    f"all-to-all-v rank {rank} {name} payload mismatch"
                )
            if float(variant.get("operation_seconds", 0.0)) <= 0:
                raise AssertionError(
                    f"all-to-all-v rank {rank} {name} timing is missing"
                )


def _run_alltoallv_case(
    nodes: list[NodeSpec],
    *,
    endpoint: str,
    remote_root: str,
    session: str,
    timeout: float,
) -> list[dict[str, Any]]:
    report_dir = f"{remote_root}/alltoallv"
    timeout_ms = max(1, round(timeout * 1000))
    processes: list[tuple[NodeSpec, subprocess.Popen[str]]] = []
    for rank, node in enumerate(nodes):
        report_path = f"{report_dir}/rank_{rank}.json"
        env = _distributed_env(
            node=node,
            endpoint=endpoint,
            timeout_ms=timeout_ms,
            hybrid=False,
        )
        argv = [
            node.python,
            str(ALLTOALLV_WORKER_RELATIVE),
            f"--rank={rank}",
            "--world-size=2",
            f"--session={session}-alltoallv",
            f"--timeout-ms={timeout_ms}",
            f"--nccl-lib={node.repo}/build/libnccl.so.2",
            f"--report={report_path}",
        ]
        processes.append(
            (
                node,
                _start_remote(
                    node,
                    _shell_command(
                        node,
                        argv,
                        env=env,
                        create_dir=report_dir,
                    ),
                ),
            )
        )
    _collect_processes(processes, timeout=timeout, label="physical all-to-all-v")
    reports = [
        _read_remote_json(node, f"{report_dir}/rank_{rank}.json")
        for rank, node in enumerate(nodes)
    ]
    _validate_alltoallv_reports(reports)
    return reports


def _validate_cluster_report(
    path: Path,
    *,
    expected_collectives: set[str],
    expect_point_to_point: bool,
    expect_timeout: bool,
) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != "cluster_report.v1":
        raise AssertionError("unexpected cluster report schema")
    cluster = report.get("cluster", {})
    if cluster.get("world_size") != 2 or cluster.get("node_count") != 2:
        raise AssertionError(f"unexpected physical topology: {cluster}")
    if cluster.get("coordinator_transport") != "tcp":
        raise AssertionError(f"unexpected coordinator transport: {cluster}")
    if expected_collectives:
        for collective in sorted(expected_collectives):
            if report["collectives"][collective]["calls"] <= 0:
                raise AssertionError(f"distributed training did not issue {collective}")
        node_pairs = report.get("node_pairs", [])
        if len(node_pairs) != 1 or int(node_pairs[0].get("total_bytes", 0)) <= 0:
            raise AssertionError(f"physical node-pair traffic is empty: {node_pairs}")
    if expect_point_to_point:
        point_to_point = report.get("point_to_point", {})
        for field in ("operations", "sends", "bytes"):
            if int(point_to_point.get(field, 0)) <= 0:
                raise AssertionError(
                    f"physical point-to-point traffic is empty: {point_to_point}"
                )
        ranks = report.get("ranks", [])
        if len(ranks) != 2 or any(
            int(rank.get("point_to_point_calls", 0)) <= 0 for rank in ranks
        ):
            raise AssertionError(f"physical rank P2P accounting is empty: {ranks}")
        node_pairs = report.get("node_pairs", [])
        if (
            len(node_pairs) != 1
            or int(node_pairs[0].get("point_to_point_operations", 0)) <= 0
        ):
            raise AssertionError(
                f"physical node-pair P2P accounting is empty: {node_pairs}"
            )
    if expect_timeout:
        if sum(int(item["timeouts"]) for item in report.get("ranks", [])) <= 0:
            raise AssertionError("missing-peer case did not increment rank timeouts")
    return report


def _cluster_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "cluster": report["cluster"],
        "collectives": report["collectives"],
        "point_to_point": report["point_to_point"],
        "node_pairs": report["node_pairs"],
        "ranks": report["ranks"],
        "operation_timeline": {
            "retained_entries": report["operation_timeline"]["retained_entries"],
            "dropped_entries": report["operation_timeline"]["dropped_entries"],
        },
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# FakeGPU Physical Multi-Host Validation",
        "",
        f"- Status: `{report['status']}`",
        f"- Session: `{report['session']}`",
        f"- Git commit: `{report['git_commit']}`",
        f"- Coordinator: `{report['coordinator_endpoint']}`",
        "",
        "## Nodes",
        "",
        "| Rank | Node | GPU | Compute capability | PyTorch | CUDA |",
        "|---:|---|---|---:|---|---|",
    ]
    for rank, node in enumerate(report["nodes"]):
        capability = ".".join(str(value) for value in node["compute_capability"])
        lines.append(
            f"| {rank} | `{node['name']}` | {node['gpu_name']} | {capability} "
            f"| {node['torch_version']} | {node['torch_cuda_version']} |"
        )

    lines.extend(["", "## Cases", ""])
    cases = report["cases"]
    if "ddp" in cases:
        lines.append(
            "- Hybrid DDP: averaged gradient `[1.5, 3.0]`; updated "
            "parameters `[0.85, -0.30]` on both ranks."
        )
    if "ddp_options" in cases:
        lines.append(
            "- DDP options: `no_sync`, `find_unused_parameters`, "
            "`static_graph`, and gradient bucket views completed with "
            "matching parameters on both ranks."
        )
    if "fsdp" in cases:
        lines.append(
            "- Hybrid FSDP: each rank held one parameter shard; "
            "reduce-scatter gradients, full-parameter reconstruction, and "
            "full-state-dict restoration matched the analytical result."
        )
    if "fsdp2" in cases:
        lines.append(
            "- Hybrid FSDP2: DTensor row shards, averaged gradients, optimizer "
            "updates, and full-parameter reconstruction matched in FP32."
        )
    if "fsdp2_mixed" in cases:
        lines.append(
            "- FSDP2 mixed precision: FP16 and BF16 parameter all-gathers "
            "completed with FP32 reduce-scatter gradients on both nodes."
        )
    if "fsdp2_low_reduce" in cases:
        lines.append(
            "- FSDP2 low-precision reduction: FP16 and BF16 reduce-scatter "
            "matched the analytical averaged gradients across both nodes."
        )
    if "deepspeed_zero2" in cases:
        lines.append(
            "- Hybrid DeepSpeed ZeRO-2: FP32 forward, backward, gradient "
            "accumulation, optimizer updates, and cross-rank consistency "
            "completed with one rank on each physical host."
        )
    if "deepspeed_zero3" in cases:
        lines.append(
            "- Hybrid DeepSpeed ZeRO-3: FP32 parameter partitioning, forward, "
            "backward, gradient accumulation, optimizer updates, and "
            "cross-rank consistency completed with matching DeepSpeed "
            "versions on both physical hosts."
        )
    if "deepspeed_pipeline" in cases:
        lines.append(
            "- Hybrid DeepSpeed Pipeline: one pipeline stage ran on each "
            "physical host; FP32 forward activations, backward gradients, "
            "the optimizer step, and cross-stage parameters matched the "
            "analytical result over TCP P2P communication."
        )
    if "alltoallv" in cases:
        lines.append(
            "- Physical all-to-all-v: nonuniform and sparse split plans, "
            "including a zero-sized cross-host direction, transferred the "
            "expected FP32 payloads over the TCP coordinator."
        )
    if "collective_mismatch" in cases:
        results = [item["mismatch_result"] for item in cases["collective_mismatch"]]
        lines.append(
            f"- Collective mismatch: both ranks returned `{results}` and "
            "the async error remained visible."
        )
    if "missing_peer" in cases:
        lines.append(
            "- Missing peer: communicator initialization returned a timeout "
            "diagnostic within the configured limit."
        )

    cluster = report["cluster_summary"]
    lines.extend(
        [
            "",
            "## Communication Summary",
            "",
            "| Operation | Calls | Logical bytes |",
            "|---|---:|---:|",
        ]
    )
    for name, item in cluster["collectives"].items():
        if int(item["calls"]) > 0:
            lines.append(f"| `{name}` | {item['calls']} | {item['bytes']} |")

    lines.extend(
        [
            "",
            "## Node-Pair Communication",
            "",
            "| Node A | Node B | Total bytes | Peak bytes/op | Operations | "
            "Collective ops | P2P ops |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for pair in cluster["node_pairs"]:
        lines.append(
            f"| `{pair['node_a']}` | `{pair['node_b']}` "
            f"| {pair['total_bytes']} "
            f"| {pair['peak_combined_bytes_per_operation']} "
            f"| {pair['operations']} "
            f"| {pair['collective_operations']} "
            f"| {pair['point_to_point_operations']} |"
        )
    lines.extend(
        [
            "",
            "The communication table is coordinator accounting. Throughput and "
            "modeled time in the cluster report are not raw NIC/NCCL counters.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    nodes: list[NodeSpec] = args.node
    if len(nodes) != 2:
        raise ValueError("physical validation currently requires exactly two nodes")
    if len({node.name for node in nodes}) != len(nodes):
        raise ValueError("node names must be unique")
    if not SESSION_PATTERN.fullmatch(args.session):
        raise ValueError(
            "session may contain only letters, digits, underscore, dot, and dash"
        )

    cases = args.case or [
        "ddp",
        "ddp-options",
        "fsdp",
        "fsdp2",
        "fsdp2-mixed",
        "fsdp2-low-reduce",
        "alltoallv",
        "collective-mismatch",
        "missing-peer",
    ]
    git_commit = _local_head()
    preflight = [_remote_preflight(node) for node in nodes]
    if not args.allow_head_mismatch:
        mismatches = [
            f"{node.name}={payload['head']}"
            for node, payload in zip(nodes, preflight)
            if payload["head"] != git_commit
        ]
        if mismatches:
            raise RuntimeError(
                f"remote Git commits differ from local {git_commit}: "
                + ", ".join(mismatches)
            )
    deepspeed_cases = {
        "deepspeed-zero2": 2,
        "deepspeed-zero3": 3,
    }
    selected_deepspeed_cases = {
        name: zero_stage
        for name, zero_stage in deepspeed_cases.items()
        if name in cases
    }
    selected_deepspeed_pipeline = "deepspeed-pipeline" in cases
    if selected_deepspeed_cases or selected_deepspeed_pipeline:
        unavailable = [
            node.name
            for node, payload in zip(nodes, preflight)
            if not payload.get("deepspeed_version")
            or (
                bool(selected_deepspeed_cases)
                and not payload.get("deepspeed_worker_exists")
            )
            or (
                selected_deepspeed_pipeline
                and not payload.get("deepspeed_pipeline_worker_exists")
            )
        ]
        if unavailable:
            raise RuntimeError(
                "DeepSpeed validation is unavailable on: " + ", ".join(unavailable)
            )
    if "deepspeed-zero3" in selected_deepspeed_cases:
        versions = {str(payload["deepspeed_version"]) for payload in preflight}
        if len(versions) != 1:
            details = ", ".join(
                f"{node.name}={payload['deepspeed_version']}"
                for node, payload in zip(nodes, preflight)
            )
            raise RuntimeError(
                "DeepSpeed ZeRO-3 requires the same DeepSpeed version on "
                f"every physical rank; found {details}"
            )
    if selected_deepspeed_pipeline:
        _require_matching_deepspeed_pipeline_stack(nodes, preflight)
    remote_root = f"/tmp/fakegpu-physical-{args.session}"
    endpoint = f"{args.coordinator_host}:{args.coordinator_port}"
    coordinator_node = nodes[0]
    remote_cluster_path = f"{remote_root}/cluster-report.json"
    remote_markdown_path = f"{remote_root}/cluster-report.md"
    coordinator_argv = [
        coordinator_node.python,
        "-m",
        "fakegpu",
        "coordinator",
        "--listen",
        f"0.0.0.0:{args.coordinator_port}",
        "--cluster-config",
        f"{coordinator_node.repo}/{CLUSTER_CONFIG_RELATIVE}",
        "--report",
        remote_cluster_path,
        "--markdown-report",
        remote_markdown_path,
        "--build-dir",
        f"{coordinator_node.repo}/build",
    ]
    coordinator_env = {
        "FAKEGPU_CLUSTER_REPORT_MAX_OPERATIONS": str(args.timeline_limit),
    }
    coordinator = _start_remote(
        coordinator_node,
        _shell_command(
            coordinator_node,
            coordinator_argv,
            env=coordinator_env,
            create_dir=remote_root,
        ),
    )
    stopped = False
    case_reports: dict[str, Any] = {}
    try:
        _wait_for_coordinator(endpoint, coordinator, args.startup_timeout)
        if "ddp" in cases:
            case_reports["ddp"] = _run_ddp_case(
                nodes,
                endpoint=endpoint,
                master_host=args.coordinator_host,
                master_port=args.master_port,
                remote_root=remote_root,
                timeout=args.case_timeout,
            )
        if "ddp-options" in cases:
            case_reports["ddp_options"] = {
                variant: _run_ddp_case(
                    nodes,
                    endpoint=endpoint,
                    master_host=args.coordinator_host,
                    master_port=args.master_port,
                    remote_root=remote_root,
                    timeout=args.case_timeout,
                    variant=variant,
                    report_name=f"ddp-options-{variant}",
                )
                for variant in DDP_OPTION_VARIANTS
            }
        if "fsdp" in cases:
            case_reports["fsdp"] = _run_fsdp_case(
                nodes,
                endpoint=endpoint,
                master_host=args.coordinator_host,
                master_port=args.master_port,
                remote_root=remote_root,
                timeout=args.case_timeout,
            )
        if "fsdp2" in cases:
            case_reports["fsdp2"] = _run_fsdp2_case(
                nodes,
                endpoint=endpoint,
                master_host=args.coordinator_host,
                master_port=args.master_port,
                remote_root=remote_root,
                timeout=args.case_timeout,
                precision="fp32",
            )
        if "fsdp2-mixed" in cases:
            case_reports["fsdp2_mixed"] = {
                precision: _run_fsdp2_case(
                    nodes,
                    endpoint=endpoint,
                    master_host=args.coordinator_host,
                    master_port=args.master_port,
                    remote_root=remote_root,
                    timeout=args.case_timeout,
                    precision=precision,
                )
                for precision in ("fp16", "bf16")
            }
        if "fsdp2-low-reduce" in cases:
            case_reports["fsdp2_low_reduce"] = {
                precision: _run_fsdp2_case(
                    nodes,
                    endpoint=endpoint,
                    master_host=args.coordinator_host,
                    master_port=args.master_port,
                    remote_root=remote_root,
                    timeout=args.case_timeout,
                    precision=precision,
                    reduce_precision="parameter",
                )
                for precision in ("fp16", "bf16")
            }
        for case_name, zero_stage in selected_deepspeed_cases.items():
            report_name = case_name.replace("-", "_")
            case_reports[report_name] = _run_deepspeed_case(
                nodes,
                endpoint=endpoint,
                master_host=args.coordinator_host,
                master_port=args.master_port,
                remote_root=remote_root,
                timeout=args.case_timeout,
                zero_stage=zero_stage,
                precision="fp32",
            )
        if selected_deepspeed_pipeline:
            case_reports["deepspeed_pipeline"] = _run_deepspeed_pipeline_case(
                nodes,
                endpoint=endpoint,
                master_host=args.coordinator_host,
                master_port=args.master_port,
                remote_root=remote_root,
                timeout=args.case_timeout,
            )
        if "alltoallv" in cases:
            case_reports["alltoallv"] = _run_alltoallv_case(
                nodes,
                endpoint=endpoint,
                remote_root=remote_root,
                session=args.session,
                timeout=args.case_timeout,
            )
        if "collective-mismatch" in cases:
            case_reports["collective_mismatch"] = _run_mismatch_case(
                nodes,
                endpoint=endpoint,
                remote_root=remote_root,
                session=args.session,
                timeout=args.case_timeout,
            )
        if "missing-peer" in cases:
            case_reports["missing_peer"] = _run_missing_peer_case(
                nodes[1],
                endpoint=endpoint,
                remote_root=remote_root,
                session=args.session,
                timeout=args.case_timeout,
            )

        response = _request(endpoint, "SHUTDOWN", timeout=3.0)
        if not response.startswith("OK "):
            raise RuntimeError(f"coordinator rejected shutdown: {response}")
        stopped = True
        stdout, stderr = coordinator.communicate(timeout=10)
        if coordinator.returncode != 0:
            raise RuntimeError(
                f"remote coordinator exited with {coordinator.returncode}\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}"
            )
    finally:
        if coordinator.poll() is None:
            if not stopped:
                try:
                    _request(endpoint, "SHUTDOWN", timeout=1.0)
                    coordinator.wait(timeout=5)
                except Exception:
                    coordinator.kill()
                    coordinator.wait(timeout=5)

    output_dir = args.output_dir.resolve() / args.session
    output_dir.mkdir(parents=True, exist_ok=True)
    cluster_path = output_dir / "cluster-report.json"
    cluster_path.write_text(
        _read_remote_text(coordinator_node, remote_cluster_path),
        encoding="utf-8",
    )
    (output_dir / "cluster-report.md").write_text(
        _read_remote_text(coordinator_node, remote_markdown_path),
        encoding="utf-8",
    )
    expected_collectives: set[str] = set()
    if "ddp" in cases or "ddp-options" in cases or selected_deepspeed_cases:
        expected_collectives.update({"all_reduce", "all_gather"})
    if selected_deepspeed_pipeline:
        expected_collectives.add("all_gather")
    if "alltoallv" in cases:
        expected_collectives.add("all_to_all")
    if (
        "fsdp" in cases
        or "fsdp2" in cases
        or "fsdp2-mixed" in cases
        or "fsdp2-low-reduce" in cases
    ):
        expected_collectives.update({"all_gather", "reduce_scatter"})
    cluster_report = _validate_cluster_report(
        cluster_path,
        expected_collectives=expected_collectives,
        expect_point_to_point=selected_deepspeed_pipeline,
        expect_timeout="missing-peer" in cases,
    )

    report: dict[str, Any] = {
        "schema_version": "fakegpu.physical_multihost_validation.v1",
        "status": "success",
        "session": args.session,
        "git_commit": git_commit,
        "coordinator_endpoint": endpoint,
        "nodes": [
            {
                "name": node.name,
                "ssh_target": node.ssh,
                "shell": node.shell,
                "head": payload["head"],
                "python": payload["python"],
                "torch_version": payload["torch_version"],
                "torch_cuda_version": payload["torch_cuda_version"],
                "gpu_name": payload["gpu_name"],
                "compute_capability": payload["compute_capability"],
                "deepspeed_version": payload.get("deepspeed_version"),
            }
            for node, payload in zip(nodes, preflight)
        ],
        "cases": case_reports,
        "cluster_report": str(cluster_path),
        "cluster_markdown_report": str(output_dir / "cluster-report.md"),
        "cluster_summary": _cluster_summary(cluster_report),
    }
    report_path = output_dir / "physical-multihost-validation.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path = output_dir / "physical-multihost-validation.md"
    _write_markdown(markdown_path, report)
    report["report_path"] = str(report_path)
    report["markdown_path"] = str(markdown_path)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeatable Hybrid DDP/FSDP/FSDP2/DeepSpeed Pipeline and TCP "
            "failure checks on two SSH hosts with the same FakeGPU Git commit."
        )
    )
    parser.add_argument(
        "--node",
        type=parse_node_spec,
        action="append",
        required=True,
        help=(
            "Rank-ordered node spec: "
            "name=NAME;ssh=TARGET;repo=/path;python=/path;shell=posix|wsl"
        ),
    )
    parser.add_argument("--coordinator-host", required=True)
    parser.add_argument("--coordinator-port", type=int, default=29591)
    parser.add_argument("--master-port", type=int, default=29592)
    parser.add_argument(
        "--case",
        action="append",
        choices=[
            "ddp",
            "ddp-options",
            "fsdp",
            "fsdp2",
            "fsdp2-mixed",
            "fsdp2-low-reduce",
            "deepspeed-zero2",
            "deepspeed-zero3",
            "deepspeed-pipeline",
            "alltoallv",
            "collective-mismatch",
            "missing-peer",
        ],
        default=[],
    )
    parser.add_argument(
        "--session",
        default=(time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "build" / "physical_multihost_validation",
    )
    parser.add_argument("--startup-timeout", type=float, default=20.0)
    parser.add_argument("--case-timeout", type=float, default=120.0)
    parser.add_argument("--timeline-limit", type=int, default=4096)
    parser.add_argument(
        "--allow-head-mismatch",
        action="store_true",
        help="Allow remote repositories to differ from the local Git commit.",
    )
    args = parser.parse_args(argv)

    for port_name in ("coordinator_port", "master_port"):
        port = int(getattr(args, port_name))
        if port < 1 or port > 65535:
            parser.error(f"--{port_name.replace('_', '-')} must be within 1..65535")
    if args.coordinator_port == args.master_port:
        parser.error("coordinator and torch rendezvous ports must differ")
    if args.startup_timeout <= 0 or args.case_timeout <= 0:
        parser.error("timeouts must be greater than zero")
    if args.timeline_limit < 0 or args.timeline_limit > 1_000_000:
        parser.error("--timeline-limit must be within 0..1000000")

    try:
        report = _run(args)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "session": report["session"],
                    "git_commit": report["git_commit"],
                    "report": report["report_path"],
                    "markdown": report["markdown_path"],
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        print(
            f"physical multi-host validation failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        if os.environ.get("FAKEGPU_VALIDATION_TRACEBACK") == "1":
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
