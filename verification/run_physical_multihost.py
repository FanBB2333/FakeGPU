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

try:
    from verification.elastic_ddp_training_state_validation import (
        validate_elastic_ddp_training_state_reports,
    )
except ModuleNotFoundError:
    from elastic_ddp_training_state_validation import (  # type: ignore[no-redef]
        validate_elastic_ddp_training_state_reports,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
CLUSTER_CONFIG_RELATIVE = Path("verification/data/cluster_tcp_4r.yaml")
DDP_WORKER_RELATIVE = Path("test/test_ddp_hybrid_numerics.py")
ELASTIC_DDP_WORKER_RELATIVE = Path("verification/elastic_ddp_worker.py")
ELASTIC_DDP_CHECKPOINT_WORKER_RELATIVE = Path(
    "verification/elastic_ddp_checkpoint_worker.py"
)
ELASTIC_DDP_TRAINING_STATE_WORKER_RELATIVE = Path(
    "verification/elastic_ddp_training_state_worker.py"
)
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
EXPECTED_PROCESS_EXIT_CODE = 86


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
    "elastic_ddp_worker_exists": pathlib.Path("verification/elastic_ddp_worker.py").is_file(),
    "elastic_ddp_checkpoint_worker_exists": pathlib.Path("verification/elastic_ddp_checkpoint_worker.py").is_file(),
    "elastic_ddp_training_state_worker_exists": pathlib.Path("verification/elastic_ddp_training_state_worker.py").is_file(),
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
        "elastic_ddp_worker_exists",
        "elastic_ddp_checkpoint_worker_exists",
        "elastic_ddp_training_state_worker_exists",
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
    allowed_returncodes: Sequence[set[int]] | None = None,
) -> list[dict[str, Any]]:
    accepted = (
        list(allowed_returncodes)
        if allowed_returncodes is not None
        else [{0} for _ in processes]
    )
    if len(accepted) != len(processes):
        raise ValueError("allowed_returncodes must match the process count")
    deadline = time.monotonic() + timeout
    outputs: list[dict[str, Any]] = []
    failures: list[str] = []
    try:
        for index, (node, process) in enumerate(processes):
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
            if process.returncode not in accepted[index]:
                failures.append(
                    f"{label} failed on {node.name} with {process.returncode}\n"
                    f"accepted return codes: {sorted(accepted[index])}\n"
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


def _remote_route_source(
    node: NodeSpec,
    *,
    destination_host: str,
    destination_port: int,
) -> str:
    code = f"""
import socket

addresses = socket.getaddrinfo(
    {destination_host!r},
    {destination_port},
    type=socket.SOCK_DGRAM,
)
if not addresses:
    raise RuntimeError("rendezvous host did not resolve")
family, socket_type, protocol, _, address = addresses[0]
with socket.socket(family, socket_type, protocol) as probe:
    probe.connect(address)
    print(probe.getsockname()[0])
""".strip()
    completed = _run_remote(
        node,
        _shell_command(node, [node.python, "-c", code]),
        timeout=20.0,
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"could not determine route source on {node.name}")
    return lines[-1]


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


def _safe_report_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return normalized or "node"


def _validate_elastic_ddp_restart_reports(
    initial: list[dict[str, Any]],
    restarted: list[dict[str, Any]],
    *,
    node_names: list[str],
    failed_node: str,
) -> dict[str, Any]:
    if len(initial) != 2 or len(restarted) != 2:
        raise AssertionError(
            "elastic DDP requires two initial and two restarted worker reports"
        )
    if len(set(node_names)) != 2 or failed_node not in node_names:
        raise AssertionError("elastic DDP node metadata is invalid")

    initial_by_node = {str(item.get("node_name")): item for item in initial}
    restarted_by_node = {str(item.get("node_name")): item for item in restarted}
    if set(initial_by_node) != set(node_names):
        raise AssertionError(f"initial elastic node reports mismatch: {initial}")
    if set(restarted_by_node) != set(node_names):
        raise AssertionError(f"restarted elastic node reports mismatch: {restarted}")

    expected_states = {
        node_name: (
            "expected_process_exit"
            if node_name == failed_node
            else "waiting_for_restart"
        )
        for node_name in node_names
    }
    for node_name, report in initial_by_node.items():
        if report.get("backend") != "nccl":
            raise AssertionError(f"physical elastic backend mismatch: {report}")
        if report.get("status") != expected_states[node_name]:
            raise AssertionError(f"unexpected initial elastic state: {report}")
        if int(report.get("restart_count", -1)) != 0:
            raise AssertionError(f"invalid initial restart count: {report}")
        if int(report.get("restart_arrival_count", -1)) != 1:
            raise AssertionError(f"invalid initial arrival count: {report}")
        if int(report.get("local_restart_count", -1)) != 0:
            raise AssertionError(f"invalid initial local restart count: {report}")
        if report.get("observed_local_restart_counts") != [0, 0]:
            raise AssertionError(f"initial restart generation mismatch: {report}")
        if int(report.get("max_restarts", -1)) != 1:
            raise AssertionError(f"invalid elastic restart limit: {report}")
        if int(report.get("world_size", 0)) != 2:
            raise AssertionError(f"invalid initial elastic world size: {report}")
        if abs(float(report.get("initial_all_reduce_value", 0.0)) - 3.0) > 1e-6:
            raise AssertionError(f"initial elastic all-reduce mismatch: {report}")
        if report.get("initial_process_group_destroyed_before_exit") is not False:
            raise AssertionError(
                f"physical elastic worker cleaned up before failure: {report}"
            )
    failed = initial_by_node[failed_node]
    if (
        failed.get("selected_for_initial_exit") is not True
        or int(failed.get("expected_process_exit_code", -1))
        != EXPECTED_PROCESS_EXIT_CODE
    ):
        raise AssertionError(f"elastic failure marker is invalid: {failed}")
    survivors = [
        report
        for node_name, report in initial_by_node.items()
        if node_name != failed_node
    ]
    if any(report.get("selected_for_initial_exit") is not False for report in survivors):
        raise AssertionError(f"unexpected elastic failure selection: {survivors}")
    if sorted(int(item.get("rank", -1)) for item in initial) != [0, 1]:
        raise AssertionError(f"initial elastic ranks are invalid: {initial}")

    expected_gradient = [[1.5, 3.0]]
    expected_parameters = [[0.85, -0.3]]
    for report in restarted:
        if report.get("backend") != "nccl":
            raise AssertionError(f"restarted elastic backend mismatch: {report}")
        if report.get("status") != "success":
            raise AssertionError(f"restarted elastic worker failed: {report}")
        if int(report.get("restart_count", -1)) != 1:
            raise AssertionError(f"restarted elastic count is invalid: {report}")
        if int(report.get("restart_arrival_count", -1)) != 2:
            raise AssertionError(
                f"restarted elastic arrival count is invalid: {report}"
            )
        observed_counts = report.get("observed_local_restart_counts")
        if (
            not isinstance(observed_counts, list)
            or len(observed_counts) != 2
            or max(int(value) for value in observed_counts) != 1
        ):
            raise AssertionError(f"restarted generation did not converge: {report}")
        if int(report.get("max_restarts", -1)) != 1:
            raise AssertionError(f"restarted elastic limit is invalid: {report}")
        if int(report.get("world_size", 0)) != 2:
            raise AssertionError(f"restarted elastic world size is invalid: {report}")
        if not _nested_allclose(report.get("gradient"), expected_gradient):
            raise AssertionError(f"restarted elastic gradient mismatch: {report}")
        if not _nested_allclose(
            report.get("parameters_after_step"),
            expected_parameters,
        ):
            raise AssertionError(f"restarted elastic parameters mismatch: {report}")
        gathered = report.get("gathered_parameter_vectors")
        if not _nested_allclose(gathered, expected_parameters * 2):
            raise AssertionError(
                f"restarted elastic parameters differ across ranks: {report}"
            )
        if not report.get("physical_device_name"):
            raise AssertionError(f"physical GPU metadata is missing: {report}")
    if sorted(int(item.get("rank", -1)) for item in restarted) != [0, 1]:
        raise AssertionError(f"restarted elastic ranks are invalid: {restarted}")
    if int(restarted_by_node[failed_node].get("local_restart_count", -1)) != 1:
        raise AssertionError(
            "the failed node did not report its local worker restart"
        )
    if any(
        int(initial_by_node[node_name]["pid"])
        == int(restarted_by_node[node_name]["pid"])
        for node_name in node_names
    ):
        raise AssertionError("torchrun did not replace every physical worker")

    run_ids = {
        str(item.get("run_id", "")) for item in initial + restarted
    }
    if len(run_ids) != 1 or not next(iter(run_ids)):
        raise AssertionError(f"elastic run IDs are inconsistent: {run_ids}")
    run_id = next(iter(run_ids))
    return {
        "schema_version": "fakegpu.elastic_ddp_validation.v1",
        "status": "success",
        "backend": "nccl",
        "world_size": 2,
        "failed_node": failed_node,
        "failure_exit_code": EXPECTED_PROCESS_EXIT_CODE,
        "restart_count": 1,
        "run_id": run_id,
        "initial_rank_assignments": {
            node_name: int(initial_by_node[node_name]["rank"])
            for node_name in node_names
        },
        "restarted_rank_assignments": {
            node_name: int(restarted_by_node[node_name]["rank"])
            for node_name in node_names
        },
        "local_restart_counts": {
            node_name: int(
                restarted_by_node[node_name]["local_restart_count"]
            )
            for node_name in node_names
        },
        "initial_pids": {
            node_name: int(initial_by_node[node_name]["pid"])
            for node_name in node_names
        },
        "restarted_pids": {
            node_name: int(restarted_by_node[node_name]["pid"])
            for node_name in node_names
        },
        "gradient": restarted[0]["gradient"],
        "parameters_after_step": restarted[0]["parameters_after_step"],
        "initial_workers": initial,
        "restarted_workers": restarted,
    }


def _validate_elastic_ddp_checkpoint_reports(
    initial: list[dict[str, Any]],
    restarted: list[dict[str, Any]],
    *,
    node_names: list[str],
    failed_node: str,
) -> dict[str, Any]:
    if len(initial) != 2 or len(restarted) != 2:
        raise AssertionError(
            "elastic checkpoint recovery requires two initial and two "
            "restarted worker reports"
        )
    if len(set(node_names)) != 2 or failed_node not in node_names:
        raise AssertionError("elastic checkpoint node metadata is invalid")

    initial_by_node = {str(item.get("node_name")): item for item in initial}
    restarted_by_node = {
        str(item.get("node_name")): item for item in restarted
    }
    if set(initial_by_node) != set(node_names):
        raise AssertionError(
            f"initial checkpoint node reports mismatch: {initial}"
        )
    if set(restarted_by_node) != set(node_names):
        raise AssertionError(
            f"restarted checkpoint node reports mismatch: {restarted}"
        )

    expected_states = {
        node_name: (
            "expected_process_exit"
            if node_name == failed_node
            else "waiting_for_restart"
        )
        for node_name in node_names
    }
    expected_gradient = [[1.5, 3.0]]
    expected_step_one_parameters = [[0.85, -0.3]]
    expected_step_one_momentum = [[1.5, 3.0]]
    expected_step_two_parameters = [[0.565, -0.87]]
    expected_step_two_momentum = [[2.85, 5.7]]
    expected_restored_state = [[0.85, -0.3, 1.5, 3.0]] * 2
    expected_final_state = [[0.565, -0.87, 2.85, 5.7]] * 2

    for node_name, report in initial_by_node.items():
        if report.get("backend") != "nccl":
            raise AssertionError(
                f"physical checkpoint backend mismatch: {report}"
            )
        if report.get("status") != expected_states[node_name]:
            raise AssertionError(
                f"unexpected initial checkpoint state: {report}"
            )
        if int(report.get("restart_count", -1)) != 0:
            raise AssertionError(
                f"invalid initial checkpoint restart count: {report}"
            )
        if int(report.get("restart_arrival_count", -1)) != 1:
            raise AssertionError(
                f"invalid initial checkpoint arrival count: {report}"
            )
        if int(report.get("local_restart_count", -1)) != 0:
            raise AssertionError(
                f"invalid initial local restart count: {report}"
            )
        if report.get("observed_local_restart_counts") != [0, 0]:
            raise AssertionError(
                f"initial checkpoint generation mismatch: {report}"
            )
        if int(report.get("max_restarts", -1)) != 1:
            raise AssertionError(
                f"invalid checkpoint restart limit: {report}"
            )
        if int(report.get("world_size", 0)) != 2:
            raise AssertionError(
                f"invalid initial checkpoint world size: {report}"
            )
        if int(report.get("completed_steps", -1)) != 1:
            raise AssertionError(
                f"checkpoint was saved at an invalid step: {report}"
            )
        if report.get("initial_process_group_destroyed_before_exit") is not False:
            raise AssertionError(
                f"checkpoint worker cleaned up before failure: {report}"
            )
        if (
            not _nested_allclose(
                report.get("gradient_after_step_one"), expected_gradient
            )
            or not _nested_allclose(
                report.get("parameters_after_step_one"),
                expected_step_one_parameters,
            )
            or not _nested_allclose(
                report.get("momentum_after_step_one"),
                expected_step_one_momentum,
            )
            or not _nested_allclose(
                report.get("gathered_state_after_step_one"),
                expected_restored_state,
            )
        ):
            raise AssertionError(
                f"initial checkpoint training state mismatch: {report}"
            )
        saved = report.get("saved_checkpoint")
        if (
            not isinstance(saved, dict)
            or int(saved.get("bytes", 0)) <= 0
            or not saved.get("sha256")
        ):
            raise AssertionError(
                f"saved checkpoint metadata is invalid: {report}"
            )
        if not report.get("physical_device_name"):
            raise AssertionError(f"physical GPU metadata is missing: {report}")

    failed = initial_by_node[failed_node]
    if (
        failed.get("selected_for_initial_exit") is not True
        or int(failed.get("expected_process_exit_code", -1))
        != EXPECTED_PROCESS_EXIT_CODE
    ):
        raise AssertionError(
            f"elastic checkpoint failure marker is invalid: {failed}"
        )
    if sorted(int(item.get("rank", -1)) for item in initial) != [0, 1]:
        raise AssertionError(
            f"initial checkpoint ranks are invalid: {initial}"
        )
    survivors = [
        report
        for node_name, report in initial_by_node.items()
        if node_name != failed_node
    ]
    if any(
        report.get("selected_for_initial_exit") is not False
        for report in survivors
    ):
        raise AssertionError(
            f"unexpected checkpoint failure selection: {survivors}"
        )

    for node_name, report in restarted_by_node.items():
        if report.get("backend") != "nccl" or report.get("status") != "success":
            raise AssertionError(
                f"restarted checkpoint worker failed: {report}"
            )
        if int(report.get("restart_count", -1)) != 1:
            raise AssertionError(
                f"restarted checkpoint generation is invalid: {report}"
            )
        if int(report.get("restart_arrival_count", -1)) != 2:
            raise AssertionError(
                f"restarted checkpoint arrival count is invalid: {report}"
            )
        if int(report.get("max_restarts", -1)) != 1:
            raise AssertionError(
                f"restarted checkpoint limit is invalid: {report}"
            )
        if int(report.get("world_size", 0)) != 2:
            raise AssertionError(
                f"restarted checkpoint world size is invalid: {report}"
            )
        observed_counts = report.get("observed_local_restart_counts")
        if (
            not isinstance(observed_counts, list)
            or len(observed_counts) != 2
            or max(int(value) for value in observed_counts) != 1
        ):
            raise AssertionError(
                f"checkpoint restart generation did not converge: {report}"
            )
        if int(report.get("restored_completed_steps", -1)) != 1:
            raise AssertionError(
                f"checkpoint training step was not restored: {report}"
            )
        if int(report.get("completed_steps", -1)) != 2:
            raise AssertionError(
                f"resumed checkpoint step is invalid: {report}"
            )
        if report.get("checkpoint_loaded") is not True:
            raise AssertionError(f"checkpoint was not loaded: {report}")
        saved = initial_by_node[node_name]["saved_checkpoint"]
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
        if (
            not _nested_allclose(
                report.get("restored_parameters"),
                expected_step_one_parameters,
            )
            or not _nested_allclose(
                report.get("restored_momentum"), expected_step_one_momentum
            )
            or not _nested_allclose(
                report.get("gathered_restored_state"),
                expected_restored_state,
            )
            or not _nested_allclose(
                report.get("gradient_after_step_two"), expected_gradient
            )
            or not _nested_allclose(
                report.get("parameters_after_step_two"),
                expected_step_two_parameters,
            )
            or not _nested_allclose(
                report.get("momentum_after_step_two"),
                expected_step_two_momentum,
            )
            or not _nested_allclose(
                report.get("gathered_state_after_step_two"),
                expected_final_state,
            )
        ):
            raise AssertionError(
                f"resumed checkpoint training state mismatch: {report}"
            )
        if not report.get("physical_device_name"):
            raise AssertionError(f"physical GPU metadata is missing: {report}")

    if sorted(int(item.get("rank", -1)) for item in restarted) != [0, 1]:
        raise AssertionError(
            f"restarted checkpoint ranks are invalid: {restarted}"
        )
    if int(restarted_by_node[failed_node].get("local_restart_count", -1)) != 1:
        raise AssertionError(
            "the failed checkpoint node did not report a local restart"
        )
    if any(
        int(initial_by_node[node_name]["pid"])
        == int(restarted_by_node[node_name]["pid"])
        for node_name in node_names
    ):
        raise AssertionError(
            "torchrun did not replace every checkpoint worker"
        )

    run_ids = {
        str(item.get("run_id", "")) for item in initial + restarted
    }
    if len(run_ids) != 1 or not next(iter(run_ids)):
        raise AssertionError(
            f"elastic checkpoint run IDs are inconsistent: {run_ids}"
        )
    return {
        "schema_version": "fakegpu.elastic_ddp_checkpoint_validation.v1",
        "status": "success",
        "backend": "nccl",
        "world_size": 2,
        "failed_node": failed_node,
        "failure_exit_code": EXPECTED_PROCESS_EXIT_CODE,
        "restart_count": 1,
        "completed_steps": 2,
        "run_id": next(iter(run_ids)),
        "local_restart_counts": {
            node_name: int(
                restarted_by_node[node_name]["local_restart_count"]
            )
            for node_name in node_names
        },
        "initial_pids": {
            node_name: int(initial_by_node[node_name]["pid"])
            for node_name in node_names
        },
        "restarted_pids": {
            node_name: int(restarted_by_node[node_name]["pid"])
            for node_name in node_names
        },
        "restored_parameters": restarted[0]["restored_parameters"],
        "restored_momentum": restarted[0]["restored_momentum"],
        "parameters_after_step_two": restarted[0]["parameters_after_step_two"],
        "momentum_after_step_two": restarted[0]["momentum_after_step_two"],
        "initial_workers": initial,
        "restarted_workers": restarted,
    }


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


def _run_elastic_ddp_restart_case(
    nodes: list[NodeSpec],
    *,
    endpoint: str,
    rendezvous_host: str,
    rendezvous_port: int,
    remote_root: str,
    session: str,
    timeout: float,
) -> dict[str, Any]:
    report_dir = f"{remote_root}/elastic-ddp-restart"
    timeout_ms = min(10_000, max(2_000, round(timeout * 100)))
    process_group_timeout = max(10, min(60, round(timeout / 2)))
    failed_node = nodes[1]
    local_addresses = {
        node.name: _remote_route_source(
            node,
            destination_host=rendezvous_host,
            destination_port=rendezvous_port,
        )
        for node in nodes
    }
    processes: list[tuple[NodeSpec, subprocess.Popen[str]]] = []
    for node_index, node in enumerate(nodes):
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
            "--max-restarts=1",
            "--monitor-interval=0.2",
            "--rdzv-backend=c10d",
            f"--rdzv-endpoint={rendezvous_host}:{rendezvous_port}",
            f"--rdzv-id={session}-elastic-ddp",
            f"--rdzv-conf=is_host={'true' if node_index == 0 else 'false'}",
            f"--local-addr={local_addresses[node.name]}",
            str(ELASTIC_DDP_WORKER_RELATIVE),
            f"--report-dir={report_dir}",
            f"--node-name={node.name}",
            "--backend=nccl",
            "--trace-store",
            f"--timeout-seconds={process_group_timeout}",
            f"--survivor-wait-seconds={process_group_timeout}",
        ]
        if node == failed_node:
            argv.append("--fail-this-node")
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
        label="Hybrid elastic DDP worker restart",
    )

    initial = [
        _read_remote_json(
            node,
            f"{report_dir}/attempt_0_{_safe_report_component(node.name)}_local_0.json",
        )
        for node in nodes
    ]
    restarted = [
        _read_remote_json(
            node,
            f"{report_dir}/attempt_1_{_safe_report_component(node.name)}_local_0.json",
        )
        for node in nodes
    ]
    summary = _validate_elastic_ddp_restart_reports(
        initial,
        restarted,
        node_names=[node.name for node in nodes],
        failed_node=failed_node.name,
    )
    summary["rendezvous_endpoint"] = (
        f"{rendezvous_host}:{rendezvous_port}"
    )
    summary["local_addresses"] = local_addresses
    return summary


def _run_elastic_ddp_checkpoint_case(
    nodes: list[NodeSpec],
    *,
    endpoint: str,
    rendezvous_host: str,
    rendezvous_port: int,
    remote_root: str,
    session: str,
    timeout: float,
) -> dict[str, Any]:
    report_dir = f"{remote_root}/elastic-ddp-checkpoint/reports"
    checkpoint_dir = f"{remote_root}/elastic-ddp-checkpoint/checkpoints"
    timeout_ms = min(10_000, max(2_000, round(timeout * 100)))
    process_group_timeout = max(10, min(60, round(timeout / 2)))
    failed_node = nodes[1]
    local_addresses = {
        node.name: _remote_route_source(
            node,
            destination_host=rendezvous_host,
            destination_port=rendezvous_port,
        )
        for node in nodes
    }
    processes: list[tuple[NodeSpec, subprocess.Popen[str]]] = []
    for node_index, node in enumerate(nodes):
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
            "--max-restarts=1",
            "--monitor-interval=0.2",
            "--rdzv-backend=c10d",
            f"--rdzv-endpoint={rendezvous_host}:{rendezvous_port}",
            f"--rdzv-id={session}-elastic-ddp-checkpoint",
            f"--rdzv-conf=is_host={'true' if node_index == 0 else 'false'}",
            f"--local-addr={local_addresses[node.name]}",
            str(ELASTIC_DDP_CHECKPOINT_WORKER_RELATIVE),
            f"--report-dir={report_dir}",
            f"--checkpoint-dir={checkpoint_dir}",
            f"--node-name={node.name}",
            "--backend=nccl",
            "--trace-store",
            f"--timeout-seconds={process_group_timeout}",
            f"--survivor-wait-seconds={process_group_timeout}",
        ]
        if node == failed_node:
            argv.append("--fail-this-node")
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
        label="Hybrid elastic DDP checkpoint recovery",
    )

    initial = [
        _read_remote_json(
            node,
            f"{report_dir}/attempt_0_"
            f"{_safe_report_component(node.name)}_local_0.json",
        )
        for node in nodes
    ]
    restarted = [
        _read_remote_json(
            node,
            f"{report_dir}/attempt_1_"
            f"{_safe_report_component(node.name)}_local_0.json",
        )
        for node in nodes
    ]
    summary = _validate_elastic_ddp_checkpoint_reports(
        initial,
        restarted,
        node_names=[node.name for node in nodes],
        failed_node=failed_node.name,
    )
    summary["rendezvous_endpoint"] = (
        f"{rendezvous_host}:{rendezvous_port}"
    )
    summary["local_addresses"] = local_addresses
    return summary


def _run_elastic_ddp_training_state_case(
    nodes: list[NodeSpec],
    *,
    endpoint: str,
    rendezvous_host: str,
    rendezvous_port: int,
    remote_root: str,
    session: str,
    timeout: float,
) -> dict[str, Any]:
    report_dir = f"{remote_root}/elastic-ddp-training-state/reports"
    checkpoint_dir = (
        f"{remote_root}/elastic-ddp-training-state/checkpoints"
    )
    timeout_ms = min(10_000, max(2_000, round(timeout * 100)))
    process_group_timeout = max(10, min(60, round(timeout / 2)))
    failed_node = nodes[1]
    local_addresses = {
        node.name: _remote_route_source(
            node,
            destination_host=rendezvous_host,
            destination_port=rendezvous_port,
        )
        for node in nodes
    }
    processes: list[tuple[NodeSpec, subprocess.Popen[str]]] = []
    for node_index, node in enumerate(nodes):
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
            "--max-restarts=1",
            "--monitor-interval=0.2",
            "--rdzv-backend=c10d",
            f"--rdzv-endpoint={rendezvous_host}:{rendezvous_port}",
            f"--rdzv-id={session}-elastic-ddp-training-state",
            f"--rdzv-conf=is_host={'true' if node_index == 0 else 'false'}",
            f"--local-addr={local_addresses[node.name]}",
            str(ELASTIC_DDP_TRAINING_STATE_WORKER_RELATIVE),
            f"--report-dir={report_dir}",
            f"--checkpoint-dir={checkpoint_dir}",
            f"--node-name={node.name}",
            "--backend=nccl",
            "--trace-store",
            "--resume-rank-shift=1",
            f"--timeout-seconds={process_group_timeout}",
            f"--survivor-wait-seconds={process_group_timeout}",
        ]
        if node == failed_node:
            argv.append("--fail-this-node")
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
        label="Hybrid elastic DDP training-state recovery",
    )

    initial = [
        _read_remote_json(
            node,
            f"{report_dir}/attempt_0_"
            f"{_safe_report_component(node.name)}_local_0.json",
        )
        for node in nodes
    ]
    restarted = [
        _read_remote_json(
            node,
            f"{report_dir}/attempt_1_"
            f"{_safe_report_component(node.name)}_local_0.json",
        )
        for node in nodes
    ]
    summary = validate_elastic_ddp_training_state_reports(
        initial,
        restarted,
        backend="nccl",
        require_physical_gpu=True,
    )
    initial_by_node = {
        str(item["node_name"]): item for item in initial
    }
    if initial_by_node[failed_node.name].get("selected_for_initial_exit") is not True:
        raise AssertionError(
            "the configured physical training-state failure node did not exit"
        )
    summary["failed_node"] = failed_node.name
    summary["rank_nodes"] = {
        str(item["rank"]): str(item["node_name"]) for item in restarted
    }
    summary["rendezvous_endpoint"] = (
        f"{rendezvous_host}:{rendezvous_port}"
    )
    summary["local_addresses"] = local_addresses
    return summary


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


def _run_fault_shrink_case(
    nodes: list[NodeSpec],
    *,
    endpoint: str,
    remote_root: str,
    session: str,
    timeout: float,
) -> list[dict[str, Any]]:
    world_size = 4
    failed_rank = 2
    timeout_ms = min(10_000, max(2_000, round(timeout * 100)))
    rank_nodes = [nodes[0], nodes[1], nodes[0], nodes[1]]
    report_dir = f"{remote_root}/fault-shrink"
    processes: list[tuple[NodeSpec, subprocess.Popen[str]]] = []
    for rank, node in enumerate(rank_nodes):
        report_path = f"{report_dir}/rank_{rank}.json"
        env = _distributed_env(
            node=node,
            endpoint=endpoint,
            timeout_ms=timeout_ms,
            hybrid=False,
        )
        env.update(
            {
                "FAKEGPU_NCCL_FAULT_RANK": str(failed_rank),
                "FAKEGPU_NCCL_FAULT_SEQNO": "1",
                "FAKEGPU_NCCL_FAULT_OPERATION": "all_reduce",
            }
        )
        argv = [
            node.python,
            str(FAULT_WORKER_RELATIVE),
            "--case=fault-shrink",
            f"--rank={rank}",
            f"--world-size={world_size}",
            f"--failed-rank={failed_rank}",
            f"--session={session}-fault-shrink",
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

    _collect_processes(
        processes,
        timeout=timeout,
        label="physical rank failure and communicator shrink",
    )
    reports = [
        _read_remote_json(node, f"{report_dir}/rank_{rank}.json")
        for rank, node in enumerate(rank_nodes)
    ]
    if any(report.get("status") != "success" for report in reports):
        raise AssertionError(f"fault-shrink validation failed: {reports}")
    if any(int(report.get("failure_result", 0)) == 0 for report in reports):
        raise AssertionError(f"fault injection did not reach every rank: {reports}")
    if any(int(report.get("async_error", 0)) == 0 for report in reports):
        raise AssertionError(f"fault async error was not persistent: {reports}")
    if reports[failed_rank].get("participated_in_shrink") is not False:
        raise AssertionError("excluded rank unexpectedly participated in shrink")

    survivors = [
        report for report in reports if int(report["rank"]) != failed_rank
    ]
    if [int(report.get("child_rank", -1)) for report in survivors] != [0, 1, 2]:
        raise AssertionError(f"shrunk ranks were not contiguous: {survivors}")
    if any(int(report.get("child_world_size", 0)) != 3 for report in survivors):
        raise AssertionError(f"shrunk world size mismatch: {survivors}")
    if any(
        abs(float(report.get("recovered_all_reduce_value", 0.0)) - 7.0)
        > 1e-6
        for report in survivors
    ):
        raise AssertionError(f"post-shrink All-Reduce mismatch: {survivors}")
    return reports


def _run_process_exit_shrink_case(
    nodes: list[NodeSpec],
    *,
    endpoint: str,
    remote_root: str,
    session: str,
    timeout: float,
) -> list[dict[str, Any]]:
    world_size = 4
    failed_rank = 2
    timeout_ms = min(10_000, max(2_000, round(timeout * 100)))
    rank_nodes = [nodes[0], nodes[1], nodes[0], nodes[1]]
    report_dir = f"{remote_root}/process-exit-shrink"
    processes: list[tuple[NodeSpec, subprocess.Popen[str]]] = []
    for rank, node in enumerate(rank_nodes):
        report_path = f"{report_dir}/rank_{rank}.json"
        argv = [
            node.python,
            str(FAULT_WORKER_RELATIVE),
            "--case=process-exit-shrink",
            f"--rank={rank}",
            f"--world-size={world_size}",
            f"--failed-rank={failed_rank}",
            f"--session={session}-process-exit-shrink",
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
                        env=_distributed_env(
                            node=node,
                            endpoint=endpoint,
                            timeout_ms=timeout_ms,
                            hybrid=False,
                        ),
                        create_dir=report_dir,
                    ),
                ),
            )
        )

    allowed_returncodes = [
        {EXPECTED_PROCESS_EXIT_CODE} if rank == failed_rank else {0}
        for rank in range(world_size)
    ]
    _collect_processes(
        processes,
        timeout=timeout,
        label="physical process exit and communicator shrink",
        allowed_returncodes=allowed_returncodes,
    )
    reports = [
        _read_remote_json(node, f"{report_dir}/rank_{rank}.json")
        for rank, node in enumerate(rank_nodes)
    ]
    exited = reports[failed_rank]
    if (
        exited.get("status") != "expected_process_exit"
        or int(exited.get("process_exit_code", -1)) != EXPECTED_PROCESS_EXIT_CODE
        or exited.get("expected_exit_performed") is not True
        or exited.get("expected_failure_observed") is not False
        or exited.get("participated_in_collective") is not False
        or exited.get("participated_in_shrink") is not False
    ):
        raise AssertionError(f"failed rank exit marker was invalid: {exited}")

    survivors = [
        report for report in reports if int(report["rank"]) != failed_rank
    ]
    if any(report.get("status") != "success" for report in survivors):
        raise AssertionError(f"process-exit survivors failed: {survivors}")
    if any(int(report.get("failure_result", 0)) == 0 for report in survivors):
        raise AssertionError(f"exited rank was not detected: {survivors}")
    if any(
        "timeout" not in str(report.get("failure_last_error", "")).lower()
        for report in survivors
    ):
        raise AssertionError(f"process-exit timeout diagnostic missing: {survivors}")
    if any(int(report.get("async_error", 0)) == 0 for report in survivors):
        raise AssertionError(f"process-exit async error was not persistent: {survivors}")
    if [int(report.get("child_rank", -1)) for report in survivors] != [0, 1, 2]:
        raise AssertionError(f"process-exit shrink ranks were invalid: {survivors}")
    if any(int(report.get("child_world_size", 0)) != 3 for report in survivors):
        raise AssertionError(f"process-exit child world size was invalid: {survivors}")
    if any(
        abs(float(report.get("recovered_all_reduce_value", 0.0)) - 7.0)
        > 1e-6
        for report in survivors
    ):
        raise AssertionError(f"process-exit recovery result mismatch: {survivors}")
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


def _validate_alltoallv_reports(
    reports: list[dict[str, Any]],
    *,
    elements_per_unit: int = 1,
) -> None:
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
        if report.get("elements_per_unit") != elements_per_unit:
            raise AssertionError(f"all-to-all-v rank {rank} element scale mismatch")
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
            expected_send = [value * elements_per_unit for value in expected["send"]]
            expected_recv = [value * elements_per_unit for value in expected["recv"]]
            if variant.get("send_splits") != expected_send:
                raise AssertionError(
                    f"all-to-all-v rank {rank} {name} send split mismatch"
                )
            if variant.get("recv_splits") != expected_recv:
                raise AssertionError(
                    f"all-to-all-v rank {rank} {name} receive split mismatch"
                )
            if variant.get("payload_validated") is not True:
                raise AssertionError(
                    f"all-to-all-v rank {rank} {name} payload mismatch"
                )
            if "received_values" in variant and variant.get(
                "received_values"
            ) != variant.get("expected_values"):
                raise AssertionError(
                    f"all-to-all-v rank {rank} {name} inline payload mismatch"
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
    elements_per_unit: int,
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
            f"--elements-per-unit={elements_per_unit}",
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
    _validate_alltoallv_reports(
        reports,
        elements_per_unit=elements_per_unit,
    )
    return reports


def _validate_cluster_report(
    path: Path,
    *,
    expected_collectives: set[str],
    expect_point_to_point: bool,
    expect_timeout: bool,
    expected_world_size: int = 2,
    expect_recovery: bool = False,
    expect_process_exit: bool = False,
    expect_elastic_restart: bool = False,
    expected_minimum_communicators: int = 0,
) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != "cluster_report.v1":
        raise AssertionError("unexpected cluster report schema")
    cluster = report.get("cluster", {})
    if (
        cluster.get("world_size") != expected_world_size
        or cluster.get("node_count") != 2
    ):
        raise AssertionError(f"unexpected physical topology: {cluster}")
    if cluster.get("coordinator_transport") != "tcp":
        raise AssertionError(f"unexpected coordinator transport: {cluster}")
    minimum_communicators = max(
        expected_minimum_communicators,
        2 if expect_elastic_restart else 0,
    )
    if int(cluster.get("communicators", 0)) < minimum_communicators:
        raise AssertionError(
            "elastic recovery did not initialize new communicators "
            f"({minimum_communicators}): {cluster}"
        )
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
        active_ranks = [
            rank
            for rank in ranks
            if int(rank.get("point_to_point_calls", 0)) > 0
        ]
        if len(active_ranks) < 2:
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
    if expect_recovery or expect_process_exit:
        resilience = report.get("resilience", {})
        expected_events = int(expect_recovery) + int(expect_process_exit)
        if int(resilience.get("failure_count", 0)) < expected_events:
            raise AssertionError("rank failure was not recorded")
        if int(resilience.get("recovery_count", 0)) < expected_events:
            raise AssertionError("communicator recovery was not recorded")
        failures = resilience["failure_events"]
        if expect_recovery:
            injected = [item for item in failures if item.get("source") == "injected"]
            if len(injected) != 1:
                raise AssertionError(f"unexpected injected failures: {injected}")
            failure = injected[0]
            if int(failure.get("global_rank", -1)) != 2:
                raise AssertionError(f"unexpected failed rank: {failure}")
            if failure.get("observed_ranks") != [0, 1, 2, 3]:
                raise AssertionError(f"incomplete failure observers: {failure}")
            if int(failure.get("attempted_payload_bytes", 0)) != 16:
                raise AssertionError(
                    f"unexpected attempted failure payload: {failure}"
                )
        if expect_process_exit:
            timed_out = [
                item
                for item in failures
                if item.get("source") == "collective_timeout"
            ]
            if len(timed_out) != 1:
                raise AssertionError(f"unexpected timeout failures: {timed_out}")
            failure = timed_out[0]
            if int(failure.get("global_rank", -1)) != 2:
                raise AssertionError(f"unexpected absent rank: {failure}")
            if failure.get("observed_ranks") != [0, 1, 3]:
                raise AssertionError(f"unexpected timeout observers: {failure}")
            if int(failure.get("attempted_payload_bytes", 0)) != 12:
                raise AssertionError(
                    f"unexpected timed-out payload: {failure}"
                )
            if failure.get("error_code") != "timeout_waiting_for_collective":
                raise AssertionError(f"unexpected timeout error: {failure}")
        for recovery in resilience["recovery_events"]:
            if recovery.get("excluded_ranks") != [2]:
                raise AssertionError(f"unexpected shrink exclusions: {recovery}")
            if recovery.get("surviving_ranks") != [0, 1, 3]:
                raise AssertionError(f"unexpected surviving ranks: {recovery}")
    return report


def _cluster_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "cluster": report["cluster"],
        "collectives": report["collectives"],
        "point_to_point": report["point_to_point"],
        "node_pairs": report["node_pairs"],
        "ranks": report["ranks"],
        "resilience": report.get(
            "resilience",
            {
                "failure_count": 0,
                "recovery_count": 0,
                "failure_events": [],
                "recovery_events": [],
            },
        ),
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
    if "elastic_ddp_restart" in cases:
        elastic = cases["elastic_ddp_restart"]
        lines.append(
            "- Elastic DDP restart: one worker exited with code "
            f"`{elastic['failure_exit_code']}` while its NCCL communicator "
            "was active; torchrun replaced both workers and the restarted "
            "gradient `[1.5, 3.0]` and parameters `[0.85, -0.30]` matched "
            "on both physical hosts."
        )
    if "elastic_ddp_checkpoint" in cases:
        checkpoint = cases["elastic_ddp_checkpoint"]
        lines.append(
            "- Elastic DDP checkpoint recovery: both workers loaded step `1` "
            "model parameters `[0.85, -0.30]` and SGD momentum `[1.5, 3.0]`; "
            "the resumed step produced parameters `[0.565, -0.87]` and "
            f"momentum `[2.85, 5.7]` after worker exit code "
            f"`{checkpoint['failure_exit_code']}`."
        )
    if "elastic_ddp_training_state" in cases:
        training_state = cases["elastic_ddp_training_state"]
        lines.append(
            "- Elastic DDP training-state recovery: each host replicated a "
            "complete rank-state bundle; AdamW moments, StepLR, rank-local "
            "RNG, a DistributedSampler cursor, and one pending accumulation "
            "micro-step were restored with rank mapping `0 -> 1`, `1 -> 0` "
            "after worker exit code "
            f"`{training_state['failure_exit_code']}`; step 2 produced "
            "parameters `[0.983838, -0.014662]` on both hosts."
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
    if "fault_shrink" in cases:
        survivors = [
            item
            for item in cases["fault_shrink"]
            if item.get("participated_in_shrink")
        ]
        child_ranks = [item["child_rank"] for item in survivors]
        lines.append(
            "- Injected rank failure and recovery: global rank `2` failed on "
            f"All-Reduce; surviving ranks became `{child_ranks}` and the "
            "post-shrink sum was `7.0` on both physical hosts."
        )
    if "process_exit_shrink" in cases:
        survivors = [
            item
            for item in cases["process_exit_shrink"]
            if item.get("participated_in_shrink")
        ]
        child_ranks = [item["child_rank"] for item in survivors]
        lines.append(
            "- Process exit and recovery: global rank `2` exited with code "
            f"`{EXPECTED_PROCESS_EXIT_CODE}` after communicator initialization; "
            f"the survivors inferred its absence from an All-Reduce timeout, "
            f"became child ranks `{child_ranks}`, and recovered sum `7.0`."
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
        "elastic-ddp-restart",
        "elastic-ddp-checkpoint",
        "elastic-ddp-training-state",
        "ddp-options",
        "fsdp",
        "fsdp2",
        "fsdp2-mixed",
        "fsdp2-low-reduce",
        "alltoallv",
        "collective-mismatch",
        "fault-shrink",
        "process-exit-shrink",
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
        if "elastic-ddp-restart" in cases:
            case_reports["elastic_ddp_restart"] = (
                _run_elastic_ddp_restart_case(
                    nodes,
                    endpoint=endpoint,
                    rendezvous_host=args.coordinator_host,
                    rendezvous_port=args.elastic_port,
                    remote_root=remote_root,
                    session=args.session,
                    timeout=args.case_timeout,
                )
            )
        if "elastic-ddp-checkpoint" in cases:
            case_reports["elastic_ddp_checkpoint"] = (
                _run_elastic_ddp_checkpoint_case(
                    nodes,
                    endpoint=endpoint,
                    rendezvous_host=args.coordinator_host,
                    rendezvous_port=args.elastic_checkpoint_port,
                    remote_root=remote_root,
                    session=args.session,
                    timeout=args.case_timeout,
                )
            )
        if "elastic-ddp-training-state" in cases:
            case_reports["elastic_ddp_training_state"] = (
                _run_elastic_ddp_training_state_case(
                    nodes,
                    endpoint=endpoint,
                    rendezvous_host=args.coordinator_host,
                    rendezvous_port=args.elastic_training_state_port,
                    remote_root=remote_root,
                    session=args.session,
                    timeout=args.case_timeout,
                )
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
                elements_per_unit=args.alltoallv_elements_per_unit,
            )
        if "collective-mismatch" in cases:
            case_reports["collective_mismatch"] = _run_mismatch_case(
                nodes,
                endpoint=endpoint,
                remote_root=remote_root,
                session=args.session,
                timeout=args.case_timeout,
            )
        if "fault-shrink" in cases:
            case_reports["fault_shrink"] = _run_fault_shrink_case(
                nodes,
                endpoint=endpoint,
                remote_root=remote_root,
                session=args.session,
                timeout=args.case_timeout,
            )
        if "process-exit-shrink" in cases:
            case_reports["process_exit_shrink"] = (
                _run_process_exit_shrink_case(
                    nodes,
                    endpoint=endpoint,
                    remote_root=remote_root,
                    session=args.session,
                    timeout=args.case_timeout,
                )
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
    if (
        "ddp" in cases
        or "elastic-ddp-restart" in cases
        or "elastic-ddp-checkpoint" in cases
        or "elastic-ddp-training-state" in cases
        or "ddp-options" in cases
        or selected_deepspeed_cases
    ):
        expected_collectives.update({"all_reduce", "all_gather"})
    if selected_deepspeed_pipeline:
        expected_collectives.add("all_gather")
    if (
        "elastic-ddp-checkpoint" in cases
        or "elastic-ddp-training-state" in cases
    ):
        expected_collectives.add("broadcast")
    if "alltoallv" in cases:
        expected_collectives.add("all_to_all")
    if "fault-shrink" in cases or "process-exit-shrink" in cases:
        expected_collectives.add("all_reduce")
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
        expected_world_size=(
            4
            if "fault-shrink" in cases or "process-exit-shrink" in cases
            else 2
        ),
        expect_recovery="fault-shrink" in cases,
        expect_process_exit="process-exit-shrink" in cases,
        expect_elastic_restart=(
            "elastic-ddp-restart" in cases
            or "elastic-ddp-checkpoint" in cases
            or "elastic-ddp-training-state" in cases
        ),
        expected_minimum_communicators=(
            2
            * sum(
                name in cases
                for name in (
                    "elastic-ddp-restart",
                    "elastic-ddp-checkpoint",
                    "elastic-ddp-training-state",
                )
            )
        ),
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
            "Run repeatable Hybrid DDP/FSDP/FSDP2/DeepSpeed Pipeline, elastic "
            "restart/checkpoint/training-state recovery, and TCP failure "
            "checks on two SSH hosts at one Git commit."
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
    parser.add_argument("--elastic-port", type=int, default=29593)
    parser.add_argument("--elastic-checkpoint-port", type=int, default=29594)
    parser.add_argument("--elastic-training-state-port", type=int, default=29595)
    parser.add_argument(
        "--case",
        action="append",
        choices=[
            "ddp",
            "elastic-ddp-restart",
            "elastic-ddp-checkpoint",
            "elastic-ddp-training-state",
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
            "fault-shrink",
            "process-exit-shrink",
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
        "--alltoallv-elements-per-unit",
        type=int,
        default=1,
        help=(
            "scale each nonuniform/sparse all-to-all-v split unit by this "
            "many FP32 elements"
        ),
    )
    parser.add_argument(
        "--allow-head-mismatch",
        action="store_true",
        help="Allow remote repositories to differ from the local Git commit.",
    )
    args = parser.parse_args(argv)

    for port_name in (
        "coordinator_port",
        "master_port",
        "elastic_port",
        "elastic_checkpoint_port",
        "elastic_training_state_port",
    ):
        port = int(getattr(args, port_name))
        if port < 1 or port > 65535:
            parser.error(f"--{port_name.replace('_', '-')} must be within 1..65535")
    if len(
        {
            args.coordinator_port,
            args.master_port,
            args.elastic_port,
            args.elastic_checkpoint_port,
            args.elastic_training_state_port,
        }
    ) != 5:
        parser.error("coordinator and torch rendezvous ports must be distinct")
    if args.startup_timeout <= 0 or args.case_timeout <= 0:
        parser.error("timeouts must be greater than zero")
    if args.timeline_limit < 0 or args.timeline_limit > 1_000_000:
        parser.error("--timeline-limit must be within 0..1000000")
    if not 1 <= args.alltoallv_elements_per_unit <= 4_194_304:
        parser.error("--alltoallv-elements-per-unit must be within 1..4194304")

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
