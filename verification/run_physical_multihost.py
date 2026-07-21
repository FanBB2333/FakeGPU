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
FAULT_WORKER_RELATIVE = Path("verification/remote_nccl_fault_worker.py")
SESSION_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


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
            raise argparse.ArgumentTypeError(
                f"duplicate node field: {normalized}"
            )
        fields[normalized] = raw_value.strip()

    allowed = {"name", "ssh", "repo", "python", "shell"}
    unknown = sorted(set(fields) - allowed)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown node fields: {', '.join(unknown)}"
        )
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
        raise argparse.ArgumentTypeError(
            "node python must be an absolute Linux path"
        )
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
    raise TimeoutError(
        f"coordinator did not become ready at {endpoint}: {last_error}"
    )


def _local_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def _remote_preflight(node: NodeSpec) -> dict[str, Any]:
    code = """
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
    "fault_worker_exists": pathlib.Path("verification/remote_nccl_fault_worker.py").is_file(),
}
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
        "fault_worker_exists",
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


def _run_ddp_case(
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
        report_dir = f"{remote_root}/ddp"
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
    _collect_processes(processes, timeout=timeout, label="Hybrid DDP")

    reports = [
        _read_remote_json(node, f"{remote_root}/ddp/rank_{rank}.json")
        for rank, node in enumerate(nodes)
    ]
    for rank, report in enumerate(reports):
        if report.get("status") != "success":
            raise AssertionError(f"DDP rank {rank} failed: {report}")
        if report.get("gradient") != [[1.5, 3.0]]:
            raise AssertionError(
                f"DDP rank {rank} gradient mismatch: {report.get('gradient')}"
            )
        parameters = report.get("parameters_after_step")
        if not (
            isinstance(parameters, list)
            and len(parameters) == 1
            and abs(float(parameters[0][0]) - 0.85) <= 1e-6
            and abs(float(parameters[0][1]) + 0.3) <= 1e-6
        ):
            raise AssertionError(
                f"DDP rank {rank} parameter mismatch: {parameters}"
            )
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


def _validate_cluster_report(
    path: Path,
    *,
    expect_ddp: bool,
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
    if expect_ddp:
        for collective in ("all_reduce", "all_gather"):
            if report["collectives"][collective]["calls"] <= 0:
                raise AssertionError(f"DDP did not issue {collective}")
        node_pairs = report.get("node_pairs", [])
        if len(node_pairs) != 1 or int(node_pairs[0].get("total_bytes", 0)) <= 0:
            raise AssertionError(f"physical node-pair traffic is empty: {node_pairs}")
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
    if "collective_mismatch" in cases:
        results = [
            item["mismatch_result"] for item in cases["collective_mismatch"]
        ]
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

    cases = args.case or ["ddp", "collective-mismatch", "missing-peer"]
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
    cluster_report = _validate_cluster_report(
        cluster_path,
        expect_ddp="ddp" in cases,
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
            "Run repeatable Hybrid DDP and TCP failure checks on two SSH hosts "
            "that already have the same FakeGPU Git commit."
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
        choices=["ddp", "collective-mismatch", "missing-peer"],
        default=[],
    )
    parser.add_argument(
        "--session",
        default=(
            time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
        ),
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
