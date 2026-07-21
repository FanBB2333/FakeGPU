#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPO_ROOT / "test" / "test_fsdp2_hybrid_numerics.py"
CLUSTER_CONFIGS = {
    2: REPO_ROOT / "verification" / "data" / "cluster_hybrid_2r.yaml",
    4: REPO_ROOT / "verification" / "data" / "cluster_hybrid_4r.yaml",
}


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _shutdown_unix_coordinator(socket_path: Path) -> None:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(2.0)
        sock.connect(str(socket_path))
        sock.sendall(b"SHUTDOWN\n")
        sock.recv(4096)


def _wait_for_unix_socket(
    socket_path: Path,
    process: subprocess.Popen[str],
) -> None:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if socket_path.exists():
            return
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(
                f"coordinator exited with code {process.returncode}\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}"
            )
        time.sleep(0.05)
    raise TimeoutError(f"coordinator socket was not created: {socket_path}")


def _close_coordinator(
    coordinator: subprocess.Popen[str],
    socket_path: Path,
) -> None:
    if coordinator.poll() is not None:
        return
    try:
        _shutdown_unix_coordinator(socket_path)
        coordinator.wait(timeout=3)
    except Exception:
        coordinator.kill()
        coordinator.wait(timeout=5)


def _load_rank_reports(
    report_dir: Path,
    world_size: int,
) -> list[dict[str, Any]]:
    reports = []
    for rank in range(world_size):
        path = report_dir / f"rank_{rank}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing rank report: {path}")
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    return reports


def _matrix_close(
    actual: Any,
    expected: list[list[float]],
    tolerance: float,
) -> bool:
    if not isinstance(actual, list) or len(actual) != len(expected):
        return False
    for actual_row, expected_row in zip(actual, expected):
        if not isinstance(actual_row, list) or len(actual_row) != len(
            expected_row
        ):
            return False
        for actual_value, expected_value in zip(actual_row, expected_row):
            if abs(float(actual_value) - expected_value) > tolerance:
                return False
    return True


def _expected_values(
    world_size: int,
) -> tuple[list[list[float]], list[list[float]]]:
    average_scale = (world_size + 1.0) / 2.0
    gradient_row = [
        average_scale * float(index + 1) for index in range(world_size)
    ]
    parameter = []
    for row in range(world_size):
        parameter.append(
            [
                (1.0 if row == column else 0.0)
                - 0.1 * gradient_row[column]
                for column in range(world_size)
            ]
        )
    return [gradient_row], parameter


def _validate_rank_reports(
    reports: list[dict[str, Any]],
    world_size: int,
    precision: str,
) -> None:
    expected_gradient, expected_parameter = _expected_values(world_size)
    tolerance = 1e-6 if precision == "fp32" else 2e-2
    for rank, report in enumerate(reports):
        if report.get("status") != "success":
            raise AssertionError(f"FSDP2 rank {rank} failed: {report}")
        if report.get("precision") != precision:
            raise AssertionError(
                f"rank {rank} precision mismatch: {report.get('precision')}"
            )
        if report.get("global_parameter_shape") != [world_size, world_size]:
            raise AssertionError(
                f"rank {rank} global shape mismatch: {report}"
            )
        if report.get("local_shard_shape") != [1, world_size]:
            raise AssertionError(
                f"rank {rank} local shape mismatch: {report}"
            )
        if not _matrix_close(
            report.get("local_shard_gradient"),
            expected_gradient,
            tolerance,
        ):
            raise AssertionError(
                f"rank {rank} gradient mismatch: "
                f"{report.get('local_shard_gradient')}"
            )
        if not _matrix_close(
            report.get("local_shard_after_step"),
            [expected_parameter[rank]],
            tolerance,
        ):
            raise AssertionError(
                f"rank {rank} local parameter mismatch: "
                f"{report.get('local_shard_after_step')}"
            )
        if not _matrix_close(
            report.get("full_parameter_after_step"),
            expected_parameter,
            tolerance,
        ):
            raise AssertionError(
                f"rank {rank} full parameter mismatch: "
                f"{report.get('full_parameter_after_step')}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run FSDP2/DTensor numeric validation on one physical CUDA GPU "
            "while fake NCCL simulates two or four ranks."
        )
    )
    parser.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build")
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--world-size", type=int, choices=(2, 4), default=2)
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="fp32",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    build_dir = args.build_dir.resolve()
    coordinator_bin = build_dir / "fakegpu-coordinator"
    nccl_lib = (
        build_dir / "libnccl.dylib"
        if sys.platform == "darwin"
        else build_dir / "libnccl.so.2"
    )
    cluster_config = CLUSTER_CONFIGS[args.world_size]
    for artifact in (coordinator_bin, nccl_lib, WORKER, cluster_config):
        if not artifact.exists():
            print(f"missing required artifact: {artifact}", file=sys.stderr)
            return 2

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.report_dir is None:
        temporary = tempfile.TemporaryDirectory(
            prefix="fakegpu-hybrid-fsdp2-numerics-"
        )
        report_root = Path(temporary.name)
    else:
        report_root = args.report_dir.resolve()
        report_root.mkdir(parents=True, exist_ok=True)

    socket_path = report_root / "coordinator.sock"
    cluster_report_path = report_root / "cluster_report.json"
    rank_report_dir = report_root / "ranks"
    coordinator_env = dict(os.environ)
    coordinator_env.pop("LD_PRELOAD", None)
    coordinator_env.pop("DYLD_INSERT_LIBRARIES", None)
    coordinator_env.update(
        {
            "FAKEGPU_MODE": "simulate",
            "FAKEGPU_DIST_MODE": "simulate",
            "FAKEGPU_CLUSTER_CONFIG": str(cluster_config),
            "FAKEGPU_COORDINATOR_TRANSPORT": "unix",
            "FAKEGPU_COORDINATOR_ADDR": str(socket_path),
            "FAKEGPU_CLUSTER_REPORT_PATH": str(cluster_report_path),
        }
    )
    coordinator = subprocess.Popen(
        [
            str(coordinator_bin),
            "--transport",
            "unix",
            "--address",
            str(socket_path),
        ],
        cwd=REPO_ROOT,
        env=coordinator_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        _wait_for_unix_socket(socket_path, coordinator)
        worker_env = dict(os.environ)
        worker_env.pop("DYLD_INSERT_LIBRARIES", None)
        worker_env["LD_PRELOAD"] = str(nccl_lib)
        worker_env.update(
            {
                "FAKEGPU_MODE": "hybrid",
                "FAKEGPU_DIST_MODE": "simulate",
                "FAKEGPU_CLUSTER_CONFIG": str(cluster_config),
                "FAKEGPU_COORDINATOR_TRANSPORT": "unix",
                "FAKEGPU_COORDINATOR_ADDR": str(socket_path),
                "FAKEGPU_COORDINATOR_TIMEOUT_MS": "90000",
            }
        )
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            f"--nproc-per-node={args.world_size}",
            "--master-addr=127.0.0.1",
            f"--master-port={_find_free_port()}",
            str(WORKER),
            "--report-dir",
            str(rank_report_dir),
            "--precision",
            args.precision,
        ]
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=worker_env,
            text=True,
            capture_output=True,
            timeout=args.timeout,
            check=False,
        )
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        if completed.returncode != 0:
            return completed.returncode

        rank_reports = _load_rank_reports(rank_report_dir, args.world_size)
        _validate_rank_reports(
            rank_reports,
            args.world_size,
            args.precision,
        )

        _shutdown_unix_coordinator(socket_path)
        coordinator.wait(timeout=5)
        cluster_report = json.loads(
            cluster_report_path.read_text(encoding="utf-8")
        )
        for collective in ("all_gather", "reduce_scatter"):
            if cluster_report["collectives"][collective]["calls"] < 1:
                raise AssertionError(f"FSDP2 did not issue {collective}")
        node_pairs = cluster_report.get("node_pairs")
        if not isinstance(node_pairs, list) or len(node_pairs) != 1:
            raise AssertionError(
                f"expected one complete node-pair entry, got {node_pairs!r}"
            )
        node_pair = node_pairs[0]
        if int(node_pair.get("total_bytes", 0)) <= 0:
            raise AssertionError(
                f"FSDP2 node-pair traffic was empty: {node_pair}"
            )
        markdown_report_path = Path(
            cluster_report["cluster"]["markdown_report_path"]
        )
        if not markdown_report_path.is_file():
            raise AssertionError(
                "cluster Markdown report was not generated: "
                f"{markdown_report_path}"
            )

        summary = {
            "status": "success",
            "world_size": args.world_size,
            "precision": args.precision,
            "physical_device_name": rank_reports[0]["physical_device_name"],
            "physical_compute_capability": rank_reports[0][
                "physical_compute_capability"
            ],
            "torch_version": rank_reports[0]["torch_version"],
            "torch_cuda_version": rank_reports[0]["torch_cuda_version"],
            "parameter_dtype": rank_reports[0]["parameter_dtype"],
            "gradient_dtype": rank_reports[0]["gradient_dtype"],
            "local_shard_shape": rank_reports[0]["local_shard_shape"],
            "all_gather_calls": cluster_report["collectives"]["all_gather"][
                "calls"
            ],
            "reduce_scatter_calls": cluster_report["collectives"][
                "reduce_scatter"
            ]["calls"],
            "node_pair_total_bytes": node_pair["total_bytes"],
            "node_pair_peak_bytes_per_operation": node_pair[
                "peak_combined_bytes_per_operation"
            ],
            "cluster_markdown_report": str(markdown_report_path),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    finally:
        _close_coordinator(coordinator, socket_path)
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
