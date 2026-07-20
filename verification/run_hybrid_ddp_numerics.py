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
WORKER = REPO_ROOT / "test" / "test_ddp_hybrid_numerics.py"
CLUSTER_CONFIG = REPO_ROOT / "verification" / "data" / "cluster_hybrid_2r.yaml"


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


def _load_rank_reports(report_dir: Path, world_size: int) -> list[dict[str, Any]]:
    reports = []
    for rank in range(world_size):
        path = report_dir / f"rank_{rank}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing rank report: {path}")
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run two real-CUDA ranks on one physical GPU while fake NCCL "
            "validates DDP gradient averaging."
        )
    )
    parser.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build")
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    build_dir = args.build_dir.resolve()
    coordinator_bin = build_dir / "fakegpu-coordinator"
    nccl_lib = (
        build_dir / "libnccl.dylib"
        if sys.platform == "darwin"
        else build_dir / "libnccl.so.2"
    )
    for artifact in (coordinator_bin, nccl_lib, WORKER, CLUSTER_CONFIG):
        if not artifact.exists():
            print(f"missing required artifact: {artifact}", file=sys.stderr)
            return 2

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.report_dir is None:
        temporary = tempfile.TemporaryDirectory(
            prefix="fakegpu-hybrid-ddp-numerics-"
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
            "FAKEGPU_CLUSTER_CONFIG": str(CLUSTER_CONFIG),
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
                "FAKEGPU_CLUSTER_CONFIG": str(CLUSTER_CONFIG),
                "FAKEGPU_COORDINATOR_TRANSPORT": "unix",
                "FAKEGPU_COORDINATOR_ADDR": str(socket_path),
                "FAKEGPU_COORDINATOR_TIMEOUT_MS": "60000",
            }
        )
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--nproc-per-node=2",
            "--master-addr=127.0.0.1",
            f"--master-port={_find_free_port()}",
            str(WORKER),
            "--report-dir",
            str(rank_report_dir),
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

        rank_reports = _load_rank_reports(rank_report_dir, 2)
        for report in rank_reports:
            if report.get("status") != "success":
                raise AssertionError(f"rank failed: {report}")
            if report.get("gradient") != [[1.5, 3.0]]:
                raise AssertionError(f"unexpected gradient: {report}")
            parameters = report.get("parameters_after_step")
            if not (
                isinstance(parameters, list)
                and len(parameters) == 1
                and abs(float(parameters[0][0]) - 0.85) <= 1e-6
                and abs(float(parameters[0][1]) + 0.3) <= 1e-6
            ):
                raise AssertionError(f"unexpected parameters: {report}")

        _shutdown_unix_coordinator(socket_path)
        coordinator.wait(timeout=5)
        cluster_report = json.loads(
            cluster_report_path.read_text(encoding="utf-8")
        )
        if cluster_report["collectives"]["all_reduce"]["calls"] < 1:
            raise AssertionError("DDP did not issue a simulated all-reduce")
        if cluster_report["collectives"]["all_gather"]["calls"] < 1:
            raise AssertionError("parameter consistency did not issue all-gather")

        summary = {
            "status": "success",
            "physical_device_name": rank_reports[0]["physical_device_name"],
            "physical_compute_capability": rank_reports[0][
                "physical_compute_capability"
            ],
            "torch_version": rank_reports[0]["torch_version"],
            "torch_cuda_version": rank_reports[0]["torch_cuda_version"],
            "gradient": rank_reports[0]["gradient"],
            "parameters_after_step": rank_reports[0][
                "parameters_after_step"
            ],
            "all_reduce_calls": cluster_report["collectives"]["all_reduce"][
                "calls"
            ],
            "all_gather_calls": cluster_report["collectives"]["all_gather"][
                "calls"
            ],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    finally:
        if coordinator.poll() is None:
            try:
                _shutdown_unix_coordinator(socket_path)
                coordinator.wait(timeout=3)
            except Exception:
                coordinator.kill()
                coordinator.wait(timeout=5)
        coordinator.communicate(timeout=5)
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
