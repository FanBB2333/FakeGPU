#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
def find_real_nccl_path() -> Path:
    env_value = os.environ.get("FAKEGPU_REAL_NCCL_PATH")
    if env_value:
        path = Path(env_value)
        if path.is_file():
            return path
        raise SystemExit(f"FAKEGPU_REAL_NCCL_PATH does not exist: {path}")

    import torch

    site_packages = Path(torch.__file__).resolve().parent.parent
    candidates = [
        site_packages / "nvidia" / "nccl" / "lib" / "libnccl.so.2",
    ]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "lib" / "libnccl.so.2")
    candidates.extend(
        [
            Path("/usr/lib/x86_64-linux-gnu/libnccl.so.2"),
            Path("/usr/lib64/libnccl.so.2"),
            Path("/usr/local/cuda/lib64/libnccl.so.2"),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SystemExit("unable to locate a real libnccl.so.2; set FAKEGPU_REAL_NCCL_PATH")


def detect_world_size() -> int:
    import torch

    device_count = torch.cuda.device_count()
    if device_count <= 0:
        raise SystemExit("no CUDA device is available for Step 20 validation")
    return 2 if device_count >= 2 else 1


def run_workers(
    *,
    run_mode: str,
    world_size: int,
    lib_path: Path,
    report_dir: Path,
    env_updates: dict[str, str],
) -> list[dict[str, Any]]:
    worker_code = (REPO_ROOT / "verification" / "nccl_proxy_worker.py").read_text(encoding="utf-8")
    uid_path = report_dir / f"{run_mode}_uid.txt"
    processes: list[tuple[int, subprocess.Popen[str], Path]] = []
    for rank in range(world_size):
        report_path = report_dir / f"{run_mode}_rank_{rank}.json"
        env = dict(os.environ)
        env.update(env_updates)
        env.update(
            {
                "RUN_MODE": run_mode,
                "WORLD_SIZE": str(world_size),
                "RANK": str(rank),
                "LOCAL_RANK": "0",
                "REPORT_PATH": str(report_path),
                "NCCL_UID_PATH": str(uid_path),
                "NCCL_LIB_PATH": str(lib_path),
            }
        )
        proc = subprocess.Popen(
            [sys.executable, "-"],
            cwd=REPO_ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.stdin is None:
            raise RuntimeError("worker stdin was not created")
        proc.stdin.write(worker_code)
        proc.stdin.close()
        proc.stdin = None
        processes.append((rank, proc, report_path))

    reports: list[dict[str, Any]] = []
    failures: list[str] = []
    for rank, proc, report_path in processes:
        try:
            stdout, stderr = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            partial = report_path.read_text(encoding="utf-8") if report_path.is_file() else "(no report)"
            failures.append(
                f"rank {rank} timed out\n"
                f"partial_report:\n{partial}\n"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr}"
            )
            continue
        if proc.returncode != 0:
            failures.append(
                f"rank {rank} failed with exit code {proc.returncode}\n"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr}"
            )
            continue
        if not report_path.is_file():
            failures.append(f"rank {rank} did not produce report {report_path}")
            continue
        reports.append(json.loads(report_path.read_text(encoding="utf-8")))

    if failures:
        raise RuntimeError("\n\n".join(failures))
    reports.sort(key=lambda item: int(item["rank"]))
    return reports


def wait_for_socket(path: Path, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise TimeoutError(f"coordinator socket was not created: {path}")


def shutdown_coordinator(socket_path: Path) -> None:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(str(socket_path))
        sock.sendall(b"SHUTDOWN\n")
        sock.recv(4096)


def compare_reports(
    baseline: list[dict[str, Any]],
    proxy: list[dict[str, Any]],
) -> None:
    if len(baseline) != len(proxy):
        raise AssertionError(
            f"baseline rank count {len(baseline)} != proxy rank count {len(proxy)}"
        )
    for base_item, proxy_item in zip(baseline, proxy):
        if base_item["rank"] != proxy_item["rank"]:
            raise AssertionError(
                f"rank mismatch: baseline rank {base_item['rank']} vs proxy rank {proxy_item['rank']}"
            )
        if base_item["all_reduce_value"] != proxy_item["all_reduce_value"]:
            raise AssertionError(
                f"all_reduce mismatch on rank {base_item['rank']}: "
                f"{base_item['all_reduce_value']} != {proxy_item['all_reduce_value']}"
            )
        if base_item["broadcast_value"] != proxy_item["broadcast_value"]:
            raise AssertionError(
                f"broadcast mismatch on rank {base_item['rank']}: "
                f"{base_item['broadcast_value']} != {proxy_item['broadcast_value']}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate experimental FakeGPU NCCL proxy mode")
    parser.add_argument("--world-size", type=int, default=0)
    parser.add_argument(
        "--fake-nccl-lib",
        type=Path,
        default=REPO_ROOT / "build" / "libnccl.so.2",
    )
    parser.add_argument(
        "--coordinator-bin",
        type=Path,
        default=REPO_ROOT / "build" / "fakegpu-coordinator",
    )
    parser.add_argument(
        "--cluster-config",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    if not args.fake_nccl_lib.is_file():
        raise SystemExit(f"missing fake NCCL library: {args.fake_nccl_lib}")
    if not args.coordinator_bin.is_file():
        raise SystemExit(f"missing coordinator binary: {args.coordinator_bin}")

    real_nccl_path = find_real_nccl_path()
    world_size = args.world_size if args.world_size > 0 else detect_world_size()
    if world_size not in (1, 2):
        raise SystemExit("Step 20 validation currently supports world_size 1 or 2")

    cluster_config = args.cluster_config
    if cluster_config is None:
        cluster_config = (
            REPO_ROOT / "verification" / "data" / ("cluster_proxy_2r.yaml" if world_size == 2 else "cluster_proxy_1r.yaml")
        )
    if not cluster_config.is_file():
        raise SystemExit(f"missing cluster config: {cluster_config}")

    with tempfile.TemporaryDirectory(prefix="fakegpu-proxy-step20-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        baseline_report_dir = tmpdir_path / "baseline"
        proxy_report_dir = tmpdir_path / "proxy"
        baseline_report_dir.mkdir()
        proxy_report_dir.mkdir()

        baseline_reports = run_workers(
            run_mode="baseline",
            world_size=world_size,
            lib_path=real_nccl_path,
            report_dir=baseline_report_dir,
            env_updates={},
        )

        socket_path = tmpdir_path / "coordinator.sock"
        cluster_report_path = tmpdir_path / "cluster_report.json"
        coordinator_env = dict(os.environ)
        coordinator_env.update(
            {
                "FAKEGPU_MODE": "hybrid",
                "FAKEGPU_DIST_MODE": "proxy",
                "FAKEGPU_CLUSTER_CONFIG": str(cluster_config),
                "FAKEGPU_COORDINATOR_TRANSPORT": "unix",
                "FAKEGPU_COORDINATOR_ADDR": str(socket_path),
                "FAKEGPU_CLUSTER_REPORT_PATH": str(cluster_report_path),
            }
        )
        coordinator = subprocess.Popen(
            [str(args.coordinator_bin), "--transport", "unix", "--address", str(socket_path)],
            cwd=REPO_ROOT,
            env=coordinator_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            wait_for_socket(socket_path)

            proxy_base_env = {
                "FAKEGPU_MODE": "hybrid",
                "FAKEGPU_DIST_MODE": "proxy",
                "FAKEGPU_CLUSTER_CONFIG": str(cluster_config),
                "FAKEGPU_COORDINATOR_TRANSPORT": "unix",
                "FAKEGPU_COORDINATOR_ADDR": str(socket_path),
                "FAKEGPU_REAL_NCCL_PATH": str(real_nccl_path),
            }
            proxy_reports = run_workers(
                run_mode="proxy",
                world_size=world_size,
                lib_path=args.fake_nccl_lib,
                report_dir=proxy_report_dir,
                env_updates=proxy_base_env,
            )
        finally:
            try:
                shutdown_coordinator(socket_path)
            except Exception:
                coordinator.kill()
            stdout, stderr = coordinator.communicate(timeout=5)
            if coordinator.returncode not in (0, None):
                raise RuntimeError(
                    "coordinator exited with error:\n"
                    f"stdout:\n{stdout}\n"
                    f"stderr:\n{stderr}"
                )

        compare_reports(baseline_reports, proxy_reports)

        check_cmd = [
            sys.executable,
            str(REPO_ROOT / "verification" / "check_cluster_report.py"),
            "--path",
            str(cluster_report_path),
            "--expect-world-size",
            str(world_size),
            "--expect-node-count",
            str(world_size),
            "--expect-collective",
            "all_reduce",
            "--expect-collective",
            "broadcast",
            "--min-ranks",
            str(world_size),
        ]
        if world_size > 1:
            check_cmd.append("--expect-links")
        check_result = subprocess.run(
            check_cmd,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if check_result.returncode != 0:
            raise RuntimeError(
                "cluster report validation failed:\n"
                f"stdout:\n{check_result.stdout}\n"
                f"stderr:\n{check_result.stderr}"
            )

        summary = {
            "world_size": world_size,
            "cluster_config": str(cluster_config),
            "real_nccl_path": str(real_nccl_path),
            "fake_nccl_lib": str(args.fake_nccl_lib),
            "cluster_report_path": str(cluster_report_path),
            "baseline": baseline_reports,
            "proxy": proxy_reports,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
