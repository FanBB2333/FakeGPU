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
WORKER = REPO_ROOT / "verification" / "remote_nccl_fault_worker.py"
CLUSTER_CONFIG = REPO_ROOT / "verification" / "data" / "cluster_tcp_2r.yaml"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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


def _wait_for_ready(
    endpoint: str,
    coordinator: subprocess.Popen[str],
    timeout: float = 5.0,
) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if coordinator.poll() is not None:
            stdout, stderr = coordinator.communicate()
            raise RuntimeError(
                f"coordinator exited with code {coordinator.returncode}\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}"
            )
        try:
            response = _request(endpoint, "PING", timeout=0.25)
            if response.startswith("OK ") and "status=ready" in response:
                return
        except OSError as exc:
            last_error = exc
        time.sleep(0.05)
    raise TimeoutError(f"coordinator did not become ready: {last_error}")


def _worker_env(endpoint: str, timeout_ms: int) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("LD_PRELOAD", None)
    env.pop("DYLD_INSERT_LIBRARIES", None)
    env.update(
        {
            "FAKEGPU_MODE": "simulate",
            "FAKEGPU_DIST_MODE": "simulate",
            "FAKEGPU_CLUSTER_CONFIG": str(CLUSTER_CONFIG),
            "FAKEGPU_COORDINATOR_TRANSPORT": "tcp",
            "FAKEGPU_COORDINATOR_ADDR": endpoint,
            "FAKEGPU_COORDINATOR_TIMEOUT_MS": str(timeout_ms),
            "FAKEGPU_STAGING_FORCE_SOCKET": "1",
        }
    )
    return env


def _worker_command(
    *,
    case: str,
    rank: int,
    session: str,
    timeout_ms: int,
    nccl_lib: Path,
    report_path: Path,
) -> list[str]:
    return [
        sys.executable,
        str(WORKER),
        "--case",
        case,
        "--rank",
        str(rank),
        "--world-size",
        "2",
        "--session",
        session,
        "--timeout-ms",
        str(timeout_ms),
        "--nccl-lib",
        str(nccl_lib),
        "--report",
        str(report_path),
    ]


def _run_mismatch(
    endpoint: str,
    nccl_lib: Path,
    temp_dir: Path,
) -> list[dict[str, Any]]:
    timeout_ms = 1000
    session = "local-collective-mismatch"
    processes: list[tuple[int, Path, subprocess.Popen[str]]] = []
    for rank in range(2):
        report_path = temp_dir / f"mismatch-rank-{rank}.json"
        process = subprocess.Popen(
            _worker_command(
                case="collective-mismatch",
                rank=rank,
                session=session,
                timeout_ms=timeout_ms,
                nccl_lib=nccl_lib,
                report_path=report_path,
            ),
            cwd=REPO_ROOT,
            env=_worker_env(endpoint, timeout_ms),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        processes.append((rank, report_path, process))

    reports: list[dict[str, Any]] = []
    failures: list[str] = []
    for rank, report_path, process in processes:
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate(timeout=5)
            failures.append(f"rank {rank} timed out\n{stdout}\n{stderr}")
            continue
        if report_path.is_file():
            reports.append(json.loads(report_path.read_text(encoding="utf-8")))
        if process.returncode != 0:
            failures.append(
                f"rank {rank} exited with {process.returncode}\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}"
            )
    if failures:
        raise AssertionError("\n\n".join(failures))
    reports.sort(key=lambda item: int(item["rank"]))
    assert len(reports) == 2
    assert all(item["status"] == "success" for item in reports)
    assert all(item["expected_failure_observed"] is True for item in reports)
    assert all(int(item["mismatch_result"]) != 0 for item in reports)
    assert all(int(item["async_error"]) != 0 for item in reports)
    assert all(int(item["retry_result"]) != 0 for item in reports)
    return reports


def _run_missing_peer(
    endpoint: str,
    nccl_lib: Path,
    temp_dir: Path,
) -> dict[str, Any]:
    timeout_ms = 250
    report_path = temp_dir / "missing-peer-rank-0.json"
    completed = subprocess.run(
        _worker_command(
            case="missing-peer",
            rank=0,
            session="local-missing-peer",
            timeout_ms=timeout_ms,
            nccl_lib=nccl_lib,
            report_path=report_path,
        ),
        cwd=REPO_ROOT,
        env=_worker_env(endpoint, timeout_ms),
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"missing-peer worker exited with {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "success"
    assert report["expected_failure_observed"] is True
    assert int(report["comm_init_result"]) != 0
    assert "timeout" in str(report["comm_init_last_error"]).lower()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate mismatch and missing-peer failures over TCP."
    )
    parser.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build")
    args = parser.parse_args()

    build_dir = args.build_dir.resolve()
    coordinator_bin = build_dir / "fakegpu-coordinator"
    nccl_lib = (
        build_dir / "libnccl.dylib"
        if sys.platform == "darwin"
        else build_dir / "libnccl.so.2"
    )
    for path in (coordinator_bin, nccl_lib, WORKER, CLUSTER_CONFIG):
        if not path.exists():
            print(f"missing required artifact: {path}", file=sys.stderr)
            return 2

    with tempfile.TemporaryDirectory(prefix="fakegpu-remote-faults-") as raw:
        temp_dir = Path(raw)
        endpoint = f"127.0.0.1:{_free_port()}"
        cluster_report_path = temp_dir / "cluster-report.json"
        coordinator_env = _worker_env(endpoint, 1000)
        coordinator_env["FAKEGPU_CLUSTER_REPORT_PATH"] = str(cluster_report_path)
        coordinator = subprocess.Popen(
            [
                str(coordinator_bin),
                "--transport",
                "tcp",
                "--address",
                endpoint,
            ],
            cwd=REPO_ROOT,
            env=coordinator_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            _wait_for_ready(endpoint, coordinator)
            mismatch_reports = _run_mismatch(endpoint, nccl_lib, temp_dir)
            missing_peer_report = _run_missing_peer(
                endpoint,
                nccl_lib,
                temp_dir,
            )
            shutdown = _request(endpoint, "SHUTDOWN")
            assert shutdown.startswith("OK "), shutdown
            coordinator.wait(timeout=5)
            if coordinator.returncode != 0:
                stdout, stderr = coordinator.communicate()
                raise AssertionError(
                    f"coordinator exited with {coordinator.returncode}\n"
                    f"stdout:\n{stdout}\nstderr:\n{stderr}"
                )
        finally:
            if coordinator.poll() is None:
                try:
                    _request(endpoint, "SHUTDOWN", timeout=0.5)
                    coordinator.wait(timeout=3)
                except Exception:
                    coordinator.kill()
                    coordinator.wait(timeout=5)

        cluster_report = json.loads(
            cluster_report_path.read_text(encoding="utf-8")
        )
        assert cluster_report["schema_version"] == "cluster_report.v1"
        assert cluster_report["cluster"]["coordinator_transport"] == "tcp"
        assert cluster_report["cluster"]["world_size"] == 2
        assert cluster_report["cluster"]["node_count"] == 2
        assert cluster_report["collectives"]["all_reduce"]["calls"] == 0
        assert cluster_report["operation_timeline"]["retained_entries"] == 0
        rank_timeouts = {
            int(item["rank"]): int(item["timeouts"])
            for item in cluster_report["ranks"]
        }
        assert rank_timeouts.get(0, 0) >= 1

        print(
            json.dumps(
                {
                    "status": "success",
                    "mismatch_results": [
                        int(item["mismatch_result"])
                        for item in mismatch_reports
                    ],
                    "missing_peer_result": int(
                        missing_peer_report["comm_init_result"]
                    ),
                    "rank_timeouts": rank_timeouts,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
