#!/usr/bin/env python3

from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = Path(os.environ.get("FAKEGPU_BUILD_DIR", REPO_ROOT / "build"))
COORDINATOR_BIN = BUILD_DIR / "fakegpu-coordinator"

TWO_NODE_CLUSTER_YAML = """version: 1
cluster:
  name: smoke
  default_backend: nccl
nodes:
  - id: node0
    host: h0
    ranks: [0, 1]
    gpus:
      - profile: a100
      - profile: a100
  - id: node1
    host: h1
    ranks: [2, 3]
    gpus:
      - profile: a100
      - profile: a100
fabric:
  intra_node:
    type: nvlink
    bandwidth_gbps: 100
    latency_us: 1
  inter_node:
    type: tcp
    bandwidth_gbps: {inter_node_bandwidth}
    latency_us: 5
"""


def request(socket_path: str, payload: str) -> str:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(socket_path)
        sock.sendall((payload + "\n").encode("utf-8"))
        data = b""
        while not data.endswith(b"\n"):
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
    return data.decode("utf-8").strip()


def request_packet(
    socket_path: str,
    line: str,
    payload: bytes = b"",
) -> tuple[str, bytes]:
    if payload:
        line += f" payload_bytes={len(payload)}"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(10.0)
        sock.connect(socket_path)
        sock.sendall(line.encode("utf-8") + b"\n" + payload)
        data = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
    header, _, response_payload = data.partition(b"\n")
    return header.decode("utf-8").strip(), response_payload


def concurrent_requests(*calls) -> list:
    results: list = [None] * len(calls)

    def run(index: int) -> None:
        results[index] = calls[index]()

    threads = [threading.Thread(target=run, args=(index,)) for index in range(len(calls))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


def coordinator_rss_kib(pid: int) -> int:
    return int(
        subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


@contextlib.contextmanager
def running_coordinator(tmpdir: str, extra_env: dict[str, str] | None = None):
    socket_path = os.path.join(tmpdir, "coordinator.sock")
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("FAKEGPU_")
    }
    env.update(extra_env or {})
    proc = subprocess.Popen(
        [str(COORDINATOR_BIN), "--transport", "unix", "--address", socket_path],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        startup_timeout_seconds = 5.0
        deadline = time.monotonic() + startup_timeout_seconds
        while time.monotonic() < deadline:
            if os.path.exists(socket_path):
                break
            if proc.poll() is not None:
                raise AssertionError(
                    "coordinator exited before creating its socket "
                    f"(code {proc.returncode})"
                )
            time.sleep(0.05)
        if not os.path.exists(socket_path):
            raise AssertionError(
                "socket was not created within "
                f"{startup_timeout_seconds:g} seconds"
            )
        yield socket_path, proc
    finally:
        if proc.poll() is None:
            proc.kill()
        stdout, stderr = proc.communicate()
        if proc.returncode not in (None, 0):
            sys.stderr.write(stdout)
            sys.stderr.write(stderr)


def check_empty_report(tmpdir: str) -> None:
    report_path = Path(tmpdir) / "cluster-report.json"
    markdown_path = Path(tmpdir) / "cluster-report.md"
    with running_coordinator(
        tmpdir,
        {
            "FAKEGPU_CLUSTER_REPORT_PATH": str(report_path),
            "FAKEGPU_CLUSTER_REPORT_MARKDOWN_PATH": str(markdown_path),
        },
    ) as (socket_path, proc):
        response = request(socket_path, "PING")
        if "OK" not in response or "status=ready" not in response or "version=1" not in response:
            raise AssertionError(f"unexpected ping response: {response}")

        response = request(socket_path, "SHUTDOWN")
        if "OK" not in response or "status=shutting_down" not in response:
            raise AssertionError(f"unexpected shutdown response: {response}")

        proc.wait(timeout=2.0)
        if os.path.exists(socket_path):
            raise AssertionError("socket path still exists after shutdown")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("schema_version") != "cluster_report.v1":
            raise AssertionError(f"unexpected empty report: {report}")
        if any(
            int(stats.get("calls", -1)) != 0
            for stats in report.get("collectives", {}).values()
        ):
            raise AssertionError(f"empty report recorded traffic: {report}")
        if (
            report.get("cluster", {}).get("communicators") != 0
            or report.get("node_pairs") != []
            or report.get("ranks") != []
            or report.get("operation_timeline", {}).get("retained_entries")
            != 0
            or not markdown_path.is_file()
        ):
            raise AssertionError(f"empty report is incomplete: {report}")


def check_socket_point_to_point(tmpdir: str) -> None:
    sent = bytes(range(8))
    with running_coordinator(tmpdir) as (socket_path, proc):
        init = concurrent_requests(
            *(
                lambda rank=rank: request_packet(
                    socket_path,
                    f"INIT_COMM unique_id=p2p world_size=2 rank={rank} timeout_ms=3000",
                )
                for rank in range(2)
            )
        )
        if any(not header.startswith("OK") for header, _ in init):
            raise AssertionError(f"unexpected init responses: {init}")
        common = "comm_id=1 seqno=1 count=2 bytes=8 dtype=float32 buffer_transport=socket timeout_ms=3000"
        send, recv = concurrent_requests(
            lambda: request_packet(socket_path, f"SEND rank=0 peer=1 {common}", sent),
            lambda: request_packet(socket_path, f"RECV rank=1 peer=0 {common}"),
        )
        if not send[0].startswith("OK") or not recv[0].startswith("OK"):
            raise AssertionError(f"socket point-to-point failed: {send}, {recv}")
        if recv[1] != sent:
            raise AssertionError(f"RECV returned {recv[1]!r}, expected {sent!r}")
        short_send, _ = request_packet(
            socket_path,
            f"SEND rank=0 peer=1 {common.replace('seqno=1', 'seqno=2')}",
            sent[:4],
        )
        if "socket_payload_size_mismatch" not in short_send:
            raise AssertionError(f"short socket SEND was accepted: {short_send}")
        request(socket_path, "SHUTDOWN")
        proc.wait(timeout=2.0)


def check_finished_client_threads_are_released(tmpdir: str) -> None:
    with running_coordinator(tmpdir) as (socket_path, proc):
        for _ in range(100):
            request(socket_path, "PING")
        rss_before = coordinator_rss_kib(proc.pid)
        for _ in range(2000):
            request(socket_path, "PING")
        growth_kib = coordinator_rss_kib(proc.pid) - rss_before
        if growth_kib > 10 * 1024:
            raise AssertionError(
                f"coordinator RSS grew by {growth_kib} KiB over 2000 requests"
            )
        request(socket_path, "SHUTDOWN")
        proc.wait(timeout=2.0)


def check_non_finite_fabric_value_rejected(tmpdir: str) -> None:
    config_path = Path(tmpdir) / "cluster.yaml"
    config_path.write_text(
        TWO_NODE_CLUSTER_YAML.format(inter_node_bandwidth="nan"),
        encoding="utf-8",
    )
    report_path = Path(tmpdir) / "nan-report.json"
    socket_path = os.path.join(tmpdir, "coordinator.sock")
    with running_coordinator(
        tmpdir,
        {
            "FAKEGPU_DIST_MODE": "simulate",
            "FAKEGPU_COORDINATOR_TRANSPORT": "unix",
            "FAKEGPU_COORDINATOR_ADDR": socket_path,
            "FAKEGPU_CLUSTER_CONFIG": str(config_path),
            "FAKEGPU_CLUSTER_REPORT_PATH": str(report_path),
        },
    ) as (socket_path, proc):
        request(socket_path, "SHUTDOWN")
        proc.wait(timeout=2.0)
    json.loads(report_path.read_text(encoding="utf-8"))


def check_cluster_config_without_distributed_env(tmpdir: str) -> None:
    config_path = Path(tmpdir) / "cluster.yaml"
    config_path.write_text(
        TWO_NODE_CLUSTER_YAML.format(inter_node_bandwidth="25"),
        encoding="utf-8",
    )
    report_path = Path(tmpdir) / "config-report.json"
    with running_coordinator(
        tmpdir,
        {
            "FAKEGPU_CLUSTER_CONFIG": str(config_path),
            "FAKEGPU_CLUSTER_REPORT_PATH": str(report_path),
        },
    ) as (socket_path, proc):
        concurrent_requests(
            *(
                lambda rank=rank: request_packet(
                    socket_path,
                    f"INIT_COMM unique_id=cfg world_size=4 rank={rank} timeout_ms=3000",
                )
                for rank in range(4)
            )
        )
        allreduce = concurrent_requests(
            *(
                lambda rank=rank: request_packet(
                    socket_path,
                    f"ALLREDUCE comm_id=1 rank={rank} seqno=1 count=256 bytes=1024 "
                    "root=-1 timeout_ms=3000 proxy_only=1 dtype=float32 reduce_op=sum",
                )
                for rank in range(4)
            )
        )
        if any(not header.startswith("OK") for header, _ in allreduce):
            raise AssertionError(f"unexpected allreduce responses: {allreduce}")
        request(socket_path, "SHUTDOWN")
        proc.wait(timeout=2.0)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cluster = report.get("cluster", {})
    if cluster.get("name") != "smoke" or cluster.get("node_count") != 2 or not report.get("links"):
        raise AssertionError(f"cluster config was not applied: {report}")


def main() -> int:
    if not COORDINATOR_BIN.exists():
        print(f"missing coordinator binary: {COORDINATOR_BIN}", file=sys.stderr)
        return 2

    checks = (
        check_empty_report,
        check_socket_point_to_point,
        check_finished_client_threads_are_released,
        check_non_finite_fabric_value_rejected,
        check_cluster_config_without_distributed_env,
    )
    for check in checks:
        with tempfile.TemporaryDirectory(prefix="fgpu-coord-", dir="/tmp") as tmpdir:
            check(tmpdir)

    print("coordinator smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
