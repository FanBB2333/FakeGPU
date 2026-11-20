#!/usr/bin/env python3

from __future__ import annotations

import ctypes
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fakegpu._bandwidth_worker import (
    NCCL_FLOAT32,
    NCCL_SUM,
    NcclUniqueId,
    _configure_nccl,
    _require_success,
)
from fakegpu.distributed_cli import _nccl_library_path


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = Path(os.environ.get("FAKEGPU_BUILD_DIR", REPO_ROOT / "build"))
COORDINATOR_BIN = BUILD_DIR / "fakegpu-coordinator"


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


def check_socket_reduce(socket_path: str) -> None:
    os.environ.update(
        FAKEGPU_DIST_MODE="simulate",
        FAKEGPU_COORDINATOR_TRANSPORT="unix",
        FAKEGPU_COORDINATOR_ADDR=socket_path,
        FAKEGPU_COORDINATOR_TIMEOUT_MS="5000",
        FAKEGPU_STAGING_FORCE_SOCKET="1",
    )
    lib = ctypes.CDLL(str(_nccl_library_path(BUILD_DIR)))
    _configure_nccl(lib)
    lib.ncclReduce.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
    ]
    uid = NcclUniqueId()
    uid.internal = b"socket-reduce-test"

    def worker(rank: int) -> None:
        comm = ctypes.c_void_p()
        send = ctypes.c_float(rank + 2)
        recv = ctypes.c_float(-1)
        _require_success(lib, comm, lib.ncclCommInitRank(ctypes.byref(comm), 2, uid, rank), "init")
        result = lib.ncclReduce(
            ctypes.byref(send), ctypes.byref(recv) if rank == 1 else None,
            1, NCCL_FLOAT32, NCCL_SUM, 1, comm, None,
        )
        _require_success(lib, comm, result, "reduce")
        assert recv.value == (5 if rank == 1 else -1)
        result = lib.ncclAllReduce(
            ctypes.byref(send), ctypes.byref(recv), 1, NCCL_FLOAT32, NCCL_SUM, comm, None,
        )
        _require_success(lib, comm, result, "allreduce after reduce")
        assert recv.value == 5
        _require_success(lib, comm, lib.ncclCommDestroy(comm), "destroy")

    with ThreadPoolExecutor(2) as pool:
        list(pool.map(worker, range(2)))


def check_coordinator(*, reduce: bool) -> int:
    if not COORDINATOR_BIN.exists():
        print(f"missing coordinator binary: {COORDINATOR_BIN}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(
        prefix="fgpu-coord-",
        dir="/tmp",
    ) as tmpdir:
        socket_path = os.path.join(tmpdir, "coordinator.sock")
        report_path = Path(tmpdir) / "cluster-report.json"
        markdown_path = Path(tmpdir) / "cluster-report.md"
        env = dict(os.environ)
        env["FAKEGPU_CLUSTER_REPORT_PATH"] = str(report_path)
        env["FAKEGPU_CLUSTER_REPORT_MARKDOWN_PATH"] = str(markdown_path)
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

            response = request(socket_path, "PING")
            if "OK" not in response or "status=ready" not in response or "version=1" not in response:
                raise AssertionError(f"unexpected ping response: {response}")

            if reduce:
                check_socket_reduce(socket_path)

            response = request(socket_path, "SHUTDOWN")
            if "OK" not in response or "status=shutting_down" not in response:
                raise AssertionError(f"unexpected shutdown response: {response}")

            proc.wait(timeout=2.0)
            if os.path.exists(socket_path):
                raise AssertionError("socket path still exists after shutdown")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("schema_version") != "cluster_report.v1":
                raise AssertionError(f"unexpected empty report: {report}")
            if reduce:
                assert report["collectives"]["reduce"]["calls"] == 1
                assert report["collectives"]["all_reduce"]["calls"] == 1
                return 0
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

            return 0
        finally:
            if proc.poll() is None:
                proc.kill()
            stdout, stderr = proc.communicate()
            if proc.returncode not in (None, 0):
                sys.stderr.write(stdout)
                sys.stderr.write(stderr)


def main() -> int:
    result = check_coordinator(reduce=False) or check_coordinator(reduce=True)
    if result == 0:
        print("coordinator smoke test passed, including a socket reduce followed by all-reduce")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
