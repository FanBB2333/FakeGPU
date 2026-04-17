#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import ctypes
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
COORDINATOR_BIN = REPO_ROOT / "build" / "fakegpu-coordinator"
NCCL_LIB = REPO_ROOT / "build" / "libnccl.so.2"
CLUSTER_CONFIG_2R = REPO_ROOT / "verification" / "data" / "cluster_proxy_2r.yaml"
CLUSTER_CONFIG_1R = REPO_ROOT / "verification" / "data" / "cluster_proxy_1r.yaml"


def find_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request_tcp(host: str, port: int, payload: str) -> str:
    with socket.create_connection((host, port), timeout=1.0) as sock:
        sock.sendall((payload + "\n").encode("utf-8"))
        data = b""
        while not data.endswith(b"\n"):
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
    return data.decode("utf-8").strip()


def wait_for_ready(host: str, port: int, timeout_s: float = 3.0) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            response = request_tcp(host, port, "PING")
            if response.startswith("OK ") and "status=ready" in response:
                return
        except OSError as exc:
            last_error = exc
        time.sleep(0.05)
    raise TimeoutError(f"tcp coordinator did not become ready on {host}:{port}: {last_error}")


def write_report(report_path: Path, payload: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_report(report_path: Path) -> dict[str, Any]:
    return json.loads(report_path.read_text(encoding="utf-8"))


def run_success_worker(args: argparse.Namespace) -> int:
    report: dict[str, Any] = {
        "mode": "success",
        "rank": args.rank,
        "world_size": args.world_size,
        "status": "starting",
        "stage": "bootstrap",
    }

    class ncclUniqueId(ctypes.Structure):
        _fields_ = [("internal", ctypes.c_char * 128)]

    ncclComm_t = ctypes.c_void_p
    comm = ncclComm_t()
    try:
        lib = ctypes.CDLL(str(args.nccl_lib), mode=ctypes.RTLD_GLOBAL)
        report["stage"] = "lib_loaded"
        write_report(args.report_path, report)

        lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(ncclUniqueId)]
        lib.ncclGetUniqueId.restype = ctypes.c_int
        lib.ncclCommInitRank.argtypes = [
            ctypes.POINTER(ncclComm_t),
            ctypes.c_int,
            ncclUniqueId,
            ctypes.c_int,
        ]
        lib.ncclCommInitRank.restype = ctypes.c_int
        lib.ncclAllReduce.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ncclComm_t,
            ctypes.c_void_p,
        ]
        lib.ncclAllReduce.restype = ctypes.c_int
        lib.ncclCommDestroy.argtypes = [ncclComm_t]
        lib.ncclCommDestroy.restype = ctypes.c_int

        unique_id = ncclUniqueId()
        if args.rank == 0:
            result = lib.ncclGetUniqueId(ctypes.byref(unique_id))
            report["unique_id_result"] = int(result)
            if result != 0:
                raise AssertionError(f"ncclGetUniqueId failed with code {result}")
            payload = ctypes.string_at(ctypes.byref(unique_id), ctypes.sizeof(unique_id))
            args.uid_path.write_text(base64.b64encode(payload).decode("ascii"), encoding="utf-8")
        else:
            deadline = time.time() + 10.0
            while time.time() < deadline and not args.uid_path.exists():
                time.sleep(0.05)
            if not args.uid_path.exists():
                raise TimeoutError(f"timed out waiting for unique id file: {args.uid_path}")
            raw = base64.b64decode(args.uid_path.read_text(encoding="utf-8").strip())
            ctypes.memmove(
                ctypes.addressof(unique_id),
                raw,
                min(len(raw), ctypes.sizeof(unique_id)),
            )
        report["stage"] = "uid_ready"
        write_report(args.report_path, report)

        init_result = lib.ncclCommInitRank(ctypes.byref(comm), args.world_size, unique_id, args.rank)
        report["comm_init_result"] = int(init_result)
        if init_result != 0:
            raise AssertionError(f"ncclCommInitRank failed with code {init_result}")
        report["stage"] = "comm_ready"
        write_report(args.report_path, report)

        send = (ctypes.c_float * 4)(*([float(args.rank + 1)] * 4))
        recv = (ctypes.c_float * 4)()
        result = lib.ncclAllReduce(
            ctypes.cast(send, ctypes.c_void_p),
            ctypes.cast(recv, ctypes.c_void_p),
            4,
            7,
            0,
            comm,
            None,
        )
        report["all_reduce_result"] = int(result)
        if result != 0:
            raise AssertionError(f"ncclAllReduce failed with code {result}")
        values = [float(recv[index]) for index in range(4)]
        report["all_reduce_value"] = values
        expected = [float(sum(range(1, args.world_size + 1)))] * 4
        report["all_reduce_expected"] = expected
        if values != expected:
            raise AssertionError(f"unexpected all_reduce output: {values} != {expected}")

        destroy_result = lib.ncclCommDestroy(comm)
        report["destroy_result"] = int(destroy_result)
        if destroy_result != 0:
            raise AssertionError(f"ncclCommDestroy failed with code {destroy_result}")
        comm = ncclComm_t()

        report["status"] = "success"
        report["stage"] = "done"
        write_report(args.report_path, report)
        return 0
    except Exception as exc:  # noqa: BLE001
        report["status"] = "error"
        report["exception_type"] = type(exc).__name__
        report["exception_message"] = str(exc)
        report["traceback"] = traceback.format_exc()
        write_report(args.report_path, report)
        return 1


def run_disconnect_worker(args: argparse.Namespace) -> int:
    report: dict[str, Any] = {
        "mode": "disconnect",
        "status": "starting",
        "stage": "bootstrap",
    }

    class ncclUniqueId(ctypes.Structure):
        _fields_ = [("internal", ctypes.c_char * 128)]

    ncclComm_t = ctypes.c_void_p
    comm = ncclComm_t()
    try:
        lib = ctypes.CDLL(str(args.nccl_lib), mode=ctypes.RTLD_GLOBAL)
        report["stage"] = "lib_loaded"
        write_report(args.report_path, report)

        lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(ncclUniqueId)]
        lib.ncclGetUniqueId.restype = ctypes.c_int
        lib.ncclCommInitRank.argtypes = [
            ctypes.POINTER(ncclComm_t),
            ctypes.c_int,
            ncclUniqueId,
            ctypes.c_int,
        ]
        lib.ncclCommInitRank.restype = ctypes.c_int
        lib.ncclCommDestroy.argtypes = [ncclComm_t]
        lib.ncclCommDestroy.restype = ctypes.c_int
        lib.ncclCommAbort.argtypes = [ncclComm_t]
        lib.ncclCommAbort.restype = ctypes.c_int
        lib.ncclGetLastError.argtypes = [ncclComm_t]
        lib.ncclGetLastError.restype = ctypes.c_char_p

        unique_id = ncclUniqueId()
        result = lib.ncclGetUniqueId(ctypes.byref(unique_id))
        report["unique_id_result"] = int(result)
        if result != 0:
            raise AssertionError(f"ncclGetUniqueId failed with code {result}")

        init_result = lib.ncclCommInitRank(ctypes.byref(comm), 1, unique_id, 0)
        report["comm_init_result"] = int(init_result)
        if init_result != 0:
            raise AssertionError(f"ncclCommInitRank failed with code {init_result}")
        report["stage"] = "comm_ready"
        args.ready_path.write_text("ready\n", encoding="utf-8")
        write_report(args.report_path, report)

        deadline = time.time() + 10.0
        while time.time() < deadline and not args.continue_path.exists():
            time.sleep(0.05)
        if not args.continue_path.exists():
            raise TimeoutError(f"timed out waiting for continue flag: {args.continue_path}")

        destroy_result = lib.ncclCommDestroy(comm)
        report["destroy_result"] = int(destroy_result)
        last_error = lib.ncclGetLastError(comm)
        report["destroy_last_error"] = last_error.decode("utf-8", errors="replace") if last_error else ""
        abort_result = lib.ncclCommAbort(comm)
        report["abort_result"] = int(abort_result)
        comm = ncclComm_t()

        report["status"] = "success"
        report["stage"] = "done"
        write_report(args.report_path, report)
        return 0
    except Exception as exc:  # noqa: BLE001
        report["status"] = "error"
        report["exception_type"] = type(exc).__name__
        report["exception_message"] = str(exc)
        report["traceback"] = traceback.format_exc()
        write_report(args.report_path, report)
        return 1


def worker_entry() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-mode", choices=["success", "disconnect"], required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--uid-path", type=Path)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--ready-path", type=Path)
    parser.add_argument("--continue-path", type=Path)
    parser.add_argument("--nccl-lib", type=Path, required=True)
    args = parser.parse_args()
    if args.worker_mode == "success":
        if args.uid_path is None:
            raise SystemExit("--uid-path is required for success worker")
        return run_success_worker(args)
    if args.ready_path is None or args.continue_path is None:
        raise SystemExit("--ready-path and --continue-path are required for disconnect worker")
    return run_disconnect_worker(args)


def spawn_success_workers(tmpdir: Path, endpoint: str) -> list[dict[str, Any]]:
    uid_path = tmpdir / "uid.txt"
    reports: list[Path] = []
    procs: list[subprocess.Popen[str]] = []
    env_base = {
        "FAKEGPU_DIST_MODE": "simulate",
        "FAKEGPU_COORDINATOR_TRANSPORT": "tcp",
        "FAKEGPU_COORDINATOR_ADDR": endpoint,
        "FAKEGPU_CLUSTER_CONFIG": str(CLUSTER_CONFIG_2R),
    }

    for rank in range(2):
        report_path = tmpdir / f"success_rank_{rank}.json"
        reports.append(report_path)
        env = dict(os.environ)
        env.setdefault("XONSH_HISTORY_BACKEND", "dummy")
        env.update(env_base)
        proc = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-mode",
                "success",
                "--rank",
                str(rank),
                "--world-size",
                "2",
                "--uid-path",
                str(uid_path),
                "--report-path",
                str(report_path),
                "--nccl-lib",
                str(NCCL_LIB),
            ],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        procs.append(proc)

    failures: list[str] = []
    payloads: list[dict[str, Any]] = []
    for rank, (proc, report_path) in enumerate(zip(procs, reports)):
        stdout, stderr = proc.communicate(timeout=20)
        if proc.returncode != 0:
            failures.append(
                f"rank {rank} failed with exit code {proc.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"
            )
            continue
        payloads.append(load_report(report_path))
    if failures:
        raise RuntimeError("\n\n".join(failures))
    return payloads


def run_disconnect_case(tmpdir: Path, endpoint: str, host: str, port: int) -> dict[str, Any]:
    report_path = tmpdir / "disconnect_report.json"
    ready_path = tmpdir / "disconnect_ready.flag"
    continue_path = tmpdir / "disconnect_continue.flag"
    env = dict(os.environ)
    env.setdefault("XONSH_HISTORY_BACKEND", "dummy")
    env.update(
        {
            "FAKEGPU_DIST_MODE": "simulate",
            "FAKEGPU_COORDINATOR_TRANSPORT": "tcp",
            "FAKEGPU_COORDINATOR_ADDR": endpoint,
            "FAKEGPU_CLUSTER_CONFIG": str(CLUSTER_CONFIG_1R),
        }
    )
    proc = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-mode",
            "disconnect",
            "--report-path",
            str(report_path),
            "--ready-path",
            str(ready_path),
            "--continue-path",
            str(continue_path),
            "--nccl-lib",
            str(NCCL_LIB),
        ],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    deadline = time.time() + 10.0
    while time.time() < deadline and not ready_path.exists():
        time.sleep(0.05)
    if not ready_path.exists():
        stdout, stderr = proc.communicate(timeout=5)
        raise RuntimeError(
            "disconnect worker did not become ready\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )

    shutdown_response = request_tcp(host, port, "SHUTDOWN")
    if not shutdown_response.startswith("OK "):
        proc.kill()
        stdout, stderr = proc.communicate(timeout=5)
        raise RuntimeError(
            f"failed to stop tcp coordinator: {shutdown_response}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )

    continue_path.write_text("continue\n", encoding="utf-8")
    stdout, stderr = proc.communicate(timeout=20)
    if proc.returncode != 0:
        raise RuntimeError(
            f"disconnect worker failed with exit code {proc.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )
    return load_report(report_path)


def main() -> int:
    if "--worker-mode" in sys.argv:
        return worker_entry()

    if not COORDINATOR_BIN.exists():
        print(f"missing coordinator binary: {COORDINATOR_BIN}", file=sys.stderr)
        return 2
    if not NCCL_LIB.exists():
        print(f"missing fake NCCL library: {NCCL_LIB}", file=sys.stderr)
        return 2

    host = "127.0.0.1"
    with tempfile.TemporaryDirectory(prefix="fakegpu-remote-coordinator-") as tmp:
        tmpdir = Path(tmp)
        port = find_free_tcp_port()
        endpoint = f"{host}:{port}"
        cluster_report = tmpdir / "remote_cluster_report.json"
        coord_env = dict(os.environ)
        coord_env.update(
            {
                "FAKEGPU_DIST_MODE": "simulate",
                "FAKEGPU_COORDINATOR_TRANSPORT": "tcp",
                "FAKEGPU_COORDINATOR_ADDR": endpoint,
                "FAKEGPU_CLUSTER_CONFIG": str(CLUSTER_CONFIG_2R),
                "FAKEGPU_CLUSTER_REPORT_PATH": str(cluster_report),
            }
        )
        proc = subprocess.Popen(
            [str(COORDINATOR_BIN), "--transport", "tcp", "--address", endpoint],
            cwd=REPO_ROOT,
            env=coord_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            wait_for_ready(host, port)
            success_reports = spawn_success_workers(tmpdir, endpoint)
            shutdown_response = request_tcp(host, port, "SHUTDOWN")
            if not shutdown_response.startswith("OK "):
                raise AssertionError(f"unexpected SHUTDOWN response: {shutdown_response}")
            proc.wait(timeout=5)
            if proc.returncode != 0:
                stdout, stderr = proc.communicate()
                raise RuntimeError(
                    f"tcp coordinator exited with code {proc.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"
                )

            check_result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "verification" / "check_cluster_report.py"),
                    "--path",
                    str(cluster_report),
                    "--expect-world-size",
                    "2",
                    "--expect-node-count",
                    "2",
                    "--expect-collective",
                    "all_reduce",
                    "--expect-links",
                    "--min-ranks",
                    "2",
                ],
                cwd=REPO_ROOT,
                env={**os.environ, "XONSH_HISTORY_BACKEND": os.environ.get("XONSH_HISTORY_BACKEND", "dummy")},
                capture_output=True,
                text=True,
                check=False,
            )
            if check_result.returncode != 0:
                raise RuntimeError(
                    "cluster report validation failed\n"
                    f"stdout:\n{check_result.stdout}\nstderr:\n{check_result.stderr}"
                )

            cluster_payload = json.loads(cluster_report.read_text(encoding="utf-8"))
            if cluster_payload["cluster"]["coordinator_transport"] != "tcp":
                raise AssertionError(
                    f"expected cluster transport tcp, got {cluster_payload['cluster']['coordinator_transport']}"
                )

            disconnect_port = find_free_tcp_port()
            disconnect_endpoint = f"{host}:{disconnect_port}"
            disconnect_env = dict(os.environ)
            disconnect_env.update(
                {
                    "FAKEGPU_DIST_MODE": "simulate",
                    "FAKEGPU_COORDINATOR_TRANSPORT": "tcp",
                    "FAKEGPU_COORDINATOR_ADDR": disconnect_endpoint,
                    "FAKEGPU_CLUSTER_CONFIG": str(CLUSTER_CONFIG_1R),
                }
            )
            disconnect_proc = subprocess.Popen(
                [str(COORDINATOR_BIN), "--transport", "tcp", "--address", disconnect_endpoint],
                cwd=REPO_ROOT,
                env=disconnect_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                wait_for_ready(host, disconnect_port)
                disconnect_report = run_disconnect_case(
                    tmpdir,
                    disconnect_endpoint,
                    host,
                    disconnect_port,
                )
            finally:
                if disconnect_proc.poll() is None:
                    disconnect_proc.kill()
                disconnect_proc.communicate(timeout=5)

            if disconnect_report.get("destroy_result") == 0:
                raise AssertionError("expected ncclCommDestroy to fail after coordinator shutdown")
            if "connect() failed" not in disconnect_report.get("destroy_last_error", ""):
                raise AssertionError(
                    "expected clear connect() failed error after coordinator shutdown, "
                    f"got: {disconnect_report.get('destroy_last_error')}"
                )
            if disconnect_report.get("abort_result") != 0:
                raise AssertionError(
                    f"expected ncclCommAbort to succeed after shutdown, got {disconnect_report.get('abort_result')}"
                )

            print(
                json.dumps(
                    {
                        "success_reports": success_reports,
                        "disconnect_report": disconnect_report,
                        "cluster_report": str(cluster_report),
                        "endpoint": endpoint,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.communicate(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
