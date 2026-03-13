#!/usr/bin/env python3

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COORDINATOR_BIN = REPO_ROOT / "build" / "fakegpu-coordinator"


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


def parse_fields(response: str) -> dict[str, str]:
    tokens = response.split()
    fields: dict[str, str] = {}
    for token in tokens[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def run_parallel_init(socket_path: str, unique_id: str, world_size: int, timeout_ms: int) -> int:
    responses = [""] * world_size
    threads: list[threading.Thread] = []

    def worker(rank: int) -> None:
        responses[rank] = request(
            socket_path,
            f"INIT_COMM unique_id={unique_id} world_size={world_size} rank={rank} timeout_ms={timeout_ms}",
        )

    for rank in range(world_size):
        thread = threading.Thread(target=worker, args=(rank,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join(timeout=5.0)
        if thread.is_alive():
            raise AssertionError("init worker thread did not finish")

    comm_ids: set[int] = set()
    for rank, response in enumerate(responses):
        if not response.startswith("OK "):
            raise AssertionError(f"rank {rank} init failed: {response}")
        fields = parse_fields(response)
        comm_ids.add(int(fields["comm_id"]))
    if len(comm_ids) != 1:
        raise AssertionError(f"expected one communicator id, got {sorted(comm_ids)}")
    return next(iter(comm_ids))


def main() -> int:
    if not COORDINATOR_BIN.exists():
        print(f"missing coordinator binary: {COORDINATOR_BIN}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="fakegpu-barrier-") as tmpdir:
        socket_path = os.path.join(tmpdir, "coordinator.sock")
        proc = subprocess.Popen(
            [str(COORDINATOR_BIN), "--transport", "unix", "--address", socket_path],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            deadline = time.time() + 1.0
            while time.time() < deadline:
                if os.path.exists(socket_path):
                    break
                time.sleep(0.05)
            if not os.path.exists(socket_path):
                raise AssertionError("socket was not created within 1 second")

            comm_id = run_parallel_init(socket_path, "barrier_ok_group", 2, 1000)

            success_responses = ["", ""]
            success_threads: list[threading.Thread] = []

            def success_worker(rank: int) -> None:
                success_responses[rank] = request(
                    socket_path,
                    f"BARRIER comm_id={comm_id} rank={rank} seqno=1 timeout_ms=1000",
                )

            for rank in range(2):
                thread = threading.Thread(target=success_worker, args=(rank,))
                thread.start()
                success_threads.append(thread)

            for thread in success_threads:
                thread.join(timeout=5.0)
                if thread.is_alive():
                    raise AssertionError("success barrier worker did not finish")

            for rank, response in enumerate(success_responses):
                if not response.startswith("OK "):
                    raise AssertionError(f"rank {rank} barrier failed: {response}")
                fields = parse_fields(response)
                if fields.get("op") != "barrier" or fields.get("seqno") != "1":
                    raise AssertionError(f"rank {rank} returned unexpected barrier response: {response}")

            timed_out_comm_id = run_parallel_init(socket_path, "barrier_timeout_group", 2, 1000)
            timeout_responses = ["", ""]

            def early_rank() -> None:
                timeout_responses[0] = request(
                    socket_path,
                    f"BARRIER comm_id={timed_out_comm_id} rank=0 seqno=1 timeout_ms=200",
                )

            thread = threading.Thread(target=early_rank)
            thread.start()
            time.sleep(0.4)
            timeout_responses[1] = request(
                socket_path,
                f"BARRIER comm_id={timed_out_comm_id} rank=1 seqno=1 timeout_ms=200",
            )
            thread.join(timeout=5.0)
            if thread.is_alive():
                raise AssertionError("timeout barrier worker did not finish")

            for rank, response in enumerate(timeout_responses):
                if "code=timeout_waiting_for_barrier" not in response:
                    raise AssertionError(f"rank {rank} should receive timeout_waiting_for_barrier: {response}")

            shutdown_response = request(socket_path, "SHUTDOWN")
            if not shutdown_response.startswith("OK "):
                raise AssertionError(f"unexpected SHUTDOWN response: {shutdown_response}")

            proc.wait(timeout=2.0)
            print("coordinator barrier test passed")
            return 0
        finally:
            if proc.poll() is None:
                proc.kill()
            stdout, stderr = proc.communicate()
            if proc.returncode not in (None, 0):
                sys.stderr.write(stdout)
                sys.stderr.write(stderr)


if __name__ == "__main__":
    raise SystemExit(main())
