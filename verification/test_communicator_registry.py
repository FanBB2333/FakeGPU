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


def run_parallel_init(socket_path: str, unique_id: str, world_size: int, timeout_ms: int) -> list[str]:
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

    return responses


def assert_all_ok(responses: list[str], expected_world_size: int) -> int:
    comm_ids: set[int] = set()
    for rank, response in enumerate(responses):
        if not response.startswith("OK "):
            raise AssertionError(f"rank {rank} returned non-OK response: {response}")
        fields = parse_fields(response)
        if fields.get("seqno") != "0":
            raise AssertionError(f"rank {rank} returned unexpected seqno: {response}")
        if fields.get("world_size") != str(expected_world_size):
            raise AssertionError(f"rank {rank} returned unexpected world size: {response}")
        comm_ids.add(int(fields["comm_id"]))
    if len(comm_ids) != 1:
        raise AssertionError(f"expected a single comm_id, got: {sorted(comm_ids)}")
    return next(iter(comm_ids))


def main() -> int:
    if not COORDINATOR_BIN.exists():
        print(f"missing coordinator binary: {COORDINATOR_BIN}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="fakegpu-coordinator-registry-") as tmpdir:
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

            hello_response = request(socket_path, "HELLO client=registry_test")
            if not hello_response.startswith("OK ") or "version=1" not in hello_response:
                raise AssertionError(f"unexpected HELLO response: {hello_response}")

            comm_id = assert_all_ok(run_parallel_init(socket_path, "two_rank_group", 2, 1000), 2)
            for rank in range(2):
                destroy_response = request(socket_path, f"DESTROY_COMM comm_id={comm_id} rank={rank}")
                if not destroy_response.startswith("OK "):
                    raise AssertionError(f"destroy failed for rank {rank}: {destroy_response}")

            comm_id = assert_all_ok(run_parallel_init(socket_path, "four_rank_group", 4, 1000), 4)
            for rank in range(4):
                destroy_response = request(socket_path, f"DESTROY_COMM comm_id={comm_id} rank={rank}")
                if not destroy_response.startswith("OK "):
                    raise AssertionError(f"destroy failed for rank {rank}: {destroy_response}")

            responses: list[str] = [""] * 2

            def rank_zero() -> None:
                responses[0] = request(
                    socket_path,
                    "INIT_COMM unique_id=dup_group world_size=2 rank=0 timeout_ms=1000",
                )

            thread = threading.Thread(target=rank_zero)
            thread.start()
            time.sleep(0.1)
            responses[1] = request(
                socket_path,
                "INIT_COMM unique_id=dup_group world_size=2 rank=0 timeout_ms=1000",
            )
            thread.join(timeout=2.0)
            if thread.is_alive():
                raise AssertionError("duplicate rank init thread did not finish")
            if "code=duplicate_rank" not in responses[1]:
                raise AssertionError(f"expected duplicate_rank error, got: {responses[1]}")
            if "code=duplicate_rank" not in responses[0]:
                raise AssertionError(f"expected group failure for waiting rank, got: {responses[0]}")

            mismatch_first = {}

            def rank_zero_mismatch() -> None:
                mismatch_first["response"] = request(
                    socket_path,
                    "INIT_COMM unique_id=mismatch_group world_size=2 rank=0 timeout_ms=1000",
                )

            thread = threading.Thread(target=rank_zero_mismatch)
            thread.start()
            time.sleep(0.1)
            mismatch_second = request(
                socket_path,
                "INIT_COMM unique_id=mismatch_group world_size=4 rank=1 timeout_ms=1000",
            )
            thread.join(timeout=2.0)
            if thread.is_alive():
                raise AssertionError("world size mismatch thread did not finish")
            if "code=world_size_mismatch" not in mismatch_second:
                raise AssertionError(f"expected world_size_mismatch error, got: {mismatch_second}")
            if "code=world_size_mismatch" not in mismatch_first.get("response", ""):
                raise AssertionError(f"expected group failure for waiting rank, got: {mismatch_first!r}")

            timeout_response = request(
                socket_path,
                "INIT_COMM unique_id=timeout_group world_size=2 rank=0 timeout_ms=200",
            )
            if "code=timeout_waiting_for_ranks" not in timeout_response:
                raise AssertionError(f"expected timeout_waiting_for_ranks, got: {timeout_response}")

            shutdown_response = request(socket_path, "SHUTDOWN")
            if not shutdown_response.startswith("OK "):
                raise AssertionError(f"unexpected SHUTDOWN response: {shutdown_response}")

            proc.wait(timeout=2.0)
            print("communicator registry test passed")
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
