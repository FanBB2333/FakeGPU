#!/usr/bin/env python3
"""Drive the fake libnccl through ctypes and check its error paths."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = Path(os.environ.get("FAKEGPU_BUILD_DIR", REPO_ROOT / "build"))
COORDINATOR_BIN = BUILD_DIR / "fakegpu-coordinator"
LIBNCCL = BUILD_DIR / ("libnccl.dylib" if sys.platform == "darwin" else "libnccl.so")
NCCL_SYSTEM_ERROR = 2
NCCL_FLOAT32 = 7
NCCL_SUM = 0
SPLIT_WORLD = 3
REDUCE_WORLD = 2


class UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * 128)]


def load_nccl() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(LIBNCCL))
    lib.ncclGetLastError.restype = ctypes.c_char_p
    lib.ncclGetLastError.argtypes = [ctypes.c_void_p]
    return lib


def child_partial_loader() -> int:
    lib = load_nccl()
    for attempt in (1, 2):
        version = ctypes.c_int(0)
        rc = lib.ncclGetVersion(ctypes.byref(version))
        if rc != NCCL_SYSTEM_ERROR:
            message = lib.ncclGetLastError(None).decode()
            print(f"ncclGetVersion call {attempt} returned {rc} ({message})", file=sys.stderr)
            return 1
    return 0


def check_partial_real_nccl() -> None:
    compiler = shutil.which("cc")
    if compiler is None:
        print("no C compiler found; skipped the partial real NCCL check")
        return
    with tempfile.TemporaryDirectory(prefix="fgpu-nccl-") as tmpdir:
        source = Path(tmpdir) / "partial_nccl.c"
        source.write_text("int ncclGetVersion(int* v) { *v = 99999; return 0; }\n", encoding="utf-8")
        library = Path(tmpdir) / "libpartialnccl.so"
        subprocess.run([compiler, "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
        env = dict(os.environ, FAKEGPU_DIST_MODE="passthrough", FAKEGPU_REAL_NCCL_PATH=str(library))
        result = subprocess.run([sys.executable, __file__, "--partial-loader"], env=env)
        if result.returncode != 0:
            raise AssertionError("a real NCCL library with missing symbols was treated as loaded")


def child_split_rank(rank: int) -> int:
    lib = load_nccl()
    lib.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, UniqueId, ctypes.c_int]
    comm = ctypes.c_void_p()
    unique_id = UniqueId(internal=os.environ["NCCL_SMOKE_UNIQUE_ID"].encode())
    rc = lib.ncclCommInitRank(ctypes.byref(comm), SPLIT_WORLD, unique_id, rank)
    if rc != 0:
        print(f"rank {rank}: ncclCommInitRank returned {rc}", file=sys.stderr)
        return 1
    if rank != 0:
        time.sleep(1.0)
        return 0
    newcomm = ctypes.c_void_p()
    rc = lib.ncclCommSplit(comm, 0, 0, ctypes.byref(newcomm), None)
    if rc != NCCL_SYSTEM_ERROR:
        message = lib.ncclGetLastError(comm).decode()
        print(f"ncclCommSplit without peers returned {rc} ({message})", file=sys.stderr)
        return 1
    return 0


def child_reduce_rank(rank: int) -> int:
    lib = load_nccl()
    lib.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, UniqueId, ctypes.c_int]
    comm = ctypes.c_void_p()
    unique_id = UniqueId(internal=os.environ["NCCL_SMOKE_UNIQUE_ID"].encode())
    rc = lib.ncclCommInitRank(ctypes.byref(comm), REDUCE_WORLD, unique_id, rank)
    if rc != 0:
        print(f"rank {rank}: ncclCommInitRank returned {rc}", file=sys.stderr)
        return 1
    send = (ctypes.c_float * 3)(rank + 1.0, rank + 2.0, rank + 3.0)
    recv = (ctypes.c_float * 3)()
    # The second reduce checks that the communicator is still in step.
    for _ in range(2):
        rc = lib.ncclReduce(send, recv, ctypes.c_size_t(3), NCCL_FLOAT32, NCCL_SUM, 0, comm, None)
        if rc != 0:
            message = lib.ncclGetLastError(comm).decode()
            print(f"rank {rank}: ncclReduce returned {rc} ({message})", file=sys.stderr)
            return 1
    if rank == 0 and list(recv) != [3.0, 5.0, 7.0]:
        print(f"rank 0: ncclReduce produced {list(recv)}", file=sys.stderr)
        return 1
    return 0


def run_ranks(child: str, world: int, timeout_ms: int, extra_env: dict[str, str] | None = None) -> bool:
    """Run ``world`` rank processes against a fresh coordinator; True if all succeed."""

    with tempfile.TemporaryDirectory(prefix="fgpu-nccl-", dir="/tmp") as tmpdir:
        socket_path = os.path.join(tmpdir, "coordinator.sock")
        env = dict(os.environ, FAKEGPU_CLUSTER_REPORT_PATH=os.path.join(tmpdir, "cluster-report.json"))
        coordinator = subprocess.Popen(
            [str(COORDINATOR_BIN), "--transport", "unix", "--address", socket_path],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + 5.0
            while not os.path.exists(socket_path):
                if time.monotonic() > deadline or coordinator.poll() is not None:
                    raise AssertionError("coordinator did not create its socket")
                time.sleep(0.05)
            env.update(
                FAKEGPU_DIST_MODE="simulate",
                FAKEGPU_COORDINATOR_TRANSPORT="unix",
                FAKEGPU_COORDINATOR_ADDR=socket_path,
                FAKEGPU_COORDINATOR_TIMEOUT_MS=str(timeout_ms),
                NCCL_SMOKE_UNIQUE_ID=f"nccl-smoke{child}-{os.getpid()}",
                **(extra_env or {}),
            )
            ranks = [
                subprocess.Popen([sys.executable, __file__, child, str(rank)], env=env)
                for rank in range(world)
            ]
            return all([rank.wait(timeout=30) == 0 for rank in ranks])
        finally:
            coordinator.kill()
            coordinator.wait()


def check_split_timeout() -> None:
    if not run_ranks("--split-rank", SPLIT_WORLD, 300):
        raise AssertionError("ncclCommSplit timeout was not reported as ncclSystemError")


def check_reduce_over_socket_payloads() -> None:
    if not run_ranks("--reduce-rank", REDUCE_WORLD, 3000, {"FAKEGPU_STAGING_FORCE_SOCKET": "1"}):
        raise AssertionError("ncclReduce failed with socket payloads")


def main() -> int:
    if sys.argv[1:] == ["--partial-loader"]:
        return child_partial_loader()
    if sys.argv[1:2] == ["--split-rank"]:
        return child_split_rank(int(sys.argv[2]))
    if sys.argv[1:2] == ["--reduce-rank"]:
        return child_reduce_rank(int(sys.argv[2]))
    if not LIBNCCL.exists():
        print(f"missing NCCL library: {LIBNCCL}", file=sys.stderr)
        return 2
    check_partial_real_nccl()
    check_split_timeout()
    check_reduce_over_socket_payloads()
    print("nccl stubs smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
