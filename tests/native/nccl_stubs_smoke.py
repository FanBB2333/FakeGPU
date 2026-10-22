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
LIBCUDART = BUILD_DIR / ("libcudart.dylib" if sys.platform == "darwin" else "libcudart.so.12")
NCCL_SUCCESS = 0
NCCL_SYSTEM_ERROR = 2
NCCL_INVALID_ARGUMENT = 4
NCCL_INVALID_USAGE = 5
NCCL_FLOAT32 = 7
NCCL_SUM = 0
NCCL_SCALAR_HOST_IMMEDIATE = 1
# NCCL_SIM_INFO_INITIALIZER magic from nccl.h.
NCCL_SIM_INFO_MAGIC = 0x74685283
SPLIT_WORLD = 3
REDUCE_WORLD = 2


class UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * 128)]


class SimInfo(ctypes.Structure):
    # ncclSimInfo_v22200 from nccl.h.
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("magic", ctypes.c_uint),
        ("version", ctypes.c_uint),
        ("estimatedTime", ctypes.c_float),
    ]


def load_nccl() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(LIBNCCL))
    lib.ncclGetLastError.restype = ctypes.c_char_p
    lib.ncclGetLastError.argtypes = [ctypes.c_void_p]
    lib.ncclCommDestroy.argtypes = [ctypes.c_void_p]
    lib.ncclCommFinalize.argtypes = [ctypes.c_void_p]
    lib.ncclRedOpDestroy.argtypes = [ctypes.c_int, ctypes.c_void_p]
    lib.ncclRedOpCreatePreMulSum.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    return lib


def expect_rc(actual: int, expected: int, what: str) -> None:
    if actual != expected:
        raise AssertionError(f"{what} returned {actual}, expected {expected}")


def child_lifecycle() -> int:
    """Return codes that real NCCL gives without any communicator traffic."""

    lib = load_nccl()
    # group.cc, ncclGroupEndInternal: an unmatched end is ncclInvalidUsage.
    expect_rc(lib.ncclGroupEnd(), NCCL_INVALID_USAGE, "unmatched ncclGroupEnd")
    expect_rc(lib.ncclGroupSimulateEnd(None), NCCL_INVALID_USAGE, "unmatched ncclGroupSimulateEnd")
    for _ in range(2):
        expect_rc(lib.ncclGroupStart(), NCCL_SUCCESS, "ncclGroupStart")
    expect_rc(lib.ncclGroupEnd(), NCCL_SUCCESS, "inner ncclGroupEnd")
    expect_rc(lib.ncclGroupEnd(), NCCL_SUCCESS, "outer ncclGroupEnd")
    expect_rc(lib.ncclGroupEnd(), NCCL_INVALID_USAGE, "third ncclGroupEnd")
    # An uninitialized ncclSimInfo_t is rejected; an initialized one is not.
    sim_info = SimInfo(size=ctypes.sizeof(SimInfo))
    lib.ncclGroupStart()
    expect_rc(lib.ncclGroupSimulateEnd(ctypes.byref(sim_info)), NCCL_INVALID_ARGUMENT, "ncclGroupSimulateEnd without magic")
    sim_info.magic = NCCL_SIM_INFO_MAGIC
    sim_info.estimatedTime = -1.0
    lib.ncclGroupStart()
    expect_rc(lib.ncclGroupSimulateEnd(ctypes.byref(sim_info)), NCCL_SUCCESS, "ncclGroupSimulateEnd")
    # init.cc: destroying or finalizing a null communicator is a no-op.
    expect_rc(lib.ncclCommDestroy(None), NCCL_SUCCESS, "ncclCommDestroy(NULL)")
    expect_rc(lib.ncclCommFinalize(None), NCCL_SUCCESS, "ncclCommFinalize(NULL)")
    # enqueue.cc, ncclRedOpDestroy: built-in operators and null communicators.
    expect_rc(lib.ncclRedOpDestroy(NCCL_SUM, None), NCCL_INVALID_ARGUMENT, "ncclRedOpDestroy(ncclSum)")
    expect_rc(lib.ncclRedOpDestroy(64, None), NCCL_INVALID_ARGUMENT, "ncclRedOpDestroy without a communicator")
    return 0


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
    # A preMulSum operator can be destroyed once; NCCL rejects the second
    # destroy as unknown to the communicator.
    op = ctypes.c_int(-1)
    scalar = ctypes.c_float(0.5)
    expect_rc(
        lib.ncclRedOpCreatePreMulSum(ctypes.byref(op), ctypes.byref(scalar), NCCL_FLOAT32, NCCL_SCALAR_HOST_IMMEDIATE, comm),
        NCCL_SUCCESS,
        "ncclRedOpCreatePreMulSum",
    )
    expect_rc(lib.ncclRedOpDestroy(op.value, comm), NCCL_SUCCESS, "ncclRedOpDestroy")
    expect_rc(lib.ncclRedOpDestroy(op.value, comm), NCCL_INVALID_ARGUMENT, "second ncclRedOpDestroy")
    # init.cc, ncclCommFinalize: a second finalize is ncclInvalidArgument.
    expect_rc(lib.ncclCommFinalize(comm), NCCL_SUCCESS, "ncclCommFinalize")
    expect_rc(lib.ncclCommFinalize(comm), NCCL_INVALID_ARGUMENT, "second ncclCommFinalize")
    expect_rc(lib.ncclCommDestroy(comm), NCCL_SUCCESS, "ncclCommDestroy after finalize")
    return 0


def child_empty_p2p_rank(rank: int) -> int:
    """Zero-count send/recv are valid NCCL operations, grouped or not."""

    lib = load_nccl()
    lib.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, UniqueId, ctypes.c_int]
    comm = ctypes.c_void_p()
    unique_id = UniqueId(internal=os.environ["NCCL_SMOKE_UNIQUE_ID"].encode())
    expect_rc(lib.ncclCommInitRank(ctypes.byref(comm), REDUCE_WORLD, unique_id, rank), NCCL_SUCCESS, "ncclCommInitRank")
    peer = 1 - rank
    if rank == 0:
        expect_rc(lib.ncclSend(None, ctypes.c_size_t(0), NCCL_FLOAT32, peer, comm, None), NCCL_SUCCESS, "empty ncclSend")
    else:
        expect_rc(lib.ncclRecv(None, ctypes.c_size_t(0), NCCL_FLOAT32, peer, comm, None), NCCL_SUCCESS, "empty ncclRecv")
    lib.ncclGroupStart()
    expect_rc(lib.ncclSend(None, ctypes.c_size_t(0), NCCL_FLOAT32, peer, comm, None), NCCL_SUCCESS, "grouped empty ncclSend")
    expect_rc(lib.ncclRecv(None, ctypes.c_size_t(0), NCCL_FLOAT32, peer, comm, None), NCCL_SUCCESS, "grouped empty ncclRecv")
    expect_rc(lib.ncclGroupEnd(), NCCL_SUCCESS, "ncclGroupEnd of empty point-to-point")
    # The communicator stays in step: a real transfer still pairs up.
    buf = (ctypes.c_float * 2)(rank + 1.0, rank + 2.0)
    if rank == 0:
        expect_rc(lib.ncclSend(buf, ctypes.c_size_t(2), NCCL_FLOAT32, peer, comm, None), NCCL_SUCCESS, "ncclSend")
    else:
        expect_rc(lib.ncclRecv(buf, ctypes.c_size_t(2), NCCL_FLOAT32, peer, comm, None), NCCL_SUCCESS, "ncclRecv")
        if list(buf) != [1.0, 2.0]:
            raise AssertionError(f"ncclRecv produced {list(buf)}")
    expect_rc(lib.ncclSend(None, ctypes.c_size_t(0), 99, peer, comm, None), NCCL_INVALID_ARGUMENT, "empty ncclSend with a bad datatype")
    return 0


def child_init_all(_rank: int) -> int:
    """ncclCommInitAll validates its device list like init.cc does."""

    lib = load_nccl()
    lib.ncclCommInitAll.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.ncclCommCuDevice.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    comms = (ctypes.c_void_p * 2)()
    expect_rc(lib.ncclCommInitAll(comms, 2, (ctypes.c_int * 2)(0, 4096)), NCCL_INVALID_ARGUMENT, "ncclCommInitAll with a missing device")
    expect_rc(lib.ncclCommInitAll(comms, 2, (ctypes.c_int * 2)(0, 0)), NCCL_INVALID_USAGE, "ncclCommInitAll with a duplicate device")
    expect_rc(lib.ncclCommInitAll(comms, 2, (ctypes.c_int * 2)(1, 0)), NCCL_SUCCESS, "ncclCommInitAll")
    device = ctypes.c_int(-1)
    for index, expected in enumerate((1, 0)):
        expect_rc(lib.ncclCommCuDevice(comms[index], ctypes.byref(device)), NCCL_SUCCESS, "ncclCommCuDevice")
        if device.value != expected:
            raise AssertionError(f"rank {index} reports device {device.value}, expected {expected}")
    for comm in comms:
        lib.ncclCommDestroy(comm)
    return 0


def child_current_device_rank(rank: int) -> int:
    """ncclCommInitRank binds to the caller's CUDA device, not LOCAL_RANK.

    NCCL init.cc reads cudaGetDevice() into comm->cudaDev, and
    ncclCommCuDevice returns it. Each rank selects the other rank's index.
    """

    runtime = ctypes.CDLL(str(LIBCUDART), mode=ctypes.RTLD_GLOBAL)
    os.environ["LOCAL_RANK"] = str(rank)
    expect_rc(runtime.cudaSetDevice(1 - rank), 0, "cudaSetDevice")
    lib = load_nccl()
    lib.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, UniqueId, ctypes.c_int]
    lib.ncclCommCuDevice.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    comm = ctypes.c_void_p()
    unique_id = UniqueId(internal=os.environ["NCCL_SMOKE_UNIQUE_ID"].encode())
    expect_rc(lib.ncclCommInitRank(ctypes.byref(comm), REDUCE_WORLD, unique_id, rank), NCCL_SUCCESS, "ncclCommInitRank")
    device = ctypes.c_int(-1)
    expect_rc(lib.ncclCommCuDevice(comm, ctypes.byref(device)), NCCL_SUCCESS, "ncclCommCuDevice")
    if device.value != 1 - rank:
        raise AssertionError(f"rank {rank} communicator is on device {device.value}, expected {1 - rank}")
    lib.ncclCommDestroy(comm)
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


def check_empty_point_to_point() -> None:
    if not run_ranks("--empty-p2p-rank", REDUCE_WORLD, 3000):
        raise AssertionError("zero-count ncclSend/ncclRecv were not accepted")


def check_init_all_device_list() -> None:
    if not run_ranks("--init-all", 1, 3000):
        raise AssertionError("ncclCommInitAll did not validate its device list")


def check_comm_binds_current_device() -> None:
    if not run_ranks("--current-device-rank", REDUCE_WORLD, 3000):
        raise AssertionError("ncclCommInitRank did not bind the communicator to cudaGetDevice()")


def check_lifecycle_contracts() -> None:
    env = dict(os.environ, FAKEGPU_DIST_MODE="simulate")
    if subprocess.run([sys.executable, __file__, "--lifecycle"], env=env).returncode != 0:
        raise AssertionError("NCCL group and lifecycle return codes differ from NCCL")


def main() -> int:
    if sys.argv[1:] == ["--partial-loader"]:
        return child_partial_loader()
    if sys.argv[1:] == ["--lifecycle"]:
        return child_lifecycle()
    if sys.argv[1:2] == ["--split-rank"]:
        return child_split_rank(int(sys.argv[2]))
    if sys.argv[1:2] == ["--reduce-rank"]:
        return child_reduce_rank(int(sys.argv[2]))
    if sys.argv[1:2] == ["--empty-p2p-rank"]:
        return child_empty_p2p_rank(int(sys.argv[2]))
    if sys.argv[1:2] == ["--init-all"]:
        return child_init_all(int(sys.argv[2]))
    if sys.argv[1:2] == ["--current-device-rank"]:
        return child_current_device_rank(int(sys.argv[2]))
    if not LIBNCCL.exists():
        print(f"missing NCCL library: {LIBNCCL}", file=sys.stderr)
        return 2
    check_partial_real_nccl()
    check_lifecycle_contracts()
    check_empty_point_to_point()
    check_init_all_device_list()
    check_comm_binds_current_device()
    check_split_timeout()
    check_reduce_over_socket_payloads()
    print("nccl stubs smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
