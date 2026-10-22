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
NCCL_FLOAT16 = 6
NCCL_FLOAT32 = 7
NCCL_FLOAT8E4M3 = 10
NCCL_SUM = 0
NCCL_AVG = 4
NCCL_SCALAR_DEVICE = 0
NCCL_SCALAR_HOST_IMMEDIATE = 1
# NCCL_SIM_INFO_INITIALIZER magic from nccl.h.
NCCL_SIM_INFO_MAGIC = 0x74685283
SPLIT_WORLD = 3
REDUCE_WORLD = 2
P2P_WORLD = 3


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


def child_premul_rank(rank: int) -> int:
    """PreMulSum operators behave as in NCCL's enqueue.cc.

    A device scalar (ncclScalarDevice, which torch passes for a tensor factor)
    is read like a host one, the operator is copied into a grouped operation
    so it may be destroyed before ncclGroupEnd, and an operator only works with
    its own datatype and communicator.
    """

    lib = load_nccl()
    lib.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, UniqueId, ctypes.c_int]
    comm = ctypes.c_void_p()
    unique_id = UniqueId(internal=os.environ["NCCL_SMOKE_UNIQUE_ID"].encode())
    expect_rc(lib.ncclCommInitRank(ctypes.byref(comm), REDUCE_WORLD, unique_id, rank), NCCL_SUCCESS, "ncclCommInitRank")
    send = (ctypes.c_float * 2)(rank + 1.0, 10.0 * (rank + 1))
    recv = (ctypes.c_float * 2)()

    factor = ctypes.c_float(0.5)
    op = ctypes.c_int(-1)
    expect_rc(
        lib.ncclRedOpCreatePreMulSum(ctypes.byref(op), ctypes.byref(factor), NCCL_FLOAT32, NCCL_SCALAR_DEVICE, comm),
        NCCL_SUCCESS,
        "ncclRedOpCreatePreMulSum with a device scalar",
    )
    expect_rc(lib.ncclAllReduce(send, recv, ctypes.c_size_t(2), NCCL_FLOAT32, op.value, comm, None), NCCL_SUCCESS, "device-scalar preMulSum")
    if list(recv) != [1.5, 15.0]:
        raise AssertionError(f"rank {rank}: device-scalar preMulSum produced {list(recv)}")
    half = (ctypes.c_uint16 * 2)()
    expect_rc(lib.ncclAllReduce(half, half, ctypes.c_size_t(2), NCCL_FLOAT16, op.value, comm, None), NCCL_INVALID_ARGUMENT, "preMulSum with another datatype")
    expect_rc(lib.ncclRedOpDestroy(op.value, comm), NCCL_SUCCESS, "ncclRedOpDestroy")

    factor = ctypes.c_float(2.0)
    expect_rc(
        lib.ncclRedOpCreatePreMulSum(ctypes.byref(op), ctypes.byref(factor), NCCL_FLOAT32, NCCL_SCALAR_HOST_IMMEDIATE, comm),
        NCCL_SUCCESS,
        "ncclRedOpCreatePreMulSum",
    )
    lib.ncclGroupStart()
    expect_rc(lib.ncclAllReduce(send, recv, ctypes.c_size_t(2), NCCL_FLOAT32, op.value, comm, None), NCCL_SUCCESS, "grouped preMulSum")
    expect_rc(lib.ncclRedOpDestroy(op.value, comm), NCCL_SUCCESS, "ncclRedOpDestroy inside the group")
    expect_rc(lib.ncclGroupEnd(), NCCL_SUCCESS, "ncclGroupEnd")
    if list(recv) != [6.0, 60.0]:
        raise AssertionError(f"rank {rank}: grouped preMulSum produced {list(recv)}")

    child = ctypes.c_void_p()
    expect_rc(lib.ncclCommSplit(comm, 0, rank, ctypes.byref(child), None), NCCL_SUCCESS, "ncclCommSplit")
    expect_rc(
        lib.ncclRedOpCreatePreMulSum(ctypes.byref(op), ctypes.byref(factor), NCCL_FLOAT32, NCCL_SCALAR_HOST_IMMEDIATE, comm),
        NCCL_SUCCESS,
        "ncclRedOpCreatePreMulSum",
    )
    expect_rc(lib.ncclAllReduce(send, recv, ctypes.c_size_t(2), NCCL_FLOAT32, op.value, child, None), NCCL_INVALID_ARGUMENT, "preMulSum on another communicator")
    expect_rc(lib.ncclRedOpDestroy(op.value, child), NCCL_INVALID_ARGUMENT, "ncclRedOpDestroy on another communicator")
    return 0


def child_fp8_rank(rank: int) -> int:
    """FP8 follows NCCL 2.27.5 (enqueue.cc, taskAppend).

    Below compute capability 9.0 on any rank every non-empty FP8 collective is
    ncclInvalidArgument (an Ampere GPU running NCCL 2.27.5 returned it for
    all-reduce, reduce, reduce-scatter, broadcast and all-gather) and leaves the
    communicator usable; send and receive work on any device.
    """

    lib = load_nccl()
    lib.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, UniqueId, ctypes.c_int]
    comm = ctypes.c_void_p()
    unique_id = UniqueId(internal=os.environ["NCCL_SMOKE_UNIQUE_ID"].encode())
    expect_rc(lib.ncclCommInitRank(ctypes.byref(comm), REDUCE_WORLD, unique_id, rank), NCCL_SUCCESS, "ncclCommInitRank")
    # e4m3 bytes: rank 0 holds (1, 16), rank 1 holds (2, 1).
    send = (ctypes.c_uint8 * 2)(*((0x38, 0x58) if rank == 0 else (0x40, 0x38)))
    recv = (ctypes.c_uint8 * 4)()
    size = ctypes.c_size_t
    calls = {
        "all-reduce": lambda: lib.ncclAllReduce(send, recv, size(2), NCCL_FLOAT8E4M3, NCCL_SUM, comm, None),
        "reduce": lambda: lib.ncclReduce(send, recv, size(2), NCCL_FLOAT8E4M3, NCCL_SUM, 0, comm, None),
        "reduce-scatter": lambda: lib.ncclReduceScatter(send, recv, size(1), NCCL_FLOAT8E4M3, NCCL_SUM, comm, None),
        "broadcast": lambda: lib.ncclBroadcast(send, recv, size(2), NCCL_FLOAT8E4M3, 0, comm, None),
        "all-gather": lambda: lib.ncclAllGather(send, recv, size(2), NCCL_FLOAT8E4M3, comm, None),
    }
    if os.environ["NCCL_SMOKE_FP8"] == "rejected":
        for name, call in calls.items():
            expect_rc(call(), NCCL_INVALID_ARGUMENT, f"FP8 {name} below compute capability 9.0")
        expect_rc(lib.ncclAllReduce(send, recv, size(0), NCCL_FLOAT8E4M3, NCCL_SUM, comm, None), NCCL_SUCCESS, "empty FP8 all-reduce")
        value = (ctypes.c_float * 1)(1.0)
        expect_rc(lib.ncclAllReduce(value, value, size(1), NCCL_FLOAT32, NCCL_SUM, comm, None), NCCL_SUCCESS, "all-reduce after FP8 rejections")
        if value[0] != 2.0:
            raise AssertionError(f"rank {rank}: all-reduce after FP8 rejections produced {value[0]}")
    else:
        expect_rc(calls["all-reduce"](), NCCL_SUCCESS, "FP8 all-reduce")
        # 1 + 2 = 3; 16 + 1 rounds back to 16 in e4m3.
        if list(recv[:2]) != [0x44, 0x58]:
            raise AssertionError(f"rank {rank}: FP8 all-reduce produced {list(recv[:2])}")
        expect_rc(lib.ncclAllReduce(send, recv, size(2), NCCL_FLOAT8E4M3, NCCL_AVG, comm, None), NCCL_SUCCESS, "FP8 average")
        # 0.5 * 1 + 0.5 * 2 = 1.5; 0.5 * 16 + 0.5 * 1 = 8.5 rounds to 8.
        if list(recv[:2]) != [0x3C, 0x50]:
            raise AssertionError(f"rank {rank}: FP8 average produced {list(recv[:2])}")
    peer = 1 - rank
    if rank == 0:
        expect_rc(lib.ncclSend(send, size(2), NCCL_FLOAT8E4M3, peer, comm, None), NCCL_SUCCESS, "FP8 ncclSend")
    else:
        expect_rc(lib.ncclRecv(recv, size(2), NCCL_FLOAT8E4M3, peer, comm, None), NCCL_SUCCESS, "FP8 ncclRecv")
        if list(recv[:2]) != [0x38, 0x58]:
            raise AssertionError(f"rank 1: FP8 ncclRecv produced {list(recv[:2])}")
    return 0


def init_rank(lib: ctypes.CDLL, world: int, rank: int) -> ctypes.c_void_p:
    lib.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, UniqueId, ctypes.c_int]
    comm = ctypes.c_void_p()
    unique_id = UniqueId(internal=os.environ["NCCL_SMOKE_UNIQUE_ID"].encode())
    expect_rc(lib.ncclCommInitRank(ctypes.byref(comm), world, unique_id, rank), NCCL_SUCCESS, "ncclCommInitRank")
    return comm


def expect_values(actual, expected: list[float], what: str) -> None:
    if list(actual) != expected:
        raise AssertionError(f"{what} produced {list(actual)}, expected {expected}")


def child_exchange_p2p_rank(rank: int) -> int:
    """Sends and receives pair by (sender, receiver, per-pair sequence).

    NCCL queues each peer's sends and receives in posting order (enqueue.cc,
    taskAppend), so a group holding both a send and a receive works, a rank's
    k-th send to a peer meets the peer's k-th receive, and grouped and
    ungrouped calls pair with each other. Self pairs and their errors follow
    NCCL 2.27.5 on an Ampere GPU (sm_80): ordered self pairs gave "3 4 1 2", a
    lone send to self is ncclInvalidUsage (at ncclGroupEnd inside a group,
    count 0 included) and leaves no async error.
    """

    lib = load_nccl()
    lib.ncclCommGetAsyncError.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    comm = init_rank(lib, REDUCE_WORLD, rank)
    size = ctypes.c_size_t
    floats = ctypes.c_float * 2
    peer = 1 - rank
    send = floats(rank + 1.0, rank + 2.0)
    recv = floats()
    lib.ncclGroupStart()
    expect_rc(lib.ncclSend(send, size(2), NCCL_FLOAT32, peer, comm, None), NCCL_SUCCESS, "grouped ncclSend")
    expect_rc(lib.ncclRecv(recv, size(2), NCCL_FLOAT32, peer, comm, None), NCCL_SUCCESS, "grouped ncclRecv")
    expect_rc(lib.ncclGroupEnd(), NCCL_SUCCESS, "ncclGroupEnd of a send/recv exchange")
    expect_values(recv, [peer + 1.0, peer + 2.0], f"rank {rank}: grouped exchange")

    first, second = floats(1.0, 2.0), floats(3.0, 4.0)
    lib.ncclGroupStart()
    if rank == 0:
        lib.ncclSend(first, size(2), NCCL_FLOAT32, 1, comm, None)
        lib.ncclSend(second, size(2), NCCL_FLOAT32, 1, comm, None)
    else:
        first, second = floats(), floats()
        lib.ncclRecv(first, size(2), NCCL_FLOAT32, 0, comm, None)
        lib.ncclRecv(second, size(2), NCCL_FLOAT32, 0, comm, None)
    expect_rc(lib.ncclGroupEnd(), NCCL_SUCCESS, "ncclGroupEnd of two sends to one peer")
    if rank == 1:
        expect_values(first, [1.0, 2.0], "first receive")
        expect_values(second, [3.0, 4.0], "second receive")

    recv = floats()
    if rank == 0:
        lib.ncclGroupStart()
        lib.ncclSend(send, size(2), NCCL_FLOAT32, peer, comm, None)
        lib.ncclRecv(recv, size(2), NCCL_FLOAT32, peer, comm, None)
        expect_rc(lib.ncclGroupEnd(), NCCL_SUCCESS, "grouped exchange with an ungrouped peer")
    else:
        expect_rc(lib.ncclSend(send, size(2), NCCL_FLOAT32, peer, comm, None), NCCL_SUCCESS, "ungrouped ncclSend")
        expect_rc(lib.ncclRecv(recv, size(2), NCCL_FLOAT32, peer, comm, None), NCCL_SUCCESS, "ungrouped ncclRecv")
    expect_values(recv, [peer + 1.0, peer + 2.0], f"rank {rank}: mixed exchange")

    source = (ctypes.c_float * 4)(1.0, 2.0, 3.0, 4.0)
    target = (ctypes.c_float * 4)()
    lib.ncclGroupStart()
    lib.ncclSend(source, size(2), NCCL_FLOAT32, rank, comm, None)
    lib.ncclSend(ctypes.byref(source, 8), size(2), NCCL_FLOAT32, rank, comm, None)
    lib.ncclRecv(ctypes.byref(target, 8), size(2), NCCL_FLOAT32, rank, comm, None)
    lib.ncclRecv(target, size(2), NCCL_FLOAT32, rank, comm, None)
    expect_rc(lib.ncclGroupEnd(), NCCL_SUCCESS, "ncclGroupEnd of ordered self pairs")
    expect_values(target, [3.0, 4.0, 1.0, 2.0], "ordered self pairs")

    expect_rc(lib.ncclSend(source, size(2), NCCL_FLOAT32, rank, comm, None), NCCL_INVALID_USAGE, "ungrouped send to self")
    expect_rc(lib.ncclRecv(target, size(2), NCCL_FLOAT32, rank, comm, None), NCCL_INVALID_USAGE, "ungrouped recv from self")
    expect_rc(lib.ncclSend(source, size(0), NCCL_FLOAT32, rank, comm, None), NCCL_INVALID_USAGE, "ungrouped empty send to self")
    for count in (2, 0):
        lib.ncclGroupStart()
        expect_rc(lib.ncclSend(source, size(count), NCCL_FLOAT32, rank, comm, None), NCCL_SUCCESS, "grouped lone send to self")
        expect_rc(lib.ncclGroupEnd(), NCCL_INVALID_USAGE, f"ncclGroupEnd of a lone send to self of {count} elements")
    async_error = ctypes.c_int(-1)
    expect_rc(lib.ncclCommGetAsyncError(comm, ctypes.byref(async_error)), NCCL_SUCCESS, "ncclCommGetAsyncError")
    expect_rc(async_error.value, NCCL_SUCCESS, "async error after unmatched self operations")
    value = (ctypes.c_float * 1)(1.0)
    expect_rc(lib.ncclAllReduce(value, value, size(1), NCCL_FLOAT32, NCCL_SUM, comm, None), NCCL_SUCCESS, "all-reduce after point-to-point")
    expect_values(value, [2.0], "all-reduce after point-to-point")
    return 0


def child_partial_p2p_rank(rank: int) -> int:
    """Ranks outside a send/recv pair are not involved, and all-to-all-v works.

    The all-to-all-v is issued like torch's all_to_all_single with splits:
    every rank posts a send and a receive to every rank, itself included,
    even when the count is zero.
    """

    lib = load_nccl()
    comm = init_rank(lib, P2P_WORLD, rank)
    size = ctypes.c_size_t
    if rank < 2:
        peer = 1 - rank
        send = (ctypes.c_float * 2)(rank + 1.0, rank + 2.0)
        recv = (ctypes.c_float * 2)()
        lib.ncclGroupStart()
        lib.ncclSend(send, size(2), NCCL_FLOAT32, peer, comm, None)
        lib.ncclRecv(recv, size(2), NCCL_FLOAT32, peer, comm, None)
        expect_rc(lib.ncclGroupEnd(), NCCL_SUCCESS, f"rank {rank}: exchange without rank 2")
        expect_values(recv, [peer + 1.0, peer + 2.0], f"rank {rank}: exchange without rank 2")
    value = (ctypes.c_float * 1)(rank + 1.0)
    expect_rc(lib.ncclAllReduce(value, value, size(1), NCCL_FLOAT32, NCCL_SUM, comm, None), NCCL_SUCCESS, f"rank {rank}: all-reduce")
    expect_values(value, [6.0], f"rank {rank}: all-reduce after a partial exchange")

    def split(src: int, dst: int) -> int:
        return (src + dst) % P2P_WORLD

    def payload(src: int, dst: int) -> list[float]:
        return [100.0 * src + 10.0 * dst + k for k in range(split(src, dst))]

    sends = [(ctypes.c_float * 2)(*payload(rank, peer)) for peer in range(P2P_WORLD)]
    recvs = [(ctypes.c_float * 2)() for _ in range(P2P_WORLD)]
    lib.ncclGroupStart()
    for peer in range(P2P_WORLD):
        lib.ncclSend(sends[peer], size(split(rank, peer)), NCCL_FLOAT32, peer, comm, None)
        lib.ncclRecv(recvs[peer], size(split(peer, rank)), NCCL_FLOAT32, peer, comm, None)
    expect_rc(lib.ncclGroupEnd(), NCCL_SUCCESS, f"rank {rank}: all-to-all-v")
    for peer in range(P2P_WORLD):
        expected = payload(peer, rank)
        expect_values(recvs[peer][: len(expected)], expected, f"rank {rank}: all-to-all-v from rank {peer}")
    return 0


def child_multi_comm(_rank: int) -> int:
    """A group may span communicators, here both ranks of ncclCommInitAll.

    An Ampere GPU running NCCL 2.27.5 completed groups holding all-reduces on
    three communicators and send/recv on two, and returned ncclInvalidUsage
    for a send to self on one communicator with the receive on another.
    An error from an operation enqueued in a group drops the group and
    ncclGroupEnd returns it (group.cc, ncclGroupEndInternal).
    """

    lib = load_nccl()
    lib.ncclCommInitAll.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    handles = (ctypes.c_void_p * 2)()
    expect_rc(lib.ncclCommInitAll(handles, 2, (ctypes.c_int * 2)(0, 1)), NCCL_SUCCESS, "ncclCommInitAll")
    comms = [ctypes.c_void_p(handle) for handle in handles]
    size = ctypes.c_size_t
    floats = ctypes.c_float * 2
    sends = [floats(1.0, 2.0), floats(3.0, 4.0)]

    def grouped_all_reduce() -> list[ctypes.Array]:
        recvs = [floats(), floats()]
        lib.ncclGroupStart()
        for index in range(2):
            expect_rc(
                lib.ncclAllReduce(sends[index], recvs[index], size(2), NCCL_FLOAT32, NCCL_SUM, comms[index], None),
                NCCL_SUCCESS,
                "grouped all-reduce",
            )
        expect_rc(lib.ncclGroupEnd(), NCCL_SUCCESS, "ncclGroupEnd over two communicators")
        return recvs

    for recv in grouped_all_reduce():
        expect_values(recv, [4.0, 6.0], "all-reduce over two communicators")

    recvs = [floats(), floats()]
    lib.ncclGroupStart()
    for index in range(2):
        lib.ncclSend(sends[index], size(2), NCCL_FLOAT32, 1 - index, comms[index], None)
        lib.ncclRecv(recvs[index], size(2), NCCL_FLOAT32, 1 - index, comms[index], None)
    expect_rc(lib.ncclGroupEnd(), NCCL_SUCCESS, "ncclGroupEnd of an exchange over two communicators")
    for index in range(2):
        expect_values(recvs[index], list(sends[1 - index]), f"rank {index}: exchange")

    untouched = floats(-1.0, -1.0)
    lib.ncclGroupStart()
    expect_rc(lib.ncclAllReduce(sends[0], untouched, size(2), NCCL_FLOAT32, NCCL_SUM, comms[0], None), NCCL_SUCCESS, "valid grouped all-reduce")
    expect_rc(lib.ncclAllReduce(sends[1], recvs[1], size(2), 99, NCCL_SUM, comms[1], None), NCCL_INVALID_ARGUMENT, "grouped all-reduce with a bad datatype")
    expect_rc(lib.ncclGroupEnd(), NCCL_INVALID_ARGUMENT, "ncclGroupEnd after an enqueue error")
    expect_values(untouched, [-1.0, -1.0], "operation of a dropped group")

    lib.ncclGroupStart()
    lib.ncclSend(sends[0], size(2), NCCL_FLOAT32, 0, comms[0], None)
    lib.ncclRecv(recvs[1], size(2), NCCL_FLOAT32, 1, comms[1], None)
    expect_rc(lib.ncclGroupEnd(), NCCL_INVALID_USAGE, "send to self paired across communicators")

    for recv in grouped_all_reduce():
        expect_values(recv, [4.0, 6.0], "all-reduce after failed groups")
    for comm in comms:
        lib.ncclCommDestroy(comm)
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


def check_premul_sum_operators() -> None:
    if not run_ranks("--premul-rank", REDUCE_WORLD, 3000):
        raise AssertionError("PreMulSum operators do not behave as in NCCL")


def check_fp8_compute_capability_gate() -> None:
    # A100 profiles are sm_80, H100 profiles sm_90; one sm_80 rank decides.
    for profiles, expected in (("a100:2", "rejected"), ("h100:2", "allowed"), ("h100,a100", "rejected")):
        if not run_ranks("--fp8-rank", REDUCE_WORLD, 3000, {"FAKEGPU_PROFILES": profiles, "NCCL_SMOKE_FP8": expected}):
            raise AssertionError(f"FP8 collectives on {profiles} were not {expected} as in NCCL")


def check_point_to_point_pairing() -> None:
    if not run_ranks("--exchange-p2p-rank", REDUCE_WORLD, 3000):
        raise AssertionError("ncclSend/ncclRecv do not pair up as in NCCL")
    # Socket payloads carry several sends and receives in one request.
    if not run_ranks("--partial-p2p-rank", P2P_WORLD, 3000, {"FAKEGPU_STAGING_FORCE_SOCKET": "1"}):
        raise AssertionError("partial point-to-point or all-to-all-v failed")


def check_multi_communicator_groups() -> None:
    if not run_ranks("--multi-comm", 1, 3000):
        raise AssertionError("a group spanning two communicators failed")


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
    if sys.argv[1:2] == ["--premul-rank"]:
        return child_premul_rank(int(sys.argv[2]))
    if sys.argv[1:2] == ["--fp8-rank"]:
        return child_fp8_rank(int(sys.argv[2]))
    if sys.argv[1:2] == ["--exchange-p2p-rank"]:
        return child_exchange_p2p_rank(int(sys.argv[2]))
    if sys.argv[1:2] == ["--partial-p2p-rank"]:
        return child_partial_p2p_rank(int(sys.argv[2]))
    if sys.argv[1:2] == ["--multi-comm"]:
        return child_multi_comm(int(sys.argv[2]))
    if not LIBNCCL.exists():
        print(f"missing NCCL library: {LIBNCCL}", file=sys.stderr)
        return 2
    check_partial_real_nccl()
    check_lifecycle_contracts()
    check_empty_point_to_point()
    check_point_to_point_pairing()
    check_multi_communicator_groups()
    check_init_all_device_list()
    check_comm_binds_current_device()
    check_premul_sum_operators()
    check_fp8_compute_capability_gate()
    check_split_timeout()
    check_reduce_over_socket_payloads()
    print("nccl stubs smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
