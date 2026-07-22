#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any


NCCL_SUCCESS = 0
NCCL_FLOAT32 = 7


class NcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * 128)]


def _session_token(session: str) -> bytes:
    return hashlib.sha256(
        f"fakegpu-remote-alltoallv-v1:{session}".encode("utf-8")
    ).hexdigest()[:32].encode("ascii")


def _unique_id(session: str) -> NcclUniqueId:
    unique_id = NcclUniqueId()
    token = _session_token(session)
    ctypes.memmove(ctypes.addressof(unique_id), token, len(token))
    return unique_id


def _configure_nccl(lib: ctypes.CDLL) -> None:
    comm_type = ctypes.c_void_p
    lib.ncclCommInitRank.argtypes = [
        ctypes.POINTER(comm_type),
        ctypes.c_int,
        NcclUniqueId,
        ctypes.c_int,
    ]
    lib.ncclCommInitRank.restype = ctypes.c_int
    lib.ncclGroupStart.argtypes = []
    lib.ncclGroupStart.restype = ctypes.c_int
    lib.ncclGroupEnd.argtypes = []
    lib.ncclGroupEnd.restype = ctypes.c_int
    for name in ("ncclSend", "ncclRecv"):
        operation = getattr(lib, name)
        operation.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            comm_type,
            ctypes.c_void_p,
        ]
        operation.restype = ctypes.c_int
    lib.ncclCommDestroy.argtypes = [comm_type]
    lib.ncclCommDestroy.restype = ctypes.c_int
    lib.ncclCommAbort.argtypes = [comm_type]
    lib.ncclCommAbort.restype = ctypes.c_int
    lib.ncclGetLastError.argtypes = [comm_type]
    lib.ncclGetLastError.restype = ctypes.c_char_p


def _last_error(lib: ctypes.CDLL, comm: ctypes.c_void_p) -> str:
    raw = lib.ncclGetLastError(comm)
    return raw.decode("utf-8", errors="replace") if raw else ""


def _require_success(
    result: int,
    operation: str,
    lib: ctypes.CDLL,
    comm: ctypes.c_void_p,
) -> None:
    if result != NCCL_SUCCESS:
        raise AssertionError(
            f"{operation} failed with result {result}: {_last_error(lib, comm)}"
        )


def split_count(sender: int, receiver: int, *, sparse: bool) -> int:
    if not sparse:
        return 1 + (sender * 2 + receiver) % 3
    if sender == receiver:
        return 2
    return 0 if (sender + 2 * receiver) % 3 == 2 else 1


def split_plan(rank: int, world_size: int, *, sparse: bool) -> tuple[list[int], list[int]]:
    sends = [
        split_count(rank, peer, sparse=sparse) for peer in range(world_size)
    ]
    receives = [
        split_count(peer, rank, sparse=sparse) for peer in range(world_size)
    ]
    return sends, receives


def expected_receive_values(
    rank: int,
    world_size: int,
    *,
    sparse: bool,
) -> list[float]:
    values: list[float] = []
    for sender in range(world_size):
        for index in range(split_count(sender, rank, sparse=sparse)):
            values.append(float(1000 * sender + 100 * rank + index))
    return values


def _offsets(splits: list[int]) -> list[int]:
    offsets: list[int] = []
    total = 0
    for count in splits:
        offsets.append(total)
        total += count
    return offsets


def _buffer_pointer(buffer: Any, element_offset: int) -> ctypes.c_void_p:
    return ctypes.cast(
        ctypes.byref(buffer, element_offset * ctypes.sizeof(ctypes.c_float)),
        ctypes.c_void_p,
    )


def _run_variant(
    *,
    rank: int,
    world_size: int,
    sparse: bool,
    lib: ctypes.CDLL,
    comm: ctypes.c_void_p,
) -> dict[str, Any]:
    send_splits, recv_splits = split_plan(rank, world_size, sparse=sparse)
    send_offsets = _offsets(send_splits)
    recv_offsets = _offsets(recv_splits)
    send_total = sum(send_splits)
    recv_total = sum(recv_splits)
    send_buffer = (ctypes.c_float * max(1, send_total))()
    recv_buffer = (ctypes.c_float * max(1, recv_total))()

    for peer, count in enumerate(send_splits):
        for index in range(count):
            send_buffer[send_offsets[peer] + index] = float(
                1000 * rank + 100 * peer + index
            )
    for index in range(recv_total):
        recv_buffer[index] = -1.0

    started = time.monotonic()
    _require_success(int(lib.ncclGroupStart()), "ncclGroupStart", lib, comm)
    for peer in range(world_size):
        send_count = send_splits[peer]
        if send_count:
            _require_success(
                int(
                    lib.ncclSend(
                        _buffer_pointer(send_buffer, send_offsets[peer]),
                        send_count,
                        NCCL_FLOAT32,
                        peer,
                        comm,
                        None,
                    )
                ),
                f"ncclSend(peer={peer})",
                lib,
                comm,
            )
        recv_count = recv_splits[peer]
        if recv_count:
            _require_success(
                int(
                    lib.ncclRecv(
                        _buffer_pointer(recv_buffer, recv_offsets[peer]),
                        recv_count,
                        NCCL_FLOAT32,
                        peer,
                        comm,
                        None,
                    )
                ),
                f"ncclRecv(peer={peer})",
                lib,
                comm,
            )
    _require_success(int(lib.ncclGroupEnd()), "ncclGroupEnd", lib, comm)
    elapsed = time.monotonic() - started

    received = [float(recv_buffer[index]) for index in range(recv_total)]
    expected = expected_receive_values(rank, world_size, sparse=sparse)
    if received != expected:
        raise AssertionError(
            f"all-to-all-v payload mismatch: received={received}, expected={expected}"
        )
    remote_peer = 1 - rank if world_size == 2 else None
    cross_host_send_bytes = (
        send_splits[remote_peer] * ctypes.sizeof(ctypes.c_float)
        if remote_peer is not None
        else None
    )
    return {
        "name": "sparse" if sparse else "nonuniform",
        "send_splits": send_splits,
        "recv_splits": recv_splits,
        "send_values": [float(send_buffer[index]) for index in range(send_total)],
        "received_values": received,
        "expected_values": expected,
        "send_bytes": send_total * ctypes.sizeof(ctypes.c_float),
        "recv_bytes": recv_total * ctypes.sizeof(ctypes.c_float),
        "cross_host_send_bytes": cross_host_send_bytes,
        "operation_seconds": elapsed,
    }


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": "fakegpu.remote_alltoallv.rank.v1",
        "rank": args.rank,
        "world_size": args.world_size,
        "session": args.session,
        "timeout_ms": args.timeout_ms,
        "coordinator": os.environ.get("FAKEGPU_COORDINATOR_ADDR", ""),
        "status": "starting",
    }
    lib: ctypes.CDLL | None = None
    comm = ctypes.c_void_p()
    try:
        lib = ctypes.CDLL(str(args.nccl_lib), mode=ctypes.RTLD_GLOBAL)
        _configure_nccl(lib)
        started = time.monotonic()
        init_result = int(
            lib.ncclCommInitRank(
                ctypes.byref(comm),
                args.world_size,
                _unique_id(args.session),
                args.rank,
            )
        )
        report["comm_init_result"] = init_result
        report["comm_init_seconds"] = time.monotonic() - started
        _require_success(init_result, "ncclCommInitRank", lib, comm)

        report["variants"] = [
            _run_variant(
                rank=args.rank,
                world_size=args.world_size,
                sparse=sparse,
                lib=lib,
                comm=comm,
            )
            for sparse in (False, True)
        ]
        destroy_result = int(lib.ncclCommDestroy(comm))
        report["destroy_result"] = destroy_result
        _require_success(destroy_result, "ncclCommDestroy", lib, comm)
        comm.value = None
        report["status"] = "success"
        return report
    except Exception as exc:
        report.update(
            {
                "status": "error",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        if lib is not None and comm.value:
            try:
                lib.ncclCommAbort(comm)
            except Exception:
                pass
        return report


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate grouped nonuniform and sparse all-to-all-v over TCP."
    )
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--timeout-ms", type=int, required=True)
    parser.add_argument("--nccl-lib", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.world_size <= 1:
        parser.error("--world-size must be greater than one")
    if args.rank < 0 or args.rank >= args.world_size:
        parser.error("--rank must be within [0, world-size)")
    if args.timeout_ms <= 0:
        parser.error("--timeout-ms must be greater than zero")
    if not args.nccl_lib.is_file():
        parser.error(f"fake NCCL library does not exist: {args.nccl_lib}")

    report = run_worker(args)
    _write_report(args.report, report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
