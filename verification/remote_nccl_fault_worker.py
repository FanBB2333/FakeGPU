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
NCCL_SUM = 0
NCCL_PROD = 1


class NcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * 128)]


def _session_token(session: str) -> bytes:
    return hashlib.sha256(
        f"fakegpu-remote-fault-v1:{session}".encode("utf-8")
    ).hexdigest()[:32].encode("ascii")


def _configure_nccl(lib: ctypes.CDLL) -> None:
    comm_type = ctypes.c_void_p
    lib.ncclCommInitRank.argtypes = [
        ctypes.POINTER(comm_type),
        ctypes.c_int,
        NcclUniqueId,
        ctypes.c_int,
    ]
    lib.ncclCommInitRank.restype = ctypes.c_int
    lib.ncclAllReduce.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        comm_type,
        ctypes.c_void_p,
    ]
    lib.ncclAllReduce.restype = ctypes.c_int
    lib.ncclCommGetAsyncError.argtypes = [
        comm_type,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.ncclCommGetAsyncError.restype = ctypes.c_int
    lib.ncclCommDestroy.argtypes = [comm_type]
    lib.ncclCommDestroy.restype = ctypes.c_int
    lib.ncclCommAbort.argtypes = [comm_type]
    lib.ncclCommAbort.restype = ctypes.c_int
    lib.ncclGetLastError.argtypes = [comm_type]
    lib.ncclGetLastError.restype = ctypes.c_char_p


def _last_error(lib: ctypes.CDLL, comm: ctypes.c_void_p) -> str:
    raw = lib.ncclGetLastError(comm)
    if not raw:
        return ""
    return raw.decode("utf-8", errors="replace")


def _unique_id(session: str) -> NcclUniqueId:
    unique_id = NcclUniqueId()
    token = _session_token(session)
    ctypes.memmove(ctypes.addressof(unique_id), token, len(token))
    return unique_id


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_collective_mismatch(
    args: argparse.Namespace,
    lib: ctypes.CDLL,
    comm: ctypes.c_void_p,
    report: dict[str, Any],
) -> None:
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
    if init_result != NCCL_SUCCESS:
        raise AssertionError(
            f"ncclCommInitRank unexpectedly failed with result {init_result}: "
            f"{_last_error(lib, comm)}"
        )

    send = (ctypes.c_float * 4)(*([float(args.rank + 1)] * 4))
    recv = (ctypes.c_float * 4)()
    reduction = NCCL_SUM if args.rank == 0 else NCCL_PROD
    mismatch_result = int(
        lib.ncclAllReduce(
            ctypes.cast(send, ctypes.c_void_p),
            ctypes.cast(recv, ctypes.c_void_p),
            4,
            NCCL_FLOAT32,
            reduction,
            comm,
            None,
        )
    )
    report["mismatch_result"] = mismatch_result
    report["mismatch_last_error"] = _last_error(lib, comm)
    if mismatch_result == NCCL_SUCCESS:
        raise AssertionError("mismatched reduction operators unexpectedly succeeded")

    async_error = ctypes.c_int(NCCL_SUCCESS)
    query_result = int(lib.ncclCommGetAsyncError(comm, ctypes.byref(async_error)))
    report["async_query_result"] = query_result
    report["async_error"] = int(async_error.value)
    if query_result != NCCL_SUCCESS:
        raise AssertionError(
            f"ncclCommGetAsyncError failed with result {query_result}"
        )
    if async_error.value == NCCL_SUCCESS:
        raise AssertionError("mismatch did not persist as an async communicator error")

    retry_started = time.monotonic()
    retry_result = int(
        lib.ncclAllReduce(
            ctypes.cast(send, ctypes.c_void_p),
            ctypes.cast(recv, ctypes.c_void_p),
            4,
            NCCL_FLOAT32,
            NCCL_SUM,
            comm,
            None,
        )
    )
    retry_seconds = time.monotonic() - retry_started
    report["retry_result"] = retry_result
    report["retry_seconds"] = retry_seconds
    if retry_result == NCCL_SUCCESS:
        raise AssertionError("poisoned communicator retry unexpectedly succeeded")
    if retry_seconds >= min(2.0, max(0.5, args.timeout_ms / 1000.0)):
        raise AssertionError(
            f"poisoned communicator did not fail quickly: {retry_seconds:.3f}s"
        )

    destroy_result = int(lib.ncclCommDestroy(comm))
    report["destroy_result"] = destroy_result
    if destroy_result != NCCL_SUCCESS:
        raise AssertionError(
            f"ncclCommDestroy failed after mismatch with result {destroy_result}"
        )
    comm.value = None


def _run_missing_peer(
    args: argparse.Namespace,
    lib: ctypes.CDLL,
    comm: ctypes.c_void_p,
    report: dict[str, Any],
) -> None:
    started = time.monotonic()
    init_result = int(
        lib.ncclCommInitRank(
            ctypes.byref(comm),
            args.world_size,
            _unique_id(args.session),
            args.rank,
        )
    )
    elapsed = time.monotonic() - started
    report["comm_init_result"] = init_result
    report["comm_init_seconds"] = elapsed
    report["comm_init_last_error"] = _last_error(lib, comm)
    if init_result == NCCL_SUCCESS:
        raise AssertionError("communicator initialization succeeded without every rank")
    maximum_seconds = max(2.0, args.timeout_ms / 1000.0 * 3.0)
    if elapsed >= maximum_seconds:
        raise AssertionError(
            f"missing-peer failure exceeded {maximum_seconds:.3f}s: {elapsed:.3f}s"
        )
    if "timeout" not in report["comm_init_last_error"].lower():
        raise AssertionError(
            "missing-peer failure did not include a timeout diagnostic: "
            f"{report['comm_init_last_error']!r}"
        )


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": "fakegpu.remote_fault.rank.v1",
        "case": args.case,
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
        if args.case == "collective-mismatch":
            _run_collective_mismatch(args, lib, comm, report)
        elif args.case == "missing-peer":
            _run_missing_peer(args, lib, comm, report)
        else:  # pragma: no cover - argparse rejects this path
            raise AssertionError(f"unsupported case: {args.case}")
        report["status"] = "success"
        report["expected_failure_observed"] = True
        return report
    except Exception as exc:
        report.update(
            {
                "status": "error",
                "expected_failure_observed": False,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Exercise expected FakeGPU NCCL failures through a TCP coordinator."
    )
    parser.add_argument(
        "--case",
        choices=["collective-mismatch", "missing-peer"],
        required=True,
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
