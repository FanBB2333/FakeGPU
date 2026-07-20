from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import statistics
import time
import traceback
from array import array
from pathlib import Path
from typing import Any


NCCL_SUCCESS = 0
NCCL_FLOAT32 = 7
NCCL_SUM = 0


class NcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * 128)]


def _session_token(session: str) -> bytes:
    return hashlib.sha256(
        f"fakegpu-tcp-bandwidth-v1:{session}".encode("utf-8")
    ).hexdigest()[:32].encode("ascii")


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _last_error(lib: ctypes.CDLL, comm: ctypes.c_void_p) -> str:
    try:
        value = lib.ncclGetLastError(comm)
    except Exception:
        return ""
    if not value:
        return ""
    return value.decode("utf-8", errors="replace")


def _require_success(
    lib: ctypes.CDLL,
    comm: ctypes.c_void_p,
    result: int,
    operation: str,
) -> None:
    if result == NCCL_SUCCESS:
        return
    detail = _last_error(lib, comm)
    suffix = f": {detail}" if detail else ""
    raise RuntimeError(f"{operation} failed with NCCL result {result}{suffix}")


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
    lib.ncclCommDestroy.argtypes = [comm_type]
    lib.ncclCommDestroy.restype = ctypes.c_int
    lib.ncclCommAbort.argtypes = [comm_type]
    lib.ncclCommAbort.restype = ctypes.c_int
    lib.ncclGetLastError.argtypes = [comm_type]
    lib.ncclGetLastError.restype = ctypes.c_char_p


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": "fakegpu.tcp_bandwidth.rank.v1",
        "rank": args.rank,
        "world_size": args.world_size,
        "endpoint": os.environ.get("FAKEGPU_COORDINATOR_ADDR", ""),
        "payload_bytes": args.payload_bytes,
        "warmup_iterations": args.warmup,
        "iterations": args.iterations,
        "status": "starting",
    }
    comm = ctypes.c_void_p()
    lib: ctypes.CDLL | None = None

    try:
        if args.payload_bytes <= 0 or args.payload_bytes % ctypes.sizeof(ctypes.c_float) != 0:
            raise ValueError("payload bytes must be a positive multiple of 4")

        lib = ctypes.CDLL(str(args.nccl_lib), mode=ctypes.RTLD_GLOBAL)
        _configure_nccl(lib)

        unique_id = NcclUniqueId()
        token = _session_token(args.session)
        ctypes.memmove(ctypes.addressof(unique_id), token, len(token))

        result = lib.ncclCommInitRank(
            ctypes.byref(comm),
            args.world_size,
            unique_id,
            args.rank,
        )
        _require_success(lib, comm, int(result), "ncclCommInitRank")

        count = args.payload_bytes // ctypes.sizeof(ctypes.c_float)
        send = array("f", [float(args.rank + 1)]) * count
        recv = array("f", [0.0]) * count
        send_pointer = ctypes.c_void_p(send.buffer_info()[0])
        recv_pointer = ctypes.c_void_p(recv.buffer_info()[0])

        def all_reduce() -> float:
            started = time.perf_counter()
            result_code = lib.ncclAllReduce(
                send_pointer,
                recv_pointer,
                count,
                NCCL_FLOAT32,
                NCCL_SUM,
                comm,
                None,
            )
            elapsed = time.perf_counter() - started
            _require_success(lib, comm, int(result_code), "ncclAllReduce")
            return elapsed

        for _ in range(args.warmup):
            all_reduce()

        timings = [all_reduce() for _ in range(args.iterations)]
        expected = float(args.world_size * (args.world_size + 1) // 2)
        sample_indices = sorted({0, count // 2, count - 1})
        samples = [float(recv[index]) for index in sample_indices]
        if any(abs(value - expected) > 1e-5 for value in samples):
            raise AssertionError(
                f"all-reduce result mismatch: samples={samples}, expected={expected}"
            )

        destroy_result = int(lib.ncclCommDestroy(comm))
        _require_success(lib, comm, destroy_result, "ncclCommDestroy")
        comm = ctypes.c_void_p()

        total_seconds = sum(timings)
        payload_bits = float(args.payload_bytes * 8)
        report.update(
            {
                "status": "success",
                "timings_seconds": timings,
                "total_seconds": total_seconds,
                "average_seconds": statistics.fmean(timings),
                "median_seconds": statistics.median(timings),
                "p95_seconds": _percentile(timings, 0.95),
                "min_seconds": min(timings),
                "max_seconds": max(timings),
                "algorithmic_bandwidth_gbps": (
                    payload_bits * args.iterations / total_seconds / 1e9
                ),
                "rank_socket_payload_throughput_gbps": (
                    payload_bits * 2.0 * args.iterations / total_seconds / 1e9
                ),
                "expected_value": expected,
                "sample_values": samples,
            }
        )
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--payload-bytes", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--nccl-lib", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    report = run_worker(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
