#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _die(msg: str) -> None:
    print(f"[check_report] ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _require(obj: dict, key: str, *, ctx: str) -> object:
    if key not in obj:
        _die(f"missing key '{key}' in {ctx}")
    return obj[key]


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate FakeGPU JSON report schema/metrics")
    ap.add_argument("--path", default="fake_gpu_report.json", help="Path to fake_gpu_report.json")
    ap.add_argument("--expect-io", action="store_true", help="Require non-zero device IO counters")
    ap.add_argument("--expect-flops", action="store_true", help="Require non-zero GEMM/Matmul FLOPs counters")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.is_file():
        _die(f"report file not found: {path}")

    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        _die("report root must be a JSON object")

    version = report.get("report_version", 0)
    if not isinstance(version, int) or version < 2:
        _die(f"unexpected report_version={version!r} (expected >= 2)")

    host_io = _require(report, "host_io", ctx="root")
    if not isinstance(host_io, dict):
        _die("host_io must be an object")
    _require(host_io, "memcpy_calls", ctx="host_io")
    _require(host_io, "memcpy_bytes", ctx="host_io")

    devices = _require(report, "devices", ctx="root")
    if not isinstance(devices, list) or not devices:
        _die("devices must be a non-empty array")

    total_io_bytes = 0
    total_flops = 0
    saw_memory_peak = False

    for idx, dev in enumerate(devices):
        ctx = f"devices[{idx}]"
        if not isinstance(dev, dict):
            _die(f"{ctx} must be an object")

        _require(dev, "name", ctx=ctx)
        _require(dev, "uuid", ctx=ctx)
        total_mem = _require(dev, "total_memory", ctx=ctx)
        used_peak = _require(dev, "used_memory_peak", ctx=ctx)
        used_cur = _require(dev, "used_memory_current", ctx=ctx)
        if not all(isinstance(v, int) for v in (total_mem, used_peak, used_cur)):
            _die(f"{ctx} memory fields must be integers")
        if used_peak > 0:
            saw_memory_peak = True

        io = _require(dev, "io", ctx=ctx)
        if not isinstance(io, dict):
            _die(f"{ctx}.io must be an object")
        for key in ("h2d", "d2h", "d2d", "peer_tx", "peer_rx", "memset"):
            entry = _require(io, key, ctx=f"{ctx}.io")
            if not isinstance(entry, dict):
                _die(f"{ctx}.io.{key} must be an object")
            _require(entry, "calls", ctx=f"{ctx}.io.{key}")
            _require(entry, "bytes", ctx=f"{ctx}.io.{key}")
        total_bytes = _require(io, "total_bytes", ctx=f"{ctx}.io")
        if not isinstance(total_bytes, int):
            _die(f"{ctx}.io.total_bytes must be an integer")
        total_io_bytes += total_bytes

        compute = _require(dev, "compute", ctx=ctx)
        if not isinstance(compute, dict):
            _die(f"{ctx}.compute must be an object")
        for key in ("cublas_gemm", "cublaslt_matmul"):
            entry = _require(compute, key, ctx=f"{ctx}.compute")
            if not isinstance(entry, dict):
                _die(f"{ctx}.compute.{key} must be an object")
            _require(entry, "calls", ctx=f"{ctx}.compute.{key}")
            _require(entry, "flops", ctx=f"{ctx}.compute.{key}")
        dev_flops = _require(compute, "total_flops", ctx=f"{ctx}.compute")
        if not isinstance(dev_flops, int):
            _die(f"{ctx}.compute.total_flops must be an integer")
        total_flops += dev_flops

    if not saw_memory_peak:
        _die("no device reported non-zero used_memory_peak (did the workload allocate anything?)")
    if args.expect_io and total_io_bytes <= 0:
        _die("expected non-zero device IO counters but total_io_bytes is 0")
    if args.expect_flops and total_flops <= 0:
        _die("expected non-zero FLOPs counters but total_flops is 0")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

