#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any


def rank_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def write_report(report_dir: Path | None, rank: int, payload: dict[str, Any]) -> None:
    if report_dir is None:
        return
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"rank_{rank}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal torch.distributed barrier smoke test for FakeGPU")
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--delay-rank", type=int, default=-1)
    parser.add_argument("--delay-seconds", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=int, default=5)
    args = parser.parse_args()

    rank = rank_env("RANK", 0)
    world_size = rank_env("WORLD_SIZE", 1)
    local_rank = rank_env("LOCAL_RANK", 0)
    stage = "bootstrap"
    report: dict[str, Any] = {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "pid": os.getpid(),
        "status": "starting",
    }

    dist = None

    def persist() -> None:
        write_report(args.report_dir, rank, report)

    def handle_signal(signum: int, _frame: Any) -> None:
        report["status"] = "gap"
        report["failed_stage"] = stage
        report["exception_type"] = "Signal"
        report["exception_message"] = f"terminated by signal {signum}"
        persist()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    persist()

    try:
        stage = "import_torch"
        import torch
        import torch.distributed as dist  # type: ignore[no-redef]

        report["torch_version"] = torch.__version__
        persist()

        stage = "cuda_probe"
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda_device_count"] = int(torch.cuda.device_count())
        persist()

        stage = "set_device"
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        report["device"] = str(device)
        persist()

        stage = "init_process_group"
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=args.timeout_seconds),
        )
        report["init_process_group"] = "ok"
        persist()

        stage = "pre_barrier_delay"
        if rank == args.delay_rank and args.delay_seconds > 0:
            time.sleep(args.delay_seconds)
            report["delay_seconds"] = args.delay_seconds
        persist()

        stage = "barrier"
        begin = time.monotonic()
        dist.barrier(device_ids=[local_rank])
        report["barrier_elapsed_ms"] = round((time.monotonic() - begin) * 1000.0, 3)
        report["barrier"] = "ok"
        persist()

        stage = "destroy_process_group"
        dist.destroy_process_group()
        report["destroy_process_group"] = "ok"
        report["status"] = "success"
        persist()
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        report["status"] = "gap"
        report["failed_stage"] = stage
        report["exception_type"] = type(exc).__name__
        report["exception_message"] = str(exc)
        report["traceback"] = traceback.format_exc()
        persist()
        print(
            f"[Rank {rank}] barrier gap at {stage}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        if dist is not None:
            try:
                if dist.is_initialized():
                    dist.destroy_process_group()
            except Exception:  # noqa: BLE001
                pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
