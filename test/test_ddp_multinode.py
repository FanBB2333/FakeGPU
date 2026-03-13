#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
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
    report_path = report_dir / f"rank_{rank}.json"
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal torch.distributed + NCCL smoke test for FakeGPU")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory for per-rank exploratory reports.",
    )
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

    try:
        stage = "import_torch"
        import torch
        import torch.distributed as dist  # type: ignore[no-redef]

        report["torch_version"] = torch.__version__

        stage = "cuda_probe"
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda_device_count"] = int(torch.cuda.device_count())

        stage = "init_process_group"
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=10),
        )
        report["init_process_group"] = "ok"

        stage = "set_device"
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        report["device"] = str(device)

        stage = "all_reduce"
        all_reduce_value = torch.tensor([float(rank + 1)], device=device, dtype=torch.float32)
        dist.all_reduce(all_reduce_value)
        expected_sum = float(world_size * (world_size + 1) // 2)
        report["all_reduce_value"] = float(all_reduce_value.detach().cpu().item())
        report["all_reduce_expected"] = expected_sum
        if abs(report["all_reduce_value"] - expected_sum) > 1e-6:
            raise AssertionError(
                f"all_reduce produced {report['all_reduce_value']}, expected {expected_sum}"
            )

        stage = "broadcast"
        broadcast_value = torch.tensor([123 if rank == 0 else 0], device=device, dtype=torch.int32)
        dist.broadcast(broadcast_value, src=0)
        report["broadcast_value"] = int(broadcast_value.detach().cpu().item())
        if report["broadcast_value"] != 123:
            raise AssertionError(
                f"broadcast produced {report['broadcast_value']}, expected 123"
            )

        stage = "destroy_process_group"
        dist.destroy_process_group()
        report["destroy_process_group"] = "ok"
        report["status"] = "success"
        write_report(args.report_dir, rank, report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        report["status"] = "gap"
        report["failed_stage"] = stage
        report["exception_type"] = type(exc).__name__
        report["exception_message"] = str(exc)
        report["traceback"] = traceback.format_exc()
        write_report(args.report_dir, rank, report)
        print(
            f"[Rank {rank}] exploratory gap at {stage}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        if dist is not None:
            try:
                if dist.is_initialized():
                    dist.destroy_process_group()
            except Exception:  # noqa: BLE001
                pass
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
