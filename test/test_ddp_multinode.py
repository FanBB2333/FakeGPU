#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
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
    parser = argparse.ArgumentParser(description="Minimal DDP multinode validation for FakeGPU")
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--output-dim", type=int, default=4)
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
        import torch.nn as nn
        from torch.nn.parallel import DistributedDataParallel as DDP

        class TinyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(args.input_dim, args.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(args.hidden_dim, args.output_dim),
                )

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return self.net(inputs)

        report["torch_version"] = torch.__version__

        stage = "cuda_probe"
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda_device_count"] = int(torch.cuda.device_count())

        stage = "init_process_group"
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=20),
        )
        report["init_process_group"] = "ok"

        stage = "set_device"
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        report["device"] = str(device)

        stage = "barrier_before_training"
        dist.barrier(device_ids=[local_rank])
        report["barrier_before_training"] = "ok"

        stage = "broadcast_seed"
        seed_value = torch.tensor([2026 if rank == 0 else 0], device=device, dtype=torch.int32)
        dist.broadcast(seed_value, src=0)
        report["broadcast_value"] = int(seed_value.detach().cpu().item())
        if report["broadcast_value"] != 2026:
            raise AssertionError(
                f"broadcast produced {report['broadcast_value']}, expected 2026"
            )

        stage = "build_model"
        model = TinyModel().to(device)
        ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
        report["ddp"] = "ok"

        stage = "train"
        report["epochs_completed"] = 0
        report["steps_completed"] = 0
        for epoch in range(args.epochs):
            for step in range(args.steps_per_epoch):
                inputs = torch.full(
                    (args.batch_size, args.input_dim),
                    float(rank + step + 1),
                    device=device,
                    dtype=torch.float32,
                )
                targets = torch.zeros(
                    (args.batch_size, args.output_dim),
                    device=device,
                    dtype=torch.float32,
                )

                outputs = ddp_model(inputs)
                loss = torch.nn.functional.mse_loss(outputs, targets)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                report["steps_completed"] += 1
                report["last_loss"] = float(loss.detach().cpu().item())

            report["epochs_completed"] = epoch + 1

        stage = "post_train_all_reduce"
        all_reduce_value = torch.tensor([float(rank + 1)], device=device, dtype=torch.float32)
        dist.all_reduce(all_reduce_value)
        expected_sum = float(world_size * (world_size + 1) // 2)
        report["post_train_all_reduce"] = float(all_reduce_value.detach().cpu().item())
        report["post_train_all_reduce_expected"] = expected_sum
        if abs(report["post_train_all_reduce"] - expected_sum) > 1e-6:
            raise AssertionError(
                "post-train all_reduce produced "
                f"{report['post_train_all_reduce']}, expected {expected_sum}"
            )

        stage = "barrier_after_training"
        dist.barrier(device_ids=[local_rank])
        report["barrier_after_training"] = "ok"

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
            f"[Rank {rank}] DDP gap at {stage}: {type(exc).__name__}: {exc}",
            file=os.sys.stderr,
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
