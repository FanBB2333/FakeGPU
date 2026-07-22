#!/usr/bin/env python3
"""Validate one configurable DeepSpeed AutoTP training step."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED_OUTPUT = [3.0, 7.0]
EXPECTED_LOSS = 34.0
EXPECTED_COLUMN = [
    [0.98, -0.04, -0.06, -0.08],
    [-0.02, 0.96, -0.06, -0.08],
    [-0.08, -0.16, 0.76, -0.32],
    [-0.08, -0.16, -0.24, 0.68],
]
EXPECTED_ROW = [
    [0.98, 0.96, -0.06, -0.08],
    [-0.08, -0.16, 0.76, 0.68],
]


def _write_report(report_dir: Path, rank: int, report: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"rank_{rank}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _tensor_close(
    actual: object,
    expected: list[Any],
    tolerance: float,
) -> bool:
    if not isinstance(actual, list) or len(actual) != len(expected):
        return False
    for actual_value, expected_value in zip(actual, expected):
        if isinstance(expected_value, list):
            if not _tensor_close(actual_value, expected_value, tolerance):
                return False
        elif abs(float(actual_value) - float(expected_value)) > tolerance:
            return False
    return True


def _gather_full_weight(
    torch: Any,
    dist: Any,
    local_weight: Any,
    *,
    partition_dim: int,
) -> Any:
    gathered = [torch.empty_like(local_weight) for _ in range(2)]
    dist.all_gather(gathered, local_weight)
    return torch.cat(gathered, dim=partition_dim)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--zero-stage", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default="fp32",
    )
    args = parser.parse_args(argv)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    report: dict[str, Any] = {
        "schema_version": "fakegpu.deepspeed_autotp.rank.v1",
        "rank": rank,
        "world_size": world_size,
        "zero_stage": args.zero_stage,
        "precision": args.precision,
        "status": "starting",
    }
    stage = "import_torch"
    dist = None

    try:
        import torch
        import torch.distributed as dist  # type: ignore[no-redef]

        stage = "import_deepspeed"
        import deepspeed
        from deepspeed.runtime.tensor_parallel.config import TPTrainingConfig
        from deepspeed.utils.logging import logger as deepspeed_logger

        del TPTrainingConfig
        deepspeed_logger.setLevel(logging.WARNING)
        logging.getLogger("deepspeed").setLevel(logging.WARNING)
        if world_size != 2:
            raise AssertionError(f"expected world_size=2, got {world_size}")
        if not torch.cuda.is_available():
            raise RuntimeError("real CUDA is not available")
        if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("physical CUDA device does not support BF16")

        dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
        report.update(
            {
                "torch_version": str(torch.__version__),
                "torch_cuda_version": str(torch.version.cuda),
                "deepspeed_version": str(deepspeed.__version__),
                "physical_device_name": torch.cuda.get_device_name(0),
                "physical_compute_capability": list(
                    torch.cuda.get_device_capability(0)
                ),
            }
        )

        stage = "init_torch_process_group"
        torch.cuda.set_device(0)
        os.environ["LOCAL_RANK"] = "0"
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=90),
            device_id=torch.device("cuda", 0),
        )

        stage = "init_deepspeed_backend"
        deepspeed.init_distributed(
            dist_backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=90),
            dist_init_required=False,
        )

        class TinyAutoTPModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.column = torch.nn.Linear(4, 4, bias=False)
                self.row = torch.nn.Linear(4, 2, bias=False)
                with torch.no_grad():
                    self.column.weight.copy_(torch.eye(4))
                    self.row.weight.copy_(
                        torch.tensor(
                            [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]
                        )
                    )

            def forward(self, inputs: Any) -> Any:
                return self.row(self.column(inputs))

        stage = "initialize_engine"
        model = TinyAutoTPModel()
        config = {
            "train_batch_size": 1,
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "zero_optimization": {"stage": args.zero_stage},
            "fp16": {"enabled": False},
            "bf16": {"enabled": args.precision == "bf16"},
            "optimizer": {"type": "SGD", "params": {"lr": 0.01}},
            "tensor_parallel": {
                "autotp_size": 2,
                "dtype": dtype,
                "partition_config": {
                    "use_default_specs": False,
                    "strict_mode": True,
                    "layer_specs": [
                        {
                            "patterns": ["^column\\.weight$"],
                            "partition_type": "column",
                        },
                        {
                            "patterns": ["^row\\.weight$"],
                            "partition_type": "row",
                        },
                    ],
                },
            },
            "steps_per_print": 1_000_000,
        }
        engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=config,
            dist_init_required=False,
        )
        column = engine.module.column
        row = engine.module.row
        report.update(
            {
                "engine_type": type(engine).__name__,
                "optimizer_type": type(optimizer).__name__,
                "effective_zero_stage": int(engine.zero_optimization_stage()),
                "autotp_size": int(engine.autotp_size()),
                "tensor_parallel_rank": int(
                    engine.mpu.get_tensor_model_parallel_rank()
                ),
                "tensor_parallel_world_size": int(
                    engine.mpu.get_tensor_model_parallel_world_size()
                ),
                "column_layer_type": type(column).__name__,
                "row_layer_type": type(row).__name__,
                "column_local_shape": list(column.weight.shape),
                "row_local_shape": list(row.weight.shape),
            }
        )
        if list(column.weight.shape) != [2, 4]:
            raise AssertionError(
                f"column-parallel shard shape mismatch: {column.weight.shape}"
            )
        if list(row.weight.shape) != [2, 2]:
            raise AssertionError(
                f"row-parallel shard shape mismatch: {row.weight.shape}"
            )

        stage = "training_step"
        torch.cuda.reset_peak_memory_stats(0)
        inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=dtype, device="cuda:0")
        target = torch.tensor([[1.0, -1.0]], dtype=dtype, device="cuda:0")
        output = engine(inputs)
        loss = torch.nn.functional.mse_loss(output, target, reduction="mean")
        output_values = output.detach().float().cpu().reshape(-1).tolist()
        loss_value = float(loss.detach().float().cpu().item())
        engine.backward(loss)
        engine.step()
        torch.cuda.synchronize()

        full_column = _gather_full_weight(
            torch,
            dist,
            column.weight.detach(),
            partition_dim=0,
        )
        full_row = _gather_full_weight(
            torch,
            dist,
            row.weight.detach(),
            partition_dim=1,
        )
        full_column_values = full_column.float().cpu().tolist()
        full_row_values = full_row.float().cpu().tolist()
        report.update(
            {
                "output": output_values,
                "loss": loss_value,
                "global_steps": int(engine.global_steps),
                "full_column_weight_after_step": full_column_values,
                "full_row_weight_after_step": full_row_values,
                "peak_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(0)
                ),
            }
        )

        tolerance = 1e-6 if args.precision == "fp32" else 2e-2
        if not _tensor_close(output_values, EXPECTED_OUTPUT, tolerance):
            raise AssertionError(
                f"AutoTP output mismatch: {output_values} != {EXPECTED_OUTPUT}"
            )
        if abs(loss_value - EXPECTED_LOSS) > tolerance:
            raise AssertionError(
                f"AutoTP loss mismatch: {loss_value} != {EXPECTED_LOSS}"
            )
        if int(engine.global_steps) != 1:
            raise AssertionError(
                f"expected one optimizer step, got {engine.global_steps}"
            )
        if not _tensor_close(full_column_values, EXPECTED_COLUMN, tolerance):
            raise AssertionError(
                f"AutoTP column weight mismatch: {full_column_values}"
            )
        if not _tensor_close(full_row_values, EXPECTED_ROW, tolerance):
            raise AssertionError(
                f"AutoTP row weight mismatch: {full_row_values}"
            )

        stage = "barrier"
        dist.barrier(device_ids=[0])
        dist.destroy_process_group()
        dist = None
        report.update({"status": "success", "stage": "complete"})
        _write_report(args.report_dir, rank, report)
        print(json.dumps(report, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        report.update(
            {
                "status": "error",
                "stage": stage,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        _write_report(args.report_dir, rank, report)
        print(
            f"rank {rank} failed at {stage}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    finally:
        if dist is not None and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
