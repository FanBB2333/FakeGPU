#!/usr/bin/env python3
"""Validate one DeepSpeed pipeline-parallel optimizer step."""

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

EXPECTED_INITIAL = [
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 1.0],
]
EXPECTED_LOSS = {1: 6.25, 2: 4.25}
EXPECTED_FINAL = {
    1: [
        [0.5, -1.0, -0.5, 0.0],
        [0.5, 0.0],
    ],
    2: [
        [0.45, -0.35, -0.55, 0.65],
        [0.45, 0.65],
    ],
}


def _write_report(report_dir: Path, rank: int, report: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"rank_{rank}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _vector_close(
    actual: object,
    expected: list[float],
    tolerance: float,
) -> bool:
    return (
        isinstance(actual, list)
        and len(actual) == len(expected)
        and all(
            abs(float(actual_value) - expected_value) <= tolerance
            for actual_value, expected_value in zip(actual, expected)
        )
    )


def _gather_stage_parameters(
    torch: Any,
    dist: Any,
    local_parameters: list[float],
) -> list[list[float]]:
    device = torch.device("cuda:0")
    local_count = torch.tensor(
        [len(local_parameters)],
        dtype=torch.int64,
        device=device,
    )
    gathered_counts = [torch.empty_like(local_count) for _ in range(2)]
    dist.all_gather(gathered_counts, local_count)
    counts = [int(value.item()) for value in gathered_counts]
    max_count = max(counts)

    padded = torch.zeros(max_count, dtype=torch.float32, device=device)
    if local_parameters:
        padded[: len(local_parameters)] = torch.tensor(
            local_parameters,
            dtype=torch.float32,
            device=device,
        )
    gathered = [torch.empty_like(padded) for _ in range(2)]
    dist.all_gather(gathered, padded)
    return [
        tensor[:count].cpu().tolist()
        for tensor, count in zip(gathered, counts)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default="fp32",
    )
    parser.add_argument(
        "--activation-checkpoint-interval",
        type=int,
        choices=(0, 1),
        default=0,
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        choices=(1, 2),
        default=1,
    )
    args = parser.parse_args(argv)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    report: dict[str, Any] = {
        "schema_version": "fakegpu.deepspeed_pipeline.rank.v1",
        "rank": rank,
        "world_size": world_size,
        "precision": args.precision,
        "activation_checkpoint_interval": args.activation_checkpoint_interval,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "status": "starting",
    }
    stage = "import_torch"
    dist = None

    try:
        import torch
        import torch.distributed as dist  # type: ignore[no-redef]

        stage = "import_deepspeed"
        import deepspeed
        from deepspeed.pipe import LayerSpec, PipelineModule
        from deepspeed.utils.logging import logger as deepspeed_logger

        deepspeed_logger.setLevel(logging.WARNING)
        logging.getLogger("deepspeed").setLevel(logging.WARNING)
        if world_size != 2:
            raise AssertionError(f"expected world_size=2, got {world_size}")
        if not torch.cuda.is_available():
            raise RuntimeError("real CUDA is not available")
        if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("physical CUDA device does not support BF16")

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

        stage = "init_process_group"
        torch.cuda.set_device(0)
        os.environ["LOCAL_RANK"] = "0"
        deepspeed.init_distributed(
            dist_backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=90),
        )

        class FirstStage(torch.nn.Linear):
            def __init__(self) -> None:
                super().__init__(2, 2, bias=False)
                with torch.no_grad():
                    self.weight.copy_(torch.eye(2))

        class LastStage(torch.nn.Linear):
            def __init__(self) -> None:
                super().__init__(2, 1, bias=False)
                with torch.no_grad():
                    self.weight.copy_(torch.ones(1, 2))

        stage = "construct_pipeline"
        model = PipelineModule(
            layers=[LayerSpec(FirstStage), LayerSpec(LastStage)],
            num_stages=2,
            loss_fn=torch.nn.MSELoss(reduction="mean"),
            partition_method="uniform",
            activation_checkpoint_interval=(
                args.activation_checkpoint_interval
            ),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        config = {
            "train_batch_size": args.gradient_accumulation_steps,
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "zero_optimization": {"stage": 0},
            "fp16": {"enabled": False},
            "bf16": {"enabled": args.precision == "bf16"},
            "pipeline": {
                "activation_checkpoint_interval": (
                    args.activation_checkpoint_interval
                ),
                "use_reentrant": True,
            },
            "steps_per_print": 1_000_000,
        }
        engine, engine_optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=config,
            dist_init_required=False,
        )
        local_parameters = [
            parameter.detach().float().cpu().reshape(-1).tolist()
            for parameter in engine.module.parameters()
        ]
        if len(local_parameters) != 1:
            raise AssertionError(
                f"expected one parameter on stage {rank}, got {local_parameters}"
            )
        initial_local = local_parameters[0]
        tolerance = 1e-6 if args.precision == "fp32" else 2e-2
        if not _vector_close(initial_local, EXPECTED_INITIAL[rank], tolerance):
            raise AssertionError(
                f"pipeline stage {rank} initial parameter mismatch: "
                f"{initial_local} != {EXPECTED_INITIAL[rank]}"
            )

        report.update(
            {
                "engine_type": type(engine).__name__,
                "optimizer_type": type(engine_optimizer).__name__,
                "pipe_parallel_size": int(engine.num_stages),
                "pipe_stage_id": int(engine.stage_id),
                "gradient_accumulation_steps": int(
                    engine.gradient_accumulation_steps()
                ),
                "initial_local_parameter": initial_local,
                "local_parameter_names": [
                    name for name, _ in engine.module.named_parameters()
                ],
            }
        )

        stage = "train_batch"
        torch.cuda.reset_peak_memory_stats(0)
        micro_batches = iter(
            [
                (
                    torch.tensor([[1.0, 2.0]], dtype=torch.float32),
                    torch.tensor([[0.5]], dtype=torch.float32),
                ),
                (
                    torch.tensor([[2.0, -1.0]], dtype=torch.float32),
                    torch.tensor([[-0.5]], dtype=torch.float32),
                ),
            ][: args.gradient_accumulation_steps]
        )
        loss = engine.train_batch(data_iter=micro_batches)
        torch.cuda.synchronize()
        loss_value = float(loss.detach().float().cpu().item())
        final_local = next(engine.module.parameters()).detach().float().cpu()
        final_local_values = final_local.reshape(-1).tolist()
        all_stage_parameters = _gather_stage_parameters(
            torch,
            dist,
            final_local_values,
        )
        report.update(
            {
                "loss": loss_value,
                "global_steps": int(engine.global_steps),
                "final_local_parameter": final_local_values,
                "all_stage_parameters": all_stage_parameters,
                "peak_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(0)
                ),
            }
        )

        if int(engine.global_steps) != 1:
            raise AssertionError(
                f"expected one pipeline optimizer step, got {engine.global_steps}"
            )
        expected_loss = EXPECTED_LOSS[args.gradient_accumulation_steps]
        if abs(loss_value - expected_loss) > tolerance:
            raise AssertionError(
                f"pipeline loss mismatch: {loss_value} != {expected_loss}"
            )
        expected_parameters = EXPECTED_FINAL[
            args.gradient_accumulation_steps
        ]
        for stage_id, (actual, expected) in enumerate(
            zip(all_stage_parameters, expected_parameters)
        ):
            if not _vector_close(actual, expected, tolerance):
                raise AssertionError(
                    f"pipeline stage {stage_id} update mismatch: "
                    f"{actual} != {expected}"
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
        if dist is not None:
            try:
                if dist.is_initialized():
                    dist.destroy_process_group()
            except Exception:
                pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
