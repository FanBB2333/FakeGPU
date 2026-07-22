#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import os
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any


def _write_report(report_dir: Path, rank: int, report: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"rank_{rank}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _gather_weight(engine: Any, deepspeed: Any) -> list[list[float]]:
    parameters = list(engine.module.parameters())
    if len(parameters) != 1:
        raise AssertionError(
            f"expected one model parameter, got {len(parameters)}"
        )
    with deepspeed.zero.GatheredParameters(
        parameters,
        modifier_rank=None,
    ):
        return parameters[0].detach().float().cpu().tolist()


def _local_parameter_partition(engine: Any) -> list[float]:
    parameters = list(engine.module.parameters())
    if len(parameters) != 1:
        raise AssertionError(
            f"expected one model parameter, got {len(parameters)}"
        )
    parameter = parameters[0]
    partition = getattr(parameter, "ds_tensor", None)
    if partition is None:
        partition = parameter
    return partition.detach().float().cpu().reshape(-1).tolist()


def _vector_close(
    actual: list[float],
    expected: list[float],
    tolerance: float,
) -> bool:
    return len(actual) == len(expected) and all(
        abs(float(actual_value) - float(expected_value)) <= tolerance
        for actual_value, expected_value in zip(actual, expected)
    )


def _matrix_close(
    actual: object,
    expected: list[list[float]],
    tolerance: float,
) -> bool:
    if not isinstance(actual, list) or len(actual) != len(expected):
        return False
    for actual_row, expected_row in zip(actual, expected):
        if not isinstance(actual_row, list) or len(actual_row) != len(
            expected_row
        ):
            return False
        if any(
            abs(float(actual_value) - expected_value) > tolerance
            for actual_value, expected_value in zip(actual_row, expected_row)
        ):
            return False
    return True


def _optimizer_state_devices(optimizer: Any) -> list[str]:
    devices: set[str] = set()

    def visit(value: Any) -> None:
        if hasattr(value, "device") and hasattr(value, "shape"):
            devices.add(str(value.device))
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(optimizer.state)
    return sorted(devices)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate DeepSpeed ZeRO numerics with real CUDA compute and "
            "FakeGPU-simulated NCCL communication."
        )
    )
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--zero-stage", type=int, choices=(0, 1, 2, 3))
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default="fp32",
    )
    parser.add_argument("--offload-optimizer", action="store_true")
    parser.add_argument("--offload-parameters", action="store_true")
    args = parser.parse_args()
    if args.offload_optimizer and args.zero_stage not in (2, 3):
        parser.error("optimizer offload requires ZeRO stage 2 or 3")
    if args.offload_parameters and args.zero_stage != 3:
        parser.error("parameter offload requires ZeRO stage 3")
    if args.offload_parameters and not args.offload_optimizer:
        parser.error("parameter offload validation also requires optimizer offload")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    original_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    report: dict[str, Any] = {
        "schema_version": "fakegpu.hybrid_deepspeed_numerics.rank.v1",
        "rank": rank,
        "world_size": world_size,
        "zero_stage": args.zero_stage,
        "precision": args.precision,
        "offload_optimizer": bool(args.offload_optimizer),
        "offload_parameters": bool(args.offload_parameters),
        "torchrun_local_rank": original_local_rank,
        "physical_device_index": 0,
        "status": "starting",
    }
    stage = "import_torch"
    dist = None

    try:
        import torch
        import torch.distributed as dist  # type: ignore[no-redef]

        stage = "import_deepspeed"
        import deepspeed
        from deepspeed.utils.logging import logger as deepspeed_logger

        deepspeed_logger.setLevel(logging.WARNING)
        logging.getLogger("deepspeed").setLevel(logging.WARNING)
        if world_size not in (2, 4):
            raise AssertionError(
                f"expected world_size to be 2 or 4, got {world_size}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("real CUDA is not available")
        if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("physical CUDA device does not support BF16")

        report["torch_version"] = str(torch.__version__)
        report["torch_cuda_version"] = str(torch.version.cuda)
        report["deepspeed_version"] = str(deepspeed.__version__)
        report["physical_device_name"] = torch.cuda.get_device_name(0)
        report["physical_compute_capability"] = list(
            torch.cuda.get_device_capability(0)
        )

        stage = "set_device"
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")

        stage = "init_process_group"
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=90),
        )

        # torchrun assigns distinct local ranks, but hybrid mode intentionally
        # maps every logical rank to the one physical CUDA device. DeepSpeed
        # reads LOCAL_RANK while constructing its engine.
        os.environ["LOCAL_RANK"] = "0"
        parameter_dtype = (
            torch.bfloat16 if args.precision == "bf16" else torch.float32
        )

        stage = "construct_engine"
        model = torch.nn.Linear(
            2,
            1,
            bias=False,
            device=device,
            dtype=parameter_dtype,
        )
        with torch.no_grad():
            model.weight.copy_(
                torch.tensor(
                    [[1.0, 0.0]],
                    dtype=parameter_dtype,
                    device=device,
                )
            )
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.1,
            momentum=0.9,
        )
        zero_config = {
            "stage": args.zero_stage,
            "contiguous_gradients": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 32,
            "allgather_partitions": True,
            "allgather_bucket_size": 32,
            "overlap_comm": False,
            "stage3_prefetch_bucket_size": 0,
            "stage3_param_persistence_threshold": 0,
        }
        if args.offload_optimizer:
            zero_config["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": False,
            }
        if args.offload_parameters:
            zero_config["offload_param"] = {
                "device": "cpu",
                "pin_memory": False,
            }
        config = {
            "train_batch_size": world_size * 2,
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 2,
            "zero_optimization": zero_config,
            "zero_allow_untested_optimizer": True,
            "zero_force_ds_cpu_optimizer": False,
            "fp16": {"enabled": False},
            "bf16": {"enabled": args.precision == "bf16"},
            "steps_per_print": 1_000_000,
        }
        engine, engine_optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=config,
            dist_init_required=False,
        )
        report["engine_type"] = type(engine).__name__
        report["optimizer_type"] = type(engine_optimizer).__name__
        report["effective_zero_stage"] = int(
            engine.zero_optimization_stage()
        )
        report["gradient_accumulation_steps"] = int(
            engine.gradient_accumulation_steps()
        )
        report["parameter_dtype"] = str(next(engine.module.parameters()).dtype)
        report["initial_parameter"] = _gather_weight(engine, deepspeed)
        initial_local_partition = _local_parameter_partition(engine)
        report["initial_local_partition"] = initial_local_partition

        expected_initial = [[1.0, 0.0]]
        tolerance = 1e-6 if args.precision == "fp32" else 1e-2
        if not _matrix_close(
            report["initial_parameter"],
            expected_initial,
            tolerance,
        ):
            raise AssertionError(
                f"unexpected initial parameter: {report['initial_parameter']}"
            )

        stage = "forward_backward"
        micro_parameters: list[list[list[float]]] = []
        losses: list[float] = []
        for micro_step, micro_scale in enumerate((1.0, 2.0), start=1):
            rank_scale = float(rank + 1) * micro_scale
            inputs = torch.tensor(
                [[rank_scale, 2.0 * rank_scale]],
                dtype=parameter_dtype,
                device=device,
            )
            loss = engine(inputs).sum()
            losses.append(float(loss.detach().float().cpu().item()))
            engine.backward(loss)
            engine.step()
            torch.cuda.synchronize()
            report[f"micro_step_{micro_step}_global_steps"] = int(
                engine.global_steps
            )
            if micro_step == 1:
                local_partition = _local_parameter_partition(engine)
                report["local_partition_after_micro_step_1"] = local_partition
                if not _vector_close(
                    local_partition,
                    initial_local_partition,
                    tolerance,
                ):
                    raise AssertionError(
                        "DeepSpeed updated the local parameter partition "
                        "before the gradient accumulation boundary: "
                        f"{local_partition} != {initial_local_partition}"
                    )
                # ZeRO-3 keeps parameters active between accumulated micro
                # steps. Re-gathering them here violates DeepSpeed's own
                # lifecycle, so the full initial value represents the verified
                # unchanged local partitions.
                micro_parameters.append(report["initial_parameter"])
            else:
                micro_parameters.append(_gather_weight(engine, deepspeed))

        report["local_losses"] = losses
        report["parameters_after_micro_steps"] = micro_parameters
        report["global_steps"] = int(engine.global_steps)
        basic_optimizer = getattr(engine_optimizer, "optimizer", optimizer)
        parameter = next(engine.module.parameters())
        local_partition = getattr(parameter, "ds_tensor", parameter)
        report["offload"] = {
            "optimizer_requested": bool(args.offload_optimizer),
            "parameters_requested": bool(args.offload_parameters),
            "basic_optimizer_type": type(basic_optimizer).__name__,
            "optimizer_state_devices": _optimizer_state_devices(
                basic_optimizer
            ),
            "local_parameter_partition_device": str(
                local_partition.device
            ),
        }

        rank_average = (world_size + 1.0) / 2.0
        micro_average = 1.5
        gradient_scale = rank_average * micro_average
        expected_final = [
            [1.0 - 0.1 * gradient_scale, -0.2 * gradient_scale]
        ]
        report["expected_parameter_after_step"] = expected_final
        if not _matrix_close(
            micro_parameters[0],
            expected_initial,
            tolerance,
        ):
            raise AssertionError(
                "DeepSpeed updated parameters before the gradient "
                f"accumulation boundary: {micro_parameters[0]}"
            )
        if not _matrix_close(
            micro_parameters[1],
            expected_final,
            tolerance,
        ):
            raise AssertionError(
                "DeepSpeed parameter update mismatch: "
                f"{micro_parameters[1]} != {expected_final}"
            )
        if engine.global_steps != 1:
            raise AssertionError(
                f"expected one optimizer step, got {engine.global_steps}"
            )

        stage = "cross_rank_consistency"
        final_parameter = torch.tensor(
            micro_parameters[1],
            dtype=torch.float32,
            device=device,
        ).reshape(-1)
        gathered = [
            torch.empty_like(final_parameter) for _ in range(world_size)
        ]
        dist.all_gather(gathered, final_parameter)
        torch.cuda.synchronize()
        report["gathered_parameters"] = [
            value.cpu().tolist() for value in gathered
        ]
        if any(
            not torch.allclose(
                value,
                final_parameter,
                atol=tolerance,
                rtol=0,
            )
            for value in gathered
        ):
            raise AssertionError(
                "DeepSpeed parameters differ across ranks: "
                f"{report['gathered_parameters']}"
            )

        stage = "barrier"
        dist.barrier(device_ids=[0])
        dist.destroy_process_group()
        dist = None

        report["status"] = "success"
        report["stage"] = "complete"
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
            file=os.sys.stderr,
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
