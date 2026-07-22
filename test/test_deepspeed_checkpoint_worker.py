#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "fakegpu.hybrid_deepspeed_checkpoint.rank.v1"


def _write_report(report_dir: Path, rank: int, report: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"rank_{rank}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _matrix_close(
    actual: list[list[float]],
    expected: list[list[float]],
    tolerance: float,
) -> bool:
    return len(actual) == len(expected) and all(
        len(actual_row) == len(expected_row)
        and all(
            abs(float(actual_value) - float(expected_value)) <= tolerance
            for actual_value, expected_value in zip(
                actual_row,
                expected_row,
            )
        )
        for actual_row, expected_row in zip(actual, expected)
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


def _make_engine(
    *,
    torch: Any,
    deepspeed: Any,
    device: Any,
    parameter_dtype: Any,
    world_size: int,
    zero_stage: int,
    precision: str,
) -> tuple[Any, Any]:
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
                [[0.75, -0.25]],
                dtype=parameter_dtype,
                device=device,
            )
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.05,
        betas=(0.8, 0.9),
        eps=1e-6,
        weight_decay=0.0,
        foreach=False,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=0.5,
    )
    config = {
        "train_batch_size": world_size,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": zero_stage,
            "contiguous_gradients": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 32,
            "allgather_partitions": True,
            "allgather_bucket_size": 32,
            "overlap_comm": False,
            "stage3_prefetch_bucket_size": 0,
            "stage3_param_persistence_threshold": 0,
        },
        "zero_allow_untested_optimizer": True,
        "fp16": {"enabled": False},
        "bf16": {"enabled": precision == "bf16"},
        "steps_per_print": 1_000_000,
    }
    engine, _, _, engine_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=config,
        dist_init_required=False,
    )
    return engine, engine_scheduler


def _train_step(
    *,
    torch: Any,
    engine: Any,
    device: Any,
    parameter_dtype: Any,
    rank: int,
    step: int,
) -> float:
    scale = float(rank + 1)
    if step == 1:
        inputs = (1.0 * scale, -0.5 * scale)
        targets = (0.25 * scale,)
    elif step == 2:
        inputs = (-0.25 * scale, 1.5 * scale)
        targets = (-0.75 * scale,)
    else:
        raise ValueError(f"unsupported validation step: {step}")
    input_tensor = torch.tensor(
        [inputs],
        dtype=parameter_dtype,
        device=device,
    )
    target_tensor = torch.tensor(
        [targets],
        dtype=parameter_dtype,
        device=device,
    )
    prediction = engine(input_tensor)
    loss = torch.nn.functional.mse_loss(
        prediction.float(),
        target_tensor.float(),
    )
    loss_value = float(loss.detach().cpu().item())
    if not math.isfinite(loss_value):
        raise AssertionError(f"non-finite loss at step {step}: {loss_value}")
    engine.backward(loss)
    engine.step()
    torch.cuda.synchronize()
    return loss_value


def _learning_rate(engine: Any) -> float:
    values = engine.get_lr()
    if len(values) != 1:
        raise AssertionError(f"unexpected learning-rate groups: {values}")
    return float(values[0])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate DeepSpeed ZeRO checkpoint save and resume with real "
            "CUDA compute and FakeGPU-simulated NCCL communication."
        )
    )
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--zero-stage", type=int, choices=(0, 1, 2, 3))
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default="fp32",
    )
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    original_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    tag = "global_step1"
    client_token = "fakegpu-deepspeed-checkpoint-v1"
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "starting",
        "stage": "starting",
        "rank": rank,
        "world_size": world_size,
        "zero_stage": args.zero_stage,
        "precision": args.precision,
        "checkpoint_tag": tag,
        "torchrun_local_rank": original_local_rank,
        "physical_device_index": 0,
    }
    stage = "import_torch"
    dist = None
    engine = None

    try:
        import torch
        import torch.distributed as dist  # type: ignore[no-redef]

        stage = "import_deepspeed"
        import deepspeed
        from deepspeed.runtime.fp16.loss_scaler import LossScaler
        from deepspeed.runtime.zero.config import ZeroStageEnum
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
        device = torch.device("cuda:0")
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=180),
        )
        os.environ["LOCAL_RANK"] = "0"
        parameter_dtype = (
            torch.bfloat16 if args.precision == "bf16" else torch.float32
        )
        tolerance = 1e-6 if args.precision == "fp32" else 2e-2

        stage = "construct_reference_engine"
        engine, scheduler = _make_engine(
            torch=torch,
            deepspeed=deepspeed,
            device=device,
            parameter_dtype=parameter_dtype,
            world_size=world_size,
            zero_stage=args.zero_stage,
            precision=args.precision,
        )
        report["effective_zero_stage"] = int(
            engine.zero_optimization_stage()
        )
        report["engine_type"] = type(engine).__name__
        report["optimizer_type"] = type(engine.optimizer).__name__
        report["scheduler_type"] = type(scheduler).__name__

        stage = "train_before_checkpoint"
        first_loss = _train_step(
            torch=torch,
            engine=engine,
            device=device,
            parameter_dtype=parameter_dtype,
            rank=rank,
            step=1,
        )
        saved_parameter = _gather_weight(engine, deepspeed)
        saved_lr = _learning_rate(engine)
        if int(engine.global_steps) != 1:
            raise AssertionError(
                f"expected global step 1 before save, got {engine.global_steps}"
            )

        stage = "save_checkpoint"
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_ok = engine.save_checkpoint(
            str(args.checkpoint_dir),
            tag=tag,
            client_state={
                "validation_token": client_token,
                "saved_global_steps": 1,
                "saved_parameter": saved_parameter,
            },
            save_latest=True,
        )
        if save_ok is False:
            raise AssertionError("DeepSpeed save_checkpoint returned False")
        dist.barrier(device_ids=[0])

        stage = "train_uninterrupted_reference"
        reference_second_loss = _train_step(
            torch=torch,
            engine=engine,
            device=device,
            parameter_dtype=parameter_dtype,
            rank=rank,
            step=2,
        )
        uninterrupted_parameter = _gather_weight(engine, deepspeed)
        uninterrupted_lr = _learning_rate(engine)
        if int(engine.global_steps) != 2:
            raise AssertionError(
                "uninterrupted engine did not reach global step 2: "
                f"{engine.global_steps}"
            )

        del engine, scheduler
        engine = None
        gc.collect()
        torch.cuda.empty_cache()

        stage = "construct_resume_engine"
        engine, scheduler = _make_engine(
            torch=torch,
            deepspeed=deepspeed,
            device=device,
            parameter_dtype=parameter_dtype,
            world_size=world_size,
            zero_stage=args.zero_stage,
            precision=args.precision,
        )

        stage = "load_checkpoint"
        safe_globals_added = False
        if hasattr(torch.serialization, "add_safe_globals"):
            # DeepSpeed 0.15 stores this enum in its model-state payload but
            # predates PyTorch's weights_only=True default. The checkpoint was
            # created by this process, so explicitly allow only the two types
            # reported by get_unsafe_globals_in_checkpoint().
            torch.serialization.add_safe_globals(
                [ZeroStageEnum, LossScaler]
            )
            safe_globals_added = True
        load_path, client_state = engine.load_checkpoint(
            str(args.checkpoint_dir),
            tag=tag,
            load_module_strict=True,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )
        if load_path is None:
            raise AssertionError("DeepSpeed load_checkpoint returned no path")
        restored_parameter = _gather_weight(engine, deepspeed)
        restored_lr = _learning_rate(engine)
        if not _matrix_close(
            restored_parameter,
            saved_parameter,
            tolerance,
        ):
            raise AssertionError(
                "restored model parameter differs from checkpoint: "
                f"{restored_parameter} != {saved_parameter}"
            )
        if int(engine.global_steps) != 1:
            raise AssertionError(
                "restored engine did not recover global step 1: "
                f"{engine.global_steps}"
            )
        if not math.isclose(restored_lr, saved_lr, rel_tol=0, abs_tol=1e-12):
            raise AssertionError(
                f"restored learning rate differs: {restored_lr} != {saved_lr}"
            )
        if client_state.get("validation_token") != client_token:
            raise AssertionError(
                f"client state token was not restored: {client_state}"
            )
        if int(client_state.get("saved_global_steps", -1)) != 1:
            raise AssertionError(
                f"client state step was not restored: {client_state}"
            )

        stage = "train_resumed_step"
        resumed_second_loss = _train_step(
            torch=torch,
            engine=engine,
            device=device,
            parameter_dtype=parameter_dtype,
            rank=rank,
            step=2,
        )
        resumed_parameter = _gather_weight(engine, deepspeed)
        resumed_lr = _learning_rate(engine)
        if not _matrix_close(
            resumed_parameter,
            uninterrupted_parameter,
            tolerance,
        ):
            raise AssertionError(
                "resumed optimizer result differs from uninterrupted result: "
                f"{resumed_parameter} != {uninterrupted_parameter}"
            )
        if not math.isclose(
            resumed_second_loss,
            reference_second_loss,
            rel_tol=0,
            abs_tol=tolerance,
        ):
            raise AssertionError(
                "resumed loss differs from uninterrupted loss: "
                f"{resumed_second_loss} != {reference_second_loss}"
            )
        if not math.isclose(
            resumed_lr,
            uninterrupted_lr,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise AssertionError(
                "resumed scheduler result differs from uninterrupted result: "
                f"{resumed_lr} != {uninterrupted_lr}"
            )
        if int(engine.global_steps) != 2:
            raise AssertionError(
                f"resumed engine did not reach step 2: {engine.global_steps}"
            )

        final_tensor = torch.tensor(
            resumed_parameter,
            dtype=torch.float32,
            device=device,
        ).reshape(-1)
        gathered = [torch.empty_like(final_tensor) for _ in range(world_size)]
        dist.all_gather(gathered, final_tensor)
        torch.cuda.synchronize()
        gathered_parameters = [value.cpu().tolist() for value in gathered]
        if any(
            not torch.allclose(
                value,
                final_tensor,
                atol=tolerance,
                rtol=0,
            )
            for value in gathered
        ):
            raise AssertionError(
                f"resumed parameters differ across ranks: {gathered_parameters}"
            )

        stage = "checkpoint_inventory"
        dist.barrier(device_ids=[0])
        checkpoint_files = sorted(
            str(path.relative_to(args.checkpoint_dir))
            for path in args.checkpoint_dir.rglob("*")
            if path.is_file()
        )
        if "latest" not in checkpoint_files:
            raise AssertionError(
                f"checkpoint did not write latest tag: {checkpoint_files}"
            )

        report.update(
            {
                "status": "success",
                "stage": "complete",
                "checkpoint_load_path": str(load_path),
                "torch_load_safe_globals_added": safe_globals_added,
                "checkpoint_files": checkpoint_files,
                "client_state": {
                    "validation_token": client_state.get(
                        "validation_token"
                    ),
                    "saved_global_steps": client_state.get(
                        "saved_global_steps"
                    ),
                },
                "losses": {
                    "first": first_loss,
                    "uninterrupted_second": reference_second_loss,
                    "resumed_second": resumed_second_loss,
                },
                "learning_rates": {
                    "saved": saved_lr,
                    "restored": restored_lr,
                    "uninterrupted_after_second": uninterrupted_lr,
                    "resumed_after_second": resumed_lr,
                },
                "parameters": {
                    "saved": saved_parameter,
                    "restored": restored_parameter,
                    "uninterrupted_after_second": uninterrupted_parameter,
                    "resumed_after_second": resumed_parameter,
                    "gathered_after_resume": gathered_parameters,
                },
                "global_steps_after_resume": int(engine.global_steps),
            }
        )
        _write_report(args.report_dir, rank, report)
        print(json.dumps(report, sort_keys=True), flush=True)

        dist.destroy_process_group()
        dist = None
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
