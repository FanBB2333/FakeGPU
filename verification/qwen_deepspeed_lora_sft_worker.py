#!/usr/bin/env python3
"""Measure one Qwen3.5 LoRA SFT step under Hybrid DeepSpeed ZeRO."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any


os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA_VERSION = "fakegpu.qwen_deepspeed_lora_sft.rank.v1"


def _write_report(report_dir: Path, rank: int, report: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"rank_{rank}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _gathered_squared_norm(parameter: Any, deepspeed: Any) -> float:
    with deepspeed.zero.GatheredParameters(
        [parameter],
        modifier_rank=None,
    ):
        return float(
            parameter.detach().float().square().sum().cpu().item()
        )


def _max_phase_bytes(records: list[dict[str, Any]]) -> int:
    if not records:
        return 0
    return max(int(record["peak_allocated_bytes"]) for record in records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--zero-stage", type=int, choices=(0, 1, 2, 3))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--attention-implementation",
        choices=("eager", "sdpa"),
        default="sdpa",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--checkpoint-implementation",
        choices=("reentrant", "non-reentrant"),
        default="reentrant",
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-target-modules", default="all-linear")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--data-seed", type=int, default=20260722)
    parser.add_argument("--process-group-timeout", type=float, default=600.0)
    args = parser.parse_args(argv)
    args.training_method = "lora"
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    if args.sequence_length < 2:
        parser.error("--sequence-length must be at least two")
    if args.gradient_accumulation_steps <= 0:
        parser.error("--gradient-accumulation-steps must be greater than zero")
    if args.lora_rank <= 0 or args.lora_alpha <= 0:
        parser.error("LoRA rank and alpha must be greater than zero")
    if not 0.0 <= args.lora_dropout < 1.0:
        parser.error("--lora-dropout must be in [0, 1)")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be greater than zero")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    original_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "starting",
        "stage": "starting",
        "rank": rank,
        "world_size": world_size,
        "zero_stage": args.zero_stage,
        "torchrun_local_rank": original_local_rank,
        "physical_device_index": 0,
    }
    stage = "import_torch"
    dist = None
    sampler = None
    started = time.monotonic()

    try:
        import torch
        import torch.distributed as dist  # type: ignore[no-redef]
        import transformers
        from transformers import AutoConfig, Qwen3_5ForCausalLM

        stage = "import_deepspeed"
        import deepspeed
        from deepspeed.utils.logging import logger as deepspeed_logger

        from verification.qwen_sft_memory_worker import (
            _NvmlProcessSampler,
            _configure_training_model,
            _dtype,
            _load_text_config,
            _make_batches,
            _memory_record,
            _parameter_summary,
            _reset_peak,
        )

        deepspeed_logger.setLevel(logging.WARNING)
        logging.getLogger("deepspeed").setLevel(logging.WARNING)
        if world_size not in (2, 4):
            raise ValueError(
                f"expected world_size to be 2 or 4, got {world_size}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("real CUDA is not available")
        if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("physical CUDA device does not support BF16")

        transformers.utils.import_utils._torchvision_available = False
        transformers.utils.import_utils._torchvision_version = "0.0"
        model_dir = Path(args.model_dir).expanduser().resolve()
        if not model_dir.is_dir():
            raise FileNotFoundError(f"model directory not found: {model_dir}")

        dtype = _dtype(torch, args.dtype)
        text_config = _load_text_config(AutoConfig, model_dir)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        sampler = _NvmlProcessSampler(enabled=True)

        report.update(
            {
                "model_dir": str(model_dir),
                "training_method": "lora",
                "dtype": args.dtype,
                "attention_implementation": args.attention_implementation,
                "gradient_checkpointing": bool(
                    args.gradient_checkpointing
                ),
                "checkpoint_implementation": (
                    args.checkpoint_implementation
                    if args.gradient_checkpointing
                    else None
                ),
                "gradient_accumulation_steps": (
                    args.gradient_accumulation_steps
                ),
                "batch_size": args.batch_size,
                "sequence_length": args.sequence_length,
                "learning_rate": args.learning_rate,
                "lora": {
                    "rank": args.lora_rank,
                    "alpha": args.lora_alpha,
                    "dropout": args.lora_dropout,
                    "target_modules": args.lora_target_modules,
                },
                "seed": args.seed,
                "data_seed": args.data_seed,
                "torch_version": str(torch.__version__),
                "torch_cuda_version": str(torch.version.cuda),
                "transformers_version": str(transformers.__version__),
                "deepspeed_version": str(deepspeed.__version__),
                "physical_device_name": torch.cuda.get_device_name(0),
                "physical_compute_capability": list(
                    torch.cuda.get_device_capability(0)
                ),
            }
        )

        stage = "init_process_group"
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=args.process_group_timeout),
        )
        os.environ["LOCAL_RANK"] = "0"

        timeline: list[dict[str, Any]] = []
        torch.cuda.empty_cache()
        _reset_peak(torch, "real")
        timeline.append(_memory_record(torch, sampler, "real", "baseline"))

        stage = "load_model"
        load_started = time.monotonic()
        model = Qwen3_5ForCausalLM.from_pretrained(
            str(model_dir),
            dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
            attn_implementation=args.attention_implementation,
        )
        model.config.use_cache = False
        checkpointing_requested = bool(args.gradient_checkpointing)
        args.gradient_checkpointing = False
        model, training_metadata = _configure_training_model(
            model,
            args,
            static=False,
        )
        args.gradient_checkpointing = checkpointing_requested
        if checkpointing_requested:
            use_reentrant = args.checkpoint_implementation == "reentrant"
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": use_reentrant,
                }
            )
            model.enable_input_require_grads()
            training_metadata["checkpointing_analysis"] = (
                "executed_reentrant"
                if use_reentrant
                else "executed_non_reentrant"
            )
        model.train()
        model.to(device)
        torch.cuda.synchronize()
        model_load_seconds = time.monotonic() - load_started
        parameter_summary = _parameter_summary(model)
        model_load_record = _memory_record(
            torch,
            sampler,
            "real",
            "after_model_load",
        )
        timeline.append(model_load_record)

        trainable_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise AssertionError("LoRA did not create trainable parameters")
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=args.learning_rate,
            foreach=False,
        )
        config = {
            "train_batch_size": (
                args.batch_size
                * world_size
                * args.gradient_accumulation_steps
            ),
            "train_micro_batch_size_per_gpu": args.batch_size,
            "gradient_accumulation_steps": (
                args.gradient_accumulation_steps
            ),
            "gradient_clipping": 1.0,
            "zero_optimization": {
                "stage": args.zero_stage,
                "contiguous_gradients": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5_000_000,
                "allgather_partitions": True,
                "allgather_bucket_size": 5_000_000,
                "overlap_comm": False,
                "stage3_prefetch_bucket_size": 5_000_000,
                "stage3_param_persistence_threshold": 100_000,
            },
            "zero_allow_untested_optimizer": True,
            "fp16": {"enabled": False},
            "bf16": {"enabled": args.dtype == "bfloat16"},
            "steps_per_print": 1_000_000,
        }

        stage = "construct_engine"
        engine_started = time.monotonic()
        engine, engine_optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=config,
            dist_init_required=False,
        )
        torch.cuda.synchronize()
        engine_seconds = time.monotonic() - engine_started
        engine_record = _memory_record(
            torch,
            sampler,
            "real",
            "after_deepspeed_engine",
        )
        timeline.append(engine_record)
        report["engine"] = {
            "type": type(engine).__name__,
            "optimizer_type": type(engine_optimizer).__name__,
            "effective_zero_stage": int(engine.zero_optimization_stage()),
            "gradient_accumulation_steps": int(
                engine.gradient_accumulation_steps()
            ),
        }

        probe_name, update_probe = next(
            (name, parameter)
            for name, parameter in engine.module.named_parameters()
            if parameter.requires_grad and "lora_B" in name
        )
        update_probe_before = _gathered_squared_norm(
            update_probe,
            deepspeed,
        )

        stage = "prepare_batches"
        batches = _make_batches(
            torch,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            vocab_size=int(text_config.vocab_size),
            seed=(
                args.data_seed
                + rank * args.gradient_accumulation_steps
            ),
            count=args.gradient_accumulation_steps,
        )
        device_batches = [
            {
                "input_ids": batch["input_ids"].to(device),
                "labels": batch["labels"].to(device),
                "position_ids": batch["position_ids"].to(device),
            }
            for batch in batches["items"]
        ]
        input_record = _memory_record(
            torch,
            sampler,
            "real",
            "after_inputs",
        )
        timeline.append(input_record)

        forward_records: list[dict[str, Any]] = []
        backward_records: list[dict[str, Any]] = []
        optimizer_records: list[dict[str, Any]] = []
        losses: list[float] = []
        microstep_seconds: list[dict[str, float]] = []
        global_steps: list[int] = []
        for microstep, device_batch in enumerate(device_batches, start=1):
            stage = f"microstep_{microstep}_forward"
            _reset_peak(torch, "real")
            phase_started = time.monotonic()
            outputs = engine(**device_batch, use_cache=False)
            loss = outputs.loss
            forward_seconds = time.monotonic() - phase_started
            loss_value = float(loss.detach().float().cpu().item())
            if not math.isfinite(loss_value):
                raise AssertionError(f"non-finite loss: {loss_value}")
            losses.append(loss_value)
            forward_record = _memory_record(
                torch,
                sampler,
                "real",
                f"microstep_{microstep}_after_forward",
            )
            timeline.append(forward_record)
            forward_records.append(forward_record)

            stage = f"microstep_{microstep}_backward"
            _reset_peak(torch, "real")
            phase_started = time.monotonic()
            engine.backward(loss)
            backward_seconds = time.monotonic() - phase_started
            backward_record = _memory_record(
                torch,
                sampler,
                "real",
                f"microstep_{microstep}_after_backward",
            )
            timeline.append(backward_record)
            backward_records.append(backward_record)

            stage = f"microstep_{microstep}_optimizer"
            _reset_peak(torch, "real")
            phase_started = time.monotonic()
            engine.step()
            optimizer_seconds = time.monotonic() - phase_started
            optimizer_record = _memory_record(
                torch,
                sampler,
                "real",
                f"microstep_{microstep}_after_optimizer",
            )
            timeline.append(optimizer_record)
            optimizer_records.append(optimizer_record)
            global_steps.append(int(engine.global_steps))
            microstep_seconds.append(
                {
                    "forward": forward_seconds,
                    "backward": backward_seconds,
                    "optimizer": optimizer_seconds,
                }
            )
            del outputs, loss

        if engine.global_steps != 1:
            raise AssertionError(
                f"expected one optimizer step, got {engine.global_steps}"
            )

        stage = "validate_update"
        update_probe_after = _gathered_squared_norm(
            update_probe,
            deepspeed,
        )
        if not math.isfinite(update_probe_after):
            raise AssertionError("LoRA update probe is non-finite")
        if update_probe_after <= update_probe_before:
            raise AssertionError(
                "DeepSpeed did not update the LoRA adapter probe: "
                f"{update_probe_before} -> {update_probe_after}"
            )

        probe_tensor = torch.tensor(
            [update_probe_after],
            dtype=torch.float64,
            device=device,
        )
        gathered_probes = [
            torch.empty_like(probe_tensor) for _ in range(world_size)
        ]
        dist.all_gather(gathered_probes, probe_tensor)
        torch.cuda.synchronize()
        gathered_probe_values = [
            float(value.cpu().item()) for value in gathered_probes
        ]
        if any(
            not math.isclose(
                value,
                update_probe_after,
                rel_tol=1e-5,
                abs_tol=1e-12,
            )
            for value in gathered_probe_values
        ):
            raise AssertionError(
                "LoRA parameters differ across ranks: "
                f"{gathered_probe_values}"
            )

        stage = "barrier"
        dist.barrier(device_ids=[0])

        phase_records = [
            *forward_records,
            *backward_records,
            *optimizer_records,
        ]
        memory_phases = {
            "model_load_current_bytes": int(
                model_load_record["allocated_bytes"]
            ),
            "engine_current_bytes": int(engine_record["allocated_bytes"]),
            "inputs_current_bytes": int(input_record["allocated_bytes"]),
            "forward_peak_bytes": _max_phase_bytes(forward_records),
            "backward_peak_bytes": _max_phase_bytes(backward_records),
            "optimizer_peak_bytes": _max_phase_bytes(optimizer_records),
            "overall_peak_bytes": _max_phase_bytes(phase_records),
        }
        report.update(
            {
                "status": "success",
                "stage": "complete",
                "peft_version": training_metadata["peft_version"],
                "checkpointing_analysis": training_metadata[
                    "checkpointing_analysis"
                ],
                "parameters": parameter_summary,
                "batch": {
                    "masked_prefix_tokens": batches[
                        "masked_prefix_tokens"
                    ],
                    "fingerprint_sha256": batches[
                        "fingerprint_sha256"
                    ],
                    "microbatch_fingerprints_sha256": batches[
                        "microbatch_fingerprints_sha256"
                    ],
                },
                "microbatch_losses": losses,
                "loss": sum(losses) / len(losses),
                "global_steps_after_microsteps": global_steps,
                "optimizer_update_probe": {
                    "name": probe_name,
                    "dtype": str(update_probe.dtype),
                    "before_squared_norm": update_probe_before,
                    "after_squared_norm": update_probe_after,
                    "gathered_after_squared_norms": (
                        gathered_probe_values
                    ),
                },
                "phase_seconds": {
                    "model_load": model_load_seconds,
                    "engine_initialize": engine_seconds,
                    "microsteps": microstep_seconds,
                },
                "elapsed_seconds": time.monotonic() - started,
                "memory_phases": memory_phases,
                "timeline": timeline,
            }
        )
        _write_report(args.report_dir, rank, report)
        print(
            json.dumps(
                {
                    "status": "success",
                    "rank": rank,
                    "zero_stage": args.zero_stage,
                    "loss": report["loss"],
                    "overall_peak_bytes": memory_phases[
                        "overall_peak_bytes"
                    ],
                    "update_probe_after": update_probe_after,
                },
                sort_keys=True,
            ),
            flush=True,
        )

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
                "elapsed_seconds": time.monotonic() - started,
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
    finally:
        if sampler is not None:
            sampler.close()


if __name__ == "__main__":
    raise SystemExit(main())
