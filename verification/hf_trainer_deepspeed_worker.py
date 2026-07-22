#!/usr/bin/env python3
"""Run one Hugging Face Trainer workload through Hybrid DeepSpeed."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any


os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA_VERSION = "fakegpu.hf_trainer_deepspeed.rank.v1"


def _write_report(report_dir: Path, rank: int, report: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"rank_{rank}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _make_dataset(
    torch: Any,
    *,
    size: int,
    sequence_length: int,
    vocab_size: int,
    seed: int,
) -> tuple[Any, str]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    items = []
    digest = hashlib.sha256()
    for _ in range(size):
        input_ids = torch.randint(
            0,
            vocab_size,
            (sequence_length,),
            generator=generator,
            dtype=torch.long,
        )
        labels = input_ids.clone()
        labels[: max(1, sequence_length // 4)] = -100
        attention_mask = torch.ones_like(input_ids)
        for tensor in (input_ids, labels, attention_mask):
            digest.update(tensor.numpy().tobytes())
        items.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )

    class _TokenDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(items)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return {key: value.clone() for key, value in items[index].items()}

    return _TokenDataset(), digest.hexdigest()


def _gather_parameter(parameter: Any, deepspeed: Any) -> Any:
    with deepspeed.zero.GatheredParameters(
        [parameter],
        modifier_rank=None,
    ):
        return parameter.detach().float().cpu().clone()


def _tensor_sha256(tensor: Any) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def _select_probe(model: Any, workload: str) -> tuple[str, Any]:
    candidates = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if workload == "qwen-lora":
        candidates = [
            item for item in candidates if "lora_B" in item[0]
        ]
    if not candidates:
        raise AssertionError("model has no trainable probe parameter")
    return candidates[0]


def _json_safe_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_safe_config(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_config(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload",
        choices=("tiny", "qwen-lora"),
        default="tiny",
    )
    parser.add_argument("--model-dir")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--trainer-output-dir", type=Path, required=True)
    parser.add_argument("--zero-stage", type=int, choices=(2, 3), required=True)
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default="bf16",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args(argv)
    if args.workload == "qwen-lora" and not args.model_dir:
        parser.error("--model-dir is required for --workload qwen-lora")
    if args.batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        parser.error("batch size and accumulation steps must be positive")
    if args.sequence_length < 2 or args.max_steps <= 0:
        parser.error("sequence length must be >= 2 and max steps positive")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    original_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "starting",
        "stage": "starting",
        "rank": rank,
        "world_size": world_size,
        "torchrun_local_rank": original_local_rank,
        "physical_device_index": 0,
        "workload": args.workload,
        "zero_stage": args.zero_stage,
        "precision": args.precision,
    }
    stage = "import_torch"
    dist = None
    started = time.monotonic()

    try:
        import torch
        import torch.distributed as dist  # type: ignore[no-redef]
        import transformers
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            Qwen3_5ForCausalLM,
            Trainer,
            TrainingArguments,
        )

        stage = "import_deepspeed"
        import accelerate
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

        transformers.utils.import_utils._torchvision_available = False
        transformers.utils.import_utils._torchvision_version = "0.0"
        torch.cuda.set_device(0)
        os.environ["LOCAL_RANK"] = "0"
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)

        deepspeed_config = {
            "train_micro_batch_size_per_gpu": "auto",
            "train_batch_size": "auto",
            "gradient_accumulation_steps": "auto",
            "gradient_clipping": "auto",
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
                "stage3_gather_16bit_weights_on_model_save": True,
            },
            "zero_allow_untested_optimizer": True,
            "fp16": {"enabled": "auto"},
            "bf16": {"enabled": "auto"},
            "steps_per_print": 1_000_000,
        }

        stage = "construct_training_arguments"
        args.trainer_output_dir.mkdir(parents=True, exist_ok=True)
        training_args = TrainingArguments(
            output_dir=str(args.trainer_output_dir),
            do_train=True,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            max_grad_norm=1.0,
            bf16=args.precision == "bf16",
            fp16=False,
            gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs=(
                {"use_reentrant": True}
                if args.gradient_checkpointing
                else None
            ),
            optim="adamw_torch",
            logging_strategy="steps",
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            disable_tqdm=True,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            remove_unused_columns=True,
            ddp_find_unused_parameters=False,
            deepspeed=deepspeed_config,
            seed=args.seed,
            data_seed=args.seed,
        )

        stage = "load_model"
        if args.workload == "tiny":
            config = AutoConfig.for_model(
                "qwen2",
                vocab_size=256,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=max(64, args.sequence_length),
                use_cache=False,
            )
            model = AutoModelForCausalLM.from_config(config)
            vocab_size = int(config.vocab_size)
            peft_version = None
        else:
            from peft import LoraConfig, get_peft_model
            import peft

            model_dir = Path(args.model_dir).expanduser().resolve()
            if not model_dir.is_dir():
                raise FileNotFoundError(
                    f"model directory not found: {model_dir}"
                )
            dtype = (
                torch.bfloat16
                if args.precision == "bf16"
                else torch.float32
            )
            model = Qwen3_5ForCausalLM.from_pretrained(
                str(model_dir),
                dtype=dtype,
                low_cpu_mem_usage=True,
                local_files_only=True,
                attn_implementation="sdpa",
            )
            model = get_peft_model(
                model,
                LoraConfig(
                    r=args.lora_rank,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=0.0,
                    target_modules="all-linear",
                    bias="none",
                    task_type="CAUSAL_LM",
                ),
                autocast_adapter_dtype=False,
            )
            model.config.use_cache = False
            if args.gradient_checkpointing:
                model.enable_input_require_grads()
            vocab_size = int(model.config.vocab_size)
            peft_version = str(peft.__version__)

        model.config.use_cache = False
        model.train()
        dataset_size = max(
            8,
            world_size
            * args.batch_size
            * args.gradient_accumulation_steps
            * args.max_steps
            * 2,
        )
        train_dataset, dataset_fingerprint = _make_dataset(
            torch,
            size=dataset_size,
            sequence_length=args.sequence_length,
            vocab_size=vocab_size,
            seed=args.seed + 1,
        )

        probe_name, probe_parameter = _select_probe(model, args.workload)
        probe_before = _gather_parameter(probe_parameter, deepspeed)

        stage = "construct_trainer"
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )

        stage = "trainer_train"
        train_output = trainer.train()
        torch.cuda.synchronize()
        if not math.isfinite(float(train_output.training_loss)):
            raise AssertionError(
                f"Trainer returned non-finite loss: {train_output.training_loss}"
            )
        if int(trainer.state.global_step) != args.max_steps:
            raise AssertionError(
                "Trainer global step mismatch: "
                f"{trainer.state.global_step} != {args.max_steps}"
            )

        engine = trainer.model_wrapped
        if not isinstance(engine, deepspeed.DeepSpeedEngine):
            raise AssertionError(
                f"Trainer did not create DeepSpeedEngine: {type(engine)}"
            )
        effective_zero_stage = int(engine.zero_optimization_stage())
        if effective_zero_stage != args.zero_stage:
            raise AssertionError(
                f"Trainer created ZeRO-{effective_zero_stage}, expected {args.zero_stage}"
            )

        stage = "validate_parameter_update"
        probe_after = _gather_parameter(probe_parameter, deepspeed)
        if probe_after.shape != probe_before.shape:
            raise AssertionError(
                f"probe shape changed: {probe_before.shape} -> {probe_after.shape}"
            )
        update_norm = float(
            (probe_after - probe_before).float().norm().item()
        )
        if not math.isfinite(update_norm) or update_norm <= 0:
            raise AssertionError(
                f"Trainer did not update probe {probe_name}: {update_norm}"
            )

        probe_device = probe_after.to(device="cuda:0", dtype=torch.float32)
        gathered = [
            torch.empty_like(probe_device) for _ in range(world_size)
        ]
        dist.all_gather(gathered, probe_device)
        torch.cuda.synchronize()
        max_rank_difference = max(
            float((value - probe_device).abs().max().cpu().item())
            for value in gathered
        )
        tolerance = 1e-6 if args.precision == "fp32" else 2e-2
        if max_rank_difference > tolerance:
            raise AssertionError(
                "Trainer parameters differ across ranks: "
                f"max difference {max_rank_difference}"
            )

        resolved_config = _json_safe_config(
            training_args.hf_deepspeed_config.config
        )
        stage = "barrier"
        dist.barrier(device_ids=[0])
        peak_allocated = int(torch.cuda.max_memory_allocated(0))
        current_allocated = int(torch.cuda.memory_allocated(0))
        report.update(
            {
                "status": "success",
                "stage": "complete",
                "torch_version": str(torch.__version__),
                "torch_cuda_version": str(torch.version.cuda),
                "transformers_version": str(transformers.__version__),
                "accelerate_version": str(accelerate.__version__),
                "deepspeed_version": str(deepspeed.__version__),
                "peft_version": peft_version,
                "physical_device_name": torch.cuda.get_device_name(0),
                "physical_compute_capability": list(
                    torch.cuda.get_device_capability(0)
                ),
                "trainer_identity": {
                    "process_index": int(training_args.process_index),
                    "local_process_index": int(
                        training_args.local_process_index
                    ),
                    "world_size": int(training_args.world_size),
                    "is_world_process_zero": bool(
                        trainer.is_world_process_zero()
                    ),
                },
                "engine": {
                    "type": type(engine).__name__,
                    "optimizer_type": type(engine.optimizer).__name__,
                    "effective_zero_stage": effective_zero_stage,
                },
                "training": {
                    "global_step": int(trainer.state.global_step),
                    "training_loss": float(train_output.training_loss),
                    "metrics": _json_safe_config(train_output.metrics),
                    "batch_size": args.batch_size,
                    "gradient_accumulation_steps": (
                        args.gradient_accumulation_steps
                    ),
                    "max_steps": args.max_steps,
                    "sequence_length": args.sequence_length,
                    "gradient_checkpointing": bool(
                        args.gradient_checkpointing
                    ),
                },
                "dataset": {
                    "size": dataset_size,
                    "fingerprint_sha256": dataset_fingerprint,
                },
                "parameter_update_probe": {
                    "name": probe_name,
                    "numel": int(probe_after.numel()),
                    "before_sha256": _tensor_sha256(probe_before),
                    "after_sha256": _tensor_sha256(probe_after),
                    "before_squared_norm": float(
                        probe_before.square().sum().item()
                    ),
                    "after_squared_norm": float(
                        probe_after.square().sum().item()
                    ),
                    "update_norm": update_norm,
                    "gathered_after_sha256": [
                        _tensor_sha256(value.float().cpu())
                        for value in gathered
                    ],
                    "max_rank_difference": max_rank_difference,
                },
                "memory": {
                    "current_allocated_bytes": current_allocated,
                    "peak_allocated_bytes": peak_allocated,
                },
                "resolved_deepspeed_config": resolved_config,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        _write_report(args.report_dir, rank, report)
        print(
            json.dumps(
                {
                    "status": "success",
                    "rank": rank,
                    "workload": args.workload,
                    "zero_stage": args.zero_stage,
                    "global_step": trainer.state.global_step,
                    "training_loss": train_output.training_loss,
                    "update_norm": update_norm,
                    "peak_allocated_bytes": peak_allocated,
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
