#!/usr/bin/env python3
"""Measure one Qwen SFT step under two-rank Hybrid FSDP FULL_SHARD."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable


os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA_VERSION = "fakegpu.qwen_fsdp_sft_memory.rank.v1"


def _write_report(report_dir: Path, rank: int, report: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"rank_{rank}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _unique_tensors(items: Iterable[Any]) -> list[Any]:
    unique: dict[int, Any] = {}
    for item in items:
        unique[id(item)] = item
    return list(unique.values())


def _tensor_bytes(items: Iterable[Any]) -> int:
    return sum(
        int(item.numel()) * int(item.element_size())
        for item in _unique_tensors(items)
    )


def _parameter_unit_specs(model: Any, layer_type: type[Any]) -> list[dict[str, Any]]:
    wrapped_parameter_ids: set[int] = set()
    units: list[dict[str, Any]] = []

    def append_unit(name: str, parameters: Iterable[Any]) -> None:
        unique = _unique_tensors(parameters)
        if not unique:
            return
        element_sizes = {int(parameter.element_size()) for parameter in unique}
        if len(element_sizes) != 1:
            raise ValueError(
                f"FSDP unit {name!r} mixes parameter element sizes: "
                f"{sorted(element_sizes)}"
            )
        units.append(
            {
                "name": name,
                "numel": sum(int(parameter.numel()) for parameter in unique),
                "element_size": element_sizes.pop(),
                "parameter_tensors": len(unique),
            }
        )

    for name, module in model.named_modules():
        if not isinstance(module, layer_type):
            continue
        parameters = _unique_tensors(module.parameters())
        wrapped_parameter_ids.update(id(parameter) for parameter in parameters)
        append_unit(name, parameters)

    root_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in wrapped_parameter_ids
    ]
    append_unit("<root>", root_parameters)
    return units


def _optimizer_state_tensors(optimizer: Any) -> list[Any]:
    tensors = []
    for state in optimizer.state.values():
        for value in state.values():
            if hasattr(value, "numel") and hasattr(value, "element_size"):
                tensors.append(value)
    return tensors


def _gradient_tensors(model: Any) -> list[Any]:
    return [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--attention-implementation",
        choices=("eager", "sdpa"),
        default="sdpa",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--data-seed", type=int, default=20260721)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--process-group-timeout", type=float, default=300.0)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    if args.sequence_length < 2:
        parser.error("--sequence-length must be at least two")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be greater than zero")
    if args.process_group_timeout <= 0:
        parser.error("--process-group-timeout must be greater than zero")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "starting",
        "rank": rank,
        "world_size": world_size,
    }
    stage = "import_torch"
    dist = None
    sampler = None
    started = time.monotonic()
    try:
        import torch
        import torch.distributed as dist  # type: ignore[no-redef]
        import transformers
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardingStrategy,
        )
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy
        from transformers import AutoConfig, Qwen3_5ForCausalLM

        from fakegpu.fsdp_memory import build_full_shard_plan
        from verification.qwen_sft_memory_worker import (
            _NvmlProcessSampler,
            _dtype,
            _load_text_config,
            _make_batches,
            _memory_record,
            _parameter_summary,
            _reset_peak,
        )

        if world_size != 2:
            raise ValueError(f"expected world_size=2, got {world_size}")
        if not torch.cuda.is_available():
            raise RuntimeError("real CUDA is not available")

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
                "dtype": args.dtype,
                "attention_implementation": args.attention_implementation,
                "optimizer": "adamw_single_tensor",
                "training_method": "full",
                "gradient_checkpointing": bool(args.gradient_checkpointing),
                "gradient_accumulation_steps": 1,
                "batch_size": args.batch_size,
                "sequence_length": args.sequence_length,
                "seed": args.seed,
                "data_seed": args.data_seed,
                "torch_version": str(torch.__version__),
                "torch_cuda_version": str(torch.version.cuda),
                "transformers_version": str(transformers.__version__),
                "gpu_name": str(torch.cuda.get_device_name(0)),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
            }
        )

        stage = "init_process_group"
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=args.process_group_timeout),
        )

        timeline: list[dict[str, Any]] = []
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
        model.train()
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        layer_type = type(model.model.layers[0])
        logical_parameters = _parameter_summary(model)
        unit_specs = _parameter_unit_specs(model, layer_type)
        sharding_plan = build_full_shard_plan(
            unit_specs,
            world_size=world_size,
        )

        stage = "construct_fsdp"
        fsdp_model = FSDP(
            model,
            auto_wrap_policy=ModuleWrapPolicy({layer_type}),
            device_id=device,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            backward_prefetch=None,
            forward_prefetch=False,
            limit_all_gathers=True,
            use_orig_params=False,
        )
        torch.cuda.synchronize()
        load_seconds = time.monotonic() - load_started
        sharded_parameters = _parameter_summary(fsdp_model)
        wrap_record = _memory_record(
            torch,
            sampler,
            "real",
            "after_fsdp_wrap",
        )
        timeline.append(wrap_record)

        actual_shard_bytes = int(sharded_parameters["parameter_bytes"])
        expected_shard_bytes = int(
            sharding_plan["local_shard_parameter_bytes"]
        )
        if actual_shard_bytes != expected_shard_bytes:
            raise AssertionError(
                "FSDP local parameter storage did not match the unit plan: "
                f"{actual_shard_bytes} != {expected_shard_bytes}"
            )

        stage = "prepare_batch"
        batches = _make_batches(
            torch,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            vocab_size=int(text_config.vocab_size),
            seed=args.data_seed,
            count=1,
        )
        batch = batches["items"][0]
        device_batch = {
            "input_ids": batch["input_ids"].to(device),
            "labels": batch["labels"].to(device),
            "position_ids": batch["position_ids"].to(device),
        }
        optimizer = torch.optim.AdamW(
            fsdp_model.parameters(),
            lr=args.learning_rate,
            foreach=False,
        )
        optimizer.zero_grad(set_to_none=True)
        input_record = _memory_record(
            torch,
            sampler,
            "real",
            "after_inputs",
        )
        timeline.append(input_record)

        stage = "forward"
        _reset_peak(torch, "real")
        forward_started = time.monotonic()
        outputs = fsdp_model(**device_batch, use_cache=False)
        loss = outputs.loss
        forward_seconds = time.monotonic() - forward_started
        forward_record = _memory_record(
            torch,
            sampler,
            "real",
            "after_forward",
        )
        timeline.append(forward_record)

        stage = "backward"
        _reset_peak(torch, "real")
        backward_started = time.monotonic()
        loss.backward()
        backward_seconds = time.monotonic() - backward_started
        backward_record = _memory_record(
            torch,
            sampler,
            "real",
            "after_backward",
        )
        timeline.append(backward_record)

        gradients = _gradient_tensors(fsdp_model)
        if not gradients:
            raise AssertionError("FSDP did not produce local gradients")
        if not all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients):
            raise AssertionError("FSDP produced a non-finite local gradient")
        gradient_bytes = _tensor_bytes(gradients)
        if gradient_bytes != expected_shard_bytes:
            raise AssertionError(
                "FSDP local gradient storage did not match the parameter shard: "
                f"{gradient_bytes} != {expected_shard_bytes}"
            )

        stage = "optimizer"
        _reset_peak(torch, "real")
        optimizer_started = time.monotonic()
        optimizer.step()
        optimizer_seconds = time.monotonic() - optimizer_started
        optimizer_record = _memory_record(
            torch,
            sampler,
            "real",
            "after_optimizer",
        )
        timeline.append(optimizer_record)
        optimizer_state_bytes = _tensor_bytes(
            _optimizer_state_tensors(optimizer)
        )

        stage = "synchronize"
        loss_value = float(loss.detach().float().cpu().item())
        if not torch.isfinite(torch.tensor(loss_value)):
            raise AssertionError(f"non-finite loss: {loss_value}")
        loss_digest = hashlib.sha256(f"{loss_value:.9f}".encode("ascii")).hexdigest()
        dist.barrier(device_ids=[0])

        phase_records = (forward_record, backward_record, optimizer_record)
        memory_phases = {
            "model_shard_current_bytes": int(wrap_record["allocated_bytes"]),
            "inputs_current_bytes": int(input_record["allocated_bytes"]),
            "forward_current_bytes": int(forward_record["allocated_bytes"]),
            "forward_peak_bytes": int(forward_record["peak_allocated_bytes"]),
            "backward_current_bytes": int(backward_record["allocated_bytes"]),
            "backward_peak_bytes": int(backward_record["peak_allocated_bytes"]),
            "optimizer_current_bytes": int(optimizer_record["allocated_bytes"]),
            "optimizer_peak_bytes": int(optimizer_record["peak_allocated_bytes"]),
            "graph_peak_bytes": max(
                int(forward_record["peak_allocated_bytes"]),
                int(backward_record["peak_allocated_bytes"]),
            ),
            "overall_peak_bytes": max(
                int(record["peak_allocated_bytes"])
                for record in phase_records
            ),
        }
        report.update(
            {
                "status": "success",
                "stage": "complete",
                "parameters": {
                    "logical": logical_parameters,
                    "local_shard": sharded_parameters,
                    "local_gradient_bytes": gradient_bytes,
                    "optimizer_state_tensor_bytes": optimizer_state_bytes,
                },
                "fsdp": {
                    "sharding_strategy": "FULL_SHARD",
                    "auto_wrap": "transformer_layer",
                    "forward_prefetch": False,
                    "backward_prefetch": None,
                    "limit_all_gathers": True,
                    "use_orig_params": False,
                    "sharding_plan": sharding_plan,
                },
                "batch": {
                    "masked_prefix_tokens": batches["masked_prefix_tokens"],
                    "fingerprint_sha256": batches["fingerprint_sha256"],
                },
                "loss": loss_value,
                "loss_digest_sha256": loss_digest,
                "phase_seconds": {
                    "model_load_and_wrap": load_seconds,
                    "forward": forward_seconds,
                    "backward": backward_seconds,
                    "optimizer": optimizer_seconds,
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
                    "status": report["status"],
                    "rank": rank,
                    "gpu_name": report["gpu_name"],
                    "loss": loss_value,
                    "overall_peak_bytes": memory_phases["overall_peak_bytes"],
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
