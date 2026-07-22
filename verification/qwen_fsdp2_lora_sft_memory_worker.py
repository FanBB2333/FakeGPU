#!/usr/bin/env python3
"""Measure one Qwen LoRA SFT step under two-rank Hybrid FSDP2."""

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

SCHEMA_VERSION = "fakegpu.qwen_fsdp2_lora_sft_memory.rank.v1"


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


def _local_tensor(value: Any) -> Any:
    to_local = getattr(value, "to_local", None)
    return to_local() if callable(to_local) else value


def _local_logical_bytes(items: Iterable[Any]) -> int:
    return sum(
        int(local.numel()) * int(local.element_size())
        for local in (_local_tensor(item) for item in _unique_tensors(items))
    )


def _local_storage_bytes(items: Iterable[Any], *, cuda_only: bool = False) -> int:
    storages: dict[int, int] = {}
    fallback = 0
    for item in _unique_tensors(items):
        local = _local_tensor(item)
        if cuda_only and str(getattr(local, "device", "")).split(":", 1)[0] != "cuda":
            continue
        try:
            storage = local.untyped_storage()
            key = int(getattr(storage, "_cdata", id(storage)))
            storages[key] = max(storages.get(key, 0), int(storage.nbytes()))
        except Exception:
            fallback += int(local.numel()) * int(local.element_size())
    return sum(storages.values()) + fallback


def _local_parameter_summary(model: Any) -> dict[str, Any]:
    parameters = _unique_tensors(model.parameters())
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    frozen = [parameter for parameter in parameters if not parameter.requires_grad]
    return {
        "parameter_tensors": len(parameters),
        "parameter_storage_bytes": _local_storage_bytes(parameters),
        "logical_parameter_bytes": _local_logical_bytes(parameters),
        "trainable_parameter_tensors": len(trainable),
        "trainable_parameter_storage_bytes": _local_storage_bytes(trainable),
        "logical_trainable_parameter_bytes": _local_logical_bytes(trainable),
        "frozen_parameter_tensors": len(frozen),
        "frozen_parameter_storage_bytes": _local_storage_bytes(frozen),
        "logical_frozen_parameter_bytes": _local_logical_bytes(frozen),
        "dtypes": sorted({str(parameter.dtype) for parameter in parameters}),
        "devices": sorted({str(parameter.device) for parameter in parameters}),
    }


def _parameter_unit_specs(model: Any, layer_type: type[Any]) -> list[dict[str, Any]]:
    named_parameters = list(model.named_parameters())
    metadata_by_id = {
        id(parameter): (index, name, parameter)
        for index, (name, parameter) in enumerate(named_parameters)
    }
    wrapped_parameter_ids: set[int] = set()
    units: list[dict[str, Any]] = []

    def append_unit(
        name: str,
        parameters: Iterable[Any],
        *,
        is_root: bool,
    ) -> None:
        unique = _unique_tensors(parameters)
        if not unique:
            return
        specs = []
        for parameter in unique:
            index, parameter_name, _ = metadata_by_id[id(parameter)]
            specs.append(
                {
                    "name": parameter_name,
                    "parameter_index": index,
                    "shape": list(parameter.shape),
                    "numel": int(parameter.numel()),
                    "dtype": str(parameter.dtype),
                    "element_size": int(parameter.element_size()),
                    "gradient_element_size": int(parameter.element_size()),
                    "trainable": bool(parameter.requires_grad),
                }
            )
        units.append(
            {
                "name": name,
                "is_root": is_root,
                "parameters": specs,
            }
        )

    for name, module in model.named_modules():
        if not isinstance(module, layer_type):
            continue
        parameters = _unique_tensors(module.parameters())
        wrapped_parameter_ids.update(id(parameter) for parameter in parameters)
        append_unit(name, parameters, is_root=False)

    root_parameters = [
        parameter
        for _, parameter in named_parameters
        if id(parameter) not in wrapped_parameter_ids
    ]
    append_unit("<root>", root_parameters, is_root=True)
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
        if parameter.requires_grad and parameter.grad is not None
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
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-target-modules", default="all-linear")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--data-seed", type=int, default=20260721)
    parser.add_argument("--process-group-timeout", type=float, default=300.0)
    args = parser.parse_args(argv)
    args.training_method = "lora"
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    if args.sequence_length < 2:
        parser.error("--sequence-length must be at least two")
    if args.lora_rank <= 0 or args.lora_alpha <= 0:
        parser.error("LoRA rank and alpha must be greater than zero")
    if not 0.0 <= args.lora_dropout < 1.0:
        parser.error("--lora-dropout must be in [0, 1)")
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
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard
        from transformers import AutoConfig, Qwen3_5ForCausalLM

        from fakegpu.fsdp_memory import build_fully_shard_plan
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
                "training_method": "lora",
                "lora": {
                    "rank": args.lora_rank,
                    "alpha": args.lora_alpha,
                    "dropout": args.lora_dropout,
                    "target_modules": args.lora_target_modules,
                },
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
        model, training_metadata = _configure_training_model(
            model,
            args,
            static=False,
        )
        model.train()

        layer_modules = list(model.base_model.model.model.layers)
        layer_type = type(layer_modules[0])
        logical_parameters = _parameter_summary(model)
        unit_specs = _parameter_unit_specs(model, layer_type)
        sharding_plan = build_fully_shard_plan(
            unit_specs,
            world_size=world_size,
        )

        stage = "construct_fsdp2"
        model.to(device)
        mesh = init_device_mesh(
            "cuda",
            (world_size,),
            mesh_dim_names=("dp",),
        )
        for layer in layer_modules:
            fully_shard(
                layer,
                mesh=mesh,
                reshard_after_forward=True,
            )
        fully_shard(
            model,
            mesh=mesh,
            reshard_after_forward=True,
        )
        torch.cuda.synchronize()
        load_seconds = time.monotonic() - load_started
        local_parameters = _local_parameter_summary(model)
        wrap_record = _memory_record(torch, sampler, "real", "after_fsdp2_wrap")
        timeline.append(wrap_record)

        expected_rank_shard = sharding_plan["rank_shards"][rank]
        actual_shard_bytes = int(local_parameters["parameter_storage_bytes"])
        expected_shard_bytes = int(
            expected_rank_shard["parameter_storage_bytes"]
        )
        if actual_shard_bytes != expected_shard_bytes:
            raise AssertionError(
                "FSDP2 local parameter storage did not match the plan: "
                f"{actual_shard_bytes} != {expected_shard_bytes}"
            )
        actual_trainable_bytes = int(
            local_parameters["logical_trainable_parameter_bytes"]
        )
        expected_trainable_bytes = int(
            expected_rank_shard["logical_trainable_parameter_bytes"]
        )
        if actual_trainable_bytes != expected_trainable_bytes:
            raise AssertionError(
                "FSDP2 local trainable parameters did not match the plan: "
                f"{actual_trainable_bytes} != {expected_trainable_bytes}"
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
        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=args.learning_rate,
            foreach=False,
        )
        optimizer.zero_grad(set_to_none=True)
        input_record = _memory_record(torch, sampler, "real", "after_inputs")
        timeline.append(input_record)

        stage = "forward"
        _reset_peak(torch, "real")
        forward_started = time.monotonic()
        outputs = model(**device_batch, use_cache=False)
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

        gradients = _gradient_tensors(model)
        if not gradients:
            raise AssertionError("FSDP2 did not produce local LoRA gradients")
        if not all(
            bool(torch.isfinite(_local_tensor(gradient)).all().item())
            for gradient in gradients
        ):
            raise AssertionError("FSDP2 produced a non-finite LoRA gradient")
        gradient_storage_bytes = _local_storage_bytes(gradients)
        expected_gradient_bytes = int(
            expected_rank_shard["gradient_storage_bytes"]
        )
        if gradient_storage_bytes != expected_gradient_bytes:
            raise AssertionError(
                "FSDP2 local gradient storage did not match the plan: "
                f"{gradient_storage_bytes} != {expected_gradient_bytes}"
            )
        update_probe = next(
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and "lora_B" in name
        )
        update_probe_before = float(
            _local_tensor(update_probe).detach().float().square().sum().cpu().item()
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
        optimizer_state_bytes = _local_storage_bytes(
            _optimizer_state_tensors(optimizer),
            cuda_only=True,
        )
        expected_optimizer_state_bytes = 2 * expected_trainable_bytes
        if optimizer_state_bytes != expected_optimizer_state_bytes:
            raise AssertionError(
                "FSDP2 AdamW state did not match the local trainable shard: "
                f"{optimizer_state_bytes} != {expected_optimizer_state_bytes}"
            )
        update_probe_after = float(
            _local_tensor(update_probe).detach().float().square().sum().cpu().item()
        )
        if update_probe_after == update_probe_before:
            raise AssertionError("AdamW did not update the LoRA adapter probe")

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
                int(record["peak_allocated_bytes"]) for record in phase_records
            ),
        }
        report.update(
            {
                "status": "success",
                "stage": "complete",
                "peft_version": training_metadata["peft_version"],
                "parameters": {
                    "logical": logical_parameters,
                    "local_shard": local_parameters,
                    "local_gradient_storage_bytes": gradient_storage_bytes,
                    "local_gradient_logical_bytes": _local_logical_bytes(gradients),
                    "optimizer_state_tensor_bytes": optimizer_state_bytes,
                },
                "fsdp2": {
                    "api": "fully_shard",
                    "mesh": [world_size],
                    "shard_placement": "Shard(0)",
                    "reshard_after_forward": True,
                    "mixed_precision_policy": None,
                    "auto_backward_prefetch": True,
                    "sharding_plan": sharding_plan,
                },
                "batch": {
                    "masked_prefix_tokens": batches["masked_prefix_tokens"],
                    "fingerprint_sha256": batches["fingerprint_sha256"],
                },
                "loss": loss_value,
                "loss_digest_sha256": loss_digest,
                "optimizer_update_probe": {
                    "before_squared_norm": update_probe_before,
                    "after_squared_norm": update_probe_after,
                },
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
