#!/usr/bin/env python3
"""Measure or statically estimate a text-only Qwen3.5 SFT optimizer step."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any


os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA_VERSION = "fakegpu.qwen_sft_memory_worker.v3"


class _NvmlProcessSampler:
    def __init__(self, enabled: bool):
        self.reason: str | None = None
        self._pynvml: Any = None
        self._handle: Any = None
        if not enabled:
            self.reason = "disabled outside real-CUDA mode"
            return
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as exc:
            self.reason = f"{type(exc).__name__}: {exc}"

    def sample(self) -> dict[str, Any]:
        if self._pynvml is None or self._handle is None:
            return {"status": "unavailable", "reason": self.reason or "NVML unavailable"}
        pynvml = self._pynvml
        memory = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        process_memory: int | None = None
        process_query = None
        for name in (
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetComputeRunningProcesses_v2",
            "nvmlDeviceGetComputeRunningProcesses",
        ):
            process_query = getattr(pynvml, name, None)
            if process_query is not None:
                break
        if process_query is not None:
            try:
                for process in process_query(self._handle):
                    if int(getattr(process, "pid", -1)) != os.getpid():
                        continue
                    used = int(getattr(process, "usedGpuMemory", 0) or 0)
                    process_memory = used if process_memory is None else max(process_memory, used)
            except Exception as exc:
                self.reason = f"process query failed: {type(exc).__name__}: {exc}"
        return {
            "status": "available",
            "device_used_memory": int(memory.used),
            "device_free_memory": int(memory.free),
            "device_total_memory": int(memory.total),
            "process_memory": process_memory,
            "process_memory_status": "available" if process_memory is not None else "unavailable",
            "process_memory_reason": self.reason if process_memory is None else None,
        }

    def close(self) -> None:
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass


def _memory_record(torch: Any, sampler: _NvmlProcessSampler, mode: str, label: str) -> dict[str, Any]:
    if mode == "real":
        torch.cuda.synchronize()
    record: dict[str, Any] = {
        "label": label,
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    if mode == "fakecuda":
        from fakegpu.torch_patch import memory_snapshot

        devices = memory_snapshot().get("devices") or []
        record["fakecuda_snapshot"] = devices[0] if devices else None
    else:
        record["nvml"] = sampler.sample()
    return record


def _reset_peak(torch: Any, mode: str) -> None:
    if mode == "real":
        torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def _unique_storage_bytes(items: list[Any]) -> int:
    storages: dict[int, int] = {}
    fallback = 0
    for item in items:
        try:
            storage = item.untyped_storage()
            key = int(getattr(storage, "_cdata", id(storage)))
            storages[key] = max(storages.get(key, 0), int(storage.nbytes()))
        except Exception:
            fallback += int(item.numel()) * int(item.element_size())
    return sum(storages.values()) + fallback


def _parameter_summary(model: Any) -> dict[str, Any]:
    parameters = list(model.parameters())
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    buffers = list(model.buffers())

    def nbytes(items: list[Any]) -> int:
        return sum(int(item.numel()) * int(item.element_size()) for item in items)

    physical_parameter_bytes = _unique_storage_bytes(parameters)
    buffer_bytes = _unique_storage_bytes(buffers)
    return {
        "parameter_tensors": len(parameters),
        "parameter_count": sum(int(parameter.numel()) for parameter in parameters),
        "parameter_bytes": physical_parameter_bytes,
        "logical_parameter_bytes": nbytes(parameters),
        "buffer_tensors": len(buffers),
        "buffer_bytes": buffer_bytes,
        "logical_buffer_bytes": nbytes(buffers),
        "persistent_storage_bytes": physical_parameter_bytes + buffer_bytes,
        "trainable_parameter_tensors": len(trainable),
        "trainable_parameter_count": sum(int(parameter.numel()) for parameter in trainable),
        "trainable_parameter_bytes": nbytes(trainable),
        "dtypes": sorted({str(parameter.dtype) for parameter in parameters}),
        "devices": sorted({str(parameter.device) for parameter in parameters}),
    }


def _quantized_parameter_summary(
    summary: dict[str, Any],
    plan: dict[str, Any],
    *,
    static_surrogate: bool,
) -> dict[str, Any]:
    result = dict(summary)
    if static_surrogate:
        physical_parameter_bytes = int(summary["parameter_bytes"]) - int(
            plan["original_weight_bytes"]
        )
        # Meta buffers may be views of one placeholder storage even though the
        # checkpoint loader materializes them independently on CUDA.
        buffer_bytes = int(summary["logical_buffer_bytes"]) + int(
            plan["quantized_storage_bytes"]
        )
    else:
        result["parameter_count"] = int(summary["parameter_count"]) + int(
            plan["original_parameter_count"]
        )
        result["logical_parameter_bytes"] = int(summary["logical_parameter_bytes"]) + int(
            plan["original_parameter_bytes"]
        )
        physical_parameter_bytes = int(summary["parameter_bytes"])
        buffer_bytes = int(summary["buffer_bytes"])
    result["physical_parameter_bytes"] = physical_parameter_bytes
    result["buffer_bytes"] = buffer_bytes
    result["persistent_storage_bytes"] = physical_parameter_bytes + buffer_bytes
    # Keep the historical field useful for model-load comparisons: for a
    # quantized model it means all persistent model tensor storage.
    result["parameter_bytes"] = result["persistent_storage_bytes"]
    result["quantized_weight_count"] = int(plan["quantized_weight_count"])
    result["quantized_storage_bytes"] = int(plan["quantized_storage_bytes"])
    return result


def _make_batch(torch: Any, *, batch_size: int, sequence_length: int, vocab_size: int, seed: int) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    input_ids = torch.randint(
        low=min(100, max(0, vocab_size - 1)),
        high=vocab_size,
        size=(batch_size, sequence_length),
        dtype=torch.long,
        generator=generator,
    )
    labels = input_ids.clone()
    masked_prefix_tokens = sequence_length // 2
    labels[:, :masked_prefix_tokens] = -100
    position_ids = (
        torch.arange(sequence_length, dtype=torch.long)
        .view(1, 1, sequence_length)
        .expand(3, batch_size, sequence_length)
        .clone()
    )
    digest = hashlib.sha256()
    digest.update(input_ids.numpy().tobytes())
    digest.update(labels.numpy().tobytes())
    return {
        "input_ids": input_ids,
        "labels": labels,
        "position_ids": position_ids,
        "masked_prefix_tokens": masked_prefix_tokens,
        "fingerprint_sha256": digest.hexdigest(),
    }


def _make_batches(
    torch: Any,
    *,
    batch_size: int,
    sequence_length: int,
    vocab_size: int,
    seed: int,
    count: int,
) -> dict[str, Any]:
    batches = [
        _make_batch(
            torch,
            batch_size=batch_size,
            sequence_length=sequence_length,
            vocab_size=vocab_size,
            seed=seed + index,
        )
        for index in range(count)
    ]
    digest = hashlib.sha256()
    for batch in batches:
        digest.update(batch["fingerprint_sha256"].encode("ascii"))
    return {
        "items": batches,
        "masked_prefix_tokens": batches[0]["masked_prefix_tokens"],
        "fingerprint_sha256": digest.hexdigest(),
        "microbatch_fingerprints_sha256": [
            batch["fingerprint_sha256"] for batch in batches
        ],
    }


def _load_text_config(AutoConfig: Any, model_dir: Path) -> Any:
    root_config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
    return getattr(root_config, "text_config", root_config)


def _construct_meta_model(
    torch: Any,
    model_class: Any,
    config: Any,
    *,
    dtype: Any,
) -> Any:
    """Construct parameters in the target dtype without casting FP32 buffers."""

    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with torch.device("meta"):
            return model_class(config)
    finally:
        torch.set_default_dtype(previous_dtype)


def _configure_training_model(model: Any, args: argparse.Namespace, *, static: bool) -> tuple[Any, dict[str, Any]]:
    peft_version: str | None = None
    quantization_plan: dict[str, Any] | None = None
    if args.training_method == "qlora" and not static:
        from verification.native_nf4 import convert_model_to_native_nf4_lora

        model, quantization_plan = convert_model_to_native_nf4_lora(
            model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            block_size=args.quantization_block_size,
            double_quantization=args.quantization_double_quantization,
            scale_block_size=args.quantization_scale_block_size,
            target_modules=args.lora_target_modules,
            chunk_blocks=args.quantization_chunk_blocks,
        )
    elif args.training_method in {"lora", "qlora"}:
        import peft
        from peft import LoraConfig, get_peft_model

        if args.training_method == "qlora":
            from verification.native_nf4 import plan_nf4_lora

            quantization_plan = plan_nf4_lora(
                model,
                rank=args.lora_rank,
                block_size=args.quantization_block_size,
                double_quantization=args.quantization_double_quantization,
                scale_block_size=args.quantization_scale_block_size,
                target_modules=args.lora_target_modules,
            )

        peft_version = str(peft.__version__)
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=args.lora_target_modules,
            ),
        )

    checkpointing_analysis = "disabled"
    if args.gradient_checkpointing:
        if static:
            checkpointing_analysis = "uncheckpointed_upper_bound"
        else:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if args.training_method in {"lora", "qlora"}:
                model.enable_input_require_grads()
            checkpointing_analysis = "executed_non_reentrant"

    return model, {
        "peft_version": peft_version,
        "checkpointing_analysis": checkpointing_analysis,
        "quantization_plan": quantization_plan,
    }


def _checkpoint_graph_estimate(estimate: dict[str, Any]) -> dict[str, Any]:
    uncheckpointed_peak = int(estimate["first_step_graph_phase_peak_bytes"])
    parameter_bytes = int(estimate["parameter_bytes"])
    trainable_bytes = int(estimate["trainable_parameter_bytes"])
    if trainable_bytes * 2 >= parameter_bytes:
        return {
            "method": "analytical_full_parameter_graph_floor",
            "estimated_peak_bytes": uncheckpointed_peak,
            "uncheckpointed_upper_bound_bytes": uncheckpointed_peak,
            "loss_temporary_bytes": None,
            "notes": [
                "The full-parameter peak is dominated by parameter-gradient and tied-weight temporaries that checkpointing does not remove."
            ],
        }

    loss_fragments = ("log_softmax", "nll_loss", "cross_entropy")
    loss_temporary_bytes = sum(
        int(storage.get("bytes", 0))
        for storage in estimate["graph"].get("top_peak_storages", [])
        if storage.get("category") == "temporary"
        and any(
            fragment in str(storage.get("producer_target", "")).lower()
            for fragment in loss_fragments
        )
    )
    if loss_temporary_bytes <= 0:
        return {
            "method": "uncheckpointed_upper_bound",
            "estimated_peak_bytes": uncheckpointed_peak,
            "uncheckpointed_upper_bound_bytes": uncheckpointed_peak,
            "loss_temporary_bytes": None,
            "notes": [
                "The graph peak did not expose recognized loss temporaries, so the uncheckpointed graph remains the conservative estimate."
            ],
        }

    peak_categories = estimate["graph"]["peak_bytes_by_category"]
    retained_loss_floor = (
        parameter_bytes
        + int(estimate["buffer_bytes"])
        + int(estimate["input_bytes"])
        + int(peak_categories.get("output", 0))
        + loss_temporary_bytes
        + int(estimate["workspace_peak_contribution_bytes"])
    )
    post_graph_floor = int(estimate["post_graph_live_bytes"]) + int(
        estimate["workspace_peak_contribution_bytes"]
    )
    return {
        "method": "analytical_loss_and_optimizer_floor",
        "estimated_peak_bytes": max(retained_loss_floor, post_graph_floor),
        "uncheckpointed_upper_bound_bytes": uncheckpointed_peak,
        "loss_temporary_bytes": loss_temporary_bytes,
        "retained_loss_floor_bytes": retained_loss_floor,
        "post_graph_floor_bytes": post_graph_floor,
        "notes": [
            "Checkpointed decoder activations are removed from the graph peak while parameters, outputs, loss temporaries, gradients, and profiled workspaces remain."
        ],
    }


def _apply_nf4_static_adjustment(
    estimate: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Replace persistent BF16 linear weights with packed NF4 storage.

    The ATen graph is captured from the mathematically equivalent BF16 LoRA
    surrogate.  Frozen weights remain live throughout that graph, so their
    storage can be substituted exactly.  The reference backend's largest
    per-layer dequantization workspace is then added analytically.
    """

    adjusted = copy.deepcopy(estimate)
    original_weight_bytes = int(plan["original_weight_bytes"])
    quantized_storage_bytes = int(plan["quantized_storage_bytes"])
    materialized_buffer_delta = int(plan.get("base_buffer_materialization_delta_bytes", 0))
    storage_delta = (
        quantized_storage_bytes - original_weight_bytes + materialized_buffer_delta
    )
    dequant_workspace = int(plan["largest_dequantization_workspace_bytes"])

    adjusted["unquantized_parameter_bytes"] = int(estimate["parameter_bytes"])
    adjusted["parameter_bytes"] = int(estimate["parameter_bytes"]) - original_weight_bytes
    adjusted["frozen_parameter_bytes"] = int(estimate["frozen_parameter_bytes"]) - original_weight_bytes
    adjusted["buffer_bytes"] = (
        int(estimate["buffer_bytes"])
        + quantized_storage_bytes
        + materialized_buffer_delta
    )
    adjusted["persistent_model_storage_bytes"] = (
        int(adjusted["parameter_bytes"]) + int(adjusted["buffer_bytes"])
    )
    adjusted["post_graph_live_bytes"] = int(estimate["post_graph_live_bytes"]) + storage_delta
    adjusted["first_step_graph_phase_peak_bytes"] = (
        int(estimate["first_step_graph_phase_peak_bytes"])
        + storage_delta
        + dequant_workspace
    )
    adjusted["graph_phase_peak_bytes"] = (
        int(estimate["graph_phase_peak_bytes"])
        + storage_delta
        + dequant_workspace
    )
    optimizer_phase = estimate.get("optimizer_phase_peak_bytes")
    adjusted["optimizer_phase_peak_bytes"] = (
        int(optimizer_phase) + storage_delta if optimizer_phase is not None else None
    )
    adjusted["first_step_estimated_peak_bytes"] = max(
        int(adjusted["first_step_graph_phase_peak_bytes"]),
        int(adjusted["optimizer_phase_peak_bytes"] or 0),
    )
    adjusted["steady_state_estimated_peak_bytes"] = max(
        int(adjusted["graph_phase_peak_bytes"]),
        int(adjusted["optimizer_phase_peak_bytes"] or 0),
    )
    adjusted["estimated_peak_bytes"] = int(adjusted["steady_state_estimated_peak_bytes"])
    if int(adjusted["graph_phase_peak_bytes"]) > int(adjusted["optimizer_phase_peak_bytes"] or 0):
        adjusted["peak_phase"] = "graph"
    elif int(adjusted["graph_phase_peak_bytes"]) < int(adjusted["optimizer_phase_peak_bytes"] or 0):
        adjusted["peak_phase"] = "optimizer"
    else:
        adjusted["peak_phase"] = "graph_and_optimizer"

    graph = adjusted["graph"]
    graph["peak_live_bytes"] = int(graph["peak_live_bytes"]) + storage_delta
    graph["final_live_bytes"] = int(graph["final_live_bytes"]) + storage_delta
    graph["total_unique_storage_bytes"] = int(graph["total_unique_storage_bytes"]) + storage_delta
    for categories_name in ("peak_bytes_by_category", "final_bytes_by_category"):
        categories = graph[categories_name]
        categories["frozen_parameter"] = int(categories.get("frozen_parameter", 0)) - original_weight_bytes
        categories["buffer"] = (
            int(categories.get("buffer", 0))
            + quantized_storage_bytes
            + materialized_buffer_delta
        )

    workspace = adjusted["workspace_estimate"]
    workspace["total_bytes"] = int(workspace["total_bytes"]) + dequant_workspace
    workspace["effective_peak_contribution_bytes"] = int(
        workspace["effective_peak_contribution_bytes"]
    ) + dequant_workspace
    workspace["profiles"].append(
        {
            "kind": "native_nf4_dequantization_workspace",
            "bytes": dequant_workspace,
            "confidence": "analytical_reference_implementation",
            "module_count": int(plan["module_count"]),
            "description": "largest per-layer int32 lookup and BF16 dequantization workspace",
        }
    )
    adjusted["workspace_estimate_bytes"] = int(estimate["workspace_estimate_bytes"]) + dequant_workspace
    adjusted["workspace_peak_contribution_bytes"] = int(
        estimate["workspace_peak_contribution_bytes"]
    ) + dequant_workspace
    adjusted["tracking_confidence"] = "S2_aot_training_liveness_plus_analytical_nf4"
    adjusted["quantization"] = {
        key: value for key, value in plan.items() if key != "modules"
    }
    adjusted["unmodeled_components"] = list(adjusted["unmodeled_components"]) + [
        "fused_external_4bit_backend_kernel_behavior"
    ]
    adjusted["notes"] = list(adjusted["notes"]) + [
        "Frozen BF16 linear storage is replaced exactly by the native packed NF4 buffers.",
        "The largest PyTorch reference dequantization workspace is combined analytically with the captured graph.",
    ]
    return adjusted


def _common_report(args: argparse.Namespace, torch: Any, transformers: Any, model_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "mode": args.mode,
        "model_dir": str(model_dir),
        "profile": args.profile if args.mode == "fakecuda" else None,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "transformers_version": str(transformers.__version__),
        "dtype": args.dtype,
        "attention_implementation": args.attention_implementation,
        "optimizer": "adamw_single_tensor",
        "training_method": args.training_method,
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": args.lora_target_modules,
        }
        if args.training_method in {"lora", "qlora"}
        else None,
        "quantization": {
            "backend": "pytorch_native_nf4",
            "format": "nf4_blockwise",
            "block_size": args.quantization_block_size,
            "double_quantization": bool(args.quantization_double_quantization),
            "scale_block_size": args.quantization_scale_block_size,
        }
        if args.training_method == "qlora"
        else None,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "seed": args.seed,
        "data_seed": args.data_seed,
    }


def _run_static(args: argparse.Namespace, torch: Any, transformers: Any, model_dir: Path) -> dict[str, Any]:
    from transformers import AutoConfig, Qwen3_5ForCausalLM

    from fakegpu.memory_estimator import estimate_module_memory

    dtype = _dtype(torch, args.dtype)
    config = _load_text_config(AutoConfig, model_dir)
    config.use_cache = False
    config._attn_implementation = args.attention_implementation
    model = _construct_meta_model(
        torch,
        Qwen3_5ForCausalLM,
        config,
        dtype=dtype,
    )
    model, training_metadata = _configure_training_model(model, args, static=True)
    model.train()
    parameters = _parameter_summary(model)
    quantization_plan = training_metadata["quantization_plan"]
    if quantization_plan is not None:
        quantization_plan["base_buffer_materialization_delta_bytes"] = (
            int(parameters["logical_buffer_bytes"]) - int(parameters["buffer_bytes"])
        )
        parameters = _quantized_parameter_summary(
            parameters,
            quantization_plan,
            static_surrogate=True,
        )
    batches = _make_batches(
        torch,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        vocab_size=int(config.vocab_size),
        seed=args.data_seed,
        count=args.gradient_accumulation_steps,
    )
    batch = batches["items"][0]
    started = time.monotonic()
    estimate = estimate_module_memory(
        model,
        (batch["input_ids"],),
        example_kwargs={
            "labels": batch["labels"],
            "position_ids": batch["position_ids"],
            "use_cache": False,
        },
        mode="training",
        loss_fn=lambda output: output.loss,
        optimizer="adamw",
        retain_forward_outputs=True,
        target_device="cuda",
    )
    if quantization_plan is not None:
        estimate = _apply_nf4_static_adjustment(estimate, quantization_plan)
    checkpoint_estimate: dict[str, Any] | None = None
    base_graph_estimate = int(estimate["first_step_graph_phase_peak_bytes"])
    if args.gradient_checkpointing:
        checkpoint_estimate = _checkpoint_graph_estimate(estimate)
        base_graph_estimate = int(checkpoint_estimate["estimated_peak_bytes"])
        training_metadata["checkpointing_analysis"] = checkpoint_estimate["method"]
    accumulation_graph_estimate = base_graph_estimate
    accumulation_analysis = "single_microbatch_exact"
    if args.gradient_accumulation_steps > 1:
        accumulation_graph_estimate += int(
            estimate["optimizer_temporary"]["largest_parameter_tensor_bytes"]
        )
        accumulation_analysis = "in_place_largest_gradient_temporary"
    optimizer_peak = int(estimate["optimizer_phase_peak_bytes"])
    first_step_peak = max(accumulation_graph_estimate, optimizer_peak)
    report = _common_report(args, torch, transformers, model_dir)
    report.update(
        {
            "gpu_name": "static FakeTensor CUDA target",
            "peft_version": training_metadata["peft_version"],
            "quantization_details": (
                {key: value for key, value in quantization_plan.items() if key != "modules"}
                if quantization_plan is not None
                else None
            ),
            "parameters": parameters,
            "batch": {
                "masked_prefix_tokens": batches["masked_prefix_tokens"],
                "fingerprint_sha256": batches["fingerprint_sha256"],
                "microbatch_fingerprints_sha256": batches[
                    "microbatch_fingerprints_sha256"
                ],
            },
            "static_analysis": {
                "checkpointing": training_metadata["checkpointing_analysis"],
                "gradient_accumulation": accumulation_analysis,
                "checkpoint_estimate": checkpoint_estimate,
                "quantization": (
                    "analytical_pytorch_native_nf4_workspace"
                    if quantization_plan is not None
                    else "disabled"
                ),
            },
            "elapsed_seconds": time.monotonic() - started,
            "memory_phases": {
                "overall_peak_bytes": first_step_peak,
                "first_step_estimated_peak_bytes": first_step_peak,
                "steady_state_estimated_peak_bytes": int(estimate["steady_state_estimated_peak_bytes"]),
                "first_step_graph_phase_peak_bytes": int(estimate["first_step_graph_phase_peak_bytes"]),
                "checkpoint_graph_estimated_peak_bytes": (
                    int(checkpoint_estimate["estimated_peak_bytes"])
                    if checkpoint_estimate is not None
                    else None
                ),
                "accumulation_graph_estimated_peak_bytes": accumulation_graph_estimate,
                "graph_phase_peak_bytes": int(estimate["graph_phase_peak_bytes"]),
                "optimizer_phase_peak_bytes": int(estimate["optimizer_phase_peak_bytes"]),
                "parameter_bytes": int(
                    estimate.get(
                        "persistent_model_storage_bytes",
                        int(estimate["parameter_bytes"]) + int(estimate["buffer_bytes"]),
                    )
                ),
                "optimizer_state_bytes": int(estimate["optimizer_state_bytes"]),
                "optimizer_temporary_bytes": int(estimate["optimizer_temporary_bytes"]),
                "workspace_estimate_bytes": int(estimate["workspace_estimate_bytes"]),
                "peak_phase": str(estimate["peak_phase"]),
            },
            "static_estimate": estimate,
        }
    )
    return report


def _run_execution(args: argparse.Namespace, torch: Any, transformers: Any, model_dir: Path) -> dict[str, Any]:
    from transformers import AutoConfig, Qwen3_5ForCausalLM

    if args.mode == "real" and not torch.cuda.is_available():
        raise RuntimeError("real mode requires a CUDA device")
    dtype = _dtype(torch, args.dtype)
    text_config = _load_text_config(AutoConfig, model_dir)
    torch.manual_seed(args.seed)
    if args.mode == "real":
        torch.cuda.manual_seed_all(args.seed)

    sampler = _NvmlProcessSampler(enabled=args.mode == "real")
    timeline: list[dict[str, Any]] = []
    phase_seconds: dict[str, float] = {}
    started = time.monotonic()
    try:
        if args.mode == "real":
            torch.cuda.empty_cache()
        _reset_peak(torch, args.mode)
        timeline.append(_memory_record(torch, sampler, args.mode, "baseline"))

        load_started = time.monotonic()
        model = Qwen3_5ForCausalLM.from_pretrained(
            str(model_dir),
            dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
            attn_implementation=args.attention_implementation,
        )
        model.config.use_cache = False
        model, training_metadata = _configure_training_model(model, args, static=False)
        model.train()
        model.to("cuda:0")
        phase_seconds["model_load"] = time.monotonic() - load_started
        parameters = _parameter_summary(model)
        quantization_plan = training_metadata["quantization_plan"]
        if quantization_plan is not None:
            parameters = _quantized_parameter_summary(
                parameters,
                quantization_plan,
                static_surrogate=False,
            )
        model_load_record = _memory_record(torch, sampler, args.mode, "after_model_load")
        timeline.append(model_load_record)

        batches = _make_batches(
            torch,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            vocab_size=int(text_config.vocab_size),
            seed=args.data_seed,
            count=args.gradient_accumulation_steps,
        )
        device_batches = [
            {
                "input_ids": batch["input_ids"].to("cuda:0"),
                "labels": batch["labels"].to("cuda:0"),
                "position_ids": batch["position_ids"].to("cuda:0"),
            }
            for batch in batches["items"]
        ]
        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=args.learning_rate,
            foreach=False,
        )
        timeline.append(_memory_record(torch, sampler, args.mode, "after_inputs"))

        forward_records: list[dict[str, Any]] = []
        backward_records: list[dict[str, Any]] = []
        microstep_seconds: list[dict[str, float]] = []
        losses: list[float] = []
        for microstep, device_batch in enumerate(device_batches, start=1):
            _reset_peak(torch, args.mode)
            forward_started = time.monotonic()
            outputs = model(**device_batch, use_cache=False)
            loss = outputs.loss / args.gradient_accumulation_steps
            forward_seconds = time.monotonic() - forward_started
            forward_record = _memory_record(
                torch,
                sampler,
                args.mode,
                f"microstep_{microstep}_after_forward",
            )
            timeline.append(forward_record)
            forward_records.append(forward_record)

            _reset_peak(torch, args.mode)
            backward_started = time.monotonic()
            loss.backward()
            backward_seconds = time.monotonic() - backward_started
            backward_record = _memory_record(
                torch,
                sampler,
                args.mode,
                f"microstep_{microstep}_after_backward",
            )
            timeline.append(backward_record)
            backward_records.append(backward_record)
            microstep_seconds.append(
                {"forward": forward_seconds, "backward": backward_seconds}
            )
            losses.append(
                float(loss.detach().float().cpu().item())
                * args.gradient_accumulation_steps
            )
            if microstep < args.gradient_accumulation_steps:
                del outputs, loss

        phase_seconds["forward"] = sum(item["forward"] for item in microstep_seconds)
        phase_seconds["backward"] = sum(item["backward"] for item in microstep_seconds)

        _reset_peak(torch, args.mode)
        optimizer_started = time.monotonic()
        optimizer.step()
        phase_seconds["optimizer"] = time.monotonic() - optimizer_started
        optimizer_record = _memory_record(torch, sampler, args.mode, "after_optimizer")
        timeline.append(optimizer_record)

        phase_records = [*forward_records, *backward_records, optimizer_record]
        overall_peak = max(int(item["peak_allocated_bytes"]) for item in phase_records)
        forward_peak = max(int(item["peak_allocated_bytes"]) for item in forward_records)
        backward_peak = max(int(item["peak_allocated_bytes"]) for item in backward_records)
        report = _common_report(args, torch, transformers, model_dir)
        report.update(
            {
                "gpu_name": str(torch.cuda.get_device_name(0)),
                "peft_version": training_metadata["peft_version"],
                "quantization_details": (
                    {key: value for key, value in quantization_plan.items() if key != "modules"}
                    if quantization_plan is not None
                    else None
                ),
                "parameters": parameters,
                "batch": {
                    "masked_prefix_tokens": batches["masked_prefix_tokens"],
                    "fingerprint_sha256": batches["fingerprint_sha256"],
                    "microbatch_fingerprints_sha256": batches[
                        "microbatch_fingerprints_sha256"
                    ],
                },
                "loss": sum(losses) / len(losses),
                "microbatch_losses": losses,
                "microstep_seconds": microstep_seconds,
                "phase_seconds": phase_seconds,
                "elapsed_seconds": time.monotonic() - started,
                "timeline": timeline,
                "memory_phases": {
                    "model_load_current_bytes": int(model_load_record["allocated_bytes"]),
                    "forward_current_bytes": int(forward_records[-1]["allocated_bytes"]),
                    "forward_peak_bytes": forward_peak,
                    "backward_current_bytes": int(backward_records[-1]["allocated_bytes"]),
                    "backward_peak_bytes": backward_peak,
                    "microstep_forward_peak_bytes": [
                        int(item["peak_allocated_bytes"]) for item in forward_records
                    ],
                    "microstep_backward_peak_bytes": [
                        int(item["peak_allocated_bytes"]) for item in backward_records
                    ],
                    "optimizer_current_bytes": int(optimizer_record["allocated_bytes"]),
                    "optimizer_peak_bytes": int(optimizer_record["peak_allocated_bytes"]),
                    "overall_peak_bytes": overall_peak,
                },
            }
        )
        return report
    finally:
        sampler.close()


def _dtype(torch: Any, name: str) -> Any:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode == "fakecuda":
        os.environ.setdefault("FAKEGPU_TERMINAL_REPORT", "0")
        import fakegpu

        fakegpu.init(runtime="fakecuda", profile=args.profile, device_count=1, force=True)

    import torch
    import transformers

    transformers.utils.import_utils._torchvision_available = False
    transformers.utils.import_utils._torchvision_version = "0.0"
    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_dir}")
    if args.mode == "static":
        return _run_static(args, torch, transformers, model_dir)
    return _run_execution(args, torch, transformers, model_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["real", "fakecuda", "static"], required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", default="rtx-pro-5000-blackwell")
    parser.add_argument("--training-method", choices=["full", "lora", "qlora"], default="full")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-target-modules", default="all-linear")
    parser.add_argument("--quantization-block-size", type=int, default=64)
    parser.add_argument(
        "--quantization-double-quantization",
        "--double-quantization",
        action="store_true",
    )
    parser.add_argument("--quantization-scale-block-size", type=int, default=256)
    parser.add_argument("--quantization-chunk-blocks", type=int, default=16_384)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--attention-implementation", choices=["eager", "sdpa"], default="sdpa")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--data-seed", type=int, default=20260721)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    if args.sequence_length < 2:
        parser.error("--sequence-length must be at least two")
    if args.gradient_accumulation_steps <= 0:
        parser.error("--gradient-accumulation-steps must be greater than zero")
    if args.lora_rank <= 0:
        parser.error("--lora-rank must be greater than zero")
    if args.lora_alpha <= 0:
        parser.error("--lora-alpha must be greater than zero")
    if not 0.0 <= args.lora_dropout < 1.0:
        parser.error("--lora-dropout must be in [0, 1)")
    if args.quantization_block_size <= 0 or args.quantization_block_size % 2:
        parser.error("--quantization-block-size must be a positive even integer")
    if args.quantization_scale_block_size <= 0:
        parser.error("--quantization-scale-block-size must be greater than zero")
    if args.quantization_chunk_blocks <= 0:
        parser.error("--quantization-chunk-blocks must be greater than zero")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be greater than zero")

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = run(args)
    except Exception as exc:
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "mode": args.mode,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), file=sys.stderr)
        return 1
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "mode": report["mode"],
                "model_dir": report["model_dir"],
                "parameter_bytes": report["parameters"]["parameter_bytes"],
                "overall_peak_bytes": report["memory_phases"]["overall_peak_bytes"],
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
