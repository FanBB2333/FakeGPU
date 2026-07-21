#!/usr/bin/env python3
"""Measure or statically estimate one text-only Qwen3.5 SFT step."""

from __future__ import annotations

import argparse
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

SCHEMA_VERSION = "fakegpu.qwen_sft_memory_worker.v1"


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


def _parameter_summary(model: Any) -> dict[str, Any]:
    parameters = list(model.parameters())
    trainable = [parameter for parameter in parameters if parameter.requires_grad]

    def nbytes(items: list[Any]) -> int:
        return sum(int(item.numel()) * int(item.element_size()) for item in items)

    return {
        "parameter_tensors": len(parameters),
        "parameter_count": sum(int(parameter.numel()) for parameter in parameters),
        "parameter_bytes": nbytes(parameters),
        "trainable_parameter_tensors": len(trainable),
        "trainable_parameter_count": sum(int(parameter.numel()) for parameter in trainable),
        "trainable_parameter_bytes": nbytes(trainable),
        "dtypes": sorted({str(parameter.dtype) for parameter in parameters}),
        "devices": sorted({str(parameter.device) for parameter in parameters}),
    }


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


def _load_text_config(AutoConfig: Any, model_dir: Path) -> Any:
    root_config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
    return getattr(root_config, "text_config", root_config)


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
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)
    model = model.to(dtype=dtype)
    model.train()
    parameters = _parameter_summary(model)
    batch = _make_batch(
        torch,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        vocab_size=int(config.vocab_size),
        seed=args.data_seed,
    )
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
    report = _common_report(args, torch, transformers, model_dir)
    report.update(
        {
            "gpu_name": "static FakeTensor CUDA target",
            "parameters": parameters,
            "batch": {
                "masked_prefix_tokens": batch["masked_prefix_tokens"],
                "fingerprint_sha256": batch["fingerprint_sha256"],
            },
            "elapsed_seconds": time.monotonic() - started,
            "memory_phases": {
                "overall_peak_bytes": int(estimate["estimated_peak_bytes"]),
                "graph_phase_peak_bytes": int(estimate["graph_phase_peak_bytes"]),
                "optimizer_phase_peak_bytes": int(estimate["optimizer_phase_peak_bytes"]),
                "parameter_bytes": int(estimate["parameter_bytes"]),
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
        model.train()
        model.to("cuda:0")
        phase_seconds["model_load"] = time.monotonic() - load_started
        parameters = _parameter_summary(model)
        model_load_record = _memory_record(torch, sampler, args.mode, "after_model_load")
        timeline.append(model_load_record)

        batch = _make_batch(
            torch,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            vocab_size=int(text_config.vocab_size),
            seed=args.data_seed,
        )
        input_ids = batch["input_ids"].to("cuda:0")
        labels = batch["labels"].to("cuda:0")
        position_ids = batch["position_ids"].to("cuda:0")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, foreach=False)
        timeline.append(_memory_record(torch, sampler, args.mode, "after_inputs"))

        _reset_peak(torch, args.mode)
        forward_started = time.monotonic()
        outputs = model(
            input_ids=input_ids,
            labels=labels,
            position_ids=position_ids,
            use_cache=False,
        )
        loss = outputs.loss
        phase_seconds["forward"] = time.monotonic() - forward_started
        forward_record = _memory_record(torch, sampler, args.mode, "after_forward")
        timeline.append(forward_record)

        _reset_peak(torch, args.mode)
        backward_started = time.monotonic()
        loss.backward()
        phase_seconds["backward"] = time.monotonic() - backward_started
        backward_record = _memory_record(torch, sampler, args.mode, "after_backward")
        timeline.append(backward_record)

        _reset_peak(torch, args.mode)
        optimizer_started = time.monotonic()
        optimizer.step()
        phase_seconds["optimizer"] = time.monotonic() - optimizer_started
        optimizer_record = _memory_record(torch, sampler, args.mode, "after_optimizer")
        timeline.append(optimizer_record)

        phase_records = [forward_record, backward_record, optimizer_record]
        overall_peak = max(int(item["peak_allocated_bytes"]) for item in phase_records)
        report = _common_report(args, torch, transformers, model_dir)
        report.update(
            {
                "gpu_name": str(torch.cuda.get_device_name(0)),
                "parameters": parameters,
                "batch": {
                    "masked_prefix_tokens": batch["masked_prefix_tokens"],
                    "fingerprint_sha256": batch["fingerprint_sha256"],
                },
                "loss": float(loss.detach().float().cpu().item()),
                "phase_seconds": phase_seconds,
                "elapsed_seconds": time.monotonic() - started,
                "timeline": timeline,
                "memory_phases": {
                    "model_load_current_bytes": int(model_load_record["allocated_bytes"]),
                    "forward_current_bytes": int(forward_record["allocated_bytes"]),
                    "forward_peak_bytes": int(forward_record["peak_allocated_bytes"]),
                    "backward_current_bytes": int(backward_record["allocated_bytes"]),
                    "backward_peak_bytes": int(backward_record["peak_allocated_bytes"]),
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
