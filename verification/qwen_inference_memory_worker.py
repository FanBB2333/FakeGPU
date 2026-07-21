#!/usr/bin/env python3

from __future__ import annotations

import argparse
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


class _NvmlProcessSampler:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.reason: str | None = None
        self._pynvml: Any = None
        self._handle: Any = None
        if not enabled:
            self.reason = "disabled for fakecuda mode"
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
            return {
                "status": "unavailable",
                "reason": self.reason or "NVML unavailable",
            }
        pynvml = self._pynvml
        memory = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        process_memory: int | None = None
        process_query = None
        for name in (
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetComputeRunningProcesses_v2",
            "nvmlDeviceGetComputeRunningProcesses",
        ):
            candidate = getattr(pynvml, name, None)
            if candidate is not None:
                process_query = candidate
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

        snapshot = memory_snapshot()
        devices = snapshot.get("devices") or []
        record["fakecuda_snapshot"] = devices[0] if devices else None
    else:
        record["nvml"] = sampler.sample()
    return record


def _parameter_summary(model: Any) -> dict[str, Any]:
    tensors = list(model.parameters())
    return {
        "parameter_tensors": len(tensors),
        "parameter_count": sum(int(tensor.numel()) for tensor in tensors),
        "parameter_bytes": sum(
            int(tensor.numel()) * int(tensor.element_size()) for tensor in tensors
        ),
        "dtypes": sorted({str(tensor.dtype) for tensor in tensors}),
        "devices": sorted({str(tensor.device) for tensor in tensors}),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode == "fakecuda":
        os.environ.setdefault("FAKEGPU_TERMINAL_REPORT", "0")
        if args.smi_state:
            os.environ["FAKEGPU_SMI_STATE_PATH"] = str(
                Path(args.smi_state).expanduser().resolve()
            )
            os.environ["FAKEGPU_SMI_RUNTIME_OVERHEAD_BYTES"] = str(
                args.smi_runtime_overhead_bytes
            )
        import fakegpu

        fakegpu.init(
            runtime="fakecuda",
            profile=args.profile,
            device_count=1,
            force=True,
        )

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from fakegpu.flop_counter import MatmulFlopCounterMode

    transformers.utils.import_utils._torchvision_available = False
    transformers.utils.import_utils._torchvision_version = "0.0"

    if args.mode == "real" and not torch.cuda.is_available():
        raise RuntimeError("real mode requires a CUDA device")
    torch.manual_seed(args.seed)
    if args.mode == "real":
        torch.cuda.manual_seed_all(args.seed)

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_dir}")

    sampler = _NvmlProcessSampler(enabled=args.mode == "real")
    timeline: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        if args.mode == "real":
            torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        timeline.append(_memory_record(torch, sampler, args.mode, "baseline"))

        load_started = time.monotonic()
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            dtype=dtype,
            device_map="cuda:0",
            low_cpu_mem_usage=True,
            local_files_only=True,
            attn_implementation=args.attention_implementation,
        )
        model.eval()
        load_seconds = time.monotonic() - load_started
        model_load_record = _memory_record(torch, sampler, args.mode, "after_model_load")
        timeline.append(model_load_record)
        parameter_summary = _parameter_summary(model)
        torch.cuda.reset_peak_memory_stats()

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            local_files_only=True,
        )
        messages = [{"role": "user", "content": args.prompt}]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([rendered], return_tensors="pt").to("cuda:0")
        prompt_tokens = int(model_inputs["input_ids"].shape[1])
        timeline.append(_memory_record(torch, sampler, args.mode, "after_inputs"))

        generated_ids: list[int] = []
        measured_flops: list[int] = []
        inference_started = time.monotonic()
        with torch.inference_mode():
            with MatmulFlopCounterMode() as counter:
                outputs = model(**model_inputs, use_cache=True)
            measured_flops.append(int(counter.total_flops))
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated_ids.append(int(next_token.raw_data.item() if hasattr(next_token, "raw_data") else next_token.item()))
            attention_mask = model_inputs.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(model_inputs["input_ids"])

            for _step in range(max(0, args.generated_tokens - 1)):
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            (attention_mask.shape[0], 1),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ],
                    dim=1,
                )
                with MatmulFlopCounterMode() as counter:
                    outputs = model(
                        input_ids=next_token,
                        attention_mask=attention_mask,
                        past_key_values=outputs.past_key_values,
                        use_cache=True,
                    )
                measured_flops.append(int(counter.total_flops))
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                generated_ids.append(int(next_token.raw_data.item() if hasattr(next_token, "raw_data") else next_token.item()))
        inference_seconds = time.monotonic() - inference_started
        inference_record = _memory_record(torch, sampler, args.mode, "after_inference")
        timeline.append(inference_record)

        from fakegpu.llm_estimator import estimate_decoder_inference

        estimate = estimate_decoder_inference(
            model_dir,
            batch_size=1,
            prompt_tokens=prompt_tokens,
            generated_tokens=args.generated_tokens,
            dtype=args.dtype,
            use_cache=True,
            attention_implementation=args.attention_implementation,
            runtime_overhead_bytes=args.smi_runtime_overhead_bytes,
        )
        expected_steps = [
            int(estimate["compute"]["prefill_flops"]),
            *[
                int(item["matmul_flops"])
                for item in estimate["compute"]["decode_steps"]
            ],
        ]
        comparisons = []
        for index, (measured, expected) in enumerate(zip(measured_flops, expected_steps, strict=True)):
            comparisons.append(
                {
                    "step": index,
                    "measured_flops": measured,
                    "estimated_flops": expected,
                    "signed_error_flops": measured - expected,
                    "absolute_error_percent": (
                        round(100.0 * abs(measured - expected) / expected, 6)
                        if expected
                        else None
                    ),
                }
            )

        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
        report = {
            "schema_version": "fakegpu.qwen_inference_memory_worker.v1",
            "status": "success",
            "mode": args.mode,
            "pid": os.getpid(),
            "model_dir": str(model_dir),
            "profile": args.profile if args.mode == "fakecuda" else None,
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
            "transformers_version": str(transformers.__version__),
            "gpu_name": str(torch.cuda.get_device_name(0)),
            "dtype": args.dtype,
            "attention_implementation": args.attention_implementation,
            "prompt": args.prompt,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_ids,
            "generated_text": decoded,
            "parameters": parameter_summary,
            "load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
            "elapsed_seconds": time.monotonic() - started,
            "timeline": timeline,
            "memory_phases": {
                "model_load_current_bytes": int(model_load_record["allocated_bytes"]),
                "model_load_peak_bytes": int(model_load_record["peak_allocated_bytes"]),
                "inference_current_bytes": int(inference_record["allocated_bytes"]),
                "inference_peak_bytes": int(inference_record["peak_allocated_bytes"]),
            },
            "measured_matmul_flops_by_step": measured_flops,
            "flop_comparisons": comparisons,
            "static_estimate": estimate,
        }
        if args.hold_seconds > 0:
            time.sleep(args.hold_seconds)
        return report
    finally:
        sampler.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["real", "fakecuda"], required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", default="rtx-pro-5000-blackwell")
    parser.add_argument("--prompt", default="Say hello in one short sentence.")
    parser.add_argument("--generated-tokens", type=int, default=2)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--attention-implementation", choices=["eager", "sdpa"], default="eager")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smi-state")
    parser.add_argument("--smi-runtime-overhead-bytes", type=int, default=0)
    parser.add_argument("--hold-seconds", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.generated_tokens <= 0:
        parser.error("--generated-tokens must be greater than zero")
    if args.smi_runtime_overhead_bytes < 0:
        parser.error("--smi-runtime-overhead-bytes must be non-negative")
    if args.hold_seconds < 0:
        parser.error("--hold-seconds must be non-negative")

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = run(args)
    except Exception as exc:
        report = {
            "schema_version": "fakegpu.qwen_inference_memory_worker.v1",
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
    print(json.dumps({
        "status": report["status"],
        "mode": report["mode"],
        "prompt_tokens": report["prompt_tokens"],
        "generated_tokens": report["generated_tokens"],
        "parameter_bytes": report["parameters"]["parameter_bytes"],
        "model_load_current_bytes": report["memory_phases"]["model_load_current_bytes"],
        "inference_peak_bytes": report["memory_phases"]["inference_peak_bytes"],
        "measured_matmul_flops": sum(report["measured_matmul_flops_by_step"]),
        "estimated_matmul_flops": report["static_estimate"]["compute"]["total_flops"],
        "output": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
