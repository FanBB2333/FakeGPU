from __future__ import annotations

import argparse
from typing import Sequence

from .llm_estimator import estimate_decoder_inference
from .structured_io import emit_json


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fakegpu estimate-llm",
        description=(
            "Estimate decoder inference memory, matrix FLOPs, communication, "
            "and optional profile-aware latency without loading weights."
        ),
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--generated-tokens", type=int, default=1)
    parser.add_argument(
        "--exclude-decode-steps",
        action="store_true",
        help="Aggregate decode FLOPs without embedding one record per token.",
    )
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--kv-cache-strategy",
        choices=["dynamic", "static", "quantized", "paged"],
        default="dynamic",
    )
    parser.add_argument(
        "--kv-cache-bits",
        type=int,
        choices=[2, 4, 8],
        default=4,
        help="Storage bit width for the quantized KV-cache strategy.",
    )
    parser.add_argument(
        "--kv-cache-residual-tokens",
        type=int,
        default=128,
        help="Recent full-precision tokens retained by a quantized cache.",
    )
    parser.add_argument(
        "--kv-cache-block-tokens",
        type=int,
        default=16,
        help="Allocation block size for the paged KV-cache strategy.",
    )
    parser.add_argument(
        "--kv-cache-max-tokens",
        type=int,
        help="Per-sequence reservation for the static KV-cache strategy.",
    )
    parser.add_argument(
        "--kv-cache-window-tokens",
        type=int,
        help="Optional sliding-window context limit.",
    )
    parser.add_argument("--attention-implementation", choices=["eager", "sdpa"], default="eager")
    parser.add_argument("--runtime-overhead-bytes", type=int, default=0)
    parser.add_argument(
        "--adapter-dir",
        action="append",
        default=[],
        help="PEFT/LoRA adapter directory; may be repeated.",
    )
    parser.add_argument("--expert-parallel-size", type=int, default=1)
    parser.add_argument(
        "--target-profile",
        help="GPU profile used for an analytical roofline interval.",
    )
    parser.add_argument(
        "--compute-acceleration-factor",
        type=float,
        default=1.0,
        help="Explicit matrix/tensor throughput factor over scalar FP32.",
    )
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args(argv)

    try:
        report = estimate_decoder_inference(
            args.model_dir,
            batch_size=args.batch_size,
            prompt_tokens=args.prompt_tokens,
            generated_tokens=args.generated_tokens,
            include_decode_steps=not args.exclude_decode_steps,
            dtype=args.dtype,
            use_cache=not args.no_cache,
            attention_implementation=args.attention_implementation,
            runtime_overhead_bytes=args.runtime_overhead_bytes,
            adapter_dirs=args.adapter_dir,
            expert_parallel_size=args.expert_parallel_size,
            target_profile=args.target_profile,
            compute_acceleration_factor=args.compute_acceleration_factor,
            kv_cache_strategy=args.kv_cache_strategy,
            kv_cache_bits=args.kv_cache_bits,
            kv_cache_residual_tokens=args.kv_cache_residual_tokens,
            kv_cache_block_tokens=args.kv_cache_block_tokens,
            kv_cache_max_tokens=args.kv_cache_max_tokens,
            kv_cache_window_tokens=args.kv_cache_window_tokens,
        )
    except (OSError, ValueError) as exc:
        parser.exit(2, f"fakegpu estimate-llm: {exc}\n")

    if args.json_path:
        output = emit_json(args.json_path, report)
        if output is not None:
            print(f"LLM estimate: {output}")

    memory = report["memory"]
    compute = report["compute"]
    checkpoint = report["checkpoint"]
    print("FakeGPU LLM inference estimate")
    print(f"  parameters: {checkpoint['parameter_count']:,}")
    print(f"  checkpoint: {_format_bytes(checkpoint['checkpoint_bytes'])}")
    print(f"  tensor peak: {_format_bytes(memory['estimated_tensor_peak_bytes'])}")
    print(f"  process peak: {_format_bytes(memory['estimated_process_peak_bytes'])}")
    print(
        "  KV cache: "
        f"{report['kv_cache']['strategy']} "
        f"({_format_bytes(memory['kv_cache_bytes_after_generation'])})"
    )
    print(f"  matrix FLOPs: {compute['total_flops']:,}")
    if report["communication"]["enabled"]:
        print(
            "  expert-parallel traffic: "
            f"{_format_bytes(report['communication']['total_bytes'])}"
        )
    if report["performance"] is not None:
        interval = report["performance"]["latency_interval_seconds"]
        print(
            "  analytical latency: "
            f"{interval['lower'] * 1_000:.3f} / "
            f"{interval['expected'] * 1_000:.3f} / "
            f"{interval['upper'] * 1_000:.3f} ms"
        )
    return 0


def _format_bytes(value: int) -> str:
    return f"{int(value) / 2**30:.3f} GiB"
