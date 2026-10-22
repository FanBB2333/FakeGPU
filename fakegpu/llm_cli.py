from __future__ import annotations

import argparse
from typing import Sequence

from ._cli import (
    add_json_path_argument,
    command_prog,
    emit_json_report,
    usage_error,
)
from .llm_estimator import (
    DEFAULT_KV_QUANTIZATION_GROUP_SIZE,
    MLA_KV_CACHE_LAYOUTS,
    VISION_ESTIMATE_WARNING,
    estimate_decoder_inference,
)
from .performance_model import COMPUTE_PRECISIONS


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=command_prog(__name__),
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
        "--logits-to-keep",
        type=int,
        default=0,
        help="Tokens projected to vocabulary logits per forward: 0 means all (default); 1 models last-token-only generation.",
    )
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
        help=(
            "Quantized-cache residual length: generated tokens stay in full "
            "precision until this many accumulate, then the whole cache is "
            "re-quantized."
        ),
    )
    parser.add_argument(
        "--kv-cache-group-size",
        type=int,
        default=DEFAULT_KV_QUANTIZATION_GROUP_SIZE,
        help=(
            "Quantized-cache q_group_size; each group stores a scale and a "
            "zero point in the compute dtype."
        ),
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
        help=(
            "Override: one attention window for every layer instead of the "
            "model's configured sliding-window layers."
        ),
    )
    parser.add_argument(
        "--mla-kv-cache-layout",
        choices=sorted(MLA_KV_CACHE_LAYOUTS),
        default="auto",
        help=(
            "Cache layout for multi-head latent attention models: "
            "decompressed per-head keys and values (Hugging Face "
            "transformers) or the compressed latent plus rope key (vLLM). "
            "auto picks latent for the paged strategy, decompressed "
            "otherwise."
        ),
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
        help="Explicit factor applied to the selected compute peak.",
    )
    parser.add_argument(
        "--compute-precision",
        choices=COMPUTE_PRECISIONS,
        default="fp32",
        help="Profile peak for the roofline: fp32 (scalar FMA, default) or a tensor path such as tensor_bf16.",
    )
    add_json_path_argument(parser)
    args = parser.parse_args(argv)

    try:
        report = estimate_decoder_inference(
            args.model_dir,
            batch_size=args.batch_size,
            prompt_tokens=args.prompt_tokens,
            generated_tokens=args.generated_tokens,
            logits_to_keep=args.logits_to_keep,
            include_decode_steps=not args.exclude_decode_steps,
            dtype=args.dtype,
            use_cache=not args.no_cache,
            attention_implementation=args.attention_implementation,
            runtime_overhead_bytes=args.runtime_overhead_bytes,
            adapter_dirs=args.adapter_dir,
            expert_parallel_size=args.expert_parallel_size,
            target_profile=args.target_profile,
            compute_acceleration_factor=args.compute_acceleration_factor,
            compute_precision=args.compute_precision,
            kv_cache_strategy=args.kv_cache_strategy,
            kv_cache_bits=args.kv_cache_bits,
            kv_cache_residual_tokens=args.kv_cache_residual_tokens,
            kv_cache_block_tokens=args.kv_cache_block_tokens,
            kv_cache_max_tokens=args.kv_cache_max_tokens,
            kv_cache_window_tokens=args.kv_cache_window_tokens,
            kv_cache_group_size=args.kv_cache_group_size,
            mla_kv_cache_layout=args.mla_kv_cache_layout,
        )
    except (OSError, ValueError) as exc:
        usage_error(parser, exc)

    if args.json_path:
        emit_json_report(args.json_path, report, "LLM estimate")

    memory = report["memory"]
    compute = report["compute"]
    checkpoint = report["checkpoint"]
    print("FakeGPU LLM inference estimate")
    if report["model"].get("has_vision_encoder"):
        print(f"  warning: {VISION_ESTIMATE_WARNING}")
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
