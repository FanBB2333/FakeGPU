from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .llm_estimator import estimate_decoder_inference


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
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--no-cache", action="store_true")
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
            dtype=args.dtype,
            use_cache=not args.no_cache,
            attention_implementation=args.attention_implementation,
            runtime_overhead_bytes=args.runtime_overhead_bytes,
            adapter_dirs=args.adapter_dir,
            expert_parallel_size=args.expert_parallel_size,
            target_profile=args.target_profile,
            compute_acceleration_factor=args.compute_acceleration_factor,
        )
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"fakegpu estimate-llm: {exc}\n")

    if args.json_path:
        path = Path(args.json_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"LLM estimate: {path}")

    memory = report["memory"]
    compute = report["compute"]
    checkpoint = report["checkpoint"]
    print("FakeGPU LLM inference estimate")
    print(f"  parameters: {checkpoint['parameter_count']:,}")
    print(f"  checkpoint: {_format_bytes(checkpoint['checkpoint_bytes'])}")
    print(f"  tensor peak: {_format_bytes(memory['estimated_tensor_peak_bytes'])}")
    print(f"  process peak: {_format_bytes(memory['estimated_process_peak_bytes'])}")
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
