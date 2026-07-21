from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .llm_estimator import estimate_decoder_inference


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fakegpu estimate-llm",
        description="Estimate dense decoder inference memory and matrix FLOPs without loading weights.",
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--generated-tokens", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--attention-implementation", choices=["eager", "sdpa"], default="eager")
    parser.add_argument("--runtime-overhead-bytes", type=int, default=0)
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
    return 0


def _format_bytes(value: int) -> str:
    return f"{int(value) / 2**30:.3f} GiB"
