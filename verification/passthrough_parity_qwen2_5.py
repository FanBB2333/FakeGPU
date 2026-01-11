#!/usr/bin/env python3
"""
Generate a small, deterministic-ish output artifact for passthrough parity checks.

This script runs a single forward pass on Qwen2.5-0.5B-Instruct and saves:
- next_token (int): argmax token id at the last position
- last_logits (torch.FloatTensor): float32 logits for the last position on CPU
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path


def _configure_env_for_determinism() -> None:
    os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    os.environ.setdefault("TORCH_SDPA_KERNEL", "math")
    # Must be set before the first CUDA context is created.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


_configure_env_for_determinism()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Path to write a torch.save() artifact.")
    parser.add_argument(
        "--model-dir",
        default=os.path.expanduser("~/models/Qwen/Qwen2.5-0.5B-Instruct"),
        help="Local model directory (default: ~/models/Qwen/Qwen2.5-0.5B-Instruct).",
    )
    parser.add_argument("--prompt", default="Say hello", help="Prompt string.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    ns = parser.parse_args(argv)

    out_path = Path(ns.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Completely disable torchvision to avoid optional dependency issues.
        transformers.utils.import_utils._torchvision_available = False
        transformers.utils.import_utils._torchvision_version = "0.0"

        if not torch.cuda.is_available():
            print("CUDA not available; passthrough parity requires a real GPU.")
            return 2

        torch.manual_seed(ns.seed)
        torch.cuda.manual_seed_all(ns.seed)

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

        device = torch.device("cuda:0")

        model_dir = Path(ns.model_dir).expanduser()
        if not model_dir.is_dir():
            print(f"Model dir not found: {model_dir}")
            return 2

        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch.float16,
            device_map="cuda:0",
        )
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

        messages = [{"role": "user", "content": ns.prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**model_inputs, use_cache=False)

        last_logits = outputs.logits[:, -1, :].float().cpu()
        next_token = int(torch.argmax(last_logits, dim=-1)[0].item())

        torch.save(
            {
                "next_token": next_token,
                "last_logits": last_logits,
                "meta": {
                    "prompt": ns.prompt,
                    "seed": ns.seed,
                    "torch_version": getattr(torch, "__version__", "unknown"),
                    "transformers_version": getattr(transformers, "__version__", "unknown"),
                },
            },
            str(out_path),
        )
        print(f"Wrote: {out_path}")
        print(f"next_token: {next_token}")
        return 0
    except Exception as e:
        print("passthrough_parity_qwen2_5 failed:", e)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

