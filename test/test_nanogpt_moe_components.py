#!/usr/bin/env python3
"""Smoke coverage for nanoGPT MoE components and wrapper integration."""

from __future__ import annotations

import importlib.util
import runpy
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
NANOGPT_DIR = REPO_ROOT / "test" / "real_scene" / "nanoGPT"
if str(NANOGPT_DIR) not in sys.path:
    sys.path.insert(0, str(NANOGPT_DIR))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    moe_model = _load_module("nanogpt_moe_model", NANOGPT_DIR / "moe_model.py")
    model = moe_model.MoEGPT(
        moe_model.MoEGPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=2,
            n_head=2,
            n_embd=16,
            dropout=0.0,
            bias=False,
            num_experts=4,
            num_experts_per_tok=2,
            expert_parallel=False,
        )
    )
    idx = torch.randint(0, 32, (2, 8))
    logits, loss = model(idx, idx)
    assert logits.shape == (2, 8, 32)
    assert loss is not None

    train_moe_source = (NANOGPT_DIR / "train_moe.py").read_text(encoding="utf-8")
    compile(train_moe_source, str(NANOGPT_DIR / "train_moe.py"), "exec")

    homo = runpy.run_path(str(NANOGPT_DIR / "config" / "train_moe_homo.py"))
    assert homo["expert_parallel"] is True
    assert homo["dtype"] == "bfloat16"

    hetero = runpy.run_path(str(NANOGPT_DIR / "config" / "train_moe_hetero.py"))
    assert hetero["expert_parallel"] is True
    assert hetero["dtype"] == "float16"

    wrapper = _load_module("nanogpt_train_wrapper_moe", NANOGPT_DIR / "train_wrapper.py")
    assert wrapper._resolve_train_script("gpt") == "train.py"  # type: ignore[attr-defined]
    assert wrapper._resolve_train_script("moe") == "train_moe.py"  # type: ignore[attr-defined]

    print("nanogpt MoE component smoke passed")


if __name__ == "__main__":
    main()
