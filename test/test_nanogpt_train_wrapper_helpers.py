#!/usr/bin/env python3
"""Regression checks for nanoGPT validation wrapper helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER_PATH = REPO_ROOT / "test" / "real_scene" / "nanoGPT" / "train_wrapper.py"

spec = importlib.util.spec_from_file_location("nanogpt_train_wrapper", WRAPPER_PATH)
wrapper = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(wrapper)


def _assert_has_flag(args: list[str], expected: str) -> None:
    assert expected in args, args


def main() -> None:
    assert wrapper._resolve_train_script("gpt") == "train.py"  # type: ignore[attr-defined]
    assert wrapper._resolve_train_script("moe") == "train_moe.py"  # type: ignore[attr-defined]

    prepared = wrapper._prepare_train_args(  # type: ignore[attr-defined]
        ["config/train_shakespeare_char.py", "--max_iters=20"],
        mode="full",
        info={"fakegpu_runtime": "fakecuda", "fakegpu_backend": "custom_torch"},
    )
    _assert_has_flag(prepared, "--dtype=float32")
    _assert_has_flag(prepared, "--eval_iters=2")
    _assert_has_flag(prepared, "--batch_size=8")
    _assert_has_flag(prepared, "--block_size=64")

    untouched = wrapper._prepare_train_args(  # type: ignore[attr-defined]
        [
            "config/train_shakespeare_char.py",
            "--dtype=float16",
            "--eval_iters=7",
            "--batch_size=16",
            "--block_size=32",
        ],
        mode="full",
        info={"fakegpu_runtime": "fakecuda", "fakegpu_backend": "custom_torch"},
    )
    assert "--dtype=float16" in untouched
    assert "--eval_iters=7" in untouched
    assert "--batch_size=16" in untouched
    assert "--block_size=32" in untouched
    assert "--dtype=float32" not in untouched

    limiter = wrapper._FakeCudaMemoryLimiter(1024)  # type: ignore[attr-defined]
    small = torch.empty(128, dtype=torch.float32)
    limiter.reserve_tensor(small, context="small")
    try:
        limiter.reserve_tensor(torch.empty(1024, dtype=torch.float32), context="large")
    except RuntimeError as exc:
        assert "out of memory" in str(exc).lower()
    else:
        raise AssertionError("expected fake CUDA OOM to trigger")

    profile_memory = wrapper._resolve_profile_memory_bytes(  # type: ignore[attr-defined]
        profile="a100-1g",
        devices=None,
    )
    assert profile_memory == 1024**3

    devices_memory = wrapper._resolve_profile_memory_bytes(  # type: ignore[attr-defined]
        profile=None,
        devices="a100-1g,a100",
    )
    assert devices_memory == 1024**3

    print("nanogpt train wrapper helper regression passed")


if __name__ == "__main__":
    main()
