"""Standalone fake-CUDA runtime smoke for torch_patch RNG/checkpoint behavior."""

from __future__ import annotations

import importlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

_orig_import_module = importlib.import_module


def _blocked_import(name: str, package: str | None = None):
    if name == "torch.fakegpu":
        raise ModuleNotFoundError("blocked for standalone fallback validation")
    return _orig_import_module(name, package)


importlib.import_module = _blocked_import

from fakegpu.torch_patch import patch

patch_result = patch()

import torch


def main() -> None:
    assert patch_result.backend == "standalone"
    assert torch.cuda.is_available() is True

    states = torch.cuda.get_rng_state_all()
    assert len(states) == torch.cuda.device_count()
    for state in states:
        assert state.dtype == torch.uint8
        assert state.numel() > 0

    torch.cuda.manual_seed_all(1234)
    saved_states = torch.cuda.get_rng_state_all()
    torch.cuda.set_rng_state_all(saved_states)
    first = torch.randn(4, 4, device="cuda")
    torch.cuda.set_rng_state_all(saved_states)
    second = torch.randn(4, 4, device="cuda")
    assert torch.equal(first.cpu(), second.cpu())

    print("torch patch standalone runtime smoke passed")


if __name__ == "__main__":
    main()
