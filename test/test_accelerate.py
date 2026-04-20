"""Test ``accelerate`` integration with FakeGPU.

Verifies that the HuggingFace Accelerate library correctly detects FakeGPU
as CUDA, can initialize an Accelerator, prepare models/optimizers, and
run training steps — including mixed-precision (fp16).

Requires: accelerate, torch
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = Path(__file__).resolve().parent


def _run_snippet(code: str, *, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), str(TEST_DIR), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env["FAKEGPU_TERMINAL_REPORT"] = "0"
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


def test_accelerate_detects_cuda() -> None:
    """accelerate.utils.is_cuda_available() returns True after patch."""
    code = textwrap.dedent(
        """
        from hf_test_utils import patch_fakegpu
        patch_fakegpu(profile="a100", device_count=1)

        from accelerate.utils import is_cuda_available
        assert is_cuda_available() is True
        print("ok")
        """
    )
    proc = _run_snippet(code)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_accelerator_init_and_device() -> None:
    """Accelerator() initializes with device=cuda."""
    code = textwrap.dedent(
        """
        from hf_test_utils import patch_fakegpu
        patch_fakegpu(profile="a100", device_count=1)

        import torch
        from accelerate import Accelerator
        acc = Accelerator()
        assert acc.device.type == "cuda"
        assert str(acc.distributed_type) == "DistributedType.NO"
        print("ok")
        """
    )
    proc = _run_snippet(code)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_accelerator_prepare_model_optimizer() -> None:
    """accelerator.prepare() wraps model and optimizer correctly."""
    code = textwrap.dedent(
        """
        from hf_test_utils import patch_fakegpu
        patch_fakegpu(profile="a100", device_count=1)

        import torch
        from accelerate import Accelerator

        acc = Accelerator()
        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        model, optimizer = acc.prepare(model, optimizer)

        assert next(model.parameters()).device.type == "cuda"
        print("ok")
        """
    )
    proc = _run_snippet(code)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_accelerator_training_step() -> None:
    """Full forward → backward → step via Accelerator."""
    code = textwrap.dedent(
        """
        from hf_test_utils import patch_fakegpu
        patch_fakegpu(profile="a100", device_count=1)

        import torch
        from accelerate import Accelerator

        acc = Accelerator()
        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        model, optimizer = acc.prepare(model, optimizer)

        x = torch.randn(4, 10, device=acc.device)
        y = model(x)
        loss = y.sum()
        acc.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        assert isinstance(loss.item(), float)
        print("ok")
        """
    )
    proc = _run_snippet(code)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_accelerator_fp16_mixed_precision() -> None:
    """Accelerator with mixed_precision='fp16' works end-to-end."""
    code = textwrap.dedent(
        """
        from hf_test_utils import patch_fakegpu
        patch_fakegpu(profile="a100", device_count=1)

        import torch
        from accelerate import Accelerator

        acc = Accelerator(mixed_precision="fp16")
        assert acc.mixed_precision == "fp16"

        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        model, optimizer = acc.prepare(model, optimizer)

        x = torch.randn(4, 10, device=acc.device)
        with acc.autocast():
            y = model(x)
            loss = y.sum()
        acc.backward(loss)
        optimizer.step()

        assert isinstance(loss.item(), float)
        print("ok")
        """
    )
    proc = _run_snippet(code)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_accelerator_device_map_utilities() -> None:
    """accelerate model partitioning utilities work with FakeGPU memory info."""
    code = textwrap.dedent(
        """
        from hf_test_utils import patch_fakegpu, make_tiny_causal_lm
        patch_fakegpu(profile="a100", device_count=1)

        from accelerate.utils import infer_auto_device_map

        model = make_tiny_causal_lm()
        device_map = infer_auto_device_map(model, max_memory={0: "10GB", "cpu": "10GB"})
        assert isinstance(device_map, dict)
        assert len(device_map) > 0
        print("ok")
        """
    )
    proc = _run_snippet(code)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
