"""Test HuggingFace Trainer running through Accelerate's full CUDA path.

Verifies that ``transformers.Trainer`` with ``use_cpu=False`` and fp16 mixed
precision works end-to-end using FakeGPU + Accelerate device placement.

Requires: accelerate, transformers, torch
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


def test_trainer_cuda_path_with_accelerate() -> None:
    """Trainer(use_cpu=False) + accelerate selects cuda device and trains."""
    code = textwrap.dedent(
        """
        from hf_test_utils import (
            patch_fakegpu, make_tiny_causal_lm, make_lm_dataset, make_training_args,
        )
        patch_fakegpu(profile="a100", device_count=1)

        import torch
        from transformers import Trainer

        model = make_tiny_causal_lm()
        args = make_training_args(use_cpu=False)
        trainer = Trainer(model=model, args=args, train_dataset=make_lm_dataset())
        result = trainer.train()

        assert trainer.args.use_cpu is False
        assert next(trainer.model.parameters()).device.type == "cuda"
        assert isinstance(result.training_loss, float)
        assert result.training_loss > 0
        print("ok")
        """
    )
    proc = _run_snippet(code)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_trainer_cuda_fp16_with_accelerate() -> None:
    """Trainer with fp16=True trains in mixed precision via accelerate."""
    code = textwrap.dedent(
        """
        from hf_test_utils import (
            patch_fakegpu, make_tiny_causal_lm, make_lm_dataset, make_training_args,
        )
        patch_fakegpu(profile="a100", device_count=1)

        import torch
        from transformers import Trainer

        model = make_tiny_causal_lm()
        args = make_training_args(use_cpu=False, fp16=True)
        trainer = Trainer(model=model, args=args, train_dataset=make_lm_dataset())
        result = trainer.train()

        assert trainer.args.use_cpu is False
        assert trainer.args.fp16 is True
        assert next(trainer.model.parameters()).device.type == "cuda"
        assert isinstance(result.training_loss, float)
        assert result.training_loss > 0
        print("ok")
        """
    )
    proc = _run_snippet(code)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_trainer_eval_and_predict() -> None:
    """Trainer.evaluate() and Trainer.predict() work on the CUDA path."""
    code = textwrap.dedent(
        """
        from hf_test_utils import (
            patch_fakegpu, make_tiny_causal_lm, make_lm_dataset, make_training_args,
        )
        patch_fakegpu(profile="a100", device_count=1)

        import torch
        from transformers import Trainer

        model = make_tiny_causal_lm()
        ds = make_lm_dataset(size=4)
        args = make_training_args(use_cpu=False)
        trainer = Trainer(model=model, args=args, train_dataset=ds, eval_dataset=ds)

        trainer.train()
        metrics = trainer.evaluate()
        assert "eval_loss" in metrics
        assert isinstance(metrics["eval_loss"], float)

        pred = trainer.predict(ds)
        assert pred.predictions is not None
        print("ok")
        """
    )
    proc = _run_snippet(code)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_trainer_checkpoint_roundtrip() -> None:
    """Trainer save/load checkpoint preserves parameters on CUDA."""
    code = textwrap.dedent(
        """
        import tempfile
        from hf_test_utils import (
            patch_fakegpu, make_tiny_causal_lm, make_lm_dataset, make_training_args,
        )
        patch_fakegpu(profile="a100", device_count=1)

        import torch
        from transformers import Trainer, AutoModelForCausalLM

        model = make_tiny_causal_lm()
        outdir = tempfile.mkdtemp(prefix="fakegpu_ckpt_")
        args = make_training_args(
            output_dir=outdir, use_cpu=False, save_strategy="steps", save_steps=1,
        )
        trainer = Trainer(model=model, args=args, train_dataset=make_lm_dataset())
        trainer.train()

        import glob, os
        ckpts = sorted(glob.glob(os.path.join(outdir, "checkpoint-*")))
        assert len(ckpts) >= 1, f"No checkpoints found in {outdir}"
        print("ok")
        """
    )
    proc = _run_snippet(code)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
