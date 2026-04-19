from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = Path(__file__).resolve().parent


def _run_snippet(code: str, *, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), str(TEST_DIR), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env["XONSH_HISTORY_BACKEND"] = "dummy"
    env["FAKEGPU_TERMINAL_REPORT"] = "0"
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )


def test_trainer_runs_in_cuda_path_without_use_cpu() -> None:
    code = textwrap.dedent(
        """
        from hf_test_utils import make_lm_dataset, make_tiny_causal_lm, make_training_args, patch_fakegpu

        patch_fakegpu(profile="a100", device_count=1)

        import torch
        from transformers import Trainer

        model = make_tiny_causal_lm()
        trainer = Trainer(
            model=model,
            args=make_training_args(),
            train_dataset=make_lm_dataset(),
        )
        result = trainer.train()

        assert trainer.args.use_cpu is False
        assert trainer.args.no_cuda is False
        assert next(trainer.model.parameters()).device.type == "cuda"
        assert isinstance(result.training_loss, float)
        assert result.training_loss > 0
        print("ok")
        """
    )
    proc = _run_snippet(code)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_hf_cuda_surface_reports_cuda_build() -> None:
    code = textwrap.dedent(
        """
        from hf_test_utils import patch_fakegpu

        patch_fakegpu(profile="a100", device_count=1)

        import torch
        from transformers.utils.import_utils import is_torch_tf32_available

        assert torch.version.cuda == "12.1"
        assert torch.backends.cuda.is_built() is True
        assert torch.backends.cudnn.is_available() is True
        assert torch.cuda.nccl.version() == (2, 21, 5)
        assert is_torch_tf32_available() is True
        print("ok")
        """
    )
    proc = _run_snippet(code)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_cuda_random_fork_rng_restores_cpu_rng_state() -> None:
    code = textwrap.dedent(
        """
        from hf_test_utils import patch_fakegpu

        patch_fakegpu(profile="a100", device_count=1)

        import torch

        torch.manual_seed(0)
        before = torch.random.get_rng_state().clone()
        with torch.cuda.random.fork_rng(devices=[0]):
            torch.manual_seed(123)
            _ = torch.rand(4)
        after = torch.random.get_rng_state()
        assert torch.equal(before, after)
        print("ok")
        """
    )
    proc = _run_snippet(code)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_tree_map_rewraps_plain_dataclass_tensors() -> None:
    code = textwrap.dedent(
        """
        import dataclasses

        from hf_test_utils import patch_fakegpu

        patch_fakegpu(profile="a100", device_count=1)

        import fakegpu.torch_patch as tp
        import torch

        @dataclasses.dataclass
        class Box:
            value: object

        wrapped = tp._upstream_mod._tree_map(
            lambda obj: obj.cuda() if isinstance(obj, torch.Tensor) else obj,
            Box(torch.ones(2)),
        )
        assert wrapped.value.device.type == "cuda"
        print("ok")
        """
    )
    proc = _run_snippet(code)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
