from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = Path(__file__).resolve().parent


def _run_snippet(
    code: str,
    *,
    extra_env: dict[str, str] | None = None,
    timeout: float = 45.0,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), str(TEST_DIR), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env["FAKEGPU_TERMINAL_REPORT"] = "0"
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )


def test_fsdp_no_shard_world_size_1_train_step() -> None:
    code = textwrap.dedent(
        """
        from hf_test_utils import patch_fakegpu

        patch_fakegpu(profile="a100", device_count=1)

        import torch
        import torch.distributed as dist
        import torch.nn as nn
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        dist.init_process_group(backend="nccl", rank=0, world_size=1)
        try:
            model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 16)).cuda()
            fsdp_model = FSDP(model, device_id=torch.device("cuda", 0))
            x = torch.randn(4, 32, device="cuda")
            y = fsdp_model(x)
            loss = y.sum()
            loss.backward()
            optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-3)
            optimizer.step()
            assert isinstance(loss.item(), float)
            print("ok")
        finally:
            dist.destroy_process_group()
        """
    )
    proc = _run_snippet(code)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_fsdp_full_shard_rebinds_flat_param_storage() -> None:
    code = textwrap.dedent(
        """
        from hf_test_utils import patch_fakegpu

        patch_fakegpu(profile="a100", device_count=2)

        import torch
        import torch.distributed as dist
        import torch.nn as nn
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        dist.init_process_group(backend="nccl", rank=0, world_size=2)
        try:
            model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 16)).cuda()
            total_numel = sum(param.numel() for param in model.parameters())
            fsdp_model = FSDP(model, device_id=torch.device("cuda", 0))
            flat_param = next(fsdp_model.parameters())

            assert flat_param.data_ptr() != 0
            assert flat_param.untyped_storage().nbytes() > 0
            assert flat_param.numel() * 2 == total_numel
            print("ok")
        finally:
            dist.destroy_process_group()
        """
    )
    proc = _run_snippet(code)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_fsdp_full_shard_forward_backward_smoke() -> None:
    code = textwrap.dedent(
        """
        from hf_test_utils import patch_fakegpu

        patch_fakegpu(profile="a100", device_count=2)

        import torch
        import torch.distributed as dist
        import torch.nn as nn
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        dist.init_process_group(backend="nccl", rank=0, world_size=2)
        try:
            model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 16)).cuda()
            fsdp_model = FSDP(model, device_id=torch.device("cuda", 0))
            x = torch.randn(4, 32, device="cuda")
            y = fsdp_model(x)
            loss = y.sum()
            loss.backward()
            torch.optim.AdamW(fsdp_model.parameters(), lr=1e-3).step()

            assert y.shape == (4, 16)
            assert next(fsdp_model.parameters()).device.type == "cuda"
            print("ok")
        finally:
            dist.destroy_process_group()
        """
    )
    proc = _run_snippet(code)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
