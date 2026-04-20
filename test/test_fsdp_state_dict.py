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


def test_fsdp_full_state_dict_roundtrip() -> None:
    code = textwrap.dedent(
        """
        import io

        from hf_test_utils import patch_fakegpu

        patch_fakegpu(profile="a100", device_count=2)

        import torch
        import torch.distributed as dist
        import torch.nn as nn
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType

        dist.init_process_group(backend="nccl", rank=0, world_size=2)
        try:
            model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 16)).cuda()
            fsdp_model = FSDP(model, device_id=torch.device("cuda", 0))

            with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT):
                state_dict = fsdp_model.state_dict()

            assert set(state_dict) == {"0.weight", "0.bias", "2.weight", "2.bias"}
            assert state_dict["0.weight"].shape == (64, 32)
            assert state_dict["2.weight"].shape == (16, 64)

            buffer = io.BytesIO()
            torch.save(state_dict, buffer)
            buffer.seek(0)
            reloaded = torch.load(buffer)
            assert reloaded["0.weight"].shape == state_dict["0.weight"].shape
            print("ok")
        finally:
            dist.destroy_process_group()
        """
    )
    proc = _run_snippet(code)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_fsdp_local_state_dict_contains_flat_param() -> None:
    code = textwrap.dedent(
        """
        from hf_test_utils import patch_fakegpu

        patch_fakegpu(profile="a100", device_count=2)

        import torch
        import torch.distributed as dist
        import torch.nn as nn
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType

        dist.init_process_group(backend="nccl", rank=0, world_size=2)
        try:
            model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 16)).cuda()
            fsdp_model = FSDP(model, device_id=torch.device("cuda", 0))

            with FSDP.state_dict_type(fsdp_model, StateDictType.LOCAL_STATE_DICT):
                state_dict = fsdp_model.state_dict()

            assert "_flat_param" in state_dict
            local_shards = state_dict["_flat_param"].local_shards()
            assert len(local_shards) == 1
            assert local_shards[0].tensor.numel() > 0
            print("ok")
        finally:
            dist.destroy_process_group()
        """
    )
    proc = _run_snippet(code)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
