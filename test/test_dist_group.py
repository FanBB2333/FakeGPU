"""Test ``torch.distributed`` group management fallbacks with FakeGPU.

Verifies that ``_get_default_group()``, ``new_group()``, and related
ProcessGroup APIs work correctly after ``dist.init_process_group()``
through FakeGPU's patched distributed layer.

Requires: torch
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


def _run_snippet(code: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), str(TEST_DIR), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env["FAKEGPU_TERMINAL_REPORT"] = "0"
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def test_get_default_group_after_init() -> None:
    """_get_default_group() returns a ProcessGroup after init_process_group."""
    code = textwrap.dedent(
        """
        from hf_test_utils import patch_fakegpu
        patch_fakegpu(profile="a100", device_count=2)

        import torch.distributed as dist
        dist.init_process_group(backend="nccl", rank=0, world_size=1)

        from torch.distributed.distributed_c10d import _get_default_group
        pg = _get_default_group()
        assert pg is not None
        assert hasattr(pg, "rank") or hasattr(pg, "size")
        dist.destroy_process_group()
        print("ok")
        """
    )
    proc = _run_snippet(code)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_new_group_returns_group() -> None:
    """dist.new_group() creates a sub-group that has rank/size methods."""
    code = textwrap.dedent(
        """
        from hf_test_utils import patch_fakegpu
        patch_fakegpu(profile="a100", device_count=2)

        import torch.distributed as dist
        dist.init_process_group(backend="nccl", rank=0, world_size=1)

        ng = dist.new_group(ranks=[0])
        assert ng is not None

        dist.destroy_process_group()
        print("ok")
        """
    )
    proc = _run_snippet(code)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_distributed_init_destroy_roundtrip() -> None:
    """init → queries → destroy → re-init cycle works without errors."""
    code = textwrap.dedent(
        """
        from hf_test_utils import patch_fakegpu
        patch_fakegpu(profile="a100", device_count=1)

        import torch.distributed as dist

        # First cycle
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
        assert dist.is_initialized()
        assert dist.get_rank() == 0
        assert dist.get_world_size() == 1
        dist.destroy_process_group()
        assert not dist.is_initialized()

        # Second cycle
        dist.init_process_group(backend="nccl", rank=0, world_size=2)
        assert dist.is_initialized()
        assert dist.get_world_size() == 2
        dist.destroy_process_group()
        print("ok")
        """
    )
    proc = _run_snippet(code)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_collectives_after_init() -> None:
    """Basic collective operations work after init_process_group."""
    code = textwrap.dedent(
        """
        from hf_test_utils import patch_fakegpu
        patch_fakegpu(profile="a100", device_count=1)

        import torch
        import torch.distributed as dist

        dist.init_process_group(backend="nccl", rank=0, world_size=1)

        t = torch.ones(4)
        dist.all_reduce(t)
        dist.broadcast(t, src=0)
        dist.barrier()

        out = [torch.zeros(4)]
        dist.all_gather(out, t)

        dist.destroy_process_group()
        print("ok")
        """
    )
    proc = _run_snippet(code)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
