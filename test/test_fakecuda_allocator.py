from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from fakegpu.torch_patch import _DeviceMemoryTracker


ROOT = Path(__file__).resolve().parents[1]
MIB = 1024**2


def test_small_allocations_share_and_release_cached_segment() -> None:
    tracker = _DeviceMemoryTracker([64 * MIB], caching_allocator=True)

    assert tracker.allocate(101, 1_000, 0)
    assert tracker.allocate(102, 1_000, 0)
    assert tracker.memory_allocated(0) == 2_000
    assert tracker.memory_reserved(0) == 2 * MIB
    assert tracker.max_memory_reserved(0) == 2 * MIB

    tracker.release(101)
    assert tracker.memory_allocated(0) == 1_000
    assert tracker.memory_reserved(0) == 2 * MIB
    tracker.empty_cache()
    assert tracker.memory_reserved(0) == 2 * MIB

    tracker.release(102)
    assert tracker.memory_allocated(0) == 0
    assert tracker.memory_reserved(0) == 2 * MIB
    tracker.empty_cache()
    assert tracker.memory_reserved(0) == 0


def test_allocator_best_fit_splits_and_coalesces_blocks() -> None:
    tracker = _DeviceMemoryTracker([64 * MIB], caching_allocator=True)
    tracker.allocate(1, MIB, 0)
    tracker.allocate(2, MIB, 0)
    tracker.release(1)
    tracker.allocate(3, MIB // 2, 0)

    snapshot = tracker.allocator_snapshot()
    assert len(snapshot) == 1
    blocks = snapshot[0]["blocks"]
    assert [block["size"] for block in blocks] == [MIB // 2, MIB // 2, MIB]
    assert [block["state"] for block in blocks] == [
        "active_allocated",
        "inactive",
        "active_allocated",
    ]

    tracker.release(2)
    tracker.release(3)
    snapshot = tracker.allocator_snapshot()
    assert len(snapshot[0]["blocks"]) == 1
    assert snapshot[0]["blocks"][0]["size"] == 2 * MIB


def test_allocator_releases_cached_segments_before_retry() -> None:
    tracker = _DeviceMemoryTracker([20 * MIB], caching_allocator=True)
    tracker.allocate(1, 1_000, 0)
    tracker.release(1)
    assert tracker.memory_reserved(0) == 2 * MIB

    tracker.allocate(2, 3 * MIB, 0)
    stats = tracker.memory_stats(0)
    assert tracker.memory_reserved(0) == 20 * MIB
    assert stats["num_alloc_retries"] == 1
    assert stats["reserved_bytes.all.freed"] == 2 * MIB


def test_allocator_oom_reports_reserved_state_and_counter() -> None:
    torch = pytest.importorskip("torch")
    tracker = _DeviceMemoryTracker([2 * MIB], caching_allocator=True)
    tracker.allocate(1, MIB, 0)
    tracker.allocate(2, MIB, 0)

    with pytest.raises(torch.cuda.OutOfMemoryError, match="reserved"):
        tracker.allocate(3, 512, 0)

    stats = tracker.memory_stats(0)
    assert stats["num_ooms"] == 1
    assert stats["reserved_bytes.all.current"] == 2 * MIB
    assert stats["allocated_bytes.all.current"] == 2 * MIB


def test_fakecuda_memory_api_exposes_allocator_state() -> None:
    code = textwrap.dedent(
        """
        import gc
        import json
        import os

        os.environ["FAKEGPU_TERMINAL_REPORT"] = "0"

        import fakegpu
        fakegpu.init(runtime="fakecuda", profile="test-512m", device_count=1)

        import torch

        baseline_allocated = torch.cuda.memory_allocated()
        baseline_reserved = torch.cuda.memory_reserved()
        tensor = torch.empty(1000, device="cuda", dtype=torch.float32)
        live = {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "max_reserved": torch.cuda.max_memory_reserved(),
            "stats": torch.cuda.memory_stats(),
            "snapshot": torch.cuda.memory_snapshot(),
        }
        del tensor
        gc.collect()
        released = {
            "allocated": torch.cuda.memory_allocated(),
            "reserved_before_empty": torch.cuda.memory_reserved(),
        }
        torch.cuda.empty_cache()
        released["reserved_after_empty"] = torch.cuda.memory_reserved()
        print(json.dumps({
            "baseline_allocated": baseline_allocated,
            "baseline_reserved": baseline_reserved,
            "live": live,
            "released": released,
        }, sort_keys=True))
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    live = payload["live"]
    assert live["allocated"] - payload["baseline_allocated"] >= 4_000
    assert live["reserved"] >= 2 * MIB
    assert live["max_reserved"] >= live["reserved"]
    assert live["stats"]["reserved_bytes.all.current"] == live["reserved"]
    assert live["stats"]["requested_bytes.all.current"] == live["allocated"]
    assert live["snapshot"]
    assert payload["released"]["reserved_before_empty"] >= 2 * MIB
    assert (
        payload["released"]["reserved_after_empty"]
        <= payload["released"]["reserved_before_empty"]
    )
