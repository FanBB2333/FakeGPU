from __future__ import annotations

import json
import gc
import os
import subprocess
import sys
import textwrap
import weakref
from pathlib import Path

import pytest

from fakegpu.torch_patch import (
    _DeviceMemoryTracker,
    _TrackedSavedTensor,
    _unpack_autograd_saved_tensor,
)


ROOT = Path(__file__).resolve().parents[1]
MIB = 1024**2


def test_saved_tensor_unpack_detaches_stale_finalizer() -> None:
    tracker = _DeviceMemoryTracker([64 * MIB], caching_allocator=True)
    raw_data_ptr = 1234
    tracker.allocate_saved_tensor(
        raw_data_ptr=raw_data_ptr,
        nbytes=1_000,
        device=0,
    )
    holder = _TrackedSavedTensor(
        object(),
        raw_data_ptr,
        tracker.release_saved_tensor,
    )
    holder._finalizer = weakref.finalize(
        holder,
        holder._release,
        raw_data_ptr,
    )

    _unpack_autograd_saved_tensor(holder)
    tracker.allocate_saved_tensor(
        raw_data_ptr=raw_data_ptr,
        nbytes=1_000,
        device=0,
    )
    del holder
    gc.collect()

    # memory_allocated() counts the 512-byte-rounded block (round_size()).
    assert tracker.memory_allocated(0) == 1_024
    assert tracker.memory_stats(0)["requested_bytes.all.current"] == 1_000


def test_small_allocations_share_and_release_cached_segment() -> None:
    tracker = _DeviceMemoryTracker([64 * MIB], caching_allocator=True)

    assert tracker.allocate(101, 1_000, 0)
    assert tracker.allocate(102, 1_000, 0)
    assert tracker.memory_allocated(0) == 2_048
    assert tracker.memory_reserved(0) == 2 * MIB
    assert tracker.max_memory_reserved(0) == 2 * MIB

    tracker.release(101)
    assert tracker.memory_allocated(0) == 1_024
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


def test_allocator_keeps_small_and_large_pools_apart() -> None:
    # CUDACachingAllocator get_pool()/get_free_block() (PyTorch v2.14.1):
    # requests up to 1 MiB only reuse small-pool blocks and larger requests
    # only reuse large-pool blocks, so each pool reserves its own segment.
    tracker = _DeviceMemoryTracker([64 * MIB], caching_allocator=True)
    tracker.allocate(1, 5 * MIB, 0)
    tracker.allocate(2, 4, 0)
    assert tracker.memory_reserved(0) == 22 * MIB
    assert sorted(
        (segment["segment_type"], segment["total_size"])
        for segment in tracker.allocator_snapshot()
    ) == [("large", 20 * MIB), ("small", 2 * MIB)]

    tracker = _DeviceMemoryTracker([64 * MIB], caching_allocator=True)
    tracker.allocate(1, 512, 0)
    tracker.release(1)
    tracker.allocate(2, 3 * MIB // 2, 0)
    assert tracker.memory_reserved(0) == 22 * MIB


def test_allocator_keeps_large_pool_remainders_up_to_1mib_in_the_block() -> None:
    # should_split(): a large-pool block is split only when more than
    # kSmallSize (1 MiB) would remain; otherwise the whole block is used.
    tracker = _DeviceMemoryTracker([64 * MIB], caching_allocator=True)
    tracker.allocate(1, 19 * MIB + MIB // 2, 0)
    snapshot = tracker.allocator_snapshot()
    assert [block["size"] for block in snapshot[0]["blocks"]] == [20 * MIB]
    assert tracker.memory_stats(0)["active_bytes.all.current"] == 20 * MIB
    assert tracker.memory_stats(0)["inactive_split_bytes.all.current"] == 0

    tracker.release(1)
    tracker.allocate(2, 19 * MIB, 0)
    blocks = tracker.allocator_snapshot()[0]["blocks"]
    assert [block["size"] for block in blocks] == [20 * MIB]
    tracker.release(2)
    tracker.allocate(3, 18 * MIB, 0)
    blocks = tracker.allocator_snapshot()[0]["blocks"]
    assert [block["size"] for block in blocks] == [18 * MIB, 2 * MIB]


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
    # malloc() always retries once after release_cached_blocks(), which
    # synchronizes first, even when no cached segment could be freed.
    assert stats["num_alloc_retries"] == 1
    assert stats["num_sync_all_streams"] == 1
    assert stats["reserved_bytes.all.current"] == 2 * MIB
    assert stats["allocated_bytes.all.current"] == 2 * MIB


def test_allocator_oom_message_matches_cuda_caching_allocator() -> None:
    # Message layout and sizes from CUDACachingAllocator::malloc() and
    # c10::CachingAllocator::format_size() (PyTorch v2.14.1).
    torch = pytest.importorskip("torch")
    from fakegpu._allocator import _format_size

    assert [
        _format_size(size)
        for size in (512, 1024, 1025, MIB, MIB + 1, 1024**3, 2 * 1024**3)
    ] == [
        "512 bytes",
        "1024 bytes",
        "1.00 KiB",
        "1024.00 KiB",
        "1.00 MiB",
        "1024.00 MiB",
        "2.00 GiB",
    ]

    tracker = _DeviceMemoryTracker([64 * MIB], caching_allocator=True)
    tracker.allocate(1, 60 * MIB - 100, 0)
    with pytest.raises(torch.OutOfMemoryError) as excinfo:
        tracker.allocate(2, 5 * MIB, 0)
    assert str(excinfo.value).startswith(
        "CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total "
        "capacity of 64.00 MiB of which 4.00 MiB is free. Of the allocated "
        "memory 60.00 MiB is allocated by PyTorch, and 0 bytes is reserved by "
        "PyTorch but unallocated. If reserved but unallocated memory is large "
        "try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid "
        "fragmentation."
    )


def test_allocator_memory_stats_follow_cuda_caching_allocator() -> None:
    # Expected values follow DeviceStats updates in CUDACachingAllocator.cpp
    # and the key layout of torch.cuda.memory_stats() (PyTorch v2.14.1).
    tracker = _DeviceMemoryTracker([64 * MIB], caching_allocator=True)
    tracker.allocate(1, 4, 0)
    tracker.allocate(2, 19 * MIB + MIB // 2, 0)

    stats = tracker.memory_stats(0)
    arrays = (
        "active", "active_bytes", "allocated_bytes", "allocation",
        "inactive_split", "inactive_split_bytes", "requested_bytes",
        "reserved_bytes", "segment",
    )
    expected_keys = {
        f"{name}.{pool}.{metric}"
        for name in arrays
        for pool in ("all", "large_pool", "small_pool")
        for metric in ("allocated", "current", "freed", "peak")
    } | {
        f"{name}.{metric}"
        for name in ("oversize_allocations", "oversize_segments")
        for metric in ("allocated", "current", "freed", "peak")
    } | {
        "max_split_size", "num_alloc_retries", "num_device_alloc",
        "num_device_free", "num_oom_rejections", "num_ooms",
        "num_sync_all_streams",
    }
    assert set(stats) == expected_keys
    assert list(stats) == sorted(stats)
    assert stats["max_split_size"] == -1

    # Allocated and active bytes count whole blocks; requested bytes do not.
    assert tracker.memory_allocated(0) == 512 + 20 * MIB
    assert stats["allocated_bytes.small_pool.current"] == 512
    assert stats["allocated_bytes.large_pool.current"] == 20 * MIB
    assert stats["active_bytes.all.current"] == 512 + 20 * MIB
    assert stats["requested_bytes.all.current"] == 4 + 19 * MIB + MIB // 2
    assert stats["allocation.all.current"] == 2
    assert stats["active.all.current"] == 2
    assert stats["segment.small_pool.current"] == 1
    assert stats["reserved_bytes.large_pool.current"] == 20 * MIB
    assert stats["inactive_split.small_pool.current"] == 1
    assert stats["inactive_split_bytes.small_pool.current"] == 2 * MIB - 512
    assert stats["inactive_split_bytes.large_pool.current"] == 0
    assert stats["num_device_alloc"] == 2

    tracker.release(1)
    tracker.empty_cache()
    stats = tracker.memory_stats(0)
    assert stats["inactive_split.small_pool.current"] == 0
    assert stats["inactive_split_bytes.small_pool.freed"] == 2 * MIB - 512
    assert stats["allocated_bytes.all.freed"] == 512
    assert stats["reserved_bytes.all.freed"] == 2 * MIB
    assert stats["segment.all.current"] == 1
    assert stats["num_device_free"] == 1
    assert stats["num_sync_all_streams"] == 1

    # reset_accumulated() zeroes the allocated/freed totals and counters,
    # even for memory that is still allocated.
    tracker.reset_accumulated(0)
    stats = tracker.memory_stats(0)
    assert stats["allocated_bytes.all.allocated"] == 0
    assert stats["allocated_bytes.all.freed"] == 0
    assert stats["reserved_bytes.all.allocated"] == 0
    assert stats["num_device_alloc"] == 0
    assert stats["num_sync_all_streams"] == 0
    assert stats["allocated_bytes.all.current"] == 20 * MIB
    assert stats["allocated_bytes.all.peak"] == 512 + 20 * MIB

    tracker.reset_peak(0)
    assert tracker.max_memory_allocated(0) == 20 * MIB
    assert tracker.memory_stats_as_nested_dict(0)["allocated_bytes"]["all"] == {
        "current": 20 * MIB,
        "peak": 20 * MIB,
        "allocated": 0,
        "freed": 0,
    }


def test_allocator_category_totals_follow_allocate_reclassify_and_release() -> None:
    tracker = _DeviceMemoryTracker([64 * MIB], caching_allocator=True)

    tracker.allocate(1, 1_000, 0, metadata={"category": "activation"})
    tracker.allocate(2, 2_000, 0, metadata={"category": "parameter"})
    assert tracker.current_bytes_by_category(0) == {
        "activation": 1_000,
        "parameter": 2_000,
    }

    tracker.mark_category(1, "parameter")
    assert tracker.current_bytes_by_category(0) == {"parameter": 3_000}

    tracker.release(2)
    assert tracker.current_bytes_by_category(0) == {"parameter": 1_000}
    tracker.release(1)
    assert tracker.current_bytes_by_category(0) == {}


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

        for name in (
            "empty_cache", "memory_allocated", "max_memory_allocated",
            "memory_reserved", "max_memory_reserved", "memory_cached",
            "max_memory_cached", "mem_get_info", "reset_peak_memory_stats",
            "reset_max_memory_allocated", "reset_max_memory_cached",
            "reset_accumulated_memory_stats", "memory_stats", "memory_snapshot",
        ):
            assert getattr(torch.cuda, name) is getattr(torch.cuda.memory, name), name

        baseline_allocated = torch.cuda.memory_allocated()
        baseline_reserved = torch.cuda.memory_reserved()
        tensor = torch.empty(1000, device="cuda", dtype=torch.float32)
        live = {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "max_reserved": torch.cuda.max_memory_reserved(),
            "stats": torch.cuda.memory_stats(),
            "nested_allocated": torch.cuda.memory_stats_as_nested_dict()[
                "allocated_bytes"
            ]["all"]["current"],
            "snapshot": torch.cuda.memory_snapshot(),
        }
        del tensor
        gc.collect()
        released = {
            "allocated": torch.cuda.memory_allocated(),
            "reserved_before_empty": torch.cuda.memory_reserved(),
        }
        torch.cuda.memory.empty_cache()
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
    assert live["stats"]["allocated_bytes.all.current"] == live["allocated"]
    assert live["nested_allocated"] == live["allocated"]
    assert live["stats"]["requested_bytes.all.current"] <= live["allocated"]
    assert live["snapshot"]
    assert payload["released"]["reserved_before_empty"] >= 2 * MIB
    assert (
        payload["released"]["reserved_after_empty"]
        <= payload["released"]["reserved_before_empty"]
    )


def test_fakecuda_module_transfer_reports_buffer_category() -> None:
    code = textwrap.dedent(
        """
        import json
        import os

        os.environ["FAKEGPU_TERMINAL_REPORT"] = "0"

        import fakegpu
        fakegpu.init(runtime="fakecuda", profile="test-512m", device_count=1)

        import torch
        from fakegpu import torch_patch

        class WithBuffer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.zeros(256, 1024))

        moved = [WithBuffer().cuda(), WithBuffer().to("cuda")]
        snapshot = torch_patch.memory_snapshot()
        print(json.dumps(snapshot["devices"][0]["current_bytes_by_category"]))
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
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    categories = json.loads(completed.stdout.strip().splitlines()[-1])
    assert categories.get("buffer", 0) >= 2 * MIB
    assert "tensor" not in categories


def test_fakecuda_allocation_stacks_skip_fakegpu_frames() -> None:
    code = textwrap.dedent(
        """
        import json
        import os

        os.environ["FAKEGPU_TERMINAL_REPORT"] = "0"
        os.environ["FAKEGPU_ALLOCATION_STACKS"] = "1"
        os.environ["FAKEGPU_ALLOCATION_STACK_DEPTH"] = "2"

        import fakegpu
        fakegpu.init(runtime="fakecuda", profile="test-512m", device_count=1)

        import torch
        from fakegpu import torch_patch

        def allocate_here():
            return torch.empty(1024, 1024, device="cuda")

        tensor = allocate_here()
        snapshot = torch_patch.memory_snapshot()
        print(json.dumps(snapshot["devices"][0]["largest_allocations"][0]["stack"]))
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
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    stack = json.loads(completed.stdout.strip().splitlines()[-1])
    assert len(stack) == 2
    assert stack[-1]["function"] == "allocate_here"
    assert all("/fakegpu/" not in frame["file"] for frame in stack)


def test_fakecuda_patches_the_distributed_process_group_lifecycle() -> None:
    code = textwrap.dedent(
        """
        import json
        import os

        os.environ["FAKEGPU_TERMINAL_REPORT"] = "0"

        import fakegpu
        fakegpu.init(runtime="fakecuda", profile="test-512m", device_count=1)

        import torch.distributed as dist

        before = dist.is_initialized()
        dist.init_process_group(backend="nccl", world_size=4, rank=2)
        during = {
            "initialized": dist.is_initialized(),
            "backend": dist.get_backend(),
            "world_size": dist.get_world_size(),
            "rank": dist.get_rank(),
        }
        dist.destroy_process_group()
        after = {
            "initialized": dist.is_initialized(),
            "world_size": dist.get_world_size(),
            "rank": dist.get_rank(),
        }

        dist.init_process_group()
        defaulted = {
            "backend": dist.get_backend(),
            "world_size": dist.get_world_size(),
            "rank": dist.get_rank(),
        }
        dist.destroy_process_group()

        print(json.dumps({
            "before": before,
            "during": during,
            "after": after,
            "defaulted": defaulted,
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
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["before"] is False
    assert payload["during"] == {
        "initialized": True,
        "backend": "nccl",
        "world_size": 4,
        "rank": 2,
    }
    assert payload["after"] == {
        "initialized": False,
        "world_size": 1,
        "rank": 0,
    }
    # An argument-free call keeps the NCCL-shaped single-rank default.
    assert payload["defaulted"] == {
        "backend": "nccl",
        "world_size": 1,
        "rank": 0,
    }
