"""FakeCUDA device-memory accounting and the modeled caching allocator.

Split out of ``torch_patch`` unchanged: this module models what PyTorch's
caching allocator would do with each request — segment sizing, best-fit
reuse, splitting, coalescing, and ``empty_cache()`` — and keeps the byte,
category, and dispatch-tracking counters that ``torch.cuda``'s memory APIs
report through the patch layer.
"""

from __future__ import annotations

import os
import threading
from bisect import bisect_left, insort
from collections import OrderedDict
from typing import Any


_ALLOCATOR_ALIGNMENT_BYTES = 512


_ALLOCATOR_SMALL_SEGMENT_BYTES = 2 * 1024**2


_ALLOCATOR_MEDIUM_SEGMENT_BYTES = 20 * 1024**2


_ALLOCATOR_LARGE_ALIGNMENT_BYTES = 2 * 1024**2


_ALLOCATOR_SMALL_REQUEST_LIMIT = 1024**2


_ALLOCATOR_MEDIUM_REQUEST_LIMIT = 10 * 1024**2


_dispatch_tracking_lock = threading.RLock()


_dispatch_tracking_stats: dict[str, Any] = {}


def _reset_dispatch_tracking_stats(*, enabled: bool) -> None:
    with _dispatch_tracking_lock:
        _dispatch_tracking_stats.clear()
        _dispatch_tracking_stats.update(
            {
                "enabled": bool(enabled),
                "operator_calls": 0,
                "output_tensors": 0,
                "new_allocations": 0,
                "alias_outputs": 0,
                "inaccessible_outputs": 0,
                "operators": {},
            }
        )


_reset_dispatch_tracking_stats(enabled=False)


def _record_dispatch_tracking(
    operator: str,
    *,
    output_tensors: int,
    new_allocations: int,
    alias_outputs: int,
    inaccessible_outputs: int,
) -> None:
    with _dispatch_tracking_lock:
        _dispatch_tracking_stats["operator_calls"] += 1
        _dispatch_tracking_stats["output_tensors"] += int(output_tensors)
        _dispatch_tracking_stats["new_allocations"] += int(new_allocations)
        _dispatch_tracking_stats["alias_outputs"] += int(alias_outputs)
        _dispatch_tracking_stats["inaccessible_outputs"] += int(
            inaccessible_outputs
        )
        operators = _dispatch_tracking_stats["operators"]
        record = operators.setdefault(
            operator,
            {
                "calls": 0,
                "output_tensors": 0,
                "new_allocations": 0,
                "alias_outputs": 0,
                "inaccessible_outputs": 0,
            },
        )
        record["calls"] += 1
        record["output_tensors"] += int(output_tensors)
        record["new_allocations"] += int(new_allocations)
        record["alias_outputs"] += int(alias_outputs)
        record["inaccessible_outputs"] += int(inaccessible_outputs)


def _dispatch_tracking_snapshot() -> dict[str, Any]:
    with _dispatch_tracking_lock:
        operators = {
            name: dict(values)
            for name, values in sorted(
                _dispatch_tracking_stats["operators"].items()
            )
        }
        return {
            **{
                key: value
                for key, value in _dispatch_tracking_stats.items()
                if key != "operators"
            },
            "operators": operators,
        }


def _round_allocator_bytes(nbytes: int) -> int:
    value = max(0, int(nbytes))
    if value == 0:
        return 0
    return (
        (value + _ALLOCATOR_ALIGNMENT_BYTES - 1)
        // _ALLOCATOR_ALIGNMENT_BYTES
        * _ALLOCATOR_ALIGNMENT_BYTES
    )


def _allocator_pool(block_size: int) -> str:
    # CUDACachingAllocator get_pool(): requests up to kSmallSize (1 MiB)
    # come from the small pool, everything else from the large pool.
    return "small" if block_size <= _ALLOCATOR_SMALL_REQUEST_LIMIT else "large"


def _allocator_segment_size(block_size: int) -> tuple[int, str]:
    if block_size <= _ALLOCATOR_SMALL_REQUEST_LIMIT:
        return _ALLOCATOR_SMALL_SEGMENT_BYTES, "small"
    if block_size < _ALLOCATOR_MEDIUM_REQUEST_LIMIT:
        return _ALLOCATOR_MEDIUM_SEGMENT_BYTES, "large"
    size = (
        (block_size + _ALLOCATOR_LARGE_ALIGNMENT_BYTES - 1)
        // _ALLOCATOR_LARGE_ALIGNMENT_BYTES
        * _ALLOCATOR_LARGE_ALIGNMENT_BYTES
    )
    return size, "large"


def _format_size(size: int) -> str:
    """Mirror ``c10::CachingAllocator::format_size`` used in OOM messages."""
    if size <= 1024:
        return f"{size} bytes"
    if size <= 1024**2:
        return f"{size / 1024:.2f} KiB"
    if size <= 1024**3:
        return f"{size / 1024**2:.2f} MiB"
    return f"{size / 1024**3:.2f} GiB"


def _allocator_should_split(pool: str, remaining: int) -> bool:
    # CUDACachingAllocator should_split(): a small-pool block keeps any
    # remainder of at least kMinBlockSize as a separate free block; a
    # large-pool block is split only when more than kSmallSize remains, so
    # smaller remainders stay inside the allocated block.
    if pool == "small":
        return remaining >= _ALLOCATOR_ALIGNMENT_BYTES
    return remaining > _ALLOCATOR_SMALL_REQUEST_LIMIT


def _coalesce_allocator_blocks(
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for block in blocks:
        if (
            merged
            and merged[-1]["data_ptr"] is None
            and block["data_ptr"] is None
            and merged[-1]["offset"] + merged[-1]["size"] == block["offset"]
        ):
            merged[-1]["size"] += block["size"]
            continue
        merged.append(block)
    return merged


def _segment_inactive_split(segment: dict[str, Any]) -> tuple[int, int]:
    """Return the count and bytes of the segment's free blocks that are split.

    A free block is split, and so cannot be released, while another block of
    its segment is still active.
    """
    if segment["active_count"] <= 0:
        return 0, 0
    free_blocks = sum(1 for block in segment["blocks"] if block["data_ptr"] is None)
    return free_blocks, segment["free_bytes"]


def _public_allocation_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not str(key).startswith("_")}


# c10::CachingAllocator::StatType order and the names memory_stats() uses.
_STAT_TYPE_NAMES = ("all", "small_pool", "large_pool")
_POOL_STAT_TYPE = {"small": 1, "large": 2}
# The StatArray members of c10::CachingDeviceAllocator::DeviceStats.
_STAT_ARRAY_NAMES = (
    "allocation",
    "segment",
    "active",
    "inactive_split",
    "allocated_bytes",
    "reserved_bytes",
    "active_bytes",
    "inactive_split_bytes",
    "requested_bytes",
)
# First synthetic segment address reported by memory_snapshot().
_SNAPSHOT_ADDRESS_BASE = 0x7F0000000000
# DeviceStats.max_split_size is the size_t limit cast to int64_t when
# max_split_size_mb is not configured.
_UNLIMITED_MAX_SPLIT_SIZE = -1


class _Stat:
    """Mirror of ``c10::CachingAllocator::Stat``."""

    __slots__ = ("current", "peak", "allocated", "freed")

    def __init__(self) -> None:
        self.current = 0
        self.peak = 0
        self.allocated = 0
        self.freed = 0

    def increase(self, amount: int) -> None:
        self.current += amount
        self.peak = max(self.current, self.peak)
        self.allocated += amount

    def decrease(self, amount: int) -> None:
        self.current -= amount
        self.freed += amount

    def as_dict(self) -> dict[str, int]:
        return {
            "current": self.current,
            "peak": self.peak,
            "allocated": self.allocated,
            "freed": self.freed,
        }


class _DeviceStats:
    """Mirror of ``c10::CachingDeviceAllocator::DeviceStats`` for one device.

    Each StatArray keeps the aggregate plus the small- and large-pool
    breakdown, so every update names the pool it belongs to.
    """

    def __init__(self) -> None:
        self.arrays = {
            name: tuple(_Stat() for _ in _STAT_TYPE_NAMES)
            for name in _STAT_ARRAY_NAMES
        }
        self.num_alloc_retries = 0
        self.num_ooms = 0
        self.num_sync_all_streams = 0
        self.num_device_alloc = 0
        self.num_device_free = 0
        self.num_oom_rejections = 0
        # Nothing reaches max_split_size while it is unlimited.
        self.oversize_allocations = _Stat()
        self.oversize_segments = _Stat()

    def change(self, name: str, pool: str, delta: int) -> None:
        if delta == 0:
            return
        stats = self.arrays[name]
        for stat in (stats[0], stats[_POOL_STAT_TYPE[pool]]):
            if delta > 0:
                stat.increase(delta)
            else:
                stat.decrease(-delta)

    def total(self, name: str) -> _Stat:
        return self.arrays[name][0]

    def reset_accumulated(self) -> None:
        for stats in self.arrays.values():
            for stat in stats:
                stat.allocated = 0
                stat.freed = 0
        for stat in (self.oversize_allocations, self.oversize_segments):
            stat.allocated = 0
            stat.freed = 0
        self.num_alloc_retries = 0
        self.num_ooms = 0
        self.num_sync_all_streams = 0
        self.num_device_alloc = 0
        self.num_device_free = 0
        self.num_oom_rejections = 0

    def reset_peak(self) -> None:
        for stats in self.arrays.values():
            for stat in stats:
                stat.peak = stat.current
        for stat in (self.oversize_allocations, self.oversize_segments):
            stat.peak = stat.current

    def as_nested_dict(self) -> dict[str, Any]:
        """Return the dictionary ``torch._C._cuda_memoryStats`` builds."""
        result: dict[str, Any] = {
            "num_alloc_retries": self.num_alloc_retries,
            "num_ooms": self.num_ooms,
            "max_split_size": _UNLIMITED_MAX_SPLIT_SIZE,
            "num_sync_all_streams": self.num_sync_all_streams,
            "num_device_alloc": self.num_device_alloc,
            "num_device_free": self.num_device_free,
            "num_oom_rejections": self.num_oom_rejections,
        }
        for name in _STAT_ARRAY_NAMES:
            result[name] = {
                type_name: stat.as_dict()
                for type_name, stat in zip(_STAT_TYPE_NAMES, self.arrays[name])
            }
        result["reserved_bytes_by_private_pools"] = {}
        result["oversize_allocations"] = self.oversize_allocations.as_dict()
        result["oversize_segments"] = self.oversize_segments.as_dict()
        return result


def _flatten_memory_stats(nested: dict[str, Any]) -> OrderedDict[str, Any]:
    """Flatten nested stats the way ``torch.cuda.memory_stats`` does."""
    result: list[tuple[str, Any]] = []

    def _add(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            if prefix:
                prefix += "."
            for key, item in value.items():
                _add(prefix + str(key), item)
        else:
            result.append((prefix, value))

    _add("", nested)
    result.sort()
    return OrderedDict(result)


class _DeviceMemoryTracker:
    """Track tensor bytes and a simplified CUDA caching allocator per device."""

    def __init__(
        self,
        per_device_bytes: list[int],
        *,
        caching_allocator: bool | None = None,
    ):
        self._total = list(per_device_bytes)
        # Requested tensor bytes, which FakeGPU's own reports use.  The
        # torch.cuda memory API reports the allocator stats below instead.
        self._used = [0] * len(per_device_bytes)
        self._peak = [0] * len(per_device_bytes)
        self._stats = [_DeviceStats() for _ in per_device_bytes]
        # set_per_process_memory_fraction() cap on reserved bytes, if any.
        self._allowed_memory_maximum: list[int | None] = [
            None for _ in per_device_bytes
        ]
        self._alloc_calls = [0] * len(per_device_bytes)
        self._free_calls = [0] * len(per_device_bytes)
        self._peak_by_stage: list[dict[str, int]] = [dict() for _ in per_device_bytes]
        self._reserved_peak_by_stage: list[dict[str, int]] = [
            dict() for _ in per_device_bytes
        ]
        self._largest_allocations: list[list[dict[str, Any]]] = [
            [] for _ in per_device_bytes
        ]
        self._current_bytes_by_category: list[dict[str, int]] = [
            {} for _ in per_device_bytes
        ]
        # data_ptr -> allocation record
        self._allocs: dict[int, dict[str, Any]] = {}
        # Each segment contains ordered blocks with ``offset``, ``size``, and
        # an optional active ``data_ptr``. This is sufficient to model best-fit
        # reuse, splitting, coalescing, fragmentation, and empty_cache().
        self._segments: list[list[dict[str, Any]]] = [[] for _ in per_device_bytes]
        self._segments_by_id: dict[int, dict[str, Any]] = {}
        # Free blocks are indexed per (device, pool): like PyTorch, a request
        # only reuses blocks from its own small or large pool.
        self._free_block_sizes: dict[tuple[int, str], list[int]] = {}
        self._free_block_keys: dict[
            tuple[int, str], dict[int, list[tuple[int, int]]]
        ] = {}
        self._free_blocks_by_key: dict[
            tuple[int, int], dict[str, Any]
        ] = {}
        self._allocation_blocks: dict[int, tuple[int, int]] = {}
        self._next_segment_id = 1
        # Synthetic device addresses for memory_snapshot(); segments never
        # reuse a range, as with distinct cudaMalloc results.
        self._next_segment_address = _SNAPSHOT_ADDRESS_BASE
        if caching_allocator is None:
            caching_allocator = os.environ.get("FAKEGPU_CACHING_ALLOCATOR", "1") != "0"
        self._caching_allocator = bool(caching_allocator)
        self._lock = threading.RLock()
        self._next_synthetic_data_ptr = -1
        self._synthetic_saved_raw_ptrs: dict[int, dict[str, int]] = {}
        self._held_saved_raw_ptrs: dict[int, int] = {}
        self._pending_saved_releases: set[int] = set()

    @property
    def _reserved(self) -> list[int]:
        return [stats.total("reserved_bytes").current for stats in self._stats]

    @property
    def _reserved_peak(self) -> list[int]:
        return [stats.total("reserved_bytes").peak for stats in self._stats]

    def allocate(
        self,
        data_ptr: int,
        nbytes: int,
        device: int,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Register allocation. Raise OutOfMemoryError if exceeds limit."""
        with self._lock:
            if device < 0 or device >= len(self._total):
                return False
            if data_ptr in self._allocs:
                return False  # already tracked
            nbytes = max(0, int(nbytes))
            block_size = _round_allocator_bytes(nbytes)
            block = self._allocate_allocator_block(
                data_ptr=data_ptr,
                block_size=block_size,
                device=device,
            )
            stats = self._stats[device]
            if block is None:
                # CUDACachingAllocator::malloc(): after a failed cudaMalloc it
                # synchronizes, frees every unsplit cached block and always
                # retries once, counting the retry even if nothing was freed.
                stats.num_sync_all_streams += 1
                self._release_empty_segments(device)
                stats.num_alloc_retries += 1
                block = self._allocate_allocator_block(
                    data_ptr=data_ptr,
                    block_size=block_size,
                    device=device,
                )
            if block is None:
                stats.num_ooms += 1
                import torch

                raise torch.cuda.OutOfMemoryError(
                    self._out_of_memory_message(device, block_size)
                )
            meta = dict(metadata or {})
            stage = str(
                meta.get("stage")
                or os.environ.get("FAKEGPU_PREFLIGHT_STAGE")
                or "unknown"
            )
            meta["stage"] = stage

            # An unsplit large-pool block can be bigger than the rounded request.
            block_size = int(block["size"])
            pool = str(block["pool"])
            self._allocs[data_ptr] = {
                "device": device,
                "bytes": nbytes,
                "block_bytes": block_size,
                "segment_id": int(block["segment_id"]),
                **meta,
                "pool": pool,
            }
            category = str(meta.get("category") or "unknown")
            categories = self._current_bytes_by_category[device]
            categories[category] = categories.get(category, 0) + nbytes
            self._used[device] += nbytes
            # alloc_found_block(): allocated and active bytes count the whole
            # block, requested bytes count what the caller asked for.
            stats.change("allocation", pool, 1)
            stats.change("allocated_bytes", pool, block_size)
            stats.change("active", pool, 1)
            stats.change("active_bytes", pool, block_size)
            stats.change("requested_bytes", pool, nbytes)
            self._peak[device] = max(self._peak[device], self._used[device])
            stage_peaks = self._peak_by_stage[device]
            stage_peaks[stage] = max(stage_peaks.get(stage, 0), self._used[device])
            reserved_stage_peaks = self._reserved_peak_by_stage[device]
            reserved_stage_peaks[stage] = max(
                reserved_stage_peaks.get(stage, 0),
                stats.total("reserved_bytes").current,
            )
            self._alloc_calls[device] += 1
            self._record_largest_allocation(data_ptr, device, nbytes, meta)
            return True

    def _out_of_memory_message(self, device: int, block_size: int) -> str:
        # CUDACachingAllocator::malloc() reports the segment size it tried to
        # cudaMalloc and the free memory left after releasing cached blocks.
        # Per-process NVML usage is omitted, as PyTorch does without NVML.
        alloc_size, _ = _allocator_segment_size(block_size)
        total = self._total[device]
        reserved = self._stats[device].total("reserved_bytes").current
        allocated = self._stats[device].total("allocated_bytes").current
        free = max(0, total - reserved)
        allowed = self._allowed_memory_maximum[device]
        allowed_info = (
            "" if allowed is None else f"{_format_size(allowed)} allowed; "
        )
        return (
            f"CUDA out of memory. Tried to allocate {_format_size(alloc_size)}. "
            f"GPU {device} has a total capacity of {_format_size(total)} "
            f"of which {_format_size(free)} is free. {allowed_info}"
            f"Of the allocated memory {_format_size(allocated)} is allocated "
            f"by PyTorch, and {_format_size(reserved - allocated)} is reserved "
            "by PyTorch but unallocated. If reserved but unallocated memory is "
            "large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
            "to avoid fragmentation.  See documentation for Memory Management  "
            "(https://docs.pytorch.org/docs/stable/notes/cuda.html"
            "#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)"
        )

    def release(self, data_ptr: int) -> None:
        """Unregister allocation."""
        with self._lock:
            if self._held_saved_raw_ptrs.get(data_ptr, 0) > 0:
                self._pending_saved_releases.add(data_ptr)
                return
            self._release_now(data_ptr)

    def _release_now(self, data_ptr: int) -> None:
        rec = self._allocs.pop(data_ptr, None)
        if rec:
            dev = rec["device"]
            nbytes = rec["bytes"]
            category = str(rec.get("category") or "unknown")
            categories = self._current_bytes_by_category[dev]
            remaining = max(0, categories.get(category, 0) - nbytes)
            if remaining:
                categories[category] = remaining
            else:
                categories.pop(category, None)
            self._used[dev] = max(0, self._used[dev] - nbytes)
            stats = self._stats[dev]
            pool = rec["pool"]
            stats.change("allocation", pool, -1)
            stats.change("allocated_bytes", pool, -rec["block_bytes"])
            self._free_calls[dev] += 1
            self._free_allocator_block(data_ptr, dev)
            stats.change("active", pool, -1)
            stats.change("active_bytes", pool, -rec["block_bytes"])
            stats.change("requested_bytes", pool, -nbytes)

    def allocate_saved_tensor(
        self,
        *,
        raw_data_ptr: int,
        nbytes: int,
        device: int,
        metadata: dict[str, Any] | None = None,
    ) -> int | None:
        """Track an autograd-saved raw tensor that is hidden behind a high-level op."""
        with self._lock:
            if raw_data_ptr in self._allocs:
                self._held_saved_raw_ptrs[raw_data_ptr] = (
                    self._held_saved_raw_ptrs.get(raw_data_ptr, 0) + 1
                )
                return raw_data_ptr
            existing = self._synthetic_saved_raw_ptrs.get(raw_data_ptr)
            if existing is not None:
                existing["refs"] += 1
                return int(existing["synthetic_data_ptr"])

            synthetic_data_ptr = self._next_synthetic_data_ptr
            self._next_synthetic_data_ptr -= 1
            meta = {
                "category": "activation",
                "synthetic": True,
                "source": "autograd_saved_tensor",
                **(metadata or {}),
            }
            self.allocate(synthetic_data_ptr, nbytes, device, metadata=meta)
            self._synthetic_saved_raw_ptrs[raw_data_ptr] = {
                "synthetic_data_ptr": synthetic_data_ptr,
                "refs": 1,
            }
            return synthetic_data_ptr

    def release_saved_tensor(self, raw_data_ptr: int) -> None:
        with self._lock:
            held_refs = self._held_saved_raw_ptrs.get(raw_data_ptr)
            if held_refs is not None:
                held_refs -= 1
                if held_refs > 0:
                    self._held_saved_raw_ptrs[raw_data_ptr] = held_refs
                    return
                self._held_saved_raw_ptrs.pop(raw_data_ptr, None)
                if raw_data_ptr in self._pending_saved_releases:
                    self._pending_saved_releases.remove(raw_data_ptr)
                    self._release_now(raw_data_ptr)
                return

            existing = self._synthetic_saved_raw_ptrs.get(raw_data_ptr)
            if existing is None:
                return
            refs = int(existing.get("refs", 1)) - 1
            if refs > 0:
                existing["refs"] = refs
                return
            synthetic_data_ptr = int(existing["synthetic_data_ptr"])
            self._synthetic_saved_raw_ptrs.pop(raw_data_ptr, None)
            self._release_now(synthetic_data_ptr)

    def has_allocation(self, data_ptr: int) -> bool:
        with self._lock:
            return int(data_ptr) in self._allocs

    def _device_stat(self, device: int, name: str) -> _Stat | None:
        if device < 0 or device >= len(self._stats):
            return None
        return self._stats[device].total(name)

    def memory_allocated(self, device: int) -> int:
        # torch.cuda.memory_allocated() is allocated_bytes.all.current: the
        # allocator's block sizes, not the requested tensor bytes.
        with self._lock:
            stat = self._device_stat(device, "allocated_bytes")
            return 0 if stat is None else stat.current

    def max_memory_allocated(self, device: int) -> int:
        with self._lock:
            stat = self._device_stat(device, "allocated_bytes")
            return 0 if stat is None else stat.peak

    def memory_reserved(self, device: int) -> int:
        with self._lock:
            stat = self._device_stat(device, "reserved_bytes")
            return 0 if stat is None else stat.current

    def max_memory_reserved(self, device: int) -> int:
        with self._lock:
            stat = self._device_stat(device, "reserved_bytes")
            return 0 if stat is None else stat.peak

    def mem_get_info(self, device: int) -> tuple[int, int]:
        with self._lock:
            if device < 0 or device >= len(self._total):
                return (0, 0)
            free = self._total[device] - self._reserved[device]
            return (max(0, free), self._total[device])

    def _check_fraction_device(self, device: int) -> None:
        if device < 0 or device >= len(self._total):
            raise RuntimeError(
                f"Allocator not initialized for device {device}: did you call init?"
            )

    def set_memory_fraction(self, fraction: float, device: int) -> None:
        """Mirror ``DeviceCachingAllocator::setMemoryFraction``."""
        with self._lock:
            self._check_fraction_device(device)
            if not 0 <= fraction <= 1:
                raise RuntimeError(
                    f"invalid fraction:{fraction}. Please set within [0, 1]."
                )
            self._allowed_memory_maximum[device] = (
                round(fraction * self._total[device]) if fraction < 1.0 else None
            )

    def get_memory_fraction(self, device: int) -> float:
        with self._lock:
            self._check_fraction_device(device)
            allowed = self._allowed_memory_maximum[device]
            if allowed is None:
                return 1.0
            return allowed / self._total[device]

    def reset_peak(self, device: int) -> None:
        with self._lock:
            if 0 <= device < len(self._peak):
                self._peak[device] = self._used[device]
                self._stats[device].reset_peak()
                self._peak_by_stage[device] = {}
                self._reserved_peak_by_stage[device] = {}

    def reset_accumulated(self, device: int) -> None:
        with self._lock:
            if 0 <= device < len(self._stats):
                self._stats[device].reset_accumulated()

    def empty_cache(self, device: int | None = None) -> None:
        with self._lock:
            devices = range(len(self._total)) if device is None else (int(device),)
            for index in devices:
                if 0 <= index < len(self._total):
                    # release_cached_blocks() synchronizes before freeing.
                    self._stats[index].num_sync_all_streams += 1
                    self._release_empty_segments(index)

    def peak_by_stage(self, device: int) -> dict[str, int]:
        with self._lock:
            if device < 0 or device >= len(self._peak_by_stage):
                return {}
            return dict(self._peak_by_stage[device])

    def reserved_peak_by_stage(self, device: int) -> dict[str, int]:
        with self._lock:
            if device < 0 or device >= len(self._reserved_peak_by_stage):
                return {}
            return dict(self._reserved_peak_by_stage[device])

    def largest_allocations(self, device: int, limit: int = 10) -> list[dict[str, Any]]:
        with self._lock:
            if device < 0 or device >= len(self._largest_allocations):
                return []
            return [
                _public_allocation_record(item)
                for item in self._largest_allocations[device][:limit]
            ]

    def current_bytes_by_category(self, device: int) -> dict[str, int]:
        with self._lock:
            if device < 0 or device >= len(self._used):
                return {}
            return dict(self._current_bytes_by_category[device])

    def mark_category(self, data_ptr: int, category: str) -> None:
        with self._lock:
            record = self._allocs.get(data_ptr)
            if record is not None:
                device = record["device"]
                normalized_category = str(category or "unknown")
                previous_category = str(
                    record.get("category") or "unknown"
                )
                if (
                    0 <= device < len(self._current_bytes_by_category)
                    and normalized_category != previous_category
                ):
                    categories = self._current_bytes_by_category[device]
                    nbytes = record["bytes"]
                    remaining = max(
                        0,
                        categories.get(previous_category, 0) - nbytes,
                    )
                    if remaining:
                        categories[previous_category] = remaining
                    else:
                        categories.pop(previous_category, None)
                    categories[normalized_category] = (
                        categories.get(normalized_category, 0) + nbytes
                    )
                record["category"] = category
                if 0 <= device < len(self._largest_allocations):
                    for item in self._largest_allocations[device]:
                        if item["_data_ptr"] == data_ptr:
                            item["category"] = category

    def snapshot(self, profiles: list[dict[str, Any]]) -> dict[str, Any]:
        with self._lock:
            allocator_model = (
                "cuda_caching_allocator.v1"
                if self._caching_allocator
                else "direct_segments.v1"
            )
            tracking_confidence = (
                "C3_torch_dispatch_lifetime"
                if _dispatch_tracking_stats.get("enabled")
                else "C2_torch_tensor_lifetime"
            )
            devices: list[dict[str, Any]] = []
            for index, prof in enumerate(profiles):
                if index >= len(self._total):
                    break
                total = int(self._total[index])
                peak = int(self._peak[index])
                current = int(self._used[index])
                reserved = int(self._reserved[index])
                reserved_peak = int(self._reserved_peak[index])
                conservative_peak = max(peak, reserved_peak)
                headroom = total - conservative_peak
                headroom_percent = (100.0 * headroom / total) if total > 0 else None
                devices.append(
                    {
                        "index": index,
                        "name": str(prof.get("name", "")),
                        "profile_id": str(prof.get("profile_id", "")),
                        "total_memory": total,
                        "current_memory": current,
                        "peak_memory": peak,
                        "current_reserved_memory": reserved,
                        "peak_reserved_memory": reserved_peak,
                        "inactive_split_bytes": int(
                            self._stats[index].total("inactive_split_bytes").current
                        ),
                        "segment_count": len(self._segments[index]),
                        "headroom_bytes": headroom,
                        "headroom_percent": (
                            round(headroom_percent, 3)
                            if headroom_percent is not None
                            else None
                        ),
                        "allocation_count": int(self._alloc_calls[index]),
                        "free_count": int(self._free_calls[index]),
                        "current_bytes_by_category": self.current_bytes_by_category(
                            index
                        ),
                        "peak_by_stage": self.peak_by_stage(index),
                        "reserved_peak_by_stage": self.reserved_peak_by_stage(index),
                        "largest_allocations": self.largest_allocations(index),
                        "allocator_model": allocator_model,
                        "tracking_confidence": tracking_confidence,
                    }
                )
            return {
                "tracking_confidence": tracking_confidence,
                "allocator_model": allocator_model,
                "dispatch_tracking": _dispatch_tracking_snapshot(),
                "devices": devices,
            }

    def memory_stats_as_nested_dict(self, device: int) -> dict[str, Any]:
        with self._lock:
            if device < 0 or device >= len(self._stats):
                return _DeviceStats().as_nested_dict()
            return self._stats[device].as_nested_dict()

    def memory_stats(self, device: int) -> OrderedDict[str, Any]:
        return _flatten_memory_stats(self.memory_stats_as_nested_dict(device))

    def allocator_snapshot(self) -> list[dict[str, Any]]:
        """Return segments in the layout of ``torch.cuda.memory_snapshot()``.

        FakeCUDA records no allocation history, so ``frames`` stay empty.
        """
        with self._lock:
            result: list[dict[str, Any]] = []
            for device, segments in enumerate(self._segments):
                for segment in segments:
                    blocks = []
                    allocated_size = 0
                    requested_size = 0
                    for block in segment["blocks"]:
                        data_ptr = block.get("data_ptr")
                        record = (
                            self._allocs.get(int(data_ptr))
                            if data_ptr is not None
                            else None
                        )
                        requested = int((record or {}).get("bytes", 0))
                        if data_ptr is not None:
                            allocated_size += int(block["size"])
                            requested_size += requested
                        blocks.append(
                            {
                                "address": int(segment["address"])
                                + int(block["offset"]),
                                "size": int(block["size"]),
                                "requested_size": requested,
                                "state": (
                                    "active_allocated"
                                    if data_ptr is not None
                                    else "inactive"
                                ),
                                "frames": [],
                            }
                        )
                    result.append(
                        {
                            "device": device,
                            "address": int(segment["address"]),
                            "segment_id": int(segment["id"]),
                            "total_size": int(segment["size"]),
                            "allocated_size": allocated_size,
                            "active_size": allocated_size,
                            "requested_size": requested_size,
                            "stream": 0,
                            "segment_type": str(segment["type"]),
                            "segment_pool_id": (0, 0),
                            "is_expandable": False,
                            "frames": [],
                            "blocks": blocks,
                        }
                    )
            return result

    def _allocate_allocator_block(
        self,
        *,
        data_ptr: int,
        block_size: int,
        device: int,
    ) -> dict[str, Any] | None:
        pool = _allocator_pool(block_size)
        selected = self._take_free_allocator_block(device, pool, block_size)
        if selected is None:
            segment_size, segment_type = _allocator_segment_size(block_size)
            segment_size = max(
                block_size,
                min(segment_size, self._total[device]),
            )
            reserved = self._stats[device].total("reserved_bytes").current
            if reserved + segment_size > self._total[device]:
                return None
            # alloc_block() refuses cudaMalloc beyond the fraction's cap.
            allowed = self._allowed_memory_maximum[device]
            if allowed is not None and reserved + segment_size > allowed:
                return None
            segment = {
                "id": self._next_segment_id,
                "address": self._next_segment_address,
                "device": device,
                "size": segment_size,
                "type": segment_type,
                "active_count": 0,
                "free_bytes": segment_size,
                "blocks": [
                    {
                        "offset": 0,
                        "size": segment_size,
                        "data_ptr": None,
                    }
                ],
            }
            self._next_segment_id += 1
            self._next_segment_address += (
                (segment_size + _ALLOCATOR_LARGE_ALIGNMENT_BYTES - 1)
                // _ALLOCATOR_LARGE_ALIGNMENT_BYTES
                * _ALLOCATOR_LARGE_ALIGNMENT_BYTES
            )
            self._segments[device].append(segment)
            self._segments_by_id[int(segment["id"])] = segment
            stats = self._stats[device]
            stats.change("segment", segment_type, 1)
            stats.change("reserved_bytes", segment_type, segment_size)
            stats.num_device_alloc += 1
            block = segment["blocks"][0]
        else:
            segment, block = selected

        block_index = next(
            index
            for index, candidate in enumerate(segment["blocks"])
            if candidate is block
        )
        inactive_before = _segment_inactive_split(segment)
        remainder = block["size"] - block_size
        split = _allocator_should_split(segment["type"], remainder)
        if not split:
            block_size = block["size"]
        allocated = {
            "offset": block["offset"],
            "size": block_size,
            "data_ptr": data_ptr,
            "segment_id": segment["id"],
            "pool": segment["type"],
        }
        replacement = [allocated]
        if split:
            free_block = {
                "offset": block["offset"] + block_size,
                "size": remainder,
                "data_ptr": None,
            }
            replacement.append(free_block)
        segment["blocks"][block_index : block_index + 1] = replacement
        if split:
            self._register_free_allocator_block(segment, free_block)
        segment["active_count"] += 1
        segment["free_bytes"] = max(0, segment["free_bytes"] - block_size)
        self._record_inactive_split_change(segment, inactive_before)
        self._allocation_blocks[data_ptr] = (device, segment["id"])
        return allocated

    def _record_inactive_split_change(
        self,
        segment: dict[str, Any],
        before: tuple[int, int],
    ) -> None:
        after = _segment_inactive_split(segment)
        stats = self._stats[segment["device"]]
        stats.change("inactive_split", segment["type"], after[0] - before[0])
        stats.change("inactive_split_bytes", segment["type"], after[1] - before[1])

    def _free_allocator_block(self, data_ptr: int, device: int) -> None:
        location = self._allocation_blocks.pop(data_ptr, None)
        if location is None:
            return
        _, segment_id = location
        segment = self._segments_by_id.get(segment_id)
        if segment is None or segment["device"] != device:
            return
        inactive_before = _segment_inactive_split(segment)
        for free_block in segment["blocks"]:
            if free_block["data_ptr"] is None:
                self._unregister_free_allocator_block(segment, free_block)
        for block in segment["blocks"]:
            if block["data_ptr"] == data_ptr:
                segment["active_count"] = max(0, segment["active_count"] - 1)
                segment["free_bytes"] = min(
                    segment["size"], segment["free_bytes"] + block["size"]
                )
                block["data_ptr"] = None
                block.pop("segment_id", None)
                block.pop("pool", None)
                break
        segment["blocks"] = _coalesce_allocator_blocks(segment["blocks"])
        for free_block in segment["blocks"]:
            if free_block["data_ptr"] is None:
                self._register_free_allocator_block(segment, free_block)
        self._record_inactive_split_change(segment, inactive_before)
        if not self._caching_allocator and all(
            block["data_ptr"] is None for block in segment["blocks"]
        ):
            self._remove_segment(device, segment_id)

    def _release_segment_stats(self, segment: dict[str, Any]) -> None:
        # release_block(): one cudaFree per released segment.
        stats = self._stats[segment["device"]]
        stats.change("segment", segment["type"], -1)
        stats.change("reserved_bytes", segment["type"], -segment["size"])
        stats.num_device_free += 1

    def _release_empty_segments(self, device: int) -> int:
        released = 0
        retained = []
        for segment in self._segments[device]:
            if all(block["data_ptr"] is None for block in segment["blocks"]):
                released += segment["size"]
                for block in segment["blocks"]:
                    self._unregister_free_allocator_block(segment, block)
                self._segments_by_id.pop(segment["id"], None)
                self._release_segment_stats(segment)
            else:
                retained.append(segment)
        if released:
            self._segments[device] = retained
        return released

    def _remove_segment(self, device: int, segment_id: int) -> None:
        retained = []
        for segment in self._segments[device]:
            if segment["id"] == segment_id:
                for block in segment["blocks"]:
                    if block["data_ptr"] is None:
                        self._unregister_free_allocator_block(segment, block)
                self._segments_by_id.pop(segment_id, None)
                self._release_segment_stats(segment)
            else:
                retained.append(segment)
        self._segments[device] = retained

    def _register_free_allocator_block(
        self,
        segment: dict[str, Any],
        block: dict[str, Any],
    ) -> None:
        pool_key = (segment["device"], segment["type"])
        size = block["size"]
        key = (segment["id"], block["offset"])
        size_buckets = self._free_block_keys.setdefault(pool_key, {})
        bucket = size_buckets.get(size)
        if bucket is None:
            bucket = []
            size_buckets[size] = bucket
            insort(self._free_block_sizes.setdefault(pool_key, []), size)
        insort(bucket, key)
        self._free_blocks_by_key[key] = block

    def _unregister_free_allocator_block(
        self,
        segment: dict[str, Any],
        block: dict[str, Any],
    ) -> None:
        pool_key = (segment["device"], segment["type"])
        size = block["size"]
        key = (segment["id"], block["offset"])
        self._free_blocks_by_key.pop(key, None)
        size_buckets = self._free_block_keys.get(pool_key, {})
        bucket = size_buckets.get(size)
        if not bucket:
            return
        key_index = bisect_left(bucket, key)
        if key_index < len(bucket) and bucket[key_index] == key:
            bucket.pop(key_index)
        if bucket:
            return
        size_buckets.pop(size, None)
        sizes = self._free_block_sizes.get(pool_key, [])
        size_index = bisect_left(sizes, size)
        if size_index < len(sizes) and sizes[size_index] == size:
            sizes.pop(size_index)

    def _take_free_allocator_block(
        self,
        device: int,
        pool: str,
        minimum_size: int,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        pool_key = (device, pool)
        sizes = self._free_block_sizes.get(pool_key, [])
        size_index = bisect_left(sizes, minimum_size)
        if size_index >= len(sizes):
            return None
        size = sizes[size_index]
        bucket = self._free_block_keys[pool_key][size]
        key = bucket.pop(0)
        if not bucket:
            self._free_block_keys[pool_key].pop(size, None)
            sizes.pop(size_index)
        block = self._free_blocks_by_key.pop(key)
        segment = self._segments_by_id[key[0]]
        return segment, block

    def _record_largest_allocation(
        self, data_ptr: int, device: int, nbytes: int, metadata: dict[str, Any]
    ) -> None:
        entries = self._largest_allocations[device]
        if len(entries) >= 10 and nbytes <= entries[-1]["bytes"]:
            return
        item = {
            "_data_ptr": int(data_ptr),
            "bytes": int(nbytes),
            "device": int(device),
            "dtype": metadata.get("dtype"),
            "shape": metadata.get("shape"),
            "stage": metadata.get("stage"),
            "category": metadata.get("category", "tensor"),
        }
        if metadata.get("source"):
            item["source"] = str(metadata["source"])
        if metadata.get("operator"):
            item["operator"] = str(metadata["operator"])
        if metadata.get("stack"):
            item["stack"] = metadata["stack"]
        position = next(
            (
                index
                for index, allocation in enumerate(entries)
                if nbytes > allocation["bytes"]
            ),
            len(entries),
        )
        entries.insert(position, item)
        del entries[10:]
