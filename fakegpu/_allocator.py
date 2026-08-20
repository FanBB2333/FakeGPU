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
from typing import Any


_ALLOCATOR_ALIGNMENT_BYTES = 512


_ALLOCATOR_SMALL_SEGMENT_BYTES = 2 * 1024**2


_ALLOCATOR_MEDIUM_SEGMENT_BYTES = 20 * 1024**2


_ALLOCATOR_LARGE_ALIGNMENT_BYTES = 2 * 1024**2


_ALLOCATOR_SMALL_REQUEST_LIMIT = 1024**2


_ALLOCATOR_MEDIUM_REQUEST_LIMIT = 10 * 1024**2


_dispatch_tracking_lock = threading.RLock()


_dispatch_tracking_stats: dict[str, Any] = {
    "enabled": False,
    "operator_calls": 0,
    "output_tensors": 0,
    "new_allocations": 0,
    "alias_outputs": 0,
    "inaccessible_outputs": 0,
    "operators": {},
}


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


def _coalesce_allocator_blocks(
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for block in blocks:
        current = dict(block)
        if (
            merged
            and merged[-1].get("data_ptr") is None
            and current.get("data_ptr") is None
            and int(merged[-1]["offset"]) + int(merged[-1]["size"])
            == int(current["offset"])
        ):
            merged[-1]["size"] = int(merged[-1]["size"]) + int(current["size"])
            continue
        merged.append(current)
    return merged


def _segment_inactive_split_bytes(segment: dict[str, Any]) -> int:
    if int(segment.get("active_count", 0)) <= 0:
        return 0
    return int(segment.get("free_bytes", 0))


def _public_allocation_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not str(key).startswith("_")}


def _build_memory_stats_dict(
    current: int,
    peak: int,
    *,
    active_current: int | None = None,
    active_peak: int | None = None,
    reserved_current: int | None = None,
    reserved_peak: int | None = None,
    allocated_total: int | None = None,
    freed_total: int = 0,
    reserved_total: int | None = None,
    reserved_freed_total: int = 0,
    inactive_split_current: int = 0,
    inactive_split_peak: int = 0,
    segment_current: int = 0,
    segment_peak: int = 0,
    allocation_current: int = 0,
    allocation_peak: int = 0,
    num_alloc_retries: int = 0,
    num_ooms: int = 0,
) -> dict[str, Any]:
    current_i = int(current)
    peak_i = int(max(current, peak))
    active_current_i = int(
        max(current_i, active_current if active_current is not None else current_i)
    )
    active_peak_i = int(
        max(
            active_current_i,
            active_peak if active_peak is not None else peak_i,
        )
    )
    reserved_current_i = int(
        max(
            active_current_i,
            reserved_current if reserved_current is not None else current_i,
        )
    )
    reserved_peak_i = int(
        max(
            reserved_current_i,
            reserved_peak if reserved_peak is not None else peak_i,
        )
    )
    allocated_total_i = int(
        max(current_i, allocated_total if allocated_total is not None else peak_i)
    )
    reserved_total_i = int(
        max(
            reserved_current_i,
            reserved_total if reserved_total is not None else reserved_peak_i,
        )
    )
    return {
        "active_bytes.all.current": active_current_i,
        "active_bytes.all.peak": active_peak_i,
        "active_bytes.all.allocated": allocated_total_i,
        "active_bytes.all.freed": int(freed_total),
        "allocated_bytes.all.current": current_i,
        "allocated_bytes.all.peak": peak_i,
        "allocated_bytes.all.allocated": allocated_total_i,
        "allocated_bytes.all.freed": int(freed_total),
        "requested_bytes.all.current": current_i,
        "requested_bytes.all.peak": peak_i,
        "requested_bytes.all.allocated": allocated_total_i,
        "requested_bytes.all.freed": int(freed_total),
        "reserved_bytes.all.current": reserved_current_i,
        "reserved_bytes.all.peak": reserved_peak_i,
        "reserved_bytes.all.allocated": reserved_total_i,
        "reserved_bytes.all.freed": int(reserved_freed_total),
        "inactive_split_bytes.all.current": int(inactive_split_current),
        "inactive_split_bytes.all.peak": int(inactive_split_peak),
        "segment.all.current": int(segment_current),
        "segment.all.peak": int(segment_peak),
        "allocation.all.current": int(allocation_current),
        "allocation.all.peak": int(max(allocation_current, allocation_peak)),
        "num_alloc_retries": int(num_alloc_retries),
        "num_ooms": int(num_ooms),
    }


class _DeviceMemoryTracker:
    """Track tensor bytes and a simplified CUDA caching allocator per device."""

    def __init__(
        self,
        per_device_bytes: list[int],
        *,
        caching_allocator: bool | None = None,
    ):
        self._total = list(per_device_bytes)
        self._used = [0] * len(per_device_bytes)
        self._peak = [0] * len(per_device_bytes)
        self._reserved = [0] * len(per_device_bytes)
        self._reserved_peak = [0] * len(per_device_bytes)
        self._active = [0] * len(per_device_bytes)
        self._active_peak = [0] * len(per_device_bytes)
        self._inactive_split = [0] * len(per_device_bytes)
        self._inactive_split_peak = [0] * len(per_device_bytes)
        self._segment_peak = [0] * len(per_device_bytes)
        self._allocation_current = [0] * len(per_device_bytes)
        self._allocation_peak = [0] * len(per_device_bytes)
        self._alloc_calls = [0] * len(per_device_bytes)
        self._free_calls = [0] * len(per_device_bytes)
        self._allocated_bytes_total = [0] * len(per_device_bytes)
        self._freed_bytes_total = [0] * len(per_device_bytes)
        self._reserved_bytes_total = [0] * len(per_device_bytes)
        self._released_reserved_bytes_total = [0] * len(per_device_bytes)
        self._num_alloc_retries = [0] * len(per_device_bytes)
        self._num_ooms = [0] * len(per_device_bytes)
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
        self._free_block_sizes: list[list[int]] = [
            [] for _ in per_device_bytes
        ]
        self._free_block_keys: list[
            dict[int, list[tuple[int, int]]]
        ] = [{} for _ in per_device_bytes]
        self._free_blocks_by_key: dict[
            tuple[int, int], dict[str, Any]
        ] = {}
        self._allocation_blocks: dict[int, tuple[int, int]] = {}
        self._next_segment_id = 1
        if caching_allocator is None:
            caching_allocator = os.environ.get("FAKEGPU_CACHING_ALLOCATOR", "1") != "0"
        self._caching_allocator = bool(caching_allocator)
        self._lock = threading.RLock()
        self._next_synthetic_data_ptr = -1
        self._synthetic_saved_raw_ptrs: dict[int, dict[str, int]] = {}
        self._held_saved_raw_ptrs: dict[int, int] = {}
        self._pending_saved_releases: set[int] = set()

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
            if block is None:
                released = self._release_empty_segments(device)
                if released:
                    self._num_alloc_retries[device] += 1
                    block = self._allocate_allocator_block(
                        data_ptr=data_ptr,
                        block_size=block_size,
                        device=device,
                    )
            if block is None:
                self._num_ooms[device] += 1
                free = max(0, self._total[device] - self._reserved[device])
                import torch

                raise torch.cuda.OutOfMemoryError(
                    f"CUDA out of memory. Tried to allocate "
                    f"{nbytes / 2**20:.2f} MiB. "
                    f"GPU {device} has a total capacity of "
                    f"{self._total[device] / 2**30:.2f} GiB "
                    f"of which {free / 2**30:.2f} GiB is free. "
                    f"FakeGPU has {self._used[device] / 2**30:.2f} GiB allocated "
                    f"and {self._reserved[device] / 2**30:.2f} GiB reserved."
                )
            meta = dict(metadata or {})
            stage = str(
                meta.get("stage")
                or os.environ.get("FAKEGPU_PREFLIGHT_STAGE")
                or "unknown"
            )
            meta["stage"] = stage

            self._allocs[data_ptr] = {
                "device": device,
                "bytes": nbytes,
                "block_bytes": block_size,
                "segment_id": int(block["segment_id"]),
                **meta,
            }
            category = str(meta.get("category") or "unknown")
            categories = self._current_bytes_by_category[device]
            categories[category] = categories.get(category, 0) + nbytes
            self._used[device] += nbytes
            self._active[device] += block_size
            self._allocation_current[device] += 1
            self._allocated_bytes_total[device] += nbytes
            self._peak[device] = max(self._peak[device], self._used[device])
            stage_peaks = self._peak_by_stage[device]
            stage_peaks[stage] = max(stage_peaks.get(stage, 0), self._used[device])
            reserved_stage_peaks = self._reserved_peak_by_stage[device]
            reserved_stage_peaks[stage] = max(
                reserved_stage_peaks.get(stage, 0),
                self._reserved[device],
            )
            self._alloc_calls[device] += 1
            self._update_allocator_peaks(device)
            self._record_largest_allocation(data_ptr, device, nbytes, meta)
            return True

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
            dev = int(rec.get("device", 0))
            nbytes = int(rec.get("bytes", 0))
            category = str(rec.get("category") or "unknown")
            categories = self._current_bytes_by_category[dev]
            remaining = max(0, categories.get(category, 0) - nbytes)
            if remaining:
                categories[category] = remaining
            else:
                categories.pop(category, None)
            self._used[dev] = max(0, self._used[dev] - nbytes)
            self._active[dev] = max(
                0,
                self._active[dev] - int(rec.get("block_bytes", nbytes)),
            )
            self._allocation_current[dev] = max(
                0,
                self._allocation_current[dev] - 1,
            )
            self._freed_bytes_total[dev] += nbytes
            self._free_calls[dev] += 1
            self._free_allocator_block(data_ptr, dev)
            self._update_allocator_peaks(dev)

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

    def memory_allocated(self, device: int) -> int:
        with self._lock:
            if device < 0 or device >= len(self._used):
                return 0
            return self._used[device]

    def max_memory_allocated(self, device: int) -> int:
        with self._lock:
            if device < 0 or device >= len(self._peak):
                return 0
            return self._peak[device]

    def memory_reserved(self, device: int) -> int:
        with self._lock:
            if device < 0 or device >= len(self._reserved):
                return 0
            return self._reserved[device]

    def max_memory_reserved(self, device: int) -> int:
        with self._lock:
            if device < 0 or device >= len(self._reserved_peak):
                return 0
            return self._reserved_peak[device]

    def mem_get_info(self, device: int) -> tuple[int, int]:
        with self._lock:
            if device < 0 or device >= len(self._total):
                return (0, 0)
            free = self._total[device] - self._reserved[device]
            return (max(0, free), self._total[device])

    def reset_peak(self, device: int) -> None:
        with self._lock:
            if 0 <= device < len(self._peak):
                self._peak[device] = self._used[device]
                self._reserved_peak[device] = self._reserved[device]
                self._active_peak[device] = self._active[device]
                self._inactive_split_peak[device] = self._inactive_split[device]
                self._segment_peak[device] = len(self._segments[device])
                self._allocation_peak[device] = self._allocation_current[device]
                self._peak_by_stage[device] = {}
                self._reserved_peak_by_stage[device] = {}

    def reset_accumulated(self, device: int) -> None:
        with self._lock:
            if device < 0 or device >= len(self._total):
                return
            self._allocated_bytes_total[device] = self._used[device]
            self._freed_bytes_total[device] = 0
            self._reserved_bytes_total[device] = self._reserved[device]
            self._released_reserved_bytes_total[device] = 0
            self._num_alloc_retries[device] = 0
            self._num_ooms[device] = 0

    def empty_cache(self, device: int | None = None) -> None:
        with self._lock:
            devices = range(len(self._total)) if device is None else (int(device),)
            for index in devices:
                if 0 <= index < len(self._total):
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
                device = int(record.get("device", -1))
                normalized_category = str(category or "unknown")
                previous_category = str(
                    record.get("category") or "unknown"
                )
                if (
                    0 <= device < len(self._current_bytes_by_category)
                    and normalized_category != previous_category
                ):
                    categories = self._current_bytes_by_category[device]
                    nbytes = int(record.get("bytes", 0))
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
                        if int(item.get("_data_ptr", -1)) == data_ptr:
                            item["category"] = category

    def snapshot(self, profiles: list[dict[str, Any]]) -> dict[str, Any]:
        with self._lock:
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
                        "inactive_split_bytes": self._inactive_split_bytes(index),
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
                        "allocator_model": (
                            "cuda_caching_allocator.v1"
                            if self._caching_allocator
                            else "direct_segments.v1"
                        ),
                        "tracking_confidence": (
                            "C3_torch_dispatch_lifetime"
                            if _dispatch_tracking_stats.get("enabled")
                            else "C2_torch_tensor_lifetime"
                        ),
                    }
                )
            return {
                "tracking_confidence": (
                    "C3_torch_dispatch_lifetime"
                    if _dispatch_tracking_stats.get("enabled")
                    else "C2_torch_tensor_lifetime"
                ),
                "allocator_model": (
                    "cuda_caching_allocator.v1"
                    if self._caching_allocator
                    else "direct_segments.v1"
                ),
                "dispatch_tracking": _dispatch_tracking_snapshot(),
                "devices": devices,
            }

    def memory_stats(self, device: int) -> dict[str, Any]:
        with self._lock:
            if device < 0 or device >= len(self._total):
                return _build_memory_stats_dict(0, 0)
            active_bytes = self._active[device]
            inactive_split = self._inactive_split[device]
            return _build_memory_stats_dict(
                self._used[device],
                self._peak[device],
                active_current=active_bytes,
                active_peak=self._active_peak[device],
                reserved_current=self._reserved[device],
                reserved_peak=self._reserved_peak[device],
                allocated_total=self._allocated_bytes_total[device],
                freed_total=self._freed_bytes_total[device],
                reserved_total=self._reserved_bytes_total[device],
                reserved_freed_total=self._released_reserved_bytes_total[device],
                inactive_split_current=inactive_split,
                inactive_split_peak=self._inactive_split_peak[device],
                segment_current=len(self._segments[device]),
                segment_peak=self._segment_peak[device],
                allocation_current=self._allocation_current[device],
                allocation_peak=self._allocation_peak[device],
                num_alloc_retries=self._num_alloc_retries[device],
                num_ooms=self._num_ooms[device],
            )

    def allocator_snapshot(self) -> list[dict[str, Any]]:
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
                                "address": int(block["offset"]),
                                "size": int(block["size"]),
                                "requested_size": requested,
                                "state": (
                                    "active_allocated"
                                    if data_ptr is not None
                                    else "inactive"
                                ),
                            }
                        )
                    result.append(
                        {
                            "device": device,
                            "segment_id": int(segment["id"]),
                            "total_size": int(segment["size"]),
                            "allocated_size": allocated_size,
                            "active_size": allocated_size,
                            "requested_size": requested_size,
                            "segment_type": str(segment["type"]),
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
        selected = self._take_free_allocator_block(device, block_size)
        if selected is None:
            segment_size, segment_type = _allocator_segment_size(block_size)
            segment_size = max(
                block_size,
                min(segment_size, self._total[device]),
            )
            if self._reserved[device] + segment_size > self._total[device]:
                return None
            segment = {
                "id": self._next_segment_id,
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
            self._segments[device].append(segment)
            self._segments_by_id[int(segment["id"])] = segment
            self._reserved[device] += segment_size
            self._reserved_bytes_total[device] += segment_size
            block = segment["blocks"][0]
        else:
            segment, block = selected

        block_index = next(
            index
            for index, candidate in enumerate(segment["blocks"])
            if candidate is block
        )
        inactive_before = _segment_inactive_split_bytes(segment)
        remainder = int(block["size"]) - block_size
        allocated = {
            "offset": int(block["offset"]),
            "size": block_size,
            "data_ptr": data_ptr,
            "segment_id": int(segment["id"]),
        }
        replacement = [allocated]
        if remainder > 0:
            free_block = {
                "offset": int(block["offset"]) + block_size,
                "size": remainder,
                "data_ptr": None,
            }
            replacement.append(free_block)
        segment["blocks"][block_index : block_index + 1] = replacement
        if remainder > 0:
            self._register_free_allocator_block(segment, free_block)
        segment["active_count"] = int(segment.get("active_count", 0)) + 1
        segment["free_bytes"] = max(
            0,
            int(segment.get("free_bytes", segment["size"])) - block_size,
        )
        self._inactive_split[device] = max(
            0,
            self._inactive_split[device]
            + _segment_inactive_split_bytes(segment)
            - inactive_before,
        )
        self._allocation_blocks[data_ptr] = (device, int(segment["id"]))
        return allocated

    def _free_allocator_block(self, data_ptr: int, device: int) -> None:
        location = self._allocation_blocks.pop(data_ptr, None)
        if location is None:
            return
        _, segment_id = location
        segment = self._segments_by_id.get(segment_id)
        if segment is None or int(segment.get("device", -1)) != device:
            return
        inactive_before = _segment_inactive_split_bytes(segment)
        for free_block in segment["blocks"]:
            if free_block.get("data_ptr") is None:
                self._unregister_free_allocator_block(segment, free_block)
        for block in segment["blocks"]:
            if block.get("data_ptr") == data_ptr:
                segment["active_count"] = max(
                    0,
                    int(segment.get("active_count", 1)) - 1,
                )
                segment["free_bytes"] = min(
                    int(segment["size"]),
                    int(segment.get("free_bytes", 0)) + int(block["size"]),
                )
                block["data_ptr"] = None
                block.pop("segment_id", None)
                break
        segment["blocks"] = _coalesce_allocator_blocks(segment["blocks"])
        for free_block in segment["blocks"]:
            if free_block.get("data_ptr") is None:
                self._register_free_allocator_block(segment, free_block)
        self._inactive_split[device] = max(
            0,
            self._inactive_split[device]
            + _segment_inactive_split_bytes(segment)
            - inactive_before,
        )
        if not self._caching_allocator and all(
            block.get("data_ptr") is None for block in segment["blocks"]
        ):
            self._remove_segment(device, segment_id)

    def _release_empty_segments(self, device: int) -> int:
        released = 0
        retained = []
        for segment in self._segments[device]:
            if all(block.get("data_ptr") is None for block in segment["blocks"]):
                released += int(segment["size"])
                for block in segment["blocks"]:
                    self._unregister_free_allocator_block(segment, block)
                self._segments_by_id.pop(int(segment["id"]), None)
            else:
                retained.append(segment)
        if released:
            self._segments[device] = retained
            self._reserved[device] = max(0, self._reserved[device] - released)
            self._released_reserved_bytes_total[device] += released
        return released

    def _remove_segment(self, device: int, segment_id: int) -> None:
        retained = []
        released = 0
        for segment in self._segments[device]:
            if int(segment["id"]) == segment_id:
                released += int(segment["size"])
                for block in segment["blocks"]:
                    if block.get("data_ptr") is None:
                        self._unregister_free_allocator_block(segment, block)
                self._segments_by_id.pop(segment_id, None)
            else:
                retained.append(segment)
        self._segments[device] = retained
        if released:
            self._reserved[device] = max(0, self._reserved[device] - released)
            self._released_reserved_bytes_total[device] += released

    def _register_free_allocator_block(
        self,
        segment: dict[str, Any],
        block: dict[str, Any],
    ) -> None:
        device = int(segment["device"])
        size = int(block["size"])
        key = (int(segment["id"]), int(block["offset"]))
        size_buckets = self._free_block_keys[device]
        bucket = size_buckets.get(size)
        if bucket is None:
            bucket = []
            size_buckets[size] = bucket
            insort(self._free_block_sizes[device], size)
        insort(bucket, key)
        self._free_blocks_by_key[key] = block

    def _unregister_free_allocator_block(
        self,
        segment: dict[str, Any],
        block: dict[str, Any],
    ) -> None:
        device = int(segment["device"])
        size = int(block["size"])
        key = (int(segment["id"]), int(block["offset"]))
        self._free_blocks_by_key.pop(key, None)
        size_buckets = self._free_block_keys[device]
        bucket = size_buckets.get(size)
        if not bucket:
            return
        key_index = bisect_left(bucket, key)
        if key_index < len(bucket) and bucket[key_index] == key:
            bucket.pop(key_index)
        if bucket:
            return
        size_buckets.pop(size, None)
        sizes = self._free_block_sizes[device]
        size_index = bisect_left(sizes, size)
        if size_index < len(sizes) and sizes[size_index] == size:
            sizes.pop(size_index)

    def _take_free_allocator_block(
        self,
        device: int,
        minimum_size: int,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        sizes = self._free_block_sizes[device]
        size_index = bisect_left(sizes, minimum_size)
        if size_index >= len(sizes):
            return None
        size = sizes[size_index]
        bucket = self._free_block_keys[device][size]
        key = bucket.pop(0)
        if not bucket:
            self._free_block_keys[device].pop(size, None)
            sizes.pop(size_index)
        block = self._free_blocks_by_key.pop(key)
        segment = self._segments_by_id[key[0]]
        return segment, block

    def _active_block_bytes(self, device: int) -> int:
        return self._active[device]

    def _inactive_split_bytes(self, device: int) -> int:
        return self._inactive_split[device]

    def _update_allocator_peaks(self, device: int) -> None:
        self._active_peak[device] = max(
            self._active_peak[device],
            self._active[device],
        )
        self._reserved_peak[device] = max(
            self._reserved_peak[device],
            self._reserved[device],
        )
        self._inactive_split_peak[device] = max(
            self._inactive_split_peak[device],
            self._inactive_split[device],
        )
        self._segment_peak[device] = max(
            self._segment_peak[device],
            len(self._segments[device]),
        )
        self._allocation_peak[device] = max(
            self._allocation_peak[device],
            self._allocation_current[device],
        )

    def _record_largest_allocation(
        self, data_ptr: int, device: int, nbytes: int, metadata: dict[str, Any]
    ) -> None:
        entries = self._largest_allocations[device]
        if len(entries) >= 10 and nbytes <= int(entries[-1].get("bytes", 0)):
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
                if nbytes > int(allocation.get("bytes", 0))
            ),
            len(entries),
        )
        entries.insert(position, item)
        del entries[10:]
