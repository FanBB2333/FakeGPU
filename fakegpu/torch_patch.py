"""Monkeypatch ``torch.cuda`` so CUDA-dependent code runs on CPU.

On systems without an NVIDIA GPU (or with a CPU-only PyTorch build), this
module transparently provides CUDA-visible tensor semantics backed by CPU.
It uses a **two-layer architecture**:

1. **Base layer**: the vendored upstream ``FakeCudaTensor`` backend
   (``fakegpu/_upstream.py``, from `pytorch-fakegpu`_ by FanBB2333).  Uses
   ``torch.Tensor._make_subclass`` + ``__torch_function__`` so that
   ``tensor.device`` reports ``cuda:N`` and ``tensor.is_cuda`` returns ``True``.

2. **Enhancement layer**: FakeGPU additions applied on top — GPU profiles,
   per-device memory tracking with OOM simulation, autocast dtype validation,
   cross-device operation guards, and terminal summary reporting.

The base layer is always available because ``fakegpu._upstream`` ships with
this package; ``patch()`` raises if it cannot be activated.

.. _pytorch-fakegpu: https://github.com/FanBB2333/pytorch-fakegpu

Verified PyTorch versions: **torch 2.6.0 -- 2.11.0** (all pass 30/30 validation steps).

Usage::

    import fakegpu
    fakegpu.init(runtime="fakecuda")
    # or: fakegpu.patch_torch()

    import torch
    # Everything below "just works" on CPU.
    x = torch.randn(3, 3, device="cuda")
    assert x.device.type == "cuda"
    assert x.is_cuda is True
    model = torch.nn.Linear(3, 3).cuda()
    y = model(x)
"""

from __future__ import annotations

import atexit
import dataclasses
import functools
import importlib
import os
import sys
import threading
import traceback
import types
import weakref
import warnings
from bisect import bisect_left, insort
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from .profile_catalog import architecture_for_compute_capability, load_profiles

_patched = False
_patch_result: "PatchResult | None" = None
_upstream_mod: Any = None  # Set when upstream FakeCudaTensor backend is active

# ---------------------------------------------------------------------------
# Configuration – mirrors the active FakeGPU profile when available.
# ---------------------------------------------------------------------------

_PROFILE_CATALOG = load_profiles()
_PROFILE_CC: dict[str, tuple[int, int]] = {
    profile_id: profile.compute_capability
    for profile_id, profile in _PROFILE_CATALOG.items()
}
_PROFILE_NAMES: dict[str, str] = {
    profile_id: profile.torch_name for profile_id, profile in _PROFILE_CATALOG.items()
}
_PROFILE_TOTAL_MEMORY: dict[str, int] = {
    profile_id: profile.memory_bytes for profile_id, profile in _PROFILE_CATALOG.items()
}
_PROFILE_SUPPORTED_TYPES: dict[str, tuple[str, ...]] = {
    profile_id: profile.supported_types
    for profile_id, profile in _PROFILE_CATALOG.items()
}


def _resolve_profile_id() -> str | None:
    profiles_env = os.environ.get("FAKEGPU_PROFILES", "")
    if profiles_env:
        first_spec = profiles_env.split(",")[0].strip()
        return first_spec.split(":")[0].strip().lower()

    profile_env = os.environ.get("FAKEGPU_PROFILE", "")
    if profile_env:
        return profile_env.strip().lower()

    device_name = os.environ.get("FAKEGPU_DEVICE_NAME", "").strip().lower()
    if device_name:
        reverse_names = {value.lower(): key for key, value in _PROFILE_NAMES.items()}
        return reverse_names.get(device_name)

    return None


def _resolve_compute_capability() -> tuple[int, int]:
    profile_id = _resolve_profile_id()
    if profile_id and profile_id in _PROFILE_CC:
        return _PROFILE_CC[profile_id]
    return (8, 0)


def _resolve_device_name() -> str:
    name = os.environ.get("FAKEGPU_DEVICE_NAME", "")
    if name:
        return name
    profile_id = _resolve_profile_id()
    if profile_id:
        return _PROFILE_NAMES.get(profile_id, "NVIDIA A100-SXM4-80GB")
    return "NVIDIA A100-SXM4-80GB"


def _resolve_total_memory() -> int:
    profile_id = _resolve_profile_id()
    if profile_id:
        return _PROFILE_TOTAL_MEMORY.get(profile_id, 80 * 1024**3)
    return 80 * 1024**3


def _resolve_per_device_profiles(
    num_devices: int | None = None,
) -> list[dict[str, Any]]:
    """Resolve per-device profile info from FAKEGPU_PROFILES.

    Returns a list of dicts, one per device, each with keys:
      'profile_id', 'name', 'total_memory', 'compute_major', 'compute_minor'
    """
    profiles_env = os.environ.get("FAKEGPU_PROFILES", "")
    target_count = int(
        num_devices
        if num_devices is not None
        else os.environ.get("FAKEGPU_DEVICE_COUNT", "8")
    )
    result: list[dict[str, Any]] = []

    if profiles_env:
        for spec in profiles_env.split(","):
            spec = spec.strip()
            if not spec:
                continue
            parts = spec.split(":")
            pid = parts[0].strip().lower()
            count = (
                int(parts[1]) if len(parts) > 1 and parts[1].strip().isdigit() else 1
            )
            for _ in range(count):
                cc = _PROFILE_CC.get(pid, (8, 0))
                result.append(
                    {
                        "profile_id": pid,
                        "name": _PROFILE_NAMES.get(pid, "NVIDIA A100-SXM4-80GB"),
                        "total_memory": _PROFILE_TOTAL_MEMORY.get(pid, 80 * 1024**3),
                        "compute_major": cc[0],
                        "compute_minor": cc[1],
                    }
                )

    if not result:
        # Uniform config: all devices share the same profile
        pid = _resolve_profile_id() or "a100"
        cc = _PROFILE_CC.get(pid, (8, 0))
        entry = {
            "profile_id": pid,
            "name": _PROFILE_NAMES.get(pid, "NVIDIA A100-SXM4-80GB"),
            "total_memory": _PROFILE_TOTAL_MEMORY.get(pid, 80 * 1024**3),
            "compute_major": cc[0],
            "compute_minor": cc[1],
        }
        for _ in range(target_count):
            result.append(dict(entry))

    if len(result) != target_count and len(result) > 0:
        while len(result) < target_count:
            result.append(dict(result[-1]))
        result = result[:target_count]

    return result


_NUM_DEVICES = int(os.environ.get("FAKEGPU_DEVICE_COUNT", "8"))
_DEVICE_PROFILES: list[dict[str, Any]] = _resolve_per_device_profiles(_NUM_DEVICES)

_DEVICE_NAME = _resolve_device_name()
_COMPUTE_MAJOR, _COMPUTE_MINOR = _resolve_compute_capability()
_TOTAL_MEMORY = _resolve_total_memory()


def _refresh_runtime_profile_state(
    *, num_devices: int | None = None, device_name: str | None = None
) -> None:
    """Refresh per-device profile globals after runtime options change."""
    global \
        _NUM_DEVICES, \
        _DEVICE_PROFILES, \
        _DEVICE_NAME, \
        _COMPUTE_MAJOR, \
        _COMPUTE_MINOR, \
        _TOTAL_MEMORY

    if num_devices is None:
        num_devices = int(os.environ.get("FAKEGPU_DEVICE_COUNT", str(_NUM_DEVICES)))
    _NUM_DEVICES = int(num_devices)
    os.environ["FAKEGPU_DEVICE_COUNT"] = str(_NUM_DEVICES)

    _DEVICE_PROFILES = _resolve_per_device_profiles(_NUM_DEVICES)

    if _DEVICE_PROFILES:
        first = _DEVICE_PROFILES[0]
        _COMPUTE_MAJOR = int(first.get("compute_major", 8))
        _COMPUTE_MINOR = int(first.get("compute_minor", 0))
        _TOTAL_MEMORY = int(first.get("total_memory", 80 * 1024**3))
        if device_name is None:
            _DEVICE_NAME = str(first.get("name", _resolve_device_name()))
        else:
            _DEVICE_NAME = device_name
    elif device_name is not None:
        _DEVICE_NAME = device_name


# ---------------------------------------------------------------------------
# Cross-device guards
# ---------------------------------------------------------------------------

_CROSS_DEVICE_CHECK = os.environ.get("FAKEGPU_CROSS_DEVICE_CHECK", "1") != "0"


def _is_fake_tensor(tensor: Any) -> bool:
    """Return whether a tensor is owned by PyTorch FakeTensorMode."""
    return getattr(tensor, "fake_mode", None) is not None or (
        type(tensor).__module__ == "torch._subclasses.fake_tensor"
        and type(tensor).__name__ == "FakeTensor"
    )


def _check_same_device(*tensors: Any) -> None:
    """Raise RuntimeError if tensors span multiple fake CUDA devices."""
    if not _CROSS_DEVICE_CHECK:
        return

    first_dev: int | None = None
    for t in tensors:
        # FakeCudaTensor carries its device index as an attribute.
        dev = getattr(t, "device_index", None)
        if dev is None:
            continue  # untracked tensor (e.g. pure CPU) — skip
        if first_dev is None:
            first_dev = dev
        elif dev != first_dev:
            raise RuntimeError(
                f"Expected all tensors to be on the same device, "
                f"but found at least two devices, cuda:{first_dev} and cuda:{dev}!"
            )


def _wrap_multi_tensor_op(orig_fn: Any, torch_mod: Any) -> Any:
    """Wrap a torch function to check device consistency of tensor args."""

    @functools.wraps(orig_fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        tensors = []
        for a in args:
            if isinstance(a, torch_mod.Tensor):
                tensors.append(a)
            elif isinstance(a, (list, tuple)):
                for item in a:
                    if isinstance(item, torch_mod.Tensor):
                        tensors.append(item)
        for v in kwargs.values():
            if isinstance(v, torch_mod.Tensor):
                tensors.append(v)
        if len(tensors) >= 2:
            _check_same_device(*tensors)
        return orig_fn(*args, **kwargs)

    return wrapper


_BINARY_DUNDER_TORCH_OPS: dict[str, Any] = {
    "__add__": lambda torch_mod, self, other: torch_mod.add(self, other),
    "__radd__": lambda torch_mod, self, other: torch_mod.add(other, self),
    "__sub__": lambda torch_mod, self, other: torch_mod.sub(self, other),
    "__rsub__": lambda torch_mod, self, other: torch_mod.sub(other, self),
    "__mul__": lambda torch_mod, self, other: torch_mod.mul(self, other),
    "__rmul__": lambda torch_mod, self, other: torch_mod.mul(other, self),
    "__truediv__": lambda torch_mod, self, other: torch_mod.true_divide(self, other),
    "__rtruediv__": lambda torch_mod, self, other: torch_mod.true_divide(other, self),
    "__matmul__": lambda torch_mod, self, other: torch_mod.matmul(self, other),
    "__rmatmul__": lambda torch_mod, self, other: torch_mod.matmul(other, self),
}


def _wrap_tensor_binary_op(
    orig_fn: Any,
    dunder_name: str,
    torch_mod: Any,
) -> Any:
    """Wrap a Tensor binary method to check cross-device with torch-friendly calls."""

    @functools.wraps(orig_fn)
    def wrapper(self: Any, other: Any) -> Any:
        if isinstance(other, torch_mod.Tensor):
            _check_same_device(self, other)
        torch_op = _BINARY_DUNDER_TORCH_OPS.get(dunder_name)
        if torch_op is not None:
            return torch_op(torch_mod, self, other)
        return orig_fn(self, other)

    return wrapper
# ---------------------------------------------------------------------------
# Memory tracking: per-device memory accounting.
# ---------------------------------------------------------------------------

_MEMORY_TRACKING = os.environ.get("FAKEGPU_MEMORY_TRACKING", "1") != "0"
_DISPATCH_MEMORY_TRACKING = (
    os.environ.get("FAKEGPU_DISPATCH_MEMORY_TRACKING", "1") != "0"
)
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


# ---------------------------------------------------------------------------
# Architecture name lookup (mirrors C++ gpu_profile.cpp)
# ---------------------------------------------------------------------------


def _arch_name(major: int, minor: int) -> str:
    """Return the architecture name for a compute capability."""
    architecture = architecture_for_compute_capability(major, minor)
    return architecture.title() if architecture != "unknown" else "Unknown"


# ---------------------------------------------------------------------------
# Terminal Report Summary (atexit handler, mirrors C++ monitor.cpp)
# ---------------------------------------------------------------------------


def _fmt_bytes(b: int) -> str:
    if b >= 1024**3:
        return f"{b / 1024**3:.1f} GB"
    elif b >= 1024**2:
        return f"{b / 1024**2:.1f} MB"
    elif b >= 1024:
        return f"{b / 1024:.1f} KB"
    return f"{b} B"


def _dump_terminal_summary() -> None:
    """Print a Report Summary to stderr on process exit.

    Controlled by ``FAKEGPU_TERMINAL_REPORT`` (default: enabled).
    """
    if os.environ.get("FAKEGPU_TERMINAL_REPORT", "1") == "0":
        return
    tracker = _memory_tracker
    if tracker is None:
        return

    lines: list[str] = []
    lines.append("")
    lines.append("======================================================")
    lines.append("             FakeGPU Report Summary")
    lines.append("======================================================")

    for i, prof in enumerate(_DEVICE_PROFILES):
        if i >= len(tracker._total):
            break
        name = prof.get("name", "NVIDIA A100-SXM4-80GB")
        cc_major = prof.get("compute_major", 8)
        cc_minor = prof.get("compute_minor", 0)
        arch = _arch_name(cc_major, cc_minor)

        total = tracker._total[i]
        peak = tracker._peak[i]
        reserved_peak = tracker._reserved_peak[i]
        peak_pct = (100.0 * reserved_peak / total) if total > 0 else 0.0

        alloc = tracker._alloc_calls[i]
        free = tracker._free_calls[i]

        lines.append(f" Device {i}: {name} ({arch}, cc {cc_major}.{cc_minor})")
        lines.append(
            f"   Memory: {_fmt_bytes(peak)} allocated | "
            f"{_fmt_bytes(reserved_peak)} reserved / {_fmt_bytes(total)} "
            f"({peak_pct:.1f}%)"
        )
        lines.append(f"   Alloc: {alloc} calls | Free: {free} calls")
        lines.append("------------------------------------------------------")

    lines.append(" Peak VRAM by GPU:")
    for i, peak in enumerate(tracker._peak[: len(_DEVICE_PROFILES)]):
        lines.append(
            f"   GPU {i}: {_fmt_bytes(peak)} allocated | "
            f"{_fmt_bytes(tracker._reserved_peak[i])} reserved"
        )
    lines.append("------------------------------------------------------")

    lines.append("======================================================")
    lines.append("")

    sys.stderr.write("\n".join(lines))
    sys.stderr.flush()


# Initialized later in patch() after _DEVICE_PROFILES is finalized
_memory_tracker: _DeviceMemoryTracker | None = None
_smi_publisher: Any = None


def _smi_memory_snapshot() -> dict[str, Any]:
    tracker = _memory_tracker
    if tracker is None:
        return {
            "tracking_confidence": "C0_incomplete",
            "stage": os.environ.get("FAKEGPU_PREFLIGHT_STAGE") or "unknown",
            "runtime_backend": (
                _patch_result.backend
                if _patch_result is not None
                else "unknown"
            ),
            "memory_tracking_enabled": False,
            "devices": [],
        }
    snapshot = tracker.snapshot(_DEVICE_PROFILES)
    snapshot.update(
        {
            "stage": (
                os.environ.get("FAKEGPU_PREFLIGHT_STAGE")
                or "unknown"
            ),
            "runtime_backend": (
                _patch_result.backend
                if _patch_result is not None
                else "unknown"
            ),
            "memory_tracking_enabled": bool(_MEMORY_TRACKING),
        }
    )
    return snapshot


def _refresh_smi_publisher() -> None:
    global _smi_publisher
    if _smi_publisher is not None:
        _smi_publisher.stop()
        _smi_publisher = None
    from .smi import SmiStatePublisher, configured_state_path

    path = configured_state_path()
    if path is None or _memory_tracker is None:
        return
    try:
        interval_ms = float(os.environ.get("FAKEGPU_SMI_INTERVAL_MS", "250"))
    except ValueError:
        interval_ms = 250.0
    try:
        overhead = int(os.environ.get("FAKEGPU_SMI_RUNTIME_OVERHEAD_BYTES", "0"))
    except ValueError:
        overhead = 0
    _smi_publisher = SmiStatePublisher(
        path,
        _smi_memory_snapshot,
        interval_seconds=max(50.0, interval_ms) / 1000.0,
        runtime_overhead_bytes=max(0, overhead),
    )
    _smi_publisher.start()


def _set_tracked_data_ptr(tensor: Any, data_ptr: int) -> None:
    try:
        setattr(tensor, "_fakegpu_memory_data_ptr", int(data_ptr))
    except Exception:
        pass


def _register_tensor_for_memory_tracking(
    tensor: Any,
    device_index: int,
    *,
    metadata: dict[str, Any] | None = None,
    storage_info: tuple[Any, int, int] | None = None,
) -> bool:
    """Register a tensor's memory and set up GC cleanup via weakref."""
    if _memory_tracker is None or not _MEMORY_TRACKING:
        return False
    if _is_fake_tensor(tensor):
        return False

    # Functional transforms such as vmap expose BatchedTensorImpl objects
    # whose storage is intentionally inaccessible.  Meta tensors and some
    # external tensor subclasses have the same property.  They do not own a
    # distinct allocation that FakeGPU can track, so skip them without
    # suppressing errors raised later by the actual tracker (notably OOM).
    try:
        if storage_info is None:
            storage = tensor.untyped_storage()
            dp = int(storage.data_ptr())
            nbytes = int(storage.nbytes())
        else:
            storage, dp, nbytes = storage_info
    except Exception:
        # Inaccessible storage (vmap batched tensors, meta tensors, some
        # subclasses): nothing distinct to track, so skip without suppressing
        # errors raised later by the actual tracker (notably OOM).
        return False

    if dp == 0 or nbytes == 0:
        return False

    try:
        _set_tracked_data_ptr(tensor, int(dp))
        if _memory_tracker.has_allocation(int(dp)):
            return False
        allocation_metadata = _tensor_allocation_metadata(tensor)
        allocation_metadata.update(metadata or {})
        did_allocate = _memory_tracker.allocate(
            dp,
            nbytes,
            device_index,
            metadata=allocation_metadata,
        )
        if not did_allocate:
            return False

        # Set up weakref callback to release memory when tensor is GC'd.
        # We weakref the storage, not the tensor, because multiple tensor
        # views can share one storage.  The callback is bound to the tracker
        # that owns this allocation: a later patch() call installs a fresh
        # _DeviceMemoryTracker, and stale finalizers must not release into it.
        # Only add weakref if not already tracked (avoid double-counting)
        weakref.finalize(storage, _memory_tracker.release, int(dp))
        return True
    except (MemoryError, RuntimeError):
        raise  # Preserve simulated OOM and other tracker errors.
    except Exception:
        return False


def _tensor_allocation_metadata(tensor: Any) -> dict[str, Any]:
    raw = getattr(tensor, "raw_data", tensor)
    shape = _safe_tensor_shape(raw)
    dtype = getattr(raw, "dtype", getattr(tensor, "dtype", None))
    metadata = {
        "dtype": str(dtype) if dtype is not None else None,
        "shape": shape,
        "stage": os.environ.get("FAKEGPU_PREFLIGHT_STAGE") or "unknown",
        "category": _infer_tensor_memory_category(tensor, raw),
    }
    stack = _capture_allocation_stack_trace()
    if stack:
        metadata["stack"] = stack
    return metadata


def _capture_allocation_stack_trace() -> list[dict[str, Any]]:
    if not _truthy_env("FAKEGPU_ALLOCATION_STACKS"):
        return []

    try:
        depth = int(os.environ.get("FAKEGPU_ALLOCATION_STACK_DEPTH", "8"))
    except ValueError:
        depth = 8
    depth = max(1, min(depth, 32))

    frames = traceback.extract_stack()[:-2]
    public_frames: list[dict[str, Any]] = []
    for frame in frames:
        if _is_internal_allocation_frame(frame.filename):
            continue
        item: dict[str, Any] = {
            "file": frame.filename,
            "line": int(frame.lineno),
            "function": frame.name,
        }
        if frame.line:
            item["code"] = frame.line.strip()
        public_frames.append(item)

    if not public_frames:
        for frame in frames[-depth:]:
            item = {
                "file": frame.filename,
                "line": int(frame.lineno),
                "function": frame.name,
            }
            if frame.line:
                item["code"] = frame.line.strip()
            public_frames.append(item)

    return public_frames[-depth:]


def _is_internal_allocation_frame(filename: str) -> bool:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(filename)
    return path == os.path.abspath(__file__) or path.startswith(
        current_dir + os.sep
    )


def _truthy_env(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _safe_tensor_shape(tensor: Any) -> list[int] | None:
    try:
        return [int(dim) for dim in tuple(tensor.shape)]
    except Exception:
        return None


def _infer_tensor_memory_category(tensor: Any, raw: Any) -> str:
    import torch

    if isinstance(tensor, torch.nn.Parameter) or isinstance(
        raw, torch.nn.Parameter
    ):
        return "parameter"

    try:
        if getattr(raw, "grad_fn", None) is not None:
            return "activation"
    except Exception:
        pass

    try:
        if bool(getattr(raw, "requires_grad", False)):
            return "activation"
    except Exception:
        pass

    if os.environ.get("FAKEGPU_PREFLIGHT_STAGE"):
        return "temporary"

    return "tensor"


def _mark_tensor_memory_category(tensor: Any, category: str) -> None:
    tracker = _memory_tracker
    if tracker is None or _is_fake_tensor(tensor):
        return
    dp = getattr(tensor, "_fakegpu_memory_data_ptr", None)
    if dp is not None:
        tracker.mark_category(int(dp), category)
        return
    try:
        dp = tensor.untyped_storage().data_ptr()
    except Exception:
        raw = getattr(tensor, "raw_data", None)
        if raw is None:
            return
        try:
            dp = raw.untyped_storage().data_ptr()
        except Exception:
            return
    if dp:
        tracker.mark_category(int(dp), category)


def _iter_tensors(value: Any):
    import torch

    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_tensors(item)


def _install_upstream_dispatch_memory_tracking(
    upstream: Any,
    torch_mod: Any,
) -> None:
    fake_tensor_cls = getattr(upstream, "FakeCudaTensor", None)
    if (
        fake_tensor_cls is None
        or not _MEMORY_TRACKING
        or not _DISPATCH_MEMORY_TRACKING
    ):
        _reset_dispatch_tracking_stats(enabled=False)
        return
    if getattr(fake_tensor_cls, "_fakegpu_dispatch_tracking", False):
        _reset_dispatch_tracking_stats(enabled=True)
        return

    try:
        from torch.utils._python_dispatch import TorchDispatchMode
    except (ImportError, AttributeError):
        _reset_dispatch_tracking_stats(enabled=False)
        return

    class _AllocationTrackingMode(TorchDispatchMode):
        def __init__(self, device_index: int) -> None:
            super().__init__()
            self.device_index = int(device_index)

        def __torch_dispatch__(
            self,
            func: Any,
            types: Any,
            args: tuple[Any, ...] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> Any:
            result = func(*args, **(kwargs or {}))
            tracker = _memory_tracker
            operator = str(func)
            output_tensors = 0
            new_allocations = 0
            alias_outputs = 0
            inaccessible_outputs = 0
            for tensor in _iter_tensors(result):
                output_tensors += 1
                try:
                    storage = tensor.untyped_storage()
                    data_ptr = int(storage.data_ptr())
                    nbytes = int(storage.nbytes())
                    if data_ptr == 0 or nbytes == 0:
                        inaccessible_outputs += 1
                        continue
                except Exception:
                    inaccessible_outputs += 1
                    continue

                allocated = _register_tensor_for_memory_tracking(
                    tensor,
                    self.device_index,
                    metadata={
                        "source": "torch_dispatch",
                        "operator": operator,
                    },
                    storage_info=(storage, data_ptr, nbytes),
                )
                if allocated:
                    new_allocations += 1
                elif tracker is not None and tracker.has_allocation(data_ptr):
                    alias_outputs += 1
                else:
                    inaccessible_outputs += 1
            _record_dispatch_tracking(
                operator,
                output_tensors=output_tensors,
                new_allocations=new_allocations,
                alias_outputs=alias_outputs,
                inaccessible_outputs=inaccessible_outputs,
            )
            return result

    @classmethod
    def _tracked_torch_function(
        cls,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        del cls, types
        call_kwargs = kwargs or {}
        device_index = upstream._infer_device_index(args, call_kwargs)
        cpu_args = upstream._tree_map(upstream.unwrap_tensor, args)
        cpu_kwargs = upstream._tree_map(upstream.unwrap_tensor, call_kwargs)
        with _AllocationTrackingMode(device_index):
            result = func(*cpu_args, **cpu_kwargs)
        return upstream._tree_map(
            lambda obj: upstream._wrap_function_result(obj, device_index),
            result,
        )

    _tracked_torch_function._fakegpu_dispatch_tracking = True  # type: ignore[attr-defined]
    fake_tensor_cls.__torch_function__ = _tracked_torch_function
    fake_tensor_cls._fakegpu_dispatch_tracking = True
    _reset_dispatch_tracking_stats(enabled=True)


def memory_snapshot() -> dict[str, Any]:
    """Return FakeGPU torch-layer memory metadata for preflight reports."""
    tracker = _memory_tracker
    if tracker is None:
        return {
            "tracking_confidence": "C0_incomplete",
            "devices": [],
        }
    return tracker.snapshot(_DEVICE_PROFILES)


class _TrackedSavedTensor:
    """Small holder used by autograd saved_tensors_hooks.

    Carries the bound release callback of the tracker that allocated the
    saved-tensor bookkeeping, so releasing after a re-patch() never hits a
    replacement tracker.
    """

    __slots__ = ("tensor", "raw_data_ptr", "_release", "__weakref__")

    def __init__(
        self,
        tensor: Any,
        raw_data_ptr: int | None,
        release: Any,
    ) -> None:
        self.tensor = tensor
        self.raw_data_ptr = raw_data_ptr
        self._release = release


_saved_tensors_hooks_cm: Any = None


def _active_fake_device_index() -> int:
    return int(_upstream_mod._CURRENT_DEVICE)


def _pack_autograd_saved_tensor(tensor: Any) -> Any:
    tracker = _memory_tracker
    if tracker is None or not _MEMORY_TRACKING:
        return tensor
    import torch

    try:
        if not isinstance(tensor, torch.Tensor):
            return tensor
        if _is_fake_tensor(tensor):
            return tensor
        storage = tensor.untyped_storage()
        try:
            raw_data_ptr = int(storage.data_ptr())
        except RuntimeError as exc:
            if "Cannot access data pointer" in str(exc):
                return tensor
            raise
        nbytes = int(storage.nbytes())
        if raw_data_ptr == 0 or nbytes == 0:
            return tensor
        device_index = _active_fake_device_index()
        if device_index < 0 or device_index >= len(tracker._total):
            return tensor
        metadata = {
            "dtype": str(getattr(tensor, "dtype", None)),
            "shape": _safe_tensor_shape(tensor),
            "stage": os.environ.get("FAKEGPU_PREFLIGHT_STAGE") or "unknown",
        }
        synthetic_ptr = tracker.allocate_saved_tensor(
            raw_data_ptr=raw_data_ptr,
            nbytes=nbytes,
            device=device_index,
            metadata=metadata,
        )
        if synthetic_ptr is None:
            return tensor
        holder = _TrackedSavedTensor(
            tensor,
            raw_data_ptr,
            tracker.release_saved_tensor,
        )
        weakref.finalize(holder, holder._release, raw_data_ptr)
        return holder
    except (MemoryError, RuntimeError):
        raise
    except Exception:
        return tensor


def _unpack_autograd_saved_tensor(value: Any) -> Any:
    if isinstance(value, _TrackedSavedTensor):
        raw_data_ptr = value.raw_data_ptr
        if raw_data_ptr is not None:
            value._release(int(raw_data_ptr))
            value.raw_data_ptr = None
        return value.tensor
    return value


def _install_autograd_saved_tensor_tracking(torch_mod: Any) -> None:
    global _saved_tensors_hooks_cm
    if _saved_tensors_hooks_cm is not None or _memory_tracker is None:
        return
    try:
        hooks = torch_mod.autograd.graph.saved_tensors_hooks(
            _pack_autograd_saved_tensor,
            _unpack_autograd_saved_tensor,
        )
        hooks.__enter__()
        _saved_tensors_hooks_cm = hooks

        def _close_hooks() -> None:
            global _saved_tensors_hooks_cm
            cm = _saved_tensors_hooks_cm
            _saved_tensors_hooks_cm = None
            if cm is not None:
                cm.__exit__(None, None, None)

        atexit.register(_close_hooks)
    except Exception:
        _saved_tensors_hooks_cm = None


@contextmanager
def _suspend_autograd_saved_tensor_tracking():
    """Temporarily remove the global hook for torch.func/export graph capture."""

    global _saved_tensors_hooks_cm
    hooks = _saved_tensors_hooks_cm
    if hooks is None:
        yield
        return

    _saved_tensors_hooks_cm = None
    hooks.__exit__(None, None, None)
    try:
        yield
    finally:
        try:
            import torch

            restored = torch.autograd.graph.saved_tensors_hooks(
                _pack_autograd_saved_tensor,
                _unpack_autograd_saved_tensor,
            )
            restored.__enter__()
            _saved_tensors_hooks_cm = restored
        except Exception:
            _saved_tensors_hooks_cm = None


def _install_upstream_memory_category_hooks(upstream: Any, torch_mod: Any) -> None:
    try:
        module_cls = torch_mod.nn.Module
    except Exception:
        module_cls = None
    if module_cls is not None:
        original_module_cuda = getattr(module_cls, "cuda", None)
        if callable(original_module_cuda) and not getattr(
            original_module_cuda, "_fakegpu_category_patch", False
        ):

            @functools.wraps(original_module_cuda)
            def _tracked_module_cuda(self, *args, **kwargs):
                result = original_module_cuda(self, *args, **kwargs)
                _mark_module_memory_categories(result)
                return result

            _tracked_module_cuda._fakegpu_category_patch = True  # type: ignore[attr-defined]
            module_cls.cuda = _tracked_module_cuda

        original_module_to = getattr(module_cls, "to", None)
        if callable(original_module_to) and not getattr(
            original_module_to, "_fakegpu_category_patch", False
        ):

            @functools.wraps(original_module_to)
            def _tracked_module_to(self, *args, **kwargs):
                result = original_module_to(self, *args, **kwargs)
                _mark_module_memory_categories(result)
                return result

            _tracked_module_to._fakegpu_category_patch = True  # type: ignore[attr-defined]
            module_cls.to = _tracked_module_to

    original_register_parameter = getattr(upstream, "register_parameter", None)
    if callable(original_register_parameter) and not getattr(
        original_register_parameter, "_fakegpu_category_patch", False
    ):

        @functools.wraps(original_register_parameter)
        def _tracked_register_parameter(tensor):
            result = original_register_parameter(tensor)
            _mark_tensor_memory_category(tensor, "parameter")
            return result

        _tracked_register_parameter._fakegpu_category_patch = True  # type: ignore[attr-defined]
        upstream.register_parameter = _tracked_register_parameter

    fake_tensor_cls = getattr(upstream, "FakeCudaTensor", None)
    if fake_tensor_cls is not None:
        original_backward = getattr(fake_tensor_cls, "backward", None)
        if callable(original_backward) and not getattr(
            original_backward, "_fakegpu_category_patch", False
        ):

            @functools.wraps(original_backward)
            def _tracked_backward(self, *args, **kwargs):
                try:
                    return original_backward(self, *args, **kwargs)
                finally:
                    _mark_registered_parameter_grads(upstream)

            _tracked_backward._fakegpu_category_patch = True  # type: ignore[attr-defined]
            fake_tensor_cls.backward = _tracked_backward

        grad_property = getattr(fake_tensor_cls, "grad", None)
        if isinstance(grad_property, property) and not getattr(
            grad_property.fget, "_fakegpu_category_patch", False
        ):
            original_get = grad_property.fget
            original_set = grad_property.fset

            def _tracked_grad_get(self):
                result = original_get(self) if original_get is not None else None
                if result is not None:
                    _mark_tensor_memory_category(result, "gradient")
                return result

            def _tracked_grad_set(self, value) -> None:
                if original_set is not None:
                    original_set(self, value)
                if value is not None:
                    _mark_tensor_memory_category(value, "gradient")

            _tracked_grad_get._fakegpu_category_patch = True  # type: ignore[attr-defined]
            fake_tensor_cls.grad = property(_tracked_grad_get, _tracked_grad_set)

    _install_optimizer_state_category_patch(torch_mod)


def _mark_registered_parameter_grads(upstream: Any) -> None:
    for param in list(getattr(upstream, "_REGISTERED_PARAMETERS", ())):
        _mark_tensor_memory_category(param, "parameter")
        try:
            grad = param.grad
        except Exception:
            grad = None
        if grad is not None:
            _mark_tensor_memory_category(grad, "gradient")


def _install_optimizer_state_category_patch(torch_mod: Any) -> None:
    try:
        optim_mod = torch_mod.optim
    except Exception:
        return

    class_names = (
        "Optimizer",
        "SGD",
        "Adam",
        "AdamW",
        "RMSprop",
        "Adagrad",
        "Adadelta",
    )
    for name in class_names:
        cls = getattr(optim_mod, name, None)
        if cls is None:
            continue
        original_step = getattr(cls, "step", None)
        if not callable(original_step) or getattr(
            original_step, "_fakegpu_category_patch", False
        ):
            continue

        @functools.wraps(original_step)
        def _tracked_step(self, *args, __orig_step=original_step, **kwargs):
            try:
                return __orig_step(self, *args, **kwargs)
            finally:
                _mark_optimizer_state_tensors(self)

        _tracked_step._fakegpu_category_patch = True  # type: ignore[attr-defined]
        cls.step = _tracked_step


def _mark_optimizer_state_tensors(optimizer: Any) -> None:
    try:
        states = optimizer.state.values()
    except Exception:
        return
    for state in states:
        for tensor in _iter_tensors(state):
            _mark_tensor_memory_category(tensor, "optimizer_state")


def _mark_module_memory_categories(module: Any) -> None:
    try:
        parameters = module.parameters()
    except Exception:
        parameters = ()
    for param in parameters:
        _mark_tensor_memory_category(param, "parameter")

    try:
        buffers = module.buffers()
    except Exception:
        buffers = ()
    for buffer in buffers:
        _mark_tensor_memory_category(buffer, "buffer")


@dataclass(frozen=True)
class PatchResult:
    backend: str
    num_devices: int
    device_name: str


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------
def _normalize_device_index(device: Any) -> int:
    import torch

    def _get_current() -> int:
        return _upstream_mod._CURRENT_DEVICE

    if device is None:
        return _get_current()
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        return device.index if device.index is not None else _get_current()
    return _get_current()


_orig_tensor_pin_memory: Any = None
_orig_torch_compile: Any = None


def _patched_tensor_pin_memory(self: Any, device: Any = None) -> Any:
    """Pinned-memory is a semantic no-op on FakeGPU's CPU-backed runtime."""
    return self


def _torch_minor_version(torch_mod: Any) -> tuple[int, int]:
    version = str(getattr(torch_mod, "__version__", "0.0")).split("+", 1)[0]
    parts = version.split(".")
    if len(parts) < 2:
        return (0, 0)
    try:
        return (int(parts[0]), int(parts[1]))
    except ValueError:
        return (0, 0)


def _install_compile_compat_shim(torch_mod: Any) -> None:
    """Install a no-op torch.compile compatibility shim on crash-prone minors."""
    global _orig_torch_compile

    if _torch_minor_version(torch_mod) < (2, 8):
        return

    if _orig_torch_compile is None:
        _orig_torch_compile = torch_mod.compile

    def _fakegpu_compile(model: Any = None, *args: Any, **kwargs: Any) -> Any:
        if model is None:

            def _decorator(fn: Any) -> Any:
                return fn

            return _decorator
        return model

    torch_mod.compile = _fakegpu_compile
    compiler_mod = getattr(torch_mod, "compiler", None)
    if compiler_mod is not None:
        compiler_mod.compile = _fakegpu_compile


# ---------------------------------------------------------------------------
# Fake CUDA Stream / Event
# ---------------------------------------------------------------------------
def _install_torch_accelerator_compat(torch_mod: Any) -> None:
    """Route the generic accelerator API through FakeGPU's CUDA surface.

    PyTorch 2.13 optimizers query ``torch.accelerator.current_stream()``
    before every step. On a CPU runner compiled with CUDA support, the
    unpatched implementation enters the real driver even though
    ``torch.cuda`` has already been redirected by FakeGPU.
    """

    accelerator = getattr(torch_mod, "accelerator", None)
    if accelerator is None:
        return

    def _current_accelerator(check_available: bool = False):
        if check_available and not torch_mod.cuda.is_available():
            return None
        return torch_mod.device("cuda")

    def _current_device_index() -> int:
        return int(torch_mod.cuda.current_device())

    def _set_device_index(device: Any) -> None:
        if isinstance(device, int) and device < 0:
            return
        torch_mod.cuda.set_device(device)

    def _current_stream(device: Any = None):
        return torch_mod.cuda.current_stream(device)

    def _set_stream(stream: Any) -> None:
        torch_mod.cuda.set_stream(stream)

    def _synchronize(device: Any = None) -> None:
        torch_mod.cuda.synchronize(device)

    accelerator.current_accelerator = _current_accelerator
    accelerator.current_device_index = _current_device_index
    accelerator.set_device_index = _set_device_index
    accelerator.current_stream = _current_stream
    accelerator.set_stream = _set_stream
    accelerator.synchronize = _synchronize
    accelerator.is_available = lambda: bool(torch_mod.cuda.is_available())
    accelerator.device_count = lambda: int(torch_mod.cuda.device_count())


# ---------------------------------------------------------------------------
# Fake device properties
# ---------------------------------------------------------------------------


class _FakeDeviceProperties:
    """Mimics ``torch.cuda.get_device_properties()`` return value.

    Reads per-device profile data from ``_DEVICE_PROFILES`` when available,
    falling back to the module-level scalar defaults.
    """

    def __init__(self, index: int = 0):
        prof = _DEVICE_PROFILES[index] if index < len(_DEVICE_PROFILES) else {}
        self.name = prof.get("name", _DEVICE_NAME)
        self.major = prof.get("compute_major", _COMPUTE_MAJOR)
        self.minor = prof.get("compute_minor", _COMPUTE_MINOR)
        self.total_memory = prof.get("total_memory", _TOTAL_MEMORY)
        self.multi_processor_count = 108
        self.is_multi_gpu_board = False
        self.is_integrated = False
        self.max_threads_per_multi_processor = 2048
        self.max_threads_per_block = 1024
        self.regs_per_block = 65536
        self.regs_per_multiprocessor = 65536
        self.warp_size = 32
        self.gcnArchName = ""

    def __repr__(self) -> str:
        return (
            f"_FakeDeviceProperties(name='{self.name}', major={self.major}, "
            f"minor={self.minor}, total_memory={self.total_memory // (1024**2)}MB, "
            f"multi_processor_count={self.multi_processor_count})"
        )


# ---------------------------------------------------------------------------
# torch.cuda module‑level stubs
# ---------------------------------------------------------------------------
def _stub_is_bf16_supported(device: Any = None) -> bool:
    return _COMPUTE_MAJOR >= 8
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
def _cpu_rng_state():
    import torch

    return torch.random.get_rng_state()


def _set_cpu_rng_state(new_state: Any) -> None:
    import torch

    state = new_state.cpu() if hasattr(new_state, "cpu") else new_state
    torch.random.set_rng_state(state)


def _stub_get_rng_state(device: Any = "cuda"):
    return _cpu_rng_state()


def _stub_get_rng_state_all() -> list[Any]:
    state = _cpu_rng_state()
    return [state.clone() for _ in range(_NUM_DEVICES)]


def _stub_set_rng_state(new_state: Any, device: Any = "cuda") -> None:
    _set_cpu_rng_state(new_state)


def _stub_set_rng_state_all(new_states: Any) -> None:
    states = list(new_states)
    if not states:
        return
    current = _upstream_mod._CURRENT_DEVICE
    index = current if current < len(states) else 0
    _set_cpu_rng_state(states[index])


# ---------------------------------------------------------------------------
# Shared compatibility helpers
# ---------------------------------------------------------------------------
def _reported_cuda_version(torch_mod: Any) -> str:
    configured_cuda_version = os.environ.get("FAKEGPU_CUDA_VERSION", "").strip()
    installed_cuda_version = getattr(torch_mod.version, "cuda", None)
    return str(configured_cuda_version or installed_cuda_version or "12.1")


def _patch_hf_cuda_surface(torch_mod: Any) -> None:
    """Expose CUDA metadata expected by HuggingFace and Accelerate."""
    torch_mod.version.cuda = _reported_cuda_version(torch_mod)

    backends_cuda = getattr(torch_mod.backends, "cuda", None)
    if backends_cuda is None:
        backends_cuda = types.SimpleNamespace()
        torch_mod.backends.cuda = backends_cuda
    backends_cuda.is_built = lambda: True

    matmul_backend = getattr(backends_cuda, "matmul", None)
    if matmul_backend is None:
        matmul_backend = types.SimpleNamespace()
        backends_cuda.matmul = matmul_backend
    matmul_backend.allow_tf32 = False
    if not hasattr(matmul_backend, "allow_fp16_reduced_precision_reduction"):
        matmul_backend.allow_fp16_reduced_precision_reduction = True

    cudnn_backend = getattr(torch_mod.backends, "cudnn", None)
    if cudnn_backend is None:
        cudnn_backend = types.SimpleNamespace()
        torch_mod.backends.cudnn = cudnn_backend
    cudnn_backend.is_available = lambda: True
    cudnn_backend.enabled = True
    cudnn_backend.benchmark = False
    cudnn_backend.deterministic = False
    cudnn_backend.allow_tf32 = False

    # Lightning Fabric calls torch._C._cuda_clearCublasWorkspaces in _clear_cuda_memory()
    if not hasattr(torch_mod._C, "_cuda_clearCublasWorkspaces"):
        torch_mod._C._cuda_clearCublasWorkspaces = lambda: None

    # Patch matmul precision getters/setters to avoid C++ per-backend state conflicts.
    # In torch 2.9+, setting allow_tf32 via backends.cuda.matmul uses the new per-backend
    # API, while torch.get_float32_matmul_precision() uses the legacy global getter.
    # Mixing these throws RuntimeError. We manage precision state in Python instead.
    _matmul_precision = {"value": "highest"}

    _precision_to_tf32 = {"highest": False, "high": True, "medium": True}
    _tf32_to_precision = {False: "highest", True: "high"}

    def _fake_set_float32_matmul_precision(precision: str) -> None:
        if precision not in ("highest", "high", "medium"):
            raise ValueError(
                f"Invalid precision {precision!r}, must be 'highest', 'high', or 'medium'"
            )
        _matmul_precision["value"] = precision
        matmul_backend.allow_tf32 = _precision_to_tf32[precision]

    def _fake_get_float32_matmul_precision() -> str:
        return _matmul_precision["value"]

    torch_mod.set_float32_matmul_precision = _fake_set_float32_matmul_precision
    torch_mod.get_float32_matmul_precision = _fake_get_float32_matmul_precision

    # Also intercept writes to matmul.allow_tf32 so they stay consistent
    _orig_matmul_type = type(matmul_backend)
    if hasattr(_orig_matmul_type, "allow_tf32") and isinstance(
        getattr(_orig_matmul_type, "allow_tf32", None), property
    ):
        # Real cuBLASModule has allow_tf32 as a property — override the class setter
        _orig_setter = _orig_matmul_type.allow_tf32.fset

        @_orig_matmul_type.allow_tf32.setter  # type: ignore[attr-defined]
        def _intercept_tf32(self: Any, value: bool) -> None:
            _matmul_precision["value"] = _tf32_to_precision.get(value, "highest")
            if _orig_setter is not None:
                try:
                    _orig_setter(self, value)
                except Exception:
                    pass  # C++ backend unavailable — fine, Python state is authoritative
    else:
        # SimpleNamespace or plain object — wrap with a descriptor is overkill;
        # just sync on get
        _real_get = torch_mod.get_float32_matmul_precision

        def _synced_get() -> str:
            tf32_val = getattr(matmul_backend, "allow_tf32", False)
            return _tf32_to_precision.get(tf32_val, "highest")

        torch_mod.get_float32_matmul_precision = _synced_get


def _patch_transformers_utils() -> None:
    """Patch transformers.utils helpers for LLaMA-Factory / LitGPT compatibility."""
    try:
        import transformers.utils.import_utils as _tu
    except ImportError:
        return

    if not getattr(_tu, "is_torch_cuda_available", lambda: False)():
        _tu.is_torch_cuda_available = lambda: True

    _tu.is_torch_bf16_gpu_available = lambda: _COMPUTE_MAJOR >= 8
    # Transformers caches this probe. Replacing it avoids retaining a false
    # result from imports that happened before FakeGPU enabled CUDA semantics.
    _tu.is_torch_tf32_available = lambda: _COMPUTE_MAJOR >= 8

    # Also patch the top-level re-exports if they exist
    try:
        import transformers.utils as _tu_top

        _tu_top.is_torch_cuda_available = _tu.is_torch_cuda_available
        _tu_top.is_torch_bf16_gpu_available = _tu.is_torch_bf16_gpu_available
        _tu_top.is_torch_tf32_available = _tu.is_torch_tf32_available
    except (ImportError, AttributeError):
        pass


def _build_fake_fork_rng(torch_mod: Any):
    @contextmanager
    def _fake_fork_rng(devices=None, enabled: bool = True, device_type: str = "cuda"):
        if not enabled:
            yield
            return
        cpu_state = torch_mod.random.get_rng_state()
        try:
            yield
        finally:
            torch_mod.random.set_rng_state(cpu_state)

    return _fake_fork_rng


def _patch_cuda_rng_surface(torch_mod: Any) -> None:
    fake_fork_rng = _build_fake_fork_rng(torch_mod)
    torch_mod.random.fork_rng = fake_fork_rng
    try:
        import torch.cuda.random as _random

        _random.fork_rng = fake_fork_rng
    except Exception:
        pass


def _patch_nccl_surface(torch_mod: Any) -> None:
    nccl_mod = getattr(torch_mod.cuda, "nccl", None)
    if nccl_mod is None:
        nccl_mod = types.SimpleNamespace()
        torch_mod.cuda.nccl = nccl_mod
    nccl_mod.version = lambda: (2, 21, 5)


def _patch_accelerate_utils() -> None:
    """Patch ``accelerate`` detection helpers so the library treats FakeGPU as real CUDA.

    When ``fakegpu.patch_torch()`` has already set ``torch.cuda.is_available()``
    to ``True``, accelerate's ``is_cuda_available()`` normally picks that up.
    However some helpers cache their result at import time or check extra
    conditions (like ``torch.cuda.device_count() > 0``).  We explicitly patch
    them here so that accelerate **always** sees CUDA as available regardless
    of import order.
    """
    try:
        import accelerate.utils.imports as _aui
    except ImportError:
        return

    _aui.is_cuda_available = lambda: True
    if hasattr(_aui, "is_bf16_available"):
        _aui.is_bf16_available = lambda: True

    # Re-export to top-level accelerate.utils if the names are there
    try:
        import accelerate.utils as _au

        _au.is_cuda_available = _aui.is_cuda_available
        if hasattr(_au, "is_bf16_available"):
            _au.is_bf16_available = _aui.is_bf16_available
    except (ImportError, AttributeError):
        pass


def _patch_dist_group_fallback(torch_mod: Any) -> None:
    """Ensure ``torch.distributed._get_default_group`` and ``new_group`` are
    available even when the PyTorch ``fake`` distributed backend fails to init.

    When the ``fake`` backend succeeds (PyTorch ≥ 2.1), these functions are
    already provided by the real C++ ProcessGroup layer.  This fallback only
    activates when they are absent or broken, so that downstream code like
    FSDP can still call ``_get_default_group()`` without crashing.
    """
    import torch.distributed as dist

    # ---- _get_default_group fallback ----
    try:
        from torch.distributed.distributed_c10d import (
            _get_default_group as _real_get_default_group,
        )
    except ImportError:
        _real_get_default_group = None

    if _real_get_default_group is None:
        # Very old PyTorch — provide a simple stub
        _fake_default_group = types.SimpleNamespace(
            rank=lambda: 0,
            size=lambda: 1,
        )

        def _fallback_get_default_group():
            return _fake_default_group

        dist._get_default_group = _fallback_get_default_group  # type: ignore[attr-defined]

    # ---- new_group fallback ----
    if not hasattr(dist, "new_group") or dist.new_group is None:

        def _fallback_new_group(
            ranks=None, timeout=None, backend=None, pg_options=None
        ):
            return types.SimpleNamespace(
                rank=lambda: 0,
                size=lambda: len(ranks) if ranks else 1,
            )

        dist.new_group = _fallback_new_group  # type: ignore[attr-defined]


def _patch_upstream_fakecuda_tensor_compat(upstream: Any, torch_mod: Any) -> None:
    fake_tensor_cls = getattr(upstream, "FakeCudaTensor", None)
    if fake_tensor_cls is None:
        return

    # torch.utils.checkpoint groups non-CPU tensors by ``tensor.get_device()``.
    # The subclass stores CPU data internally, so inheriting Tensor.get_device
    # leaks the backing CPU value (-1) and checkpoint tries to enter cuda:-1.
    fake_tensor_cls.get_device = lambda self: int(self.device_index)

    if getattr(fake_tensor_cls, "_fakegpu_set_patched", False):
        return

    def _patched_set_(self, source, storage_offset=0, size=None, stride=None):
        raw_source = upstream.unwrap_tensor(source)
        with torch_mod.no_grad():
            if size is None and stride is None and storage_offset == 0:
                torch_mod.Tensor.set_(self, raw_source)
            else:
                if size is None:
                    size = tuple(raw_source.shape)
                if stride is None:
                    stride = tuple(raw_source.stride())
                torch_mod.Tensor.set_(self, raw_source, storage_offset, size, stride)

        # FSDP mutates FlatParameter storage via ``set_()``. Keep the
        # CPU-side shadow tensor in sync so subsequent unwraps see the new
        # storage instead of the previously freed one.
        self.raw_data = self.as_subclass(torch_mod.Tensor)

        if isinstance(source, fake_tensor_cls):
            self.device_index = source.device_index
        return self

    fake_tensor_cls.set_ = _patched_set_
    fake_tensor_cls.record_stream = lambda self, stream: None
    fake_tensor_cls.is_cpu = property(lambda self: False)
    fake_tensor_cls._fakegpu_set_patched = True
def _patch_upstream_all_gather_object(upstream: Any, torch_mod: Any) -> None:
    if getattr(upstream, "_fakegpu_all_gather_object_patched", False):
        return

    def _clone_gathered_object_for_rank(obj: Any, rank: int) -> Any:
        import copy
        import re

        try:
            from torch.distributed._shard.metadata import ShardMetadata
            from torch.distributed._shard.sharded_tensor.metadata import (
                ShardedTensorMetadata,
            )
        except Exception:
            ShardMetadata = None
            ShardedTensorMetadata = None

        if ShardMetadata is not None and isinstance(obj, ShardMetadata):
            shard_offsets = list(obj.shard_offsets)
            if shard_offsets:
                shard_offsets[0] = int(obj.shard_sizes[0]) * rank
            placement = re.sub(
                r"rank:\d+/", f"rank:{rank}/", str(obj.placement), count=1
            )
            return dataclasses.replace(
                obj,
                shard_offsets=shard_offsets,
                placement=placement,
            )

        if ShardedTensorMetadata is not None and isinstance(obj, ShardedTensorMetadata):
            return dataclasses.replace(
                obj,
                shards_metadata=[
                    _clone_gathered_object_for_rank(shard, rank)
                    for shard in obj.shards_metadata
                ],
            )

        if isinstance(obj, list):
            return [_clone_gathered_object_for_rank(item, rank) for item in obj]
        if isinstance(obj, tuple):
            return tuple(_clone_gathered_object_for_rank(item, rank) for item in obj)
        if isinstance(obj, dict):
            return obj.__class__(
                (key, _clone_gathered_object_for_rank(value, rank))
                for key, value in obj.items()
            )
        return copy.deepcopy(obj)

    def _patched_all_gather_object(
        object_list: list[Any], obj: Any, group: Any = None
    ) -> None:
        for index in range(len(object_list)):
            object_list[index] = _clone_gathered_object_for_rank(obj, index)
        return None

    upstream._dist_all_gather_object = _patched_all_gather_object
    torch_mod.distributed.all_gather_object = _patched_all_gather_object
    upstream._fakegpu_all_gather_object_patched = True


def _patch_upstream_process_group_compat(upstream: Any, torch_mod: Any) -> None:
    if getattr(upstream, "_fakegpu_process_group_patched", False):
        return

    orig_dist_init = getattr(upstream, "_fakegpu_orig_dist_init", None)
    orig_dist_destroy = getattr(upstream, "_fakegpu_orig_dist_destroy", None)

    def _patched_dist_init_process_group(
        backend: str | None = None,
        init_method: Any = None,
        timeout: Any = None,
        world_size: int = -1,
        rank: int = -1,
        store: Any = None,
        group_name: str = "",
        pg_options: Any = None,
        device_id: Any = None,
    ) -> None:
        upstream._DIST_INITIALIZED = True
        upstream._DIST_BACKEND = "nccl" if backend is None else str(backend)
        upstream._DIST_WORLD_SIZE = 1 if world_size in (-1, None) else int(world_size)
        upstream._DIST_RANK = 0 if rank in (-1, None) else int(rank)

        if orig_dist_init is not None:
            env_set: list[str] = []
            if "MASTER_ADDR" not in os.environ:
                os.environ["MASTER_ADDR"] = "localhost"
                env_set.append("MASTER_ADDR")
            if "MASTER_PORT" not in os.environ:
                os.environ["MASTER_PORT"] = "29500"
                env_set.append("MASTER_PORT")
            try:
                orig_dist_init(
                    backend="fake",
                    rank=upstream._DIST_RANK,
                    world_size=upstream._DIST_WORLD_SIZE,
                )
            except Exception:
                pass
            finally:
                for key in env_set:
                    os.environ.pop(key, None)
        return None

    def _patched_dist_destroy_process_group(group: Any = None) -> None:
        if orig_dist_destroy is not None:
            try:
                orig_dist_destroy(group)
            except Exception:
                pass
        upstream._DIST_INITIALIZED = False
        upstream._DIST_WORLD_SIZE = 1
        upstream._DIST_RANK = 0
        return None

    upstream._dist_init_process_group = _patched_dist_init_process_group
    upstream._dist_destroy_process_group = _patched_dist_destroy_process_group
    torch_mod.distributed.init_process_group = _patched_dist_init_process_group
    torch_mod.distributed.destroy_process_group = _patched_dist_destroy_process_group
    upstream._fakegpu_process_group_patched = True


def _patch_fsdp_device_handling() -> None:
    """Patch FSDP internal device resolution for FakeGPU compatibility.

    On macOS, ``torch.device(0)`` resolves to ``mps:0`` via the C++
    accelerator lookup, which cannot be overridden from Python.  This causes
    FSDP's device_id resolution and ``_FSDPDeviceHandle.from_device()`` to
    produce MPS-backed handles while model parameters report ``cuda:0``
    (from FakeCudaTensor), leading to device mismatch errors.

    We fix this by:
    1. Wrapping ``_FSDPDeviceHandle.from_device`` to remap any non-cuda
       device to cuda before creating the handle.
    2. Wrapping ``_get_device_from_device_id`` to fix integer device_id
       resolution (``torch.device(0)`` → ``mps:0`` → remapped to ``cuda:0``).
    """
    try:
        from torch.distributed.fsdp._common_utils import _FSDPDeviceHandle
    except ImportError:
        return  # FSDP not available

    _orig_from_device = _FSDPDeviceHandle.from_device.__func__

    @classmethod  # type: ignore[misc]
    def _patched_from_device(cls, device):
        import torch as _torch

        # Remap non-cuda device types to cuda for FakeGPU
        if device.type not in ("cuda", "cpu", "meta"):
            device = _torch.device(
                "cuda", device.index if device.index is not None else 0
            )
        return _orig_from_device(cls, device)

    _FSDPDeviceHandle.from_device = _patched_from_device

    # Also patch _get_device_from_device_id for the consistency check in
    # _get_compute_device (compares device_from_device_id vs param.device).
    try:
        import torch.distributed.fsdp._init_utils as _fsdp_init
        import torch.distributed.fsdp.fully_sharded_data_parallel as _fsdp_mod
    except ImportError:
        return

    _orig_get_device = getattr(_fsdp_init, "_get_device_from_device_id", None)
    if _orig_get_device is not None:

        @functools.wraps(_orig_get_device)
        def _patched_get_device(device_id, rank, device_handle):
            result = _orig_get_device(device_id, rank, device_handle)
            if result is not None and result.type not in ("cuda", "cpu", "meta"):
                import torch as _torch

                result = _torch.device(
                    "cuda", result.index if result.index is not None else 0
                )
            return result

        _fsdp_init._get_device_from_device_id = _patched_get_device
        # Also patch the direct import in the FSDP module
        if hasattr(_fsdp_mod, "_get_device_from_device_id"):
            _fsdp_mod._get_device_from_device_id = _patched_get_device


def _patch_fsdp_runtime_compat(fake_tensor_cls: type | None) -> None:
    try:
        import torch
        import torch.distributed.fsdp._runtime_utils as _fsdp_runtime
    except ImportError:
        return

    _orig_register = getattr(_fsdp_runtime, "_register_post_backward_hook", None)
    if _orig_register is None or getattr(_orig_register, "_fakegpu_patched", False):
        return

    @functools.wraps(_orig_register)
    def _patched_register_post_backward_hook(state, handle):
        if not handle or fake_tensor_cls is None:
            return _orig_register(state, handle)

        flat_param = handle.flat_param
        if not isinstance(flat_param, fake_tensor_cls):
            return _orig_register(state, handle)

        if not torch.is_grad_enabled():
            return

        already_registered = hasattr(flat_param, "_post_backward_hook_state")
        if already_registered or not flat_param.requires_grad:
            return

        # FakeCudaTensor aliases do not expose the expected AccumulateGrad via
        # ``expand_as(...).grad_fn.next_functions`` after FSDP's internal
        # storage rebinding. Use the newer post-accumulate hook when available
        # to avoid asserting during forward.
        register_hook = getattr(flat_param, "register_post_accumulate_grad_hook", None)
        if register_hook is not None:
            hook = functools.partial(_fsdp_runtime._post_backward_hook, state, handle)
            hook_handle = register_hook(hook)
            flat_param._post_backward_hook_state = (None, hook_handle)
            return

        return

    _patched_register_post_backward_hook._fakegpu_patched = True
    _fsdp_runtime._register_post_backward_hook = _patched_register_post_backward_hook


def _install_fakegpu_autocast(torch_mod: Any) -> None:
    _strict_compat = os.environ.get("FAKEGPU_STRICT_COMPAT", "1") != "0"
    if not _strict_compat:
        return

    _OrigAutocast = torch_mod.amp.autocast

    class _PatchedAutocast(_OrigAutocast):
        """Autocast wrapper that redirects fake CUDA autocast to CPU."""

        def __init__(self, device_type: str = "cuda", **kwargs):
            requested_device = device_type
            actual_device = "cpu" if device_type == "cuda" else device_type
            self._fakegpu_requested_device_type = requested_device
            self._fakegpu_actual_device_type = actual_device
            super().__init__(actual_device, **kwargs)

        def __enter__(self):
            if (
                getattr(self, "_fakegpu_requested_device_type", None) == "cuda"
                and getattr(self, "fast_dtype", None) == torch_mod.bfloat16
                and _COMPUTE_MAJOR < 8
            ):
                raise RuntimeError(
                    f"Current CUDA Device does not support bfloat16. "
                    f"Please switch dtype to float16 "
                    f"(compute capability {_COMPUTE_MAJOR}.{_COMPUTE_MINOR}, "
                    f"need >= 8.0 for bf16)."
                )
            return super().__enter__()

    torch_mod.amp.autocast = _PatchedAutocast
    if hasattr(torch_mod.cuda.amp, "autocast"):
        torch_mod.cuda.amp.autocast = _PatchedAutocast


def _activate_upstream(num_devices: int, device_name: str) -> Any:
    """Load and enable the upstream FakeCudaTensor backend.

    Prefers an installed ``torch.fakegpu`` module (custom PyTorch build) and
    falls back to the vendored ``fakegpu._upstream`` that ships with this
    package.  Raises when neither can be activated: the vendored module is
    always present in a healthy install, so a failure means the installation
    is broken and must not be silently degraded.
    """
    global _upstream_mod

    # 1. Prefer an installed torch.fakegpu (custom PyTorch build).  Only an
    # ImportError means "not installed"; anything else is a broken build and
    # must surface instead of silently falling back to the vendored module.
    try:
        upstream = importlib.import_module("torch.fakegpu")
    except ImportError:
        from . import _upstream as upstream

    if not hasattr(upstream, "enable"):
        raise RuntimeError(
            "fakegpu.torch_patch: upstream backend has no enable() function: "
            f"{upstream!r}"
        )

    orig_dist_init = None
    orig_dist_destroy = None
    try:
        import torch.distributed as _dist

        orig_dist_init = _dist.init_process_group
        orig_dist_destroy = _dist.destroy_process_group
    except Exception:
        pass

    # Configure device count and name before enable()
    upstream._NUM_DEVICES = num_devices
    upstream._DEVICE_NAME = device_name

    os.environ["TORCH_FAKEGPU_DEVICE_COUNT"] = str(num_devices)
    os.environ["TORCH_FAKEGPU_DEVICE_NAME"] = device_name

    upstream.enable()
    upstream._fakegpu_orig_dist_init = orig_dist_init
    upstream._fakegpu_orig_dist_destroy = orig_dist_destroy
    _upstream_mod = upstream
    return upstream


def _apply_enhancements_over_upstream(upstream: Any, torch_mod: Any) -> None:
    """Layer FakeGPU enhancements on top of the upstream FakeCudaTensor backend.

    The upstream ``enable()`` has already patched core CUDA redirection
    (Tensor.to/cuda, Module.to/cuda, DataParallel, DDP, distributed,
    factory functions, torch.load).  This function adds:

    * Per-device GPU profile support
    * Memory tracking with OOM simulation
    * Autocast dtype validation
    * GradScaler passthrough
    * Cross-device validation
    * Terminal report on exit
    """
    global _memory_tracker

    import torch.cuda

    # Let analysis helpers distinguish FakeGPU's simulated availability from a
    # real CUDA runtime.  In particular, PyTorch 2.13 FakeTensor tracing may
    # initialize a physical CUDA context when is_available() reports true.
    torch_mod.cuda._fakegpu_simulated = True

    # ---- 0. Device index bounds validation ----
    # The upstream uses a different error message ("Invalid fake CUDA device index N").
    # Replace _normalize_device_index entirely so that all paths (set_device,
    # torch.load, etc.) that call it produce our "invalid device ordinal" message
    # matching real CUDA behaviour and our test suite.
    _orig_normalize_cuda_device = upstream._normalize_cuda_device

    def _checked_normalize_device_index(device):
        """Full replacement for upstream._normalize_device_index."""
        normalized = _orig_normalize_cuda_device(device)
        if normalized is None:
            return upstream._CURRENT_DEVICE
        index = 0 if normalized.index is None else int(normalized.index)
        if index < 0 or index >= _NUM_DEVICES:
            raise RuntimeError(
                f"CUDA error: invalid device ordinal "
                f"(requested cuda:{index}, available: {_NUM_DEVICES})"
            )
        return index

    upstream._normalize_device_index = _checked_normalize_device_index

    # Also override set_device to use our bounds-checked normalize
    def _validated_set_device(device):
        idx = _checked_normalize_device_index(device)
        upstream._CURRENT_DEVICE = idx

    torch.cuda.set_device = _validated_set_device
    upstream.set_device = _validated_set_device

    # Override _normalize_cuda_device to validate bounds for factory functions.
    # The upstream version doesn't check _NUM_DEVICES, so torch.randn(device="cuda:99")
    # would silently create a tensor on a non-existent device.
    def _bounds_checked_normalize_cuda(device, *, allow_none=False):
        result = _orig_normalize_cuda_device(device, allow_none=allow_none)
        if result is not None and result.type == "cuda":
            idx = 0 if result.index is None else int(result.index)
            if idx < 0 or idx >= _NUM_DEVICES:
                raise RuntimeError(
                    f"CUDA error: invalid device ordinal "
                    f"(requested cuda:{idx}, available: {_NUM_DEVICES})"
                )
        return result

    upstream._normalize_cuda_device = _bounds_checked_normalize_cuda

    # ---- 1. Memory tracker ----
    if _MEMORY_TRACKING:
        per_device_bytes = [p["total_memory"] for p in _DEVICE_PROFILES]
        _memory_tracker = _DeviceMemoryTracker(per_device_bytes)
        _install_autograd_saved_tensor_tracking(torch_mod)

    # ---- 2. Hook upstream.wrap_tensor for memory tracking ----
    _orig_wrap_tensor = upstream.wrap_tensor

    def _hooked_wrap_tensor(t, device_index=None):
        # Validate device index bounds
        actual_idx = (
            upstream._CURRENT_DEVICE if device_index is None else int(device_index)
        )
        if actual_idx < 0 or actual_idx >= _NUM_DEVICES:
            raise RuntimeError(
                f"CUDA error: invalid device ordinal "
                f"(requested cuda:{actual_idx}, available: {_NUM_DEVICES})"
            )
        result = _orig_wrap_tensor(t, device_index=device_index)
        if _memory_tracker is not None:
            actual_idx = getattr(result, "device_index", 0)
            tracked_data_ptr = getattr(
                t,
                "_fakegpu_memory_data_ptr",
                None,
            )
            if (
                tracked_data_ptr is not None
                and _memory_tracker.has_allocation(int(tracked_data_ptr))
            ):
                _set_tracked_data_ptr(result, int(tracked_data_ptr))
            else:
                _register_tensor_for_memory_tracking(result, actual_idx)
        return result

    upstream.wrap_tensor = _hooked_wrap_tensor
    _install_upstream_dispatch_memory_tracking(upstream, torch_mod)
    _install_upstream_memory_category_hooks(upstream, torch_mod)
    _patch_upstream_fakecuda_tensor_compat(upstream, torch_mod)
    _patch_upstream_all_gather_object(upstream, torch_mod)
    _patch_upstream_process_group_compat(upstream, torch_mod)
    _install_torch_accelerator_compat(torch_mod)

    def _dynamo_friendly_tree_map(fn, obj):
        # torch.Size is a tuple subclass but must be preserved as-is so that
        # FSDP (and other code) can call .numel() on tensor.size() results.
        if isinstance(obj, torch_mod.Size):
            return obj
        if isinstance(obj, tuple):
            return tuple(_dynamo_friendly_tree_map(fn, item) for item in obj)
        if isinstance(obj, list):
            return [_dynamo_friendly_tree_map(fn, item) for item in obj]
        if isinstance(obj, dict):
            items = (
                (key, _dynamo_friendly_tree_map(fn, value))
                for key, value in obj.items()
            )
            if isinstance(obj, dict) and type(obj) is dict:
                return dict(items)
            return obj.__class__(items)
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            mapped_fields = {
                field.name: _dynamo_friendly_tree_map(fn, getattr(obj, field.name))
                for field in dataclasses.fields(obj)
            }
            try:
                return type(obj)(**mapped_fields)
            except TypeError:
                new_obj = object.__new__(type(obj))
                for name, value in mapped_fields.items():
                    object.__setattr__(new_obj, name, value)
                return new_obj
        return fn(obj)

    upstream._tree_map = _dynamo_friendly_tree_map

    fake_ddp_cls = getattr(torch_mod.nn.parallel, "DistributedDataParallel", None)
    if fake_ddp_cls is not None and not hasattr(fake_ddp_cls, "_get_active_ddp_module"):
        fake_ddp_cls._get_active_ddp_module = staticmethod(lambda: None)

    global _orig_tensor_pin_memory
    if _orig_tensor_pin_memory is None:
        _orig_tensor_pin_memory = torch_mod.Tensor.pin_memory
    torch_mod.Tensor.pin_memory = _patched_tensor_pin_memory  # type: ignore[assignment]
    _install_compile_compat_shim(torch_mod)

    # ---- 3. Per-device GPU profiles ----
    def _profiled_get_device_name(device=None):
        idx = _normalize_device_index(device)
        if idx < len(_DEVICE_PROFILES):
            return _DEVICE_PROFILES[idx].get("name", _DEVICE_NAME)
        return _DEVICE_NAME

    def _profiled_get_device_capability(device=None):
        idx = _normalize_device_index(device)
        if idx < len(_DEVICE_PROFILES):
            prof = _DEVICE_PROFILES[idx]
            return (
                prof.get("compute_major", _COMPUTE_MAJOR),
                prof.get("compute_minor", _COMPUTE_MINOR),
            )
        return (_COMPUTE_MAJOR, _COMPUTE_MINOR)

    def _profiled_get_device_properties(device=None):
        idx = _normalize_device_index(device)
        if idx < 0 or idx >= _NUM_DEVICES:
            raise RuntimeError(
                f"CUDA error: invalid device ordinal "
                f"(requested {idx}, available: {_NUM_DEVICES})"
            )
        return _FakeDeviceProperties(idx)

    torch.cuda.get_device_name = _profiled_get_device_name
    torch.cuda.get_device_capability = _profiled_get_device_capability
    torch.cuda.get_device_properties = _profiled_get_device_properties

    # Compute-capability-aware bf16 check (upstream always returns True)
    torch.cuda.is_bf16_supported = _stub_is_bf16_supported
    _patch_hf_cuda_surface(torch_mod)

    # ---- 4. Tracked memory query functions ----
    if _memory_tracker is not None:

        def _active_tracker():
            if _memory_tracker is None:
                raise RuntimeError("FakeGPU memory tracking is not initialized")
            return _memory_tracker

        def _tracked_memory_allocated(device=None):
            idx = _normalize_device_index(device)
            return _active_tracker().memory_allocated(idx)

        def _tracked_max_memory_allocated(device=None):
            idx = _normalize_device_index(device)
            return _active_tracker().max_memory_allocated(idx)

        def _tracked_memory_reserved(device=None):
            idx = _normalize_device_index(device)
            return _active_tracker().memory_reserved(idx)

        def _tracked_max_memory_reserved(device=None):
            idx = _normalize_device_index(device)
            return _active_tracker().max_memory_reserved(idx)

        def _tracked_mem_get_info(device=None):
            idx = _normalize_device_index(device)
            return _active_tracker().mem_get_info(idx)

        def _tracked_empty_cache():
            _active_tracker().empty_cache()

        def _tracked_reset_peak_memory_stats(device=None):
            idx = _normalize_device_index(device)
            _active_tracker().reset_peak(idx)

        def _tracked_reset_accumulated_memory_stats(device=None):
            idx = _normalize_device_index(device)
            _active_tracker().reset_accumulated(idx)

        def _tracked_memory_stats(device=None):
            idx = _normalize_device_index(device)
            return _active_tracker().memory_stats(idx)

        def _tracked_memory_snapshot():
            return _active_tracker().allocator_snapshot()

        torch.cuda.empty_cache = _tracked_empty_cache
        torch.cuda.memory_allocated = _tracked_memory_allocated
        torch.cuda.max_memory_allocated = _tracked_max_memory_allocated
        torch.cuda.memory_reserved = _tracked_memory_reserved
        torch.cuda.max_memory_reserved = _tracked_max_memory_reserved
        torch.cuda.memory_cached = _tracked_memory_reserved
        torch.cuda.max_memory_cached = _tracked_max_memory_reserved
        torch.cuda.mem_get_info = _tracked_mem_get_info
        torch.cuda.reset_peak_memory_stats = _tracked_reset_peak_memory_stats
        torch.cuda.reset_max_memory_allocated = _tracked_reset_peak_memory_stats
        torch.cuda.reset_max_memory_cached = _tracked_reset_peak_memory_stats
        torch.cuda.reset_accumulated_memory_stats = (
            _tracked_reset_accumulated_memory_stats
        )
        torch.cuda.memory_stats = _tracked_memory_stats
        torch.cuda.memory_snapshot = _tracked_memory_snapshot

        # Also patch torch.cuda.memory submodule
        try:
            import torch.cuda.memory as _memory_mod

            _memory_mod.empty_cache = _tracked_empty_cache
            _memory_mod.memory_allocated = _tracked_memory_allocated
            _memory_mod.max_memory_allocated = _tracked_max_memory_allocated
            _memory_mod.memory_reserved = _tracked_memory_reserved
            _memory_mod.max_memory_reserved = _tracked_max_memory_reserved
            _memory_mod.memory_cached = _tracked_memory_reserved
            _memory_mod.max_memory_cached = _tracked_max_memory_reserved
            _memory_mod.mem_get_info = _tracked_mem_get_info
            _memory_mod.reset_peak_memory_stats = _tracked_reset_peak_memory_stats
            _memory_mod.reset_max_memory_allocated = _tracked_reset_peak_memory_stats
            _memory_mod.reset_max_memory_cached = _tracked_reset_peak_memory_stats
            _memory_mod.reset_accumulated_memory_stats = (
                _tracked_reset_accumulated_memory_stats
            )
            _memory_mod.memory_stats = _tracked_memory_stats
            _memory_mod.memory_snapshot = _tracked_memory_snapshot
        except Exception:
            pass

    # ---- 5. Autocast / GradScaler ----
    _install_fakegpu_autocast(torch_mod)
    _install_optimizer_state_category_patch(torch_mod)

    try:
        from torch.amp import GradScaler as _RealGradScaler

        class _FakeGradScaler(_RealGradScaler):
            def __init__(self, *args: Any, **kwargs: Any):
                kwargs.setdefault("enabled", False)
                super().__init__(*args, **kwargs)

        torch_mod.cuda.amp.GradScaler = _FakeGradScaler  # type: ignore[attr-defined]
        torch_mod.amp.GradScaler = _FakeGradScaler  # type: ignore[attr-defined]
    except Exception:
        pass

    # ---- 6. Cross-device validation ----
    if _CROSS_DEVICE_CHECK:
        import torch.nn.functional as F

        _MULTI_TENSOR_OPS = [
            "matmul",
            "mm",
            "bmm",
            "cat",
            "stack",
            "where",
            "addmm",
            "addcmul",
            "addcdiv",
        ]
        for op_name in _MULTI_TENSOR_OPS:
            orig = getattr(torch_mod, op_name, None)
            if orig is not None:
                setattr(
                    torch_mod,
                    op_name,
                    _wrap_multi_tensor_op(orig, torch_mod),
                )

        _LOSS_OPS = ["cross_entropy", "mse_loss", "nll_loss", "binary_cross_entropy"]
        for op_name in _LOSS_OPS:
            orig = getattr(F, op_name, None)
            if orig is not None:
                setattr(F, op_name, _wrap_multi_tensor_op(orig, torch_mod))

        _FUNCTIONAL_OPS = [
            "linear",
            "conv1d",
            "conv2d",
            "conv3d",
            "embedding",
            "batch_norm",
            "layer_norm",
        ]
        for op_name in _FUNCTIONAL_OPS:
            orig = getattr(F, op_name, None)
            if orig is not None:
                setattr(F, op_name, _wrap_multi_tensor_op(orig, torch_mod))

        _BINARY_DUNDERS = [
            "__add__",
            "__radd__",
            "__sub__",
            "__rsub__",
            "__mul__",
            "__rmul__",
            "__truediv__",
            "__rtruediv__",
            "__matmul__",
            "__rmatmul__",
        ]
        for dunder in _BINARY_DUNDERS:
            orig = getattr(torch_mod.Tensor, dunder, None)
            if orig is not None:
                setattr(
                    torch_mod.Tensor,
                    dunder,
                    _wrap_tensor_binary_op(orig, dunder, torch_mod),
                )

    # ---- 7. RNG state functions (not provided by upstream) ----
    torch.cuda.get_rng_state = _stub_get_rng_state
    torch.cuda.get_rng_state_all = _stub_get_rng_state_all
    torch.cuda.set_rng_state = _stub_set_rng_state
    torch.cuda.set_rng_state_all = _stub_set_rng_state_all
    try:
        import torch.cuda.random as _random

        _random.get_rng_state = _stub_get_rng_state
        _random.get_rng_state_all = _stub_get_rng_state_all
        _random.set_rng_state = _stub_set_rng_state
        _random.set_rng_state_all = _stub_set_rng_state_all
    except Exception:
        pass
    _patch_cuda_rng_surface(torch_mod)
    _patch_nccl_surface(torch_mod)
    _patch_dist_group_fallback(torch_mod)
    _patch_fsdp_device_handling()
    _patch_fsdp_runtime_compat(getattr(upstream, "FakeCudaTensor", None))
    _patch_transformers_utils()
    _patch_accelerate_utils()

    # ---- 8. Terminal report ----
    atexit.register(_dump_terminal_summary)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def patch(
    *, num_devices: int | None = None, device_name: str | None = None
) -> PatchResult:
    """Apply monkey-patches to ``torch`` so CUDA code runs transparently on CPU.

    Safe to call multiple times. Later calls refresh device/profile state while
    reusing the installed monkey patches.

    Parameters
    ----------
    num_devices:
        Number of fake CUDA devices to expose.  Defaults to ``$FAKEGPU_DEVICE_COUNT`` or 8.
    device_name:
        Name reported by ``torch.cuda.get_device_name()``.
    Returns
    -------
    PatchResult
        Describes the activated upstream FakeCudaTensor backend.
    """

    global \
        _patched, \
        _NUM_DEVICES, \
        _DEVICE_NAME, \
        _patch_result, \
        _memory_tracker

    os.environ["FAKEGPU_RUNTIME"] = "fakecuda"
    import torch
    import torch.cuda

    _refresh_runtime_profile_state(num_devices=num_devices, device_name=device_name)

    if _patched:
        _upstream_mod._NUM_DEVICES = _NUM_DEVICES
        _upstream_mod._DEVICE_NAME = _DEVICE_NAME
        if getattr(_upstream_mod, "_CURRENT_DEVICE", 0) >= _NUM_DEVICES:
            _upstream_mod._CURRENT_DEVICE = 0
        torch.cuda._cached_device_count = _NUM_DEVICES
        if _MEMORY_TRACKING:
            _memory_tracker = _DeviceMemoryTracker(
                [profile["total_memory"] for profile in _DEVICE_PROFILES]
            )
        _reset_dispatch_tracking_stats(
            enabled=bool(_MEMORY_TRACKING and _DISPATCH_MEMORY_TRACKING)
        )
        _refresh_smi_publisher()
        _patch_transformers_utils()
        _patch_accelerate_utils()
        _patch_result = PatchResult(
            backend=_patch_result.backend,
            num_devices=_NUM_DEVICES,
            device_name=_DEVICE_NAME,
        )
        return _patch_result

    upstream = _activate_upstream(_NUM_DEVICES, _DEVICE_NAME)
    _apply_enhancements_over_upstream(upstream, torch)
    _refresh_smi_publisher()
    _patched = True
    _patch_result = PatchResult(
        backend="upstream",
        num_devices=_NUM_DEVICES,
        device_name=_DEVICE_NAME,
    )
    warnings.warn(
        "fakegpu.torch_patch: enabled upstream FakeCudaTensor backend with "
        "FakeGPU enhancements (memory tracking, GPU profiles, cross-device "
        "validation).",
        stacklevel=2,
    )
    return _patch_result



def is_patched() -> bool:
    """Return True if the torch‑cuda patch has been applied."""
    return _patched
