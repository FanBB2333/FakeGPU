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
import traceback
import types
import weakref
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from ._allocator import (
    _DeviceMemoryTracker,
    _record_dispatch_tracking,
    _reset_dispatch_tracking_stats,
)
from ._cross_device import (
    _CROSS_DEVICE_CHECK,
    _is_fake_tensor,
    _wrap_multi_tensor_op,
    _wrap_tensor_binary_op,
)
from ._profiles import (
    _resolve_compute_capability,
    _resolve_device_name,
    _resolve_per_device_profiles,
    _resolve_total_memory,
)
from .profile_catalog import architecture_for_compute_capability

_patched = False
_patch_result: "PatchResult | None" = None
_upstream_mod: Any = None  # Set when upstream FakeCudaTensor backend is active

# ---------------------------------------------------------------------------
# Configuration – mirrors the active FakeGPU profile when available.
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Memory tracking: per-device memory accounting.
# ---------------------------------------------------------------------------

_MEMORY_TRACKING = os.environ.get("FAKEGPU_MEMORY_TRACKING", "1") != "0"
_DISPATCH_MEMORY_TRACKING = (
    os.environ.get("FAKEGPU_DISPATCH_MEMORY_TRACKING", "1") != "0"
)
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

    __slots__ = (
        "tensor",
        "raw_data_ptr",
        "_release",
        "_finalizer",
        "__weakref__",
    )

    def __init__(
        self,
        tensor: Any,
        raw_data_ptr: int | None,
        release: Any,
    ) -> None:
        self.tensor = tensor
        self.raw_data_ptr = raw_data_ptr
        self._release = release
        self._finalizer: weakref.finalize | None = None


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
        holder._finalizer = weakref.finalize(
            holder,
            holder._release,
            raw_data_ptr,
        )
        return holder
    except (MemoryError, RuntimeError):
        raise
    except Exception:
        return tensor


def _unpack_autograd_saved_tensor(value: Any) -> Any:
    if isinstance(value, _TrackedSavedTensor):
        raw_data_ptr = value.raw_data_ptr
        if value._finalizer is not None:
            value._finalizer.detach()
            value._finalizer = None
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

    # Configure device count and name before enable()
    upstream._NUM_DEVICES = num_devices
    upstream._DEVICE_NAME = device_name

    os.environ["TORCH_FAKEGPU_DEVICE_COUNT"] = str(num_devices)
    os.environ["TORCH_FAKEGPU_DEVICE_NAME"] = device_name

    upstream.enable()
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
