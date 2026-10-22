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
import time
import traceback
import weakref
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from . import _monitor_activity

from ._allocator import (
    _DeviceMemoryTracker,
    _record_dispatch_tracking,
    _reset_dispatch_tracking_stats,
)
from ._cross_device import (
    _BINARY_DUNDER_TORCH_OPS,
    _CROSS_DEVICE_CHECK,
    _is_fake_tensor,
    _wrap_multi_tensor_op,
    _wrap_tensor_binary_op,
)
from ._ecosystem_compat import (
    _patch_accelerate_utils,
    _patch_fsdp_device_handling,
    _patch_fsdp_runtime_compat,
    _patch_upstream_all_gather_object,
    _patch_upstream_fakecuda_tensor_compat,
)
from ._torch_reporting import (
    _arch_name,  # noqa: F401 - preserve the historical private import path
    _fmt_bytes,  # noqa: F401 - preserve the historical private import path
    render_terminal_summary as _render_terminal_summary,
)
from ._profiles import (
    _profile_entry,
    _resolve_per_device_profiles,
    _resolve_profile_id,
)
_patched = False
_patch_result: "PatchResult | None" = None
_upstream_mod: Any = None  # Set when upstream FakeCudaTensor backend is active

# ---------------------------------------------------------------------------
# Configuration – mirrors the active FakeGPU profile when available.
# ---------------------------------------------------------------------------

_NUM_DEVICES = int(os.environ.get("FAKEGPU_DEVICE_COUNT", "8"))
_DEVICE_PROFILES: list[dict[str, Any]] = _resolve_per_device_profiles(_NUM_DEVICES)

_default_profile = _profile_entry(_resolve_profile_id() or "")
_DEVICE_NAME = os.environ.get("FAKEGPU_DEVICE_NAME") or _default_profile["name"]
_COMPUTE_MAJOR = _default_profile["compute_major"]
_COMPUTE_MINOR = _default_profile["compute_minor"]
_TOTAL_MEMORY = _default_profile["total_memory"]


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
        _COMPUTE_MAJOR = first["compute_major"]
        _COMPUTE_MINOR = first["compute_minor"]
        _TOTAL_MEMORY = first["total_memory"]
        _DEVICE_NAME = first["name"]
    if device_name is not None:
        _DEVICE_NAME = device_name


# ---------------------------------------------------------------------------
# Memory tracking: per-device memory accounting.
# ---------------------------------------------------------------------------

_MEMORY_TRACKING = os.environ.get("FAKEGPU_MEMORY_TRACKING", "1") != "0"
_DISPATCH_MEMORY_TRACKING = (
    os.environ.get("FAKEGPU_DISPATCH_MEMORY_TRACKING", "1") != "0"
)

def _dump_terminal_summary() -> None:
    """Print a Report Summary to stderr on process exit.

    Controlled by ``FAKEGPU_TERMINAL_REPORT`` (default: enabled).
    """
    if os.environ.get("FAKEGPU_TERMINAL_REPORT", "1") == "0":
        return
    _render_terminal_summary(
        _memory_tracker,
        _DEVICE_PROFILES,
        transfers=_transfer_snapshot(),
    )


# Initialized later in patch() after _DEVICE_PROFILES is finalized
_memory_tracker: _DeviceMemoryTracker | None = None
_smi_publisher: Any = None
# Modeled host and peer copies (fakegpu.link_traffic.LinkTraffic)
_link_traffic: Any = None


def _transfer_snapshot() -> dict[int, dict[str, Any]]:
    traffic = _link_traffic
    if traffic is None:
        return {}
    try:
        return traffic.snapshot()
    except Exception:
        return {}


def _smi_memory_snapshot() -> dict[str, Any]:
    snapshot = _memory_tracker.snapshot(_DEVICE_PROFILES)
    if _monitor_activity.enabled():
        activity = _monitor_activity.snapshot()
        for device in snapshot["devices"]:
            device["activity"] = activity.get(device["index"], _monitor_activity.empty())
    transfers = _transfer_snapshot()
    for device in snapshot["devices"]:
        if device["index"] in transfers:
            device["transfer"] = transfers[device["index"]]
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
    # ``fakegpu run`` lists a process limited by CUDA_VISIBLE_DEVICES under
    # the run's GPU indices, as nvidia-smi does on a real machine.
    try:
        physical = [int(item) for item in os.environ.get("FAKEGPU_PHYSICAL_INDICES", "").split(",") if item]
    except ValueError:
        physical = []
    for device in snapshot["devices"]:
        if 0 <= device["index"] < len(physical):
            device["index"] = physical[device["index"]]
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
    selected = [
        frame
        for frame in frames
        if not os.path.abspath(frame.filename).startswith(_PACKAGE_PREFIX)
    ] or frames
    return [_frame_record(frame) for frame in selected[-depth:]]


_PACKAGE_PREFIX = os.path.dirname(os.path.abspath(__file__)) + os.sep


def _frame_record(frame: traceback.FrameSummary) -> dict[str, Any]:
    item: dict[str, Any] = {
        "file": frame.filename,
        "line": int(frame.lineno),
        "function": frame.name,
    }
    if frame.line:
        item["code"] = frame.line.strip()
    return item


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


def _install_upstream_dispatch_memory_tracking(upstream: Any) -> None:
    if not _MEMORY_TRACKING or not _DISPATCH_MEMORY_TRACKING:
        _reset_dispatch_tracking_stats(enabled=False)
        return

    from torch.utils._python_dispatch import TorchDispatchMode

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
            observe = _monitor_activity.enabled()
            started = time.perf_counter_ns() if observe else 0
            result = func(*args, **(kwargs or {}))
            if observe:
                elapsed = time.perf_counter_ns() - started
                # CPU execution is a load proxy, not an A100 kernel duration.
                # Views/allocations alone do not imply an active GPU engine.
                name = str(func).split(".")[1]
                if name not in {"empty", "empty_strided", "view", "reshape", "detach", "alias", "as_strided"}:
                    tensors = list(_iter_tensors((args, kwargs or {}, result)))
                    io_bytes = sum(int(t.numel()) * int(t.element_size()) for t in tensors)
                    _monitor_activity.record(self.device_index, elapsed, io_bytes)
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
                elif tracker.has_allocation(data_ptr):
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

    upstream.FakeCudaTensor.__torch_function__ = _tracked_torch_function
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


def _enter_saved_tensor_hooks() -> None:
    global _saved_tensors_hooks_cm
    try:
        import torch

        hooks = torch.autograd.graph.saved_tensors_hooks(
            _pack_autograd_saved_tensor,
            _unpack_autograd_saved_tensor,
        )
        hooks.__enter__()
        _saved_tensors_hooks_cm = hooks
    except Exception:
        _saved_tensors_hooks_cm = None


def _close_saved_tensor_hooks() -> None:
    global _saved_tensors_hooks_cm
    cm = _saved_tensors_hooks_cm
    _saved_tensors_hooks_cm = None
    if cm is not None:
        cm.__exit__(None, None, None)


def _install_autograd_saved_tensor_tracking() -> None:
    _enter_saved_tensor_hooks()
    atexit.register(_close_saved_tensor_hooks)


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
        _enter_saved_tensor_hooks()


def _install_upstream_memory_category_hooks(upstream: Any, torch_mod: Any) -> None:
    module_cls = torch_mod.nn.Module
    for name in ("cuda", "to"):
        original = getattr(module_cls, name)

        @functools.wraps(original)
        def _tracked_module_transfer(self, *args, __original=original, **kwargs):
            result = __original(self, *args, **kwargs)
            _mark_module_memory_categories(result)
            return result

        setattr(module_cls, name, _tracked_module_transfer)

    original_register_parameter = upstream.register_parameter

    @functools.wraps(original_register_parameter)
    def _tracked_register_parameter(tensor):
        result = original_register_parameter(tensor)
        _mark_tensor_memory_category(tensor, "parameter")
        return result

    upstream.register_parameter = _tracked_register_parameter

    fake_tensor_cls = upstream.FakeCudaTensor
    original_backward = fake_tensor_cls.backward

    @functools.wraps(original_backward)
    def _tracked_backward(self, *args, **kwargs):
        try:
            return original_backward(self, *args, **kwargs)
        finally:
            _mark_registered_parameter_grads(upstream)

    fake_tensor_cls.backward = _tracked_backward

    original_get = fake_tensor_cls.grad.fget
    original_set = fake_tensor_cls.grad.fset

    def _tracked_grad_get(self):
        result = original_get(self)
        if result is not None:
            _mark_tensor_memory_category(result, "gradient")
        return result

    def _tracked_grad_set(self, value) -> None:
        original_set(self, value)
        if value is not None:
            _mark_tensor_memory_category(value, "gradient")

    fake_tensor_cls.grad = property(_tracked_grad_get, _tracked_grad_set)

    _install_optimizer_state_category_patch(torch_mod)


def _mark_registered_parameter_grads(upstream: Any) -> None:
    for param in list(upstream._REGISTERED_PARAMETERS):
        _mark_tensor_memory_category(param, "parameter")
        try:
            grad = param.grad
        except Exception:
            grad = None
        if grad is not None:
            _mark_tensor_memory_category(grad, "gradient")


def _install_optimizer_state_category_patch(torch_mod: Any) -> None:
    optim_mod = torch_mod.optim

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
        cls = getattr(optim_mod, name)
        # Classes such as AdamW inherit an already wrapped step.
        if "step" not in vars(cls):
            continue
        original_step = cls.step

        @functools.wraps(original_step)
        def _tracked_step(self, *args, __orig_step=original_step, **kwargs):
            try:
                return __orig_step(self, *args, **kwargs)
            finally:
                _mark_optimizer_state_tensors(self)

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
def _checked_ordinal(index: int) -> int:
    if index < 0 or index >= _NUM_DEVICES:
        import torch

        # C10_CUDA_CHECK raises c10::AcceleratorError, a RuntimeError subclass.
        raise getattr(torch, "AcceleratorError", RuntimeError)(
            f"CUDA error: invalid device ordinal "
            f"(requested cuda:{index}, available: {_NUM_DEVICES})"
        )
    return index


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


def _cuda_device_index(device: Any, *, optional: bool = False) -> int:
    """``torch.cuda._utils._get_device_index`` for fake devices.

    Integers pass through unchanged (negative ones included), non-CUDA
    devices raise ``ValueError``, and a device without an index resolves to
    the current device only when ``optional``.
    """
    import torch

    if isinstance(device, int):
        return device
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        if device.type != "cuda":
            raise ValueError(f"Expected a cuda device, but got: {device}")
        if device.index is not None:
            return int(device.index)
    if isinstance(device, _CudaDevice):
        return device.idx
    if optional:
        return int(_upstream_mod._CURRENT_DEVICE)
    raise ValueError(
        "Expected a torch.device with a specified index or an integer, "
        f"but got:{device}"
    )


def _exchange_device(device: int) -> int:
    """``torch.cuda._exchange_device``: select ``device``, return the previous one."""
    if device < 0:
        return -1
    previous = int(_upstream_mod._CURRENT_DEVICE)
    _upstream_mod._CURRENT_DEVICE = _checked_ordinal(int(device))
    return previous


class _CudaDevice:
    """``torch.cuda.device``: a no-op for ``None`` or a negative index."""

    def __init__(self, device: Any):
        self.idx = _cuda_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = _exchange_device(self.idx)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        self.idx = _exchange_device(self.prev_idx)
        return False


class _CudaDeviceOf(_CudaDevice):
    """``torch.cuda.device_of``: a no-op for objects not on a CUDA device."""

    def __init__(self, obj: Any):
        idx = obj.get_device() if obj.is_cuda else -1
        super().__init__(idx)


def _set_device(device: Any) -> None:
    """``torch.cuda.set_device``: a no-op for a negative index."""
    index = _cuda_device_index(device)
    if index >= 0:
        _upstream_mod._CURRENT_DEVICE = _checked_ordinal(index)


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
    if _torch_minor_version(torch_mod) < (2, 8):
        return

    def _fakegpu_compile(model: Any = None, *args: Any, **kwargs: Any) -> Any:
        if model is None:

            def _decorator(fn: Any) -> Any:
                return fn

            return _decorator
        return model

    torch_mod.compile = _fakegpu_compile
    torch_mod.compiler.compile = _fakegpu_compile


def _install_torch_accelerator_compat(torch_mod: Any) -> None:
    """Route the generic accelerator API through FakeGPU's CUDA surface.

    PyTorch 2.13 optimizers query ``torch.accelerator.current_stream()``
    before every step. On a CPU runner compiled with CUDA support, the
    unpatched implementation enters the real driver even though
    ``torch.cuda`` has already been redirected by FakeGPU.
    """

    accelerator = torch_mod.accelerator

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


def _hide_mps_backend(torch_mod: Any) -> None:
    """Report no MPS backend, as on a CUDA machine.

    Accelerate's ``PartialState.default_device`` and transformers'
    ``TrainingArguments.device`` pick MPS before CUDA when
    ``torch.backends.mps`` is available and built. On Apple Silicon a
    single-process ``Accelerator`` or ``Trainer`` would then train on MPS and
    bypass the simulated GPUs.
    """

    mps = getattr(torch_mod.backends, "mps", None)
    if mps is None:
        return

    # Cached like the original: torch._dynamo constant-folds
    # ``is_available.__wrapped__``.
    @functools.lru_cache
    def _is_available() -> bool:
        return False

    mps.is_available = _is_available
    mps.is_built = lambda: False


# ---------------------------------------------------------------------------
# Fake device properties
# ---------------------------------------------------------------------------


class _FakeCUuuid:
    """``torch._C._CUuuid``: ``bytes`` lists the 16 octets and ``str`` gives
    the 8-4-4-4-12 form without the ``GPU-`` prefix, as torch 2.9.1 prints a
    real GPU's UUID."""

    __slots__ = ("_value",)

    def __init__(self, value: str):
        import uuid as _uuid

        self._value = _uuid.UUID(value)

    @property
    def bytes(self) -> list[int]:
        return list(self._value.bytes)

    def __str__(self) -> str:
        return str(self._value)


def _device_identity(index: int, profile_id: str) -> tuple[_FakeCUuuid, int, int, int]:
    """The UUID and PCI address fakegpu-smi and NVML publish for ``cuda:index``.

    Under ``CUDA_VISIBLE_DEVICES`` the device keeps its physical index
    (FAKEGPU_PHYSICAL_INDICES), as its identity does on a real machine.
    """

    import socket

    from ._smi_state import _synthetic_pci_address, _synthetic_uuid

    physical = index
    try:
        indices = [int(item) for item in os.environ.get("FAKEGPU_PHYSICAL_INDICES", "").split(",") if item]
    except ValueError:
        indices = []
    if 0 <= index < len(indices):
        physical = indices[index]
    gpu_uuid = _synthetic_uuid(socket.gethostname(), physical, profile_id)
    domain, bus, device = _synthetic_pci_address(physical)
    return _FakeCUuuid(gpu_uuid[len("GPU-"):]), bus, device, domain


class _FakeDeviceProperties:
    """Mimics ``torch.cuda.get_device_properties()`` return value.

    Reads per-device profile data from ``_DEVICE_PROFILES``.
    """

    def __init__(self, index: int = 0):
        prof = _DEVICE_PROFILES[index]
        self.name = prof["name"]
        # torch's ``uuid`` (a ``_CUuuid``) and the PCI bus, device (slot) and
        # domain numbers of the bus id NVML and fakegpu-smi report.
        self.uuid, self.pci_bus_id, self.pci_device_id, self.pci_domain_id = _device_identity(
            index, str(prof.get("profile_id") or "")
        )
        self.major = prof["compute_major"]
        self.minor = prof["compute_minor"]
        self.total_memory = prof["total_memory"]
        # Reference profiles without a published SM count keep the A100 value.
        self.multi_processor_count = prof["sm_count"] or 108
        self.is_multi_gpu_board = False
        self.is_integrated = False
        self.max_threads_per_multi_processor = prof["max_threads_per_multiprocessor"]
        self.max_threads_per_block = 1024
        self.regs_per_block = prof["regs_per_block"]
        self.regs_per_multiprocessor = prof["regs_per_multiprocessor"]
        self.warp_size = 32
        # Fields of torch._C._CudaDeviceProperties; the clocks are in kHz.
        self.shared_memory_per_block = prof["shared_mem_per_block"]
        self.shared_memory_per_block_optin = prof["shared_mem_per_block_optin"]
        self.shared_memory_per_multiprocessor = prof["shared_mem_per_sm"]
        self.L2_cache_size = prof["l2_cache_bytes"]
        self.memory_bus_width = prof["memory_bus_width_bits"]
        self.clock_rate = prof["core_clock_mhz"] * 1000
        self.memory_clock_rate = prof["memory_clock_mhz"] * 1000
        # CUDA builds of PyTorch bind gcnArchName to the device name.
        self.gcnArchName = self.name

    def __repr__(self) -> str:
        return (
            f"_FakeDeviceProperties(name='{self.name}', major={self.major}, "
            f"minor={self.minor}, total_memory={self.total_memory // (1024**2)}MB, "
            f"multi_processor_count={self.multi_processor_count}, uuid={self.uuid}, "
            f"pci_bus_id={self.pci_bus_id}, pci_device_id={self.pci_device_id}, "
            f"pci_domain_id={self.pci_domain_id}, L2_cache_size={self.L2_cache_size // (1024**2)}MB)"
        )


# ---------------------------------------------------------------------------
# torch.cuda module‑level stubs
# ---------------------------------------------------------------------------
def _current_compute_major() -> int:
    index = int(_upstream_mod._CURRENT_DEVICE) if _upstream_mod is not None else 0
    if 0 <= index < len(_DEVICE_PROFILES):
        return int(_DEVICE_PROFILES[index]["compute_major"])
    return int(_COMPUTE_MAJOR)


def _stub_is_bf16_supported(including_emulation: bool = True) -> bool:
    """``torch.cuda.is_bf16_supported`` for the current fake device.

    PyTorch reports native support from compute capability 8.0. Below that,
    with emulation included (the default), it returns whether a bfloat16
    tensor can be created on the device, which CUDA devices allow.
    """
    return _current_compute_major() >= 8 or bool(including_emulation)


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
    installed_cuda_version = torch_mod.version.cuda
    return str(configured_cuda_version or installed_cuda_version or "12.1")


def _patch_hf_cuda_surface(torch_mod: Any) -> None:
    """Expose CUDA metadata expected by HuggingFace and Accelerate."""
    torch_mod.version.cuda = _reported_cuda_version(torch_mod)

    backends_cuda = torch_mod.backends.cuda
    backends_cuda.is_built = lambda: True

    matmul_backend = backends_cuda.matmul
    matmul_backend.allow_tf32 = False

    cudnn_backend = torch_mod.backends.cudnn
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

    # Writes to matmul.allow_tf32 cannot be intercepted, so the getter keeps
    # the stored precision only while it agrees with the backend flag.
    def _fake_get_float32_matmul_precision() -> str:
        stored = _matmul_precision["value"]
        tf32 = bool(matmul_backend.allow_tf32)
        if _precision_to_tf32[stored] == tf32:
            return stored
        return _tf32_to_precision[tf32]

    torch_mod.set_float32_matmul_precision = _fake_set_float32_matmul_precision
    torch_mod.get_float32_matmul_precision = _fake_get_float32_matmul_precision


def _patch_transformers_utils() -> None:
    """Patch transformers.utils helpers for LLaMA-Factory / LitGPT compatibility."""
    try:
        import transformers.utils.import_utils as _tu
    except ImportError:
        return

    if not getattr(_tu, "is_torch_cuda_available", lambda: False)():
        _tu.is_torch_cuda_available = lambda: True

    # Transformers caches these probes in some releases. Replacing them avoids
    # retaining a false result from imports that happened before FakeGPU
    # enabled CUDA semantics. Current releases answer bf16 with
    # torch.cuda.is_bf16_supported() and TF32 with the current device's
    # compute capability.
    _tu.is_torch_bf16_gpu_available = lambda: _stub_is_bf16_supported()
    _tu.is_torch_tf32_available = lambda: _current_compute_major() >= 8

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
    def _fake_fork_rng(
        devices=None,
        enabled: bool = True,
        _caller: str = "fork_rng",
        _devices_kw: str = "devices",
        device_type: str | None = None,
    ):
        # Signature of torch.random.fork_rng; fake CUDA RNG state is the CPU
        # generator's, so forking the CPU state covers every device.
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
    """Back CUDA RNG state with the CPU generator (upstream provides none)."""
    import torch.cuda.random as cuda_random

    for namespace in (torch_mod.cuda, cuda_random):
        namespace.get_rng_state = _stub_get_rng_state
        namespace.get_rng_state_all = _stub_get_rng_state_all
        namespace.set_rng_state = _stub_set_rng_state
        namespace.set_rng_state_all = _stub_set_rng_state_all
    fake_fork_rng = _build_fake_fork_rng(torch_mod)
    torch_mod.random.fork_rng = fake_fork_rng
    cuda_random.fork_rng = fake_fork_rng


def _patch_nccl_surface(torch_mod: Any) -> None:
    torch_mod.cuda.nccl.version = lambda: (2, 21, 5)



def _install_fakegpu_autocast(torch_mod: Any) -> None:
    _strict_compat = os.environ.get("FAKEGPU_STRICT_COMPAT", "1") != "0"
    if not _strict_compat:
        return

    _OrigAutocast = torch_mod.amp.autocast

    class _PatchedAutocast(_OrigAutocast):
        """Autocast wrapper that redirects fake CUDA autocast to CPU."""

        def __init__(
            self,
            device_type: str,
            dtype: Any = None,
            enabled: bool = True,
            cache_enabled: bool | None = None,
        ):
            requested_device = device_type
            actual_device = "cpu" if device_type == "cuda" else device_type
            self._fakegpu_requested_device_type = requested_device
            self._fakegpu_actual_device_type = actual_device
            # CPU autocast defaults to bfloat16; keep the CUDA default.
            if device_type == "cuda" and dtype is None:
                dtype = torch_mod.get_autocast_dtype("cuda")
            # The check torch.amp.autocast runs for an enabled CUDA autocast.
            if (
                device_type == "cuda"
                and enabled
                and dtype == torch_mod.bfloat16
                and not torch_mod.cuda.is_bf16_supported()
            ):
                raise RuntimeError(
                    "Current CUDA Device does not support bfloat16. "
                    "Please switch dtype to float16."
                )
            super().__init__(
                actual_device,
                dtype=dtype,
                enabled=enabled,
                cache_enabled=cache_enabled,
            )

        def __enter__(self):
            if self._fakegpu_requested_device_type == "cuda":
                # Track the CUDA state as CUDA autocast does, so a nested
                # autocast("cuda") without a dtype inherits this one and
                # torch.is_autocast_enabled("cuda") answers for the block.
                # Fake CUDA tensors compute on the CPU, so the CUDA flag
                # changes no kernel.
                self._fakegpu_prev_cuda_state = (
                    torch_mod.is_autocast_enabled("cuda"),
                    torch_mod.get_autocast_dtype("cuda"),
                )
                torch_mod.set_autocast_enabled("cuda", self._enabled)
                torch_mod.set_autocast_dtype("cuda", self.fast_dtype)
            return super().__enter__()

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self._fakegpu_requested_device_type == "cuda":
                enabled, dtype = self._fakegpu_prev_cuda_state
                torch_mod.set_autocast_enabled("cuda", enabled)
                torch_mod.set_autocast_dtype("cuda", dtype)
            return super().__exit__(exc_type, exc_val, exc_tb)

    class _PatchedCudaAmpAutocast(_PatchedAutocast):
        """Legacy ``torch.cuda.amp.autocast(enabled, dtype, cache_enabled)``."""

        def __init__(
            self,
            enabled: bool = True,
            dtype: Any = torch_mod.float16,
            cache_enabled: bool = True,
        ):
            super().__init__(
                "cuda", dtype=dtype, enabled=enabled, cache_enabled=cache_enabled
            )

    # ``torch.autocast`` is a separate binding of the same class; Accelerate
    # and most training code use it.
    torch_mod.autocast = _PatchedAutocast
    torch_mod.amp.autocast = _PatchedAutocast
    torch_mod.cuda.amp.autocast = _PatchedCudaAmpAutocast
    # custom_fwd/custom_bwd enter the module-level name.
    torch_mod.amp.autocast_mode.autocast = _PatchedAutocast


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


def _new_link_traffic() -> Any:
    from .link_traffic import LinkTraffic
    from .platforms import topology_from_environment

    profile_ids = [str(profile.get("profile_id") or "") for profile in _DEVICE_PROFILES]
    return LinkTraffic(lambda: topology_from_environment(profile_ids, os.environ))


def _install_to_overloads(upstream: Any, torch_mod: Any) -> None:
    """Accept every ``Tensor.to`` and ``Module.to`` overload on fake devices.

    The upstream conversions only look for a target device: ``to(device,
    dtype)`` left the dtype positional after the device was moved into a
    keyword (a TypeError), ``to(other)`` dropped ``other.dtype``, and
    ``Module.to`` dropped the dtype whenever the target was CUDA.
    """

    tensor_to = upstream._tensor_to

    @functools.wraps(tensor_to)
    def _tensor_to_any_overload(self: Any, *args: Any, **kwargs: Any) -> Any:
        if args and not isinstance(args[0], torch_mod.dtype):
            first, rest = args[0], args[1:]
            if isinstance(first, torch_mod.Tensor):
                # to(other, non_blocking=False, copy=False)
                parsed = {"device": first.device, "dtype": first.dtype}
                parsed.update(zip(("non_blocking", "copy"), rest))
            else:
                # to(device=None, dtype=None, non_blocking=False, copy=False)
                parsed = {"device": first}
                parsed.update(zip(("dtype", "non_blocking", "copy"), rest))
            kwargs = {**parsed, **kwargs}
            args = ()
        return tensor_to(self, *args, **kwargs)

    upstream._tensor_to = _tensor_to_any_overload

    module_to = upstream._module_to

    @functools.wraps(module_to)
    def _module_to_any_overload(self: Any, *args: Any, **kwargs: Any) -> Any:
        first = args[0] if args else None
        if isinstance(first, torch_mod.Tensor):
            # to(tensor, non_blocking=False)
            target, dtype = first.device, first.dtype
        elif isinstance(first, torch_mod.dtype):
            # to(dtype, non_blocking=False)
            target, dtype = None, first
        else:
            # to(device=None, dtype=None, non_blocking=False)
            target = first if args else kwargs.get("device")
            dtype = args[1] if len(args) > 1 else kwargs.get("dtype")
        memory_format = kwargs.get("memory_format")
        if upstream._normalize_cuda_device(target, allow_none=True) is None:
            return module_to(self, *args, **kwargs)
        # nn.Module.to converts floating-point tensors to ``dtype`` and moves
        # them in one step; convert first so no copy in the old dtype lands
        # on the device.
        conversion = {}
        if dtype is not None:
            conversion["dtype"] = dtype
        if memory_format is not None:
            conversion["memory_format"] = memory_format
        if conversion:
            module_to(self, **conversion)
        return module_to(self, target)

    upstream._module_to = _module_to_any_overload


def _install_module_moves_in_place(upstream: Any, torch_mod: Any) -> None:
    """Keep each ``Parameter`` object when a module moves to a fake device.

    ``nn.Module._apply`` moves a parameter by assigning ``param.data``, so an
    optimizer built before ``model.cuda()`` (as Accelerate's ``prepare``
    expects) keeps updating the model, and a module already on the target
    device is left as is. The upstream conversion created new parameters.
    """
    try:
        from torch.utils import swap_tensors
    except ImportError:  # torch < 2.3: a moved module gets new Parameter objects
        swap_tensors = None

    fake_cls = upstream.FakeCudaTensor

    def _module_cuda_impl(module: Any, device: Any = None, _memo: Any = None) -> Any:
        target = upstream._normalize_module_device(device)
        index = target.index or 0
        if _memo is None:
            _memo = {"parameters": {}, "buffers": {}}

        for child in module.children():
            _module_cuda_impl(child, target, _memo)

        for name, param in list(module._parameters.items()):
            if param is None:
                continue
            moved = _memo["parameters"].get(id(param))
            if moved is None:
                if isinstance(param, fake_cls) and param.device_index == index:
                    moved = param
                else:
                    converted = param.detach().clone().requires_grad_(param.requires_grad)
                    moved = torch_mod.nn.Parameter(
                        upstream.wrap_tensor(converted, device_index=index),
                        requires_grad=param.requires_grad,
                    )
                    moved.raw_data = converted
                    moved.device_index = index
                    if param.grad is not None:
                        moved.grad = upstream.wrap_tensor(
                            param.grad.detach(), device_index=index
                        )
                    if swap_tensors is not None:
                        try:
                            # ``param`` becomes the moved parameter in place.
                            swap_tensors(param, moved)
                            moved = param
                        except RuntimeError:
                            pass  # e.g. weak references: fall back to a new object
                    upstream.register_parameter(moved)
                _memo["parameters"][id(param)] = moved
            module._parameters[name] = moved

        for name, buf in list(module._buffers.items()):
            if buf is None:
                continue
            converted = _memo["buffers"].get(id(buf))
            if converted is None:
                if isinstance(buf, fake_cls) and buf.device_index == index:
                    converted = buf
                else:
                    converted = upstream.wrap_tensor(buf.detach().clone(), device_index=index)
                    converted.raw_data = buf.detach().clone()
                    converted.device_index = index
                _memo["buffers"][id(buf)] = converted
            module._buffers[name] = converted

        return module

    upstream._module_cuda_impl = _module_cuda_impl


def _install_stream_event_semantics(upstream: Any, torch_mod: Any) -> None:
    """Give fake streams and events the attributes and checks torch has.

    Timing is not modeled, so ``elapsed_time`` stays 0 ms, but it raises the
    ``ValueError`` CUDA events raise when timing is off or a side was never
    recorded (c10/cuda/CUDAEvent.h).
    """

    class _Stream(upstream._FakeStream):
        device_type = 1  # c10::DeviceType::CUDA

        def __init__(
            self,
            device: Any = None,
            priority: int = 0,
            stream_id: int | None = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(device, priority, stream_id, **kwargs)
            self.priority = int(priority)
            self._fakegpu_context: list[tuple[Any, int]] = []

        @property
        def device(self) -> Any:
            return torch_mod.device("cuda", self.device_index)

        def record_event(self, event: Any = None) -> Any:
            if event is None:
                event = _Event()
            event.record(self)
            return event

        def __enter__(self) -> "_Stream":
            # torch.Stream.__enter__ makes the stream current on its device
            # and selects that device until __exit__.
            previous_device = int(upstream._CURRENT_DEVICE)
            previous_stream = upstream._current_stream(self.device_index)
            self._fakegpu_context.append((previous_stream, previous_device))
            _set_stream(self)
            return self

        def __exit__(self, *args: Any) -> None:
            previous_stream, previous_device = self._fakegpu_context.pop()
            _set_stream(previous_stream)
            upstream._CURRENT_DEVICE = previous_device
            return None

    def _set_stream(stream: Any) -> None:
        # torch.cuda.set_stream also selects the stream's device
        # (THCPModule_setStream_wrap); None is a no-op.
        if stream is None:
            return
        upstream._CURRENT_DEVICE = _checked_ordinal(int(stream.device_index))
        upstream._set_stream(stream)

    def _stream(stream: Any) -> Any:
        # The real StreamContext: a no-op for None, restores both devices'
        # streams on exit.
        return torch_mod.cuda.StreamContext(stream)

    class _Event(upstream._FakeEvent):
        def __init__(
            self,
            enable_timing: bool = False,
            blocking: bool = False,
            interprocess: bool = False,
            external: bool = False,
        ) -> None:
            super().__init__(enable_timing, blocking, interprocess)
            self._fakegpu_timing = bool(enable_timing)
            self._fakegpu_device: Any = None

        @property
        def device(self) -> Any:
            # None until the event is first recorded.
            return self._fakegpu_device

        def record(self, stream: Any = None) -> None:
            if stream is None:
                stream = torch_mod.cuda.current_stream()
            self._fakegpu_device = torch_mod.device("cuda", stream.device_index)

        def elapsed_time(self, end_event: Any) -> float:
            if not (self._fakegpu_timing and getattr(end_event, "_fakegpu_timing", False)):
                raise ValueError(
                    "Both events must be created with argument 'enable_timing=True'."
                )
            if self._fakegpu_device is None or end_event._fakegpu_device is None:
                raise ValueError(
                    "Both events must be recorded before calculating elapsed time."
                )
            return 0.0

    upstream._FakeStream = _Stream
    upstream._FakeEvent = _Event
    torch_mod.cuda.Stream = _Stream
    torch_mod.cuda.Event = _Event
    torch_mod.cuda.set_stream = _set_stream
    torch_mod.cuda.stream = _stream


def _install_transfer_recording(upstream: Any, torch_mod: Any) -> None:
    """Record host-device and peer copies in ``_link_traffic``.

    Covers ``Tensor.to/.cuda/.cpu``, ``Module.to/.cuda``,
    ``torch.tensor``/``torch.as_tensor``/``torch.asarray``, ``copy_``
    between devices, and ``item``/``tolist`` reads.
    """

    fake_cls = upstream.FakeCudaTensor

    def endpoint(value: Any) -> Any:
        """GPU index, ``None`` for host memory, ``False`` for anything else."""
        if isinstance(value, fake_cls):
            return int(value.device_index)
        if isinstance(value, torch_mod.Tensor) and value.device.type == "cpu":
            return None
        return False

    def tensor_bytes(tensor: Any) -> int | None:
        raw = upstream.unwrap_tensor(tensor)
        try:
            return int(raw.numel()) * int(raw.element_size())
        except Exception:
            return None

    def record_bytes(source: Any, target: Any, nbytes: int | None) -> None:
        traffic = _link_traffic
        if (
            traffic is None
            or nbytes is None
            or source is False
            or target is False
            or source == target
        ):
            return
        traffic.record(source, target, nbytes)

    def record(source: Any, target: Any, tensor: Any) -> None:
        record_bytes(source, target, tensor_bytes(tensor))

    def module_tensors(module: Any) -> list[Any]:
        seen: set[int] = set()
        found = []
        for tensor in (*module.parameters(), *module.buffers()):
            if tensor is not None and id(tensor) not in seen:
                seen.add(id(tensor))
                found.append(tensor)
        return found

    def record_module(module: Any, convert: Any) -> Any:
        # Moves keep the Parameter objects, so read the sources first.
        before = [(endpoint(t), tensor_bytes(t)) for t in module_tensors(module)]
        result = convert()
        for (source, nbytes), new in zip(before, module_tensors(module)):
            record_bytes(source, endpoint(new), nbytes)
        return result

    tensor_to = upstream._tensor_to

    @functools.wraps(tensor_to)
    def _counted_tensor_to(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = tensor_to(self, *args, **kwargs)
        if result is not self and isinstance(result, torch_mod.Tensor):
            record(endpoint(self), endpoint(result), self)
        return result

    upstream._tensor_to = _counted_tensor_to

    tensor_cpu = upstream._tensor_cpu

    @functools.wraps(tensor_cpu)
    def _counted_tensor_cpu(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = tensor_cpu(self, *args, **kwargs)
        if isinstance(self, fake_cls):
            record(int(self.device_index), None, self)
        return result

    upstream._tensor_cpu = _counted_tensor_cpu

    module_to = upstream._module_to
    module_cuda = upstream._module_cuda

    @functools.wraps(module_to)
    def _counted_module_to(self: Any, *args: Any, **kwargs: Any) -> Any:
        target = kwargs.get("device")
        if target is None and args:
            target = args[0]
        if upstream._normalize_cuda_device(target, allow_none=True) is None:
            # Moves to the host go through Tensor.to, which records them.
            return module_to(self, *args, **kwargs)
        return record_module(self, lambda: module_to(self, *args, **kwargs))

    @functools.wraps(module_cuda)
    def _counted_module_cuda(self: Any, device: Any = None) -> Any:
        return record_module(self, lambda: module_cuda(self, device))

    upstream._module_to = _counted_module_to
    upstream._module_cuda = _counted_module_cuda

    torch_mod.Tensor.to = upstream._tensor_to
    torch_mod.Tensor.cpu = upstream._tensor_cpu
    torch_mod.nn.Module.to = upstream._module_to
    torch_mod.nn.Module.cuda = upstream._module_cuda

    for name in ("tensor", "as_tensor", "asarray"):
        factory = getattr(torch_mod, name)

        def _counted_factory(*args: Any, _factory: Any = factory, **kwargs: Any) -> Any:
            result = _factory(*args, **kwargs)
            if isinstance(result, torch_mod.Tensor):
                data = args[0] if args else kwargs.get("data", kwargs.get("obj"))
                source = endpoint(data) if isinstance(data, torch_mod.Tensor) else None
                record(source, endpoint(result), result)
            return result

        setattr(torch_mod, name, functools.wraps(factory)(_counted_factory))

    torch_function = fake_cls.__dict__.get("__torch_function__")
    inner = getattr(torch_function, "__func__", None)
    if inner is not None:
        copy_ = torch_mod.Tensor.copy_
        reads = (torch_mod.Tensor.item, torch_mod.Tensor.tolist)

        def _counted_torch_function(
            cls: Any,
            func: Any,
            types: Any,
            args: tuple[Any, ...] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> Any:
            result = inner(cls, func, types, args, kwargs)
            if func is copy_ and len(args) >= 2:
                record(endpoint(args[1]), endpoint(args[0]), args[1])
            elif args and any(func is read for read in reads):
                record(endpoint(args[0]), None, args[0])
            return result

        fake_cls.__torch_function__ = classmethod(_counted_torch_function)


def _install_device_index_validation(upstream: Any, torch_mod: Any) -> Any:
    """Raise CUDA's "invalid device ordinal" error for out-of-range devices.

    Returns the bounds-checked replacement for ``upstream._normalize_device_index``.
    """
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
        return _checked_ordinal(int(normalized.index))

    upstream._normalize_device_index = _checked_normalize_device_index

    # Device selection follows torch.cuda: set_device, device and device_of
    # ignore negative indices, and _exchange_device switches the device.
    torch_mod.cuda.set_device = _set_device
    upstream.set_device = _set_device
    torch_mod.cuda.device = _CudaDevice
    torch_mod.cuda.device_of = _CudaDeviceOf
    torch_mod.cuda._exchange_device = _exchange_device
    torch_mod.cuda._maybe_exchange_device = _exchange_device

    def _synchronize(device: Any = None) -> None:
        # torch.cuda.synchronize enters torch.cuda.device(device) first.
        with _CudaDevice(device):
            return None

    torch_mod.cuda.synchronize = _synchronize

    # Override _normalize_cuda_device to validate bounds for factory functions.
    # The upstream version doesn't check _NUM_DEVICES, so torch.randn(device="cuda:99")
    # would silently create a tensor on a non-existent device.
    def _bounds_checked_normalize_cuda(device, *, allow_none=False):
        result = _orig_normalize_cuda_device(device, allow_none=allow_none)
        if result is not None:
            _checked_ordinal(result.index)
        return result

    upstream._normalize_cuda_device = _bounds_checked_normalize_cuda
    return _checked_normalize_device_index


def _install_wrap_tensor_tracking(upstream: Any) -> None:
    """Hook ``upstream.wrap_tensor`` for bounds checks and memory tracking."""
    _orig_wrap_tensor = upstream.wrap_tensor

    def _hooked_wrap_tensor(t, device_index=None):
        _checked_ordinal(
            upstream._CURRENT_DEVICE if device_index is None else int(device_index)
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


def _install_dynamo_friendly_tree_map(upstream: Any, torch_mod: Any) -> None:
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


def _install_profiled_device_queries(upstream: Any, torch_mod: Any) -> None:
    """Answer device name/capability/properties and peer access per profile."""

    # As in torch.cuda, name and capability come from the properties, and an
    # index outside the devices fails with AssertionError("Invalid device id").
    def _profiled_get_device_name(device=None):
        return _profiled_get_device_properties(device).name

    def _profiled_get_device_capability(device=None):
        properties = _profiled_get_device_properties(device)
        return properties.major, properties.minor

    def _profiled_get_device_properties(device=None):
        idx = _cuda_device_index(device, optional=True)
        if idx < 0 or idx >= _NUM_DEVICES:
            raise AssertionError("Invalid device id")
        return _FakeDeviceProperties(idx)

    torch_mod.cuda.get_device_name = _profiled_get_device_name
    torch_mod.cuda.get_device_capability = _profiled_get_device_capability
    torch_mod.cuda.get_device_properties = _profiled_get_device_properties

    # Peer access follows the modeled NVLink/PCIe topology (see
    # fakegpu.link_topology); without a platform every pair has access.
    def _topology_can_device_access_peer(device, peer_device):
        source = _cuda_device_index(device, optional=True)
        target = _cuda_device_index(peer_device)
        if source < 0 or source >= _NUM_DEVICES:
            raise AssertionError("Invalid device id")
        if target < 0 or target >= _NUM_DEVICES:
            raise AssertionError("Invalid peer device id")
        from .platforms import topology_from_environment

        topology = topology_from_environment(
            [str(profile.get("profile_id") or "") for profile in _DEVICE_PROFILES],
            os.environ,
        )
        return topology.can_access_peer(source, target)

    torch_mod.cuda.can_device_access_peer = _topology_can_device_access_peer
    upstream.can_device_access_peer = _topology_can_device_access_peer


def _install_tracked_memory_api(torch_mod: Any) -> None:
    """Serve the ``torch.cuda`` memory API from ``_memory_tracker``."""

    def _stats_device(device):
        # CUDACachingAllocator::assertValidDevice() rejects devices the
        # allocator has no stats for instead of reporting zeros.
        index = _normalize_device_index(device)
        if index < 0 or index >= _NUM_DEVICES:
            raise RuntimeError(
                f"Invalid device argument {index}: did you call init?"
            )
        return index

    def _tracked_memory_allocated(device=None):
        return _memory_tracker.memory_allocated(_stats_device(device))

    def _tracked_max_memory_allocated(device=None):
        return _memory_tracker.max_memory_allocated(_stats_device(device))

    def _tracked_memory_reserved(device=None):
        return _memory_tracker.memory_reserved(_stats_device(device))

    def _tracked_max_memory_reserved(device=None):
        return _memory_tracker.max_memory_reserved(_stats_device(device))

    def _tracked_mem_get_info(device=None):
        # cudaMemGetInfo runs under a CUDAGuard for the requested device.
        return _memory_tracker.mem_get_info(
            _checked_ordinal(_normalize_device_index(device))
        )

    def _tracked_empty_cache():
        _memory_tracker.empty_cache()

    def _tracked_reset_peak_memory_stats(device=None):
        _memory_tracker.reset_peak(_stats_device(device))

    def _tracked_reset_accumulated_memory_stats(device=None):
        _memory_tracker.reset_accumulated(_stats_device(device))

    def _tracked_memory_stats(device=None):
        return _memory_tracker.memory_stats(_stats_device(device))

    def _tracked_memory_snapshot(mempool_id=None, include_traces=True):
        # Only the default pool exists, so a private pool id has no segments.
        if mempool_id is not None and tuple(mempool_id[:2]) != (0, 0):
            return []
        return _memory_tracker.allocator_snapshot()

    # Both public namespaces must expose the same tracker and aliases.
    memory_api = {
        "empty_cache": _tracked_empty_cache,
        "memory_allocated": _tracked_memory_allocated,
        "max_memory_allocated": _tracked_max_memory_allocated,
        "memory_reserved": _tracked_memory_reserved,
        "max_memory_reserved": _tracked_max_memory_reserved,
        "memory_cached": _tracked_memory_reserved,
        "max_memory_cached": _tracked_max_memory_reserved,
        "mem_get_info": _tracked_mem_get_info,
        "reset_peak_memory_stats": _tracked_reset_peak_memory_stats,
        "reset_max_memory_allocated": _tracked_reset_peak_memory_stats,
        "reset_max_memory_cached": _tracked_reset_peak_memory_stats,
        "reset_accumulated_memory_stats": _tracked_reset_accumulated_memory_stats,
        "memory_stats": _tracked_memory_stats,
        "memory_snapshot": _tracked_memory_snapshot,
    }
    for namespace in (torch_mod.cuda, torch_mod.cuda.memory):
        for name, function in memory_api.items():
            setattr(namespace, name, function)

    # torch.cuda.memory_stats_as_nested_dict() reads this binding directly.
    def _tracked_cuda_memory_stats(device):
        return _memory_tracker.memory_stats_as_nested_dict(_stats_device(device))

    torch_mod._C._cuda_memoryStats = _tracked_cuda_memory_stats

    # torch.cuda.set/get_per_process_memory_fraction() and
    # get_allocator_backend() validate in Python and then call these.
    def _tracked_set_memory_fraction(fraction, device):
        _memory_tracker.set_memory_fraction(float(fraction), int(device))

    def _tracked_get_memory_fraction(device):
        return _memory_tracker.get_memory_fraction(int(device))

    torch_mod._C._cuda_setMemoryFraction = _tracked_set_memory_fraction
    torch_mod._C._cuda_getMemoryFraction = _tracked_get_memory_fraction
    torch_mod._C._cuda_getAllocatorBackend = lambda: "native"


def _install_fake_grad_scaler(torch_mod: Any) -> None:
    class _FakeGradScaler(torch_mod.amp.GradScaler):
        def __init__(self, *args: Any, **kwargs: Any):
            kwargs.setdefault("enabled", False)
            super().__init__(*args, **kwargs)

    torch_mod.cuda.amp.GradScaler = _FakeGradScaler  # type: ignore[attr-defined]
    torch_mod.amp.GradScaler = _FakeGradScaler  # type: ignore[attr-defined]


def _install_cross_device_checks(torch_mod: Any) -> None:
    import torch.nn.functional as F

    for op_name in (
        "matmul",
        "mm",
        "bmm",
        "cat",
        "stack",
        "where",
        "addmm",
        "addcmul",
        "addcdiv",
    ):
        setattr(
            torch_mod,
            op_name,
            _wrap_multi_tensor_op(getattr(torch_mod, op_name), torch_mod),
        )

    for op_name in (
        # losses
        "cross_entropy",
        "mse_loss",
        "nll_loss",
        "binary_cross_entropy",
        # layers
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "embedding",
        "batch_norm",
        "layer_norm",
    ):
        setattr(F, op_name, _wrap_multi_tensor_op(getattr(F, op_name), torch_mod))

    for dunder in _BINARY_DUNDER_TORCH_OPS:
        setattr(
            torch_mod.Tensor,
            dunder,
            _wrap_tensor_binary_op(
                getattr(torch_mod.Tensor, dunder), dunder, torch_mod
            ),
        )


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
    global _memory_tracker, _link_traffic

    # Let analysis helpers distinguish FakeGPU's simulated availability from a
    # real CUDA runtime.  In particular, PyTorch 2.13 FakeTensor tracing may
    # initialize a physical CUDA context when is_available() reports true.
    torch_mod.cuda._fakegpu_simulated = True

    _install_device_index_validation(upstream, torch_mod)
    _install_stream_event_semantics(upstream, torch_mod)
    _install_module_moves_in_place(upstream, torch_mod)

    if _MEMORY_TRACKING:
        per_device_bytes = [p["total_memory"] for p in _DEVICE_PROFILES]
        _memory_tracker = _DeviceMemoryTracker(per_device_bytes)
        _install_autograd_saved_tensor_tracking()

    _install_wrap_tensor_tracking(upstream)
    _install_upstream_dispatch_memory_tracking(upstream)
    # Patched before transfer recording, which must wrap its torch.asarray.
    _patch_upstream_fakecuda_tensor_compat(upstream, torch_mod)
    # Copies take the modeled time of their route over the platform links
    # (fakegpu.link_traffic); recorded only, nothing waits.  Installed before
    # the category hooks so they wrap the recording Module.to/cuda.
    _link_traffic = _new_link_traffic()
    _install_to_overloads(upstream, torch_mod)
    _install_transfer_recording(upstream, torch_mod)
    _install_upstream_memory_category_hooks(upstream, torch_mod)
    _patch_upstream_all_gather_object(upstream, torch_mod)
    _install_torch_accelerator_compat(torch_mod)
    _hide_mps_backend(torch_mod)
    _install_dynamo_friendly_tree_map(upstream, torch_mod)

    fake_ddp_cls = torch_mod.nn.parallel.DistributedDataParallel
    if not hasattr(fake_ddp_cls, "_get_active_ddp_module"):
        fake_ddp_cls._get_active_ddp_module = staticmethod(lambda: None)

    torch_mod.Tensor.pin_memory = _patched_tensor_pin_memory  # type: ignore[assignment]
    _install_compile_compat_shim(torch_mod)

    _install_profiled_device_queries(upstream, torch_mod)

    # Compute-capability-aware bf16 check (upstream always returns True)
    torch_mod.cuda.is_bf16_supported = _stub_is_bf16_supported
    _patch_hf_cuda_surface(torch_mod)

    if _memory_tracker is not None:
        _install_tracked_memory_api(torch_mod)

    _install_fakegpu_autocast(torch_mod)
    _install_fake_grad_scaler(torch_mod)

    if _CROSS_DEVICE_CHECK:
        _install_cross_device_checks(torch_mod)

    _patch_cuda_rng_surface(torch_mod)
    _patch_nccl_surface(torch_mod)
    _patch_fsdp_device_handling()
    _patch_fsdp_runtime_compat(upstream.FakeCudaTensor)
    _patch_transformers_utils()
    _patch_accelerate_utils()

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
        _memory_tracker, \
        _link_traffic

    os.environ["FAKEGPU_RUNTIME"] = "fakecuda"
    import torch
    import torch.cuda

    _refresh_runtime_profile_state(num_devices=num_devices, device_name=device_name)

    if _patched:
        _upstream_mod._NUM_DEVICES = _NUM_DEVICES
        _upstream_mod._DEVICE_NAME = _DEVICE_NAME
        if _upstream_mod._CURRENT_DEVICE >= _NUM_DEVICES:
            _upstream_mod._CURRENT_DEVICE = 0
        torch.cuda._cached_device_count = _NUM_DEVICES
        if _MEMORY_TRACKING:
            _memory_tracker = _DeviceMemoryTracker(
                [profile["total_memory"] for profile in _DEVICE_PROFILES]
            )
        _reset_dispatch_tracking_stats(
            enabled=bool(_MEMORY_TRACKING and _DISPATCH_MEMORY_TRACKING)
        )
        _link_traffic = _new_link_traffic()
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
    _patch_result = PatchResult(
        backend="upstream",
        num_devices=_NUM_DEVICES,
        device_name=_DEVICE_NAME,
    )
    # The publisher writes its first state synchronously on start().
    _refresh_smi_publisher()
    _patched = True
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
