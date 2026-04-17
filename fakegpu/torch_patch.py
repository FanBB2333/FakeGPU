"""Monkeypatch ``torch.cuda`` so CUDA-dependent code runs on CPU.

On systems without an NVIDIA GPU (or with a CPU-only PyTorch build), this
module transparently redirects all CUDA device references to the CPU backend.
Tensor operations, module transfers, and factory functions are intercepted at
the Python level so that the C++ dispatcher never sees a ``cuda`` device.

Usage::

    import fakegpu
    fakegpu.init(runtime="fakecuda")
    # or: fakegpu.patch_torch()

    import torch
    # Everything below "just works" on CPU.
    x = torch.randn(3, 3, device="cuda")
    model = torch.nn.Linear(3, 3).cuda()
    y = model(x)
"""

from __future__ import annotations

import atexit
import functools
import importlib
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Any

_patched = False
_patch_result: "PatchResult | None" = None

# ---------------------------------------------------------------------------
# Configuration – mirrors the active FakeGPU profile when available.
# ---------------------------------------------------------------------------

_PROFILE_CC: dict[str, tuple[int, int]] = {
    "gtx980": (5, 2),
    "p100": (6, 0),
    "v100": (7, 0),
    "t4": (7, 5),
    "a40": (8, 6),
    "a100": (8, 0),
    "a100-1g": (8, 0),
    "h100": (9, 0),
    "l40s": (8, 9),
    "b100": (11, 0),
    "b200": (11, 0),
}

_PROFILE_NAMES: dict[str, str] = {
    "gtx980": "NVIDIA GeForce GTX 980",
    "p100": "Tesla P100-PCIE-16GB",
    "v100": "Tesla V100-SXM2-32GB",
    "t4": "Tesla T4",
    "a40": "NVIDIA A40",
    "a100": "NVIDIA A100-SXM4-80GB",
    "a100-1g": "NVIDIA A100-SXM4-1GB",
    "h100": "NVIDIA H100 80GB HBM3",
    "l40s": "NVIDIA L40S",
    "b100": "NVIDIA B100",
    "b200": "NVIDIA B200",
}

_PROFILE_TOTAL_MEMORY: dict[str, int] = {
    "gtx980": 4 * 1024**3,
    "p100": 16 * 1024**3,
    "v100": 32 * 1024**3,
    "t4": 16 * 1024**3,
    "a40": 48 * 1024**3,
    "a100": 80 * 1024**3,
    "a100-1g": 1 * 1024**3,
    "h100": 80 * 1024**3,
    "l40s": 48 * 1024**3,
    "b100": 192 * 1024**3,
    "b200": 192 * 1024**3,
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


def _resolve_per_device_profiles(num_devices: int | None = None) -> list[dict[str, Any]]:
    """Resolve per-device profile info from FAKEGPU_PROFILES.

    Returns a list of dicts, one per device, each with keys:
      'profile_id', 'name', 'total_memory', 'compute_major', 'compute_minor'
    """
    profiles_env = os.environ.get("FAKEGPU_PROFILES", "")
    target_count = int(num_devices if num_devices is not None else os.environ.get("FAKEGPU_DEVICE_COUNT", "8"))
    result: list[dict[str, Any]] = []

    if profiles_env:
        for spec in profiles_env.split(","):
            spec = spec.strip()
            if not spec:
                continue
            parts = spec.split(":")
            pid = parts[0].strip().lower()
            count = int(parts[1]) if len(parts) > 1 and parts[1].strip().isdigit() else 1
            for _ in range(count):
                cc = _PROFILE_CC.get(pid, (8, 0))
                result.append({
                    "profile_id": pid,
                    "name": _PROFILE_NAMES.get(pid, "NVIDIA A100-SXM4-80GB"),
                    "total_memory": _PROFILE_TOTAL_MEMORY.get(pid, 80 * 1024**3),
                    "compute_major": cc[0],
                    "compute_minor": cc[1],
                })

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

_current_device: int = 0


def _refresh_runtime_profile_state(*, num_devices: int | None = None, device_name: str | None = None) -> None:
    """Refresh per-device profile globals after runtime options change."""
    global _NUM_DEVICES, _DEVICE_PROFILES, _DEVICE_NAME, _COMPUTE_MAJOR, _COMPUTE_MINOR, _TOTAL_MEMORY

    if num_devices is not None:
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
# Device registry: tracks which fake CUDA device each tensor lives on.
# Key = storage data_ptr (stable across views/slices)
# Value = logical device index
# ---------------------------------------------------------------------------

_CROSS_DEVICE_CHECK = os.environ.get("FAKEGPU_CROSS_DEVICE_CHECK", "1") != "0"

_device_registry: dict[int, int] = {}


def _register_tensor_device(tensor: Any, device_index: int) -> None:
    """Register a tensor's storage in the device registry and memory tracker."""
    try:
        dp = tensor.untyped_storage().data_ptr()
        if dp != 0:
            _device_registry[dp] = device_index
            _register_tensor_for_memory_tracking(tensor, device_index)
    except (MemoryError, RuntimeError):
        raise  # Re-raise OOM and related errors
    except Exception:
        pass


def _get_tensor_device(tensor: Any) -> int | None:
    """Look up the fake CUDA device index for a tensor, or None if untracked."""
    try:
        dp = tensor.untyped_storage().data_ptr()
        return _device_registry.get(dp)
    except Exception:
        return None


def _check_same_device(*tensors: Any) -> None:
    """Raise RuntimeError if tensors span multiple fake CUDA devices."""
    if not _CROSS_DEVICE_CHECK:
        return
    import torch

    first_dev: int | None = None
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            continue
        dev = _get_tensor_device(t)
        if dev is None:
            continue  # untracked tensor (e.g. pure CPU) — skip
        if first_dev is None:
            first_dev = dev
        elif dev != first_dev:
            raise RuntimeError(
                f"Expected all tensors to be on the same device, "
                f"but found at least two devices, cuda:{first_dev} and cuda:{dev}!"
            )


def _wrap_multi_tensor_op(orig_fn: Any) -> Any:
    """Wrap a torch function to check device consistency of tensor args."""

    @functools.wraps(orig_fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        import torch

        tensors = []
        for a in args:
            if isinstance(a, torch.Tensor):
                tensors.append(a)
            elif isinstance(a, (list, tuple)):
                for item in a:
                    if isinstance(item, torch.Tensor):
                        tensors.append(item)
        for v in kwargs.values():
            if isinstance(v, torch.Tensor):
                tensors.append(v)
        if len(tensors) >= 2:
            _check_same_device(*tensors)
        return orig_fn(*args, **kwargs)

    return wrapper


def _wrap_tensor_binary_op(orig_fn: Any) -> Any:
    """Wrap a Tensor binary method (e.g. __add__) to check cross-device."""

    @functools.wraps(orig_fn)
    def wrapper(self: Any, other: Any) -> Any:
        import torch

        if isinstance(other, torch.Tensor):
            _check_same_device(self, other)
        return orig_fn(self, other)

    return wrapper


def _extract_cuda_device_index(device: Any) -> int | None:
    """Extract CUDA device index from a device-like argument, or None if not CUDA."""
    import torch

    if isinstance(device, int):
        return device
    if isinstance(device, str):
        try:
            device = torch.device(device)
        except Exception:
            return None
    if isinstance(device, torch.device) and device.type == "cuda":
        return device.index if device.index is not None else _current_device
    return None


# ---------------------------------------------------------------------------
# Memory tracking: per-device memory accounting.
# ---------------------------------------------------------------------------

_MEMORY_TRACKING = os.environ.get("FAKEGPU_MEMORY_TRACKING", "1") != "0"


class _DeviceMemoryTracker:
    """Track per-device memory allocations in the torch_patch layer."""

    def __init__(self, per_device_bytes: list[int]):
        self._total = list(per_device_bytes)
        self._used = [0] * len(per_device_bytes)
        self._peak = [0] * len(per_device_bytes)
        self._alloc_calls = [0] * len(per_device_bytes)
        self._free_calls = [0] * len(per_device_bytes)
        # data_ptr -> (device_index, nbytes)
        self._allocs: dict[int, tuple[int, int]] = {}

    def allocate(self, data_ptr: int, nbytes: int, device: int) -> None:
        """Register allocation. Raise OutOfMemoryError if exceeds limit."""
        import torch

        if device < 0 or device >= len(self._total):
            return
        if data_ptr in self._allocs:
            return  # already tracked
        if self._used[device] + nbytes > self._total[device]:
            free = self._total[device] - self._used[device]
            raise torch.cuda.OutOfMemoryError(
                f"CUDA out of memory. Tried to allocate "
                f"{nbytes / 2**20:.2f} MiB. "
                f"GPU {device} has a total capacity of "
                f"{self._total[device] / 2**30:.2f} GiB "
                f"of which {free / 2**30:.2f} GiB is free."
            )
        self._allocs[data_ptr] = (device, nbytes)
        self._used[device] += nbytes
        self._peak[device] = max(self._peak[device], self._used[device])
        self._alloc_calls[device] += 1

    def release(self, data_ptr: int) -> None:
        """Unregister allocation."""
        rec = self._allocs.pop(data_ptr, None)
        if rec:
            dev, nbytes = rec
            self._used[dev] = max(0, self._used[dev] - nbytes)
            self._free_calls[dev] += 1

    def memory_allocated(self, device: int) -> int:
        if device < 0 or device >= len(self._used):
            return 0
        return self._used[device]

    def max_memory_allocated(self, device: int) -> int:
        if device < 0 or device >= len(self._peak):
            return 0
        return self._peak[device]

    def mem_get_info(self, device: int) -> tuple[int, int]:
        if device < 0 or device >= len(self._total):
            return (0, 0)
        free = self._total[device] - self._used[device]
        return (max(0, free), self._total[device])

    def reset_peak(self, device: int) -> None:
        if 0 <= device < len(self._peak):
            self._peak[device] = self._used[device]


# ---------------------------------------------------------------------------
# Architecture name lookup (mirrors C++ gpu_profile.cpp)
# ---------------------------------------------------------------------------

_CC_TO_ARCH: dict[tuple[int, int], str] = {
    (5, 2): "Maxwell",
    (6, 0): "Pascal",
    (7, 0): "Volta",
    (7, 5): "Turing",
    (8, 0): "Ampere",
    (8, 6): "Ampere",
    (8, 9): "Ada",
    (9, 0): "Hopper",
    (10, 0): "Blackwell",
    (11, 0): "Blackwell",
}


def _arch_name(major: int, minor: int) -> str:
    """Return the architecture name for a compute capability."""
    exact = _CC_TO_ARCH.get((major, minor))
    if exact:
        return exact
    # Fallback: match by major only
    for (ma, _mi), name in _CC_TO_ARCH.items():
        if ma == major:
            return name
    return "Unknown"


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
        peak_pct = (100.0 * peak / total) if total > 0 else 0.0

        alloc = tracker._alloc_calls[i]
        free = tracker._free_calls[i]

        lines.append(f" Device {i}: {name} ({arch}, cc {cc_major}.{cc_minor})")
        lines.append(f"   Memory: {_fmt_bytes(peak)} / {_fmt_bytes(total)} peak ({peak_pct:.1f}%)")
        lines.append(f"   Alloc: {alloc} calls | Free: {free} calls")
        lines.append("------------------------------------------------------")

    lines.append("======================================================")
    lines.append("")

    sys.stderr.write("\n".join(lines))
    sys.stderr.flush()


# Initialized later in patch() after _DEVICE_PROFILES is finalized
_memory_tracker: _DeviceMemoryTracker | None = None


import weakref


def _register_tensor_for_memory_tracking(tensor: Any, device_index: int) -> None:
    """Register a tensor's memory and set up GC cleanup via weakref."""
    if _memory_tracker is None or not _MEMORY_TRACKING:
        return
    try:
        storage = tensor.untyped_storage()
        dp = storage.data_ptr()
        nbytes = storage.nbytes()
        if dp == 0 or nbytes == 0:
            return
        _memory_tracker.allocate(dp, nbytes, device_index)

        # Set up weakref callback to release memory when tensor is GC'd.
        # We weakref the storage, not the tensor, because multiple tensor
        # views can share one storage.
        def _release_cb(data_ptr=dp):
            if _memory_tracker is not None:
                _memory_tracker.release(data_ptr)
            _device_registry.pop(data_ptr, None)

        # Only add weakref if not already tracked (avoid double-counting)
        weakref.finalize(storage, _release_cb)
    except (MemoryError, RuntimeError):
        raise  # Re-raise OOM and related errors
    except Exception:
        pass


@dataclass(frozen=True)
class PatchResult:
    backend: str
    num_devices: int
    device_name: str

# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


def _normalize_device(device: Any) -> Any:
    """Redirect ``cuda`` devices to ``cpu``, raising for invalid ordinals."""
    import torch

    if device is None:
        return device
    if isinstance(device, int):
        # bare int in a context expecting a CUDA ordinal → validate then cpu
        if device >= _NUM_DEVICES:
            raise RuntimeError(
                f"CUDA error: invalid device ordinal "
                f"(requested cuda:{device}, available: {_NUM_DEVICES})"
            )
        return torch.device("cpu")
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device) and device.type == "cuda":
        idx = device.index if device.index is not None else _current_device
        if idx >= _NUM_DEVICES:
            raise RuntimeError(
                f"CUDA error: invalid device ordinal "
                f"(requested cuda:{idx}, available: {_NUM_DEVICES})"
            )
        return torch.device("cpu")
    return device


def _normalize_device_index(device: Any) -> int:
    import torch

    if device is None:
        return _current_device
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        return device.index or _current_device
    return _current_device


def _device_kwarg_wrapper(fn: Any) -> Any:
    """Wrap a callable so that its ``device`` keyword is redirected."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        target_dev: int | None = None
        if "device" in kwargs and kwargs["device"] is not None:
            target_dev = _extract_cuda_device_index(kwargs["device"])
            kwargs["device"] = _normalize_device(kwargs["device"])
        result = fn(*args, **kwargs)
        if target_dev is not None:
            _register_tensor_device(result, target_dev)
        return result

    return wrapper


# ---------------------------------------------------------------------------
# Tensor.to / Tensor.cuda wrappers
# ---------------------------------------------------------------------------

_orig_tensor_to: Any = None
_orig_tensor_cuda: Any = None


def _patched_tensor_to(self: Any, *args: Any, **kwargs: Any) -> Any:
    import torch

    # Extract target CUDA device index before normalization
    target_dev: int | None = None
    if "device" in kwargs:
        target_dev = _extract_cuda_device_index(kwargs["device"])
        kwargs["device"] = _normalize_device(kwargs["device"])

    if args:
        first = args[0]
        if isinstance(first, str):
            if target_dev is None:
                target_dev = _extract_cuda_device_index(first)
            args = (_normalize_device(first),) + args[1:]
        elif isinstance(first, torch.device) and first.type == "cuda":
            if target_dev is None:
                target_dev = _extract_cuda_device_index(first)
            args = (torch.device("cpu"),) + args[1:]
        elif isinstance(first, int) and len(args) >= 2 and isinstance(args[1], torch.dtype):
            if target_dev is None:
                target_dev = first
            args = (torch.device("cpu"),) + args[1:]

    result = _orig_tensor_to(self, *args, **kwargs)
    if target_dev is not None:
        _register_tensor_device(result, target_dev)
    return result


def _patched_tensor_cuda(
    self: Any,
    device: Any = None,
    non_blocking: bool = False,
    memory_format: Any = None,
) -> Any:
    import torch

    target_dev = _extract_cuda_device_index(device) if device is not None else _current_device
    if memory_format is not None and memory_format is not torch.preserve_format:
        result = self.contiguous(memory_format=memory_format)
    else:
        result = self
    if target_dev is not None:
        _register_tensor_device(result, target_dev)
    return result


# ---------------------------------------------------------------------------
# Module.to / Module.cuda wrappers
# ---------------------------------------------------------------------------

_orig_module_to: Any = None


def _patched_module_to(self: Any, *args: Any, **kwargs: Any) -> Any:
    import torch

    target_dev: int | None = None

    if "device" in kwargs:
        target_dev = _extract_cuda_device_index(kwargs["device"])
        kwargs["device"] = _normalize_device(kwargs["device"])

    if args:
        first = args[0]
        if isinstance(first, str):
            target_dev = _extract_cuda_device_index(first) if target_dev is None else target_dev
            args = (_normalize_device(first),) + args[1:]
        elif isinstance(first, torch.device) and first.type == "cuda":
            target_dev = _extract_cuda_device_index(first) if target_dev is None else target_dev
            args = (torch.device("cpu"),) + args[1:]

    result = _orig_module_to(self, *args, **kwargs)

    # Register all parameters/buffers in the device registry
    if target_dev is not None:
        for param in result.parameters():
            _register_tensor_device(param.data, target_dev)
        for buf in result.buffers():
            _register_tensor_device(buf, target_dev)

    return result


def _patched_module_cuda(self: Any, device: Any = None) -> Any:
    # Module is already on CPU; no transfer needed.
    return self


# ---------------------------------------------------------------------------
# Fake CUDA Stream / Event
# ---------------------------------------------------------------------------


class _FakeStream:
    """Minimal stub for ``torch.cuda.Stream``."""

    def __init__(self, device: Any = None, priority: int = 0, **kwargs: Any):
        self.device_index = _normalize_device_index(device)
        self.cuda_stream = 0

    def synchronize(self) -> None:
        pass

    def wait_stream(self, stream: Any) -> None:
        pass

    def wait_event(self, event: Any) -> None:
        pass

    def record_event(self, event: Any = None) -> Any:
        if event is None:
            event = _FakeEvent()
        return event

    def query(self) -> bool:
        return True

    def __enter__(self) -> "_FakeStream":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _FakeEvent:
    """Minimal stub for ``torch.cuda.Event``."""

    def __init__(self, enable_timing: bool = False, blocking: bool = False, interprocess: bool = False):
        self._time: float = 0.0

    def record(self, stream: Any = None) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def wait(self, stream: Any = None) -> None:
        pass

    def query(self) -> bool:
        return True

    def elapsed_time(self, other: "_FakeEvent") -> float:
        return 0.0


# ---------------------------------------------------------------------------
# Fake device properties
# ---------------------------------------------------------------------------


class _FakeDeviceProperties:
    """Mimics ``torch.cuda.get_device_properties()`` return value."""

    def __init__(self, index: int = 0):
        self.name = _DEVICE_NAME
        self.major = _COMPUTE_MAJOR
        self.minor = _COMPUTE_MINOR
        self.total_memory = _TOTAL_MEMORY
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


def _stub_is_available() -> bool:
    return True


def _stub_is_bf16_supported(device: Any = None) -> bool:
    return _COMPUTE_MAJOR >= 8


def _stub_device_count() -> int:
    return _NUM_DEVICES


def _stub_current_device() -> int:
    return _current_device


def _stub_set_device(device: Any) -> None:
    global _current_device
    import torch

    if isinstance(device, torch.device):
        idx = device.index or 0
    elif isinstance(device, str):
        idx = torch.device(device).index or 0
    elif isinstance(device, int):
        idx = device
    else:
        idx = 0

    if idx < 0 or idx >= _NUM_DEVICES:
        raise RuntimeError(
            f"CUDA error: invalid device ordinal "
            f"(requested {idx}, available: {_NUM_DEVICES})"
        )
    _current_device = idx


def _stub_get_device_name(device: Any = None) -> str:
    return _DEVICE_NAME


def _stub_get_device_capability(device: Any = None) -> tuple[int, int]:
    return (_COMPUTE_MAJOR, _COMPUTE_MINOR)


def _stub_get_device_properties(device: Any = None) -> _FakeDeviceProperties:
    idx = 0
    if device is not None:
        import torch

        if isinstance(device, torch.device):
            idx = device.index or 0
        elif isinstance(device, int):
            idx = device
    if idx < 0 or idx >= _NUM_DEVICES:
        raise RuntimeError(
            f"CUDA error: invalid device ordinal "
            f"(requested {idx}, available: {_NUM_DEVICES})"
        )
    return _FakeDeviceProperties(idx)


def _stub_get_arch_list() -> list[str]:
    return ["sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]


def _stub_mem_get_info(device: Any = None) -> tuple[int, int]:
    return (_TOTAL_MEMORY, _TOTAL_MEMORY)


def _stub_synchronize(device: Any = None) -> None:
    pass


def _stub_empty_cache() -> None:
    pass


def _stub_memory_allocated(device: Any = None) -> int:
    return 0


def _stub_memory_reserved(device: Any = None) -> int:
    return 0


def _stub_max_memory_allocated(device: Any = None) -> int:
    return 0


def _stub_max_memory_reserved(device: Any = None) -> int:
    return 0


def _stub_memory_cached(device: Any = None) -> int:
    return 0


def _stub_max_memory_cached(device: Any = None) -> int:
    return 0


def _stub_reset_peak_memory_stats(device: Any = None) -> None:
    pass


def _stub_reset_max_memory_allocated(device: Any = None) -> None:
    pass


def _stub_reset_max_memory_cached(device: Any = None) -> None:
    pass


def _stub_reset_accumulated_memory_stats(device: Any = None) -> None:
    pass


def _stub_memory_stats(device: Any = None) -> dict[str, Any]:
    return {}


def _stub_memory_summary(device: Any = None, abbreviated: bool = False) -> str:
    return "FakeGPU: no real CUDA memory to report.\n"


def _stub_memory_snapshot() -> list[Any]:
    return []


def _stub_manual_seed(seed: int) -> None:
    import torch

    torch.random.default_generator.manual_seed(int(seed))


def _stub_manual_seed_all(seed: int) -> None:
    _stub_manual_seed(seed)


def _stub_seed() -> int:
    import torch

    return int(torch.random.default_generator.seed())


def _stub_seed_all() -> None:
    _stub_seed()


def _stub_initial_seed() -> int:
    import torch

    return int(torch.random.default_generator.initial_seed())


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
    index = _current_device if _current_device < len(states) else 0
    _set_cpu_rng_state(states[index])


def _stub_is_initialized() -> bool:
    return True


def _stub_lazy_init() -> None:
    pass


def _stub_init() -> None:
    pass


def _stub_ipc_collect() -> None:
    pass


def _stub_can_device_access_peer(device: int, peer_device: int) -> bool:
    return True


def _stub_get_gencode_flags() -> str:
    return ""


def _stub_cudart() -> Any:
    return None


# ---------------------------------------------------------------------------
# Wrap non-default-stream context manager
# ---------------------------------------------------------------------------

class _FakeStreamCtx:
    """Replacement for ``torch.cuda.stream(s)``."""
    def __init__(self, stream: Any) -> None:
        pass
    def __enter__(self) -> "_FakeStreamCtx":
        return self
    def __exit__(self, *args: Any) -> None:
        pass


class _FakeCudaGenerator:
    """CPU-backed stand-in for ``torch.cuda.default_generators`` entries."""

    def __init__(self, index: int):
        self.index = index

    def get_state(self):
        return _cpu_rng_state()

    def set_state(self, new_state: Any) -> None:
        _set_cpu_rng_state(new_state)

    def manual_seed(self, seed: int):
        _stub_manual_seed(seed)
        return self

    def seed(self) -> int:
        return _stub_seed()

    def initial_seed(self) -> int:
        return _stub_initial_seed()


def _make_default_generators() -> tuple[_FakeCudaGenerator, ...]:
    return tuple(_FakeCudaGenerator(i) for i in range(_NUM_DEVICES))


# ---------------------------------------------------------------------------
# Shared compatibility helpers
# ---------------------------------------------------------------------------


def _install_legacy_cuda_types(torch: Any, *, device: str) -> None:
    """Install ``torch.cuda.FloatTensor``-style factories."""

    _LEGACY_TYPES = {
        "FloatTensor": torch.float32,
        "DoubleTensor": torch.float64,
        "HalfTensor": torch.float16,
        "BFloat16Tensor": torch.bfloat16,
        "IntTensor": torch.int32,
        "LongTensor": torch.int64,
        "ShortTensor": torch.int16,
        "ByteTensor": torch.uint8,
        "CharTensor": torch.int8,
        "BoolTensor": torch.bool,
    }

    def _make_legacy_factory(dtype: Any) -> type:
        _dtype = dtype

        class _LegacyCudaTensor:
            def __new__(cls, *args: Any, **kwargs: Any) -> Any:
                kwargs = dict(kwargs)
                kwargs["device"] = device
                if args and not isinstance(args[0], int):
                    return torch.tensor(args[0], dtype=_dtype, **kwargs)
                return torch.empty(*args, dtype=_dtype, **kwargs)

        return _LegacyCudaTensor

    for tname, dt in _LEGACY_TYPES.items():
        setattr(torch.cuda, tname, _make_legacy_factory(dt))


def _maybe_enable_custom_torch_fakegpu(
    torch: Any,
    *,
    num_devices: int | None = None,
    device_name: str | None = None,
) -> bool:
    """Enable the installed custom torch fake-CUDA backend when available."""

    os.environ["TORCH_FAKEGPU_ENABLE"] = "1"
    os.environ["FAKEGPU_TORCH_ENABLE"] = "1"
    if num_devices is not None:
        os.environ["TORCH_FAKEGPU_DEVICE_COUNT"] = str(num_devices)
    if device_name is not None:
        os.environ["TORCH_FAKEGPU_DEVICE_NAME"] = device_name

    try:
        torch_fakegpu = importlib.import_module("torch.fakegpu")
    except Exception:
        return False

    if num_devices is not None and hasattr(torch_fakegpu, "_NUM_DEVICES"):
        torch_fakegpu._NUM_DEVICES = int(num_devices)
    if device_name is not None and hasattr(torch_fakegpu, "_DEVICE_NAME"):
        torch_fakegpu._DEVICE_NAME = str(device_name)

    if not hasattr(torch_fakegpu, "enable"):
        return False

    torch_fakegpu.enable()
    _install_legacy_cuda_types(torch, device="cuda")
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def patch(*, num_devices: int | None = None, device_name: str | None = None) -> PatchResult:
    """Apply monkey-patches to ``torch`` so CUDA code runs transparently on CPU.

    Safe to call multiple times; only the first call has effect.

    Parameters
    ----------
    num_devices:
        Number of fake CUDA devices to expose.  Defaults to ``$FAKEGPU_DEVICE_COUNT`` or 8.
    device_name:
        Name reported by ``torch.cuda.get_device_name()``.
    Returns
    -------
    PatchResult
        Describes whether the installed custom torch backend was used or the
        standalone CPU-backed fallback was activated.
    """

    global _patched, _NUM_DEVICES, _DEVICE_NAME, _patch_result
    if _patched:
        return _patch_result

    import torch
    import torch.cuda
    import torch.nn

    _refresh_runtime_profile_state(num_devices=num_devices, device_name=device_name)

    if _maybe_enable_custom_torch_fakegpu(
        torch,
        num_devices=_NUM_DEVICES,
        device_name=_DEVICE_NAME,
    ):
        _patched = True
        _patch_result = PatchResult(
            backend="custom_torch",
            num_devices=_NUM_DEVICES,
            device_name=_DEVICE_NAME,
        )
        warnings.warn(
            "fakegpu.torch_patch: enabled the installed custom torch fake-CUDA backend.",
            stacklevel=2,
        )
        return _patch_result

    # ---- torch.cuda module functions ----
    torch.cuda.is_available = _stub_is_available
    torch.cuda.device_count = _stub_device_count
    torch.cuda.current_device = _stub_current_device
    torch.cuda.set_device = _stub_set_device
    torch.cuda.get_device_name = _stub_get_device_name
    torch.cuda.get_device_capability = _stub_get_device_capability
    torch.cuda.get_device_properties = _stub_get_device_properties
    torch.cuda.get_arch_list = _stub_get_arch_list
    torch.cuda.mem_get_info = _stub_mem_get_info
    torch.cuda.synchronize = _stub_synchronize
    torch.cuda.empty_cache = _stub_empty_cache
    torch.cuda.memory_allocated = _stub_memory_allocated
    torch.cuda.memory_reserved = _stub_memory_reserved
    torch.cuda.max_memory_allocated = _stub_max_memory_allocated
    torch.cuda.max_memory_reserved = _stub_max_memory_reserved
    torch.cuda.memory_cached = _stub_memory_cached
    torch.cuda.max_memory_cached = _stub_max_memory_cached
    torch.cuda.reset_peak_memory_stats = _stub_reset_peak_memory_stats
    torch.cuda.reset_max_memory_allocated = _stub_reset_max_memory_allocated
    torch.cuda.reset_max_memory_cached = _stub_reset_max_memory_cached
    torch.cuda.reset_accumulated_memory_stats = _stub_reset_accumulated_memory_stats
    torch.cuda.memory_stats = _stub_memory_stats
    torch.cuda.memory_summary = _stub_memory_summary
    torch.cuda.memory_snapshot = _stub_memory_snapshot
    torch.cuda.manual_seed = _stub_manual_seed
    torch.cuda.manual_seed_all = _stub_manual_seed_all
    torch.cuda.seed = _stub_seed
    torch.cuda.seed_all = _stub_seed_all
    torch.cuda.initial_seed = _stub_initial_seed
    torch.cuda.get_rng_state = _stub_get_rng_state
    torch.cuda.get_rng_state_all = _stub_get_rng_state_all
    torch.cuda.set_rng_state = _stub_set_rng_state
    torch.cuda.set_rng_state_all = _stub_set_rng_state_all
    torch.cuda.ipc_collect = _stub_ipc_collect
    torch.cuda.can_device_access_peer = _stub_can_device_access_peer
    torch.cuda.get_gencode_flags = _stub_get_gencode_flags
    torch.cuda.default_generators = _make_default_generators()

    # ---- Initialize memory tracker ----
    global _memory_tracker
    if _MEMORY_TRACKING:
        per_device_bytes = [p["total_memory"] for p in _DEVICE_PROFILES]
        _memory_tracker = _DeviceMemoryTracker(per_device_bytes)

    # ---- Register atexit terminal summary ----
    atexit.register(_dump_terminal_summary)

    # Override memory stubs to use tracker
    if _memory_tracker is not None:
        _tracker = _memory_tracker  # local ref for closures

        def _tracked_memory_allocated(device=None):
            idx = _normalize_device_index(device)
            return _tracker.memory_allocated(idx)

        def _tracked_max_memory_allocated(device=None):
            idx = _normalize_device_index(device)
            return _tracker.max_memory_allocated(idx)

        def _tracked_mem_get_info(device=None):
            idx = _normalize_device_index(device)
            return _tracker.mem_get_info(idx)

        def _tracked_reset_peak_memory_stats(device=None):
            idx = _normalize_device_index(device)
            _tracker.reset_peak(idx)

        torch.cuda.memory_allocated = _tracked_memory_allocated
        torch.cuda.max_memory_allocated = _tracked_max_memory_allocated
        torch.cuda.mem_get_info = _tracked_mem_get_info
        torch.cuda.reset_peak_memory_stats = _tracked_reset_peak_memory_stats

    # Internal helpers PyTorch relies on
    torch.cuda._is_compiled = lambda: True
    torch.cuda._lazy_init = _stub_lazy_init
    torch.cuda.is_initialized = _stub_is_initialized
    torch.cuda.init = _stub_init
    torch.cuda._initialized = True
    torch.cuda._cached_device_count = _NUM_DEVICES

    # ---- Patch internal helpers that check for CUDA compilation ----
    torch.cuda._exchange_device = lambda device: 0
    torch.cuda._get_device = lambda device: 0
    if hasattr(torch.cuda, "_maybe_exchange_device"):
        torch.cuda._maybe_exchange_device = lambda device: 0

    # ---- Patch torch.load to validate + normalize map_location ----
    _orig_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        # Extract map_location from kwargs or positional args
        ml = kwargs.get("map_location", None)
        if ml is None and len(args) >= 2:
            ml = args[1]

        # Validate CUDA device index in map_location
        if ml is not None:
            if isinstance(ml, str):
                try:
                    dev = torch.device(ml)
                    if dev.type == "cuda":
                        idx = dev.index if dev.index is not None else _current_device
                        if idx >= _NUM_DEVICES:
                            raise RuntimeError(
                                f"CUDA error: invalid device ordinal "
                                f"(map_location={ml}, available: {_NUM_DEVICES})"
                            )
                except RuntimeError:
                    raise
                except Exception:
                    pass
            elif isinstance(ml, torch.device) and ml.type == "cuda":
                idx = ml.index if ml.index is not None else _current_device
                if idx >= _NUM_DEVICES:
                    raise RuntimeError(
                        f"CUDA error: invalid device ordinal "
                        f"(map_location={ml}, available: {_NUM_DEVICES})"
                    )

        # Normalize device in map_location
        if "map_location" in kwargs:
            ml_val = kwargs["map_location"]
            if isinstance(ml_val, (str, torch.device)):
                kwargs["map_location"] = _normalize_device(ml_val)
        elif len(args) >= 2:
            ml_val = args[1]
            if isinstance(ml_val, (str, torch.device)):
                args = (args[0], _normalize_device(ml_val)) + args[2:]

        return _orig_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load

    # ---- torch._C stubs for internal imports ----
    # Several PyTorch subsystems (torch._dynamo, torch._inductor, torch.cuda.graphs)
    # do top-level ``from torch._C import _cuda_*`` which fails on CPU-only builds.
    _c_stubs = {
        "_cuda_getCurrentRawStream": lambda device_index=0: 0,
        "_cuda_isCurrentStreamCapturing": lambda: False,
        "_cuda_getDeviceCount": lambda: _NUM_DEVICES,
        "_cuda_getDevice": lambda: _current_device,
        "_cuda_setDevice": lambda device: None,
        "_cuda_init": lambda: None,
        "_cuda_emptyCache": lambda: None,
        "_cuda_resetPeakMemoryStats": lambda device: None,
        "_cuda_memoryStats": lambda device: {},
        "_cuda_memorySnapshot": lambda: [],
    }
    for attr, stub in _c_stubs.items():
        setattr(torch._C, attr, stub)

    # ---- Fix stale module-level bindings from 'from torch._C import _cuda_*' ----
    # On CPU-only builds, these are dummy classes that raise RuntimeError.
    # The modules are already imported by the time patch() runs, so we must
    # patch the local bindings directly.
    import torch.cuda.graphs as _graphs
    if hasattr(_graphs, "_cuda_isCurrentStreamCapturing"):
        _graphs._cuda_isCurrentStreamCapturing = _c_stubs["_cuda_isCurrentStreamCapturing"]

    import torch.cuda.memory as _memory
    import torch.cuda.random as _random
    _mem_stubs = {
        "_cuda_CUDAAllocator": lambda: None,
        "_cuda_beginAllocateCurrentThreadToPool": lambda *a: None,
        "_cuda_beginAllocateToPool": lambda *a: None,
        "_cuda_endAllocateToPool": lambda *a: None,
        "_cuda_releasePool": lambda *a: None,
    }
    for attr, stub in _mem_stubs.items():
        if hasattr(_memory, attr):
            setattr(_memory, attr, stub)
        if not hasattr(torch._C, attr):
            setattr(torch._C, attr, stub)

    _random.manual_seed = _stub_manual_seed
    _random.manual_seed_all = _stub_manual_seed_all
    _random.seed = _stub_seed
    _random.seed_all = _stub_seed_all
    _random.initial_seed = _stub_initial_seed
    _random.get_rng_state = _stub_get_rng_state
    _random.get_rng_state_all = _stub_get_rng_state_all
    _random.set_rng_state = _stub_set_rng_state
    _random.set_rng_state_all = _stub_set_rng_state_all

    # Fake CUDAGraph class if missing
    if not hasattr(torch._C, "_CUDAGraph"):
        class _FakeCUDAGraph:
            def __init__(self): pass
            def capture_begin(self, *a, **kw): pass
            def capture_end(self): pass
            def replay(self): pass
            def reset(self): pass
            def pool(self): return 0
        torch._C._CUDAGraph = _FakeCUDAGraph  # type: ignore[attr-defined]

    if not hasattr(torch._C, "_graph_pool_handle"):
        torch._C._graph_pool_handle = lambda: 0  # type: ignore[attr-defined]

    # Safe stubs that should avoid AttributeError if code touches them
    if hasattr(torch.cuda, "is_bf16_supported"):
        torch.cuda.is_bf16_supported = _stub_is_bf16_supported

    # ---- Stream / Event ----
    torch.cuda.Stream = _FakeStream  # type: ignore[misc]
    torch.cuda.Event = _FakeEvent  # type: ignore[misc]
    torch.cuda.stream = _FakeStreamCtx  # type: ignore[misc]
    torch.cuda.current_stream = lambda device=None: _FakeStream(device=device)
    torch.cuda.default_stream = lambda device=None: _FakeStream(device=device)
    torch.cuda.set_stream = lambda stream: _stub_set_device(getattr(stream, "device_index", 0))

    # ---- Tensor.to / Tensor.cuda ----
    global _orig_tensor_to, _orig_tensor_cuda
    _orig_tensor_to = torch.Tensor.to
    _orig_tensor_cuda = torch.Tensor.cuda
    torch.Tensor.to = _patched_tensor_to  # type: ignore[assignment]
    torch.Tensor.cuda = _patched_tensor_cuda  # type: ignore[assignment]

    # ---- Propagate device registry through clone/contiguous/detach ----
    _orig_tensor_clone = torch.Tensor.clone

    def _patched_tensor_clone(self, *args, **kwargs):
        result = _orig_tensor_clone(self, *args, **kwargs)
        dev = _get_tensor_device(self)
        if dev is not None:
            _register_tensor_device(result, dev)
        return result

    torch.Tensor.clone = _patched_tensor_clone

    _orig_tensor_contiguous = torch.Tensor.contiguous

    def _patched_tensor_contiguous(self, *args, **kwargs):
        result = _orig_tensor_contiguous(self, *args, **kwargs)
        dev = _get_tensor_device(self)
        if dev is not None:
            _register_tensor_device(result, dev)
        return result

    torch.Tensor.contiguous = _patched_tensor_contiguous

    # detach() is a method in PyTorch 2.x, not a property
    _orig_tensor_detach_fn = torch.Tensor.detach

    def _patched_tensor_detach(self):
        result = _orig_tensor_detach_fn(self)
        dev = _get_tensor_device(self)
        if dev is not None:
            _register_tensor_device(result, dev)
        return result

    torch.Tensor.detach = _patched_tensor_detach

    # ---- Cross-device validation patches ----
    if _CROSS_DEVICE_CHECK:
        import torch.nn.functional as F

        # Multi-input torch functions
        _MULTI_TENSOR_OPS = [
            "matmul", "mm", "bmm", "cat", "stack", "where",
            "addmm", "addcmul", "addcdiv",
        ]
        for op_name in _MULTI_TENSOR_OPS:
            orig = getattr(torch, op_name, None)
            if orig is not None:
                setattr(torch, op_name, _wrap_multi_tensor_op(orig))

        # Loss functions
        _LOSS_OPS = ["cross_entropy", "mse_loss", "nll_loss", "binary_cross_entropy"]
        for op_name in _LOSS_OPS:
            orig = getattr(F, op_name, None)
            if orig is not None:
                setattr(F, op_name, _wrap_multi_tensor_op(orig))

        # Also wrap F.linear for model forward cross-device checks
        _FUNCTIONAL_OPS = ["linear", "conv1d", "conv2d", "conv3d",
                           "embedding", "batch_norm", "layer_norm"]
        for op_name in _FUNCTIONAL_OPS:
            orig = getattr(F, op_name, None)
            if orig is not None:
                setattr(F, op_name, _wrap_multi_tensor_op(orig))

        # Tensor binary dunder methods
        _BINARY_DUNDERS = [
            "__add__", "__radd__", "__sub__", "__rsub__",
            "__mul__", "__rmul__", "__truediv__", "__rtruediv__",
            "__matmul__", "__rmatmul__",
        ]
        for dunder in _BINARY_DUNDERS:
            orig = getattr(torch.Tensor, dunder, None)
            if orig is not None:
                setattr(torch.Tensor, dunder, _wrap_tensor_binary_op(orig))

    # ---- Module.to / Module.cuda ----
    global _orig_module_to
    _orig_module_to = torch.nn.Module.to
    torch.nn.Module.to = _patched_module_to  # type: ignore[assignment]
    torch.nn.Module.cuda = _patched_module_cuda  # type: ignore[assignment]

    # ---- Tensor creation functions (redirect device='cuda' → 'cpu') ----
    _FACTORY_NAMES = [
        "tensor",
        "as_tensor",
        "zeros",
        "ones",
        "empty",
        "full",
        "rand",
        "randn",
        "randint",
        "arange",
        "linspace",
        "logspace",
        "eye",
        "zeros_like",
        "ones_like",
        "empty_like",
        "full_like",
        "rand_like",
        "randn_like",
        "randint_like",
        "scalar_tensor",
        "sparse_coo_tensor",
    ]
    for name in _FACTORY_NAMES:
        orig = getattr(torch, name, None)
        if orig is not None:
            setattr(torch, name, _device_kwarg_wrapper(orig))

    # ---- torch.cuda.FloatTensor and friends (legacy) ----
    # Code like ``torch.cuda.FloatTensor(3, 4)`` should produce a CPU tensor.
    _install_legacy_cuda_types(torch, device="cpu")

    # ---- GradScaler passthrough ----
    try:
        from torch.amp import GradScaler as _RealGradScaler

        class _FakeGradScaler(_RealGradScaler):
            def __init__(self, *args: Any, **kwargs: Any):
                kwargs.setdefault("enabled", False)
                super().__init__(*args, **kwargs)

        torch.cuda.amp.GradScaler = _FakeGradScaler  # type: ignore[attr-defined]
        torch.amp.GradScaler = _FakeGradScaler  # type: ignore[attr-defined]
    except Exception:
        pass

    # ---- Autocast dtype validation (defense-in-depth) ----
    _strict_compat = os.environ.get("FAKEGPU_STRICT_COMPAT", "1") != "0"

    if _strict_compat and hasattr(torch.amp, "autocast"):
        _OrigAutocast = torch.amp.autocast

        class _PatchedAutocast(_OrigAutocast):
            def __enter__(self):
                if (
                    getattr(self, "device_type", None) == "cuda"
                    and getattr(self, "fast_dtype", None) == torch.bfloat16
                    and _COMPUTE_MAJOR < 8
                ):
                    raise RuntimeError(
                        f"Current CUDA Device does not support bfloat16. "
                        f"Please switch dtype to float16 "
                        f"(compute capability {_COMPUTE_MAJOR}.{_COMPUTE_MINOR}, "
                        f"need >= 8.0 for bf16)."
                    )
                return super().__enter__()

        torch.amp.autocast = _PatchedAutocast
        # Also patch the cuda-specific alias if it exists
        if hasattr(torch.cuda.amp, "autocast"):
            torch.cuda.amp.autocast = _PatchedAutocast

    # ---- Patch torch.device to allow 'cuda' construction ----
    # torch.device('cuda') already works; no patch needed.

    # ---- NCCL stubs ----
    try:
        import torch.distributed as dist
        if hasattr(dist, "is_nccl_available"):
            dist.is_nccl_available = lambda: True
    except Exception:
        pass

    _patched = True
    _patch_result = PatchResult(
        backend="standalone",
        num_devices=_NUM_DEVICES,
        device_name=_DEVICE_NAME,
    )

    warnings.warn(
        "fakegpu.torch_patch: CUDA operations are transparently redirected to CPU. "
        "Tensor.device will report 'cpu'. Computations are real but run on the CPU backend.",
        stacklevel=2,
    )
    return _patch_result


def is_patched() -> bool:
    """Return True if the torch‑cuda patch has been applied."""
    return _patched
