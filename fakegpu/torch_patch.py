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

_NUM_DEVICES = int(os.environ.get("FAKEGPU_DEVICE_COUNT", "8"))
_DEVICE_NAME = os.environ.get("FAKEGPU_DEVICE_NAME", "NVIDIA A100-SXM4-80GB")
_COMPUTE_MAJOR = 8
_COMPUTE_MINOR = 0
_TOTAL_MEMORY = 80 * 1024**3  # 80 GiB per device

_current_device: int = 0


@dataclass(frozen=True)
class PatchResult:
    backend: str
    num_devices: int
    device_name: str

# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


def _normalize_device(device: Any) -> Any:
    """Redirect ``cuda`` devices to ``cpu``, leaving others untouched."""
    import torch

    if device is None:
        return device
    if isinstance(device, int):
        # bare int in a context expecting a CUDA ordinal → cpu
        return torch.device("cpu")
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device) and device.type == "cuda":
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
        if "device" in kwargs and kwargs["device"] is not None:
            kwargs["device"] = _normalize_device(kwargs["device"])
        return fn(*args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Tensor.to / Tensor.cuda wrappers
# ---------------------------------------------------------------------------

_orig_tensor_to: Any = None
_orig_tensor_cuda: Any = None


def _patched_tensor_to(self: Any, *args: Any, **kwargs: Any) -> Any:
    import torch

    # keyword device
    if "device" in kwargs:
        kwargs["device"] = _normalize_device(kwargs["device"])

    # positional device (first arg can be device‑like, dtype, or Tensor)
    if args:
        first = args[0]
        if isinstance(first, str):
            args = (_normalize_device(first),) + args[1:]
        elif isinstance(first, torch.device) and first.type == "cuda":
            args = (torch.device("cpu"),) + args[1:]
        # int → might be device ordinal when second arg is dtype
        elif isinstance(first, int) and len(args) >= 2 and isinstance(args[1], torch.dtype):
            args = (torch.device("cpu"),) + args[1:]

    return _orig_tensor_to(self, *args, **kwargs)


def _patched_tensor_cuda(
    self: Any,
    device: Any = None,
    non_blocking: bool = False,
    memory_format: Any = None,
) -> Any:
    # Already on CPU; return self to avoid a pointless copy.
    import torch

    if memory_format is not None and memory_format is not torch.preserve_format:
        return self.contiguous(memory_format=memory_format)
    return self


# ---------------------------------------------------------------------------
# Module.to / Module.cuda wrappers
# ---------------------------------------------------------------------------

_orig_module_to: Any = None


def _patched_module_to(self: Any, *args: Any, **kwargs: Any) -> Any:
    import torch

    if "device" in kwargs:
        kwargs["device"] = _normalize_device(kwargs["device"])

    if args:
        first = args[0]
        if isinstance(first, str):
            args = (_normalize_device(first),) + args[1:]
        elif isinstance(first, torch.device) and first.type == "cuda":
            args = (torch.device("cpu"),) + args[1:]

    return _orig_module_to(self, *args, **kwargs)


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
        device = device.index or 0
    elif isinstance(device, str):
        device = torch.device(device).index or 0
    _current_device = device


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

    if num_devices is not None:
        _NUM_DEVICES = num_devices
    if device_name is not None:
        _DEVICE_NAME = device_name

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

    # ---- Patch torch.load to normalize map_location ----
    _orig_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        if "map_location" in kwargs:
            ml = kwargs["map_location"]
            if isinstance(ml, (str, torch.device)):
                ml = _normalize_device(ml)
                kwargs["map_location"] = ml
        elif len(args) >= 2:
            ml = args[1]
            if isinstance(ml, (str, torch.device)):
                args = (args[0], _normalize_device(ml)) + args[2:]
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
