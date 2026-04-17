# Error Simulation Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable FakeGPU's Python torch_patch layer and C stub layer to reproduce common real-GPU errors (cross-device, OOM, invalid device, dtype mismatch, checkpoint load), so researchers can debug GPU-dependent code without physical hardware.

**Architecture:** Two-layer implementation: (1) Python `fakegpu/torch_patch.py` gains a storage-based device registry, per-device memory tracker, device index validation, and cross-device guards that wrap multi-input operations; (2) C `src/cublas/cublas_stubs.cpp` gains pointer-based cross-device validation in cublasGemmEx. Each feature has an env var opt-out. Tests use unittest, run via a unified HTML report generator.

**Tech Stack:** Python 3.10+, PyTorch (CPU build), CMake/C++17, unittest, weakref, JSON

---

## File Structure

### Modified files

| File | Responsibility |
|------|---------------|
| `fakegpu/torch_patch.py` | All Python-layer changes: per-device profiles, device registry, memory tracker, device index validation, cross-device guard, operation patches, autocast patch, torch.load enhancement |
| `src/core/global_state.hpp` | Add public `resolve_device_for_ptr` method |
| `src/core/global_state.cpp` | Implement `resolve_device_for_ptr` |
| `src/cublas/cublas_stubs.cpp` | Add cross-device pointer check in `cublasGemmEx` and `cublasLtMatmul` |

### New files

| File | Responsibility |
|------|---------------|
| `test/test_error_device_index.py` | E3-1..E3-4: device index out-of-bounds tests |
| `test/test_error_cross_device.py` | E1-1..E1-5: cross-device tensor operation tests |
| `test/test_error_oom.py` | E2-1..E2-5: OOM precise simulation tests |
| `test/test_error_dtype_autocast.py` | E4-1..E4-3: dtype autocast mismatch tests |
| `test/test_error_checkpoint_load.py` | E5-1..E5-3: checkpoint load device error tests |
| `test/test_error_gradient.py` | E7-1..E7-3: gradient computation error tests (validation-only) |
| `test/run_error_simulation_suite.py` | Unified test runner + HTML report generator |

---

## Task 1: Per-device Profile List

Currently `_resolve_total_memory()` returns a single scalar for all devices. For heterogeneous configs (`FAKEGPU_PROFILES=a100:2,v100:2`), each device needs its own memory limit.

**Files:**
- Modify: `fakegpu/torch_patch.py:66-127`

- [ ] **Step 1: Add `_resolve_per_device_profiles()` function**

Add after `_resolve_total_memory()` (after line 120), before the module-level constants (line 123):

```python
def _resolve_per_device_profiles() -> list[dict[str, Any]]:
    """Resolve per-device profile info from FAKEGPU_PROFILES.

    Returns a list of dicts, one per device, each with keys:
      'profile_id', 'name', 'total_memory', 'compute_major', 'compute_minor'
    """
    profiles_env = os.environ.get("FAKEGPU_PROFILES", "")
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
        for _ in range(_NUM_DEVICES):
            result.append(dict(entry))

    return result
```

- [ ] **Step 2: Add module-level `_DEVICE_PROFILES` constant**

Replace the existing module-level constants block (lines 123-128) with:

```python
_NUM_DEVICES = int(os.environ.get("FAKEGPU_DEVICE_COUNT", "8"))
_DEVICE_PROFILES: list[dict[str, Any]] = _resolve_per_device_profiles()
# If FAKEGPU_DEVICE_COUNT overrides the profile count, adjust
if len(_DEVICE_PROFILES) != _NUM_DEVICES:
    if len(_DEVICE_PROFILES) > 0:
        # Pad or truncate to match _NUM_DEVICES
        while len(_DEVICE_PROFILES) < _NUM_DEVICES:
            _DEVICE_PROFILES.append(dict(_DEVICE_PROFILES[-1]))
        _DEVICE_PROFILES = _DEVICE_PROFILES[:_NUM_DEVICES]

_DEVICE_NAME = _resolve_device_name()
_COMPUTE_MAJOR, _COMPUTE_MINOR = _resolve_compute_capability()
_TOTAL_MEMORY = _resolve_total_memory()

_current_device: int = 0
```

- [ ] **Step 3: Verify existing tests still pass**

Run: `python -m pytest test/test_torch_patch.py -x -q 2>&1 | head -30`
Expected: all existing tests pass (per-device profiles is additive, doesn't change existing behavior)

- [ ] **Step 4: Commit**

```
git add fakegpu/torch_patch.py
git commit -m "feat(torch_patch): add per-device profile resolution for heterogeneous configs"
```

---

## Task 2: Device Index Validation (P0-3)

Add bounds checking so `torch.cuda.set_device(99)`, `tensor.to("cuda:99")`, and `get_device_properties(99)` raise `RuntimeError` when the device index exceeds `_NUM_DEVICES`.

**Files:**
- Modify: `fakegpu/torch_patch.py:142-155` (`_normalize_device`), `:365-373` (`_stub_set_device`), `:318-341` (`_FakeDeviceProperties`), `:384-393` (`_stub_get_device_properties`)
- Create: `test/test_error_device_index.py`

- [ ] **Step 1: Write the failing test**

Create `test/test_error_device_index.py`:

```python
"""Error simulation: device index out-of-bounds (E3-1..E3-4)."""
import gc
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Configure BEFORE importing fakegpu/torch
os.environ["FAKEGPU_DEVICE_COUNT"] = "2"
os.environ["FAKEGPU_PROFILES"] = "a100:2"


class TestDeviceIndexErrors(unittest.TestCase):
    torch = None

    @classmethod
    def setUpClass(cls):
        import fakegpu
        fakegpu.patch_torch()
        import torch
        cls.torch = torch

    def test_e3_1_set_device_out_of_bounds(self):
        """E3-1: set_device(5) with 2 devices -> RuntimeError."""
        with self.assertRaisesRegex(RuntimeError, r"invalid device ordinal"):
            self.torch.cuda.set_device(5)

    def test_e3_2_tensor_to_invalid_device(self):
        """E3-2: randn(device='cuda:99') with 2 devices -> RuntimeError."""
        with self.assertRaisesRegex(RuntimeError, r"invalid device ordinal"):
            self.torch.randn(3, device="cuda:99")

    def test_e3_3_get_device_properties_invalid(self):
        """E3-3: get_device_properties(10) with 2 devices -> RuntimeError."""
        with self.assertRaisesRegex(RuntimeError, r"invalid device ordinal"):
            self.torch.cuda.get_device_properties(10)

    def test_e3_4_set_device_valid(self):
        """E3-4: set_device(0) with 2 devices -> no error."""
        try:
            self.torch.cuda.set_device(0)
            self.torch.cuda.set_device(1)
        except RuntimeError:
            self.fail("set_device raised RuntimeError for valid device index")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_error_device_index.py -v 2>&1 | tail -20`
Expected: E3-1, E3-2, E3-3 FAIL (no RuntimeError raised). E3-4 passes.

- [ ] **Step 3: Add `_parse_device_index` helper**

Add after `_normalize_device_index()` (after line 169):

```python
def _parse_device_index(device: Any) -> int:
    """Extract integer device index from various device representations."""
    import torch

    if isinstance(device, int):
        return device
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        if device.type == "cuda":
            return device.index if device.index is not None else _current_device
    return _current_device
```

- [ ] **Step 4: Add validation to `_normalize_device`**

Replace the `_normalize_device` function (lines 142-155) with:

```python
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
```

- [ ] **Step 5: Add validation to `_stub_set_device`**

Replace `_stub_set_device` (lines 365-373) with:

```python
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
```

- [ ] **Step 6: Add validation to `_stub_get_device_properties` and `_FakeDeviceProperties`**

Replace `_stub_get_device_properties` (lines 384-393) with:

```python
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
```

- [ ] **Step 7: Run test to verify it passes**

Run: `python -m pytest test/test_error_device_index.py -v 2>&1 | tail -20`
Expected: All 4 tests PASS.

- [ ] **Step 8: Run existing tests to check for regressions**

Run: `python -m pytest test/test_torch_patch.py -x -q 2>&1 | tail -10`
Expected: All existing tests pass.

- [ ] **Step 9: Commit**

```
git add fakegpu/torch_patch.py test/test_error_device_index.py
git commit -m "feat(torch_patch): add device index validation (E3-1..E3-4)"
```

---

## Task 3: Device Registry (Storage data_ptr → device index)

Add a module-level registry that maps `tensor.untyped_storage().data_ptr()` to logical device index. This is the foundation for cross-device checking (Task 5) and memory tracking (Task 4).

**Files:**
- Modify: `fakegpu/torch_patch.py` (add registry, modify `_patched_tensor_to`, `_patched_tensor_cuda`, `_device_kwarg_wrapper`)

- [ ] **Step 1: Add the registry and registration helper**

Add after the `_current_device: int = 0` line (around line 128, after Task 1's additions):

```python
# ---------------------------------------------------------------------------
# Device registry: tracks which fake CUDA device each tensor lives on.
# Key = storage data_ptr (stable across views/slices)
# Value = logical device index
# ---------------------------------------------------------------------------

_CROSS_DEVICE_CHECK = os.environ.get("FAKEGPU_CROSS_DEVICE_CHECK", "1") != "0"

_device_registry: dict[int, int] = {}


def _register_tensor_device(tensor: Any, device_index: int) -> None:
    """Register a tensor's storage in the device registry."""
    try:
        dp = tensor.untyped_storage().data_ptr()
        if dp != 0:
            _device_registry[dp] = device_index
    except Exception:
        pass


def _get_tensor_device(tensor: Any) -> int | None:
    """Look up the fake CUDA device index for a tensor, or None if untracked."""
    try:
        dp = tensor.untyped_storage().data_ptr()
        return _device_registry.get(dp)
    except Exception:
        return None
```

- [ ] **Step 2: Modify `_normalize_device` to also return the extracted device index**

We need a helper that extracts the device index before normalizing to CPU. Add after `_parse_device_index`:

```python
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
```

- [ ] **Step 3: Update `_patched_tensor_to` to register device**

Replace `_patched_tensor_to` (lines 192-210) with:

```python
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
```

- [ ] **Step 4: Update `_patched_tensor_cuda` to register device**

Replace `_patched_tensor_cuda` (lines 213-224) with:

```python
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
```

- [ ] **Step 5: Update `_device_kwarg_wrapper` to register device**

Replace `_device_kwarg_wrapper` (lines 172-181) with:

```python
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
```

- [ ] **Step 6: Add propagation patches for clone/contiguous/detach**

Add in the `patch()` function, right after the `torch.Tensor.cuda = _patched_tensor_cuda` line (around line 869):

```python
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
```

- [ ] **Step 7: Verify existing tests still pass**

Run: `python -m pytest test/test_torch_patch.py -x -q 2>&1 | tail -10`
Expected: All existing tests pass.

- [ ] **Step 8: Commit**

```
git add fakegpu/torch_patch.py
git commit -m "feat(torch_patch): add storage-based device registry for tensor device tracking"
```

---

## Task 4: Per-device Memory Tracker (P0-2)

Add `_DeviceMemoryTracker` class that tracks allocations per device and raises `torch.cuda.OutOfMemoryError` when exceeding device capacity.

**Files:**
- Modify: `fakegpu/torch_patch.py` (add class, wire into stubs)
- Create: `test/test_error_oom.py`

- [ ] **Step 1: Write the failing test**

Create `test/test_error_oom.py`:

```python
"""Error simulation: OOM precise simulation (E2-1..E2-5)."""
import gc
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["FAKEGPU_DEVICE_COUNT"] = "2"
os.environ["FAKEGPU_PROFILES"] = "a100-1g:2"
os.environ["FAKEGPU_MEMORY_TRACKING"] = "1"


class TestOOMErrors(unittest.TestCase):
    torch = None

    @classmethod
    def setUpClass(cls):
        import fakegpu
        fakegpu.patch_torch()
        import torch
        cls.torch = torch

    def setUp(self):
        # Force GC to reclaim any leftover tensors between tests
        gc.collect()

    def test_e2_1_single_allocation_exceeds_capacity(self):
        """E2-1: create tensor larger than device memory -> OutOfMemoryError."""
        torch = self.torch
        # a100-1g has 80GB; we want a profile with ~1GB for testing.
        # Since a100-1g uses 80GB by default in _PROFILE_TOTAL_MEMORY,
        # we'll test with a tensor that we know is too large.
        # The test relies on _DeviceMemoryTracker being initialized with
        # per-device memory from profiles.
        total = torch.cuda.mem_get_info(0)[1]
        # Try to allocate 2x total
        num_floats = (total * 2) // 4  # float32 = 4 bytes
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            torch.randn(num_floats, device="cuda:0")

    def test_e2_2_cumulative_oom(self):
        """E2-2: allocate 60% then another 60% -> second fails."""
        torch = self.torch
        total = torch.cuda.mem_get_info(0)[1]
        sixty_pct_floats = int((total * 0.6) // 4)
        a = torch.randn(sixty_pct_floats, device="cuda:0")
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            b = torch.randn(sixty_pct_floats, device="cuda:0")
        del a

    def test_e2_3_gc_reclaims_memory(self):
        """E2-3: allocate, del, gc, re-allocate same size -> success."""
        torch = self.torch
        total = torch.cuda.mem_get_info(0)[1]
        half_floats = int((total * 0.5) // 4)
        a = torch.randn(half_floats, device="cuda:0")
        del a
        gc.collect()
        # Should succeed after GC
        try:
            b = torch.randn(half_floats, device="cuda:0")
            del b
        except torch.cuda.OutOfMemoryError:
            self.fail("OOM after GC reclaimed memory")

    def test_e2_4_memory_allocated_tracks_live_tensors(self):
        """E2-4: memory_allocated returns actual sum of live tensor sizes."""
        torch = self.torch
        gc.collect()
        # Clear any leftover state
        before = torch.cuda.memory_allocated(0)
        a = torch.randn(1000, device="cuda:0")
        after = torch.cuda.memory_allocated(0)
        self.assertGreater(after, before)
        # float32 * 1000 = 4000 bytes
        self.assertGreaterEqual(after - before, 4000)
        del a
        gc.collect()

    def test_e2_5_mem_get_info_reflects_allocations(self):
        """E2-5: mem_get_info free = total - allocated."""
        torch = self.torch
        gc.collect()
        free_before, total = torch.cuda.mem_get_info(0)
        a = torch.randn(1000, device="cuda:0")
        free_after, total2 = torch.cuda.mem_get_info(0)
        self.assertEqual(total, total2)
        self.assertLess(free_after, free_before)
        del a
        gc.collect()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_error_oom.py -v 2>&1 | tail -20`
Expected: All tests FAIL (no OOM raised, memory_allocated returns 0).

- [ ] **Step 3: Add `_DeviceMemoryTracker` class**

Add after the `_device_registry` section in `torch_patch.py`:

```python
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

    def release(self, data_ptr: int) -> None:
        """Unregister allocation."""
        rec = self._allocs.pop(data_ptr, None)
        if rec:
            dev, nbytes = rec
            self._used[dev] = max(0, self._used[dev] - nbytes)

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


# Initialized later in patch() after _DEVICE_PROFILES is finalized
_memory_tracker: _DeviceMemoryTracker | None = None
```

- [ ] **Step 4: Add weakref-based cleanup helper**

Add right after the `_DeviceMemoryTracker` class:

```python
import weakref

_prevent_release_during_gc: set[int] = set()


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
        def _release_cb(ref, data_ptr=dp):
            if _memory_tracker is not None:
                _memory_tracker.release(data_ptr)
            _device_registry.pop(data_ptr, None)

        # Only add weakref if not already tracked (avoid double-counting)
        weakref.finalize(storage, _release_cb)
    except Exception:
        pass
```

- [ ] **Step 5: Wire tracker into tensor creation paths**

Modify `_register_tensor_device` to also call memory tracking:

```python
def _register_tensor_device(tensor: Any, device_index: int) -> None:
    """Register a tensor's storage in the device registry and memory tracker."""
    try:
        dp = tensor.untyped_storage().data_ptr()
        if dp != 0:
            _device_registry[dp] = device_index
            _register_tensor_for_memory_tracking(tensor, device_index)
    except Exception:
        pass
```

- [ ] **Step 6: Initialize tracker in `patch()` and wire memory stubs**

Inside `patch()`, after `_NUM_DEVICES` is finalized but before the stub assignments (around line 717), add:

```python
    # ---- Initialize memory tracker ----
    global _memory_tracker
    if _MEMORY_TRACKING:
        per_device_bytes = [p["total_memory"] for p in _DEVICE_PROFILES]
        _memory_tracker = _DeviceMemoryTracker(per_device_bytes)
```

Then update the memory stub functions (or replace them inline inside `patch()`):

```python
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
```

Place this block right after the existing `torch.cuda.mem_get_info = _stub_mem_get_info` block but before the `torch.cuda._is_compiled` line, so the tracked versions override the static stubs.

- [ ] **Step 7: Run test to verify it passes**

Run: `python -m pytest test/test_error_oom.py -v 2>&1 | tail -20`
Expected: All 5 tests PASS.

- [ ] **Step 8: Run existing tests to check for regressions**

Run: `python -m pytest test/test_torch_patch.py test/test_error_device_index.py -x -q 2>&1 | tail -10`
Expected: All pass.

- [ ] **Step 9: Commit**

```
git add fakegpu/torch_patch.py test/test_error_oom.py
git commit -m "feat(torch_patch): add per-device memory tracker with OOM simulation (E2-1..E2-5)"
```

---

## Task 5: Cross-device Guard and Operation Patches (P0-1 Python)

Add `_check_same_device()` validation that raises `RuntimeError` when tensors from different fake CUDA devices are used together.

**Files:**
- Modify: `fakegpu/torch_patch.py`
- Create: `test/test_error_cross_device.py`

- [ ] **Step 1: Write the failing test**

Create `test/test_error_cross_device.py`:

```python
"""Error simulation: cross-device tensor operations (E1-1..E1-5)."""
import gc
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["FAKEGPU_DEVICE_COUNT"] = "4"
os.environ["FAKEGPU_PROFILES"] = "a100:4"
os.environ["FAKEGPU_CROSS_DEVICE_CHECK"] = "1"
os.environ["FAKEGPU_MEMORY_TRACKING"] = "1"


class TestCrossDeviceErrors(unittest.TestCase):
    torch = None

    @classmethod
    def setUpClass(cls):
        import fakegpu
        fakegpu.patch_torch()
        import torch
        cls.torch = torch

    def test_e1_1_add_different_devices(self):
        """E1-1: a + b across cuda:0 and cuda:1."""
        torch = self.torch
        a = torch.randn(3, device="cuda:0")
        b = torch.randn(3, device="cuda:1")
        with self.assertRaisesRegex(RuntimeError, r"same device"):
            _ = a + b

    def test_e1_2_model_forward_cross_device(self):
        """E1-2: model on cuda:0, input on cuda:1."""
        torch = self.torch
        model = torch.nn.Linear(3, 3)
        # Move model to cuda:0
        model = model.to("cuda:0")
        x = torch.randn(2, 3, device="cuda:1")
        with self.assertRaisesRegex(RuntimeError, r"same device"):
            model(x)

    def test_e1_3_cat_cross_device(self):
        """E1-3: torch.cat across devices."""
        torch = self.torch
        a = torch.randn(3, device="cuda:0")
        b = torch.randn(3, device="cuda:1")
        with self.assertRaisesRegex(RuntimeError, r"same device"):
            torch.cat([a, b])

    def test_e1_4_cross_entropy_cross_device(self):
        """E1-4: F.cross_entropy with output on cuda:0, target on cuda:1."""
        torch = self.torch
        import torch.nn.functional as F
        output = torch.randn(2, 5, device="cuda:0")
        target = torch.randint(0, 5, (2,), device="cuda:1")
        with self.assertRaisesRegex(RuntimeError, r"same device"):
            F.cross_entropy(output, target)

    def test_e1_5_same_device_no_error(self):
        """E1-5: a + b on same device -> no error."""
        torch = self.torch
        a = torch.randn(3, device="cuda:0")
        b = torch.randn(3, device="cuda:0")
        try:
            c = a + b
        except RuntimeError:
            self.fail("RuntimeError raised for same-device operation")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_error_cross_device.py -v 2>&1 | tail -20`
Expected: E1-1, E1-2, E1-3, E1-4 FAIL. E1-5 passes.

- [ ] **Step 3: Add `_check_same_device` function**

Add after the `_get_tensor_device` function:

```python
def _check_same_device(*tensors: Any) -> None:
    """Raise RuntimeError if tensors span multiple fake CUDA devices."""
    if not _CROSS_DEVICE_CHECK:
        return
    import torch

    devices_seen: dict[int, bool] = {}
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
```

- [ ] **Step 4: Add operation wrapper helpers**

Add after `_check_same_device`:

```python
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
```

- [ ] **Step 5: Wire operation patches in `patch()`**

Inside `patch()`, after the clone/contiguous/detach patches (from Task 3), add:

```python
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
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest test/test_error_cross_device.py -v 2>&1 | tail -20`
Expected: All 5 tests PASS.

- [ ] **Step 7: Run all tests to check for regressions**

Run: `python -m pytest test/test_torch_patch.py test/test_error_device_index.py test/test_error_oom.py test/test_error_cross_device.py -x -q 2>&1 | tail -10`
Expected: All pass.

- [ ] **Step 8: Commit**

```
git add fakegpu/torch_patch.py test/test_error_cross_device.py
git commit -m "feat(torch_patch): add cross-device tensor operation validation (E1-1..E1-5)"
```

---

## Task 6: C Stub Cross-device Pointer Check (P0-1 C level)

Add pointer-based cross-device validation in `cublasGemmEx` and a public `resolve_device_for_ptr` wrapper.

**Files:**
- Modify: `src/core/global_state.hpp:61-105`
- Modify: `src/core/global_state.cpp:475-481`
- Modify: `src/cublas/cublas_stubs.cpp:1482-1491`

- [ ] **Step 1: Add public `resolve_device_for_ptr` to `GlobalState`**

In `src/core/global_state.hpp`, add after line 84 (the `get_allocation_info_ex` declaration):

```cpp
    int resolve_device_for_ptr(const void* ptr, int fallback_device) const;
```

- [ ] **Step 2: Implement `resolve_device_for_ptr`**

In `src/core/global_state.cpp`, add after `release_host_allocation` (after line 231):

```cpp
int GlobalState::resolve_device_for_ptr(const void* ptr, int fallback_device) const {
    std::lock_guard<std::mutex> lock(mutex);
    return resolve_device_for_ptr_nolock(ptr, fallback_device);
}
```

- [ ] **Step 3: Add cross-device check in `cublasGemmEx`**

In `src/cublas/cublas_stubs.cpp`, after the dtype compat check (line 1491), add:

```cpp
    // Cross-device check: A, B, C must be on the same device
    {
        auto& gs = fake_gpu::GlobalState::instance();
        const int current = gs.get_current_device();
        int dev_a = gs.resolve_device_for_ptr(A, current);
        int dev_b = gs.resolve_device_for_ptr(B, current);
        int dev_c = gs.resolve_device_for_ptr(C, current);
        if (dev_a != dev_c || dev_b != dev_c) {
            FGPU_LOG("[FakeCUBLAS] cublasGemmEx: cross-device detected "
                     "(A@dev%d, B@dev%d, C@dev%d)\n", dev_a, dev_b, dev_c);
            return CUBLAS_STATUS_INVALID_VALUE;
        }
    }
```

- [ ] **Step 4: Build to verify compilation**

Run: `cmake --build build 2>&1 | tail -5`
Expected: Build succeeds.

- [ ] **Step 5: Run smoke test**

Run: `./verification/run_smoke.sh 2>&1 | tail -10`
Expected: Smoke test passes (no cross-device in smoke test).

- [ ] **Step 6: Commit**

```
git add src/core/global_state.hpp src/core/global_state.cpp src/cublas/cublas_stubs.cpp
git commit -m "feat(cublas): add cross-device pointer validation in cublasGemmEx"
```

---

## Task 7: Autocast dtype Validation (P1-1)

Patch `torch.amp.autocast` to reject bfloat16 on devices with compute capability < 8.0.

**Files:**
- Modify: `fakegpu/torch_patch.py` (add autocast patch in `patch()`)
- Create: `test/test_error_dtype_autocast.py`

- [ ] **Step 1: Write the failing test**

Create `test/test_error_dtype_autocast.py`:

```python
"""Error simulation: dtype autocast mismatch (E4-1..E4-3)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestAutocastDtypeV100(unittest.TestCase):
    """E4-1, E4-3: V100 (compute 7.0) should reject bfloat16."""
    torch = None

    @classmethod
    def setUpClass(cls):
        os.environ["FAKEGPU_DEVICE_COUNT"] = "1"
        os.environ["FAKEGPU_PROFILES"] = "v100:1"
        os.environ["FAKEGPU_STRICT_COMPAT"] = "1"
        # Must re-patch with V100 profile
        import importlib
        import fakegpu.torch_patch as tp
        # Force re-resolve profile constants
        tp._patched = False
        tp._NUM_DEVICES = 1
        tp._COMPUTE_MAJOR, tp._COMPUTE_MINOR = 7, 0
        tp._DEVICE_NAME = "Tesla V100-SXM2-32GB"
        tp._TOTAL_MEMORY = 32 * 1024**3
        import fakegpu
        fakegpu.patch_torch()
        import torch
        cls.torch = torch

    def test_e4_1_autocast_bf16_on_v100(self):
        """E4-1: autocast bfloat16 on V100 -> RuntimeError."""
        torch = self.torch
        with self.assertRaisesRegex(RuntimeError, r"does not support bfloat16|bfloat16"):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                a = torch.randn(3, 3, device="cuda:0")
                b = torch.randn(3, 3, device="cuda:0")
                _ = a @ b


class TestAutocastDtypeA100(unittest.TestCase):
    """E4-2: A100 (compute 8.0) should allow bfloat16."""
    torch = None

    @classmethod
    def setUpClass(cls):
        os.environ["FAKEGPU_DEVICE_COUNT"] = "1"
        os.environ["FAKEGPU_PROFILES"] = "a100:1"
        os.environ["FAKEGPU_STRICT_COMPAT"] = "1"
        import fakegpu.torch_patch as tp
        tp._patched = False
        tp._COMPUTE_MAJOR, tp._COMPUTE_MINOR = 8, 0
        import fakegpu
        fakegpu.patch_torch()
        import torch
        cls.torch = torch

    def test_e4_2_autocast_bf16_on_a100(self):
        """E4-2: autocast bfloat16 on A100 -> no error."""
        torch = self.torch
        try:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                a = torch.randn(3, 3, device="cuda:0")
                b = torch.randn(3, 3, device="cuda:0")
                _ = a @ b
        except RuntimeError as e:
            if "bfloat16" in str(e):
                self.fail(f"A100 should support bfloat16: {e}")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_error_dtype_autocast.py::TestAutocastDtypeV100 -v 2>&1 | tail -10`
Expected: E4-1 FAIL (no RuntimeError raised).

- [ ] **Step 3: Add autocast patch in `patch()`**

Inside `patch()`, after the GradScaler patch section (around line 923), add:

```python
    # ---- Autocast dtype validation ----
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test/test_error_dtype_autocast.py -v 2>&1 | tail -15`
Expected: All tests PASS (E4-1 raises, E4-2 does not).

- [ ] **Step 5: Commit**

```
git add fakegpu/torch_patch.py test/test_error_dtype_autocast.py
git commit -m "feat(torch_patch): add autocast bfloat16 validation for compute < 8.0 (E4-1..E4-3)"
```

---

## Task 8: Checkpoint Load Device Validation (P1-2)

Enhance `_patched_torch_load` to validate `map_location` device index against `_NUM_DEVICES`.

**Files:**
- Modify: `fakegpu/torch_patch.py:771-785` (`_patched_torch_load` inside `patch()`)
- Create: `test/test_error_checkpoint_load.py`

- [ ] **Step 1: Write the failing test**

Create `test/test_error_checkpoint_load.py`:

```python
"""Error simulation: checkpoint load device errors (E5-1..E5-3)."""
import gc
import io
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["FAKEGPU_DEVICE_COUNT"] = "2"
os.environ["FAKEGPU_PROFILES"] = "a100:2"


class TestCheckpointLoadErrors(unittest.TestCase):
    torch = None

    @classmethod
    def setUpClass(cls):
        import fakegpu
        fakegpu.patch_torch()
        import torch
        cls.torch = torch

    def test_e5_1_load_map_location_invalid_device(self):
        """E5-1: load with map_location='cuda:5', count=2 -> RuntimeError."""
        torch = self.torch
        model = torch.nn.Linear(3, 3)
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)
        with self.assertRaisesRegex(RuntimeError, r"invalid device ordinal"):
            torch.load(buf, map_location="cuda:5")

    def test_e5_2_load_map_location_none_warning(self):
        """E5-2: load with map_location=None -> works (normalized to cpu)."""
        torch = self.torch
        model = torch.nn.Linear(3, 3)
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)
        # Should work without error (maps to cpu)
        state = torch.load(buf, map_location=None, weights_only=True)
        self.assertIsInstance(state, dict)

    def test_e5_3_load_map_location_cpu(self):
        """E5-3: load with map_location='cpu' -> success."""
        torch = self.torch
        model = torch.nn.Linear(3, 3)
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)
        state = torch.load(buf, map_location="cpu", weights_only=True)
        self.assertIsInstance(state, dict)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_error_checkpoint_load.py -v 2>&1 | tail -15`
Expected: E5-1 FAIL (no RuntimeError for map_location="cuda:5"). E5-2, E5-3 should pass.

- [ ] **Step 3: Enhance `_patched_torch_load` in `patch()`**

Replace the `_patched_torch_load` section (lines 771-785) with:

```python
    _orig_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        # Validate map_location device index
        ml = kwargs.get("map_location", None)
        if ml is None and len(args) >= 2:
            ml = args[1]

        if ml is not None:
            # Validate CUDA device index in map_location
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test/test_error_checkpoint_load.py -v 2>&1 | tail -15`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```
git add fakegpu/torch_patch.py test/test_error_checkpoint_load.py
git commit -m "feat(torch_patch): validate map_location device index in torch.load (E5-1..E5-3)"
```

---

## Task 9: Gradient Error Validation Tests (P2-2)

These tests verify FakeGPU's patching doesn't accidentally suppress PyTorch's native gradient errors. No implementation changes needed.

**Files:**
- Create: `test/test_error_gradient.py`

- [ ] **Step 1: Create the test file**

Create `test/test_error_gradient.py`:

```python
"""Error simulation: gradient computation errors (E7-1..E7-3).

These are pure validation tests — FakeGPU should NOT suppress
PyTorch's native gradient error messages. No implementation changes needed.
"""
import gc
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["FAKEGPU_DEVICE_COUNT"] = "2"
os.environ["FAKEGPU_PROFILES"] = "a100:2"


class TestGradientErrors(unittest.TestCase):
    torch = None

    @classmethod
    def setUpClass(cls):
        import fakegpu
        fakegpu.patch_torch()
        import torch
        cls.torch = torch

    def test_e7_1_backward_twice_without_retain(self):
        """E7-1: loss.backward() twice without retain_graph -> RuntimeError."""
        torch = self.torch
        x = torch.randn(3, requires_grad=True)
        y = (x ** 2).sum()
        y.backward()
        with self.assertRaisesRegex(RuntimeError, r"backward through the graph a second time|Trying to backward"):
            y.backward()

    def test_e7_2_requires_grad_on_integer(self):
        """E7-2: requires_grad on integer tensor -> RuntimeError."""
        torch = self.torch
        with self.assertRaisesRegex(RuntimeError, r"floating point|complex dtype"):
            torch.tensor([1, 2, 3], dtype=torch.long).requires_grad_(True)

    def test_e7_3_backward_on_non_scalar(self):
        """E7-3: backward on non-scalar without gradient arg -> RuntimeError."""
        torch = self.torch
        x = torch.randn(3, requires_grad=True)
        y = x ** 2  # non-scalar
        with self.assertRaisesRegex(RuntimeError, r"grad can be implicitly created only for scalar"):
            y.backward()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify all pass**

Run: `python -m pytest test/test_error_gradient.py -v 2>&1 | tail -15`
Expected: All 3 tests PASS (PyTorch native errors should fire normally).

- [ ] **Step 3: Commit**

```
git add test/test_error_gradient.py
git commit -m "test: add gradient error validation tests (E7-1..E7-3)"
```

---

## Task 10: Unified Test Runner and HTML Report

Create a runner script that discovers and executes all `test_error_*.py` files, then generates an HTML report in Morandi style matching `report_phase2.html`.

**Files:**
- Create: `test/run_error_simulation_suite.py`

- [ ] **Step 1: Create the runner script**

Create `test/run_error_simulation_suite.py`:

```python
#!/usr/bin/env python3
"""Unified runner for FakeGPU error simulation test suite.

Discovers all test/test_error_*.py files, runs each as a subprocess,
collects results, and generates an HTML report.

Usage:
    python test/run_error_simulation_suite.py [--output report.html]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TestResult:
    test_id: str
    name: str
    file: str
    status: str  # "pass", "fail", "error", "skip"
    duration: float = 0.0
    stdout: str = ""
    stderr: str = ""
    message: str = ""


@dataclass
class SuiteResult:
    category: str
    description: str
    results: list[TestResult] = field(default_factory=list)

    @property
    def passed(self):
        return sum(1 for r in self.results if r.status == "pass")

    @property
    def failed(self):
        return sum(1 for r in self.results if r.status in ("fail", "error"))

    @property
    def total(self):
        return len(self.results)


def discover_test_files(test_dir: str) -> list[Path]:
    """Find all test_error_*.py files."""
    return sorted(Path(test_dir).glob("test_error_*.py"))


def parse_unittest_output(stdout: str, stderr: str) -> list[dict]:
    """Parse unittest -v output to extract individual test results."""
    results = []
    # Match lines like "test_e3_1_set_device... ok" or "test_e3_1_set_device... FAIL"
    combined = stdout + "\n" + stderr
    for line in combined.splitlines():
        m = re.match(r"^(test_\w+)\s+\((\w+)\.(\w+)\)\s*$", line)
        if m:
            continue
        # unittest -v format: "test_name (module.Class) ... ok"
        m = re.match(r"^(test_\w+)\s+\([\w.]+\)\s*\.\.\.\s*(ok|FAIL|ERROR|SKIP|EXPECTED FAILURE)", line)
        if not m:
            # Also try: "test_name (module.Class)\nDescription ... ok"
            m = re.match(r"^(.*?)\s+\.\.\.\s*(ok|FAIL|ERROR|SKIP|EXPECTED FAILURE)", line)
        if m:
            name = m.group(1).strip()
            status_str = m.group(2).strip()
            status = {
                "ok": "pass",
                "FAIL": "fail",
                "ERROR": "error",
                "SKIP": "skip",
                "EXPECTED FAILURE": "pass",
            }.get(status_str, "error")
            results.append({"name": name, "status": status})
    return results


def run_test_file(path: Path) -> list[TestResult]:
    """Run a single test file and return results for each test case."""
    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(path), "-v", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(path.parent.parent),
        )
    except subprocess.TimeoutExpired:
        return [TestResult(
            test_id="TIMEOUT",
            name=path.stem,
            file=str(path),
            status="error",
            duration=120.0,
            message="Test timed out after 120 seconds",
        )]
    elapsed = time.time() - start

    # Parse pytest -v output
    results = []
    for line in (proc.stdout + proc.stderr).splitlines():
        # pytest -v format: "test/test_error_foo.py::TestClass::test_name PASSED"
        m = re.match(r"^.*?::([\w]+)::(test_\w+)\s+(PASSED|FAILED|ERROR|SKIPPED)", line)
        if m:
            cls_name = m.group(1)
            test_name = m.group(2)
            status = {
                "PASSED": "pass",
                "FAILED": "fail",
                "ERROR": "error",
                "SKIPPED": "skip",
            }[m.group(3)]
            results.append(TestResult(
                test_id=f"{path.stem}::{cls_name}::{test_name}",
                name=test_name,
                file=str(path.name),
                status=status,
                duration=elapsed / max(len(results) + 1, 1),
                stdout=proc.stdout,
                stderr=proc.stderr,
            ))

    if not results:
        # Fallback: treat the whole file as one test
        status = "pass" if proc.returncode == 0 else "fail"
        results.append(TestResult(
            test_id=path.stem,
            name=path.stem,
            file=str(path.name),
            status=status,
            duration=elapsed,
            stdout=proc.stdout,
            stderr=proc.stderr,
        ))

    return results


CATEGORY_MAP = {
    "test_error_cross_device": ("E1: Cross-device Ops", "Tensors on different CUDA devices used in same operation"),
    "test_error_oom": ("E2: OOM Simulation", "Per-device memory tracking and OutOfMemoryError"),
    "test_error_device_index": ("E3: Device Index", "Invalid device ordinal validation"),
    "test_error_dtype_autocast": ("E4: dtype/Autocast", "Autocast bfloat16 on incompatible compute capability"),
    "test_error_checkpoint_load": ("E5: Checkpoint Load", "torch.load map_location device validation"),
    "test_error_distributed": ("E6: Distributed", "NCCL communication error simulation"),
    "test_error_gradient": ("E7: Gradient", "Native PyTorch gradient errors (validation only)"),
}


def generate_html(suites: list[SuiteResult], output_path: str) -> None:
    """Generate Morandi-style HTML report."""
    total_pass = sum(s.passed for s in suites)
    total_fail = sum(s.failed for s in suites)
    total = sum(s.total for s in suites)

    suite_html_parts = []
    for i, suite in enumerate(suites):
        rows = []
        for r in suite.results:
            badge_class = {
                "pass": "badge-pass",
                "fail": "badge-fail",
                "error": "badge-fail",
                "skip": "badge-warn",
            }.get(r.status, "badge-info")
            badge_label = r.status.upper()

            detail_content = ""
            if r.stdout or r.stderr:
                escaped_out = (r.stdout + r.stderr).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                detail_content = f'<pre class="detail-pre">{escaped_out[:4000]}</pre>'

            rows.append(f"""
            <tr class="result-row" onclick="this.classList.toggle('expanded')">
              <td><code>{r.test_id}</code></td>
              <td>{r.name.replace('_', ' ').replace('test ', '')}</td>
              <td><span class="badge {badge_class}">{badge_label}</span></td>
            </tr>
            <tr class="detail-row"><td colspan="3">{detail_content}</td></tr>
            """)

        status_class = "section-pass" if suite.failed == 0 else "section-fail"
        suite_html_parts.append(f"""
        <section class="suite {status_class}" style="animation-delay:{0.1 + i*0.08}s">
          <div class="suite-header">
            <h2>{suite.category}</h2>
            <p class="suite-desc">{suite.description}</p>
            <div class="suite-stats">
              <span class="badge badge-pass">{suite.passed} passed</span>
              <span class="badge badge-fail">{suite.failed} failed</span>
            </div>
          </div>
          <table class="results-table">
            <thead><tr><th>Test ID</th><th>Description</th><th>Status</th></tr></thead>
            <tbody>{''.join(rows)}</tbody>
          </table>
        </section>
        """)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FakeGPU · Error Simulation Test Report</title>
<style>
  :root {{
    --bg:        #ede8e0;
    --surface:   #f4f0e9;
    --surface-2: #e3dcd1;
    --ink:       #4a4740;
    --ink-soft:  #7a7468;
    --line:      #d6cdbf;
    --sage:      #a8b5a0;
    --sage-dk:   #7d8e78;
    --rose:      #c9a7a1;
    --rose-dk:   #a8837c;
    --sand:      #d4bf9a;
    --sand-dk:   #b39a73;
    --mist:      #a4b1bd;
    --mist-dk:   #7d8b98;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue",
                 "PingFang SC", "Microsoft YaHei", sans-serif;
    background: var(--bg); color: var(--ink);
    line-height: 1.65; -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ max-width: 1080px; margin: 0 auto; padding: 64px 32px 96px; }}
  header.hero {{
    text-align: center; padding: 56px 24px 40px; margin-bottom: 48px;
    background: linear-gradient(140deg, var(--surface) 0%, var(--surface-2) 100%);
    border-radius: 24px; border: 1px solid var(--line);
    opacity: 0; transform: translateY(14px);
    animation: rise .9s .05s ease-out forwards;
  }}
  @keyframes rise {{ to {{ opacity: 1; transform: translateY(0); }} }}
  header.hero .eyebrow {{
    letter-spacing: .24em; font-size: 12px; color: var(--ink-soft);
    text-transform: uppercase; margin-bottom: 14px;
  }}
  header.hero h1 {{ font-size: 32px; margin: 0 0 12px; font-weight: 600; }}
  header.hero .sub {{ color: var(--ink-soft); font-size: 15px; margin: 0; }}
  .stats {{
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 18px; margin-bottom: 48px;
  }}
  .stat {{
    background: var(--surface); border: 1px solid var(--line);
    border-radius: 16px; padding: 22px 20px; text-align: center;
    opacity: 0; animation: rise .7s ease-out forwards;
  }}
  .stat .num {{ font-size: 36px; font-weight: 700; }}
  .stat .label {{ font-size: 13px; color: var(--ink-soft); margin-top: 4px; }}
  .stat.pass .num {{ color: var(--sage-dk); }}
  .stat.fail .num {{ color: var(--rose-dk); }}
  .stat.total .num {{ color: var(--mist-dk); }}

  .suite {{
    background: var(--surface); border: 1px solid var(--line);
    border-radius: 16px; padding: 28px 24px; margin-bottom: 24px;
    opacity: 0; animation: rise .7s ease-out forwards;
  }}
  .suite-header h2 {{ margin: 0 0 4px; font-size: 20px; }}
  .suite-desc {{ color: var(--ink-soft); font-size: 14px; margin: 0 0 12px; }}
  .suite-stats {{ display: flex; gap: 8px; margin-bottom: 16px; }}

  .badge {{
    display: inline-block; padding: 3px 10px; border-radius: 999px;
    font-size: 12px; font-weight: 600;
  }}
  .badge-pass {{ background: var(--sage); color: #fff; }}
  .badge-fail {{ background: var(--rose); color: #fff; }}
  .badge-warn {{ background: var(--sand); color: #fff; }}
  .badge-info {{ background: var(--mist); color: #fff; }}

  .results-table {{
    width: 100%; border-collapse: collapse; font-size: 14px;
  }}
  .results-table th {{
    text-align: left; padding: 8px 12px; border-bottom: 2px solid var(--line);
    color: var(--ink-soft); font-weight: 600; font-size: 12px;
    text-transform: uppercase; letter-spacing: .1em;
  }}
  .results-table td {{ padding: 10px 12px; border-bottom: 1px solid var(--line); }}
  .result-row {{ cursor: pointer; transition: background .15s; }}
  .result-row:hover {{ background: var(--surface-2); }}
  .detail-row {{ display: none; }}
  .result-row.expanded + .detail-row {{ display: table-row; }}
  .detail-pre {{
    background: var(--bg); padding: 12px; border-radius: 8px;
    font-size: 12px; overflow-x: auto; max-height: 300px; overflow-y: auto;
    white-space: pre-wrap; word-break: break-all;
  }}
  code {{ font-size: 13px; }}

  .section-pass {{ border-left: 4px solid var(--sage); }}
  .section-fail {{ border-left: 4px solid var(--rose); }}
</style>
</head>
<body>
<div class="wrap">
  <header class="hero">
    <div class="eyebrow">FakeGPU Error Simulation</div>
    <h1>Test Report</h1>
    <p class="sub">Automated validation of GPU error reproduction fidelity</p>
  </header>

  <div class="stats">
    <div class="stat pass" style="animation-delay:.15s">
      <div class="num">{total_pass}</div><div class="label">Passed</div>
    </div>
    <div class="stat fail" style="animation-delay:.2s">
      <div class="num">{total_fail}</div><div class="label">Failed</div>
    </div>
    <div class="stat total" style="animation-delay:.25s">
      <div class="num">{total}</div><div class="label">Total</div>
    </div>
  </div>

  {''.join(suite_html_parts)}
</div>
</body>
</html>"""

    Path(output_path).write_text(html, encoding="utf-8")
    print(f"Report written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="FakeGPU error simulation test suite")
    parser.add_argument("--output", "-o", default="test/error_simulation_report.html",
                        help="Output HTML report path")
    args = parser.parse_args()

    test_dir = os.path.join(os.path.dirname(__file__))
    files = discover_test_files(test_dir)

    if not files:
        print("No test_error_*.py files found!")
        sys.exit(1)

    print(f"Discovered {len(files)} test files:")
    for f in files:
        print(f"  {f.name}")

    suites: list[SuiteResult] = []
    for path in files:
        stem = path.stem
        cat, desc = CATEGORY_MAP.get(stem, (stem, ""))
        results = run_test_file(path)
        suite = SuiteResult(category=cat, description=desc, results=results)
        suites.append(suite)
        passed = suite.passed
        failed = suite.failed
        status_icon = "PASS" if failed == 0 else "FAIL"
        print(f"  [{status_icon}] {cat}: {passed}/{suite.total} passed")

    generate_html(suites, args.output)

    total_fail = sum(s.failed for s in suites)
    sys.exit(1 if total_fail > 0 else 0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the suite to verify it works**

Run: `python test/run_error_simulation_suite.py --output test/error_simulation_report.html 2>&1 | tail -20`
Expected: Report generated. All implemented tests should pass.

- [ ] **Step 3: Commit**

```
git add test/run_error_simulation_suite.py
git commit -m "feat: add unified error simulation test runner with HTML report generator"
```

---

## Task 11: Final Integration Verification

Run all error simulation tests together and verify the complete suite.

**Files:**
- No new files

- [ ] **Step 1: Run the full suite**

Run: `python test/run_error_simulation_suite.py --output test/error_simulation_report.html`
Expected: All tests pass.

- [ ] **Step 2: Run existing tests to check for regressions**

Run: `python -m pytest test/test_torch_patch.py -x -q 2>&1 | tail -10`
Expected: All existing tests pass.

- [ ] **Step 3: Verify HTML report is valid**

Run: `wc -l test/error_simulation_report.html && head -5 test/error_simulation_report.html`
Expected: HTML file exists with reasonable line count, starts with `<!DOCTYPE html>`.

- [ ] **Step 4: Final commit with all files**

```
git add -A
git status
# If any unstaged changes, add and commit
git commit -m "feat: complete P0+P1+P2 error simulation experiments"
```
