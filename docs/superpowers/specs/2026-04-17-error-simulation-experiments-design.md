# Error Simulation Experiments Design Spec

**Date:** 2026-04-17
**Goal:** Enable FakeGPU to reproduce common real-GPU errors (cross-device, OOM, invalid device, dtype mismatch, checkpoint, distributed, gradient) so researchers can debug GPU-dependent code without physical hardware.

**Layers:** Both C stub (`libfake_gpu.so`, Linux LD_PRELOAD) and Python (`fakegpu/torch_patch.py`, macOS / CPU-only PyTorch).

**Delivery:** 3 batches (P0 → P1 → P2), each batch adds implementation + independent test scripts + unified HTML report.

---

## Current State & Gap Analysis

FakeGPU already simulates:
- CUDA OOM at C level (`register_allocation_nolock` rejects when `used_memory + size > total_memory`)
- Invalid device ordinal at C level (`cudaSetDevice` returns `cudaErrorInvalidDevice`)
- cuBLAS dtype compatibility (`check_device_dtype_compat` in `cublasGemmEx`)
- NCCL argument errors (~50+ error codes mapped)
- cuBLAS null pointer / dimension validation
- Stream/event handle validation in driver stubs

FakeGPU does NOT simulate:
- **Cross-device tensor operation errors** — pointers from different devices mix freely; torch_patch discards device index entirely
- **Python-level OOM** — `torch_patch.py` reports unlimited free memory; `memory_allocated()` always returns 0
- **Python-level device index validation** — `set_device(99)` and `tensor.to("cuda:99")` succeed silently
- **dtype mismatch in autocast context** — only cuBLAS GEMM is checked, not the autocast entry point
- **Checkpoint load device errors** — no validation of `map_location` against device count
- **cuBLAS cross-device pointer validation** — A, B, C matrix pointers are never checked for device consistency
- `cudaGetLastError` / `cudaPeekAtLastError` — `last_error` is never set by stubs

---

## Batch P0: Cross-device / OOM / Device Index

### P0-1: Cross-device Tensor Operation Errors

#### Problem
Most common multi-GPU bug: `expected all tensors to be on the same device, but found at least two devices, cuda:0 and cuda:1`. Currently FakeGPU never raises this because:
- **torch_patch**: all `cuda:N` tensors silently become CPU; device index is discarded
- **C stubs**: cuBLAS only null-checks matrix pointers, never validates device affinity

#### Design: torch_patch layer

**Tensor device registry (Storage-based, Strategy C):**

```python
# Module-level in torch_patch.py
_device_registry: dict[int, int] = {}
# Key: tensor.untyped_storage().data_ptr()  (stable across views/slices)
# Value: logical device index (0, 1, 2, ...)
```

Why `data_ptr()` not `id(tensor)`:
- Views, slices, transposes share the underlying `Storage` object
- `a.view(2,8)` has a different `id()` but the same `storage().data_ptr()`
- This means device tag is automatically inherited by all views

**Registration points** — existing patched functions gain one extra line:

| Function | Change |
|----------|--------|
| `_patched_tensor_to` | After normalizing device, record `_device_registry[result.untyped_storage().data_ptr()] = target_device_index` |
| `_patched_tensor_cuda` | Same |
| `_device_kwarg_wrapper` (factory fns) | After creation, register if device was `cuda:N` |
| `_patched_tensor_clone` (NEW) | Propagate tag from source to cloned tensor |
| `_patched_tensor_contiguous` (NEW) | Propagate tag if new storage allocated |
| `_patched_tensor_detach` (NEW) | Propagate tag |

**Cleanup:** When a storage is freed (refcount → 0), its `data_ptr` becomes invalid. This is benign — stale entries in `_device_registry` are harmless because no live tensor will have that `data_ptr`. Periodic cleanup via `_stub_empty_cache` or a simple size cap (e.g. evict oldest entries when dict exceeds 100k).

**Validation guard:**

```python
def _check_same_device(*tensors: torch.Tensor) -> None:
    """Raise RuntimeError if tensors span multiple fake CUDA devices."""
    devices_seen: dict[int, torch.Tensor] = {}  # device_idx -> first tensor
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            continue
        dp = t.untyped_storage().data_ptr()
        dev = _device_registry.get(dp)
        if dev is None:
            continue  # untracked tensor (e.g. CPU constant) — skip
        if devices_seen and dev not in devices_seen:
            other_dev = next(iter(devices_seen))
            raise RuntimeError(
                f"Expected all tensors to be on the same device, "
                f"but found at least two devices, cuda:{other_dev} and cuda:{dev}!"
            )
        devices_seen[dev] = t
```

**Patched multi-input operations:**

These operations are wrapped to call `_check_same_device` on all tensor arguments before delegating to the original:

- `torch.matmul`, `torch.mm`, `torch.bmm`
- `torch.cat`, `torch.stack`
- `torch.where` (3-arg form)
- `torch.addmm`, `torch.addcmul`, `torch.addcdiv`
- `F.cross_entropy`, `F.mse_loss`, `F.nll_loss`, `F.binary_cross_entropy`
- `Tensor.__add__`, `__mul__`, `__sub__`, `__truediv__`, `__matmul__` (and their `r` variants)

**Opt-out:** `FAKEGPU_CROSS_DEVICE_CHECK=0` env var disables the guard for users who don't want it.

#### Design: C stub layer

In `cublas_stubs.cpp`, add to `cublasGemmEx` (after the dtype compat check):

```cpp
// Cross-device check: A, B, C must be on the same device
int dev_a = gs.resolve_device_for_ptr(A, current);
int dev_b = gs.resolve_device_for_ptr(B, current);
int dev_c = gs.resolve_device_for_ptr(C, current);
if (dev_a != dev_c || dev_b != dev_c) {
    FGPU_LOG("[FakeCUBLAS] cublasGemmEx: cross-device detected "
             "(A@dev%d, B@dev%d, C@dev%d)\n", dev_a, dev_b, dev_c);
    return CUBLAS_STATUS_INVALID_VALUE;
}
```

Add a public `resolve_device_for_ptr` (locking wrapper) to `GlobalState` if not already exposed.

#### Test cases

| ID | File | Scenario | Expected |
|----|------|----------|----------|
| E1-1 | `test_error_cross_device.py` | `a=randn(device="cuda:0"); b=randn(device="cuda:1"); a+b` | `RuntimeError: ...cuda:0 and cuda:1` |
| E1-2 | same | `model.cuda(0); x=randn(device="cuda:1"); model(x)` | `RuntimeError: ...cuda:0 and cuda:1` (in Linear.forward) |
| E1-3 | same | `torch.cat([randn(device="cuda:0"), randn(device="cuda:1")])` | `RuntimeError: ...cuda:0 and cuda:1` |
| E1-4 | same | `F.cross_entropy(output_on_0, target_on_1)` | `RuntimeError: ...cuda:0 and cuda:1` |
| E1-5 | same | `a=randn(device="cuda:0"); b=a.to("cuda:0"); a+b` | No error (same device) |
| E1-6 | `test_error_cross_device_native.cpp` | cuBLAS GEMM with A@dev0, C@dev1 | `CUBLAS_STATUS_INVALID_VALUE` |

---

### P0-2: OOM Precise Simulation (torch_patch layer)

#### Problem
`torch_patch.py` always reports unlimited free memory. Researchers cannot test OOM handling, memory-efficient techniques (gradient checkpointing, offloading), or validate model fits within a target GPU's VRAM.

#### Design

**Per-device memory tracker** (new class in `torch_patch.py`):

```python
class _DeviceMemoryTracker:
    def __init__(self, num_devices: int, bytes_per_device: int):
        self._total = [bytes_per_device] * num_devices
        self._used = [0] * num_devices
        self._peak = [0] * num_devices
        # data_ptr -> (device_index, nbytes)
        self._allocs: dict[int, tuple[int, int]] = {}

    def allocate(self, data_ptr: int, nbytes: int, device: int) -> None:
        """Register allocation. Raise OutOfMemoryError if exceeds limit."""
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
        return self._used[device]

    def max_memory_allocated(self, device: int) -> int:
        return self._peak[device]

    def mem_get_info(self, device: int) -> tuple[int, int]:
        free = self._total[device] - self._used[device]
        return (free, self._total[device])
```

**Hook points:**
- `_patched_tensor_to("cuda:N")`: after CPU `.to()`, call `tracker.allocate(storage.data_ptr(), storage.nbytes(), N)`
- `_patched_tensor_cuda()`: same
- Factory functions with `device="cuda:N"`: register after creation
- GC / tensor deletion: use `weakref.ref(tensor, callback)` to call `tracker.release(data_ptr)` when tensor is collected
- Replace static memory stubs: `_stub_memory_allocated`, `_stub_max_memory_allocated`, `_stub_mem_get_info` delegate to tracker

**Interaction with `_DeviceMemoryTracker` and `_device_registry`:** They share the same `data_ptr` keyspace. When a tensor is registered in the device registry, it is also tracked for memory. They are two maps serving different purposes (device affinity vs. memory accounting).

**Per-device memory limits** come from `_TOTAL_MEMORY` (existing, resolved from profile). For heterogeneous configs (`FAKEGPU_PROFILES=a100:2,v100:2`), each device gets its profile's memory limit.

**Opt-out:** `FAKEGPU_MEMORY_TRACKING=0` disables memory tracking (all stubs return 0 as before).

#### Test cases

| ID | Scenario | Expected |
|----|----------|----------|
| E2-1 | `FAKEGPU_PROFILES=a100-1g:1` (1GiB), create 2GiB tensor on cuda:0 | `OutOfMemoryError` with capacity message |
| E2-2 | Allocate 60% VRAM, then another 60% | Second allocation: `OutOfMemoryError` |
| E2-3 | Allocate tensor, `del tensor`, force GC, re-allocate same size | Success (memory reclaimed) |
| E2-4 | `torch.cuda.memory_allocated(0)` after creating tensors | Returns actual sum of live tensor sizes |
| E2-5 | `torch.cuda.mem_get_info(0)` after allocations | `free = total - allocated`, `total` matches profile |

---

### P0-3: Device Index Out-of-Bounds

#### Problem
`torch_patch.py` accepts any device index silently. `torch.cuda.set_device(99)` succeeds. `tensor.to("cuda:99")` silently becomes CPU. Researchers writing multi-GPU code don't get the error they'd see on real hardware.

#### Design

**Validation in `_normalize_device()`:**

```python
def _normalize_device(dev) -> torch.device:
    """Normalize device, raising for invalid CUDA ordinals."""
    # ... existing parsing logic ...
    if device_type == "cuda":
        if idx >= _NUM_DEVICES:
            raise RuntimeError(
                f"CUDA error: invalid device ordinal "
                f"(requested cuda:{idx}, available: {_NUM_DEVICES})"
            )
    return torch.device("cpu")
```

**Validation in `_stub_set_device()`:**

```python
def _stub_set_device(device) -> None:
    idx = _parse_device_index(device)
    if idx < 0 or idx >= _NUM_DEVICES:
        raise RuntimeError(
            f"CUDA error: invalid device ordinal "
            f"(requested {idx}, available: {_NUM_DEVICES})"
        )
    global _current_device
    _current_device = idx
```

**Validation in `_FakeDeviceProperties.__init__`:**

```python
def __init__(self, device_index):
    if device_index >= _NUM_DEVICES:
        raise RuntimeError(f"CUDA error: invalid device ordinal")
    # ... existing code ...
```

#### Test cases

| ID | Scenario | Expected |
|----|----------|----------|
| E3-1 | `FAKEGPU_DEVICE_COUNT=2; torch.cuda.set_device(5)` | `RuntimeError: invalid device ordinal` |
| E3-2 | `torch.randn(3, device="cuda:99")` with count=2 | `RuntimeError: invalid device ordinal` |
| E3-3 | `torch.cuda.get_device_properties(10)` with count=2 | `RuntimeError: invalid device ordinal` |
| E3-4 | `torch.cuda.set_device(0)` with count=2 | No error |

---

## Batch P1: dtype Mismatch + Checkpoint Loading

### P1-1: dtype Mismatch in Autocast

#### Problem
Researchers using `torch.amp.autocast("cuda", dtype=torch.bfloat16)` on a simulated V100 (no BF16 support) should see a meaningful error, not silent execution on CPU.

#### Design

Patch `torch.amp.autocast.__init__` or `__enter__`:

```python
_strict_compat = os.environ.get("FAKEGPU_STRICT_COMPAT", "1") != "0"

def _patched_autocast_enter(self):
    if self.device_type == "cuda" and self.fast_dtype == torch.bfloat16:
        if _COMPUTE_MAJOR < 8:
            if _strict_compat:
                raise RuntimeError(
                    f"Current CUDA Device does not support bfloat16. "
                    f"Please switch dtype to float16 (compute capability "
                    f"{_COMPUTE_MAJOR}.{_COMPUTE_MINOR}, need >= 8.0 for bf16)."
                )
    return _orig_autocast_enter(self)
```

Also verify that the existing C-level `check_device_dtype_compat` fires for non-autocast manual bf16 operations.

**Opt-out:** Governed by `FAKEGPU_STRICT_COMPAT` (already exists).

#### Test cases

| ID | Scenario | Expected |
|----|----------|----------|
| E4-1 | `FAKEGPU_PROFILES=v100:1`, `autocast("cuda", dtype=bf16)` | `RuntimeError: ...does not support bfloat16` |
| E4-2 | `FAKEGPU_PROFILES=a100:1`, `autocast("cuda", dtype=bf16)` | No error |
| E4-3 | `FAKEGPU_PROFILES=v100:1`, manual `tensor.bfloat16().mm(other.bfloat16())` | Warning or error depending on strict mode |

---

### P1-2: Checkpoint Load Device Errors

#### Problem
Saving a checkpoint from `cuda:0` and loading it on a different device configuration is a common source of errors. FakeGPU should reproduce these.

#### Design

Enhance the existing `_patched_torch_load`:

```python
def _patched_torch_load(*args, **kwargs):
    map_location = kwargs.get("map_location", None)
    # ... existing normalization ...

    # Validate map_location device index
    if isinstance(map_location, str) and map_location.startswith("cuda"):
        idx = _parse_device_index(map_location)
        if idx >= _NUM_DEVICES:
            raise RuntimeError(
                f"CUDA error: invalid device ordinal "
                f"(map_location={map_location}, available: {_NUM_DEVICES})"
            )

    # If map_location is None and checkpoint has cuda tensors,
    # simulate the real error
    if map_location is None:
        kwargs["map_location"] = "cpu"  # existing behavior
        # Optionally warn that original was cuda

    return _orig_torch_load(*args, **kwargs)
```

For the "deserialize on CUDA without map_location" case, we can optionally wrap the loaded state_dict to detect cuda tensors and emit a warning.

#### Test cases

| ID | Scenario | Expected |
|----|----------|----------|
| E5-1 | Save model on cuda:0, load `map_location="cuda:5"`, count=2 | `RuntimeError: invalid device ordinal` |
| E5-2 | Save with fake cuda device tag, load `map_location=None` | Warning about cuda→cpu remapping |
| E5-3 | Save on cuda:0, load `map_location="cpu"` | Success |

---

## Batch P2: Distributed + Gradient

### P2-1: Distributed Communication Errors

NCCL stubs already implement extensive error checking. These experiments validate that real workflow patterns trigger the correct errors.

#### Test cases

| ID | Scenario | Expected |
|----|----------|----------|
| E6-1 | `all_reduce` with mismatched tensor sizes across ranks | `ncclInvalidArgument` / RuntimeError |
| E6-2 | `broadcast` with `root` >= `world_size` | `ncclInvalidArgument` |
| E6-3 | `send`/`recv` with `peer == rank` | `ncclInvalidArgument` |

#### Implementation
Minimal — mostly writing test scripts that exercise existing NCCL stub error paths via `torch.distributed`. May need a small multi-process harness (2 processes via `torch.multiprocessing.spawn`).

---

### P2-2: Gradient Computation Errors

These errors are PyTorch-internal and should fire identically under torch_patch since they don't depend on CUDA. The experiments verify that FakeGPU's patching doesn't accidentally suppress them.

#### Test cases

| ID | Scenario | Expected |
|----|----------|----------|
| E7-1 | `loss.backward()` twice without `retain_graph=True` | `RuntimeError: trying to backward through the graph a second time` |
| E7-2 | `torch.tensor([1, 2], dtype=torch.long).requires_grad_(True)` | `RuntimeError: only Tensors of floating point and complex dtype can require gradients` |
| E7-3 | `.backward()` on non-scalar without `gradient` arg | `RuntimeError: grad can be implicitly created only for scalar outputs` |

#### Implementation
No implementation changes. Pure validation test scripts.

---

## Test Infrastructure

### File structure

```
test/
  test_error_cross_device.py          # E1-1..E1-6
  test_error_oom.py                   # E2-1..E2-5
  test_error_device_index.py          # E3-1..E3-4
  test_error_dtype_autocast.py        # E4-1..E4-3
  test_error_checkpoint_load.py       # E5-1..E5-3
  test_error_distributed.py           # E6-1..E6-3
  test_error_gradient.py              # E7-1..E7-3
  run_error_simulation_suite.py       # Unified runner + HTML report generator

verification/
  test_cross_device_cublas.cpp        # E1-6 (C-level test)
```

### Test script pattern

Each test file follows this pattern:

```python
"""Error simulation: <category>."""
import sys, os, unittest

# Ensure FakeGPU is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

class TestCrossDeviceErrors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["FAKEGPU_DEVICE_COUNT"] = "4"
        os.environ.setdefault("FAKEGPU_PROFILES", "a100:4")
        import fakegpu
        fakegpu.patch_torch()
        import torch
        cls.torch = torch

    def test_add_different_devices(self):
        """E1-1: a+b across cuda:0 and cuda:1"""
        a = self.torch.randn(3, device="cuda:0")
        b = self.torch.randn(3, device="cuda:1")
        with self.assertRaisesRegex(RuntimeError, r"cuda:0.*cuda:1|cuda:1.*cuda:0"):
            _ = a + b

    # ... more test methods ...

if __name__ == "__main__":
    unittest.main()
```

### HTML report runner

`run_error_simulation_suite.py`:
1. Discovers all `test_error_*.py` files
2. Runs each via `subprocess.run([sys.executable, path], capture_output=True, timeout=60)`
3. Parses exit code + stdout/stderr
4. Generates HTML report in Morandi style (same as `report_phase2.html`)
5. Writes to `test/error_simulation_report.html`

---

## Implementation Priority & Dependencies

```
P0 (first batch):
  torch_patch.py changes:
    1. _device_registry (storage data_ptr -> device index)
    2. _DeviceMemoryTracker (per-device memory accounting)
    3. Device index validation in _normalize_device, _stub_set_device
    4. _check_same_device guard + operation patches
    5. clone/contiguous/detach propagation patches

  C stub changes:
    6. cublasGemmEx cross-device pointer check
    7. GlobalState::resolve_device_for_ptr public locking wrapper

  Tests:
    8. test_error_cross_device.py (E1-1..E1-5)
    9. test_error_oom.py (E2-1..E2-5)
    10. test_error_device_index.py (E3-1..E3-4)
    11. test_cross_device_cublas.cpp + test_error_cross_device_native.py (E1-6)
    12. run_error_simulation_suite.py

P1 (second batch, depends on P0):
  13. Autocast dtype validation patch
  14. torch.load device validation enhancement
  15. test_error_dtype_autocast.py (E4-1..E4-3)
  16. test_error_checkpoint_load.py (E5-1..E5-3)

P2 (third batch, independent):
  17. test_error_distributed.py (E6-1..E6-3) — uses existing NCCL stubs
  18. test_error_gradient.py (E7-1..E7-3) — pure validation, no impl changes
```

---

## Environment Variables Summary

| Variable | Default | Purpose |
|----------|---------|---------|
| `FAKEGPU_DEVICE_COUNT` | 8 | Number of simulated devices |
| `FAKEGPU_PROFILES` | `a100:8` | Device profiles (controls memory, compute cap, dtype support) |
| `FAKEGPU_CROSS_DEVICE_CHECK` | 1 | Enable cross-device tensor validation |
| `FAKEGPU_MEMORY_TRACKING` | 1 | Enable per-device memory accounting in torch_patch |
| `FAKEGPU_STRICT_COMPAT` | 1 | Strict dtype compatibility (error vs warning) |
| `FAKEGPU_TERMINAL_REPORT` | 1 | Terminal summary on exit |

---

## Success Criteria

A test is considered passing if FakeGPU raises the **same exception class** with a **message that contains the same key phrases** as real CUDA/PyTorch would. Exact message matching is not required (CUDA driver version strings, memory addresses, etc. will differ), but the error class and diagnostic keywords must match.

Specifically:
- Cross-device: `RuntimeError` containing `"same device"` and both device names
- OOM: `torch.cuda.OutOfMemoryError` containing `"out of memory"` and capacity info
- Invalid device: `RuntimeError` containing `"invalid device ordinal"`
- dtype: `RuntimeError` containing `"does not support bfloat16"` or similar
- Checkpoint: `RuntimeError` containing `"invalid device ordinal"` or `"Attempting to deserialize"`
- Distributed: `RuntimeError` containing NCCL error description
- Gradient: `RuntimeError` with PyTorch's native message (unmodified)
