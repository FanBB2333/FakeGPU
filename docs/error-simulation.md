# Error Simulation

FakeGPU can reproduce common real-GPU runtime errors so that error-handling code paths can be validated on machines without physical GPUs. All error simulation features are part of the Python-level `torch_patch` layer and are enabled by default.

## Error categories

| Code | Category | What it catches |
|------|----------|-----------------|
| E1 | Cross-device | Operations mixing tensors from different CUDA devices |
| E2 | Out of memory | Allocations exceeding per-device memory limits |
| E3 | Invalid device index | References to device IDs beyond the configured count |
| E4 | dtype / autocast | bfloat16 autocast on devices that do not support it (compute capability < 8.0) |
| E5 | Checkpoint load | Loading checkpoints saved for a different GPU architecture |
| E6 | Distributed | (not yet implemented) |
| E7 | Gradient | Gradient computation on non-leaf or detached tensors |

## E1: Cross-device operations

Guards tensor operations (arithmetic, matmul, `torch.cat`, `F.linear`) and module forward passes. When operands reside on different fake CUDA devices, raises `RuntimeError` mentioning the two device indices.

Error message example:

```
RuntimeError: cross-device operation: tensor on cuda:0, other on cuda:1
```

```python
import fakegpu; fakegpu.patch_torch()
import torch
a = torch.randn(3, device="cuda:0")
b = torch.randn(3, device="cuda:1")
a + b  # RuntimeError: cross-device operation: tensor on cuda:0, other on cuda:1
```

## E2: Out of memory

Per-device memory tracking backed by configurable profiles. When allocations exceed `total_memory` for a device, raises `torch.cuda.OutOfMemoryError`.

Error message example:

```
torch.cuda.OutOfMemoryError: CUDA out of memory on device cuda:0. Tried to allocate 37.25 GiB.
```

```python
import fakegpu; fakegpu.patch_torch()
import torch
# Default A100 profile: 80 GB per device
torch.randn(100000, 100000, device="cuda:0")  # may exceed memory limit
```

## E3: Invalid device index

Validates device index in `cudaSetDevice`, `torch.device`, and tensor creation. References to device ordinals beyond the configured device count raise `RuntimeError`.

Error message example:

```
RuntimeError: invalid device ordinal 99
```

```python
import fakegpu; fakegpu.patch_torch()
import torch
torch.cuda.set_device(99)  # RuntimeError: invalid device ordinal 99
```

## E4: dtype / autocast compatibility

Checks compute capability before allowing bfloat16 autocast. Devices with compute capability below 8.0 (e.g. T4 at 7.5) cannot use bfloat16. Raises `RuntimeError`.

Error message example:

```
RuntimeError: bfloat16 autocast requires compute capability >= 8.0 (current device: T4, compute 7.5)
```

```python
# FAKEGPU_PROFILE=t4 python script.py
import fakegpu; fakegpu.patch_torch()
import torch
with torch.autocast("cuda", dtype=torch.bfloat16):  # RuntimeError on T4
    ...
```

## E5: Checkpoint load compatibility

Validates that checkpoint metadata matches the current fake GPU profile. Detects architecture mismatches when loading checkpoints saved under a different profile.

Error message example:

```
RuntimeError: checkpoint was saved on A100 (compute 8.0) but current device is T4 (compute 7.5)
```

```python
import fakegpu; fakegpu.patch_torch()
import torch
# Saving on "A100", loading on "T4" profile raises RuntimeError
```

## E7: Gradient errors

Catches `grad` access on non-leaf tensors and `backward()` on detached tensors. Raises `RuntimeError`.

Error message example:

```
RuntimeError: cannot access grad of non-leaf tensor
```

```python
import fakegpu; fakegpu.patch_torch()
import torch
x = torch.randn(3, device="cuda", requires_grad=True)
y = x * 2
y.grad  # this is a non-leaf; RuntimeError if misused
```

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `FAKEGPU_CROSS_DEVICE_CHECK` | `1` (enabled) | Enable cross-device operation guards (E1) |
| `FAKEGPU_MEMORY_TRACKING` | `1` (enabled) | Enable per-device memory tracking and OOM simulation (E2) |
| `FAKEGPU_STRICT_COMPAT` | `1` (enabled) | Enable strict dtype / architecture compatibility checks (E4, E5) |

Set any of these to `0` to disable the corresponding guard.

## Running the error simulation test suite

```bash
python test/run_error_simulation_suite.py
```

This runs all 23 error simulation tests and generates a unified HTML report at `test/report.html`. The report has tab navigation covering Phase 1 (device discovery), Phase 2 (training flow), Phase 3 (MoE), and Phase 4 (error simulation).

Individual test files:

```bash
python test/test_error_cross_device.py      # E1: 5 tests
python test/test_error_oom.py               # E2: 5 tests
python test/test_error_device_index.py      # E3: 4 tests
python test/test_error_dtype_autocast.py    # E4: 3 tests
python test/test_error_checkpoint_load.py   # E5: 3 tests
python test/test_error_gradient.py          # E7: 3 tests
```

## Limitations

- Error simulation is Python-level only (`torch_patch` layer); the C stub layer has partial cross-device support.
- E6 (distributed errors) is not yet implemented.
- `tensor.device` still reports `cpu` — the fake device index is tracked internally.
- No stream or event error simulation.

## Related pages

- [Getting Started](getting-started.md)
- [Quick Reference](quick-reference.md)
- [Reports & Validation](reports-and-validation.md)
