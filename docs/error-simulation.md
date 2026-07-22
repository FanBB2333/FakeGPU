# Error Simulation

FakeGPU can reproduce common real-GPU runtime errors so that error-handling code paths can be validated on machines without physical GPUs. E1–E5 and E7 are Python-level `torch_patch` checks. E6 is implemented by the native NCCL shim and distributed coordinator and is opt-in.

## Error categories

| Code | Category | What it catches |
|------|----------|-----------------|
| E1 | Cross-device | Operations mixing tensors from different CUDA devices |
| E2 | Out of memory | Allocations exceeding per-device memory limits |
| E3 | Invalid device index | References to device IDs beyond the configured count |
| E4 | dtype / autocast | bfloat16 autocast on devices that do not support it (compute capability < 8.0) |
| E5 | Checkpoint load | Loading checkpoints saved for a different GPU architecture |
| E6 | Distributed | Deterministic collective rank failure, persistent async error, communicator shrink, and survivor recovery |
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

## E6: Distributed rank failure and recovery

In `FAKEGPU_DIST_MODE=simulate`, one global rank can be failed before a
selected direct collective is submitted. Every participating rank receives
`ncclRemoteError`, and `ncclCommGetAsyncError` keeps returning that error on
the parent communicator. Surviving ranks can call `ncclCommShrink` with an
explicit exclusion list and `NCCL_SHRINK_ABORT`; the child communicator uses
contiguous local ranks while preserving global-rank identity in reports.

The maintained four-rank experiment fails global rank 2 on the first
All-Reduce, shrinks `[0, 1, 2, 3]` to global ranks `[0, 1, 3]`, and verifies a
second All-Reduce on the recovered communicator:

```bash
FAKEGPU_MODE=simulate \
FAKEGPU_DIST_MODE=simulate \
FAKEGPU_NCCL_FAULT_RANK=2 \
FAKEGPU_NCCL_FAULT_SEQNO=1 \
FAKEGPU_NCCL_FAULT_OPERATION=all_reduce \
./build/fakegpu_nccl_direct_test --scenario fault-shrink
```

The TCP suite also initializes four workers and then terminates rank 2 with
`os._exit(86)`, without communicator cleanup. Ranks 0, 1, and 3 submit an
All-Reduce, receive a persistent `ncclSystemError` after the collective
timeout, explicitly exclude rank 2 with `ncclCommShrink`, and verify the
recovered sum `7.0`. The report marks this as `source=collective_timeout` and
keeps the submitted ranks `[0, 1, 3]` separate from the inferred absent rank.

A separate framework check runs one worker per physical host under
`torchrun --max-restarts=1`. After the initial All-Reduce, one worker exits
with code `86` while its communicator is active. Both agents replace their
workers, atomically increment per-rank arrival counters in the store, and enter
the same recovery generation despite possibly asymmetric local restart
counters. The restarted DDP step verifies
averaged gradients, optimizer parameters, and cross-rank equality. Run the
local Gloo version with `./ftest elastic_ddp`; the physical version is
`verification/run_physical_multihost.py --case elastic-ddp-restart`.

The checkpoint variant performs one DDP step with SGD momentum, atomically
writes host-local model/optimizer/step state, and then injects the same worker
exit. Replacement workers must load the exact files, verify equal restored
tensor state across ranks, and continue with the analytically expected second
update. Use `./ftest elastic_ddp_checkpoint` locally or
`verification/run_physical_multihost.py --case elastic-ddp-checkpoint` on two
GPU hosts.

The accumulated-state variant injects failure after the first of two gradient
accumulation microsteps. It restores AdamW first and second moments, StepLR,
completed optimizer steps, pending rank-local gradients, rank-local RNG, and a
`DistributedSampler` cursor before completing the synchronized second
microstep. Every host checkpoint contains a bundle for all saved ranks; the
maintained test reverses the logical rank mapping to verify that recovery does
not depend on the checkpoint file owner's original rank. Use
`./ftest elastic_ddp_training_state` locally or
`verification/run_physical_multihost.py --case elastic-ddp-training-state` on
two GPU hosts.

Use `python3 verification/test_fault_injection_recovery.py` for report-schema
checks, or `./ftest distributed_resilience` for the complete maintained
failure suite. Cluster JSON and Markdown reports contain the failed rank,
operation, observed ranks, attempted payload, exclusions, survivors, and
recovery time.

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
| `FAKEGPU_NCCL_FAULT_RANK` | unset | Global rank to fail during a direct simulated collective (E6) |
| `FAKEGPU_NCCL_FAULT_SEQNO` | unset | Positive communicator sequence number at which to fail (E6) |
| `FAKEGPU_NCCL_FAULT_OPERATION` | `all_reduce` | Collective selector: `all_reduce`, `reduce`, `broadcast`, `all_gather`, `reduce_scatter`, or `all_to_all` (E6) |

Set one of the first three guard variables to `0` to disable that guard. The
E6 rank and seqno variables must be used together; operation is optional and
defaults to `all_reduce`.

## Running the error simulation test suite

```bash
python test/run_error_simulation_suite.py
```

This runs the Python-layer error tests and generates a unified HTML report at `test/report.html`. The report has tab navigation covering Phase 1 (device discovery), Phase 2 (training flow), Phase 3 (MoE), and Phase 4 (error simulation). E6 uses the native build and is maintained separately:

```bash
python3 verification/test_fault_injection_recovery.py
./ftest elastic_ddp
./ftest elastic_ddp_checkpoint
./ftest elastic_ddp_training_state
./ftest distributed_resilience
```

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

- E1–E5 and E7 are Python-level checks; E6 runs through the native NCCL shim and coordinator.
- E6 direct failure injection targets collectives in `simulate` mode. The maintained Hybrid check also covers fixed-size `torchrun` worker-group restart; grouped/P2P injection, heartbeat detection, automatic membership changes, and world-size changes remain unsupported.
- The maintained shrink path requires an explicit exclusion list; excluded ranks must not call `ncclCommShrink`.
- `tensor.device` still reports `cpu` — the fake device index is tracked internally.
- No stream or event error simulation.

## Related pages

- [Getting Started](getting-started.md)
- [Quick Reference](quick-reference.md)
- [Reports & Validation](reports-and-validation.md)
