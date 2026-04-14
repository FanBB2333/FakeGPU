# Phase 2 Custom Torch Route

This page documents the current status of the Phase 2 route: a custom PyTorch build that exposes CUDA-facing semantics on CPU-only macOS and Linux hosts while pairing with FakeGPU.

## Goal

Phase 2 exists for workloads that cannot stop at a `PrivateUse1` device name such as `fgpu`.

The target is pragmatic compatibility for local debug and smoke validation:

- `tensor.device.type == "cuda"`
- `tensor.is_cuda is True`
- `module.cuda()` and `module.to("cuda")`
- common `torch.cuda.*` control flow
- enough `torch.distributed` / `DataParallel` / checkpoint behavior to keep training scripts moving

It is not a native CUDA backend. Execution still happens on CPU.

## Current architecture

- Base: upstream `pytorch/pytorch` `v2.11.0`
- Fork: local `pytorch-fakegpu` repository
- Integration point: `torch.fakegpu.enable()`
- Bridge: `fakegpu.torch_patch.patch()` prefers the installed custom torch route when available

Implementation is intentionally Python-layer heavy:

- wrapped tensors expose fake CUDA-visible properties
- tensor/module/device factory entry points are monkeypatched
- `torch.load(..., map_location=...)` loads on CPU first, then recursively rewrites tensors to fake CUDA objects
- selected `torch.cuda`, `torch.nn.parallel`, and `torch.distributed` surfaces are replaced with single-process shims

## Supported surface

### CUDA-visible tensor and module semantics

- `torch.device("cuda")`, `torch.device("cuda:N")`
- tensor creation with `device="cuda"` / `device="cuda:N"`
- `.cuda()`, `.to("cuda")`, `.cpu()`
- `tensor.device`, `tensor.is_cuda`
- `module.cuda()`, `module.to("cuda")`
- legacy tensor factories such as `torch.cuda.FloatTensor`

### CUDA management APIs

- `torch.cuda.is_available()`, `device_count()`, `current_device()`, `set_device()`
- `get_device_name()`, `get_device_capability()`, `get_device_properties()`
- `Stream`, `Event`, `stream(...)`, `current_stream()`, `default_stream()`, `set_stream()`, `device_of(...)`
- `manual_seed()`, `manual_seed_all()`, `seed()`, `seed_all()`, `initial_seed()`
- `get_rng_state()`, `get_rng_state_all()`, `set_rng_state()`, `set_rng_state_all()`
- `memory_allocated()`, `memory_reserved()`, `mem_get_info()`
- `memory_stats()`, `memory_summary()`, `memory_snapshot()`
- matching aliases under `torch.cuda.memory` and `torch.cuda.random`

### Parallel and distributed shims

- `torch.nn.DataParallel`
- `torch.nn.parallel.DistributedDataParallel`
- `torch.nn.parallel.comm.broadcast`, `scatter`, `gather`, `reduce_add`, `reduce_add_coalesced`
- single-process `torch.distributed` compatibility for:
  - `init_process_group`, `destroy_process_group`
  - `barrier`
  - `all_reduce`, `broadcast`
  - `all_gather`, `all_gather_into_tensor`, `all_gather_object`
  - `reduce`, `gather`, `scatter`
  - `reduce_scatter`, `reduce_scatter_tensor`
  - `all_to_all`, `all_to_all_single`
  - `broadcast_object_list`
  - private aliases `_all_gather_base`, `_reduce_scatter_base`

### Checkpoint and training compatibility

- `torch.save(...)`
- `torch.load(..., map_location="cuda:N")`
- `torch.load(..., map_location=torch.device("cuda:N"))`
- `torch.load(..., map_location={"cpu": "cuda:N"})`
- `torch.load(..., map_location={torch.device("cpu"): torch.device("cuda:N")})`
- recursive alias preservation for `OrderedDict`, lists, tuples, and shared tensors
- checkpoint restore for model, optimizer, scheduler, AMP scaler, and CUDA RNG state
- `torch.amp.autocast(device_type="cuda")`
- `torch.amp.GradScaler("cuda")`

## Known limits

- All compute is CPU-backed. This route is for compatibility, not performance.
- CUDA memory accounting is stubbed. Memory stats always report zero or fixed fake totals.
- Streams and events are API-compatible stubs only. They do not provide real asynchronous execution.
- Distributed support is single-process semantic compatibility only. No real multi-rank transport or collective execution happens in the custom torch shim.
- `torch.load(..., map_location=<callable>)` keeps upstream storage-callback behavior. Fake-CUDA target translation for callable return values is not implemented.
- This route does not make CUDA extensions, custom kernels, or storage-level CUDA allocators work on CPU-only builds.

## Maintained validation baseline

These tests are the maintained Phase 2 smoke baseline in the FakeGPU repository:

- `test/test_phase2_custom_torch_smoke.py`
- `test/test_phase2_cuda_api_surface.py`
- `test/test_phase2_parallel_api_surface.py`
- `test/test_phase2_distributed_api_surface.py`
- `test/test_phase2_checkpoint_state_surface.py`
- `test/test_phase2_torch_load_map_location_surface.py`
- `test/test_phase2_patch_bridge.py`
- `test/test_torch_patch.py`
- `test/test_torch_training.py`

## Recommended usage

### 1. Install the custom torch wheel

Build from the local `pytorch-fakegpu` fork, then install into the same environment used by FakeGPU.

### 2. Pick one activation path

Direct custom torch tests:

```bash
TORCH_FAKEGPU_ENABLE=1 python test/test_phase2_custom_torch_smoke.py
```

Existing training scripts that already use FakeGPU patching:

```python
from fakegpu.torch_patch import patch
patch()
```

The bridge keeps old user scripts working while preferring the installed custom torch fake-CUDA backend.

## When to stop extending Phase 2

Phase 2 is already good enough when your goal is:

- local script bring-up
- CPU-only debugging of CUDA-facing training code
- checkpoint and optimizer smoke validation
- single-process compatibility for code that expects `torch.cuda` and basic `torch.distributed` presence

If the next requirement is real CUDA execution, real allocator behavior, or real distributed communication, this route is no longer the right layer to extend.
