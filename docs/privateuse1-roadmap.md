# FakeGPU PrivateUse1 Prototype Report

## Summary

Phase 1 adds a CPU-backed `fgpu` device on top of PyTorch `PrivateUse1`.

The current prototype is intentionally scoped for local debug on non-CUDA hosts:

- `torch.device("fgpu")` works
- `tensor.to("fgpu")` works
- `torch.fgpu.set_device()` / `torch.fgpu.current_device()` work
- `with torch.fgpu.device(i)` scopes the default `fgpu` target device
- `module.fgpu()` works
- tied/shared parameters keep aliasing across `.fgpu()`
- forward / backward / Adam one-step smoke works
- `torch.save()` works
- `torch.load(..., map_location="fgpu")` works via CPU load plus recursive tensor conversion
- `torch.load(..., map_location="fgpu")` preserves `OrderedDict` containers and shared tensor references
- `torch.amp.autocast(device_type="fgpu", enabled=False)` works

## Implementation Notes

- Python-visible `fgpu` registration uses `torch.utils.rename_privateuse1_backend("fgpu")`.
- Backend state now tracks `device_count`, `current_device`, and scoped device contexts in Python.
- Tensor compute does not rely on native PrivateUse1 autograd kernels.
- `FgpuTensor.__torch_function__` unwraps to CPU `raw_data`, executes CPU ops, and wraps results back into `fgpu` tensors.
- `FgpuTensor.backward()` delegates to `raw_data.backward()` and then syncs grads back to wrapped parameters.
- Each wrapped tensor carries a `device_index`, so Python-visible `tensor.device` is no longer hard-coded to `fgpu:0`.
- `module.fgpu()` is overridden to preserve CPU `raw_data` for parameters and buffers; the auto-generated `Module.fgpu()` from PyTorch dropped this metadata through `Parameter(...)` reconstruction.
- `module.fgpu()` now memoizes converted parameters and buffers across the whole module tree so tied weights remain shared after conversion.
- `torch.load(..., map_location="fgpu")` is patched to load on CPU first, then recursively convert tensors to `fgpu`. This avoids the current storage-level `PrivateUse1` restore crash on CPU-only builds.
- The recursive `torch.load(..., map_location="fgpu")` conversion now memoizes visited objects so shared tensor aliases survive conversion and `OrderedDict` containers stay ordered.

## Supported in Phase 1

- `torch.device("fgpu:0")`
- `tensor.device.type == "fgpu"`
- `tensor.is_fgpu`
- `module.fgpu()`
- CPU-backed forward/backward through wrapped tensors
- single-step optimizer smoke for Adam
- `torch.load(..., map_location=torch.device("fgpu"))`
- `torch.amp.autocast(device_type="fgpu", enabled=False)`

## Explicitly Unsupported in Phase 1

- `device="cuda"`
- `tensor.is_cuda`
- `tensor.device.type == "cuda"`
- `DataParallel`
- distributed collectives on `fgpu`
- native `PrivateUse1` storage restore without the `torch.load` patch
- `torch.Generator(device="fgpu")`
- claims of parity with CUDA backend semantics

## Decision Checklist

- Phase 1 keeps a separate `fgpu` device name.
- Phase 1 does not satisfy CUDA semantic compatibility.
- If the remaining blockers are `is_cuda`, `tensor.device == cuda`, `DataParallel`, CUDA-native RNG, or third-party hard-coded CUDA checks, continue to Stage 2.
- If the current `fgpu` path is sufficient for local model debug, stop after Phase 1 and maintain only the custom-device route.
