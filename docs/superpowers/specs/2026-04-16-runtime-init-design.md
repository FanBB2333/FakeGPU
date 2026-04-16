# FakeGPU Runtime Init Design

## Goal

Make `import fakegpu` side-effect free, and route initialization through a single
public API:

```python
fakegpu.init(runtime="auto" | "native" | "fakecuda")
```

`auto` prefers an installed `torch.fakegpu` backend when present, and otherwise
falls back to the native preload-based FakeGPU runtime.

## Requirements

- `import fakegpu` must not import `torch`, `fakegpu.privateuse1`, or
  `fakegpu.torch_patch`.
- Users can explicitly choose:
  - `runtime="native"` for the preload/shared-library route
  - `runtime="fakecuda"` for the Python-level fake-CUDA route
  - `runtime="auto"` to prefer `torch.fakegpu` when installed, else use native
- The standalone fake-CUDA fallback should cover checkpoint RNG save/restore
  flows used by the maintained training smoke.
- Existing advanced entry points remain available:
  - `fakegpu.patch_torch()`
  - `fakegpu.init_privateuse1()`

## Design

### Public API

- Keep `fakegpu.init(...)` as the single top-level initializer.
- Add a `runtime` keyword with values `auto`, `native`, and `fakecuda`.
- Default to `auto`.

### Import Boundaries

- `fakegpu.__init__` only imports lightweight helpers from `fakegpu._api`,
  `fakegpu._runtime`, and `_version`.
- `patch_torch()` and `init_privateuse1()` become lazy wrappers that import
  their heavy modules only when called.

### Runtime Routing

- `native` delegates to the existing preload-based initializer in `_api.py`.
- `fakecuda` delegates to `fakegpu.torch_patch.patch()`.
- `auto` checks whether `torch.fakegpu` is installed without importing `torch`
  first. If found, it uses `fakecuda`; otherwise it uses `native`.

### Return Value

- The top-level initializer returns a runtime-aware result object that keeps
  compatibility for native callers via `.lib_dir` and `.handles`.
- `native` fills those fields from the preload init result.
- `fakecuda` leaves them empty and records which fake-CUDA backend was enabled.

### Standalone Fake-CUDA Gaps To Fix

- Patch `torch.cuda` and `torch.cuda.random` RNG helpers so
  `get_rng_state[_all]()` and `set_rng_state[_all]()` work in the fallback path.
- Expose a stable `torch.cuda.default_generators` tuple in the fallback path.
- Keep RNG behavior CPU-backed and deterministic even though tensors are created
  on CPU after device normalization.

## Validation

- `python3 -c "import fakegpu"` must succeed in an environment where `torch`
  either does not exist or does not expose `torch._C._acc`.
- `fakegpu.init(runtime="native")` must still initialize the preload runtime.
- `fakegpu.init(runtime="auto")` must choose `fakecuda` when `torch.fakegpu`
  exists and `native` otherwise.
- The standalone fallback must pass the maintained training smoke checkpoint
  restore flow.
