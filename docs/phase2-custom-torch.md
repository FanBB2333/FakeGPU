# Torch Patch System

FakeGPU's Python-level torch patch provides full CUDA-visible tensor semantics on CPU-only hosts. Unlike previous versions that required installing a separate `pytorch-fakegpu` custom PyTorch build, the upstream `FakeCudaTensor` code is now vendored directly into the FakeGPU package.

## Architecture

The torch patch uses a two-layer architecture:

| Layer | Source | Role |
|-------|--------|------|
| **Base layer** | Vendored `FakeCudaTensor` backend (`fakegpu/_upstream.py`, originally from `pytorch-fakegpu` by FanBB2333) | Core CUDA redirection using `torch.Tensor._make_subclass` + `__torch_function__` protocol |
| **Enhancement layer** | `fakegpu/torch_patch.py` | GPU profiles, memory tracking with OOM simulation, autocast dtype validation, GradScaler passthrough, cross-device operation validation, terminal summary reporting |

The base layer handles the hard problem (making `tensor.device` and `tensor.is_cuda` report CUDA), while the enhancement layer adds production-quality behavior on top.

## Data flow

1. `fakegpu.patch_torch()` (or `fakegpu.init(runtime="fakecuda")`) calls `patch()` in `torch_patch.py`.
2. `patch()` calls `_activate_upstream(num_devices, device_name)` which:
   - First tries `import torch.fakegpu` (installed custom PyTorch build, if present).
   - Falls back to `fakegpu._upstream` (vendored copy, always available).
   - Calls `upstream.enable()` to install the base FakeCudaTensor patching.
3. `patch()` then calls `_apply_enhancements_over_upstream(upstream, torch)` which layers FakeGPU additions on top.

## Key mechanism: FakeCudaTensor

The core problem is that `tensor.device` and `tensor.is_cuda` are C-level descriptors on `torch.Tensor` that cannot be monkeypatched on regular tensors. `FakeCudaTensor` solves this with a tensor subclass:

- Uses `torch.Tensor._make_subclass(cls, raw_data, requires_grad)` to create a subclass where the underlying data is a CPU tensor.
- Overrides the `device` property to return `torch.device(f"cuda:{device_index}")`.
- Overrides the `is_cuda` property to return `True`.
- Uses the `__torch_function__` protocol: unwrap args to CPU, execute the CPU op, rewrap the result as `FakeCudaTensor`.

## Enhancement layer

`_apply_enhancements_over_upstream` applies the following sections in order:

| Section | What it does |
|---------|-------------|
| 0 | **Device index bounds validation** — replaces upstream's permissive `_normalize_device_index` with a bounds-checked version that raises "CUDA error: invalid device ordinal" matching real CUDA behavior |
| 1 | **Memory tracker initialization** — per-device memory limits from GPU profiles |
| 2 | **Tensor creation hooks** — hooks `upstream.wrap_tensor` for automatic memory tracking on tensor creation |
| 3 | **GPU profiles** — 11 presets (see below); overrides `get_device_name`, `get_device_capability`, `get_device_properties` |
| 4 | **Memory query functions** — tracked memory queries replacing upstream's zero-returning stubs |
| 5 | **Autocast dtype validation** — bf16 requires compute capability >= 8.0; GradScaler passthrough |
| 6 | **Cross-device operation validation** — validates tensor ops, loss functions, functional ops, and binary dunders |
| 7 | **RNG state functions** — `get_rng_state`, `set_rng_state`, etc. (not provided by upstream) |
| 8 | **Terminal summary report** — prints memory usage summary on exit |

### GPU profiles

11 built-in profiles are available:

| Profile | Architecture | Compute Capability |
|---------|-------------|-------------------|
| `gtx980` | Maxwell | 5.2 |
| `p100` | Pascal | 6.0 |
| `v100` | Volta | 7.0 |
| `t4` | Turing | 7.5 |
| `a40` | Ampere | 8.6 |
| `a100` | Ampere | 8.0 |
| `a100-1g` | Ampere | 8.0 |
| `h100` | Hopper | 9.0 |
| `l40s` | Ada Lovelace | 8.9 |
| `b100` | Blackwell | 11.0 |
| `b200` | Blackwell | 11.0 |

## Supported surface

From the combination of both layers:

- `tensor.device == cuda:N`
- `tensor.is_cuda == True`
- `nn.DataParallel`
- `nn.DistributedDataParallel`
- `torch.distributed.*` (single-process shims for all collective ops)
- Autocast / GradScaler with dtype validation
- GPU profiles (11 presets)
- Memory tracking with OOM simulation
- Cross-device validation
- `torch.load` with `map_location` normalization
- Factory functions (`torch.randn`, `torch.zeros`, etc.) with `device="cuda"`
- Legacy tensor factories (`torch.cuda.FloatTensor`, etc.)
- Stream/Event API-compatible stubs

## Configuration

| Environment variable | Description | Default |
|---------------------|-------------|---------|
| `FAKEGPU_DEVICE_COUNT` | Number of fake devices | `8` |
| `FAKEGPU_PROFILE` | GPU profile preset (e.g. `a100`, `h100`) | — |
| `FAKEGPU_PROFILES` | Mixed profile configuration (e.g. `a100:4,h100:4`) | — |
| `FAKEGPU_DEVICE_NAME` | Custom device name | — |
| `FAKEGPU_STRICT_COMPAT` | Enable/disable strict compatibility checks | `1` |

## Usage

```python
import fakegpu
fakegpu.patch_torch()
import torch

# Everything below works on CPU
x = torch.randn(3, 3, device="cuda")
assert x.device.type == "cuda"
assert x.is_cuda is True

model = torch.nn.Linear(3, 3).cuda()
y = model(x)
```

With explicit runtime selection:

```python
import fakegpu
fakegpu.init(runtime="fakecuda")
```

## Verified PyTorch version

torch 2.9.1 is the only version tested so far.

## Known limitations

- All compute is CPU-backed — no actual GPU execution.
- `__torch_function__` overhead: ~2-3x slower than direct CPU tensor operations (measured via benchmark suite).
- Stream/Event are API-compatible stubs only (no real async).
- Distributed is single-process semantic compatibility only.
- CUDA extensions, custom kernels, and storage-level CUDA allocators do not work.
- Some internal PyTorch paths may bypass `__torch_function__` (rare).

## Test suite

12 test files, 58 tests total. All pass individually; cross-file isolation issues exist when running in the same process (pre-existing, due to module-level `_NUM_DEVICES` global state).

Key test files: `test_benchmark_overhead.py`, `test_dataloader_pin_memory.py`, `test_error_*.py`, `test_patch_advanced.py`, `test_hf_trainer.py`.

## Vendored upstream maintenance

`fakegpu/_upstream.py` is a verbatim copy of the upstream `FakeCudaTensor` code. Do not modify it directly — apply all enhancements in `torch_patch.py`. To re-sync, replace the file (simple file replacement). The attribution header at the top of the file is preserved.
