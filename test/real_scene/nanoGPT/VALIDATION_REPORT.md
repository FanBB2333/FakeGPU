# FakeGPU Real-Scene Validation Report — nanoGPT

**Date:** 2026-04-16
**Test model:** nanoGPT character-level Shakespeare (baby GPT, ~10.65M params)
**Training iterations:** 20
**Runner:** `test/real_scene/nanoGPT/train_wrapper.py`

## Test Environment

### macOS
- Platform: macOS, no physical NVIDIA GPU
- Baseline/partial Python: `/Users/l1ght/miniforge3/envs/py311/bin/python`
- Baseline/partial PyTorch: `2.9.1`
- Full Python: `/Users/l1ght/miniforge3/envs/fakegpu/bin/python`
- Full PyTorch: `2.11.0a0+gitb926036`
- `torch.fakegpu`: available in the full environment
- FakeGPU build: `.dylib` stubs present in `build/`

### Linux
- Platform: Linux, RTX 3090 Ti (24GB)
- Baseline/partial Python: `/home/l1ght/anaconda3/envs/fakegpu/bin/python`
- Baseline/partial PyTorch: `2.9.1+cu128`
- Full Python: `/home/l1ght/anaconda3/envs/fakegpu-macos/bin/python`
- Full PyTorch: `2.11.0+fakegpu`
- `torch.fakegpu`: available in the full environment
- FakeGPU build: `.so` stubs present in `~/repos/fakeGPU/build/`

## Test Results Summary

| # | Test | Platform | Config | Result | Notes |
|---|------|----------|--------|--------|-------|
| 1A | Baseline | macOS | No FakeGPU | FAIL | Expected. `torch.cuda.is_available()` is `False`; `model.to(device)` fails with `Torch not compiled with CUDA enabled`. |
| 1B | Partial | macOS | `fakegpu` + native PyTorch | FAIL | Expected. Native runtime initializes, but CPU-only PyTorch still reports no CUDA. |
| 1C | Full | macOS | `fakegpu` + `torch.fakegpu` | PASS | Training completed 20 iterations after patching `pin_memory()` to no-op when fakecuda probing detects no pinned-memory support. |
| 2 | OOM | macOS | Full + 1GB virtual VRAM | BLOCKED | Current `custom_torch` backend ignores total-memory override; oversized `cuda` allocation still succeeds. |
| 3A | Baseline | Linux | Real GPU, no FakeGPU | PASS | Re-run after freeing remote GPU resources passed end-to-end in 6.9s. |
| 3B | Partial | Linux | `fakegpu` + native PyTorch | FAIL | Fake devices enumerate correctly, but forward pass fails with `M should be less than maximum CUDA grid size`. |
| 3C | Full | Linux | `fakegpu` + `torch.fakegpu` | INCONCLUSIVE | Full runtime initializes and fake devices appear, but training did not reach `step 0` within a reasonable window even after reducing `eval_iters` to 20. |

## Detailed Results

### Test 1A: macOS Baseline
- Log: `logs/macos_1a_baseline.log`
- Result: `FAIL`
- Key output:
  - `torch.cuda.is_available(): False`
  - `AssertionError: Torch not compiled with CUDA enabled`
- Analysis:
  - This matches the baseline expectation on a machine with no CUDA-capable PyTorch build.

### Test 1B: macOS Partial FakeGPU
- Log: `logs/macos_1b_partial.log`
- Result: `FAIL`
- Key output:
  - `fakegpu.init(runtime='native') -> runtime=native, backend=native`
  - `torch.cuda.is_available(): False`
  - `AssertionError: Torch not compiled with CUDA enabled`
- Analysis:
  - The C-layer stubs load, but CPU-only PyTorch never enters a CUDA path, so interception is ineffective on macOS.

### Test 1C: macOS Full FakeGPU
- Log: `logs/macos_1c_full.log`
- Result: `PASS`
- Key output:
  - `fakegpu.init(runtime='auto') -> runtime=fakecuda, backend=custom_torch`
  - `Patched torch.Tensor.pin_memory() to no-op for fakecuda after probe failure`
  - `iter 20: loss 2.7556`
  - `Training completed successfully in 592.2s`
- Analysis:
  - Full Python-level patching is sufficient to run nanoGPT on macOS.
  - One extra compatibility patch was required: on this host/build combination, `pin_memory()` raises even though fake CUDA is enabled, so the wrapper now probes and disables pinned-memory calls when unsupported.

### Test 2: macOS OOM Simulation
- Log: `logs/macos_2_oom.log`
- Result: `BLOCKED`
- Key output:
  - `runtime fakecuda backend custom_torch`
  - `torch.cuda.mem_get_info() (85899345920, 85899345920)`
  - `allocation_succeeded 350000000 cuda:0`
  - `oom_probe_result NO_OOM`
- Analysis:
  - The current `custom_torch` backend does not honor `FAKEGPU_TOTAL_MEMORY` / `TORCH_FAKEGPU_TOTAL_MEMORY`.
  - Because an allocation larger than 1GB still succeeds and reported memory remains 80GB, the planned OOM simulation is not meaningful in the current implementation.

### Test 3A: Linux Baseline
- Log: `logs/linux_3a_baseline.log`
- Result: `PASS`
- Key output:
  - `torch.cuda.is_available(): True`
  - `Device 0: NVIDIA GeForce RTX 3090 Ti`
  - `iter 20: loss 2.7335`
  - `Training completed successfully in 6.9s`
- Analysis:
  - The first attempt failed only because another process was occupying ~22GB of VRAM.
  - After remote GPU resources were freed, the baseline run passed cleanly and should be treated as the authoritative result.

### Test 3B: Linux Partial FakeGPU
- Log: `logs/linux_3b_partial.log`
- Result: `FAIL`
- Key output:
  - `fakegpu.init(runtime='native') -> runtime=native, backend=native`
  - `torch.cuda.device_count(): 8`
  - `Device 0: Fake NVIDIA A100-SXM4-80GB`
  - `RuntimeError: M should be less than maximum CUDA grid size`
- Analysis:
  - Native interception clearly affects device discovery and properties on Linux.
  - However, this nanoGPT workload still trips a CUDA-kernel/runtime limitation before completing the validation run, so the expected PASS did not materialize.

### Test 3C: Linux Full FakeGPU
- Log: `logs/linux_3c_full.log`
- Result: `INCONCLUSIVE`
- Key output:
  - `fakegpu.init(runtime='auto') -> runtime=fakecuda, backend=custom_torch`
  - `Patched torch.Tensor.pin_memory() to no-op for fakecuda after probe failure`
  - `torch.cuda.device_count(): 8`
  - No `step 0` / training-iteration output before the run was stopped
- Analysis:
  - Initialization succeeds and the same pinned-memory compatibility issue is handled.
  - Even after reducing `eval_iters` to 20, the Linux `torch 2.11.0+fakegpu` full backend did not complete the initial validation stage within a reasonable runtime window. This needs backend-level profiling before it can be called PASS or FAIL.

## Analysis & Conclusions

### macOS
- Baseline and partial behave exactly as the architecture predicts: CPU-only PyTorch never becomes CUDA-capable through native-library interception alone.
- Full fakecuda works for real nanoGPT training, but only after disabling pinned-memory calls that are unsupported on this host/backend combination.

### Linux
- Real-GPU baseline is healthy once the machine is not under external VRAM pressure.
- Native FakeGPU interception on Linux is strong enough to replace device enumeration, but this workload still fails deeper in the runtime with a CUDA grid-size error.
- Full fakecuda initialization succeeds, but the current Linux fakegpu build is too slow or otherwise stalled in early execution to complete this validation run within the test window.

### OOM Fidelity
- The planned OOM simulation is not currently testable on the full fakecuda path because the `custom_torch` backend ignores the configured memory limit.
- Before re-attempting Task 2, FakeGPU needs either:
  - real allocation accounting in the `custom_torch` backend, or
  - a separate backend path that enforces virtual VRAM limits.

### Practical Outcome
- Confirmed working:
  - macOS full fakecuda path
  - Linux real-GPU baseline
- Confirmed failing:
  - macOS baseline
  - macOS partial
  - Linux partial
- Not yet fully validated:
  - macOS OOM simulation
  - Linux full fakecuda training completion
