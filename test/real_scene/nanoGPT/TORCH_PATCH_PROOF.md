# FakeGPU torch_patch Proof Experiments

**Date:** 2026-04-17
**Scope:** CPU-backed fakecuda (`fakegpu.init(runtime='fakecuda')`) terminal summary semantics
**Artifacts:** `torch_patch_proof_results.json`, `test/report.html`, `test/real_scene/nanoGPT/VALIDATION_REPORT.md`

## Key Findings

- The current fakecuda `torch_patch` summary is weight/storage-oriented, not a full training-peak allocator trace.
- Load-only peak memory scales correctly from the existing 2.41M MoE baseline to 520M and 1.0B parameter configurations.
- The pure `torch_patch` path now honors the `a100-1g` profile limit and keeps device-count/profile state consistent.

## Experiment Summary

| ID | Experiment | Result | Key observation |
|---|---|---|---|
| P3-5 | Summary Scope Probe (Elementwise + Clone) | PASS | `x + y` stayed at 8.0 MB after two 4 MB inputs, so op outputs are still outside the current tracker scope. |
| P3-6 | Load-Only Scaling (520M MoE) | PASS | 520.22M params produced a 1.94 GiB peak, matching fp32 weight size rather than full training peak. |
| P3-7 | Load-Only Scaling (1.0B MoE) | PASS | 1001.61M params produced a 3.73 GiB peak, showing summary scaling remains linear for load-only weight tracking. |
| P3-8 | Small-VRAM OOM (1.0B MoE on a100-1g) | PASS | `a100-1g` now reports 1.00 GiB with 2 synchronized profile entries and raises OOM as expected. |

## Detailed Results

### P3-5: Summary Scope Probe (Elementwise + Clone)
- Description: Proves the current torch_patch summary tracks explicit fake-CUDA storages but does not account for most op-produced activation tensors.
- Result: `PASS`
- Note: This is an expected limitation probe, not a bug regression. It documents that op-produced tensors like `x + y` are not yet tracked.
- Metrics:
```text
after_xy: 8.0 MB
after_add: 8.0 MB
after_clone: 8.0 MB
after_zeros_like: 8.0 MB
peak: 8.0 MB
alloc_calls: 2
```
- Report Summary:
```text
======================================================
             FakeGPU Report Summary
======================================================
 Device 0: NVIDIA A100-SXM4-80GB (Ampere, cc 8.0)
   Memory: 8.0 MB / 80.0 GB peak (0.0%)
   Alloc: 2 calls | Free: 2 calls
------------------------------------------------------
======================================================
```

### P3-6: Load-Only Scaling (520M MoE)
- Description: Shows that Report Summary peak memory scales with parameter count for weight-only model loading.
- Result: `PASS`
- Note: Peak memory matches fp32 parameter bytes, which is the correct behavior for the current weight-only tracker.
- Metrics:
```text
params: 520.22M
fp32_weight_bytes: 1.94 GiB
memory_allocated: 1.94 GiB
memory_peak: 1.94 GiB
alloc_calls: 255
```
- Report Summary:
```text
======================================================
             FakeGPU Report Summary
======================================================
 Device 0: NVIDIA A100-SXM4-80GB (Ampere, cc 8.0)
   Memory: 1.9 GB / 80.0 GB peak (2.4%)
   Alloc: 255 calls | Free: 255 calls
------------------------------------------------------
======================================================
```

### P3-7: Load-Only Scaling (1.0B MoE)
- Description: Confirms the same weight-tracking behavior at roughly 1B parameters.
- Result: `PASS`
- Note: The 1B-class model still reports weight-size peak memory rather than full optimizer+activation training memory.
- Metrics:
```text
params: 1001.61M
fp32_weight_bytes: 3.73 GiB
memory_allocated: 3.73 GiB
memory_peak: 3.73 GiB
alloc_calls: 507
```
- Report Summary:
```text
======================================================
             FakeGPU Report Summary
======================================================
 Device 0: NVIDIA A100-SXM4-80GB (Ampere, cc 8.0)
   Memory: 3.7 GB / 80.0 GB peak (4.7%)
   Alloc: 507 calls | Free: 507 calls
------------------------------------------------------
======================================================
```

### P3-8: Small-VRAM OOM (1.0B MoE on a100-1g)
- Description: Verifies pure torch_patch fakecuda now honors the a100-1g profile memory limit and device-count/profile synchronization.
- Result: `PASS`
- Note: This probe exercises pure torch_patch fakecuda without the wrapper's separate memory limiter.
- Metrics:
```text
params: 1001.61M
device_count: 2
profile_count: 2
tracker_count: 2
reported_total_memory: 1.00 GiB
error: CUDA out of memory. Tried to allocate 9.00 MiB. GPU 0 has a total capacity of 1.00 GiB of which 0.00 GiB is free.
```
- Report Summary:
```text
======================================================
             FakeGPU Report Summary
======================================================
 Device 0: NVIDIA A100-SXM4-1GB (Ampere, cc 8.0)
   Memory: 1021.3 MB / 1.0 GB peak (99.7%)
   Alloc: 123 calls | Free: 123 calls
------------------------------------------------------
 Device 1: NVIDIA A100-SXM4-1GB (Ampere, cc 8.0)
   Memory: 0 B / 1.0 GB peak (0.0%)
   Alloc: 0 calls | Free: 0 calls
------------------------------------------------------
======================================================
```
