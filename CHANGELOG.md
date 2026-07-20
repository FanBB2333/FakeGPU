# Changelog

## Unreleased

### Added

- `fakegpu demo`, a minimal CPU-backed PyTorch forward/backward/optimizer example with profile and JSON output.
- `fakegpu doctor`, including installation checks, structured diagnostics, and a complete profile listing.
- Ten reference profiles: P4, A30, A10, Jetson AGX Orin 64GB, L4, H200, B300, Jetson T5000, RTX PRO 6000 Blackwell, and GB10.
- A checked-in snapshot and updater for NVIDIA's current and legacy model-to-compute-capability tables.

### Changed

- Python and native runtimes now consume the same 24-profile YAML catalog.
- Every profile declares an explicit compute capability, provenance, memory kind, and measured/reference/synthetic status.
- Native and Python architecture mapping now covers Maxwell through Blackwell, including Blackwell compute capabilities 10.0, 10.3, 11.0, 12.0, and 12.1.
- Native smoke validation now exercises all 15 represented compute capabilities.

### Fixed

- Corrected the previous B100/B200 compute-capability mismatch between the Python registry and native profiles.
- Added FP8 and FP4 profile capability parsing so Hopper, Ada, and Blackwell profiles remain available to native builds.
- Included profile YAML and NVIDIA catalog data in built wheels.

## v1.5.2 - 2026-07-20

Compared with `v1.5.1`.

### Added

- PyTorch compatibility coverage for Hugging Face Trainer, PEFT, TRL, LLaMA-Factory, Lightning Fabric, LitGPT, torchtune, Accelerate, and maintained FSDP paths.
- AI workload preflight CLI with stage-aware execution, fit/OOM classification, JSON schema validation, Markdown reports, allocation categories, shared-storage handling, logical-device attribution, and optional allocation stack traces.
- Exact `rtx3090ti` and `rtx-pro-5000-blackwell` profiles, including compute capability 12.0 and measured CUDA device attributes for the current 72 GB calibration server, plus the lightweight `test-512m` profile.
- Generic `real_gpu_calibration` suite that records the detected GPU, selects a matching fakecuda profile, executes real/passthrough/hybrid/fakecuda comparisons, checks deterministic result signatures, and verifies Hybrid clamp OOM behavior.
- Gradient accumulation and gradient checkpointing calibration workloads.
- Cross-GPU calibration aggregation that selects exact workload/profile matches and retains allocator, requested-byte, NVML process, and NVML device-delta observations.
- Target-device fake-tensor ATen forward/backward storage-liveness estimator with alias handling, phase-aware optimizer memory, CUDA Flash Attention auxiliary-storage profiles, per-stack backend-resident calibration, and cross-GPU validation reports.
- Operator-lifetime-aware workspace profiles, including a two-GPU-calibrated FP32 Efficient Attention backward profile and phase-resolved real CUDA measurements.

### Changed

- README reorganized around task selection, short runnable examples, runtime modes, capability boundaries, reports, and current validation evidence.
- Main-branch test artifacts now retain compact summaries while large generated HTML reports live on the reports branch.
- Empirical preflight calibration prefers an observed physical-memory upper bound for an exact GPU profile and workload signature instead of a universal multiplier.
- Static validation now separates allocator-allocated and allocator-requested errors and compares forward, backward, and optimizer phases independently.
- Package, native report, and C++ runtime versions are synchronized at `1.5.2`.

### Fixed

- Repeated `patch_torch()` calls now refresh device/profile state, memory tracking limits, and framework capability probes.
- Pytest no longer executes standalone Qwen, native CUDA, or PrivateUse1 scripts during collection.
- Per-profile `compute_major` overrides allow workstation Blackwell (12.0) and datacenter Blackwell profiles to coexist.
- Native CUDA Driver and Runtime stubs are split into separate libraries so NVML and Driver preloads do not leak Runtime symbols.
- Passthrough now runs as a zero-interposition real-GPU baseline; Hybrid keeps the real CUDA Runtime and cuBLAS while interposing the Driver/NVML surfaces it owns.
- Hybrid forwards the CUDA 12.x Driver symbols resolved directly by PyTorch/cuBLAS, restoring real GEMM, backward, and optimizer execution on RTX PRO 5000.
- FakeCudaTensor now reports its logical CUDA index through `get_device()`, enabling `torch.utils.checkpoint`.
- Memory tracking now skips storage-less functional-transform tensors such as `BatchedTensorImpl`, enabling current Transformers Qwen2 masking code that uses `vmap`.
- NVML exports the NVLink queries used by PyTorch OOM diagnostics, so Hybrid clamp raises `torch.cuda.OutOfMemoryError` instead of an internal missing-symbol assertion.
- Pytest collection isolates the standalone native Driver script so its `LD_PRELOAD` state cannot leak into Transformers/torchvision subprocess tests; runtime benchmarks now use adjacent, batched samples.

### Validation

- Full local suite: 182 passed and 1 skipped.
- Static-memory validation: 13 workloads across RTX 3090 Ti and RTX PRO 5000 Blackwell, for 26 GPU observations.
- Static peak bytes matched across both GPU/software stacks.
- Maximum allocated-byte absolute error: 0.077160%.
- Maximum requested-byte absolute error: 0.001358%.

## v1.5.1 - 2026-04-18

Compared with `v1.4.0`.

### Added

- Python-level PyTorch patch (`fakegpu.torch_patch`) with vendored FakeCudaTensor backend for CPU-only PyTorch builds (e.g. macOS).
- PrivateUse1 backend prototype (`fakegpu/privateuse1/`) for native PyTorch device registration.
- Per-GPU peak VRAM summaries in the report output.
- Unified test report entry with hash routing in HTML report.
- GPU error simulation experiments with unified report integration.
- Torch 2.6.0–2.11.0 compatibility matrix test results.
- Phase 5/6/7 and B3 validation suites for torch_patch coverage.
- Real-scene validation with nanoGPT and MoE training flows.
- Terminal Report Summary output for torch_patch.
- GPU profile-based dtype compatibility enforcement.
- v4 GPU profile and kernel GEMM stats in report.

### Changed

- Verified PyTorch compatibility expanded to torch 2.6.0, 2.7.1, 2.8.0, 2.9.1, 2.10.0, 2.11.0.
- Report version unified with FakeGPU package version.
- Torch patch documentation rewritten for vendored upstream architecture.

### Fixed

- macOS compatibility and NCCL loading improvements.
- macOS verification portability.
- torch_patch device context manager and legacy autocast API.
- torch 2.6–2.11 compatibility in torch_patch.
- Runtime detection of editable torch fakegpu installs.
- Native smoke metrics preservation in reports.

### Included work

- `feat(torch_patch): add Python-level CUDA-to-CPU redirect for CPU-only PyTorch`
- `feat(privateuse1): add fgpu phase1 prototype`
- `feat(torch_patch): vendor upstream FakeCudaTensor as layered backend`
- `feat(report): add per-gpu peak VRAM summaries`
- `feat(report): add unified test report entry with hash routing`
- `feat(error-sim): add GPU error simulation experiments with unified report`
- `fix(torch_patch): support torch 2.6-2.11 compatibility`
- `fix(macos): improve compatibility and nccl loading`

## v1.4.0 - 2026-04-14

Compared with `v1.3.0`.

### Added

- A bilingual MkDocs documentation site with English as the default language and Simplified Chinese companion pages.
- Validation guides for bring-up, distributed simulation, cluster reporting, and common `fgpu` workflows.
- Packaging and docs-build adjustments so release and documentation workflows resolve the project version consistently.

### Changed

- The maintained validation baseline is now documented more explicitly, including direct NCCL checks, simulate-mode DDP smoke paths, and hybrid multinode coverage.
- Documentation wording now distinguishes maintained smoke coverage from full PyTorch, NCCL, or Transformers parity claims.

### Fixed

- GitHub release/docs workflow issues that previously broke versioned packaging and pages deployment behavior.
- Grouped NCCL submission on the implicit default stream in simulate mode.
- The `test/run_multinode_sim.sh` rendezvous setup so local torch distributed launch uses an explicit loopback address and port.
- Fake CUDA stream/event completion bookkeeping for synchronous fake NCCL work, which unblocks simulate-mode `ProcessGroupNCCL` / DDP flows and removes the observed `ALLGATHER` timeout during teardown.

### Included work

- `fix(ci): repair release and docs workflows`
- `fix(docs): stop auto-enabling GitHub Pages`
- `docs: add bilingual docs site and validation guide`
- `fix(nccl): restore grouped default-stream semantics and align docs`
- `fix(ddp): complete fake nccl stream events`
