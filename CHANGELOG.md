# Changelog

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
