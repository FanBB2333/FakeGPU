# Changelog

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
