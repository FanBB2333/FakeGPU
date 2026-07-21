# Changelog

## Unreleased

### Added

- A Git-revision-aware physical multi-host controller that launches heterogeneous Hybrid DDP, collective-mismatch, and missing-peer cases over SSH, including Windows-to-WSL command execution and combined JSON/Markdown reports.
- A `distributed_resilience` suite covering persistent async errors, TCP communicator timeouts, disabled operation retention, bounded rolling retention, and small-message report stress.
- Two-rank real-CUDA Hybrid FSDP validation covering parameter sharding, reduce-scatter gradient averaging, optimizer updates, full-parameter reconstruction, and full-state-dict serialization/restoration.
- Two- and four-rank real-CUDA FSDP2/DeviceMesh/DTensor validation covering FP32, FP16, and BF16 parameters, FP32 or parameter-dtype gradient reduction, optimizer updates, and full-tensor reconstruction.
- DDP numerical variants for `no_sync`, rank-dependent unused parameters, `static_graph`, and gradient bucket views, available on one physical GPU and across two SSH hosts.
- Fake NCCL FP16/BF16 collective payloads, `ncclAvg`, and FP16/BF16 `ncclRedOpCreatePreMulSum` execution in simulate mode.
- Per-operation collective data type and reduction operator fields in cluster JSON and Markdown timelines.
- Physical multi-host controller cases for FSDP2 FP32, mixed-precision parameters, and low-precision gradient reduction.
- Default cross-field cluster-report checks that reconcile collective/P2P counters with the operation timeline and reconcile directional links with node-pair totals.
- A checkpoint-only dense-decoder inference estimator that reads safetensors headers without loading tensor payloads and reports parameter storage, KV cache, transient tensors, process memory, and prefill/decode matrix FLOPs.
- A virtual `fakegpu nvidia-smi` view backed by per-process FakeCUDA memory state, with optional same-stack NVML runtime-overhead calibration.
- Real-CUDA/FakeCUDA Qwen inference workers and a comparison report covering model load, inference peak, NVML process memory, generated tokens, and observed/static FLOPs.
- A matrix-heavy FLOP counter that preserves PyTorch operator decomposition under inference mode and supports grouped-query SDPA with different query and KV head counts.
- Reproducible Qwen3.5 SFT workers for full-parameter, LoRA, and PyTorch-native packed-NF4 QLoRA execution on real CUDA, FakeCUDA, and static ATen graphs.
- Optional QLoRA nested scale quantization using the bitsandbytes dynamic 8-bit map, 256-value second-level blocks, and an FP32 mean offset.

### Fixed

- Preserved non-default CUDA stream ordering at the Hybrid host-staging boundary by copying on the collective stream and synchronizing before coordinator access. This restores correct FSDP pre-divided gradients with PyTorch 2.12/CUDA 13 while retaining PyTorch 2.9/CUDA 12 behavior.
- Replaced PyTorch's equal-head fused-SDPA FLOP assumption in the Qwen verifier, allowing Qwen3 grouped-query attention (32 query heads and 8 KV heads) to be measured without assertion failures.
- Isolated the `Tensor.to("cuda")` redirect microbenchmark from random-tensor creation variance while preserving its 100-microsecond incremental-overhead limit.
- Corrected native NF4 first-level statistics to FP32, counted every materialized quantization buffer in static estimates, and preserved explicit FP32 buffers in meta-model construction.
- Added the known native-NF4 dequantization workspace to FakeCUDA compute peaks while retaining the unadjusted CPU tracker phases for inspection.

### Validation

- Full local suite: 271 passed and 1 skipped.
- The distributed resilience suite passed on macOS, the RTX PRO 5000 Linux host, and the RTX 3090 Ti WSL host; its 256-operation case retained the newest 64 timeline entries and counted 192 discarded entries.
- The automated heterogeneous two-host run used the same `3e6c8b2` commit on both hosts. Hybrid DDP produced gradient `[1.5, 3.0]` and parameters `[0.85, -0.30]`; both mismatched ranks persisted async error 5; the missing-peer case timed out in 0.755 seconds. The cluster report recorded six successful collectives, 192 bytes between the node pair, a 64-byte per-operation peak, and one expected timeout.
- Single-host Hybrid DDP/FSDP passed on both the RTX PRO 5000 Blackwell (compute capability 12.0, PyTorch 2.9.1/CUDA 12.8) and RTX 3090 Ti (compute capability 8.6, PyTorch 2.12.1/CUDA 13.0). FSDP produced local shard gradients `[1.5]` and `[3.0]`, reconstructed `[0.85, -0.30]`, and restored the same values from a full state dict.
- The complete heterogeneous two-host run used commit `fb852c7` and passed basic DDP, all three DDP option cases, FSDP, collective mismatch, and missing-peer timeout. Its report reconciled 34 completed collectives, 1,104 node-pair bytes, a 128-byte per-operation peak, and one expected timeout with no discarded timeline entries.
- The FSDP2 matrix passed 20 single-host combinations across the RTX PRO 5000 and RTX 3090 Ti: two/four ranks, FP32/FP16/BF16 parameters, and FP16/BF16 parameter-dtype gradient reductions.
- Heterogeneous two-host FSDP2 passed FP32/FP16/BF16 parameter cases plus FP16/BF16 gradient reduction. The low-precision report retained eight operations, 160 node-pair bytes, and a 32-byte per-operation peak while identifying `float16`/`bfloat16` payloads and `sum`/`avg` reductions.
- Qwen3-8B BF16 SDPA validation on the RTX PRO 5000 matched 8,190,735,360 parameters and both generated token IDs. FakeCUDA load and inference-peak errors versus the real CUDA allocator were 0.012914% and 0.064877%; the checkpoint-only peak error was 0.067234%; calibrated virtual-SMI process memory differed from NVML by 0.063182%.
- Qwen3-8B observed matrix FLOPs matched exactly between CPU-backed FakeCUDA and real CUDA at 151,415,620,864; the shape estimate differed by 1,280 FLOPs (0.000001%).
- Five direct-scale and five nested-scale QLoRA cases using Qwen3.5-0.8B/2B all passed with static peak errors at or below 1.732%. Nested scales reduced storage by 0.370-0.372 bit per quantized weight; the complete 0.8B FakeCUDA step differed from real CUDA by 0.165% overall.
- RTX 3090 Ti Ampere and RTX PRO 5000 Blackwell reproduced both maintained 0.8B nested-scale cases byte-for-byte, including real/static peaks, persistent storage, and random-batch fingerprints.

## v1.5.3 - 2026-07-20

Compared with `v1.5.2`.

### Added

- `fakegpu demo`, a minimal CPU-backed PyTorch forward/backward/optimizer example with profile and JSON output.
- `fakegpu doctor`, including installation checks, structured diagnostics, and a complete profile listing.
- Ten reference profiles: P4, A30, A10, Jetson AGX Orin 64GB, L4, H200, B300, Jetson T5000, RTX PRO 6000 Blackwell, and GB10.
- A checked-in snapshot and updater for NVIDIA's current and legacy model-to-compute-capability tables.
- `fakegpu coordinator` and `fakegpu bandwidth` commands for chosen-port TCP coordination, local logical-node simulation, physical multi-host rank launch, correctness checks, and end-to-end throughput reports.
- A real-CUDA DDP numerical check for averaged gradients and cross-rank parameter consistency through simulated NCCL collectives.
- Complete node-pair communication matrices in cluster JSON reports, plus an automatically generated Markdown project report with collective, P2P, node-pair, rank, and recent-operation tables.
- A formal `cluster_report.v1` JSON contract and default schema validation.
- A bounded coordinator-observed communication timeline with global communicator ranks, logical/socket payloads, rendezvous time, execution time, and topology-modeled time.

### Changed

- Python and native runtimes now consume the same 24-profile YAML catalog.
- Every profile declares an explicit compute capability, provenance, memory kind, and measured/reference/synthetic status.
- Native and Python architecture mapping now covers Maxwell through Blackwell, including Blackwell compute capabilities 10.0, 10.3, 11.0, 12.0, and 12.1.
- Native smoke validation now exercises all 15 represented compute capabilities.
- TCP coordinators now carry collective and point-to-point data in socket payloads so ranks can communicate across physical hosts; `FAKEGPU_COORDINATOR_TIMEOUT_MS` controls rank rendezvous and collective waits.
- Link and node-pair reports now retain directional totals, per-operation peak payloads, modeled average/peak throughput, estimated time, and contention penalties.
- Cluster reports distinguish collective and P2P operations at the link, node-pair, and rank levels.
- Package, native report, and C++ runtime versions are synchronized at `1.5.3`.

### Fixed

- Corrected the previous B100/B200 compute-capability mismatch between the Python registry and native profiles.
- Added FP8 and FP4 profile capability parsing so Hopper, Ada, and Blackwell profiles remain available to native builds.
- Included profile YAML and NVIDIA catalog data in built wheels.
- Removed fake NCCL's hard dependency on the FakeGPU CUDA Driver, allowing simulated communication to coexist with a process that uses the physical CUDA Driver.
- Kept macOS collective shared-memory names within Darwin's 31-character POSIX limit.
- Sized hybrid NCCL pointer-attribute storage for both CUDA 12 and CUDA 13 Runtime ABIs.
- Preserved local-to-global rank membership across communicator splits so subgroup traffic is attributed only to participating cluster nodes.
- Counted successful NCCL send/recv operations in cluster traffic reports without double-counting the matching receive endpoint.

### Validation

- Full local suite: 222 passed and 1 skipped.
- Dedicated four-node accounting coverage verifies a reordered `[2, 0]` subgroup plus independent `0 → 1` and `3 → 2` P2P transfers.
- TCP payload validation verifies non-zero request/response bytes and coordinator-observed timing for every retained all-reduce operation.
- Hybrid DDP numerical validation passed on an RTX PRO 5000 with PyTorch 2.9.1/CUDA 12.8 and an RTX 3090 Ti with PyTorch 2.12.1/CUDA 13.0.
- A heterogeneous physical two-host Hybrid DDP check passed between those software stacks: fake NCCL carried broadcast, all-reduce, and all-gather over TCP, producing the expected averaged gradient and identical updated parameters without coordinator timeouts.
- A physical two-host Tailscale test completed correct 1 MiB and 16 MiB all-reduces with zero coordinator timeouts; the 16 MiB × 5 case measured about 0.261 Gbit/s algorithmic throughput and 0.521 Gbit/s bidirectional socket payload per rank.

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
