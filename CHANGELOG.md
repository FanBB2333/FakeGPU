# Changelog

## Unreleased

### Fixed

- FakeCUDA autocast now defaults to float16 like CUDA instead of the CPU
  bfloat16 default, so `torch.amp.autocast("cuda")` no longer fails on
  pre-Ampere profiles; the positional `dtype` form and the legacy
  `torch.cuda.amp.autocast(enabled, dtype, cache_enabled)` signature work.
- Module buffers moved with `Module.to`/`.cuda` are reported in the `buffer`
  memory category again instead of `tensor`.
- Calling `patch_torch()` again with more devices now records copies on the
  added devices; transfer counters restart with the new topology.
- FakeCUDA `torch.cuda.get_device_properties()` reports the profile's SM
  count, threads per multiprocessor, and register limits instead of A100
  values for every profile, matching the native runtime.
- The first virtual-SMI state written by a FakeCUDA process reports the
  `upstream` backend instead of `unknown`.
- Under FakeCUDA, `torch.get_float32_matmul_precision()` returns `medium`
  after `torch.set_float32_matmul_precision("medium")` instead of `high`.
- `fakegpu bandwidth --size`/`--bytes` rejects a bare `ib` suffix with a usage
  error instead of crashing with `KeyError`.
- `fakegpu bandwidth --listen` without `--cluster-report` no longer prints the
  path of a cluster report inside an already deleted temporary directory.
- `fakegpu simulate-topology` reports an unknown rank used as both source and
  destination as a usage error instead of crashing with `KeyError`.
- `fakegpu replay-trace` accepts `"links": null` in generic and FakeGPU
  cluster timeline events instead of crashing with `TypeError`.
- Modeled link traffic drops copy segments older than the rate window when a
  copy is recorded, so its memory no longer grows with every copy in
  processes that do not publish SMI state.
- `fakegpu platform show`/`env` report an invalid machine configuration with
  exit code 2 instead of silently showing the default HGX A100 machine.
- Count MoE layers with both `mlp_only_layers` and `decoder_sparse_step`, as
  Transformers' Qwen MoE models do. Saved Qwen MoE configs always list
  `mlp_only_layers`, so a sparse step above one was ignored and every
  non-listed layer was counted as MoE in LLM, serving, and fine-tuning
  estimates.
- Treat the leading `first_k_dense_replace` layers of DeepSeek-style MoE
  configs as dense. They were counted as MoE layers, so LLM, serving, and
  fine-tuning estimates, serving plans, and workload signatures for these
  models change.
- Report a malformed `model.safetensors.index.json` (non-object index,
  non-object `metadata`, or non-numeric `total_size`) as a usage error
  instead of an uncaught `AttributeError` or `TypeError`.
- Fine-tuning pipeline and transfer plans now resolve a fractional
  `logging_steps` against the planned duration for every logging strategy.
  With `logging_strategy: no`, a longer plan previously failed its own
  execution-policy validation instead of producing a plan.
- Retried experiment attempts no longer carry the previous attempt's
  `finished_at_utc` in `state.json` while running or after an interruption.
- The Hugging Face native runner now names the first unsupported option in
  sorted order when a configuration sets several, instead of a
  hash-seed-dependent one.
- `fakegpu validate` records a case whose command cannot be started (missing
  executable or working directory) as failed and still writes the report,
  instead of aborting the run with a traceback.
- `fakegpu validate` reports unreadable manifests and unusable values (a null
  `timeout_seconds`, non-object `defaults.env`/`requires`/`expect`, a
  non-numeric `defaults.timeout_seconds`) as manifest errors (exit 2) before
  running any command; a non-numeric `expect.exit_code` or
  `expect.duration_seconds_max` is recorded as a case failure.
- `fakegpu preflight` rejects an invalid `--memory-safety-margin` with a usage
  error (exit 2) before running the workload, instead of failing afterwards
  without writing a report.
- `cuEventRecord` with an unknown or NULL event no longer advances the
  stream timeline, so the stream stays idle for `cuStreamQuery`.
- `cudaGetDriverEntryPointByVersion` reports `driverStatus` 0 (success) for
  resolved symbols and 1 (symbol not found) otherwise; the values were
  inverted.
- `cuGetProcAddress_v2` with a NULL `pfn` and a non-NULL `symbolStatus` now
  returns `CUDA_ERROR_INVALID_VALUE` with status 1 instead of crashing.
- `cuCtxPushCurrent` now also switches the Runtime's current device, as
  `cuCtxSetCurrent` does, so pinned host allocations and peer-access checks
  use the pushed context's device.
- `cudaMemPoolGetAttribute` writes 4 bytes for the int-valued reuse-policy
  attributes instead of overwriting 8 bytes of caller memory.
- `cuStreamAddCallback` follows `FAKEGPU_UNSUPPORTED_API` like
  `cudaStreamAddCallback`: it records the unregistered callback, warns once
  under `warn` and returns `CUDA_ERROR_NOT_SUPPORTED` under `error`.
- Hybrid `cuGetProcAddress` no longer returns the caller's stale `*pfn` when
  the real driver resolves a symbol to NULL and `dlsym` also misses it.
- Hybrid `cuMemAllocFromPoolAsync` applies a single allocation decision, so
  one over-budget request counts once in the report's OOM fallback count.
- NCCL passthrough no longer treats a real NCCL library with missing
  required symbols as loaded: every later call now fails with
  `ncclSystemError` instead of reaching the partially resolved library.
- `ncclCommSplit` now reports a coordinator timeout as `ncclSystemError` and
  duplicate split ranks or split timeout mismatches as `ncclInvalidUsage`,
  matching `ncclCommShrink`, instead of `ncclInternalError`. Ranks that
  disagree on explicit all-to-all splits also get `ncclInvalidUsage`.
- Complete `ncclSend`/`ncclRecv` pairs over the TCP coordinator. The receiver
  no longer declares a socket payload it never sends, so the coordinator stops
  waiting for it and the sender no longer times out.
- Release each finished coordinator connection thread instead of keeping
  every handler until shutdown, so coordinator memory no longer grows with
  the number of requests.
- Reject `nan` and `inf` fabric bandwidth, latency, and oversubscription
  values in cluster YAML files instead of writing them into the cluster report
  as invalid JSON.
- Apply `FAKEGPU_CLUSTER_CONFIG` to a directly started `fakegpu-coordinator`
  even when `FAKEGPU_DIST_MODE` and `FAKEGPU_COORDINATOR_ADDR` are unset. The
  cluster report previously named the config path but reported no nodes or
  link estimates; an unreadable or invalid config is now reported on stderr
  at startup and omitted from the report.
- Accept coordinator request headers up to 1 MiB, so large NCCL groups and
  all-to-all-v split lists over many ranks are no longer rejected with
  `request line is too long` once the header passes about 4 KiB.
- `nvmlDeviceGetHandleByPciBusId` now returns the device with the requested
  PCI bus id and `NVML_ERROR_NOT_FOUND` for an unknown id instead of always
  returning device 0.
- Builds without CPU simulation now count `cublasLtMatmul` calls and FLOPs
  in the report, like `cublasGemmEx` already did.
- With CPU simulation, `cublasSgemm_v2`, `cublasDgemm_v2` and `cublasGemmEx`
  accept `k = 0` with padded leading dimensions and scale C by beta instead
  of returning `CUBLAS_STATUS_INVALID_VALUE`.
- Without CPU simulation, the synthetic `cublasI[sd]amax_v2` and
  `cublasI[sd]amin_v2` results are 1-based like cuBLAS instead of possibly
  returning the invalid index 0.
- The report's `gemm_by_dtype` breakdown now includes
  `cublasGemmStridedBatchedEx`, `cublasGemmBatchedEx` and `cublasSgemmEx`
  calls, not only `cublasGemmEx` and `cublasLtMatmul`.
- Native `FAKEGPU_FAULT_EVENTS`, `FAKEGPU_MIG_LAYOUT` and
  `FAKEGPU_NVLINK_GROUPS` integers with a `-` sign (such as device `-0` or
  count `-18446744073709551615`) are rejected, as in the Python model,
  instead of being accepted as 0 or wrapped values.
- A native `FAKEGPU_FAULT_EVENTS` event with an empty field, such as
  `0::critical`, reports the Python model's `must use
  DEVICE:CODE:SEVERITY[:COUNT]` error instead of a code or severity error.
- The native `FAKEGPU_NVLINK_FABRIC` error for a group naming a GPU beyond
  the device count includes the group, as in the Python model.

## v1.6.0 - 2026-10-17

### Added

- Native macOS Python/Conda development support, installed-wheel CI, and a
  state-file-to-NVML bridge for the standard `nvitop` interface.
- A background simulated GPU machine with a control API, process events,
  idle-device monitoring, and an interactive Textual console with per-device
  editing and memory, utilization, power, temperature, and link charts.
- A catalog of 307 public-reference GPU configurations spanning Maxwell
  through Blackwell, including GeForce, data-center, and workstation cards.
- Per-precision throughput specifications, host platforms, GPU slots, PCIe
  and NVLink topology, and modeled transfer time and link throughput.
- Reference telemetry for every profile and an eight-device synthetic A100
  PyTorch example whose live allocations appear in `nvitop`. Sensor values
  are modeled; the example does not require physical A100 hardware.
- `fakegpu estimate-llm --exclude-decode-steps` (and the
  `include_decode_steps=False` argument on `estimate_decoder_inference`)
  to summarize decode FLOPs in closed form instead of embedding one
  record per generated token.
- `fakegpu calibrate` for phase-aware prediction/measurement error analysis,
  safety recommendations, and reusable exact-scope calibration bundles.
- `fakegpu plan-training` for DeepSpeed, Accelerate, FSDP, and FSDP2 config
  normalization with rank-local phase memory and CPU-offload estimates.
- `fakegpu replay-trace` for PyTorch Profiler, NCCL-style, and FakeGPU
  timelines with rank waits/overlap, pair/link totals, memory samples,
  operator profiles, fusion handling, and recovery events.
- `fakegpu simulate-topology` for rack/switch/multi-NIC/NVLink-domain graphs,
  deterministic ECMP, ring/tree/hierarchical algorithms, contention, and
  critical-link reports.
- `fakegpu analyze-kernel` for PTX/SASS/CUDA static instruction, register,
  shared-memory, FLOP, and profile occupancy analysis.
- Phase-normalized static and LLM memory timelines with virtual-SMI
  projections.
- CPU numeric coverage for FP16 GEMM plus real-valued `scal`, `gemv`, and
  `ger` cuBLAS paths.
- Call-based workspace coverage summaries, a public
  `require_workspace_coverage()` gate, validation thresholds, and cross-report
  coverage aggregation.
- A native unsupported-API policy with `allow`, one-time `warn`, and
  `error`/CUDA 801 behavior, plus per-device report events.
- Lower/expected/upper workspace intervals, explicit bounds for unmatched
  calls, and a strict complete-upper-bound gate.
- PyTorch-dispatch storage tracking for operator-created tensors, aliases,
  inaccessible outputs, and per-operation statistics in preflight and virtual
  SMI state.
- A versioned native API capability manifest with strict source and built
  export audits.
- `fakegpu analyze-repo` for entrypoint, dependency, compiled-acceleration,
  and distributed-configuration readiness analysis.
- `fakegpu estimate-roofline` for profile-derived scalar-compute and memory
  ceilings with explicit lower/expected/upper latency assumptions.
- MoE active-expert FLOPs, expert-parallel traffic, quantized safetensors
  storage, PEFT adapter storage, and optional profile roofline output in
  `fakegpu estimate-llm`.
- Draft-model speculative-decoding memory planning in `fakegpu plan-serving`,
  including simultaneous target/draft weights and KV caches, conservative
  lookahead slots, verification transients, acceptance assumptions, and
  homogeneous or mixed-request admission.
- Path-independent serving workload signatures for exact-scope calibration,
  with separate GPU-profile dimensions and rejection of speculative-config
  drift.
- `fakegpu calibrate observe-serving` for aggregating repeated prefill/decode
  peaks into signature- and profile-scoped observation reports with explicit
  sample-sufficiency status.
- `fakegpu calibrate collect-serving` for repeated external runner execution,
  a versioned peak-sample protocol, physical-CUDA metadata checks, environment
  consistency, a reported 2% profile-capacity tolerance, timeouts, and
  command-argument-safe observation reports.
- `fakegpu calibrate sample-transformers` for checkpoint-verified homogeneous
  prefill/decode measurements with dynamic KV caches and lazy optional
  framework imports.
- `fakegpu calibrate emit-serving-sample` and
  `build_cuda_serving_sample()` for adding validated CUDA, PyTorch, and
  framework identity to phase peaks produced by vLLM or custom runners.
- Runtime-aware vLLM serving budgets with automatic or explicit paged-KV
  reservation, whole-block capacity, initialization fit, and scheduler
  admission against the reserved pool.
- Automatic MLA recognition from decoder configuration, with compressed
  latent-plus-RoPE KV-cache storage and MLA-specific attention transient and
  matrix-operation formulas.
- Scriptable FakeGPU-SMI runtime queries for runtime policy, software,
  tracking, catalog, dispatch, and bounded publisher-health fields.

### Changed

- Removed the two hardware-measured GPU profiles from the public catalog.
  Configurations that depended on a removed profile must select a supported
  public-reference profile; these profiles are not calibration equivalents.
- Redacted remaining personal home-directory paths from public review
  records while retaining their original-digest provenance. Added release
  privacy checks, neutral source-archive ownership metadata, and explicit
  exclusions for private local notes.
- Expanded CI with a full Python regression job, Linux/macOS native builds,
  and installed-wheel checks for bundled profiles and native libraries.
- Updated runtime, preflight, report, memory-estimation, and distributed
  documentation to describe the implemented behavior and remaining limits.
- Expanded the public analysis API with repository, native-capability, and
  profile-roofline reports.
- Removed the PrivateUse1 backend prototype (`fakegpu/privateuse1/`) and its
  `fakegpu.init_privateuse1()` export; it depended on private
  `torch._C._acc` APIs that no supported PyTorch release provides.
- Aligned the command-line conventions across every `fakegpu` command:
  `plan-serving`, `estimate-llm`, and `bandwidth` accept `--json` without a
  path (the report goes to stdout) like the other report commands,
  `bandwidth --json` writes through the shared JSON writer, `coordinator`
  and `bandwidth` report unusable arguments with exit code 2 like every
  other command instead of 1, and `validate` and `workspace-profiles`
  prefix input errors with the command name.
- Shortened the English, Simplified Chinese, and Traditional Chinese READMEs
  to an overview, a feature list, and basic usage. Detailed guides, evidence
  tables, and the command reference moved to `docs/en/`, `docs/zh-CN/`, and
  `docs/zh-TW/`, and the documentation contract tests now check those pages.

### Fixed

- Read GemmEx alpha/beta in the scalar storage type selected by computeType,
  including FP16 and legacy real-valued compute enums, across ordinary and both
  batched entry points. Unknown compute types now obey the unsupported-API
  policy without reading scalars or modifying output.
- Route unimplemented cuBLASLt auxiliary/gradient epilogues and unknown epilogue
  combinations through the unsupported-API policy instead of partially executing
  them as ordinary activations. Preserve output buffers on these paths and test
  that all six implemented epilogues still compute under strict policy. Match
  cuBLASLt's documented tanh GELU approximation instead of using erf-based GELU.
- Avoid reading old GEMM output when beta is zero, so NaN/Inf in C cannot
  contaminate the result. Reject non-identity cuBLASLt TRANSC settings before
  output writes in every policy and build mode; only TRANSC=N is supported by
  the cuBLASLt contract. Add native numerical and output-preservation regressions.
- Allow the research validation manifest to rerun in the same report directory
  by reusing its fixture directories while rewriting model inputs and reports.
- Enforce the native unsupported-API policy across 78 cuBLAS arithmetic entry
  points, including synthetic outputs, GEMM/Lt fallback configurations, and
  skipped CPU computation. Strict mode rejects these paths before output writes;
  reports and the capability catalog now expose them. Add compiled-library
  policy tests for CPU-enabled and CPU-disabled builds.
- Read nested `text_config` decoder dimensions and current `dtype` metadata,
  and warn in LLM/serving JSON and terminal reports when vision/projector
  activations and FLOPs are unmodeled, including speculative draft models.
- Decoder inference can model last-token-only vocabulary projection with
  `estimate-llm --logits-to-keep 1` (Python: `logits_to_keep=1`), including
  FLOPs, logits memory, and traffic in cached and uncached generation.
  The default of zero retains the conservative all-token projection.
- Unified native runtime statistics across the CUDA Runtime, Driver, cuBLAS,
  and NVML compatibility libraries so one library cannot overwrite a report
  with an empty or partial DSO-local snapshot.
- Routed PyTorch's generic `torch.accelerator` device, stream, and
  synchronization APIs through FakeCUDA, including `is_capturing()` support
  required by optimizer steps in PyTorch 2.13 CPU runners.
- Kept CUDA-targeted static memory analysis driver-free on PyTorch 2.13 by
  separating the logical target device from the CPU FakeTensor execution
  device when no real CUDA runtime is available.
- Made CI independent of Ruff's changing zero-configuration defaults, added
  the PEFT dependency required by the full regression suite, and isolated
  microsecond-scale overhead measurements from process-global PyTorch state.

### Validation

- The release is checked with Python regressions, Linux/macOS native tests,
  declarative validation manifests, and installed-package monitoring probes.
- Privacy review covers tracked source, decompressed review archives,
  documentation screenshots, and distribution contents. Historical source
  digests still describe private originals, not the redacted public copies.

## v1.5.5 - 2026-07-23

Compared with `v1.5.4`.

### Added

- A CUDA-style caching allocator model with 512-byte blocks, small/medium/large segments, best-fit reuse, split/coalesce, cached empty segments, retry, OOM counters, `empty_cache()`, reserved-memory APIs, memory statistics, and allocator snapshots.
- Allocated/reserved/inactive-split reporting in preflight and virtual SMI, with simulated process memory based on allocator-reserved bytes plus optional same-stack runtime-overhead calibration.
- A JSON/YAML backend-workspace profile registry with exact GPU/software/dtype/shape matching, fixed/linear/tiled formulas, explicit lifetimes, priorities, catalog validation, and a `fakegpu workspace-profiles` command.
- Ten exact-stack cuBLAS/cuDNN matrix and convolution workspace profiles measured on the Ampere GPU and Blackwell GPU.
- A `fakegpu validate` command for declarative JSON/TOML/YAML test matrices, prerequisite handling, timeouts, output/file/JSON assertions, per-case logs, and unified Git/host-aware JSON and Markdown reports.
- Fixed allocator-trace and workspace-capture validators plus a maintained cross-profile smoke manifest.
- A GitHub Actions matrix for Python 3.10–3.12 CPU PyTorch checks and native CMake smoke/CPU-simulation builds.
- Virtual `nvidia-smi` process tables with host/profile/stage identity, simulated and tracked current/peak memory, bounded live refresh, dynamic state-directory discovery, and NDJSON sampling.
- A two-rank Hybrid FSDP Qwen SFT memory experiment that reports per-rank allocator phases, parameter/gradient/AdamW shards, and all-gather/reduce-scatter traffic.
- A FULL_SHARD memory projection that transforms single-GPU ATen storage liveness using per-unit padding, local optimizer storage, and collective workspaces.
- A two- or four-rank FSDP2 LoRA experiment and per-parameter DTensor projection for frozen BF16 base weights, FP32 adapters, gradient-production liveness, AdamW state, and overlapping collective buffers.
- Fake NCCL `int8`, `uint8`, `uint32`, and `uint64` payload support, including reduction execution and cluster-report schema coverage.
- A reusable DeepSpeed matrix for ZeRO 0–3, FP32/BF16, two/four ranks, gradient accumulation, optimizer numerics, and communication reports.
- A Qwen3.5 LoRA SFT DeepSpeed runner with per-rank phase memory, adapter-update consistency, gradient checkpointing, and JSON/Markdown summaries.
- A ZeRO checkpoint validator covering all-rank save, fresh-engine restore, optimizer/scheduler/client-state resume, uninterrupted-result comparison, and FP32 consolidation.
- A Hugging Face Trainer + DeepSpeed validator for self-contained tiny models and local Qwen LoRA weights.
- DeepSpeed CPU optimizer and ZeRO-3 parameter-offload axes with explicit state-device validation.
- Optional physical multi-host DeepSpeed ZeRO-2/3 cases with per-host package preflight and combined communication reports.
- A parameterized deterministic DataLoader replay matrix covering shuffled `DistributedSampler` epochs, worker counts, prefetch depths, batch sizes, replacement PIDs, and PyTorch/Python/NumPy RNG streams locally or across SSH hosts.

### Changed

- Package, native report, and C++ runtime versions are synchronized at `1.5.5`.

### Fixed

- Preserved conservative preflight headroom after introducing caching by using the larger of requested and reserved peaks while retaining both per-stage series.
- Calibrated Qwen virtual-SMI process memory against real reserved memory rather than requested tensor bytes, avoiding double-counting allocator cache.
- Linked POSIX shared-memory calls against `librt` on Linux, restoring the manylinux2014 coordinator wheel build.
- Included the active full-gradient buffer retained until FSDP reduce-scatter emits a local shard, reducing the Qwen graph-peak prediction error from 8.420% to at most 0.758%.
- Accepted FSDP2's byte-packed mixed-dtype all-gathers and separated activation-, communication-, backward-, and optimizer-dominated event floors instead of assuming that every buffer peaks together.
- Separated forward/loss liveness at the first explicit backward operator and modeled the retained FSDP2 nested-unit buffers at the backward activation peak.
- Accepted null buffers for zero-element NCCL collectives while retaining strict non-empty validation, restoring DeepSpeed 0.19.2 ZeRO-3 under PyTorch 2.12.
- Selected reentrant Qwen activation checkpointing for DeepSpeed ZeRO-3 while retaining non-reentrant mode as an explicit compatibility probe.
- Executed NCCL product, minimum, and maximum reductions used by DeepSpeed checkpoint metadata instead of accepting them as sum-only collectives.
- Registered only the legacy DeepSpeed checkpoint types identified by PyTorch's safe loader, preserving `weights_only` protection under DeepSpeed 0.15.3/PyTorch 2.8.
- Used short temporary Unix coordinator sockets in every DeepSpeed runner, avoiding platform path-length failures for descriptive report directories.
- Rejected physical ZeRO-3 launches when DeepSpeed versions differ, replacing an in-training collective-order failure with a package preflight diagnostic.
- Drained the remaining DataLoader epoch before persistent-worker shutdown, avoiding active-prefetch teardown failures on PyTorch 2.8 without changing the captured replay prefix.
- Emitted schema-valid JSON and Markdown cluster reports for zero-operation coordinator sessions, including configured zero-traffic node pairs.

### Validation

- Full local suite: 396 passed and 1 skipped.
- The 15-stage allocator trace matched real CUDA allocated and reserved bytes exactly on both Ampere GPU/PyTorch 2.12.1/CUDA 13.0 and Blackwell GPU/PyTorch 2.9.1/CUDA 12.8.
- Ten controlled workspace observations were captured across both stacks. The FP32 `N=2,C=64,H=W=128,K=64` convolution used 8,536,576 bytes on Ampere and 16,925,184 bytes on Blackwell; exact-stack matching prevents unsupported cross-architecture extrapolation.
- The declarative smoke manifest passed all six expanded cases on macOS, Ampere GPU WSL, and Blackwell GPU Linux.
- The five-scenario DataLoader replay matrix matched sample-order and PyTorch/Python/NumPy RNG digests across macOS PyTorch 2.9.1, Blackwell GPU Linux PyTorch 2.8.0, and Ampere GPU WSL2 PyTorch 2.12.1. Two physical repeats each started 52 fresh workers per host with no cross-run PID overlap.
- Qwen3.5-0.8B BF16 FULL_SHARD sequence-16/128 experiments passed on Blackwell GPU and Ampere GPU. Overall per-rank errors were 0.730% and 0.633%; both stacks reproduced allocator peaks, shard sizes, optimizer state, and 8,011,593,488 node-pair bytes exactly.
- Qwen3.5-0.8B FSDP2 LoRA sequence-16/64/128 experiments passed the 3% per-phase limit with two or four ranks on both GPUs. Every phase was within 1.974%; sequence-128 overall errors were 0.071%-0.161%.
- DeepSpeed ZeRO 0–3 FP32/BF16 numerics passed on Blackwell GPU with DeepSpeed 0.15.3 and Ampere GPU with DeepSpeed 0.19.2; four-rank BF16 ZeRO-3 also passed on the Blackwell GPU.
- Qwen3.5-0.8B BF16 LoRA ZeRO-2/3 passed on both GPUs. The maintained Blackwell GPU sequence-64, accumulation-2 case used 3.052 GiB per rank without checkpointing and 2.409 GiB with reentrant checkpointing; the complete node-pair reports recorded 15.196 GiB and 17.134 GiB respectively.
- ZeRO-3 FP32 and ZeRO-2/3 BF16 checkpoint save/restore/continue/consolidate checks passed on both GPUs and DeepSpeed versions.
- Tiny Hugging Face Trainer ZeRO-2/3 and Qwen3.5-0.8B LoRA Trainer ZeRO-3 passed on both GPU stacks; the Blackwell GPU sequence-64 run also passed accumulation-2 with reentrant checkpointing.
- FP32 ZeRO-2 optimizer offload and ZeRO-3 optimizer + parameter offload produced the exact analytical update on both GPUs, with the requested state verified on CPU.
- Physical Blackwell GPU ↔ Ampere GPU ZeRO-2 passed across DeepSpeed 0.15.3/0.19.2, producing identical parameters and reporting seven TCP collectives, 176 node-pair bytes, and a 32-byte per-operation peak.

## v1.5.4 - 2026-07-21

Compared with `v1.5.3`.

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

### Changed

- Package, native report, and C++ runtime versions are synchronized at `1.5.4`.

### Fixed

- Preserved non-default CUDA stream ordering at the Hybrid host-staging boundary by copying on the collective stream and synchronizing before coordinator access. This restores correct FSDP pre-divided gradients with PyTorch 2.12/CUDA 13 while retaining PyTorch 2.9/CUDA 12 behavior.
- Replaced PyTorch's equal-head fused-SDPA FLOP assumption in the Qwen verifier, allowing Qwen3 grouped-query attention (32 query heads and 8 KV heads) to be measured without assertion failures.
- Isolated the `Tensor.to("cuda")` redirect microbenchmark from random-tensor creation variance while preserving its 100-microsecond incremental-overhead limit.
- Corrected native NF4 first-level statistics to FP32, counted every materialized quantization buffer in static estimates, and preserved explicit FP32 buffers in meta-model construction.
- Added the known native-NF4 dequantization workspace to FakeCUDA compute peaks while retaining the unadjusted CPU tracker phases for inspection.

### Validation

- Full local suite: 271 passed and 1 skipped.
- The distributed resilience suite passed on macOS, the Blackwell GPU Linux host, and the Ampere GPU WSL host; its 256-operation case retained the newest 64 timeline entries and counted 192 discarded entries.
- The automated heterogeneous two-host run used the same `3e6c8b2` commit on both hosts. Hybrid DDP produced gradient `[1.5, 3.0]` and parameters `[0.85, -0.30]`; both mismatched ranks persisted async error 5; the missing-peer case timed out in 0.755 seconds. The cluster report recorded six successful collectives, 192 bytes between the node pair, a 64-byte per-operation peak, and one expected timeout.
- Single-host Hybrid DDP/FSDP passed on both the Blackwell GPU (compute capability 12.0, PyTorch 2.9.1/CUDA 12.8) and Ampere GPU (compute capability 8.6, PyTorch 2.12.1/CUDA 13.0). FSDP produced local shard gradients `[1.5]` and `[3.0]`, reconstructed `[0.85, -0.30]`, and restored the same values from a full state dict.
- The complete heterogeneous two-host run used commit `fb852c7` and passed basic DDP, all three DDP option cases, FSDP, collective mismatch, and missing-peer timeout. Its report reconciled 34 completed collectives, 1,104 node-pair bytes, a 128-byte per-operation peak, and one expected timeout with no discarded timeline entries.
- The FSDP2 matrix passed 20 single-host combinations across the Blackwell GPU and Ampere GPU: two/four ranks, FP32/FP16/BF16 parameters, and FP16/BF16 parameter-dtype gradient reductions.
- Heterogeneous two-host FSDP2 passed FP32/FP16/BF16 parameter cases plus FP16/BF16 gradient reduction. The low-precision report retained eight operations, 160 node-pair bytes, and a 32-byte per-operation peak while identifying `float16`/`bfloat16` payloads and `sum`/`avg` reductions.
- Qwen3-8B BF16 SDPA validation on the Blackwell GPU matched 8,190,735,360 parameters and both generated token IDs. FakeCUDA load and inference-peak errors versus the real CUDA allocator were 0.012914% and 0.064877%; the checkpoint-only peak error was 0.067234%; calibrated virtual-SMI process memory differed from NVML by 0.063182%.
- Qwen3-8B observed matrix FLOPs matched exactly between CPU-backed FakeCUDA and real CUDA at 151,415,620,864; the shape estimate differed by 1,280 FLOPs (0.000001%).
- Five direct-scale and five nested-scale QLoRA cases using Qwen3.5-0.8B/2B all passed with static peak errors at or below 1.732%. Nested scales reduced storage by 0.370-0.372 bit per quantized weight; the complete 0.8B FakeCUDA step differed from real CUDA by 0.165% overall.
- Ampere GPU and Blackwell GPU reproduced both maintained 0.8B nested-scale cases byte-for-byte, including real/static peaks, persistent storage, and random-batch fingerprints.

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
- Hybrid DDP numerical validation passed on an Blackwell GPU with PyTorch 2.9.1/CUDA 12.8 and an Ampere GPU with PyTorch 2.12.1/CUDA 13.0.
- A heterogeneous physical two-host Hybrid DDP check passed between those software stacks: fake NCCL carried broadcast, all-reduce, and all-gather over TCP, producing the expected averaged gradient and identical updated parameters without coordinator timeouts.
- A physical two-host Tailscale test completed correct 1 MiB and 16 MiB all-reduces with zero coordinator timeouts; the 16 MiB × 5 case measured about 0.261 Gbit/s algorithmic throughput and 0.521 Gbit/s bidirectional socket payload per rank.

## v1.5.2 - 2026-07-20

Compared with `v1.5.1`.

### Added

- PyTorch compatibility coverage for Hugging Face Trainer, PEFT, TRL, LLaMA-Factory, Lightning Fabric, LitGPT, torchtune, Accelerate, and maintained FSDP paths.
- AI workload preflight CLI with stage-aware execution, fit/OOM classification, JSON schema validation, Markdown reports, allocation categories, shared-storage handling, logical-device attribution, and optional allocation stack traces.
- Exact `ampere` and `blackwell` profiles, including compute capability 12.0 and measured CUDA device attributes for the current 72 GB calibration server, plus the lightweight `test-512m` profile.
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
- Hybrid forwards the CUDA 12.x Driver symbols resolved directly by PyTorch/cuBLAS, restoring real GEMM, backward, and optimizer execution on Blackwell GPU.
- FakeCudaTensor now reports its logical CUDA index through `get_device()`, enabling `torch.utils.checkpoint`.
- Memory tracking now skips storage-less functional-transform tensors such as `BatchedTensorImpl`, enabling current Transformers Qwen2 masking code that uses `vmap`.
- NVML exports the NVLink queries used by PyTorch OOM diagnostics, so Hybrid clamp raises `torch.cuda.OutOfMemoryError` instead of an internal missing-symbol assertion.
- Pytest collection isolates the standalone native Driver script so its `LD_PRELOAD` state cannot leak into Transformers/torchvision subprocess tests; runtime benchmarks now use adjacent, batched samples.

### Validation

- Full local suite: 182 passed and 1 skipped.
- Static-memory validation: 13 workloads across Ampere GPU and Blackwell GPU, for 26 GPU observations.
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
