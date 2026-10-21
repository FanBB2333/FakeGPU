# Validation and reliability

[Documentation index](README.md) · [简体中文](../zh-CN/validation.md) ·
[繁體中文](../zh-TW/validation.md)

This page collects the real-GPU evidence behind FakeGPU's memory estimates and
explains how each workload is labeled. To produce new evidence yourself, see
[Calibrating against a real GPU](calibration.md).

Public experiment records name the GPU architecture only. Supported model
names in the profile catalog describe compatibility, not the hardware used for
testing.

<a id="memory-estimation-evidence"></a>

## Real-GPU memory-estimation evidence

> [!NOTE]
> In the recorded validation envelopes below, the stack-calibrated static
> estimator stayed within **0.08%** on 26 controlled GPU observations and
> within **1.921%** across ten Qwen full/LoRA SFT cases.

Error is `|predicted - observed| / observed × 100%`. Agreement is
`100% - error`.

| Validated envelope | Real-GPU reference | Evidence | Absolute percentage error | Agreement |
|---|---|---:|---:|---:|
| [Controlled ATen MLP and Transformer grid with backend-resident calibration][validation-static] | Ampere GPU and Blackwell GPU; PyTorch/CUDA 2.12/13.0 and 2.9/12.8 | 13 workloads, 26 observations | **0.08% maximum** | **≥99.92%** |
| [Qwen3-8B BF16 SDPA inference][validation-inference] | Blackwell GPU; PyTorch 2.9.1/CUDA 12.8 | Model load and inference peak | 0.0129% load; **0.0672% peak** | 99.9871%; **99.9328%** |
| [Qwen 0.8B/2B full and LoRA SFT][validation-sft] | Blackwell GPU; PyTorch 2.8/CUDA 12.8 | 10 training cases | **0.102%–1.921%** | **98.079%–99.898%** |
| [Qwen 0.8B/2B native NF4 QLoRA][validation-qlora] | Blackwell GPU; PyTorch 2.8/CUDA 12.8 | 10 quantized training cases | **0.628%–1.732%** | **98.268%–99.372%** |

How to read these numbers:

- Qwen rows use `torch.cuda.max_memory_allocated()`. CUDA context and
  reserved-but-unused allocator memory are excluded.
- The ATen row adds one backend-resident measurement from the exact stack. Do
  not reuse it on another stack.
- Ranges are the smallest and largest case errors, not averages.
  Underestimation is the failure mode that matters for OOM risk.
- A `99.x%` agreement is not spare capacity. Still apply a safety margin.

These are fixed-workload results, not a general accuracy claim. Other models,
shapes, backends, kernels, allocators, software versions, or GPUs need their
own calibration. The links point to an immutable snapshot with full
configurations and byte counts. CI checks this table against a
[machine-readable evidence summary](../../tests/data/memory_validation_evidence.json).

<a id="llm-reliability"></a>

## Research workload reliability

Reliability is reported per workload and environment signature:

- `GPU-validated` applies only to the recorded model revision, shapes, dtype,
  attention backend, allocator, software stack, and GPU.
- `CPU-validated` means maintained execution or analysis passed without a GPU.
- `Modeled` means analytical coverage without matching real-GPU evidence.
- `Planned` is not yet a supported accuracy claim.

Serving plans carry a path-independent `workload_signature`, so evidence cannot
silently carry over to a different serving configuration. Collect repeated
real-CUDA samples with `calibrate observe-serving` or
`calibrate collect-serving`.

### Verification snapshot (2026-08-05)

Verified with every `scripts/test.sh` suite and both validation manifests on
macOS 26.5 arm64, Python 3.11.9, and PyTorch 2.9.1 CPU. The Transformers
sampler also ran on a Blackwell GPU (PyTorch 2.11.0+cu128,
Transformers 5.14.1).

| Validation layer | Maintained check | Result |
|---|---|---|
| Python runtime, estimators, CLIs, schemas, and documentation contracts | Complete `pytest` suite | **223 passed** |
| Declarative validation matrices | 7 smoke executions plus 39 research architecture, cache, generic/vLLM serving, training, calibration, and diffusion executions | **46 passed** |
| Native interception | Build, library boundaries, exports, preload, memory types, coordinator, and unsupported-API policy | **Passed** |
| FakeGPU-SMI diagnostics | Bounded state, topology/NVLink/MIG views, NVML peer/MIG queries, health fields, and event reporting | **Passed** |
| Monitoring exporter | Prometheus/JSON snapshots, bounded history/cardinality, malformed-state degradation, and HTTP endpoints | **Passed** |
| Native capability inventory (updated 2026-10-21) | 5 groups, 107 explicit APIs, 94 policy-enforced APIs | **Passed** |
| GPU profile catalog | 307 profiles across 16 compute capabilities (catalog updated 2026-10-16) | **Passed** |
| CPU numerical simulation | GEMM, cuBLASLt, batched GEMM, BLAS1/2, and FP16: 8 maintained test groups | **Passed** |
| CUDA-enabled PyTorch native matmul | Requires a CUDA build of PyTorch | **Not run on this CPU-only host** |
| Transformers real-CUDA serving sampler | Qwen/Qwen3-0.6B, BF16 SDPA, 2 × 128-token prefill + 8 generated tokens, 3 isolated samples per phase | **Passed; `ready_for_comparison`** |

GitHub CI runs the Python suite on Python 3.10–3.12, and the native smoke and
CPU simulation suites on Linux and macOS. The sampler row checks repeated
execution and CUDA identity. It says nothing about accuracy.

### Maintained research workload matrix

| Workload class | Covered variations | Evidence | Status |
|---|---|---|---|
| Offline decoder inference | Qwen3-8B, BF16, SDPA, model load, prefill, and decode peak | Blackwell GPU prediction-versus-observation data | `GPU-validated` |
| Full and adapter SFT | Qwen 0.8B/2B full fine-tuning and LoRA | Ten Blackwell GPU training cases | `GPU-validated` |
| Quantized adapter SFT | Qwen 0.8B/2B native NF4 QLoRA | Ten Blackwell GPU training cases | `GPU-validated` |
| General decoder analysis | Dense and MoE metadata; MHA, GQA, MQA, and compressed-latent MLA; adapters, quantized checkpoints, eager/SDPA attention, KV cache, and expert-parallel traffic | Formula, fixture, CLI regression tests, and the four-case architecture matrix | `CPU-validated` + `Modeled` |
| Distributed training plans | DeepSpeed, Accelerate, FSDP/FSDP2, sharding, checkpointing, and CPU/NVMe offload | Configuration, byte-accounting, topology, and trace tests | `CPU-validated` + `Modeled` |
| KV-cache allocation | Dynamic growth, static reservation, 2/4/8-bit quantized storage, paged block rounding, and sliding-window limits | Formula, API, `--kv-cache-strategy` CLI, and `tests/data/research_validation.yaml` matrix tests | `CPU-validated` + `Modeled` |
| Online serving capacity | Homogeneous and mixed-length request sets under continuous batching, ordered admission, chunked prefill, grouped prefix caching, paged KV blocks, generic or vLLM runtime budgets, profile or explicit KV capacity, and admission headroom | Checkpoint headers, architecture and block-allocation formulas, `plan-serving --runtime vllm`, `--vllm-kv-cache-memory-bytes`, `--vllm-non-kv-cache-memory-bytes`, unit tests, and generic/vLLM chat/RAG cases in `tests/data/research_validation.yaml` | `CPU-validated` + `Modeled` |
| Diffusion image generation | Stable Diffusion v1.5, SDXL Base, and PixArt-Sigma; local UNets, cross-attention DiTs, and SD3/Flux-style joint-attention transformers; CFG, offload, attention/VAE slicing, and VAE tiling | Fixed-revision and local component headers, architecture configs, phase formulas, CLI/unit tests, and a three-case local-architecture matrix in `tests/data/research_validation.yaml` | `CPU-validated` + `Modeled` |
| speculative decoding | Independent autoregressive draft-model weights and KV cache, target lookahead KV, batched verification transients, optional fixed acceptance assumptions, and homogeneous/mixed request admission | Checkpoint headers, architecture formulas, `plan-serving --draft-model-dir`, `--speculative-tokens`, `--speculative-acceptance-rate`, and unit tests; no maintained real-GPU observation yet | `CPU-validated` + `Modeled` |
| Multi-GPU LLM execution | TP, PP, CP, EP, MoE imbalance, and combined FSDP/ZeRO execution | Analytical topology and coordinator coverage only | `Modeled` |
| Diffusion training | Cached-feature UNet full/LoRA, FP32 AdamW/EMA storage, BF16 autocast and native activation/allocator envelopes | Twelve SD/SDXL native processes, independent holdouts and reload checks | `GPU-validated` for documented cases |

<a id="research-scenario-effects"></a>

### Architecture-aware estimate comparisons

These examples come from the checked-in formulas and are verified by
documentation contract tests. They are formula results, not real-GPU
observations, and they do not replace a calibration on the same configuration.
GiB values are binary units.

| Research scenario | Controlled comparison | Modeled effect | Validation |
|---|---|---:|---|
| LLM inference KV cache | 32-layer GQA, 8 KV heads, head dim 128, batch 1, BF16; context 4K → 32K | **0.50 → 4.00 GiB** (8× cache) | Exact byte formula |
| Attention cache architecture | 32 layers, 8 active sequences, 4K BF16 paged cache; standard head dim 128; MHA 32 KV heads / GQA 8 / MQA 1 / MLA latent 512 + RoPE 64 | **16.00 / 4.00 / 0.50 / 1.125 GiB** | Architecture-specific cache-layout formulas and four-case manifest |
| Online serving prefix cache | Same decoder, 8 active sequences, 4K prompt + 256 generated tokens, paged BF16 KV; independent caches → one shared 1K prefix | **4.25 → 3.38 GiB** (−20.6%) | Shared/private paged-block formula |
| Mixed chat/completion serving | Same decoder; prompts 4K/8K/2K and generation 256/512/1024; independent KV → two chat requests sharing a 1K system prefix | **1.97 → 1.84 GiB** decode KV (−6.3%) | Heterogeneous request-set formula and chat/RAG matrix |
| Speculative target calls | 5 draft tokens per verification step and an independent 70% per-token acceptance assumption | **1.00 → 2.94** expected output tokens per target step; not a throughput claim | Geometric acceptance formula plus dual-model memory tests |
| LLM training | 8B BF16 parameters, AdamW, 4 GPUs, checkpointed activations; replicated → full shard | **134.11 → 33.99 GiB** per rank (−74.7%) | Training-plan phase model |
| Stable Diffusion v1.5 generation | 512², batch 1, FP16, CFG; all weights resident → model offload | **2.61 → 1.72 GiB** (−34.3%) | Component + phase model |
| SDXL batch generation | 1024², batch 4, FP16, CFG; regular VAE decode → VAE slicing | **26.64 → 11.64 GiB** (−56.3%) | Sequential VAE batch-shape model |
| SDXL high-resolution generation | 2048², batch 1, FP16, CFG; full-frame VAE → 512² VAE tiles | **26.64 → 7.92 GiB** (−70.3%) | Sequential VAE tile-shape model |
| PixArt-Sigma DiT generation | 1024², batch 1, FP16, CFG, model offload; eager → SDPA denoise phase | **2.32 → 1.28 GiB** (−44.7%) | Patch-token, cross-attention, and phase model |

- Diffusion profiles use component parameter counts from fixed revisions of
  [Stable Diffusion v1.5][diffusion-sd15], [SDXL Base 1.0][diffusion-sdxl],
  and [PixArt-Sigma XL 2][diffusion-pixart]. Runtime context, fragmentation,
  private workspaces, and transfer overlap are excluded.
- Reports stay `Modeled`, with `accuracy.status` set to `uncalibrated`, until a
  matching observation exists. `estimate-diffusion --model-dir` exercises
  local architecture inspection.
- Cache formulas follow
  [Transformers cache strategies](https://huggingface.co/docs/transformers/kv_cache).
  Speculative mode follows the draft/target flow of the
  [original speculative-decoding paper](https://arxiv.org/abs/2211.17192) and
  [Transformers assisted decoding](https://huggingface.co/docs/transformers/assisted_decoding).
  Serving terms follow [vLLM serving](https://docs.vllm.ai/en/stable/).
- Arrival distributions, cache eviction, preemption, adaptive draft length,
  KV rollback workspaces, throughput, and latency are out of scope. Binary
  CUDA extensions and arbitrary kernels need a passthrough or hybrid real-GPU
  observation.

### Publishing new validation rows

A new or refreshed public validation row should record:

- at least five isolated observations and the maximum observed peak
- predicted and observed bytes for every reported phase
- maximum underprediction as the main OOM-risk metric, at most 5% for a
  `GPU-validated` row
- median, 95th-percentile, and maximum absolute percentage error
- prediction-interval coverage and false-safe decisions (predicted fit, real
  OOM)
- the model revision, command, shapes, dtype, backend, allocator settings,
  GPU, driver, CUDA, PyTorch, and framework versions

Use `calibrate verify` to apply these limits before publishing. Rows that miss
the target stay `Modeled` or experimental.

[validation-static]: https://github.com/FanBB2333/FakeGPU/blob/cc7f221356b582e3e06618f226d9e05bd7da64ef/docs/ai-researcher-preflight.md#4-static-aten-storage-liveness-validation
[validation-inference]: https://github.com/FanBB2333/FakeGPU/blob/cc7f221356b582e3e06618f226d9e05bd7da64ef/docs/llm-inference-estimation.md#maintained-qwen3-8b-result
[validation-sft]: https://github.com/FanBB2333/FakeGPU/blob/cc7f221356b582e3e06618f226d9e05bd7da64ef/docs/llm-sft-memory-estimation.md#blackwell-gpu-matrix
[validation-qlora]: https://github.com/FanBB2333/FakeGPU/blob/cc7f221356b582e3e06618f226d9e05bd7da64ef/docs/llm-sft-memory-estimation.md#native-nf4-qlora-matrices
[diffusion-sd15]: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/451f4fe16113bff5a5d2269ed5ad43b0592e9a14
[diffusion-sdxl]: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/462165984030d82259a11f4367a4eed129e94a7b
[diffusion-pixart]: https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS/tree/e102b3591cc82e97071b8b4cb90d834d0c487207
