<a id="readme-top"></a>

<div align="center">

# FakeGPU

**Validate CUDA-facing applications, estimate GPU memory, and simulate
distributed GPU workflows without a production GPU cluster.**

[![Tests][test-shield]][test-url]
[![Release][release-shield]][release-url]
[![Python][python-shield]][python-url]
[![License][license-shield]][license-url]

[English](README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md)

[Report a bug](https://github.com/FanBB2333/FakeGPU/issues/new?labels=bug) ·
[Request a feature](https://github.com/FanBB2333/FakeGPU/issues/new?labels=enhancement)

</div>

> [!IMPORTANT]
> FakeGPU is a development, compatibility-testing, and capacity-planning tool.
> It does not provide numerical or performance parity for arbitrary CUDA
> kernels. Passthrough, hybrid, and calibration workflows still require a real
> CUDA stack.

## Table of contents

1. [About the project](#about-the-project)
   - [What FakeGPU answers](#what-fakegpu-answers)
   - [Typical use cases](#use-cases)
   - [Real-GPU memory-estimation evidence](#memory-estimation-evidence)
   - [Research workload reliability](#llm-reliability)
   - [How it works](#how-it-works)
   - [Built with](#built-with)
2. [Getting started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [Verify the installation](#verify-the-installation)
3. [Usage](#usage)
   - [Run PyTorch code on FakeCUDA](#run-pytorch-code-on-fakecuda)
   - [Inspect FakeGPU devices and processes](#inspect-fakegpu-devices-and-processes)
   - [Export monitoring metrics](#export-monitoring-metrics)
   - [Intercept native CUDA libraries](#intercept-native-cuda-libraries)
   - [Check memory before a run](#check-memory-before-a-run)
   - [Plan online LLM serving capacity](#plan-online-llm-serving-capacity)
   - [Analyze a repository or model](#analyze-a-repository-or-model)
4. [Command reference](#command-reference)
5. [GPU profiles](#gpu-profiles)
6. [Development](#development)
   - [Build](#build)
   - [Test](#test)
   - [Reusable scripts](#reusable-scripts)
7. [Project structure](#project-structure)
8. [Limitations](#limitations)
9. [Roadmap](#roadmap)
10. [Contributing](#contributing)
11. [License](#license)
12. [Acknowledgments](#acknowledgments)

## About the project

FakeGPU simulates CUDA-facing environments for development, CI, compatibility
checks, and capacity planning. It exposes configurable NVIDIA-like devices to
applications while maintained operations run on CPU, records simulated memory
and communication, and provides static estimators for workloads that should
not be loaded at all.

Physical GPUs are optional for the simulation and analysis paths. A compatible
physical CUDA stack is required only for passthrough, hybrid, and calibration
runs.

In native hybrid mode, `CUDA_VISIBLE_DEVICES` selects physical GPUs by index or
UUID. When this variable is present, FakeGPU keeps physical NVML discovery so
frameworks can resolve that selection correctly; without it, hybrid mode keeps
the modeled NVML inventory. The CUDA runtime and cuBLAS remain native in both
cases. Rebuilding a child environment with `fakegpu.env()` removes stale
FakeGPU preloads from the selected library directory when changing modes.

### What FakeGPU answers

| Question | Recommended path | Physical GPU |
|---|---|---:|
| Does PyTorch code follow the expected CUDA-facing control flow? | Python FakeCUDA runtime | No |
| Can an unmodified process load and call CUDA-family shared libraries? | Native interception | No |
| Will a selected GPU profile fit a workload? | Preflight or static memory estimator | No |
| How much checkpoint, KV-cache, adapter, or MoE memory should an LLM use? | LLM estimator | No |
| Which homogeneous or mixed-length online LLM requests fit under a memory budget? | Serving planner | No |
| Does a target plus draft model for speculative decoding fit in memory? | Serving planner | No |
| How do resolution, batch size, CFG, VAE tiling, and offload affect diffusion generation memory? | Diffusion estimator | No |
| Where are the GPU-only entry points and dependencies in a repository? | Repository analyzer | No |
| What does a distributed training configuration imply for rank-local memory? | Training planner | No |
| Where do compute, communication, wait, and memory overlap in a trace? | Trace replay | No |
| How does an estimate compare with an actual CUDA run? | Passthrough or hybrid calibration | Yes |

<a id="use-cases"></a>

### Typical use cases

| When this is useful | What FakeGPU provides | Start with |
|---|---|---|
| Choosing a GPU before renting capacity or starting a long job | Profile-aware checkpoint, KV-cache, activation, optimizer, and workspace estimates | `estimate-llm`, `preflight` |
| Sizing chat, RAG, completion, summarization, or draft-assisted traffic before deployment | Per-request prompt/generation lengths, continuous-batch admission, chunked-prefill transients, shared-prefix KV groups, speculative target/draft memory, and explicit memory headroom | `plan-serving`, `validate` |
| Developing CUDA-oriented PyTorch code on a laptop or CPU-only CI runner | CUDA-visible control-flow checks while maintained tensor operations execute on CPU | `fakegpu.init(...)`, `demo`, `validate` |
| Comparing full fine-tuning, LoRA, QLoRA, checkpointing, offload, or sharding plans | Phase-aware and rank-local memory estimates before allocating a cluster | `plan-training`, Python memory estimator |
| Comparing UNet and diffusion-transformer generation shapes and memory optimizations | Architecture-specific text encoder, denoising, and VAE-decode estimates from fixed profiles or a local pipeline | `estimate-diffusion`, `validate` |
| Reviewing an unfamiliar GPU repository or native extension | GPU entry-point, dependency, kernel, and unsupported-API inventory | `analyze-repo`, `analyze-kernel`, `capabilities` |
| Designing or debugging a distributed workflow | Collective routing, link contention, rank waits, memory timelines, and TCP payload validation | `simulate-topology`, `replay-trace`, `bandwidth` |
| Observing simulated devices and processes in CI or a local lab | Bounded Prometheus metrics, exporter health, and short in-memory history | `nvidia-smi`, `metrics` |
| Turning a small real-GPU trial into evidence for repeated runs | Prediction-versus-observation reports and signature-scoped calibration data | `calibrate`, `preflight --memory-calibration` |

<a id="memory-estimation-evidence"></a>

### Real-GPU memory-estimation evidence

> [!NOTE]
> In the recorded validation envelopes below, the stack-calibrated static
> estimator stayed within **0.08%** on 26 controlled GPU observations and
> within **1.921%** across ten Qwen full/LoRA SFT cases.

Absolute percentage error is
`|predicted - observed| / observed × 100%`. “Agreement” below is its
complement, `100% - error`, shown only as a more intuitive reading of the same
measurement.

| Validated envelope | Real-GPU reference | Evidence | Absolute percentage error | Agreement |
|---|---|---:|---:|---:|
| [Controlled ATen MLP and Transformer grid with backend-resident calibration][validation-static] | Ampere GPU and Blackwell GPU; PyTorch/CUDA 2.12/13.0 and 2.9/12.8 | 13 workloads, 26 observations | **0.08% maximum** | **≥99.92%** |
| [Qwen3-8B BF16 SDPA inference][validation-inference] | Blackwell GPU; PyTorch 2.9.1/CUDA 12.8 | Model load and inference peak | 0.0129% load; **0.0672% peak** | 99.9871%; **99.9328%** |
| [Qwen 0.8B/2B full and LoRA SFT][validation-sft] | Blackwell GPU; PyTorch 2.8/CUDA 12.8 | 10 training cases | **0.102%–1.921%** | **98.079%–99.898%** |
| [Qwen 0.8B/2B native NF4 QLoRA][validation-qlora] | Blackwell GPU; PyTorch 2.8/CUDA 12.8 | 10 quantized training cases | **0.628%–1.732%** | **98.268%–99.372%** |

How to read these numbers:

- The Qwen rows use `torch.cuda.max_memory_allocated()` as the reference. CUDA
  context memory and reserved-but-unused allocator memory are excluded.
- The controlled ATen row adds one backend-resident measurement from the exact
  GPU and software stack; that value must not be reused on another stack.
- Ranges show the minimum and maximum case error, not an average. Maximum
  underestimation is the important failure mode when evaluating OOM risk.
- A `99.x%` agreement value is not spare capacity. Capacity decisions should
  still apply a workload-specific safety margin or factor.

These are fixed-workload measurements, not a universal accuracy claim.
Different models, shapes, attention backends, quantization kernels, allocators,
PyTorch/CUDA versions, or GPUs require a matching calibration. The links above
point to the immutable validation snapshot containing the full configurations
and measured byte counts. A
[machine-readable evidence summary](https://github.com/FanBB2333/FakeGPU/blob/main/tests/data/memory_validation_evidence.json)
is checked against the README in CI.

On a CUDA host, regenerate the maintained controlled comparison with:

```bash
python3 scripts/validation/static_memory_validation.py \
  --output build/static-memory-validation.json \
  --markdown build/static-memory-validation.md \
  --max-underestimate-percent 5
```

On a CPU-only host, add `--static-only`; this checks the estimation path but
does not produce a real-GPU accuracy measurement. For a `plan-serving` report,
normalize repeated real-GPU phase peaks before comparison:

```bash
python3 -m fakegpu calibrate observe-serving \
  build/speculative-serving-plan.json \
  --prefill-peak-bytes 24800000000,24900000000,25000000000 \
  --prefill-peak-bytes 24950000000,25050000000 \
  --decode-peak-bytes 25200000000,25300000000,25250000000 \
  --decode-peak-bytes 25350000000,25400000000 \
  --source torch.cuda.max_memory_reserved \
  --strict \
  --json build/serving-observation.json
```

The command retains every sample and uses the maximum for each phase. Five
samples per phase are required by default; `--strict` returns status 1 when
that minimum is not met. `ready_for_comparison` records sample sufficiency,
not a `GPU-validated` claim. The plan must contain `--target-profile` so the
observation keeps the same GPU identity.

Automate the repeated executions with a framework-specific runner:

```bash
python3 -m fakegpu calibrate collect-serving \
  build/speculative-serving-plan.json \
  --repetitions 5 \
  --timeout-seconds 900 \
  --strict \
  --json build/serving-observation.json \
  -- python3 benchmark_serving.py
```

For a homogeneous, non-speculative Transformers workload, FakeGPU includes a
runner, so no separate benchmark file is needed. The plan must describe the
same execution semantics and use a dynamic KV cache:

```bash
python3 -m fakegpu plan-serving \
  --model-dir /mnt/z/models/Qwen/Qwen3-0.6B \
  --active-sequences 4 \
  --max-batch-size 8 \
  --prompt-tokens 2048 \
  --generated-tokens 128 \
  --dtype bfloat16 \
  --attention-implementation sdpa \
  --kv-cache-strategy dynamic \
  --target-profile a100 \
  --json build/transformers-serving-plan.json

python3 -m fakegpu calibrate collect-serving \
  build/transformers-serving-plan.json \
  --repetitions 5 \
  --timeout-seconds 900 \
  --strict \
  --json build/transformers-serving-observation.json \
  -- python3 -m fakegpu calibrate sample-transformers
```

The model path keeps the Hugging Face `owner/repository` hierarchy. On hosts
without `/mnt/z`, change only the models root and retain
`Qwen/Qwen3-0.6B`.

`sample-transformers` rechecks the local model configuration and Safetensors
headers against the plan, then measures full prefill and token-at-a-time decode
with the returned KV cache. Its default metric is
`torch.cuda.max_memory_reserved`; use `--metric allocated` for tensor-only
allocator peaks. It deliberately rejects mixed request sets, speculative
decoding, paged or quantized caches, shared-prefix or chunked-prefill plans,
quantized checkpoints, and adapters until those execution paths have matching
measurement implementations. PyTorch and Transformers remain optional and
must already be installed in the CUDA benchmark environment.

vLLM or custom instrumentation that already has trustworthy phase peaks can
use the protocol builder without recreating CUDA metadata:

```bash
python3 -m fakegpu calibrate emit-serving-sample \
  build/vllm-serving-plan.json \
  --prefill-peak-bytes 24900000000 \
  --decode-peak-bytes 25300000000 \
  --metric nvml.process_family_peak_bytes \
  --framework vllm \
  --framework-version 0.10.2
```

The same adapter is available as `fakegpu.build_cuda_serving_sample(...)` for
use inside a Python benchmark. vLLM reserves engine and KV-cache memory from
its configured GPU budget, while its cache-usage metric is a fraction of
blocks rather than a process-memory byte peak. FakeGPU therefore requires the
vLLM benchmark to provide real phase bytes instead of labeling cache usage as
prefill or decode memory.

The runner is invoked in a new process for every repetition and must emit one
`fakegpu.serving_peak_sample.v1` JSON object. It may print the object as its
last line or prefix a one-line object with `FAKEGPU_SERVING_SAMPLE=` when the
framework also writes logs to stdout:

```json
{
  "schema_version": "fakegpu.serving_peak_sample.v1",
  "workload_signature": "<FAKEGPU_SERVING_WORKLOAD_SIGNATURE>",
  "run_index": 1,
  "metric": "torch.cuda.max_memory_reserved",
  "phases": {
    "prefill": {"peak_bytes": 24900000000},
    "decode": {"peak_bytes": 25300000000}
  },
  "environment": {
    "backend": "cuda",
    "simulated": false,
    "gpu_name": "NVIDIA A100-SXM4-80GB",
    "gpu_uuid": "GPU-...",
    "compute_capability": [8, 0],
    "total_memory_bytes": 85899345920,
    "software": {
      "framework": "transformers",
      "framework_version": "...",
      "cuda_version": "...",
      "torch_version": "..."
    }
  }
}
```

FakeGPU supplies the plan path, workload signature, target profile, compute
capability, run index, and run count through `FAKEGPU_SERVING_*` environment
variables. The same protocol can wrap Transformers, vLLM, or a custom server
without adding them as FakeGPU dependencies. The built-in Transformers runner
and vLLM/custom protocol builder populate this contract automatically.
Collection fails if the runner
reports simulated CUDA, a target capacity outside a 2% profile tolerance, a
different compute capability, inconsistent software metadata, a timeout, or a
nonzero exit. The report retains the actual/profile capacity difference. Raw
command arguments are not stored; the report retains only the executable name
and a command fingerprint.
`simulated: false` is still runner-supplied metadata rather than independent
hardware attestation, so retained benchmark logs and calibration gates remain
necessary for a `GPU-validated` claim.

To compare compatible prediction and observation reports for any workload:

```bash
python3 -m fakegpu calibrate compare \
  build/prediction.json \
  build/observation.json \
  --json build/calibration-comparison.json

python3 -m fakegpu calibrate verify \
  build/calibration-comparison.json \
  --max-underestimate-percent 5 \
  --max-absolute-percentage-error-percent 5 \
  --min-interval-coverage-percent 90 \
  --capacity-bytes 25769803776 \
  --json build/calibration-verification.json
```

The comparison reports per-phase signed and absolute error, interval coverage,
and a recommended memory safety margin and factor. `calibrate verify` exits
with status 1 when a configured gate fails. It checks maximum underestimation,
median/p95/maximum absolute percentage error, prediction-interval coverage,
false-safe fit decisions at the supplied capacity, and workload-dimension
consistency. Apply results only to the same workload signature, shapes, dtype,
software stack, and GPU profile.

<a id="llm-reliability"></a>

### Research workload reliability report

FakeGPU reports reliability per workload and environment signature. A
`GPU-validated` result applies only to the recorded model revision, shapes,
dtype, attention backend, allocator, software stack, and GPU. `CPU-validated`
means that maintained execution or analysis behavior passed without a physical
GPU. `Modeled` identifies analytical coverage without matching real-GPU
evidence, while `Planned` is not yet a supported accuracy claim.
Serving plans expose a path-independent `workload_signature` so calibration
evidence cannot silently cross serving configurations.
Use `calibrate observe-serving` to retain repeated phase samples before
comparison and reliability gates.
Use `calibrate collect-serving` with the built-in `sample-transformers`
runner, or with vLLM/custom measurements wrapped by
`build_cuda_serving_sample`, when repeated real-CUDA evidence is needed.

#### Verification snapshot (2026-08-05)

This repository state was verified on 2026-08-05 with every suite from
`scripts/test.sh` and both declarative validation manifests on macOS 26.5
arm64, Python 3.11.9, and PyTorch 2.9.1 CPU. The Transformers sampling adapter
was also exercised on an Blackwell GPU with PyTorch 2.11.0+cu128 and
Transformers 5.14.1:

| Validation layer | Maintained check | Result |
|---|---|---|
| Python runtime, estimators, CLIs, schemas, and README contracts | Complete `pytest` suite | **223 passed** |
| Declarative validation matrices | 7 smoke executions plus 39 research architecture, cache, generic/vLLM serving, training, calibration, and diffusion executions | **46 passed** |
| Native interception | Build, library boundaries, exports, preload, memory types, coordinator, and unsupported-API policy | **Passed** |
| FakeGPU-SMI diagnostics | Bounded state, topology/NVLink/MIG views, NVML peer/MIG queries, health fields, and event reporting | **Passed** |
| Monitoring exporter | Prometheus/JSON snapshots, bounded history/cardinality, malformed-state degradation, and HTTP endpoints | **Passed** |
| Native capability inventory (updated 2026-09-05) | 5 groups, 104 explicit APIs, 102 policy-enforced APIs | **Passed** |
| GPU profile catalog | 82 profiles across 15 compute capabilities | **Passed** |
| CPU numerical simulation | GEMM, cuBLASLt, batched GEMM, BLAS1/2, and FP16: 8 maintained test groups | **Passed** |
| CUDA-enabled PyTorch native matmul | Requires a CUDA build of PyTorch | **Not run on this CPU-only host** |
| Transformers real-CUDA serving sampler | Qwen/Qwen3-0.6B, BF16 SDPA, 2 × 128-token prefill + 8 generated tokens, 3 isolated samples per phase | **Passed; `ready_for_comparison`** |

GitHub CI separately runs the Python suite on Python 3.10–3.12 and the native
smoke and CPU simulation suites on Linux and macOS. Published accuracy results
above come from their linked immutable validation snapshots. The new
Transformers sampler result validates repeated execution, CUDA identity, and
report collection; without a comparison gate, it is not an accuracy claim.

#### Maintained research workload matrix

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

#### Architecture-aware estimate comparisons

The previous “reproducible modeled effects” label only meant that fixed inputs
produce the same formula result; it did not mean the estimate matched a real
GPU observation. This section is now named “architecture-aware estimate
comparisons” to make that distinction explicit. The examples are generated by
checked-in formulas and verified by README contract tests. GiB values are
binary units, and the results do not replace same-configuration GPU calibration.

| Research scenario | Controlled comparison | Modeled effect | Validation |
|---|---|---:|---|
| LLM inference KV cache | 32-layer GQA, 8 KV heads, head dim 128, batch 1, BF16; context 4K → 32K | **0.50 → 4.00 GiB** (8× cache) | Exact byte formula |
| Attention cache architecture | 32 layers, 8 active sequences, 4K BF16 paged cache; standard head dim 128; MHA 32 KV heads / GQA 8 / MQA 1 / MLA latent 512 + RoPE 64 | **16.00 / 4.00 / 0.50 / 1.125 GiB** | Architecture-specific cache-layout formulas and four-case manifest |
| Online serving prefix cache | Same decoder, 8 active sequences, 4K prompt + 256 generated tokens, paged BF16 KV; independent caches → one shared 1K prefix | **4.25 → 3.38 GiB** (−20.6%) | Shared/private paged-block formula |
| Mixed chat/completion serving | Same decoder; prompts 4K/8K/2K and generation 256/512/1024; independent KV → two chat requests sharing a 1K system prefix | **1.97 → 1.84 GiB** decode KV (−6.3%) | Heterogeneous request-set formula and chat/RAG matrix |
| Speculative target calls | 5 draft tokens per verification step and an independent 70% per-token acceptance assumption | **1.00 → 2.94** expected output tokens per target step; not a throughput claim | Geometric acceptance formula plus dual-model memory tests |
| LLM training | 8B BF16 parameters, AdamW, 4 GPUs, checkpointed activations; replicated → full shard | **104.31 → 26.54 GiB** per rank (−74.6%) | Training-plan phase model |
| Stable Diffusion v1.5 generation | 512², batch 1, FP16, CFG; all weights resident → model offload | **2.61 → 1.72 GiB** (−34.3%) | Component + phase model |
| SDXL batch generation | 1024², batch 4, FP16, CFG; regular VAE decode → VAE slicing | **26.64 → 11.64 GiB** (−56.3%) | Sequential VAE batch-shape model |
| SDXL high-resolution generation | 2048², batch 1, FP16, CFG; full-frame VAE → 512² VAE tiles | **26.64 → 8.28 GiB** (−68.9%) | Sequential VAE tile-shape model |
| PixArt-Sigma DiT generation | 1024², batch 1, FP16, CFG, model offload; eager → SDPA denoise phase | **2.32 → 1.28 GiB** (−44.7%) | Patch-token, cross-attention, and phase model |

The diffusion profiles use component parameter counts from fixed revisions of
[Stable Diffusion v1.5][diffusion-sd15] and
[SDXL Base 1.0][diffusion-sdxl], and
[PixArt-Sigma XL 2][diffusion-pixart]. Runtime context, allocator
fragmentation, backend-private workspaces, and transfer overlap remain outside
these numbers.
Architecture-aware reports remain `Modeled` with
`accuracy.status` set to `uncalibrated` until a matching observation exists.
Local architecture inspection is exercised through
`estimate-diffusion --model-dir`.

The cache formulas follow the workload shapes exposed by
[Transformers cache strategies](https://huggingface.co/docs/transformers/kv_cache).
The serving planner models homogeneous shapes and ordered heterogeneous active
request sets. Its speculative mode follows the independent draft/target flow
described by the
[original speculative-decoding paper](https://arxiv.org/abs/2211.17192) and
[Transformers assisted decoding](https://huggingface.co/docs/transformers/assisted_decoding): a smaller
model proposes several tokens and the target verifies them in one forward
pass. Request arrival distributions, cache eviction, preemption, adaptive
draft length, KV rollback workspaces, and end-to-end throughput or latency
remain outside the supported estimate. The workload terminology follows
[vLLM serving](https://docs.vllm.ai/en/stable/). Binary CUDA extensions and
arbitrary kernels remain outside CPU FakeCUDA execution; those workloads
require analysis plus a passthrough or hybrid real-GPU observation.

New or refreshed public validation rows should record:

- at least five isolated observations and the maximum observed peak;
- predicted and observed bytes for every reported phase;
- maximum underprediction as the primary OOM-risk metric, with a publication
  target of at most 5% for a `GPU-validated` row;
- median, 95th-percentile, and maximum absolute percentage error;
- prediction-interval coverage and the count of false-safe decisions, where
  FakeGPU predicted a fit but the real workload reached OOM; and
- the model revision, command, shapes, dtype, backend, allocator settings,
  GPU, driver, CUDA, PyTorch, and framework versions.

Use `calibrate verify` to apply these limits to machine-readable comparison
reports before publishing a result.

Rows that miss the target remain `Modeled` or are marked experimental instead
of being presented as validated. Agreement percentages remain secondary to
maximum underprediction and false-safe OOM decisions.

### How it works

| Path | What the application sees | What actually runs |
|---|---|---|
| **Python FakeCUDA** | CUDA devices, CUDA-looking tensors, memory APIs, and common training flows | Maintained PyTorch operations execute on CPU through `FakeCudaTensor` |
| **Native interception** | `libcuda`, `libcudart`, `libcublas`, `libnvidia-ml`, and `libnccl` entry points | Selected operations use host memory or CPU math; unsupported behavior is classified and reported |
| **Analysis and reporting** | Memory, FLOP, roofline, topology, and communication reports | ATen graphs, safetensors metadata, runtime traces, calibration data, and coordinator events are analyzed |

### Built with

- [Python](https://www.python.org/) 3.10+ for the runtime, estimators, CLI, and
  reports
- C++17 and [CMake](https://cmake.org/) for native interception libraries and
  the coordinator
- [PyTorch](https://pytorch.org/) for CPU-backed FakeCUDA execution and ATen
  graph capture
- YAML and JSON schemas for GPU profiles, validation manifests, and reports

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting started

### Prerequisites

- Linux or macOS
- Python 3.10 or newer
- CMake 3.14 or newer
- A C++17 compiler
- PyTorch for the Python FakeCUDA runtime

On Debian or Ubuntu, install `build-essential`. On macOS, install the Xcode
Command Line Tools.

### Installation

Clone the repository:

```bash
git clone https://github.com/FanBB2333/FakeGPU.git
cd FakeGPU
```

Build the native libraries and install the package:

```bash
scripts/build.sh
FAKEGPU_BUILD_DIR="$PWD/build" python3 -m pip install .
```

For development directly from a checkout:

```bash
python3 -m pip install pytest PyYAML jsonschema ruff
export PYTHONPATH="$PWD"
```

### Verify the installation

```bash
python3 -m fakegpu doctor --list-profiles
python3 -m fakegpu demo --profile l4
```

`doctor` checks the profile catalog, native libraries, and PyTorch environment.
`demo` performs a small forward, backward, and optimizer step on CPU while the
program sees a CUDA device.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

### Run PyTorch code on FakeCUDA

Initialize FakeGPU before importing PyTorch:

```python
import fakegpu

fakegpu.init(runtime="fakecuda", profile="a100", device_count=2)

import torch

device = torch.device("cuda:0")
model = torch.nn.Linear(8, 4).to(device)
x = torch.randn(2, 8, device=device)
loss = model(x).square().mean()
loss.backward()

print(torch.cuda.device_count())      # 2
print(torch.cuda.get_device_name(0))  # NVIDIA A100
print(loss.item())
```

Maintained operations execute on CPU while device placement, memory limits,
training control flow, and error handling use the simulated CUDA surface.

### Inspect FakeGPU devices and processes

State publishing is opt-in. Set a state directory before the workload calls
`fakegpu.init(...)` or starts through the native launcher; `build/` is ignored
by Git:

```bash
export FAKEGPU_SMI_STATE_DIR=build/smi

# Python FakeCUDA workload.
python3 your_script.py

# Unmodified native CUDA/NVML workload.
python3 -m fakegpu --build-dir build ./your_native_workload
```

From another terminal, inspect the running workload:

```bash
# Compact device and process table; add "-l 1" to refresh every second.
python3 -m fakegpu nvidia-smi --state-dir build/smi

# Device inventory and detailed runtime/profile/allocator report.
python3 -m fakegpu nvidia-smi --state-dir build/smi -L
python3 -m fakegpu nvidia-smi --state-dir build/smi -q

# NVIDIA-style modeled topology and NVLink views.
python3 -m fakegpu nvidia-smi topo -m --state-dir build/smi
python3 -m fakegpu nvidia-smi nvlink -s --state-dir build/smi

# Modeled faults, compatibility warnings, publisher failures, and stale state.
python3 -m fakegpu nvidia-smi events --state-dir build/smi

# Modeled MIG GPU and compute instances.
python3 -m fakegpu nvidia-smi mig -lgi --state-dir build/smi
python3 -m fakegpu nvidia-smi mig -lci --state-dir build/smi

# Script-friendly GPU and process queries.
python3 -m fakegpu nvidia-smi --state-dir build/smi \
  --query-gpu=index,name,uuid,pci.bus_id,profile.id,compute_cap,memory.total,memory.used,memory.free,allocator.model,native.kernel_launches,native.gemm_calls,native.io_bytes \
  --format=csv
python3 -m fakegpu nvidia-smi --state-dir build/smi \
  --query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory,peak_gpu_memory,stage,status \
  --format=csv,noheader,nounits
python3 -m fakegpu nvidia-smi --state-dir build/smi \
  --query-runtime=pid,fakegpu.version,runtime.backend,runtime.mode,policy.oom,tracking.dispatch,catalog.profiles,catalog.native_apis,dispatch.calls,publisher.failed_writes,status \
  --format=json
```

The detailed report includes the FakeGPU version, runtime backend and policies,
Python/PyTorch/CUDA versions, state freshness, profile catalog and native API
coverage, synthetic device identity, compute properties, memory categories,
allocator activity, dispatch tracking, and per-process peaks. Use `-i` with an
index, UUID, PCI bus ID, or profile ID to select devices; use `--json` for the
complete normalized inventory. State schema v2 is emitted, while v1 files
remain readable.
`--query-runtime` exposes the same runtime, policy, software, tracking,
catalog, dispatch, and publisher-health details as script-friendly CSV or
JSON. Use `--help-query-runtime` to list every supported field.

Native interception additionally publishes allocation lifetime, transfer
volume, kernel launches, GEMM calls/FLOP, compatibility events, and unsupported
API counts. State is refreshed while the process is running and marked exited
when the process shuts down. `FAKEGPU_SMI_DETAIL_LIMIT` bounds retained detail
entries and `FAKEGPU_SMI_MAX_STATE_BYTES` limits each state file; `-q` reports
publisher write counts, failures, latency, and serialized size.

NVLink modeling is disabled by default. Set
`FAKEGPU_NVLINK_GROUPS="0,1;2,3"` before starting the workload to create
full-mesh peer relationships inside each semicolon-separated group. The
optional `FAKEGPU_NVLINK_BANDWIDTH_GBPS` value defaults to `900`. The same
model drives state files, `topo -m`, `nvlink -s`, the
`topology.*`/`nvlink.*` query fields, and native NVML link-state, capability,
remote-device-type, and remote-PCI queries. Invalid groups leave every link
inactive and publish a configuration error.

MIG modeling is disabled by default. Configure instances before starting the
workload with
`FAKEGPU_MIG_LAYOUT="DEVICE:PROFILE:MEMORY_MIB[:COUNT];..."`, for example
`FAKEGPU_MIG_LAYOUT="0:1g.10gb:10240:2;1:2g.20gb:20480"`. Each modeled GPU
instance owns one compute instance. The layout drives state files, `-L`, `-q`,
`mig -lgi`, `mig -lci`, the `mig.*` query fields, and native NVML MIG mode,
handle, UUID, parent, instance-ID, and memory-capacity queries. Layouts that
exceed parent memory, eight slices, or eight instances per device are rejected
without creating partial instances. Per-instance runtime allocation attribution
is not implemented, so state files mark used/free instance memory unobserved.

Fault injection is also disabled by default. Set entries using
`FAKEGPU_FAULT_EVENTS="DEVICE:CODE:SEVERITY[:COUNT];..."`, for example
`FAKEGPU_FAULT_EVENTS="0:XID_79:critical;1:NVLINK_CRC:error:3"`. Codes are
labels; supported severities are `info`, `warning`, `error`, and `critical`.
The `events` view combines configured fault events with unsupported native API
calls, publisher failures, stale state, and model-configuration errors.
`health.*` query fields expose each device's modeled status, maximum severity,
event counts, and the fact that hardware health is unobserved. Invalid input
activates no faults and is reported as a configuration error.

UUIDs and PCI bus IDs are stable simulated identifiers. Temperature, fan
speed, live power draw, and hardware GPU utilization remain `N/A` because the
CPU-backed runtime cannot observe them; profile power and clock values are
shown separately as static specifications. Topology labels and configured
bandwidth, fault codes, and health status are modeled inputs, not hardware
measurements or observed ECC/Xid data.

<a id="export-monitoring-metrics"></a>

### Export monitoring metrics

The normalized FakeGPU-SMI state can be exported without adding a monitoring
dependency. A one-shot command emits Prometheus text by default or a normalized
JSON snapshot:

```bash
python3 -m fakegpu metrics --state-dir build/smi
python3 -m fakegpu metrics --state-dir build/smi --json
```

For local collection, start the bounded in-memory exporter:

```bash
python3 -m fakegpu metrics --state-dir build/smi --serve \
  --host 127.0.0.1 --port 9400 --interval 1 \
  --history-size 300 --max-process-series 128

curl http://127.0.0.1:9400/metrics
curl http://127.0.0.1:9400/healthz
curl http://127.0.0.1:9400/api/v1/history
```

`/metrics` exposes the latest device, process, runtime, MIG, topology, health,
and publisher values; `/healthz` reports source and scrape status;
`/api/v1/history` returns normalized recent samples. Process series retain the
highest-memory processes first and are limited to 128 by default and 256 at
most; `--max-process-series 0` disables them. History keeps 300 samples by
default and at most 1,440. Both limits are enforced in memory, and no monitoring
history is written to disk or tracked by Git. Use Prometheus or another scraper
for durable retention.

The server listens only on `127.0.0.1` by default and has no authentication.
Place it behind an authenticated proxy or network policy before binding a
non-loopback address. Exported values preserve the same modeled-versus-observed
semantics as FakeGPU-SMI; they do not turn unavailable hardware telemetry into
measurements.

### Intercept native CUDA libraries

Build the native libraries, then let the module launcher prepare
`LD_PRELOAD` or `DYLD_INSERT_LIBRARIES` for an unmodified command:

```bash
python3 -m fakegpu --build-dir build --profile a100 \
  python3 your_script.py

python3 -m fakegpu --build-dir build --devices "a100:2,h100:2" \
  python3 your_script.py

python3 -m fakegpu --build-dir build \
  --mode simulate \
  --unsupported-api error \
  python3 your_script.py
```

Unsupported native calls can be recorded, warned about, or returned as
`cudaErrorNotSupported` or `CUDA_ERROR_NOT_SUPPORTED`.

### Check memory before a run

Run a command to a target stage and write reports under an ignored build
directory:

```bash
python3 -m fakegpu preflight \
  --runtime fakecuda \
  --profile a100 \
  --stage forward \
  --report-dir build/preflight \
  --strict \
  -- python3 train.py
```

Preflight tracks the memory visible on the executed path and classifies whether
the selected profile fits the workload.

<a id="plan-online-llm-serving-capacity"></a>

### Plan online LLM serving capacity

Estimate a homogeneous active-request pool without loading checkpoint tensors:

```bash
python3 -m fakegpu plan-serving \
  --model-dir /models/example \
  --active-sequences 16 \
  --max-batch-size 64 \
  --prompt-tokens 4096 \
  --generated-tokens 256 \
  --prefill-chunk-tokens 512 \
  --shared-prefix-tokens 1024 \
  --kv-cache-strategy paged \
  --kv-cache-block-tokens 16 \
  --target-profile a100 \
  --memory-utilization 0.9 \
  --json build/serving-plan.json
```

The report separates model weights, prefill and decode transients, shared and
private KV segments, runtime/scheduler overhead, and usable device headroom.
`--device-memory-gib` can replace `--target-profile` for a custom capacity.
Prefix sharing is supported for dynamic and paged caches; paged shared and
private segments are rounded independently. Capacity search never exceeds
`--max-batch-size`.

#### Match vLLM runtime reservation

vLLM preallocates a paged KV pool instead of growing only with the active
requests. Select its memory policy when the deployment target is vLLM:

```bash
python3 -m fakegpu plan-serving \
  --model-dir /mnt/z/models/Qwen/Qwen3-0.6B \
  --active-sequences 16 \
  --max-batch-size 64 \
  --prompt-tokens 4096 \
  --generated-tokens 256 \
  --runtime vllm \
  --kv-cache-strategy paged \
  --kv-cache-block-tokens 16 \
  --target-profile "$TARGET_GPU_PROFILE" \
  --memory-utilization 0.92 \
  --json build/vllm-serving-plan.json
```

Automatic sizing follows the vLLM V1 policy: requested model-executor memory
minus profiled non-KV memory, rounded down to whole cache blocks. The report
keeps logical request KV demand separate from the reserved pool and exposes
`num_gpu_blocks`, cache token capacity, block-rounding tail, initialization
fit, and concurrency at `max_model_len`. If a matching vLLM startup profile is
available, pass its weights + peak activations + non-torch + CUDA-graph total
with `--vllm-non-kv-cache-memory-bytes`; this replaces FakeGPU's modeled
non-KV value. `--vllm-kv-cache-memory-bytes` provides an exact per-GPU cache
override and, like vLLM, ignores `--memory-utilization` for cache sizing.

When `--memory-utilization` is omitted, vLLM mode uses `0.92`, matching the
current [vLLM CacheConfig reference](https://docs.vllm.ai/en/latest/api/vllm/config/cache/).
Specify the value explicitly for a version-locked report. The budget formula
tracks vLLM's documented
[GPU-worker profiling path](https://docs.vllm.ai/en/latest/api/vllm/v1/worker/gpu_worker/),
but initial free memory, other processes, tensor/pipeline parallelism, hybrid
cache groups, allocator fragmentation, and version-specific kernels still
need same-stack observation. The research manifest checks both automatic and
explicit-cache modes. Speculative decoding currently remains available only
with the generic runtime model; vLLM-specific speculative schedulers and
cache managers are not represented yet.

Add an independent autoregressive draft checkpoint to estimate speculative
decoding memory:

```bash
python3 -m fakegpu plan-serving \
  --model-dir /models/target \
  --draft-model-dir /models/draft \
  --active-sequences 8 \
  --max-batch-size 32 \
  --prompt-tokens 4096 \
  --generated-tokens 256 \
  --speculative-tokens 5 \
  --speculative-acceptance-rate 0.7 \
  --kv-cache-strategy paged \
  --target-profile a100 \
  --json build/speculative-serving-plan.json
```

The report keeps target and draft weights plus both KV caches resident, adds
conservative lookahead slots, and compares target verification with draft
proposal transients. The acceptance assumption reports expected output tokens
per target call; it does not change the memory peak or claim an end-to-end
speedup. Equal vocabulary sizes are required, while tokenizer identity remains
unverified because checkpoint headers cannot prove token-ID equivalence.

Every homogeneous or mixed-request serving report also emits a path-independent
`workload_signature`. It covers target/draft architecture and weight storage,
request shapes, KV/prefill settings, and speculative configuration while
excluding absolute model paths and device capacity. Put the same field in a
measurement report before using `calibrate verify`; the calibration gate
compares the signature and checks `target.profile` as a separate dimension, so
evidence from a different proposal length or GPU profile is rejected.

For chat, RAG, completion, or summarization traffic with different lengths,
provide an ordered request manifest:

```json
{
  "schema_version": "fakegpu.serving_requests.v1",
  "requests": [
    {
      "id": "chat-a",
      "prompt_tokens": 4096,
      "generated_tokens": 256,
      "prefix_group": "system-prompt",
      "shared_prefix_tokens": 1024
    },
    {
      "id": "chat-b",
      "prompt_tokens": 8192,
      "generated_tokens": 512,
      "prefix_group": "system-prompt",
      "shared_prefix_tokens": 1024
    },
    {
      "id": "completion",
      "prompt_tokens": 2048,
      "generated_tokens": 1024
    }
  ]
}
```

```bash
python3 -m fakegpu plan-serving \
  --model-dir /models/example \
  --requests serving-requests.json \
  --max-batch-size 64 \
  --prefill-chunk-tokens 512 \
  --prefill-concurrency 2 \
  --kv-cache-strategy paged \
  --target-profile a100 \
  --memory-utilization 0.9 \
  --json build/mixed-serving-plan.json
```

Manifest mode calculates every request independently, stores each named
prefix group once, and admits the longest fitting prefix of the file without
reordering it. The report includes admitted and rejected request IDs,
per-request transients, component-wise concurrent-prefill bounds, grouped KV
segments, and the requested capacity headroom. Group members must use the same
positive `shared_prefix_tokens` value; shared prefixes require dynamic or paged
KV storage. Named groups are treated as resident cache hits; cache population,
lookup probability, and eviction behavior are not predicted.

Arrival timing, cache eviction, preemption/reordering, tensor parallelism,
adaptive speculative scheduling, KV rollback workspaces, throughput, and
latency are listed as unmodeled.
Until a matching online-serving observation is supplied, `validation_status`
remains `Modeled` and `accuracy.status` remains `uncalibrated`.

### Analyze a repository or model

```bash
# Find GPU entry points, dependencies, native sources, and compatibility risks.
python3 -m fakegpu analyze-repo .

# Estimate checkpoint, KV-cache, transient, adapter, and MoE memory.
python3 -m fakegpu estimate-llm \
  --model-dir /models/example \
  --batch-size 1 \
  --prompt-tokens 128 \
  --generated-tokens 32 \
  --dtype bfloat16 \
  --kv-cache-strategy paged \
  --kv-cache-block-tokens 16 \
  --target-profile a100 \
  --json build/llm-estimate.json

# Match a generation runtime that projects only the last token to logits.
python3 -m fakegpu estimate-llm \
  --model-dir /models/example --prompt-tokens 128 --generated-tokens 32 \
  --logits-to-keep 1 --json build/last-token-estimate.json
# The default, --logits-to-keep 0, projects all tokens. Use the runtime's
# actual setting: this changes LM-head FLOPs and memory, not layer token counts.

# Compare diffusion generation phases and memory optimizations.
python3 -m fakegpu estimate-diffusion --list-profiles
python3 -m fakegpu estimate-diffusion \
  --model-profile stable-diffusion-xl-base-1.0 \
  --height 1024 --width 1024 --batch-size 4 \
  --attention-backend sdpa --vae-slicing \
  --offload model --target-profile a100 \
  --json build/diffusion-estimate.json

# Inspect a local Diffusers pipeline without loading tensor payloads.
python3 -m fakegpu estimate-diffusion \
  --model-dir /models/pixart-or-flux \
  --height 1024 --width 1024 --text-tokens 300 \
  --dtype bfloat16 --attention-backend sdpa \
  --offload model --target-profile a100 \
  --json build/local-diffusion-estimate.json

# Audit source and built native exports against the capability manifest.
python3 -m fakegpu capabilities \
  --source-root . \
  --build-dir build \
  --strict
```

The LLM estimator reads safetensors headers without materializing checkpoint
weights. Choose `dynamic`, `static`, `quantized`, or `paged` with
`--kv-cache-strategy`; the JSON report separates logical storage,
quantization savings, static reservation, paged-block overhead, and optional
sliding-window limits. Quantized cache accounting retains 128 recent tokens
at the compute dtype by default; change it with
`--kv-cache-residual-tokens`.
MHA, GQA, and MQA use separate key/value storage. When the model config
contains `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, and
`v_head_dim`, the estimator identifies MLA and uses its compressed KV latent
plus decoupled RoPE component for cache bytes. MLA backend workspaces and
projection absorption remain runtime-specific and require calibration.

The diffusion estimator separates text encoding, repeated denoising, and VAE
decode phases. With `--model-dir`, it reads only `model_index.json`, component
`config.json` files, and selected safetensors headers. It does not import
remote custom code or read tensor payloads. Checkpoint storage and runtime
weight bytes at the requested dtype are reported separately. The declared local
Euler/PNDM scheduler configuration determines denoiser evaluation counts;
unsupported or unspecified schedulers report `denoiser_evaluations: null`.
Requested inference steps remain separate from actual scheduler evaluations.
The built-in `pixart-sigma-xl-2-1024-ms` profile provides a fixed-revision
transformer example alongside the two Stable Diffusion UNet profiles.

UNets, cross-attention patch transformers (DiT/PixArt), and SD3/Flux-style
joint-attention transformers use distinct activation formulas. Image-token
counts include the VAE scale, patch size, and Flux latent packing; CFG doubles
the denoiser batch only for architectures that use positive and negative
branches. Use `--weight-variant fp16|bf16|fp32` when a local directory contains
multiple checkpoint families.

`--offload model`, `--attention-slicing`, `--vae-slicing`, and `--vae-tiling`
expose common Diffusers memory trade-offs. Until a matching real-GPU comparison
is supplied, the report remains `Modeled`, `accuracy.status` is
`uncalibrated`, and no unsupported error percentage or prediction interval is
emitted.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Command reference

### Fine-tuning time and VRAM

`estimate-finetune` reads local model metadata and Hugging Face Trainer/PEFT,
LLaMA-Factory, or Axolotl configurations without importing those frameworks or
loading model weights. It estimates full/LoRA/4-bit QLoRA training on one GPU,
including optimizer updates, gradient accumulation, trainable/frozen storage,
activations, workspace, and the total **training-loop** time for a sample count
and epochs. `max_steps` takes precedence over epochs. Samples mean processed
training sequences after filtering, splitting, and packing; sequence length
includes template tokens and padding. An equal number of raw documents is not
an equal amount of training work. Pass `--max-steps -1` to clear a configured
step limit and estimate the requested epochs.

```bash
python -m fakegpu estimate-finetune \
  --config examples/finetuning/huggingface-lora.yaml \
  --model /path/to/Qwen3.5-0.8B --text-only \
  --profile "$TARGET_GPU_PROFILE" --num-samples 10000 --epochs 3 \
  --sequence-length 128 --json estimate.json
```

Estimation config examples are provided in
[`examples/finetuning/`](examples/finetuning/) for all three frameworks.
Configuration adaptation is covered by CPU tests; it does not imply that
every framework/runtime combination has been benchmarked. Distributed/offloaded execution,
unsupported optimizer variants, and unresolved adapter targets produce explicit
errors. `plan-training` remains the distributed memory-planning entry point.

Qwen3.5's linear/full attention layers are counted separately. Chunked DeltaNet
backward workspaces, optional fast kernels, and MoE dispatch remain partial
analytical coverage; an uncalibrated report cannot certify OOM safety.
`--text-only` explicitly selects the text backbone of a multimodal checkpoint.

`profile-finetune-data` scans every row of local JSONL files and records a content
fingerprint, token lengths, truncation, padding, and processed sequence counts.
Raw `text` or `messages` rows use a local tokenizer; their counts describe the
selected preprocessing scenario. Generic `--packing concatenate` is a token
budget scenario and does not reproduce a framework's packing algorithm. For
counts from an actual training split, export its final `input_ids` rows and use
`--processed`; this preserves row boundaries and rejects rows exceeding the cap.
Optional `attention_mask` and `labels` retain padding and supervision accounting.

```bash
python -m fakegpu profile-finetune-data \
  --data /path/to/processed-train.jsonl --format input_ids --processed \
  --sequence-length 128 --json dataset.json
python -m fakegpu estimate-finetune \
  --config examples/finetuning/huggingface-lora.yaml \
  --model /path/to/Qwen3.5-0.8B --text-only \
  --profile "$TARGET_GPU_PROFILE" --dataset-report dataset.json \
  --epochs 3 --max-steps -1 --json estimate.json
```

`--dataset-report` binds counts and the sequence cap together. It replaces an
estimated config sample count, rejects conflicting explicit overrides, and
requires the report cap to match the training config. A `--source-framework`
label records provenance only; it does not attest that the framework ran.
Packed training requires final processed sequence counts, rather than a raw-row
or generic concatenation scenario.

`run-finetune` invokes the installed Hugging Face Trainer/PEFT, LLaMA-Factory or Axolotl native pipeline on
one physical CUDA GPU. It supports full fine-tuning, LoRA, and bitsandbytes 4-bit
QLoRA with a local unquantized text checkpoint and unpacked data. Use a complete native config, including dataset
and output paths; relative paths resolve from the config directory. The report
records native preprocessing, the effective training dataset, actual batches,
optimizer updates, allocator peaks, evaluation/checkpoint events, and final
saving. Warmup is excluded from steady step samples but remains in total wall
time. This command performs the training requested by the native config.

Each `observation.updates[].batches` entry records a consumed microbatch's shape,
one-based epoch position, token slots, nonpadding tokens, and supervised labels
before and after the usual causal shift. `observation.batch_profile` reconciles
these counts and groups timings by the complete ordered accumulation window;
mixed shapes and partial windows keep separate groups. A group needs two steady
updates before `sufficient` is true; this is sample availability, not an accuracy
certificate. Missing or nonbinary attention masks leave token counts unknown.
Position-ID resets do not identify original packed examples. Collection overhead
is reported separately and excluded from update samples, while total wall time
includes it. These synchronized observations have a separate calibration identity.
`validate-finetune-windows` compares timing predictions against a separately
recorded dynamic run. It requires at least two independent calibration reports
from the same model, data, software and native execution settings. Each complete,
ordered window must have two retained observations in every calibration run,
after excluding global warmup and each window's first occurrence. Process means
receive equal weight; uncovered windows are rejected. New reports include a run
UUID, PID and UTC start time to catch accidental duplicate sampling.

The comparison predicts only the holdout's recorded steady windows. Its timings
are used for error measurement, never calibration. This is historical evidence
validation: it neither loads current weights nor attests the current environment.
It does not infer future shuffle order, warmup, total wall time or dynamic VRAM;
window statistics are not complete input equivalence. Periodic evaluation/save
schedules require the explicit event-context mode below. Native packing and MoE
window calibration remain unsupported.
```bash
python -m fakegpu validate-finetune-windows \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --holdout-run holdout.json --json window-validation.json
```


`observation.phase_events` records the actual evaluation and save call tree,
including training-start evaluation, periodic checkpoints, and final framework
calls. Only root intervals enter phase totals. The ledger separates raw wall time
from event metadata collection and records actual evaluation batch/content
fingerprints and files present after saves. A file inventory is not a measurement
of disk writes or durability. `preceding_event_ids` links an update to events
that occurred after the previous update.

`validate-finetune-events` compares evaluation and save subtotals against an
independent recorded run. At least two calibration runs are required; initial,
first, repeated and final events remain separate, and repeated groups need two
samples per calibration run. Matching workloads use equally weighted process
means; missing coverage is rejected. The first version requires unpacked dense
text models, loss-only evaluation without custom metrics, and local
model/adapter or resumable checkpoint saves.
It does not predict future event counts or total pipeline time: saves can alter
the allocator/cache state of subsequent training windows.

```bash
python -m fakegpu validate-finetune-events \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --holdout-run holdout.json --json event-validation.json
```

`validate-finetune-windows --event-context` also matches the recorded history
before each training window. It retains the latest evaluation and checkpoint
boundaries separately, including ordered calls, their workloads and phases, and
the number of optimizer updates since each boundary. An evaluation cannot erase
the context of an earlier save. Reports need the event ledger and matching actual
hardware; event support has the same limits as `validate-finetune-events`.

Global warmup and each window/context combination's first occurrence are excluded.
Every retained combination still needs two samples per calibration run. This
validates recurring windows following periodic events; it does not predict the
first save's recovery, establish that older history has no effect, or predict
complete training time. Evaluation/save durations remain separate subtotals.

```bash
python -m fakegpu validate-finetune-windows --event-context \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --holdout-run holdout.json --json event-window-validation.json
```

Add `--epoch-context` to match the recorded position of every microbatch within
its epoch as well. This distinguishes otherwise identical windows that include
epoch-start data loading or worker startup. It uses the complete ordered,
one-based microstep positions, without binding the absolute epoch number.
Epoch metadata is required; warmup, first-combination exclusion and independent
sample requirements still apply. Use this flag on its own for runs without
periodic events, or combine it with `--event-context`:

```bash
python -m fakegpu validate-finetune-windows --event-context --epoch-context \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --holdout-run holdout.json --json epoch-window-validation.json
```

`plan-finetune-pipeline` projects future epochs or optimizer updates from a
calibration run's complete observed epoch, before the target training runs.
`estimate-finetune-pipeline` combines at least two independent calibrations into
a complete native execution budget: setup, warmup and first occurrences, ordered
training windows, evaluation, checkpoints, Trainer gaps, and finalization.
New run reports provide a non-overlapping timeline and separate raw wall time
from measured instrumentation. The span ends after the native call and final
CUDA synchronization; Python process startup, report construction, and CLI
output are outside it. Time ranges combine the extrema of per-process component means; they are not
confidence intervals. Every component must have independent coverage; missing costs fail
explicitly.

The future schedule adapter supports unpacked, sequential text datasets with
retained tail batches: Hugging Face Trainer (Transformers 5.6.0) and LLaMA-Factory 0.9.5 /
Transformers 5.6.0 require `disable_shuffling: true`; Axolotl 0.18.0 /
Transformers 5.14.1 requires `curriculum_sampling: true`. It verifies the actual
observed batch sequence and eval/save schedule, including fractional epochs
and terminal events. The dataset, model, hardware and execution settings must
match calibration; changing the number of dataset rows requires a new matching
calibration. Logging is disabled or performed every update; checkpoint rotation
and best-model reload require additional calibration support.

```bash
python -m fakegpu plan-finetune-pipeline \
  --run-report calibration-1.json --epochs 30 --json future-plan.json
python -m fakegpu estimate-finetune-pipeline \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --plan future-plan.json --memory-budget-gib 24 --reserve-gib 1 --json budget.json
python -m fakegpu validate-finetune-pipeline \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --holdout-run holdout.json --json validation.json
```

For default random sampling and changed training datasets, use the explicit
empirical transfer workflow. It reads the native prepared dataloader, plans
CPU sampler/collator outputs up to the requested horizon, and verifies consumed
row indices and input tensor hashes during calibration. Supported stacks are
HF/Factory with Accelerate 1.11.0 and Axolotl with Accelerate 1.13.0, at the
framework versions above. Random sampling uses the actual native data seed plus
epoch; Axolotl 0.18.0 does not forward its YAML `data_seed` to Trainer.

```bash
python -m fakegpu calibrate-finetune \
  --config target-training.json --framework huggingface \
  --profile "$TARGET_GPU_PROFILE" --python /path/to/framework/python \
  --cuda-device 0 --workspace ./calibration-target \
  --calibration-steps 24 --warmup-steps 2 --epochs 3 \
  --timeout-seconds 600 --json calibration.json
python -m fakegpu budget-finetune-workspace \
  --workspace ./calibration-target --epochs 2 --json revised-budget.json
```

The workflow runs exactly two fresh native processes with explicit step/time
limits and new empty kernel-cache directories by default. It preserves every
attempt, seals reports and predictions with SHA256, and supports `--resume`
after interruption. Successful evidence is checked before reuse; it never
silently expands a calibration or reruns it because its error is large. CPU-only
rebudgeting needs no GPU or installed training framework and cannot exceed the
saved sampling horizon. Keep the workspace at its original absolute path and
outside input directories recorded in full, such as the model or a local dataset
directory. A nested workspace is rejected before runtime probing or training.
Only finite local datasets are supported.
Workspaces use POSIX process and lock semantics.

Add repeated `--reuse-calibration-run PATH` arguments to reuse compatible
training evidence from another dataset. Two fresh target-dataset calibrations
still supply startup and finalization costs; these costs are never multiplied
by a sample-count ratio. Training groups preserve ordered tensor shapes,
accumulation tails, epoch boundaries, warmup, first shape use and event context.
They **approximate** costs across different token/mask/supervision distributions;
full descriptor distributions remain visible in the report. Every group needs
two independent processes, and repeated groups need two observations in each.
Missing coverage returns exit status 1 and a list of gaps without a total budget.
Even a shorter duration can require an unobserved terminal transition.
Memory is a separate empirical envelope of complete calibration-process peaks,
not a guarantee for the combined long run or a smaller GPU.

`run-finetune --planning-epochs N` (or `--planning-max-steps N`) also records the
CPU horizon directly. `plan-finetune-transfer`, `estimate-finetune-transfer`
and `validate-finetune-transfer` expose planning, budgeting and independent
holdout comparison without the workspace manager. Existing strict pipeline
commands keep their original matching rules. Streaming, custom samplers
and automatic batch-size changes remain unsupported.

For length grouping, set `train_sampling_strategy: group_by_length` in HF or
LLaMA-Factory, or `group_by_length: true` in Axolotl, and explicitly add
`--length-grouping-seed 123` to `calibrate-finetune` or `run-finetune`.
This option uses the native Trainer grouping algorithm with an independent CPU
generator seeded by `123 + epoch`; it changes the sampler's random stream.
Native global-RNG length grouping cannot provide a reproducible future plan.
A planning horizon is required. With one finite evaluation dataset, evaluation
uses a separate generator with the same explicit seed for every invocation,
including persistent-worker loaders; it does not advance the training sampler.
The evaluation grouping policy and length digest are bound to calibration
identity. Custom evaluation samplers and prepared loaders are rejected.
Sampler plan v2 retains the complete ordered grouping lengths and
grouping batch size in calibration identity. Different length sequences need
their own evidence. Coverage still requires two independent runs and sufficient
repeat observations; a short calibration may be insufficient for many shapes.

The [packing semantics audit](tests/native/finetune_packing_semantics.py) changes
one segment and checks the next segment's logits and gradients in tiny real
Qwen3.5 models, with a dense-attention control. The tested ordinary HF/Factory
hybrid paths leak across a block mask and reset positions; the tested Axolotl
patch has a Transformers 5.14.1 decoder-interface incompatibility. These are
kernel diagnostics of the ordinary framework paths. See
[`refactor.md`](refactor.md) for the tested versions and retained results.

The opt-in [segment reference audit](tests/native/finetune_segment_reference.py)
checks a project-owned Qwen3.5 interface that executes each linear-attention
segment with empty convolution and recurrent state. It validates block masks,
padding and causal label boundaries, and compares all layers and gradients with
independent native sequences, including checkpoint recomputation and checkpoint
reload. With dense `qwen3_5_text` LoRA, `--packing-seed 123` explicitly selects
this reference interface for `run-finetune` or `calibrate-finetune`. It requires a
planning horizon and native packing (`packing: true` plus `neat_packing: true`
for Factory, or `sample_packing: true` for Axolotl), with evaluation disabled.
HF processed rows additionally require `position_ids` and contiguous positive
segment IDs in `attention_mask`, followed by optional zero right padding.
The first label of every subsequent segment must be ignored; the row's first label is removed by the causal shift. Native source sampling uses an
independent seed plus epoch; the native bin packer, tails and dropped rows remain
in effect. Sampler plan v3 records complete epoch bins, logical-sequence counts,
physical tensor rows, block geometry, dtypes and consumed input hashes. Budget
identity retains the same dataset and packing seed; different data needs its own
calibration. This path splits recurrent computation per segment and makes no
varlen-kernel throughput claim. Full/QLoRA packing and grouped packing are not
covered. The PEFT direct-forward regression is in
[the native audit](tests/native/finetune_segment_peft.py). The
[packed-budget evidence](tests/data/finetune_packing_budget_evidence.json) retains
four native cases, their prediction errors and allocator underestimation.
Native packing reports require the sampler-aware transfer commands; the old
fixed-shape projection cannot interpret their source-row counts.

Calibration workspaces isolate Axolotl prepared datasets and explicit Factory
tokenized datasets under their own `prepared/` directory. They derive those
files from the recorded raw inputs, preserving external caches. Resume rejects
changed raw inputs; ordinary `run-finetune` retains the native cache policy.

The memory result is the observed full-run CUDA allocated/reserved peak envelope
for a covered workload on the same GPU and allocator. This does not project to a
smaller physical card. Per-window peaks are not added. The optional reserve is
a user-selected margin; fitting this envelope does not prove OOM safety.
`validate-finetune-pipeline` compares against an independent holdout run and
reports timing errors and memory underestimation. Actual allocation failures are tested separately.

`run-finetune --cache-regime empty_kernel_cache` requires five explicitly set,
distinct existing empty directories: `TRITON_CACHE_DIR`,
`TORCHINDUCTOR_CACHE_DIR`, `CUDA_CACHE_PATH`, `TORCH_EXTENSIONS_DIR`, and
`PYTORCH_KERNEL_CACHE_PATH`. `populated_kernel_cache` records every file's
relative path, size and SHA256; replicate the same seed into fresh directories
for independent runs. The runner never creates, clears or warms these caches.
These modes control only the named directories, not OS file caches or all
library caches. Default `uncontrolled` reports make no cache-state claim and
cannot be mixed with controlled calibration. File manifests do not certify
cache hits or external file dependencies: Triton group metadata can retain
absolute references to the original cache. Keep those files available and
unchanged. This experiment separately fixed and audited that dependency.

Unpacked Hugging Face native runs use a flat config and processed JSONL rows containing
exactly `input_ids`, `attention_mask` and `labels`: equal-length integer arrays,
a binary attention mask, and 2 through `sequence_length` tokens before padding.
The local tokenizer must have a pad token. No implicit tokenization or truncation
is applied. See [the native LoRA example](examples/finetuning/huggingface-native-lora.yaml)
and pass `--framework huggingface` to `run-finetune`. The original
`probe-finetune` fixed-shape workflow remains available.

The [complete pipeline validation record](tests/data/finetune_pipeline_validation_evidence.json)
summarizes 30 native processes across all three frameworks, including separate
allocator OOM controls. See [`refactor.md`](refactor.md) for phase-level errors,
cache conditions and the remaining coverage limits.

The [sampler and transfer validation record](tests/data/finetune_transfer_validation_evidence.json)
adds 18 Qwen3.5-0.8B native processes on Blackwell GPU: three framework tests
changing 25 to 73 training rows, plus random sampling with periodic eval/save.
Each condition has one independent holdout. The [Qwen3.5-2B record](tests/data/finetune_qwen2b_evidence.json)
adds 512/1024-token LoRA and 512-token NF4 QLoRA with nonreentrant checkpointing:
two calibration processes and two independent holdouts per condition, all on GPU 1.
The six total-time errors range from 0.054% to 1.578%; these measurements do not establish
general accuracy or OOM safety; phase errors and rejected coverage are retained.

Large reports and plans can use `.json.gz` with the same `--json` and input-path
options as JSON. Compression streams to an atomic replacement and is reproducible;
plain JSON and stdout keep their existing format. The native planner reuses its
just-completed full dataset scan and still verifies every planned batch twice.
This reduces duplicate work; exact plans still grow with their declared horizon.

Standard gradient checkpointing is supported. LLaMA-Factory uses
`disable_gradient_checkpointing: false` with `use_reentrant_gc: true` or `false`;
Axolotl uses `gradient_checkpointing: true` with
`gradient_checkpointing_kwargs: {use_reentrant: true}` (or set `use_reentrant`
to `false`). Effective
checkpointing and quantization settings are read from the loaded model and
included in calibration identity. Changing them requires a new calibration.

Native text runs require a text checkpoint; `--text-only` on an estimator does
not convert the original multimodal weights for a native run. For local Qwen3.5,
first export and verify a separate text checkpoint on CPU and create synthetic
Alpaca data for framework integration checks:

```bash
python tests/native/prepare_finetune_fixture.py \
  --model /path/to/Qwen3.5-0.8B \
  --output-dir /tmp/fakegpu-finetune/qwen35-text \
  --dataset-dir /tmp/fakegpu-finetune/data
```

Use a Transformers environment supporting `Qwen3_5ForCausalLM` for export.
The native examples target LLaMA-Factory 0.9.5 with Transformers 5.6.0, and
Axolotl 0.18.0 with Transformers 5.14.1. Install the selected framework and its
compatible CUDA PyTorch, PEFT, and dataset dependencies in its own environment.
The configs use the paths above and require a new or empty output directory;
choose a new `output_dir` before repeating a run. Axolotl's prepared-data cache
is retained separately at `/tmp/fakegpu-finetune/axolotl-prepared`.

```bash
CUDA_VISIBLE_DEVICES=1 python -m fakegpu run-finetune \
  --config examples/finetuning/axolotl-native-lora.yaml --framework axolotl \
  --profile "$TARGET_GPU_PROFILE" --warmup-steps 2 --output run.json
python -m fakegpu estimate-finetune \
  --run-report run.json --epochs 3 --max-steps -1 --json estimate.json
```

For LLaMA-Factory, use `--framework llamafactory` with
[`llamafactory-native-lora.yaml`](examples/finetuning/llamafactory-native-lora.yaml). Reuse
`--run-report` in the same Python environment: the CLI checks the current local
weights, model config, and software identity against the recorded execution.
The effective dataset fingerprint and measured shapes remain bound to the
report; the GPU profile defaults to the report's profile. This legacy
`estimate-finetune` single-report interface requires fixed full batches. Its
native pipeline total reuses one measured
setup/warmup/final-save cost only for the same dataset and local cache state,
with no periodic evaluation or checkpoint schedule. A different `--num-samples`
provides a training-loop scenario; its unmeasured preprocessing and loading costs
are not projected. These scenarios are not confidence intervals and exclude
Python process startup and downloads.

For a 512-token NF4 QLoRA run with checkpointing, use
[`llamafactory-native-qlora.yaml`](examples/finetuning/llamafactory-native-qlora.yaml)
or [`axolotl-native-qlora.yaml`](examples/finetuning/axolotl-native-qlora.yaml).
These examples enable double quantization and BF16 compute/storage, and require
bitsandbytes in the selected framework environment. Create their longer dataset
from the exported tokenizer without exporting the model again:

```bash
python tests/native/prepare_finetune_fixture.py \
  --model /tmp/fakegpu-finetune/qwen35-text --dataset-only \
  --dataset-dir /tmp/fakegpu-finetune/data-long \
  --sequence-length 1024 --num-samples 64
```

The requested fixture length is a lower bound checked before native
preprocessing. Each framework applies its configured 512- or 1024-token cutoff;
the run report records the sequences and batches actually used.

Independent calibration and holdout processes can be collected with
[`tests/native/finetune_framework_matrix.py`](tests/native/finetune_framework_matrix.py).
The matrix takes sequence length, microbatch size, accumulation, and seed from
the native config. Calibration and holdout lengths are independently selectable:

```bash
python tests/native/finetune_framework_matrix.py \
  --framework axolotl --config examples/finetuning/axolotl-native-qlora.yaml \
  --python /path/to/axolotl/venv/bin/python \
  --output-dir /tmp/fakegpu-finetune/axolotl-qlora-matrix \
  --gpu-index 1 --calibration-steps 6 --warmup-steps 2 \
  --holdout-runs 2 --holdout-steps 16
```

Step counts include warmup. Defaults remain 6 calibration updates, 2 warmup
updates, and three 32-update holdouts. Every run uses a fresh output directory;
the matrix preserves the source seed and resolves local paths before copying
the config. Predictions use the production estimator and actual microstep
counts, including partial accumulation windows at epoch boundaries.

The [native validation archive](tests/data/finetune_framework_validation_evidence.json)
retains the earlier 128-token Qwen3.5-0.8B LoRA runs for both frameworks on Blackwell GPU, including
three independent one-epoch holdouts each. Maximum native wall-time errors were
3.931% for Factory and 1.125% for the cached Axolotl repeat. It also retains a
364.885% error when Axolotl's first-use calibration was applied to subsequent
cached runs: the first update took 66.542 seconds. Cache/JIT state is not
automatically fingerprinted; a matching model/software signature alone cannot
make those fixed costs reusable. See [`refactor.md`](refactor.md) for the exact
workload, versions, memory results, and remaining coverage.

The [extended validation archive](tests/data/finetune_extended_validation_evidence.json)
covers six configurations across both frameworks at 512/1024 tokens and batch
sizes 1/2, including gradient checkpointing and native NF4 QLoRA. Each has one
6-update calibration and two independent 16-update holdouts, with 2 warmup
updates per process: 18 measured processes in total. Every configuration had a
short priming run first, and shared caches were retained. Exact results and
configuration boundaries are recorded in [`refactor.md`](refactor.md).

For a measured estimate, run a short real-GPU probe and reuse its observation
in the **same Python environment**, with the same model, GPU, precision, adapter,
attention implementation, batch, sequence length, and accumulation settings:

```bash
CUDA_VISIBLE_DEVICES=1 python -m fakegpu probe-finetune \
  --config examples/finetuning/huggingface-lora.yaml \
  --model /path/to/Qwen3.5-0.8B --text-only \
  --profile "$TARGET_GPU_PROFILE" --warmup-steps 2 --measure-steps 5 \
  --output observation.json
python -m fakegpu estimate-finetune \
  --config examples/finetuning/huggingface-lora.yaml \
  --model /path/to/Qwen3.5-0.8B --text-only \
  --profile "$TARGET_GPU_PROFILE" --num-samples 10000 --epochs 3 \
  --observation observation.json --json calibrated.json
```

The built-in probe executes Hugging Face Trainer with fixed-length synthetic
arithmetic text; Transformers with support for the selected architecture,
Accelerate, and CUDA PyTorch are required, plus PEFT for adapters and
bitsandbytes for QLoRA. It records warmup separately, includes optimizer work
and batch fetching in measured updates, and retains both startup and steady
memory peaks. Its measurements do not establish convergence or another
framework's runtime performance. Dataset preparation, model loading, evaluation,
and checkpoint I/O are excluded from projected training-loop time. Analytical
scenario ranges and measured sample extrema are not confidence intervals.
Probe collection and observation reuse hash the local checkpoint files, so
replacing weights at the same path invalidates the old calibration. Reading
those files adds startup I/O, especially for large checkpoints.

The [Qwen3.5-0.8B validation record](tests/data/finetune_validation_evidence.json)
contains independent calibration/holdout runs for full, LoRA, and bitsandbytes
QLoRA on an Blackwell GPU. It retains the analytical errors as well as calibrated
comparisons; these short runs do not establish accuracy for longer or different
workloads. Reproduce them with [`tests/native/finetune_matrix.py`](tests/native/finetune_matrix.py).

The detailed feature acceptance matrix and remaining implementation phases are
maintained in [`refactor.md`](refactor.md).

`benchmark-serving` measures an already running completion endpoint with vLLM's
`ignore_eos` and final streaming usage fields. Its explicit token-ID manifest sets
arrivals and output limits; the report retains SSE events, failures, client queue
time and first nonempty text latency. It does not retry requests or infer device
phase peaks. Use the served model name configured by your server.

```bash
python -m fakegpu benchmark-serving \
  --base-url http://127.0.0.1:8000/v1 --model local-model \
  --requests examples/serving/online-requests.json \
  --concurrency 4 --repetitions 3 --json build/online.json.gz
```

The [native serving record](tests/data/serving_native_evidence.json) includes 20
8K/32K Transformers processes and 72 mixed vLLM requests with prefix caching off/on.
At 32K, allocated phase errors were 2.02%/7.34%; reserved decode was underestimated
by 24.07%. Analytical tensor storage therefore does not establish allocator or OOM
safety. Cache conditions, first-round overhead and all samples remain in the record.

Cached-feature UNet training supports full tuning, LoRA and optional `--ema-decay`.
`plan-diffusion-training` hashes the exact local weights and FP32 cached tensors;
`run-diffusion-training` uses one real CUDA GPU, BF16 autocast, FP32 trainable
parameters and AdamW. VAE/text-encoder training and raw-image preparation are
excluded. The fixture below generates random features for integration tests.
Use a compatible Diffusers/PEFT environment; the validated stack is recorded below.

```bash
python tests/native/prepare_diffusion_fixture.py \
  --model-dir /path/to/stable-diffusion-v1-4 --output-dir /tmp/diff-fixture \
  --num-samples 5 --epochs 3 --method lora --gradient-checkpointing
CUDA_VISIBLE_DEVICES=1 python -m fakegpu run-diffusion-training \
  --plan /tmp/diff-fixture/plan.json --output-dir /tmp/diff-cal-1 \
  --json /tmp/diff-cal-1.json
CUDA_VISIBLE_DEVICES=1 python -m fakegpu run-diffusion-training \
  --plan /tmp/diff-fixture/plan.json --output-dir /tmp/diff-cal-2 \
  --json /tmp/diff-cal-2.json
python -m fakegpu estimate-diffusion-training --plan /tmp/diff-fixture/plan.json \
  --epochs 5 --calibration-run /tmp/diff-cal-1.json \
  --calibration-run /tmp/diff-cal-2.json --json /tmp/diff-budget.json.gz
```

Activation and reserved peaks stay unknown without compatible native observations.
Budgets require two independent processes and repeated samples for every future
window; first use of a shape is charged separately. `--cache-regime` records the
selected kernel-cache state (default `uncontrolled`); controlled modes require the
five explicit cache directories documented in the native workflow. Allocator,
GPU UUID, software, input content and execution identity must match. The observed
allocator envelope is not a physical OOM guarantee.

Use `validate-diffusion-training` with an independent `--holdout-run` and the same
calibrations; optional `--max-total-time-error-percent` and
`--max-memory-underestimate-percent` return status 1 when a limit is exceeded.
The [native diffusion record](tests/data/diffusion_native_evidence.json) includes
SD/SDXL generation observations and twelve cached-feature training processes.
Six training holdouts have 0.044%–3.185% total-time error and no allocator-envelope
underestimate. Generation VAE allocated peaks were underestimated by 12.0%–13.8%,
and reserved peaks by up to 31.6%; those failures remain visible.

The [subsequent phase-memory validation](tests/data/phase_memory_evidence.json)
fixes the VAE upsampling shape omission: four independent resolution cohorts have
0.045%–1.172% VAE allocated error. Six conditions, each with five calibrations and
five later holdouts (60 fresh processes), pass both allocated/reserved gates using
explicit native phase budgets. All held-out peaks equal their calibration maxima
in this fixed-input BF16 matrix. Earlier uncalibrated failures remain preserved;
these results do not imply transfer to new shapes or physical OOM safety.

Allocated tensor storage and allocator-reserved memory are separate quantities.
`calibrate phase-sample` collects both in one fresh CUDA process; `phase-estimate`
seals per-phase maxima from at least five independent processes, and `phase-validate`
checks a later independent run. Use a new output path for every sample and budget.

```bash
# plan.json: a compatible local plan-serving or estimate-diffusion report
for i in 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=1 python -m fakegpu calibrate phase-sample plan.json \
    --json "/tmp/phase-cal-$i.json"
done
python -m fakegpu calibrate phase-estimate \
  --calibration-sample /tmp/phase-cal-1.json \
  --calibration-sample /tmp/phase-cal-2.json \
  --calibration-sample /tmp/phase-cal-3.json \
  --calibration-sample /tmp/phase-cal-4.json \
  --calibration-sample /tmp/phase-cal-5.json --json /tmp/phase-budget.json.gz
CUDA_VISIBLE_DEVICES=1 python -m fakegpu calibrate phase-sample plan.json \
  --json /tmp/phase-holdout.json
python -m fakegpu calibrate phase-validate --budget /tmp/phase-budget.json.gz \
  --holdout-sample /tmp/phase-holdout.json --max-absolute-error-percent 20 \
  --max-underestimate-percent 10 --json /tmp/phase-validation.json
```

Sampling supports homogeneous Transformers dynamic-cache serving, and local
SD/SDXL with SDPA and no offload/slicing/tiling. Each plan, full selected weights
and configuration, GPU UUID, Torch/CUDA/cuDNN versions, allocator and execution
implementation must match. Reports expose both metrics and the analytic prediction
separately; no shape transfer or hidden safety multiplier is applied. Kernel caches
are uncontrolled. Offline estimation/validation needs only the standard library.
Default validation limits are 20% absolute error and 10% underestimation; failures
return status 1. Observed envelopes are not physical OOM guarantees.

The VAE shape model includes the high-resolution upsampling convolution before
channel reduction; its working set can be larger than the following block output.
Native SD/SDXL allocated/reserved accuracy evidence is recorded separately below.

For FSDP2 `full_shard`, `plan-training --parameter-shapes layout.json` accepts
`{"shapes": [[12, 8], [12], [3, 12], [3]]}` plus the matching total
`--parameter-bytes` and uniform `--parameter-element-bytes`. It models native
per-parameter dim-0 chunks, exposes bytes for each rank, and budgets the largest
rank. Without explicit shapes the existing aggregate balanced-shard approximation
remains; it does not cover uneven parameter partitions. Optimizer dtype and
workspace assumptions still apply. This path does not model custom placements.

The [versioned native validation matrix](tests/data/research_native_evidence.json)
records six FP32/FP16/BF16 LoRA/attention CPU_SIM cases, tiny Qwen3 MoE routing and
gradients on CPU/GPU 1, and two CPU ranks running native DDP/FSDP2. These are finite
numeric and semantic checks. Physical TP/PP/CP/EP, multi-GPU ZeRO/FSDP and multi-node
collectives are outside the current project scope and remain unvalidated. Arbitrary
CUDA device code is not executed by CPU_SIM; strict unsupported-API checks remain
enabled in the numeric probe. Reproduce the host math with `scripts/test.sh cpu`
and the local semantics with `tests/native/research_distributed.py --help`.

Finite experiment sweeps use an explicit argv array, parameter grid and repetition count.
The CPU-only example below binds its worker script, executable, FakeGPU source and
selected resource environment to stable run IDs. Execution is sequential; every
attempt retains stdout, exit status, errors, artifact hashes and numeric metrics.

```bash
python -m fakegpu plan-experiments --config examples/experiments/cpu-sweep.json \
  --workspace /tmp/research-sweep --json /tmp/research-plan.json
python -m fakegpu run-experiments --config examples/experiments/cpu-sweep.json \
  --workspace /tmp/research-sweep --json /tmp/research-results.json.gz
python -m fakegpu run-experiments --config examples/experiments/cpu-sweep.json \
  --workspace /tmp/research-sweep --resume --json /tmp/research-resumed.json
python -m fakegpu compare-experiments --workspace /tmp/research-sweep \
  --plot /tmp/research-sweep.svg --x size --metric seconds \
  --json /tmp/research-comparison.json
```

Resume skips verified successes; `--resume --retry-failed` explicitly creates new
attempts for failures or interruptions. Input/source/environment changes require a
new workspace. Compare works offline and keeps workspaces separate. Declared inputs
are attested; authors must pin external packages and record native runtime identity.
Commands must stay in the foreground. JSON key/index metric paths support nested
objects/arrays. Plotting alone requires Matplotlib; bars show observed min/max, not
confidence intervals. The example measures CPU work, not GPU training throughput.

The [release regression record](tests/data/research_release_evidence.json) covers
1734 passing Python tests (one CPU-only-build branch skipped), the native CPU
suite, 104 packaged Python modules, 42 command help checks and 68 offline
functional CLI checks. All 152 historical predictions remain byte-identical.

| Command | Purpose |
|---|---|
| `fakegpu doctor` | Check the installation, native libraries, PyTorch, and profiles |
| `fakegpu demo` | Run a small CPU-backed, CUDA-visible training step |
| `fakegpu preflight` | Execute a workload to a target stage and classify fit or OOM |
| `fakegpu analyze-repo` | Inventory repository entry points and GPU-only risks |
| `fakegpu analyze-kernel` | Inspect CUDA, PTX, and SASS resources and operations |
| `fakegpu estimate-llm` | Estimate decoder memory, communication, and FLOPs |
| `fakegpu plan-serving` | Plan homogeneous or mixed-request admission, chunked prefill, and grouped shared-prefix KV storage |
| `fakegpu benchmark-serving` | Measure scheduled mixed streaming requests and retain usage, latency, SSE events, and failures |
| `fakegpu plan-experiments` | Plan a bounded parameter sweep and inspect exact commands |
| `fakegpu run-experiments` | Run or explicitly resume experiments while preserving every attempt |
| `fakegpu compare-experiments` | Verify raw artifacts, compare repetitions and optionally export a plot |
| `fakegpu estimate-diffusion` | Estimate diffusion text-encode, denoise, and VAE-decode memory phases |
| `fakegpu plan-diffusion-training` | Inspect cached-feature UNet training plans and persistent storage |
| `fakegpu run-diffusion-training` | Run cached-feature UNet full/LoRA training on one real GPU |
| `fakegpu estimate-diffusion-training` | Budget compatible native cached-feature training observations |
| `fakegpu validate-diffusion-training` | Compare an independent native diffusion training holdout |
| `fakegpu estimate-roofline` | Produce a profile-aware analytical latency interval |
| `fakegpu plan-training` | Normalize distributed training configs and estimate rank memory |
| `fakegpu estimate-finetune` | Estimate single-GPU full/LoRA/QLoRA training steps, time, and VRAM from framework configs |
| `fakegpu probe-finetune` | Collect fixed-shape Hugging Face Trainer timing and memory on a physical CUDA GPU |
| `fakegpu profile-finetune-data` | Scan local JSONL data for token lengths, truncation, padding, and processed sequence counts |
| `fakegpu run-finetune` | Run native HF Trainer/PEFT, LLaMA-Factory or Axolotl training and record data, CUDA timing, memory, and final saving |
| `fakegpu validate-finetune-windows` | Validate dynamic window timing using independent calibration and holdout runs |
| `fakegpu validate-finetune-events` | Validate evaluation and checkpoint costs from independent native event records |
| `fakegpu plan-finetune-pipeline` | Plan future native epochs/updates from a calibrated sequential dataset |
| `fakegpu estimate-finetune-pipeline` | Estimate complete native execution time and a covered memory envelope |
| `fakegpu validate-finetune-pipeline` | Compare full-span time and memory predictions against independent native runs |
| `fakegpu plan-finetune-transfer` | Plan verified random or sequential native inputs |
| `fakegpu estimate-finetune-transfer` | Estimate explicit empirical shape-transfer budgets |
| `fakegpu validate-finetune-transfer` | Compare transfer budgets with independent native runs |
| `fakegpu calibrate-finetune` | Run bounded native calibration in a resumable workspace |
| `fakegpu budget-finetune-workspace` | Rebudget saved native evidence using only the CPU |
| `fakegpu simulate-topology` | Model collective routes and link contention |
| `fakegpu replay-trace` | Summarize compute, communication, wait, and memory timelines |
| `fakegpu calibrate` | Build serving observations, compare memory reports, and enforce reliability gates |
| `fakegpu capabilities` | List or strictly audit native API classifications |
| `fakegpu nvidia-smi` | Inspect devices, processes, modeled topology/MIG, health status, and reliability events |
| `fakegpu metrics` | Export bounded Prometheus/JSON metrics and serve short in-memory history |
| `fakegpu workspace-profiles` | Validate and inspect workspace estimation profiles |
| `fakegpu validate` | Run a declarative JSON, TOML, or YAML validation matrix |
| `fakegpu coordinator` | Manage the distributed simulation coordinator |
| `fakegpu bandwidth` | Validate simulated TCP payloads and report throughput |

Use `python3 -m fakegpu --help` for the complete list and
`python3 -m fakegpu <command> --help` for command-specific options.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## GPU profiles

The catalog contains 82 YAML profiles covering consumer, workstation,
data-center, and embedded NVIDIA GPUs from Maxwell through Blackwell. Profiles
are shared by the Python and native runtimes.

```bash
python3 -m fakegpu doctor --list-profiles
python3 -m fakegpu demo --profile rtx4090
python3 -m fakegpu --build-dir build --devices "t4,a100:2,h100" \
  python3 your_script.py
python3 scripts/update_nvidia_gpu_catalog.py --check
```

Set `FAKEGPU_PROFILE` or pass `--profile` to select one profile. Use
`--devices` for a heterogeneous list.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Development

### Build

All reusable native build behavior is exposed through one script:

```bash
scripts/build.sh
scripts/build.sh --release
scripts/build.sh --debug
scripts/build.sh --build-dir build-custom -- -DSOME_CMAKE_OPTION=value
```

Build directories and compiled artifacts are ignored by Git.

### Test

The maintained regression surface is grouped into four commands:

```bash
scripts/test.sh python
scripts/test.sh smoke
scripts/test.sh cpu
scripts/test.sh all
```

| Suite | Coverage |
|---|---|
| `python` | Maintained Python regression tests |
| `smoke` | Native loading, reports, capabilities, SMI topology/health, and coordinator |
| `cpu` | CPU-backed cuBLAS simulation |
| `all` | All maintained suites |

Run a declarative validation manifest directly when needed:

```bash
python3 -m fakegpu validate \
  --manifest tests/data/validation_smoke.yaml \
  --report-dir build/validation-smoke \
  --strict

python3 -m fakegpu validate \
  --manifest tests/data/research_validation.yaml \
  --report-dir build/validation-research \
  --strict
```

### Reusable scripts

| Path | Purpose |
|---|---|
| `scripts/build.sh` | Configure and compile native targets |
| `scripts/test.sh` | Run maintained test suites |
| `scripts/update_nvidia_gpu_catalog.py` | Check or update profile metadata |
| `scripts/validation/` | Shared report and artifact validators |
| `scripts/linux/` | Linux GPU-management helpers |
| `scripts/macos/` | macOS-to-Linux-VM helpers |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Project structure

```text
FakeGPU/
├── fakegpu/    Python package, CLI, runtimes, and estimators
├── profiles/   YAML GPU profile catalog
├── schemas/    JSON report and validation schemas
├── scripts/    Reusable build, test, platform, and validation tools
├── src/        Native C++ interception and coordinator implementation
└── tests/      Maintained regression tests and minimal native fixtures
```

Generated build directories, compiled libraries, test reports, caches, local
environments, binary assets, and design drafts are excluded through
`.gitignore`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Limitations

- Native simulation does not execute arbitrary CUDA kernels.
- FakeCUDA covers maintained Python and PyTorch behavior, not binary CUDA
  extensions.
- Static analysis cannot resolve every dynamic import, generated kernel,
  runtime shape, or data-dependent branch.
- Memory estimates can miss backend-private allocations, custom operators,
  allocator policies, and unmatched workspaces.
- Native cuBLAS algorithms without numeric implementations, unsupported GEMM/Lt
  dtypes, layouts or pointer modes, and disabled or skipped CPU computation follow
  `FAKEGPU_UNSUPPORTED_API`.
  `warn` records and warns about each fallback; `error` returns
  `CUBLAS_STATUS_NOT_SUPPORTED` before writing synthetic output or success metadata.
  Supported CPU calculations still execute in strict mode; this does not turn
  arbitrary CUDA kernels into executable implementations.
- For models declaring a vision encoder, `estimate-llm` and `plan-serving`
  include checkpoint weights but leave vision/projector activations and FLOPs
  unmodeled. Both JSON and terminal reports flag this incomplete coverage.
  Decoder token counts must already include the visual tokens inserted by the
  processor; validate full multimodal memory before using a fit/admission result.
- Diffusion profiles model fixed reference pipelines. Custom ControlNet,
  IP-Adapter, LoRA, safety-checker, refiner, video, and DiT components require
  additional component metadata and matching real-GPU validation.
- Online-serving plans accept explicit mixed lengths and independent
  autoregressive draft models. They do not model request arrival distributions,
  cache eviction, preemption or scheduler reordering, self-speculative/Medusa/
  EAGLE variants, tokenizer remapping, tensor-parallel execution, throughput,
  or latency.
- Roofline output is an analytical interval, not measured kernel latency.
- Distributed timing includes coordinator work, memory copies, sockets, and
  process scheduling; it is not an NCCL, NVLink, or RDMA benchmark.
- Modeled fault events are report-only control-plane inputs. They do not alter
  CUDA execution or represent NVML ECC counters, Xid observations, or hardware
  failures.
- Hybrid and passthrough modes require a compatible physical CUDA stack.
- macOS System Integrity Protection can remove `DYLD_*` variables from system
  binaries. Prefer a Homebrew, conda, or pyenv Python for native interception.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Roadmap

- [x] CPU-backed PyTorch FakeCUDA runtime
- [x] Native CUDA, NVML, cuBLAS, and NCCL interception
- [x] Architecture-aware GPU profile catalog
- [x] Runtime, static, LLM, and distributed memory analysis
- [x] Framework-aware single-GPU fine-tuning estimates and matching Hugging Face Trainer probes
- [x] Full dataset scans and native LLaMA-Factory/Axolotl run reports for single-GPU text full fine-tuning, LoRA, and 4-bit QLoRA
- [x] Phase-aware Stable Diffusion v1.5 and SDXL generation memory estimation
- [x] Repository, kernel, topology, and trace analysis
- [x] Detailed FakeGPU-SMI device, runtime, allocator, and process queries
- [ ] Expand executable native CUDA operations and cuBLAS coverage
- [x] Publish live native-runtime state and activity through FakeGPU-SMI
- [x] Add modeled topology and NVLink views with NVML peer queries
- [x] Add modeled fault injection and health/reliability event views
- [x] Add modeled MIG views and native NVML MIG handle queries
- [x] Export bounded device, process, and runtime metrics with in-memory history for Prometheus
- [x] Add modeled online-serving plans for continuous batching, mixed request lengths, chunked prefill, and grouped prefix caching
- [x] Add modeled draft-model speculative-decoding memory and admission plans
- [x] Add documented real-GPU long-context and online-serving validation
- [x] Add SD/SDXL GPU observations, cached-feature UNet full/LoRA/EMA budgets, and architecture-specific modeled DiT memory
- [ ] Validate distributed and MoE estimates across more GPU and software stacks
- [x] Validate LLaMA-Factory and Axolotl native runners on the documented Qwen3.5 single-GPU scenarios

See the [open issues](https://github.com/FanBB2333/FakeGPU/issues) for proposed
features and known limitations.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

Bug reports, focused test cases, profile corrections, documentation
improvements, and implementation patches are welcome.

1. Fork the repository.
2. Create a branch: `git checkout -b feat/your-change`.
3. Add or update tests for the changed behavior.
4. Run `scripts/test.sh all`.
5. Commit with a clear
   [Conventional Commit](https://www.conventionalcommits.org/) message.
6. Push the branch and open a pull request.

For estimation or compatibility issues, include the exact command, selected
profile, software versions, and generated report.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments

- README structure inspired by
  [Best-README-Template](https://github.com/othneildrew/Best-README-Template)
- CPU-backed framework validation built around
  [PyTorch](https://pytorch.org/)
- Native builds powered by [CMake](https://cmake.org/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

[test-shield]: https://img.shields.io/github/actions/workflow/status/FanBB2333/FakeGPU/test.yml?branch=main&style=for-the-badge&label=tests
[test-url]: https://github.com/FanBB2333/FakeGPU/actions/workflows/test.yml
[release-shield]: https://img.shields.io/github/v/release/FanBB2333/FakeGPU?include_prereleases&sort=semver&style=for-the-badge
[release-url]: https://github.com/FanBB2333/FakeGPU/releases
[python-shield]: https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white
[python-url]: https://www.python.org/
[license-shield]: https://img.shields.io/github/license/FanBB2333/FakeGPU?style=for-the-badge
[license-url]: LICENSE
[validation-static]: https://github.com/FanBB2333/FakeGPU/blob/df254c21eebc0a5bbf13992f3f5a8e995cfa8708/docs/ai-researcher-preflight.md#4-static-aten-storage-liveness-validation
[validation-inference]: https://github.com/FanBB2333/FakeGPU/blob/df254c21eebc0a5bbf13992f3f5a8e995cfa8708/docs/llm-inference-estimation.md#maintained-qwen3-8b-result
[validation-sft]: https://github.com/FanBB2333/FakeGPU/blob/df254c21eebc0a5bbf13992f3f5a8e995cfa8708/docs/llm-sft-memory-estimation.md#blackwell-matrix
[validation-qlora]: https://github.com/FanBB2333/FakeGPU/blob/df254c21eebc0a5bbf13992f3f5a8e995cfa8708/docs/llm-sft-memory-estimation.md#native-nf4-qlora-matrices
[diffusion-sd15]: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/451f4fe16113bff5a5d2269ed5ad43b0592e9a14
[diffusion-sdxl]: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/462165984030d82259a11f4367a4eed129e94a7b
[diffusion-pixart]: https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS/tree/e102b3591cc82e97071b8b4cb90d834d0c487207
