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
   - [Fine-tuning time and VRAM](#fine-tuning-time-and-vram)
   - [Online serving benchmark](#online-serving-benchmark)
   - [Diffusion training](#diffusion-training)
   - [Phase memory calibration](#phase-memory-calibration)
   - [Distributed plans and experiment sweeps](#distributed-plans-and-experiment-sweeps)
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
checks, and capacity planning. Applications see configurable NVIDIA-like
devices while maintained operations run on CPU; FakeGPU records simulated
memory and communication and provides static estimators for workloads that
should not be loaded at all.

A physical GPU is needed only for passthrough, hybrid, and calibration runs.

Examples use `TARGET_GPU_PROFILE` for a catalog ID you select with
`python3 -m fakegpu doctor --list-profiles`. Public experiment records disclose
architecture only; supported model names describe compatibility, not ownership.

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

Error is `|predicted - observed| / observed × 100%`; agreement is
`100% - error`.

| Validated envelope | Real-GPU reference | Evidence | Absolute percentage error | Agreement |
|---|---|---:|---:|---:|
| [Controlled ATen MLP and Transformer grid with backend-resident calibration][validation-static] | Ampere GPU and Blackwell GPU; PyTorch/CUDA 2.12/13.0 and 2.9/12.8 | 13 workloads, 26 observations | **0.08% maximum** | **≥99.92%** |
| [Qwen3-8B BF16 SDPA inference][validation-inference] | Blackwell GPU; PyTorch 2.9.1/CUDA 12.8 | Model load and inference peak | 0.0129% load; **0.0672% peak** | 99.9871%; **99.9328%** |
| [Qwen 0.8B/2B full and LoRA SFT][validation-sft] | Blackwell GPU; PyTorch 2.8/CUDA 12.8 | 10 training cases | **0.102%–1.921%** | **98.079%–99.898%** |
| [Qwen 0.8B/2B native NF4 QLoRA][validation-qlora] | Blackwell GPU; PyTorch 2.8/CUDA 12.8 | 10 quantized training cases | **0.628%–1.732%** | **98.268%–99.372%** |

How to read these numbers:

- Qwen rows use `torch.cuda.max_memory_allocated()`; CUDA context and
  reserved-but-unused allocator memory are excluded.
- The ATen row adds one backend-resident measurement from the exact stack; do
  not reuse it on another stack.
- Ranges are min/max case errors, not averages. Underestimation is the
  failure mode that matters for OOM risk.
- `99.x%` agreement is not spare capacity; still apply a safety margin.

These are fixed-workload results, not a universal accuracy claim. Other
models, shapes, backends, kernels, allocators, software versions, or GPUs need
a matching calibration. The links point to an immutable snapshot with full
configurations and byte counts; CI checks this table against a
[machine-readable evidence summary](https://github.com/FanBB2333/FakeGPU/blob/main/tests/data/memory_validation_evidence.json).

On a CUDA host, regenerate the controlled comparison with:

```bash
python3 scripts/validation/static_memory_validation.py \
  --output build/static-memory-validation.json \
  --markdown build/static-memory-validation.md \
  --max-underestimate-percent 5
```

On a CPU-only host, `--static-only` checks the estimation path without
producing an accuracy measurement. To compare a `plan-serving` report with
repeated real-GPU phase peaks:

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

Every sample is kept and each phase uses its maximum. Five samples per phase
are required by default (`--strict` exits 1 otherwise); `ready_for_comparison`
means enough samples, not `GPU-validated`. The plan must set
`--target-profile`.

To automate repeated runs with a framework-specific runner:

```bash
python3 -m fakegpu calibrate collect-serving \
  build/speculative-serving-plan.json \
  --repetitions 5 \
  --timeout-seconds 900 \
  --strict \
  --json build/serving-observation.json \
  -- python3 benchmark_serving.py
```

Homogeneous, non-speculative Transformers workloads can use the built-in
runner; the plan must use a dynamic KV cache:

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

On hosts without `/mnt/z`, change only the models root and keep the
`Qwen/Qwen3-0.6B` layout. `sample-transformers`:

- rechecks the local config and Safetensors headers against the plan, then
  measures full prefill and token-by-token decode;
- reports `torch.cuda.max_memory_reserved` by default (`--metric allocated`
  for tensor-only peaks);
- rejects mixed requests, speculative decoding, paged or quantized caches,
  shared-prefix or chunked-prefill plans, quantized checkpoints, and adapters;
- needs PyTorch and Transformers already installed in the CUDA environment.

vLLM or custom instrumentation with trustworthy phase peaks can build the
sample directly:

```bash
python3 -m fakegpu calibrate emit-serving-sample \
  build/vllm-serving-plan.json \
  --prefill-peak-bytes 24900000000 \
  --decode-peak-bytes 25300000000 \
  --metric nvml.process_family_peak_bytes \
  --framework vllm \
  --framework-version 0.10.2
```

The same adapter is `fakegpu.build_cuda_serving_sample(...)` in Python. vLLM's
cache-usage metric is a block fraction, not a byte peak, so the benchmark must
supply real phase bytes.

Each repetition runs in a new process and must print one
`fakegpu.serving_peak_sample.v1` object as its last line, or on a line prefixed
with `FAKEGPU_SERVING_SAMPLE=`:

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

- FakeGPU passes the plan path, workload signature, target profile, compute
  capability, run index, and run count through `FAKEGPU_SERVING_*` variables.
  The built-in runner and the protocol builder fill this contract
  automatically, and neither framework becomes a FakeGPU dependency.
- Collection fails on simulated CUDA, capacity outside a 2% profile tolerance,
  a different compute capability, inconsistent software metadata, a timeout,
  or a nonzero exit.
- Raw command arguments are not stored, only the executable name and a
  command fingerprint.
- `simulated: false` is runner-supplied, not hardware attestation; a
  `GPU-validated` claim still needs retained logs and calibration gates.

To compare and gate any compatible prediction and observation:

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

`compare` reports per-phase signed/absolute error, interval coverage, and a
recommended safety margin. `verify` exits 1 when a gate fails: underestimation,
median/p95/max error, interval coverage, false-safe fits at the given capacity,
or workload consistency. Apply results only to the same workload signature,
shapes, dtype, software stack, and GPU profile.

<a id="llm-reliability"></a>

### Research workload reliability report

Reliability is reported per workload and environment signature:

- `GPU-validated`: applies only to the recorded model revision, shapes, dtype,
  attention backend, allocator, software stack, and GPU.
- `CPU-validated`: maintained execution or analysis passed without a GPU.
- `Modeled`: analytical coverage without matching real-GPU evidence.
- `Planned`: not yet a supported accuracy claim.

Serving plans expose a path-independent `workload_signature`, so evidence
cannot silently cross serving configurations. Collect repeated real-CUDA
samples with `calibrate observe-serving` or `calibrate collect-serving`.

#### Verification snapshot (2026-08-05)

Verified with every `scripts/test.sh` suite and both validation manifests on
macOS 26.5 arm64, Python 3.11.9, and PyTorch 2.9.1 CPU. The Transformers
sampler also ran on a Blackwell GPU (PyTorch 2.11.0+cu128,
Transformers 5.14.1).

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

GitHub CI runs the Python suite on Python 3.10–3.12 and the native smoke and
CPU simulation suites on Linux and macOS. The sampler row validates repeated
execution and CUDA identity, not accuracy.

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

These examples come from checked-in formulas and are verified by README
contract tests. They are formula results, not real-GPU observations, and do
not replace same-configuration calibration. GiB values are binary units.

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

- Diffusion profiles use component parameter counts from fixed revisions of
  [Stable Diffusion v1.5][diffusion-sd15], [SDXL Base 1.0][diffusion-sdxl],
  and [PixArt-Sigma XL 2][diffusion-pixart]. Runtime context, fragmentation,
  private workspaces, and transfer overlap are excluded.
- Reports stay `Modeled` with `accuracy.status` set to `uncalibrated` until a
  matching observation exists. `estimate-diffusion --model-dir` exercises
  local architecture inspection.
- Cache formulas follow
  [Transformers cache strategies](https://huggingface.co/docs/transformers/kv_cache).
  Speculative mode follows the draft/target flow of the
  [original speculative-decoding paper](https://arxiv.org/abs/2211.17192) and
  [Transformers assisted decoding](https://huggingface.co/docs/transformers/assisted_decoding);
  serving terms follow [vLLM serving](https://docs.vllm.ai/en/stable/).
- Arrival distributions, cache eviction, preemption, adaptive draft length,
  KV rollback workspaces, and throughput or latency are out of scope. Binary
  CUDA extensions and arbitrary kernels need a passthrough or hybrid real-GPU
  observation.

New or refreshed public validation rows should record:

- at least five isolated observations and the maximum observed peak;
- predicted and observed bytes for every reported phase;
- maximum underprediction as the primary OOM-risk metric, at most 5% for a
  `GPU-validated` row;
- median, 95th-percentile, and maximum absolute percentage error;
- prediction-interval coverage and false-safe decisions (predicted fit, real
  OOM); and
- the model revision, command, shapes, dtype, backend, allocator settings,
  GPU, driver, CUDA, PyTorch, and framework versions.

Use `calibrate verify` to apply these limits before publishing. Rows that miss
the target stay `Modeled` or experimental.

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

- `-q` shows the FakeGPU version, runtime policies, software versions, state
  freshness, catalog and native API coverage, device identity, memory
  categories, allocator activity, dispatch tracking, and per-process peaks.
- `-i` selects devices by index, UUID, PCI bus ID, or profile ID; `--json`
  prints the full normalized inventory. State schema v2 is written; v1 stays
  readable.
- `--query-runtime` exposes the same runtime details as CSV or JSON;
  `--help-query-runtime` lists every field.
- Native interception also publishes allocation lifetime, transfer volume,
  kernel launches, GEMM calls/FLOP, compatibility events, and unsupported API
  counts. `FAKEGPU_SMI_DETAIL_LIMIT` and `FAKEGPU_SMI_MAX_STATE_BYTES` bound
  retained detail and state-file size.

NVLink, MIG, and fault modeling are disabled by default. Set them before the
workload starts; invalid input activates nothing and is reported as a
configuration error.

- **NVLink**: `FAKEGPU_NVLINK_GROUPS="0,1;2,3"` creates full-mesh peers within
  each group; `FAKEGPU_NVLINK_BANDWIDTH_GBPS` defaults to `900`. This drives
  `topo -m`, `nvlink -s`, the `topology.*`/`nvlink.*` fields, and native NVML
  link queries.
- **MIG**: `FAKEGPU_MIG_LAYOUT="DEVICE:PROFILE:MEMORY_MIB[:COUNT];..."`, e.g.
  `FAKEGPU_MIG_LAYOUT="0:1g.10gb:10240:2;1:2g.20gb:20480"`. Each GPU instance
  owns one compute instance; layouts exceeding parent memory, eight slices, or
  eight instances are rejected. Per-instance memory use is not observed.
- **Faults**: `FAKEGPU_FAULT_EVENTS="DEVICE:CODE:SEVERITY[:COUNT];..."`, e.g.
  `FAKEGPU_FAULT_EVENTS="0:XID_79:critical;1:NVLINK_CRC:error:3"`, with
  severities `info`, `warning`, `error`, and `critical`. The `events` view also
  shows unsupported API calls, publisher failures, and stale state; `health.*`
  fields expose modeled device status.

UUIDs and PCI bus IDs are stable simulated values. Temperature, fan speed, live
power, and hardware utilization show `N/A`; topology, bandwidth, faults, and
health are modeled inputs, not hardware measurements or ECC/Xid data.

<a id="export-monitoring-metrics"></a>

### Export monitoring metrics

Export the normalized FakeGPU-SMI state as Prometheus text (default) or JSON,
without extra monitoring dependencies:

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

- `/metrics` serves the latest device, process, runtime, MIG, topology,
  health, and publisher values; `/healthz` reports source and scrape status;
  `/api/v1/history` returns recent samples.
- Process series keep the highest-memory processes: 128 by default, 256 at
  most, `--max-process-series 0` disables them. History keeps 300 samples by
  default, 1,440 at most.
- Everything stays in memory; use Prometheus or another scraper for durable
  retention.
- The server binds `127.0.0.1` without authentication; put it behind an
  authenticated proxy before exposing it. Exported values keep FakeGPU-SMI's
  modeled-versus-observed semantics.

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

In native hybrid mode, the CUDA runtime and cuBLAS stay native. If
`CUDA_VISIBLE_DEVICES` is set (by index or UUID), physical NVML discovery is
kept so frameworks resolve that selection; otherwise the modeled NVML
inventory is used. `fakegpu.env()` removes stale FakeGPU preloads when a child
environment switches modes.

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

The report separates weights, prefill/decode transients, shared and private KV
segments, runtime overhead, and usable headroom. Use `--device-memory-gib`
instead of `--target-profile` for a custom capacity. Prefix sharing needs a
dynamic or paged cache, and capacity search never exceeds `--max-batch-size`.

#### Match vLLM runtime reservation

vLLM preallocates a paged KV pool instead of growing it with active requests.
Select its memory policy when deploying on vLLM:

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

- Automatic sizing follows vLLM V1: requested executor memory minus profiled
  non-KV memory, rounded down to whole blocks. The report shows
  `num_gpu_blocks`, cache token capacity, initialization fit, and concurrency
  at `max_model_len`, separate from logical request KV demand.
- `--vllm-non-kv-cache-memory-bytes` replaces the modeled non-KV value with a
  vLLM startup profile (weights + peak activations + non-torch + CUDA graphs).
- `--vllm-kv-cache-memory-bytes` sets an exact per-GPU cache size and, like
  vLLM, ignores `--memory-utilization`.
- `--memory-utilization` defaults to `0.92`, matching the
  [vLLM CacheConfig reference](https://docs.vllm.ai/en/latest/api/vllm/config/cache/);
  set it explicitly for version-locked reports.
- The budget follows vLLM's
  [GPU-worker profiling path](https://docs.vllm.ai/en/latest/api/vllm/v1/worker/gpu_worker/),
  but free memory, other processes, tensor/pipeline parallelism, hybrid cache
  groups, fragmentation, and version-specific kernels still need same-stack
  observation. Speculative decoding is available only with the generic
  runtime model.

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

Target and draft weights and both KV caches stay resident, with conservative
lookahead slots. The acceptance rate only changes expected output tokens per
target call, not peak memory or speedup. Vocabulary sizes must match;
tokenizer identity is not verified.

Every serving report emits a path-independent `workload_signature` covering
architecture, weight storage, request shapes, KV/prefill, and speculative
settings, but not model paths or device capacity. `calibrate verify` checks it
and `target.profile` separately, so evidence from another configuration or
GPU is rejected.

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

- Requests are evaluated in file order; each prefix group is stored once, and
  the longest fitting prefix of the list is admitted without reordering.
- The report lists admitted and rejected IDs, per-request transients,
  concurrent-prefill bounds, grouped KV segments, and headroom.
- Group members must share the same positive `shared_prefix_tokens`. Named
  groups are treated as resident cache hits.
- Arrival timing, eviction, preemption, tensor parallelism, adaptive
  speculation, KV rollback, throughput, and latency are unmodeled. Reports
  stay `Modeled` and `uncalibrated` until a matching observation is supplied.

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

LLM estimator:

- Reads safetensors headers only. `--kv-cache-strategy` accepts `dynamic`,
  `static`, `quantized`, or `paged`; the report separates logical storage,
  quantization savings, static reservation, paged overhead, and sliding-window
  limits.
- Quantized caches keep 128 recent tokens at the compute dtype by default
  (`--kv-cache-residual-tokens`).
- MHA, GQA, and MQA use separate key/value storage. Configs with
  `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, and `v_head_dim` are
  treated as MLA (compressed latent + decoupled RoPE); MLA backend workspaces
  need calibration.

Diffusion estimator:

- Separates text encoding, repeated denoising, and VAE decode. `--model-dir`
  reads only `model_index.json`, component `config.json` files, and selected
  safetensors headers, without remote code or tensor payloads.
- Checkpoint storage and runtime weight bytes are reported separately. Local
  Euler/PNDM scheduler configs set denoiser evaluation counts; other
  schedulers report `denoiser_evaluations: null`.
- UNets, cross-attention transformers (DiT/PixArt), and SD3/Flux-style
  joint-attention transformers use distinct activation formulas. CFG doubles
  the denoiser batch only where both branches exist.
- `--weight-variant fp16|bf16|fp32` selects a checkpoint family. Built-in
  profiles include `pixart-sigma-xl-2-1024-ms` and two Stable Diffusion UNets.
- `--offload model`, `--attention-slicing`, `--vae-slicing`, and
  `--vae-tiling` model common Diffusers trade-offs. Without a matching
  real-GPU comparison, reports stay `Modeled` and `uncalibrated` with no error
  percentage or prediction interval.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Command reference

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
`python3 -m fakegpu <command> --help` for command-specific options. Large
reports and plans can use `.json.gz` wherever JSON paths are accepted;
compressed output is written atomically and reproducibly.

### Fine-tuning time and VRAM

`estimate-finetune` estimates single-GPU full, LoRA, and 4-bit QLoRA memory and
**training-loop** time from local model metadata and a Hugging Face
Trainer/PEFT, LLaMA-Factory, or Axolotl config, without importing those
frameworks or loading weights.

- Samples are processed training sequences after filtering, splitting, and
  packing; sequence length includes template tokens and padding.
- `max_steps` overrides epochs; `--max-steps -1` clears a configured limit.

```bash
python -m fakegpu estimate-finetune \
  --config examples/finetuning/huggingface-lora.yaml \
  --model /path/to/Qwen3.5-0.8B --text-only \
  --profile "$TARGET_GPU_PROFILE" --num-samples 10000 --epochs 3 \
  --sequence-length 128 --json estimate.json
```

- Config examples for all three frameworks are in
  [`examples/finetuning/`](examples/finetuning/). Adaptation is CPU-tested,
  not benchmarked for every framework/runtime combination.
- Distributed/offloaded execution, unsupported optimizers, and unresolved
  adapter targets fail explicitly; use `plan-training` for distributed plans.
- Qwen3.5 linear/full attention layers are counted separately; DeltaNet
  backward workspaces, fast kernels, and MoE dispatch are only partially
  modeled. `--text-only` selects the text backbone of a multimodal checkpoint.

`profile-finetune-data` scans every row of local JSONL files and records a
content fingerprint, token lengths, truncation, padding, and sequence counts.
Raw `text`/`messages` rows use a local tokenizer, and `--packing concatenate`
is a token-budget scenario, not a framework's packing algorithm. For counts
from a real training split, export its final `input_ids` rows (optionally with
`attention_mask` and `labels`) and pass `--processed`:

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

`--dataset-report` binds sample counts and the sequence cap: it replaces the
config's sample count, rejects conflicting overrides, and requires a matching
cap. `--source-framework` is a provenance label only. Packed training needs
final processed sequence counts.

`run-finetune` runs the installed native HF Trainer/PEFT, LLaMA-Factory, or
Axolotl pipeline on one physical CUDA GPU (full, LoRA, or bitsandbytes 4-bit
QLoRA; local unquantized text checkpoint; unpacked data unless noted below).
It performs the training in the complete native config (relative paths resolve
from the config directory) and records:

- native preprocessing, the effective dataset, actual batches, optimizer
  updates, allocator peaks, evaluation/checkpoint events, and final saving;
- `observation.updates[].batches`: each microbatch's shape, epoch position,
  token slots, nonpadding tokens, and supervised labels;
- `observation.batch_profile`: timings grouped by complete ordered
  accumulation window; a group is `sufficient` after two steady updates
  (sample availability, not accuracy);
- `observation.phase_events`: the actual evaluation/save call tree; only root
  intervals count toward phase totals, and post-save file inventories are not
  disk-write measurements.

Warmup and collection overhead are excluded from steady samples but included
in total wall time.

`validate-finetune-windows` compares timing predictions with a separately
recorded holdout run:

- It needs at least two independent calibration reports with the same model,
  data, software, and native settings.
- Every window needs two retained observations per calibration run, after
  excluding global warmup and the window's first occurrence; uncovered windows
  are rejected.
- It validates historical evidence only: no shuffle order, warmup, total wall
  time, or dynamic VRAM prediction. Native packing and MoE windows are
  unsupported.

```bash
python -m fakegpu validate-finetune-windows \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --holdout-run holdout.json --json window-validation.json
```

`validate-finetune-events` validates evaluation and save subtotals the same
way. Initial, first, repeated, and final events stay separate. It currently
supports unpacked dense text models, loss-only evaluation, and local
model/adapter or resumable checkpoint saves, and does not predict event counts
or total pipeline time.

```bash
python -m fakegpu validate-finetune-events \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --holdout-run holdout.json --json event-validation.json
```

`--event-context` also matches the latest evaluation and checkpoint boundaries
before each window (ordered calls, workloads, phases, and updates since each
boundary). Each window/context combination needs two samples per calibration
run. It validates recurring windows after periodic events, not first-save
recovery or complete training time.

```bash
python -m fakegpu validate-finetune-windows --event-context \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --holdout-run holdout.json --json event-window-validation.json
```

`--epoch-context` also matches each microbatch's position within its epoch, to
separate windows that include epoch-start loading or worker startup. Use it
alone or with `--event-context`:

```bash
python -m fakegpu validate-finetune-windows --event-context --epoch-context \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --holdout-run holdout.json --json epoch-window-validation.json
```

`plan-finetune-pipeline` projects future epochs or updates from a calibration
run's complete observed epoch. `estimate-finetune-pipeline` combines at least
two calibrations into a complete native budget: setup, warmup, ordered
training windows, evaluation, checkpoints, Trainer gaps, and finalization.

- The span ends after the native call and final CUDA sync; Python startup,
  report construction, and CLI output are excluded.
- Time ranges are extrema of per-process means, not confidence intervals.
  Missing components fail explicitly.
- Supported schedules are unpacked, sequential text datasets with retained
  tail batches: HF Trainer (Transformers 5.6.0) and LLaMA-Factory 0.9.5 with
  `disable_shuffling: true`; Axolotl 0.18.0 (Transformers 5.14.1) with
  `curriculum_sampling: true`.
- Dataset, model, hardware, and execution settings must match calibration; a
  different row count needs a new calibration. Logging must be off or every
  update; checkpoint rotation and best-model reload are unsupported.

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

For default random sampling or a changed dataset, use the empirical transfer
workflow. It plans CPU sampler/collator outputs from the native prepared
dataloader and verifies consumed row indices and input tensor hashes during
calibration. Supported stacks: HF/Factory with Accelerate 1.11.0 and Axolotl
with Accelerate 1.13.0. Random sampling uses the native data seed plus epoch
(Axolotl 0.18.0 does not forward its YAML `data_seed` to Trainer).

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

- `calibrate-finetune` runs exactly two fresh native processes with step/time
  limits and new empty kernel-cache directories by default. It keeps every
  attempt, seals reports with SHA256, supports `--resume`, and never silently
  extends or reruns a calibration.
- CPU-only rebudgeting needs no GPU or framework but cannot exceed the saved
  sampling horizon.
- Keep the workspace at its original absolute path and outside recorded input
  directories (nested workspaces are rejected). Only finite local datasets and
  POSIX lock semantics are supported. Prepared Axolotl/Factory datasets are
  isolated under the workspace's `prepared/` directory; resume rejects changed
  raw inputs.
- `--reuse-calibration-run PATH` (repeatable) reuses compatible training
  evidence from another dataset. Two fresh target-dataset calibrations still
  provide startup and finalization costs, which are never scaled by sample
  count. Training costs across different token/mask/supervision distributions
  are **approximate**; missing coverage exits 1 with a list of gaps.
- `run-finetune --planning-epochs N` (or `--planning-max-steps N`) records the
  CPU horizon directly. `plan-finetune-transfer`, `estimate-finetune-transfer`,
  and `validate-finetune-transfer` expose the same steps without the workspace
  manager. Streaming, custom samplers, and automatic batch-size changes are
  unsupported.

Memory results are the observed full-run allocated/reserved peak envelope on
the same GPU and allocator. They do not project to smaller cards; the optional
reserve is a user-chosen margin, not proof of OOM safety.
`validate-finetune-pipeline` reports timing errors and memory underestimation
against an independent holdout.

Length grouping: set `train_sampling_strategy: group_by_length` (HF/Factory) or
`group_by_length: true` (Axolotl) and pass `--length-grouping-seed 123` to
`calibrate-finetune` or `run-finetune`.

- It uses the native grouping algorithm with an independent CPU generator
  seeded by `123 + epoch`; native global-RNG grouping is not reproducible.
- A planning horizon is required. One finite evaluation dataset uses the same
  explicit seed without advancing the training sampler; custom evaluation
  samplers and prepared loaders are rejected.
- Sampler plan v2 binds the grouping lengths and batch size into calibration
  identity, so different length sequences need their own evidence.

Packing:

- The [packing semantics audit](tests/native/finetune_packing_semantics.py)
  shows that the tested ordinary HF/Factory hybrid paths leak across block
  masks and reset positions in tiny Qwen3.5 models, and that the tested
  Axolotl patch is incompatible with the Transformers 5.14.1 decoder interface.
- The opt-in [segment reference audit](tests/native/finetune_segment_reference.py)
  checks a project-owned Qwen3.5 interface that runs each linear-attention
  segment with empty convolution and recurrent state, matching independent
  native sequences including checkpoint recomputation and reload.
- Enable it with `--packing-seed 123` on `run-finetune` or
  `calibrate-finetune` for dense `qwen3_5_text` LoRA, with a planning horizon,
  native packing (`packing: true` plus `neat_packing: true` for Factory, or
  `sample_packing: true` for Axolotl), and evaluation disabled.
- HF processed rows also need `position_ids` and contiguous positive segment
  IDs in `attention_mask` (optional zero right padding). The first label of
  every later segment must be ignored.
- Sampler plan v3 records epoch bins, logical-sequence counts, tensor rows,
  block geometry, dtypes, and input hashes; different data needs its own
  calibration. Native packing reports require the transfer commands.
- No varlen-kernel throughput claim; full/QLoRA and grouped packing are not
  covered. See also the [PEFT direct-forward audit](tests/native/finetune_segment_peft.py)
  and the [packed-budget evidence](tests/data/finetune_packing_budget_evidence.json).

`run-finetune --cache-regime empty_kernel_cache` requires five distinct,
existing, empty directories set explicitly: `TRITON_CACHE_DIR`,
`TORCHINDUCTOR_CACHE_DIR`, `CUDA_CACHE_PATH`, `TORCH_EXTENSIONS_DIR`, and
`PYTORCH_KERNEL_CACHE_PATH`. `populated_kernel_cache` records each file's
relative path, size, and SHA256. The runner never creates, clears, or warms
these caches. Default `uncontrolled` reports make no cache claim and cannot be
mixed with controlled calibration. Triton metadata can reference the original
cache path, so keep those files unchanged.

Native run setup:

- Unpacked HF runs (`--framework huggingface`) use a flat config and processed
  JSONL rows with exactly `input_ids`, `attention_mask`, and `labels`:
  equal-length integer arrays, a binary mask, and 2 to `sequence_length`
  tokens. No implicit tokenization is applied and the tokenizer needs a pad
  token. See [the native LoRA example](examples/finetuning/huggingface-native-lora.yaml).
- Gradient checkpointing: LLaMA-Factory uses
  `disable_gradient_checkpointing: false` with `use_reentrant_gc: true` or
  `false`; Axolotl uses `gradient_checkpointing: true` with
  `gradient_checkpointing_kwargs: {use_reentrant: true}` (or `false`).
  Effective checkpointing and quantization settings are part of calibration
  identity.
- Native runs need a text checkpoint; `--text-only` does not convert weights.
  For local Qwen3.5, export a text checkpoint on CPU and create synthetic
  Alpaca data for integration checks:

```bash
python tests/native/prepare_finetune_fixture.py \
  --model /path/to/Qwen3.5-0.8B \
  --output-dir /tmp/fakegpu-finetune/qwen35-text \
  --dataset-dir /tmp/fakegpu-finetune/data
```

Export needs a Transformers version with `Qwen3_5ForCausalLM`. The native
examples target LLaMA-Factory 0.9.5 (Transformers 5.6.0) and Axolotl 0.18.0
(Transformers 5.14.1); install each with compatible CUDA PyTorch, PEFT, and
dataset dependencies in its own environment. Use a new `output_dir` for every
run; Axolotl's prepared cache lives in `/tmp/fakegpu-finetune/axolotl-prepared`.

```bash
CUDA_VISIBLE_DEVICES=1 python -m fakegpu run-finetune \
  --config examples/finetuning/axolotl-native-lora.yaml --framework axolotl \
  --profile "$TARGET_GPU_PROFILE" --warmup-steps 2 --output run.json
python -m fakegpu estimate-finetune \
  --run-report run.json --epochs 3 --max-steps -1 --json estimate.json
```

- For LLaMA-Factory, use `--framework llamafactory` with
  [`llamafactory-native-lora.yaml`](examples/finetuning/llamafactory-native-lora.yaml).
- Reuse `--run-report` in the same Python environment; the CLI rechecks local
  weights, model config, and software identity, and defaults to the report's
  GPU profile.
- This single-report `estimate-finetune` interface needs fixed full batches.
  A different `--num-samples` gives a training-loop scenario without
  preprocessing or loading costs; Python startup and downloads are excluded.

For 512-token NF4 QLoRA with checkpointing, use
[`llamafactory-native-qlora.yaml`](examples/finetuning/llamafactory-native-qlora.yaml)
or [`axolotl-native-qlora.yaml`](examples/finetuning/axolotl-native-qlora.yaml)
(double quantization, BF16, requires bitsandbytes). Create the longer dataset
from the exported tokenizer:

```bash
python tests/native/prepare_finetune_fixture.py \
  --model /tmp/fakegpu-finetune/qwen35-text --dataset-only \
  --dataset-dir /tmp/fakegpu-finetune/data-long \
  --sequence-length 1024 --num-samples 64
```

The fixture length is a lower bound; each framework applies its configured
512- or 1024-token cutoff, and the run report records the sequences used.

Collect independent calibration and holdout processes with
[`tests/native/finetune_framework_matrix.py`](tests/native/finetune_framework_matrix.py).
It takes sequence length, microbatch size, accumulation, and seed from the
native config:

```bash
python tests/native/finetune_framework_matrix.py \
  --framework axolotl --config examples/finetuning/axolotl-native-qlora.yaml \
  --python /path/to/axolotl/venv/bin/python \
  --output-dir /tmp/fakegpu-finetune/axolotl-qlora-matrix \
  --gpu-index 1 --calibration-steps 6 --warmup-steps 2 \
  --holdout-runs 2 --holdout-steps 16
```

Step counts include warmup (defaults: 6 calibration updates, 2 warmup updates,
three 32-update holdouts). Every run uses a fresh output directory, and
predictions use the production estimator with actual microstep counts.

For a measured estimate, run a short real-GPU probe and reuse its observation
in the **same Python environment**, with the same model, GPU, precision,
adapter, attention implementation, batch, sequence length, and accumulation:

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

The probe runs Hugging Face Trainer on fixed-length synthetic text and needs
Transformers, Accelerate, and CUDA PyTorch, plus PEFT for adapters and
bitsandbytes for QLoRA. Warmup is recorded separately and both startup and
steady peaks are kept. Dataset preparation, model loading, evaluation, and
checkpoint I/O are excluded from projected time. Checkpoint files are hashed,
so replacing weights invalidates the calibration.

Validation records (exact workloads, versions, and limits are in
[`refactor.md`](refactor.md)):

- [Complete pipeline validation](tests/data/finetune_pipeline_validation_evidence.json):
  30 native processes across all three frameworks, including allocator OOM
  controls.
- [Sampler and transfer validation](tests/data/finetune_transfer_validation_evidence.json):
  18 Qwen3.5-0.8B processes on Blackwell GPU, changing 25 to 73 training rows
  plus random sampling with periodic eval/save; one holdout per condition.
- [Qwen3.5-2B record](tests/data/finetune_qwen2b_evidence.json): 512/1024-token
  LoRA and 512-token NF4 QLoRA with checkpointing, two calibrations and two
  holdouts per condition; total-time errors 0.054%–1.578%.
- [Native validation archive](tests/data/finetune_framework_validation_evidence.json):
  earlier 128-token LoRA runs; maximum wall-time errors 3.931% (Factory) and
  1.125% (cached Axolotl). Reusing Axolotl's first-use calibration on later
  cached runs gave a 364.885% error (first update: 66.542 s), so cache/JIT
  costs are not reusable from a model/software signature alone.
- [Extended validation archive](tests/data/finetune_extended_validation_evidence.json):
  six configurations at 512/1024 tokens and batch 1/2, including checkpointing
  and NF4 QLoRA; 18 measured processes.
- [Qwen3.5-0.8B probe record](tests/data/finetune_validation_evidence.json):
  full, LoRA, and QLoRA calibration/holdout runs, reproducible with
  [`tests/native/finetune_matrix.py`](tests/native/finetune_matrix.py).

These short runs do not establish general accuracy or OOM safety.

### Online serving benchmark

`benchmark-serving` measures an already running completion endpoint using
vLLM's `ignore_eos` and final streaming usage fields. A token-ID manifest sets
arrivals and output limits; the report keeps SSE events, failures, client queue
time, and first nonempty text latency. It does not retry requests or infer
device phase peaks. Use your server's served model name.

```bash
python -m fakegpu benchmark-serving \
  --base-url http://127.0.0.1:8000/v1 --model local-model \
  --requests examples/serving/online-requests.json \
  --concurrency 4 --repetitions 3 --json build/online.json.gz
```

The [native serving record](tests/data/serving_native_evidence.json) covers 20
8K/32K Transformers processes and 72 mixed vLLM requests with prefix caching
off/on. At 32K, allocated phase errors were 2.02%/7.34%, but reserved decode
was underestimated by 24.07%: tensor-storage estimates do not establish
allocator or OOM safety.

### Diffusion training

Cached-feature UNet training supports full tuning, LoRA, and optional
`--ema-decay`. `plan-diffusion-training` hashes the local weights and FP32
cached tensors; `run-diffusion-training` uses one CUDA GPU with BF16 autocast,
FP32 trainable parameters, and AdamW. VAE/text-encoder training and raw-image
preparation are excluded. The fixture below generates random features for
integration tests:

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

- Activation and reserved peaks are unknown without native observations.
- Budgets need two independent processes and repeated samples for every future
  window; the first use of a shape is charged separately.
- `--cache-regime` behaves as in the fine-tuning workflow (default
  `uncontrolled`). Allocator, GPU UUID, software, inputs, and execution
  identity must match.
- `validate-diffusion-training` compares an independent `--holdout-run`;
  `--max-total-time-error-percent` and `--max-memory-underestimate-percent`
  exit 1 when exceeded.

The [native diffusion record](tests/data/diffusion_native_evidence.json) has
SD/SDXL generation observations and twelve training processes. Six training
holdouts show 0.044%–3.185% total-time error with no allocator-envelope
underestimate; generation VAE allocated peaks were underestimated by
12.0%–13.8% and reserved peaks by up to 31.6%.

### Phase memory calibration

Allocated tensor storage and allocator-reserved memory are tracked separately.
`calibrate phase-sample` collects both in one fresh CUDA process,
`phase-estimate` seals per-phase maxima from at least five processes, and
`phase-validate` checks a later independent run. Use a new output path for
every sample and budget.

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

- Sampling supports homogeneous Transformers dynamic-cache serving and local
  SD/SDXL with eager or SDPA attention, attention slicing, VAE slicing, VAE
  tiling, and their combinations, without offload.
- For tiling, `--vae-tile-size` sets both native sample and latent tile
  limits. Effective processors, VAE flags, and tile limits are part of the
  native identity.
- The plan, selected weights and config, GPU UUID, Torch/CUDA/cuDNN versions,
  allocator, and implementation must match. No shape transfer or hidden
  safety multiplier is applied; kernel caches are uncontrolled.
- Default limits are 20% absolute error and 10% underestimation, exiting 1 on
  failure. Offline estimation and validation need only the standard library.

Evidence (experimental; limited to the tested GPU, software, and fixed inputs):

- [Phase-memory validation](tests/data/phase_memory_evidence.json): after
  adding the VAE upsampling convolution, four resolution cohorts show
  0.045%–1.172% VAE allocated error. Six conditions with five calibrations and
  five holdouts each (60 processes) pass both gates.
- [Broader memory validation](tests/data/memory_generalization_evidence.json):
  25 single-GPU conditions in 74 processes across SD 1.4/SDXL,
  Qwen3-0.6B/1.7B/Qwen2.5-0.5B, FP16/BF16/FP32, batches, rectangular images,
  guidance, eager attention, slicing, and tiling. Worst allocated errors are
  10.93% absolute and 3.58% underestimate (gates: 20%/10%). Exact-condition
  budgets match all twelve holdouts; 68 cross-condition reuses are rejected.
  These are not `GPU-validated` rows: 21 conditions have fewer than five
  observations, and optimized modes used an archived native probe rather
  than the public CLI.
- [Public optimized-mode validation](tests/data/optimized_phase_memory_evidence.json):
  seven SD/SDXL conditions in 70 processes; all 210 holdout comparisons match
  their calibration maxima, with analytical allocated error at most 6.69%
  absolute and 3.38% underestimate. 42 cross-mode and 21 effective-mode
  identity mismatches are rejected.

None of these establish transfer to new shapes, prediction intervals, or
physical OOM safety.

### Distributed plans and experiment sweeps

For FSDP2 `full_shard`, `plan-training --parameter-shapes layout.json` accepts
`{"shapes": [[12, 8], [12], [3, 12], [3]]}` plus the matching total
`--parameter-bytes` and uniform `--parameter-element-bytes`. It models native
per-parameter dim-0 chunks and budgets the largest rank. Without shapes, the
balanced-shard approximation is used; custom placements are not modeled.

The [versioned native validation matrix](tests/data/research_native_evidence.json)
records six FP32/FP16/BF16 LoRA/attention CPU_SIM cases, tiny Qwen3 MoE
routing and gradients on CPU and GPU, and two CPU ranks running native
DDP/FSDP2. Physical TP/PP/CP/EP, multi-GPU ZeRO/FSDP, and multi-node
collectives are out of scope. Reproduce with `scripts/test.sh cpu` and
`tests/native/research_distributed.py --help`.

Finite experiment sweeps use an explicit argv array, parameter grid, and
repetition count. Runs are sequential, bound to stable run IDs, and keep
stdout, exit status, errors, artifact hashes, and numeric metrics:

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

`--resume` skips verified successes; `--resume --retry-failed` retries
failures and interruptions. Changed inputs, sources, or environments need a new
workspace. Commands must stay in the foreground, and metric paths support
nested JSON. Plotting needs Matplotlib; bars show observed min/max, not
confidence intervals. The example measures CPU work, not GPU throughput.

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

The [release regression record](tests/data/research_release_evidence.json)
covers 1734 passing Python tests (one CPU-only-build branch skipped), the
native CPU suite, 104 packaged Python modules, 42 command help checks, and 68
offline CLI checks; all 152 historical predictions remain byte-identical.

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
- Native cuBLAS algorithms without numeric implementations, unsupported
  GEMM/Lt dtypes, layouts, or pointer modes, and skipped CPU computation follow
  `FAKEGPU_UNSUPPORTED_API`: `warn` records a warning, and `error` returns
  `CUBLAS_STATUS_NOT_SUPPORTED` before writing output.
- For models with a vision encoder, `estimate-llm` and `plan-serving` count
  checkpoint weights but not vision/projector activations or FLOPs (flagged in
  reports). Decoder token counts must already include visual tokens.
- Diffusion profiles model fixed reference pipelines. Custom ControlNet,
  IP-Adapter, LoRA, safety-checker, refiner, video, and DiT components need
  extra metadata and real-GPU validation.
- Online-serving plans do not model request arrival distributions, cache
  eviction, preemption or reordering, self-speculative/Medusa/EAGLE variants,
  tokenizer remapping, tensor-parallel execution, throughput, or latency.
- Roofline output is an analytical interval, not measured kernel latency.
- Distributed timing includes coordinator work, memory copies, sockets, and
  process scheduling; it is not an NCCL, NVLink, or RDMA benchmark.
- Modeled fault events are report-only inputs; they do not alter CUDA
  execution or represent real ECC/Xid data.
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
[validation-static]: https://github.com/FanBB2333/FakeGPU/blob/6efed08ce20f9628481328ef31f90195df78c7fb/docs/ai-researcher-preflight.md#4-static-aten-storage-liveness-validation
[validation-inference]: https://github.com/FanBB2333/FakeGPU/blob/6efed08ce20f9628481328ef31f90195df78c7fb/docs/llm-inference-estimation.md#maintained-qwen3-8b-result
[validation-sft]: https://github.com/FanBB2333/FakeGPU/blob/6efed08ce20f9628481328ef31f90195df78c7fb/docs/llm-sft-memory-estimation.md#blackwell-gpu-matrix
[validation-qlora]: https://github.com/FanBB2333/FakeGPU/blob/6efed08ce20f9628481328ef31f90195df78c7fb/docs/llm-sft-memory-estimation.md#native-nf4-qlora-matrices
[diffusion-sd15]: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/451f4fe16113bff5a5d2269ed5ad43b0592e9a14
[diffusion-sdxl]: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/462165984030d82259a11f4367a4eed129e94a7b
[diffusion-pixart]: https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS/tree/e102b3591cc82e97071b8b4cb90d834d0c487207
