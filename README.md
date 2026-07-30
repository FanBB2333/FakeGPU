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

### What FakeGPU answers

| Question | Recommended path | Physical GPU |
|---|---|---:|
| Does PyTorch code follow the expected CUDA-facing control flow? | Python FakeCUDA runtime | No |
| Can an unmodified process load and call CUDA-family shared libraries? | Native interception | No |
| Will a selected GPU profile fit a workload? | Preflight or static memory estimator | No |
| How much checkpoint, KV-cache, adapter, or MoE memory should an LLM use? | LLM estimator | No |
| Which homogeneous or mixed-length online LLM requests fit under a memory budget? | Serving planner | No |
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
| Sizing chat, RAG, completion, or summarization traffic before deployment | Per-request prompt/generation lengths, continuous-batch admission, chunked-prefill transients, shared-prefix KV groups, and explicit memory headroom | `plan-serving`, `validate` |
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
| [Controlled ATen MLP and Transformer grid with backend-resident calibration][validation-static] | RTX 3090 Ti and RTX PRO 5000; PyTorch/CUDA 2.12/13.0 and 2.9/12.8 | 13 workloads, 26 observations | **0.08% maximum** | **≥99.92%** |
| [Qwen3-8B BF16 SDPA inference][validation-inference] | RTX PRO 5000; PyTorch 2.9.1/CUDA 12.8 | Model load and inference peak | 0.0129% load; **0.0672% peak** | 99.9871%; **99.9328%** |
| [Qwen 0.8B/2B full and LoRA SFT][validation-sft] | RTX PRO 5000; PyTorch 2.8/CUDA 12.8 | 10 training cases | **0.102%–1.921%** | **98.079%–99.898%** |
| [Qwen 0.8B/2B native NF4 QLoRA][validation-qlora] | RTX PRO 5000; PyTorch 2.8/CUDA 12.8 | 10 quantized training cases | **0.628%–1.732%** | **98.268%–99.372%** |

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
does not produce a real-GPU accuracy measurement. To compare compatible
prediction and observation reports for your own workload:

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

#### Current repository verification

This repository state was verified on 2026-07-31 with `scripts/test.sh all`
and both declarative validation manifests on macOS 26.5 arm64, Python 3.11.9,
and PyTorch 2.9.1 CPU:

| Validation layer | Maintained check | Result |
|---|---|---|
| Python runtime, estimators, CLIs, schemas, and README contracts | Complete `pytest` suite | **195 passed** |
| Declarative validation matrices | 6 smoke executions plus 28 research cache, serving, training, calibration, and diffusion executions | **34 passed** |
| Native interception | Build, library boundaries, exports, preload, memory types, coordinator, and unsupported-API policy | **Passed** |
| FakeGPU-SMI diagnostics | Bounded state, topology/NVLink/MIG views, NVML peer/MIG queries, health fields, and event reporting | **Passed** |
| Monitoring exporter | Prometheus/JSON snapshots, bounded history/cardinality, malformed-state degradation, and HTTP endpoints | **Passed** |
| Native capability inventory | 5 groups, 26 explicit APIs, 24 policy-enforced APIs | **Passed** |
| GPU profile catalog | 82 profiles across 15 compute capabilities | **Passed** |
| CPU numerical simulation | GEMM, cuBLASLt, batched GEMM, BLAS1/2, and FP16: 8 maintained test groups | **Passed** |
| CUDA-enabled PyTorch native matmul | Requires a CUDA build of PyTorch | **Not run on this CPU-only host** |

GitHub CI separately runs the Python suite on Python 3.10–3.12 and the native
smoke and CPU simulation suites on Linux and macOS. The real-GPU results above
come from their linked immutable validation snapshots; they were checked
against the structured evidence and formulas but were not regenerated on this
CPU-only host.

#### Maintained research workload matrix

| Workload class | Covered variations | Evidence | Status |
|---|---|---|---|
| Offline decoder inference | Qwen3-8B, BF16, SDPA, model load, prefill, and decode peak | RTX PRO 5000 prediction-versus-observation data | `GPU-validated` |
| Full and adapter SFT | Qwen 0.8B/2B full fine-tuning and LoRA | Ten RTX PRO 5000 training cases | `GPU-validated` |
| Quantized adapter SFT | Qwen 0.8B/2B native NF4 QLoRA | Ten RTX PRO 5000 training cases | `GPU-validated` |
| General decoder analysis | Dense and MoE metadata, adapters, quantized checkpoints, eager/SDPA attention, KV cache, and expert-parallel traffic | Formula, fixture, and CLI regression tests | `CPU-validated` + `Modeled` |
| Distributed training plans | DeepSpeed, Accelerate, FSDP/FSDP2, sharding, checkpointing, and CPU/NVMe offload | Configuration, byte-accounting, topology, and trace tests | `CPU-validated` + `Modeled` |
| KV-cache allocation | Dynamic growth, static reservation, 2/4/8-bit quantized storage, paged block rounding, and sliding-window limits | Formula, API, `--kv-cache-strategy` CLI, and `tests/data/research_validation.yaml` matrix tests | `CPU-validated` + `Modeled` |
| Online serving capacity | Homogeneous and mixed-length request sets under continuous batching, ordered admission, chunked prefill, grouped prefix caching, paged KV blocks, profile or explicit capacity, and admission headroom | Checkpoint headers, architecture formulas, `plan-serving`, unit tests, and chat/RAG cases in `tests/data/research_validation.yaml` | `CPU-validated` + `Modeled` |
| Diffusion image generation | Stable Diffusion v1.5, SDXL Base, and PixArt-Sigma; local UNets, cross-attention DiTs, and SD3/Flux-style joint-attention transformers; CFG, offload, attention/VAE slicing, and VAE tiling | Fixed-revision and local component headers, architecture configs, phase formulas, CLI/unit tests, and `tests/data/research_validation.yaml` | `CPU-validated` + `Modeled` |
| speculative decoding | Draft-model weights, draft KV cache, acceptance rate, and target verification batches | No dedicated estimator or maintained real-GPU evidence yet | `Planned` |
| Multi-GPU LLM execution | TP, PP, CP, EP, MoE imbalance, and combined FSDP/ZeRO execution | Analytical topology and coordinator coverage only | `Modeled` |
| Diffusion training | Optimizer, gradient, activation, EMA, and parameter-efficient tuning | No dedicated estimator or real-GPU evidence yet | `Planned` |

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
| Online serving prefix cache | Same decoder, 8 active sequences, 4K prompt + 256 generated tokens, paged BF16 KV; independent caches → one shared 1K prefix | **4.25 → 3.38 GiB** (−20.6%) | Shared/private paged-block formula |
| Mixed chat/completion serving | Same decoder; prompts 4K/8K/2K and generation 256/512/1024; independent KV → two chat requests sharing a 1K system prefix | **1.97 → 1.84 GiB** decode KV (−6.3%) | Heterogeneous request-set formula and chat/RAG matrix |
| LLM training | 8B BF16 parameters, AdamW, 4 GPUs, checkpointed activations; replicated → full shard | **104.31 → 26.54 GiB** per rank (−74.6%) | Training-plan phase model |
| Stable Diffusion v1.5 generation | 512², batch 1, FP16, CFG; all weights resident → model offload | **2.30 → 1.65 GiB** (−28.1%) | Component + phase model |
| SDXL batch generation | 1024², batch 4, FP16, CFG; regular VAE decode → VAE slicing | **11.49 → 7.74 GiB** (−32.7%) | Sequential VAE batch-shape model |
| SDXL high-resolution generation | 2048², batch 1, FP16, CFG; full-frame VAE → 512² VAE tiles | **11.49 → 7.27 GiB** (−36.7%) | Sequential VAE tile-shape model |
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
request sets. Request arrival distributions, cache eviction, preemption,
throughput, latency, and speculative decoding remain outside the supported
estimate. The workload terminology follows
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
```

The detailed report includes the FakeGPU version, runtime backend and policies,
Python/PyTorch/CUDA versions, state freshness, profile catalog and native API
coverage, synthetic device identity, compute properties, memory categories,
allocator activity, dispatch tracking, and per-process peaks. Use `-i` with an
index, UUID, PCI bus ID, or profile ID to select devices; use `--json` for the
complete normalized inventory. State schema v2 is emitted, while v1 files
remain readable.

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
speculative draft models, throughput, and latency are listed as unmodeled.
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

The diffusion estimator separates text encoding, repeated denoising, and VAE
decode phases. With `--model-dir`, it reads only `model_index.json`, component
`config.json` files, and selected safetensors headers. It does not import
remote custom code or read tensor payloads. Checkpoint storage and runtime
weight bytes at the requested dtype are reported separately.
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

| Command | Purpose |
|---|---|
| `fakegpu doctor` | Check the installation, native libraries, PyTorch, and profiles |
| `fakegpu demo` | Run a small CPU-backed, CUDA-visible training step |
| `fakegpu preflight` | Execute a workload to a target stage and classify fit or OOM |
| `fakegpu analyze-repo` | Inventory repository entry points and GPU-only risks |
| `fakegpu analyze-kernel` | Inspect CUDA, PTX, and SASS resources and operations |
| `fakegpu estimate-llm` | Estimate decoder memory, communication, and FLOPs |
| `fakegpu plan-serving` | Plan homogeneous or mixed-request admission, chunked prefill, and grouped shared-prefix KV storage |
| `fakegpu estimate-diffusion` | Estimate diffusion text-encode, denoise, and VAE-decode memory phases |
| `fakegpu estimate-roofline` | Produce a profile-aware analytical latency interval |
| `fakegpu plan-training` | Normalize distributed training configs and estimate rank memory |
| `fakegpu simulate-topology` | Model collective routes and link contention |
| `fakegpu replay-trace` | Summarize compute, communication, wait, and memory timelines |
| `fakegpu calibrate` | Compare memory reports and enforce reliability gates |
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
- Diffusion profiles model fixed reference pipelines. Custom ControlNet,
  IP-Adapter, LoRA, safety-checker, refiner, video, and DiT components require
  additional component metadata and matching real-GPU validation.
- Online-serving plans accept explicit mixed lengths but do not model request
  arrival distributions, cache eviction, preemption or scheduler reordering,
  speculative decoding, tensor-parallel execution, throughput, or latency.
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
- [ ] Add real-GPU LLM validation for long-context and online-serving workloads
- [ ] Add real-GPU diffusion validation plus training and DiT memory models
- [ ] Validate distributed and MoE estimates across more GPU and software stacks

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
[validation-sft]: https://github.com/FanBB2333/FakeGPU/blob/df254c21eebc0a5bbf13992f3f5a8e995cfa8708/docs/llm-sft-memory-estimation.md#rtx-pro-5000-matrix
[validation-qlora]: https://github.com/FanBB2333/FakeGPU/blob/df254c21eebc0a5bbf13992f3f5a8e995cfa8708/docs/llm-sft-memory-estimation.md#native-nf4-qlora-matrices
[diffusion-sd15]: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/451f4fe16113bff5a5d2269ed5ad43b0592e9a14
[diffusion-sdxl]: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/462165984030d82259a11f4367a4eed129e94a7b
[diffusion-pixart]: https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS/tree/e102b3591cc82e97071b8b4cb90d834d0c487207
