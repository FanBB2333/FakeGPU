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
   - [How it works](#how-it-works)
   - [Built with](#built-with)
2. [Getting started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [Verify the installation](#verify-the-installation)
3. [Usage](#usage)
   - [Run PyTorch code on FakeCUDA](#run-pytorch-code-on-fakecuda)
   - [Inspect FakeGPU devices and processes](#inspect-fakegpu-devices-and-processes)
   - [Intercept native CUDA libraries](#intercept-native-cuda-libraries)
   - [Check memory before a run](#check-memory-before-a-run)
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
| Where are the GPU-only entry points and dependencies in a repository? | Repository analyzer | No |
| What does a distributed training configuration imply for rank-local memory? | Training planner | No |
| Where do compute, communication, wait, and memory overlap in a trace? | Trace replay | No |
| How does an estimate compare with an actual CUDA run? | Passthrough or hybrid calibration | Yes |

<a id="use-cases"></a>

### Typical use cases

| When this is useful | What FakeGPU provides | Start with |
|---|---|---|
| Choosing a GPU before renting capacity or starting a long job | Profile-aware checkpoint, KV-cache, activation, optimizer, and workspace estimates | `estimate-llm`, `preflight` |
| Developing CUDA-oriented PyTorch code on a laptop or CPU-only CI runner | CUDA-visible control-flow checks while maintained tensor operations execute on CPU | `fakegpu.init(...)`, `demo`, `validate` |
| Comparing full fine-tuning, LoRA, QLoRA, checkpointing, offload, or sharding plans | Phase-aware and rank-local memory estimates before allocating a cluster | `plan-training`, Python memory estimator |
| Reviewing an unfamiliar GPU repository or native extension | GPU entry-point, dependency, kernel, and unsupported-API inventory | `analyze-repo`, `analyze-kernel`, `capabilities` |
| Designing or debugging a distributed workflow | Collective routing, link contention, rank waits, memory timelines, and TCP payload validation | `simulate-topology`, `replay-trace`, `bandwidth` |
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

### LLM reliability report

FakeGPU reports reliability per workload and environment signature. A
`GPU-validated` result applies only to the recorded model revision, shapes,
dtype, attention backend, allocator, software stack, and GPU. `CPU-validated`
means that maintained execution or analysis behavior passed without a physical
GPU. `Modeled` identifies analytical coverage without matching real-GPU
evidence, while `Planned` is not yet a supported accuracy claim.

#### Current repository verification

This repository state was verified on 2026-07-26 with `scripts/test.sh all`
and both declarative validation manifests on macOS 26.5 arm64, Python 3.11.9,
and PyTorch 2.9.1 CPU:

| Validation layer | Maintained check | Result |
|---|---|---|
| Python runtime, estimators, CLIs, schemas, and README contracts | Complete `pytest` suite | **161 passed** |
| Declarative validation matrices | 6 smoke executions plus 8 LLM cache, training-plan, and calibration executions | **14 passed** |
| Native interception | Build, library boundaries, exports, preload, memory types, coordinator, and unsupported-API policy | **Passed** |
| Native capability inventory | 5 groups, 26 explicit APIs, 24 policy-enforced APIs | **Passed** |
| GPU profile catalog | 82 profiles across 15 compute capabilities | **Passed** |
| CPU numerical simulation | GEMM, cuBLASLt, batched GEMM, BLAS1/2, and FP16: 8 maintained test groups | **Passed** |
| CUDA-enabled PyTorch native matmul | Requires a CUDA build of PyTorch | **Not run on this CPU-only host** |

GitHub CI separately runs the Python suite on Python 3.10–3.12 and the native
smoke and CPU simulation suites on Linux and macOS. The real-GPU results above
come from their linked immutable validation snapshots; they were checked
against the structured evidence and formulas but were not regenerated on this
CPU-only host.

#### Maintained LLM workload matrix

| Workload class | Covered variations | Evidence | Status |
|---|---|---|---|
| Offline decoder inference | Qwen3-8B, BF16, SDPA, model load, prefill, and decode peak | RTX PRO 5000 prediction-versus-observation data | `GPU-validated` |
| Full and adapter SFT | Qwen 0.8B/2B full fine-tuning and LoRA | Ten RTX PRO 5000 training cases | `GPU-validated` |
| Quantized adapter SFT | Qwen 0.8B/2B native NF4 QLoRA | Ten RTX PRO 5000 training cases | `GPU-validated` |
| General decoder analysis | Dense and MoE metadata, adapters, quantized checkpoints, eager/SDPA attention, KV cache, and expert-parallel traffic | Formula, fixture, and CLI regression tests | `CPU-validated` + `Modeled` |
| Distributed training plans | DeepSpeed, Accelerate, FSDP/FSDP2, sharding, checkpointing, and CPU/NVMe offload | Configuration, byte-accounting, topology, and trace tests | `CPU-validated` + `Modeled` |
| KV-cache allocation | Dynamic growth, static reservation, 2/4/8-bit quantized storage, paged block rounding, and sliding-window limits | Formula, API, `--kv-cache-strategy` CLI, and `tests/data/llm_validation.yaml` matrix tests | `CPU-validated` + `Modeled` |
| Online serving scheduling | continuous batching, chunked prefill, prefix caching, and speculative decoding | No maintained real-GPU evidence yet | `Planned` |
| Multi-GPU LLM execution | TP, PP, CP, EP, MoE imbalance, and combined FSDP/ZeRO execution | Analytical topology and coordinator coverage only | `Modeled` |

The cache formulas follow the workload shapes exposed by
[Transformers cache strategies](https://huggingface.co/docs/transformers/kv_cache).
Online serving schedules remain planned and follow
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

UUIDs and PCI bus IDs are stable simulated identifiers. Temperature, fan
speed, live power draw, and hardware GPU utilization remain `N/A` because the
CPU-backed runtime cannot observe them; profile power and clock values are
shown separately as static specifications.

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
| `fakegpu estimate-roofline` | Produce a profile-aware analytical latency interval |
| `fakegpu plan-training` | Normalize distributed training configs and estimate rank memory |
| `fakegpu simulate-topology` | Model collective routes and link contention |
| `fakegpu replay-trace` | Summarize compute, communication, wait, and memory timelines |
| `fakegpu calibrate` | Compare memory reports and enforce reliability gates |
| `fakegpu capabilities` | List or strictly audit native API classifications |
| `fakegpu nvidia-smi` | Inspect FakeGPU devices, profiles, runtime state, allocator memory, and processes |
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
| `smoke` | Native library loading, reports, capabilities, and coordinator |
| `cpu` | CPU-backed cuBLAS simulation |
| `all` | All maintained suites |

Run a declarative validation manifest directly when needed:

```bash
python3 -m fakegpu validate \
  --manifest tests/data/validation_smoke.yaml \
  --report-dir build/validation-smoke \
  --strict

python3 -m fakegpu validate \
  --manifest tests/data/llm_validation.yaml \
  --report-dir build/validation-llm \
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
- Roofline output is an analytical interval, not measured kernel latency.
- Distributed timing includes coordinator work, memory copies, sockets, and
  process scheduling; it is not an NCCL, NVLink, or RDMA benchmark.
- Hybrid and passthrough modes require a compatible physical CUDA stack.
- macOS System Integrity Protection can remove `DYLD_*` variables from system
  binaries. Prefer a Homebrew, conda, or pyenv Python for native interception.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Roadmap

- [x] CPU-backed PyTorch FakeCUDA runtime
- [x] Native CUDA, NVML, cuBLAS, and NCCL interception
- [x] Architecture-aware GPU profile catalog
- [x] Runtime, static, LLM, and distributed memory analysis
- [x] Repository, kernel, topology, and trace analysis
- [x] Detailed FakeGPU-SMI device, runtime, allocator, and process queries
- [ ] Expand executable native CUDA operations and cuBLAS coverage
- [x] Publish live native-runtime state and activity through FakeGPU-SMI
- [ ] Add topology, MIG, NVLink, and fault views
- [ ] Export historical device and process metrics to monitoring systems
- [ ] Add real-GPU LLM validation for long-context and online-serving workloads
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
