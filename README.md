<a id="readme-top"></a>

<div align="center">

# FakeGPU

**Validate CUDA-facing applications, estimate GPU memory, and simulate distributed GPU workflows without a production GPU cluster.**

[![Test][test-shield]][test-url]
[![Release][release-shield]][release-url]
[![Python][python-shield]][python-url]
[![License][license-shield]][license-url]

[English](README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md)

[Explore the docs](https://fanbb2333.github.io/FakeGPU/) ·
[Report a bug](https://github.com/FanBB2333/FakeGPU/issues/new?labels=bug) ·
[Request a feature](https://github.com/FanBB2333/FakeGPU/issues/new?labels=enhancement)

</div>

![FakeGPU workflows: CPU-backed PyTorch execution, GPU profile switching, memory preflight, and static workload estimation](docs/assets/readme/tldr-workflows.png)

> [!IMPORTANT]
> FakeGPU is a development, testing, and capacity-planning tool. It does not
> provide numerical or performance parity for arbitrary CUDA kernels, and its
> TCP benchmark is not an NCCL, NVLink, or RDMA performance predictor.

## Table of contents

1. [About the project](#about-the-project)
   - [What FakeGPU answers](#what-fakegpu-answers)
   - [How it works](#how-it-works)
   - [Built with](#built-with)
2. [Getting started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [Verify the installation](#verify-the-installation)
3. [Usage](#usage)
   - [Run PyTorch code on FakeCUDA](#run-pytorch-code-on-fakecuda)
   - [Intercept native CUDA libraries](#intercept-native-cuda-libraries)
   - [Check memory before a run](#check-memory-before-a-run)
   - [Analyze a repository or model](#analyze-a-repository-or-model)
   - [Monitor virtual GPU memory](#monitor-virtual-gpu-memory)
   - [Simulate multiple nodes over TCP](#simulate-multiple-nodes-over-tcp)
4. [Feature coverage](#feature-coverage)
   - [Command reference](#command-reference)
   - [Runtime modes](#runtime-modes)
5. [GPU profiles](#gpu-profiles)
6. [Reports and validation](#reports-and-validation)
7. [Architecture](#architecture)
8. [Limitations](#limitations)
9. [Roadmap](#roadmap)
10. [Documentation](#documentation)
11. [Contributing](#contributing)
12. [License](#license)
13. [Acknowledgments](#acknowledgments)

## About the project

FakeGPU is a CUDA, CUDA Runtime, cuBLAS, NVML, and NCCL interception toolkit.
It exposes configurable NVIDIA-like devices to applications while running
maintained operations on CPU, records simulated memory and communication, and
provides static estimators for workloads that should not be loaded at all.

The project is designed for CI, local development, compatibility testing,
capacity planning, and repeatable experiments. Physical GPUs are optional for
the simulation and analysis paths; passthrough, hybrid, and calibration runs
use a real CUDA stack.

### What FakeGPU answers

| Question | Recommended path | Physical GPU |
|---|---|---:|
| Does my PyTorch code follow the expected CUDA-facing control flow? | Python FakeCUDA runtime | No |
| Can an unmodified process discover and call CUDA-family shared libraries? | Native interception | No |
| Will a selected GPU profile fit this workload? | Preflight or static memory estimator | No |
| How much checkpoint, KV-cache, adapter, or MoE memory should an LLM use? | LLM estimator | No |
| Where are the GPU-only entry points and dependencies in a repository? | Repository analyzer | No |
| What is the analytical compute/memory latency range for a profile? | Roofline estimator | No |
| Does multi-rank control flow and recovery behave as expected? | Distributed simulator | No |
| How does the estimate compare with an actual CUDA run? | Passthrough/hybrid calibration | Yes |

### How it works

FakeGPU provides three complementary paths:

| Path | What the application sees | What actually runs |
|---|---|---|
| **Python FakeCUDA** | CUDA devices, CUDA-looking tensors, memory APIs, and common training flows | Maintained PyTorch operations execute on CPU through `FakeCudaTensor` |
| **Native interception** | `libcuda`, `libcudart`, `libcublas`, `libnvidia-ml`, and `libnccl` entry points | Selected operations use host memory or CPU math; unsupported behavior is classified and reported |
| **Analysis and reporting** | Memory, FLOP, roofline, topology, and communication reports | ATen graphs, safetensors metadata, runtime traces, calibration data, and coordinator events are analyzed |

### Built with

- Python 3.10+ for the runtime, estimators, CLI, and reports
- C++17 and CMake for native interception libraries and the coordinator
- PyTorch for CPU-backed FakeCUDA execution and ATen graph capture
- YAML/JSON schemas for GPU profiles, test matrices, and reports

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting started

### Prerequisites

- Linux or macOS
- Python 3.10 or newer
- CMake 3.14 or newer
- A C++17 compiler (`build-essential` on Debian/Ubuntu or Xcode Command Line
  Tools on macOS)
- PyTorch for the Python FakeCUDA path

### Installation

Clone the repository:

```bash
git clone https://github.com/FanBB2333/FakeGPU.git
cd FakeGPU
```

Build and validate the native libraries:

```bash
cmake -S . -B build
cmake --build build -j
./ftest smoke
```

Install the Python package, including its compiled native artifacts:

```bash
python3 -m pip install .
```

For development from a checkout, install the optional validation dependencies
and keep the source tree on `PYTHONPATH`:

```bash
python3 -m pip install PyYAML jsonschema pytest
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

This route is intended for device placement, training control flow, error
handling, memory accounting, and framework compatibility checks. The result is
real CPU math, so large models can be much slower than CUDA.

### Intercept native CUDA libraries

`fgpu` configures the preload environment for an unmodified command:

```bash
./fgpu --profile a100 --device-count 2 python3 your_script.py
./fgpu --devices "a100:2,h100:2" python3 your_script.py
./fgpu --mode simulate --unsupported-api error python3 your_script.py
```

Unsupported native calls use one of three explicit policies:

| Policy | Behavior |
|---|---|
| `warn` | Report each unsupported API once and continue when possible |
| `error` | Return `cudaErrorNotSupported` or `CUDA_ERROR_NOT_SUPPORTED` |
| `allow` | Record the event without printing a warning |

### Check memory before a run

Run a command until a target stage and write JSON, Markdown, stdout, and stderr
artifacts:

```bash
python3 -m fakegpu preflight \
  --runtime fakecuda \
  --profile a100 \
  --stage forward \
  --report-dir preflight-report \
  --allocation-stacks \
  --strict \
  -- python3 train.py --small-config
```

Preflight tracks parameters, buffers, gradients, optimizer state, activations,
tensor aliases, dispatch-created storage, saved tensors, caching-allocator
state, and bounded workspace estimates visible on the executed path.

For an ATen graph that does not allocate real CUDA memory:

```python
from fakegpu import estimate_module_memory, require_workspace_coverage

report = estimate_module_memory(
    model,
    (example_input,),
    mode="training",
    optimizer="adamw",
    target_device="auto",
)

require_workspace_coverage(
    report,
    minimum_fraction=1.0,
    allow_extrapolated=False,
    require_upper_bound=True,
)
print(report["estimated_peak_interval_bytes"])
```

### Analyze a repository or model

```bash
# Find entry points, frameworks, CUDA sources, and validation risks.
fakegpu analyze-repo .

# Estimate checkpoint, KV-cache, transient, adapter, and MoE memory.
fakegpu estimate-llm \
  --model-dir /models/qwen \
  --batch-size 1 \
  --prompt-tokens 128 \
  --generated-tokens 32 \
  --dtype bfloat16 \
  --target-profile a100 \
  --json build/llm-estimate.json

# Estimate an analytical profile-aware latency interval.
fakegpu estimate-roofline \
  --profile a100 \
  --flops 1000000000000 \
  --memory-bytes 4000000000 \
  --launch-count 100

# Audit source and built native exports against the capability manifest.
fakegpu capabilities --source-root . --build-dir build --strict
```

The LLM estimator reads safetensors headers without materializing checkpoint
weights. It supports dense and common MoE decoder metadata, quantized
checkpoint storage, repeated PEFT adapters, expert-parallel traffic, KV cache,
eager/SDPA transients, matrix FLOPs, and optional roofline intervals.

### Monitor virtual GPU memory

Publish a process state file from one terminal:

```bash
FAKEGPU_SMI_STATE_DIR=/tmp/fakegpu-smi python3 your_inference.py
```

Inspect it from another terminal:

```bash
fakegpu nvidia-smi --state-dir /tmp/fakegpu-smi
fakegpu nvidia-smi --state-dir /tmp/fakegpu-smi --loop 1 --count 10
```

The table separates requested tensor bytes, allocator-reserved bytes, and
optional calibrated runtime overhead. It is FakeGPU telemetry, not data from
the host NVIDIA driver.

### Simulate multiple nodes over TCP

Run a self-contained two-node loopback check on a chosen port:

```bash
fakegpu bandwidth \
  --listen 127.0.0.1:29591 \
  --nodes 2 \
  --ranks-per-node 1 \
  --size 4MiB \
  --iterations 10
```

FakeGPU starts a coordinator, creates the topology, moves collective payloads
through TCP, validates the reduction, reports end-to-end throughput, and
writes node-pair traffic statistics. A separately hosted coordinator supports
physical multi-host experiments.

See [Distributed Simulation Usage](docs/distributed-sim-usage.md) for
`torchrun`, DDP, FSDP, DeepSpeed, topology, and recovery examples.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Feature coverage

| Area | Implemented coverage | Boundary |
|---|---|---|
| FakeCUDA runtime | Common tensor creation/manipulation, modules, autograd, optimizers, device propagation, memory APIs, mixed precision, checkpointing, DataLoader paths, and dispatch-level storage lifetime tracking | Binary CUDA extensions and unsupported PyTorch operators need targeted validation |
| Native CUDA stack | Selected Driver, Runtime, NVML, cuBLAS/cuBLASLt, and NCCL symbols; host-backed memory; CPU GEMM; configurable unsupported-API policy | Arbitrary CUDA kernels do not execute in native simulate mode |
| Memory estimation | Runtime peaks, simplified caching allocator, ATen graph liveness, optimizer phases, workspace intervals, calibration bundles, and OOM checks | Backend-private allocations and unmatched custom kernels require calibration |
| LLM analysis | Dense/MoE inference, quantized checkpoint bytes, PEFT adapters, KV cache, transients, expert traffic, FLOPs, SFT references, and FSDP/FSDP2 projections | Fused quantization kernels, expert imbalance, and arbitrary architectures are not inferred automatically |
| Performance model | Scalar-compute and memory-bandwidth roofline with lower/expected/upper efficiency assumptions | Analytical interval only; Tensor Core acceleration must be supplied explicitly |
| Repository analysis | Python entry points, imports, frameworks, configs, CUDA/PTX sources, compiled extensions, and recommended experiments | Dynamic imports, generated kernels, and data-dependent branches need runtime checks |
| Distributed simulation | TCP/Unix coordinator, collectives, P2P, subgroups, heterogeneous topologies, node-pair reports, timeout/failure injection, communicator shrink, and fixed-size elastic restart workflows | Does not reproduce NCCL protocols, NVLink, or RDMA timing |
| Framework compatibility | Focused Transformers, Accelerate, PEFT, TRL, DDP, FSDP/FSDP2, DeepSpeed ZeRO/Pipeline/AutoTP/AutoEP, torchtune, Lightning, LitGPT, and nanoGPT workflows | Compatibility tests cover documented versions and options, not every upstream API |
| Monitoring and reports | Virtual `nvidia-smi`, per-device native report, preflight artifacts, cluster matrix, validation manifests, and JSON schema checks | Virtual telemetry reflects tracked or calibrated state only |
| GPU catalog | 82 consumer, workstation, data-center, embedded, and test profiles across eight NVIDIA architectures | Profile specifications do not guarantee kernel-level performance equivalence |

Coverage terms:

- **Maintained**: included in the regular regression surface.
- **Validated**: checked by a dedicated numerical or physical-host experiment
  within its documented model, shape, software, and architecture range.
- **Compatibility-tested**: checked by focused framework workflows.
- **Experimental**: available for its documented prototype scope without a
  general compatibility guarantee.

### Command reference

| Command | Purpose |
|---|---|
| `fakegpu doctor` | Check installation, libraries, PyTorch, and GPU profiles |
| `fakegpu demo` | Run a small CPU-backed CUDA-visible training step |
| `fakegpu preflight` | Execute a workload to a target stage and classify fit/OOM |
| `fakegpu analyze-repo` | Inventory repository entry points and GPU-only risks |
| `fakegpu estimate-llm` | Estimate decoder checkpoint, runtime memory, communication, and FLOPs |
| `fakegpu estimate-roofline` | Produce a profile-aware analytical latency interval |
| `fakegpu capabilities` | List or strictly audit native API classifications |
| `fakegpu nvidia-smi` | Display virtual per-process GPU memory |
| `fakegpu workspace-profiles` | Validate and list workspace catalogs |
| `fakegpu validate` | Run a declarative JSON/TOML/YAML test matrix |
| `fakegpu coordinator` | Start, probe, stop, or report on the distributed coordinator |
| `fakegpu bandwidth` | Validate simulated TCP payloads and measure end-to-end throughput |

### Runtime modes

Python runtime:

| Runtime | Behavior |
|---|---|
| `fakecuda` | Patch PyTorch with FakeCudaTensor behavior and execute maintained operations on CPU |
| `native` | Load FakeGPU shared libraries into the current process |
| `auto` | Prefer `fakecuda` when available, otherwise use `native` |

Native compute mode:

| `FAKEGPU_MODE` | Behavior | Physical GPU |
|---|---|---:|
| `simulate` | Virtual identity and memory; maintained cuBLAS/cuBLASLt paths can execute on CPU | No |
| `passthrough` | Unmodified real-CUDA baseline without FakeGPU CUDA/NVML injection | Yes |
| `hybrid` | Real CUDA compute with selected Driver/NVML virtualization and OOM policy handling | Yes |

Distributed mode:

| `FAKEGPU_DIST_MODE` | Behavior |
|---|---|
| `disabled` | Do not install a FakeGPU distributed layer |
| `simulate` | Use coordinator-managed collective and point-to-point semantics |
| `proxy` | Keep real NCCL movement and add FakeGPU control-plane reporting |
| `passthrough` | Forward directly to the real NCCL stack |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## GPU profiles

Profiles live under `profiles/<architecture>/<segment>/*.yaml` and are shared
by the Python and native runtimes.

| Architecture | Profiles | Compute capability | Product coverage |
|---|---:|---|---|
| Maxwell | 1 | 5.2 | GeForce GTX 900 series |
| Pascal | 9 | 6.0, 6.1 | GeForce GTX 10 and Tesla P series |
| Volta | 1 | 7.0 | Tesla V series |
| Turing | 12 | 7.5 | GeForce RTX 20, Quadro RTX, and T4 |
| Ampere | 22 | 8.0, 8.6, 8.7 | GeForce RTX 30, RTX A, A-series accelerators, and Jetson |
| Ada | 17 | 8.9 | GeForce RTX 40, RTX Ada, and L-series accelerators |
| Hopper | 2 | 9.0 | H-series accelerators |
| Blackwell | 18 | 10.0, 10.3, 11.0, 12.0, 12.1 | GeForce RTX 50, RTX PRO, B-series, Jetson, and GB10 |

Every profile declares its architecture and compute capability. Validators
reject mismatched combinations. Specification sources and
measured/reference/synthetic status are recorded in each YAML file.

```bash
fakegpu doctor --list-profiles
./fgpu --profile rtx4090 --device-count 2 python3 your_script.py
./fgpu --devices "t4,a100:2,h100" python3 your_script.py
python3 scripts/update_nvidia_gpu_catalog.py --check
```

See [profiles/README.md](profiles/README.md) for provenance and validation
rules.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Reports and validation

| Artifact | Producer | Main contents |
|---|---|---|
| `fake_gpu_report.json` | Native runtime | Per-device memory, IO, API calls, unsupported behavior, and maintained GEMM FLOPs |
| `cluster_report.json/.md` | Distributed coordinator | Collective/P2P totals, complete node-pair matrix, peaks, topology, timeline, failures, and recovery |
| `preflight_report.json/.md` | Preflight CLI | Stage progress, fit/OOM result, memory categories, workspace coverage, and confidence |
| LLM estimate | `fakegpu estimate-llm` | Checkpoint, KV cache, transients, adapters, MoE traffic, FLOPs, and roofline interval |
| Static memory report | `./ftest static_memory_validation` | Graph liveness, optimizer phases, workspace profiles, and optional CUDA comparison |
| Declarative validation report | `fakegpu validate` | Expanded case matrix, prerequisites, assertions, host/Git identity, duration, and logs |
| Virtual SMI state | FakeCUDA runtime | Per-process requested, reserved, simulated current/peak bytes, stage, and confidence |

Maintained local checks:

```bash
./ftest smoke
./ftest cpu_sim
./ftest static_memory_validation
python3 -m pytest -q
python3 -m fakegpu validate \
  --manifest verification/data/validation_smoke.yaml \
  --report-dir build/validation-smoke \
  --strict
```

The current regression baseline contains 425 passing tests and one optional
skip. Native smoke, CPU numerical simulation, strict native capability audit,
wheel installation, and strict MkDocs builds are also checked. Accuracy
numbers apply only to the documented workloads and calibration signatures.

Detailed numerical, distributed, framework, and cross-architecture evidence is
kept in [Reports and Validation](docs/reports-and-validation.md) so this page
remains useful as an entry point.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Architecture

```text
GPU-facing application
├── Python runtime: fakegpu.init(runtime="fakecuda")
│   └── FakeCudaTensor + policies + memory tracker
│       └── maintained PyTorch operations execute on CPU
│
├── Native runtime: ./fgpu or fakegpu.init(runtime="native")
│   └── libcuda / libcudart / libcublas / libnvidia-ml / libnccl
│       ├── profiles, allocations, streams, and metrics
│       ├── host memory and CPU-backed maintained math
│       └── optional real CUDA forwarding in hybrid mode
│
└── Analysis
    ├── repository and dependency inventory
    ├── ATen graph and safetensors estimators
    └── roofline, calibration, and report validation

Distributed coordinator
└── logical nodes, TCP/Unix transport, collectives, failures, and reports
```

For a file-level map, see
[Architecture and Project Structure](docs/project-structure.md).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Limitations

- Native simulate mode does not execute arbitrary CUDA kernels. Use
  `FAKEGPU_UNSUPPORTED_API=error` when a compatibility no-op would invalidate
  the test.
- FakeCudaTensor covers maintained Python/PyTorch behavior, not binary CUDA
  extensions.
- Static repository analysis cannot resolve every dynamic import, generated
  kernel, runtime shape, or data-dependent branch.
- Runtime and static memory estimates can miss backend-private allocations,
  custom operators, allocator policies, and unmatched workspaces. Use a
  matching real-GPU calibration for capacity decisions.
- The LLM estimator does not reproduce fused quantization kernels, infer expert
  imbalance, or execute arbitrary model architectures.
- Roofline output is an analytical interval, not measured kernel latency.
- Distributed timing includes coordinator work, memory copies, sockets, and
  process scheduling. It is not raw network or NCCL performance.
- Hybrid and passthrough modes require a compatible physical CUDA stack.
- macOS SIP can remove `DYLD_*` variables from system binaries. Prefer a
  Homebrew, conda, or pyenv Python for native interception.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Roadmap

- [x] CPU-backed PyTorch FakeCUDA runtime
- [x] Native CUDA/NVML/cuBLAS/NCCL interception
- [x] Configurable architecture-aware GPU profile catalog
- [x] Runtime, static, LLM, MoE, quantization, and adapter memory estimates
- [x] Strict native API capability manifest and export audit
- [x] Repository analyzer and profile-aware roofline estimator
- [x] TCP multi-node simulation and complete node-pair communication reports
- [x] Focused DDP, FSDP/FSDP2, DeepSpeed, and elastic recovery validation
- [ ] Expand executable native CUDA and cuBLAS operation coverage
- [ ] Add more calibration bundles across software stacks and workload classes
- [ ] Extend generated-kernel and custom-extension detection
- [ ] Expand topology models for hierarchical and high-radix fabrics

See the [open issues](https://github.com/FanBB2333/FakeGPU/issues) for proposed
features and known limitations.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Documentation

- [Getting Started](docs/getting-started.md)
- [Quick Reference](docs/quick-reference.md)
- [AI Researcher Preflight](docs/ai-researcher-preflight.md)
- [Repository and Roofline Analysis](docs/repository-and-performance-analysis.md)
- [LLM Inference Estimation](docs/llm-inference-estimation.md)
- [LLM SFT Memory Estimation](docs/llm-sft-memory-estimation.md)
- [Distributed Simulation Usage](docs/distributed-sim-usage.md)
- [DeepSpeed Validation](docs/deepspeed-validation.md)
- [Error Simulation](docs/error-simulation.md)
- [Reports and Validation](docs/reports-and-validation.md)
- [Declarative Validation Manifests](docs/validation-manifests.md)
- [Architecture and Project Structure](docs/project-structure.md)

Preview the documentation site locally:

```bash
python3 -m pip install -e ".[docs]"
mkdocs serve
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

Bug reports, focused test cases, profile corrections, documentation
improvements, and implementation patches are welcome.

1. Fork the repository.
2. Create a branch: `git checkout -b feat/your-change`.
3. Add or update tests for the changed behavior.
4. Run the relevant `ftest` target and Python tests.
5. Commit with a clear
   [Conventional Commit](https://www.conventionalcommits.org/) message.
6. Push the branch and open a pull request.

Please include the exact command, profile, software versions, and generated
report when submitting an estimation or compatibility issue.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments

- README structure inspired by
  [Best-README-Template](https://github.com/othneildrew/Best-README-Template)
- GPU model and compute-capability references from
  [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda/gpus) and the
  [legacy GPU table](https://developer.nvidia.com/cuda/gpus/legacy)
- CPU-backed framework validation built around
  [PyTorch](https://pytorch.org/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

[test-shield]: https://github.com/FanBB2333/FakeGPU/actions/workflows/test.yml/badge.svg?branch=main
[test-url]: https://github.com/FanBB2333/FakeGPU/actions/workflows/test.yml
[release-shield]: https://img.shields.io/github/v/release/FanBB2333/FakeGPU?include_prereleases&sort=semver
[release-url]: https://github.com/FanBB2333/FakeGPU/releases
[python-shield]: https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white
[python-url]: https://www.python.org/
[license-shield]: https://img.shields.io/github/license/FanBB2333/FakeGPU
[license-url]: LICENSE
