<a id="readme-top"></a>

<div align="center">

# FakeGPU

**Validate CUDA-facing applications, estimate GPU memory, and simulate
distributed GPU workflows without a production GPU cluster.**

[![Tests][test-shield]][test-url]
[![Release][release-shield]][release-url]
[![Python][python-shield]][python-url]
[![License][license-shield]][license-url]

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
  --target-profile a100 \
  --json build/llm-estimate.json

# Audit source and built native exports against the capability manifest.
python3 -m fakegpu capabilities \
  --source-root . \
  --build-dir build \
  --strict
```

The LLM estimator reads safetensors headers without materializing checkpoint
weights.

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
| `fakegpu calibrate` | Compare predicted and observed memory |
| `fakegpu capabilities` | List or strictly audit native API classifications |
| `fakegpu nvidia-smi` | Display virtual per-process GPU memory |
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
- [ ] Expand executable native CUDA operations and cuBLAS coverage
- [ ] Add calibration evidence for more software stacks and workload classes

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
