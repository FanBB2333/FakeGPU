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
2. [Getting started](#getting-started)
3. [Usage](#usage)
4. [Documentation](#documentation)
5. [Development](#development)
6. [Limitations](#limitations)
7. [Contributing](#contributing)
8. [License](#license)
9. [Acknowledgments](#acknowledgments)

## About the project

FakeGPU makes a machine without a GPU look like it has one. Applications see
configurable NVIDIA-like devices, while the supported operations actually run
on the CPU. Along the way, FakeGPU records memory use and communication, so you
can check whether code works and whether a workload would fit before you pay
for real hardware.

It works in three ways:

- **Python FakeCUDA** lets PyTorch code use `cuda` devices while tensors stay
  on the CPU.
- **Native interception** loads in place of `libcuda`, `libcudart`, `libcublas`,
  `libnvidia-ml`, and `libnccl`, so unmodified programs can run.
- **Static estimators** read model metadata and configs and report memory,
  FLOPs, and communication without loading weights.

A physical GPU is only needed for passthrough, hybrid, and calibration runs.

<a id="use-cases"></a>

### What you can do

- **Choose a GPU before renting one.** Estimate checkpoint, KV-cache,
  activation, and optimizer memory with `estimate-llm` or run a workload to a
  chosen stage with `preflight`.
- **Size an LLM service.** `plan-serving` checks how many chat, RAG, or
  speculative-decoding requests fit in memory, including a vLLM-style budget.
- **Develop CUDA code on a laptop or in CPU-only CI.** Call
  `fakegpu.init(...)` in your script, or try `demo` and `validate`.
- **Compare training plans.** `plan-training` estimates per-rank memory for
  full fine-tuning, LoRA, QLoRA, sharding, and offload.
- **Plan diffusion generation.** `estimate-diffusion` shows how resolution,
  batch size, CFG, VAE tiling, and offload change memory.
- **Review an unfamiliar GPU repository.** `analyze-repo`, `analyze-kernel`,
  and `capabilities` list GPU entry points, kernels, and unsupported APIs.
- **Debug distributed workflows.** `simulate-topology`, `replay-trace`, and
  `bandwidth` cover collective routing, rank waits, and payload checks.
- **Watch simulated devices.** `nvidia-smi` and `metrics` show devices and
  processes and export Prometheus metrics.
- **Check estimates against a real GPU.** `calibrate` compares predictions with
  measured peaks and enforces error limits.

On the recorded validation workloads, the calibrated memory estimator stayed
within 0.08% on 26 controlled GPU observations and within 1.921% across ten
Qwen SFT cases. These results cover fixed workloads only. See
[Validation and reliability](docs/en/validation.md) for the full evidence.

### Built with

- [Python](https://www.python.org/) 3.10+ for the runtime, estimators, and CLI
- C++17 and [CMake](https://cmake.org/) for the native libraries
- [PyTorch](https://pytorch.org/) for CPU-backed FakeCUDA execution

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting started

You need Linux or macOS, Python 3.10+, CMake 3.14+, a C++17 compiler, and
PyTorch for the Python runtime. On Debian or Ubuntu, install
`build-essential`. On macOS, install the Xcode Command Line Tools.

```bash
git clone https://github.com/FanBB2333/FakeGPU.git
cd FakeGPU
scripts/build.sh
FAKEGPU_BUILD_DIR="$PWD/build" python3 -m pip install '.[torch]'
```

Check that everything works:

```bash
python3 -m fakegpu doctor --list-profiles
python3 -m fakegpu demo --profile l4
```

`doctor` checks the profile catalog, native libraries, and PyTorch. `demo` runs
a small training step on the CPU while the program sees a CUDA device.

For a native macOS Conda environment, follow the
[macOS setup instructions](docs/en/development.md#macos-conda-environment).

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
loss = model(torch.randn(2, 8, device=device)).square().mean()
loss.backward()

print(torch.cuda.device_count())      # 2
print(torch.cuda.get_device_name(0))  # NVIDIA A100
```

### Run an unmodified program

The launcher sets up `LD_PRELOAD` or `DYLD_INSERT_LIBRARIES` for you:

```bash
python3 -m fakegpu --build-dir build --devices "a100:2,h100:2" \
  python3 your_script.py
```

More options are in [Native interception](docs/en/native-interception.md).

### Check memory before a run

```bash
# Run train.py up to the forward pass and report whether it fits.
python3 -m fakegpu preflight --runtime fakecuda --profile a100 \
  --stage forward --report-dir build/preflight -- python3 train.py

# Estimate LLM inference memory from safetensors headers only.
python3 -m fakegpu estimate-llm --model-dir /models/example \
  --prompt-tokens 128 --generated-tokens 32 --target-profile a100

# Plan an online serving pool.
python3 -m fakegpu plan-serving --model-dir /models/example \
  --active-sequences 16 --prompt-tokens 4096 --generated-tokens 256 \
  --target-profile a100
```

### Inspect simulated devices

```bash
export FAKEGPU_SMI_STATE_DIR=build/smi
python3 your_script.py &
python3 -m fakegpu nvidia-smi --state-dir build/smi
```

For pip-installed nvitop on macOS/Linux, use `fakegpu nvitop --state-dir build/smi`; see the [NVML bridge guide](docs/en/fakegpu-smi.md#use-pip-installed-nvitop-on-macos-or-linux).

To keep a simulated machine running in the background, install the `console`
extra and run `fakegpu console`: FakeCUDA processes join the machine, and the
console shows GPUs, processes and events. See the
[console guide](docs/en/console.md).

Run `python3 -m fakegpu --help` for every command, and
`python3 -m fakegpu <command> --help` for its options.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Documentation

The detailed guides live in [`docs/`](docs/en/README.md):

- [Command reference](docs/en/commands.md)
- [Memory analysis and estimation](docs/en/estimation.md):
  `preflight`, `analyze-repo`, `estimate-llm`, `estimate-diffusion`
- [Online LLM serving](docs/en/serving.md): `plan-serving`, vLLM budgets,
  speculative decoding, mixed requests, `benchmark-serving`
- [Fine-tuning time and VRAM](docs/en/finetuning.md)
- [Diffusion training](docs/en/diffusion-training.md)
- [Distributed plans and experiment sweeps](docs/en/distributed.md)
- [FakeGPU-SMI and monitoring](docs/en/fakegpu-smi.md)
- [Native interception](docs/en/native-interception.md)
- [Calibrating against a real GPU](docs/en/calibration.md)
- [Validation and reliability](docs/en/validation.md)
- [Development and testing](docs/en/development.md)
- [Limitations](docs/en/limitations.md)

GPU profiles are YAML files in [`profiles/`](profiles/). The catalog has 307
selectable configurations from Maxwell through Blackwell, including desktop,
laptop, memory-capacity, and board variants. See the [full inventory and sources](profiles/README.md). Pick one with `--profile` or
`FAKEGPU_PROFILE`, or pass a mixed list with `--devices`. Examples that target
a physical device use `TARGET_GPU_PROFILE`, which you set to a catalog ID from
`python3 -m fakegpu doctor --list-profiles`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Development

```bash
scripts/build.sh          # build the native libraries
scripts/test.sh python    # Python tests
scripts/test.sh all       # every maintained suite
```

Build options, test suites, and the project layout are described in
[Development and testing](docs/en/development.md).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Limitations

- Native simulation does not run arbitrary CUDA kernels, and FakeCUDA does not
  cover binary CUDA extensions.
- Memory estimates can miss backend-private allocations, custom operators, and
  allocator behavior. Calibrate on real hardware before relying on them.
- Timing results from the simulator are not NCCL, NVLink, or kernel
  benchmarks.
- Hybrid and passthrough modes need a working CUDA stack.

The complete list is in [Limitations](docs/en/limitations.md). Planned work is
tracked in the [open issues](https://github.com/FanBB2333/FakeGPU/issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

Bug reports, test cases, profile corrections, documentation fixes, and patches
are welcome.

1. Fork the repository and create a branch.
2. Add or update tests for the changed behavior.
3. Run `scripts/test.sh all`.
4. Commit with a [Conventional Commit](https://www.conventionalcommits.org/)
   message and open a pull request.

For estimation or compatibility issues, please include the command, profile,
software versions, and generated report.

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
