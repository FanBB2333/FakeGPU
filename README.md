# FakeGPU

FakeGPU simulates CUDA-facing environments for development, CI, compatibility
checks, and capacity planning without requiring a production GPU cluster.

It provides:

- a CPU-backed PyTorch runtime with CUDA-like devices and memory accounting;
- native `libcuda`, `libcudart`, `libcublas`, `libnvidia-ml`, and `libnccl`
  interception;
- GPU profile selection, preflight memory checks, repository analysis, and
  workload estimation;
- JSON reports for memory, compute, communication, and unsupported APIs.

FakeGPU does not provide numerical or performance parity for arbitrary CUDA
kernels. Passthrough, hybrid, and calibration workflows still require a real
CUDA stack.

## Requirements

- Python 3.10 or newer
- CMake 3.14 or newer
- A C++17 compiler
- Linux or macOS
- PyTorch for the Python FakeCUDA runtime

## Install

Build the native libraries:

```bash
scripts/build.sh
```

Install the package:

```bash
python3 -m pip install .
```

For development from a checkout:

```bash
python3 -m pip install pytest PyYAML jsonschema
export PYTHONPATH="$PWD"
```

Verify the environment:

```bash
python3 -m fakegpu doctor --list-profiles
python3 -m fakegpu demo --profile l4
```

## Python FakeCUDA

Initialize FakeGPU before importing PyTorch:

```python
import fakegpu

fakegpu.init(runtime="fakecuda", profile="a100", device_count=2)

import torch

model = torch.nn.Linear(8, 4).to("cuda:0")
x = torch.randn(2, 8, device="cuda:0")
loss = model(x).square().mean()
loss.backward()

print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(loss.item())
```

Maintained operations execute on CPU while device placement, memory limits,
training control flow, and error handling use the simulated CUDA surface.

## Native interception

From a source checkout, `python -m fakegpu` can prepare the preload environment
and launch an unmodified command:

```bash
python3 -m fakegpu --build-dir build --profile a100 \
  python3 your_script.py

python3 -m fakegpu --build-dir build --devices "a100:2,h100:2" \
  python3 your_script.py
```

Unsupported native calls can be recorded, warned about, or returned as
`cudaErrorNotSupported`/`CUDA_ERROR_NOT_SUPPORTED`.

## Analysis commands

```bash
# Inspect GPU entry points and dependencies in a repository.
python3 -m fakegpu analyze-repo .

# Check whether a command fits a selected profile.
python3 -m fakegpu preflight \
  --runtime fakecuda \
  --profile a100 \
  --stage forward \
  --report-dir build/preflight \
  --strict \
  -- python3 train.py

# Run a declarative validation manifest.
python3 -m fakegpu validate \
  --manifest tests/data/validation_smoke.yaml \
  --report-dir build/validation-smoke \
  --strict
```

Run `python3 -m fakegpu --help` for the complete command list.

## GPU profiles

Profiles are stored as YAML under `profiles/` and embedded into native builds.

```bash
python3 -m fakegpu doctor --list-profiles
python3 -m fakegpu demo --profile rtx4090
```

Set `FAKE_GPU_PROFILE` or pass `--profile` to select a device. Use `--devices`
for a heterogeneous list.

## Development

The maintained test surface is intentionally small:

```bash
scripts/test.sh python
scripts/test.sh smoke
scripts/test.sh cpu
scripts/test.sh all
```

Build options are available through one reusable entry point:

```bash
scripts/build.sh --release
scripts/build.sh --debug
scripts/build.sh --build-dir build-custom -- -DSOME_CMAKE_OPTION=value
```

Repository layout:

```text
fakegpu/   Python package
profiles/  GPU profile catalog
schemas/   JSON report and validation schemas
scripts/   Reusable build, test, platform, and validation tools
src/       Native C++ implementation
tests/     Maintained regression tests and minimal native fixtures
```

Build directories, compiled libraries, test reports, caches, local environments,
binary assets, and design drafts are ignored by Git.

## License

FakeGPU is distributed under the [MIT License](LICENSE).
