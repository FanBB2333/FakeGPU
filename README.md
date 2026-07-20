# FakeGPU

FakeGPU is a CUDA, cuBLAS, NVML, and NCCL interception toolkit for validating GPU-facing code without depending on a production GPU cluster. It provides a Python fake-CUDA path, native shared-library interception, configurable GPU profiles, distributed control-flow simulation, and memory preflight reports.

[Getting started](docs/getting-started.md) · [Quick reference](docs/quick-reference.md) · [中文文档](docs/index.zh.md) · [Changelog](CHANGELOG.md)

> FakeGPU is a development and validation tool. It does not provide numerical or performance parity for arbitrary CUDA kernels.

## Choose a path

| Goal | GPU required | Recommended entry point |
|---|---:|---|
| Exercise PyTorch CUDA-style code on CPU | No | `fakegpu.init(runtime="fakecuda")` |
| Intercept CUDA, NVML, cuBLAS, or NCCL shared-library calls | No | `./fgpu --mode simulate ...` |
| Check whether a workload reaches a stage or exceeds a GPU profile | No | `python3 -m fakegpu preflight ...` |
| Estimate training memory from an ATen graph | No | `./ftest static_memory_validation` |
| Compare an unmodified real-GPU baseline | Yes | `./fgpu --mode passthrough ...` |
| Keep real CUDA compute while virtualizing selected surfaces | Yes | `./fgpu --mode hybrid --oom-policy clamp ...` |
| Simulate multi-rank collective control flow | No | `FAKEGPU_DIST_MODE=simulate` |

## Quick start

### Build the native libraries

Requirements: Linux or macOS, Python 3.10+, CMake 3.14+, and a C++17 compiler.

```bash
git clone https://github.com/FanBB2333/FakeGPU.git
cd FakeGPU

cmake -S . -B build
cmake --build build
./ftest smoke
```

The default build executes maintained cuBLAS/cuBLASLt operations on CPU. Generic CUDA kernel launches remain no-ops in native simulate mode.

### Run PyTorch code with fake CUDA semantics

From the repository root, with PyTorch installed:

```python
import fakegpu

fakegpu.init(runtime="fakecuda", profile="a100", device_count=2)

import torch

device = torch.device("cuda:0")
model = torch.nn.Linear(8, 4).to(device)
x = torch.randn(2, 8, device=device)
loss = model(x).square().mean()
loss.backward()

print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(loss.item())
```

This route uses FakeCudaTensor semantics and CPU execution. It is suitable for training-flow, device-placement, error-handling, and framework compatibility checks. Custom CUDA extensions and arbitrary CUDA kernels are outside its maintained scope.

### Run a native intercepted command

`./fgpu` selects the built libraries and sets the preload environment:

```bash
./fgpu --profile a100 --device-count 2 nvidia-smi  # when nvidia-smi is installed
./fgpu --mode simulate python3 your_script.py
./fgpu --devices "a100:2,h100:2" python3 your_script.py
```

The same native route can be enabled before importing CUDA-using libraries:

```python
import fakegpu

fakegpu.init(runtime="native", profile="a100", device_count=2)

import torch
```

## AI workload preflight

Use preflight to check whether a command reaches a target stage and whether tracked memory fits a selected profile:

```bash
python3 -m fakegpu preflight \
  --runtime fakecuda \
  --profile a100 \
  --device-count 1 \
  --stage forward \
  --report-dir preflight-report \
  --allocation-stacks \
  --strict \
  -- python3 train.py --small-config
```

The command writes:

- `preflight_report.json`
- `preflight_report.md`
- `preflight_stdout.log`
- `preflight_stderr.log`

Preflight tracks tensor lifetimes, stage peaks, parameters, buffers, gradients, optimizer state, activations, shared-storage aliases, and saved tensors visible through PyTorch hooks. A matching empirical calibration report can replace the raw fake-CUDA estimate with an observed physical-memory upper bound.

See [AI Researcher Preflight](docs/ai-researcher-preflight.md) for calibration inputs, confidence levels, and current limits.

## Static memory estimation

The static estimator captures a target-device ATen forward/backward graph without allocating real CUDA memory. It models:

- unique tensor storages and aliases
- storage lifetime through final graph use
- parameters, buffers, gradients, and optimizer state
- separate graph and optimizer phases
- matched CUDA Attention workspace profiles
- operator-local workspace at the corresponding graph-liveness point

Run the maintained workload grid:

```bash
./ftest static_memory_validation
```

Or call the API directly:

```python
from fakegpu import estimate_module_memory

report = estimate_module_memory(
    model,
    (example_input,),
    mode="training",
    optimizer="adamw",
    target_device="auto",
)
print(report["estimated_peak_bytes"])
```

CUDA-specific ATen traces require a CUDA-enabled PyTorch build, but the trace itself does not allocate physical GPU memory.

Backend context memory, allocator fragmentation, unmatched workspaces, custom CUDA operators, and fused/foreach optimizer temporaries require device- and software-specific calibration.

## Runtime model

FakeGPU separates the Python runtime path, native compute mode, and distributed mode.

### Python runtime

| Runtime | Behavior |
|---|---|
| `fakecuda` | Patches PyTorch with FakeCudaTensor behavior and executes maintained operations on CPU |
| `native` | Loads FakeGPU shared libraries into the current process |
| `auto` | Selects `fakecuda` when available, otherwise uses `native` |

### Native compute mode

| `FAKEGPU_MODE` | Behavior | Real GPU |
|---|---|---:|
| `simulate` | Virtual device identity and memory; maintained cuBLAS/cuBLASLt paths can execute on CPU | No |
| `passthrough` | Unmodified real-CUDA baseline with no FakeGPU CUDA/NVML injection | Yes |
| `hybrid` | Real CUDA compute with selected Driver/NVML virtualization and OOM policy handling | Yes |

Hybrid OOM policies:

```text
clamp | managed | mapped_host | spill_cpu
```

`clamp` is the maintained validation path. Oversubscription policies are experimental and do not guarantee numerical parity.

### Distributed mode

| `FAKEGPU_DIST_MODE` | Behavior |
|---|---|
| `disabled` | No FakeGPU distributed layer |
| `simulate` | Coordinator-managed collective and point-to-point semantics |
| `proxy` | Real NCCL data movement with FakeGPU control-plane reporting |
| `passthrough` | Thin forwarding to real NCCL |

Compute and distributed modes can be combined. The recommended CPU-only mode pair is:

```bash
FAKEGPU_MODE=simulate
FAKEGPU_DIST_MODE=simulate
```

The simulated distributed path also needs a coordinator endpoint and cluster configuration. For complete coordinator and `torchrun` examples, see [Distributed Simulation Usage](docs/distributed-sim-usage.md).

## Capability map

| Surface | Maintained behavior | Important boundary |
|---|---|---|
| CUDA Driver/Runtime | Device discovery, memory allocation/copy, streams/events, selected Driver forwarding | The full CUDA API is not implemented |
| NVML | Device identity, memory information, common monitoring queries | Some telemetry fields are synthetic or unavailable |
| cuBLAS/cuBLASLt | Selected GEMM/matmul operations with CPU-backed execution | Unsupported algorithms may remain stubbed |
| PyTorch fake-CUDA | Common tensor, module, autograd, optimizer, Transformers, PEFT, Accelerate, and FSDP smoke paths | Custom CUDA extensions are not emulated |
| NCCL-style communication | Collective and point-to-point control flow, topology-aware reporting | Not a protocol-level NCCL/RDMA/NVLink model |
| Memory preflight | Runtime tracking, ATen static analysis, empirical GPU calibration | Results apply to the validated shape and software envelope |
| Error simulation | OOM, invalid device, cross-device, dtype/autocast, gradient, and checkpoint cases | Error timing can differ from a real driver |

## GPU profiles

Profiles are stored in `profiles/*.yaml` and compiled into native builds. Presets include:

```text
GTX 980, P100, V100, T4, A40, A100, H100, L40S,
B100, B200, RTX 3090 Ti, RTX PRO 5000 Blackwell
```

Select one uniform profile or a heterogeneous device list:

```bash
./fgpu --profile rtx3090ti --device-count 2 python3 your_script.py
./fgpu --devices "t4,a100:2,h100" python3 your_script.py
```

Equivalent environment variables:

```bash
FAKEGPU_PROFILE=a100
FAKEGPU_DEVICE_COUNT=8
FAKEGPU_PROFILES=a100:4,h100:4
```

## Reports

| Report | Produced by | Contents |
|---|---|---|
| `fake_gpu_report.json` | Native runtime | Per-device peak memory, IO, calls, and maintained GEMM FLOPs |
| Cluster report | Distributed coordinator | Collective counts, bytes, estimated timing, and link statistics |
| `preflight_report.json/.md` | Preflight CLI | Stage status, fit/OOM result, memory categories, and confidence |
| Real-GPU calibration report | `./ftest real_gpu_calibration` | Real, passthrough, hybrid, fakecuda, allocator, and NVML observations |
| Static memory validation report | `./ftest static_memory_validation` | Graph liveness, optimizer phases, workspace profiles, and real-CUDA comparison |

Output paths can be configured with:

```bash
FAKEGPU_REPORT_PATH=/path/to/fake_gpu_report.json
FAKEGPU_CLUSTER_REPORT_PATH=/path/to/cluster_report.json
```

## Validation snapshot

The maintained cross-GPU static-memory grid was checked on:

| GPU | Compute capability | PyTorch / CUDA |
|---|---:|---|
| NVIDIA GeForce RTX 3090 Ti | 8.6 | PyTorch 2.12.1 / CUDA 13.0 |
| NVIDIA RTX PRO 5000 72GB Blackwell | 12.0 | PyTorch 2.9.1 / CUDA 12.8 |

Across 13 MLP/Transformer workloads and 26 GPU observations:

- static peak bytes matched across both hosts
- maximum allocated-byte absolute error was `0.077160%`
- maximum requested-byte absolute error was `0.001358%`
- FP32 Efficient Attention requested-byte differences were at most 28 bytes
- no maintained Attention workload used an unprofiled or extrapolated profile

These results validate the maintained parameter grid. They do not establish accuracy for arbitrary models, shapes, PyTorch releases, or CUDA backends.

## Test suites

```bash
./ftest smoke
./ftest cpu_sim
./ftest python
./ftest preflight_oom
./ftest static_memory_validation
./ftest real_gpu_calibration
python3 -m pytest -q
```

| Suite | Purpose |
|---|---|
| `smoke` | Build, library boundaries, preload, profiles, and memory types |
| `cpu_sim` | CPU-backed cuBLAS/cuBLASLt correctness |
| `python` | Basic PyTorch native-interception path |
| `preflight_oom` | Fit/OOM classification and report validation |
| `static_memory_validation` | ATen graph memory estimation; optional real-CUDA comparison |
| `real_gpu_calibration` | Real/passthrough/hybrid/fakecuda comparison on a supported GPU |

Additional distributed and framework checks are listed in [Reports and Validation](docs/reports-and-validation.md).

## Architecture

```text
GPU-facing application
├── Python runtime: fakegpu.init(runtime="fakecuda")
│   └── FakeCudaTensor + FakeGPU policies
│       └── maintained PyTorch operations execute on CPU
│
└── Native runtime: ./fgpu or fakegpu.init(runtime="native")
    └── libcuda / libcudart / libcublas / libnvidia-ml / libnccl
        ├── GlobalState: profiles, allocations, streams, metrics
        ├── system RAM and CPU-backed maintained math paths
        ├── optional real CUDA forwarding in hybrid mode
        └── coordinator for simulated distributed operations

Reports: device JSON · cluster JSON · preflight · calibration · static memory
```

## Limitations

- Native simulate mode does not execute arbitrary CUDA kernels.
- FakeCudaTensor covers Python/PyTorch behavior, not binary CUDA extensions.
- Supported cuBLAS/cuBLASLt operations can be numerically checked on CPU; unsupported operations may be stubs.
- Distributed simulation checks semantics and control flow, not exact NCCL protocol or production-network performance.
- Static and runtime memory estimates can omit backend-internal allocations outside matched profiles.
- Hybrid mode requires a real GPU and remains limited to validated Driver/runtime surfaces.
- macOS system binaries can remove `DYLD_*` variables because of SIP; use a Homebrew, conda, or pyenv Python.

## Documentation

- [Getting Started](docs/getting-started.md)
- [Quick Reference](docs/quick-reference.md)
- [AI Researcher Preflight](docs/ai-researcher-preflight.md)
- [Architecture and Project Structure](docs/project-structure.md)
- [Torch Patch System](docs/phase2-custom-torch.md)
- [Reports and Validation](docs/reports-and-validation.md)
- [Distributed Simulation Usage](docs/distributed-sim-usage.md)
- [Distributed Design Notes](docs/multi-node-design.md)
- [cuBLASLt Compatibility Notes](docs/cublaslt-fix.md)
- [Error Simulation](docs/error-simulation.md)

Preview the documentation site:

```bash
python3 -m pip install -e ".[docs]"
mkdocs serve
```

## License

[MIT](LICENSE)
