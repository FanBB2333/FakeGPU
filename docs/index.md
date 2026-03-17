# FakeGPU

FakeGPU is a CUDA, cuBLAS, NVML, and NCCL interception layer for environments where you either do not have GPUs or do not want to depend on a real cluster during early validation. It focuses on device discovery, memory flow, selected math paths, distributed control flow, and observability.

English is the default documentation language. A Simplified Chinese translation is available from the language switcher in the header.

## What FakeGPU is good at

- Running CUDA-facing code paths on machines without GPUs
- Exposing configurable fake GPU inventories to frameworks such as PyTorch
- Executing supported cuBLAS/cuBLASLt operators on CPU for correctness-oriented smoke coverage
- Simulating NCCL-style collective and point-to-point flows on a single host with multiple ranks
- Emitting per-device and cluster-level JSON reports for memory, IO, FLOPs, and communication activity

## What FakeGPU is not trying to be

- A protocol-level recreation of NCCL, NVLink, RDMA, or InfiniBand internals
- A substitute for full numerical parity across arbitrary CUDA kernels
- A benchmark tool for predicting exact production-cluster performance

## Current code state

The repository currently ships:

- Fake `libcuda`, `libcudart`, `libcublas`, `libnvidia-ml`, and `libnccl` shared libraries
- A Python package and CLI wrapper (`fakegpu`, `./fgpu`) that manage preloading for you
- Embedded GPU profiles compiled from `profiles/*.yaml`
- A coordinator process for single-host multi-process distributed simulation
- JSON reporting for both single-process device activity and distributed cluster activity

## Validation baseline

These entry points are part of the maintained baseline and are good first checks after a build:

```bash
./ftest smoke
./ftest cpu_sim
./ftest python
```

In practice this covers:

- library preload and fake device discovery
- multi-architecture profile exposure
- pointer attribute and memory-type tracking
- CPU-backed cuBLAS/cuBLASLt correctness checks
- basic PyTorch CUDA tensor, matmul, and device-property paths

## Recommended mode pairs

| Goal | Compute mode | Distributed mode |
|---|---|---|
| Bring up CUDA or PyTorch code on a machine without GPUs | `simulate` | `disabled` |
| Simulate multi-rank or multi-node communication on one host | `simulate` | `simulate` |
| Keep local compute real, but virtualize cross-node communication | `hybrid` | `simulate` |
| Compare against real NCCL behavior with extra reporting | `hybrid` | `proxy` or `passthrough` |

For first-time setup, use:

```bash
FAKEGPU_MODE=simulate
FAKEGPU_DIST_MODE=simulate
```

## Architecture in one view

1. `./fgpu` or `fakegpu.init()` resolves the built libraries and sets the correct preload variables.
2. `BackendConfig` reads `FAKEGPU_*` environment variables and chooses compute and distributed modes.
3. `GlobalState` lazily instantiates fake devices from the compiled-in YAML profiles.
4. CUDA/NVML stubs answer device queries and memory operations; supported cuBLAS/cuBLASLt paths can run on CPU.
5. The fake NCCL layer and coordinator synchronize communicators, collectives, and point-to-point operations.
6. The monitor writes `fake_gpu_report.json` and, when enabled, a cluster-level communication report.

## Documentation map

- [Getting Started](getting-started.md)
- [Quick Reference](quick-reference.md)
- [Architecture](project-structure.md)
- [Reports & Validation](reports-and-validation.md)
- [Distributed Simulation Usage Guide](distributed-sim-usage.md)
- [Distributed Design Notes](multi-node-design.md)
- [cuBLASLt Compatibility Notes](cublaslt-fix.md)
