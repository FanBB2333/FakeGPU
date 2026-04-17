# Architecture

This page explains how the repository is organized and how the main runtime pieces fit together.

## Runtime flow

### 1. Launcher and Python API

- `./fgpu` is a thin shell wrapper around the same preload idea exposed by the Python package.
- `fakegpu/_api.py` resolves the build or library directory, selects which shared objects to preload for each mode, and exposes `init()`, `env()`, and `run()`.

### 2. Backend configuration

- `src/core/backend_config.hpp` parses `FAKEGPU_MODE`, `FAKEGPU_OOM_POLICY`, distributed settings, and optional real-library overrides.
- The configuration determines whether FakeGPU stays fully simulated, mixes real compute with fake device identity, or forwards to real CUDA/NCCL.

### 3. Device inventory and profiles

- `src/core/global_state.*` owns fake devices, current-device tracking, allocation maps, and runtime counters.
- `src/core/gpu_profile.*` loads GPU presets from `profiles/*.yaml`.
- CMake embeds those YAML files into generated headers at configure time, so runtime profile lookup does not depend on external files.

### 4. CUDA and NVML interception

- `src/cuda/` implements CUDA Driver and Runtime API stubs and passthrough helpers.
- `src/nvml/` implements fake NVML responses so tools and frameworks can query device state.
- In simulate mode, device memory is backed by host allocations tracked by `GlobalState`.

### 5. cuBLAS and CPU-backed compute

- `src/cublas/` provides cuBLAS and cuBLASLt compatibility.
- When `ENABLE_FAKEGPU_CPU_SIMULATION=ON`, maintained GEMM and matmul paths execute on CPU so validation can assert meaningful results instead of pure placeholder values.

### 6. Distributed simulation

- `src/nccl/` exposes the fake `libnccl.so.2` surface.
- `src/distributed/` contains communicator registration, coordinator transport, topology modeling, staging buffers, and collective execution.
- The most validated distributed path is single-host multi-process execution coordinated through Unix sockets or loopback TCP.

### 7. Monitoring and reports

- `src/monitor/monitor.cpp` dumps `fake_gpu_report.json` on shutdown.
- When distributed mode is enabled and `FAKEGPU_CLUSTER_REPORT_PATH` is set, FakeGPU also writes a cluster-level report with collective, link, and per-rank timing data.

### 8. Python torch patch layer

- `fakegpu/torch_patch.py` provides the Python-level CUDA redirection for CPU-only hosts.
- The patch uses a two-layer architecture:
  - **Base layer**: vendored upstream `FakeCudaTensor` backend (`fakegpu/_upstream.py`) provides core CUDA redirection via `torch.Tensor._make_subclass` + `__torch_function__` protocol. Tensors report `device == cuda:N` and `is_cuda == True`.
  - **Enhancement layer**: FakeGPU additions including per-device GPU profiles, memory tracking with OOM simulation, autocast dtype validation, cross-device operation guards, and terminal summary reporting.
- Error simulation (cross-device, OOM, invalid device, dtype mismatch, checkpoint, gradient) is part of the enhancement layer.
- The error simulation test suite lives in `test/test_error_*.py` with a unified runner at `test/run_error_simulation_suite.py`.
- See [Torch Patch System](phase2-custom-torch.md) for full architecture details.

## Source tree

| Path | Responsibility |
|---|---|
| `src/core/` | global state, device metadata, logging, backend selection |
| `src/cuda/` | CUDA Driver and Runtime interception |
| `src/cublas/` | cuBLAS/cuBLASLt shims and CPU-backed math |
| `src/nvml/` | fake NVML implementation |
| `src/nccl/` | fake NCCL entry points plus mode dispatch |
| `src/distributed/` | coordinator protocol, communicator state, topology, staging |
| `src/monitor/` | JSON reporting |
| `fakegpu/` | Python package, CLI, vendored FakeCudaTensor backend, and enhancement layer (GPU profiles, memory tracker, cross-device guards, error simulation) |
| `profiles/` | GPU preset YAML definitions |
| `test/` | user-facing smoke, PyTorch, DDP, and comparison scripts |
| `verification/` | lower-level probes, direct tests, and sample configs |
| `docs/` | MkDocs content |

## Build outputs

The standard build generates:

- `build/libcuda.so.1`
- `build/libcudart.so.12`
- `build/libcublas.so.12`
- `build/libnvidia-ml.so.1`
- `build/libnccl.so.2`
- `build/fakegpu-coordinator`

On macOS the corresponding `.dylib` names are produced instead.

## Profile system

- The default inventory is eight A100-class fake devices.
- You can switch to a single preset with `FAKEGPU_PROFILE` and `FAKEGPU_DEVICE_COUNT`.
- You can mix presets with `FAKEGPU_PROFILES`, for example `a100:4,h100:4` or `t4,l40s`.
- The Python API and CLI forward the same knobs.

## Good files to read first

- `README.md`
- [Getting Started](getting-started.md)
- [Reports & Validation](reports-and-validation.md)
- [Distributed Simulation Usage Guide](distributed-sim-usage.md)
- [Distributed Design Notes](multi-node-design.md)
- [Error Simulation](error-simulation.md)
- [Torch Patch System](phase2-custom-torch.md)
