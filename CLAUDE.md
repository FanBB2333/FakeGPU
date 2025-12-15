# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FakeGPU is a library that intercepts NVIDIA GPU API calls (CUDA and NVML) to simulate GPU devices without physical hardware. It uses LD_PRELOAD to inject fake implementations that:
- Report fake GPU devices to applications
- Track memory allocations using system RAM
- Generate resource usage reports
- Enable debugging GPU-dependent code without actual GPUs

**Primary use case**: Testing and debugging GPU-dependent applications (especially PyTorch/ML frameworks) on systems without physical GPUs.

## Build System

The project uses CMake with a modular structure. Build commands:

```bash
# Configure and build
cmake -S . -B build
cmake --build build

# Output: build/libfake_gpu.so
```

The build creates a single shared library (`libfake_gpu.so`) from four OBJECT libraries:
- `fake_gpu_core`: Device management and global state
- `fake_gpu_nvml`: NVML API stubs
- `fake_gpu_cuda`: CUDA Runtime API stubs
- `fake_gpu_monitor`: Resource tracking and report generation

## Testing and Verification

### Smoke Test (C)
```bash
./verification/run_smoke.sh
```
Builds the library, compiles a C test binary, runs it with LD_PRELOAD, and displays the generated report.

### Python/NVML Test
```bash
./verification/run_python_test.sh
```
Tests with pynvml (Python NVML bindings). Requires conda environment named "patent" by default (configurable via `CONDA_ENV`).

### PyTorch Test
```bash
# Included in run_python_test.sh if PyTorch is installed
FAKE_GPU_LIB=./build/libfake_gpu.so LD_PRELOAD=./build/libfake_gpu.so python verification/test_pytorch.py
```

All tests generate `fake_gpu_report.json` at the project root showing memory usage statistics.

## Architecture

### Core Components

**GlobalState (src/core/global_state.{hpp,cpp})**
- Singleton managing all fake GPU devices
- Tracks memory allocations across devices
- Thread-safe with mutex protection
- Initializes 2 fake GPUs by default (configurable in global_state.cpp)

**Device (src/core/device.{hpp,cpp})**
- Represents a single fake GPU
- Tracks: name, UUID, PCI bus ID, total/used/peak memory
- Default: 16GB total memory per device

### API Stubs

**NVML Stubs (src/nvml/nvml_stubs.cpp)**
Intercepts NVIDIA Management Library calls:
- `nvmlInit()`, `nvmlShutdown()`
- `nvmlDeviceGetCount()`, `nvmlDeviceGetHandleByIndex()`
- `nvmlDeviceGetName()`, `nvmlDeviceGetUUID()`
- `nvmlDeviceGetMemoryInfo()`

**CUDA Stubs (src/cuda/cuda_stubs.cpp)**
Intercepts CUDA Runtime API calls:
- `cudaGetDeviceCount()`, `cudaSetDevice()`
- `cudaMalloc()`, `cudaFree()` - uses real malloc/free with tracking
- `cudaMemcpy()` - direct memcpy since both "device" and host use RAM
- `cudaGetDeviceProperties()`
- `cudaLaunchKernel()` - logs but does nothing (no actual computation)

**Monitor (src/monitor/monitor.cpp)**
- Static singleton that dumps `fake_gpu_report.json` on program exit
- Reports per-device memory statistics in JSON format

### Memory Management

Memory allocations via `cudaMalloc()`:
1. Allocate real system RAM with `malloc()`
2. Register allocation in GlobalState with size and device ID
3. Update device's `used_memory` and `used_memory_peak`
4. Reject allocations exceeding device's `total_memory`

On `cudaFree()`:
1. Release from GlobalState tracking
2. Update device's `used_memory`
3. Free system RAM with `free()`

### Key Design Decisions

- **No actual GPU computation**: `cudaLaunchKernel()` is a no-op stub
- **System RAM as "device memory"**: All allocations use malloc/free
- **LD_PRELOAD injection**: Library must be preloaded to intercept API calls
- **Automatic reporting**: Monitor destructor writes report on program exit
- **Fixed device properties**: 2 devices, 16GB each, Ampere architecture (compute 8.0)

## Modifying Device Configuration

To change number of devices or memory limits, edit `src/core/global_state.cpp`:
- Device count: Modify loop in `GlobalState::initialize()`
- Memory per device: Modify `Device::Device()` constructor in `src/core/device.cpp`

## Common Patterns

When adding new API stubs:
1. Add function signature to appropriate `*_defs.hpp` file
2. Implement in corresponding `*_stubs.cpp` file
3. Call `GlobalState::instance().initialize()` if needed
4. Add printf logging for debugging
5. Update device memory tracking if allocation-related

## Limitations

- No actual GPU computation (kernels are no-ops)
- PyTorch may not detect fake GPUs unless CUDA runtime is fully intercepted
- Driver API (not Runtime API) calls are not intercepted
- No multi-GPU synchronization simulation
- No stream/event simulation
