# Native interception

[Documentation index](README.md) · [简体中文](../zh-CN/native-interception.md) ·
[繁體中文](../zh-TW/native-interception.md)

Native interception lets an unmodified program load FakeGPU's versions of the
CUDA-family libraries. It is useful when you cannot change the code, or when
the program calls CUDA from C++ instead of PyTorch.

## How FakeGPU runs your code

| Path | What the application sees | What actually runs |
|---|---|---|
| **Python FakeCUDA** | CUDA devices, CUDA-looking tensors, memory APIs, and common training flows | Maintained PyTorch operations execute on CPU through `FakeCudaTensor` |
| **Native interception** | `libcuda`, `libcudart`, `libcublas`, `libnvidia-ml`, and `libnccl` entry points | Selected operations use host memory or CPU math. Unsupported behavior is classified and reported |
| **Analysis and reporting** | Memory, FLOP, roofline, topology, and communication reports | ATen graphs, safetensors metadata, runtime traces, calibration data, and coordinator events are analyzed |

## Launch a program

Build the native libraries with `scripts/build.sh`, then let the module
launcher set up `LD_PRELOAD` or `DYLD_INSERT_LIBRARIES`:

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

## Unsupported calls

Unsupported native calls can be recorded, reported as a warning, or returned
as `cudaErrorNotSupported` or `CUDA_ERROR_NOT_SUPPORTED`.

The same policy, `FAKEGPU_UNSUPPORTED_API`, covers cuBLAS algorithms without a
numeric implementation, unsupported GEMM/Lt dtypes, layouts, or pointer modes,
and skipped CPU computation. With `warn` a warning is recorded. With `error`
the call returns `CUBLAS_STATUS_NOT_SUPPORTED` before writing output.

To check which APIs are covered, audit the source and built exports against
the capability manifest:

```bash
python3 -m fakegpu capabilities --source-root . --build-dir build --strict
```

## Library names and CUDA versions

On Linux the launcher preloads `libcudart.so.12`, `libcublas.so.12`,
`libcuda.so.1`, `libnvidia-ml.so.1` and `libnccl.so.2`. FakeGPU's runtime and
cuBLAS follow the CUDA 12 ABI and report CUDA 12.0, so native interception
targets programs built for CUDA 12, including PyTorch builds whose
`torch.version.cuda` starts with `12.`.

A CUDA 13 build links `libcudart.so.13` and `libcublas.so.13`. FakeGPU
provides neither name nor the CUDA 13 ABI. The real CUDA 13 libraries still
load, preloaded FakeGPU functions take over the calls they share a name with,
and the other calls reach the real runtime, so such a run is not a supported
simulation. The default Linux `torch` 2.14 wheels on PyPI depend on CUDA
13.0. Install a CUDA 12 build of PyTorch for native interception, or use
Python FakeCUDA, which does not depend on the CUDA libraries.

The build directory also links `libcudart.so`, which nvitop loads on Linux,
and on macOS `libnvidia-ml.so.1`, which nvidia-ml-py (`pynvml`) loads on every
platform except Windows. The wheel ships copies of these names. A copy loads
as a second library, but it shares FakeGPU's process state, so `pynvml` under
the launcher sees the simulated GPUs and their memory.

## Hybrid mode

In native hybrid mode, the CUDA runtime and cuBLAS stay native. If
`CUDA_VISIBLE_DEVICES` is set (by index or UUID), physical NVML discovery is
kept so frameworks resolve that selection. Otherwise the modeled NVML
inventory is used. `fakegpu.env()` removes stale FakeGPU preloads when a child
environment switches modes.

## macOS

macOS System Integrity Protection can strip `DYLD_*` variables from system
binaries. Use a Homebrew, conda, or pyenv Python for native interception.
