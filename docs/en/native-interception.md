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

## Device visibility

In simulate mode the native libraries apply `CUDA_VISIBLE_DEVICES` the way
the CUDA driver does. CUDA ordinals follow the mask, while NVML and
`nvidia-smi` keep listing every GPU under its physical index. An entry is an
index, a `GPU-` UUID as `nvidia-smi -L` prints it (or enough of its beginning
to match one GPU), or a single `MIG-` id. The list stops at the first entry
that names no GPU, so an empty mask or `-1` hides every GPU; a GPU listed
twice makes `cuInit` fail with `CUDA_ERROR_INVALID_DEVICE`.
`fakegpu.init(runtime="fakecuda")` applies the same mask to the GPUs it
simulates and leaves `torch.cuda` unpatched when the mask hides all of them.

`CUDA_DEVICE_ORDER=PCI_BUS_ID` enumerates GPUs in PCI bus order. The CUDA
default, `FASTEST_FIRST`, uses a heuristic NVIDIA does not publish; FakeGPU
does not model it and enumerates in PCI bus order as well. On a real machine
with different GPUs, set `CUDA_DEVICE_ORDER=PCI_BUS_ID` to get the same order.

Each simulated GPU has one UUID and one PCI bus id. CUDA (`cuDeviceGetUuid`,
`cudaDeviceGetPCIBusId`, `cuDeviceGetByPCIBusId`), NVML, `nvidia-smi` and
`torch.cuda.get_device_properties()` all report them.

## Hybrid mode

In native hybrid mode, the CUDA runtime and cuBLAS stay native. If
`CUDA_VISIBLE_DEVICES` is set (by index or UUID), physical NVML discovery is
kept so frameworks resolve that selection, and the real driver applies the
mask. Otherwise the modeled NVML inventory is used. `fakegpu.env()` removes stale FakeGPU preloads when a child
environment switches modes.

## macOS

macOS System Integrity Protection can strip `DYLD_*` variables from system
binaries. Use a Homebrew, conda, or pyenv Python for native interception.
