# Framework compatibility

[Documentation index](README.md) · [简体中文](../zh-CN/frameworks.md) ·
[繁體中文](../zh-TW/frameworks.md)

Support is verified per execution path. Passing these checks does not imply
that every GPU backend or arbitrary CUDA kernel is implemented.

| Path | Supported execution | Maintained numerical checks |
|---|---|---|
| PyTorch FakeCUDA | CUDA-facing tensors backed by CPU data | Existing runtime and training suites |
| scikit-learn Array API + FakeCUDA | Selected estimators consume simulated CUDA tensors | StandardScaler, PCA and Ridge against NumPy results |
| Ordinary scikit-learn | Normal CPU execution | Preprocessing, regression and classification pipelines |
| TensorFlow CPU adapter | Explicit GPU device scopes redirected to CPU | Eager matmul, gradients, traced training and Keras training |
| CUDA C/C++ + native libraries | Host-backed allocation, copies, streams/events and selected CPU cuBLAS operations | Compiled C++ GEMM pipeline, allocation cleanup, module registration and host-thread launch configuration |
| CUDA kernels with PTX or compatible cubin/SASS (Linux) | CPU interpretation through native Driver/Runtime APIs | CUDA C++ with/without PTX, standalone cubins, FP64, warp intrinsics, structured parameters, shared memory, barriers, atomics, globals and device functions |

## PyTorch CUDA memory API

FakeCUDA models PyTorch's CUDA caching allocator
(`c10/cuda/CUDACachingAllocator.cpp`, followed for torch 2.9 to 2.14) and runs
torch's own `torch.cuda.memory` functions on top of it. `memory_allocated()`,
`memory_reserved()`, `memory_stats()`, `memory_summary()`, `memory_snapshot()`
and `torch.cuda.memory._snapshot()` report what the allocator would hold.
Allocation traces compared with a real Ampere GPU (sm_80) under torch 2.9.1
and 2.10.0 match in allocated and reserved bytes, segment layout and
allocator counters.

- `PYTORCH_CUDA_ALLOC_CONF`, and from torch 2.10 `PYTORCH_ALLOC_CONF`, are
  parsed as the installed torch parses them. `max_split_size_mb`,
  `max_non_split_rounding_mb`, `roundup_power2_divisions`,
  `garbage_collection_threshold`, `expandable_segments`,
  `large_segment_size_mb` (torch 2.11+) and `per_process_memory_fraction`
  (torch 2.10+) change the modeled memory. A malformed setting raises torch's
  error at `torch.cuda.init()` and on allocation.
- Each stream has its own cached blocks. After `Tensor.record_stream()` a
  freed block stays in use until the next allocation; FakeCUDA work finishes
  immediately.
- `torch.cuda.memory._record_memory_history()` records allocation, free and
  segment events with Python frames. There are no C++ frames.
- `torch.cuda.mem_get_info()` leaves out a modeled CUDA context of 601.5 MiB.
  This is total minus free minus reserved memory after the first cuBLAS call,
  measured once on one Ampere GPU (sm_80) with driver 580 and torch 2.9.1.
  Other GPUs, drivers and libraries hold different amounts. Set
  `FAKEGPU_CUDA_CONTEXT_BYTES` to your own measurement or to `0`.
  Allocations can still use the whole profile memory.
- Not modeled: the `cudaMallocAsync` backend (the setting is accepted and
  reported, but the native allocator stays in use), private memory pools and
  CUDA graphs, the pinned host allocator (its statistics stay zero), and the
  sentence about process memory that out-of-memory errors add when NVML is
  available. With `FAKEGPU_MEMORY_TRACKING=0` allocations are not tracked,
  and the API reports an empty allocator on the profiled devices.

## scikit-learn with simulated CUDA tensors

Enable SciPy's Array API support before starting Python:

```bash
SCIPY_ARRAY_API=1 python your_script.py
```

```python
import fakegpu
fakegpu.init(runtime="fakecuda", profile="a100", device_count=2)

import sklearn
import torch
from sklearn.preprocessing import StandardScaler

x = torch.asarray([[1., 2.], [3., 4.]], device="cuda:1")
with sklearn.config_context(array_api_dispatch=True):
    result = StandardScaler().fit_transform(x)
assert result.device == torch.device("cuda:1")
```

The profile describes a simulated device. The maintained checks use
scikit-learn 1.8.0 and PyTorch 2.9.1; Array API support depends on the estimator
and solver. Ordinary NumPy-backed estimators remain CPU computations.

## TensorFlow device scopes on CPU

```python
import tensorflow as tf
from fakegpu.tensorflow import simulate

with simulate(device_count=2):
    with tf.device("/GPU:1"):
        x = tf.constant([[1., 2.]])
        result = tf.matmul(x, tf.transpose(x))
        print(result.numpy())  # [[5.]]
```

Enter the adapter before creating or tracing the operations to simulate.
It temporarily replaces `tf.device` process-wide; enter it before starting
worker threads. Exiting restores the original function, including on errors.
Graphs traced inside retain CPU placement. TensorFlow discovery and tensor
devices still report the actual CPU placement. The adapter does not emulate
TensorFlow GPU memory accounting, NCCL, XLA CUDA or custom GPU kernels.
The maintained check uses TensorFlow CPU 2.20.0.

## Run the checks

Install the optional validation dependencies on Linux in your test environment
(on macOS, use `tensorflow==2.20.0` in place of `tensorflow-cpu`):

```bash
python -m pip install 'tensorflow-cpu==2.20.0' 'scikit-learn==1.8.0' 'torch==2.9.1' PyYAML
```

From a checkout:

```bash
PYTHONPATH=. python -m fakegpu validate \
  --manifest tests/data/framework_compatibility.yaml \
  --report-dir build/validation-frameworks --strict
scripts/test.sh smoke
```

The manifest runs actual numerical assertions. Missing dependencies are
reported as skipped and fail with `--strict`. CI runs all three framework cases
and the native C++ pipeline on Linux and macOS. Linux additionally executes
`tests/native/ptx_execution.py` without requiring a CUDA toolkit. With `nvcc`,
smoke compiles `tests/native/cuda_kernel_launch.cu` and checks actual results
under all three unsupported-API policies, with PTX and again with SASS-only fatbins.
`tests/native/cubin_execution.py` also checks Driver loading, parameter ABI,
global relocations, pointer copies, CPU cuBLAS interoperability, image selection
and errors using real cubins.
Parent, child and grandchild launches are validated using direct PTX and linked
SASS (`nvcc -rdc=true`) programs.
Linux CI installs the CUDA compiler
so this path cannot silently skip.

## Custom CUDA kernels on CPU

Linux builds enable pinned [PantheonSim](https://github.com/pantheongpu/pantheonsim)
PTX and SASS interpreters. They execute device instructions without dispatching
on kernel names. Driver-loaded PTX/cubins and Runtime-registered fatbins work,
including LZ4 and Zstandard compression. Zstandard images require the system
`libzstd.so.1` at runtime. A fatbin selects the newest compatible PTX target;
when none is present, it selects a compatible cubin.

A program can retain PTX or contain only GPU machine code:

```bash
# Retain PTX for targets compatible with the selected simulated device.
nvcc -arch=sm_80 --cudart shared your_program.cu -o your_program
# Or build a SASS-only program for a synthetic Ampere profile.
nvcc -gencode arch=compute_80,code=sm_80 --cudart shared your_program.cu -o your_program
FAKEGPU_PROFILES=a100 python -m fakegpu --build-dir build --mode simulate --unsupported-api error ./your_program
```

Cubins must match the selected profile's architecture: ordinary images require
the same compute-capability major and an equal or newer minor; architecture-specific
images require an exact match. Incompatible images return CUDA error 209.
The upstream SASS decoder covers Turing through Blackwell; FakeGPU's compiled
numerical regression checks cover synthetic Turing, Ampere, Ada and Hopper
targets, plus Blackwell (`sm_100` and `sm_120`) when the compiler supports them.
This is not validation of every instruction or compiler version on those architectures.
Pre-Turing cubins remain unsupported; compatible PTX can still be used.

Kernel launches execute synchronously on the CPU. Globals, constants and
device allocations persist between launches. Unsupported instructions and
launch features return errors under every policy. Invalid handles and
out-of-bounds accesses also fail; execution errors may occur after partial
writes. Ordinary global pointers returned by cubin kernels can be copied and
filled through CUDA APIs. Module-relative constant/local/shared pointers
returned from a kernel cannot be used for host copies; those return unsupported.
Constants remain accessible through CUDA symbol APIs.

This path does not supply GPU performance or complete CUDA compatibility.
Device-side child launches execute through the same backend and complete before
the host launch returns. This covers nested launches, not the entire device-runtime
API. CUDA graphs, cooperative launches and textures remain outside validated coverage. Contexts remain simplified;
device resource reset returns unsupported. A framework still needs its other
CUDA API dependencies supported. The TensorFlow adapter above remains a
separate CPU execution path.

PTX/SASS execution currently requires Linux; macOS retains the existing native
simulation and Python adapters, with these module loads explicitly unsupported.
See [development](development.md) for build dependencies and
[native interception](native-interception.md) for application launching.
