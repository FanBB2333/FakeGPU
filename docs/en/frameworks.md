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
| CUDA kernels with PTX (Linux) | General CPU interpretation through native Driver/Runtime APIs | Compiled CUDA C++ plus direct PTX: arithmetic, branches, grids, local/shared memory, barriers, atomics, globals and device functions |

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
under all three unsupported-API policies. Linux CI installs the CUDA compiler
so this path cannot silently skip.

## Custom CUDA kernels on CPU

Linux builds enable a pinned [PantheonSim](https://github.com/pantheongpu/pantheonsim)
PTX interpreter. It executes device instructions without dispatching on kernel
names. Both PTX modules loaded through the Driver API and Runtime-registered
fatbins are supported, including LZ4 and Zstandard compressed PTX emitted by `nvcc`.
Zstandard images require the system `libzstd.so.1` at runtime.
For example, compile a program with PTX retained and a shared CUDA runtime:

```bash
nvcc -arch=sm_80 --cudart shared your_program.cu -o your_program
python -m fakegpu --build-dir build --mode simulate --unsupported-api error ./your_program
```

Kernel launches execute synchronously on the CPU. Module globals and constants
persist between launches. Unsupported instructions or images without PTX
return an error under every policy. Invalid handles and out-of-bounds accesses
also fail; kernel errors can occur after partial writes. The interpreter does
not supply GPU performance, complete CUDA compatibility, SASS-only cubin
execution, CUDA graphs, cooperative launches, textures, or device-side child
launches. Contexts remain simplified; device resource reset returns unsupported.
A framework still needs its other CUDA API dependencies supported.
The TensorFlow adapter above remains a separate CPU execution path.

PTX execution currently requires Linux; macOS retains the existing native
simulation and Python adapters, with PTX loads explicitly unsupported. See
[development](development.md) for build dependencies and
[native interception](native-interception.md) for application launching.
