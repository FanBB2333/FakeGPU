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
| `nvcc` device kernels | Registration and launch policy; arbitrary kernel computation is unsupported | A compiled `.cu` program verifies launch errors and unchanged output when a kernel is skipped |

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
reported as skipped and fail with `--strict`. CI installs the versions above
and runs all three framework cases. Native smoke runs the compiled C++
pipeline on Linux and macOS. With a CUDA 12 toolkit (`nvcc`, verified with
12.4), it additionally builds
`tests/native/cuda_kernel_launch.cu` with `--cudart shared` and verifies the
`allow`, `warn` and `error` policies. Without `nvcc`, that check reports a skip.

Use `--unsupported-api error` when validating native numerical workloads:
unsupported launches then fail instead of silently leaving output unchanged.
See [native interception](native-interception.md) for launching an application.
These checks do not establish CUDA-kernel numerical or timing parity.
