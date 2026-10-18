# 框架兼容性

[文档索引](README.md) · [English](../en/frameworks.md) ·
[繁體中文](../zh-TW/frameworks.md)

支持范围按执行路径验证。通过以下检查，不代表所有 GPU 后端或任意 CUDA 内核都已实现。

| 路径 | 支持的执行方式 | 维护中的数值检查 |
|---|---|---|
| PyTorch FakeCUDA | CUDA 接口张量，数据与计算位于 CPU | 已有运行时和训练测试 |
| scikit-learn Array API + FakeCUDA | 部分估计器使用模拟 CUDA 张量 | StandardScaler、PCA、Ridge 与 NumPy 结果对照 |
| 普通 scikit-learn | 正常 CPU 执行 | 预处理、回归与分类 pipeline |
| TensorFlow CPU 适配器 | 显式 GPU device scope 重定向到 CPU | eager 矩阵乘、梯度、图训练及 Keras 训练 |
| CUDA C/C++ + 原生库 | 主机内存支持的分配、拷贝、stream/event 及部分 CPU cuBLAS 运算 | 编译后的 C++ GEMM 流程、内存释放、模块注册及主机线程启动配置 |
| 包含 PTX 的 CUDA 内核（Linux） | 通过原生 Driver/Runtime API 在 CPU 上解释执行 | 编译 CUDA C++ 及直接加载 PTX，验证算术、分支、网格、局部/共享内存、屏障、原子操作、全局变量和设备函数 |

## scikit-learn 使用模拟 CUDA 张量

在启动 Python 前开启 SciPy 的 Array API：

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

这里的 profile 描述模拟设备。维护中的检查使用 scikit-learn 1.8.0 和 PyTorch 2.9.1；
Array API 支持取决于具体估计器和求解器。普通 NumPy 输入的估计器仍是 CPU 计算。

## TensorFlow GPU 作用域在 CPU 上运行

```python
import tensorflow as tf
from fakegpu.tensorflow import simulate

with simulate(device_count=2):
    with tf.device("/GPU:1"):
        x = tf.constant([[1., 2.]])
        result = tf.matmul(x, tf.transpose(x))
        print(result.numpy())  # [[5.]]
```

在创建或追踪待模拟的操作之前进入适配器。它临时替换整个进程的 `tf.device`，
应在启动工作线程前进入；退出时恢复原函数，发生异常时也会恢复。作用域内追踪的图
保留 CPU 放置。TensorFlow 的设备发现及张量设备仍显示实际 CPU，不模拟 TensorFlow
GPU 显存统计、NCCL、XLA CUDA 或自定义 GPU 内核。维护中的检查使用 TensorFlow CPU 2.20.0。

## 运行验证

在 Linux 测试环境中安装可选验证依赖（macOS 将 `tensorflow-cpu` 换为
`tensorflow==2.20.0`）：

```bash
python -m pip install 'tensorflow-cpu==2.20.0' 'scikit-learn==1.8.0' 'torch==2.9.1' PyYAML
```

在源码目录中执行：

```bash
PYTHONPATH=. python -m fakegpu validate \
  --manifest tests/data/framework_compatibility.yaml \
  --report-dir build/validation-frameworks --strict
scripts/test.sh smoke
```

验证矩阵实际运行数值断言。依赖缺失会标记为跳过，`--strict` 会将其判为失败。
CI 运行三个框架用例以及 Linux/macOS 原生 C++ 流程。Linux 额外运行无需 CUDA
工具包的 `tests/native/ptx_execution.py`；有 `nvcc` 时编译
`tests/native/cuda_kernel_launch.cu`，检查三种不支持 API 策略下的实际计算结果。
Linux CI 安装 CUDA 编译器，确保该路径持续验证。

## 在 CPU 上运行自定义 CUDA 内核

Linux 构建默认启用固定版本的 [PantheonSim](https://github.com/pantheongpu/pantheonsim)
PTX 解释器，按设备指令执行，不依赖预设内核名称。支持 Driver API 加载的 PTX
模块及 Runtime 注册的 fatbin，包括 `nvcc` 的 LZ4 和 Zstandard 压缩 PTX。
加载 Zstandard 镜像时，系统需要提供 `libzstd.so.1`。
编译时保留 PTX，并使用动态 CUDA runtime，例如：

```bash
nvcc -arch=sm_80 --cudart shared your_program.cu -o your_program
python -m fakegpu --build-dir build --mode simulate --unsupported-api error ./your_program
```

内核在 CPU 上同步执行，全局变量和常量在多次启动之间保留。
不支持的指令、缺少 PTX 的镜像在所有策略下都返回错误，无效句柄和越界访问也会失败；
执行错误可能发生在部分写入之后。此路径不提供 GPU 性能或完整 CUDA 兼容性，
暂不支持纯 SASS cubin、CUDA Graph、协作式启动、纹理及设备端子内核启动。
上下文仍采用简化模型，设备资源重置会返回不支持。
框架还需要其依赖的其他 CUDA API 已被支持；上述 TensorFlow 适配器仍是独立的 CPU 路径。

PTX 执行目前要求 Linux；macOS 保留现有原生模拟和 Python 适配器，加载 PTX 会明确
返回不支持。构建要求见[开发与测试](development.md)，启动方式见[原生库拦截](native-interception.md)。
