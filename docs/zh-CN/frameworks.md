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
| `nvcc` 设备内核 | 支持注册和启动策略；不支持任意内核计算 | 编译 `.cu` 程序，检查启动错误及跳过内核后输出不变 |

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
CI 安装上述版本并运行全部三个框架用例。原生 smoke 在 Linux/macOS 编译并运行 C++
流程；装有 CUDA 12 工具包（`nvcc`，已验证 12.4）时，额外以 `--cudart shared` 编译
`tests/native/cuda_kernel_launch.cu`，验证 `allow`、`warn`、`error` 三种策略。
没有 `nvcc` 时明确报告跳过。

验证原生数值工作负载时使用 `--unsupported-api error`，让不支持的内核启动返回错误，
避免把未改变的输出当成正确计算。应用启动方式见[原生库拦截](native-interception.md)。
这些检查不构成任意 CUDA 内核的数值或耗时一致性保证。
