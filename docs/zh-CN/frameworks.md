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
| 包含 PTX 或兼容 cubin/SASS 的 CUDA 内核（Linux） | 通过原生 Driver/Runtime API 在 CPU 上解释执行 | 包含或不含 PTX 的 CUDA C++、独立 cubin，验证双精度、warp 指令、结构体参数、共享内存、屏障、原子操作、全局变量和设备函数 |

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
`tests/native/cuda_kernel_launch.cu`，检查三种不支持 API 策略下包含 PTX 和纯 SASS 两种编译结果的数值。
`tests/native/cubin_execution.py` 额外验证 Driver 加载、参数 ABI、全局变量重定位、
指针复制、CPU cuBLAS 互操作、镜像选择及错误路径。直接 PTX 和 `nvcc -rdc=true` 链接后的 SASS 程序
均验证父内核、子内核和孙内核的数值。
Linux CI 安装 CUDA 编译器，确保该路径持续验证。

## 在 CPU 上运行自定义 CUDA 内核

Linux 构建默认启用固定版本的 [PantheonSim](https://github.com/pantheongpu/pantheonsim)
PTX 和 SASS 解释器，按设备指令执行，不依赖预设内核名称。支持 Driver API 加载的
PTX/cubin 及 Runtime 注册的 fatbin，包括 LZ4 和 Zstandard 压缩镜像。
Zstandard 镜像需要系统提供 `libzstd.so.1`。fatbin 优先选择目标版本最高的兼容 PTX；
没有兼容 PTX 时选择兼容的 cubin。

程序可以保留 PTX，也可以只包含 GPU 机器码：

```bash
# 保留 PTX。
nvcc -arch=sm_80 --cudart shared your_program.cu -o your_program
# 或为合成 Ampere profile 编译纯 SASS 程序。
nvcc -gencode arch=compute_80,code=sm_80 --cudart shared your_program.cu -o your_program
FAKEGPU_PROFILES=a100 python -m fakegpu --build-dir build --mode simulate --unsupported-api error ./your_program
```

cubin 必须匹配所选 profile：普通镜像要求计算能力主版本相同，设备次版本不低于镜像；
架构专用镜像要求精确匹配。不兼容时返回 CUDA 错误 209。上游 SASS 解码器覆盖 Turing
至 Blackwell；编译数值回归覆盖合成 Turing、Ampere、Ada 和 Hopper 目标，编译器支持时
还验证 Blackwell（`sm_100`、`sm_120`）。不能据此认为所有指令和编译器版本均已支持。早于 Turing 的 cubin 仍不支持，可使用兼容 PTX。

内核在 CPU 上同步执行，全局变量、常量和设备分配在多次启动间保留。不支持的指令和
启动特性在所有策略下都返回错误，无效句柄和越界访问也会失败；执行错误可能发生在
部分写入之后。cubin 内核返回的普通全局指针支持 CUDA 复制和填充；返回的模块相对
常量、局部和共享内存指针不支持主机复制，会明确报错。常量仍可通过 CUDA 符号 API 读写。

设备端子内核通过同一后端执行，在主机启动调用返回前完成，已验证嵌套启动，但不代表
完整设备运行时 API 均已支持。此路径不提供 GPU 性能或完整 CUDA 兼容性。
CUDA Graph、协作式启动和纹理仍不属于已验证的覆盖范围。上下文仍采用简化模型，
设备资源重置返回不支持。
框架仍需要其依赖的其他 CUDA API 已受支持；上述 TensorFlow 适配器仍是独立的 CPU 路径。

PTX/SASS 执行目前要求 Linux；macOS 保留现有原生模拟和 Python 适配器，加载这些
模块会明确返回不支持。构建要求见[开发与测试](development.md)，启动方式见
[原生库拦截](native-interception.md)。
