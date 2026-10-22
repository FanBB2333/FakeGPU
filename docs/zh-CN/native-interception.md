# 原生库拦截

[文档索引](README.md) · [English](../en/native-interception.md) ·
[繁體中文](../zh-TW/native-interception.md)

原生库拦截让未修改的程序加载 FakeGPU 版本的 CUDA 系列动态库。代码改不了，或者
程序直接从 C++ 调用 CUDA、不经过 PyTorch 时，这种方式很有用。

## FakeGPU 如何运行你的代码

| 路径 | 应用看到的内容 | 实际执行方式 |
|---|---|---|
| **Python FakeCUDA** | CUDA 设备、CUDA 风格 tensor、显存 API 和常见训练流程 | 已维护的 PyTorch 运算通过 `FakeCudaTensor` 在 CPU 上执行 |
| **原生库拦截** | `libcuda`、`libcudart`、`libcublas`、`libnvidia-ml` 和 `libnccl` 入口 | 选定的运算使用主机内存或 CPU 计算。不支持的行为会被分类并写入报告 |
| **分析与报告** | 显存、FLOP、Roofline、拓扑和通信报告 | 分析 ATen 图、safetensors 元数据、运行 trace、校准数据和 coordinator 事件 |

## 启动程序

先用 `scripts/build.sh` 构建原生库，再让模块启动器设置 `LD_PRELOAD` 或
`DYLD_INSERT_LIBRARIES`：

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

## 不支持的调用

不支持的原生调用可以记录下来、显示警告，或返回 `cudaErrorNotSupported` 或
`CUDA_ERROR_NOT_SUPPORTED`。

同一个策略 `FAKEGPU_UNSUPPORTED_API` 也覆盖 cuBLAS 中没有数值实现的算法、
GEMM/Lt 不支持的 dtype、布局和指针模式，以及跳过 CPU 计算的分支。设为 `warn`
时记录警告。设为 `error` 时，调用会在写入输出前返回
`CUBLAS_STATUS_NOT_SUPPORTED`。

想知道覆盖了哪些 API，可以根据能力 manifest 检查源码和已构建库的导出符号：

```bash
python3 -m fakegpu capabilities --source-root . --build-dir build --strict
```

## 库名与 CUDA 版本

Linux 上启动器 preload `libcudart.so.12`、`libcublas.so.12`、`libcuda.so.1`、
`libnvidia-ml.so.1` 和 `libnccl.so.2`。FakeGPU 的 runtime 与 cuBLAS 遵循 CUDA 12
ABI，报告 CUDA 12.0，因此原生拦截面向为 CUDA 12 构建的程序，包括
`torch.version.cuda` 以 `12.` 开头的 PyTorch。

CUDA 13 构建链接 `libcudart.so.13` 和 `libcublas.so.13`。FakeGPU 不提供这两个名字，
也不实现 CUDA 13 ABI。真实的 CUDA 13 库仍会加载，与 preload 的 FakeGPU 函数同名的
调用由 FakeGPU 接管，其余调用进入真实 runtime，因此这种运行不是受支持的模拟。PyPI
上 Linux 默认的 `torch` 2.14 wheel 依赖 CUDA 13.0。原生拦截请安装 CUDA 12 版本的
PyTorch，或使用不依赖 CUDA 库的 Python FakeCUDA。

构建目录还提供 `libcudart.so`（nvitop 在 Linux 上按这个名字加载），以及 macOS 上的
`libnvidia-ml.so.1`（nvidia-ml-py，即 `pynvml`，在 Windows 以外的平台都按这个名字
加载）。wheel 中这些名字是复制的文件。副本会作为第二个库加载，但与 FakeGPU 共享
进程状态，所以启动器下的 `pynvml` 能看到模拟 GPU 及其显存。

## 设备可见性

simulate 模式下，原生库按 CUDA driver 的规则处理 `CUDA_VISIBLE_DEVICES`：CUDA
ordinal 跟随这个掩码，NVML 和 `nvidia-smi` 仍按物理索引列出所有 GPU。每一项可以是
索引、`nvidia-smi -L` 显示的 `GPU-` UUID（或足以唯一匹配一块 GPU 的前缀），也可以是
单个 `MIG-` id。列表在第一个找不到 GPU 的项处结束，所以空字符串或 `-1` 会隐藏所有
GPU；同一块 GPU 出现两次时，`cuInit` 返回 `CUDA_ERROR_INVALID_DEVICE`。
`fakegpu.init(runtime="fakecuda")` 对模拟的 GPU 应用同样的掩码；掩码隐藏全部 GPU
时不再 patch `torch.cuda`。

`CUDA_DEVICE_ORDER=PCI_BUS_ID` 按 PCI 总线顺序枚举 GPU。CUDA 的默认值
`FASTEST_FIRST` 使用 NVIDIA 未公开的启发式规则，FakeGPU 不建模它，同样按 PCI 总线
顺序枚举。在混插不同 GPU 的真实机器上，设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 可以得到
相同顺序。

每块模拟 GPU 只有一个 UUID 和一个 PCI bus id，CUDA（`cuDeviceGetUuid`、
`cudaDeviceGetPCIBusId`、`cuDeviceGetByPCIBusId`）、NVML、`nvidia-smi` 和
`torch.cuda.get_device_properties()` 报告的都是它们。

## Hybrid 模式

原生 hybrid 模式下，CUDA runtime 和 cuBLAS 保持原生。设置了
`CUDA_VISIBLE_DEVICES`（按索引或 UUID）时会保留物理 NVML 发现，让框架能正确解析
这个选择，掩码交给真实 driver 处理。否则使用建模的 NVML 设备清单。子环境切换模式时，`fakegpu.env()` 会移除
残留的 FakeGPU preload。

## macOS

macOS System Integrity Protection 可能删除系统程序的 `DYLD_*` 环境变量。原生拦截
建议使用 Homebrew、conda 或 pyenv 的 Python。
