# FakeGPU 使用指南

## 概述

FakeGPU 是一个虚拟GPU库，通过拦截NVIDIA GPU API调用来模拟GPU设备，无需物理硬件即可测试和调试GPU相关代码。

## 快速开始

### 1. 构建库

```bash
cmake -S . -B build
cmake --build build
```

这会生成 `build/libfake_gpu.so` 共享库。

### 2. 基本使用方法

有两种方式使用FakeGPU：

#### 方法A: 使用 LD_PRELOAD（推荐）

```bash
LD_PRELOAD=./build/libfake_gpu.so python your_script.py
```

#### 方法B: 在Python中预加载

```python
import os
from ctypes import CDLL

# 在导入任何GPU库之前加载
fake_gpu_lib = './build/libfake_gpu.so'
CDLL(fake_gpu_lib, mode=os.RTLD_GLOBAL)

# 现在可以导入GPU库
import torch  # 或其他GPU库
```

## 使用示例

### 示例1: 纯CUDA应用

运行提供的CUDA示例：

```bash
LD_PRELOAD=./build/libfake_gpu.so python cuda_example.py
```

这个示例展示：
- CUDA Runtime API的设备管理
- 内存分配和释放
- 数据传输（Host <-> Device）
- 模拟机器学习训练的内存分配模式

### 示例2: 使用pynvml

```bash
LD_PRELOAD=./build/libfake_gpu.so python demo_usage.py
```

这个示例展示：
- 使用pynvml检测GPU设备
- 查询设备信息（名称、UUID、内存）
- 直接调用CUDA Runtime API

### 示例3: PyTorch应用

```python
from ctypes import CDLL
import os

# 预加载虚拟GPU库
CDLL('./build/libfake_gpu.so', mode=os.RTLD_GLOBAL)

import torch

# 检查CUDA可用性
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"设备数量: {torch.cuda.device_count()}")

# 分配张量
device = torch.device("cuda:0")
x = torch.randn(1000, 1000, device=device)
print(f"分配的内存: {torch.cuda.memory_allocated(0) / (1024**2):.2f} MB")
```

注意：PyTorch可能使用真实的GPU驱动进行设备检测，但内存分配操作会被虚拟GPU拦截。

## 功能特性

### 支持的API

#### CUDA Runtime API
- `cudaGetDeviceCount()` - 获取设备数量
- `cudaSetDevice()` / `cudaGetDevice()` - 设备管理
- `cudaMalloc()` / `cudaFree()` - 内存分配/释放
- `cudaMemcpy()` - 数据传输
- `cudaGetDeviceProperties()` - 获取设备属性
- `cudaLaunchKernel()` - 启动kernel（no-op）

#### CUDA Driver API
- `cuInit()` - 初始化
- `cuDeviceGetCount()` / `cuDeviceGet()` - 设备查询
- `cuDeviceGetName()` / `cuDeviceGetAttribute()` - 设备信息
- `cuMemAlloc()` / `cuMemFree()` - 内存管理
- `cuMemcpy*()` - 数据传输
- `cuCtx*()` - 上下文管理

#### NVML API
- `nvmlInit()` / `nvmlShutdown()` - 初始化/关闭
- `nvmlDeviceGetCount()` - 获取设备数量
- `nvmlDeviceGetHandleByIndex()` - 获取设备句柄
- `nvmlDeviceGetName()` / `nvmlDeviceGetUUID()` - 设备信息
- `nvmlDeviceGetMemoryInfo()` - 内存信息

### 默认配置

- **设备数量**: 8个虚拟GPU
- **每个设备内存**: 80 GB（模拟NVIDIA A100）
- **计算能力**: 8.0（Ampere架构）
- **设备名称**: "Fake NVIDIA A100-SXM4-80GB"

### 内存追踪

虚拟GPU会追踪所有内存分配：
- 使用系统RAM模拟GPU内存
- 记录每个设备的内存使用情况
- 追踪峰值内存使用
- 程序结束时生成报告

## 内存使用报告

程序结束时，FakeGPU会自动生成 `fake_gpu_report.json` 文件：

```json
{
  "devices": [
    {
      "name": "Fake NVIDIA A100-SXM4-80GB",
      "uuid": "GPU-REDACTED",
      "total_memory": 85899345920,
      "used_memory_peak": 178257920,
      "used_memory_current": 0
    }
  ]
}
```

字段说明：
- `total_memory`: 设备总内存（字节）
- `used_memory_peak`: 峰值内存使用（字节）
- `used_memory_current`: 当前内存使用（字节）

## 实际应用场景

### 1. 无GPU环境测试

在没有物理GPU的机器上测试GPU代码：

```bash
# CI/CD环境
LD_PRELOAD=./build/libfake_gpu.so pytest tests/gpu_tests.py
```

### 2. 内存泄漏调试

追踪GPU内存分配，发现内存泄漏：

```bash
LD_PRELOAD=./build/libfake_gpu.so python train.py
# 检查 fake_gpu_report.json 中的 used_memory_current
# 如果不为0，说明有内存未释放
```

### 3. 多GPU模拟

在单GPU或无GPU机器上测试多GPU代码：

```python
# 虚拟GPU默认提供8个设备
for i in range(torch.cuda.device_count()):
    print(f"设备 {i}: {torch.cuda.get_device_properties(i).name}")
```

### 4. 开发GPU工具

开发和测试GPU监控、管理工具：

```python
import pynvml

pynvml.nvmlInit()
for i in range(pynvml.nvmlDeviceGetCount()):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"设备 {i} 内存使用: {info.used / info.total * 100:.1f}%")
```

## 限制和注意事项

### 功能限制

1. **不执行实际计算**
   - `cudaLaunchKernel()` 是no-op操作
   - kernel不会真正执行
   - 计算结果可能不正确

2. **PyTorch集成限制**
   - PyTorch可能使用真实GPU驱动进行设备检测
   - 在有真实NVIDIA驱动的系统上，设备属性来自真实GPU
   - 但内存操作会被虚拟GPU拦截

3. **Driver API部分支持**
   - 基本的Driver API已实现
   - 某些高级功能可能不完整

### 适用场景

适合：
- 测试代码逻辑和流程
- 调试内存分配问题
- 开发GPU相关工具
- CI/CD环境的基本测试
- 教学和演示

不适合：
- 需要实际GPU计算的场景
- 性能测试和基准测试
- 验证计算结果的正确性

## 配置和定制

### 修改设备数量

编辑 `src/core/global_state.cpp` 中的 `initialize()` 函数：

```cpp
void GlobalState::initialize() {
    // 修改这里的循环次数来改变设备数量
    for (int i = 0; i < 8; ++i) {  // 改为你想要的数量
        devices.emplace_back(i);
    }
}
```

### 修改设备内存

编辑 `src/core/device.cpp` 中的 `Device` 构造函数：

```cpp
Device::Device(int id) : device_id(id) {
    // 修改这里的值来改变内存大小
    total_memory = 80ULL * 1024 * 1024 * 1024;  // 80 GB
}
```

### 修改设备名称

编辑 `src/core/device.cpp` 中的设备名称：

```cpp
name = "Fake NVIDIA A100-SXM4-80GB";  // 改为你想要的名称
```

重新构建库以应用更改：

```bash
cmake --build build
```

## 调试和日志

### 查看详细日志

虚拟GPU会输出详细的调试信息，标记为：
- `[FakeCUDA]` - CUDA Runtime API调用
- `[FakeCUDA-Driver]` - CUDA Driver API调用
- `[FakeNVML]` - NVML API调用
- `[GlobalState]` - 全局状态管理
- `[Monitor]` - 内存监控

示例输出：

```
[FakeCUDA] cudaGetDeviceCount returning 8
[FakeCUDA] cudaMalloc allocated 104857600 bytes at 0x7f8e9c000000 on device 0
[FakeCUDA] cudaFree released 104857600 bytes from device 0
```

### 禁用日志

如果需要禁用日志输出，可以重定向stderr：

```bash
LD_PRELOAD=./build/libfake_gpu.so python your_script.py 2>/dev/null
```

## 故障排除

### 问题1: 库未加载

错误：`找不到库文件`

解决：
```bash
# 确保库已构建
ls -la build/libfake_gpu.so

# 使用绝对路径
LD_PRELOAD=/path/to/fakeGPU/build/libfake_gpu.so python script.py
```

### 问题2: PyTorch初始化失败

错误：`RuntimeError: CUDA driver error: initialization error`

原因：PyTorch需要完整的Driver API支持

解决方案：
1. 在没有真实NVIDIA驱动的环境中运行
2. 使用容器环境
3. 直接使用CUDA Runtime API而不是PyTorch

### 问题3: 符号未定义

错误：`undefined symbol: nvmlInit_v2`

原因：某些API变体未实现

解决：使用基本版本的API（如 `nvmlInit` 而不是 `nvmlInit_v2`）

## 性能考虑

- 内存分配使用系统RAM，速度取决于系统内存
- 没有实际GPU计算，不会有GPU计算开销
- 适合快速原型开发和测试
- 不适合性能基准测试

## 贡献和反馈

如果遇到问题或有改进建议，请查看项目文档或提交issue。

## 许可证

请参考项目根目录的LICENSE文件。
