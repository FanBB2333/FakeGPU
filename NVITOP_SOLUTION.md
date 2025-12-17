# nvitop 与 FakeGPU 使用指南

## 问题诊断

### 症状
当使用以下命令运行nvitop时会出现问题：
```bash
LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH \
LD_PRELOAD=./build/libcudart.so.12:./build/libcuda.so.1:./build/libnvidia-ml.so.1:./build/libcublas.so.12 \
timeout 300 nvitop
```

**可能出现的问题：**
1. 终端光标消失
2. 程序段错误（Segmentation fault）
3. 终端看起来卡住

### 根本原因

nvitop的TUI（文本用户界面）模式可能调用了一些FakeGPU尚未完全实现的NVML函数，或者某些函数返回的数据结构不完整，导致nvitop在渲染界面时出现段错误。

**已验证：**
- ✓ nvitop的Python API工作正常
- ✗ nvitop的TUI模式会触发段错误

## 解决方案

### 方案1：使用Python API（推荐）

我们提供了一个nvitop包装脚本，使用Python API而不是TUI模式：

```bash
LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH \
LD_PRELOAD=./build/libcudart.so.12:./build/libcuda.so.1:./build/libnvidia-ml.so.1:./build/libcublas.so.12 \
python3 test_nvitop_wrapper.py
```

**功能：**
- 显示所有GPU设备
- 显示内存使用情况
- 显示利用率、温度、功耗
- 每秒自动刷新
- 按Ctrl+C退出

**示例输出：**
```
nvitop - 2025-12-17 10:30:45
================================================================================

GPU | Name                           |   Mem-Usage | Util |  Temp |   Power
--------------------------------------------------------------------------------
  0 | Fake NVIDIA A100-SXM4-80GB     |   0.0/ 80.0GB |  50% |  65°C |  300.0W
  1 | Fake NVIDIA A100-SXM4-80GB     |   0.0/ 80.0GB |  50% |  65°C |  300.0W
  2 | Fake NVIDIA A100-SXM4-80GB     |   0.0/ 80.0GB |  50% |  65°C |  300.0W
...
```

### 方案2：Python脚本查询

一次性查询GPU信息：

```bash
LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH \
LD_PRELOAD=./build/libcudart.so.12:./build/libcuda.so.1:./build/libnvidia-ml.so.1:./build/libcublas.so.12 \
python3 -c "
from nvitop import Device
devices = Device.all()
for i, dev in enumerate(devices):
    print(f'GPU {i}: {dev.name()}')
    mem = dev.memory_info()
    print(f'  Memory: {mem.used/1024**3:.2f}/{mem.total/1024**3:.2f} GB')
    print(f'  Util: {dev.gpu_utilization()}%')
    print()
"
```

### 方案3：恢复终端（如果已经卡住）

如果终端已经卡住，光标消失：

1. **方法A：盲输入reset命令**
   ```bash
   # 直接输入以下内容（即使看不到光标）
   reset
   ```
   然后按Enter键

2. **方法B：使用stty恢复**
   ```bash
   stty sane
   ```

3. **方法C：重新打开终端**
   如果以上方法都不行，关闭终端窗口并打开新的终端

## 技术细节

### nvitop Python API 可用功能

使用FakeGPU时，以下nvitop功能可以正常工作：

✓ **设备枚举：**
- `Device.all()` - 获取所有设备
- `Device(index)` - 获取指定设备

✓ **设备信息：**
- `dev.name()` - GPU名称
- `dev.uuid()` - GPU UUID
- `dev.pci_info()` - PCI信息

✓ **性能指标：**
- `dev.memory_info()` - 内存信息（总量、已用、空闲）
- `dev.gpu_utilization()` - GPU利用率
- `dev.memory_utilization()` - 内存利用率
- `dev.temperature()` - 温度
- `dev.power_usage()` - 功耗
- `dev.power_limit()` - 功耗限制

✗ **不支持的功能（TUI模式）：**
- 原生nvitop TUI界面（会导致段错误）

### 当前FakeGPU实现的NVML函数

FakeGPU已实现30+个NVML函数，包括：
- 设备管理：nvmlInit, nvmlShutdown, nvmlDeviceGetCount, etc.
- 内存查询：nvmlDeviceGetMemoryInfo, nvmlDeviceGetMemoryInfo_v2
- 性能查询：nvmlDeviceGetUtilizationRates
- 温度/功耗：nvmlDeviceGetTemperature, nvmlDeviceGetPowerUsage
- 时钟频率：nvmlDeviceGetClockInfo, nvmlDeviceGetMaxClockInfo

完整列表可通过以下命令查看：
```bash
nm -D ./build/libnvidia-ml.so.1 | grep ' T nvml'
```

## 推荐使用方式

### 开发和调试环境

使用Python包装脚本：
```bash
# 持续监控模式
LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH \
LD_PRELOAD=./build/libcudart.so.12:./build/libcuda.so.1:./build/libnvidia-ml.so.1:./build/libcublas.so.12 \
python3 test_nvitop_wrapper.py

# 或创建alias方便使用
alias fakegpu-nvitop='LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH LD_PRELOAD=./build/libcudart.so.12:./build/libcuda.so.1:./build/libnvidia-ml.so.1:./build/libcublas.so.12 python3 test_nvitop_wrapper.py'
```

### 脚本集成

在自动化脚本中查询GPU状态：
```python
#!/usr/bin/env python3
import os
os.environ['LD_LIBRARY_PATH'] = './build:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['LD_PRELOAD'] = './build/libcudart.so.12:./build/libcuda.so.1:./build/libnvidia-ml.so.1:./build/libcublas.so.12'

from nvitop import Device

devices = Device.all()
for dev in devices:
    # 你的监控逻辑
    pass
```

## 未来改进方向

要完全支持nvitop的TUI模式，可能需要：
1. 实现更多NVML函数（进程查询、拓扑结构等）
2. 确保所有返回的数据结构完全符合NVML规范
3. 使用valgrind等工具调试段错误的具体位置

当前的Python API方案已经能够满足大部分监控需求。

## 参考

- [nvitop官方文档](https://github.com/XuehaiPan/nvitop)
- [NVML API参考](https://docs.nvidia.com/deploy/nvml-api/)
- [FakeGPU项目CLAUDE.md](CLAUDE.md)
