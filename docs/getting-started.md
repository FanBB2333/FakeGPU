# 快速开始

这份文档面向“第一次把 FakeGPU 跑起来”。

## 1. 前置条件

- Python 3.10+
- CMake 3.14+
- Linux 或 macOS
- 如果要跑 PyTorch / DDP 相关测试，需要本地 Python 环境里可导入 `torch`

## 2. 构建项目

```bash
cmake -S . -B build
cmake --build build
```

默认会启用 CPU-backed compute，用 CPU 执行已支持的 cuBLAS / cuBLASLt 路径。

如果你只想保留 stub / no-op 行为，可以关闭 CPU simulation：

```bash
cmake -S . -B build -DENABLE_FAKEGPU_CPU_SIMULATION=OFF
cmake --build build
```

## 3. 运行方式

### 3.1 通过包装器运行

```bash
./fgpu python3 your_script.py
```

这是最推荐的入口，因为它会帮你处理 `LD_PRELOAD` / `DYLD_INSERT_LIBRARIES`。

### 3.2 在 Python 进程内动态启用

```python
import fakegpu

fakegpu.init()

import torch
print(torch.cuda.device_count())
```

`fakegpu.init()` 需要在导入 `torch` 或其他 CUDA 相关库之前调用。

## 4. 常用模式

### 4.1 计算模式

| 模式 | 说明 |
|---|---|
| `simulate` | 全部 CUDA 调用返回 fake 数据，不需要真实 GPU |
| `passthrough` | 转发到真实 CUDA 库，适合对比验证 |
| `hybrid` | 设备信息虚拟化，但计算尽量走真实 GPU |

### 4.2 分布式通信模式

| 模式 | 说明 |
|---|---|
| `disabled` | 不启用 FakeGPU 分布式层 |
| `simulate` | coordinator 执行 collective / p2p 语义 |
| `proxy` | 真实 NCCL 执行通信，FakeGPU 负责观测与报告 |
| `passthrough` | 尽量薄地转发到真实 NCCL |

## 5. 第一批建议命令

```bash
./ftest smoke
./ftest cpu_sim
./test/run_multinode_sim.sh 2
```

如果本地装好了 `torch`，再继续：

```bash
./ftest python
./test/run_ddp_multinode.sh 4
```

## 6. 关键环境变量

```bash
FAKEGPU_MODE={simulate,passthrough,hybrid}
FAKEGPU_DIST_MODE={disabled,simulate,proxy,passthrough}
FAKEGPU_OOM_POLICY={clamp,managed,mapped_host,spill_cpu}
FAKEGPU_PROFILE=a100
FAKEGPU_DEVICE_COUNT=8
FAKEGPU_CLUSTER_CONFIG=/abs/path/to/cluster.yaml
FAKEGPU_COORDINATOR_TRANSPORT={unix,tcp}
FAKEGPU_COORDINATOR_ADDR=/tmp/fakegpu.sock
FAKEGPU_CLUSTER_REPORT_PATH=/path/to/cluster-report.json
```

## 7. 下一步阅读

- [快速参考](quick-reference.md)
- [项目结构](project-structure.md)
- [分布式模拟使用说明](distributed-sim-usage.md)
