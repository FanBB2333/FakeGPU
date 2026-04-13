# 快速开始

这份文档面向“第一次把 FakeGPU 跑起来”的场景。

## 1. 前置条件

- Python 3.10+
- CMake 3.14+
- 支持 C++17 的编译器
- Linux 或 macOS
- 如果要跑 PyTorch 或 hybrid 分布式路径，需要当前 Python 环境里能导入 `torch`

## 2. 构建项目

```bash
cmake -S . -B build
cmake --build build
```

默认会启用 CPU-backed cuBLAS / cuBLASLt 路径，这样已维护的算子可以在 CPU 上执行并做结果校验。

如果你只想保留 stub / no-op 行为：

```bash
cmake -S . -B build -DENABLE_FAKEGPU_CPU_SIMULATION=OFF
cmake --build build
```

## 3. 先跑基础验证

```bash
./ftest smoke
./ftest cpu_sim
```

如果当前环境已经装好了 PyTorch：

```bash
./ftest python
```

建议接着跑当前维护中的分布式验证：

```bash
python3 verification/test_coordinator_smoke.py
python3 test/test_allreduce_correctness.py
./test/run_hybrid_multinode.sh 2
```

## 4. 运行第一条 FakeGPU 命令

最推荐的入口是包装器，它会自动设置好 preload：

```bash
./fgpu nvidia-smi
./fgpu python3 -c "import torch; print(torch.cuda.device_count())"
```

如果系统里已经有 `nvidia-smi`，那么 `./fgpu nvidia-smi` 是最直观的第一条 smoke 命令。当前仓库默认 profile 下的样例输出大致如下：

```text
Tue Mar 17 21:57:16 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.211.01             Driver Version: 570.195.03     CUDA Version: 12.8     |
|=========================================+========================+======================|
|   0  Fake NVIDIA A100-SXM4-80GB     Off |   00000000:01:00.0 Off |                  Off |
| 50%   N/A    P0              1W /  300W |       0MiB /  81920MiB |     50%      Default |
+-----------------------------------------+------------------------+----------------------+
|   1  Fake NVIDIA A100-SXM4-80GB     Off |   00000000:02:00.0 Off |                  Off |
| 50%   N/A    P0              1W /  300W |       0MiB /  81920MiB |     50%      Default |
+-----------------------------------------------------------------------------------------+
```

时间戳、Driver 行和设备数量会随环境变化。温度字段可能显示 `N/A`，因为这部分 NVML 信息目前还没有完全模拟。

也可以直接在命令行里指定 profile 和模式：

```bash
./fgpu --profile t4 --device-count 2 python3 your_script.py
./fgpu --mode hybrid --oom-policy clamp python3 your_script.py
```

## 5. 在 Python 进程内启用

如果你不想通过 `./fgpu` 启动，也可以在 Python 里尽早调用 `fakegpu.init()`。注意它必须在 `torch` 或其他 CUDA 相关库导入之前执行。

```python
import fakegpu

fakegpu.init(profile="a100", device_count=4)

import torch
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
```

也支持按设备混合 profile：

```python
fakegpu.init(devices="a100:2,h100:2")
```

## 6. 计算模式和通信模式

### 计算模式

| 模式 | 含义 |
|---|---|
| `simulate` | 设备信息、内存和计算都由 FakeGPU 处理，不需要真实 GPU |
| `hybrid` | 设备信息虚拟化，但尽量让真实 CUDA 执行计算 |
| `passthrough` | 尽量薄地转发到真实 CUDA 库 |

### 通信模式

| 模式 | 含义 |
|---|---|
| `disabled` | 不启用 FakeGPU 分布式层 |
| `simulate` | coordinator 执行 collective / p2p 语义 |
| `proxy` | 真实 NCCL 做数据面，FakeGPU 保留控制面与报告 |
| `passthrough` | 尽量薄地转发到真实 NCCL |

## 7. 关键环境变量

```bash
FAKEGPU_MODE={simulate,passthrough,hybrid}
FAKEGPU_DIST_MODE={disabled,simulate,proxy,passthrough}
FAKEGPU_OOM_POLICY={clamp,managed,mapped_host,spill_cpu}
FAKEGPU_PROFILE=a100
FAKEGPU_DEVICE_COUNT=8
FAKEGPU_PROFILES=a100:4,h100:4
FAKEGPU_CLUSTER_CONFIG=/abs/path/to/cluster.yaml
FAKEGPU_COORDINATOR_TRANSPORT={unix,tcp}
FAKEGPU_COORDINATOR_ADDR=/tmp/fakegpu.sock
FAKEGPU_CLUSTER_REPORT_PATH=/path/to/cluster-report.json
FAKEGPU_REPORT_PATH=/path/to/fake_gpu_report.json
```

## 8. 本地预览文档

```bash
python3 -m pip install -e ".[docs]"
mkdocs serve
```

## 9. 下一步阅读

- [快速参考](quick-reference.md)
- [项目结构与架构](project-structure.md)
- [报告与验证](reports-and-validation.md)
- [分布式模拟使用说明](distributed-sim-usage.md)
