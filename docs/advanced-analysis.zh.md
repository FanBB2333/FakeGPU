# 校准、训练、Kernel、Trace 与拓扑分析

以下命令把新增实验转为数据文件，不需要为每个测试修改 FakeGPU runtime。

| 命令 | 输入 | 输出 |
|---|---|---|
| `fakegpu calibrate compare` | 预测与实测 JSON 报告 | 分阶段误差、区间覆盖、误差来源和安全余量/系数 |
| `fakegpu calibrate bundle` | 对比或实测报告 | 适用范围明确的工作负载校准 bundle |
| `fakegpu plan-training` | DeepSpeed、Accelerate、FSDP 或 FSDP2 配置 | 统一的分片/卸载设置和单 rank 显存时间线 |
| `fakegpu analyze-kernel` | PTX、SASS 文本或 CUDA 源码 | 指令、寄存器、共享内存、可识别 FLOP 和可选 occupancy 上限 |
| `fakegpu replay-trace` | PyTorch Profiler/Chrome、NCCL 风格或 FakeGPU 时间线 | 各 rank 计算、通信、等待、重叠、节点对、关键链路、显存和恢复 |
| `fakegpu simulate-topology` | JSON/YAML 机架、交换机、网卡和 rank 图 | ring、tree 或 hierarchical 路由与竞争报告 |

## 对比预测与实测

```bash
fakegpu calibrate compare \
  build/static-estimate.json \
  build/observed-memory.json \
  --workload tiny-sft \
  --json build/calibration-comparison.json

fakegpu calibrate bundle \
  build/calibration-comparison.json \
  --output build/workload-calibrations.json
```

两份报告都有 `memory_timeline` 时按阶段匹配，否则使用各自的标准峰值。
输出包含有符号误差、绝对误差、百分比误差、预测区间覆盖情况、可能遗漏的
分量，以及保守的 preflight 参数。这些值只能用于相同的 workload
signature、软件栈、dtype、shape 和 GPU profile。

## 导入训练配置

```bash
fakegpu plan-training deepspeed.json \
  --parameter-bytes 16000000000 \
  --activation-bytes 4000000000 \
  --world-size 8 \
  --optimizer adamw \
  --json build/training-plan.json
```

解析内容包括 ZeRO stage、参数/optimizer 卸载、精度、梯度累积、activation
checkpoint、通信重叠和 bucket 大小。Accelerate 配置可以嵌套 DeepSpeed
或 FSDP 设置；FSDP/FSDP2 的分片与 prefetch 设置会转换为同一种报告格式。

显存模型分别给出 model load、forward、backward 和 optimizer 阶段。梯度
累积不会被错误地计算成多份 gradient storage。Backend workspace、
allocator 碎片和按 module 粒度变化的 prefetch 重叠仍需校准。

## 检查生成式与原生 Kernel

```bash
fakegpu analyze-repo /path/to/project --json build/repository.json
fakegpu analyze-kernel kernel.ptx \
  --profile rtx4090 \
  --threads-per-block 256 \
  --json build/kernel.json
```

仓库分析器可以解析常见 Python import、函数别名和装饰器，并检测 Triton、
CuPy RawKernel、Numba CUDA、NVRTC、内嵌 CUDA/PTX 字符串、
`cpp_extension`、`CUDAExtension`、CMake CUDA language/toolkit、`nvcc`、
自定义 operator 注册、原生源码和已编译扩展。

PTX 分析包含静态指令及分类、声明的寄存器、静态共享内存、entry point、
可识别的标量/矩阵运算，以及 GPU profile 的资源上限。SASS 输入必须是文本
反汇编。分支和循环的实际执行次数仍需 trace 或 profile。

## 使用自定义 Operator Profile 回放 Trace

```bash
fakegpu replay-trace trace.json \
  --operator-profiles verification/data/operator_profiles.example.json \
  --topology verification/data/topology_leaf_spine.json \
  --json build/trace-replay.json
```

支持 Chrome/PyTorch `traceEvents`、通用 `events` 和 FakeGPU
`operation_timeline.entries`。事件参数可以携带 rank、源/目标 rank、字节数、
dtype、shape、显存采样、链路、`fusion_id` 和重启 generation。融合算子
profile 会排除相同 `fusion_id` 的子事件，避免重复统计 launch 和 FLOP。

报告按 rank 给出计算时间、通信时间、计算/通信重叠、可见通信等待、显式
等待、通信总量、峰值有效带宽和未跟踪时间；同时包含所有 rank 对、链路
统计、可与虚拟 `nvidia-smi` 对比的显存采样，以及失败、重试、重启和
communicator 事件。

## 模拟分层网络

```bash
fakegpu simulate-topology \
  verification/data/topology_leaf_spine.json \
  --collective all-reduce \
  --algorithm hierarchical \
  --bytes-per-rank 100000000 \
  --json build/topology-report.json
```

节点可以声明机架和多张网卡。链路可连接网卡、leaf、spine 或其它高基数
层级，并显式设置带宽和延迟。Rank 可以使用节点的全部网卡，也可以固定到
一张网卡；等价路径采用确定性的 flow hash。可选 `nvlink_domain` 会添加
分析用的域内链路。

模拟器为每次逻辑传输选路，按链路串行化同一轮的并发流量，并报告 round、
路径、竞争惩罚、rank 汇总、rank 对汇总和关键链路。它不模拟 NCCL、RDMA
或交换机缓存协议。
