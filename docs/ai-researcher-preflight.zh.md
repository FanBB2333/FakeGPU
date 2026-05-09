# AI Researcher 提交前预检查

这页描述面向 AI researcher 的 preflight 工作流：在把训练或推理命令提交到大型 GPU 集群前，先在本地检查它能不能跑到指定阶段，是否可能 OOM，以及峰值显存离目标 GPU 规格还有多少余量。

这一版目标很窄：先回答“能不能跑”和“会不会 OOM”。FakeGPU 暂时不预测 GPU 利用率、step time、吞吐、排队时间或真实集群性能。

## 本地硬件前提

当前可用的真实校准机器是一张 NVIDIA GeForce RTX 3090 Ti，24GB 显存。

这张卡适合用来：

- 检查真实 CUDA / PyTorch / transformers 环境是否可用
- 校准能放进 24GB 的小型和中型 workload
- 对比真实 PyTorch `torch.cuda.max_memory_allocated()` 和 FakeGPU 报告
- 验证 24GB 边界内的真实 OOM 行为
- 对 fakecuda 的显存跟踪做受控 sanity check

它不能直接证明：

- 80GB A100 / H100 集群上的完整任务一定能跑
- 真实多卡 NCCL、NVLink、RDMA 或 InfiniBand 行为正确
- 集群吞吐或 GPU 利用率足够好
- simulate 或 fakecuda 模式下 CUDA kernel 的数值结果与真实 GPU 一致

所以 3090 Ti 应该被当成校准点，而不是目标集群的替代品。

## Preflight 需要回答什么

对于这样的命令：

```bash
python train.py --model qwen --batch-size 4 --seq-len 4096
```

preflight 至少要回答：

- 命令是否能启动并正确导入依赖？
- 模型加载是否完成？
- forward 是否完成？
- backward 是否完成？
- optimizer step 是否完成？
- 每张 logical GPU 的峰值显存是多少？
- 相对目标 GPU profile 还剩多少显存余量？
- 如果失败，是 OOM 还是运行时/配置错误？
- 这份显存结论的可信度是多少？

## 当前目标工作流

初版 runner 已经提供一个面向 Python 命令的统一入口：

```bash
fakegpu preflight \
  --runtime fakecuda \
  --devices a100:8 \
  --stage train_step \
  --steps 1 \
  --report-dir preflight-a100 \
  --strict \
  -- python train.py --model qwen --batch-size 4 --seq-len 4096
```

预期输出：

- `preflight_report.json`
- `preflight_report.md`
- `preflight_stdout.log`
- `preflight_stderr.log`

报告状态是以下之一：

| 状态 | 含义 |
|---|---|
| `PASS_FIT` | 指定阶段完成，未触发已跟踪 OOM。 |
| `FAIL_OOM` | workload 超过目标 profile，或运行时抛出 OOM。 |
| `FAIL_RUNTIME` | 依赖、数据、模型加载、代码或环境问题导致失败。 |
| `WARN_INCOMPLETE_TRACKING` | 命令跑完了，但显存跟踪不完整，不能给出强结论。 |

初版 fakecuda runner 会在执行 Python script、module 或 `-c` 代码前自动初始化 fakecuda。非 Python 命令仍可通过 native、hybrid 或 passthrough 模式运行，但 fakecuda 不能自动 patch。

## 推荐流程

### 1. 检查 FakeGPU 基线

```bash
./ftest smoke
./ftest cpu_sim
./ftest python
./ftest preflight_oom
```

这些命令用于确认构建、preload、报告、GPU profile、CPU-backed cuBLAS 路径和基础 PyTorch CUDA 表面可用。

### 2. 跑 fakecuda OOM probe

先用小显存 profile 跑 preflight：

```bash
fakegpu preflight \
  --runtime fakecuda \
  --devices a100-1g:1 \
  --stage forward \
  --report-dir preflight-a100-1g \
  --strict \
  -- python train.py --small-config
```

再换成目标 profile：

```bash
fakegpu preflight \
  --runtime fakecuda \
  --devices a100:8 \
  --stage forward \
  --report-dir preflight-a100 \
  --strict \
  -- python train.py --cluster-config
```

重要限制：fakecuda preflight 现在会跟踪 torch 层 tensor 生命周期、分阶段峰值、top allocations、可选 allocation stack trace，粗略区分 parameters、buffers、gradients、optimizer state、activations 和 temporaries，并处理共享 storage alias 和基础 logical-device 归属。autograd 保存的 activation 仍需要继续验证，所以通过结果只能作为提交前信号，不能证明完整集群任务一定能放下。

### 3. 用 3090 Ti 做真实校准

先在真实 GPU 上跑缩小版 workload：

```bash
python train.py --small-config
```

记录真实 PyTorch 显存：

```python
import torch

print(torch.cuda.max_memory_allocated())
print(torch.cuda.mem_get_info())
```

如果当前 CUDA 安装能被 FakeGPU 加载，再对比 passthrough 或 hybrid：

```bash
./fgpu --mode passthrough python train.py --small-config
./fgpu --mode hybrid --oom-policy clamp python train.py --small-config
```

这里不要求完全一致。目标是了解真实 3090 Ti 显存和 FakeGPU 报告在小型受控 workload 上的误差。未来 preflight 报告应该把这类误差作为 calibration context。

## Stage 标记

未来 runner 应该支持可选 Python 标记：

```python
import fakegpu

with fakegpu.stage("model_load"):
    model = load_model()

with fakegpu.stage("forward"):
    outputs = model(**batch)
    loss = outputs.loss

with fakegpu.stage("backward"):
    loss.backward()

with fakegpu.stage("optimizer_step"):
    optimizer.step()
```

没有标记时，FakeGPU 仍然可以报告进程状态和峰值显存，但失败阶段只能归到 `unknown_or_last_seen`。

## 报告要求

preflight 报告应包含：

- 命令和工作目录
- FakeGPU 版本和 git commit
- runtime mode
- 目标 GPU profile
- 本地校准 GPU，如果存在
- 状态
- 已到达阶段
- 每设备总显存、峰值显存和余量
- 当前显存按类别统计
- 最大分配列表
- 按 stage 统计的峰值显存
- warning
- error
- stdout / stderr 日志路径
- tracking confidence

建议的可信度等级：

| 等级 | 含义 |
|---|---|
| `C0_incomplete` | 命令跑了，但显存跟踪不足以判断 OOM。 |
| `C1_weight_storage` | 主要跟踪权重和显式 fake-CUDA storage。 |
| `C2_torch_tensor_lifetime` | torch tensor 生命周期跟踪足够用于 fakecuda preflight。 |
| `C3_native_cuda_allocations` | native simulate 模式下 CUDA allocation 跟踪较完整。 |
| `C4_real_gpu_calibrated` | 该类 workload 有 3090 Ti 真实校准数据。 |

## 建议下一步

下一版实现应优先做：

1. 超出可见 op output 的 autograd saved activation 覆盖。
2. 3090 Ti 小型受控 workload 校准报告。
3. 同一 workload 在小/大 profile 下的 pass-fail matrix。
4. 面向报告消费者的 `preflight_report.schema.json`。
5. 文档中明确区分 fit/no-fit 检查和性能预测。
