# AI Researcher 提交前预检查

这页描述面向 AI researcher 的 preflight 工作流：在把训练或推理命令提交到大型 GPU 集群前，先在本地检查它能不能跑到指定阶段，是否可能 OOM，以及峰值显存离目标 GPU 规格还有多少余量。

这一版目标很窄：先回答“能不能跑”和“会不会 OOM”。FakeGPU 暂时不预测 GPU 利用率、step time、吞吐、排队时间或真实集群性能。

## 本地硬件前提

当前可用的真实校准机器是一张 NVIDIA RTX PRO 5000 72GB Blackwell，Compute Capability 为 12.0。

这张卡适合用来：

- 检查真实 CUDA / PyTorch / transformers 环境是否可用
- 校准能放进这张卡实际可用显存的小型和中型 workload
- 对比真实 PyTorch `torch.cuda.max_memory_allocated()` 和 FakeGPU 报告
- 验证这张卡实际显存边界内的真实 OOM 行为
- 对 fakecuda 的显存跟踪做受控 sanity check

它不能直接证明：

- 在这张卡上校准的 workload 一定能适配另一种目标 GPU profile
- 真实多卡 NCCL、NVLink、RDMA 或 InfiniBand 行为正确
- 集群吞吐或 GPU 利用率足够好
- simulate 或 fakecuda 模式下 CUDA kernel 的数值结果与真实 GPU 一致

RTX PRO 5000 只是校准点，不能替代目标集群验证。

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

重要限制：fakecuda preflight 现在会跟踪 torch 层 tensor 生命周期、PyTorch hooks 能看到的 autograd saved tensor、分阶段峰值、top allocations、可选 allocation stack trace，粗略区分 parameters、buffers、gradients、optimizer state、activations 和 temporaries，并处理共享 storage alias 和基础 logical-device 归属。CUDA 后端内部的 workspace 和 optimizer 临时分配仍可能不可见，Transformer-heavy workload 尤其明显。所以 `PASS_FIT` 只能作为提交前信号，不能证明完整集群任务一定能放下。

### 3. 用真实 GPU 做校准

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

内置校准套件可以直接运行：

```bash
./ftest real_gpu_calibration
```

内置套件包含 tensor allocation probe、torch MLP 训练步、torch Tiny Transformer 训练步、梯度累积、梯度 checkpointing、本地随机初始化的 Hugging Face tiny GPT-2 训练步，以及 PEFT LoRA tiny GPT-2 训练步。它不会下载模型权重。

这里不要求完全一致。目标是了解当前真实 GPU 的显存数据和 FakeGPU 报告在小型受控 workload 上的误差。校准套件会为当前服务器自动选择 `rtx-pro-5000-blackwell`，并写出 `build/real_gpu_calibration/calibration_real_gpu.json` 和 Markdown 报告。默认每个 real/native worker 先 warmup 1 次，再测量 3 次；报告保留 PyTorch allocated、reserved、requested 峰值分布和 NVML 显存采样，并把最大观测值作为实测上界。NVML 能识别当前 PID 时会记录进程峰值；WSL 无法提供 PID 映射时，会明确标记进程采样不可用并保留设备显存增量。每个 workload 都会运行 real CUDA、passthrough、Hybrid clamp 和 fakecuda，原生模式的结果签名必须与 real CUDA 一致。最后一个超容量 tensor probe 会验证 Hybrid clamp 能返回 `torch.cuda.OutOfMemoryError`，而不会真的耗尽物理显存。

不同校准 GPU 的报告可以直接聚合，不需要拟合一个通用倍率：

```bash
python3 verification/aggregate_real_gpu_calibrations.py \
  reports/3090ti/calibration_real_gpu.json \
  reports/pro5000/calibration_real_gpu.json \
  --output build/calibration_bundle.json \
  --markdown build/calibration_bundle.md
```

对于已有精确签名的 workload，preflight 可以使用匹配 profile 的物理显存上界：

```bash
fakegpu preflight \
  --runtime fakecuda \
  --profile rtx3090ti \
  --memory-calibration build/calibration_bundle.json \
  --calibration-workload tiny_transformer_step \
  --report-dir preflight-empirical \
  --strict \
  -- python train.py --small-config
```

只有所有目标设备都找到匹配的 profile 观测时，报告才会把可信度提升为 `C4_real_gpu_calibrated`。物理显存上界优先使用 NVML 进程峰值，把 CUDA context 和后端分配一起计入；WSL 无法提供进程数据时，会取 PyTorch allocator 峰值与 NVML 设备增量中的较大值，并记录这一数据来源。这套数据不会跨模型形状外推；batch size、序列长度、模型维度或 optimizer 配置变化后，需要生成新的 workload 签名和样本。

如需为每个维护中的 workload 分别生成 preflight 报告，可以运行：

```bash
workloads=(
  tensor_256mb
  mlp_train_step
  tiny_transformer_step
  gradient_accumulation_step
  gradient_checkpointing_step
  hf_tiny_gpt2_step
  peft_lora_tiny_step
)
for workload in "${workloads[@]}"; do
  python3 -m fakegpu preflight \
    --runtime fakecuda \
    --profile rtx-pro-5000-blackwell \
    --device-count 1 \
    --stage "$workload" \
    --report-dir "build/preflight-$workload" \
    --strict \
    -- python3 verification/calibration_real_gpu.py \
      --worker fakecuda --workload "$workload" \
      --profile rtx-pro-5000-blackwell
done
```

每份报告都会包含 workload stage、峰值显存和最终 status。

如果没有精确匹配的实测 workload，而且缺失显存更像固定的后端 workspace gap，可以使用加性 margin。比如校准报告显示 `after_backward` 大约缺 18 MiB：

```bash
fakegpu preflight \
  --runtime fakecuda \
  --devices a100:8 \
  --stage optimizer_step \
  --memory-safety-margin 18MiB \
  --report-dir preflight-a100-margin \
  --strict \
  -- python train.py --cluster-config
```

只有当校准显示 gap 会随 workload 规模增长时，再使用 `--memory-safety-factor`。使用任一 safety 选项后，报告都会同时保留原始 tracked peak 和用于 fit/OOM 判定的 estimated peak。

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
| `C4_real_gpu_calibrated` | 该类 workload 有明确记录 GPU 型号的真实校准数据。 |

## 建议下一步

下一版实现应优先做：

1. 继续减少 Transformer workload 中 CUDA 后端内部 workspace 和 optimizer 分配的低估。
2. 在当前真实校准 GPU 上增加手动大 tensor OOM probe。
3. 为更真实的 HF 和 LoRA workload 增加小/大 profile pass-fail matrix。
4. 更多把 `preflight_report.json` 作为 Slurm 提交说明附件的 workload 示例。
5. 文档中明确区分 fit/no-fit 检查和性能预测。
