# LLM SFT 显存估算

FakeGPU 提供一组可重复的 Qwen3.5 文本 SFT 验证工具，用同一批随机 token
分别执行真实 CUDA、CPU-backed FakeCUDA 和 FakeTensor ATen 静态分析。worker
支持全参数或 LoRA、gradient checkpointing、gradient accumulation，并测量
forward、backward 和 single-tensor AdamW update。

## 验证负载

默认配置如下：

- `Qwen3_5ForCausalLM` 文本模型，不加载视觉塔和 MTP 权重
- BF16、SDPA、batch size 1、sequence length 16
- 固定随机种子生成合法 token ID；前 8 个 label 设为 `-100`，模拟指令部分
  不参与 loss
- 默认使用全参数训练；可选择 LoRA `all-linear`
- AdamW `foreach=False`，`use_cache=False`
- 显式文本 position IDs，使真实执行和静态图走相同的 Qwen3.5 混合
  linear/full-attention 分支

随机 token 不是有语义的训练语料，但在固定模型分支后，显存主要由 tensor
shape、dtype、参数可训练性和 optimizer 决定。因此它适合验证显存，不用于
评价训练质量。

## 执行三个后端

下面的命令在相同 Python 环境中执行。FakeCUDA 命令隐藏物理 GPU，避免误用
真实 CUDA：

```bash
PYTHON=/home/l1ght/anaconda3/envs/torch/bin/python
MODEL=/home/l1ght/models/Qwen/Qwen3.5-0.8B

$PYTHON verification/qwen_sft_memory_worker.py \
  --mode real \
  --model-dir "$MODEL" \
  --sequence-length 16 \
  --output build/qwen-sft/0.8b-real.json

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
$PYTHON verification/qwen_sft_memory_worker.py \
  --mode fakecuda \
  --profile rtx-pro-5000-blackwell \
  --model-dir "$MODEL" \
  --sequence-length 16 \
  --output build/qwen-sft/0.8b-fake.json

$PYTHON verification/qwen_sft_memory_worker.py \
  --mode static \
  --model-dir "$MODEL" \
  --sequence-length 16 \
  --output build/qwen-sft/0.8b-static.json

$PYTHON verification/compare_qwen_sft_memory.py \
  --real build/qwen-sft/0.8b-real.json \
  --fakecuda build/qwen-sft/0.8b-fake.json \
  --static build/qwen-sft/0.8b-static.json \
  --output build/qwen-sft/0.8b-comparison.json
```

将 `MODEL` 和输出文件名替换为 `Qwen3.5-2B` 即可复测 2B 模型。比较器会检查
模型配置、参数量、随机 batch 指纹和显存误差，另行生成 Markdown 表格。

以下选项可以组合使用：

```bash
# LoRA rank 8
--training-method lora --lora-rank 8

# non-reentrant gradient checkpointing
--gradient-checkpointing

# 两个 microbatch 后执行一次 optimizer update
--gradient-accumulation-steps 2
```

多个 case 完成后，可以生成统一矩阵：

```bash
$PYTHON verification/summarize_qwen_sft_matrix.py \
  --input-dir build/qwen-sft-matrix \
  --output build/qwen-sft-matrix/matrix.json
```

## RTX PRO 5000 配置矩阵

以下数据于 2026-07-21 在 NVIDIA RTX PRO 5000 72GB Blackwell 上测得，
软件为 PyTorch 2.8.0+cu128、CUDA 12.8、Transformers 5.12.1。真实基准使用
`torch.cuda.max_memory_allocated()`，不包含 CUDA context 和 allocator 中已
reserved 但尚未使用的空间。

| Case | 方法 | Checkpoint | Accum. | Seq. | 可训练参数 | 真实峰值 | 静态误差 | FakeCUDA 误差 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0.8B full | full | 否 | 2 | 16 | 752,393,024 | 6.626 GiB | 0.989% | 0.985% |
| 0.8B full | full | 否 | 1 | 128 | 752,393,024 | 6.687 GiB | 1.121% | 1.119% |
| 0.8B full | full | 是 | 1 | 128 | 752,393,024 | 6.679 GiB | 1.006% | 1.006% |
| 0.8B LoRA | LoRA r8 | 否 | 1 | 16 | 5,411,328 | 1.756 GiB | 1.921% | 2.439% |
| 0.8B LoRA | LoRA r8 | 否 | 1 | 128 | 5,411,328 | 2.696 GiB | 1.271% | — |
| 0.8B LoRA | LoRA r8 | 是 | 1 | 128 | 5,411,328 | 1.876 GiB | 1.506% | — |
| 2B full | full | 否 | 2 | 16 | 1,881,825,088 | 15.939 GiB | 0.102% | — |
| 2B LoRA | LoRA r8 | 否 | 1 | 16 | 8,409,600 | 3.868 GiB | 0.266% | 0.500% |
| 2B LoRA | LoRA r8 | 否 | 1 | 128 | 8,409,600 | 4.959 GiB | 0.359% | — |
| 2B LoRA | LoRA r8 | 是 | 1 | 128 | 8,409,600 | 3.974 GiB | 0.281% | — |

10 个 case 全部通过。静态整体误差为 0.102%–1.921%；已执行 FakeCUDA 的
case 误差为 0.500%–2.439%。LoRA checkpointing 在 sequence 128 下将 0.8B
峰值减少 30.4%，将 2B 峰值减少 19.9%。全参数 AdamW 的整体峰值由 optimizer
update 主导，因此 checkpointing 主要降低 forward 峰值，对整体峰值影响很小。

## Ampere 与 Blackwell

RTX 3090 Ti（Compute Capability 8.6、PyTorch 2.12.1+cu130）复测结果如下：

| Case | RTX 3090 Ti 真实峰值 | RTX PRO 5000 真实峰值 | 静态误差 |
|---|---:|---:|---:|
| 0.8B full，sequence 16 | 6.630 GiB | 6.630 GiB | 1.047% |
| 0.8B LoRA r8 + checkpoint，sequence 128 | 1.876 GiB | 1.876 GiB | 1.506% |

两个 CUDA allocator 路径的峰值几乎相同，当前 shape/storage 估算可以跨
Ampere 8.6 与 Blackwell 12.0 使用。backend workspace 仍需针对新的 attention
实现和超出校准范围的 shape 单独验证。

## 首步与稳态显存

AdamW 的 moment state 在第一次 `optimizer.step()` 中才创建。静态估算器
现在分别给出：

- `first_step_graph_phase_peak_bytes`：首步 forward/backward 图，不提前加入
  optimizer state
- `first_step_estimated_peak_bytes`：首个完整训练步的最大值
- `graph_phase_peak_bytes`：已有 optimizer state 时的稳态计算图峰值
- `steady_state_estimated_peak_bytes`：稳态完整训练步峰值

这种区分避免把尚未创建的两份 Adam moment 错加到首步 backward，也比
“参数 + 梯度 + 两份 optimizer state”这一固定公式更能反映 activation、
别名 storage、算子临时结果和 workspace 生命周期。

gradient accumulation 使用 eager autograd 的原地累加模型：已有梯度保持
常驻，额外峰值取最大的单个可训练参数梯度，而不重复计算全部梯度。0.8B
和 2B 的 accumulation 2 backward 误差分别为 1.305% 和 0.157%。

PyTorch checkpoint 的 saved-tensor hook 目前不能直接嵌套在 `torch.func.grad`
图捕获中。全参数训练保留由 tied weight 和参数梯度决定的 graph floor；LoRA
训练则从未 checkpoint 的图中移除 decoder activation，只保留参数、输出、
loss 临时张量、梯度、optimizer 和已建模 workspace。无法识别 loss operator
时，报告会明确标记为 `upper_bound`。

## 当前边界

CPU-backed FakeCUDA 的 forward 和整体峰值在这两组测试中较接近真实 CUDA，
但它的运行时 tracker 看不到部分 CPU kernel 内部的短生命周期 backward 临时
张量。对应的 FakeCUDA-only backward 数值会低估约 21%–26%，因此比较器将
ATen storage-liveness 静态图作为 backward 的判定值，仍保留运行时数值用于
诊断。

当前结果覆盖全参数与 LoRA、单卡、BF16、single-tensor AdamW、gradient
checkpointing、gradient accumulation 2，以及 Transformers 的 PyTorch
fallback linear-attention 实现。QLoRA、Flash Linear Attention、自定义
CUDA/Triton kernel、量化 optimizer 和多卡 sharding 仍需分别验证，不能直接
套用上述误差。
