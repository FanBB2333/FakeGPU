# LLM SFT 显存估算

FakeGPU 提供一组可重复的 Qwen3.5 文本 SFT 验证工具，用同一批随机 token
分别执行真实 CUDA、CPU-backed FakeCUDA 和 FakeTensor ATen 静态分析。当前
worker 测量一个完整的全参数训练步：forward、backward 和 single-tensor
AdamW update。

## 验证负载

默认配置如下：

- `Qwen3_5ForCausalLM` 文本模型，不加载视觉塔和 MTP 权重
- BF16、SDPA、batch size 1、sequence length 16
- 固定随机种子生成合法 token ID；前 8 个 label 设为 `-100`，模拟指令部分
  不参与 loss
- 全参数训练，AdamW `foreach=False`，`use_cache=False`
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

## RTX PRO 5000 实测

以下数据于 2026-07-21 在 NVIDIA RTX PRO 5000 72GB Blackwell 上测得，
软件为 PyTorch 2.8.0+cu128、CUDA 12.8、Transformers 5.12.1。真实基准使用
`torch.cuda.max_memory_allocated()`，不包含 CUDA context 和 allocator 中已
reserved 但尚未使用的空间。

| 模型 | 可训练参数 | 真实峰值 | FakeCUDA 峰值 / 误差 | 静态首步峰值 / 误差 | 静态 backward 误差 |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 752,393,024 | 6.630 GiB | 6.561 GiB / 1.045% | 6.560 GiB / 1.047% | 1.442% |
| Qwen3.5-2B | 1,881,825,088 | 15.939 GiB | 15.923 GiB / 0.101% | 15.923 GiB / 0.102% | 0.173% |

两组比较报告均通过：整体峰值误差低于 2%，静态 backward 峰值误差低于
5%。FakeCUDA 与真实 CUDA 的 loss 差分别为 0.471% 和 1.200%；这是 CPU 与
CUDA BF16 kernel 的数值差异，随机 batch 指纹和模型权重保持一致。

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

## 当前边界

CPU-backed FakeCUDA 的 forward 和整体峰值在这两组测试中较接近真实 CUDA，
但它的运行时 tracker 看不到部分 CPU kernel 内部的短生命周期 backward 临时
张量。对应的 FakeCUDA-only backward 数值会低估约 21%–26%，因此比较器将
ATen storage-liveness 静态图作为 backward 的判定值，仍保留运行时数值用于
诊断。

本次结果只覆盖全参数、单卡、BF16、single-tensor AdamW 和 Transformers 的
PyTorch fallback linear-attention 实现。LoRA/QLoRA、gradient checkpointing、
gradient accumulation、Flash Linear Attention、自定义 CUDA/Triton kernel、
量化 optimizer 和多卡 sharding 需要分别验证，不能直接套用上述误差。
