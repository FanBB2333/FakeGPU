# LLM SFT 显存估算

FakeGPU 提供一组可重复的 Qwen3.5 文本 SFT 验证工具，用同一批随机 token
分别执行真实 CUDA、CPU-backed FakeCUDA 和 FakeTensor ATen 静态分析。worker
支持全参数、LoRA 和 packed-NF4 QLoRA 参考实现，也支持 gradient
checkpointing 与 gradient accumulation，并测量 forward、backward 和
single-tensor AdamW update。

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

训练方式选择 LoRA 或 QLoRA 之一，再按需组合 checkpointing 与 accumulation：

```bash
# LoRA rank 8
--training-method lora --lora-rank 8

# 无外部量化依赖的 NF4 QLoRA 参考实现
--training-method qlora --lora-rank 8

# 对第一层 scale 再做 8-bit 量化
--training-method qlora --lora-rank 8 --quantization-double-quantization

# non-reentrant gradient checkpointing
--gradient-checkpointing

# 两个 microbatch 后执行一次 optimizer update
--gradient-accumulation-steps 2
```

### 原生 NF4 QLoRA 参考实现

`--training-method qlora` 用于验证量化训练显存，不依赖额外量化包。它会将
输出 embedding 之外的目标 linear 权重压成每字节两个 NF4 code，每 64 个
权重保存一个 FP32 absmax。LoRA A/B 参数保持 FP32，与 PEFT 默认的 adapter
dtype 一致；optimizer 仍为 single-tensor AdamW。

`--quantization-double-quantization` 会进一步压缩 scale：先减去第一层 absmax
的均值，再使用 bitsandbytes 的 signed dynamic 8-bit map 编码；每 256 个
scale 保存一个第二层 FP32 absmax，并保留一个 FP32 offset。该格式参照
[bitsandbytes 官方实现](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/functional.py)
与 [QLoRA 论文](https://arxiv.org/abs/2305.14314)。论文给出的平均节省量约为
每个参数 0.37 bit。

自定义 autograd function 每次只反量化一层权重。backward 计算 input gradient
时重新反量化，不会让所有 BF16 权重矩阵同时常驻。静态分析先捕获等价的
PEFT LoRA ATen 图，再用真实 packed storage 替换冻结权重，并加入最大单层
反量化 workspace。

这条路径可以验证量化权重、adapter gradient、optimizer state 与反量化临时
张量的生命周期。它不是 bitsandbytes kernel 实现：NF4 与二级 scale lookup
使用普通 PyTorch 算子，没有 fused 4-bit matmul、paged optimizer，也不代表
bitsandbytes allocator 的行为。FakeCUDA 报告中的 `runtime_memory_phases` 保留
CPU tracker 原始值，`memory_phases` 则在计算阶段加入已知的单层反量化
workspace。

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

## 原生 NF4 QLoRA 矩阵

以下数据使用同一套 RTX PRO 5000 软件环境。静态结果标记为 `analytical`，
因为 packed storage 替换与反量化 workspace 是在 LoRA ATen 图上解析加入的。

### FP32 scale 直接存储

| Case | Checkpoint | Seq. | 可训练参数 | 真实峰值 | 静态峰值 | 静态误差 | FakeCUDA 误差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.8B QLoRA | 否 | 16 | 5,411,328 | 1.072 GiB | 1.079 GiB | 0.698% | 0.150% |
| 0.8B QLoRA | 否 | 128 | 5,411,328 | 2.007 GiB | 2.019 GiB | 0.628% | — |
| 0.8B QLoRA | 是 | 128 | 5,411,328 | 1.192 GiB | 1.205 GiB | 1.085% | — |
| 2B QLoRA | 否 | 16 | 8,409,600 | 2.039 GiB | 2.067 GiB | 1.386% | — |
| 2B QLoRA | 是 | 128 | 8,409,600 | 2.136 GiB | 2.172 GiB | 1.689% | — |

### 8-bit 二级 scale 存储

| Case | Checkpoint | Seq. | 可训练参数 | 真实峰值 | 静态峰值 | 静态误差 | FakeCUDA 误差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.8B QLoRA | 否 | 16 | 5,411,328 | 1.051 GiB | 1.058 GiB | 0.700% | 0.165% |
| 0.8B QLoRA | 否 | 128 | 5,411,328 | 1.986 GiB | 1.998 GiB | 0.629% | — |
| 0.8B QLoRA | 是 | 128 | 5,411,328 | 1.171 GiB | 1.184 GiB | 1.094% | — |
| 2B QLoRA | 否 | 16 | 8,409,600 | 1.979 GiB | 2.008 GiB | 1.422% | — |
| 2B QLoRA | 是 | 128 | 8,409,600 | 2.077 GiB | 2.113 GiB | 1.732% | — |

| 模型 | 量化权重数 | 直接存储 | 直接存储 bit/权重 | 二级量化存储 | 二级量化 bit/权重 | 节省存储 |
|---|---:|---:|---:|---:|---:|---:|
| 0.8B | 497,614,848 | 280,098,816 B | 4.503 | 257,085,816 B | 4.133 | 23,013,000 B |
| 2B | 1,372,717,056 | 772,343,808 B | 4.501 | 708,524,040 B | 4.129 | 63,819,768 B |

二级量化对 0.8B 和 2B 分别节省 0.370 与 0.372 bit/权重。固定随机 batch
上的 SFT loss 与 FP32 scale 版本相比，最大变化小于 0.018。0.8B 短序列还
完成了 FakeCUDA 的完整 optimizer step，全部比较项在默认误差阈值内通过。

## Ampere 与 Blackwell

RTX 3090 Ti（Compute Capability 8.6、PyTorch 2.12.1+cu130）复测结果如下：

| Case | RTX 3090 Ti 真实峰值 | RTX PRO 5000 真实峰值 | 静态误差 |
|---|---:|---:|---:|
| 0.8B full，sequence 16 | 6.630 GiB | 6.630 GiB | 1.047% |
| 0.8B LoRA r8 + checkpoint，sequence 128 | 1.876 GiB | 1.876 GiB | 1.506% |

两个 CUDA allocator 路径的峰值几乎相同，当前 shape/storage 估算可以跨
Ampere 8.6 与 Blackwell 12.0 使用。backend workspace 仍需针对新的 attention
实现和超出校准范围的 shape 单独验证。

使用二级 scale 的 NF4 路径在两张卡上得到逐字节相同的结果：0.8B sequence
16 的真实峰值为 1.051 GiB，checkpointed sequence 128 为 1.171 GiB；对应
静态峰值为 1.058 GiB 和 1.184 GiB。持久存储和随机 batch 指纹也完全一致。

## 双 rank FSDP 全分片

FSDP 投影以单卡 ATen storage 生命周期报告为基础，按 FSDP unit 分片参数、
梯度、AdamW state 和 optimizer 临时张量；buffer 与 activation 仍由各 rank
保留完整副本。graph 峰值还包括最大的 padded parameter all-gather，以及
reduce-scatter 生成本地 shard 之前保留的完整 unit 梯度。

生成配置完全一致的静态报告，然后执行真实 CUDA 与 fake NCCL 控制器：

```bash
$PYTHON verification/qwen_sft_memory_worker.py \
  --mode static --model-dir "$MODEL" --sequence-length 16 \
  --output build/qwen-fsdp-sft-memory/static.json

$PYTHON verification/run_qwen_fsdp_sft_memory.py \
  --model-dir "$MODEL" \
  --static-report build/qwen-fsdp-sft-memory/static.json \
  --output-dir build/qwen-fsdp-sft-memory/result
```

控制器对 24 个 decoder layer 和一个 root unit 应用 FULL_SHARD，关闭参数
prefetch，并在一张物理 GPU 上启动两个 rank。两个进程拥有独立的 CUDA
allocator，因此可以验证当前主机条件下的每 rank 显存和 collective 语义；
执行耗时不能作为多 GPU NCCL 性能数据。

| GPU | Seq. | graph 实测 | graph 预测 | graph 误差 | 整体实测 | 整体预测 | 整体误差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RTX PRO 5000 Blackwell | 16 | 3.091 GiB | 3.067 GiB | 0.758% | 3.308 GiB | 3.284 GiB | 0.730% |
| RTX 3090 Ti Ampere | 16 | 3.091 GiB | 3.067 GiB | 0.758% | 3.308 GiB | 3.284 GiB | 0.730% |
| RTX PRO 5000 Blackwell | 128 | 3.143 GiB | 3.130 GiB | 0.396% | 3.357 GiB | 3.336 GiB | 0.633% |
| RTX 3090 Ti Ampere | 128 | 3.143 GiB | 3.130 GiB | 0.396% | 3.357 GiB | 3.336 GiB | 0.633% |

1,504,786,048 字节的 BF16 模型在每个 rank 上形成 752,393,024 字节的参数
shard。最大的 all-gather unit 为 508,561,408 字节，额外的 reduce-scatter
梯度 workspace 为 254,280,704 字节。一个完整 step 记录 49 次 all-gather
（5,002,021,376 字节）、25 次 reduce-scatter（3,009,572,096 字节），加上
最终 barrier 后，node-pair 总量为 8,011,593,488 字节。两套 GPU 软件环境的
参数、梯度、optimizer state、阶段峰值和通信字节数完全一致。

## 双/四 rank FSDP2 LoRA

LoRA 的冻结基座参数为 BF16，PEFT adapter 为 FP32。FSDP2 投影按原始参数
分别计算维度 0 的 shard 和 padding，不把一个 unit 强制展平为同一 dtype；
forward/loss、backward 梯度、collective buffer 与 optimizer 采用独立的
生命周期事件。

```bash
$PYTHON verification/qwen_sft_memory_worker.py \
  --mode static --model-dir "$MODEL" --training-method lora \
  --lora-rank 8 --sequence-length 16 \
  --output build/qwen-fsdp2-lora-s16/static.json

$PYTHON verification/run_qwen_fsdp2_lora_sft_memory.py \
  --model-dir "$MODEL" \
  --static-report build/qwen-fsdp2-lora-s16/static.json \
  --sequence-length 16 --world-size 4 \
  --output-dir build/qwen-fsdp2-lora-s16/result
```

| GPU | Ranks | Seq. | 整体实测 | 整体预测 | 整体误差 | 最大阶段误差 |
|---|---:|---:|---:|---:|---:|---:|
| RTX PRO 5000 Blackwell | 2 | 16 | 1.961 GiB | 1.998 GiB | 1.881% | 1.881% |
| RTX 3090 Ti Ampere | 2 | 16 | 1.961 GiB | 1.998 GiB | 1.906% | 1.906% |
| RTX PRO 5000 Blackwell | 4 | 16 | 1.593 GiB | 1.600 GiB | 0.487% | 1.228% |
| RTX 3090 Ti Ampere | 4 | 16 | 1.594 GiB | 1.600 GiB | 0.380% | 1.644% |
| RTX PRO 5000 Blackwell | 4 | 64 | 1.801 GiB | 1.768 GiB | 1.881% | 1.881% |
| RTX 3090 Ti Ampere | 4 | 64 | 1.803 GiB | 1.768 GiB | 1.974% | 1.974% |
| RTX PRO 5000 Blackwell | 2 | 128 | 2.689 GiB | 2.691 GiB | 0.106% | 0.373% |
| RTX 3090 Ti Ampere | 2 | 128 | 2.687 GiB | 2.691 GiB | 0.161% | 0.431% |
| RTX PRO 5000 Blackwell | 4 | 128 | 2.324 GiB | 2.326 GiB | 0.071% | 1.107% |
| RTX 3090 Ti Ampere | 4 | 128 | 2.322 GiB | 2.326 GiB | 0.148% | 1.477% |

1,526,431,360 字节的 mixed-dtype 模型在每个 rank 上占用 763,215,680
字节参数 storage。每个 rank 包含 10,822,656 字节 FP32 adapter 参数与同等
大小的梯度，以及 21,645,312 字节 AdamW tensor state。一个 step 记录 50 次
字节打包 all-gather（6,105,725,440 字节）、24 次 FP32 reduce-scatter
（43,290,624 字节）；包含 barrier 的 node-pair 总量为 6,149,016,080 字节。
Ampere/PyTorch 2.12/CUDA 13 与 Blackwell/PyTorch 2.8/CUDA 12.8 的静态投影、
shard 大小和通信总量一致。
四 rank 时，每个 rank 的参数、adapter 梯度和 AdamW state 分别为
381,607,840、5,411,328 和 10,822,656 字节。估算器以首个显式 backward
ATen 算子区分 forward/loss 与 backward，并分别计算 backward activation
和梯度生成阶段与 FSDP2 buffer 的重叠。

已有测试维度不需要修改源码。sequence length、LoRA 配置、dtype 和 2/4 rank
都由命令参数控制。不同 sharding 策略、量化参数表示、optimizer 或算子
workspace 会引入新的执行语义，此时需要增加可复用的模型和参数化测试，不应
为某一个 workload 添加专用公式。

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

当前结果覆盖单卡全参数与 LoRA BF16 训练、使用 FP32 scale 或二级 scale 的
原生 NF4 QLoRA 参考实现、single-tensor AdamW、gradient checkpointing、
gradient accumulation 2、Transformers 的 PyTorch fallback linear-attention，
以及 sequence 16/128 的双 rank 全参数 FSDP、sequence 16/64/128 的双/四
rank mixed-dtype FSDP2 LoRA。
外部 bitsandbytes fused kernel 与 allocator、paged/量化 optimizer、Flash
Linear Attention、自定义 CUDA/Triton kernel、FSDP2 QLoRA、显式 mixed
precision policy 和超过四个 rank 的场景仍需分别验证，不能直接套用上述误差。
