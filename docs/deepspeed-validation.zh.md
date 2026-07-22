# DeepSpeed 验证

FakeGPU 已提供维护中的 DeepSpeed 训练路径：Hybrid 模式让 CUDA 计算落在
一张真实 GPU 上，同时用 FakeGPU 模拟两个或四个逻辑 rank 的 NCCL 通信。
验证器会检查 optimizer 数值和通信记录，不只是确认进程能够启动。

## 快速验证

选择已经安装 PyTorch 和 DeepSpeed 的 Python 环境：

```bash
PYTHON=/path/to/python

$PYTHON verification/run_hybrid_deepspeed_numerics.py \
  --zero-stage all --precision bf16 --world-size 2 \
  --report-dir build/deepspeed-numerics
```

这组小型矩阵会检查：

- DeepSpeed Engine 初始化
- ZeRO 0、1、2、3
- FP32 或 BF16 参数
- 两步梯度累积只触发一次 optimizer 更新
- 所有 rank 的更新后参数一致
- broadcast、all-reduce、all-gather、reduce-scatter 报告
- 完整的节点对通信总量和单次操作峰值

同一命令支持 `--world-size 4`。如果只需要验证通信最复杂的阶段，可以使用
`--zero-stage 3`。

## Qwen LoRA SFT

真实模型路径会从本地加载 Qwen3.5 权重、应用 PEFT LoRA，通过 DeepSpeed
完成前向、反向和 AdamW 更新，并在每个 rank 上核对完整 LoRA 参数。

```bash
MODEL=/home/l1ght/models/Qwen/Qwen3.5-0.8B

$PYTHON verification/run_qwen_deepspeed_lora_sft.py \
  --model-dir "$MODEL" \
  --output-dir build/qwen-deepspeed-zero3 \
  --zero-stage 3 --world-size 2 \
  --dtype bfloat16 --sequence-length 16
```

梯度累积和 activation checkpoint 都已经是命令行参数，增加这类测试不需要修改
项目代码：

```bash
$PYTHON verification/run_qwen_deepspeed_lora_sft.py \
  --model-dir "$MODEL" \
  --output-dir build/qwen-deepspeed-zero3-s64-gas2-gc \
  --zero-stage 3 --world-size 2 \
  --dtype bfloat16 --sequence-length 64 \
  --gradient-accumulation-steps 2 \
  --gradient-checkpointing
```

DeepSpeed 路径默认使用重入 checkpoint。当前 Qwen/PEFT 软件栈在 DeepSpeed
0.15.3 ZeRO-3 下使用非重入重算时，会让分片参数的 `[0]` 占位符进入 PyTorch
元数据检查。验证器保留了 `--checkpoint-implementation non-reentrant`，用于后续
版本的兼容性复测，但它不是当前维护中的通过配置。

## 当前实测结果

小型数值矩阵已在两套环境中通过：

| GPU | 软件环境 | 已通过矩阵 |
|---|---|---|
| RTX PRO 5000 72GB Blackwell，CC 12.0 | PyTorch 2.8.0 + CUDA 12.8，DeepSpeed 0.15.3 | 双 rank FP32/BF16 ZeRO 0–3；四 rank BF16 ZeRO-3 |
| GeForce RTX 3090 Ti Ampere，CC 8.6 | PyTorch 2.12.1 + CUDA 13.0，DeepSpeed 0.19.2 | 双 rank FP32 ZeRO-0/3 和 BF16 ZeRO 0–3 |

Qwen3.5-0.8B 使用 batch size 1、BF16 基座、LoRA rank 8 和两个逻辑 rank。
下表的显存是单个 rank 的 PyTorch allocated 峰值；实验中的两个 rank 共用一张
物理 GPU。

| GPU / 配置 | 单 rank 峰值 | 逻辑节点对通信量 | 结果 |
|---|---:|---:|---|
| PRO 5000，ZeRO-2，序列 16 | 1.754 GiB | 2.884 GiB | loss 有限，LoRA 更新一致 |
| PRO 5000，ZeRO-3，序列 16 | 2.899 GiB | 12.273 GiB | 2,259 次 all-gather 和 2 次 reduce-scatter 通过 |
| RTX 3090 Ti，ZeRO-3，序列 16 | 2.904 GiB | 12.273 GiB | DeepSpeed 0.19.2 下训练约束一致 |
| PRO 5000，ZeRO-3，序列 64，累积 2 | 3.052 GiB | 15.196 GiB | 两个微步后只更新一次 |
| PRO 5000，上项 + checkpoint | 2.409 GiB | 17.134 GiB | 重入重算通过 |

在维护中的序列 64 用例里，checkpoint 让 allocated 峰值降低 21.1%，模拟通信量
增加 12.8%。这个小模型的 ZeRO-3 峰值高于 ZeRO-2，因为参数预取和聚合缓冲区
的开销超过了分片节省。该结果只对应表中的软件栈和配置，不能直接外推到更大模型
或真实多 GPU 性能。

每次运行都会写出：

- `summary.json` 和 `summary.md`
- 每个 rank 的详细 JSON
- `cluster-report.json` 和 `cluster-report.md`
- collective 调用次数和完整节点对通信表

## 没有 `nvcc` 的 WSL 环境

DeepSpeed 0.19.2 会在导入时探测可选 CUDA operator。当前 3090 Ti WSL 环境
有 CUDA runtime，但没有 `nvcc`，因此纯 PyTorch optimizer 测试使用官方提供的
探测跳过开关：

```bash
DS_IGNORE_CUDA_DETECTION=1 $PYTHON \
  verification/run_hybrid_deepspeed_numerics.py \
  --zero-stage all --precision bf16 --world-size 2 \
  --report-dir build/deepspeed-numerics
```

这个设置不能证明 JIT 编译的 DeepSpeed operator 可用。测试这类 operator 之前，
仍需要安装与 PyTorch 匹配的 CUDA compiler。

## 当前能力边界

已验证的路径：

- 客户端 PyTorch SGD 和 AdamW optimizer
- DeepSpeed Engine backward 和 step
- 双逻辑 rank ZeRO 0–3，以及一个四 rank ZeRO-3 用例
- FP32 和 BF16 通信
- 梯度累积
- Transformers Qwen3.5 + PEFT LoRA SFT
- 重入 activation checkpoint
- 任意节点对的逻辑通信报告

仍需要专门实验的路径：

- DeepSpeed fused optimizer 和其他 JIT CUDA operator
- CPU optimizer/parameter offload 和 NVMe offload
- ZeRO checkpoint 保存、恢复和合并
- pipeline、tensor、sequence、expert 和 MoE 并行
- DeepSpeed 多物理主机启动
- Hugging Face `Trainer` 直接接入 DeepSpeed 配置

因此，维护中的测试通过表示上面列出的训练路径可用，不表示 DeepSpeed 全部 API
或性能都与真实集群等价。
