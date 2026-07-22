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

### CPU offload

同一验证器可以直接选择 optimizer 和参数 offload：

```bash
$PYTHON verification/run_hybrid_deepspeed_numerics.py \
  --zero-stage 2 --precision fp32 --world-size 2 \
  --offload optimizer \
  --report-dir build/deepspeed-offload-zero2

$PYTHON verification/run_hybrid_deepspeed_numerics.py \
  --zero-stage 3 --precision fp32 --world-size 2 \
  --offload optimizer-and-parameter \
  --report-dir build/deepspeed-offload-zero3
```

这些用例使用调用方提供的 PyTorch SGD，并设置
`zero_force_ds_cpu_optimizer=false`，不会编译 DeepSpeed CPUAdam。验证器会确认
optimizer state 位于 CPU；ZeRO-3 参数 offload 还会确认本地参数分片位于 CPU。

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

## Checkpoint 保存与恢复

checkpoint 验证器会执行一次更新，让所有 rank 保存状态，创建新的 engine，恢复
model、optimizer、scheduler 和 client state，继续训练，并与不中断训练的结果
比较。ZeRO-2/3 还会重建 FP32 state dict。

```bash
$PYTHON verification/run_hybrid_deepspeed_checkpoint.py \
  --zero-stage 3 --precision fp32 --world-size 2 \
  --report-dir build/deepspeed-checkpoint
```

PyTorch 2.8 加载 DeepSpeed 0.15.3 checkpoint 时，会要求登记三个旧版
DeepSpeed 类型。验证器只放行 PyTorch 从自身生成的 checkpoint 中识别出的类型，
不会全局关闭 `weights_only` 保护。DeepSpeed 要求所有 rank 都参与 checkpoint
保存，详见 [DeepSpeed checkpoint 文档](https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html)。

## Hugging Face Trainer

Trainer 验证器既支持不依赖模型文件的小型网络，也支持本地 Qwen LoRA 权重。
它通过 `TrainingArguments(deepspeed=...)` 传入配置，并检查每个 rank 上真实发生的
参数更新。

```bash
$PYTHON verification/run_hf_trainer_deepspeed.py \
  --workload tiny --zero-stage 2 --precision bf16 \
  --gradient-accumulation-steps 2 \
  --report-dir build/hf-trainer-deepspeed-tiny

$PYTHON verification/run_hf_trainer_deepspeed.py \
  --workload qwen-lora --model-dir "$MODEL" \
  --zero-stage 3 --precision bf16 --sequence-length 64 \
  --gradient-accumulation-steps 2 --gradient-checkpointing \
  --report-dir build/hf-trainer-deepspeed-qwen
```

配置遵循
[Transformers DeepSpeed 集成文档](https://huggingface.co/docs/transformers/main/en/deepspeed)，
包含由 Trainer 解析的 `auto` 字段。

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

下列附加路径也已在两套 GPU/软件环境中通过：

| 路径 | 覆盖范围 | 结果 |
|---|---|---|
| ZeRO checkpoint | ZeRO-3 FP32；ZeRO-2/3 BF16 | 保存、创建新 engine 后恢复、AdamW/StepLR/client-state 续训、与不中断训练一致、FP32 合并 |
| Hugging Face Trainer | 小型 ZeRO-2/3 BF16；Qwen3.5-0.8B LoRA ZeRO-3 BF16 | loss 有限、参数真实更新、梯度累积、rank 一致、通信报告完整 |
| CPU offload | ZeRO-2 optimizer；ZeRO-3 optimizer + 参数，FP32 | 更新结果符合解析值，optimizer state 和指定参数分片位于 CPU |

## 物理双主机 DeepSpeed

SSH 控制脚本可以让每台物理主机各运行一个 DeepSpeed rank：

```bash
python3 verification/run_physical_multihost.py \
  --node 'name=blackwell;ssh=gpu-a;repo=/home/user/repos/fakeGPU;python=/opt/torch/bin/python;shell=posix' \
  --node 'name=ampere-wsl;ssh=user@gpu-b;repo=/home/user/repos/fakeGPU;python=/opt/torch/bin/python;shell=wsl' \
  --coordinator-host 100.x.y.z \
  --case deepspeed-zero2
```

维护中的 RTX PRO 5000 ↔ RTX 3090 Ti ZeRO-2 实验完成了 7 次 TCP
collective；节点对通信总量为 176 B，单次峰值为 32 B；两个 rank 都得到参数
`[0.775, -0.45]`。两端分别使用 DeepSpeed 0.15.3 和 0.19.2。

ZeRO-3 的参数 trace 会产生与 DeepSpeed 版本有关的 collective 序列。本次两台
主机版本不同：未加检查时，0.15.3 在同一序号执行 all-gather，0.19.2 执行
broadcast。因此，`--case deepspeed-zero3` 要求两端 DeepSpeed 版本一致；版本
不同时会在训练前给出明确的预检错误。ZeRO-3 已分别在两张 GPU 上通过单机多
rank 验证。

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
- ZeRO checkpoint 保存、恢复、续训和 FP32 合并
- Hugging Face `Trainer` 小型网络与 Qwen LoRA 路径
- CPU optimizer offload 和 ZeRO-3 CPU 参数 offload
- 物理双主机 TCP ZeRO-2
- 任意节点对的逻辑通信报告

仍需要专门实验的路径：

- DeepSpeed fused optimizer 和其他 JIT CUDA operator
- NVMe offload
- pipeline、tensor、sequence、expert 和 MoE 并行
- DeepSpeed 版本一致时的物理双主机 ZeRO-3

因此，维护中的测试通过表示上面列出的训练路径可用，不表示 DeepSpeed 全部 API
或性能都与真实集群等价。
