# 微调时间与显存

[文档索引](README.md) · [English](../en/finetuning.md) ·
[繁體中文](../zh-TW/finetuning.md)

FakeGPU 可以根据配置文件估算单卡微调，也可以运行真实的训练框架并记录实际情况，
再用独立运行检验之后的预测。支持的框架有 Hugging Face Trainer/PEFT、
LLaMA-Factory 和 Axolotl。

目录：

1. [根据配置估算](#estimate-from-a-config)
2. [扫描数据集](#profile-the-dataset)
3. [记录原生运行](#record-a-native-run)
4. [验证时间窗口和事件](#validate-timing-windows-and-events)
5. [预算完整流水线](#budget-a-complete-pipeline)
6. [随机采样与更换数据集](#random-sampling-and-changed-datasets)
7. [长度分组与 packing](#length-grouping-and-packing)
8. [Kernel 缓存](#kernel-caches)
9. [原生运行准备](#native-run-setup)
10. [用 Hugging Face Trainer 采样](#probe-with-hugging-face-trainer)
11. [验证记录](#validation-records)

<a id="estimate-from-a-config"></a>

## 根据配置估算

`estimate-finetune` 读取本地模型元数据和框架配置，估算单卡全量、LoRA、4-bit
QLoRA 的显存与**训练循环**耗时。它不导入训练框架，也不加载权重。

- 样本数指过滤、切分及 packing 后的训练序列数。序列长度包括模板 token 与 padding。
- `max_steps` 优先于 epochs。`--max-steps -1` 可清除配置中的步数限制。
- 未打包的 Axolotl 配置按 Axolotl 自己的规则换算 epochs：
  `ceil(samples * epochs / (micro_batch_size * gradient_accumulation_steps))`
  次更新，Axolotl 会把它作为 `max_steps` 传给 Trainer。

```bash
python -m fakegpu estimate-finetune \
  --config examples/finetuning/huggingface-lora.yaml \
  --model /path/to/Qwen3.5-0.8B --text-only \
  --profile "$TARGET_GPU_PROFILE" --num-samples 10000 --epochs 3 \
  --sequence-length 128 --json estimate.json
```

- 三类框架的配置示例见 [`examples/finetuning/`](../../examples/finetuning/)。
  配置适配经过 CPU 测试，但没有在所有框架与运行环境组合上做过基准测试。
- 分布式或 offload 执行、未覆盖的优化器、无法解析的 adapter 目标会明确报错。
  分布式规划请用 `plan-training`。
- Qwen3.5 的线性 attention 层与完整 attention 层分开计算。DeltaNet backward
  workspace、快速 kernel 和 MoE dispatch 只有部分建模。`--text-only` 选择多模态
  checkpoint 的文本骨干。

<a id="profile-the-dataset"></a>

## 扫描数据集

`profile-finetune-data` 全量扫描本地 JSONL 的每一行，记录内容指纹、token 长度、
截断、padding 与序列数。原始 `text` 或 `messages` 行使用本地 tokenizer。
`--packing concatenate` 只是一种 token 预算情景，不复现任何框架的 packing 算法。

要使用实际训练集的计数，请导出最终的 `input_ids` 行（可附带 `attention_mask` 和
`labels`），并加上 `--processed`：

```bash
python -m fakegpu profile-finetune-data \
  --data /path/to/processed-train.jsonl --format input_ids --processed \
  --sequence-length 128 --json dataset.json
python -m fakegpu estimate-finetune \
  --config examples/finetuning/huggingface-lora.yaml \
  --model /path/to/Qwen3.5-0.8B --text-only \
  --profile "$TARGET_GPU_PROFILE" --dataset-report dataset.json \
  --epochs 3 --max-steps -1 --json estimate.json
```

`--dataset-report` 同时固定样本数和序列上限。它会替换配置中的样本数，拒绝冲突的
覆盖值，并要求上限一致。`--source-framework` 只是来源标签。Packed 训练需要最终
处理后的序列计数。

<a id="record-a-native-run"></a>

## 记录原生运行

`run-finetune` 在一张物理 CUDA 显卡上运行已安装的 HF Trainer/PEFT、LLaMA-Factory
或 Axolotl 流水线。支持全量、LoRA 和 bitsandbytes 4-bit QLoRA，要求本地未量化的
文本 checkpoint。除下文另有说明外，数据不做 packing。训练使用完整的原生配置，
相对路径以配置文件所在目录解析。运行报告记录：

- 原生预处理、有效训练数据、实际 batch、优化器更新、allocator 峰值、评估与
  checkpoint 事件，以及最终保存
- `observation.updates[].batches`：每个 microbatch 的形状、epoch 位置、token 槽位、
  非 padding token 和有效 labels
- `observation.batch_profile`：按完整、有序的梯度累积窗口分组的耗时。每组有两个
  稳态 update 后标记为 `sufficient`。这只说明样本够用，与精度无关。
- `observation.phase_events`：实际的评估与保存调用树。阶段总量只累计根区间，
  保存后的文件清单也不代表磁盘写入量。

Warmup 和采集开销不计入稳态样本，但包含在总墙钟时间中。

<a id="validate-timing-windows-and-events"></a>

## 验证时间窗口和事件

`validate-finetune-windows` 用单独记录的 holdout 运行验证耗时预测。

- 至少需要两份独立的校准报告，模型、数据、软件和原生设置都要相同。
- 剔除全局 warmup 和窗口首次出现之后，每个窗口在每份校准报告中仍需保留两个
  观测。缺少覆盖的窗口会被拒绝。
- 它只校验历史证据，不预测 shuffle 顺序、warmup、总墙钟时间或动态显存。原生
  packing 与 MoE 窗口尚未支持。

```bash
python -m fakegpu validate-finetune-windows \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --holdout-run holdout.json --json window-validation.json
```

`validate-finetune-events` 用同样的方式验证评估和保存的子总时间。开始、首次、重复
和最终事件分别处理。目前支持无 packing 的 dense 文本模型、loss-only 评估，以及
本地模型/adapter 或可恢复 checkpoint 的保存。它不预测事件次数，也不预测完整
流水线时间。

```bash
python -m fakegpu validate-finetune-events \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --holdout-run holdout.json --json event-validation.json
```

`--event-context` 还会匹配每个窗口之前最近的评估和 checkpoint 边界，包括有序调用、
工作量、阶段及各边界之后的 update 数。每个“窗口＋上下文”组合在每份校准报告中
需要两个样本。它验证的是周期事件之后重复出现的窗口，不覆盖首次保存后的恢复耗时，
也不覆盖完整训练时间。

```bash
python -m fakegpu validate-finetune-windows --event-context \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --holdout-run holdout.json --json event-window-validation.json
```

`--epoch-context` 还会匹配每个 microbatch 在 epoch 内的位置，用来区分包含新一轮
取数或 worker 启动的窗口。可以单独使用，也可以与 `--event-context` 组合：

```bash
python -m fakegpu validate-finetune-windows --event-context --epoch-context \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --holdout-run holdout.json --json epoch-window-validation.json
```

<a id="budget-a-complete-pipeline"></a>

## 预算完整流水线

`plan-finetune-pipeline` 根据校准运行中完整观测到的 epoch 推算未来的轮次或更新数。
`estimate-finetune-pipeline` 组合至少两次校准，得到完整的原生预算。预算涵盖
setup、warmup、有序训练窗口、评估、checkpoint、Trainer 间隙与收尾。

- 计时止于原生调用及最后一次 CUDA 同步。Python 启动、报告构造和 CLI 输出不计入。
- 时间区间是各进程均值的极值，不是置信区间。任何部件缺少覆盖都会明确报错。
- 支持无 packing、保留尾批次的顺序文本数据：HF Trainer（Transformers 5.6.0）与
  LLaMA-Factory 0.9.5 需要 `disable_shuffling: true`，Axolotl 0.18.0
  （Transformers 5.14.1）需要 `curriculum_sampling: true`。
- 数据、模型、硬件和执行设置必须与校准一致，数据行数变了就要重新校准。日志必须
  关闭，或者每次 update 都记录。不支持 checkpoint 轮转删除和最佳模型重载。

```bash
python -m fakegpu plan-finetune-pipeline \
  --run-report calibration-1.json --epochs 30 --json future-plan.json
python -m fakegpu estimate-finetune-pipeline \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --plan future-plan.json --memory-budget-gib 24 --reserve-gib 1 --json budget.json
python -m fakegpu validate-finetune-pipeline \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --holdout-run holdout.json --json validation.json
```

显存结果是同一 GPU 和 allocator 上完整运行期间观测到的 allocated 与 reserved 峰值，
不能推算到更小的显卡。可选的 reserve 只是你自己定的余量，不能证明不会 OOM。
`validate-finetune-pipeline` 对照独立的 holdout 报告时间误差和显存低估。

<a id="random-sampling-and-changed-datasets"></a>

## 随机采样与更换数据集

默认随机采样或更换了训练数据时，使用经验迁移流程。它从原生 prepared dataloader
出发，在 CPU 上规划 sampler 和 collator 的输出，并在校准时核对实际消费的行号和
输入张量哈希。支持的组合是 HF 和 LLaMA-Factory 配合 Accelerate 1.11.0，以及
Axolotl 配合 Accelerate 1.13.0。随机采样使用原生 data seed 加 epoch。
（Axolotl 0.18.0 不会把 YAML 中的 `data_seed` 转发给 Trainer。）

```bash
python -m fakegpu calibrate-finetune \
  --config target-training.json --framework huggingface \
  --profile "$TARGET_GPU_PROFILE" --python /path/to/framework/python \
  --cuda-device 0 --workspace ./calibration-target \
  --calibration-steps 24 --warmup-steps 2 --epochs 3 \
  --timeout-seconds 600 --json calibration.json
python -m fakegpu budget-finetune-workspace \
  --workspace ./calibration-target --epochs 2 --json revised-budget.json
```

- `calibrate-finetune` 固定启动两个全新的原生进程，限制步数和运行时间，默认各自
  使用新的空 kernel 缓存目录。它保留所有尝试，用 SHA256 封存报告，支持
  `--resume`。它不会悄悄延长或重跑校准。
- 在 CPU 上重新预算不需要 GPU 或框架，但不能超出已保存的采样范围。
- 工作区要保持原来的绝对路径，并放在已记录的输入目录之外。嵌套的工作区会被拒绝。
  只支持有限的本地数据集和 POSIX 锁语义。Axolotl 和 LLaMA-Factory 的 prepared
  数据隔离在工作区的 `prepared/` 目录中，恢复时会拒绝已变更的原始输入。
- `--reuse-calibration-run PATH`（可重复）复用另一份数据集上兼容的训练证据。启动
  和收尾成本仍由目标数据集上的两次新校准提供，且不按样本数放大。不同 token、mask
  和监督分布之间的训练成本只是**近似**。缺少覆盖时返回 1 并列出缺口。
- `run-finetune --planning-epochs N`（或 `--planning-max-steps N`）直接记录 CPU
  规划范围。`plan-finetune-transfer`、`estimate-finetune-transfer` 和
  `validate-finetune-transfer` 不经过工作区管理也能完成相同步骤。streaming、
  自定义 sampler 和自动调整 batch size 尚未支持。

<a id="length-grouping-and-packing"></a>

## 长度分组与 packing

### 长度分组

HF 或 LLaMA-Factory 配置 `train_sampling_strategy: group_by_length`，Axolotl 配置
`group_by_length: true`，并向 `calibrate-finetune` 或 `run-finetune` 传入
`--length-grouping-seed 123`。

- 保留原生分组算法，使用以 `123 + epoch` 为种子的独立 CPU 生成器。依赖全局 RNG
  的原生分组无法复现。
- 必须指定规划范围。一个有限的评估数据集使用同一个显式种子，不推进训练 sampler。
  自定义评估 sampler 和提前准备的 loader 会被拒绝。
- Sampler plan v2 把分组长度和 batch size 纳入校准身份，长度组合不同就需要各自的
  证据。

### Packing

- [packing 语义审计](../../tests/native/finetune_packing_semantics.py)表明，在极小
  的 Qwen3.5 模型中，所测的普通 HF 和 LLaMA-Factory 混合路径在分块 mask 和重置
  位置后仍会跨段泄漏。它还表明所测的 Axolotl patch 与 Transformers 5.14.1 的
  decoder 接口不兼容。
- 可选的[分段参考审计](../../tests/native/finetune_segment_reference.py)检查项目
  自有的 Qwen3.5 接口。该接口让每段 linear attention 从空的卷积和递归状态开始，
  结果与独立的原生序列一致，包括 checkpoint 重算与重载。
- 在 `run-finetune` 或 `calibrate-finetune` 上传入 `--packing-seed 123` 启用该路径。
  它要求 dense `qwen3_5_text` LoRA、指定规划范围、启用原生 packing（LLaMA-Factory
  设置 `packing: true` 和 `neat_packing: true`，Axolotl 设置
  `sample_packing: true`），并关闭评估。
- HF processed 行另外需要 `position_ids`，`attention_mask` 中要用连续的正整数
  segment ID（右侧 padding 可以为零）。后续每段的第一个 label 必须忽略。
- Sampler plan v3 记录 epoch bins、逻辑序列数、张量行数、分块几何、dtype 和输入
  哈希，所以不同数据需要单独校准。原生 packing 报告必须使用 transfer 命令。
- 不对 varlen kernel 的吞吐做任何承诺。全量、QLoRA 和分组 packing 未覆盖。另见
  [PEFT direct-forward 审计](../../tests/native/finetune_segment_peft.py)和
  [packing 预算证据](../../tests/data/finetune_packing_budget_evidence.json)。

<a id="kernel-caches"></a>

## Kernel 缓存

`run-finetune --cache-regime empty_kernel_cache` 要求显式设置五个互不相同、已存在
且为空的目录：`TRITON_CACHE_DIR`、`TORCHINDUCTOR_CACHE_DIR`、`CUDA_CACHE_PATH`、
`TORCH_EXTENSIONS_DIR` 和 `PYTORCH_KERNEL_CACHE_PATH`。`populated_kernel_cache`
记录其中每个文件的相对路径、大小和 SHA256。runner 从不创建、清理或预热这些缓存。

默认的 `uncontrolled` 模式不对缓存状态做任何承诺，也不能与受控校准混用。Triton
元数据可能引用原来的缓存路径，所以这些文件要保持不变。

<a id="native-run-setup"></a>

## 原生运行准备

- 无 packing 的 HF 运行（`--framework huggingface`）使用平铺配置，以及每行恰好包含
  `input_ids`、`attention_mask` 和 `labels` 的 processed JSONL。三者须为等长整数
  数组，mask 为二值，长度在 2 到 `sequence_length` 个 token 之间。不做分词，
  tokenizer 需要有 pad token。参见[原生 LoRA 示例](../../examples/finetuning/huggingface-native-lora.yaml)。
- LLaMA-Factory 的梯度检查点使用 `disable_gradient_checkpointing: false`，并用
  `use_reentrant_gc: true` 或 `false` 选择实现。Axolotl 使用
  `gradient_checkpointing: true` 与 `gradient_checkpointing_kwargs: {use_reentrant: true}`
  （或 `false`）。实际生效的检查点与量化设置都纳入校准身份。
- 原生运行需要文本 checkpoint，`--text-only` 不会转换权重。对于本地 Qwen3.5，先在
  CPU 上导出文本 checkpoint，再生成用于集成检查的合成 Alpaca 数据：

```bash
python tests/native/prepare_finetune_fixture.py \
  --model /path/to/Qwen3.5-0.8B \
  --output-dir /tmp/fakegpu-finetune/qwen35-text \
  --dataset-dir /tmp/fakegpu-finetune/data
```

导出需要支持 `Qwen3_5ForCausalLM` 的 Transformers 版本。原生示例针对
LLaMA-Factory 0.9.5（Transformers 5.6.0）和 Axolotl 0.18.0（Transformers 5.14.1）。
请分别在独立环境中安装，并配上兼容的 CUDA PyTorch、PEFT 和数据依赖。每次运行
使用新的 `output_dir`。Axolotl 的预处理缓存位于
`/tmp/fakegpu-finetune/axolotl-prepared`。

```bash
CUDA_VISIBLE_DEVICES=1 python -m fakegpu run-finetune \
  --config examples/finetuning/axolotl-native-lora.yaml --framework axolotl \
  --profile "$TARGET_GPU_PROFILE" --warmup-steps 2 --output run.json
python -m fakegpu estimate-finetune \
  --run-report run.json --epochs 3 --max-steps -1 --json estimate.json
```

- LLaMA-Factory 使用 `--framework llamafactory` 和
  [`llamafactory-native-lora.yaml`](../../examples/finetuning/llamafactory-native-lora.yaml)。
- 在同一个 Python 环境中复用 `--run-report`。CLI 会重新核对本地权重、模型配置和
  软件身份，GPU profile 默认沿用报告中的值。
- 这种单报告 `estimate-finetune` 接口要求固定形状的完整 batch。改变
  `--num-samples` 只得到训练循环情景，不含预处理和加载开销。Python 启动和下载也
  不计入。

512-token NF4 QLoRA 配合梯度检查点的示例见
[`llamafactory-native-qlora.yaml`](../../examples/finetuning/llamafactory-native-qlora.yaml)
或 [`axolotl-native-qlora.yaml`](../../examples/finetuning/axolotl-native-qlora.yaml)。
两者都使用 double quantization 和 BF16，需要 bitsandbytes。用导出的 tokenizer
生成较长的数据集：

```bash
python tests/native/prepare_finetune_fixture.py \
  --model /tmp/fakegpu-finetune/qwen35-text --dataset-only \
  --dataset-dir /tmp/fakegpu-finetune/data-long \
  --sequence-length 1024 --num-samples 64
```

fixture 长度只是下限。各框架会按配置截断到 512 或 1024 个 token，运行报告记录
实际使用的序列。

用 [`tests/native/finetune_framework_matrix.py`](../../tests/native/finetune_framework_matrix.py)
采集独立的校准和 holdout 进程。序列长度、microbatch 大小、梯度累积和 seed 都从
原生配置中读取：

```bash
python tests/native/finetune_framework_matrix.py \
  --framework axolotl --config examples/finetuning/axolotl-native-qlora.yaml \
  --python /path/to/axolotl/venv/bin/python \
  --output-dir /tmp/fakegpu-finetune/axolotl-qlora-matrix \
  --gpu-index 1 --calibration-steps 6 --warmup-steps 2 \
  --holdout-runs 2 --holdout-steps 16
```

步数包含 warmup。默认是 6 次校准更新、2 次 warmup 更新，以及三次各 32 次更新的
holdout。每次运行使用新的输出目录，预测调用生产估算器，并使用实际的 microstep
计数。

<a id="probe-with-hugging-face-trainer"></a>

## 用 Hugging Face Trainer 采样

想要基于实测的估算，可以先在真实 GPU 上短跑采样，再在**同一个 Python 环境**中
复用这份观测。模型、GPU、精度、adapter、attention 实现、batch、序列长度和梯度
累积也必须一致。

```bash
CUDA_VISIBLE_DEVICES=1 python -m fakegpu probe-finetune \
  --config examples/finetuning/huggingface-lora.yaml \
  --model /path/to/Qwen3.5-0.8B --text-only \
  --profile "$TARGET_GPU_PROFILE" --warmup-steps 2 --measure-steps 5 \
  --output observation.json
python -m fakegpu estimate-finetune \
  --config examples/finetuning/huggingface-lora.yaml \
  --model /path/to/Qwen3.5-0.8B --text-only \
  --profile "$TARGET_GPU_PROFILE" --num-samples 10000 --epochs 3 \
  --observation observation.json --json calibrated.json
```

probe 用 Hugging Face Trainer 和固定长度的合成文本运行，需要 Transformers、
Accelerate 和 CUDA PyTorch。adapter 另需 PEFT，QLoRA 另需 bitsandbytes。warmup
单独记录，启动峰值和稳态峰值都会保留。预测耗时不含数据准备、模型加载、评估和
checkpoint I/O。checkpoint 文件会计算哈希，替换权重后旧校准就会失效。

<a id="validation-records"></a>

## 验证记录

精确的工作负载、版本和适用范围见 [`refactor.md`](../../refactor.md)。

- [完整流水线验证](../../tests/data/finetune_pipeline_validation_evidence.json)：
  三个框架共 30 个原生进程，包括 allocator OOM 对照。
- [采样与迁移验证](../../tests/data/finetune_transfer_validation_evidence.json)：
  Blackwell GPU 上 18 个 Qwen3.5-0.8B 进程，训练数据从 25 行改为 73 行，并包含
  随机采样配合周期性评估和保存。每个条件一次 holdout。
- [Qwen3.5-2B 记录](../../tests/data/finetune_qwen2b_evidence.json)：
  512/1024-token LoRA 与 512-token NF4 QLoRA（启用梯度检查点），每个条件两次校准、
  两次 holdout。总时间误差为 0.054%–1.578%。
- [原生框架验证归档](../../tests/data/finetune_framework_validation_evidence.json)：
  早先的 128-token LoRA 运行。最大墙钟误差为 3.931%（LLaMA-Factory）和 1.125%
  （已缓存的 Axolotl）。把 Axolotl 首次运行的校准用于之后已缓存的进程时，误差达到
  364.885%（首次 update 用了 66.542 秒）。可见模型和软件签名相同，并不意味着
  cache 和 JIT 开销可以复用。
- [扩展验证归档](../../tests/data/finetune_extended_validation_evidence.json)：
  六组配置，覆盖 512/1024 tokens、batch 1/2、梯度检查点与 NF4 QLoRA，共 18 个测量
  进程。
- [Qwen3.5-0.8B probe 记录](../../tests/data/finetune_validation_evidence.json)：
  全量、LoRA 和 QLoRA 的校准与 holdout 运行，可用
  [`tests/native/finetune_matrix.py`](../../tests/native/finetune_matrix.py) 复现。

这些短跑结果不能说明通用精度或 OOM 安全。
