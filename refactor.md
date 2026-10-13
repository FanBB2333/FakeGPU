# FakeGPU 重构分析报告

> 最新功能与验证结果见下方 **2026-10-13 / 真实 batch 观测与校验**；前一阶段见 **原生 QLoRA 与更多训练形状**、**原生框架与真实数据闭环**；此前交付见 **微调首批交付**；实施前承诺审计与后续路线见 **功能闭环计划**；重构结果见 **406**。历史章节的行号、规模和待办状态不能直接代表当前代码。
> **约束：重构保持既有功能；经复现的 bug 单独修复、单独提交。**

## 2026-10-13 / 真实 batch 观测与校验

继续前一阶段接续计划的第一项，先完成动态 batch 校准所需的实际工作量观测。接续时工作区干净，`4b677929` 已提交；基线完整回归 **861 passed / 1 skipped**，最终 QLoRA 证据的 15 个源码哈希全部匹配。下面是本轮新增功能的独立验证，不能将历史 GPU 校准精度迁移到新的采样协议。

- `observation.updates[].batches` 只记录成功返回的真实 `training_step`，保留形状、从 1 开始的 epoch/microstep 位置、token 槽位、非 padding tokens 和有效 labels。因果 labels 按常规 `labels[:, 1:]` 定义单独计数；不复制或保存 token ID 内容。
- `_finetune_batches.py` 生成 `fakegpu.finetune_batches.v1` 汇总，并按完整、有序的梯度累积窗口分组耗时。混合形状、不同 padding/监督计数及不完整窗口分别保留；不把一次 update 的秒数分摊为各 shape 的实测耗时，也不把全局 allocator 峰值分给各 shape。至少两个非 warmup 样本才标记该组 `sufficient`，这不是准确率或置信区间。
- 缺失、非二值或 4D attention mask 的 token 数保持未知，并报告 coverage；position IDs 的零位置只记录可观察重置，不能认定为原始 packed 样本边界。原生 packing 与动态形状外推仍有明确 gate，未在本轮解除。
- 每个 microstep 在统计计时前先等待前序 CUDA 工作，避免将真实训练等待误扣成 metadata 成本。统计开销另列并从 update 样本扣除，原始 wall time 保留该开销。新增可选签名字段 `batch_observation_version` 区分这种同步采样；旧报告不增加新字段，仍可复用。
- 两个估算 API 都检查 ledger、profile、workload、独立耗时数组与签名的一致性。新版本报告删除统计字段不能静默降级；旧版无统计字段的报告保持兼容。三语 README 同步说明观测口径与限制。

### 本轮验收

完整维护回归 **943 passed / 1 skipped，63.80 秒**；Ruff 全仓与 `git diff --check` 通过。独立安装 wheel 的 **71 个 Python 文件**与源码逐字节一致；`python -S` 下可访问 CLI、核验新报告，并复用两个框架上一阶段的 QLoRA 报告推算 **128 microsteps / 64 updates**。

真实 CPU Trainer 在 Transformers **4.57.3、5.6.0、5.14.1** 三个版本通过。13 条变长样本、microbatch 3、accumulation 2、两个 epochs 得到 **10 microsteps / 6 updates / 26 条消费序列**。`max_steps=4` 时 collator 已预取 8 个 batch，但模型只消费 7 个，ledger 仅记录这 7 个，得到 **19 条序列**。测试另检查左右 padding、有效及 causal labels、未知 mask、epoch 尾部、异步 GPU 等待计时和 hook 恢复。

物理 **Blackwell GPU / Qwen3.5-0.8B text / LoRA / cap 512 / SDPA / 无 packing 或 GC**，两个框架各运行一次两 epoch。使用同一份 13 条变长合成 Alpaca 数据、microbatch 3、accumulation 2、seed 42，关闭 prompt 监督。每个 epoch 的 batch 行数为 `[3, 3, 3, 3, 1]`，累积窗口为 `[2, 2, 1]`。

| 框架 | 实际 microsteps / updates / 序列 | 非 padding tokens | padding 槽位 | 有效 labels | 形状 / 窗口组 |
|---|---:|---:|---:|---:|---:|
| LLaMA-Factory 0.9.5 / Transformers 5.6.0 | 10 / 6 / 26 | 4,384 | 1,728 | 3,780 | 7 / 6 |
| Axolotl 0.18.0 / Transformers 5.14.1 | 10 / 6 / 26 | 5,216 | 2,165 | 3,754 | 7 / 6 |

独立审核直接读取两框架实际生成的 Arrow 缓存，重算有序行 SHA、tokens、labels，并逐 epoch 比较消费记录，全部一致；14 个执行源码哈希在两次运行期间保持不变。两份最终 adapter 均含 24 个张量、319,488 个参数，全部有限且 LoRA B 非零。原始报告、配置、日志和安装包保存在 `/tmp/fakegpu-batch-observation-20261013/`，仓库审阅归档为 [`finetune_batch_validation_evidence.json`](tests/data/finetune_batch_validation_evidence.json)。归档替换主机路径且仅保留观测证据，不能作为 `--run-report` 输入。

两次运行均没有包含两个稳态样本的相同窗口组，`sufficient` 全部为 false。Axolotl 在排除首 update 后仍有约 19 秒的样本，原值完整保留；本轮没有控制初始 kernel/cache 状态，不将它们当作已稳定的 shape 校准。实际 native 输入未暴露 position IDs，因此对应 coverage 为零。此次验收证明记录和计数正确，不提供动态耗时误差数字或长期显存保证。

### 接续工作

为同一完整窗口增加独立重复观测与 holdout，确认 shape/cache 覆盖后再实现动态外推；缺少覆盖时继续拒绝。原生 packing 还需要逐框架验证真实子序列边界及模型 attention 的隔离语义，不能仅凭 mask 或 position reset 开放。之后继续周期 evaluation/checkpoint 成本模型与原始 R24–R26 路线。

## 2026-10-13 / 原生 QLoRA 与更多训练形状

继续完成 P3 的单 GPU 原生微调扩展，同时把真实模型检查从执行器拆为独立的 `_finetune_model.py`。原先的 128-token LoRA 报告保持可由 Python API 复用；只有实际出现的新量化信息或 checkpoint 参数才增加签名字段。

| 交付项 | 实际行为 |
|---|---|
| 原生 4-bit QLoRA | LLaMA-Factory / Axolotl 从本地未量化纯文本权重加载 bitsandbytes NF4/FP4；检查真实模型方法、单 LoRA adapter、参数更新和最终保存。本轮真机验证 NF4，不代表 FP4 精度已验收。 |
| 参数与显存检查 | 从 `quant_state.shape` 计算逻辑参数，而不是把压缩张量的 `numel()` 当原模型数量；分别记录 packed weights、按 CUDA storage 去重的量化元数据、非量化冻结权重及其真实 dtype。估算器直接采用已检查的权重字节预算，激活/workspace 仍需要校准。 |
| 标准 checkpointing | HF/Axolotl 接受布尔 `gradient_checkpointing_kwargs.use_reentrant`；Factory 使用原生 `use_reentrant_gc`。从加载模型的实际 checkpoint 函数读取最终模式，并绑定校准签名；自定义 checkpoint 实现仍不支持。 |
| 校准身份 | 加入实际量化布局、存储和 GC 参数。bitsandbytes 首次 forward 可能改变有效 compute dtype，因此在训练完成后再次核对；不会仅记录加载时的请求值。 |
| 验收工具 | matrix 沿用配置中的 sequence length、microbatch、accumulation 与 seed；支持显式校准/warmup 步数，使用生产 API 检查真实 microsteps 和样本数。数据生成器支持只加载 tokenizer 生成长数据，默认 128-token 数据内容保持不变。 |
| 示例与文档 | 三语 README 同步；增加 Factory/Axolotl 原生 QLoRA 示例。原始日志、配置、报告与独立复核结果归档，不将历史验证冒充当前源码验收。 |

### 经确认的问题与独立修复

- `6b6a3934`：Factory 的实际 4-bit 存储类型随 compute dtype，修正此前默认按 uint8 处理的配置。
- `30b72935`：真机测试发现 Factory 要求 `quantization_method: bnb`；写为 `bitsandbytes` 会被其加载分支静默忽略。实际模型检查成功拦截了未量化模型，未生成错误校准报告；现提前拒绝错误拼写，示例已修正。失败配置和日志摘要保留在本轮证据中。
- `68107588`：提前拒绝会改变尚未建模参数策略的 `unfrozen_parameters`、`pure_bf16` 与 `export_quantization_bit`，避免把解冻基座当作 adapter、按 FP32 预算 BF16 adapter 或进入另一种量化流程。实际 LoRA 检查同时确保基座冻结、A/B adapter 可训练。

### 最终验证与实测边界

最终完整回归 **861 passed / 1 skipped，59.18 秒**；Ruff 全仓通过。wheel 独立安装的 **70 个 Python 模块**与工作树逐文件一致，`python -S` 下无需训练依赖即可访问 CLI。安装包复用真实 QLoRA 报告推算 **64 条序列 × 2 epochs = 128 microsteps / 64 updates**，并使用实测量化权重预算。上一阶段 Factory/Axolotl 的原始 LoRA 报告也通过当前 Python API 外推检查。

实测为 **Blackwell GPU / Qwen3.5-0.8B text backbone / 64 条合成 Alpaca 数据 / accumulation 2 / seed 42 / SDPA / 无 packing / 无周期性 eval 或 checkpoint / 最终 adapter 保存**。GC 场景使用 `use_reentrant=true`；QLoRA 为 NF4、double quantization、BF16 compute/storage 与 FP32 adapter。框架版本沿用上一阶段，Factory 环境新增 bitsandbytes **0.50.2**，Axolotl 使用 **0.49.1**；两者依旧分别使用 Torch 与 FLA DeltaNet 路径，不能将其速度差归结为单纯的框架差异。

修复后冻结 **15 个执行/验收源码文件**，重新运行 **6 组、18 个独立 CUDA 训练进程**。每组 1 次 6-update 校准、2 次各 16-update holdout，每次排除前 2 updates 后比较稳态耗时。每个 holdout 为 32 microsteps：batch 2 使用 64 条数据，batch 1 使用 32 条数据；这不是长时间收敛验收。各组在正式校准前已有独立短跑，保留了共享编译/文件缓存；没有清空全局缓存，不能称为受控冷启动测试。

| 框架 / 场景 | 最大稳态 update 时间 APE | 最大整体 wall time APE | allocated / reserved 峰值范围（GiB） |
|---|---:|---:|---:|
| LLaMA-Factory / LoRA / 512 / batch 2 / 无 GC | 3.904% | 1.823% | 9.012 / 9.525 |
| LLaMA-Factory / LoRA / 1024 / batch 1 / GC | 0.369% | 0.326% | 9.607 / 10.160 |
| LLaMA-Factory / NF4 QLoRA / 512 / batch 1 / GC | 1.925% | 1.366% | 4.375 / 4.664 |
| Axolotl / LoRA / 512 / batch 2 / 无 GC | 3.476% | 0.563% | 6.002 / 6.525 |
| Axolotl / LoRA / 1024 / batch 1 / GC | 2.705% | 0.086% | 4.316 / 4.850 |
| Axolotl / NF4 QLoRA / 512 / batch 1 / GC | 0.250% | 0.133% | 2.192 / 2.480 |

表中峰值范围包括各组 calibration 和两次 holdout。独立核验逐项重算时间及显存误差、检查进程 PID、实际 shape/GC/量化签名、有限 loss 和落盘 adapter 的有限值/非零 LoRA B。原始报告与额外预热记录见 [`finetune_extended_validation_evidence.json`](tests/data/finetune_extended_validation_evidence.json)；该归档替换了主机路径，只用于审阅，原始本地报告才可作为 `--run-report` 输入。工作目录为 `/tmp/fakegpu-phase3/`，此前部分成功测试和失败测试也完整保留。

解析显存仍包含未完全覆盖的激活与 kernel workspace，短跑校准不证明其他模型、显卡、shape 或长训练零误差。整体耗时仍依赖相同 dataset/cache 状态；之前的首次编译外推失败证据继续保留，不因本轮先预热后的结果而删除。

### 接续设计与验收顺序

1. **原生 packing 与动态 batch**：从实际 sampler/collator 记录每个 batch 的序列数、有效 tokens、padding、labels 与边界；按 shape 分组校准。验收数据包含不同长度及 epoch 尾部，确保最终 microsteps 与真实 Trainer 一致，不能将通用拼接行数直接作为框架 packed 序列数。
2. **完整流程成本**：为周期性 evaluation/checkpoint 建立事件次数与阶段成本模型；分别收集首次运行和缓存后重复观测。无法确认 cache 状态时保留单独场景，不用一次固定成本代表所有运行。
3. **精度与范围扩展**：增加更长 holdout、独立数据集、OOM/低估边界与其他 GPU；再依照 R24–R26 推进 serving、diffusion 与多卡/多机。当前单 GPU 微调交付不代表这些原始路线全部完成。

## 2026-10-13 / 原生框架与真实数据闭环

本次继续 P1 的数据工作量闭环和 P3 的原生框架执行。命令目录扩展到 **24 项**；既有 HF Trainer full/LoRA/QLoRA probe 与历史观测签名保持兼容。

| 交付项 | 行为与边界 |
|---|---|
| `profile-finetune-data` / `finetune_data.py` | 全量扫描本地 JSONL，支持 text、messages/chat template 和 input_ids；记录实际 token 长度分位数、截断、padding、有效 labels、文件/tokenizer 内容指纹。显式 generic concatenate 只表示所选算法的场景，不冒充任一框架的 packing。 |
| `finetune_data_report.py` / `--dataset-report` | 独立检查报告的样本数、长度、token/padding 算术、来源与配置冲突；用实测序列数替换配置占位数量。raw scenario 与实际导出的 processed sequences 明确区分。运行时 sampler/collator packing 不能凭上游 dataset 行数推断。 |
| `run-finetune` / `finetune_runner.py` | 调用 LLaMA-Factory `run_exp` 或 Axolotl `load_cfg` / `do_train`；让框架执行自身数据、模型、Trainer 与最终保存流程。首版限本地纯文本 checkpoint、单 GPU、无分片/offload、full 或未量化 LoRA、无 packing、显式 SDPA/eager。 |
| `_finetune_observer.py` | 受控作用域内捕获框架创建的 Trainer，采集有效训练 split、实际参数 dtype/数量、kernel 实现、TF32、batch shapes、参数变化、finite loss 和完整累积窗口耗时。退出或异常都恢复临时方法；保存嵌套调用只计一次。 |
| `estimate-finetune --run-report` / `estimate_finetuning_run` | 校验框架、GPU、软件、模型元数据、checkpoint 内容、processed data 与 native execution signature。固定完整 batch 可按样本数/epochs 外推训练；`--max-steps -1` 清除短跑限制。相同 dataset/cache 且无周期性 eval/save 时，另提供一次实测固定开销加稳态训练的整体时间场景。 |
| 原生验收工具 | `tests/native/prepare_finetune_fixture.py` 在 CPU 独立导出 Qwen3.5 text checkpoint，验证无 missing/unexpected weights 且导出前后 logits 一致；`finetune_framework_matrix.py` 用独立进程执行校准与较长 holdout，对照实际训练步与整体耗时。两份 native YAML 与三语 README 提供复现流程。 |

整体时间场景包含 runner 内加载、预处理、warmup、最终保存、checkpoint hashing 和观测开销，不包含 Python 进程启动或下载。单次固定开销不能代表 cache/JIT 波动的分布；逐步 logging/观测残差的增长也可能引入长期外推误差。原始 wall time、阶段耗时、allocated/reserved 峰值和逐 update 样本都保留；不将分析范围或样本 extrema 称为置信区间。

### 本轮实际发现并修复的问题

- LLaMA-Factory collator 返回 `BatchEncoding`；采集器改为接受 `Mapping`，补充对应回归，避免训练第一步错误退出。
- Axolotl 原生 Alpaca 处理会先分别截断 prompt/response 再拼接。默认超长策略会丢弃本次全部 64 条样本；示例显式设置官方 `excess_length_strategy: truncate`，保留原生流程并得到 64 × 128 token。
- 不将框架配置中的参数存储 dtype 当作实际加载 dtype。报告根据真实模型重建冻结 BF16、adapter FP32 的计数，并将实际 kernel/collator/TF32 纳入签名。
- 数据/运行报告检查独立于 CLI；重复的 workload 字段与逐步样本必须一致，避免编辑外层计数后产生貌似匹配的错误估算。
- 拒绝原生框架从随机参数重新初始化，以及自定义模型加载/配置覆写与未绑定版本的 kernel 选项。这些选项会绕开默认 checkpoint 文件选择或改变已声明架构；不能哈希一份本地权重后声称观测了另一种加载方式。新增 7 项 CPU 拒绝回归，已验证训练路径不变。

### 验证记录

最终完整回归 **768 passed / 1 skipped，58.94 秒**；Ruff 全仓检查通过。原始工作目录为 `/tmp/fakegpu-phase2/`。先前 `91b796d9` 的 571 passed / 1 skipped 属于上一轮基线。独立安装 wheel 的 **69 个 Python 文件**与工作树逐文件 SHA256 一致；无训练依赖的环境可列出 24 项命令并访问新增命令帮助；安装包成功复用真实 Factory 报告推算 1 epoch / 64 microsteps / 32 updates。

使用独立导出的 **Qwen3.5-0.8B text backbone、64 条合成数据、sequence length 128、microbatch 1、accumulation 2、BF16 frozen weights / FP32 LoRA、rank 8 q_proj/v_proj、无 GC、无 packing、SDPA、Blackwell GPU**。每个框架分别 1 次 6-update 校准、3 次独立进程的 32-update 完整 epoch 验收，每次排除前 2 次 update 后计算稳态样本。两个框架使用自身模板与预处理，因此 raw data 一致不代表 processed tokens 或 kernel 一致。

| 框架 / 软件栈 | 最大稳态 update 时间 APE | 最大原生整体 wall time APE | 峰值 allocated / reserved |
|---|---:|---:|---:|
| LLaMA-Factory 0.9.5 / Transformers 5.6.0 / Torch 2.9.1 / PEFT 0.18.1 | 4.906% | 3.931% | 2.334 / 2.408 GiB |
| Axolotl 0.18.0 / Transformers 5.14.1 / Torch 2.11.0 / PEFT 0.19.1；已建立编译缓存 | 2.164% | 1.125% | 2.349 / 2.363 GiB |

这两组的 allocated/reserved 峰值在各自 3 次 holdout 中均与校准相同，不能据此宣称任意数据、shape 或长期训练显存零误差。全部运行有 finite loss、实际 LoRA 参数变化与最终 adapter 保存。Factory 使用 Torch DeltaNet 路径，Axolotl 使用 FLA 0.4.1 chunk kernel，实际实现保存在签名中；不把两者速度当作同一内核的框架优劣比较。

**首次运行与后续缓存状态不一致的负证据也保留。** Axolotl 初次校准的第一 update 为 **66.542 秒**，后续进程第一 update 约 **1.106–1.128 秒**。运行后 Triton autotune 文件时间与该首次 update 窗口相符，支持编译/调优开销这一解释，但没有独立剖析各项耗时，也没有清空并控制初始全局缓存。因此称为 first-use calibration，不能称为受控全冷缓存实验。该校准外推到后续缓存进程时，稳态 update 最大 APE 仍为 **1.276%**，而整体时间最大 APE 达 **364.885%**。这证明整体时间的固定成本必须匹配 cache/JIT 状态；后续 cached-repeat 结果不能掩盖这组失败边界。

完整 3 组、12 次运行的时间样本、数据/模型身份、参数变化、实际 kernel、配置、报告原始 SHA256 和对照摘要见 [`finetune_framework_validation_evidence.json`](tests/data/finetune_framework_validation_evidence.json)。归档做了路径替换，供审核复现使用，不能直接作为 `--run-report` 输入；原始本地报告才可在匹配的环境中重新核验权重后复用。

### 下一阶段

P3 首批原生文本 SFT 流程交付后，优先扩展较长序列、microbatch、checkpointing、native QLoRA 与真实 packing 的计数和校准；为 eval/周期 checkpoint 与冷/热 cache 建独立开销模型。随后按原始 R24–R26 矩阵推进长上下文/online serving、diffusion 真机与训练、多 GPU/多机 MoE。当前 GPU 验收不能替代这些尚未完成的路线，也不能把单卡结果直接外推为多卡吞吐。

## 2026-10-13 / 微调首批交付

### 本轮实现与阶段状态

**P0 / 配置与验收契约、P1 / 离线微调计划的首版已交付；P2 / HF Trainer/PEFT 真机闭环已完成指定短跑场景的验证。** 项目新增 `estimate-finetune` 和 `probe-finetune`，现有 **22 个 CLI**。这次交付解决“三类框架配置 → 单卡微调工作量、时间及 VRAM 估计 → 同配置短跑校准”的主链路；下面原始 P0–P5 计划仍保留，不能将首批交付等同于全部路线完成。

| 交付项 | 实际行为与边界 |
|---|---|
| `finetune_config.py` / 三类框架适配 | 读取 Hugging Face Trainer/PEFT、LLaMA-Factory、Axolotl 配置；归一模型与数据路径、方法、精度、adapter、optimizer、量化、batch/accumulation、序列长度和训练停止条件；冲突或不支持语义显式报错。无需导入这些框架或加载权重即可估算。Factory/Axolotl 的真实训练入口尚未在本轮运行。 |
| `finetune_estimator.py` / Python API 与 `estimate-finetune` | 从本地模型元数据和处理后的样本数、epochs/max_steps 推导 microsteps、optimizer updates、处理 token 数；分项估计权重/adapter、梯度、optimizer/master state、激活和 workspace，输出 training-loop 总时间范围及显存/覆盖报告。首版限单 GPU、无 sharding/offload；分布式仍由 `plan-training` 处理。 |
| Qwen3.5 架构处理 | 分别计算 full-attention 与 linear-attention 层，区分训练/冻结参数和 LoRA 的实际目标形状；`--text-only` 显式选择文本骨干。DeltaNet backward、可选快速 kernel 和 MoE dispatch 仍为部分解析覆盖，未校准结果不能认证 OOM 安全。 |
| `finetune_probe.py` / `probe-finetune` | 真正执行 Hugging Face Trainer；支持本轮 full、PEFT LoRA 与 bitsandbytes NF4 QLoRA。拒绝模拟 CUDA、错误 GPU、未支持框架/packing/分片；记录软件与工作负载签名、原始 update 耗时、loss、参数数量以及 startup/steady allocator 峰值。计时覆盖 batch fetching、累积的 forward/backward 与 optimizer update。 |
| 校准约束 | 对模型元数据、本地 checkpoint 内容 SHA256、实际模型路径、GPU profile、框架、精度、optimizer、adapter、量化、attention、shape 和软件版本进行签名匹配。dataset 数量/epochs 可以外推，改变执行 shape 不可直接复用；来源信息不足的手填观测与已验证采样器元数据分别标记。 |
| 样例、回归与真机矩阵 | 三类配置位于 `examples/finetuning/`；新增配置、估算器、CLI、probe 协议回归；`tests/native/finetune_matrix.py` 可重复生成配置、启动独立进程、保存观测与估算并比较 holdout。 |

输入样本数指过滤、切分和 packing **之后的训练序列数**；序列长度包含模板与 padding。首版不会自动将任意原始数据集转换为框架一致的 token/packed 数量。时间范围是 **training loop**，不含下载、预处理、模型加载、评估和 checkpoint 保存；解析场景范围与实测极值都不是统计置信区间。尾 batch 和不完整 accumulation 仍按匹配形状的平均开销外推，报告保留这一限制。

### 最终代码复核中的修复

- **空 PEFT 配置不再误判为 full。** 显式出现 `peft: {}` 或 `peft_config: {}` 仍表示请求 adapter；按 LoRA 处理并校验必需目标，与 full 配置冲突时明确报错。
- **不再静默忽略执行选项。** 自定义 `gradient_checkpointing_kwargs` / `use_reentrant_gc`、`device_map` 和非空 `optim_args` / `optimizer_kwargs` 目前不受该单卡估算/采样协议支持，因此显式拒绝；不会将用户的重算实现、模型放置、offload 或额外 optimizer 状态语义丢掉后继续输出同配置估计。
- **校准绑定权重内容。** `_local_checkpoint.py` 按 Transformers 的本地权重选择规则，对实际选中的单文件或 index + 引用 shards 流式计算 SHA256 内容清单。probe 和使用观测的 CLI 都重新绑定该身份，因此同一路径替换权重后不会继续接受旧观测；传入旧摘要或哈希期间文件变化会明确报错。哈希不反序列化权重，其 I/O 耗时单独记录并排除在 training-loop 外。
- **按真实 PyTorch optimizer 存储规则修正显存项。** 显式 `adamw_torch` / `adamw_torch_fused` 的 moments 跟随可训练参数 dtype，不额外添加 FP32 master weights；相应 SGD 状态也按实现计数。其他 optimizer 实现保留明确标注的保守预算。该修复通过真实 PyTorch optimizer state 字节数测试验证，**没有根据此次 GPU 峰值拟合 activation/workspace 系数**；它修正首批 Full 结果中重复预算的状态显存，不消除未知 workspace 风险。
- **区分估算与实际 OOM。** 解析下界超过容量时返回 `estimated_oom`；观测峰值加额外预算超过容量时返回 `budget_exceeds_capacity`。这两种状态都不冒充执行中已经发生的 OOM；部分覆盖仍保留 `uncertain`。
- **隔离真实 Trainer 计数测试的运行状态。** 修复前序 mock/框架导入留下的 Torch/CUDA 缓存影响真实 CPU Trainer 回归的问题；改为独立子进程、清空可见 CUDA 设备并使用独立 Triton 缓存目录。依赖缺失或 Transformers 版本低于该测试适用范围时跳过；已安装且版本满足但导入失败仍报错。Transformers **4.57.3** 与 **5.12.1** 各实际执行 **12 个用例**，对照非整除样本、小数 epochs、max_steps 和 accumulation 尾部的 step 计数。

这些修复已通过专项与完整回归，下面记录修复后重新执行的真机矩阵。代码修复与单卡校准完成都不代表下方全部研究路线已完成。

### Qwen3.5-0.8B 真实 CUDA 证据

本轮使用 406 的物理 GPU 1 **Blackwell GPU**，Qwen3.5-0.8B 本地预训练权重的 **text-only backbone**（752,393,024 个参数），合成定长算术问答。运行栈为 PyTorch **2.8.0+cu128**、Transformers **5.12.1**、Accelerate **1.14.0**、PEFT **0.19.1**，QLoRA 使用 bitsandbytes **0.50.2**。未安装 `flash-linear-attention` 与 `causal-conv1d`，不能将此结果直接迁移到启用这些快速 kernel 的栈。

统一配置：BF16 参数与计算、SDPA、microbatch **1**、gradient accumulation **2**、sequence length **128**、AdamW、learning rate **2e-5**、不启用 checkpointing/packing。LoRA 为 `q_proj/v_proj`、rank **8**、alpha **16**、dropout **0**，可训练参数 **319,488**；QLoRA 为 NF4、double quantization、BF16 compute、uint8 storage。QLoRA 的非量化张量/adapter 实际精度还受到 PEFT k-bit preparation 影响，不能只按 BF16 权重总量推算显存。

每种方法启动 **6 个独立 CUDA 进程**：1 个 calibration（seed 42，**2 个预热 + 4 个计时 optimizer updates**，12 条序列）和 **5 个 holdout**（seed 43–47，每个 **2 个预热 + 20 个计时 updates**，44 条序列），三种方法共 **18 个独立进程**。记录矩阵期间没有并发执行其他 FakeGPU 测试套件。所有运行得到有限正数 loss 和计时样本；这是执行与预测证据，不是模型质量评估。

| 方法 | 5 次 holdout 峰值 allocated / reserved（GiB） | 5 次 holdout 均值范围（秒 / optimizer update） | 校准均值预测的最大时间绝对误差 | 未校准 allocated 显存期望值误差 |
|---|---:|---:|---:|---:|
| Full | **7.108 / 8.936** | **0.8921–0.9425** | **4.932%** | **低估 10.400%** |
| LoRA | **2.367 / 2.445** | **0.7539–0.7781** | **1.866%** | **低估 8.106%** |
| NF4 QLoRA | **2.670 / 2.869** | **0.8223–0.8441** | **3.387%** | **低估 26.110%** |

每种方法的 calibration 与全部 5 次 holdout allocator 峰值相同，因此此处匹配观测的显存复用误差为 **0%**；这仅适用于已记录的同 shape、同软件栈，不能解释为通用显存估计精度。表中保留修正 optimizer 存储规则后仍存在的未校准显存低估，作为后续完善激活与 workspace 覆盖的依据。allocated/reserved 均来自 PyTorch allocator，不包含 CUDA context 和其他非 PyTorch 设备分配。

例如对同 shape 的 **1,024 条处理后序列 × 3 epochs**，计划给出 3,072 microsteps、1,536 optimizer updates，校准后 training-loop 期望约 **22.94 / 19.66 / 21.76 分钟**（Full / LoRA / QLoRA）。这些数字是短跑外推，**本轮未完整执行这三个各处理 3,072 条序列的训练任务**。每个进程中的计时点不是独立进程观测；新增 5 次独立 holdout 提供了更充分的同配置重复证据，仍未覆盖长时间 allocator 漂移、完整数据流水线、OOM 边界或跨模型/硬件精度，不作普适认证。

仓库中保存的数据源为 [`tests/data/finetune_validation_evidence.json`](tests/data/finetune_validation_evidence.json)，包含 **18 份观测**、每种方法 5 份 holdout 比较及最坏误差、软件版本、checkpoint 内容身份、执行代码 SHA256 和已知限制；`source_file_sha256` 已与该次验收时的源码逐项核对一致。为便于分发，其中的本地模型路径已改为显示名称，**该文件用于记录证据，不能直接作为 `--observation` 校准输入**；可复用的原始 observation、日志与配置保存在 `/tmp/fakegpu-research-20261013/qwen35-final/`。复现入口如下，应在具有上述可选依赖的 CUDA Python 环境执行：

```bash
CUDA_VISIBLE_DEVICES=1 python tests/native/finetune_matrix.py \
  --model /mnt/gm7/models/Qwen/Qwen3.5-0.8B \
  --profile "$TARGET_GPU_PROFILE" \
  --report-dir /tmp/fakegpu-finetune-reproduction --qlora \
  --holdout-runs 5 --holdout-steps 20
```

### 本轮回归与下一步

以下为本轮代码修复后的完整验证结果；后续校准协议或代码改动应重新运行相应检查。

| 检查 | 本轮结果 |
|---|---|
| 完整 Python 回归 | **571 passed、1 skipped**，57.31 秒；日志 `/tmp/fakegpu-research-20261013/python-release-final.log`；跳过项为 CPU-only PyTorch 专属场景 |
| 真实 Trainer step 计数对照 | Transformers **4.57.3** 和 **5.12.1** 各 **12 个实际训练用例**通过；与估算器的 optimizer/microstep 计数一致 |
| 原生 smoke 与 CPU simulation | 均通过，包括库边界、capability、coordinator、82 profiles / 15 CC、8 组 cuBLAS 数值测试与 PyTorch matmul 对照 |
| 严格 validation manifests | smoke **7/7**、research **39/39**；均无失败或跳过 |
| 发布包 | 最终 wheel 内本轮模块与工作区逐字节一致；在 `cwd=/tmp`、`PYTHONPATH` 仅包含解包 wheel 和 bitsandbytes extras 的条件下，复用真实 LoRA 观测执行估算成功，得到 **1,536 optimizer steps**、`matched_real_measurement_extrapolation` 与 `observed_probe_fits` |
| 真实微调矩阵 | Full / LoRA / NF4 QLoRA，**18 个独立 CUDA 进程**全部完成；每种方法 1 次校准、5 次独立 holdout，原始采样与 JSON 比较完整 |

### P3 框架真实执行的环境准备审计

只读核对本地 `/home/l1ght/repos/LLaMA-Factory`：工作区干净，`main` 为 `820ed764c47ede8f6ae56da89dc48aadc1b000ba`（2025-05-16），版本 **0.9.3.dev0**；该快照已有 Qwen3 模板，未包含 Qwen3.5 支持。其依赖上限为 Transformers **4.51.3**、PEFT **0.15.1**、Accelerate **1.6.0**，与现有用于本轮验证的环境不兼容。`fakegpu` 与 `torch` 两个 conda 环境均未安装 LLaMA-Factory、Axolotl 或 TRL；现有 Transformers 分别为 **4.57.3** 和 **5.12.1**。

因此 P3 的下一项交付是建立支持 Qwen3.5 的现代框架隔离环境、锁定依赖与版本，再接入真实训练入口和同工作量观测协议。不能通过关闭版本校验强行运行旧快照，并将配置适配测试报告为框架执行通过。本轮没有改动该旧 LLaMA-Factory 仓库或现有环境的依赖约束。

继续按依赖推进：先扩展 P2 的独立长跑、更多 sequence/batch/checkpointing 组合及低估/OOM 边界；再为 LLaMA-Factory、Axolotl 增加各自真实 runner 和相同工作量对照。补齐 tokenizer/chat-template/packing 的实际数据统计后，才能将原始数据文件直接作为可靠的训练工作量输入。尚未完成的 README 路线包括长上下文与在线 serving 广泛真测、diffusion 真机/训练、更多 GPU/多机 MoE 与分布式验证、原生可执行算子扩展；它们仍按下方验收矩阵逐项交付，不因本次微调入口完成而勾选。

## 2026-10-13 / 功能闭环计划：README 承诺与微调成本预测

### 目标与验收口径

用户确认按 **Hugging Face Trainer/PEFT → LLaMA-Factory → Axolotl** 的顺序推进，使用 Qwen3.5 系列与本地合成数据建立验收。两个目标是：逐项实现 README 声明的功能；给定模型、框架配置、样本规模、训练轮次和目标 GPU，预测总训练时间与峰值 VRAM，并能通过小规模真实训练校准。

本节是基于 `fcc0f848` 的实施前审计与交付计划，不能把这里列出的待办当作已交付功能。审计逐项核对了三个 README、20 项 CLI 注册、生产模块、维护测试和历史证据摘要。2026-10-13 单独重跑 README 契约测试为 **4 passed**；该检查证明声明中的命令、证据数字与示例约定一致，不等同于所有硬件场景已验证。该基线的完整回归结果另见下方重构验证表。

采用三个状态维度：**实现**表示有可达代码；**维护验证**表示有对应测试或可复现的本地执行；**真实精度**表示与指定模型、形状、软件栈和真实 GPU 的独立观测对照。公式测试通过不能升级为真机精度通过，配置能解析不能升级为整个框架能训练。

### README 功能验收矩阵

下面覆盖三语 README 的使用场景、命令表、已勾选 roadmap 和未勾选 roadmap。相同行为在不同章节重复出现时合并为一个验收项。

| ID | 功能承诺 / 入口 | 基线实现与维护证据 | 尚需完成的验收或边界 |
|---|---|---|---|
| R01 | 安装、原生库、GPU 目录；`doctor` | `_api.py`、`profile_catalog.py`、`doctor.py`；82 profiles / 15 CC；wheel 独立导入及 native 初始化已验证 | 目录元数据不等于各卡的实测性能；各发布平台仍要跑自己的安装矩阵 |
| R02 | CPU PyTorch FakeCUDA、训练控制流；`demo` | `torch_patch.py`、`_upstream.py`、`_allocator.py`；runtime、allocator、cross-device、profile、stage 测试 | 仅维护的 PyTorch 运算；不承诺二进制 CUDA 扩展或任意 kernel 数值执行 |
| R03 | 原生 CUDA/NVML/cuBLAS/NCCL 拦截与模式切换 | `src/`、`_api.py`；native smoke、库边界、物理 GPU 掩码与 hybrid/passthrough 对照 | `simulate`、`hybrid`、`passthrough` 必须分别列明真实计算和模拟设备的边界 |
| R04 | 原生 API 分类和严格策略；`capabilities` | `native_api_capabilities.json`、`capabilities.py`；104 显式 API、policy/source/export 审计 | 未执行 API 不得伪装为正确数值结果；新增数值支持必须有参考对照与失败路径 |
| R05 | 运行前 fit/OOM 检查；`preflight` | `preflight.py`、`_preflight_*`；阶段执行、OOM、timeout、报告回归 | 未覆盖的算子/workspace 不得给无条件安全结论；backend 私有内存需同栈校准 |
| R06 | 静态 tensor/workspace 显存与 FLOP | `memory_estimator.py`、`workspace_*`、`flop_counter.py`；静态生命周期、算子 profile 和 workspace 测试 | 动态 shape、自定义算子、未知 workspace 的误差仍需单独观测 |
| R07 | 仓库、依赖、CUDA/PTX/SASS 资源分析；`analyze-repo` / `analyze-kernel` | `repository_analyzer.py`、`kernel_analysis.py`；仓库夹具、occupancy/PTX/SASS 测试 | 静态扫描不能证明所有动态入口可运行 |
| R08 | LLM 权重、KV、adapter、MoE、FLOPs；`estimate-llm` | `llm_estimator.py`；safetensors header、MHA/GQA/MQA/MLA、量化、adapter、MoE 回归 | Qwen3.5 hybrid attention 尚不能直接套用全层 GQA 公式；vision 激活与 FLOPs 已在英文 README 标为未建模 |
| R09 | KV 动态/静态/量化/分页/滑窗 | `llm_estimator.py`、`_serving_kv.py`；字节公式、block rounding、CLI 与 research manifest | 模型 cache 实现、量化元数据、kernel workspace 需匹配真实 runtime |
| R10 | 混合请求、continuous-batch 容量、chunked prefill、prefix sharing；`plan-serving` | `serving_plan.py`、`_serving_*`；generic/vLLM 预算、请求清单、准入测试 | 已实现容量模型；到达流量、排队、抢占、淘汰、吞吐/延迟不是该功能当前承诺 |
| R11 | 独立 draft speculative 显存与准入 | `_serving_speculative.py`；双模型权重/cache、verify transient、signature 回归 | 未验证真实 speculative runtime；固定 acceptance 假设不能当作实测加速比 |
| R12 | SD1.5、SDXL、PixArt、本地 UNet/cross/joint DiT 生成显存；`estimate-diffusion` | `diffusion_estimator.py`、`_diffusion_*`；分阶段、CFG、offload、切片/平铺及本地三架构测试 | **DiT generation 已有实现**；通用 DiT、video、自定义组件和所有 diffusion 真机精度尚未完成 |
| R13 | GPU-aware 分析延迟区间；`estimate-roofline` | `performance_model.py`；全部 profile 可解析、有序区间与瓶颈测试 | 分析区间不是实测 kernel 时间，也尚不是完整训练总时长 |
| R14 | DeepSpeed/Accelerate/FSDP/FSDP2 单 rank 训练显存；`plan-training` | `training_plan.py`；配置归一、阶段、分片、offload 与 research manifest | 当前需要显式 parameter/activation bytes；没有三类微调框架的数据规模、轮次与时长闭环 |
| R15 | collective 路由与链路争用；`simulate-topology` | `topology.py`；算法、ECMP、争用与完整报告对照 | 分析链路模型，不是实际集群网络性能认证 |
| R16 | compute/communication/wait/memory 时间线；`replay-trace` | `trace_replay.py`；合成 trace、rank 等待和韧性事件测试 | 后续需真实 profiler / distributed trace 的输入适配与时间基准校正 |
| R17 | 校准、重复采集、比较和可靠性门禁；`calibrate` | `calibration.py`、`_calibration_*`；协议/schema、签名拒绝、重复样本、verify/bundle 测试 | 当前内建采集器主要面向 serving；训练耗时/显存需要独立的采样协议与观测 |
| R18 | SMI 设备/进程、native 活动、NVLink/MIG/故障/健康视图；`nvidia-smi` | `smi.py`、`_smi_*`、`src/nvml`；SMI、native NVML 和 bounded publisher 回归 | 拓扑/MIG/故障为建模控制面；不是硬件分区管理、真实 ECC 或故障注入执行 |
| R19 | Prometheus/JSON、有界历史与健康服务；`metrics` | `metrics.py`；cardinality、历史隔离、HTTP、坏状态降级测试 | 内存历史是短期观测，不承诺持久化监控服务 |
| R20 | workspace 目录验证；`workspace-profiles` | `workspace_profiles.py`、`workspace_cli.py`；profile schema、匹配与 CLI 测试 | 未匹配的 profile 应显式保留覆盖缺口 |
| R21 | 声明式 JSON/TOML/YAML 校验；`validate` | `validation.py`、schema、两份 manifest；7 smoke + 39 research 全通过 | 条目必须区分合成/CPU/真实 GPU；退出码、JSON 检查通过不自动证明精度 |
| R22 | coordinator、TCP payload 和数值 collective；`coordinator` / `bandwidth` | `distributed_cli.py`、`src/distributed`；两缓冲路径数值测试、本机双逻辑节点 TCP 通过 | 尚缺真实跨机器与多栈验证；不可替代 NCCL/NVLink/RDMA benchmark |
| R23 | roadmap：扩大原生可执行 CUDA/cuBLAS | 已有 8 组 cuBLAS 数值测试，其余高风险 API 明确分类 | **待扩展**。先从所选研究 workload 实际触发的 unsupported API 建有限清单，再逐个实现与参考比对 |
| R24 | roadmap：长上下文/online serving 真实 GPU 验证 | 短序列 Transformers sampler 已有历史执行证据，generic/vLLM 容量已建模 | **待验证**。增加真实长上下文、动态请求、prefix/cache/预留策略对应观测和误差门禁 |
| R25 | roadmap：diffusion 真机、训练及 DiT 显存 | generation 和部分 DiT 已实现；无 dedicated diffusion training estimator | **部分实现**。拆成 generation 真测、training 模型、扩展 DiT 三项独立验收 |
| R26 | roadmap：更多 GPU/软件栈的 distributed/MoE | 分析模型、coordinator 与历史固定 GPU 结果存在 | **待验证**。指定版本、实际 TP/PP/EP/ZeRO/FSDP 组合和多机基准；本机逻辑节点不计入跨机完成 |

基线 README 需要同步的说明：三语均保留历史精度范围和 immutable source revision；不能把历史 Qwen SFT/NF4 结果当作本轮框架验证。英文已有多模态未建模声明，简繁中文应同步。`Current repository verification` 的 2026-08-05 / 223 passed 是历史记录，应标明日期或用新的完整证据替换；不要与下方 2026-10-13 的 323 passed 混用。roadmap 的 DiT 一项应拆细，避免让已实现 generation 与尚未完成 training 互相掩盖。

### 微调预测的最小完整产品

输入必须包含：模型路径/版本、目标 GPU、框架与版本、训练方法、优化器及精度、microbatch、gradient accumulation、序列长度或真实 token 分布、样本数/数据文件、epochs 或 max_steps。仅“样本数 + epochs + GPU”不足以确定显存或时长；未给的必要参数应在报告中明确显示假设，不能静默当作短序列。

| 层次 | 需要实现的行为 | 验收方法 |
|---|---|---|
| 框架配置适配 | 三类配置归一到同一训练描述，保留原字段来源；读取 full/LoRA/QLoRA、精度、checkpointing、packing、batch/accumulation、数据规模和轮次 | 三份等价配置得到相同核心描述；冲突精度、无效 batch、未知量化和不支持训练 stage 明确报错 |
| 数据规模 | 保留样本数、有效 token 数、padding 后执行 token 数、长度分位数、被截断样本数；明确是否已应用 chat template/packing | 固定 seed 合成 JSONL；真实 tokenizer 后记录统计；验证 packing 开/关、尾批次与截断 |
| 训练计划 | 展示每 epoch microsteps、optimizer updates、总 token、有效全局 batch；max_steps 优先级和尾部 accumulation 与指定框架版本一致 | 与真正 Trainer 的 global_step、实际 batch 数对照，覆盖不能整除与小数 epochs |
| VRAM | 分别计权重、可训练参数、梯度、optimizer/master state、saved activations、logits/loss、临时 workspace；显示 allocated / reserved / process 口径 | 比较 load、forward、backward、首次及稳态 optimizer step；峰值取各阶段最大值，不能乘以 epochs |
| 总时长 | 分离 load/tokenize/compile/warmup、训练 microstep、optimizer step、eval、checkpoint save；给分析区间与校准预测来源 | 用预热后多个独立样本预测未参与校准的较长训练；报告总时间误差和 step 吞吐波动 |
| 校准约束 | 对模型结构/版本、shape、方法、optimizer、backend、kernel、GPU 和软件栈建立签名；保留原始 step 样本 | 匹配配置可复用；不同 GPU/框架/序列/量化的样本不直接覆盖预测，外推必须显式标注 |
| 输出 | 人类可读表 + JSON，包含训练计划、阶段显存、fit/headroom、总时长范围、覆盖缺口与复现配置 | CLI/API/schema 一致；缺证据显示 `Modeled` 或 `uncalibrated`，不虚构“置信度” |

时间应按执行工作量推导，例如在固定 shape 的明确假设下使用 `训练 microsteps × 对应 microstep 时间 + optimizer updates × update 开销 + 显式固定/周期性开销`。若直接测的是包含 update 的完整 optimizer step，则不能重复加 update 开销。LoRA 的可训练参数少，不代表前向/反向计算量按同一比例减少；QLoRA 还包含反量化 kernel 的成本。未校准 GPU 上的 roofline 结果保持分析区间，不能自动沿用另一张卡的 tokens/s。

适配器以官方参数语义为准。HF Trainer 使用 `per_device_train_batch_size`、`gradient_accumulation_steps`、`num_train_epochs`/`max_steps`；LLaMA-Factory 的 `max_samples` 是**每个 dataset** 上限，多个数据集不能直接当成总样本数；Axolotl 的 `sample_packing` 会改变实际 batch 数，不能只乘样本数和 `sequence_len`。[HF Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)、[LLaMA-Factory arguments](https://llamafactory.readthedocs.io/en/latest/advanced/arguments.html)、[Axolotl config reference](https://docs.axolotl.ai/docs/config-reference.html)

### Qwen3.5 首批验收与架构约束

首个验收模型选 **Qwen/Qwen3.5-0.8B**，随后扩展到同系列 2B。0.8B 官方配置是 dense text backbone 加 vision encoder，24 层中包含 18 个 Gated DeltaNet 和 6 个 full-attention 层；不能因为家族介绍含 MoE 就把 0.8B 判为 MoE，也不能按 24 个标准 attention 层计算全部 cache/activation。官方 Transformers 文档分别提供 dense 与 MoE 架构；DeltaNet 的 kernel 选择会影响时间与显存，必须记录 `fla`、`causal_conv1d` 与实际执行路径。[官方 0.8B 模型卡](https://huggingface.co/Qwen/Qwen3.5-0.8B)、[官方 config](https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/main/config.json)、[Transformers Qwen3.5](https://huggingface.co/docs/transformers/model_doc/qwen3_5)

- 初期使用纯文本 SFT、单卡、固定长度与不 packing 的样本；明确加载的是完整 conditional-generation 模型还是 text-only causal LM。即使输入只有文本，若加载了视觉权重，驻留显存仍需计入；视觉模块是否训练应单独记录。
- 构造带固定 seed 的简单问答/指令 JSONL，并保留 tokenizer、chat template、截断后的统计。短 smoke 可用少量样本和 128-token shape；随后扩展到 512/1024-token、不同 microbatch、accumulation 与 checkpointing。短 smoke 不作为长上下文精度结论。
- 先跑 full 与 LoRA，再跑明确 backend 的 QLoRA；原生 NF4 历史实验不能替代 bitsandbytes 4-bit 或其他量化实现。LoRA 应按实际匹配模块的输入/输出维度计数，不可假设所有层都有相同 `q_proj/k_proj/v_proj`。
- 真实采样拒绝 FakeCUDA/simulate；记录 loss 有限、参数确实更新、模型/框架/torch/CUDA/GPU 版本、峰值 allocated/reserved 及同步后的耗时。
- 校准与验收分开：短跑样本拟合 step 时间，独立更长运行验收总时长；新发布的显存精度数据按 README 要求保留至少五次独立观测及最坏低估。合成数据只验证执行与预测，不评价模型质量。
- 图像/视频、多机 MoE、long-context DeltaNet backward、fused loss、特殊 optimizer 在各自完成支持与观测前标记为待扩展；纯文本单卡结果不覆盖它们。

### 按依赖排序的交付阶段

| 顺序 | 交付物 | 依赖与退出条件 |
|---|---|---|
| P0：承诺审计与配置契约 | 本验收矩阵；三语边界同步；三个框架配置适配、版本化 JSON 报告与样例 | 先锁定 full/LoRA/QLoRA 和纯文本 SFT 范围；解析成功与执行验证分别列状态 |
| P1：离线微调计划 | 参数/adapter 计数、数据工作量、阶段 VRAM、profile 时间区间；Qwen3.5 hybrid-aware 处理或明确不支持项 | CPU 公式与属性测试、框架等价配置、极端/非法输入、CLI/API 一致；无需先下载全部大模型 |
| P2：HF Trainer/PEFT 真机闭环 | Qwen3.5-0.8B + 合成 JSONL；分阶段测量、训练观测协议、校准与独立验证 | 真正 Trainer 跑 full/LoRA，随后指定 backend QLoRA；吻合 step 计数，输出未参与校准运行的显存/时间误差 |
| P3：LLaMA-Factory 与 Axolotl 执行适配 | 各框架真实入口、环境检查、配置/数据导出、同格式测量结果 | 两个框架分别真实执行相同小模型/数据；记录自身依赖/模板/packing；配置归一测试不能替代这一步 |
| P4：补齐 README 研究验证缺口 | R24 长上下文/serving、R25 diffusion generation 真测与训练估计、R26 distributed/MoE 多栈 | 复用观测/比较协议；每类给独立数据、精度门禁、失败/OOM 记录和明确支持矩阵 |
| P5：原生数值扩展与实验工具包 | 按 R23 的需求清单扩展；统一实验 ID、参数 sweep、恢复、compare、图表与原始样本索引 | 每个原生 API 有真实数值参考与严格错误测试；完整实验从配置到报告可复现，旧功能回归通过 |

P4 的 diffusion 与多机工作在 P2 协议稳定后可并行；P5 中实验产物目录和签名应在 P2 先使用最小实现，再扩展成通用 runner。每个阶段独立提交，不以勾完 roadmap 为目的降低已公开的验证要求。

本机可立即开展 P0/P1 的 CPU 归一与公式验证、P2 的 Blackwell GPU 单卡小模型真实训练，以及 R24 的单卡 Transformers 长序列采样。P3 需要逐一验证安装栈和实际启动命令；R26 的真实多机/多栈证据不能仅靠本机增加逻辑 rank 补齐。当前机器另一张 secondary GPU 卡与 Blackwell GPU 属于不同设备环境，不能未经对应校准就当作同构多卡吞吐测试。

## 2026-10-13 / 406：完成度、逐部件检查与实施结果

### 基线选择

在 `/home/l1ght/repos/fakeGPU` 执行 `git pull --ff-only origin main` 后确认分叉：本地独有 `8b7a7ec3`，远端独有 107 个提交，远端顶端为 `3ba1dcaf`。本地提交是较早的快照合入，远端已经有模块拆分、功能扩展和后续 CUDA/cuBLAS 修复，因此以 `origin/main` 为基线更合理。

先创建 `backup/406-main-before-refactor-20261013` 保存本地原始提交，再 rebase。冲突处保留远端已经演进的实现，本地独有且仍适用的运行时配置回归测试迁移到 `tests/test_runtime_config.py`，形成 `e6f15727`。没有把旧快照覆盖回新代码，也没有推送远端。旧目录下本机已有的 Shakespeare 数据保留在原处，不纳入提交。

### 项目完成度

**现有项目已具备可用的研发预检、模拟显存/OOM、容量规划和分析工具链。** Python 运行时、原生库、命令行、报告、校验清单以及源码包到 wheel 的安装链路均有可运行实现和测试。当前有 82 个 GPU profile、15 种计算能力、20 个 CLI 命令。

尚不能据此认定为完整的 CUDA 替代实现，或对任意模型、硬件与分布式部署给出精确数值和性能保证。项目没有统一的需求验收总表，因此不把测试通过率当作“完成百分比”。以下按实际能力、代码边界和本轮证据评估。

| 部件与主要代码 | 当前完成情况、检查结论及本轮处理 |
|---|---|
| 公开 API / 运行时：`_api.py`、`_runtime.py`、`__init__.py` | `init/env/run`、native/FakeCUDA 选择、配置转发和环境注入链路可用。检查模式间库边界，修复物理 GPU 掩码导致 hybrid 枚举失败，以及重复构造环境时遗留旧 preload 的问题；公开函数参数保持兼容。 |
| PyTorch 模拟：`_upstream.py`、`torch_patch.py`、`_ecosystem_compat.py`、`_cross_device.py`、`_torch_reporting.py` | 具备 CPU 支撑的 FakeCUDA、设备迁移、autograd/optimizer 分类、distributed/FSDP 兼容与报告。合并重复迁移钩子和 CUDA memory API 安装表，删除对保证存在的框架模块的静默兜底；保留 storage 访问、可选优化器及框架差异的必要保护。 |
| 显存与 profile：`_allocator.py`、`_profiles.py`、`profile_catalog.py`、`_stage.py` | 分配、缓存、峰值、释放、OOM、stage 与混合设备配置均有实现。统一 profile 元数据构造；保持未知 profile 的既有默认值、补齐/截断和独立设备字典语义；82 个 profile 全部验证。 |
| 原生公共层：`src/core/`、`src/monitor/` | 全局状态、profile、监控、真实库加载及 hybrid 显存管理仍是活跃路径。删除未参与构建、无人使用的 `dl_intercept.cpp` 原型；保留真正被 driver/runtime 使用的加载与转发逻辑。 |
| CUDA Driver / Runtime：`src/cuda/` | 分配、复制、设备查询及 native 转发链路可用；通用模拟 kernel launch 仍不能等同于实际执行任意 CUDA kernel。删除无消费者的旧 dispatch 头和未链接进产物的旧 runtime 对象目标及配套源码，保留实际 `cudart_stubs.cpp` 实现。 |
| cuBLAS：`src/cublas/` | 维护范围内有 CPU 数值执行、精度/compute type 与 Lt epilogue 策略；unsupported API 有明确策略。沿用远端最新修复，不把有语义差异的路径机械合并；8 个 C++ 数值用例及 PyTorch matmul 校验通过。 |
| NCCL / collective：`src/nccl/`、`src/distributed/collective_*`、buffer/staging | 有 collective 数值执行、归约和两种缓冲传输；高级设备/RMA API 仍有不支持项。抽取 5 处相同错误构造，broadcast 复用根缓冲，ReduceScatter 直接写归约结果切片；新增两种传输的数值与错误验证。 |
| 协调器 / 传输：`cluster_*`、`communicator.*`、`transport_*`、`topology_model.*` | 本地共享内存与 TCP 协调可运行，报告与超时诊断路径保留。本机两个逻辑节点的 TCP payload 校验通过；没有把它当成真实跨主机/NVLink/RDMA 性能验证。 |
| NVML / SMI：`src/nvml/`、`native_smi.*`、`smi.py`、`_smi_*` | 模型设备、进程、显存、拓扑、MIG 和故障视图有实现。保留旧 state schema 读取和可选字段的兼容语义；smoke 验证原生 state 与内存类型等报告。模拟字段不能替代物理设备遥测。 |
| Metrics：`metrics.py` | Prometheus、健康状态、有界历史与 HTTP 服务可用。内部只读渲染复用已发布快照，深拷贝移出 history 锁；公开快照/历史仍返回独立副本，新增嵌套修改隔离回归。 |
| 静态显存 / workspace / calibration：`memory_estimator.py`、`operator_profiles.py`、`workspace_*`、`calibration.py`、`_calibration_*` | 具备算子/工作区估计、签名约束的校准与比较流程。保留对未知算子、外部样本和缺失元数据的诊断；估计适用性依赖已覆盖的算子和匹配的软件/工作负载，不能从一张卡的小模型验证外推到全部模型。 |
| LLM / serving：`llm_estimator.py`、`llm_cli.py`、`serving_plan.py`、`_serving_*` | 权重、KV、prefill/decode、混合请求、prefix cache、draft speculative 等容量模型已有完整报告路径。远端已有共享容量/准入逻辑，未重新并成充满 mode 分支的巨函数；仍缺广泛的真实长上下文与在线调度验证。 |
| diffusion / training：`diffusion_estimator.py`、`_diffusion_*`、`training_plan.py` | 固定/本地 diffusion 元数据与训练配置的阶段、rank 显存估算可用。保持模型类型和缺失组件元数据的显式报错；自定义 pipeline、DiT/训练覆盖与更多 MoE/分布式配置仍需扩展和真机验证。 |
| 仓库 / kernel / 性能分析：`repository_analyzer.py`、`kernel_analysis.py`、`performance_model.py`、`flop_counter.py` | 静态入口扫描、kernel/API 清单、FLOP 与 roofline 分析可用。分析结果存在模型假设和未知项，保留这些边界，未用静默默认值掩盖无法分析的内容。 |
| topology / trace：`topology.py`、`trace_replay.py` | 路由、collective 流量、拥塞与时间线报告可用。路由时累计链路最大 payload，消除每条链路反复扫描全部 flow；36 份完整报告与修改前一致，包括 ECMP/算法与消息规模组合。 |
| preflight / CLI / validation：`preflight.py`、`_preflight_*`、`_cli.py`、`validation.py`、`structured_io.py` | 子进程预检、报告聚合、CLI 注册和严格清单校验可用。保留不同命令已经公开的输出/退出码约定与真实可达的错误分支；7 个 smoke + 39 个 research case 全部成功且无跳过。 |
| 构建 / 发布 / CI：CMake、`setup.py`、`scripts/`、`.github/workflows/` | 原生 Release 构建、Python/native 测试入口和 wheel 打包链完整。删除无用构建目标，把 collective 回归加入 smoke；从 sdist 重建 wheel 后独立导入、发现内置库并初始化成功。本轮实跑 Linux/Python 3.12，未声称跑过全部 CI 平台。 |

### 本轮修改及保留边界

- 删除 4 个无有效消费者的原生文件，共 **1,351 行**：`cuda_mode_dispatch.hpp`、`cuda_stubs.cpp`、`cuda_defs.hpp`、`dl_intercept.cpp`。删除对应无用/注释构建项。实际产物的 820 个 CUDA/cuBLAS/NVML/NCCL C 导出保持一致。
- 收敛 profile 解析、PyTorch 迁移与 memory API 钩子、collective 错误构造。减少无意义默认值、重复安装、缓冲复制、metrics 深拷贝和 topology 重复扫描。
- 不以“减少 except”为目的移除真实边界保护：可选框架能力、外部 state/schema、不可访问 storage、未知模型/算子及 unsupported API 仍需要兼容或明确报错。
- 基线中已有多轮重构成果；历史报告列出的很多问题已经解决。本轮没有按旧行号重复修复，也没有为了减文件长度而继续拆分已有稳定模块。

测试中复现并修复的两个 bug：

1. **hybrid + `CUDA_VISIBLE_DEVICES` 枚举错误。** 指定物理 GPU 1 时，PyTorch 会把物理编号再次应用到 FakeGPU 的单设备 NVML 清单，得到 `device_count() == 0`，随后 `get_device_name(0)` 失败。现在只在有物理可见性掩码的 hybrid 环境中保留真实 NVML；未设置掩码时维持模型 NVML。子进程按自己的环境作决定，覆盖数字、UUID 和空掩码。
2. **模式切换残留 preload。** 对已有环境反复调用 `env()`，从 simulate 切到 hybrid/passthrough 后仍可能加载旧的 fake cudart/cuBLAS，重复调用还会累加路径。现在移除当前库目录的有/无版本号 FakeGPU preload，再加入所选模式需要的库，保留其他库及其顺序；Linux 同时支持空白与冒号分隔。此修复针对子进程环境构造，不承诺卸载当前进程已经加载的库。

### 验证证据与复现

环境：406，Linux，Python **3.12.12**，PyTorch **2.9.1+cu128**；真实计算使用物理 GPU 1 **Blackwell GPU**，驱动 **580.178.04**。构建目录为 `build-refactor-20261013`，本轮日志/JSON 在 `/tmp/fakegpu-refactor-20261013/`（临时本机产物，不作为仓库资产提交）。

| 验证 | 结果 |
|---|---|
| rebase 后修改前 Python 基线 | 308 passed、1 skipped |
| 最终完整 Python 回归 | **323 passed、1 skipped**；跳过项专用于 CPU-only torch，本机安装的是 CUDA torch |
| 两处 bug 的新增回归 | 修复前出现失败，修复后通过；涵盖掩码、子进程环境、native init 与重复模式切换 |
| 修改前后完整报告对比 | topology 36 组、profile 配置 30 组，共 66 组完全一致 |
| Native Release + smoke | 库边界、104 项 capability 清单、unsupported/cuBLAS 策略、内存类型、coordinator、82 profile / 15 CC 矩阵通过 |
| 新增 collective C++ 回归 | SocketPayload / SharedMemory 共 **28 个数值用例 + 28 个非法大小用例**通过；覆盖 6 种 collective 及 5 种归约运算 |
| Native CPU simulation | 8 个 cuBLAS C++ 用例通过；CUDA torch 的模拟 matmul 与 CPU 结果相符 |
| 严格 validation manifests | smoke **7/7**、research **39/39**，无失败、无跳过 |
| 本机 TCP payload 集成 | 2 个逻辑节点，每节点 1 rank，64 KiB，1 次预热 + 3 次迭代通过 |
| 真实 GPU：real / passthrough / hybrid | MLP + AdamW 3 步训练的 loss、参数 SHA256 与 PyTorch 峰值显存一致；峰值 **18,323,968 bytes**；物理编号和 GPU UUID 选择均通过 |
| 源码包与 wheel | sdist → wheel 编译成功；从解包后的 wheel 独立导入，82 profile、内置原生库发现、masked hybrid 环境和 native 初始化通过 |
| 静态检查 | Ruff、`git diff --check`、shell 语法通过；修改模块的公开顶层函数签名与基线一致 |

主要可复现入口（先激活有 PyTorch/pytest 的 Python 环境）：

```bash
FAKEGPU_TERMINAL_REPORT=0 scripts/test.sh python
BUILD_DIR="$PWD/build-refactor-20261013" FAKEGPU_TERMINAL_REPORT=0 scripts/test.sh smoke
CPU_BUILD_DIR="$PWD/build-refactor-20261013" FAKEGPU_TERMINAL_REPORT=0 scripts/test.sh cpu
python -m fakegpu validate --manifest tests/data/validation_smoke.yaml --report-dir /tmp/fakegpu-validation-smoke --strict
python -m fakegpu validate --manifest tests/data/research_validation.yaml --report-dir /tmp/fakegpu-validation-research --strict
python tests/native/hybrid_visibility.py --build-dir build-refactor-20261013 --device 1
```

`hybrid_visibility.py` 要求真实 CUDA 设备，失败会报错，不通过跳过来伪装成功。本轮结果支持上述已测路径的行为保持与缺陷修复；仍未覆盖 macOS 实机、其他 PyTorch 版本、真正跨主机集群和大规模真实模型。README Roadmap 中的原生算子扩展、在线 serving/长上下文、diffusion 与多栈分布式验证仍属于后续工作。

## 实施进度（滚动更新）

**2026-08-16 第一批落地**：§8 建议顺序中的第 2 步（纯删除）全部完成，第 3 步（公共设施）完成两项，第 6 步（性能）完成两项。净删约 **2,500 行**（+250/−2,700 量级），功能不变。

| Commit | 阶段 | 内容 |
|---|---|---|
| `e451bd2` | §1.1 | 删除 torch_patch standalone 不可达路径（约 50 个定义、~1,200 行）；`_activate_upstream` 改为显式 raise；`_runtime` 恒真探测简化；顺带删除与 upstream 逐字相同的 no-op 补丁 |
| `bc77d9f` | §1.2 | 删除零消费者的 `fsdp_memory.py`（970 行；可从 git 历史恢复） |
| `6b00725` | §1.3–1.7 | 恒 None 遥测链整链删除（含 `native_smi.cpp`）；死状态字段（`memory_utilization_percent`、`reserved_stage_peaks`）；死 env 导出与 `efficiency` 死参数；各处局部死分支；`_preflight_bootstrap` 冗余兜底 |
| `432c21c` | §2.2 | 同体重复 except、不可能失败的 try、恒真 hasattr、setup.py 死 wheel 兜底、17 处"子类⊂基类"冗余异常项 |
| `91243b8` | §5.2 | workspace catalog 每 FX 节点重载 → 每图一次；torch 版本串 lru_cache |
| `f6891c8` | §3.4+§5.4 | `structured_io.emit_json` 收敛 8 处 print-or-write `--json` 块并消除双重序列化；dtype 表与 `_iter_tensor_leaves` 各去重一份 |

**验证矩阵**：每阶段本地全量 `pytest`（222 passed）+ ruff；`scripts/test.sh smoke` 完整通过（含 native C++ 重建，覆盖 `native_smi.cpp` 改动）；CI 同款 manifest 校验（7+39 cases）通过；远端真机单测——**406**（Python 3.12 + torch 2.9.1+cu128，`~/refactor-test/FakeGPU`）与 **gem12**（torch 2.12.1+cu130）均 221 passed + 1 skipped（skip 为 CPU-only 构建专属测试，预期）。

**2026-08-17 第二批落地**：§8 第 3 步（公共设施）剩余项之一完成。

| Commit | 阶段 | 内容 |
|---|---|---|
| `a9da7f8` | §3.3 | 新增 `_api._FakeGpuRuntimeConfig`（frozen dataclass，10 个 FAKEGPU_* 配置字段）收敛 `_api.py`/`_runtime.py` 内部转发；`init()`/`env()`/`run()` 三个公开入口的关键字签名保持逐字不变（外部调用方零感知），仅内部把 10 个散装 kwarg 换成一个 `config` 对象；新增 `_build_env()` 消除 `run()`→`env()` 的整段 10-kwarg 转发；`_apply_env_inplace`/`_apply_config_env_inplace`/`_apply_config_env`/`_runtime._init_native_runtime` 四处签名同步收敛。净减约 **77 行**（220 处/− 变化集中在 `_api.py`），行为不变 |

验证：本地全量 `pytest`（222 passed，与改动前一致）+ ruff clean；CI 同款 manifest 校验（7+39 cases）通过；`scripts/test.sh smoke`（含 native 重建）通过；额外手工冒烟——`fakegpu.env()`/`fakegpu.init(runtime="native")`/`fakegpu.run()` 三个公开入口直接调用验证派生的 `FAKEGPU_*` 环境变量、`device_count` 校验报错、子进程环境注入均与改动前一致。远端 406/gem12 真机复测见第一批的验证矩阵说明；`f6891c8`（emit_json，改动了 CLI `--json` 输出代码路径）与 `a9da7f8` 已于 2026-08-17 在两台机器上复跑通过（各 221 passed + 1 skipped）。

**2026-08-17 §7 bug 批量修复**：12 项中 10 项确认为真实缺陷并修复（各一个 `fix:` commit），1 项（#6）在修复过程中发现是**未完成且有 bug 的半成品**（假设了 catalog schema 里不存在的 `"group"` 字段，`native_capability_report()` 一调用就 `KeyError`）——已改正为正确实现；1 项（#12）复核后判定**不是 bug，未改**。

| Commit | Bug # | 内容 |
|---|---|---|
| `1a7a337` | #1 | rank 正则去掉多余转义，ShardMetadata placement 重写恢复生效 |
| `3ab2fb8` | #2 | 释放回调改为绑定分配时的 tracker 实例（storage finalizer + saved-tensor 两处），二次 `patch()` 后不再向新 tracker 误释放 |
| `14a9f5f` | #3 | `successful_writes` 改为写入前预增、失败回滚，去掉序列化里的 `+1` 特例 |
| `d519e97` | #4/#5 | `_evaluate_json_check` 捕获元组补 `IndexError`；`finished_at_ns`/`duration_seconds` 改为同一时钟读数 |
| `2f483f5` | #6 | 过滤下沉到 `native_capability_report()`，在算 summary **之前**过滤 apis（并修正了 WIP 里假设不存在的 `"group"` 关联字段导致的 KeyError）；`group_count` 保持目录级、不随 API 过滤变化（匹配既有测试 `== 5`） |
| `6a87c50` | #7 | 超时 rank 仍读取已写出的诊断 JSON；顺序 join 改为共享 deadline，避免 N 个慢 rank 把超时叠加放大 |
| `a270e59` | #8 | 文本编码器均未暴露宽度且 denoiser 也未声明时改为 raise（而非静默取 1）；真正无文本编码器角色的管线不受影响 |
| `19e1999` | #9 | `_bool_value` 只保留已通过测试验证的 `"auto"`（DeepSpeed 真实哨兵值，**不是 bug，保留**）；仅对词表外的未知字符串收紧为 raise，不再默认 True |
| `97dc3f6` | #10 | `assert` 改为显式 `if...raise`，避免 `python -O` 下失效 |
| `4a3ae3b` | #11 | `_process_name` 只取入口脚本 basename，不再拼接可能含路径/密钥的后续 argv |

**#12 复核结论（未改）**：`_validate_collector_environment_contract` 只在校验的 `FAKEGPU_SERVING_*` 环境变量已设置时才比对，起初怀疑"未设置应报错"。但 `build_cuda_serving_sample` 是文档化的公开 API（README "The same adapter is available as `fakegpu.build_cuda_serving_sample(...)` for use inside a Python benchmark"），供 vLLM/自定义 runner **在 FakeGPU 自身采集子进程之外直接调用**；`tests/test_analysis_extensions.py:757` 就是在这些环境变量全部未设置的情况下调用并断言成功。若改成"未设置即报错"会直接打破这个文档化用例和对应测试。当前"仅在存在时比对"是该函数**双重用途设计下的正确行为**，不是缺陷，保留不动。

验证：本轮改动后本地全量 `pytest`（222 passed）+ ruff clean + CI 同款 manifest 校验（7+39 cases）通过；`_process_name`/`native_capability_report` 过滤逻辑另有手工冒烟验证。**本轮未做远端 406/gem12 复测**——多数改动是纯逻辑修正（正则、计数器时序、assert→raise、字符串校验）或范围明确的行为收紧，本地+CI 已提供等价确认；如需更保险可再补一次远端跑。

**2026-08-20 第三批落地**：§8 第 1 步（待决策项）全部拍板，第 3 步（公共设施）收尾，第 6 步（兜底收敛与性能）剩余项完成。

| Commit | 阶段 | 内容 |
|---|---|---|
| `f35018d` | §1.6 / 待决策 | 删除 `privateuse1/`（866 行）与 `init_privateuse1` 导出；其余四项拍板为**保留**并补上缺失覆盖 |
| `9a4d327` | §2.3 + §5 | 静默兜底收窄（SMI publisher/catalog、workspace_profiles、repository_analyzer、calibration sample 定位改为 warning 或 raise）；allocator 分类记账与自由块索引、metrics 深拷贝、trace_replay 按 rank 建索引、topology 链路计数、preflight 复用子进程输出与预编译正则、repository_analyzer 单遍解析、catalog 缓存键归一化 |
| `672695c` | §5.4 | `decode_steps` 闭式聚合 + `--exclude-decode-steps` / `include_decode_steps=False`（默认行为与报告 shape 不变） |
| `c660cdb` | §3.6 | `_cli.py` 承载命令注册表、共享 flag 工厂与 `--json`/`--strict`/退出码约定；21 个命令模块的 `prog=` 改为从注册表派生，4 种 `--json` 拼法收敛为 2 种 |

**待决策项的拍板结论**（原文档列的 5 项，逐项复核仓库证据后）：

- `privateuse1/`：**删除**。它经私有 `torch._C._acc` API 注册设备模块，而支持窗口内（torch 2.6–2.13）没有任何版本提供该 API——`init_privateuse1()` 在任何已安装环境里都跑不起来，仓库内也无其他调用路径；原测试只断言导出可调用，从未执行。可从 `c31fee1` 恢复。
- `MatmulFlopCounterMode`：**保留**。实测可用（`aten::mm` 计数正确），是文档化公开导出，缺的只是覆盖——已补执行级测试。
- `capabilities.audit_native_exports`：**保留**。`fakegpu capabilities --build-dir` 是真实可达路径。
- `schemas/*.schema.json`：**保留**。复核时三个 schema 均已有契约或测试用途，删除会破坏现有功能（与 2026-08-16 扫描时的"完全孤立"结论不同）。
- legacy SMI schema：**保留**。v1 兼容读取仍有价值，`test_virtual_smi.py` 是其现行消费者。

**§3.6 中有意保留的差异**：`--json` 的两种形态（可选路径 vs 布尔开关）不合并——布尔那组命令的全部输出就是 JSON 文档本身，改成可选路径会让紧跟的位置参数被 argparse 吞掉；`--strict` 的退出码各命令不同（1/2/3），都写在各自 `--help` 里，统一会破坏已声明的行为，只把声明方式收敛到一处。`capabilities --strict` 的 2 与"用法错误"的 2 重合，是既有取值，未动。

验证：本地全量 `pytest`（233 passed，含 3 个新增测试）+ ruff clean；CI 同款 manifest 校验（7+39 cases）通过。**本轮尚未跑 `scripts/test.sh smoke`/`cpu` 与远端 406/gem12 复测**——改动集中在 Python 侧 CLI 与纯逻辑路径，未触碰 `src/`；如需最终确认可按 §3 建议补一次。

**2026-08-20 第四批落地**：§8 第 4 步（大合并）与第 5 步（机械拆分）完成主体。

| Commit | 阶段 | 内容 |
|---|---|---|
| `5d71400` | §3.2 上半 | 两条 serving 主路径共享容量解析、vLLM/通用准入分支、limiting_factor 阶梯与 inputs 回显（`_ServingCapacity`/`_ServingAdmission` 等 4 个 helper） |
| `242aa02` | §3.2 下半 | `_serving_batch_memory` 与 `_serving_request_set_memory` 共享驻留量与相位峰值计算（`_serving_memory_breakdown`） |
| `c6a1866` | §3.1 | 删除 torch_patch 对 `_upstream` process-group 补丁的逐字复刻；先补了该生命周期此前**完全缺失**的测试 |
| `f2f0039` | §4.2 | `torch_patch` → `_allocator.py`（990 行） |
| `9f632c4` | §4.2 | `smi` → `_smi_environment.py` + `_smi_state.py`；`metrics` 改为从 `_smi_state` 取 inventory，不再 import smi 私有函数 |
| `cd57d21` | §4.2 | `calibration` → `_calibration_protocol.py` + `_calibration_cuda.py`；顺带把懒加载的 `llm_estimator` import 提到模块级（复核确认根本不存在循环 import） |
| `eaef076` | §4.2 | `serving_plan` → `_serving_types/_serving_kv/_serving_vllm/_serving_speculative`，并补 `__all__` 明确公开面 |
| `f8510a4` | §4.2 | `diffusion_estimator` → `_diffusion_types.py` + `_diffusion_pipeline.py` + `_diffusion_activation.py` |
| `2ebb7bc` | §4.2 | `smi` → `_smi_query.py` + `_smi_render.py` |
| `f52e6bd` | §4.2 | `torch_patch` → `_profiles.py` + `_cross_device.py` |

四个超大文件的规模变化：`smi.py` 4353→957、`torch_patch.py` 4346→2073、`serving_plan.py` 3722→2501、`calibration.py` 2873→1871、`diffusion_estimator.py` 2327→779。全包最大文件从 4353 行降到 2501 行。

**§3.2 合并的实际结果与文档预期不同**：文档估计"约 −400 行净减"，实际两个 commit 合计**净增 17 行**（planner 主体 −218、memory 函数 −24，共享 helper +300 左右）。原因是真正逐字相同的片段与模式特有片段交错，抽出来的每个 helper 都要写一份显式关键字签名，签名开销吃掉了去重收益。**去重本身是达成的**（准入逻辑、limiting_factor、inputs 回显、相位峰值各只剩一份），但不要指望它减行。两个入口没有进一步合并成单个函数体——那会把重复换成一个贯穿全函数的 mode 标志，不是改善。

**等价性验证**：§3.2 两个 commit 各跑了一轮 2,734 例全报告比对（两条路径 × generic/vLLM × 容量扫描 × 三种 KV 策略 × utilization × speculative × profile + 10 个非法输入），与合并前实现**逐字节一致**，覆盖全部 6 种 limiting_factor 分支。

**拆分中发现与文档不符处**：
- `_PROFILE_SUPPORTED_TYPES`（§1.7 列为"从未读取"）实际被 `tests/test_cli_commands.py:196` 读取，**不是死代码**，已随 `_profiles.py` 保留。
- `calibration` 里 `llm_estimator` 的函数级懒加载（§4.2 说是"为躲循环 import"）实际上没有循环可躲——`llm_estimator` 不 import 包内任何模块，纯粹是位置放错。
- `torch_patch` 的 `_NUM_DEVICES`/`_DEVICE_PROFILES`/`_DEVICE_NAME`/`_TOTAL_MEMORY` 是 `_refresh_runtime_profile_state` 会重新绑定的可变模块状态，**不能**随 profile 解析函数一起搬走（搬走后 from-import 拿到的是旧值副本，刷新不可见），已刻意留在 `torch_patch`。
- 移动私有函数会让 monkeypatch 目标失效：`tests/test_analysis_extensions.py` 对 `_transformers_serving_workload` 的 patch 已改指新模块（继续 patch 旧名字不会报错，只会静默失去拦截）。

验证：本地全量 `pytest`（234 passed）+ ruff clean + CI 同款 manifest 校验（7+39 cases）+ `scripts/test.sh smoke`（含 native 重建）+ `scripts/test.sh cpu` 全部通过。**远端 406/gem12 尚未复测**。

**2026-08-22 第五批落地**：继续完成 §4.2 的 `torch_patch` 机械拆分，保持旧私有导入路径可用。

| Commit | 阶段 | 内容 |
|---|---|---|
| `b1916fe` | §4.2 | 将 Accelerate、distributed、FSDP 兼容补丁移到 `_ecosystem_compat.py`；将架构名、字节格式化和终端内存报告移到 `_torch_reporting.py`。`torch_patch.py` 从 2,087 行降至 1,753 行；原模块保留 `_patch_*`、`_arch_name`、`_fmt_bytes`、`_dump_terminal_summary` 兼容入口，参数、异常和报告文本不变。 |

验证：本地全量 `pytest`（262 passed）+ `ruff` clean + `scripts/test.sh smoke` + `scripts/test.sh cpu` 全部通过；抽出的 6 个兼容函数与拆分前 AST 源码逐字一致。推送 `00a6c4d` 后按 nvidb 队列复测：406（job 1486，Blackwell GPU，torch 2.8.0+cu128）真实 CUDA/FakeGPU 集成探针通过；gem12（job 1485，Ampere GPU，torch 2.12.1+cu130）全量 `pytest` 为 **261 passed + 1 skipped**，随后 patch probe 通过。406 环境没有 pytest，因此采用同一 CUDA/torch 环境的集成探针，未改动远端环境。

**§4/§5 剩余未做**：`preflight.py`（1362 行，四个关注点未分离）；`calibration` 的比较/验证/bundle 与 CLI 未再分；`serving_plan` 的两个 pool model（约 1,100 行）与 CLI 未再分；`smi`/各 CLI 的 `main` 仍是大函数（§4.1 的超大函数清单只处理了其中一部分）。§3.6 遗留：`llm_cli` 与 `serving_plan` 约 12 个同名 flag 默认值不同，本轮只统一了声明方式。另注意 `diffusion_estimator` 的 `_local_profile` 产物绕过 `_validate_profile` 的漂移风险（§4.2 提到）现在两者同在 `_diffusion_pipeline.py`，修它会改变行为，属独立事项。

**复核后有意不改的条目**（与文档原建议不同处）：
- `_aggregate_report` 的 error 分支（§1.5/2.3）：rc=0 但报告为 error 的边缘情况下可达，保留。
- `_stage.py` 重复 env 赋值（§2.2）：嵌套 stage 退出时有恢复语义，保留。
- `_format_bytes` 11 份（§3.4）：实测 6 份实现 5 种输出格式（负数/None/整除格式化各不同），全量合并会改 CLI 输出，仅 demo/doctor 两份本来相同，不值得为此建共享模块。**放弃**。
- `_percentile` 2 份（§3.4）：calibration 是线性插值、bandwidth_worker 是就近排名，**不同算法**，不可合并。
- `_ceil_div` 3 份（§3.4）：单行惯用式，零漂移风险，合并收益不抵跨模块依赖，保留。
- `--include-exited`（smi/metrics）、`distributed_cli` 的 `--markdown-report` 等：有真实工作路径，属缺测试而非死代码，保留。
- smi 发布的 state JSON 不再含恒 null 的 `telemetry.*`、`topology.numa_node/pcie_generation` 键：死数据移除，`typical_power_usage_mw` 等真实值字段保留。

**待决策项**：已于 2026-08-20 全部拍板，结论见上方第三批落地一节（删 1 留 4）。

---

## 0. 规模总览

| 部分 | 规模 |
|---|---|
| `fakegpu/`（主包） | 约 37,100 行 Python（含 `privateuse1/` 866 行） |
| `src/`（C/C++ stubs） | 约 34,600 行 |
| `tests/` | 约 9,200 行 |
| 最大的文件 | `smi.py` 4353 行、`torch_patch.py` 4346 行、`serving_plan.py` 3722 行、`calibration.py` 2873 行、`diffusion_estimator.py` 2327 行 |

ruff（F401/F811/F841）检查是干净的——没有未使用的 import / 变量这类低级冗余。真正的体积问题集中在四类：**成块的不可达/无消费者代码**、**大面积复制粘贴**、**多层重复校验与兜底**、**超大文件/函数**。粗略估计，前两类合计可以删掉或合并 **4,000–5,000 行** 而不改变任何可观察行为。

---

## 1. 角度一：删除死代码与不可达路径（收益最大，风险最低）

### 1.1 `torch_patch.py` 的 standalone fallback 路径（约 900 行，不可达）

`_activate_upstream`（`torch_patch.py:3333-3384`）只有在 `import torch.fakegpu` **和** `from . import _upstream` 同时失败时才返回 `None`。而 `_upstream.py` 是随包内置（vendored）的模块，正常安装下必然可导入，因此：

- `patch()` 的 standalone 分支（`torch_patch.py:3857-4341`）在正常安装下永远走不到；
- 只被该分支引用的约 43 个 `_stub_*` 函数（`:2354-2691`）、standalone 版 `_FakeStream`/`_FakeEvent`/`_FakeStreamCtx`/`_FakeDeviceProperties`（`:2207-2346, 2699-2736`）、`_patched_tensor_to`/`_patched_module_cuda` 等（`:2056-2199`）随之全部变成死代码；
- 例外：`_stub_is_bf16_supported`、`_stub_get/set_rng_state*`、`_normalize_device_index` 也被 upstream 路径使用，需保留。

同类问题：`_runtime.py:108-136` 的 `_detect_custom_torch_fakegpu_available()` 末尾 `find_spec("fakegpu._upstream") is not None` 恒为 True，导致 `init(runtime="auto")` 永远选不到 `"native"`，前面的 `sys.path` 扫描与 `torch.fakegpu` 探测（`:109-127`）全部不可达。`tests/test_runtime_init.py:71-82` 只在 mock 掉该函数的情况下测试，所以测试抓不到。

**处置建议**：要么整体删除 standalone 分支并同步修改模块 docstring（`torch_patch.py:16-19`），要么用显式环境变量把它变成可达、可测的路径——现在它既不可达也无覆盖，是最差的状态。

### 1.2 `fsdp_memory.py` 整个模块无消费者（970 行）

4 个公开函数（`build_full_shard_plan:7`、`build_fully_shard_plan:74`、`estimate_full_shard_sft_memory:461`、`estimate_fully_shard_sft_memory:598`）在 `fakegpu/`、`tests/`、`scripts/`、README、`__init__.py` 中**零引用**。docstring（`:467`）声明的唯一消费者 `qwen_sft_memory_worker.py` 已不在仓库里。git 历史显示它曾被积极维护（CHANGELOG 记录过 real-GPU 验证），属于"消费者被删掉后遗留"而非从未用过。

**处置建议**：先确认是否有仓库外的消费者；没有则删除，有则补 `__init__.py` 导出 + 测试。二者取一，不要维持现状。

### 1.3 永远为 `None` 的遥测子系统（贯穿 5 层）

`gpu_utilization_percent` / `temperature_c` / `fan_speed_percent` / `power_usage_mw` 从未被写入非 None 值（Python 侧 `smi.py:1391-1397`，C++ 侧 `src/core/native_smi.cpp:874-879` 均硬编码 `None`），却被 `smi.py:3308-3320` setdefault、`:153-173` 暴露为查询字段、`:2881-2888` 渲染，`_format_percent`/`_format_temperature`（`:4264-4279`）几乎专为它们存在。同类："永远是 None"的 `topology.numa_node`/`pcie_generation`（`smi.py:578-579, 3781-3782` 及查询/渲染点）。

**处置建议**：如果近期没有实现真实遥测的计划，整链删除；否则至少收敛到一处 `TELEMETRY_FIELDS` 表，避免 5 处手写 None。

### 1.4 写了没人读的状态字段（每 250ms 白写一次）

- `reserved_stage_peaks`：`smi.py:3087` 初始化、`:3162-3165` 累加，无任何读取方。
- `memory_utilization_percent`（`smi.py:1353-1357`）：读取方自己重算（`:3367-3369`）；同类还有 `free_memory`、`headroom_bytes/percent`、`identity_source`（恒为 `"synthetic"`，`:1331`）。
- `trace_replay.py:452-453` 的 `event["memory_bytes"]` 写后无读；`:488-489` 的 `excluded_from_aggregate`/`exclusion_reason` 打上标记后立刻在 `:490-491` 被过滤丢弃，整个复制遍历可以简化为一个 filter。

### 1.5 无消费者的 CLI 参数与环境变量

- `preflight --steps`（`preflight.py:69`）导出的 `FAKEGPU_PREFLIGHT_STEPS` 与 `FAKEGPU_PREFLIGHT_TARGET_STAGE`（`:639-643`）全仓库无读取方，是完整的 no-op。
- `FAKEGPU_PREFLIGHT_MEMORY_SAFETY_FACTOR`/`_MARGIN`（`preflight.py:839, 853`）只有读取处，无任何写入/文档/测试。
- `distributed_cli.py` 的 `--markdown-report`、`--cluster-markdown-report`、`--interconnect-*`、`--ranks-per-node` 无文档无测试，且部分组合下被静默忽略（`:706-718`）；`_aggregate_report` 的 error 分支（`:499`）因上游先 raise 而不可达，`_print_bandwidth_summary` 恒打印 PASS（`:555`）。
- `smi.py --include-exited`、`metrics.py --include-exited` 无任何测试或脚本使用。
- `performance_model.py` 的 `efficiency` 参数（`:98`）无任何调用方传入，其约 36 行 override 校验（`:298-333`）全部是死代码。

### 1.6 只导出不使用、或只在模块内自用的公开 API

- `MatmulFlopCounterMode`（`flop_counter.py`，88 行整个模块）：`__init__.py` 导出，但无内部调用、无测试。
- `capabilities.audit_native_exports`（`capabilities.py:222-328` + 6 个 helper，约 150 行）：仅 `--build-dir` 一条路径可达，无测试无 CI 调用。
- `privateuse1/` 子包（866 行）：仅经 `init_privateuse1` 暴露，测试只断言 callable（`tests/test_runtime_init.py:54`），从未真正执行——属于实验性平行后端，需要决定去留。
- `validation.expand_validation_cases`/`render_validation_markdown`、`workspace_profiles.default_workspace_profile_path`、`profile_catalog.CatalogValidation`/`profile_directory` 等公开命名但仅模块内使用；`calibration.py`/`repository_analyzer.py` 的 `__all__` 与 `__init__.py` 实际导出不一致——建议统一"什么算公开 API"的口径，缩小公开面。

### 1.7 局部死分支（小而明确）

- `diffusion_estimator.py:1416-1429`：`StableDiffusion` 分支与 fallback 返回完全相同的 dict。
- `serving_plan.py:3150-3164`、`:1796-1806`、`llm_estimator.py:862-864`：结构上不可能触发的防御性 raise。
- `kernel_analysis.py:150-152`：左侧 opcode 集合被右侧正则完全覆盖；`:287` 的 `limiting = []` 立即被重新赋值。
- `torch_patch.py:78-81` `_PROFILE_SUPPORTED_TYPES` 导入时构建、从未读取；`:2690` `_stub_cudart` 从未绑定；`_orig_tensor_cuda`/`_orig_tensor_pin_memory`/`_orig_torch_compile`（`:2051-2053` 等）被保存"以便恢复"但不存在任何 unpatch 路径。
- `_preflight_bootstrap.py:127-158`：32 行 fallback 伸手进 `torch_patch` 私有属性，重新推导公开函数 `memory_snapshot()` 已经返回的内容。
- `schemas/*.schema.json` 三个 JSON Schema 完全孤立——无代码、无 MANIFEST.in、无 CI 引用，而 `validation.py:278-318` 手写了一份更弱且已漂移的校验。要么让代码真正使用 schema，要么删掉。

---

## 2. 角度二：收敛兜底/防御逻辑（对可读性影响最大）

全包 61 处宽 `except`，其中 `torch_patch.py` 独占 60 个 `try`、39 个 `except Exception`、27 个 `hasattr`。原则建议：**版本兼容性 guard 保留并注释清楚；"不可能失败"的 guard 删除；会吞掉真实故障的宽捕获收窄并留诊断。**

### 2.1 应当保留的（真实版本/平台差异）

`torch.distributed.fsdp` 私有模块 import guard（`torch_patch.py:3205-3258`）、`torch._C._cuda_*` 的 CUDA/CPU 构建差异探测（`:2812, 4104-4128`）、`allow_tf32` 的 property 分叉（`:2840-2853`）、`hasattr(os, "RTLD_NOW")`（`_api.py:168`）等。这类保留，但建议统一加一行"为哪个 torch 版本/构建差异服务"的注释，防止未来被误删或被继续模仿。

### 2.2 应当删除的（守护不可能发生的条件）

- 完全相同的重复 handler：`torch_patch.py:1372-1375` 与 `:1603-1608`，窄捕获后紧跟同体宽捕获。
- 不可能抛异常的操作外套 try：`getattr(x, k, default)`（`:1684-1690`）、`os.path.dirname`（`:1461-1469`）、`torch.nn.Module` 属性访问（`:1808-1812, 1909-1912`）、已在 `patch()` 里无条件 import 过的子模块再次 try-import（`:3642-3662, 3743-3751, 1536-1539`）。
- 在声明支持窗口（torch 2.6–2.11）内恒为 True 的 `hasattr`：`torch.compile`（`:2130`）、`torch.amp.autocast`（`:3299, 3329`）、对自带 vendored 模块 `_upstream` 的 4 次属性探测（`:3358-3374, 3808-3810`）。
- `_stage.py:26, 30` 两次无条件重复赋同一个环境变量。
- `setup.py:12-15, 114-130` 对 `wheel` 的 try/except——`pyproject.toml` 已在 build-system.requires 里声明它。
- 各 CLI `except` 元组里"子类 ⊂ 基类"的冗余项：`FileNotFoundError ⊂ OSError`、`json.JSONDecodeError ⊂ ValueError`、自定义 `*Error ⊂ ValueError`（`serving_plan.py:2202-2208`、`training_plan.py:431-437`、`topology.py:483-489`、`capabilities.py:365-370`、`smi.py:1862` 等 6+ 处）。副作用：所有命令实际都在裸捕 `ValueError`，内部 bug 会被当作用户输入错误报出。

### 2.3 应当收窄的（宽捕获掩盖真实故障）

- `torch_patch.py:3350-3356`：vendored 模块 import 失败被静默降级到（不可达的）standalone 路径——打包坏了却无任何告警；`:3344-3347` 应只捕 `ImportError`，否则损坏的 custom-torch 构建与"未安装"不可区分。
- `torch_patch.py:1759-1776, 1794-1804`：saved-tensor 追踪安装/恢复失败被吞，激活内存追踪静默关闭且 hooks 永久丢失。
- `workspace_profiles.py:371-379`：`import torch` 失败退化成空字符串**作为 profile 匹配键**参与 fnmatch——一次 import 失败静默改变匹配结果；`:363-368` 把 `get_profile` 的一切失败折叠成"unknown target GPU profile"。
- `repository_analyzer.py:558-564, 600-605`：损坏的 `pyproject.toml` 静默视为"无依赖/无入口"，最终却报告 `static_analysis_complete: True`；`:326-358` git 失败静默退化为 `os.walk`（ignore 策略不同，文件清单会变）。
- `smi.py:1495-1500`：publisher 主循环裸捕后 `continue`，状态文件持续超限会永远失败而无信号；`:1226-1229` `stop()` 吞掉最终写入失败，进程可能永远显示 running。
- `smi.py:4019-4053`：坏掉的 profile catalog（`ProfileCatalogError` 是 `ValueError`）被 `_catalog_metadata` 折叠为 `{}`，所有输出静默降级为 N/A。
- `calibration.py:2548-2561`：定位 sample 的第四层 fallback 会捡起任意一行日志 JSON，随后在下游报出令人困惑的 schema 错误，而不是"未找到 sample"。
- `demo.py:101-155`：54 行主体套一个 `except Exception` 折叠为 exit 2，torch import 失败与算术错误不可区分。
- `metrics.py:1207-1218` `_number` 把负数/NaN 一律钳到 0（时钟回拨产生的负 age 变成 0，无告警）。

### 2.4 重复出现的兜底"模式"应工厂化

- "patch 某属性、防重复打补丁"六连块（`torch_patch.py:1813-1941, 3260-3294`）→ 一个 `once_patched(target, name)` 装饰器，约省 90 行。
- "invalid device ordinal" raise 有 11 份拷贝、3 种消息格式（`:1991-4028` 各处）→ 一个 `_require_valid_device_index()`，顺带统一报错文案。
- "`_memory_tracker` 非 None 则委托否则默认值"10 连块（`:2443-2506`）→ 表驱动。
- 多层 Optional 传播链（`calibration.py` `_first_integer→_memory_points→_canonical_point→_match_points` 四层 None/空集合接力才 raise 一个错误；`workspace_profiles` 同型）→ 在边界一次性 raise，中间层不再传 Optional。

### 2.5 同一输入被 3–4 层重复校验

一次 `estimate_serving_plan` 调用中，`prompt_tokens` 等参数在 `serving_plan.py:570-573`、`:62-65`、`llm_estimator.py:66-82` 各验一遍；`kv_cache_strategy` 验 4 遍（argparse choices + 三层函数），且每层抛不同的异常类型，还都在二分搜索内部重复执行。`training_plan.py:48-373` 对 `normalize_training_config` 已经归一化过的输出再套一层 `x or default` 链（`:84-358` 多处）。`_normalize_serving_requests`（108 行）对同一份数据在 `serving_plan.py:1027, 2648, 2670, 2693` 反复全量归一化。`build_cuda_serving_sample` 甚至校验自己刚构造的输出（`calibration.py:399-414`）。写侧/读侧独立重算同一派生值的还有 `smi.py` 的 MIG mode、health status、NVLink 汇总（写 `:1091-1097` 等 vs 读 `:3631-3715` 等）。

**方向**：确立"入口边界校验一次，内部信任"的约定；内部函数用注释或类型标注声明前置条件，删除重复层。

---

## 3. 角度三：消除重复（第二大删减来源）

### 3.1 `torch_patch.py` 对 `_upstream.py` 的成块复刻

- `_patched_copy_collective_tensor`（`torch_patch.py:3041-3063`）与 `_upstream.py:483-499` 语义逐字相同——整个补丁是 no-op，直接删。
- `_patched_dist_init/destroy_process_group`（`:3135-3181`）复刻 `_upstream.py:381-438`，仅状态读取方式不同。
- 12 个 `_tracked_*` 内存闭包在 upstream 路径（`:3581-3662`）与 standalone 路径（`:3909-3981`）写了两遍（约 150 行）；跨设备 wrapping 块两遍（`:3682-3736` vs `:4191-4249`，其中 `_BINARY_DUNDERS` 列表实际有三份）；`_FakeGradScaler` 两遍（`:3669-3677` vs `:4293-4301`）。若按 1.1 删除 standalone 路径，这些自动消失。
- `torch_patch` 影子实现了 `_upstream` 已有的 `_FakeStream`/`_FakeEvent`/`_FakeDeviceProperties`/`_normalize_device_index`/`_install_legacy_cuda_types`/torch_load 处理等 8 组类与函数——统一改为复用。

### 3.2 `serving_plan.py`：两条主路径 82% 逐行相同

- `estimate_serving_plan`（`:537-993`，457 行）与 `estimate_serving_request_set`（`:996-1494`，499 行）经机械 diff 有 **375 行逐字相同**（capacity 解析、vLLM 分支、limiting_factor 九连 if、报告尾部全部一致）。
- `_serving_batch_memory`（`:2276-2626`）与 `_serving_request_set_memory`（`:2629-3083`）64% 相同。
- 三份 token 级相同的二分搜索（`:3086-3140`）→ 一个 `_largest_fitting(limit, predicate)`。

**方向**：抽出共享的 `_build_serving_report(...)`，两个入口收敛为薄封装。这是全仓库杠杆最大的单项重构（涉及约 1760 行）。

### 3.3 `_api.py`/`_runtime.py`：10 参数配置签名写了 8 遍

`init`/`env`/`run`/`_apply_env*` 加 `_runtime` 两处，共 8 处完整签名 + 6 处逐参转发调用，约 200 行纯转发代码。引入一个 frozen 的 `FakeGpuConfig` dataclass 作为单参数即可全部收敛；`_apply_config_env_inplace` 是 `_apply_env_inplace` 的严格子集，合并为带 flag 的一个函数。

### 3.4 全包级公共 helper 缺失（建议建立 `_common`/扩展 `structured_io`）

| 重复项 | 份数 | 位置举例 |
|---|---|---|
| `_format_bytes` 字节格式化 | **11** | `doctor.py:21` 与 `demo.py:13` 逐字节相同；`serving_plan.py:3717`、`llm_cli.py:141`、`smi.py:4244`、`torch_patch.py:1203`、`preflight.py:1346`、`distributed_cli.py:543`、`diffusion_estimator.py:2280`、2 个 scripts |
| JSON/TOML/YAML 结构化读取 | 3 | `structured_io.load_mapping`、`validation.py:617`、`workspace_profiles.py:188`（各带一份"install fakegpu[validation]"提示） |
| `--json` 输出块（含双重序列化 bug，见 §7） | 8+ | `calibration.py` 一个函数内 4 份、`trace_replay.py:170`、`kernel_analysis.py:344`、`repository_analyzer.py:287` 等 |
| `write_json` 内联复刻 | 7 | `llm_cli.py:105`、`distributed_cli.py:860`、`preflight.py:160`、`serving_plan.py:2210`、`performance_model.py:254` 等（`structured_io.write_json` 已存在） |
| dtype→bytes 表 | 2（逐字节相同） | `llm_estimator.py:16-32` == `diffusion_estimator.py:31-47`（后者已 import 前者的其他函数） |
| `_positive_integer`/`_nonnegative_integer` 校验 | 3+ | `serving_plan.py:3695`、`diffusion_estimator.py:2164`、`training_plan.py:694`（仅异常类不同）；`calibration.py` 内部同型判断约 15 处 |
| `_ceil_div` | 3 | `training_plan.py:733`、`diffusion_estimator.py:2234`、`topology.py:924` |
| `_percentile` | 2 | `calibration.py:2022`、`_bandwidth_worker.py:32` |
| `git rev-parse` 子进程封装 | 3 | `preflight.py:1329`、`validation.py:700`、`repository_analyzer.py:1000` |
| `_iter_tensor_leaves`/`_tensor_bytes` | 2（逐字节相同） | `memory_estimator.py:1771` == `workspace_profiles.py:633` |
| `_positive_float`/非负 int 钳制 | 3 | `smi.py:4332`、`metrics.py:1261`、`topology.py:894` |
| 字节数量*解析*（单位表还不一致） | 2 | `distributed_cli.py:34-49` vs `preflight.py:859-886` |
| 区间合并算法（同文件内两份） | 2 | `trace_replay.py:736-742` == `:768-774` |
| severity 等级表 | 3 | `smi.py:44`、`metrics.py:1239`（每次调用重建 dict）、`native_smi.cpp:502` |
| 手写 YAML 迷你解析器 | 3 | `profile_catalog.py:423`（并强迫整数字段必须是 str）、`structured_io`（真 YAML）、`tests/native/...:88`；另有 `scripts/lib/fakegpu_uv_deps.py:18-99` 的 82 行手写 TOML 解析器和 `scripts/validation/check_preflight_report.py:84-135` 的手写迷你 JSON-Schema 校验器（CI 已装 jsonschema） |

### 3.5 Python 与 C++ 的双实现（改动必须双写）

`smi.py:433-1116` 的三个 `_modeled_*` 环境解析器（约 680 行）与 `src/core/device.cpp:271-702` 逐条重复，错误字符串逐字节一致；JSON 发布再与 `src/core/native_smi.cpp:374-880` 重复。短期至少要加一个**共享 fixture 断言两侧输出一致**；长期考虑单侧生成或以数据文件驱动。`smi.py` 内部这三个解析器本身也是同构骨架，"可选 `+` 前缀整数解析"复制了 6 遍（`:483-984` 各处），可先在 Python 侧内聚。

### 3.6 CLI 层样板重复

21 个独立 `main(argv)` 各建 `ArgumentParser`，`prog="fakegpu <name>"` 字符串须与 `__main__.py:12-33` 的注册表手工同步；`--build-dir`/`--profile`/`--devices` 在 5+ 处重复声明；`--json` 有 4 种互不兼容的拼法（store_true / dest 字符串 / type=Path）；`llm_cli.py` 与 `serving_plan.py` 重复声明约 12 个同名 flag 且**默认值不同**（`dynamic/eager` vs `paged/sdpa`）；退出码约定 5 种。`preflight`/`doctor`/`demo` 三者独立校验 profile catalog，`preflight.py:716-745` 甚至用两个独立循环解析同一个 `--devices` 字符串。**方向**：一个小型 `register(name, build_parser, handler)` 注册器 + 共享 flag 工厂，统一 `--json`/退出码约定。`CMakeLists.txt:198-282` 的 APPLE/else 双分支中 5 个相同的 `set_target_properties` 块（约 85 行→30 行）同理。

---

## 4. 角度四：拆分超大文件与函数

### 4.1 超大函数一览（重构时优先处理）

| 行数 | 位置 | 说明 |
|---:|---|---|
| 666 | `torch_patch.py:466` `_DeviceMemoryTracker` | 混杂字节记账、allocator segment 模型、stage 记账、报告格式化 |
| 573 | `torch_patch.py:3769` `patch()` | 含 485 行不可达 standalone 安装 |
| 499/457 | `serving_plan.py:996 / :537` | 两条 82% 相同的主路径 |
| 498 | `diffusion_estimator.py:311` | 内含 156 行字面量返回 dict |
| 474 | `memory_estimator.py:107` `estimate_module_memory` | |
| 421 | `metrics.py:123` `render_prometheus` | 9 个 accumulator 机械重复 |
| 412 | `calibration.py:1104` `main` | argparse 构建 + 六路 dispatch |
| 376 | `serving_plan.py:1898` `main` | 同型 |
| 375 | `torch_patch.py:3387` `_apply_enhancements_over_upstream` | 已有 0–8 节编号，逐节提取即可 |
| 365 | `smi.py:2623` `render_detail` | runtime 段与 device 段互相独立 |
| 322 | `smi.py:1503` `main` | |
| 269 | `calibration.py:748` `verify_calibration_reports` | 8 个 gate 手写 if，天然表驱动 |

各 CLI 的 `main` 都可按同一模板拆为 `_build_parser()` + 每个子命令一个 `_run_<action>(args)`，顺带消掉 dispatch 长梯子和重复 JSON 输出块。

### 4.2 模块拆分方案（均为机械移动，不改行为）

- **`torch_patch.py`（4346 → ~6 个模块）**：`_profiles.py`（profile 解析，无 torch 依赖，`:63-226`）；`_cross_device.py`（`:228-368`）；`_allocator.py`（`_DeviceMemoryTracker` 及其自由函数，约 810 行，`:371-1184`，是最干净的一刀）；`_reporting.py`（`:1187-1332`，注意 `_dump_terminal_summary` 直接摸 tracker 私有字段，先给 tracker 一个公开 `report_rows()`）；`_ecosystem_compat.py`（HF/accelerate/FSDP/NCCL 第三方 shim，`:2777-3330`）；standalone 部分按 §1.1 处置。
- **`smi.py`（4353 → ~6 个模块）**：查询 schema（`:53-403` + `:3845-3961`）；modeled 环境解析（`:406-1146`，纯函数）；publisher（`:1149-1500`，`torch_patch` 唯一依赖的部分）；inventory/归一化（`:1841-2046` + `:3042-3833`，`metrics.py` 真正需要的部分——现在 `metrics.py:18-23` 在 import `smi` 的私有函数，是缺共享模块的最明确信号）；renderers（约 900 行）；CLI。
- **`calibration.py`（2873 → 4 个模块）**：比较/验证/bundle；serving observation 协议；transformers real-CUDA 适配器（唯一 import torch/transformers 的部分，现在为躲循环 import 在 `:2131` 懒加载 `llm_estimator`——强烈暗示它本就该在别处）；CLI。
- **`serving_plan.py`（3722 → ~5 个模块）**：KV-pool 数学、vLLM 预算、speculative decoding、request manifest、CLI（详见 §3.2 合并后再拆）。
- **`diffusion_estimator.py`（2327 → 4 个模块）**：pipeline inspection、profile 加载/校验（注意 `_local_profile` 产物绕过 `_validate_profile`，两套 schema 编码会漂移）、activation 模型、CLI。
- **`preflight.py`（1355）**：报告组装 / 状态分类 / 经验校准（约 260 行）/ Markdown 渲染四个关注点分离；`render_markdown_report`（`:305-503`）约 200 行。

---

## 5. 角度五：效率

### 5.1 热路径（每次张量操作/每次分配都执行）

- `torch_patch.py` 内 20 处函数内 `import torch`，其中 `_wrap_tensor_binary_op.wrapper:343` 在**每次** `+`/`*`/`@` 上执行，`_DeviceMemoryTracker.allocate:527` 在每次分配上执行（只为拿 `torch.cuda.OutOfMemoryError`）。模块级绑定一次即可。同类：`memory_estimator.py:1771` / `workspace_profiles.py:633` 的 `_iter_tensor_leaves` 在**每层递归**都 try-import torch。
- `_allocate_allocator_block`（`torch_patch.py:929-946`）每次分配对全部 segment×block 做嵌套扫描（持锁），随分配数近似二次增长；`_free_allocator_block:1011` 线性扫段（已有 `segment_id` 却不建索引）。
- `snapshot()`（`:790-855`）O(设备×存活分配)，被后台线程每 250ms 持锁驱动，直接与分配热路径抢锁——应改为在 allocate/release 时维护增量汇总。
- 每个被追踪张量 5 次 `os.environ` 读取（`:565, 1414, 1424, 1428, 1507`），其中仅 stage 一个真正会变。

### 5.2 每个 FX 节点重新加载全目录

`memory_estimator.py:1034-1044` 在节点循环内调 `match_workspace_profile`，后者每次 `load_workspace_profiles`：每个 catalog 文件一次 `stat()`、每个 profile 一次 `dict()` 拷贝、外加 `_software_stack` 的 try-import torch（`workspace_profiles.py:51-74, 91-92`）。数千节点的图 = 数千次 syscall 与目录重建。把 catalog + software stack 提升到循环外是这批文件里**单项收益最大**的优化。

### 5.3 二分搜索内的重复计算（serving_plan）

- 每个 request 的 transient 只依赖自身（`batch_size=1`），却在每次 prefix 二分探测中全量重算（`:2718-2804` × `:3105`），6N·logN → 6N。
- request-set 路径有 `memory_by_request_count` 缓存（`:1120-1147`），homogeneous 路径的 `batch_memory`（`:665-687`）却没有，且每次探测都带着 §2.5 的多层校验重跑完整 KV 估算栈。
- `prefill_chunk_tokens is None`（默认）时 `unchunked_prefill_transient` 与 target 逐位相同却算两遍（`:2381-2388, 2758-2765`）。
- `_worst_concurrent_transient:3438-3469` 对 6 个分量各做一次全排序取前 k（k 通常 1–2）→ `heapq.nlargest`。

### 5.4 反复深拷贝与多遍扫描

- `metrics.py`：history 双端队列最多 1440 个快照，`snapshots()` 每次 `GET /api/v1/history` 深拷贝**整个队列**（`:565`）；一次 `/metrics` 抓取额外 2 次深拷贝（`:673-682`）；`health_payload:712` 为读 6 个标量深拷贝整个快照。
- `trace_replay.py`：每个 event dict 拷贝两遍（`:429, :475`）；`_rank_summary` O(ranks×events) 外加每 rank 3 次重扫；`replay_trace` 后续约 10 遍全量扫描可合并为一遍累加。
- `repository_analyzer.py`：每个 `.py` 读两遍、`pyproject.toml` 读三遍解析两遍；每文件约 7 次 `ast.walk`（一个 NodeVisitor 可合一）；files 列表 8 遍扫描。
- `compare_memory_reports`（`calibration.py:101-193`）对 comparisons 约 10 遍生成器；`verify_calibration_reports` 又从原始数据重推一遍 summary 里已有的统计量（两套公式需人工保持同步，见 §3 scripts 里还有第三、四份）。
- 8+ 处 `--json` 块的**双重序列化**：`payload = json.dumps(...)` 算完即弃，`write_json` 再序列化一遍——大报告（trace_replay 内嵌全部 event）CPU 和峰值内存×2。
- `llm_estimator.py:485-506`：`decode_steps` 每生成 token 一个 dict 全量嵌入报告（4096 token = 4096 个 dict），且逐步重算本可闭式求和的 FLOPs——建议聚合 + `include_decode_steps` 开关（报告 shape 变化需过一次消费者确认）。
- `topology.py`：`_simulate_flows` 每轮对每条链路重扫全部 flow 路径（`:556-567, 658-664`）；ring collective 对相同 (src,dst,payload) 重复跑 Dijkstra `2*(N-1)` 轮无 memo（`trace_replay.py:57` 反而有缓存，可对齐）。
- 小项：`profile_catalog.official_compute_capabilities` 无缓存每次读文件（`:220-230`）；`preflight._contains_oom_marker` 每行 stderr 编译 4 个正则（`:1223-1232`）；`preflight.build_report:178` 从磁盘重读自己刚写的日志；`capabilities.py` lru_cache 因调用形参不同缓存同一文件两份（`:29-32`）。

### 5.5 启动开销

`__main__.py:8` 的 `from ._api import env` 触发 `fakegpu/__init__.py` 全量 eager import（实测 `import fakegpu` 165ms，其中 `calibration` 82ms），使 `:42` 的 `importlib` 懒加载完全失效——每个子命令（包括只需 `_api` 的 passthrough）都付这笔钱。`__init__.py` 改为 `__getattr__` 懒导出或让 `__main__` 绕开包 `__init__` 即可。

---

## 6. 角度六：测试与验证侧的配套

- **先补保护网再动刀**：`distributed_cli.py`（876 行）零测试覆盖，是 CLI 层最大的裸奔面；`torch_patch` standalone 路径、`preflight --steps`、`--include-exited`、`llm_cli` 多个 flag 均无覆盖。重构前对将保留的路径补冒烟测试。
- `scripts/validation/` 三个 bespoke assert 脚本与 `fakegpu validate` 的声明式 manifest 机制重叠，多数可转成 manifest case；静态内存统计在 `calibration.py`、两个 scripts 里共有 4 份独立实现（§3.4），收敛后 CI 校验逻辑只剩一份。
- `tests/test_virtual_smi.py` 是唯一让 `LEGACY_SCHEMA_VERSION` 和 `render_table` 重建路径活着的消费者——确认 legacy 支持是否还需要，不需要则连测试一起删。

---

## 7. 扫描中顺带发现的疑似 bug（与重构分开处理）

这些修复会**改变行为**，不应混入"功能不变"的重构 commit；已逐一确认、单独提交（状态见下，详见"实施进度"2026-08-17 §7 批量修复一节）：

1. ✅ 已修复（`1a7a337`）`torch_patch.py:3088`：`re.sub(r"rank:\\d+/", ...)` 在 raw string 里 `\\d` 是字面反斜杠+d，替换恒为 no-op——ShardMetadata placement 重写从未生效。
2. ✅ 已修复（`3ab2fb8`）`torch_patch.py:3817-3820`：重复调用 `patch()` 时新建 `_DeviceMemoryTracker`，但旧 tracker 注册的 `weakref.finalize` 仍会对新 tracker 调 `release()`，二次 `init()` 后内存记账漂移。
3. ✅ 已修复（`14a9f5f`）`smi.py:1453-1456`：`successful_writes` 在写入**之前**+1 序列化进状态，写失败时计数超前。
4. ✅ 已修复（`d519e97`）`validation.py:539` 捕获元组缺 `IndexError`，而 `_json_pointer:583` 对 list 索引越界正是抛它——错一个 pointer 会让整次验证崩溃而非记为 case 失败。
5. ✅ 已修复（`d519e97`）`validation.py:163-164`：`finished_at_ns` 与 `duration_seconds` 来自两次独立取时钟，报告内部自相矛盾。
6. ✅ 已修复（`2f483f5`）`capabilities.py:373-388`：`--library`/`--classification` 过滤发生在 summary 计算之后，打印的汇总与表格互相矛盾。**注**：中途一版未完成的修复曾假设 `apis`/`groups` 间存在实际不存在的 `"group"` 关联字段，会导致 `KeyError`——已改正为正确实现，过滤下沉到 `native_capability_report()` 内部、summary 计算之前。
7. ✅ 已修复（`6a87c50`）`distributed_cli.py:448`：rank 超时后 `continue` 发生在读取 report 之前，超时 rank 的诊断 JSON 被丢弃；`:438` 顺序 join 使 N 个慢 rank 串行放大超时。
8. ✅ 已修复（`a270e59`）`diffusion_estimator.py:1106-1107`：text encoder 均未暴露 hidden_size 时 conditioning width 静默取 1，产出接近 0 的错误估算而非报错（扫描发现的**最高风险静默兜底**）。
9. ✅ 已修复（`19e1999`）`training_plan.py:695-696, 730`：`_positive_int("auto")` 确认是 DeepSpeed 真实哨兵值且有测试覆盖，**保留不改**；`_bool_value` 对词表外未知字符串默认 True 才是真问题，已改为 raise。
10. ✅ 已修复（`97dc3f6`）`calibration.py:713-714`：用 `assert` 做控制流，`python -O` 下消失后 `None` 会被嵌进 observation。
11. ✅ 已修复（`4a3ae3b`）`smi.py:4322-4325`：`_process_name` 拼接 `sys.argv[:3]` 写入状态文件与 Prometheus label，可能泄露命令行中的路径/敏感参数，且是无界 label 基数来源。
12. ⛔ **复核后判定不是 bug，未改**。`calibration.py:2250-2272`：`_validate_collector_environment_contract` 只在 env var 已设置时比对。起初怀疑应改成"未设置即报错"，但 `build_cuda_serving_sample` 是文档化公开 API，供 vLLM/自定义 runner 在 FakeGPU 采集子进程之外直接调用（README 明确写了这个用法），`tests/test_analysis_extensions.py:757` 就是在这些变量全部未设置时调用并断言成功——"未设置即报错"会打破这个文档化用例。当前行为是该函数双重用途设计下的正确实现。

---

## 8. 实施顺序与后续计划

原建议顺序中，第 1 步的决策项已部分拍板（`fsdp_memory.py`、standalone 路径已删），第 2 步已全部完成，第 3、6 步各完成一部分。剩余工作按以下优先级推进：

1. ~~决策项~~ ✅ 已完成（2026-08-20）：`privateuse1/` 删除（`f35018d`），其余四项拍板保留并补覆盖，逐项理由见"实施进度"。
2. ~~纯删除（§1 + §2.2）~~ ✅ 已完成（`e451bd2`/`bc77d9f`/`6b00725`/`432c21c`）。
3. **公共设施（剩余，中风险）**：~~`FakeGpuConfig` dataclass 收敛 `_api.py`/`_runtime.py` 的 8 处 10 参数签名~~ ✅ 已完成（`_FakeGpuRuntimeConfig`，2026-08-17，−77 行）。~~CLI 注册器 + 共享 flag 工厂 + 统一 `--json`/退出码约定~~ ✅ 已完成（`c660cdb`，2026-08-20）。`--json` 输出块已由 `emit_json` 收敛（`f6891c8`）。**剩余**：`llm_cli` 与 `serving_plan` 约 12 个同名 flag 默认值不同（`dynamic/eager` vs `paged/sdpa`），本轮只统一了声明方式，默认值差异原样保留，需单独确认哪些属于有意为之。
4. ~~**大合并**~~ ✅ 已完成（`5d71400`/`242aa02`/`c6a1866`，2026-08-20）：golden 测试先行，再抽共享 helper；`_patched_dist_init/destroy` 复刻已整体删除。净减行数未达预期，原因见"实施进度"第四批。
5. **机械拆分（§4.2）**：主体已完成（2026-08-20，9 个新模块，全部只做移动 + 旧模块 re-export）。剩余项见"实施进度"第四批末尾——`preflight.py`、各 CLI 的 `main`、`torch_patch` 的 `_reporting`/`_ecosystem_compat`、`serving_plan` 的两个 pool model。
6. **兜底收敛与性能（剩余）**：§2.3 逐个评估（`workspace_profiles` 的 import 失败退化、`smi.py` publisher 主循环裸捕、`calibration` 第四层 sample fallback 等——行为敏感，建议转 warning 而非静默）；§5.1 allocator `snapshot()` 增量汇总与 `_allocate_allocator_block` 索引化（需 benchmark）；§5.4 metrics 深拷贝三处、`decode_steps` 聚合 + 开关（报告 shape 变化需过消费者）。
7. ~~**bug 修复（§7，与重构完全分开走 `fix:` commit）**~~ ✅ 已完成（2026-08-17）：12 项中 10 项修复、1 项（#6）修复过程中改正了一处 WIP 引入的 KeyError、1 项（#12）复核后判定不是 bug，保留不动。详见"实施进度"与 §7 逐项状态。

原第 2–5 步预计的 4,000–5,000 行削减目标，目前已完成约 2,500 行；剩余大头集中在第 4 步（serving_plan 合并，约 −400 行净减 + 消除最大维护面）与第 5 步拆分（不减行数但改善结构）。§7 的 bug 修复批次已收尾，后续如无新发现不再单独安排。
