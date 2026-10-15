# FakeGPU 重构分析报告

> 最新进度见下方 **2026-10-15 / 连续推进验收清单**；此前交付见 **可复现长度分组与 packing 隔离边界**、**随机采样、跨数据规模与自动校准工作区**、**完整原生训练预算、缓存条件与显存边界**等阶段；实施前承诺审计与后续路线见 **功能闭环计划**；重构结果见 **406**。历史章节的行号、规模和待办状态不能直接代表当前代码。
> **约束：重构保持既有功能；经复现的 bug 单独修复、单独提交。**

## 2026-10-15 / 连续推进验收清单

接续 `23a5bbc1`，按用户要求持续完成已设计的修改及验证。下面按依赖记录状态；实现、CPU 语义验证、真机观测分别验收，不用逻辑设备代替物理设备。

- [x] Qwen3.5 分段训练参考接口（cache 关闭）：卷积与递归状态、跨层 logits/梯度、checkpointing、保存恢复及异常恢复。
- [x] 三个框架的 packing 数据边界、真实 sampler 计数、消费核验及独立时间/显存预算。
- [x] 长度分组与原生评估共存：独立采样随机流、评估输入核验及事件预算。
- [x] 大数据规划的重复扫描和存储成本，保留历史报告兼容与纯标准库离线预算。
- [x] 更多模型、形状和重复 holdout；可用 GPU 验证与缺失硬件条件分开记录。
- [x] R24 长上下文及 serving 的真实 GPU 验证。
- [x] R25 diffusion generation 真测、训练估计及独立观测（明确限定 SD/SDXL 与缓存特征 U-Net）。
- [x] R26 本机软件矩阵、CPU／GPU 1 MoE 语义及原生两 CPU rank DDP／FSDP2。
- 范围外：R26 物理多 GPU／多机 TP／PP／CP／EP、ZeRO／FSDP。用户已确认当前无法开展，不列为待办或完成阻塞；保持未验证。
- [ ] 显存精度改进：定位 serving reserved decode 和 diffusion VAE 的低估，修改后使用新的独立观测验收。
- [x] R23 本轮有限原生数值需求、实验工具与产物，以及最终全仓／wheel 回归。

当前只有 GPU 1（Blackwell GPU）用于真实测量；GPU 0 不纳入未经授权的训练或多卡验收。前阶段进程均已退出，工作区开始时干净。

### 最终发布回归与本轮完成边界

已完成本轮所有可在现有授权环境中实施的改动和验收。最终全仓 **1734 passed / 1 skipped，153.09 秒**；跳过项专用于 CPU-only PyTorch build，本环境为 CUDA build。`fakegpu / tests / examples/experiments` 全量 Ruff、shell 语法与 diff check 通过。原生 CPU 集成套件重新构建，旧八组数值、新六组低精度检查及真实 PyTorch CUDA 可见 matmul 全部通过。没有新增 C++ 数值 API 或改变既有 kernel 不执行边界。

构建并隔离安装 wheel 后，**104 个 Python 模块逐字节一致**，六个原生库／可执行文件和 **82 个 profile** 与构建／仓库一致；安装包严格 capability 审计通过。纯标准库 `python -S` 检查 **42 个命令帮助**，原有完整预算／packing／eval **41 次 CLI**、Qwen3.5-2B 与 Diffusion **21 次**、实验计划／运行／恢复／比较／绘图 **5 次**、FSDP2 显式 shard **1 次**，共 **68 次功能检查 / 110 次含 help 检查**。绘图单独使用可选 Matplotlib，其他离线检查不导入 Torch/Transformers。**152 份历史预测逐字节不变**，新校准与 holdout 原始报告、事前封存均未改写。

wheel 新审计初次把完整比较 wrapper 与摘要里的 comparison 子树直接比较，导致审计断言失败；保留脚本和日志，改为比较对应子树后通过。未改生产输出、估计值或真实测量。原生新测试的 struct 格式开发失败同样保留。最终证据见 [release](tests/data/research_release_evidence.json)和 [19 份审阅记录](tests/data/research_release_records.json.gz)，包括源码 SHA、wheel SHA、构建安装日志、两个 wheel 审计脚本、全部 CLI 产物 SHA 索引与全仓日志。产物 `/tmp/fakegpu-final-release-20261015/`，仅本地提交，未 push／发布。

用户后续确认真实多 GPU／多机当前无法开展，已移出本轮验收范围，不再等待硬件信息。GPU 0 未使用。显存精度改进按后续请求继续推进；下述为修正前基线。32K serving reserved decode **24.07%** 低估、diffusion generation VAE allocated **12.0%–13.8%** 与 reserved 最多 **31.6%** 低估保留为公开精度边界；不因实现项完成而改为通过，也不把 allocator 经验包络当成物理 OOM 保证。

### R23 有限数值需求与 R26 本机矩阵

按明确研究计算建立有限需求：`Y = XW + 0.3 * (XA)B` 与四头 scaled QKᵀ。FP32/FP16/BF16 六组原生 cuBLAS host API，对照独立 Python float64 reduction 和每层独立 dtype 编码，所有输出最大绝对差 **0**。实际 **12 GEMM / 8526 FLOPs / 27 alloc+free / 4928 bytes H2D / 1080 bytes D2H**，独立重建计数一致；整数 GEMM 严格返回 NOT_SUPPORTED=15，输出保持不变。实际触发的七个 host API 已具备实现，本轮增加组合、dtype、参考与维护套件覆盖，没有凭空增加 CUDA 支持声明。任意 kernel launch、融合 softmax、backward 和 optimizer 设备代码仍不执行，按 capability catalog 与严格策略明确拒绝。

`tests/native/research_numerics.py` 纳入 `scripts/test.sh cpu`。原八组 C++ 测试、六组新检查及真实 PyTorch CUDA 可见矩阵乘法均通过；unsupported policy、cuBLAS fallback policy 与 **104 个显式 API / 102 个严格策略 API** 的 capability 审计通过。新脚本重命名时误改 struct 格式导致一次集成失败，已修正并保存失败日志，再完整执行 CPU 套件通过。该开发失败不删除或替换已有参考观测。

| 本机栈／工作负载 | 已验证范围 | 保留边界 |
|---|---|---|
| Torch 2.9.1 + Transformers 4.57.3 / tiny Qwen3 MoE | CPU + GPU 1，eager/SDPA × FP32/BF16，共八场景；独立逐 token 专家加权和、输入与参数梯度、完整模型 loss/梯度有限 | 随机 tiny 模型，不作预训练质量、吞吐或 VRAM 精度声明 |
| 同一 Torch / Gloo DDP 与 FSDP2 | 一个主机两个 CPU rank；合并 batch 梯度与 AdamW 更新一致；显式 per-parameter shard 字节核对 | CPU 逻辑 ranks 不代表物理 GPU 或机器 |
| HF/Factory：Transformers 5.6.0；Axolotl 0.18.0：Transformers 5.14.1、Torch 2.11 | 前述单 GPU 原生 SFT／packing／eval／预算证据 | 不外推物理分布式性能 |
| vLLM 0.19.1 + Torch 2.10 | 前述混合流式请求和单 GPU cache 观测 | 非多机或任意服务负载精度保证 |
| Diffusers 0.35.2 + Torch 2.9.1 | 前述 SD/SDXL generation 和缓存特征训练 | VAE generation 显存门槛失败保留 |
| 物理多卡／多机 TP/PP/CP/EP、ZeRO/FSDP | **未验证：用户确认移出当前范围** | GPU 0 未使用；当前唯一授权设备为 GPU 1 |

MoE 共 **91,008 参数**，checkpoint header 与实际参数一致；四个专家的 36 次 token 分配为 **7/11/8/10**。FP32 输出差最多 **4.66e-10**，BF16 参数梯度差最多 **1.91e-6**，所有完整模型梯度有限。GPU 身份和驱动 **580.178.04** 已记录。自动 standalone rendezvous 未启动 worker，保留启动记录并只停止本轮进程组，改为显式 loopback static rendezvous 后 DDP/FSDP2 成功。修正不均匀 shard 后再次两 rank 核验，旧 v2 与新 v3 报告分别保留。没有任何跨机完成声明。

独立标准库审计为 **errors=[]**；证据及版本矩阵见 [research native](tests/data/research_native_evidence.json)，[45 份压缩审阅记录](tests/data/research_native_records.json.gz)包括测量时的 runner、当前 runner、原始报告、失败启动、全部软件版本与检查日志。原始目录 `/tmp/fakegpu-research-validation-20261015/`。R23 本轮有限需求已经完成，后续新 workload 需要新的有限需求和参考；R26 真实物理分布式验收按用户说明移出当前范围，保持未验证。

### FSDP2 不均匀 shard（独立 bug 修复）

原生 Torch 2.9.1 CPU 两 rank、FP64 tiny MLP 的四个参数形状为 `[12,8] / [12] / [3,12] / [3]`。FSDP2 默认按每参数第 0 维 chunk，真实参数和每份 AdamW moment 为 **640 / 536 bytes**；原总字节除二得到 **588 bytes**，不能作为最大 rank 的准确值。

`estimate_training_plan` 新增显式 `parameter_shapes`，CLI 对应 `--parameter-shapes`；只接受已验证的 FSDP2 full_shard、统一 dtype、非空正整数形状和匹配的总参数字节。逐参数计算 rank 布局，预算采用最大 rank，梯度、moments、master 分区一致；默认未给形状时保留均衡总量近似，并在三语说明边界。不猜测未知布局，不扩展为自定义 DTensor placement 或异 dtype optimizer 支持。

先复现 **8 failed / 1 passed**，修正及 CLI 错误处理后相关 **37 passed**。原生两 CPU rank 再核验，预测布局 **[640,536]** 与两个 rank 的真实 moment 张量一致，预算 moments **1280 bytes**。DDP/FSDP2 梯度相对独立合并 batch 最大绝对差 **5.55e-17**，AdamW 更新差 **4.34e-19**。旧报告与新复核分别保存，没有覆盖；152 份历史预测仍逐字节一致。原始记录 `/tmp/fakegpu-research-validation-20261015/`，版本化矩阵随 R26 归档。CPU 同步不计入物理多卡／多机验证。

### P5 实验批量运行与产物管理

新增纯标准库 `plan/run/compare-experiments` 与对应公开 Python API。显式有限参数网格、重复次数、最大运行数、argv 模板、声明输入、数值指标 JSON 路径和超时；计划只读，运行顺序执行。稳定 ID 绑定输入完整 SHA、可执行文件、FakeGPU 源码及指定环境。每次尝试单独保存 stdout、执行状态、错误原因、产物 SHA 和指标索引，恢复核验成功记录，失败／中断只有 `--resume --retry-failed` 才创建新尝试，原始失败记录不覆盖。共享进程执行器仅新增可选 cwd/purpose，既有校准默认参数和错误行为保留。

比较可在输入文件删除后用 `python -S` 离线执行，逐份核验成功产物及历史尝试记录；不同工作区分别统计，不混合环境。可选 Matplotlib 导出独立 SVG/PNG，显示重复均值与观测范围，不标为置信区间。外部包版本需作者固定，命令须在前台运行；没有声称对未声明外部依赖做完整沙箱或自动环境快照。

19 项专用测试连同工作流与 CLI 回归共 **81 passed / 33.07 秒**，覆盖失败保留、显式重试、活跃进程拒绝重叠、超时清理、自修改输入、产物篡改、压缩报告与无 ML 依赖离线比较。CPU 示例三种规模各三次，共九个进程，独立重建整型 checksum 全部一致，恢复报告逐字段相同，SVG 导出通过。三语命令表保持一致。证据见 [实验工具](tests/data/experiment_toolkit_evidence.json)及 [57 份审阅记录](tests/data/experiment_toolkit_records.json.gz)，原始目录 `/tmp/fakegpu-experiments-20261015/`。

### R25 generation 与缓存特征训练

Diffusers 0.35.2 / Torch 2.9.1+cu128 / Transformers 4.57.3 / PEFT 0.18.1，GPU 1。SD 1.4 512²、SDXL 1024² 各五个独立 generation 进程，BF16、SDPA、CFG 7.5、请求 8 inference steps。三阶段 allocated 误差分别为 SD **1.578% / 5.000% / 12.001%**，SDXL **1.825% / 3.651% / 13.819%**；VAE reserved 最多低估 **28.802% / 31.640%**。10% 低估门槛明确未通过，不事后拟合系数或替换原始样本。

原观测脚本的 `.sample` 检查漏掉 tuple 返回值，原十次最终 NumPy 图像均已验证有限，CUDA 峰值仍有效；原 scope 文案不能作为逐 forward 有限性的证明。新增可复用 `tests/native/diffusion_generation.py` 正确识别 Tensor/tuple/ModelOutput，再分别补一次语义检查，所有 primary outputs 有限，峰值和图像 SHA 与原矩阵完全相同。补充检查没有替代原五次精度样本。初始可选 torchao 导入不兼容，使用隔离环境排除该可选包；未改动原有环境。

新增 `plan/run/estimate/validate-diffusion-training` 与公开 Python API。输入为明确的 FP32 latent/noise/text-feature safetensors 和 I64 timesteps，支持标准四通道 UNet、SDXL text_time conditioning、full/LoRA、非重入 GC、可选 FP32 EMA、native AdamW。参数/梯度/optimizer/EMA 精确持久存储是下界，activation/reserved 不用常数补齐；总时间与显存包络要求匹配 GPU UUID、软件、权重/数据 SHA、allocator、缓存与执行实现的两个独立原生进程。按真实尾批和 accumulation 窗口计数，首次窗口形状开销单独建组，缺少重复覆盖则不出总预算。

CPU 八个真实 tiny UNet 场景覆盖 full/LoRA × 普通/text_time × GC 关闭/开启；尾批梯度与独立合并 batch 最大差 **6.33e-8**，普通权重和 EMA 导出/重载输出差均为 **0**，EMA 导出未改写训练参数。初次 LoRA 重载缺少显式 safetensors 格式的失败保留，补齐格式后通过。纯标准库 API、输入文件删除后的压缩报告离线预算、CLI 误差门槛及拒绝覆盖旧原生报告均通过。

正式矩阵为五条随机缓存特征、microbatch 2/GAS 2、GC，三个条件：**SD 1.4 full＋EMA 0.9、SD 1.4 LoRA r4、SDXL LoRA r4**。每条件两次 3-epoch 校准，先封存 5-epoch 预测，再运行两次独立 holdout。共 **12 processes / 96 updates / 144 microsteps / 240 行消费**。五个指定 kernel cache 每进程新建为空，OMP/MKL=1；不宣称 OS/data cache 为空。所有正式训练成功，无自动重试、误差选择或延长。

| 条件 | 两次 total-time APE | holdout allocated / reserved GiB |
|---|---:|---:|
| SD 1.4 full＋EMA | 3.185% / 0.155% | 18.179 / 18.666 |
| SD 1.4 LoRA | 0.785% / 1.392% | 2.158 / 2.250 |
| SDXL LoRA | 0.712% / 0.044% | 6.336 / 6.998 |

六次 allocator 峰值均未超过校准包络。Full＋EMA 最终保存及 SHA 校验约 **15.5 秒**，已计入总时间；setup 分项最大误差 **10.55%** 同样保留，不能用总误差掩盖。独立审计重建输入字节 SHA、逐轮消费、时间守恒、进程等权预算和预封存，逐块扫描全部导出张量有限，EMA 与最终训练权重不同，结果 **12 processes / errors=[]**。102 个生产模块在正式矩阵保持冻结并保存源码快照。随后仅为 CLI 添加可选验收阈值，生产训练与计划核心不变，diff 归档。

阶段全仓 **1702 passed / 1 skipped，146.90 秒**；最终定向回归 **77 passed**，Ruff/diff check 通过。预算仅覆盖缓存特征 denoiser，排除原始图像准备和 VAE/text-encoder 训练；实测包络不是物理 OOM 或任意长跑保证。证据及 401 份来源见 [diffusion](tests/data/diffusion_native_evidence.json)和[审阅记录](tests/data/diffusion_native_records.json.gz)，原始资料 `/tmp/fakegpu-diffusion-20261015/`。

### Diffusion scheduler 调用数（独立 bug 修复）

真实 SD 1.4 请求 8 inference steps，PNDM `skip_prk_steps=true` 实际调用 UNet **9 次**；SDXL Euler 为 8 次。原 `workload_scale.denoiser_evaluations` 一律等于请求步数，会给出错误的实际调用计数。

现在按 model_index 声明的本地 Euler/PNDM 类和 scheduler_config 计算：Euler 为 N，PNDM 跳过 PRK 时 N>1 为 N+1、N=1 为 1；PRK 模式为 N+9，并拒绝不足四步的 PRK 请求。未声明或未覆盖的 scheduler 明确返回 `null`，请求步数保留独立字段；不增加“一步一次”的兜底。scheduler 配置 SHA 纳入工作量元数据，显存计算字段全部保持一致。

先复现 **4 failed / 2 passed**，修复后 diffusion 检查 **16 passed**；Diffusers 0.35.2 四个真实 CPU scheduler 的 timesteps 数逐项一致，并与原 GPU SD/SDXL 的 9/8 次调用吻合。旧 GPU 报告和预测未改写；对两个真实 pipeline 重算后，除 workload_scale 外所有字段完全一致。原始记录位于 `/tmp/fakegpu-diffusion-20261015/scheduler-native/`。

### R24 长上下文与在线 serving

Qwen3-0.6B、GPU 1、BF16 SDPA/dynamic KV，8192/32768 prompt tokens＋128 generated tokens。每种长度对 allocated/reserved 各运行五个独立 Transformers 4.57.3 / Torch 2.9.1 进程，共 **20 次真实观测**。生产 Python 在矩阵内冻结，99 个模块另存完整源码快照。

32K 的 allocated prefill/decode APE 为 **2.020% / 7.340%**，没有低估；reserved prefill/decode 低估 **7.348% / 24.073%**。reserved 的 10% 低估门槛明确未通过，原始结果与失败门槛保留，没有按误差重跑或加常数把公式改成通过。该分析式张量模型不能当作 allocator 预留或物理 OOM 上限。

新增纯标准库 `benchmark-serving`：显式 token IDs、到达偏移、并发限额、重复轮次、流式原始事件、usage 核对、文本 SHA、客户端排队、首个非空文本和总延迟。HTTP/SSE/超时失败返回可审阅报告，不重试；错误体自身超时也保留原始 HTTP 失败。`.json.gz` 输出与 `python -S` 支持。协议与 CLI 检查 **48 passed**，包括成功/失败 CLI 状态与压缩报告。

vLLM **0.19.1**、单卡 eager、2048-token chunked prefill、4 个 active sequences、8192 max model len、25% memory utilization，prefix caching 关闭/开启两台依次启动的自有 loopback 服务。每服务三轮、每轮 12 个预声明混合请求（64/256/1024/4096 prompt、32/64/96/128 output），共 **72 requests / 5760 completion tokens**，全部通过用量和 SSE 对账。每轮保留 Prometheus 与执行后显存快照，未把快照称为峰值。

关闭 prefix 时三轮 **88.723 / 3.529 / 3.482 秒**，开启时 **4.711 / 3.422 / 3.465 秒**。首轮开销完整保留；两服务存在共享系统/kernel 缓存，不能把首轮差异归因为 prefix 开关。后续轮次约 **272–281 tokens/s**，仅代表这组重复 token-ID 合成负载。关闭时 prefix queries/hits 均为 0；开启后累计 **48,960 queries / 44,352 hits**。HTTP 首个文本块不等于设备首 token，单块可包含多个 token。

初次 serving 驱动漏传 plan，另有临时 `queue.py` 遮蔽标准库的启动问题，诊断日志保留；所有有效 v2 观测与预测未改写。原生正向请求测量后补强了 HTTP 错误体超时处理，归档原测量客户端源码及该错误路径的回归。证据见 [serving](tests/data/serving_native_evidence.json)与[审阅记录](tests/data/serving_native_records.json.gz)，原始资料 `/tmp/fakegpu-serving-20261015/`。这不覆盖多卡、speculative、量化 cache 或任意在线负载的精度保证。

### Qwen3.5-2B 与重复独立 holdout

本地 `Qwen3.5-2B-Base` 在 CPU 导出独立 text checkpoint，原模型保持只读；**1,881,825,088** 个 base 参数，导出前后 logits 精确相同，未初始化 CUDA。HF Trainer/PEFT 增加 512/1024-token BF16 LoRA 与 512-token NF4 double-quant QLoRA，均为 **417,792** 个 FP32 trainable 参数、非重入 GC、microbatch 2、GAS 2、8 条固定形状顺序合成数据。CPU 先确认 16-update 校准覆盖 24-update 目标全部 18 个成本组，模型 forward 未参与选择。

每条件两次独立 16-update 校准、先封存预测后两次独立 24-update holdout；共 **12 个原生进程 / 240 updates / 480 microsteps / 960 行消费 / 12 次根保存**。GPU 1、五个 kernel cache 每进程新建为空、OMP/MKL=1；生产 Python 在整个矩阵保持冻结。没有训练失败、按误差延长或重跑。

| 条件 | 两次 raw 总时间 APE |
|---|---:|
| 2B LoRA / 512 / GC | 0.649% / 0.607% |
| 2B LoRA / 1024 / GC | 1.288% / 0.239% |
| 2B NF4 QLoRA / 512 / GC | 0.054% / 1.578% |

独立标准库审计重建逐批顺序、原始 int64 输入 hash、消费前缀、时间守恒、成本分组/进程等权均值及显存包络；核对预测早于两次 holdout，以及 12 份 adapter 各 24 个有限 FP32 张量、12 个非零 LoRA B。结果 **12 processes / errors=[]**。保留审计开发的旧 25 行断言及 raw config 缺少显式 eval_strategy 的失败，改用本次已声明 8 行与实际 native TrainingArguments 默认值后通过；未修改任何训练报告或预测。

这是本机、这三种配置和同一数据条件的重复观测；其他 GPU、多机与更大模型不由本阶段推断。详细峰值、所有分项误差、原始 SHA 及脚本见 [2B 证据](tests/data/finetune_qwen2b_evidence.json)和[压缩审阅记录](tests/data/finetune_qwen2b_records.json.gz)，原始资料 `/tmp/fakegpu-qwen2b-20261015/`。同时验证了上一阶段复用 dataset profile 的真实训练路径。

### 大数据规划与压缩报告

原生 runner 将刚完成的 native split 全量统计传给默认/长度分组规划器，校验报告完整性、计数、cap、framework 与内存来源；调用方负责同一未变更 split 的绑定。逐批两次 collation 和输入 SHA 核验保留。epoch permutation 用线性 bytearray 检查，v1/v2/v3 计划验签移除仅为计算 SHA 的整份 deepcopy；返回报告仍独立拥有数据。

公共结构化 I/O 支持 `.json.gz`：纯标准库读取，确定性 gzip header、流式 JSON、同目录临时文件与原子替换，失败清理并保留已有目标。普通 JSON/stdout 的字节格式保持一致。大型精确计划仍随规划范围增长，本阶段不声称常数内存或无限数据支持。

十万行 lazy 合成数据、原生 SeedableRandomSampler 和确定性合成 collator 对比：读取 **200,128 → 100,128**，计划逐字节相同；本次 CPU 耗时 **19.409 → 9.730 秒**，tracemalloc 峰值 **9,527,025 → 8,718,451 bytes**。高度重复的 26,992,593-byte 合成报告压缩为 **239,705 bytes**，写入 tracemalloc 峰值 **132,588,964 → 392,860 bytes**；写入耗时 **3.365 → 3.969 秒**。这些是特定 CPU/内容的观测，不作通用速度或压缩率保证。

专用检查 **74 passed**，全仓 **1658 passed / 1 skipped，143.42 秒**；`python -S` 从压缩的真实 calibration 报告生成压缩计划和预算，与原始封存值/coverage 完全相同，未导入 ML 包。**152 份历史预测逐字节一致**。三语 README、Ruff、diff check 与可复用 benchmark 小规模运行通过；证据见 [planning/storage](tests/data/finetune_planning_storage_evidence.json)，脚本见 [benchmark](tests/native/benchmark_finetune_planning.py)，原始资料在 `/tmp/fakegpu-large-data-20261015/`。

### Packing 原生报告配置绑定（独立 bug 修复）

公共 `validate_native_run` 现在将 v3 计划与报告的 framework、microbatch、accumulation、显式整数 seed、packing 开关、collator 及软件版本逐项绑定。此前 transfer 入口有较强检查，但公共验证器可能接受互相矛盾的证据。新增七种篡改回归，相关检查 **175 passed**；四场景的 **12 份有效原生报告、封存计划、时间及显存预算和 holdout 比较均保持一致**，**152 份历史预测逐字节一致**。测试开发时修正了夹具错误的 fixed-shape 标记，未调整实际 GPU 档案。

### Packing 完整预算验收（sampler v3）

`run-finetune` / `calibrate-finetune` 新增显式 `--packing-seed`，限定 dense Qwen3.5 text LoRA、一个设备、关闭评估及已指定规划范围；Factory 要求 neat packing，HF 输入须包含已证明的 segment IDs 与 reset positions。默认 unpacked 行为及旧 v1/v2 报告保持兼容。v3 保留原生 packer、完整 epoch bins、真实尾窗和丢弃信息，区分 dataset source rows、physical packed rows、logical sequences；实际输入 SHA 必须匹配 CPU 计划。参考执行、collation 和采样实现的代码 SHA 纳入预算签名，不跨数据或显式 packing seed 池化。

完整预算沿用现有事件和时间账本，未引入按样本数比例外推或缺组兜底。每组仍要求两个独立进程及足够重复观测。原生 collated width 可以包含合并 source rows 与 padding；以已核验的 prior CPU plan 和实际消费证明几何，不再套用源数据行的 sequence cap。旧固定形状分析入口明确拒绝 v3 packing，避免 Ax real-batch 恰好具有固定张量形状时，把 source rows 误当作最终训练行数。

正式条件为本地 **Qwen3.5-0.8B / base checkpoint 752,393,024 参数、LoRA rank 8 q/v / 319,488 trainable 参数**，GPU 1 Blackwell GPU；42 条等长、重复内容的合成逻辑序列。HF/Factory 为原生导出的 14 个 prepacked source rows、cap 256，Ax 默认合并行为 cap 128，real-batch 为 cap 192。所有实际采样 seed 均为 123，GAS=2、microbatch 配置为 3，SDPA、无 GC/评估/周期保存。OMP/MKL threads=1，五个 kernel 缓存目录每进程新建且初始为空；不声称数据缓存或 OS cache 同样为空。

每场景两次 88-update 校准、预测封存后一次独立 124-update holdout，共 **12 个有效原生进程 / 1200 updates / 2079 microsteps**。累计 **8070 次 source row 消费 / 4935 个 physical packed rows / 13686 条 logical sequences**，12 个根最终保存、无评估。每场景只有一次 holdout，不作为通用精度保证。

| 场景 | raw / net 总时间 APE | holdout allocated / reserved GiB | reserved 峰值低估 |
|---|---:|---:|---:|
| HF / PEFT | 1.430% / 1.492% | 6.209 / 6.703 | 0% |
| Factory | 1.537% / 1.599% | 6.174 / 6.666 | 0% |
| Ax 默认合并行 | 6.991% / 7.514% | 3.098 / 3.465 | **2 MiB / 0.0564%** |
| Ax real-batch | 6.718% / 7.523% | 3.936 / 4.262 | 0% |

保留 Ax reserved 包络的低估，不事后加余量把验证改成零低估。Ax 首次 setup 生成工作区 prepared 数据，后续进程复用；该差异属于五个 kernel 缓存之外的条件，分项误差同样归档。完整进程 allocator 包络仍不代表物理 OOM 保证、不同 GPU 的安全容量或任意长跑增长。

保留两次早期集成失败：PEFT 绕过 pre-hook，及完整 HF 训练后发现 Accelerate 把 seed 123 覆盖成 42，均已单独修复提交。Ax 默认第一次训练的源长度 cap 误拒绝属于训练后验证器问题：原 report、execution、失败工作区与 summary 均未改写，前后源码及 diff 有 SHA 证书；重验后直接复用该 88-update 报告，另启第二个独立进程，使用公开计划/预算接口封存预测后运行 holdout。没有为这项验证器修正重复第一份训练，也没有按预测误差重跑。主矩阵计时期间只有该训练后验证器在阶段之间改变；实际 CUDA 执行相关实现保持一致。随后协议类型检查与 legacy 入口保护通过回归和独立 real-batch 矩阵验收。

独立审计分别 **9 processes / errors=[]** 与 **3 processes / errors=[]**，重建所有 native source bins、逐批输入 hash、消费前缀、时间守恒、每进程等权成本分组及封存时间，并验证 12 份 adapter 各 24 个有限张量、12 个 LoRA B 非零。首次主审计的嵌套 workspace 检查未应用验证器变更证书，失败记录保留；补齐严格限定为该单一文件的前后 SHA 核验后通过，没有重跑 GPU。

显式 seed 修复后另在 25 条变长数据上完成四种模式的原生 CPU Trainer 8-update 验收，共 32 updates；三轮 loader 与真实训练消费匹配，实际 seed 均为 123。Ax real-batch 每轮原生丢弃一行，三轮 ID 为 **22 / 14 / 15**；这与旧档案的实际 seed=42 记录分开保留。real-batch 最初 cap 128 的 CPU 候选实际每行仅一条序列，另选 cap 192 证明每行两条逻辑序列；选择发生在该模式任何 GPU 测量之前。

最终全仓 **1637 passed / 1 skipped，145.14 秒**；独立 wheel 的 **98 个 Python 模块**与源码逐字节一致，`python -S` 下 **42 项 CLI 检查**全部重放封存值（包括四个 packing 场景、旧 pipeline 与长度分组评估），**152 份历史预测逐字节一致**。Ruff、diff check 与无 ML 依赖导入通过。可读[预算证据](tests/data/finetune_packing_budget_evidence.json)及[压缩审阅记录](tests/data/finetune_packing_budget_records.json.gz)保留原始散列；资料在 `/tmp/fakegpu-packed-budget-20261015/` 与 `/tmp/fakegpu-packed-real-budget-20261015/`。这些是版本化、同 GPU 条件下的验收；不覆盖 full/QLoRA packing、评估 packing 或 varlen 吞吐。

### 分段隔离参考接口

新增 `_finetune_segments.py`，以显式 context manager 对原生 dense text Qwen3.5 的 linear decoder 实例临时替换 forward：每行每段分别调用未修改的原生 GatedDeltaNet，传入空 cache；full attention 继续使用经过校验的 block-causal mask。参数、权重名称及 checkpoint 格式保持原生，退出或异常时移除 hook 与实例方法。没有修改第三方源码，也没有开放 CLI packed 训练预算。

接口限定 Transformers **5.6.0 / 5.14.1**，要求显式 2D reset positions、4D 浮点 block-causal mask、右侧 padding、忽略每段首 token 的 causal label，且 `use_cache=False`。拒绝缓存、错误隔离 mask、空行、非连续位置、嵌套安装及第三方全局 packing patches。TF 5.14 会丢弃 recurrent 层的 4D mask，因此有效 token 标记从已校验的根输入通过 decoder kwargs 显式传递，checkpoint 重算同样保留；不由 token 内容或 padding 位置的零值猜测边界。

两套真实 Transformers 各执行 CPU FP32 与 GPU 1 BF16 审计，每组 eager/SDPA × 无/nonreentrant/reentrant checkpointing，共 **24 个场景**。模型为随机权重、四层、hidden size 64 的真实类；两行分别包含 1/3/4 与 2/4 token 的段，第二行有右侧 padding。逐段独立原生运行是对照，逐一核验四层输出及全部参数梯度，并检查反事实输入、输入梯度、保存恢复和异常恢复。

- CPU 最大 logits 差 **2.54e-7**，逐层差 **2.61e-8**，参数梯度最大相对差 **2.56e-6**。
- GPU 所有有效 token 的 logits 与逐层输出差均为 **0**；BF16 参数梯度最大相对差分别为 **2.222% / 1.316%**，保留原始值，不称数值完全一致。
- 所有场景的后段 logits 对前段扰动差、后段梯度进入前段的最大值及相同 dtype 下保存恢复输出差均为 **0**。

保留两次开发审计失败：初次 CPU 将 TF 5.14 原生 Accelerate decorator 错判为全局 patch，已按已知 decorator 的原函数核验；初次 GPU 重载对照未对齐非持久 rotary buffer dtype，新建模型为 BF16、加载模型为 FP32，输出差 0.001953125。修正的是审计对照的执行 dtype，权重保持逐项相同；失败报告与日志均保留，未通过放宽阈值消除错误。

全仓回归 **1553 passed / 1 skipped，138.90 秒**，CLI/文档与分段边界复核 **58 passed**，Ruff、diff check、无 ML 依赖导入通过；**152 份历史预测逐字节一致**。GPU 审计期间冻结 **96 个生产 Python 模块**，SHA 全部保持一致。证据见[摘要](tests/data/finetune_segment_validation_evidence.json)及[压缩审阅记录](tests/data/finetune_segment_validation_records.json.gz)，原始资料在 `/tmp/fakegpu-segments-20261015/`。这验证分段训练语义，不提供 varlen 吞吐、0.8B 微调预算或带缓存分段推理的精度结论；packed sampler 和实际成本继续按上方清单推进。

### 长度分组评估验收

分段接口已提交为 `4a8fb881`。随后接入独立固定种子的原生 evaluation LengthGroupedSampler：每次评估使用同一排序，训练仍使用 seed＋epoch；包括 persistent worker 的缓存 loader。只接受一个有限评估数据集、原生 sampler 和尚未准备的 loader；评估分组策略、batch size、行数及长度 SHA 纳入预算签名，v1 sampler 不能声明该策略。

三个框架的真实 CPU Trainer、框架 sampler 方法及 collator，各验证 seed=0/123 与无/有 persistent worker，共六次原生训练。每次训练的周期评估与两次追加评估消费一致，训练输入 SHA 仍匹配独立 CPU 未来计划。全仓回归 **1566 passed / 1 skipped，139.75 秒**，相关预算回归 **94 passed**，新增策略最终单测 **38 passed**，Ruff 与无 ML 依赖导入通过。

真机矩阵先在 CPU 上声明 25 条训练/8 条评估数据、microbatch 3、accumulation 2、seed 123、每 2 updates 评估/每 4 updates 保存、124-update 目标。48/68-step 校准缺少 22/14 个重复成本组；三个框架均在 88 steps 获得完整 CPU 覆盖（115/117/115 组），因此只选择 88-step 校准，不依据 GPU 误差调整。接下来每框架两次校准与一次独立 holdout，五个新空 kernel 缓存目录，预测先封存；生产代码冻结。资料目录 `/tmp/fakegpu-eval-grouping-20261015/`。

九次原生 GPU 运行全部完成，没有训练失败、自动重试、误差筛选或校准延长。实际共 **900 updates / 1626 microsteps / 4530 条训练消费记录**，**456 个根评估与 234 个根保存**；最后一项是根调用数，嵌套保存不重复累计。每个框架仍只有一次 holdout，不作通用精度承诺。

| 框架 | raw 总时间 APE | net 总时间 APE | holdout allocated / reserved GiB | 峰值低估 |
|---|---:|---:|---:|---:|
| HF / PEFT | 1.925% | 1.934% | 4.204 / 4.799 | 0% |
| LLaMA-Factory | 0.271% | 0.268% | 4.167 / 4.762 | 0% |
| Axolotl | 0.213% | 0.211% | 3.155 / 3.697 | 0% |

分项 raw APE 为 HF setup/update/evaluation/checkpoint/gap/finalization **0.384/1.968/1.531/6.483/0.190/1.916%**，Factory **0.030/0.196/0.533/0.762/0.626/5.947%**，Ax **0.056/0.511/0.027/0.388/0.023/0.100%**。保留保存、尾部等分项偏差，不用总误差掩盖它们。

独立审计重建全部 124-step 训练顺序与张量 SHA、固定评估顺序/逐批描述符/内容 SHA，核验九份完整消费前缀、时间守恒、成本分组和每进程等权算术、封存先于 holdout，以及九份各 24 张量 / 319,488 参数的 adapter 全部有限、12 个 LoRA B 均非零。**96 个生产 Python 模块**从回归后到全部运行及独立审计完成保持冻结。三语文档复核另为 **4 passed**。

审计补正：Ax 的原生 builder 实际默认 `eval_on_start=True`，原始 CPU 预览已保存该参数，但预览脚本的手写事件表遗漏 step 0，因此独立审计第一次拒绝 Ax。已保留失败并按原生参数重建事件表；补充 CPU 覆盖为 **115/117/116 组**，原 88-step 选择仍完整覆盖。原始预览、九份训练、预算与封存预测均未修改；真实 runner 及生产预算从开始就记录并预测了这次初始评估，Factory 的训练外最终评估也已计入。没有因此重跑 GPU。

审计最终 **9 processes / errors=[]**，**152 份历史预测逐字节一致**。证据见[摘要](tests/data/finetune_length_evaluation_evidence.json)与[压缩审阅记录](tests/data/finetune_length_evaluation_records.json.gz)。这些是同一 GPU、模型、版本、数据和缓存条件下的结果，allocator 包络不代表物理 OOM 保证，也不等同于不同长度/GPU 可直接复用。

保留 CPU 审计开发失败日志（字段名误用 `verified`，应为 `matched`）；Ax 预处理因 IPC sandbox 限制及预览脚本缺少 multiprocessing main guard 的失败也保留。main guard 已补，并只终止确认归属本轮的进程组；这些均不是原生 GPU 校准失败或精度筛选重跑。

### LLM hybrid 推理公式拒绝（独立 bug 修复）

本轮静态检查对本地 Qwen3.5-0.8B text config 调用 `llm_estimator._decoder_dimensions`，实际 `layer_types` 为 18 个 `linear_attention` 与 6 个 `full_attention`，函数返回 24 层 `grouped_query / separate_key_value`，推理入口继续使用该结果会使 KV/FLOPs 公式不匹配。README 已说明 hybrid 未覆盖，但代码没有拒绝这类配置。

新增 `_require_attention_only_decoder`，只在推理入口检查已声明的循环参数和 layer types；顶层/nested text config、缺少显式 layer list 的 Qwen3.5、声明了不支持层或错误层数量均拒绝。标准 attention 显式层列表的 memory/compute 数值保持一致。共享维度提取保留给已经单独处理 hybrid 的微调估计器，避免使 Qwen3.5 微调失效；serving 入口通过同一推理检查拒绝错误预算。

先用四项公开推理/serving API 回归复现 **4 failed**，修复后 LLM/serving/finetune 相关 **150 passed**，补全 malformed layer 与 attention-only 保持性后的专用检查 **8 passed**，Ruff 与 diff check 通过。基线及日志在 `/tmp/fakegpu-hybrid-20261015/`。此修复明确支持边界，不宣称已实现 hybrid 推理 cache/FLOPs；长上下文验收使用有对应状态模型的架构。

### Prepared dataset 缓存隔离（独立 bug 修复）

原生 CPU packing 预览将同一路径数据从 32 条更新为 42 条后，Axolotl 仍从既有 `dataset_prepared_path` 返回 32 条数据；改用新目录后才读取 42 条。工作区此前仅记录原始文件 SHA，把 prepared/tokenized 目录当作可复用缓存，无法证明外部缓存对应当前原始数据。

校准工作流现在将 Axolotl prepared 数据及显式配置的 Factory tokenized 数据放入当前新工作区的 `prepared/`，原始配置和外部缓存保持不变。两个校准进程可以复用本工作区的派生数据；恢复时仍先核对原始输入 SHA，变更输入须创建新工作区。普通 `run-finetune` 的用户缓存行为保留。三项回归先复现 **3 failed**，修复后缓存与工作区检查 **31 passed**；原生复现、错误候选及日志保留在 `/tmp/fakegpu-packed-budget-20261015/`。

### PEFT direct forward 的分段入口（独立 bug 修复）

首次 0.8B packed LoRA GPU 尝试在首个 forward 失败：PEFT BaseTuner 直接调用底层模型的 `.forward()`，绕过 Module pre-hook，导致 decoder 缺少已验证的 segment active 标记。保留该尝试的原始日志与冻结源码 SHA，未生成成功校准或预测。

分段入口改为临时包装原生模型实例的 forward，调用相同校验后再进入原函数；退出或异常删除实例覆盖，拒绝既有实例 forward 覆盖。新增直接调用回归先复现 **1 failed**，修复后分段单测 **25 passed**。两套真实 Transformers/PEFT（5.6.0/0.18.1、5.14.1/0.19.1）在 CPU 各验收 eager/SDPA × 三种 checkpointing，共 **12 场景**：最大 logits 差 **2.09e-7**，LoRA 梯度最大相对差 **1.68e-6**，后段反事实输出差为 **0**，实例方法均恢复。原始资料在 `/tmp/fakegpu-packed-budget-20261015/`；这些 CPU 结果不替代后续完整 GPU 预算。

### 等长独立段合批的参考实现重构

逐段调用原生 recurrent kernel 的 HF 88-update 诊断耗时 423.83 秒（该进程因 seed 校验失败，不是成功预算证据）。分段实现随后按实际长度分组，将等长段沿原生 batch 维拼接；每个 batch 项仍从空卷积和递归状态开始，不补造 token，最后按原始行与位置恢复。模型权重及 full-attention 路径不变。这是明确的执行重构，不能继续使用其前实现的时间校准。

两套真实 PEFT 的 12 个 CPU 场景通过：最大 logits 差 **1.79e-7**、梯度相对差 **1.11e-6**、跨段反事实输出差 **0**。三个真实 collator 的 **18 个 GPU BF16 场景**再次核验各层、梯度、隔离及保存恢复；有效 logits 与层输出差、跨段扰动与输入梯度、同 dtype 重载差均为 **0**，参数梯度最大相对差 HF/Factory **2.222%**、Ax **1.714%**。新增单测同时核验 native 调用的 batch 形状与 padding 梯度，相关检查 **56 passed**。资料见 `/tmp/fakegpu-packed-budget-20261015/batched-gpu-reference/`；后续正式预算另行冻结、校准，不把语义审计当作性能测量。

### Packing 原生数据与计数基础（完整预算尚未开放）

新增 `_finetune_packed_batches.py`，先从 HF/Factory 的显式 segment IDs 或 Ax 的原始记录 bins 证明 token、label、位置和边界对应，再保留有效的原生 4D mask 或建立隔离 mask。输出区分 physical packed rows、logical sequences、非 padding/监督/causal tokens、分段几何和实际 tensor dtypes；四类输入的 hash 保留浮点 mask 的原始小端位模式，包括 BF16。拒绝缺失 source mask、边界 label 未忽略、非法分段、原生 token/label 改写及跨段 mask。

HF/Factory 原生 collator 不补齐任意 position IDs，适配层从已证明的 source 边界重建有效位置，仅把不参与状态/attention/loss 的 padding 位置设为零。Ax 原生 builder 从 `attn_decontaminates_packing` 派生 mask dropping，原始同名 YAML 设置不能替代实际参数检查；显式参考适配在准备 loader 前保留 source mask。nested-bin 的批量取数仅覆盖当前 Dataset 实例，支持普通 flat fetch 与 pickle round-trip，由 observer 的恢复机制管理；没有修改第三方全局类。

新增 `_finetune_packed_sampler.py`，保留原生 BatchSampler / MultipackBatchSampler，仅以独立 SeedableRandomSampler 替换原始随机流。逐 epoch 克隆 sampler、保持原对象和全局 RNG，读取真实 bins、尾批和丢弃 ID；两次原生 collation 的描述符及输入 SHA 必须一致。包括 native 声明的 epoch 长度与真实执行 microsteps，不能把 `rows / configured_microbatch` 当作 multipack 的真实步数。

原生 CPU 数据实验均为 **25 条逻辑序列**，HF/Factory 使用 Factory 原生 256-token neat-packing 导出的 13 行，Ax 使用各自原生 tokenization，GAS=2。请求 seed=123，但后续预算集成发现八份原始记录的 prepared sampler 实际均为 42；Accelerate 在 loader 准备时覆盖了种子，原记录不证明 123 生效。每种模式核对三轮 CPU 计划与真正 dataloader 消费，然后以轻量 scalar loss 执行原生 Trainer 8 次 optimizer updates，主动消耗模型 CPU RNG，消费前缀与记录中的实际顺序匹配。该 scalar loss 仅验证调度，不代表 Qwen 权重训练。

| 模式 | 每轮 dataset rows 消费 | packed rows | logical sequences | microsteps / updates | 原生 8-update 消费 |
|---|---:|---:|---:|---:|---:|
| HF | 13 | 13 | 25 | 5 / 3 | 14 microsteps |
| Factory | 13 | 13 | 25 | 5 / 3 | 14 microsteps |
| Ax 默认合并行 | 25 | 6 | 25 | 6 / 3 | 16 microsteps |
| Ax real batches | 24 | 18 | 24 | 6 / 3 | 16 microsteps |

Ax real-batch 模式原生强制 `drop_last=True`，三轮实际丢弃行分别 **4 / 18 / 16**，均按原生消费核验，没有擅自保留或计作已训练。最初 128-token Factory CPU 情景实际每行仅一条逻辑序列，另建 256-token 情景验证真实多段；选择发生在任何本轮 GPU 测量之前。保留原始位置 padding、mask dropping 与缺少 nested fetch 的失败日志，没有用猜测补齐 mask 或源记录。

真实 collator 接入四层 tiny Qwen3.5 后，三个框架各执行 eager/SDPA × 无/nonreentrant/reentrant checkpointing，共 **18 个 GPU BF16 场景**。输入逐项等于手写分段张量；有效 logits/逐层输出与逐段原生对照差、跨段输出差、跨段输入梯度、同 dtype 保存恢复差均为 **0**。参数梯度最大相对差保留为 HF/Factory **2.222%**、Ax **1.316%**，不宣称 BF16 梯度逐位相同。

全仓 **1596 passed / 1 skipped，138.79 秒**，专用 packing 检查 **22 passed**，Ruff、diff check、纯标准库导入通过，**152 份历史预测逐字节一致**。**98 个生产 Python 模块**在 GPU 审计期间 SHA 不变。证据见[摘要](tests/data/finetune_packing_reference_evidence.json)与[57 份来源的审阅记录](tests/data/finetune_packing_reference_records.json.gz)，原始资料 `/tmp/fakegpu-packed-20261015/`。

后续首个修正 PEFT 入口的 HF GPU 进程完成 88 updates，但预算校验拒绝 requested seed=123 / actual seed=42，未形成有效预算。随后使 Accelerate 的 loader data seed 与显式 packing seed 一致，并在计数入口要求安装标记及实际 seed 一致。真实 `prepare_data_loader` 回归先复现 **2 failed**，修复后基础检查 **5 passed**、包含预算新协议的相关检查 **30 passed**。原始八份 CPU 档案不改写，证据摘要增加 seed 更正；新一轮原生 CPU 预览实际 seed=123，逐批消费匹配，旧计数基础不再被解释为该种子的验收。

下一步接入分段形状、原生真实 epoch 计数及消费验证的预算协议，再做 0.8B packed LoRA 的独立完整时间/显存校准。当前生产 runner 的 unpacked gate 保留；这些 CPU 计数与 tiny 模型内核结果不等于 packed 原生训练预算已经交付。

## 2026-10-14 / 可复现长度分组与 packing 隔离边界

接续 `478f8278`，推进原生长度分组，并实测 Qwen3.5 packing 的隔离边界。CLI 仍为 34 个；新增 `run-finetune` / `calibrate-finetune` 的显式 `--length-grouping-seed`，没有把依赖全局随机状态的原生排序伪装成可预排的未来计划。

| 部件 | 实现与范围 |
|---|---|
| `_finetune_length_sampler.py` | 复用已安装 Trainer 的 LengthGroupedSampler 算法，以独立 CPU generator 在每 epoch 使用显式 seed＋epoch。只替换已知原生长度 sampler，由既有 observer 管理恢复；在 loader 已准备、自定义 sampler 或存在评估时拒绝。 |
| `_finetune_sampler.py` | 新路径使用 sampler plan v2，保存原生完整有序 lengths、grouping batch size、长度 SHA、行顺序和真实输入 SHA；旧顺序/随机路径保持 v1。预排器克隆 sampler，保持原始 RNG、数据和采样状态。 |
| `_finetune_transfer_plan.py` | 时间分区与成本均值复用原有实现。新路径保留完整长度序列、分组宽度和显式 seed 的身份，不跨不同长度工作量池化 epoch 排序成本。v1/v2 与签名互校，拒绝布尔 seed 或未应用的长度分组选项。 |
| runner / workflow / CLI | 要求规划 horizon 和关闭评估，保留原生数据处理、尾 batch、累积尾窗、最终保存、缓存条件及显存观测。选择该选项会改变 sampler 的随机流；没有更改模型 seed 或默认原生采样流。工作区记录、传递并核对显式种子。 |
| `tests/native/finetune_packing_semantics.py` | 极小真实 Qwen3.5 模型的反事实输入及梯度审计；纯 attention 为正对照，普通混合 attention 和 Ax packing patch 单独核验。随机权重的内核诊断不代表本地 0.8B checkpoint 的训练预算验收。 |

### 语义与回归

三个框架的原生 sampler 方法和 collator，各执行 5 次真实 CPU Trainer 训练，共 **15 次**。覆盖 5/13/17/65 行、seed=0/123/333、不同模型 seed 和 data_seed、worker=1/prefetch=256、尾 batch、小数轮次及更大的 megabatch。小模型启用 dropout，主动消耗模型 CPU RNG；所有消费顺序及输入 SHA 仍与独立 `randperm`、megabatch 分组/排序/最大项交换重建一致。

TF 5.x 的正确参数为 `train_sampling_strategy: group_by_length`；Axolotl 0.18.0 的 builder 将自身 `group_by_length: true` 转到该参数。此前 HF 冲突检查仅识别旧 `group_by_length`，允许新版分组参数与 `disable_shuffling=true` 同时存在，随后被顺序 sampler 静默覆盖。已用 CPU 用例复现，修复于独立提交 `3cc27e61`，HF 相关 **26 项**回归通过。

冻结后全仓回归 **1529 passed / 1 skipped，137.88 秒**；Ruff、diff check 和 `python -S` 纯标准库导入通过，**152 份历史预测逐字节一致**。新回归覆盖原生全局随机流不可预排、独立种子及状态恢复、v2 签名/长度/宽度约束、消费内容核验、原工作区命令兼容和早期参数拒绝。

### Packing 隔离边界

审计使用两层、hidden size 64 的真实 Qwen3.5 类，8 个 token 分为两段；传入分块 causal mask 与重置 position IDs。只改变第一段，检查第二段 logits 的最大差和第二段损失对第一段输入 embedding 的梯度。CPU 使用 FP32，Ax 另用 GPU 1 的 BF16 检查。纯 full-attention 对照的两个量均为 **0**。

| 路径 | 后段 logits 最大差 | 后段对前段最大梯度 | 结果 |
|---|---:|---:|---|
| HF / TF 5.6.0 普通混合 attention，CPU | 0.01347984 | 0.00111157 | 未隔离 |
| Factory 0.9.5 原生 neat-packing collator＋TF 5.6.0，CPU | 0.01347984 | 0.00111157 | mask 正确，混合状态仍跨段 |
| TF 5.14.1 普通混合 attention，GPU BF16 | 0.01269531 | 0.00111389 | 未隔离 |
| Axolotl 0.18.0 的 Qwen3.5 packing patch / TF 5.14.1 | — | — | decoder 缺少 `layer_type`，接口不兼容 |

因此本轮没有开放 Qwen3.5 packed 训练预算。普通 full-attention 的 mask 隔离不代表卷积及递归状态已重置；Ax patch 存在也不代表当前软件栈可执行。第三方源码没有被修改；错误与完整 traceback 保留，不能把该诊断的 `passed`（反例及正对照成立）解释为 Ax packed 训练成功。

### 原生长度分组验收与覆盖边界

验收模型为本地 Qwen3.5-0.8B text LoRA，设备 Blackwell GPU；25 条记录、microbatch 3、accumulation 2、seed 42、显式分组 seed 123、SDPA、无 GC/packing/评估。Factory/HF 处理后长度为 64 或 128；Ax 使用原生模板并 padding 到 64 的倍数。实际 shape 包含 `[3,64] [3,128] [1,64] [1,128]`，保留尾 batch 和部分 epoch。

所有 CPU 候选及缺口均保留。最初 120-step 目标的 64 组在 40–110-step 校准中缺少重复间隙样本；更稀疏长序列分布的 53/54 组也仍缺覆盖，没有启动这两种方案的 GPU 校准。随后在原数据的预声明网格中，只有 **89-step 校准 → 99-step 目标**获得三个框架共同的完整覆盖（60 组）。选择只依据 CPU 调度及样本计数，在任何计时 GPU 运行之前完成；不按 GPU 误差筛选、延长或重跑。

同一 seed 下，原生分组的完整 24-epoch CPU 计划相对原生随机顺序减少 padding：HF/Factory **60.280%**，Ax **39.061%**。稀疏诊断数据的降幅仅 **5.172% / 1.292%**。这些是 collator 工作量计数，未执行未分组的 GPU 速度对照，不能解释为对应吞吐提升。短试跑不保证覆盖，缺失成本仍拒绝总预算。

每框架两次 89-update 校准、一次 99-update holdout，实际消费分别为 **161 microsteps / 449 条序列**和 **179 / 499**；不是把小数 epoch 请求直接乘到行数。共 **9 个独立原生训练进程，831 updates / 1503 microsteps / 4191 条消费记录**。每组只有一次 holdout，每次五个新的空 kernel 缓存目录；全部 warmup、首次 shape、Trainer 间隙和最终保存计入完整预算。预测在 holdout 启动前封存，所有输入、调度及预算数值逐项核验，没有失败训练或精度筛选重跑。

| 框架 | raw 总时间 APE | net 总时间 APE | holdout allocated / reserved GiB | 显存峰值低估 |
|---|---:|---:|---:|---:|
| HF Trainer/PEFT | 1.353% | 1.358% | 4.210 / 4.799 | 0% |
| LLaMA-Factory 0.9.5 | 0.255% | 0.262% | 4.176 / 4.762 | 0% |
| Axolotl 0.18.0 | 0.245% | 0.244% | 3.151 / 3.826 | 0% |

分项 raw wall APE 同时保留，Ax 最终保存的相对误差明显较大，不能由总量误差掩盖：

| 框架 | setup | update | checkpoint | Trainer 间隙 | finalization |
|---|---:|---:|---:|---:|---:|
| HF | 0.088% | 1.527% | 1.982% | 0.834% | 1.011% |
| Factory | 0.864% | 0.149% | 1.053% | 0.733% | 1.900% |
| Ax | 0.221% | 0.257% | 31.629% | 2.116% | 1.785% |

独立审计重新执行原生预处理，以手写 `randperm`＋megabatch 排序及最大项交换重建 99-update 计划；9 份报告的完整输入 SHA、消费前缀、时间守恒、60 个成本组的每进程均值、描述符分布和显存包络均一致。9 份 adapter 各 24 个张量 / 319,488 参数，全部有限，每份 12 个 LoRA B 均非零。**95 个生产 Python 模块**从全仓测试后冻结至所有原生运行和审计结束保持一致。

独立安装的 wheel 95 个 Python 模块与源码逐字节一致，`python -S` 下 **32 项 CLI 检查**（旧 pipeline 20、新长度分组计划/预算/holdout/工作区 12）全部与封存值一致，152 份历史预测仍逐字节一致；三语 README 与命令清单另通过 **5 项**检查。

原始证据位于 `/tmp/fakegpu-length-20261014/`，包括 CPU 失败覆盖候选、稀疏分布、packing 反例及 Ax 接口异常。仓库提供[可读摘要](tests/data/finetune_length_validation_evidence.json)和[压缩审阅记录](tests/data/finetune_length_validation_records.json.gz)，路径替换后的记录不能作为可复用校准输入。实测只对应此模型、版本、数据、GPU 和缓存条件；单次 holdout 的误差及短程 allocator 包络不构成通用精度或 OOM 保证。

### 下一步

先为 Qwen3.5 的卷积与递归状态建立可执行的分段接口，验证 logits、梯度、跨层状态与保存恢复；再接入逐框架 packed sampler 的实际计数和独立时间/显存校准。长度分组后续补可复现的评估采样、更大的数据与 planning 存储成本，以及更多模型/GPU 的重复 holdout。R24–R26 的 serving、diffusion、多卡/多机路线仍待独立推进。

## 2026-10-14 / 随机采样、跨数据规模与自动校准工作区

接续 `084ff5a4`，推进约定的前三项：更换数据规模后的完整预算、默认随机 sampler、自动校准及实验管理。新增五个命令，CLI 共 **34** 个；原有严格 pipeline、定长 probe 及历史预测保留。新路径明确使用经验形状迁移，不能把相同张量形状解释为任意 token 或监督分布的成本完全相同。

| 部件 | 行为与边界 |
|---|---|
| `_finetune_sampler.py` | 从实际 prepared dataloader 取得 sampler/collator，以独立 sampler 在 CPU 规划目标 horizon；恢复 Python/NumPy/Torch CPU RNG，不推进原始 sampler。支持顺序及 SeedableRandomSampler 的 epoch 换种子、尾 batch 和累积尾窗。逐批保存行号、四类输入张量 SHA 和实际描述符，并在原生训练中逐批核对消费前缀。 |
| `_finetune_transfer_plan.py` / `finetune_transfer_plan.py` | 从已核验 CPU 计划选取未来轮次或更新数，无需观察目标数据的完整 epoch。绑定原生版本、实际模型/内核/collator、allocator、GPU、缓存条件与日志/保存策略。复用原有事件调度器及时间账本分区，不复制调度逻辑。 |
| `finetune_transfer_estimator.py` | 工作成本只去除训练数据定位、内容 SHA 和 sampler seed 的身份约束，保留模型 seed、格式、优化器等执行选择；启动、收尾及训练外根事件额外匹配目标的原始执行签名、完整数据和 CPU horizon。首次 shape 使用、首次有序窗口、warmup、epoch 首尾和事件上下文分别分组。每组至少两个独立进程；重复组在每进程至少两次，可由不同校准进程覆盖互补组。缺失组显式报告，无残差或样本数比例兜底。 |
| `finetune_workflow.py` / 两个私有输入与存储模块 | 固定两次新进程，显式步数/超时/本次新增子进程总时间上限；默认五个新空缓存目录。保存配置、日志、状态、结果和失败尝试，SHA 校验成功产物再复用；POSIX 锁及 PID/start-time 检查避免并发，超时只清理自身进程组。封存预测后原子发布，中断可续发布；不按误差自动扩跑。 |
| `finetune_transfer_cli.py` / `finetune_workflow_cli.py` | `plan/estimate/validate-finetune-transfer` 及 `calibrate-finetune`、`budget-finetune-workspace`。可导入其他数据的校准报告并在工作区保留内容快照；目标 setup 仍只用本工作区两次校准。CPU 重预算不依赖 GPU/框架环境且不会启动训练，只允许已保存 horizon 内的时长。 |

时间预算仍覆盖 runner 入口至原生返回及最后 CUDA 同步，包含完整 setup、warmup、训练、事件、间隙和收尾；Python 进程启动及报告写出不在此计时内。实际 sampler 规划、输入 hash 验证计入既有统计开销区间，没有新增显式 CUDA synchronize 或 allocator 峰值重置。raw/net 分开报告，区间组合进程均值的极值，非置信区间。保留完整 token/mask/label 描述符分布，明确标出形状池化忽略的分布差异、模型状态及未观测 kernel/allocator 历史。

显存采用贡献校准进程的整体 allocated/reserved 峰值包络，不能相加部件峰值，也不能据此推断组合长跑、碎片化增长或更小物理 GPU 的安全上限。没有沿用旧同数据显存验收的 0% 误差作为新迁移路径保证。packing、长度分组、自定义 sampler、streaming、多卡、自动 batch size、early stopping、resume training、checkpoint 轮转和额外 profiler 仍须单独支持。

### CPU 语义与回归

已核对 HF / LLaMA-Factory 0.9.5 的 Transformers 5.6.0 + Accelerate 1.11.0，及 Axolotl 0.18.0 的 Transformers 5.14.1 + Accelerate 1.13.0。两套 Trainer 的独立随机顺序审计为 12 次训练；接入新实现后又用三个框架的原生 sampler/collator 各做四次真实 CPU Trainer 训练，共 12 次，包含 workers=1 / prefetch=256、不同 seed/data_seed、尾部与部分轮次，输入顺序和 SHA 均一致。**Axolotl 0.18.0 的原生配置构造器不转发 YAML `data_seed`**；底层 Trainer 支持该参数不等于 Axolotl YAML 已生效，正式原生验收使用其实际 model seed 42。

源码冻结后的全仓回归 **1485 passed / 1 skipped，139.65 秒**，Ruff、diff check 与纯标准库包导入通过。新增回归包括消费张量同形但 token 不同的拒绝、随机流不变、首用 shape 与首用窗口区分、目标启动身份、不同规模覆盖不足、缺少 horizon/协议、工作区锁、失败恢复、超时清理后代、产物篡改、封存发布中断、历史工作池快照及纯 CPU 缩短预算。工作流测试用真实子进程串起 sampler/原生账本/时间/显存协议；不把测试构造的输入散列当作 GPU 证据。

**152 份历史预测完整字典和序列化字节保持不变**；上一阶段四个正式条件的计划、预算及 holdout 比较共 **20 项**也保持完整字典一致。原严格模式的分区器和调度组装器提取为共用内部函数，旧输出保持不变。

### 原生 GPU 验证

设备为 Blackwell GPU，模型为已核对权重的 Qwen3.5-0.8B text LoRA；microbatch 3、accumulation 2、cap/padding 128、warmup 2、SDPA、无 GC 或 packing。三个框架各用 **25 条源数据做两次 30-update 校准，73 条目标数据做两次 6-update 校准，再独立运行 39 updates（3 epochs）**。目标校准只消费 36 条记录，未跑完目标的一个 epoch；工作池可复用源数据成本，目标启动和收尾仍使用目标自己的两次测量。HF/Factory 的源和目标 data seed 为 42/123，Ax 使用实际生效的 seed 42。每次使用五个新的空 kernel 缓存目录。

另用 HF 的 25 条数据验证随机采样与周期事件组合：eval/save 每 2/4 updates，两次 108-update 校准后独立运行 124 updates。四组共 **18 个真实训练进程**；每组只有一次 holdout。所有预测在 holdout 启动前封存，实际有序输入、未来事件顺序及全部分组数值与封存预测核对。没有按误差筛选或重跑。

| 验证条件 | raw 总时间 APE | net 总时间 APE | holdout allocated / reserved GiB | 显存峰值低估 |
|---|---:|---:|---:|---:|
| HF Trainer/PEFT，25 → 73 条 | 0.313% | 0.313% | 4.212 / 4.465 | 0% |
| LLaMA-Factory，25 → 73 条 | 0.416% | 0.421% | 4.174 / 4.428 | 0% |
| Axolotl，25 → 73 条 | 0.212% | 0.210% | 3.151 / 3.824 | 0% |
| HF，随机采样＋周期 eval/save | 0.594% | 0.597% | 4.212 / 4.465 | 0% |

总量误差不代表各阶段同样准确。下表保留每个阶段的 raw wall APE；前三组没有周期 evaluation，checkpoint 列仍包含原生最终保存。

| 条件 | setup | update | evaluation | checkpoint | Trainer 间隙 | finalization |
|---|---:|---:|---:|---:|---:|---:|
| HF，跨数据规模 | 0.213% | 0.236% | — | 30.192% | 3.949% | 8.370% |
| Factory，跨数据规模 | 0.182% | 0.488% | — | 4.915% | 0.172% | 28.247% |
| Ax，跨数据规模 | 0.166% | 0.218% | — | 3.151% | 1.289% | 9.101% |
| HF，周期事件 | 0.152% | 0.583% | 1.278% | 2.894% | 0.010% | 0.725% |

负证据也保留：最初仅 CPU 的 68/84-update 事件方案缺少 14 个间隙组的第二个样本，未启动 GPU；改为预先声明的 108/124 后，原有 115 组规则全部覆盖，没有放宽阈值。HF 主矩阵的原始验收脚本在五次训练成功后错误地假设 26-step 缩短预算一定可用；原始失败记录保留，补充审计复用五次运行，新增 GPU 进程为零。三个框架均确认 26-step 的末尾间隙缺少独立覆盖，返回 `coverage_incomplete` 且不输出总预算；20-step 的已有覆盖可直接用 CPU 重算。更短并不自动意味着有校准覆盖。

这些结果只覆盖上述模型、框架版本、数据、缓存条件与一张 GPU。显存包络并不证明较长任务或更小物理显卡不会 OOM，单次 holdout 也不足以给出普遍精度保证。

独立审计从原生预处理数据和 `randperm(seed + epoch)` 重算行序及真实 collator 张量 SHA，核对 **673 updates / 1235 microsteps / 3483 条消费记录**；不使用生产预算器重算组件分区、均值、完整分布、raw/net/metadata 守恒和显存包络。103 份保存的 adapter 张量全部有限，每份 12 个 LoRA B 均非零；周期 checkpoint 的 optimizer/scheduler/RNG/TrainerState 及评估输入单独核验。18 个 UUID/PID 独立，94 个 Python 源文件从计时前冻结至审计保持一致。审计工具最初对 Ax 最终 `save_pretrained` 及嵌套事件 ID 的假设错误、随后本地 IPC 权限失败，均保留原始输出；修正仅发生在验收工具，没有改变生产代码或重跑 GPU。

wheel 独立安装后的 **94 个 Python 模块**与源码逐字节一致；在 `python -S` 下完成 **41 项 CLI 检查**（旧 pipeline 20、新 transfer 12、CPU 工作区重预算 9），历史 152 份预测仍逐字节一致。更新后的三语 README/命令清单另通过 5 项检查。原始运行、配置、日志、CPU 预览、独立审计、驱动及预测封存位于 `/tmp/fakegpu-transfer-20261014/`；仓库保留[可读摘要](tests/data/finetune_transfer_validation_evidence.json)和[压缩审阅档案](tests/data/finetune_transfer_validation_records.json.gz)。档案中的路径已替换，仅供审阅，不能作为可复用校准报告。

### 校准工作区目录边界修复

主功能及其 18 次 GPU 验收冻结于 `5cdbc0a1`。随后用纯 CPU 用例复现：把工作区放到模型目录下面时，原实现未提前拒绝，触发了被 mock 截止的运行时探测。按原流程继续训练，工作区新产物会改变模型目录指纹。修复在 `_prepare_inputs` 递归读取前核对解析后的绝对路径，提前拒绝工作区等于或位于被整体记录的输入目录内，覆盖模型、tokenizer、配置及实际选中的数据目录；`..` 和符号链接同样按真实路径检查。Factory 仅存放注册表及单独数据文件的 `dataset_dir` 没有被整体记录，因此其中不相关的工作区子目录仍可使用。

该修复不改变采样、计时、预算算法或已有独立目录的输入内容；GPU 证据仍对应上述冻结提交，不冒充已对修改后的源码重跑。修复的基线失败记录位于 `/tmp/fakegpu-transfer-20261014/workspace-input-baseline/`，后续回归与安装包检查位于 `/tmp/fakegpu-workspace-guard-20261014/`。

原工作流 28 项与新增路径回归 18 项共 **46 passed，16.80 秒**，三语 README 另有 4 项通过，Ruff 与 diff check 通过。修复后的 wheel 94 个 Python 模块与源码一致；`python -S` 下无训练依赖即可提前拒绝错误目录，已有 HF 20-step 计划、覆盖及预算文件逐字节保持不变。源码身份变化会正常生成新的封存请求目录，原预测的 seal 和全部产物不变；验收工具最初错误地要求两版请求 ID 也相同，其失败日志及修正记录保留。未新增原生训练。

### 后续推进顺序

1. **原生 packing 与长度分组**：先核验三套框架的实际边界、attention 隔离、sampler/collator 和累积计数，再开放相应预算；不把普通拼接或 position reset 当成隔离证据。
2. **规模与覆盖扩展**：优化大数据、多 epoch 的 CPU 计划存储与扫描成本；扩大变长数据、长序列、QLoRA/GC、模型与 GPU 的独立校准矩阵，单独检查长跑显存增长和 OOM 边界。工作区路径迁移也需要独立协议，目前须保留原始绝对路径。
3. **README 剩余研究路线**：继续 R24 长上下文/在线 serving、R25 diffusion 真机与训练估计、R26 多 GPU/多机 MoE。单卡微调闭环不代表这三条路线完成。

## 2026-10-13 / 完整原生训练预算、缓存条件与显存边界

接续 `c2d04c5c`，完成本轮约定的四项：互不重叠的时间账本、首次成本与指定缓存条件校准、未来轮次/步数的完整预算、动态 allocator 峰值与实际 OOM 对照。新增三个命令，CLI 总数增至 **29**。HF Trainer/PEFT、LLaMA-Factory 与 Axolotl 共用观测和预算协议；既有定长 probe 与历史预测接口保留。

| 部件 | 实现与边界 |
|---|---|
| `_finetune_timeline.py` | update、根事件及 runner/Trainer 边界使用同源坐标；填满 setup、Trainer 间隙和 finalization，核对无重叠及 raw/net/metadata 时间守恒。嵌套事件只累计根区间，不再增加 CUDA 同步或峰值重置。 |
| `_finetune_cache.py` | `uncontrolled`、`empty_kernel_cache`、`populated_kernel_cache` 三种显式条件；后两者核对五个指定目录及全部文件内容，不自动清理或预热。不同条件不能混用。 |
| `_finetune_pipeline_plan.py` / `finetune_pipeline_estimator.py` | 完整有序窗口＋事件历史＋epoch 位置＋warmup/首次阶段分组；首次部件从独立进程取样，重复部件在每个校准进程至少两次。间隙按相邻操作与首次/重复阶段匹配，不用单次总残差兜底。至少两次独立校准，缺少覆盖则拒绝。 |
| `finetune_pipeline_plan.py` | 从校准数据的完整实测 epoch 推导未来 `epochs` 或 `max_steps`；保留尾 batch、累积尾窗、小数轮次、末尾补触发和混合 step/epoch 调度的真实顺序。只支持已核对版本的顺序、无 packing 文本数据；改变数据行数需匹配的新校准。 |
| `finetune_memory_estimator.py` / `_finetune_allocator.py` | 相同完整部件覆盖下，单独报告独立进程 allocated/reserved 峰值包络。绑定 allocator backend、环境和内存比例上限；预算及用户余量不得掩盖 allocator cap。不把窗口峰值相加，不把拟合误差或余量当 OOM 保证。 |
| `_finetune_hf_runner.py` | 新增 HF 原生入口：平铺配置、明确 processed JSONL、动态 padding、完整 Trainer 生命周期与最终保存。CPU 验证 full/LoRA，GPU 本轮验证 Qwen3.5 LoRA；既有 `probe-finetune` 继续提供定长 full/LoRA/QLoRA 短跑。 |

`plan-finetune-pipeline` 和 `estimate-finetune-pipeline` 在目标训练前运行；`validate-finetune-pipeline` 是独立训练后的误差核对。预算覆盖 runner 入口至原生调用和最终 GPU 同步，包含模型/数据处理、checkpoint 内容散列及所有已记录阶段；**不含 Python 进程启动、报告构造和 CLI 写盘**。raw wall 与仅扣除已测统计开销的 net 时间分别给出；时间范围组合各部件进程均值的极值，显存范围取完整运行峰值的极值，两者均非置信区间。统计插桩并非全部被测出，net 也不是无插桩基准。

审查补齐了会改变实际工作量的边界：关闭日志或每 update 记录；拒绝未建模的周期 logging、checkpoint 轮转删除、最佳模型重载、early stopping、额外 profiler callback 与周期 `torch_empty_cache_steps`。预算绑定实际 TrainingArguments 的执行策略，以及旧 microstep 签名排除的预处理/数据缓存选项；不改变旧签名行为。新 API 对缺失信息明确报错，旧报告继续走既有兼容入口。

### 独立验证

模型为本地导出并核对权重的 Qwen3.5-0.8B text LoRA；设备为 Blackwell GPU。13 条变长训练记录、8 条评估记录，cap 256、microbatch 3、accumulation 2、eval batch 2、eval/save 每 2/4 updates、warmup 4、seed 42、SDPA、无 GC 或 packing。Factory/HF 的实际训练 batch shape 为 `[3,72] [3,64] [3,64] [3,72] [1,40]`；Ax 使用自己的模板与 collator，shape 为 `[3,98] [3,94] [3,92] [3,102] [1,70]`。这是三套原生流程的独立验收，不是同内核速度排名。

未来调度要求 HF 与 Factory 的 `disable_shuffling=true`，Ax 的 `curriculum_sampling=true`。支持版本分别为 HF Trainer 5.6.0、Factory 0.9.5 / Transformers 5.6.0、Axolotl 0.18.0 / Transformers 5.14.1。两个实际 CPU Trainer 栈还分别执行混合 eval/save 策略与小数轮次，共 8 个场景；`epochs=2.5` 在该 fixture 上产生 8 updates / 14 microsteps / 38 条序列，最终 `TrainerState.epoch=2.8`，不把请求轮次数值直接当作实际消费比例。

每个可用缓存条件有两次 68-update 校准、三次 92-update holdout。校准消费 114 microsteps / 298 条序列，holdout 消费 154 / 402；事件周期与 epoch 长度的组合相位均覆盖。全部 warmup、首次窗口、事件和间隙参与完整预算，没有沿用上一轮的稳态剔除规则。未来计划和预测在第二次校准完成后立即封存，之后才启动 holdout；逐次核对实际工作量与封存计划完全一致，并独立重建组件键、进程均值、极值及六阶段分项。

下表为每组 3 次 holdout 的最大 APE。显存列为校准的完整进程峰值，本次各组两次校准的最小值与最大值相同。

| 框架与缓存条件 | raw 总时间最大 APE | net 总时间最大 APE | allocated / reserved GiB | 显存最大低估 |
|---|---:|---:|---:|---:|
| LLaMA-Factory / 空目录 | 2.160% | 2.164% | 3.430 / 3.754 | 0% |
| Axolotl / 空目录 | 0.835% | 0.837% | 2.797 / 3.775 | 0% |
| Axolotl / 已有缓存＋固定外部依赖 | 1.732% | 1.737% | 2.805 / 3.775 | 0% |
| HF Trainer/PEFT / 空目录 | 4.100% | 4.105% | 3.467 / 3.791 | 0% |

分阶段 raw wall 最大 APE 同时保留，不能用总量抵消掩盖局部误差：

| 条件 | setup | update | evaluation | checkpoint | Trainer 间隙 | finalization |
|---|---:|---:|---:|---:|---:|---:|
| LLaMA-Factory / 空目录 | 0.569% | 2.070% | 3.514% | 9.201% | 1.320% | 19.828% |
| Axolotl / 空目录 | 0.544% | 0.866% | 1.350% | 1.251% | 3.558% | 5.387% |
| Axolotl / 已有缓存＋固定外部依赖 | 0.867% | 2.486% | 2.653% | 1.146% | 3.186% | 11.958% |
| HF Trainer/PEFT / 空目录 | 0.754% | 3.988% | 4.710% | 11.671% | 1.439% | 32.280% |

共 **30 个真实训练进程**：20 个正式校准/holdout、4 个保留的早期 Ax 诊断、3 个成功显存对照及 3 个预期 CUDA OOM。Factory 与 HF 首次空目录运行没有生成指定缓存产物，因此已有缓存条件明确标为不可用；Ax 生成 2,247 个文件、87,112,221 bytes 的真实种子。上述误差只描述本模型、数据和环境的实测结果，不是后续任意任务的精度保证。

五个受控目录为 `TRITON_CACHE_DIR`、`TORCHINDUCTOR_CACHE_DIR`、`CUDA_CACHE_PATH`、`TORCH_EXTENSIONS_DIR` 与 `PYTORCH_KERNEL_CACHE_PATH`。每次空目录运行使用新目录；已有缓存条件从首个空目录校准的真实产物构造不可变种子，再为每个进程复制新目录并核对相同内容。没有伪造缓存文件，也没有将空产物标成“已预热”。Triton 的 276 个 group 元数据含有 1,932 条指向首个空目录校准产物的绝对引用；已有缓存验证另外将原目录固定为只读依赖，并核对全部文件内容。因此该条件是“复制目录＋固定外部依赖”，不是各进程缓存完全自包含。运行报告的目录清单本身不认证外部引用或实际 cache 命中。这些条件不控制 OS 文件缓存、数据缓存、全部第三方缓存、GPU 时钟或系统负载。

Axolotl 首轮已有缓存实验中，外部源目录到第二次校准运行时才设为只读，完整依赖审计更晚，因此最初两次校准及前两次 holdout 仅保留为诊断记录。发现后固定追加四个训练进程：在完整事前核验后重做两次校准并封存新预测，将此前尚未启动的原第三次 holdout 用作严格组第一次验证，再追加两次验证。原始结果全部保留；同一 holdout 的复用路径明确记录，不在同一估计中重复计数。补充实验是修正缓存控制协议，没有按误差挑选、删除或重写样本。

HF 首次原生训练成功后，验收发现旧 observation validator 漏掉 `huggingface` 框架名。修复只在校验器允许列表增加这一项，并补三框架回归；训练入口、observer 和原生矩阵代码没有变化。保留最初失败的验收记录与完整训练产物，新版本校验通过后复用该校准，继续第二次校准及后续 holdout，没有重新采样。旧、新源码清单、原始文件、精确差异和续接驱动均归档，不将全部运行冒称为同一源码版本。

| 框架 | 低 allocator cap | 实际结果 | 高 allocator cap | 实际结果 |
|---|---:|---|---:|---|
| llamafactory | 2.5 GiB | 原生 CUDA OOM | 4.75 GiB | 14 updates 成功 |
| axolotl | 2 GiB | 原生 CUDA OOM | 4.75 GiB | 14 updates 成功 |
| huggingface | 2.5 GiB | 原生 CUDA OOM | 4.75 GiB | 14 updates 成功 |

低/高预算按校准结果使用预先确定的公式：`floor(0.75 × min allocated / 256MiB) × 256MiB` 与 `ceil(1.25 × max reserved / 256MiB) × 256MiB`；运行相同原生配置的 14 updates，仅改变新输出目录、时长及 allocator cap。失败保留真实 CUDA OOM 类型和 traceback，成功对照保留完整原生报告及权重。**这是进程 allocator 上限下的真实训练分配失败与成功对照，不是整张显卡耗尽测试，也没有测出精确最小显存。** 这些进程不加入时间校准。

独立审计去重检查 27 份成功报告及 3 份预期 OOM。504 份周期/终点 checkpoint 与 27 份最终 adapter 共 **531 份权重**，均为 24 个张量、319,488 个参数，全部值有限且 12 个 LoRA B 张量非零；checkpoint 的 step/epoch、optimizer、scheduler、RNG、TrainerState 齐全。训练数据从实际 Arrow/processed JSONL 独立重算，逐 batch 的 tokens、labels、padding、epoch/tail 及每次评估的输入 SHA 均一致，时间区间和守恒检查通过。训练账本不含 token ID，不能据此声称逐条 sampler 身份已认证。36 处最终 metadata 的事件后改写（21 个 README、15 个 Ax debug.log）明确单列，保存时文件清单不视为永久不变或实际磁盘写入量。

源码审计覆盖 **129 个文件**：127 个跨两版保持一致，另两个是已声明的 HF validator 一行修复及对应测试。每组实际采集版本、旧文件内容与新旧补丁均核对；原生采集和训练入口在这次修复前后没有变化。

原始配置、日志、checkpoint、报告、独立审核与 wheel 保存在 `/tmp/fakegpu-pipeline-20261013/`。仓库提供[可读验收摘要](tests/data/finetune_pipeline_validation_evidence.json)及[完整压缩审阅记录](tests/data/finetune_pipeline_validation_records.json.gz)。压缩包是 UTF-8 JSON，含原始记录哈希、配置、预测、审计和复现脚本；路径已替换，仅供审阅，不能作为校准输入。

同时修复既有 probe 的两个实际问题：PEFT 的 `all-linear` 必须使用字符串简写；QLoRA 准备阶段应透传与 Trainer 一致的 gradient-checkpointing 参数。真实 CPU PEFT 对照确认 non-reentrant 不安装输入梯度 hook，reentrant 控制保留该 hook；修复单独提交。

最终全仓回归 **1370 passed / 1 skipped，100.09 秒**，Ruff 与 `git diff --check` 通过。wheel 独立安装后的 **85 个 Python 模块**与当前源码逐字节一致；`python -S` 下四个正式条件的计划、预算及三份 holdout 比较共 **20 项 CLI 检查**与提前封存的结果完全一致，不导入 Torch/Transformers。旧 Factory/Ax QLoRA 报告仍可推算 128 microsteps / 64 updates。相对 `c2d04c5c` 的 **152 份历史预测**（132 定长、4 动态窗口、4 事件、4 event-only、8 epoch 模式）在当前源码及独立 wheel 下均保持完整字典和序列化字节不变。

本轮四项交付的是有独立证据的首版完整预算。后续仍需推进随机 sampler 与 packing 语义、更多数据/模型/GPU 的长期精度和显存增长、物理容量受限设备验证，以及既有 R24–R26 的 serving、diffusion 和多卡/多机真机覆盖。不能把本次同数据长短运行验证等同于任意研究任务的完整 toolkit 已完成。

## 2026-10-13 / epoch 位置与事件上下文联合校准

接续 `98ccfceb`，处理上一阶段事件周期与 epoch 位置固定对应的问题。`estimate_finetuning_windows()` 增加显式 `epoch_context=True`，CLI 对应 `validate-finetune-windows --epoch-context`；周期事件运行可同时指定 `--event-context --epoch-context`。新选项复用已有观测与校验，不改变训练 observer 或采集协议。

- 完整有序 batch 描述之外，分组保留每个 microbatch 从 1 开始的 `epoch_microstep` 数组，区分 `[1, 2]` 与 `[3, 4]`。绝对 epoch 编号不参与分组；窗口内实际发生的 epoch 重置也保留，例如 `[3, 1]`，不推断未来边界。
- epoch 元数据的类型、连续性和重置仍由共享 batch validator 检查；新模式只补充必需字段检查。缺少位置或独立校准覆盖会明确拒绝，不回退到仅按 batch 预测。warmup、所选完整组合的首次出现及每进程最少两个样本规则保持不变。
- 默认模式与原有 event-only 模式保留原键和完整输出。CLI 共用参数解析，原生验收矩阵从配置及实际 Trainer 参数读取正整数 eval/save 间隔，交叉核对 requested/resolved 值，不再把上一轮的 4/6 写死在验收逻辑里。

本机源码审计确认：Axolotl 当前实际参数为 `dataloader_num_workers=1`、`persistent_workers=False`、`prefetch_factor=256`；Factory 为 0 workers。对应 Transformers/Accelerate 路径在新 epoch 重新建立迭代器，首次取数发生在首个 update 的观测区间内。这解释了为什么相同 batch 描述仍需区分 epoch 位置，但不是对 worker 启动耗时的独立测量。源码片段、版本和 SHA 保存在本轮证据中；没有改动框架源码、模型或 worker 配置。

正式方案沿用 13 条训练记录、8 条评估记录及上轮模型/优化器设置，保持每 epoch 3 次 update，将 eval/save 改为每 2/4 步。事件周期 4 与 epoch 长度 3 互质，组合周期为 12；相同事件背景可以落在 epoch 的不同位置。25 行候选仅做 CPU 预处理，正式方案在 GPU 计时前按覆盖周期和运行成本选定。

### 独立 GPU 验证

训练设置为 Qwen3.5-0.8B text LoRA / Blackwell GPU / cap 256 / microbatch 3 / accumulation 2 / eval batch 2 / SDPA / seed 42 / 无 GC 或 packing。Factory 0.9.5 使用 Transformers 5.6.0，Axolotl 0.18.0 使用 5.14.1；两框架原生模板和 DeltaNet 内核仍不同，不作同内核速度比较。

每框架两次 68-update calibration、两次 89-update holdout，warmup 为 4 updates，共 **8 个独立 CUDA 进程，无失败或筛选重跑**。继承现有编译/文件缓存，没有受控冷启动结论。每份 calibration 实际消费 114 microsteps / 298 条序列，holdout 为 149 / 389，包含不完整的最后 epoch；最终补触发 eval/save 与完整可恢复 checkpoint 均按实际调用核对。

两个框架均有 **12 个组合，每份 calibration 每组 4 个有效样本**。每份 holdout 排除前 20 个 update（含 4 个 warmup），剩余 69 个窗口；三种 epoch 位置各 23 个，35 个紧接事件。超过校准长度的 steps 69–89 为 21 个窗口，三种位置各 7 个，其中 11 个紧接事件。四种稳定事件上下文实际覆盖所有三种位置；同 batch、同事件背景的 `[1, 2]` 与 `[3, 4]` 不再混合为一个组。

下表为两份 holdout 中的最大 APE。epoch 首窗口、后续完整窗口与尾窗口仅指本次 `[2, 2, 1]` 累积 fixture，不能把单 microbatch 窗口在一般配置中自动解释为 epoch 尾部。

| 框架 | 全部 69 个窗口 | epoch 首窗口 | 后续完整窗口 | 尾窗口 | 后段 21 个窗口 | 单个组合 |
|---|---:|---:|---:|---:|---:|---:|
| LLaMA-Factory | 1.716% | 1.577% | 1.591% | 2.250% | 1.991% | 3.176% |
| Axolotl | 2.597% | 3.902% | 1.588% | 2.759% | 2.501% | 7.292% |

受控诊断固定**相同的校准与目标有效窗口**及进程等权聚合，只移除 epoch 分组条件：Ax 首窗口误差为 16.54%–18.60%，后续完整窗口为 19.93%–21.96%；加入位置后分别为 1.48%–3.90% 与 0.08%–1.59%。总误差却仅由 2.62% / 1.36% 变为 2.60% / 1.34%，说明总量会掩盖分项抵消。该诊断使用新组合的统一排除规则，不冒充首次排除规则不同的旧 API 输出。**Factory 的位置影响很小，没有两个框架普遍显著改善的结论**；全部正负分项均保留。

四模型诊断同时保留仅 batch、batch＋epoch、batch＋事件和完整组合。Ax 仅加 epoch 已解释主要位置差异；加入事件后，同一组 12 类窗口中的最大误差由 8.23% / 9.83% 降到 4.78% / 7.29%。但完整组合的紧接事件窗口误差比仅事件模式更大：1.37% / 0.75% 变为 1.90% / 1.28%，第一份 holdout 的后段紧接事件也从 0.17% 变为 1.53%。更细分组没有保证每个子集都改善，不能据此给出普遍精度保证或因果解释。

独立审计从实际 Arrow 重算训练统计、epoch/tail、所有评估的实际输入 SHA、labels 和 padding，全部一致。160 份周期或终点 checkpoint 与 8 份最终 adapter 均为 24 个张量、319,488 个参数，值有限且 12 个 LoRA B 张量非零；checkpoint step/epoch 与 optimizer、scheduler、RNG、TrainerState 齐全。12 项事件后的 metadata 改写（8 个 README、4 个 Ax debug.log）明确单列，保存时清单不视为永久不变。训练账本不含 token ID，不能据此证明逐条 sampler 身份。

纯数据审计独立重建四份 holdout 的组合键、排除规则、进程均值和逐窗口预测，全部匹配；**46 个采集、验收及相关测试文件**从 GPU 运行前冻结到独立审计保持一致。原始报告、配置、日志、checkpoint、审计脚本与 wheel 保存在 `/tmp/fakegpu-epoch-context-20261013/`，正式 GPU 数据位于其 `period4/` 下。路径替换后的 [`finetune_epoch_window_validation_evidence.json`](tests/data/finetune_epoch_window_validation_evidence.json) 仅供审阅，不能用作校准输入。

最终全仓回归 **1139 passed / 1 skipped，71.94 秒**，Ruff 与 `git diff --check` 通过。wheel 独立安装后的 **76 个 Python 模块**与源码逐字节一致，`python -S` 下新组合模式的四份窗口预测及四份事件预测均与原生矩阵一致，旧 QLoRA 报告仍可推算 128 microsteps / 64 updates。相对 `98ccfceb`，44 份历史定长报告的 132 个投影、4 个动态窗口、4 个事件成本及 4 个原有 event-only 窗口预测，完整字典与序列化字节均完全一致。

本轮仍仅预测实际 trace 中有独立覆盖的稳态窗口，不识别事件或 worker 启动的因果成本，不预测未来 sampler、完整 pipeline 或动态 VRAM。

### 后续推进顺序

1. **完整时间账本**：先在已记录的独立 trace 上核对互不重叠的 setup、训练窗口、根事件、其余 Trainer 开销和 finalization。当前 update 只有 raw/net/metadata 汇总，没有与 phase events 同源的起止坐标；最小下一步是记录这些坐标及 runner/Trainer 边界，独立验证区间无重叠和时间守恒，嵌套事件不重复累计。保留间隙的 raw wall，不把全局 instrumentation 或 setup 内的 checkpoint hash 时间重复相加或扣除。当前 runner 结束坐标在原生调用及 GPU 同步之后，报告构造和 CLI 写盘在外，必须明确计时范围。首次窗口及首次事件后的恢复另建独立进程样本，不能拿稳态均值代替。
2. **未来工作量与完整预算**：先支持有明确计数语义的单卡、无 packing 文本训练，按实际框架调度核对 epochs/max_steps、累积尾部、最后一步补触发及训练前后事件。把预测的次数与上述独立成本结合，再用更长 holdout 验证完整 wall time；任意 sampler 或未覆盖工作量仍需独立支持证据。
3. **显存和范围扩展**：动态 VRAM 独立验证 allocated/reserved 峰值与 OOM 边界，随后推进 packing 语义、更多模型/GPU，以及原始 R24–R26。窗口 timing 结果不用于推导动态显存，也不代表完整 toolkit 已完成。

## 2026-10-13 / 周期事件后的训练窗口校准

接续 `9fe7c9c5`，在已有动态窗口 API 中增加显式 `event_context=True`，对应 CLI `validate-finetune-windows --event-context`。默认模式继续拒绝周期事件并保持历史输出；新的模式使用事件历史匹配已记录的稳态训练窗口，不推断未来训练或完整 wall time。

- 每个窗口分别保留最近包含 evaluation、最近包含 checkpoint 的完整事件边界。边界内同一步的根调用按实际顺序保留，包括重复调用、工作量、initial/first/repeated 阶段与嵌套结构。新的评估不会擦掉此前 checkpoint 的历史。
- `optimizer_updates_since_boundary` 表示边界后已经完成的 update 数，紧接事件的窗口为 0。上下文只从已校验的 `preceding_event_ids` 向前推进，组键不包含绝对 step、run 内 event ID 或耗时。完整有序 batch 描述与这两类事件历史共同决定分组。
- 每份报告独立剔除全局 warmup 和每个“窗口＋上下文”组合的首次出现，仍要求至少两个独立 calibration、每类在每份 calibration 中至少两个剩余样本。最初一次评估/保存之后的不可重复上下文不在本轮预测范围，没有为扩大覆盖降低阈值。
- `_finetune_event_context.py` 提取事件成本与窗口校准共用的工作量、调用树和阶段定义；窗口均值、独立报告校验及预测汇总继续复用原有实现。事件上下文模式额外核对真实 hardware 字段。没有改动训练 observer、框架原始代码或原始模型。

上轮 20/29-update 报告在加入完整上下文后没有可重复的有效组合，不能作为此次校准。新的合成训练数据包含 13 条等描述记录，原生预处理后 Factory 每条 48 tokens / 22 labels，Ax 每条 80 / 21；两者仍保留每 epoch `[3, 3, 3, 3, 1]` 的 batch 行数与 `[2, 2, 1]` 的累积窗口。评估沿用 8 条变长记录，训练设置为 Qwen3.5-0.8B text LoRA / Blackwell GPU / microbatch 3 / accumulation 2 / eval batch 2 / cap 256 / SDPA / seed 42 / 无 GC 或 packing。

### 独立 GPU 验证

每框架两次 72-update calibration、两次 101-update holdout，warmup 为 4 updates；eval 每 4 步、save 每 6 步。共 **8 个独立进程，无失败或筛选重跑**，没有清空或控制已有缓存。每份 calibration 消费 120 microsteps / 312 条序列，holdout 消费 169 / 441，包含不完整的最后 epoch。最终补触发 eval/save、完整可恢复 checkpoint 和框架最终保存仍按上轮独立事件参考核对。

两个框架均有 **12 个有效组合，每份 calibration 每组 4 个样本**。每份 holdout 排除前 24 个 update，剩余 77 个窗口，其中 26 个紧接事件、51 个不是紧接事件；超过校准长度的 steps 73–101 共 29 个窗口，其中 10 个紧接事件。排除规则在校准和 holdout 中分别执行，没有按实测误差删除样本。下表是各框架两份 holdout 的最大 APE：

| 框架 | 全部 77 个有效窗口 | 紧接事件 26 个 | 后段 29 个 | 后段紧接事件 10 个 | 单个组合 |
|---|---:|---:|---:|---:|---:|
| LLaMA-Factory 0.9.5 / TF 5.6.0 | 0.523% | 0.545% | 0.263% | 0.486% | 2.060% |
| Axolotl 0.18.0 / TF 5.14.1 | 1.051% | 1.608% | 1.546% | 2.392% | 4.108% |

另用**完全相同的有效窗口**进行诊断：只按 batch 描述合并 calibration，保留相同的进程等权均值，忽略事件上下文。Ax 两份 holdout 的紧接事件窗口会低估 9.164% / 10.159%，同时高估其他窗口 4.232% / 4.651%；这些偏差在总量中抵消，总误差仍不足 1.1%。加入上下文后，紧接事件的误差为 0.518% / 1.608%。这说明本次数据中分项预测比仅看总误差更有区分度，不是无事件随机对照或因果实验。

**Factory 没有一致改善**：忽略上下文时最大单组误差为 1.339%，保留上下文为 2.060%；总误差差异很小。完整正负结果均归档，没有用两个框架的合并平均值宣称普遍收益。这里新增的是可检验的事件条件与分项证据，不是普适精度保证。

进一步诊断发现，4/6 步事件周期与每 epoch 3 次 update 存在固定相位关系：每个上下文组还对应固定的 epoch 内位置。因此增加第三个对照，按“batch 描述＋epoch 内首 microstep 位置”分组。Ax 紧接事件的误差降至 4.776% / 5.819%，仍高于完整上下文的 0.518% / 1.608%；Factory 仍无一致改善。epoch 位置确实解释了部分差异，当前证据不能把剩余差异唯一归因于事件，后续需要改变调度相位或加入无事件对照。

独立审核从实际 Arrow 重算训练指纹、逐 batch/epoch 的 tokens、labels、padding 与尾窗口，并核对每次评估的真实输入 SHA；全部一致。116 份周期 checkpoint 与 8 份最终 adapter 均为 24 个张量、319,488 个参数，值均有限且 12 个 LoRA B 张量非零。checkpoint 的 TrainerState step/epoch、optimizer/scheduler/RNG 与文件清单均通过；12 处最终 README/log 的事件后改写明确单列。训练账本没有 token ID，不能因此声称逐条 sampler 身份已经核验。

独立纯数据实现还重建四份 holdout 的上下文、组合 SHA、排除规则、每进程均值和所有子集预测，逐项匹配。22 个执行/旧验收文件与新矩阵自身共 **23 个源码哈希**保持一致；原始报告、配置、日志、checkpoint、诊断与 wheel 位于 `/tmp/fakegpu-event-windows-20261013/`。路径替换后的 [`finetune_event_window_validation_evidence.json`](tests/data/finetune_event_window_validation_evidence.json) 仅供审阅，不能作为校准输入。

最终全仓测试 **1108 passed / 1 skipped，70.92 秒**，Ruff 和 `git diff --check` 通过。wheel 独立安装后的 **76 个 Python 模块**与源码逐字节一致，`python -S` 下两个框架的四份窗口预测及四份事件预测均与矩阵一致。旧 QLoRA 报告仍可推算 128 microsteps / 64 updates；相对 `9fe7c9c5`，44 份旧定长报告的 132 个投影、4 个动态窗口与 4 个事件成本结果，完整字典及序列化字节保持完全一致。

### 边界与下一步

这里验证有限事件历史条件下的实测窗口成本，不把与普通窗口的差异解释为事件的因果开销，也不证明更早历史、cache、时钟或系统负载已无影响。下一步先改变 epoch 长度或事件间隔，区分 epoch 起始取数与事件后的恢复，并检查当前分组能否覆盖不同相位。事件本身的耗时与窗口预测分别报告。完整 pipeline 仍需补齐首次形状与首次事件后的恢复、setup/finalization、未采样框架开销及未来 sampler/调度次数，不能将本轮稳态窗口与上一轮事件秒数直接相加当作全程预测。packing 隔离语义与动态 VRAM 仍按既有路线独立推进。

## 2026-10-13 / 原生评估与 checkpoint 事件成本

接续 `f84d2615`，实现上一阶段提出的周期事件账本与独立成本验证。新增 `estimate_finetuning_events()` 和第 26 项 CLI `validate-finetune-events`；它预测已记录 holdout trace 中的评估/保存子总时间，尚不推断未来调度或完整训练时间。

- `_finetune_event_observer.py` 接管原来 observer 内的阶段包装，记录真实 `evaluate` / `predict` / `_save_checkpoint` / `save_model` / `save_pretrained` 调用树。保留父子关系、step/epoch、训练内外边界、成功状态、同步 wall time 和统计开销。同一步的独立调用分别保留，阶段总量只累计根事件。
- 评估记录实际处理数据的有序行指纹、成功消费的 batch 描述及输入内容 SHA，不保存 token 内容。任意自定义或有随机变换的数据集不额外遍历，无法安全识别时保持 unknown。保存记录调用结束后实际存在的文件及大小；这些值不是本次写入量、物理 I/O 流量或持久化完成时间。
- update 的 `preceding_event_ids` 关联两次更新之间的训练内事件，供后续检查保存对 allocator/cache 和下一次训练更新的影响。新协议纳入校准签名，旧报告缺少该可选协议时保留原有行为；账本、汇总、前继关系与阶段计时交叉验证。
- 初始、首次、重复和最终事件分别匹配。每组须在至少两个独立 calibration 中覆盖；重复事件在每个 calibration 中至少两个样本，进程均值等权聚合。holdout 秒数仅用于比较，不参与拟合。首版限无 packing 的 dense 文本模型、loss-only 且无自定义 metrics/生成的评估，以及已知文件布局的本地保存。
- 原生执行报告同时记录实际 Trainer 调度参数及解析后的步数。动态窗口与事件验证 CLI 共用参数解析、输入加载和输出逻辑；生产账本校验只有一份，验收工具另行重算事件触发和均值。

### 验收与后续范围

真实 CPU Trainer 在 Transformers **4.57.3、5.6.0、5.14.1** 三个版本验证评估调用、模型实际消费输入、checkpoint 文件、前继关系及 hook 恢复。7 updates 的保存步骤均为 `[3, 6, 7]`；5.x 额外包含最后一步补触发评估，独立的训练后评估仍保留。

本轮真实 CUDA 验收使用 **Blackwell GPU / Qwen3.5-0.8B text / LoRA / cap 256 / microbatch 3 / accumulation 2 / eval batch 2 / SDPA / seed 42 / 无 GC 或 packing**。同一份合成数据为 13 条训练记录、8 条评估记录；Factory 与 Axolotl 的实际模板不同。周期 checkpoint 保存 optimizer、scheduler、RNG 与 TrainerState，最终 adapter 保存独立统计，配置保留全部 checkpoint。

两个框架各执行一份 14-update epoch 调度 smoke，保存步骤为 `[3, 6, 9, 12, 14]`，验证不完整的最后 epoch。正式 steps 调度为每 4 步评估、每 6 步保存，各两份 20-update calibration、两份 29-update holdout。框架最后一步的补触发、Axolotl 的训练前评估、Factory 的训练后再次评估/保存均按实际调用核对。

共 **10 个独立 CUDA 进程**，无失败重跑或结果筛选；继承已有编译及文件缓存，没有受控冷启动的结论。每份正式 calibration 实际消费 34 microsteps / 90 条序列，holdout 为 49 microsteps / 129 条序列；每份 holdout 为 15 个根事件、6 类成本组，全部独立覆盖。下表报告两个 holdout 中的最大 APE：

| 框架 | 事件合计 | 评估子总时间 | 保存子总时间 | 单个事件组 |
|---|---:|---:|---:|---:|
| LLaMA-Factory 0.9.5 / TF 5.6.0 | 3.161% | 4.041% | 1.688% | 9.101% |
| Axolotl 0.18.0 / TF 5.14.1 | 0.858% | 1.691% | 1.314% | 3.106% |

Factory 最大单组误差出现在训练后的最终评估（实测 0.507 秒）；Ax 最大单组为最终 adapter 保存（实测 0.014 秒）。保留逐组结果，避免合计值抵消高估与低估。数字仅对应本次模型、数据、软件和 GPU 上的事件净时间，并非两个框架的同内核速度对比或准确率保证。

独立 CPU 审核从实际 Arrow 缓存重建所有评估输入、padding、labels 与 SHA；每次均消费 8 条评估记录，Factory 为 520 非 padding tokens / 296 有效 labels，Ax 为 776 / 288。46 份周期 checkpoint 与 10 份最终 adapter 均检查通过：每份 24 个张量、319,488 参数，所有值有限且 12 个 LoRA B 张量非零。周期 checkpoint 的文件大小、TrainerState step/epoch 以及 optimizer/scheduler/RNG 文件均匹配；最终 README/model card 和日志有 15 处在事件结束后被框架继续修改，归档明确保留快照与最终大小，不声称最终目录永久等于事件快照。

所有 21 个采集/验收源码文件在 10 次 GPU 运行至最终审计期间保持相同。原始配置、报告、日志、checkpoint 与安装包在 `/tmp/fakegpu-events-20261013/`；路径替换后的审阅归档为 [`finetune_event_validation_evidence.json`](tests/data/finetune_event_validation_evidence.json)，不能用作校准输入。

最终全仓回归 **1083 passed / 1 skipped，69.48 秒**；Ruff 与 `git diff --check` 通过。wheel 独立安装后的 **75 个 Python 模块**与源码逐字节一致，`python -S` 下四份新 CLI 预测与原始矩阵完全一致，旧 QLoRA 报告仍可推算 128 microsteps / 64 updates。另将 `f84d2615` 导出到独立目录比较：**44 份旧定长报告的 132 个投影（38 LoRA / 6 QLoRA）及 4 个动态窗口结果，完整字典和序列化字节均完全一致**。

整体 pipeline 估算仍不开放周期事件：事件本身的成本不能证明保存后的训练状态已恢复。下一步先按 `preceding_event_ids` 对保存/评估后第一个完整累积窗口做独立校准，再验证 sampler 与未来调度产生的次数和尾部，最后组合 setup、warmup、训练、事件和 finalization 的完整时间预算。packing 仍按下节已发现的隔离边界独立推进；本轮没有解除其 gate，也没有从事件耗时推导动态 VRAM。

## 2026-10-13 / 动态窗口独立校准验证

接续另一会话已提交的 `5c49621f`。新增 `estimate_finetuning_windows()` 与第 25 项 CLI `validate-finetune-windows`，将实际 batch 统计推进到独立进程的耗时预测验证。该入口比较已记录的 holdout trace；既有 `estimate_finetuning()` / `estimate-finetune --run-report` 的定长外推范围保持不变。

- 校准与 holdout 必须匹配模型、checkpoint 内容、GPU、软件、数据、原生配置和实际执行后端；对完整有序累积窗口的统计进行匹配。窗口描述保留逐行 token/labels 计数及尾窗口，不能用相同最大 shape 或全局 token 比例替代未覆盖窗口。
- 每份报告排除全局 warmup 与每类窗口的首次出现；每类待预测窗口在至少两个独立校准报告中分别具有至少两个剩余样本。先求各进程的窗口均值，再等权聚合；holdout 秒数仅计算误差，不参与拟合。报告保留逐 update 的预测、排除原因、逐组误差和总误差，缺少覆盖直接拒绝。
- 新 native run 记录 UUID、PID 与 UTC 开始时间；重复 UUID 或重复观测内容不能充当独立证据。矩阵交叉核对实际启动 PID 和时间。这些字段用于发现意外重复，不能认证人工编辑的 JSON。
- 将 native 配置及后端签名计算移为共用函数，并抽出定长/动态共有的报告校验。删除 normalizer 后重复的 method 集合检查；observer 复用 `describe_batch()` 验证，保留错误类型、失败时不消费 batch 和 hook 恢复。没有删除训练后实际模型/量化检查，它们用于发现框架改变请求设置的情况。
- 兼容性审核对 **44 份**历史原生报告逐项比较，外推结果与 `5c49621f` 完全一致（38 LoRA / 6 QLoRA）。41 份还通过新后端重建；早期缺少 kernel metadata 的报告保留旧签名校验。新版 batch 协议缺后端字段、配置被改而签名未更新等情况均拒绝。

经额外复现，原有定长 pipeline 估算会信任外层 `schedule`：若它被改为 `no`，即使实际 Trainer 参数保留周期 eval/save，仍可能错误开放整体时间外推。**`bb0a08a6` 单独修复**这一问题，交叉核对已记录的实际调度；四个回归覆盖 eval/save 的 steps/epoch，旧版缺可选参数的报告保持兼容。新窗口入口还明确拒绝周期事件，因为其对后续 allocator/cache 状态的影响尚未校准。

### 验收范围

实测为 **Blackwell GPU / Qwen3.5-0.8B text / LoRA / cap 512 / microbatch 2 / accumulation 2 / seed 42 / SDPA / 无 GC、packing 或周期 evaluation/save**。同一份 5 条合成 Alpaca 数据包含四条短记录与一条长记录。Factory 预处理后长度 58/178，实际 batch pad 到 64/184；Ax 为 90/210。每个 epoch 消费 batch 行数 `[2, 2, 1]`，累积窗口为 `[2, 1]`。

两个框架各两次独立 calibration（24 epochs / 72 microsteps / 48 updates / 120 条消费序列），各两次独立 holdout（40 epochs / 120 microsteps / 80 updates / 200 条消费序列）。另外各一份 primer 单独归档，不参与校准。共 **8 个正式 CUDA 进程 + 2 个 primer**，无失败或重跑选优。继承已有编译/文件缓存，未清空或控制全局缓存，因此不称为受控冷启动测试。

每份 calibration 的七类完整窗口，在剔除全局前两次 update 与每类首次出现后，样本数排序为 `[2, 2, 4, 5, 5, 6, 17]`；全部 holdout 有效窗口均覆盖。各 holdout 额外执行 epochs 25–40，其真实 sampler 描述流与 calibration 前缀不同；这不证明出现了新 shape 或独立新数据。表中逐组误差也保留，避免仅看聚合时间掩盖偏差。

| 框架 | holdout 窗口总时间最大 APE | epochs 25–40 最大 APE | 单个窗口组总时间最大 APE | 实测 allocated / reserved（GiB） |
|---|---:|---:|---:|---:|
| LLaMA-Factory 0.9.5 / TF 5.6.0 | 1.118% | 0.901% | 2.335% | 4.166 / 4.365 |
| Axolotl 0.18.0 / TF 5.14.1 | 1.610% | 0.425% | 4.270% | 3.310 / 3.686 |

allocator 峰值在各框架四次正式运行中相同，只作为观测记录，没有从窗口 timing 推导 VRAM。两框架的模板与 DeltaNet 内核不同，这不是同内核框架速度对比。原始配置、JSON、日志、adapter 与安装包保存在 `/tmp/fakegpu-dynamic-20261013/`；路径替换后的审阅归档为 [`finetune_window_validation_evidence.json`](tests/data/finetune_window_validation_evidence.json)，不能当作校准输入。

独立 CPU 审核直接读取实际 Arrow 缓存，重算有序行 SHA、逐 epoch tokens/labels/padding 以及原生随机采样顺序；10 份 run 全部匹配。10 份最终 adapter 均为 24 个张量、319,488 个参数，所有值有限，12 个 LoRA B 张量均非零；PID、UUID 和运行时间逐项交叉核对。

采集期间冻结的 17 个执行/验收文件全部匹配。采集后只收紧两个纯校验入口：窗口估算器拒绝周期事件，原生报告校验器核对实际调度。训练采集代码没有变化；最终重新校验全部原始报告，四份 holdout 的完整预测字典均与采集版逐字段相同。归档分别保留 `collection_source_file_sha256` 与 `validation_source_file_sha256`，不将旧哈希冒充最终源码。

最终完整回归 **999 passed / 1 skipped，64.28 秒**；Ruff 全仓与 `git diff --check` 通过。最终 wheel 独立安装后 **72 个 Python 模块**与源码逐字节一致；`python -S` 下两个框架的窗口 CLI 结果与矩阵完全一致，并可复用上一阶段 QLoRA 报告推算 **128 microsteps / 64 updates**。

独立动态验证只覆盖 holdout 中符合采样规则的 update 总耗时，不将其称为任意动态数据的训练总时间预测。它不推断未来 sampler 顺序，也不预测 warmup、周期事件、整体 wall time 或动态显存。相同统计不保证完整输入或 mask 布局相同；第一类窗口排除也不证明缓存状态已受控。首版限无 packing 的 dense 模型。

### packing 审计与下一步

当前本地框架版本存在需要先解决的语义边界，不能仅凭训练成功、有限 loss 或 position reset 开放 packing：

- Factory 普通 SFT packing 是 `greedy_knapsack`（`data/processor/supervised.py:186`），逐子样本重置 position IDs，但补齐位置的 attention mask 仍为 1。普通 packing 允许跨样本注意力；mask-active 槽位不能解释为原始非 padding 文本量，position reset 次数也不能当原样本数。可先为这个明确语义建立独立计数及验收。
- Factory neat packing 把 segment mask 转为 4D（`data/collator.py:524`）；当前 Transformers 5.6.0 的 Qwen3.5 线性层没有相应子序列卷积/递归状态重置路径。full-attention 的分块 mask 不足以证明整个 hybrid 模型隔离。
- Axolotl 的当前 export 模型类型是 `qwen3_5_text`，而 packing patch 选择器仅匹配 `qwen3_5`（`loaders/patch_manager.py:487`）；该 patch 的 `layer_type` 与当前 Transformers 5.14.1 的 `block_type` 还不兼容。SDPA packing 路径移除 attention mask，未为 linear/conv 提供所需边界参数。后续应先验证“只改变前一个 packed 样本，后一个样本 logits 不变”的隔离条件，再扩大支持。

周期事件成本的下一步先收集逐事件账本，而不是使用当前总秒数直接外推：记录真实入口、父子调用、step/epoch、训练内外边界、同步 wall time、eval 数据工作量与保存类型；嵌套 `_save_checkpoint → save_model → save_pretrained` 只累计外层区间，同 step 的独立调用不能去重。首次/重复评估、完整可恢复 checkpoint、最终 adapter 保存应分别校准。

当前两套 Transformers 在最后一步补触发 steps eval/save，比例步长按 `ceil(max_steps × ratio)` 解析，提前结束的部分 epoch 也触发 epoch-end；Factory 还可能在训练末尾额外 evaluate/save。Ax checkpoint 返回前 `gc.collect()` 和 `torch.cuda.empty_cache()` 会改变后续训练状态，必须单独观测保存后的第一个窗口。最小验收可采用 Qwen3.5 LoRA / 128 tokens：两个框架各两次 20-update 校准、一次 29-update holdout（eval 每 4 步、save 每 6 步），再用带尾 batch 的 epoch 调度验收末尾事件；核对真实 checkpoint 目录、TrainerState 与时间守恒。先交付评估/保存子总时间误差，再决定是否已足够支持完整 pipeline 预测。

以上为本机已安装版本的源码审计，路径相对各自框架包；没有修改框架原始仓库。下一阶段依次补原生 packing 语义、周期 evaluation/checkpoint 的事件账本和成本验证，再推进不同数据/模型/GPU 与原始 R24–R26。当前阶段仍不等同于完整 research toolkit 路线完成。

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
