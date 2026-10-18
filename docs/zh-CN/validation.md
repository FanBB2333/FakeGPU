# 验证与可靠性

[文档索引](README.md) · [English](../en/validation.md) ·
[繁體中文](../zh-TW/validation.md)

本页汇总 FakeGPU 显存估算背后的真实 GPU 证据，并说明每类工作负载的可靠性标签。
想自己生成新证据，请看[对照真实 GPU 校准](calibration.md)。

公开实验记录只写 GPU 架构。profile 目录里的型号表示兼容能力，不代表测试所用的
硬件。

<a id="memory-estimation-evidence"></a>

## 真实 GPU 显存估算验证

> [!NOTE]
> 在下面已记录的验证范围内，按对应软件栈校准后的静态估算器在 26 个受控 GPU
> 观测上的误差不超过 **0.08%**，在十个 Qwen 全量/LoRA SFT 用例上不超过
> **1.921%**。

误差按 `|预测值 - 实测值| / 实测值 × 100%` 计算，一致度为 `100% - 误差`。

| 已验证范围 | 真实 GPU 参考环境 | 证据规模 | 绝对百分比误差 | 一致度 |
|---|---|---:|---:|---:|
| [包含 backend 常驻显存校准的受控 ATen MLP 与 Transformer 参数网格][validation-static] | Ampere GPU 与 Blackwell GPU，PyTorch/CUDA 2.12/13.0 和 2.9/12.8 | 13 个工作负载，26 个观测 | **最大 0.08%** | **≥99.92%** |
| [Qwen3-8B BF16 SDPA 推理][validation-inference] | Blackwell GPU，PyTorch 2.9.1/CUDA 12.8 | 模型加载与推理峰值 | 加载 0.0129%，**峰值 0.0672%** | 99.9871%，**99.9328%** |
| [Qwen 0.8B/2B 全量与 LoRA SFT][validation-sft] | Blackwell GPU，PyTorch 2.8/CUDA 12.8 | 10 个训练用例 | **0.102%–1.921%** | **98.079%–99.898%** |
| [Qwen 0.8B/2B 原生 NF4 QLoRA][validation-qlora] | Blackwell GPU，PyTorch 2.8/CUDA 12.8 | 10 个量化训练用例 | **0.628%–1.732%** | **98.268%–99.372%** |

怎么看这些数字：

- Qwen 数据以 `torch.cuda.max_memory_allocated()` 为参考，不含 CUDA context，
  也不含 allocator 已预留但没用上的显存。
- ATen 一行加入了当前软件栈上实测的 backend 常驻显存。换了软件栈就不能沿用。
- 区间是各用例的最小和最大误差，不是平均值。评估 OOM 风险时要看低估。
- `99.x%` 一致度不等于显存还有余量。容量规划仍要留安全余量。

这些数字来自固定工作负载，不是通用准确率。模型、shape、backend、kernel、
allocator、软件版本或 GPU 一变，就需要重新校准。表中链接指向一个不可变快照，
里面保留了完整配置和实测字节数。CI 会检查本表与
[结构化证据摘要](../../tests/data/memory_validation_evidence.json)是否一致。

<a id="llm-reliability"></a>

## 研究场景可靠性

FakeGPU 按工作负载和环境签名报告可靠性：

- `GPU-validated`：只适用于记录中的模型 revision、shape、dtype、attention
  backend、allocator、软件栈和 GPU。
- `CPU-validated`：已维护的执行或分析行为在无 GPU 环境中通过了验证。
- `Modeled`：已有分析模型，但没有对应的真实 GPU 数据。
- `Planned`：还没有可以对外声明的准确率。

Serving 计划带有与路径无关的 `workload_signature`，校准数据不会被悄悄套用到另一种
serving 配置上。需要重复采集真实 CUDA 样本时，用 `calibrate observe-serving` 或
`calibrate collect-serving`。

### 仓库验证快照（2026-08-05）

在 macOS 26.5 arm64、Python 3.11.9 和 PyTorch 2.9.1 CPU 环境中执行了
`scripts/test.sh` 的全部测试组和两个验证 manifest。Transformers 采样器另外在
Blackwell GPU（PyTorch 2.11.0+cu128、Transformers 5.14.1）上跑过。

| 验证层 | 已维护的检查 | 结果 |
|---|---|---|
| Python runtime、估算器、CLI、schema 和文档契约 | 完整 `pytest` 测试集 | **223 个通过** |
| 声明式验证矩阵 | 7 个 smoke 用例，加 39 个 research 架构、cache、generic/vLLM serving、训练、校准和 diffusion 用例 | **46 个通过** |
| 原生库拦截 | 构建、库边界、导出符号、preload、显存类型、coordinator 和不支持 API 策略 | **通过** |
| FakeGPU-SMI 诊断 | 有上限的状态、拓扑/NVLink/MIG 视图、NVML peer/MIG 查询、健康字段和事件报告 | **通过** |
| 监控 exporter | Prometheus/JSON 快照、有上限的历史和序列数量、异常状态降级及 HTTP 接口 | **通过** |
| 原生能力清单（2026-10-18 更新） | 5 个能力组、106 个显式 API、93 个强制执行策略的 API | **通过** |
| GPU profile 目录 | 307 个 profile，覆盖 16 种 compute capability（目录更新于 2026-10-16） | **通过** |
| CPU 数值模拟 | GEMM、cuBLASLt、批量 GEMM、BLAS1/2 和 FP16，共 8 组测试 | **通过** |
| CUDA 版 PyTorch 原生矩阵乘法 | 需要带 CUDA 的 PyTorch | **当前 CPU-only 主机未执行** |
| Transformers 真实 CUDA serving 采样器 | Qwen/Qwen3-0.6B、BF16 SDPA、2 × 128-token prefill + 8 个生成 token，每阶段 3 个独立样本 | **通过，`ready_for_comparison`** |

GitHub CI 在 Python 3.10–3.12 上执行 Python 测试，并在 Linux 和 macOS 上执行
原生 smoke 与 CPU simulation。采样器这一行只检查重复执行和 CUDA 身份，跟准确率
无关。

### 已维护的研究工作负载矩阵

| 工作负载类型 | 已覆盖变化 | 验证依据 | 状态 |
|---|---|---|---|
| 离线 decoder 推理 | Qwen3-8B、BF16、SDPA、模型加载、prefill 和 decode 峰值 | Blackwell GPU 预测值与实测值 | `GPU-validated` |
| 全量与 adapter SFT | Qwen 0.8B/2B 全量微调和 LoRA | 十个 Blackwell GPU 训练用例 | `GPU-validated` |
| 量化 adapter SFT | Qwen 0.8B/2B 原生 NF4 QLoRA | 十个 Blackwell GPU 训练用例 | `GPU-validated` |
| 通用 decoder 分析 | Dense/MoE 元数据，MHA、GQA、MQA 和压缩 latent MLA，adapter、量化 checkpoint、eager/SDPA attention、KV cache 和 expert-parallel 通信量 | 公式、fixture、CLI 回归测试和四种架构矩阵 | `CPU-validated` + `Modeled` |
| 分布式训练规划 | DeepSpeed、Accelerate、FSDP/FSDP2、分片、checkpointing 和 CPU/NVMe offload | 配置、字节计算、拓扑和 trace 测试 | `CPU-validated` + `Modeled` |
| KV cache 分配 | Dynamic 增长、static 预留、2/4/8-bit quantized 存储、paged block 取整和 sliding-window 上限 | 公式、API、`--kv-cache-strategy` CLI 和 `tests/data/research_validation.yaml` 矩阵测试 | `CPU-validated` + `Modeled` |
| 在线 serving 容量 | 同构与混合长度请求集下的 continuous batching、顺序准入、chunked prefill、分组 prefix caching、paged KV block、generic 或 vLLM 运行时预算、profile 或显式 KV 容量，以及准入余量 | Checkpoint header、架构与 block 分配公式、`plan-serving --runtime vllm`、`--vllm-kv-cache-memory-bytes`、`--vllm-non-kv-cache-memory-bytes`、单元测试，以及 `tests/data/research_validation.yaml` 中的 generic/vLLM chat/RAG 用例 | `CPU-validated` + `Modeled` |
| Diffusion 图像生成 | Stable Diffusion v1.5、SDXL Base 和 PixArt-Sigma，本地 UNet、交叉注意力 DiT、SD3/Flux 类联合注意力 Transformer，CFG、offload、attention/VAE slicing 和 VAE tiling | 固定版本与本地组件 header、架构配置、阶段公式、CLI/单元测试，以及 `tests/data/research_validation.yaml` 中的三种本地架构矩阵 | `CPU-validated` + `Modeled` |
| Speculative decoding | 独立自回归 draft 模型的权重和 KV cache、target lookahead KV、批量 verification 临时显存、可选的固定接受率假设，以及同构/混合请求准入 | Checkpoint header、架构公式、`plan-serving --draft-model-dir`、`--speculative-tokens`、`--speculative-acceptance-rate` 和单元测试。还没有项目维护的真实 GPU 观测 | `CPU-validated` + `Modeled` |
| 多 GPU LLM 执行 | TP、PP、CP、EP、MoE 负载不均衡，以及 FSDP/ZeRO 组合执行 | 目前只有分析拓扑和 coordinator 验证 | `Modeled` |
| Diffusion 训练 | 缓存特征 UNet full/LoRA、FP32 AdamW/EMA 存储、BF16 autocast 与原生 activation/allocator 包络 | 12 个 SD/SDXL 原生进程、独立 holdout 与重载检查 | 已记录场景为 `GPU-validated` |

<a id="research-scenario-effects"></a>

### 架构感知估算对比

下表由仓库中的公式生成，并由文档契约测试检查。这些是公式结果，不是真实 GPU
观测，也不能替代同配置校准。GiB 使用二进制单位。

| Research 场景 | 对照条件 | 建模结果 | 验证方式 |
|---|---|---:|---|
| LLM 推理 KV cache | 32 层 GQA、8 个 KV head、head dim 128、batch 1、BF16，context 4K → 32K | **0.50 → 4.00 GiB**（cache 增长 8 倍） | 精确字节公式 |
| Attention cache 架构 | 32 层、8 个活跃请求、4K BF16 paged cache，标准 head dim 128，MHA 32 个 KV head / GQA 8 个 / MQA 1 个 / MLA latent 512 + RoPE 64 | **16.00 / 4.00 / 0.50 / 1.125 GiB** | 与架构对应的 cache layout 公式和四种 manifest 用例 |
| 在线 serving prefix cache | 同一 decoder、8 个活跃请求、4K prompt + 256 个生成 token、paged BF16 KV，独立 cache → 共享一个 1K prefix | **4.25 → 3.38 GiB**（减少 20.6%） | 共享/私有 paged block 公式 |
| 混合 chat/completion serving | 同一 decoder，prompt 为 4K/8K/2K，生成长度为 256/512/1024，独立 KV → 两个 chat 请求共享 1K system prefix | **1.97 → 1.84 GiB** decode KV（减少 6.3%） | 异构请求集公式和 chat/RAG 矩阵 |
| Speculative target 调用 | 每次 verification 生成 5 个 draft token，假设每个 token 独立接受率为 70% | 每个 target step 的预期输出 token 为 **1.00 → 2.94**，不代表吞吐提升 | 几何接受率公式和双模型显存测试 |
| LLM 训练 | 8B BF16 参数、AdamW、4 张 GPU、activation checkpointing，replicated → full shard | **104.31 → 26.54 GiB**/rank（减少 74.6%） | 训练计划阶段模型 |
| Stable Diffusion v1.5 生成 | 512²、batch 1、FP16、CFG，全部权重常驻 → model offload | **2.61 → 1.72 GiB**（减少 34.3%） | 组件与阶段模型 |
| SDXL 批量生成 | 1024²、batch 4、FP16、CFG，普通 VAE decode → VAE slicing | **26.64 → 11.64 GiB**（减少 56.3%） | 顺序 VAE batch shape 模型 |
| SDXL 高分辨率生成 | 2048²、batch 1、FP16、CFG，整图 VAE → 512² VAE tile | **26.64 → 8.28 GiB**（减少 68.9%） | 顺序 VAE tile shape 模型 |
| PixArt-Sigma DiT 生成 | 1024²、batch 1、FP16、CFG、model offload，denoise 阶段 eager → SDPA | **2.32 → 1.28 GiB**（减少 44.7%） | patch token、交叉注意力与阶段模型 |

- Diffusion profile 的组件参数量来自 [Stable Diffusion v1.5][diffusion-sd15]、
  [SDXL Base 1.0][diffusion-sdxl] 和 [PixArt-Sigma XL 2][diffusion-pixart] 的固定
  版本。runtime context、碎片、私有 workspace 和传输重叠不计入。
- 拿到匹配的观测之前，报告保持 `Modeled`，`accuracy.status` 为 `uncalibrated`。
  本地架构检查通过 `estimate-diffusion --model-dir` 使用。
- Cache 公式参考
  [Transformers cache strategies](https://huggingface.co/docs/transformers/kv_cache)。
  Speculative 模式按照
  [原始 speculative decoding 论文](https://arxiv.org/abs/2211.17192)和
  [Transformers assisted decoding](https://huggingface.co/docs/transformers/assisted_decoding)
  描述的 draft/target 流程建模。Serving 术语参考
  [vLLM serving](https://docs.vllm.ai/en/stable/)。
- 请求到达分布、cache eviction、preemption、自适应 draft 长度、KV 回退 workspace、
  吞吐量和延迟不在估算范围内。二进制 CUDA 扩展和任意 kernel 需要通过 passthrough
  或 hybrid 模式拿到真实 GPU 观测。

### 发布新的验证数据

新增或更新的公开验证数据应包含：

- 至少五次隔离执行，以及其中最大的观测峰值
- 每个报告阶段的预测字节数和实测字节数
- 最大低估，这是主要的 OOM 风险指标。标记为 `GPU-validated` 时不超过 5%
- 绝对百分比误差的中位数、95 分位数和最大值
- 预测区间覆盖率，以及 false-safe 次数（预测能放下，实际 OOM）
- 模型 revision、命令、shape、dtype、backend、allocator 设置、GPU、driver、
  CUDA、PyTorch 和框架版本

公开结果前用 `calibrate verify` 执行这些门槛检查。未达标的数据继续标记为
`Modeled` 或 experimental。

[validation-static]: https://github.com/FanBB2333/FakeGPU/blob/cc7f221356b582e3e06618f226d9e05bd7da64ef/docs/ai-researcher-preflight.md#4-static-aten-storage-liveness-validation
[validation-inference]: https://github.com/FanBB2333/FakeGPU/blob/cc7f221356b582e3e06618f226d9e05bd7da64ef/docs/llm-inference-estimation.md#maintained-qwen3-8b-result
[validation-sft]: https://github.com/FanBB2333/FakeGPU/blob/cc7f221356b582e3e06618f226d9e05bd7da64ef/docs/llm-sft-memory-estimation.md#blackwell-gpu-matrix
[validation-qlora]: https://github.com/FanBB2333/FakeGPU/blob/cc7f221356b582e3e06618f226d9e05bd7da64ef/docs/llm-sft-memory-estimation.md#native-nf4-qlora-matrices
[diffusion-sd15]: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/451f4fe16113bff5a5d2269ed5ad43b0592e9a14
[diffusion-sdxl]: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/462165984030d82259a11f4367a4eed129e94a7b
[diffusion-pixart]: https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS/tree/e102b3591cc82e97071b8b4cb90d834d0c487207
