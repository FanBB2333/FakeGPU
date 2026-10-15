<a id="readme-top"></a>

<div align="center">

# FakeGPU

**无需生产级 GPU 集群，即可验证面向 CUDA 的应用、估算 GPU 显存并模拟分布式 GPU 工作流。**

[![测试][test-shield]][test-url]
[![版本][release-shield]][release-url]
[![Python][python-shield]][python-url]
[![许可证][license-shield]][license-url]

[English](README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md)

[报告问题](https://github.com/FanBB2333/FakeGPU/issues/new?labels=bug) ·
[提出功能建议](https://github.com/FanBB2333/FakeGPU/issues/new?labels=enhancement)

</div>

> [!IMPORTANT]
> FakeGPU 用于开发、兼容性测试和容量规划，不能让任意 CUDA kernel 获得与真实
> GPU 相同的数值结果或性能。Passthrough、hybrid 和校准流程仍需要真实的
> CUDA 环境。

## 目录

1. [项目介绍](#项目介绍)
   - [FakeGPU 能回答哪些问题](#fakegpu-能回答哪些问题)
   - [典型使用场景](#use-cases)
   - [真实 GPU 显存估算验证](#memory-estimation-evidence)
   - [研究场景可靠性](#llm-reliability)
   - [工作方式](#工作方式)
   - [主要技术](#主要技术)
2. [快速开始](#快速开始)
   - [环境要求](#环境要求)
   - [安装](#安装)
   - [验证安装结果](#验证安装结果)
3. [使用方式](#使用方式)
   - [使用 FakeCUDA 运行 PyTorch](#使用-fakecuda-运行-pytorch)
   - [查看 FakeGPU 设备和进程](#查看-fakegpu-设备和进程)
   - [导出监控指标](#export-monitoring-metrics)
   - [拦截原生 CUDA 库](#拦截原生-cuda-库)
   - [运行前检查显存](#运行前检查显存)
   - [规划在线 LLM 服务容量](#plan-online-llm-serving-capacity)
   - [分析仓库或模型](#分析仓库或模型)
4. [命令参考](#命令参考)
   - [微调总时间与显存预估](#微调总时间与显存预估)
   - [在线 serving 基准测试](#在线-serving-基准测试)
   - [Diffusion 训练](#diffusion-训练)
   - [阶段显存校准](#阶段显存校准)
   - [分布式规划与实验批量运行](#分布式规划与实验批量运行)
5. [GPU profiles](#gpu-profiles)
6. [开发](#开发)
   - [构建](#构建)
   - [测试](#测试)
   - [可复用脚本](#可复用脚本)
7. [项目结构](#项目结构)
8. [限制](#限制)
9. [开发计划](#开发计划)
10. [参与贡献](#参与贡献)
11. [许可证](#许可证)
12. [致谢](#致谢)

## 项目介绍

FakeGPU 为开发、CI、兼容性检查和容量规划模拟面向 CUDA 的运行环境：应用可以
发现可配置的 NVIDIA 风格设备，已维护的运算在 CPU 上执行，模拟显存和通信会被
记录。对于不应直接加载的工作负载，FakeGPU 还提供静态估算工具。

只有 passthrough、hybrid 和校准流程需要物理 GPU。

示例中的 `TARGET_GPU_PROFILE` 由用户通过
`python3 -m fakegpu doctor --list-profiles` 选择后设置。公开实验资料只披露 GPU
架构；支持目录中的型号表示兼容能力，不表示维护者持有该设备。

### FakeGPU 能回答哪些问题

| 问题 | 建议入口 | 需要物理 GPU |
|---|---|---:|
| PyTorch 代码能否按预期执行面向 CUDA 的控制流程？ | Python FakeCUDA runtime | 否 |
| 未修改的进程能否加载并调用 CUDA 系列动态库？ | 原生库拦截 | 否 |
| 某个工作负载能否放入选定的 GPU profile？ | Preflight 或静态显存估算器 | 否 |
| LLM 的 checkpoint、KV cache、adapter 或 MoE 需要多少显存？ | LLM 估算器 | 否 |
| 给定显存预算可以容纳哪些同构或混合长度的在线 LLM 请求？ | Serving 规划器 | 否 |
| Speculative decoding 的 target 和 draft 模型能否同时放入显存？ | Serving 规划器 | 否 |
| 分辨率、batch、CFG、VAE tiling 和 offload 会如何影响 diffusion 生成显存？ | Diffusion 估算器 | 否 |
| 仓库中有哪些仅支持 GPU 的入口和依赖？ | 仓库分析器 | 否 |
| 分布式训练配置对应多少单 rank 显存？ | 训练规划器 | 否 |
| Trace 中的计算、通信、等待和显存如何重叠？ | Trace 回放 | 否 |
| 估算结果与真实 CUDA 运行结果相差多少？ | Passthrough 或 hybrid 校准 | 是 |

<a id="use-cases"></a>

### 典型使用场景

| 适用场景 | FakeGPU 提供的能力 | 建议入口 |
|---|---|---|
| 在租用 GPU 或启动长时间任务前选择硬件 | 按 profile 估算 checkpoint、KV cache、activation、optimizer 和 workspace 显存 | `estimate-llm`、`preflight` |
| 部署 chat、RAG、completion、summarization 或 draft-assisted 服务前规划容量 | 按请求估算 prompt/生成长度、continuous batch 准入、chunked prefill 临时显存、共享 prefix 分组、speculative target/draft 显存和可用显存余量 | `plan-serving`、`validate` |
| 在笔记本电脑或无 GPU 的 CI 中开发面向 CUDA 的 PyTorch 代码 | 让程序看到 CUDA 设备，同时在 CPU 上执行已维护的 tensor 运算 | `fakegpu.init(...)`、`demo`、`validate` |
| 比较全量微调、LoRA、QLoRA、checkpointing、offload 或分片方案 | 在申请集群前估算各阶段和单 rank 显存 | `plan-training`、Python 显存估算器 |
| 比较 UNet 与 diffusion Transformer 的生成 shape 和显存优化选项 | 根据固定 profile 或本地 pipeline，按架构估算 text encoder、denoising 和 VAE decode 阶段 | `estimate-diffusion`、`validate` |
| 检查不熟悉的 GPU 仓库或原生扩展 | 统计 GPU 入口、依赖、kernel 和不支持的 API | `analyze-repo`、`analyze-kernel`、`capabilities` |
| 设计或排查分布式工作流 | 分析 collective 路由、链路竞争、rank 等待、显存时间线和 TCP payload | `simulate-topology`、`replay-trace`、`bandwidth` |
| 在 CI 或本地实验环境中观察模拟设备和进程 | 提供有数量上限的 Prometheus 指标、exporter 健康状态和短期内存历史 | `nvidia-smi`、`metrics` |
| 将小规模真实 GPU 试验用于后续重复任务 | 生成预测值与实测值的对比报告，并按工作负载签名保存校准数据 | `calibrate`、`preflight --memory-calibration` |

<a id="memory-estimation-evidence"></a>

### 真实 GPU 显存估算验证

> [!NOTE]
> 在以下已记录的验证范围内，使用对应软件栈校准后的静态估算器在 26 个受控
> GPU 观测上的误差不超过 **0.08%**；十个 Qwen 全量/LoRA SFT 用例的误差
> 不超过 **1.921%**。

误差按 `|预测值 - 实测值| / 实测值 × 100%` 计算，一致度为 `100% - 误差`。

| 已验证范围 | 真实 GPU 参考环境 | 证据规模 | 绝对百分比误差 | 一致度 |
|---|---|---:|---:|---:|
| [包含 backend 常驻显存校准的受控 ATen MLP 与 Transformer 参数网格][validation-static] | Ampere GPU 与 Blackwell GPU；PyTorch/CUDA 2.12/13.0 和 2.9/12.8 | 13 个工作负载，26 个观测 | **最大 0.08%** | **≥99.92%** |
| [Qwen3-8B BF16 SDPA 推理][validation-inference] | Blackwell GPU；PyTorch 2.9.1/CUDA 12.8 | 模型加载与推理峰值 | 加载 0.0129%；**峰值 0.0672%** | 99.9871%；**99.9328%** |
| [Qwen 0.8B/2B 全量与 LoRA SFT][validation-sft] | Blackwell GPU；PyTorch 2.8/CUDA 12.8 | 10 个训练用例 | **0.102%–1.921%** | **98.079%–99.898%** |
| [Qwen 0.8B/2B 原生 NF4 QLoRA][validation-qlora] | Blackwell GPU；PyTorch 2.8/CUDA 12.8 | 10 个量化训练用例 | **0.628%–1.732%** | **98.268%–99.372%** |

这些数字需要结合测量方式理解：

- Qwen 数据以 `torch.cuda.max_memory_allocated()` 为参考，不含 CUDA context
  和 allocator 已预留但未使用的显存。
- 受控 ATen 数据加入了当前软件栈的 backend 常驻显存测量值，不能用于其他环境。
- 区间是用例中的最小/最大误差，不是平均值；评估 OOM 风险应重点看最大低估。
- `99.x%` 一致度不等于可用显存余量，容量规划仍需加安全余量。

这些数字来自固定工作负载，不是通用准确率；模型、shape、backend、kernel、
allocator、软件版本或 GPU 变化时需要重新校准。表中链接指向保留完整配置和实测
字节数的不可变快照；CI 会检查本表与
[结构化证据摘要](https://github.com/FanBB2333/FakeGPU/blob/main/tests/data/memory_validation_evidence.json)
是否一致。

在真实 CUDA 主机上，可以重新执行项目维护的受控对比：

```bash
python3 scripts/validation/static_memory_validation.py \
  --output build/static-memory-validation.json \
  --markdown build/static-memory-validation.md \
  --max-underestimate-percent 5
```

无 GPU 主机加 `--static-only` 只检查估算流程，不产生准确性结果。将
`plan-serving` 报告与多次真实 GPU 阶段峰值对比：

```bash
python3 -m fakegpu calibrate observe-serving \
  build/speculative-serving-plan.json \
  --prefill-peak-bytes 24800000000,24900000000,25000000000 \
  --prefill-peak-bytes 24950000000,25050000000 \
  --decode-peak-bytes 25200000000,25300000000,25250000000 \
  --decode-peak-bytes 25350000000,25400000000 \
  --source torch.cuda.max_memory_reserved \
  --strict \
  --json build/serving-observation.json
```

命令保留全部样本，每个阶段取最大值。默认每阶段至少五次观测（不足时 `--strict`
返回 1）；`ready_for_comparison` 只表示样本数足够，不代表 `GPU-validated`。
Plan 必须包含 `--target-profile`。

用框架对应的 runner 自动执行多次测量：

```bash
python3 -m fakegpu calibrate collect-serving \
  build/speculative-serving-plan.json \
  --repetitions 5 \
  --timeout-seconds 900 \
  --strict \
  --json build/serving-observation.json \
  -- python3 benchmark_serving.py
```

同构、非 speculative 的 Transformers 工作负载可使用内置 runner，plan 须使用
dynamic KV cache：

```bash
python3 -m fakegpu plan-serving \
  --model-dir /mnt/z/models/Qwen/Qwen3-0.6B \
  --active-sequences 4 \
  --max-batch-size 8 \
  --prompt-tokens 2048 \
  --generated-tokens 128 \
  --dtype bfloat16 \
  --attention-implementation sdpa \
  --kv-cache-strategy dynamic \
  --target-profile a100 \
  --json build/transformers-serving-plan.json

python3 -m fakegpu calibrate collect-serving \
  build/transformers-serving-plan.json \
  --repetitions 5 \
  --timeout-seconds 900 \
  --strict \
  --json build/transformers-serving-observation.json \
  -- python3 -m fakegpu calibrate sample-transformers
```

主机没有 `/mnt/z` 时只替换 models 根目录，保留 `Qwen/Qwen3-0.6B` 结构。
`sample-transformers`：

- 重新核对本地配置、Safetensors header 与 plan，再测量完整 prefill 和逐 token
  decode；
- 默认指标为 `torch.cuda.max_memory_reserved`（`--metric allocated` 取张量分配峰值）；
- 不接受混合请求、speculative decoding、paged/quantized cache、共享前缀或
  chunked prefill plan、量化 checkpoint 和 adapter；
- 需要在 CUDA 环境中预先安装 PyTorch 与 Transformers。

vLLM 或自定义测量程序已有可信阶段峰值时，可直接构建样本：

```bash
python3 -m fakegpu calibrate emit-serving-sample \
  build/vllm-serving-plan.json \
  --prefill-peak-bytes 24900000000 \
  --decode-peak-bytes 25300000000 \
  --metric nvml.process_family_peak_bytes \
  --framework vllm \
  --framework-version 0.10.2
```

Python 中对应 `fakegpu.build_cuda_serving_sample(...)`。vLLM 的 cache usage 指标
是 block 使用比例而非字节峰值，因此 benchmark 需提供真实的阶段字节数。

每次重复都会启动新进程，runner 需输出一个 `fakegpu.serving_peak_sample.v1`
对象：放在最后一行，或以 `FAKEGPU_SERVING_SAMPLE=` 作为单行前缀：

```json
{
  "schema_version": "fakegpu.serving_peak_sample.v1",
  "workload_signature": "<FAKEGPU_SERVING_WORKLOAD_SIGNATURE>",
  "run_index": 1,
  "metric": "torch.cuda.max_memory_reserved",
  "phases": {
    "prefill": {"peak_bytes": 24900000000},
    "decode": {"peak_bytes": 25300000000}
  },
  "environment": {
    "backend": "cuda",
    "simulated": false,
    "gpu_name": "NVIDIA A100-SXM4-80GB",
    "gpu_uuid": "GPU-...",
    "compute_capability": [8, 0],
    "total_memory_bytes": 85899345920,
    "software": {
      "framework": "transformers",
      "framework_version": "...",
      "cuda_version": "...",
      "torch_version": "..."
    }
  }
}
```

- FakeGPU 通过 `FAKEGPU_SERVING_*` 环境变量提供 plan 路径、工作负载签名、目标
  profile、compute capability、当前次数和总次数；内置 runner 与协议构建器会自动
  生成该对象，无需将这些框架加入 FakeGPU 依赖。
- Runner 报告模拟 CUDA、容量超出 profile 2% 容差、compute capability 不匹配、
  软件环境变化、超时或非零退出时，采集失败。
- 报告不保存原始命令参数，只记录可执行文件名和命令指纹。
- `simulated: false` 是 runner 提供的元数据，不是硬件证明；声明 `GPU-validated`
  仍需保留 benchmark 日志并通过校准门槛。

对任意兼容的预测报告和观测报告执行对比与门槛检查：

```bash
python3 -m fakegpu calibrate compare \
  build/prediction.json \
  build/observation.json \
  --json build/calibration-comparison.json

python3 -m fakegpu calibrate verify \
  build/calibration-comparison.json \
  --max-underestimate-percent 5 \
  --max-absolute-percentage-error-percent 5 \
  --min-interval-coverage-percent 90 \
  --capacity-bytes 25769803776 \
  --json build/calibration-verification.json
```

`compare` 报告各阶段有符号/绝对误差、区间覆盖率和建议安全余量。`verify` 在任一
门槛未通过时返回 1：最大低估、误差中位数/95 分位数/最大值、区间覆盖率、指定
容量下的 false-safe 判断或工作负载一致性。结果只适用于相同的工作负载签名、
shape、dtype、软件栈和 GPU profile。

<a id="llm-reliability"></a>

### 研究场景可靠性报告

FakeGPU 按工作负载和环境签名报告可靠性：

- `GPU-validated`：只适用于记录中的模型 revision、shape、dtype、attention
  backend、allocator、软件栈和 GPU。
- `CPU-validated`：已维护的执行或分析行为在无 GPU 环境中通过验证。
- `Modeled`：已有分析模型，但没有对应的真实 GPU 数据。
- `Planned`：尚未形成可对外声明的准确率。

Serving 规划报告提供与路径无关的 `workload_signature`，避免校准数据在不同服务
配置之间误用。需要重复采集真实 CUDA 数据时，使用 `calibrate observe-serving`
或 `calibrate collect-serving`。

#### 仓库验证快照（2026-08-05）

在 macOS 26.5 arm64、Python 3.11.9 和 PyTorch 2.9.1 CPU 环境中执行了
`scripts/test.sh` 全部测试组和两个声明式验证 manifest；Transformers 采样器另在
Blackwell GPU（PyTorch 2.11.0+cu128、Transformers 5.14.1）上完成真实
CUDA 验证：

| 验证层 | 已维护的检查 | 结果 |
|---|---|---|
| Python runtime、估算器、CLI、schema 和 README 契约 | 完整 `pytest` 测试集 | **223 个通过** |
| 声明式验证矩阵 | 7 个 smoke 用例，加 39 个 research 架构、cache、generic/vLLM serving、训练、校准和 diffusion 用例 | **46 个通过** |
| 原生库拦截 | 构建、库边界、导出符号、preload、显存类型、coordinator 和不支持 API 策略 | **通过** |
| FakeGPU-SMI 诊断 | 有上限的状态发布、拓扑/NVLink/MIG 视图、NVML peer/MIG 查询、健康字段和事件报告 | **通过** |
| 监控 exporter | Prometheus/JSON 快照、历史和序列数量限制、异常状态降级及 HTTP 接口 | **通过** |
| 原生能力清单（2026-09-05 更新） | 5 个能力组、104 个显式 API、102 个强制执行策略的 API | **通过** |
| GPU profile 目录 | 82 个 profile，覆盖 15 种 compute capability | **通过** |
| CPU 数值模拟 | GEMM、cuBLASLt、批量 GEMM、BLAS1/2 和 FP16，共 8 组测试 | **通过** |
| CUDA 版 PyTorch 原生矩阵乘法 | 需要带 CUDA 的 PyTorch | **当前 CPU-only 主机未执行** |
| Transformers 真实 CUDA serving 采样器 | Qwen/Qwen3-0.6B、BF16 SDPA、2 × 128-token prefill + 8 个生成 token，每阶段 3 个独立样本 | **通过；`ready_for_comparison`** |

GitHub CI 在 Python 3.10–3.12 上执行 Python 测试，并在 Linux 和 macOS 上执行
原生 smoke 与 CPU simulation。采样器一行只验证了重复执行和 CUDA 身份，不构成
准确率声明。

#### 已维护的研究工作负载矩阵

| 工作负载类型 | 已覆盖变化 | 验证依据 | 状态 |
|---|---|---|---|
| 离线 decoder 推理 | Qwen3-8B、BF16、SDPA、模型加载、prefill 和 decode 峰值 | Blackwell GPU 预测值与实测值 | `GPU-validated` |
| 全量与 adapter SFT | Qwen 0.8B/2B 全量微调和 LoRA | 十个 Blackwell GPU 训练用例 | `GPU-validated` |
| 量化 adapter SFT | Qwen 0.8B/2B 原生 NF4 QLoRA | 十个 Blackwell GPU 训练用例 | `GPU-validated` |
| 通用 decoder 分析 | Dense/MoE 元数据；MHA、GQA、MQA 和压缩 latent MLA；adapter、量化 checkpoint、eager/SDPA attention、KV cache 和 expert-parallel 通信量 | 公式、fixture、CLI 回归测试和四种架构矩阵 | `CPU-validated` + `Modeled` |
| 分布式训练规划 | DeepSpeed、Accelerate、FSDP/FSDP2、分片、checkpointing 和 CPU/NVMe offload | 配置、字节计算、拓扑和 trace 测试 | `CPU-validated` + `Modeled` |
| KV cache 分配 | Dynamic 增长、static 预留、2/4/8-bit quantized 存储、paged block 取整和 sliding-window 上限 | 公式、API、`--kv-cache-strategy` CLI 和 `tests/data/research_validation.yaml` 矩阵测试 | `CPU-validated` + `Modeled` |
| 在线 serving 容量 | 面向同构与混合长度请求集的 continuous batching、顺序准入、chunked prefill、分组 prefix caching、paged KV block、generic 或 vLLM 运行时预算、profile 或显式 KV 容量，以及准入余量 | Checkpoint header、架构与 block 分配公式、`plan-serving --runtime vllm`、`--vllm-kv-cache-memory-bytes`、`--vllm-non-kv-cache-memory-bytes`、单元测试，以及 `tests/data/research_validation.yaml` 中的 generic/vLLM chat/RAG 用例 | `CPU-validated` + `Modeled` |
| Diffusion 图像生成 | Stable Diffusion v1.5、SDXL Base 和 PixArt-Sigma；本地 UNet、交叉注意力 DiT、SD3/Flux 类联合注意力 Transformer；CFG、offload、attention/VAE slicing 和 VAE tiling | 固定版本与本地组件 header、架构配置、阶段公式、CLI/单元测试和 `tests/data/research_validation.yaml` 中的三种本地架构矩阵 | `CPU-validated` + `Modeled` |
| Speculative decoding | 独立自回归 draft 模型权重和 KV cache、target lookahead KV、批量 verification 临时显存、可选的固定接受率假设，以及同构/混合请求准入 | Checkpoint header、架构公式、`plan-serving --draft-model-dir`、`--speculative-tokens`、`--speculative-acceptance-rate` 和单元测试；尚无项目维护的真实 GPU 观测 | `CPU-validated` + `Modeled` |
| 多 GPU LLM 执行 | TP、PP、CP、EP、MoE 负载不均衡，以及 FSDP/ZeRO 组合执行 | 目前只有分析拓扑和 coordinator 验证 | `Modeled` |
| Diffusion 训练 | 缓存特征 U-Net full/LoRA、FP32 AdamW/EMA、BF16 autocast 与实测 activation/allocator 包络 | 12 个 SD/SDXL 原生进程、独立 holdout 与重载检查 | 已记录场景为 `GPU-validated` |

<a id="research-scenario-effects"></a>

#### 架构感知估算对比

下表由仓库中的公式生成，并由 README 契约测试检查。它们是公式结果，不是真实 GPU
观测，不能替代同配置校准；GiB 使用二进制单位。

| Research 场景 | 对照条件 | 建模结果 | 验证方式 |
|---|---|---:|---|
| LLM 推理 KV cache | 32 层 GQA、8 个 KV head、head dim 128、batch 1、BF16；context 4K → 32K | **0.50 → 4.00 GiB**（cache 增长 8 倍） | 精确字节公式 |
| Attention cache 架构 | 32 层、8 个活跃请求、4K BF16 paged cache；标准 head dim 128；MHA 32 个 KV head / GQA 8 个 / MQA 1 个 / MLA latent 512 + RoPE 64 | **16.00 / 4.00 / 0.50 / 1.125 GiB** | 与架构对应的 cache layout 公式和四种 manifest 用例 |
| 在线 serving prefix cache | 同一 decoder、8 个活跃请求、4K prompt + 256 个生成 token、paged BF16 KV；独立 cache → 共享 1K prefix | **4.25 → 3.38 GiB**（减少 20.6%） | 共享/私有 paged block 公式 |
| 混合 chat/completion serving | 同一 decoder；prompt 为 4K/8K/2K，生成长度为 256/512/1024；独立 KV → 两个 chat 请求共享 1K system prefix | **1.97 → 1.84 GiB** decode KV（减少 6.3%） | 异构请求集公式和 chat/RAG 矩阵 |
| Speculative target 调用 | 每次 verification 生成 5 个 draft token，并假设单 token 独立接受率为 70% | 每个 target step 的预期输出 token 为 **1.00 → 2.94**；该数值不表示吞吐提升 | 几何接受率公式和双模型显存测试 |
| LLM 训练 | 8B BF16 参数、AdamW、4 张 GPU、activation checkpointing；replicated → full shard | **104.31 → 26.54 GiB**/rank（减少 74.6%） | 训练阶段模型 |
| Stable Diffusion v1.5 生成 | 512²、batch 1、FP16、CFG；全部权重常驻 → model offload | **2.61 → 1.72 GiB**（减少 34.3%） | 组件与阶段模型 |
| SDXL 批量生成 | 1024²、batch 4、FP16、CFG；普通 VAE decode → VAE slicing | **26.64 → 11.64 GiB**（减少 56.3%） | 顺序 VAE batch shape 模型 |
| SDXL 高分辨率生成 | 2048²、batch 1、FP16、CFG；整图 VAE → 512² VAE tile | **26.64 → 8.28 GiB**（减少 68.9%） | 顺序 VAE tile shape 模型 |
| PixArt-Sigma DiT 生成 | 1024²、batch 1、FP16、CFG、model offload；eager → SDPA 的 denoise 阶段 | **2.32 → 1.28 GiB**（减少 44.7%） | patch token、交叉注意力与阶段模型 |

- Diffusion profile 的组件参数量来自 [Stable Diffusion v1.5][diffusion-sd15]、
  [SDXL Base 1.0][diffusion-sdxl] 和 [PixArt-Sigma XL 2][diffusion-pixart] 的固定
  版本，不含 runtime context、allocator 碎片、backend 私有 workspace 和传输重叠。
- 取得匹配观测前，报告保持 `Modeled`，`accuracy.status` 为 `uncalibrated`。
  本地架构检查通过 `estimate-diffusion --model-dir` 使用。
- Cache 公式参考
  [Transformers cache strategies](https://huggingface.co/docs/transformers/kv_cache)；
  Speculative 模式按照
  [原始 speculative decoding 论文](https://arxiv.org/abs/2211.17192)和
  [Transformers assisted decoding](https://huggingface.co/docs/transformers/assisted_decoding)
  描述的 draft/target 流程建模；术语参考 [vLLM serving](https://docs.vllm.ai/en/stable/)。
- 请求到达分布、cache eviction、preemption、自适应 draft 长度、KV 回退 workspace、
  吞吐量和延迟不在估算范围内。二进制 CUDA 扩展和任意 kernel 需要通过 passthrough
  或 hybrid 模式取得真实 GPU 观测。

后续新增或更新的公开验证数据应包含：

- 至少五次隔离执行，以及其中最大的观测峰值；
- 每个报告阶段的预测字节数和实测字节数；
- 将最大低估作为主要 OOM 风险指标；标记为 `GPU-validated` 时不超过 5%；
- 绝对百分比误差的中位数、95 分位数和最大值；
- 预测区间覆盖率，以及 false-safe 次数（判断可运行但实际 OOM）；
- 模型 revision、完整命令、shape、dtype、backend、allocator 设置、GPU、
  driver、CUDA、PyTorch 和框架版本。

公开结果前使用 `calibrate verify` 执行这些门槛检查；未达标的数据继续标记为
`Modeled` 或 experimental。

### 工作方式

| 路径 | 应用看到的内容 | 实际执行方式 |
|---|---|---|
| **Python FakeCUDA** | CUDA 设备、CUDA 风格 tensor、显存 API 和常见训练流程 | 已维护的 PyTorch 运算通过 `FakeCudaTensor` 在 CPU 上执行 |
| **原生库拦截** | `libcuda`、`libcudart`、`libcublas`、`libnvidia-ml` 和 `libnccl` 入口 | 选定的运算使用主机内存或 CPU 计算；不支持的行为会被分类并写入报告 |
| **分析与报告** | 显存、FLOP、Roofline、拓扑和通信报告 | 分析 ATen 图、safetensors 元数据、运行 trace、校准数据和 coordinator 事件 |

### 主要技术

- [Python](https://www.python.org/) 3.10+：runtime、估算器、CLI 和报告
- C++17 与 [CMake](https://cmake.org/)：原生拦截库和 coordinator
- [PyTorch](https://pytorch.org/)：CPU FakeCUDA 执行和 ATen 图捕获
- YAML 与 JSON Schema：GPU profiles、验证 manifest 和报告

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 快速开始

### 环境要求

- Linux 或 macOS
- Python 3.10 或更高版本
- CMake 3.14 或更高版本
- 支持 C++17 的编译器
- Python FakeCUDA runtime 需要 PyTorch

Debian 或 Ubuntu 可安装 `build-essential`，macOS 可安装 Xcode Command Line
Tools。

### 安装

克隆仓库：

```bash
git clone https://github.com/FanBB2333/FakeGPU.git
cd FakeGPU
```

构建原生库并安装 Python 包：

```bash
scripts/build.sh
FAKEGPU_BUILD_DIR="$PWD/build" python3 -m pip install .
```

直接从源码目录开发：

```bash
python3 -m pip install pytest PyYAML jsonschema ruff
export PYTHONPATH="$PWD"
```

### 验证安装结果

```bash
python3 -m fakegpu doctor --list-profiles
python3 -m fakegpu demo --profile l4
```

`doctor` 检查 profile 目录、原生库和 PyTorch 环境。`demo` 在 CPU 上完成一个
小型 forward、backward 和 optimizer step，同时让程序看到 CUDA 设备。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 使用方式

### 使用 FakeCUDA 运行 PyTorch

请在导入 PyTorch 前初始化 FakeGPU：

```python
import fakegpu

fakegpu.init(runtime="fakecuda", profile="a100", device_count=2)

import torch

device = torch.device("cuda:0")
model = torch.nn.Linear(8, 4).to(device)
x = torch.randn(2, 8, device=device)
loss = model(x).square().mean()
loss.backward()

print(torch.cuda.device_count())      # 2
print(torch.cuda.get_device_name(0))  # NVIDIA A100
print(loss.item())
```

已维护的运算在 CPU 上执行，设备放置、显存限制、训练控制流程和错误处理使用
模拟的 CUDA 接口。

### 查看 FakeGPU 设备和进程

状态发布功能需要显式启用。请在工作负载调用 `fakegpu.init(...)` 或通过原生启动器
运行前设置状态目录；`build/` 已被 Git 忽略：

```bash
export FAKEGPU_SMI_STATE_DIR=build/smi

# Python FakeCUDA 工作负载。
python3 your_script.py

# 未修改的原生 CUDA/NVML 工作负载。
python3 -m fakegpu --build-dir build ./your_native_workload
```

在另一个终端中查看正在运行的工作负载：

```bash
# 紧凑的设备与进程表；添加 "-l 1" 可每秒刷新一次。
python3 -m fakegpu nvidia-smi --state-dir build/smi

# 设备列表，以及 runtime、profile 和 allocator 详细报告。
python3 -m fakegpu nvidia-smi --state-dir build/smi -L
python3 -m fakegpu nvidia-smi --state-dir build/smi -q

# 接近 NVIDIA 命令格式的建模拓扑和 NVLink 视图。
python3 -m fakegpu nvidia-smi topo -m --state-dir build/smi
python3 -m fakegpu nvidia-smi nvlink -s --state-dir build/smi

# 建模故障、兼容性警告、发布失败和过期状态。
python3 -m fakegpu nvidia-smi events --state-dir build/smi

# 建模 MIG GPU Instance 和 Compute Instance。
python3 -m fakegpu nvidia-smi mig -lgi --state-dir build/smi
python3 -m fakegpu nvidia-smi mig -lci --state-dir build/smi

# 适合脚本处理的 GPU 与进程查询。
python3 -m fakegpu nvidia-smi --state-dir build/smi \
  --query-gpu=index,name,uuid,pci.bus_id,profile.id,compute_cap,memory.total,memory.used,memory.free,allocator.model,native.kernel_launches,native.gemm_calls,native.io_bytes \
  --format=csv
python3 -m fakegpu nvidia-smi --state-dir build/smi \
  --query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory,peak_gpu_memory,stage,status \
  --format=csv,noheader,nounits
python3 -m fakegpu nvidia-smi --state-dir build/smi \
  --query-runtime=pid,fakegpu.version,runtime.backend,runtime.mode,policy.oom,tracking.dispatch,catalog.profiles,catalog.native_apis,dispatch.calls,publisher.failed_writes,status \
  --format=json
```

- `-q` 显示 FakeGPU 版本、runtime 策略、软件版本、状态更新时间、profile 目录与
  原生 API 覆盖、设备标识、显存分类、allocator 活动、dispatch 跟踪和各进程峰值。
- `-i` 按设备索引、UUID、PCI Bus ID 或 profile ID 筛选；`--json` 输出完整的标准化
  数据。当前写入 state schema v2，仍可读取 v1。
- `--query-runtime` 以 CSV 或 JSON 输出相同的运行时信息；`--help-query-runtime`
  列出全部字段。
- 原生拦截还会发布 allocation 生命周期、传输量、kernel launch、GEMM 调用与 FLOP、
  兼容性事件和 unsupported API 次数。`FAKEGPU_SMI_DETAIL_LIMIT` 与
  `FAKEGPU_SMI_MAX_STATE_BYTES` 分别限制保留明细和单个状态文件大小。

NVLink、MIG 和故障模型默认关闭，需在工作负载启动前设置；配置无效时不会激活
任何内容，只报告配置错误。

- **NVLink**：`FAKEGPU_NVLINK_GROUPS="0,1;2,3"` 为每组设备建立全连接关系；
  `FAKEGPU_NVLINK_BANDWIDTH_GBPS` 默认为 `900`。`topo -m`、`nvlink -s`、
  `topology.*`/`nvlink.*` 字段和原生 NVML 链路查询共用该模型。
- **MIG**：`FAKEGPU_MIG_LAYOUT="DEVICE:PROFILE:MEMORY_MIB[:COUNT];..."`，例如
  `FAKEGPU_MIG_LAYOUT="0:1g.10gb:10240:2;1:2g.20gb:20480"`。每个 GPU Instance
  对应一个 Compute Instance；超出父设备显存、8 个 slice 或 8 个实例的配置会被拒绝。
  实例的 used/free 显存标记为 `unobserved`。
- **故障**：`FAKEGPU_FAULT_EVENTS="DEVICE:CODE:SEVERITY[:COUNT];..."`，例如
  `FAKEGPU_FAULT_EVENTS="0:XID_79:critical;1:NVLINK_CRC:error:3"`，严重度支持
  `info`、`warning`、`error` 和 `critical`。`events` 视图同时汇总 unsupported API
  调用、发布失败和过期状态；`health.*` 字段给出设备的建模状态。

UUID 和 PCI Bus ID 是稳定的模拟标识。温度、风扇转速、实时功耗和硬件利用率显示
为 `N/A`；拓扑、带宽、故障和健康状态属于建模输入，不是硬件测量或 ECC/Xid 数据。

<a id="export-monitoring-metrics"></a>

### 导出监控指标

无需安装监控依赖，即可将 FakeGPU-SMI 标准化状态导出为 Prometheus 文本（默认）
或 JSON：

```bash
python3 -m fakegpu metrics --state-dir build/smi
python3 -m fakegpu metrics --state-dir build/smi --json
```

需要本地连续采集时，可启动带内存历史上限的 exporter：

```bash
python3 -m fakegpu metrics --state-dir build/smi --serve \
  --host 127.0.0.1 --port 9400 --interval 1 \
  --history-size 300 --max-process-series 128

curl http://127.0.0.1:9400/metrics
curl http://127.0.0.1:9400/healthz
curl http://127.0.0.1:9400/api/v1/history
```

- `/metrics` 提供最新的设备、进程、runtime、MIG、拓扑、健康状态和发布器指标；
  `/healthz` 显示数据源与采集状态；`/api/v1/history` 返回近期样本。
- 进程序列优先保留显存占用最高的进程，默认 128 个、最多 256 个，
  `--max-process-series 0` 可关闭；历史默认 300 个样本、最多 1,440 个。
- 数据只保存在内存中；长期留存请使用 Prometheus 或其他采集系统。
- 服务默认只监听 `127.0.0.1` 且不提供认证；对外开放前应配置带认证的代理。导出
  数据沿用 FakeGPU-SMI 对建模值和观测值的区分。

### 拦截原生 CUDA 库

构建原生库后，使用模块启动器为未修改的命令设置 `LD_PRELOAD` 或
`DYLD_INSERT_LIBRARIES`：

```bash
python3 -m fakegpu --build-dir build --profile a100 \
  python3 your_script.py

python3 -m fakegpu --build-dir build --devices "a100:2,h100:2" \
  python3 your_script.py

python3 -m fakegpu --build-dir build \
  --mode simulate \
  --unsupported-api error \
  python3 your_script.py
```

不支持的原生调用可以记录、显示警告，或返回 `cudaErrorNotSupported` 或
`CUDA_ERROR_NOT_SUPPORTED`。

原生 hybrid 模式下，CUDA runtime 和 cuBLAS 保持原生。设置了
`CUDA_VISIBLE_DEVICES`（按索引或 UUID）时会保留物理 NVML 发现，以便框架正确解析
该选择；否则使用建模的 NVML 设备清单。子环境切换模式时，`fakegpu.env()` 会移除
残留的 FakeGPU preload。

### 运行前检查显存

将命令执行到指定阶段，并把报告写入 Git 忽略的构建目录：

```bash
python3 -m fakegpu preflight \
  --runtime fakecuda \
  --profile a100 \
  --stage forward \
  --report-dir build/preflight \
  --strict \
  -- python3 train.py
```

Preflight 跟踪已执行路径中的可见显存，并判断选定的 profile 能否容纳该工作负载。

<a id="plan-online-llm-serving-capacity"></a>

### 规划在线 LLM 服务容量

以下命令不会加载 checkpoint tensor，用于估算同构活跃请求池：

```bash
python3 -m fakegpu plan-serving \
  --model-dir /models/example \
  --active-sequences 16 \
  --max-batch-size 64 \
  --prompt-tokens 4096 \
  --generated-tokens 256 \
  --prefill-chunk-tokens 512 \
  --shared-prefix-tokens 1024 \
  --kv-cache-strategy paged \
  --kv-cache-block-tokens 16 \
  --target-profile a100 \
  --memory-utilization 0.9 \
  --json build/serving-plan.json
```

报告分别列出模型权重、prefill/decode 临时显存、共享与私有 KV segment、runtime
开销和可用余量。自定义容量可用 `--device-memory-gib` 代替 `--target-profile`。
Prefix 共享需要 dynamic 或 paged cache，容量搜索不会超过 `--max-batch-size`。

#### 对齐 vLLM 的运行时预留

vLLM 会预先分配 paged KV 池，而不是随活跃请求逐步增长。部署目标为 vLLM 时可选择
对应的显存策略：

```bash
python3 -m fakegpu plan-serving \
  --model-dir /mnt/z/models/Qwen/Qwen3-0.6B \
  --active-sequences 16 \
  --max-batch-size 64 \
  --prompt-tokens 4096 \
  --generated-tokens 256 \
  --runtime vllm \
  --kv-cache-strategy paged \
  --kv-cache-block-tokens 16 \
  --target-profile "$TARGET_GPU_PROFILE" \
  --memory-utilization 0.92 \
  --json build/vllm-serving-plan.json
```

- 自动模式采用 vLLM V1 的计算方式：executor 申请的显存减去 profile 得到的非 KV
  显存，再向下取整为完整 block。报告区分请求实际需要的 KV 与预留池，并给出
  `num_gpu_blocks`、cache token 容量、初始化是否可行和 `max_model_len` 下的并发量。
- `--vllm-non-kv-cache-memory-bytes` 用 vLLM 启动 profile（权重 + activation 峰值
  + 非 PyTorch 分配 + CUDA graph）替代建模的非 KV 值。
- `--vllm-kv-cache-memory-bytes` 指定每张 GPU 的精确 cache 大小；与 vLLM 一致，
  此时不再使用 `--memory-utilization`。
- `--memory-utilization` 默认 `0.92`，与
  [vLLM CacheConfig 文档](https://docs.vllm.ai/en/latest/api/vllm/config/cache/)
  一致；固定软件版本时建议显式填写。
- 预算对应 vLLM 的 [GPU worker profile 流程](https://docs.vllm.ai/en/latest/api/vllm/v1/worker/gpu_worker/)，
  但初始空闲显存、其他进程、tensor/pipeline parallelism、hybrid cache group、
  allocator 碎片和版本相关 kernel 仍需在相同软件栈上观测。Speculative decoding
  目前只适用于 generic 运行时模型。

指定一个独立的自回归 draft checkpoint，即可估算 speculative decoding 显存：

```bash
python3 -m fakegpu plan-serving \
  --model-dir /models/target \
  --draft-model-dir /models/draft \
  --active-sequences 8 \
  --max-batch-size 32 \
  --prompt-tokens 4096 \
  --generated-tokens 256 \
  --speculative-tokens 5 \
  --speculative-acceptance-rate 0.7 \
  --kv-cache-strategy paged \
  --target-profile a100 \
  --json build/speculative-serving-plan.json
```

报告同时保留 target 与 draft 权重和两套 KV cache，并加入保守的 lookahead slot。
接受率只影响每次 target 调用的预期输出 token 数，不改变显存峰值，也不表示加速比。
两个模型的 vocabulary 大小必须相同；tokenizer identity 不做验证。

每份 serving 报告都会生成与路径无关的 `workload_signature`，涵盖架构、权重存储、
请求 shape、KV/prefill 和 speculative 配置，不含模型路径与设备容量。
`calibrate verify` 会比较签名并单独检查 `target.profile`，其他配置或 GPU 的观测
不会通过验证。

对于长度不同的 chat、RAG、completion 或 summarization 请求，可提供一个按
准入顺序排列的请求清单：

```json
{
  "schema_version": "fakegpu.serving_requests.v1",
  "requests": [
    {
      "id": "chat-a",
      "prompt_tokens": 4096,
      "generated_tokens": 256,
      "prefix_group": "system-prompt",
      "shared_prefix_tokens": 1024
    },
    {
      "id": "chat-b",
      "prompt_tokens": 8192,
      "generated_tokens": 512,
      "prefix_group": "system-prompt",
      "shared_prefix_tokens": 1024
    },
    {
      "id": "completion",
      "prompt_tokens": 2048,
      "generated_tokens": 1024
    }
  ]
}
```

```bash
python3 -m fakegpu plan-serving \
  --model-dir /models/example \
  --requests serving-requests.json \
  --max-batch-size 64 \
  --prefill-chunk-tokens 512 \
  --prefill-concurrency 2 \
  --kv-cache-strategy paged \
  --target-profile a100 \
  --memory-utilization 0.9 \
  --json build/mixed-serving-plan.json
```

- 按清单顺序逐个计算请求，同名 prefix 分组只保存一次，在不重排的前提下接纳能放入
  显存的最长清单前缀。
- 报告包含已接纳和被拒绝的请求 ID、每个请求的临时显存、并发 prefill 上界、分组
  KV segment 和容量余量。
- 同组请求必须使用相同且大于零的 `shared_prefix_tokens`；命名分组视为已驻留的
  cache 命中。
- 请求到达时间、eviction、preemption、tensor parallelism、自适应 speculative 调度、
  KV 回退、吞吐量和延迟均未建模；取得匹配观测前，报告保持 `Modeled` 和
  `uncalibrated`。

### 分析仓库或模型

```bash
# 查找 GPU 入口、依赖、原生源码和兼容性风险。
python3 -m fakegpu analyze-repo .

# 估算 checkpoint、KV cache、临时 tensor、adapter 和 MoE 显存。
python3 -m fakegpu estimate-llm \
  --model-dir /models/example \
  --batch-size 1 \
  --prompt-tokens 128 \
  --generated-tokens 32 \
  --dtype bfloat16 \
  --kv-cache-strategy paged \
  --kv-cache-block-tokens 16 \
  --target-profile a100 \
  --json build/llm-estimate.json

# 比较 diffusion 生成阶段和显存优化选项。
python3 -m fakegpu estimate-diffusion --list-profiles
python3 -m fakegpu estimate-diffusion \
  --model-profile stable-diffusion-xl-base-1.0 \
  --height 1024 --width 1024 --batch-size 4 \
  --attention-backend sdpa --vae-slicing \
  --offload model --target-profile a100 \
  --json build/diffusion-estimate.json

# 检查本地 Diffusers pipeline，不加载 tensor payload。
python3 -m fakegpu estimate-diffusion \
  --model-dir /models/pixart-or-flux \
  --height 1024 --width 1024 --text-tokens 300 \
  --dtype bfloat16 --attention-backend sdpa \
  --offload model --target-profile a100 \
  --json build/local-diffusion-estimate.json

# 根据能力 manifest 检查源码和已构建原生库的导出符号。
python3 -m fakegpu capabilities \
  --source-root . \
  --build-dir build \
  --strict
```

LLM 估算器：

- 只读取 safetensors header。`--kv-cache-strategy` 可选 `dynamic`、`static`、
  `quantized` 或 `paged`；报告分别列出逻辑存储量、量化节省量、static 预留、
  paged 额外占用和 sliding-window 上限。
- 量化 cache 默认将最近 128 个 token 保留为计算 dtype（`--kv-cache-residual-tokens`）。
- MHA、GQA 和 MQA 使用独立的 key/value 存储；配置同时包含 `kv_lora_rank`、
  `qk_nope_head_dim`、`qk_rope_head_dim` 和 `v_head_dim` 时识别为 MLA（压缩
  latent + 独立 RoPE），MLA backend workspace 仍需校准。

Diffusion 估算器：

- 分别计算 text encoding、重复 denoising 和 VAE decode。`--model-dir` 只读取
  `model_index.json`、组件 `config.json` 和选定 safetensors header，不导入远端
  代码，也不读取 tensor payload。
- checkpoint 存储字节与运行时权重字节分别报告。本地 Euler/PNDM scheduler 配置
  决定 denoiser 调用数，其他 scheduler 报告 `denoiser_evaluations: null`。
- UNet、交叉注意力 Transformer（DiT/PixArt）和 SD3/Flux 类联合注意力
  Transformer 使用不同的 activation 公式；CFG 只在存在正负分支的架构上使 batch
  翻倍。
- `--weight-variant fp16|bf16|fp32` 指定权重文件族。内置 profile 包含
  `pixart-sigma-xl-2-1024-ms` 和两个 Stable Diffusion UNet。
- `--offload model`、`--attention-slicing`、`--vae-slicing` 和 `--vae-tiling`
  对应 Diffusers 常见显存选项。没有同配置真实 GPU 对比时，报告保持 `Modeled` 和
  `uncalibrated`，不给出误差百分比或预测区间。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 命令参考

| 命令 | 用途 |
|---|---|
| `fakegpu doctor` | 检查安装、原生库、PyTorch 和 profiles |
| `fakegpu demo` | 执行小型 CPU FakeCUDA 训练步骤 |
| `fakegpu preflight` | 将工作负载执行到指定阶段并判断 fit 或 OOM |
| `fakegpu analyze-repo` | 统计仓库入口和仅支持 GPU 的风险 |
| `fakegpu analyze-kernel` | 检查 CUDA、PTX 和 SASS 资源与运算 |
| `fakegpu estimate-llm` | 估算 decoder 显存、通信量和 FLOP |
| `fakegpu plan-serving` | 规划同构或混合请求准入、chunked prefill 和分组共享 prefix KV 存储 |
| `fakegpu benchmark-serving` | 测量混合流式请求，保留用量、延迟、SSE 事件和失败 |
| `fakegpu plan-experiments` | 展开限额参数网格并检查实际命令 |
| `fakegpu run-experiments` | 运行或显式恢复实验，保留每次尝试 |
| `fakegpu compare-experiments` | 核验原始产物、比较重复观测并可导出图表 |
| `fakegpu estimate-diffusion` | 估算 diffusion 的 text encode、denoise 和 VAE decode 阶段显存 |
| `fakegpu plan-diffusion-training` | 规划缓存特征 U-Net 训练与持久存储 |
| `fakegpu run-diffusion-training` | 在单个真实 GPU 执行缓存特征 U-Net full/LoRA 训练 |
| `fakegpu estimate-diffusion-training` | 用兼容原生观测预算缓存特征训练 |
| `fakegpu validate-diffusion-training` | 对照独立原生 diffusion 训练 holdout |
| `fakegpu estimate-roofline` | 生成与 profile 相关的分析延迟区间 |
| `fakegpu plan-training` | 统一分布式训练配置并估算单 rank 显存 |
| `fakegpu estimate-finetune` | 根据框架配置估算单卡全量/LoRA/QLoRA 的步数、时间与显存 |
| `fakegpu probe-finetune` | 在真实 CUDA 显卡上采集固定形状 Hugging Face Trainer 时间与显存 |
| `fakegpu profile-finetune-data` | 全量扫描本地 JSONL 的 token 长度、截断、padding 与处理后序列数 |
| `fakegpu run-finetune` | 执行 HF Trainer/PEFT、LLaMA-Factory 或 Axolotl 原生训练并记录数据、CUDA 时间、显存及最终保存 |
| `fakegpu validate-finetune-windows` | 使用独立校准与 holdout 训练验证动态窗口耗时 |
| `fakegpu validate-finetune-events` | 使用独立原生事件记录验证评估与 checkpoint 成本 |
| `fakegpu plan-finetune-pipeline` | 从校准顺序数据推导未来原生训练轮次或更新数 |
| `fakegpu estimate-finetune-pipeline` | 估算完整原生执行时间和已覆盖显存包络 |
| `fakegpu validate-finetune-pipeline` | 对照独立原生训练验证完整时间和显存预测 |
| `fakegpu plan-finetune-transfer` | 计划经核验的随机或顺序原生输入 |
| `fakegpu estimate-finetune-transfer` | 估算显式经验形状迁移预算 |
| `fakegpu validate-finetune-transfer` | 对照独立原生运行验证迁移预算 |
| `fakegpu calibrate-finetune` | 在可恢复工作区中运行限额原生校准 |
| `fakegpu budget-finetune-workspace` | 仅用 CPU 对保存的证据重预算 |
| `fakegpu simulate-topology` | 模拟 collective 路由和链路竞争 |
| `fakegpu replay-trace` | 汇总计算、通信、等待和显存时间线 |
| `fakegpu calibrate` | 整理 serving 观测、对比显存报告并执行可靠性门槛检查 |
| `fakegpu capabilities` | 列出或严格检查原生 API 分类 |
| `fakegpu nvidia-smi` | 查看设备、进程、建模拓扑/MIG、健康状态和可靠性事件 |
| `fakegpu metrics` | 导出有数量上限的 Prometheus/JSON 指标和短期内存历史 |
| `fakegpu workspace-profiles` | 验证并查看 workspace 估算 profiles |
| `fakegpu validate` | 执行 JSON、TOML 或 YAML 声明式验证矩阵 |
| `fakegpu coordinator` | 管理分布式模拟 coordinator |
| `fakegpu bandwidth` | 验证模拟 TCP payload 并报告吞吐量 |

使用 `python3 -m fakegpu --help` 查看完整列表，使用
`python3 -m fakegpu <command> --help` 查看各命令的选项。大型报告和计划在所有接受
JSON 路径的位置都可以使用 `.json.gz`，压缩输出采用原子替换且字节可复现。

### 微调总时间与显存预估

`estimate-finetune` 读取本地模型元数据和 Hugging Face Trainer/PEFT、LLaMA-Factory
或 Axolotl 配置，无需导入训练框架或加载权重，即可估算单卡全量、LoRA、4-bit QLoRA
的显存与**训练循环总耗时**。

- 样本数指过滤、切分及 packing 后的训练序列数；序列长度包括模板 token 与 padding。
- `max_steps` 优先于 epochs；`--max-steps -1` 可清除配置中的步数限制。

```bash
python -m fakegpu estimate-finetune \
  --config examples/finetuning/huggingface-lora.yaml \
  --model /path/to/Qwen3.5-0.8B --text-only \
  --profile "$TARGET_GPU_PROFILE" --num-samples 10000 --epochs 3 \
  --sequence-length 128 --json estimate.json
```

- 三类框架的配置示例见 [`examples/finetuning/`](examples/finetuning/)。配置适配经过
  CPU 测试，不代表所有框架与运行环境组合都完成了真机基准测试。
- 分布式/offload、未覆盖的优化器或无法解析的 adapter 目标会明确报错；分布式规划
  使用 `plan-training`。
- Qwen3.5 的线性与完整 attention 分层计算；DeltaNet backward workspace、快速内核
  和 MoE dispatch 只有部分解析覆盖。`--text-only` 选择多模态 checkpoint 的文本骨干。

`profile-finetune-data` 全量扫描本地 JSONL 的每条记录，统计内容摘要、token 长度、
截断、padding 与序列数。原始 `text`/`messages` 使用本地 tokenizer；
`--packing concatenate` 只是 token 预算情景，不复现框架的 packing 算法。要使用
实际训练集计数，请导出最终 `input_ids`（可附带 `attention_mask` 和 `labels`）并加上
`--processed`：

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

`--dataset-report` 同时绑定样本数和序列上限：替换配置中的样本数、拒绝冲突的覆盖，
并要求长度上限一致。`--source-framework` 只是来源标签。Packed 训练需要最终处理后的
序列计数。

`run-finetune` 在一张物理 CUDA 显卡上调用已安装的 HF Trainer/PEFT、LLaMA-Factory
或 Axolotl 原生流水线（全量、LoRA 或 bitsandbytes 4-bit QLoRA；本地未量化文本
checkpoint；除下文 packing 说明外为无 packing 数据）。它按完整原生配置执行训练
（相对路径以配置文件目录解析），并记录：

- 原生预处理、有效训练数据、实际 batch、优化器更新、allocator 峰值、评估/保存
  事件与最终保存；
- `observation.updates[].batches`：每个 microbatch 的形状、epoch 位置、token 槽位、
  非 padding token 和有效 labels；
- `observation.batch_profile`：按完整有序的梯度累积窗口分组耗时；每组有两个稳态
  update 后才标记 `sufficient`（仅表示样本充足，不是精度认证）；
- `observation.phase_events`：实际评估/保存调用树，阶段总量只累计根事件；保存后的
  文件清单不代表磁盘写入量。

Warmup 和统计开销不计入稳态样本，但包含在总墙钟时间中。

`validate-finetune-windows` 使用独立记录的 holdout 验证耗时预测：

- 至少提供两份模型、数据、软件及原生执行设置相同的独立校准报告。
- 剔除全局 warmup 和窗口首次出现后，每个窗口在每份校准报告中仍须有两个样本；
  缺少覆盖的窗口直接拒绝。
- 只校验历史证据，不预测 shuffle 顺序、warmup、总墙钟时间或动态 VRAM；原生
  packing 与 MoE 窗口尚未支持。

```bash
python -m fakegpu validate-finetune-windows \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --holdout-run holdout.json --json window-validation.json
```

`validate-finetune-events` 以同样方式验证评估/保存子总时间，开始、首次、重复和最终
事件分别匹配。目前支持无 packing 的 dense 文本模型、loss-only 评估，以及本地
模型/adapter 或可恢复 checkpoint 保存；不预测事件次数或完整流水线时间。

```bash
python -m fakegpu validate-finetune-events \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --holdout-run holdout.json --json event-validation.json
```

`--event-context` 还会匹配窗口前最近的评估和 checkpoint 边界（有序调用、工作量、
阶段及各边界后的 update 数）。每个“窗口＋上下文”组合在每份校准报告中需要两个
样本。它验证周期事件后重复出现的窗口，不预测首次保存的恢复耗时或完整训练时间。

```bash
python -m fakegpu validate-finetune-windows --event-context \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --holdout-run holdout.json --json event-window-validation.json
```

`--epoch-context` 还会匹配每个 microbatch 在 epoch 内的位置，用来区分包含新一轮
取数或 worker 启动的窗口；可单独使用，也可与 `--event-context` 组合：

```bash
python -m fakegpu validate-finetune-windows --event-context --epoch-context \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --holdout-run holdout.json --json epoch-window-validation.json
```

`plan-finetune-pipeline` 从校准运行中完整观测到的 epoch 推导未来轮次或更新次数。
`estimate-finetune-pipeline` 组合至少两次独立校准，得到完整原生预算：setup、warmup、
有序训练窗口、评估、checkpoint、Trainer 间隙与收尾。

- 计时止于原生调用及最后 CUDA 同步，不含 Python 启动、报告构造或 CLI 输出。
- 时间区间是各进程均值的极值，不是置信区间；任一部件缺少覆盖都会明确报错。
- 支持无 packing、保留尾批次的顺序文本数据：HF Trainer（Transformers 5.6.0）与
  LLaMA-Factory 0.9.5 需 `disable_shuffling: true`；Axolotl 0.18.0
  （Transformers 5.14.1）需 `curriculum_sampling: true`。
- 数据、模型、硬件和执行设置须与校准一致，改变数据行数需重新校准。日志须关闭或
  每次 update 记录；不支持 checkpoint 轮转删除和最佳模型重载。

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

默认随机采样或更换训练数据时，使用经验迁移流程：读取原生 prepared dataloader，在
CPU 上计划 sampler/collator 输出，并在校准中核对实际消费的行号和输入张量 SHA256。
支持 HF/Factory 配合 Accelerate 1.11.0、Axolotl 配合 Accelerate 1.13.0；随机采样
使用原生 data seed 加 epoch（Axolotl 0.18.0 不会把 YAML 的 `data_seed` 转发给
Trainer）。

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

- `calibrate-finetune` 固定启动两次全新原生进程，限制步数和运行时间，默认各自使用
  新的空 kernel 缓存目录。保留所有尝试，用 SHA256 封存报告，支持 `--resume`，不会
  隐式增加或重跑校准。
- 纯 CPU 重预算无需 GPU 或框架，但不能超过已保存的采样计划范围。
- 工作区需保留原始绝对路径，并放在已记录的输入目录之外（嵌套路径会被拒绝）；只支持
  有限本地数据和 POSIX 锁语义。Axolotl/Factory 的 prepared 数据隔离在工作区的
  `prepared/` 中，恢复时拒绝原始输入变更。
- 重复的 `--reuse-calibration-run PATH` 可复用另一份数据的兼容训练证据；启动和收尾
  仍由目标数据的两次新校准提供，且不按样本数放大。不同 token/mask/监督分布之间的
  训练成本复用是**经验近似**；缺少覆盖时返回 1 并列出缺口。
- `run-finetune --planning-epochs N`（或 `--planning-max-steps N`）直接保存 CPU 计划；
  `plan-finetune-transfer`、`estimate-finetune-transfer`、`validate-finetune-transfer`
  无需工作区管理即可完成相同步骤。streaming、自定义采样器和自动 batch size 调整
  尚未支持。

显存结果为同一 GPU 和 allocator 上完整运行的 allocated/reserved 峰值包络，不能推算
到更小的显卡；`--reserve-gib` 只是用户指定的余量，不证明不会 OOM。
`validate-finetune-pipeline` 对照独立 holdout 报告时间误差和显存低估。

长度分组：HF/Factory 配置 `train_sampling_strategy: group_by_length`，Axolotl 配置
`group_by_length: true`，并向 `calibrate-finetune` 或 `run-finetune` 传入
`--length-grouping-seed 123`。

- 保留原生分组算法，使用以 `123 + epoch` 为种子的独立 CPU 生成器；依赖全局 RNG 的
  原生分组不可复现。
- 必须指定规划范围。一个有限评估数据集使用同一显式种子，不推进训练 sampler；
  拒绝自定义评估 sampler 和提前准备的 dataloader。
- sampler plan v2 把分组长度和 batch size 纳入校准身份，不同长度序列需要各自的证据。

Packing：

- [packing 语义审计](tests/native/finetune_packing_semantics.py) 表明，在极小的
  Qwen3.5 模型中，所测普通 HF/Factory 混合路径在分块 mask 和重置位置后仍有跨段
  影响；所测 Axolotl patch 与 Transformers 5.14.1 的 decoder 接口不兼容。
- 可选的 [分段参考审计](tests/native/finetune_segment_reference.py) 检查项目自有的
  Qwen3.5 分段接口：每段 linear attention 使用空的卷积和递归状态，逐层输出与梯度
  与独立原生序列一致，包括 checkpoint 重算与重载。
- 对 dense `qwen3_5_text` LoRA，向 `run-finetune` 或 `calibrate-finetune` 传入
  `--packing-seed 123` 启用该路径；需指定规划范围、关闭评估并启用原生 packing
  （Factory 设置 `packing: true` 和 `neat_packing: true`，Axolotl 设置
  `sample_packing: true`）。
- HF processed 行另需 `position_ids`，`attention_mask` 使用连续正整数 segment ID
  （右侧 padding 为零）；后续各段的首个 label 必须忽略。
- sampler plan v3 记录 epoch bins、逻辑序列数、张量行数、分段几何、dtype 和输入
  SHA；不同数据需要独立校准。原生 packing 报告须使用 transfer 命令。
- 不承诺 varlen 内核吞吐；full/QLoRA packing 和 grouped packing 未覆盖。另见
  [PEFT forward 审计](tests/native/finetune_segment_peft.py) 和
  [packing 预算证据](tests/data/finetune_packing_budget_evidence.json)。

`run-finetune --cache-regime empty_kernel_cache` 要求显式设置五个互不相同、已存在且
为空的目录：`TRITON_CACHE_DIR`、`TORCHINDUCTOR_CACHE_DIR`、`CUDA_CACHE_PATH`、
`TORCH_EXTENSIONS_DIR`、`PYTORCH_KERNEL_CACHE_PATH`。`populated_kernel_cache` 记录
其中每个文件的相对路径、大小和 SHA256。runner 不创建、清理或预热缓存；默认
`uncontrolled` 不作缓存状态承诺，也不能与受控校准混用。Triton 元数据可能通过绝对
路径引用原缓存，必须保持这些文件不变。

原生运行准备：

- Unpacked HF 运行（`--framework huggingface`）使用平铺配置，以及每行恰好包含
  `input_ids`、`attention_mask`、`labels` 的 processed JSONL：等长整数数组、二值
  mask、长度 2 到 `sequence_length`。不隐式分词，tokenizer 须有 pad token。参见
  [原生 LoRA 样例](examples/finetuning/huggingface-native-lora.yaml)。
- 梯度检查点：LLaMA-Factory 使用 `disable_gradient_checkpointing: false` 并以
  `use_reentrant_gc: true` 或 `false` 选择实现；Axolotl 使用
  `gradient_checkpointing: true` 与 `gradient_checkpointing_kwargs: {use_reentrant: true}`
  （或 `false`）。实际检查点与量化设置纳入校准身份。
- 原生训练需要文本 checkpoint，`--text-only` 不会转换权重。对于本地 Qwen3.5，先在
  CPU 上导出文本 checkpoint，并生成用于集成检查的合成 Alpaca 数据：

```bash
python tests/native/prepare_finetune_fixture.py \
  --model /path/to/Qwen3.5-0.8B \
  --output-dir /tmp/fakegpu-finetune/qwen35-text \
  --dataset-dir /tmp/fakegpu-finetune/data
```

导出需要支持 `Qwen3_5ForCausalLM` 的 Transformers。原生示例针对 LLaMA-Factory 0.9.5
（Transformers 5.6.0）和 Axolotl 0.18.0（Transformers 5.14.1），请分别在独立环境
安装对应框架及兼容的 CUDA PyTorch、PEFT 和数据依赖。每次运行使用新的 `output_dir`；
Axolotl 预处理缓存位于 `/tmp/fakegpu-finetune/axolotl-prepared`。

```bash
CUDA_VISIBLE_DEVICES=1 python -m fakegpu run-finetune \
  --config examples/finetuning/axolotl-native-lora.yaml --framework axolotl \
  --profile "$TARGET_GPU_PROFILE" --warmup-steps 2 --output run.json
python -m fakegpu estimate-finetune \
  --run-report run.json --epochs 3 --max-steps -1 --json estimate.json
```

- LLaMA-Factory 使用 `--framework llamafactory` 和
  [`llamafactory-native-lora.yaml`](examples/finetuning/llamafactory-native-lora.yaml)。
- 在同一 Python 环境复用 `--run-report`：CLI 会核对本地权重、模型配置和软件身份，
  GPU profile 默认沿用报告值。
- 该单报告 `estimate-finetune` 接口要求固定形状的完整 batch。改变 `--num-samples`
  只提供训练循环情景，不包含数据准备与加载时间，也不含 Python 启动或下载。

512-token NF4 QLoRA 与梯度检查点示例见
[`llamafactory-native-qlora.yaml`](examples/finetuning/llamafactory-native-qlora.yaml) 和
[`axolotl-native-qlora.yaml`](examples/finetuning/axolotl-native-qlora.yaml)（double
quantization、BF16，需要 bitsandbytes）。用已导出的 tokenizer 生成较长数据：

```bash
python tests/native/prepare_finetune_fixture.py \
  --model /tmp/fakegpu-finetune/qwen35-text --dataset-only \
  --dataset-dir /tmp/fakegpu-finetune/data-long \
  --sequence-length 1024 --num-samples 64
```

fixture 长度只是下限；框架会按配置截断到 512 或 1024 tokens，运行报告记录实际
采用的序列。

用 [`tests/native/finetune_framework_matrix.py`](tests/native/finetune_framework_matrix.py)
运行独立校准与复测进程，序列长度、microbatch、梯度累积和 seed 沿用原生配置：

```bash
python tests/native/finetune_framework_matrix.py \
  --framework axolotl --config examples/finetuning/axolotl-native-qlora.yaml \
  --python /path/to/axolotl/venv/bin/python \
  --output-dir /tmp/fakegpu-finetune/axolotl-qlora-matrix \
  --gpu-index 1 --calibration-steps 6 --warmup-steps 2 \
  --holdout-runs 2 --holdout-steps 16
```

步数包含 warmup（默认 6 次校准更新、2 次 warmup、三次各 32 次更新的复测）。每次
训练使用独立输出目录，时间预测调用生产估算器并使用实际 microstep 计数。

已有真实显卡时，可在**同一 Python 环境**中先短跑采样，再外推数据量与 epochs；
模型、显卡、精度、adapter、attention、batch、长度和累积设置须保持一致：

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

内置 probe 用 Hugging Face Trainer 和固定长度的合成文本运行，需要 Transformers、
Accelerate 和 CUDA PyTorch；adapter 另需 PEFT，QLoRA 另需 bitsandbytes。warmup
单独记录，并保留启动/稳态显存峰值。预测耗时不含数据准备、模型加载、评估与
checkpoint I/O。checkpoint 文件会计算内容摘要，同路径替换权重会使旧校准失效。

验证记录（精确负载、版本与适用范围见 [`refactor.md`](refactor.md)）：

- [完整流水线验收](tests/data/finetune_pipeline_validation_evidence.json)：三框架共
  30 个原生进程，含独立的 allocator OOM 对照。
- [采样与迁移验收](tests/data/finetune_transfer_validation_evidence.json)：Blackwell GPU 上 18 个 Qwen3.5-0.8B 进程，包括 25 → 73 条数据迁移以及随机采样配合周期
  eval/save；每组一次独立 holdout。
- [Qwen3.5-2B 验收](tests/data/finetune_qwen2b_evidence.json)：512/1024-token LoRA
  与 512-token NF4 QLoRA（启用梯度检查点），每条件两次校准、两次 holdout；总时间
  误差 0.054%–1.578%。
- [原生框架验证归档](tests/data/finetune_framework_validation_evidence.json)：此前的
  128-token LoRA，最大墙钟误差 Factory 3.931%、已缓存的 Axolotl 1.125%。把 Axolotl
  首次运行的校准用于后续缓存进程时误差为 364.885%（首次 update 66.542 秒），因此
  模型/软件签名相同不代表 cache/JIT 开销可复用。
- [扩展验证归档](tests/data/finetune_extended_validation_evidence.json)：两框架六组配置，
  覆盖 512/1024 tokens、batch 1/2、梯度检查点与 NF4 QLoRA，共 18 个测量进程。
- [Qwen3.5-0.8B probe 记录](tests/data/finetune_validation_evidence.json)：全量、LoRA、
  QLoRA 的校准与复测，可用 [`tests/native/finetune_matrix.py`](tests/native/finetune_matrix.py)
  复现。

这些短跑结果不代表通用精度或 OOM 安全。

### 在线 serving 基准测试

`benchmark-serving` 测量已运行的 completion 服务，依赖 vLLM 的 `ignore_eos` 和最终
流式 usage 字段。token-ID 清单指定到达时间与输出上限；报告保留 SSE 事件、失败、
客户端排队与首个非空文本延迟，不重试请求，也不推断设备阶段峰值。`--model` 填写
服务端配置的模型名称。

```bash
python -m fakegpu benchmark-serving \
  --base-url http://127.0.0.1:8000/v1 --model local-model \
  --requests examples/serving/online-requests.json \
  --concurrency 4 --repetitions 3 --json build/online.json.gz
```

[真实 serving 验收](tests/data/serving_native_evidence.json) 包含 20 个 8K/32K
Transformers 进程，以及 prefix cache 关闭/开启时的 72 个混合 vLLM 请求。32K 时
allocated 两阶段误差为 2.02%/7.34%，但 reserved decode 低估 24.07%：张量公式不能
保证 allocator 或 OOM 安全。

### Diffusion 训练

缓存特征 U-Net 训练支持 full、LoRA 和可选 `--ema-decay`。`plan-diffusion-training`
绑定本地权重与 FP32 缓存张量的 SHA；`run-diffusion-training` 使用一个真实 CUDA
GPU、BF16 autocast、FP32 可训练参数和 AdamW。VAE/text encoder 训练和原始图像
预处理不在范围内；下面的 fixture 生成随机特征，仅用于集成验证：

```bash
python tests/native/prepare_diffusion_fixture.py \
  --model-dir /path/to/stable-diffusion-v1-4 --output-dir /tmp/diff-fixture \
  --num-samples 5 --epochs 3 --method lora --gradient-checkpointing
CUDA_VISIBLE_DEVICES=1 python -m fakegpu run-diffusion-training \
  --plan /tmp/diff-fixture/plan.json --output-dir /tmp/diff-cal-1 \
  --json /tmp/diff-cal-1.json
CUDA_VISIBLE_DEVICES=1 python -m fakegpu run-diffusion-training \
  --plan /tmp/diff-fixture/plan.json --output-dir /tmp/diff-cal-2 \
  --json /tmp/diff-cal-2.json
python -m fakegpu estimate-diffusion-training --plan /tmp/diff-fixture/plan.json \
  --epochs 5 --calibration-run /tmp/diff-cal-1.json \
  --calibration-run /tmp/diff-cal-2.json --json /tmp/diff-budget.json.gz
```

- 缺少原生观测时，activation 与 reserved 峰值保持未知。
- 预算要求两个独立进程，且未来每类窗口都有重复样本；首次形状开销单独计算。
- `--cache-regime` 与微调流程相同（默认 `uncontrolled`）。Allocator、GPU UUID、
  软件、输入内容和执行身份必须一致。
- `validate-diffusion-training` 对照独立 `--holdout-run`；
  `--max-total-time-error-percent` 和 `--max-memory-underestimate-percent` 超限时
  返回 1。

[真实 diffusion 记录](tests/data/diffusion_native_evidence.json) 包含 SD/SDXL
generation 观测和 12 个训练进程。六次训练 holdout 的总时间误差为 0.044%–3.185%，
allocator 包络未低估；generation VAE allocated 峰值低估 12.0%–13.8%，reserved
最多低估 31.6%。

### 阶段显存校准

Allocated 张量存储与 allocator reserved 分别统计。`calibrate phase-sample` 在一个
全新 CUDA 进程中同时采集两者，`phase-estimate` 用至少五个独立进程封存各阶段最大值，
`phase-validate` 对照之后的独立运行。每份样本和预算都使用新的输出路径。

```bash
# plan.json: a compatible local plan-serving or estimate-diffusion report
for i in 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=1 python -m fakegpu calibrate phase-sample plan.json \
    --json "/tmp/phase-cal-$i.json"
done
python -m fakegpu calibrate phase-estimate \
  --calibration-sample /tmp/phase-cal-1.json \
  --calibration-sample /tmp/phase-cal-2.json \
  --calibration-sample /tmp/phase-cal-3.json \
  --calibration-sample /tmp/phase-cal-4.json \
  --calibration-sample /tmp/phase-cal-5.json --json /tmp/phase-budget.json.gz
CUDA_VISIBLE_DEVICES=1 python -m fakegpu calibrate phase-sample plan.json \
  --json /tmp/phase-holdout.json
python -m fakegpu calibrate phase-validate --budget /tmp/phase-budget.json.gz \
  --holdout-sample /tmp/phase-holdout.json --max-absolute-error-percent 20 \
  --max-underestimate-percent 10 --json /tmp/phase-validation.json
```

- 采集支持同构 Transformers dynamic-cache serving，以及本地 SD/SDXL 的 eager/SDPA、
  attention slicing、VAE slicing、VAE tiling 及其组合，要求无 offload。
- Tiling 时，`--vae-tile-size` 同时设置原生 sample 与 latent tile 上限；实际
  processor、VAE 开关和 tile 尺寸纳入原生身份。
- 计划、所选权重和配置、GPU UUID、Torch/CUDA/cuDNN 版本、allocator 与执行实现须
  一致；不做形状迁移，也不加隐式安全系数；kernel cache 不受控。
- 默认门槛为绝对误差 20%、低估 10%，未通过返回 1。离线预算/验证只依赖标准库。

验证证据（experimental，仅适用于已测 GPU、软件和固定输入）：

- [阶段显存验收](tests/data/phase_memory_evidence.json)：补上 VAE 上采样卷积后，四种
  分辨率的 VAE allocated 误差为 0.045%–1.172%；六个条件各五次校准、五次 holdout
  （共 60 个进程）均通过两项门槛。
- [扩展显存验证](tests/data/memory_generalization_evidence.json)：25 个单卡条件、
  74 个进程，覆盖 SD 1.4/SDXL、Qwen3-0.6B/1.7B/Qwen2.5-0.5B、FP16/BF16/FP32、
  不同 batch、矩形图像、guidance、eager attention、slicing 与 tiling。最大 allocated
  绝对误差 10.93%、最大低估 3.58%（门槛 20%/10%）。精确匹配条件的预算在十二次
  holdout 中全部吻合，68 次跨条件复用被拒绝。这些不是 `GPU-validated` 行：21 个
  条件不足五次观测，优化模式使用的是归档原生探针而非公开 CLI。
- [公开优化模式验收](tests/data/optimized_phase_memory_evidence.json)：七个 SD/SDXL
  条件、70 个进程，210 项 holdout 比较均与校准最大值一致；解析式 allocated 最大
  绝对误差 6.69%、最大低估 3.38%；42 次跨模式和 21 次实际模式不匹配被拒绝。

这些结果均不代表可迁移到新形状、具备预测区间或物理 OOM 安全。

### 分布式规划与实验批量运行

FSDP2 `full_shard` 可向 `plan-training` 提供 `--parameter-shapes layout.json`，内容为
`{"shapes": [[12, 8], [12], [3, 12], [3]]}`，并给出一致的 `--parameter-bytes` 和
统一的 `--parameter-element-bytes`。它按原生每参数第 0 维 chunk 计算并以最大 rank
做预算；未提供形状时使用均衡切分近似，不覆盖自定义 placement。

[版本化原生验收矩阵](tests/data/research_native_evidence.json) 记录六组
FP32/FP16/BF16 LoRA/attention CPU_SIM 用例、CPU 与 GPU 上 tiny Qwen3 MoE 的路由与
梯度，以及两个 CPU rank 的原生 DDP/FSDP2。物理 TP/PP/CP/EP、多 GPU ZeRO/FSDP 和
多机 collective 不在当前范围内。复现入口为 `scripts/test.sh cpu` 和
`tests/native/research_distributed.py --help`。

实验批量运行使用显式 argv、参数网格和重复次数，按顺序执行并绑定稳定运行 ID，每次
尝试保留日志、退出状态、错误、产物 SHA 和数值指标：

```bash
python -m fakegpu plan-experiments --config examples/experiments/cpu-sweep.json \
  --workspace /tmp/research-sweep --json /tmp/research-plan.json
python -m fakegpu run-experiments --config examples/experiments/cpu-sweep.json \
  --workspace /tmp/research-sweep --json /tmp/research-results.json.gz
python -m fakegpu run-experiments --config examples/experiments/cpu-sweep.json \
  --workspace /tmp/research-sweep --resume --json /tmp/research-resumed.json
python -m fakegpu compare-experiments --workspace /tmp/research-sweep \
  --plot /tmp/research-sweep.svg --x size --metric seconds \
  --json /tmp/research-comparison.json
```

`--resume` 跳过已核验成功项，`--resume --retry-failed` 重试失败或中断的项。输入、
源码或环境改变需要新工作区。命令须在前台运行，指标路径支持嵌套 JSON。绘图需要
Matplotlib，误差棒是观测最小/最大值，不是置信区间。示例测量 CPU 工作，不代表 GPU
吞吐。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## GPU profiles

目录包含 82 个 YAML profile，覆盖从 Maxwell 到 Blackwell 的消费级、工作站、
数据中心和嵌入式 NVIDIA GPU。Python 与原生 runtime 共用这些 profiles。

```bash
python3 -m fakegpu doctor --list-profiles
python3 -m fakegpu demo --profile rtx4090
python3 -m fakegpu --build-dir build --devices "t4,a100:2,h100" \
  python3 your_script.py
python3 scripts/update_nvidia_gpu_catalog.py --check
```

设置 `FAKEGPU_PROFILE` 或传入 `--profile` 可选择一个 profile。使用 `--devices`
可配置异构设备列表。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 开发

### 构建

所有可复用的原生构建行为都由一个脚本提供：

```bash
scripts/build.sh
scripts/build.sh --release
scripts/build.sh --debug
scripts/build.sh --build-dir build-custom -- -DSOME_CMAKE_OPTION=value
```

构建目录和编译产物不会被 Git 管理。

### 测试

项目维护的回归测试分为四组命令：

```bash
scripts/test.sh python
scripts/test.sh smoke
scripts/test.sh cpu
scripts/test.sh all
```

| 测试组 | 覆盖内容 |
|---|---|
| `python` | 项目维护的 Python 回归测试 |
| `smoke` | 原生库加载、报告、能力检查、SMI 拓扑/健康状态和 coordinator |
| `cpu` | CPU cuBLAS 模拟 |
| `all` | 所有维护中的测试 |

需要时可以直接执行声明式验证 manifest：

```bash
python3 -m fakegpu validate \
  --manifest tests/data/validation_smoke.yaml \
  --report-dir build/validation-smoke \
  --strict

python3 -m fakegpu validate \
  --manifest tests/data/research_validation.yaml \
  --report-dir build/validation-research \
  --strict
```

[发布回归记录](tests/data/research_release_evidence.json) 包含 1734 项通过的 Python
测试（跳过一个 CPU-only build 分支）、原生 CPU 套件、104 个打包模块、42 个命令帮助
和 68 次离线 CLI 功能检查；152 份历史预测逐字节保持一致。

### 可复用脚本

| 路径 | 用途 |
|---|---|
| `scripts/build.sh` | 配置并编译原生目标 |
| `scripts/test.sh` | 执行维护中的测试组 |
| `scripts/update_nvidia_gpu_catalog.py` | 检查或更新 profile 元数据 |
| `scripts/validation/` | 共用的报告和产物验证工具 |
| `scripts/linux/` | Linux GPU 管理工具 |
| `scripts/macos/` | macOS 到 Linux VM 的辅助工具 |

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 项目结构

```text
FakeGPU/
├── fakegpu/    Python 包、CLI、runtime 和估算器
├── profiles/   YAML GPU profile 目录
├── schemas/    JSON 报告和验证 schema
├── scripts/    可复用的构建、测试、平台和验证工具
├── src/        原生 C++ 拦截和 coordinator 实现
└── tests/      维护中的回归测试和最小原生测试 fixture
```

生成的构建目录、编译库、测试报告、缓存、本地环境、二进制资源和设计草稿均通过
`.gitignore` 排除。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 限制

- 原生模拟不能执行任意 CUDA kernel。
- FakeCUDA 覆盖项目维护的 Python 和 PyTorch 行为，不支持二进制 CUDA 扩展。
- 静态分析无法解析所有动态 import、生成式 kernel、运行时 shape 或依赖数据的
  分支。
- 显存估算可能遗漏 backend 私有分配、自定义 operator、allocator 策略和未匹配的
  workspace。
- `estimate-llm` 与 `plan-serving` 对视觉模型只计入 checkpoint 权重，不含
  vision/projector 激活与 FLOP（报告中会标注）；decoder token 数需已包含视觉 token。
- cuBLAS 未实现的算法、GEMM/Lt 不支持的 dtype、布局和指针模式，以及跳过 CPU
  计算的分支，均受 `FAKEGPU_UNSUPPORTED_API` 控制：`warn` 记录警告，`error` 在
  写入输出前返回 `CUBLAS_STATUS_NOT_SUPPORTED`。
- Diffusion profile 只描述固定的参考 pipeline。自定义 ControlNet、IP-Adapter、
  LoRA、safety checker、refiner、视频和 DiT 组件需要额外的组件数据及真实 GPU 验证。
- 在线 serving 规划不估算请求到达分布、cache eviction、preemption 或调度器重排、
  self-speculative/Medusa/EAGLE、tokenizer 重映射、tensor parallelism、吞吐量和延迟。
- Roofline 输出是分析区间，不是实测 kernel 延迟。
- 分布式耗时包含 coordinator、内存复制、socket 和进程调度，不能作为 NCCL、
  NVLink 或 RDMA benchmark。
- 建模故障事件只用于报告，不会改变 CUDA 执行，也不代表真实的 ECC/Xid 数据。
- Hybrid 和 passthrough 模式需要兼容的物理 CUDA 环境。
- macOS System Integrity Protection 可能删除系统程序的 `DYLD_*` 环境变量。
  原生拦截建议使用 Homebrew、conda 或 pyenv Python。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 开发计划

- [x] CPU PyTorch FakeCUDA runtime
- [x] 原生 CUDA、NVML、cuBLAS 和 NCCL 拦截
- [x] 可识别架构的 GPU profile 目录
- [x] 运行时、静态、LLM 和分布式显存分析
- [x] 分阶段估算 Stable Diffusion v1.5 和 SDXL 生成显存
- [x] 仓库、kernel、拓扑和 trace 分析
- [x] FakeGPU-SMI 设备、runtime、allocator 和进程详细查询
- [ ] 扩展可执行的原生 CUDA 运算和 cuBLAS 覆盖范围
- [x] 适配三类框架的单卡微调预估与同配置 Hugging Face Trainer 真机采样
- [x] 全量数据扫描与 LLaMA-Factory/Axolotl 单卡文本全量微调、LoRA、4-bit QLoRA 原生执行报告
- [x] 在已记录的 Qwen3.5 单卡场景中验证 LLaMA-Factory 与 Axolotl 原生入口
- [x] 通过 FakeGPU-SMI 发布原生 runtime 的实时状态和活动
- [x] 增加建模拓扑、NVLink 视图和 NVML peer 查询
- [x] 增加建模故障注入、健康状态和可靠性事件视图
- [x] 增加建模 MIG 视图和原生 NVML MIG handle 查询
- [x] 为设备、进程和 runtime 导出有数量上限的指标及 Prometheus 内存历史
- [x] 增加 continuous batching、混合请求长度、chunked prefill 和分组 prefix caching 的在线 serving 容量模型
- [x] 增加 draft-model speculative decoding 显存和准入模型
- [ ] 为长上下文和在线服务添加真实 GPU LLM 验证
- [x] 添加 SD/SDXL GPU 观测、缓存特征 U-Net full/LoRA/EMA 预算及按架构区分的 DiT 显存模型
- [ ] 在更多 GPU 和软件栈上验证分布式与 MoE 估算

提议功能和已知限制见
[GitHub Issues](https://github.com/FanBB2333/FakeGPU/issues)。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 参与贡献

欢迎提交问题报告、针对性测试用例、profile 修正、文档改进和代码修改。

1. Fork 仓库。
2. 创建分支：`git checkout -b feat/your-change`。
3. 为修改的行为添加或更新测试。
4. 执行 `scripts/test.sh all`。
5. 使用清晰的
   [Conventional Commit](https://www.conventionalcommits.org/) 信息提交。
6. Push 分支并创建 pull request。

显存估算或兼容性问题应附带完整命令、选定的 profile、软件版本和生成的报告。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 许可证

项目采用 MIT License，详情见 [LICENSE](LICENSE)。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 致谢

- README 结构参考
  [Best-README-Template](https://github.com/othneildrew/Best-README-Template)
- 基于 [PyTorch](https://pytorch.org/) 验证 CPU 框架行为
- 使用 [CMake](https://cmake.org/) 构建原生库

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

[test-shield]: https://img.shields.io/github/actions/workflow/status/FanBB2333/FakeGPU/test.yml?branch=main&style=for-the-badge&label=tests
[test-url]: https://github.com/FanBB2333/FakeGPU/actions/workflows/test.yml
[release-shield]: https://img.shields.io/github/v/release/FanBB2333/FakeGPU?include_prereleases&sort=semver&style=for-the-badge
[release-url]: https://github.com/FanBB2333/FakeGPU/releases
[python-shield]: https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white
[python-url]: https://www.python.org/
[license-shield]: https://img.shields.io/github/license/FanBB2333/FakeGPU?style=for-the-badge
[license-url]: LICENSE
[validation-static]: https://github.com/FanBB2333/FakeGPU/blob/cc7f221356b582e3e06618f226d9e05bd7da64ef/docs/ai-researcher-preflight.md#4-static-aten-storage-liveness-validation
[validation-inference]: https://github.com/FanBB2333/FakeGPU/blob/cc7f221356b582e3e06618f226d9e05bd7da64ef/docs/llm-inference-estimation.md#maintained-qwen3-8b-result
[validation-sft]: https://github.com/FanBB2333/FakeGPU/blob/cc7f221356b582e3e06618f226d9e05bd7da64ef/docs/llm-sft-memory-estimation.md#blackwell-gpu-matrix
[validation-qlora]: https://github.com/FanBB2333/FakeGPU/blob/cc7f221356b582e3e06618f226d9e05bd7da64ef/docs/llm-sft-memory-estimation.md#native-nf4-qlora-matrices
[diffusion-sd15]: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/451f4fe16113bff5a5d2269ed5ad43b0592e9a14
[diffusion-sdxl]: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/462165984030d82259a11f4367a4eed129e94a7b
[diffusion-pixart]: https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS/tree/e102b3591cc82e97071b8b4cb90d834d0c487207
