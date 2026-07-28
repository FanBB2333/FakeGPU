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
   - [分析仓库或模型](#分析仓库或模型)
4. [命令参考](#命令参考)
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

FakeGPU 为开发、CI、兼容性检查和容量规划模拟面向 CUDA 的运行环境。应用可以
发现可配置的 NVIDIA 风格设备；已维护的运算在 CPU 上执行；模拟显存和通信会被
记录。对于不应直接加载的工作负载，FakeGPU 还提供静态估算工具。

模拟和分析功能不要求物理 GPU。只有 passthrough、hybrid 和校准流程需要兼容的
真实 CUDA 环境。

### FakeGPU 能回答哪些问题

| 问题 | 建议入口 | 需要物理 GPU |
|---|---|---:|
| PyTorch 代码能否按预期执行面向 CUDA 的控制流程？ | Python FakeCUDA runtime | 否 |
| 未修改的进程能否加载并调用 CUDA 系列动态库？ | 原生库拦截 | 否 |
| 某个工作负载能否放入选定的 GPU profile？ | Preflight 或静态显存估算器 | 否 |
| LLM 的 checkpoint、KV cache、adapter 或 MoE 需要多少显存？ | LLM 估算器 | 否 |
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

绝对百分比误差按
`|预测值 - 实测值| / 实测值 × 100%` 计算。表中的“一致度”等于
`100% - 误差`，只是同一测量结果的直观表示。

| 已验证范围 | 真实 GPU 参考环境 | 证据规模 | 绝对百分比误差 | 一致度 |
|---|---|---:|---:|---:|
| [包含 backend 常驻显存校准的受控 ATen MLP 与 Transformer 参数网格][validation-static] | RTX 3090 Ti 与 RTX PRO 5000；PyTorch/CUDA 2.12/13.0 和 2.9/12.8 | 13 个工作负载，26 个观测 | **最大 0.08%** | **≥99.92%** |
| [Qwen3-8B BF16 SDPA 推理][validation-inference] | RTX PRO 5000；PyTorch 2.9.1/CUDA 12.8 | 模型加载与推理峰值 | 加载 0.0129%；**峰值 0.0672%** | 99.9871%；**99.9328%** |
| [Qwen 0.8B/2B 全量与 LoRA SFT][validation-sft] | RTX PRO 5000；PyTorch 2.8/CUDA 12.8 | 10 个训练用例 | **0.102%–1.921%** | **98.079%–99.898%** |
| [Qwen 0.8B/2B 原生 NF4 QLoRA][validation-qlora] | RTX PRO 5000；PyTorch 2.8/CUDA 12.8 | 10 个量化训练用例 | **0.628%–1.732%** | **98.268%–99.372%** |

这些数字需要结合测量方式理解：

- Qwen 数据以 `torch.cuda.max_memory_allocated()` 为参考，不包含 CUDA context
  和 allocator 已预留但尚未使用的显存。
- 受控 ATen 数据加入了当前 GPU 与软件栈的 backend 常驻显存测量值，不能将该值
  用于其他环境。
- 区间表示所有用例中的最小和最大误差，不是平均值。评估 OOM 风险时应重点关注
  最大低估值。
- `99.x%` 一致度不表示还有相同比例的可用显存。容量规划仍应加入与工作负载
  对应的安全余量或系数。

这些数字来自固定工作负载，不能作为任意场景的通用准确率。模型、shape、
attention backend、量化 kernel、allocator、PyTorch/CUDA 版本或 GPU 发生变化时，
需要重新校准。表中链接指向不可变的验证快照，其中保留了完整配置和实测字节数。
CI 还会检查当前仓库中的
[结构化证据摘要](https://github.com/FanBB2333/FakeGPU/blob/main/tests/data/memory_validation_evidence.json)
与 README 是否一致。

在真实 CUDA 主机上，可以重新执行项目维护的受控对比：

```bash
python3 scripts/validation/static_memory_validation.py \
  --output build/static-memory-validation.json \
  --markdown build/static-memory-validation.md \
  --max-underestimate-percent 5
```

在无 GPU 主机上添加 `--static-only` 可以检查估算流程，但不会生成真实 GPU
准确性结果。对于自己的工作负载，可以对兼容的预测报告和实测报告执行：

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

对比报告包含各阶段的有符号误差、绝对误差、预测区间覆盖情况，以及建议的显存
安全余量和系数。`calibrate verify` 会检查最大低估、绝对百分比误差的中位数/
95 分位数/最大值、预测区间覆盖率、指定容量下的 false-safe 判断，以及工作负载
参数是否一致；任一门槛未通过时返回状态码 1。结果只适用于相同的工作负载签名、
shape、dtype、软件栈和 GPU profile。

<a id="llm-reliability"></a>

### 研究场景可靠性报告

FakeGPU 按工作负载和环境签名报告可靠性。`GPU-validated` 结果只适用于记录中的
模型 revision、shape、dtype、attention backend、allocator、软件栈和 GPU。
`CPU-validated` 表示已维护的执行或分析行为在无物理 GPU 的环境中通过验证。
`Modeled` 表示已有分析模型，但没有对应的真实 GPU 数据；`Planned` 表示尚未形成
可对外声明的准确率。

#### 当前仓库验证

当前仓库状态于 2026-07-28 在 macOS 26.5 arm64、Python 3.11.9 和 PyTorch
2.9.1 CPU 环境中执行了 `scripts/test.sh all` 和两个声明式验证 manifest：

| 验证层 | 已维护的检查 | 结果 |
|---|---|---|
| Python runtime、估算器、CLI、schema 和 README 契约 | 完整 `pytest` 测试集 | **183 个通过** |
| 声明式验证矩阵 | 6 个 smoke 用例，加 22 个 research cache、训练、校准和 diffusion 用例 | **28 个通过** |
| 原生库拦截 | 构建、库边界、导出符号、preload、显存类型、coordinator 和不支持 API 策略 | **通过** |
| FakeGPU-SMI 诊断 | 有上限的状态发布、拓扑/NVLink/MIG 视图、NVML peer/MIG 查询、健康字段和事件报告 | **通过** |
| 监控 exporter | Prometheus/JSON 快照、历史和序列数量限制、异常状态降级及 HTTP 接口 | **通过** |
| 原生能力清单 | 5 个能力组、26 个显式 API、24 个强制执行策略的 API | **通过** |
| GPU profile 目录 | 82 个 profile，覆盖 15 种 compute capability | **通过** |
| CPU 数值模拟 | GEMM、cuBLASLt、批量 GEMM、BLAS1/2 和 FP16，共 8 组测试 | **通过** |
| CUDA 版 PyTorch 原生矩阵乘法 | 需要带 CUDA 的 PyTorch | **当前 CPU-only 主机未执行** |

GitHub CI 还会在 Python 3.10–3.12 上执行 Python 测试，并在 Linux 和 macOS
上执行原生 smoke 与 CPU simulation。上文的真实 GPU 结果来自对应的固定版本
验证快照；本次检查验证了结构化证据和计算公式，没有在当前 CPU-only 主机上
重新测量。

#### 已维护的研究工作负载矩阵

| 工作负载类型 | 已覆盖变化 | 验证依据 | 状态 |
|---|---|---|---|
| 离线 decoder 推理 | Qwen3-8B、BF16、SDPA、模型加载、prefill 和 decode 峰值 | RTX PRO 5000 预测值与实测值 | `GPU-validated` |
| 全量与 adapter SFT | Qwen 0.8B/2B 全量微调和 LoRA | 十个 RTX PRO 5000 训练用例 | `GPU-validated` |
| 量化 adapter SFT | Qwen 0.8B/2B 原生 NF4 QLoRA | 十个 RTX PRO 5000 训练用例 | `GPU-validated` |
| 通用 decoder 分析 | Dense/MoE 元数据、adapter、量化 checkpoint、eager/SDPA attention、KV cache 和 expert-parallel 通信量 | 公式、fixture 和 CLI 回归测试 | `CPU-validated` + `Modeled` |
| 分布式训练规划 | DeepSpeed、Accelerate、FSDP/FSDP2、分片、checkpointing 和 CPU/NVMe offload | 配置、字节计算、拓扑和 trace 测试 | `CPU-validated` + `Modeled` |
| KV cache 分配 | Dynamic 增长、static 预留、2/4/8-bit quantized 存储、paged block 取整和 sliding-window 上限 | 公式、API、`--kv-cache-strategy` CLI 和 `tests/data/research_validation.yaml` 矩阵测试 | `CPU-validated` + `Modeled` |
| Diffusion 图像生成 | Stable Diffusion v1.5、SDXL Base 和 PixArt-Sigma；本地 UNet、交叉注意力 DiT、SD3/Flux 类联合注意力 Transformer；CFG、offload、attention/VAE slicing 和 VAE tiling | 固定版本与本地组件 header、架构配置、阶段公式、CLI/单元测试和 `tests/data/research_validation.yaml` | `CPU-validated` + `Modeled` |
| 在线服务调度 | continuous batching、chunked prefill、prefix caching 和 speculative decoding | 尚无项目维护的真实 GPU 数据 | `Planned` |
| 多 GPU LLM 执行 | TP、PP、CP、EP、MoE 负载不均衡，以及 FSDP/ZeRO 组合执行 | 目前只有分析拓扑和 coordinator 验证 | `Modeled` |
| Diffusion 训练 | Optimizer、gradient、activation、EMA 和参数高效微调 | 尚无专用估算器或真实 GPU 数据 | `Planned` |

<a id="research-scenario-effects"></a>

#### 架构感知估算对比

此前使用的“可复现的建模效果”只表示固定输入可以得到相同的公式结果，并不表示
估算值已经接近真实 GPU 观测。为避免混淆，本节改为“架构感知估算对比”。
下表由仓库中的公式生成，并由 README 契约测试检查；GiB 使用二进制单位。
它用于检查架构、shape 和优化选项带来的变化，不能替代同配置的真实 GPU 校准。

| Research 场景 | 对照条件 | 建模结果 | 验证方式 |
|---|---|---:|---|
| LLM 推理 KV cache | 32 层 GQA、8 个 KV head、head dim 128、batch 1、BF16；context 4K → 32K | **0.50 → 4.00 GiB**（cache 增长 8 倍） | 精确字节公式 |
| LLM 训练 | 8B BF16 参数、AdamW、4 张 GPU、activation checkpointing；replicated → full shard | **104.31 → 26.54 GiB**/rank（减少 74.6%） | 训练阶段模型 |
| Stable Diffusion v1.5 生成 | 512²、batch 1、FP16、CFG；全部权重常驻 → model offload | **2.30 → 1.65 GiB**（减少 28.1%） | 组件与阶段模型 |
| SDXL 批量生成 | 1024²、batch 4、FP16、CFG；普通 VAE decode → VAE slicing | **11.49 → 7.74 GiB**（减少 32.7%） | 顺序 VAE batch shape 模型 |
| SDXL 高分辨率生成 | 2048²、batch 1、FP16、CFG；整图 VAE → 512² VAE tile | **11.49 → 7.27 GiB**（减少 36.7%） | 顺序 VAE tile shape 模型 |
| PixArt-Sigma DiT 生成 | 1024²、batch 1、FP16、CFG、model offload；eager → SDPA 的 denoise 阶段 | **2.32 → 1.28 GiB**（减少 44.7%） | patch token、交叉注意力与阶段模型 |

Diffusion profile 中的组件参数量来自
[Stable Diffusion v1.5][diffusion-sd15] 和
[SDXL Base 1.0][diffusion-sdxl]、[PixArt-Sigma XL 2][diffusion-pixart]
的固定版本。上述数字不包含 runtime context、allocator 碎片、backend 私有
workspace 和传输重叠。
在取得匹配观测值之前，架构感知报告保持为 `Modeled`，`accuracy.status`
为 `uncalibrated`。
本地架构检查通过 `estimate-diffusion --model-dir` 使用。

Cache 公式参考
[Transformers cache strategies](https://huggingface.co/docs/transformers/kv_cache)
提供的工作负载形态。在线服务调度仍处于计划阶段，参考
[vLLM serving](https://docs.vllm.ai/en/stable/)。CPU FakeCUDA 不执行二进制
CUDA 扩展或任意 kernel；这类工作负载需要分析结果，并通过 passthrough 或
hybrid 模式取得真实 GPU 观测值。

后续新增或更新的公开验证数据应包含：

- 至少五次隔离执行，以及其中最大的观测峰值；
- 每个报告阶段的预测字节数和实测字节数；
- 将最大低估作为主要 OOM 风险指标；标记为 `GPU-validated` 时，建议不超过
  5%；
- 绝对百分比误差的中位数、95 分位数和最大值；
- 预测区间覆盖率，以及 FakeGPU 判断可运行但真实工作负载发生 OOM 的
  false-safe 次数；
- 模型 revision、完整命令、shape、dtype、backend、allocator 设置、GPU、
  driver、CUDA、PyTorch 和框架版本。

公开结果前可使用 `calibrate verify` 对机器可读的对比报告执行这些门槛检查。

没有达到上述目标的数据继续标记为 `Modeled` 或 experimental，不作为已验证
准确率展示。“一致度”只作为辅助信息，最大低估和 false-safe OOM 判断更能反映
容量规划风险。

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
```

详细报告包含 FakeGPU 版本、runtime backend 与策略、Python/PyTorch/CUDA 版本、
状态更新时间、profile 目录和原生 API 覆盖情况、模拟设备标识、计算规格、显存
分类、allocator 活动、dispatch 跟踪和各进程峰值。`-i` 可以按设备索引、UUID、
PCI Bus ID 或 profile ID 筛选；`--json` 会输出完整的标准化数据。当前发布
state schema v2，同时仍可读取 v1 状态文件。

原生拦截还会发布 allocation 生命周期、传输量、kernel launch、GEMM 调用与
FLOP、兼容性事件和 unsupported API 次数。进程运行时会定期更新状态，退出时会
将状态标记为 exited。`FAKEGPU_SMI_DETAIL_LIMIT` 用于限制保留的明细数量，
`FAKEGPU_SMI_MAX_STATE_BYTES` 用于限制单个状态文件大小；`-q` 会显示发布次数、
失败次数、耗时和序列化大小。

NVLink 模型默认关闭。在工作负载启动前设置
`FAKEGPU_NVLINK_GROUPS="0,1;2,3"`，可为分号分隔的每组设备建立全连接关系。
`FAKEGPU_NVLINK_BANDWIDTH_GBPS` 是可选的带宽参数，默认值为 `900`。状态文件、
`topo -m`、`nvlink -s`、`topology.*`/`nvlink.*` 查询字段，以及原生 NVML
的链路状态、capability、远端设备类型和远端 PCI 查询共用同一模型。配置无效时，
所有链路都会保持 inactive，状态中会给出配置错误。

MIG 模型默认关闭。请在工作负载启动前使用
`FAKEGPU_MIG_LAYOUT="DEVICE:PROFILE:MEMORY_MIB[:COUNT];..."` 配置实例，例如
`FAKEGPU_MIG_LAYOUT="0:1g.10gb:10240:2;1:2g.20gb:20480"`。每个建模 GPU
Instance 对应一个 Compute Instance。状态文件、`-L`、`-q`、`mig -lgi`、
`mig -lci`、`mig.*` 查询字段，以及原生 NVML 的 MIG mode、handle、UUID、
父设备、实例 ID 和显存容量查询共用同一配置。单个设备最多包含 8 个实例和
8 个 slice，实例显存总量不能超过父设备；无效配置不会生成部分实例。当前还不能
将 runtime allocation 归属到具体 MIG 实例，因此状态文件会将实例 used/free
显存标记为 `unobserved`。

故障注入默认同样关闭。配置格式为
`FAKEGPU_FAULT_EVENTS="DEVICE:CODE:SEVERITY[:COUNT];..."`，例如
`FAKEGPU_FAULT_EVENTS="0:XID_79:critical;1:NVLINK_CRC:error:3"`。CODE 仅作为
标签，严重度支持 `info`、`warning`、`error` 和 `critical`。`events` 视图会
汇总配置的故障事件、原生 unsupported API 调用、状态发布失败、过期状态和模型
配置错误。`health.*` 查询字段会给出每个设备的建模状态、最高严重度、事件数，
并明确标注硬件健康状态不可观测。输入无效时不会激活任何故障，只会报告配置错误。

UUID 和 PCI Bus ID 是稳定的模拟标识。CPU runtime 无法观测温度、风扇转速、
实时功耗和硬件 GPU 利用率，因此这些字段显示为 `N/A`；profile 功耗与时钟频率
会作为静态规格单独展示。拓扑关系、配置带宽、故障代码和健康状态属于建模输入，
不是硬件测量结果，也不是观测到的 ECC/Xid 数据。

<a id="export-monitoring-metrics"></a>

### 导出监控指标

FakeGPU-SMI 的标准化状态无需安装监控依赖即可导出。单次命令默认输出
Prometheus 文本，也可以输出标准化 JSON 快照：

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

`/metrics` 提供最新的设备、进程、runtime、MIG、拓扑、健康状态和发布器指标；
`/healthz` 显示数据源与采集状态；`/api/v1/history` 返回近期的标准化样本。
进程序列优先保留显存占用最高的进程，默认上限为 128，最大为 256；
`--max-process-series 0` 可以关闭进程序列。历史默认保留 300 个样本，最多
1,440 个。这些限制都在内存中执行，监控历史不会写入磁盘，也不会由 Git 管理。
如需长期留存，请使用 Prometheus 或其他采集系统。

服务默认只监听 `127.0.0.1`，自身不提供身份认证。绑定非本机地址前，应在前方
配置带身份认证的代理或网络访问策略。导出的数据沿用 FakeGPU-SMI 对“建模值”和
“观测值”的区分，不会把无法取得的硬件遥测数据表示为真实测量结果。

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

LLM 估算器只读取 safetensors header，不会把 checkpoint 权重加载到内存。
`--kv-cache-strategy` 可选 `dynamic`、`static`、`quantized` 或 `paged`；
JSON 报告会分别列出逻辑存储量、量化节省量、static 预留、paged block 额外占用
和可选的 sliding-window 上限。量化 cache 默认将最近 128 个 token 保留为计算
dtype，可通过 `--kv-cache-residual-tokens` 调整。

Diffusion 估算器分别计算 text encoding、重复 denoising 和 VAE decode 阶段。
使用 `--model-dir` 时，它只读取 `model_index.json`、组件 `config.json` 和选定
safetensors 文件的 header，不会导入远端自定义代码，也不会读取 tensor payload。
checkpoint 存储字节和指定 dtype 下的运行时权重字节会分别报告。
内置 `pixart-sigma-xl-2-1024-ms` profile 与两个 Stable Diffusion UNet profile
一起提供固定版本的 Transformer 示例。

估算器会区分 UNet、带交叉注意力的 patch Transformer（DiT/PixArt）以及
SD3/Flux 类联合注意力 Transformer。图像 token 数会结合 VAE 缩放、patch size
和 Flux latent packing 计算；CFG 只在需要正负分支批次的架构上使 denoiser batch
翻倍。`--weight-variant fp16|bf16|fp32` 可在本地目录包含多套权重时指定要检查的
文件族。

`--offload model`、`--attention-slicing`、`--vae-slicing` 和
`--vae-tiling` 对应 Diffusers 中常见的显存选项。获得同配置的真实 GPU
对比数据前，报告状态保持为 `Modeled`，`accuracy.status` 为 `uncalibrated`，
也不会生成没有观测依据的误差百分比或预测区间。

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
| `fakegpu estimate-diffusion` | 估算 diffusion 的 text encode、denoise 和 VAE decode 阶段显存 |
| `fakegpu estimate-roofline` | 生成与 profile 相关的分析延迟区间 |
| `fakegpu plan-training` | 统一分布式训练配置并估算单 rank 显存 |
| `fakegpu simulate-topology` | 模拟 collective 路由和链路竞争 |
| `fakegpu replay-trace` | 汇总计算、通信、等待和显存时间线 |
| `fakegpu calibrate` | 对比显存报告并执行可靠性门槛检查 |
| `fakegpu capabilities` | 列出或严格检查原生 API 分类 |
| `fakegpu nvidia-smi` | 查看设备、进程、建模拓扑/MIG、健康状态和可靠性事件 |
| `fakegpu metrics` | 导出有数量上限的 Prometheus/JSON 指标和短期内存历史 |
| `fakegpu workspace-profiles` | 验证并查看 workspace 估算 profiles |
| `fakegpu validate` | 执行 JSON、TOML 或 YAML 声明式验证矩阵 |
| `fakegpu coordinator` | 管理分布式模拟 coordinator |
| `fakegpu bandwidth` | 验证模拟 TCP payload 并报告吞吐量 |

使用 `python3 -m fakegpu --help` 查看完整列表，使用
`python3 -m fakegpu <command> --help` 查看各命令的选项。

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
- Diffusion profile 只描述固定的参考 pipeline。自定义 ControlNet、IP-Adapter、
  LoRA、safety checker、refiner、视频和 DiT 组件需要额外的组件数据及对应的
  真实 GPU 验证。
- Roofline 输出是分析区间，不是实测 kernel 延迟。
- 分布式耗时包含 coordinator、内存复制、socket 和进程调度，不能作为 NCCL、
  NVLink 或 RDMA benchmark。
- 建模故障事件仅用于检查控制面报告，不会改变 CUDA 执行，也不代表 NVML ECC
  计数、实际观测到的 Xid 或硬件故障。
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
- [x] 通过 FakeGPU-SMI 发布原生 runtime 的实时状态和活动
- [x] 增加建模拓扑、NVLink 视图和 NVML peer 查询
- [x] 增加建模故障注入、健康状态和可靠性事件视图
- [x] 增加建模 MIG 视图和原生 NVML MIG handle 查询
- [x] 为设备、进程和 runtime 导出有数量上限的指标及 Prometheus 内存历史
- [ ] 为长上下文和在线服务添加真实 GPU LLM 验证
- [ ] 添加真实 GPU diffusion 验证，以及训练和 DiT 显存模型
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
[validation-static]: https://github.com/FanBB2333/FakeGPU/blob/df254c21eebc0a5bbf13992f3f5a8e995cfa8708/docs/ai-researcher-preflight.md#4-static-aten-storage-liveness-validation
[validation-inference]: https://github.com/FanBB2333/FakeGPU/blob/df254c21eebc0a5bbf13992f3f5a8e995cfa8708/docs/llm-inference-estimation.md#maintained-qwen3-8b-result
[validation-sft]: https://github.com/FanBB2333/FakeGPU/blob/df254c21eebc0a5bbf13992f3f5a8e995cfa8708/docs/llm-sft-memory-estimation.md#rtx-pro-5000-matrix
[validation-qlora]: https://github.com/FanBB2333/FakeGPU/blob/df254c21eebc0a5bbf13992f3f5a8e995cfa8708/docs/llm-sft-memory-estimation.md#native-nf4-qlora-matrices
[diffusion-sd15]: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/451f4fe16113bff5a5d2269ed5ad43b0592e9a14
[diffusion-sdxl]: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/462165984030d82259a11f4367a4eed129e94a7b
[diffusion-pixart]: https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS/tree/e102b3591cc82e97071b8b4cb90d834d0c487207
