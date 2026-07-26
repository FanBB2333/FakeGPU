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
   - [工作方式](#工作方式)
   - [主要技术](#主要技术)
2. [快速开始](#快速开始)
   - [环境要求](#环境要求)
   - [安装](#安装)
   - [验证安装结果](#验证安装结果)
3. [使用方式](#使用方式)
   - [使用 FakeCUDA 运行 PyTorch](#使用-fakecuda-运行-pytorch)
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
| 检查不熟悉的 GPU 仓库或原生扩展 | 统计 GPU 入口、依赖、kernel 和不支持的 API | `analyze-repo`、`analyze-kernel`、`capabilities` |
| 设计或排查分布式工作流 | 分析 collective 路由、链路竞争、rank 等待、显存时间线和 TCP payload | `simulate-topology`、`replay-trace`、`bandwidth` |
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
```

对比报告包含各阶段的有符号误差、绝对误差、预测区间覆盖情况，以及建议的显存
安全余量和系数。这些建议只适用于相同的工作负载签名、shape、dtype、软件栈和
GPU profile。

<a id="llm-reliability"></a>

### LLM 可靠性报告

FakeGPU 按工作负载和环境签名报告可靠性。`GPU-validated` 结果只适用于记录中的
模型 revision、shape、dtype、attention backend、allocator、软件栈和 GPU。
`CPU-validated` 表示已维护的执行或分析行为在无物理 GPU 的环境中通过验证。
`Modeled` 表示已有分析模型，但没有对应的真实 GPU 数据；`Planned` 表示尚未形成
可对外声明的准确率。

#### 当前仓库验证

当前仓库状态于 2026-07-26 在 macOS 26.5 arm64、Python 3.11.9 和 PyTorch
2.9.1 CPU 环境中执行了 `scripts/test.sh all`：

| 验证层 | 已维护的检查 | 结果 |
|---|---|---|
| Python runtime、估算器、CLI、schema 和 README 契约 | 完整 `pytest` 测试集 | **151 个通过** |
| 原生库拦截 | 构建、库边界、导出符号、preload、显存类型、coordinator 和不支持 API 策略 | **通过** |
| 原生能力清单 | 5 个能力组、26 个显式 API、24 个强制执行策略的 API | **通过** |
| GPU profile 目录 | 82 个 profile，覆盖 15 种 compute capability | **通过** |
| CPU 数值模拟 | GEMM、cuBLASLt、批量 GEMM、BLAS1/2 和 FP16，共 8 组测试 | **通过** |
| CUDA 版 PyTorch 原生矩阵乘法 | 需要带 CUDA 的 PyTorch | **当前 CPU-only 主机未执行** |

GitHub CI 还会在 Python 3.10–3.12 上执行 Python 测试，并在 Linux 和 macOS
上执行原生 smoke 与 CPU simulation。上文的真实 GPU 结果来自对应的固定版本
验证快照；本次检查验证了结构化证据和计算公式，没有在当前 CPU-only 主机上
重新测量。

#### 已维护的 LLM 工作负载矩阵

| 工作负载类型 | 已覆盖变化 | 验证依据 | 状态 |
|---|---|---|---|
| 离线 decoder 推理 | Qwen3-8B、BF16、SDPA、模型加载、prefill 和 decode 峰值 | RTX PRO 5000 预测值与实测值 | `GPU-validated` |
| 全量与 adapter SFT | Qwen 0.8B/2B 全量微调和 LoRA | 十个 RTX PRO 5000 训练用例 | `GPU-validated` |
| 量化 adapter SFT | Qwen 0.8B/2B 原生 NF4 QLoRA | 十个 RTX PRO 5000 训练用例 | `GPU-validated` |
| 通用 decoder 分析 | Dense/MoE 元数据、adapter、量化 checkpoint、eager/SDPA attention、KV cache 和 expert-parallel 通信量 | 公式、fixture 和 CLI 回归测试 | `CPU-validated` + `Modeled` |
| 分布式训练规划 | DeepSpeed、Accelerate、FSDP/FSDP2、分片、checkpointing 和 CPU/NVMe offload | 配置、字节计算、拓扑和 trace 测试 | `CPU-validated` + `Modeled` |
| 长上下文与在线服务 | Dynamic/static/quantized 或 paged KV cache、continuous batching、chunked prefill、prefix caching 和 speculative decoding | 尚无项目维护的真实 GPU 数据 | `Planned` |
| 多 GPU LLM 执行 | TP、PP、CP、EP、MoE 负载不均衡，以及 FSDP/ZeRO 组合执行 | 目前只有分析拓扑和 coordinator 验证 | `Modeled` |

计划中的 cache 与在线服务用例参考
[Transformers cache strategies](https://huggingface.co/docs/transformers/kv_cache)
和 [vLLM serving](https://docs.vllm.ai/en/stable/) 提供的工作负载形态。CPU
FakeCUDA 不执行二进制 CUDA 扩展或任意 kernel；这类工作负载需要分析结果，并
通过 passthrough 或 hybrid 模式取得真实 GPU 观测值。

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
  --target-profile a100 \
  --json build/llm-estimate.json

# 根据能力 manifest 检查源码和已构建原生库的导出符号。
python3 -m fakegpu capabilities \
  --source-root . \
  --build-dir build \
  --strict
```

LLM 估算器只读取 safetensors header，不会把 checkpoint 权重加载到内存。

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
| `fakegpu estimate-roofline` | 生成与 profile 相关的分析延迟区间 |
| `fakegpu plan-training` | 统一分布式训练配置并估算单 rank 显存 |
| `fakegpu simulate-topology` | 模拟 collective 路由和链路竞争 |
| `fakegpu replay-trace` | 汇总计算、通信、等待和显存时间线 |
| `fakegpu calibrate` | 对比预测显存和实测显存 |
| `fakegpu capabilities` | 列出或严格检查原生 API 分类 |
| `fakegpu nvidia-smi` | 显示虚拟进程显存 |
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
| `smoke` | 原生库加载、报告、能力检查和 coordinator |
| `cpu` | CPU cuBLAS 模拟 |
| `all` | 所有维护中的测试 |

需要时可以直接执行声明式验证 manifest：

```bash
python3 -m fakegpu validate \
  --manifest tests/data/validation_smoke.yaml \
  --report-dir build/validation-smoke \
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
- Roofline 输出是分析区间，不是实测 kernel 延迟。
- 分布式耗时包含 coordinator、内存复制、socket 和进程调度，不能作为 NCCL、
  NVLink 或 RDMA benchmark。
- Hybrid 和 passthrough 模式需要兼容的物理 CUDA 环境。
- macOS System Integrity Protection 可能删除系统程序的 `DYLD_*` 环境变量。
  原生拦截建议使用 Homebrew、conda 或 pyenv Python。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 开发计划

- [x] CPU PyTorch FakeCUDA runtime
- [x] 原生 CUDA、NVML、cuBLAS 和 NCCL 拦截
- [x] 可识别架构的 GPU profile 目录
- [x] 运行时、静态、LLM 和分布式显存分析
- [x] 仓库、kernel、拓扑和 trace 分析
- [ ] 扩展可执行的原生 CUDA 运算和 cuBLAS 覆盖范围
- [ ] 为长上下文和在线服务添加真实 GPU LLM 验证
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
