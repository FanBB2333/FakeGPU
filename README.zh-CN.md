<a id="readme-top"></a>

<div align="center">

# FakeGPU

**无需生产 GPU 集群，即可验证 CUDA 应用、估算显存并模拟分布式 GPU 工作流。**

[![Test][test-shield]][test-url]
[![Release][release-shield]][release-url]
[![Python][python-shield]][python-url]
[![License][license-shield]][license-url]

[English](README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md)

[浏览文档](https://fanbb2333.github.io/FakeGPU/) ·
[报告问题](https://github.com/FanBB2333/FakeGPU/issues/new?labels=bug) ·
[提出功能建议](https://github.com/FanBB2333/FakeGPU/issues/new?labels=enhancement)

</div>

![FakeGPU 工作流：CPU 执行 PyTorch、切换 GPU 配置、显存预检和静态工作负载估算](docs/assets/readme/tldr-workflows.png)

> [!IMPORTANT]
> FakeGPU 用于开发、测试和容量规划，无法让任意 CUDA kernel 获得与真实
> GPU 相同的数值和性能。TCP 测试结果也不能替代 NCCL、NVLink 或 RDMA
> 性能测试。

## 目录

1. [项目介绍](#项目介绍)
   - [FakeGPU 能回答哪些问题](#fakegpu-能回答哪些问题)
   - [工作方式](#工作方式)
   - [主要技术](#主要技术)
2. [快速开始](#快速开始)
   - [环境要求](#环境要求)
   - [安装](#安装)
   - [检查安装结果](#检查安装结果)
3. [使用方式](#使用方式)
   - [使用 FakeCUDA 运行 PyTorch](#使用-fakecuda-运行-pytorch)
   - [拦截原生 CUDA 库](#拦截原生-cuda-库)
   - [运行前检查显存](#运行前检查显存)
   - [分析代码仓库或模型](#分析代码仓库或模型)
   - [查看虚拟 GPU 显存](#查看虚拟-gpu-显存)
   - [通过 TCP 模拟多台机器](#通过-tcp-模拟多台机器)
4. [功能范围](#功能范围)
   - [命令速查](#命令速查)
   - [运行模式](#运行模式)
5. [GPU Profiles](#gpu-profiles)
6. [报告与验证](#报告与验证)
7. [架构](#架构)
8. [限制](#限制)
9. [开发计划](#开发计划)
10. [文档](#文档)
11. [参与贡献](#参与贡献)
12. [许可证](#许可证)
13. [致谢](#致谢)

## 项目介绍

FakeGPU 是一套 CUDA、CUDA Runtime、cuBLAS、NVML 和 NCCL 拦截与分析工具。
应用可以发现可配置的 NVIDIA 风格设备；已维护的运算在 CPU 上执行；显存、
通信和兼容性事件会写入结构化报告。对于不适合直接加载的工作负载，FakeGPU
还提供静态显存和计算量估算。

项目适用于 CI、本地开发、兼容性测试、容量规划和可重复实验。模拟与分析功能
不需要物理 GPU；passthrough、hybrid 和校准实验需要真实 CUDA 环境。

### FakeGPU 能回答哪些问题

| 问题 | 建议入口 | 需要物理 GPU |
|---|---|---:|
| PyTorch 代码能否按预期执行 CUDA 相关控制流程？ | Python FakeCUDA runtime | 否 |
| 未修改的程序能否发现并调用 CUDA 系列动态库？ | 原生库拦截 | 否 |
| 某个工作负载能否放入指定的 GPU profile？ | Preflight 或静态显存估算 | 否 |
| LLM 的权重、KV cache、Adapter 或 MoE 显存大约是多少？ | LLM 估算器 | 否 |
| 仓库中有哪些 GPU 入口、依赖和原生扩展？ | 仓库分析器 | 否 |
| 指定 profile 的计算/带宽延迟范围是多少？ | Roofline 估算器 | 否 |
| 多 rank 控制流程、通信和恢复是否符合预期？ | 分布式模拟器 | 否 |
| 估算值与真实 CUDA 运行结果相差多少？ | Passthrough/hybrid 校准 | 是 |

### 工作方式

FakeGPU 提供三条互相配合的路径：

| 路径 | 应用看到的内容 | 实际执行方式 |
|---|---|---|
| **Python FakeCUDA** | CUDA 设备、CUDA 风格 tensor、显存 API 和常见训练流程 | `FakeCudaTensor` 将已维护的 PyTorch 运算交给 CPU |
| **原生库拦截** | `libcuda`、`libcudart`、`libcublas`、`libnvidia-ml` 和 `libnccl` 符号 | 已维护的调用使用主机内存或 CPU 运算；不支持的行为会被分类并记录 |
| **分析与报告** | 显存、FLOP、Roofline、拓扑和通信报告 | 分析 ATen 图、safetensors 元数据、运行跟踪、校准数据和 coordinator 事件 |

### 主要技术

- Python 3.10+：runtime、估算器、CLI 和报告
- C++17 与 CMake：原生拦截库和分布式 coordinator
- PyTorch：CPU FakeCUDA 执行和 ATen 图捕获
- YAML/JSON Schema：GPU profiles、测试矩阵和报告

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 快速开始

### 环境要求

- Linux 或 macOS
- Python 3.10 或更高版本
- CMake 3.14 或更高版本
- 支持 C++17 的编译器（Debian/Ubuntu 可安装 `build-essential`，macOS
  可安装 Xcode Command Line Tools）
- Python FakeCUDA 路径需要 PyTorch

### 安装

克隆仓库：

```bash
git clone https://github.com/FanBB2333/FakeGPU.git
cd FakeGPU
```

编译并验证原生库：

```bash
cmake -S . -B build
cmake --build build -j
./ftest smoke
```

安装包含原生库的 Python 包：

```bash
python3 -m pip install .
```

从源码目录开发时，可安装验证依赖并设置 `PYTHONPATH`：

```bash
python3 -m pip install PyYAML jsonschema pytest
export PYTHONPATH="$PWD"
```

### 检查安装结果

```bash
python3 -m fakegpu doctor --list-profiles
python3 -m fakegpu demo --profile l4
```

`doctor` 检查 profile 目录、原生库和 PyTorch 环境。`demo` 在 CPU 上完成一个
小型 forward、backward 和 optimizer step，同时让程序看到 CUDA 设备。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 使用方式

### 使用 FakeCUDA 运行 PyTorch

需要在导入 PyTorch 前初始化 FakeGPU：

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

该路径适合检查设备放置、训练控制流程、异常处理、显存统计和框架兼容性。
运算由 CPU 实际完成，大模型的执行速度可能远低于 CUDA。

### 拦截原生 CUDA 库

`fgpu` 为未修改的命令设置动态库预加载环境：

```bash
./fgpu --profile a100 --device-count 2 python3 your_script.py
./fgpu --devices "a100:2,h100:2" python3 your_script.py
./fgpu --mode simulate --unsupported-api error python3 your_script.py
```

不支持的原生调用有三种明确策略：

| 策略 | 行为 |
|---|---|
| `warn` | 每个不支持的 API 报告一次，并在可行时继续执行 |
| `error` | 返回 `cudaErrorNotSupported` 或 `CUDA_ERROR_NOT_SUPPORTED` |
| `allow` | 记录事件，但不打印警告 |

### 运行前检查显存

将命令执行到指定阶段，并生成 JSON、Markdown、stdout 和 stderr 文件：

```bash
python3 -m fakegpu preflight \
  --runtime fakecuda \
  --profile a100 \
  --stage forward \
  --report-dir preflight-report \
  --allocation-stacks \
  --strict \
  -- python3 train.py --small-config
```

Preflight 会统计已执行路径中可见的参数、buffer、gradient、optimizer state、
activation、tensor alias、dispatch 创建的 storage、saved tensor、缓存分配器
状态和有上下界的 workspace。

不分配真实 CUDA 显存的 ATen 图估算：

```python
from fakegpu import estimate_module_memory, require_workspace_coverage

report = estimate_module_memory(
    model,
    (example_input,),
    mode="training",
    optimizer="adamw",
    target_device="auto",
)

require_workspace_coverage(
    report,
    minimum_fraction=1.0,
    allow_extrapolated=False,
    require_upper_bound=True,
)
print(report["estimated_peak_interval_bytes"])
```

### 分析代码仓库或模型

```bash
# 查找入口、训练框架、CUDA 源文件和验证风险。
fakegpu analyze-repo .

# 估算权重、KV cache、临时 tensor、Adapter 和 MoE 显存。
fakegpu estimate-llm \
  --model-dir /models/qwen \
  --batch-size 1 \
  --prompt-tokens 128 \
  --generated-tokens 32 \
  --dtype bfloat16 \
  --target-profile a100 \
  --json build/llm-estimate.json

# 估算与 profile 相关的分析延迟区间。
fakegpu estimate-roofline \
  --profile a100 \
  --flops 1000000000000 \
  --memory-bytes 4000000000 \
  --launch-count 100

# 根据能力清单检查源码和已编译的原生符号。
fakegpu capabilities --source-root . --build-dir build --strict
```

LLM 估算器只读取 safetensors header，不会将 checkpoint 权重加载到内存。
它支持 dense 和常见 MoE decoder 元数据、量化 checkpoint 存储、多个 PEFT
Adapter、expert-parallel 通信、KV cache、eager/SDPA 临时 tensor、矩阵 FLOP
以及可选 Roofline 区间。

### 查看虚拟 GPU 显存

在一个终端中发布进程状态：

```bash
FAKEGPU_SMI_STATE_DIR=/tmp/fakegpu-smi python3 your_inference.py
```

在另一个终端中查看：

```bash
fakegpu nvidia-smi --state-dir /tmp/fakegpu-smi
fakegpu nvidia-smi --state-dir /tmp/fakegpu-smi --loop 1 --count 10
```

表格会分别显示请求的 tensor 字节数、allocator reserved 字节数和可选的校准
runtime overhead。这里展示的是 FakeGPU 状态，不是主机 NVIDIA 驱动数据。

### 通过 TCP 模拟多台机器

在指定端口运行自包含的双节点 loopback 测试：

```bash
fakegpu bandwidth \
  --listen 127.0.0.1:29591 \
  --nodes 2 \
  --ranks-per-node 1 \
  --size 4MiB \
  --iterations 10
```

FakeGPU 会启动 coordinator、创建拓扑、通过 TCP 传输 collective payload、
验证规约结果、报告端到端吞吐量，并记录任意节点对之间的通信数据。独立部署的
coordinator 可用于多台物理主机。

`torchrun`、DDP、FSDP、DeepSpeed、拓扑和恢复示例见
[分布式模拟使用说明](docs/distributed-sim-usage.zh.md)。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 功能范围

| 领域 | 已实现范围 | 限制 |
|---|---|---|
| FakeCUDA runtime | 常见 tensor 创建与操作、module、autograd、optimizer、设备传播、显存 API、混合精度、checkpoint、DataLoader 和 dispatch storage 生命周期统计 | 二进制 CUDA 扩展与未覆盖的 PyTorch operator 需要单独验证 |
| 原生 CUDA 栈 | 部分 Driver、Runtime、NVML、cuBLAS/cuBLASLt 和 NCCL 符号；主机内存；CPU GEMM；可配置的不支持 API 策略 | native simulate 模式不执行任意 CUDA kernel |
| 显存估算 | 运行峰值、简化 caching allocator、ATen 图生命周期、optimizer 阶段、workspace 区间、校准数据和 OOM 检查 | backend 内部内存和未匹配的自定义 kernel 需要校准 |
| LLM 分析 | Dense/MoE 推理、量化权重、PEFT Adapter、KV cache、临时 tensor、expert 通信、FLOP、SFT 参考和 FSDP/FSDP2 投影 | 不会自动推断融合量化 kernel、expert 不均衡和任意模型架构 |
| 性能模型 | 标量计算与显存带宽 Roofline，以及 lower/expected/upper 效率假设 | 只提供分析区间；Tensor Core 加速需要显式传入 |
| 仓库分析 | Python 入口、import、训练框架、配置、CUDA/PTX 源码、二进制扩展和建议实验 | 动态 import、生成 kernel 和数据相关分支需要运行验证 |
| 分布式模拟 | TCP/Unix coordinator、collective、P2P、subgroup、异构拓扑、节点对报告、timeout、故障注入、communicator shrink 和固定规模 elastic restart | 不复现 NCCL 协议、NVLink 或 RDMA 时序 |
| 框架兼容性 | 针对 Transformers、Accelerate、PEFT、TRL、DDP、FSDP/FSDP2、DeepSpeed ZeRO/Pipeline/AutoTP/AutoEP、torchtune、Lightning、LitGPT 和 nanoGPT 的测试 | 兼容性只适用于文档记录的版本和选项 |
| 监控与报告 | 虚拟 `nvidia-smi`、原生设备报告、preflight、cluster matrix、验证 manifest 和 JSON Schema 检查 | 虚拟数据只反映已统计或已校准的状态 |
| GPU 目录 | 82 个 profile，覆盖八种 NVIDIA 架构以及消费级、工作站、数据中心、嵌入式和测试设备 | 硬件规格不能保证 kernel 级性能等价 |

范围标记：

- **Maintained**：属于常规回归测试范围。
- **Validated**：在文档说明的模型、shape、软件和架构范围内完成专项数值或
  物理主机实验。
- **Compatibility-tested**：完成针对特定框架工作流的测试。
- **Experimental**：仅适用于对应原型范围，不提供通用兼容性保证。

### 命令速查

| 命令 | 用途 |
|---|---|
| `fakegpu doctor` | 检查安装、动态库、PyTorch 和 GPU profiles |
| `fakegpu demo` | 执行小型 CPU FakeCUDA 训练步骤 |
| `fakegpu preflight` | 将工作负载执行到指定阶段并判断 fit/OOM |
| `fakegpu analyze-repo` | 统计仓库入口和 GPU 依赖风险 |
| `fakegpu estimate-llm` | 估算 decoder 权重、运行显存、通信和 FLOP |
| `fakegpu estimate-roofline` | 生成与 profile 相关的分析延迟区间 |
| `fakegpu capabilities` | 查看或严格检查原生 API 分类 |
| `fakegpu nvidia-smi` | 显示虚拟进程显存 |
| `fakegpu workspace-profiles` | 检查并列出 workspace profiles |
| `fakegpu validate` | 执行 JSON/TOML/YAML 声明式测试矩阵 |
| `fakegpu coordinator` | 启动、探测、停止分布式 coordinator 或生成报告 |
| `fakegpu bandwidth` | 验证 TCP payload 并测量端到端吞吐量 |

### 运行模式

Python runtime：

| Runtime | 行为 |
|---|---|
| `fakecuda` | 为 PyTorch 添加 FakeCudaTensor 行为，并在 CPU 上执行已维护运算 |
| `native` | 在当前进程中加载 FakeGPU 原生动态库 |
| `auto` | 可用时选择 `fakecuda`，否则使用 `native` |

原生计算模式：

| `FAKEGPU_MODE` | 行为 | 需要物理 GPU |
|---|---|---:|
| `simulate` | 虚拟设备身份和显存；已维护的 cuBLAS/cuBLASLt 路径可使用 CPU | 否 |
| `passthrough` | 不注入 FakeGPU CUDA/NVML 的真实 CUDA 基线 | 是 |
| `hybrid` | 保留真实 CUDA 运算，同时虚拟化部分 Driver/NVML 并处理 OOM 策略 | 是 |

分布式模式：

| `FAKEGPU_DIST_MODE` | 行为 |
|---|---|
| `disabled` | 不安装 FakeGPU 分布式层 |
| `simulate` | 使用 coordinator 管理 collective 和 point-to-point 语义 |
| `proxy` | 保留真实 NCCL 数据传输并添加 FakeGPU 控制面报告 |
| `passthrough` | 直接转发到真实 NCCL |

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## GPU Profiles

Profiles 位于 `profiles/<architecture>/<segment>/*.yaml`，由 Python 与原生
runtime 共同使用。

| 架构 | Profile 数量 | Compute capability | 产品范围 |
|---|---:|---|---|
| Maxwell | 1 | 5.2 | GeForce GTX 900 系列 |
| Pascal | 9 | 6.0, 6.1 | GeForce GTX 10 和 Tesla P 系列 |
| Volta | 1 | 7.0 | Tesla V 系列 |
| Turing | 12 | 7.5 | GeForce RTX 20、Quadro RTX 和 T4 |
| Ampere | 22 | 8.0, 8.6, 8.7 | GeForce RTX 30、RTX A、A 系列加速卡和 Jetson |
| Ada | 17 | 8.9 | GeForce RTX 40、RTX Ada 和 L 系列加速卡 |
| Hopper | 2 | 9.0 | H 系列加速卡 |
| Blackwell | 18 | 10.0, 10.3, 11.0, 12.0, 12.1 | GeForce RTX 50、RTX PRO、B 系列、Jetson 和 GB10 |

每个 profile 都声明架构和 compute capability；验证器会拒绝不匹配的组合。
YAML 文件还会记录规格来源以及 measured/reference/synthetic 状态。

```bash
fakegpu doctor --list-profiles
./fgpu --profile rtx4090 --device-count 2 python3 your_script.py
./fgpu --devices "t4,a100:2,h100" python3 your_script.py
python3 scripts/update_nvidia_gpu_catalog.py --check
```

数据来源和验证规则见 [profiles/README.md](profiles/README.md)。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 报告与验证

| 文件 | 生成入口 | 主要内容 |
|---|---|---|
| `fake_gpu_report.json` | 原生 runtime | 单设备显存、IO、API 调用、不支持行为和已维护 GEMM FLOP |
| `cluster_report.json/.md` | 分布式 coordinator | Collective/P2P 总量、完整节点对矩阵、峰值、拓扑、时间线、故障和恢复 |
| `preflight_report.json/.md` | Preflight CLI | 阶段进度、fit/OOM、显存类别、workspace 覆盖率和可信度 |
| LLM estimate | `fakegpu estimate-llm` | 权重、KV cache、临时 tensor、Adapter、MoE 通信、FLOP 和 Roofline |
| 静态显存报告 | `./ftest static_memory_validation` | 图生命周期、optimizer 阶段、workspace profiles 和可选 CUDA 对比 |
| 声明式验证报告 | `fakegpu validate` | 展开的测试矩阵、前置条件、断言、主机/Git 信息、耗时和日志 |
| Virtual SMI state | FakeCUDA runtime | 单进程 requested、reserved、simulated 当前/峰值字节、阶段和可信度 |

常规本地检查：

```bash
./ftest smoke
./ftest cpu_sim
./ftest static_memory_validation
python3 -m pytest -q
python3 -m fakegpu validate \
  --manifest verification/data/validation_smoke.yaml \
  --report-dir build/validation-smoke \
  --strict
```

当前回归基线为 425 个测试通过，1 个可选测试跳过。原生 smoke、CPU 数值模拟、
严格能力检查、wheel 安装和严格 MkDocs 构建也已通过。精度数据只适用于文档
记录的工作负载和校准签名。

完整数值、分布式、框架和跨架构结果见
[报告与验证](docs/reports-and-validation.zh.md)。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 架构

```text
面向 GPU 的应用
├── Python runtime: fakegpu.init(runtime="fakecuda")
│   └── FakeCudaTensor + 策略 + 显存统计
│       └── 已维护的 PyTorch 运算在 CPU 上执行
│
├── 原生 runtime: ./fgpu 或 fakegpu.init(runtime="native")
│   └── libcuda / libcudart / libcublas / libnvidia-ml / libnccl
│       ├── profiles、allocation、stream 和指标
│       ├── 主机内存和 CPU 数值运算
│       └── hybrid 模式可转发到真实 CUDA
│
└── 分析工具
    ├── 仓库和依赖统计
    ├── ATen 图与 safetensors 估算
    └── Roofline、校准和报告检查

分布式 coordinator
└── 逻辑节点、TCP/Unix 传输、collective、故障和报告
```

文件级说明见 [架构与项目结构](docs/project-structure.zh.md)。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 限制

- Native simulate 模式不执行任意 CUDA kernel。兼容性 no-op 会影响测试结论
  时，应使用 `FAKEGPU_UNSUPPORTED_API=error`。
- FakeCudaTensor 覆盖已维护的 Python/PyTorch 行为，不支持二进制 CUDA 扩展。
- 静态仓库分析无法解析所有动态 import、生成 kernel、运行时 shape 和数据相关
  分支。
- 运行与静态显存估算可能遗漏 backend 内部内存、自定义 operator、特定 allocator
  策略和未匹配 workspace。容量规划应配合相同环境的真实 GPU 校准。
- LLM 估算器不复现融合量化 kernel，不推断 expert 不均衡，也不能自动执行任意
  模型架构。
- Roofline 结果是分析区间，不是实测 kernel 延迟。
- 分布式耗时包含 coordinator、内存复制、socket 和进程调度，不能作为原始网络
  或 NCCL 性能。
- Hybrid 和 passthrough 模式需要兼容的物理 CUDA 环境。
- macOS SIP 可能删除系统程序的 `DYLD_*` 环境变量。原生拦截建议使用
  Homebrew、conda 或 pyenv Python。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 开发计划

- [x] CPU PyTorch FakeCUDA runtime
- [x] 原生 CUDA/NVML/cuBLAS/NCCL 拦截
- [x] 可配置并检查架构的 GPU profile 目录
- [x] 运行、静态、LLM、MoE、量化和 Adapter 显存估算
- [x] 严格原生 API 能力清单和导出符号检查
- [x] 仓库分析器和 profile Roofline 估算器
- [x] TCP 多节点模拟和完整节点对通信报告
- [x] DDP、FSDP/FSDP2、DeepSpeed 和 elastic recovery 专项验证
- [ ] 扩展可执行的原生 CUDA 和 cuBLAS 操作
- [ ] 增加更多软件环境和工作负载的校准数据
- [ ] 增强生成 kernel 和自定义扩展检测
- [ ] 增加分层与高基数网络拓扑模型

提议功能和已知限制见
[GitHub Issues](https://github.com/FanBB2333/FakeGPU/issues)。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 文档

- [入门指南](docs/getting-started.zh.md)
- [快速参考](docs/quick-reference.zh.md)
- [AI 工作负载 Preflight](docs/ai-researcher-preflight.zh.md)
- [仓库与 Roofline 分析](docs/repository-and-performance-analysis.zh.md)
- [LLM 推理估算](docs/llm-inference-estimation.zh.md)
- [LLM SFT 显存估算](docs/llm-sft-memory-estimation.zh.md)
- [分布式模拟使用说明](docs/distributed-sim-usage.zh.md)
- [DeepSpeed 验证](docs/deepspeed-validation.zh.md)
- [错误模拟](docs/error-simulation.zh.md)
- [报告与验证](docs/reports-and-validation.zh.md)
- [声明式验证 Manifest](docs/validation-manifests.zh.md)
- [架构与项目结构](docs/project-structure.zh.md)

本地预览文档：

```bash
python3 -m pip install -e ".[docs]"
mkdocs serve
```

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 参与贡献

欢迎提交问题报告、最小测试用例、profile 数据修正、文档改进和代码修改。

1. Fork 仓库。
2. 创建分支：`git checkout -b feat/your-change`。
3. 为修改的行为添加或更新测试。
4. 执行对应 `ftest` target 和 Python 测试。
5. 使用清晰的
   [Conventional Commit](https://www.conventionalcommits.org/) 信息提交。
6. Push 分支并创建 pull request。

显存估算或兼容性问题应附带完整命令、profile、软件版本和生成的报告。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 许可证

项目使用 MIT License，详情见 [LICENSE](LICENSE)。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 致谢

- README 结构参考
  [Best-README-Template](https://github.com/othneildrew/Best-README-Template)
- GPU 型号和 compute capability 数据参考
  [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda/gpus) 与
  [旧版 GPU 列表](https://developer.nvidia.com/cuda/gpus/legacy)
- CPU 框架验证基于 [PyTorch](https://pytorch.org/)

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

[test-shield]: https://github.com/FanBB2333/FakeGPU/actions/workflows/test.yml/badge.svg?branch=main
[test-url]: https://github.com/FanBB2333/FakeGPU/actions/workflows/test.yml
[release-shield]: https://img.shields.io/github/v/release/FanBB2333/FakeGPU?include_prereleases&sort=semver
[release-url]: https://github.com/FanBB2333/FakeGPU/releases
[python-shield]: https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white
[python-url]: https://www.python.org/
[license-shield]: https://img.shields.io/github/license/FanBB2333/FakeGPU
[license-url]: LICENSE
