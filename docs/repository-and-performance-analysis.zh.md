# 仓库与性能静态分析

FakeGPU 提供以下无需执行 workload 的检查：

| 命令 | 用途 |
|---|---|
| `fakegpu analyze-repo` | 找出入口、GPU 专用依赖和需要验证的路径 |
| `fakegpu capabilities` | 检查 native API 是真实实现、模拟实现还是不支持 |
| `fakegpu estimate-roofline` | 根据 GPU profile 与显式 workload 参数估算延迟区间 |
| `fakegpu analyze-kernel` | 检查 PTX/SASS/CUDA 文本中的指令、资源与可识别 FLOP |

这些命令不会导入目标仓库、执行 kernel 或分配 GPU 显存。

## 扫描代码仓库

```bash
fakegpu analyze-repo /path/to/project \
  --entry train.py \
  --json build/repository-analysis.json
```

扫描器会统计 Python、CUDA、PTX/SASS、本地扩展、构建和配置文件，解析
Python import、别名、装饰器与部分关键调用，收集依赖并发现可能的入口。
它可以识别 PyTorch、
Transformers、Accelerate、DeepSpeed、PEFT、TRL、Triton、bitsandbytes、
Flash Attention、xFormers、Apex、Lightning 与 torchtune。
它也能识别 CMake CUDA/toolkit、`nvcc`、NVRTC、内嵌 CUDA/PTX 字符串、
CuPy RawKernel、Numba CUDA、`cpp_extension` 与自定义 operator 注册，并
静态分析可读取的 kernel 文件。

报告中的结论分为三类：

| 结论 | 含义 |
|---|---|
| `preflight_candidate` | 静态扫描没有发现必须使用真实 GPU 的路径 |
| `requires_targeted_validation` | `torch.compile`、DeepSpeed、FSDP 或模型并行配置需要专项实验 |
| `requires_real_gpu_or_hybrid` | CUDA、Triton 或其他编译扩展需要匹配的真实 CUDA/Hybrid 环境 |
| `analysis_incomplete` | 没有选择可运行入口，或部分 Python 文件无法解析 |

报告还会给出建议的 preflight 与 Hybrid 实验。加密或运行时下载的 kernel、
tensor 形状和依赖数据的分支不在静态扫描范围内。

## 检查 PTX、SASS 或 CUDA 源码

```bash
fakegpu analyze-kernel kernel.ptx \
  --profile rtx4090 \
  --threads-per-block 256 \
  --json build/kernel-analysis.json
```

PTX 报告包含静态 opcode 分类、entry point、声明的寄存器、静态共享内存和
可识别的运算 FLOP。指定 profile 后，还会给出受寄存器、共享内存和线程数
约束的 occupancy 上限。SASS 输入必须是文本反汇编。CUDA 源码报告包含
kernel 定义、launch、共享内存声明和运行时编译标记。

静态指令计数不会推断循环次数、分支频率、缓存行为或实际 occupancy。

## 审计 native API

```bash
fakegpu capabilities --source-root . --strict
fakegpu capabilities --source-root . --build-dir build --strict --json -
```

版本化清单覆盖 CUDA Runtime、CUDA Driver、cuBLAS、NVML、NCCL 的行为分组
和需要单独声明的高风险 API。源码审计检查 compatibility stub 与
unsupported-policy 调用是否已经登记；编译产物审计检查导出的 vendor symbol
是否匹配显式声明或已审核的分组规则。严格模式在审计不完整时返回非零状态。

运行时可以配合 `FAKEGPU_UNSUPPORTED_API=error`，让已识别但不会真正执行的
调用返回 `NotSupported`。

## 估算 roofline 延迟区间

```bash
fakegpu estimate-roofline \
  --profile a100 \
  --flops 2000000000000 \
  --memory-bytes 8000000000 \
  --launch-count 120 \
  --json build/roofline.json
```

模型根据 SM 数量、Compute Capability、profile 时钟和 CUDA core 发射宽度
计算标量 FP32 上限，根据显存时钟与总线宽度计算带宽。每个效率场景使用：

```text
耗时 = max(FLOPs / 有效计算速率,
           字节数 / 有效显存带宽)
       + launch 数量 × launch 开销
```

报告包含算术强度、ridge point、瓶颈判断、全部假设，以及
lower/expected/upper 三个结果。模型不会猜测 Tensor Core 加速比；如果有
对应 dtype、稀疏模式和矩阵形状的实测或官方数据，可以显式传入：

```bash
fakegpu estimate-roofline \
  --profile a100 \
  --flops 2000000000000 \
  --memory-bytes 8000000000 \
  --compute-acceleration-factor 8
```

这个结果是分析区间，不是 CUDA benchmark。功耗、occupancy、kernel fusion、
Tensor Core 形状限制、cache reuse、调度和并发竞争都可能改变真实耗时。

checkpoint 估算器的 `--target-profile` 会复用同一模型：

```bash
fakegpu estimate-llm \
  --model-dir /models/decoder \
  --prompt-tokens 128 \
  --target-profile a100 \
  --json build/decoder-estimate.json
```
