# 限制

[文档索引](README.md) · [English](../en/limitations.md) ·
[繁體中文](../zh-TW/limitations.md)

- Linux 原生模拟可在 CPU 上执行受支持的 PTX 和兼容的 cubin/SASS 指令；不支持的指令和
  启动特性会明确失败。macOS 尚不执行这些内核。详见[框架兼容性](frameworks.md)。
- FakeCUDA 覆盖项目维护的 Python 和 PyTorch 行为。二进制 CUDA 扩展不在覆盖
  范围内。
- 静态分析无法解析所有动态 import、生成式 kernel、运行时 shape 或依赖数据的
  分支。
- 显存估算可能遗漏 backend 私有分配、自定义 operator、allocator 策略和未匹配的
  workspace。
- cuBLAS 中没有数值实现的算法、GEMM/Lt 不支持的 dtype、布局和指针模式，以及
  跳过 CPU 计算的分支，都受 `FAKEGPU_UNSUPPORTED_API` 控制。设为 `warn` 时记录
  警告。设为 `error` 时，调用会在写入输出前返回 `CUBLAS_STATUS_NOT_SUPPORTED`。
- 对带视觉编码器的模型，`estimate-llm` 和 `plan-serving` 只计入 checkpoint
  权重，不含 vision/projector 激活与 FLOP。报告中会标注这一点。decoder token
  数需要已经包含视觉 token。
- Diffusion profile 只描述固定的参考 pipeline。自定义 ControlNet、IP-Adapter、
  LoRA、safety checker、refiner、视频和 DiT 组件需要额外的元数据及真实 GPU 验证。
- 在线 serving 规划不估算请求到达分布、cache eviction、preemption 或重排、
  self-speculative/Medusa/EAGLE 变体、tokenizer 重映射、tensor parallelism、
  吞吐量和延迟。
- Roofline 输出是分析区间，不是实测 kernel 延迟。
- 分布式耗时包含 coordinator、内存复制、socket 和进程调度，不能当作 NCCL、
  NVLink 或 RDMA benchmark。
- 建模故障事件只出现在报告里。它们不会改变 CUDA 执行，也不代表真实的 ECC/Xid
  数据。
- Hybrid 和 passthrough 模式需要兼容的物理 CUDA 环境。
- macOS System Integrity Protection 可能删除系统程序的 `DYLD_*` 环境变量。
  原生拦截建议使用 Homebrew、conda 或 pyenv 的 Python。

## 尚在计划中

- 扩展可执行的原生 CUDA 运算和 cuBLAS 覆盖范围。
- 在更多 GPU 和软件栈上验证分布式与 MoE 估算。

其他提议和已知问题见
[open issues](https://github.com/FanBB2333/FakeGPU/issues)。
