# TODOs / Roadmap: Real-GPU Passthrough + No-GPU Kernel Simulation

目标：同时实现两种运行模式，并尽可能做到 **输出 token/数值对齐真实 GPU**。

> 说明：这里的“对齐”建议分层定义（见下文 *Definition of Done*）。**bitwise 完全一致**在跨设备/跨实现下通常不现实，但可以先做到“token 一致 + logits/allclose”并逐步逼近。

---

## Definition of Done（建议分层验收）

- **L0（可跑）**：模型能加载、forward 能完成、不崩溃（当前 FakeGPU 主要做到这一级）。
- **L1（token 对齐）**：固定 seed + 固定推理配置下（禁用随机/非确定性），生成 token 与真实 GPU 一致。
- **L2（数值对齐）**：关键张量/最终 logits 与真实 GPU 在指定容差内 `allclose(rtol, atol)`。
- **L3（bitwise 对齐，选做）**：关键张量 bitwise 相同（极难，通常只在同架构同实现+严格控制下才可能）。

建议优先实现 **L1 → L2**，L3 作为研究性目标。

---

## Glossary

- **Reported Device（虚拟设备）**：通过 NVML / CUDA API 返回给上层框架的设备信息（name/memory/cc 等）。
- **Backing Device（真实设备）**：实际执行计算/分配显存的真实 GPU（模式 1）。
- **Fake Memory（假显存）**：FakeGPU 现在用系统 RAM 模拟的“device memory”（模式 2）。
- **Passthrough**：拦截不改语义，尽量调用真实 CUDA 栈，保证结果与直接运行一致。
- **Hybrid**：设备信息可虚拟化，但计算/分配尽量走真实 GPU，并提供 OOM 安全策略。

---

## 总体架构改造（两种模式共享）

- [x] **统一配置入口**
  - [x] `FAKEGPU_MODE={simulate,passthrough,hybrid}`（默认 `simulate` 保持现状）。
  - [ ] CLI：`fakegpu --mode ...`、`./fgpu --mode ...`、`fakegpu.init(mode=...)`（Python wrapper）。
  - [x] 为所有 mode 提供清晰的 fallback：无法加载真实 CUDA 时自动回退到 `simulate` 并提示。

- [x] **Backend 抽象**
  - [x] 为 `libcuda/libcudart/libcublas/libcublasLt/libnvidia-ml` 引入统一的 *dispatch layer*：
    - `FakeBackend`（现有 stub + CPU sim）
    - `RealBackend`（dlopen/dlsym 转发到真实库）
    - `HybridBackend`（设备信息/内存策略可能由 Fake 控制，计算/部分 API 转发到 Real）
  - [x] 目标：每个 API 都能在运行时按 mode 选择实现，避免散落 `if (env)`。

- [ ] **测试矩阵与基准数据（Golden）**
  - [ ] 新增 `verification/golden/` 生成与读取工具（只在"有 GPU"的机器上生成）。
  - [ ] Golden 里至少保存：
    - prompt、tokenizer 版本、transformers/torch 版本、seed、关键 env
    - 生成的 token ids
    - 可选：最后一步 logits（float32）或若干关键中间张量摘要（hash/统计量）

---

## Mode 1：真实 GPU Passthrough/Hybrid（结果对齐 + OOM 安全）

### 1A. Pure Passthrough（最先落地：保证结果与直接跑一致）

目标：在有真实 GPU 的机器上，`./fgpu --mode passthrough ...` 的结果 **与不使用 FakeGPU 完全一致**（至少达到 L1/L2）。

- [x] **真实库定位与加载**
  - [x] 增加 `FAKEGPU_REAL_CUDA_LIB_DIR`（或自动从 `ldconfig`/常见路径探测）。
  - [x] 在每个 FakeGPU so 内部 `dlopen()` 真实库（例如 `/usr/local/cuda/.../libcuda.so.1` 等），并 `dlsym()` 真实符号。
  - [x] 避免递归：不要用 `RTLD_DEFAULT`；优先用显式 handle + `dlsym(handle, ...)`。

- [x] **完整转发链**
  - [x] `libcuda`：Driver API 全量转发（或至少覆盖 PyTorch/Transformers 路径）。
  - [x] `libcudart`：Runtime API 全量转发。
  - [x] `libcublas/libcublasLt`：全量转发。
  - [x] `libnvidia-ml`：可选择转发或仍用 Fake（取决于是否需要"虚拟设备展示"）。

- [ ] **一致性与确定性**
  - [ ] 为 parity 测试提供推荐环境变量集合（如 `CUBLAS_WORKSPACE_CONFIG`、禁用 TF32、固定算法等）。
  - [x] 增加 `./ftest passthrough_parity`：同一脚本跑两次（直接 vs passthrough）并比较 token/logits。

验收：
- [ ] `test/test_load_qwen2_5.py` 在 `passthrough` 下生成 token 与直接跑一致（L1）。
- [ ] 可选：dump logits 并在 `allclose` 容差内（L2）。

---

### 1B. Hybrid：虚拟设备信息 + 真实计算（并保证不 OOM）

核心难点：**虚拟设备（比如宣称 80GB A100）与真实设备（比如 24GB 3090）不一致时，框架可能会分配超额导致 OOM**。

建议把 OOM 安全策略做成可配置的 policy，并明确"保证级别"：

- [x] **策略（按推荐顺序）**
  - [x] `oom_policy=clamp`（最稳）：report 的 `total_memory`/`memGetInfo` 等不超过真实 GPU；避免框架误判。
  - [x] `oom_policy=managed`（中等）：拦截 `cudaMalloc` → `cudaMallocManaged`（或统一走 managed），依赖 UVM 进行 oversubscription（性能差但可避免 OOM）。
  - [x] `oom_policy=mapped_host`（进阶）：超额部分用 `cudaHostAllocMapped` + `cudaHostGetDevicePointer` 返回"device pointer"，允许 kernel 访问主存（零拷贝）。
  - [x] `oom_policy=spill+cpu`（混合兜底）：超额分配落到 Fake 内存；对可 CPU-sim 的算子走 CPU；不可 CPU-sim 的 kernel 直接报错（或强制回退到 simulate）。

- [x] **Reported vs Real 的一致性规则**
  - [x] `computeCapability` 建议 **不虚拟到高于真实**（避免 cubin 兼容问题）；name 可以虚拟，memory 可虚拟（但配合 policy）。
  - [x] NVML 与 CUDA Runtime/Driver 返回值保持一致（避免 torch/transformers 读到矛盾信息）。

- [x] **内存预算与压力感知**
  - [x] 在 Hybrid 下引入 "allocation tracker + budget"：
    - 真实显存 budget：通过真实 `cudaMemGetInfo`/NVML 查询。
    - 虚拟显存 budget：来自 profile。
  - [x] 对每次 `cudaMalloc/cudaMallocAsync/cuMemAlloc*` 做决策：走真实、走 managed、走 mapped_host、或 spill。

- [x] **可观测性**
  - [x] `fake_gpu_report.json` 增加：
    - backing GPU 信息（真实设备 id/name/memory）
    - 真实分配 vs managed vs mapped_host vs spilled 的统计
    - OOM 次数/回退次数

验收：
- [ ] 在 "虚拟 80GB、真实 24GB" 配置下，运行 LLM 推理脚本不因显存 OOM 直接失败（至少能通过 managed/mapped_host 策略跑完）。
- [ ] 在 `clamp` 策略下，保证行为与真实 GPU 接近（不会因为虚报而 OOM）。

---

## Mode 2：无 GPU 的 CUDA Kernel 执行/仿真（大工程）

目标：在没有真实 GPU 的环境下，FakeGPU 仍能执行 kernel 并尽量达到 L1/L2（Qwen2.5 起步）。

> 关键现实：PyTorch/Transformers 的 CUDA 路径大量依赖自定义 kernel + 库 kernel。要做到 L1/L2，必须让这些 kernel **真的产生正确结果**。

---

### 2A. 先做可观测性：Kernel Trace + Coverage 报告（强烈建议先落地）

- [ ] **捕获模块与 kernel 信息**
  - [ ] 在 `cuModuleLoad/cuModuleLoadData/cuModuleGetFunction` 中提取：
    - fatbin/cubin/PTX 原始 bytes（可选落盘）
    - kernel 名称、函数句柄映射
  - [ ] 在 `cuLaunchKernel` 中记录：
    - grid/block dims、sharedMemBytes、stream
    - kernelParams/extra 的参数布局（尽力解析）

- [ ] **Trace 输出与可复现**
  - [ ] 新增 `FAKEGPU_KERNEL_TRACE_PATH=/path`：输出可重放的 JSON trace。
  - [ ] 提供 `verification/replay_kernel_trace.py`（后续用来回归）。

- [ ] **Coverage 报告**
  - [ ] 跑 `test/test_load_qwen2_5.py` 时输出 “Top kernels by calls/time/bytes”。
  - [ ] 自动生成 “缺失 kernel 列表 + 需要的 PTX 特性列表”。

验收：
- [ ] 跑一次 Qwen，能得到稳定的 kernel 调用清单，为后续实现提供优先级。

---

### 2B. 设计 Kernel 执行框架（可插拔执行器）

- [ ] **KernelExecutor 接口**
  - [ ] `NoopExecutor`（现状）
  - [ ] `BuiltinExecutor`：对“已知 kernel”用手写 CPU 实现（最快见效）
  - [ ] `PTXInterpreterExecutor`：解释执行 PTX 子集（通用性更强）
  - [ ] `ExternalSimulatorExecutor`：可选接外部模拟器（如 gpgpu-sim/accel-sim，进程外执行）

- [ ] **内存模型**
  - [ ] global memory：沿用 FakeGPU allocation（host malloc）。
  - [ ] shared/local：在 CPU 端按 block/thread 分配并模拟寻址。
  - [ ] 原子/同步：先实现最常见子集（`bar.sync`、basic atomics），不够再扩展。

---

### 2C. 最快达到 L1：优先做 “BuiltinExecutor（算子级 CPU 参考实现）”

思路：先不追求任意 kernel 通用执行，而是把 Qwen 推理路径上的关键 kernel 用 CPU reference 写出来，并通过 kernel 名称/签名匹配来调用。

- [ ] **选定最小闭环目标：Qwen2.5-0.5B 单步 forward**
  - [ ] embedding gather
  - [ ] RMSNorm
  - [ ] RoPE
  - [ ] attention（建议强制 `TORCH_SDPA_KERNEL=math`，把注意力拆成 matmul+softmax+matmul）
  - [ ] softmax（含 mask）
  - [ ] SiLU + MLP（其中 Linear 已由 cuBLAS CPU-sim 覆盖）
  - [ ] 残差 add / elementwise mul

- [ ] **Kernel 匹配机制**
  - [ ] 基于 kernel name 前缀/正则匹配（trace 得到）。
  - [ ] 若 name 不稳定：用 module hash + 参数规模特征匹配（shape/stride/dtype）。

- [ ] **数值一致性策略**
  - [ ] 用与 GPU 接近的 dtype/舍入（fp16/bf16），必要时在关键点用 fp32 accumulation 模拟 GPU 行为。
  - [ ] 固定随机种子与禁用非确定性路径（文档化）。

验收：
- [ ] 无 GPU 环境下，Qwen 测试脚本生成 token 与真实 GPU 一致（L1）。
- [ ] 可选：logits `allclose`（L2）。

---

### 2D. 通用化：PTX 子集解释器 / JIT（长期）

当 builtin 的覆盖越来越难维护/扩展时，引入 PTX 解释/JIT。

- [ ] **PTX 获取**
  - [ ] 从 fatbin 中优先提取 PTX（如果存在）。
  - [ ] 如果只有 cubin：需要 SASS 解释器或外部模拟器（见 2E）。

- [ ] **PTX Interpreter MVP（先能跑再补齐）**
  - [ ] 基础指令：ld/st、add/mul/fma、type convert、predication、bra。
  - [ ] 特殊寄存器：`%tid.%ctaid.%ntid.%nctaid` 等。
  - [ ] memory space：global/shared/local/const。
  - [ ] 同步：`bar.sync`。
  - [ ] half/bf16：实现 IEEE 舍入与 NaN/Inf 规则（尽量贴近 CUDA）。

- [ ] **并行执行**
  - [ ] CPU 端用线程池/OpenMP 模拟 grid/block 并行。
  - [ ] 提供 deterministic 模式（单线程或固定调度）用于对齐。

验收：
- [ ] 能执行一批常见 elementwise/reduction kernel，并通过单元测试验证。

---

### 2E. cubin/SASS 路径（可能绕不开）

现实：很多发行版可能只带 cubin（或多架构 fatbin），PTX 不一定总可用。

- [ ] **策略选择**
  - [ ] 方案 1：要求/引导构建或运行时提供 PTX（可通过设置/编译选项达成）。
  - [ ] 方案 2：集成 SASS 解释器（极大工程）。
  - [ ] 方案 3：外部模拟器（gpgpu-sim/accel-sim）离线/子进程执行，并做 IPC。

建议：先用 **trace 驱动** 评估 PTX 可得性，再决定是否投入 2E。

---

## 测试与工程化

- [ ] **新增 ftest suites**
  - [ ] `./ftest passthrough_parity`
  - [ ] `./ftest hybrid_oom`（用小脚本主动申请超过真实显存的场景，验证 policy）
  - [ ] `./ftest kernel_trace`（输出 trace + coverage）
  - [ ] `./ftest llm_parity`（真实 GPU 机器：直接 vs passthrough/hybrid/simulate 对比）

- [ ] **新增对齐测试**
  - [ ] `test/test_llm_parity_token.py`：对比 token ids
  - [ ] `test/test_llm_parity_logits.py`：对比 logits（容差可配置）

- [ ] **文档**
  - [ ] 在 `README.md` 增加 mode 说明、OOM policy 说明、对齐等级说明（L0-L3）
  - [ ] 在 `test/README_QWEN_TEST.md` 补充 parity 流程

---

## 风险与开放问题（需要尽早决策）

- [ ] **“数值对齐”的标准**：你希望 L2 的容差是多少？是否需要层级/逐层对齐？
- [ ] **PTX 可得性**：目标环境下 PyTorch kernels 是否带 PTX？如果只带 cubin，2E 成本会陡增。
- [ ] **性能目标**：无 GPU 的 kernel 仿真一般会慢几个数量级；是否接受只跑小 batch/短序列？
- [ ] **维护成本**：builtin kernel 的匹配可能随 torch/transformers 版本变化，需要版本锁定或自动化 trace 更新。

