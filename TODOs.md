# TODOs / Roadmap: AI Researcher Preflight OOM Validation

目标：让 AI researcher 在提交训练或推理任务到大型 GPU 集群前，可以在本地先运行同一条命令，判断它是否能跑到指定阶段、是否会 OOM，以及峰值显存离目标 GPU 容量还有多少余量。

当前可用的真实校准机器：单张 NVIDIA RTX PRO 5000 72GB Blackwell，Compute Capability 12.0。真实 GPU 校准必须记录检测到的硬件和匹配的 FakeGPU profile，不能沿用旧 3090 Ti 假设。

本路线图暂不追求：

- GPU SM 利用率、occupancy、吞吐或 step time 预测
- token/logits 数值对齐
- 通用 CUDA kernel 执行
- PTX / SASS 解释器
- NCCL / RDMA / NVLink 协议级仿真
- 真实多机多卡集群性能预测

---

## Definition of Done

下一版完成后，用户至少可以做到：

- [x] 用一条命令运行 preflight：
  `fakegpu preflight --devices a100:8 --stage train_step -- python train.py ...`
- [x] 得到机器可读 JSON 和人可读 Markdown 报告。
- [x] 报告给出明确状态：
  - `PASS_FIT`：指定阶段完成，未触发已跟踪 OOM。
  - `FAIL_OOM`：触发 OOM 或目标 profile 显存不足。
  - `FAIL_RUNTIME`：依赖、导入、数据、模型加载或代码逻辑失败。
  - `WARN_INCOMPLETE_TRACKING`：运行完成，但显存跟踪范围不足以给出强结论。
- [x] 报告包含每张 logical GPU 的：
  - `total_memory`
  - `peak_memory`
  - `headroom_bytes`
  - `headroom_percent`
  - `allocation_count`
  - `largest_allocations`
  - `tracking_confidence`
- [x] OOM 时报告：
  - 失败阶段
  - 异常类型和错误信息
  - 最近一次大分配
  - 当前 profile 和设备数量
- [x] 缺少 torch、transformers、本地模型、数据文件或真实 CUDA 时，不能静默跳过并报告成功。
- [x] 文档明确不同 runtime 的可信度边界。

---

## Glossary

- **Preflight**：提交集群前的本地预检查。目标是确认命令能跑到指定阶段，并给出显存风险判断。
- **Stage**：preflight 要跑到的最小阶段，例如 `import`、`model_load`、`forward`、`backward`、`optimizer_step`、`n_steps`。
- **Target Profile**：目标集群上的 GPU 规格，例如 `a100`、`h100`、`b200` 或混合配置 `a100:4,h100:4`。
- **Calibration GPU**：本地真实 GPU。当前是 RTX PRO 5000 72GB Blackwell，用来校准真实 CUDA 路径和这张卡的实际显存边界。
- **Tracking Confidence**：显存报告的可信度等级。它描述“这份报告覆盖了哪些分配”，不是模型能否正确训练的证明。

---

## Current Scope For The Real Calibration GPU

单张 RTX PRO 5000 可以支持：

- [x] 在更换后的 RTX PRO 5000 上验证真实 CUDA / PyTorch / transformers 环境。
- [x] 在真实显存范围内校准 `passthrough` 和 `hybrid clamp` 路径。
- [x] 在 RTX PRO 5000 上验证小模型、短序列、小 batch 的真实峰值显存。
- [x] 验证这张卡实际显存边界以内的 workload 和真实 OOM 行为。
- [x] 对 fakecuda 的 profile 显存检查做 sanity check。

单张 RTX PRO 5000 不能直接支持：

- [ ] 证明 80GB A100/H100 上一定能跑完整大 batch。
- [ ] 证明真实多卡 NCCL / NVLink / InfiniBand 行为。
- [ ] 预测大集群利用率、吞吐或排队后的性能。
- [ ] 验证需要多张真实 GPU 才会暴露的通信或 sharding 问题。

因此下一版的默认策略是：

1. 用 RTX PRO 5000 做真实 CUDA 校准和小规模 sanity check。
2. 用 fakecuda / simulate profile 做目标 GPU 显存 preflight。
3. 在报告中明确标出 confidence，不把估算当成真实集群证明。

---

## P0: 清理旧路线图和范围漂移

- [x] 保留已实现的统一配置入口：
  - `FAKEGPU_MODE={simulate,passthrough,hybrid}`
  - `fakegpu --mode ...`
  - `./fgpu --mode ...`
  - `fakegpu.init(mode=...)`
- [x] 删除或迁移旧 TODO 中近期无关项：
  - token/logits parity
  - PTX interpreter
  - SASS / cubin interpreter
  - ExternalSimulatorExecutor
  - Qwen token 对齐
  - 深度 NCCL 协议仿真
- [x] 保留通用修复：
  - profile 一致性
  - report schema
  - strict test
  - OOM policy 验证
  - hybrid 真实 GPU 校准
- [x] README 和 docs 中避免把 simulate/fakecuda 说成“完整 CUDA 数值执行”。

验收：

- [x] `TODOs.md` 只描述 preflight / OOM validation 近期目标。
- [x] 长期研究项移动到后续 milestone 或单独设计文档，不阻塞下一版。

---

## P1: Preflight Runner

新增 `fakegpu preflight` 子命令，作为研究者的主入口。

- [x] 支持基础参数：
  - `--devices a100:8`
  - `--profile h100 --device-count 8`
  - `--stage import|model_load|forward|backward|optimizer_step|n_steps`
  - `--steps N`
  - `--report-dir path`
  - `--runtime fakecuda|native|hybrid|passthrough`
  - `--strict`
- [x] 自动设置：
  - `FAKEGPU_PROFILES`
  - `FAKEGPU_DEVICE_COUNT`
  - `FAKEGPU_TERMINAL_REPORT`
  - `FAKEGPU_REPORT_PATH`
  - `FAKEGPU_CLUSTER_REPORT_PATH`（分布式时）
- [x] 运行用户命令并捕获：
  - stdout
  - stderr
  - exit code
  - OOM 异常
  - report 文件
- [x] 输出：
  - `preflight_report.json`
  - `preflight_report.md`
  - `preflight_stdout.log`
  - `preflight_stderr.log`
- [x] 支持初版严格模式：
  - 依赖缺失即失败
  - 模型文件缺失即失败
  - CUDA 不可用即失败（当 runtime 需要真实 CUDA 时）
- [x] 完善 strict pytest 语义：
  - pytest skip 不能算作通过

验收：

- [x] `fakegpu preflight --profile a100-1g --stage forward -- python demo_usage.py --test transformer` 能生成报告。
- [x] `--strict` 下缺依赖会返回非零退出码。

---

## P2: Stage Contract

Preflight 需要知道用户命令跑到了哪个阶段。下一版先提供轻量协议，不要求用户重写训练脚本。

- [x] 提供环境变量约定：
  - `FAKEGPU_PREFLIGHT_STAGE=import`
  - `FAKEGPU_PREFLIGHT_STAGE=model_load`
  - `FAKEGPU_PREFLIGHT_STAGE=forward`
  - `FAKEGPU_PREFLIGHT_STAGE=backward`
  - `FAKEGPU_PREFLIGHT_STAGE=optimizer_step`
- [x] 提供 Python helper：

```python
import fakegpu

with fakegpu.stage("model_load"):
    model = load_model()

with fakegpu.stage("forward"):
    loss = model(**batch).loss

with fakegpu.stage("backward"):
    loss.backward()
```

- [x] 没有显式 stage 标记时，runner 使用保守阶段：
  - 命令启动成功：`import`
  - 进程正常退出：`completed`
  - OOM：`unknown_or_last_seen`
- [x] 报告中记录 stage timeline。

验收：

- [x] 用户不用 helper 也能得到基础报告。
- [x] 使用 helper 后，OOM 报告能定位到阶段。

---

## P3: 显存跟踪可信度提升

当前 `torch_patch` 的报告已经覆盖显式 fake-CUDA storage、部分 op output，以及 PyTorch hooks 能看到的 autograd saved tensor。已知不足是 CUDA 后端内部 workspace、fused attention/optimizer temporary 等仍可能不可见。面向 OOM preflight，这一项仍是核心。

- [ ] 修复 op-produced tensor 跟踪：
  - [x] elementwise output
  - [x] matmul output
  - [x] loss output
  - [x] clone / contiguous 后的新 storage
  - [x] view 后共享 storage 的归属验证
  - [x] PyTorch hooks 能看到的 autograd saved tensor
  - [ ] CUDA 后端内部 workspace / fused attention temporary
- [x] 区分内存类别：
  - [x] parameters
  - [x] buffers
  - [x] gradients
  - [x] optimizer state
  - [x] activations
  - [x] temporary tensors
  - unknown
- [x] 支持分阶段峰值：
  - `peak_import`
  - `peak_model_load`
  - `peak_forward`
  - `peak_backward`
  - `peak_optimizer_step`
- [x] 支持 top allocations：
  - bytes
  - dtype
  - shape
  - device
  - stage
  - category
- [x] 支持 optional stack trace
- [x] 多设备场景下正确归属 logical device。
- [x] 支持 `--memory-safety-margin`，可把真实 GPU 校准得到的固定 missing bytes 用于更精确的 fit/OOM 判定。
- [x] 支持 `--memory-safety-factor`，可把真实 GPU 校准得到的 factor 用于保守 fit/OOM 判定。
- [x] 报告 `tracking_confidence`：
  - `C0_incomplete`：只跑通流程，不适合判断 OOM。
  - `C1_weight_storage`：主要覆盖权重和显式 storage。
  - `C2_torch_tensor_lifetime`：覆盖 torch 层 tensor 生命周期，适合 fakecuda preflight。
  - `C3_native_cuda_allocations`：覆盖 C/CUDA allocation，适合 native simulate。
  - `C4_real_gpu_calibrated`：在报告明确记录的真实 GPU 上有 CUDA 对照。

验收：

- [x] 现有 `test/real_scene/nanoGPT/TORCH_PATCH_PROOF.md` 已记录 common op-produced tensor output 跟踪实验。
- [x] `torch.cuda.max_memory_allocated()` 与 preflight report 的峰值一致。
- [x] `a100-1g` 下可稳定触发 OOM。

---

## P4: OOM 行为验证

- [x] 新增 `./ftest preflight_oom`。
- [x] 测试小显存 profile：
  - [x] `a100-1g`
  - [x] 自定义 512MB profile（仅用于测试）：`test-512m`
- [x] 测试大显存 profile：
  - [x] `a100`
  - [x] `h100`
- [x] 同一 workload 在小 profile 下 `FAIL_OOM`，在大 profile 下 `PASS_FIT`（`test-512m` -> `a100`）。
- [x] 验证 PyTorch OOM 表面：
  - [x] 异常类型接近 `torch.cuda.OutOfMemoryError`
  - [x] 错误信息包含 requested、total、free
- [x] 验证报告字段：
  - `status`
  - `stage`
  - `peak_memory`
  - `headroom_bytes`
  - `tracking_confidence`
  - `runtime`
- [x] `--strict` 模式下任何 skip 都失败。

验收：

- [x] `./ftest preflight_oom` 在没有真实 GPU 的环境中也能验证 fakecuda OOM。
- [x] 在更换后的 RTX PRO 5000 上运行真实 CUDA 校准用例。

---

## P5: 真实 GPU 校准流程

这部分专门面向当前可用硬件。

- [x] 新增 `./ftest real_gpu_calibration`，保留 `rtx3090ti_calibration` 兼容别名。
- [x] 新增 `profiles/rtx-pro-5000-blackwell.yaml`，并按检测到的 GPU 自动选择校准 profile。
- [x] 校准内容：
  - [x] 真实 PyTorch `torch.cuda.max_memory_allocated()`
  - [x] `passthrough` 模式峰值显存
  - [x] `hybrid --oom-policy clamp` 峰值显存与 Driver API 峰值
  - [x] fakecuda preflight 峰值显存
- [x] 使用小模型和可控张量，避免意外耗尽真实 GPU 显存：
  - [x] MLP
  - [x] Tiny Transformer
  - [x] HF tiny model
  - [x] LoRA tiny flow
  - [x] 受控 tensor allocation probe
  - [x] Hybrid clamp 超容量 tensor OOM probe
- [x] 生成校准报告：
  - [x] `calibration_real_gpu.json`
  - [x] `calibration_real_gpu.md`
- [x] 生成误差诊断字段：
  - [x] `calibration_factor`
  - [x] `missing_peak_bytes`
  - [x] `recommended_memory_safety_margin_bytes`
  - [x] `gap_analysis`
  - [x] `likely_gap_reason`
- [x] 记录误差，不要求完全一致。
- [x] 如果真实 CUDA 不可用，测试必须报告 skip/fail 原因，不能静默通过。

验收：

- [x] 在 RTX PRO 5000 上记录同一 workload 的 real、passthrough、hybrid clamp 与 fakecuda peak，并验证原生模式结果签名。
- [x] 文档说明该误差只能作为当前实现校准，不代表 A100/H100 性能。

---

## P6: 典型 AI Workload 覆盖

优先覆盖研究者最常见的提交前检查，不追求大模型完整训练。

- [x] Tiny Transformer：
  - forward
  - backward
  - optimizer step
- [x] HF tiny GPT-2 单卡训练步校准 workload。
- [x] LoRA / PEFT tiny GPT-2 训练步校准 workload。
- [x] HF Trainer 单卡 smoke。
- [x] AMP：
  - fp16
  - bf16 profile capability check
- [x] Gradient accumulation。
- [x] Gradient checkpointing。
- [x] DDP 2-rank 单机 smoke。
- [x] FSDP 基础 smoke。

验收：

- [x] 每个 workload 都有一个 preflight 示例命令。
- [x] 每个 workload 都能在报告里看到 stage、peak、status。

---

## P7: Report Schema

- [x] 定义 `preflight_report.schema.json`。
- [x] 报告字段包括：
  - `schema_version`
  - `fakegpu_version`
  - `git_commit`
  - `command`
  - `runtime`
  - `target_profiles`
  - `calibration_gpu`
  - `status`
  - `stage`
  - `devices`
  - `tracking_confidence`
  - `warnings`
  - `errors`
  - `logs`
- [x] 提供校验工具：
  - `verification/check_preflight_report.py`
- [x] Markdown 报告包含：
  - 一句话结论
  - 每卡峰值显存表
  - 失败原因
  - 可信度说明
  - 下一步建议

验收：

- [x] schema 校验加入 `./ftest preflight_oom`。
- [x] 报告可直接作为 Slurm 提交前的附件。

---

## P8: 文档

- [x] 新增 `docs/ai-researcher-preflight.md`。
- [x] 新增 `docs/ai-researcher-preflight.zh.md`。
- [x] README 增加 “AI researcher preflight” 入口。
- [x] `docs/reports-and-validation.md` 增加 preflight 报告说明。
- [x] `docs/quick-reference.md` 增加真实 GPU 校准命令。
- [x] 明确限制：
  - PASS 不代表训练结果数值正确。
  - PASS 不代表目标集群性能。
  - fakecuda profile 可以模拟 80GB 显存上限，但不等价于真实 A100/H100 kernel 行为。
  - 真实 GPU 校准结果只适用于报告中记录的硬件和 workload 类型。

---

## Deferred

以下内容从下一版移出，不再阻塞 AI researcher preflight：

- token 一致性与 logits allclose
- Qwen2.5 真实 GPU parity
- PTX interpreter
- SASS / cubin interpreter
- 外部 GPU 模拟器集成
- kernel 数值执行
- 完整 NCCL 协议复刻
- GPU 利用率预测
- step time / throughput 预测
