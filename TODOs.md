# TODOs / Roadmap: AI Researcher Preflight OOM Validation

目标：让 AI researcher 在提交训练或推理任务到大型 GPU 集群前，可以在本地先运行同一条命令，判断它是否能跑到指定阶段、是否会 OOM，以及峰值显存离目标 GPU 容量还有多少余量。

当前可用的真实校准机器：单张 NVIDIA GeForce RTX 3090 Ti，24GB 显存，Ampere 架构。下一版设计需要明确利用这张卡能验证什么，不能验证什么。

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

- [ ] 用一条命令运行 preflight：
  `fakegpu preflight --devices a100:8 --stage train_step -- python train.py ...`
- [ ] 得到机器可读 JSON 和人可读 Markdown 报告。
- [ ] 报告给出明确状态：
  - `PASS_FIT`：指定阶段完成，未触发已跟踪 OOM。
  - `FAIL_OOM`：触发 OOM 或目标 profile 显存不足。
  - `FAIL_RUNTIME`：依赖、导入、数据、模型加载或代码逻辑失败。
  - `WARN_INCOMPLETE_TRACKING`：运行完成，但显存跟踪范围不足以给出强结论。
- [ ] 报告包含每张 logical GPU 的：
  - `total_memory`
  - `peak_memory`
  - `headroom_bytes`
  - `headroom_percent`
  - `allocation_count`
  - `largest_allocations`
  - `tracking_confidence`
- [ ] OOM 时报告：
  - 失败阶段
  - 异常类型和错误信息
  - 最近一次大分配
  - 当前 profile 和设备数量
- [ ] 缺少 torch、transformers、本地模型、数据文件或真实 CUDA 时，不能静默跳过并报告成功。
- [ ] 文档明确不同 runtime 的可信度边界。

---

## Glossary

- **Preflight**：提交集群前的本地预检查。目标是确认命令能跑到指定阶段，并给出显存风险判断。
- **Stage**：preflight 要跑到的最小阶段，例如 `import`、`model_load`、`forward`、`backward`、`optimizer_step`、`n_steps`。
- **Target Profile**：目标集群上的 GPU 规格，例如 `a100`、`h100`、`b200` 或混合配置 `a100:4,h100:4`。
- **Calibration GPU**：本地真实 GPU。当前是 3090 Ti，用来校准真实 CUDA 路径和 24GB 边界。
- **Tracking Confidence**：显存报告的可信度等级。它描述“这份报告覆盖了哪些分配”，不是模型能否正确训练的证明。

---

## Current Scope For RTX 3090 Ti

单张 3090 Ti 可以支持：

- [x] 验证真实 CUDA / PyTorch / transformers 环境是否能跑。
- [x] 在 24GB 真实显存内校准 `passthrough` 和 `hybrid clamp` 路径。
- [x] 验证小模型、短序列、小 batch 的真实峰值显存。
- [x] 验证 24GB 以下 workload 的真实 OOM 行为。
- [x] 对 fakecuda 的 profile 显存检查做 sanity check。

单张 3090 Ti 不能直接支持：

- [ ] 证明 80GB A100/H100 上一定能跑完整大 batch。
- [ ] 证明真实多卡 NCCL / NVLink / InfiniBand 行为。
- [ ] 预测大集群利用率、吞吐或排队后的性能。
- [ ] 验证需要多张真实 GPU 才会暴露的通信或 sharding 问题。

因此下一版的默认策略是：

1. 用 3090 Ti 做真实 CUDA 校准和小规模 sanity check。
2. 用 fakecuda / simulate profile 做目标 GPU 显存 preflight。
3. 在报告中明确标出 confidence，不把估算当成真实集群证明。

---

## P0: 清理旧路线图和范围漂移

- [x] 保留已实现的统一配置入口：
  - `FAKEGPU_MODE={simulate,passthrough,hybrid}`
  - `fakegpu --mode ...`
  - `./fgpu --mode ...`
  - `fakegpu.init(mode=...)`
- [ ] 删除或迁移旧 TODO 中近期无关项：
  - token/logits parity
  - PTX interpreter
  - SASS / cubin interpreter
  - ExternalSimulatorExecutor
  - Qwen token 对齐
  - 深度 NCCL 协议仿真
- [ ] 保留通用修复：
  - profile 一致性
  - report schema
  - strict test
  - OOM policy 验证
  - hybrid 3090 Ti 校准
- [ ] README 和 docs 中避免把 simulate/fakecuda 说成“完整 CUDA 数值执行”。

验收：

- [ ] `TODOs.md` 只描述 preflight / OOM validation 近期目标。
- [ ] 长期研究项移动到后续 milestone 或单独设计文档，不阻塞下一版。

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
- [ ] 完善 strict pytest 语义：
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
- [ ] 区分内存类别：
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
- [x] 支持 `--memory-safety-factor`，可把 3090 Ti 校准得到的 factor 用于保守 fit/OOM 判定。
- [x] 报告 `tracking_confidence`：
  - `C0_incomplete`：只跑通流程，不适合判断 OOM。
  - `C1_weight_storage`：主要覆盖权重和显式 storage。
  - `C2_torch_tensor_lifetime`：覆盖 torch 层 tensor 生命周期，适合 fakecuda preflight。
  - `C3_native_cuda_allocations`：覆盖 C/CUDA allocation，适合 native simulate。
  - `C4_real_gpu_calibrated`：在 3090 Ti 上有真实 CUDA 对照。

验收：

- [ ] 现有 `test/real_scene/nanoGPT/TORCH_PATCH_PROOF.md` 中 “op-produced activation 未计入” 的限制被替换为新的通过实验。
- [x] `torch.cuda.max_memory_allocated()` 与 preflight report 的峰值一致。
- [x] `a100-1g` 下可稳定触发 OOM。

---

## P4: OOM 行为验证

- [x] 新增 `./ftest preflight_oom`。
- [ ] 测试小显存 profile：
  - [x] `a100-1g`
  - [x] 自定义 512MB profile（仅用于测试）：`test-512m`
- [ ] 测试大显存 profile：
  - [x] `a100`
  - [ ] `h100`
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
- [x] 在 3090 Ti 机器上额外运行真实 CUDA 校准用例。

---

## P5: 3090 Ti 校准流程

这部分专门面向当前可用硬件。

- [x] 新增 `./ftest rtx3090ti_calibration`。
- [x] 新增 `profiles/rtx3090ti.yaml`，用于文档、报告和校准元数据；它不是目标集群 profile 的替代品。
- [ ] 校准内容：
  - [x] 真实 PyTorch `torch.cuda.max_memory_allocated()`
  - [ ] `passthrough` 模式峰值显存
  - [ ] `hybrid --oom-policy clamp` 峰值显存
  - [x] fakecuda preflight 峰值显存
- [ ] 使用小模型和可控张量，避免超过 24GB：
  - [x] MLP
  - [x] Tiny Transformer
  - [ ] HF tiny model
  - [ ] LoRA tiny flow
  - [x] 受控 tensor allocation probe
  - [ ] 手动大 tensor OOM probe
- [x] 生成校准报告：
  - [x] `calibration_rtx3090ti.json`
  - [x] `calibration_rtx3090ti.md`
- [x] 生成误差诊断字段：
  - [x] `calibration_factor`
  - [x] `gap_analysis`
  - [x] `likely_gap_reason`
- [x] 记录误差，不要求完全一致。
- [x] 如果真实 CUDA 不可用，测试必须报告 skip/fail 原因，不能静默通过。

验收：

- [x] 3090 Ti 上同一 workload 的 fakecuda peak 与真实 peak 误差被记录。
- [x] 文档说明该误差只能作为当前实现校准，不代表 A100/H100 性能。

---

## P6: 典型 AI Workload 覆盖

优先覆盖研究者最常见的提交前检查，不追求大模型完整训练。

- [ ] Tiny Transformer：
  - forward
  - backward
  - optimizer step
- [ ] HF Trainer 单卡 smoke。
- [ ] LoRA / PEFT 小模型训练 smoke。
- [ ] AMP：
  - fp16
  - bf16 profile capability check
- [ ] Gradient accumulation。
- [ ] Gradient checkpointing。
- [ ] DDP 2-rank 单机 smoke。
- [ ] FSDP 基础 smoke。

验收：

- [ ] 每个 workload 都有一个 preflight 示例命令。
- [ ] 每个 workload 都能在报告里看到 stage、peak、status。

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
- [x] `docs/quick-reference.md` 增加 3090 Ti 校准命令。
- [ ] 明确限制：
  - PASS 不代表训练结果数值正确。
  - PASS 不代表目标集群性能。
  - fakecuda profile 可以模拟 80GB 显存上限，但不等价于真实 A100/H100 kernel 行为。
  - 3090 Ti 只能校准 24GB 内的真实 CUDA 行为。

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
