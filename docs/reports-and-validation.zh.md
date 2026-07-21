# 报告与验证

这页汇总 FakeGPU 自带的测试入口，以及运行时会生成哪些报告文件。

## 维护中的测试入口

| 命令 | 覆盖内容 |
|---|---|
| `./ftest smoke` | 构建、预加载、fake 设备发现、报告结构、多架构 profile、指针内存类型 |
| `./ftest cpu_sim` | CPU-backed cuBLAS / cuBLASLt 与 CPU 参考结果的一致性 |
| `./ftest python` | 基础 PyTorch CUDA 设备、张量和 matmul 路径 |
| `./ftest preflight_oom` | fakecuda fit/OOM 判定与报告 schema |
| `./ftest static_memory_validation` | fake-tensor ATen 前向/反向 storage 生命周期、optimizer 显存和可选真实 CUDA allocator 对比 |
| `./ftest real_gpu_calibration` | real/passthrough/Hybrid/fakecuda 显存与结果签名校准 |
| `fakegpu estimate-llm ...` | 仅检查 header 的 dense decoder 参数、KV cache、临时显存与矩阵 FLOPs 估算 |
| `verification/compare_qwen_memory.py ...` | 对比相同配置下的真实 CUDA/FakeCUDA 加载、推理、虚拟 SMI、token 与 FLOPs |
| `verification/compare_qwen_sft_memory.py ...` | 对比 Qwen3.5 全参数 SFT 的真实 CUDA、FakeCUDA 与 ATen 静态图峰值 |
| `python3 verification/test_coordinator_smoke.py` | coordinator 启停、请求/响应与正常关闭 |
| `python3 test/test_allreduce_correctness.py` | direct all-reduce 语义正确性 |
| `python3 verification/test_allgather_correctness.py` | direct all-gather 语义正确性 |
| `python3 verification/test_group_semantics.py` | grouped collective 提交语义 |
| `./ftest tcp_bandwidth` | 指定端口的 TCP 负载正确性与端到端模拟吞吐 |
| `./ftest distributed_resilience` | TCP collective 参数不一致、缺少 rank 超时和报告时间线容量限制 |
| `./test/run_hybrid_multinode.sh 2` | hybrid 计算 + simulate 通信的维护中多进程验证 |
| `python3 verification/run_hybrid_ddp_numerics.py --variant all` | 真实 CUDA DDP 基础路径、`no_sync`、未使用参数、静态图、bucket view、optimizer 与跨 rank 参数一致性 |
| `python3 verification/run_hybrid_fsdp_numerics.py` | 真实 CUDA FSDP 参数分片、reduce-scatter 梯度、optimizer 结果、完整参数重建与 state dict 恢复 |
| `python3 verification/run_hybrid_fsdp2_numerics.py ...` | 真实 CUDA FSDP2/DeviceMesh/DTensor 数值验证，覆盖双/四 rank、FP32/FP16/BF16 参数，以及 FP32 或参数 dtype 梯度归约 |
| `python3 verification/run_physical_multihost.py ...` | 通过 SSH 重复执行两主机 Hybrid DDP/FSDP/FSDP2、混合精度、参数不一致、超时、Git 版本和报告检查 |
| `./ftest llm` | 在本地模型文件可用时运行的可选 LLM smoke test |
| `python test/run_error_simulation_suite.py` | 统一错误模拟套件：跨设备、OOM、无效设备、dtype、checkpoint、梯度（23 个测试） |
| `python test/test_error_cross_device.py` | 跨设备张量操作守卫 |
| `python test/test_error_oom.py` | 每设备 OOM 模拟 |

前面三条是最适合在代码或构建变更后优先执行的基线验证。

## `fake_gpu_report.json`

进程退出时，FakeGPU 会写出 `fake_gpu_report.json`。如果设置了 `FAKEGPU_REPORT_PATH`，则会写到你指定的位置。

报告通常包含：

- 当前运行模式
- 每张 fake device 的条目
- 当前显存占用和峰值显存占用
- H2D / D2H / D2D / peer / memset 的 IO 计数
- 已维护 cuBLAS / cuBLASLt 路径的调用次数和 FLOP 估算
- host-to-host copy 计数

大致结构如下：

```json
{
  "report_version": "1.5.3",
  "mode": "simulate",
  "devices": [
    {
      "index": 0,
      "name": "Fake NVIDIA A100-SXM4-80GB",
      "used_memory_peak": 123456,
      "io": {
        "h2d": {"calls": 1, "bytes": 4096}
      },
      "compute": {
        "cublas_gemm": {"calls": 2, "flops": 8192}
      }
    }
  ]
}
```

## Cluster report

开启分布式并设置 `FAKEGPU_CLUSTER_REPORT_PATH` 后，FakeGPU 会写出 cluster
JSON 数据，并自动在同一目录生成 Markdown 项目报告。可以通过
`FAKEGPU_CLUSTER_REPORT_MARKDOWN_PATH` 指定 Markdown 路径，也可以将其
设为 `off`，只保留 JSON。
维护中的 DDP 和 Hybrid 验证脚本还会把这张完整节点对表直接写入最终
验证报告。

里面通常包括：

- cluster mode、world size、node count、coordinator transport
- 各类 collective 的调用次数、字节数、估算耗时
- P2P 操作数、发送次数和字节数
- 节点内 / 节点间的方向链路统计
- 配置中全部不同节点的两两组合，包括通信量为 0 的节点对
- collective/P2P 操作分类、每个方向及双向合计字节数、单次操作最大负载、传输次数、模型平均/峰值吞吐、估算耗时和争用惩罚
- 各 rank 的等待时间、超时次数、communicator 初始化次数、collective/P2P 次数
- 有大小限制的最近操作时间线，包含 communicator 全局 ranks、collective 数据类型/归约运算、逻辑和 socket 负载、汇合时间、coordinator 执行时间及拓扑模型时间

仓库根目录的 `cluster_report.schema.json` 定义 `cluster_report.v1` JSON
契约。`verification/check_cluster_report.py` 默认执行 schema 校验，也可以
通过 `--expect-point-to-point` 要求报告中存在 P2P 流量。默认校验还会把
全部已完成 collective/P2P 计数与时间线保留数、丢弃数相互核对。时间线
完整时，每类操作的次数和逻辑字节必须完全一致；方向链路的样本数、字节、
峰值、模型耗时和吞吐也必须与节点对中的对应方向一致。

Markdown 报告中的节点对表格如下：

| 节点 A | 节点 B | 双向总量 | 节点对单次峰值 | 操作数 | Collective 操作 | P2P 操作 | 传输数 | 模型平均 Gbit/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `node0` | `node1` | 8.00 GiB | 128.00 MiB | 64 | 60 | 4 | 128 | 18.420 |
| `node0` | `node2` | 0 B | 0 B | 0 | 0 | 0 | 0 | 0.000 |

“单次峰值”表示一次已完成通信操作中，归属于该方向或无序节点对的最大
负载。吞吐、耗时和争用数据来自 cluster topology model，不是网卡抓包，
也不是 NIC/NCCL 实测带宽。JSON 文件保留精确的整数字节计数，便于后续分析。

时间线中的 `coordinator_duration_us` 从第一个完整 rank 请求进入 registry
开始，到 coordinator 侧执行结束为止；它不包含客户端准备和最终响应送达。
默认保留最近 4096 条，可通过
`FAKEGPU_CLUSTER_REPORT_MAX_OPERATIONS` 调整。

这份报告很适合用来验证控制流、拓扑模型，以及通信量的大致趋势。

## Preflight report

面向 AI researcher 的 preflight 工作流会在现有 runtime 报告之上生成一份更高层的报告。

runner 会写出 `preflight_report.json`，并配套生成 `preflight_report.md`。它回答：用户命令是否跑到了指定阶段，是否在目标 GPU profile 下触发 OOM，以及还剩多少显存余量。JSON 契约由仓库根目录的 `preflight_report.schema.json` 定义，`verification/check_preflight_report.py` 默认会按这个 schema 校验报告。

状态如下：

| 状态 | 含义 |
|---|---|
| `PASS_FIT` | 指定阶段完成，未触发已跟踪 OOM。 |
| `FAIL_OOM` | 运行超过目标 profile，或抛出 OOM。 |
| `FAIL_RUNTIME` | 依赖、数据、模型加载、代码或环境设置失败。 |
| `WARN_INCOMPLETE_TRACKING` | 运行完成，但显存跟踪不够完整，不能给出强 fit/no-fit 判断。 |

每个 device entry 会包含总显存、峰值显存、余量、allocation 次数、`current_bytes_by_category`、`peak_by_stage` 和 `largest_allocations`。fakecuda 模式下，top allocations 会记录 bytes、dtype、shape、stage，以及 `parameter` / `buffer` / `gradient` / `optimizer_state` / `activation` / `temporary` / `tensor` 这类粗粒度 category。加上 `--allocation-stacks` 后，还会为这些 top allocations 记录短 Python stack trace。

静态显存验证报告会分别保留 forward、backward 和 optimizer 的 CUDA 峰值。workspace 字段会区分 profile 总字节数与实际影响峰值的增量：graph-phase persistent storage 作用于整个计算图，operator-local workspace 只与对应 ATen 节点的 live storage 相加。报告还会列出已由计算图覆盖和仍未匹配的 Attention 算子。

当前真实校准目标是一张 NVIDIA RTX PRO 5000 72GB Blackwell（Compute Capability 12.0）。维护中的套件覆盖七个受控 workload，要求 passthrough 与 Hybrid 的结果签名匹配 real CUDA，记录 Hybrid Driver allocation 峰值，并验证 Hybrid clamp 下的 PyTorch OOM 表现。它不能证明多机目标集群一定能容纳 workload，也不能证明性能表现。

用法和当前限制见 [AI Researcher 提交前预检查](ai-researcher-preflight.md)。

## LLM 推理估算与虚拟 SMI

`fakegpu estimate-llm --json <path>` 会写出
`fakegpu.llm_inference_estimate.v1` 报告，并且不会加载 checkpoint payload。
内容包括参数/checkpoint 元数据、KV cache、prefill/decode 临时显存、tensor
与进程峰值，以及逐 step 的矩阵 FLOPs。

如果在 FakeCUDA runtime 启动前设置 `FAKEGPU_SMI_STATE_PATH` 或
`FAKEGPU_SMI_STATE_DIR`，每个进程都会发布 `fakegpu.smi_state.v1` JSON。
`fakegpu nvidia-smi` 会显示当前及峰值 tracked tensor 显存，并可加入同一
软件与硬件路径实测得到的 runtime overhead。

Qwen 验证 worker 会分别记录模型加载和推理峰值；真实模式读取 NVML 进程
显存，无 GPU 模式隐藏物理 GPU 并用 CPU-backed FakeCUDA 执行，还会对比
实际 FLOPs 与形状估算。命令、实测结果和适用范围参见
[LLM 推理显存与计算量估算](llm-inference-estimation.md)。

Qwen3.5 全参数 SFT worker 会用固定随机 token 和 loss mask 对比真实 CUDA、
FakeCUDA 与静态训练图，并区分 AdamW 首步和稳态显存。命令、实测误差及
适用范围参见 [LLM SFT 显存估算](llm-sft-memory-estimation.md)。

## 统一 HTML 测试报告

`test/report.html` 是一个自包含的 HTML 报告，带有 tab 导航，覆盖：

- **Phase 1** — 设备发现与 profile 暴露
- **Phase 2** — 训练流程（nanoGPT on fake GPU）
- **Phase 3** — MoE 架构验证与 torch_patch 证明型实验
- **Phase 4** — 错误模拟实验（6 类场景，23 个测试）

重新生成：

```bash
python test/run_error_simulation_suite.py
```

该报告可以与 mkdocs 站点一起部署在 `/test/report.html`。

对应的 Markdown 证明报告见：

- `test/real_scene/nanoGPT/TORCH_PATCH_PROOF.md` — 包含 520M / 1.0B load-only 扩展实验、`a100-1g` OOM 验证，以及当前 fakecuda terminal summary 尚未覆盖大多数算子输出 activation 的范围说明。

## 稳定性建议

下面这些路径可以视为当前最稳定的基线：

- `smoke`
- `cpu_sim`
- `python`
- 单机 `simulate + simulate`
- 本地指定端口的 TCP collective 与带宽验证
- direct NCCL 验证加 simulate-mode DDP 验证（`test_coordinator_smoke.py`、`test_allreduce_correctness.py`、`test_allgather_correctness.py`、`test_group_semantics.py`、`run_multinode_sim.sh`、`run_ddp_multinode.sh`）

下面这些路径更依赖环境，或者属于扩展覆盖：

- `hybrid` 分布式运行
- 物理多机 TCP 测量
- `proxy` / `passthrough` 分布式模式
- 依赖本地模型文件和更广框架覆盖的 LLM smoke 路径

## 推荐验证顺序

1. 先完成构建。
2. 执行 `./ftest smoke`。
3. 执行 `./ftest cpu_sim`。
4. 如果装了 PyTorch，再执行 `./ftest python`。
5. 执行 `python3 verification/test_coordinator_smoke.py`。
6. 执行 `python3 test/test_allreduce_correctness.py`。
7. 执行 `python3 verification/test_allgather_correctness.py`。
8. 执行 `python3 verification/test_group_semantics.py`。
9. 执行 `./ftest tcp_bandwidth`。
10. 执行 `./ftest distributed_resilience`。
11. 执行 `./test/run_multinode_sim.sh 2`。
12. 执行 `./test/run_multinode_sim.sh 4`。
13. 执行 `./test/run_ddp_multinode.sh 4`。
14. 然后再进入 `./test/run_hybrid_multinode.sh 2`。
15. 在真实 CUDA 主机上执行 `python3 verification/run_hybrid_ddp_numerics.py --variant all`。
16. 在真实 CUDA 主机上执行 `python3 verification/run_hybrid_fsdp_numerics.py`。
17. 在真实 CUDA 主机上执行 `python3 verification/run_hybrid_fsdp2_numerics.py ...`，覆盖 FSDP2 参数与梯度精度组合。
18. 两台 GPU 主机同步到相同 commit 后，执行 `python3 verification/run_physical_multihost.py ...`。
19. 执行 `python test/run_error_simulation_suite.py`，检查错误模拟覆盖。
