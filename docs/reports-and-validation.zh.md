# 报告与验证

这页汇总 FakeGPU 自带的测试入口，以及运行时会生成哪些报告文件。

## 维护中的测试入口

| 命令 | 覆盖内容 |
|---|---|
| `./ftest smoke` | 构建、预加载、fake 设备发现、报告结构、多架构 profile、指针内存类型 |
| `./ftest cpu_sim` | CPU-backed cuBLAS / cuBLASLt 与 CPU 参考结果的一致性 |
| `./ftest python` | 基础 PyTorch CUDA 设备、张量和 matmul 路径 |
| `python3 verification/test_coordinator_smoke.py` | coordinator 启停与请求/响应闭环 |
| `python3 test/test_allreduce_correctness.py` | direct all-reduce 语义正确性 |
| `python3 verification/test_allgather_correctness.py` | direct all-gather 语义正确性 |
| `python3 verification/test_group_semantics.py` | grouped collective 提交语义 |
| `./test/run_hybrid_multinode.sh 2` | hybrid 计算 + simulate 通信的维护中多进程验证 |
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
  "report_version": "1.5.0",
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

当开启分布式并设置 `FAKEGPU_CLUSTER_REPORT_PATH` 后，FakeGPU 还会再写一份 cluster 级报告。

里面通常包括：

- cluster mode、world size、node count、coordinator transport
- 各类 collective 的调用次数、字节数、估算耗时
- 节点内 / 节点间链路统计
- 各 rank 的等待时间、超时次数、communicator 初始化次数、collective 次数

这份报告很适合用来验证控制流、拓扑模型，以及通信量的大致趋势。

## 统一 HTML 测试报告

`test/report.html` 是一个自包含的 HTML 报告，带有 tab 导航，覆盖：

- **Phase 1** — 设备发现与 profile 暴露
- **Phase 2** — 训练流程（nanoGPT on fake GPU）
- **Phase 3** — MoE 架构验证
- **Phase 4** — 错误模拟实验（6 类场景，23 个测试）

重新生成：

```bash
python test/run_error_simulation_suite.py
```

该报告可以与 mkdocs 站点一起部署在 `/test/report.html`。

## 稳定性建议

下面这些路径可以视为当前最稳定的基线：

- `smoke`
- `cpu_sim`
- `python`
- 单机 `simulate + simulate`
- direct NCCL 验证加 simulate-mode DDP 验证（`test_coordinator_smoke.py`、`test_allreduce_correctness.py`、`test_allgather_correctness.py`、`test_group_semantics.py`、`run_multinode_sim.sh`、`run_ddp_multinode.sh`）

下面这些路径更依赖环境，或者属于扩展覆盖：

- `hybrid` 分布式运行
- `proxy` / `passthrough` 分布式模式
- 依赖本地模型文件和更广框架覆盖的 LLM smoke 路径

## 推荐验证顺序

1. 先完成构建。
2. 跑 `./ftest smoke`。
3. 跑 `./ftest cpu_sim`。
4. 如果装了 PyTorch，再跑 `./ftest python`。
5. 跑 `python3 verification/test_coordinator_smoke.py`。
6. 跑 `python3 test/test_allreduce_correctness.py`。
7. 跑 `python3 verification/test_allgather_correctness.py`。
8. 跑 `python3 verification/test_group_semantics.py`。
9. 跑 `./test/run_multinode_sim.sh 2`。
10. 跑 `./test/run_multinode_sim.sh 4`。
11. 跑 `./test/run_ddp_multinode.sh 4`。
12. 然后再进入 `./test/run_hybrid_multinode.sh 2`。
13. 跑 `python test/run_error_simulation_suite.py` 做错误模拟覆盖。
