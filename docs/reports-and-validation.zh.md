# 报告与验证

这页汇总 FakeGPU 自带的测试入口，以及运行时会生成哪些报告文件。

## 维护中的测试入口

| 命令 | 覆盖内容 |
|---|---|
| `./ftest smoke` | 构建、预加载、fake 设备发现、报告结构、多架构 profile、指针内存类型 |
| `./ftest cpu_sim` | CPU-backed cuBLAS / cuBLASLt 与 CPU 参考结果的一致性 |
| `./ftest python` | 基础 PyTorch CUDA 设备、张量和 matmul 路径 |
| `./test/run_multinode_sim.sh 2` | 最小分布式模拟 smoke |
| `./test/run_ddp_multinode.sh 4` | DDP 风格多 rank 路径 |
| `./ftest llm` | 在本地模型文件可用时运行的可选 LLM smoke test |

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
  "report_version": 4,
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

## 稳定性建议

下面这些路径可以视为当前最稳定的基线：

- `smoke`
- `cpu_sim`
- `python`
- 单机 `simulate + simulate`

下面这些路径更依赖环境，或者还偏实验性质：

- `hybrid` 分布式运行
- `proxy` / `passthrough` 分布式模式
- 依赖本地模型文件和更广框架覆盖的 LLM smoke 路径

## 推荐验证顺序

1. 先完成构建。
2. 跑 `./ftest smoke`。
3. 跑 `./ftest cpu_sim`。
4. 如果装了 PyTorch，再跑 `./ftest python`。
5. 然后再进入 `./test/run_multinode_sim.sh 2`。
