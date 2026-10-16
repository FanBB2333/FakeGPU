# 分布式规划与实验批量运行

[文档索引](README.md) · [English](../en/distributed.md) ·
[繁體中文](../zh-TW/distributed.md)

## 分布式训练规划

`plan-training` 读取 DeepSpeed、Accelerate、FSDP 和 FSDP2 配置，估算每个 rank 的
显存。

对于 FSDP2 `full_shard`，`plan-training --parameter-shapes layout.json` 接受
`{"shapes": [[12, 8], [12], [3, 12], [3]]}`，同时需要一致的 `--parameter-bytes`
总量和统一的 `--parameter-element-bytes`。它按原生方式对每个参数沿第 0 维切 chunk，
并以最大的 rank 做预算。没有提供 shape 时退回均衡切分近似。自定义 placement 不在
建模范围内。

[版本化原生验证矩阵](../../tests/data/research_native_evidence.json)记录了六组
FP32/FP16/BF16 LoRA 与 attention 的 CPU_SIM 用例、tiny Qwen3 MoE 在 CPU 和 GPU 上
的路由与梯度，以及两个 CPU rank 上运行的原生 DDP 和 FSDP2。物理 TP/PP/CP/EP、
多 GPU ZeRO/FSDP 和多机 collective 不在当前范围内。复现入口为 `scripts/test.sh cpu`
和 `tests/native/research_distributed.py --help`。

## 拓扑、trace 与带宽

- `simulate-topology` 模拟 collective 路由和链路竞争。
- `replay-trace` 从 PyTorch Profiler、NCCL 风格或 FakeGPU trace 中汇总计算、通信、
  等待和显存时间线。
- `bandwidth` 验证模拟 TCP payload 并报告吞吐量。
- `coordinator` 管理分布式模拟 coordinator。

分布式耗时包含 coordinator、内存复制、socket 和进程调度，不能当作 NCCL、NVLink
或 RDMA benchmark。

## 实验批量运行

有限的批量运行使用显式的 argv 数组、参数网格和重复次数。各次运行按顺序执行，并绑定
稳定的运行 ID。每次运行都会保留 stdout、退出状态、错误、产物哈希和数值指标。

```bash
python -m fakegpu plan-experiments --config examples/experiments/cpu-sweep.json \
  --workspace /tmp/research-sweep --json /tmp/research-plan.json
python -m fakegpu run-experiments --config examples/experiments/cpu-sweep.json \
  --workspace /tmp/research-sweep --json /tmp/research-results.json.gz
python -m fakegpu run-experiments --config examples/experiments/cpu-sweep.json \
  --workspace /tmp/research-sweep --resume --json /tmp/research-resumed.json
python -m fakegpu compare-experiments --workspace /tmp/research-sweep \
  --plot /tmp/research-sweep.svg --x size --metric seconds \
  --json /tmp/research-comparison.json
```

- `--resume` 跳过已核验的成功项。`--resume --retry-failed` 还会重试失败或中断的项。
- 输入、源码或环境改变后需要新的工作区。
- 命令必须在前台运行。指标路径支持嵌套 JSON。
- 绘图需要 Matplotlib。误差棒表示观测到的最小值和最大值，不是置信区间。
- 示例测量的是 CPU 工作量，不代表 GPU 吞吐。
