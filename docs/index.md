# FakeGPU

FakeGPU 是一个 CUDA / NVML / cuBLAS 拦截库，用来在没有 GPU 的环境中暴露“可用的 GPU 形态”，让 PyTorch 以及其他依赖 CUDA 的程序先把设备发现、内存管理、基础算子和分布式控制流跑起来。

## 适合什么场景

- 在无 GPU 环境里验证 CUDA / PyTorch 控制流
- 在单机上模拟多 rank、多节点的通信拓扑
- 在有真实 GPU 的环境里，用 `hybrid` 模式验证“本地算子真实执行，集群通信虚拟化”

## 当前能力

- CUDA Driver API / Runtime API 拦截
- cuBLAS / cuBLASLt 基础兼容
- NVML 设备查询兼容
- `simulate`、`passthrough`、`hybrid` 三种计算模式
- `simulate`、`proxy`、`passthrough` 三种分布式通信模式
- cluster 级通信统计与报告输出

## 推荐上手路径

1. 先构建项目，并跑 `./ftest smoke`
2. 再用 `./fgpu python3 your_script.py` 验证单进程路径
3. 需要分布式时，从 `./test/run_multinode_sim.sh 2` 开始

## 最小体验命令

```bash
cmake -S . -B build
cmake --build build

./ftest smoke
./fgpu python3 -c "import fakegpu; fakegpu.init(); import torch; print(torch.cuda.device_count())"
```

## 模式选择

| 目标 | 推荐计算模式 | 推荐通信模式 |
|---|---|---|
| 无 GPU 机器上先把 CUDA / PyTorch 主路径跑通 | `simulate` | `disabled` |
| 单机模拟多 rank / 多节点控制流 | `simulate` | `simulate` |
| 本地算子真实执行，但跨节点通信继续虚拟化 | `hybrid` | `simulate` |
| 对接真实 NCCL 做对比验证 | `hybrid` | `proxy` 或 `passthrough` |

## 文档导航

- [快速开始](getting-started.md)
- [快速参考](quick-reference.md)
- [项目结构](project-structure.md)
- [分布式模拟使用说明](distributed-sim-usage.md)
- [多节点模拟设计文档](multi-node-design.md)
- [cuBLASLt 兼容修复记录](cublaslt-fix.md)

## 仓库地址

- GitHub: [FanBB2333/FakeGPU](https://github.com/FanBB2333/FakeGPU)
