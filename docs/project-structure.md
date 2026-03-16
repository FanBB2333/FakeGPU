# 项目结构

## 构建与产物

- 核心 CMake 目标包括 `libcuda`、`libcudart`、`libcublas`、`libnvidia-ml` 和 `libnccl`
- 默认提供 8 张虚拟 GPU，相关 profile 来自根目录 `profiles/*.yaml`
- 分布式路径还会生成 `fakegpu-coordinator`

## 源码目录

### `src/core/`

- 维护全局设备状态、日志与动态库拦截入口

### `src/cuda/`

- CUDA Driver API / Runtime API 的类型定义与 stub 实现

### `src/cublas/`

- cuBLAS / cuBLASLt 的声明与 stub 或 CPU-backed 实现

### `src/nvml/`

- NVML 设备查询相关的 fake 实现

### `src/monitor/`

- 生成 `fake_gpu_report.json` 等运行时报告

### `src/nccl/`

- 分布式通信模拟、代理和透传相关逻辑

## 其他目录

- `fakegpu/`：Python 包装层与 CLI 入口
- `profiles/`：虚拟 GPU 预设
- `test/`：C / Python / DDP / 对比测试脚本
- `verification/`：更细粒度的验证脚本与测试数据
- `docs/`：项目文档站内容

## 先读哪些文件

- 根目录 [README.md](https://github.com/FanBB2333/FakeGPU/blob/dev/README.md)
- [快速开始](getting-started.md)
- [分布式模拟使用说明](distributed-sim-usage.md)
- [多节点模拟设计文档](multi-node-design.md)
