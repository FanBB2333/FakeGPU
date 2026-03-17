# 项目结构与架构

这份文档解释仓库是怎么组织的，以及 FakeGPU 运行时的主链路是怎样串起来的。

## 运行链路

### 1. 启动器与 Python API

- `./fgpu` 是最常用的入口，本质上是在帮你准备 preload 环境。
- `fakegpu/_api.py` 负责解析 build 目录或库目录、按模式选择需要预加载的动态库，并暴露 `init()`、`env()`、`run()`。

### 2. 后端配置

- `src/core/backend_config.hpp` 会解析 `FAKEGPU_MODE`、`FAKEGPU_OOM_POLICY`、分布式配置，以及真实 CUDA 库路径覆盖项。
- 这些配置决定当前是完全模拟、混合模式，还是尽量透传到真实 CUDA / NCCL。

### 3. 设备清单与 profile

- `src/core/global_state.*` 负责 fake device、当前 device、分配映射和运行时计数器。
- `src/core/gpu_profile.*` 负责从 `profiles/*.yaml` 读取 GPU preset。
- CMake 会在 configure 阶段把这些 YAML 编译进产物，所以运行时不依赖外部 profile 文件。

### 4. CUDA 与 NVML 拦截

- `src/cuda/` 实现 CUDA Driver / Runtime 的 stub 和 passthrough 辅助逻辑。
- `src/nvml/` 提供 fake NVML 响应，让工具或框架可以查询设备状态。
- 在 `simulate` 模式下，设备内存本质上是 host allocation，再由 `GlobalState` 负责跟踪。

### 5. cuBLAS 与 CPU-backed compute

- `src/cublas/` 提供 cuBLAS / cuBLASLt 兼容层。
- 当 `ENABLE_FAKEGPU_CPU_SIMULATION=ON` 时，已维护的 GEMM / matmul 路径会在 CPU 上执行，这样测试不仅能跑通，还能做结果校验。

### 6. 分布式模拟

- `src/nccl/` 暴露 fake `libnccl.so.2` 接口。
- `src/distributed/` 负责 communicator 注册、coordinator 协议、拓扑模型、staging buffer 和 collective 执行。
- 当前验证最充分的仍然是“单机、多进程、Unix socket 或 loopback TCP 协调”的路径。

### 7. 监控与报告

- `src/monitor/monitor.cpp` 会在退出时写出 `fake_gpu_report.json`。
- 如果开启分布式并设置了 `FAKEGPU_CLUSTER_REPORT_PATH`，还会再写一份 cluster 级通信报告。

## 目录说明

| 路径 | 职责 |
|---|---|
| `src/core/` | 全局状态、设备元数据、日志、后端选择 |
| `src/cuda/` | CUDA Driver / Runtime 拦截 |
| `src/cublas/` | cuBLAS / cuBLASLt shim 与 CPU-backed math |
| `src/nvml/` | fake NVML 实现 |
| `src/nccl/` | fake NCCL 入口与模式分发 |
| `src/distributed/` | coordinator 协议、communicator、拓扑、staging |
| `src/monitor/` | JSON 报告 |
| `fakegpu/` | Python 包与 CLI |
| `profiles/` | GPU preset YAML |
| `test/` | 用户入口级 smoke / PyTorch / DDP / comparison 脚本 |
| `verification/` | 更底层的 probe、direct test 与样例配置 |
| `docs/` | MkDocs 文档内容 |

## 构建产物

标准构建会生成：

- `build/libcuda.so.1`
- `build/libcudart.so.12`
- `build/libcublas.so.12`
- `build/libnvidia-ml.so.1`
- `build/libnccl.so.2`
- `build/fakegpu-coordinator`

在 macOS 上会生成对应的 `.dylib`。

## Profile 系统

- 默认会暴露 8 张 A100 级别的 fake GPU。
- 可以用 `FAKEGPU_PROFILE` + `FAKEGPU_DEVICE_COUNT` 统一切换 preset。
- 也可以用 `FAKEGPU_PROFILES` 混合配置，例如 `a100:4,h100:4` 或 `t4,l40s`。
- Python API 和 CLI 都支持同样的参数。

## 建议先读哪些内容

- `README.md`
- [快速开始](getting-started.md)
- [报告与验证](reports-and-validation.md)
- [分布式模拟使用说明](distributed-sim-usage.md)
- [分布式设计说明](multi-node-design.md)
