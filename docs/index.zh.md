# FakeGPU

FakeGPU 是一个用于拦截 CUDA、cuBLAS、NVML 和 NCCL 调用的运行时层，目标是在没有 GPU、没有真实集群，或者还不想依赖真实集群的阶段，先把设备发现、内存流、部分算子路径、分布式控制流和资源观测跑通。

English 是默认文档版本，页面右上角可以切换到简体中文。

## FakeGPU 适合做什么

- 在无 GPU 环境里先验证 CUDA / PyTorch 主路径
- 向框架暴露可配置的虚拟 GPU 设备清单
- 对已维护的 cuBLAS / cuBLASLt 路径使用 CPU 执行，做结果可校验的 smoke test
- 在单机多进程场景下模拟 NCCL 风格的 collective 和 point-to-point 通信
- 输出设备级和 cluster 级 JSON 报告，用于观察显存、IO、FLOPs 和通信活动
- 复现常见 GPU 运行时错误（跨设备、OOM、无效设备、dtype 不匹配、checkpoint 不兼容），用于验证错误处理逻辑

## FakeGPU 不打算做什么

- 不追求精确复刻 NCCL、NVLink、RDMA、InfiniBand 的协议细节
- 不承诺任意 CUDA kernel 都具备数值正确性
- 不把自己定位成真实生产集群性能预测工具

## 当前代码状态

当前仓库已经包含：

- fake `libcuda`、`libcudart`、`libcublas`、`libnvidia-ml`、`libnccl`
- Python 包和 CLI 包装器（`fakegpu`、`./fgpu`）
- 从 `profiles/*.yaml` 编译进产物的 GPU profile
- 用于单机多进程分布式模拟的 coordinator
- 单进程设备报告和 cluster 级通信报告
- 覆盖 7 类错误场景的错误模拟实验，附带统一 HTML 测试报告

## 建议先跑的验证

```bash
./ftest smoke
./ftest cpu_sim
./ftest python
```

这几条能覆盖：

- 预加载是否正常
- fake 设备发现是否正常
- 多种 GPU profile 是否正确暴露
- 指针属性和内存类型跟踪
- CPU-backed cuBLAS / cuBLASLt 正确性
- 基础 PyTorch CUDA 张量与 matmul 路径

## 推荐模式组合

| 目标 | 计算模式 | 通信模式 |
|---|---|---|
| 无 GPU 机器上先跑通 CUDA / PyTorch 主路径 | `simulate` | `disabled` |
| 单机模拟多 rank / 多节点通信 | `simulate` | `simulate` |
| 本地算子真实执行，跨节点通信继续虚拟化 | `hybrid` | `simulate` |
| 对接真实 NCCL 做对比并保留报告 | `hybrid` | `proxy` 或 `passthrough` |

第一次上手推荐：

```bash
FAKEGPU_MODE=simulate
FAKEGPU_DIST_MODE=simulate
```

## 一眼看懂运行链路

1. `./fgpu` 或 `fakegpu.init(runtime="native")` 负责找到产物并设置 preload 环境变量。
2. `BackendConfig` 读取 `FAKEGPU_*` 环境变量，决定计算模式和分布式模式。
3. `GlobalState` 按编译进来的 YAML profile 惰性创建 fake device。
4. CUDA / NVML stub 负责设备查询与内存流；已维护的 cuBLAS / cuBLASLt 路径可走 CPU。
5. fake NCCL 层和 coordinator 负责 communicator、collective、p2p 协调。
6. monitor 在退出时写出 `fake_gpu_report.json` 和 cluster report。

## 文档导航

- [快速开始](getting-started.md)
- [快速参考](quick-reference.md)
- [项目结构与架构](project-structure.md)
- [Torch Patch 系统](phase2-custom-torch.zh.md)
- [报告与验证](reports-and-validation.md)
- [分布式模拟使用说明](distributed-sim-usage.md)
- [分布式设计说明](multi-node-design.md)
- [cuBLASLt 兼容性说明](cublaslt-fix.md)
- [错误模拟](error-simulation.md)
