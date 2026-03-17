# 分布式设计说明

这份文档说明 FakeGPU 分布式层的设计目标、当前边界，以及为什么现在的实现长成这样。

## 核心定位

FakeGPU 的分布式支持本质上是一个**分布式语义模拟器**：

- rank 可以完成初始化并交换 communicator 状态
- collective 和 point-to-point 调用可以在受控拓扑下完成
- 可以输出 cluster 级通信报告

它**不是**：

- NCCL / NVLink / RDMA / InfiniBand 的协议级复刻
- 面向真实生产性能的一比一仿真器

## 设计目标

1. 在单机上把 `torch.distributed` / NCCL 风格控制流先跑通。
2. 让“计算后端”和“通信后端”解耦，这样真实本地计算可以和虚拟通信组合使用。
3. 产出 cluster 级可观测数据，而不是只停留在单进程设备统计。
4. 优先保证行为可解释、可调试，而不是追求内部调度细节和真实 NCCL 完全一致。

## 当前实现模型

### Fake NCCL shim

- `src/nccl/nccl_stubs.cpp` 提供 `libnccl.so.2` 的兼容入口。
- 初始化阶段会先校验分布式配置是否合法。
- grouped operations、communicator split、p2p 和常见 collective 都在 FakeGPU 控制面里处理。

### Communicator registry

- `src/distributed/communicator.cpp` 维护 pending / active communicator 状态。
- collective 会等待所有参与 rank 到齐，再统一执行。
- rank 等待时间、超时次数、collective 次数等统计会进入 cluster report。

### Coordinator 进程

- `build/fakegpu-coordinator` 可以通过 Unix socket 或 TCP 接收请求。
- 它是单机多进程场景里的 rendezvous 中心。
- 这样每个进程里的 fake 库可以更简单，而 communicator 状态集中在 coordinator 里维护。

### 拓扑与时间模型

- `src/distributed/topology_model.*` 会读取 cluster config 和 fabric 参数。
- 链路带宽、时延、争用惩罚会进入估算时间模型和报告。
- 这套模型追求“可解释、可配置”，而不是 cycle-accurate。

### Staging 与数据流

- collective 和 p2p 语义需要 staging buffer 来落地。
- 本地快速路径优先使用 shared memory。
- 也保留了 socket fallback，既能兜底，也能单独拿来做验证。

## 有意保留的边界

- 当前验证最充分的仍然是“单机、多进程”的模拟路径。
- `proxy` 和 `passthrough` 更适合做对比和观测，不是第一条建议路径。
- `hybrid` 分布式路径有价值，但它更依赖真实本地 CUDA / NCCL 环境，受环境影响更大。
- 项目应该先对“已维护 collective 路径的语义正确性”负责，再谈更广义的框架兼容性。

## 为什么要把计算和通信拆开

`FAKEGPU_MODE` 描述本地计算后端：

- `simulate`
- `hybrid`
- `passthrough`

`FAKEGPU_DIST_MODE` 描述通信后端：

- `disabled`
- `simulate`
- `proxy`
- `passthrough`

这样就可以灵活组合：

- fake compute + fake communication
- real compute + fake communication
- real compute + real communication，同时保留 FakeGPU 的观测能力

## 适合当前设计的后续方向

- 扩展 communicator 生命周期和 grouped semantics 的 direct test
- 增加 `proxy` / `passthrough` 的验证覆盖
- 提高多机或类多机场景的覆盖，而不只停留在 loopback / 单机编排
- 增加故障注入和 timeout 调试能力
