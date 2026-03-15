# FakeGPU 多节点模拟设计文档

## 1. 背景

当前 FakeGPU 已经具备以下基础能力：

- 通过 `libcuda` / `libcudart` / `libcublas` / `libnvidia-ml` 拦截，在无 GPU 环境中暴露虚拟设备
- 提供 `simulate` / `passthrough` / `hybrid` 三种运行模式
- 在 `hybrid` 模式下跟踪真实显存预算并提供 OOM fallback
- 输出单机、单进程维度的资源使用报告

下一步如果要支持“多节点”场景，重点不再只是伪造设备本身，而是要补齐“分布式语义”：

- 节点、rank、local rank、world size 的拓扑关系
- 跨进程 / 跨节点的 collective 与 point-to-point 通信
- 虚拟网络拓扑与带宽/时延模型
- 分布式场景下的资源统计、死锁诊断、故障注入

本设计文档的目标，是为未来的多节点模拟能力提供一份按 Step 顺序落地的技术方案。

## 2. 目标

### 2.1 总体目标

让 FakeGPU 能在以下场景中提供可用的分布式模拟能力：

1. 在单机上模拟多节点多卡集群，运行依赖 `torch.distributed` / NCCL 的程序
2. 在无 GPU 环境中跑通分布式训练/推理的控制流与通信流
3. 在有真实 GPU 的环境中，将“本地计算”和“虚拟集群通信”组合起来做混合验证
4. 产出可观测的 cluster 级报告，用于评估显存、通信量、带宽瓶颈和同步等待

### 2.2 分层验收标准

- L0：程序能完成分布式初始化，不崩溃
- L1：常见 collective 能跑通，所有 rank 正常退出
- L2：collective 语义正确，输出张量与参考实现一致
- L3：可配置网络时延/带宽模型，报告可用于资源与性能评估
- L4：支持故障注入、超时、rank 丢失、网络抖动等调试场景

需要额外明确的是，这里的 L0 ~ L4 是**总体能力分层**，不是某个单独 Step 的直接交付承诺。

- Step 1 ~ Step 9 的最低验收应以 direct NCCL shim / communicator / collective MVP 为准
- `torch.distributed.init_process_group` 和 DDP 主路径应放到 Step 10 之后单独验收
- 如果某个 Step 或 Step 区间只完成 direct API 语义正确，不应宣称已经覆盖框架级兼容性

### 2.3 可行性判断

这项设计整体上是可行的，但前提是要把“多节点模拟”的边界定义清楚。

**可行的部分：**

- 单机多进程模拟多节点
- `torch.distributed` / DDP 控制流跑通
- collective 语义执行
- cluster 级通信统计与等待时间统计
- 可配置的简化带宽/时延模型

**第一版不应追求的部分：**

- 精确复现真实 NCCL 内核调度
- 精确复现 RDMA / InfiniBand / NVLink 协议级行为
- 与真实集群性能一一对应
- 覆盖所有 NCCL API 与框架行为细节

换句话说，第一版应被定义为 **分布式语义模拟器**，而不是 **真实网络/真实 NCCL 的协议级复刻**。

## 3. 非目标

以下内容不作为前置实现目标，尤其不作为 Step 1 ~ Step 15 的目标：

- 完整模拟 InfiniBand/RDMA 协议栈细节
- bitwise 级别复现真实 NCCL 内核与调度策略
- 追求真实集群级性能，仅要求可解释、可配置、可复现
- 一开始就覆盖所有框架，只优先覆盖 PyTorch + NCCL 主路径

## 4. 设计原则

### 4.1 语义正确性优先于性能逼真

第一优先级是 collective 的输入输出语义正确，其次才是时延与带宽建模。

### 4.2 计算与通信解耦

当前 `FAKEGPU_MODE` 主要描述“设备/计算后端”：

- `simulate`
- `passthrough`
- `hybrid`

多节点能力不应强行塞进这个枚举，而应新增独立的“分布式通信模式”配置，使计算后端与通信后端正交组合。

### 4.3 单机多进程先行，真实多机后扩展

优先支持：

- 单机、多进程、模拟多节点
- 多进程之间通过本地 socket / shared memory 与 coordinator 通信

再扩展：

- 多机 coordinator
- coordinator federation
- 真实网络互联

### 4.4 统一“语义执行”和“时间模型”

collective 的结果计算与其耗时模拟应拆成两个层次：

- 语义执行层：决定输出 buffer 内容
- 时间模型层：决定何时完成、耗时多少、是否注入抖动/丢包/超时

## 5. 关键使用场景

### 场景 A：单机无 GPU，模拟 2 节点 x 4 卡

目标：

- 在一台普通机器上，用 `torchrun` 或多个 Python 进程跑通 `torch.distributed`
- 每个 rank 看到自己的本地 GPU
- 集体通信由 FakeGPU 模拟

### 场景 B：单机有 GPU，本地计算真实执行，跨节点通信虚拟化

目标：

- 单机真实算子跑在 GPU 上
- 通信路径走 FakeGPU 的虚拟 fabric
- 用于评估“如果未来扩成多机，通信会是什么形态”

### 场景 C：故障与极限条件调试

目标：

- 模拟某个节点超时、带宽下降、部分 rank 掉线
- 复现 DDP / inference serving 中的 hang、barrier 卡死、梯度不同步问题

## 6. 配置模型

建议新增一组独立配置：

```bash
FAKEGPU_DIST_MODE={disabled,simulate,proxy,passthrough}
FAKEGPU_CLUSTER_CONFIG=/path/to/cluster.yaml
FAKEGPU_COORDINATOR_ADDR=127.0.0.1:29591
FAKEGPU_COORDINATOR_TRANSPORT={unix,tcp}
FAKEGPU_FAULT_PROFILE=/path/to/faults.yaml
FAKEGPU_CLUSTER_REPORT_PATH=/path/to/fake_gpu_cluster_report.json
```

说明：

- `disabled`：不启用分布式模拟
- `simulate`：使用 FakeGPU 自己的 collective engine
- `proxy`：记录与调度通信，但底层可转发到真实 NCCL
- `passthrough`：直接转发到真实 NCCL，仅做最薄的包装/报告

建议把模式稳定性也写清楚：

- Step 1 ~ Step 15：只承诺 `disabled`、`simulate`
- `proxy`：实验性，至少放到真实 GPU 混合运行之后
- `passthrough`：更偏观测/对比用途，不应阻塞前 15 个 Step 的实现

### 6.1 推荐的 cluster 配置文件

建议采用 YAML，风格与现有 `profiles/*.yaml` 保持一致：

```yaml
version: 1
cluster:
  name: dev-cluster
  default_backend: nccl

nodes:
  - id: node0
    host: 127.0.0.1
    ranks: [0, 1, 2, 3]
    gpus:
      - profile: a100
      - profile: a100
      - profile: a100
      - profile: a100

  - id: node1
    host: 127.0.0.1
    ranks: [4, 5, 6, 7]
    gpus:
      - profile: a100
      - profile: a100
      - profile: a100
      - profile: a100

fabric:
  intra_node:
    type: nvlink
    bandwidth_gbps: 300
    latency_us: 3

  inter_node:
    type: infiniband
    bandwidth_gbps: 200
    latency_us: 15
    oversubscription: 1.5

policies:
  collective_timeout_ms: 60000
  default_allreduce_algo: ring
  deterministic_ordering: true
```

### 6.2 运行时环境变量映射

需要兼容以下标准环境变量：

- `RANK`
- `LOCAL_RANK`
- `WORLD_SIZE`
- `MASTER_ADDR`
- `MASTER_PORT`

FakeGPU 不应替代这些变量，而应读取它们并与 cluster config 做一致性校验。

### 6.3 配置主从关系

为避免启动路径含糊，建议明确以下规则：

- `RANK` / `LOCAL_RANK` / `WORLD_SIZE` 这类运行时变量是**进程实例事实来源**
- `FAKEGPU_CLUSTER_CONFIG` 是**静态拓扑与策略来源**
- 启动时优先用运行时变量确定“当前进程是谁”，再用 cluster config 校验该 rank 是否存在、位于哪个 node、使用哪个 GPU profile
- 如果 cluster config 与运行时变量不一致，应在初始化阶段立即失败，而不是尝试隐式修正
- Step 1 可以允许“无 cluster config”的最小模式，只依赖运行时变量和默认拓扑完成单机 smoke test

## 7. 总体架构

建议引入以下新模块：

```text
FakeGPU
├── src/
│   ├── distributed/         # cluster coordinator / collective engine / transport
│   ├── nccl/                # NCCL stubs / passthrough dispatch
│   ├── core/                # 现有 GlobalState / device / mode config
│   ├── cuda/                # 现有 CUDA interception
│   ├── cublas/              # 现有 cuBLAS interception
│   └── monitor/             # 现有 report，未来扩展 cluster report
└── test/
    ├── test_nccl_basic.py
    ├── test_ddp_multinode.py
    └── run_multinode_sim.sh
```

### 7.1 核心组件

#### A. Fake NCCL Shim

职责：

- 提供 `libnccl.so` 兼容接口
- 拦截 `ncclCommInitRank`、`ncclAllReduce`、`ncclReduce`、`ncclBroadcast`、`ncclAllGather`、`ncclReduceScatter`、`ncclSend`、`ncclRecv` 等调用
- 将每个 collective 转换成 coordinator 可以理解的请求

#### B. Cluster Coordinator

职责：

- 管理 cluster、node、rank、communicator 的生命周期
- 聚合同一 collective 的各 rank 请求
- 验证参数一致性
- 调用 collective semantic engine 产出结果
- 结合 time model 决定何时完成

部署方式建议：

- Step 3 先保证 **local coordinator 接口** 跑通；实现上可先支持同机单独守护进程，也允许保留 in-process / helper 形态用于开发与验证
- Step 4 ~ Step 15 将**同机单独守护进程**作为推荐默认部署
- Step 21 再进入多机 TCP coordinator

推荐理由：

- 避免把 coordinator 生命周期绑死在 rank0
- rank0 崩溃时不会顺带拖死整个调度层
- 日志、报告、超时诊断更容易集中管理
- 后续扩到多机时迁移路径更清晰

这里的关键是先把 coordinator 抽象边界做稳定，而不是在第一步就把部署形态做死。否则 Step 3 会同时承担接口设计、进程生命周期、IPC 和 collective 语义四类复杂度。

#### C. Collective Semantic Engine

职责：

- 执行 AllReduce / Broadcast / ReduceScatter / AllGather / AllToAll / SendRecv 的语义逻辑
- 不关心真实网络细节，只关心输入输出内容

#### D. Virtual Fabric Model

职责：

- 描述 intra-node、inter-node、NIC uplink、oversubscription
- 根据 tensor 大小、算法、rank 分布估算耗时
- 为故障注入提供链路级入口

#### E. Distributed Memory Staging Layer

职责：

- 解决“不同进程里的 device pointer 彼此不可见”的问题
- 在 `simulate` 模式下，把 RAM-backed fake device memory 拷贝到跨进程共享区
- 在 `hybrid` / `passthrough` 模式下，必要时把 GPU buffer staging 到 host buffer

#### F. Cluster Report / Trace

职责：

- 输出 collective 次数、字节数、等待时间、每条 link 的利用率
- 记录 communicator 事件、rank 超时、错误码、重试与回退

### 7.2 控制面与数据面分离

建议从第一版开始就显式拆成两层：

- **控制面**
  - coordinator <-> rank 之间的元数据交互
  - communicator 注册、seqno 分配、collective 参数校验、错误回传
  - 推荐使用 Unix domain socket

- **数据面**
  - tensor staging buffer 的读写
  - 推荐优先使用 POSIX shared memory
  - shared memory 不可用时，再回退到 socket streaming

这样做的好处是：

- 控制逻辑与大块数据传输不会相互干扰
- 后续引入 chunking、限流、零拷贝优化更自然
- `simulate` 与 `hybrid` 可以共用控制面，只替换数据面 staging 策略

## 8. 与现有模式的关系

建议把“计算模式”和“通信模式”拆开：

| 计算模式 | 通信模式 | 含义 |
|---|---|---|
| `simulate` | `simulate` | 全栈虚拟，无真实 GPU，无真实 NCCL |
| `hybrid` | `simulate` | 本地计算尽量走真实 GPU，跨 rank 通信由 FakeGPU 模拟 |
| `passthrough` | `proxy` | 真实 GPU + 真实 NCCL 为主，FakeGPU 只做调度观测与可选故障注入 |
| `passthrough` | `passthrough` | 几乎纯透传，只保留轻量记录 |

不建议把多节点逻辑直接绑死在 `hybrid` 上，否则后续难以维护。

模式稳定性建议进一步收敛：

- Step 1 ~ Step 15 只正式承诺 `simulate + simulate`
- `hybrid + simulate` 作为 Step 19 之后的正式目标，但不应提前成为 Step 1 ~ Step 15 的隐性约束
- `proxy` / `passthrough` 在文档中保留为后期扩展路线即可，不应成为前 15 个 Step 的设计复杂度来源

## 9. API 接入点

### 9.1 前置 Step 最小覆盖的 NCCL API

建议优先实现：

- `ncclGetVersion`
- `ncclGetUniqueId`
- `ncclCommInitRank`
- `ncclCommDestroy`
- `ncclCommAbort`
- `ncclAllReduce`
- `ncclBroadcast`

这组 API 对第一版是够用的，目标是先跑通最小 direct collective MVP，而不是一开始就覆盖完整 NCCL 面。

更准确地说，这组 API 对第一版足够支撑 **direct NCCL collective MVP**；如果要把目标上升到 PyTorch DDP 主路径，通常还需要补足 barrier、group 语义，以及更多错误传播与时序细节。

需要注意：

- 框架层的 `barrier` 优先通过 coordinator 自己实现，或退化成一个固定大小的同步 collective
- 文档中不再把“`ncclBarrier`”当作前置 Step 必需 API，因为它不应成为设计的中心依赖
- 当前实现还额外覆盖了 `ncclCommInitAll`、`ncclCommSplit`（当前仅 `FAKEGPU_DIST_MODE=simulate`）、blocking `ncclSend` / `ncclRecv`（当前仅 `FAKEGPU_DIST_MODE=simulate`，已支持 grouped submission，但每个 seqno 仍需要 communicator 内所有 rank 都提交一个 p2p 操作）、`ncclRedOpCreatePreMulSum`（当前仅 `FAKEGPU_DIST_MODE=simulate` + `ncclScalarHostImmediate` + `int32/int64/float32/float64`）、`ncclAlltoAll`、`ncclCommGetAsyncError` 的基础 async-error 持久化/快失败语义、grouped submission 下的基础 stream 一致性约束、fake stream / event 生命周期校验，以及基础 stream capture 状态查询，可用于本地 direct smoke test

### 9.2 后续可补充

- 更细的 stream 关联

## 10. 内存与数据通路设计

### 10.1 问题定义

当前 FakeGPU 的 fake device memory 在 `simulate` 模式下通常是“进程内 RAM 指针”。  
这意味着：

- rank 0 的“device pointer”对 rank 1 没有意义
- coordinator 不能直接读取别的进程的 pointer

所以必须增加 staging 机制。

### 10.2 建议方案

每个 collective 分两层 buffer：

1. **rank local buffer**
   - 当前进程中的 device pointer
   - 来源可能是 fake RAM，也可能是真实 GPU 内存

2. **coordinator visible staging buffer**
   - shared memory 或 socket buffer
   - coordinator 只操作这个层

执行流程：

1. rank 进入 NCCL API
2. 将源 tensor 从 local buffer 拷贝到 staging buffer
3. 向 coordinator 发送 metadata + staging handle
4. coordinator 等齐所有 rank 后执行 collective
5. coordinator 将结果写回每个 rank 的 staging buffer
6. rank 再把结果从 staging buffer 回写到目的 buffer

第一版建议的具体实现：

- 每个 rank 为每次 collective 注册一个 `staging_id`
- coordinator 用 `(comm_id, seqno, rank, staging_id)` 定位这块数据
- 控制面只交换 metadata，不传大块 tensor 内容
- 数据面优先使用命名 shared memory 段，避免 coordinator 反复复制

### 10.3 不同模式下的 staging 策略

- `simulate`：直接 `memcpy`，因为 fake device memory 本质是 host RAM
- `hybrid`：优先走真实 `cudaMemcpyDtoH` / `cudaMemcpyHtoD`，但这一步需要额外处理 stream 同步、buffer 生命周期与错误传播，不应与 `simulate` 版 staging 混为同一复杂度
- `passthrough` + `proxy`：默认仍可使用 host staging；后续再优化为零拷贝或真实 NCCL proxy

### 10.4 大 tensor 处理

为避免超大 tensor 带来的 coordinator 内存峰值，建议支持：

- chunked staging
- 分片 reduce
- streaming allgather

但应明确优先级：

- Step 1 ~ Step 9：先不做 chunking，只限制 tensor 大小并给出清晰报错
- Step 13 ~ Step 18：再补 chunked staging
- Step 18 之后：再考虑 streaming allgather / reduce-scatter 优化

## 11. Communicator 生命周期

### 11.1 初始化

`ncclGetUniqueId` / `ncclCommInitRank` 需要映射到一个全局 communicator：

- `comm_id`
- `world_size`
- `rank -> node_id -> local_device`
- backend mode

coordinator 在收到所有 rank 的 init 请求后，创建 communicator 并返回就绪状态。

### 11.2 顺序性

每个 communicator 维护单调递增的 `seqno`：

- 每次 collective 自动分配或校验 `seqno`
- 同一 `seqno` 下必须收齐所有相关 rank 的请求
- 参数不一致则立即 fail，而不是默默挂死

建议在第一版就把以下校验做全：

- collective 类型是否一致
- dtype / count / root / reduce op 是否一致
- communicator / world size 是否一致
- 是否有 rank 缺失或重复提交

### 11.3 GroupStart / GroupEnd

对 `ncclGroupStart/End`，建议采用“本地缓冲 + coordinator 批量提交”：

- `GroupStart` 之后先不发请求
- `GroupEnd` 时统一发送一批 collective
- coordinator 以批为单位校验和调度

但实现优先级建议延后到 Step 12。  
第一版先支持“无 group 语义”的直连调用，更容易把死锁和参数不一致问题暴露清楚。

## 12. Collective 语义层设计

### 12.1 AllReduce

输入：

- 所有 rank 的同 shape / same dtype buffer
- reduction op：sum / prod / max / min

语义：

- 先做 reduce，再把结果分发给所有 rank

### 12.2 Broadcast

输入：

- root rank buffer
- 其他 rank 目的 buffer

语义：

- root 内容复制到所有参与 rank

### 12.3 ReduceScatter / AllGather

建议以“逻辑张量视图 + slice plan”实现，而不是把每种 collective 单独写成互不复用的流程。

### 12.4 Send / Recv

点对点通信应支持：

- tag
- src / dst
- blocking 语义
- timeout

后续也可扩展非阻塞语义，但不是前置 Step 必需。

## 13. 时间模型与网络拓扑

### 13.1 基本公式

建议初始模型使用：

```text
cost = base_latency + bytes / effective_bandwidth + contention_penalty
```

其中：

- `base_latency` 来自 fabric 配置
- `effective_bandwidth` 由 link 类型、并发度、collective 算法决定
- `contention_penalty` 由 oversubscription、同时进行的 collective 数量估算

这个模型的定位应当写清楚：

- 它用于 **相对比较** 和 **瓶颈解释**
- 不用于承诺“与真实集群耗时严格一致”
- 第一版的价值在于“可配置、可复现、能解释”，而不是“完全逼真”

### 13.2 拓扑层次

至少区分两层：

- intra-node：PCIe / NVLink
- inter-node：Ethernet / InfiniBand

后续可进一步细化：

- NUMA domain
- NIC affinity
- spine-leaf oversubscription

### 13.3 算法模型

建议先内置两类 collective 算法估算：

- Ring
- Tree

coordinator 可根据 tensor size 与拓扑简单选择，或者由配置文件强制指定。

## 14. 故障注入设计

多节点模拟的价值之一，是制造真实集群难复现的问题。

建议支持以下 fault model：

- rank join timeout
- collective timeout
- link slowdown
- packet drop 的高层等价模拟
- node unavailable
- communicator abort
- 指定 seqno 卡住

配置方式建议采用独立 YAML：

```yaml
faults:
  - kind: link_slowdown
    src_node: node0
    dst_node: node1
    factor: 0.2
    start_seqno: 50
    end_seqno: 80

  - kind: rank_timeout
    rank: 3
    at_collective: all_reduce
    after_seqno: 120
```

## 15. 报告与可观测性

建议在现有 `fake_gpu_report.json` 之外，新增 cluster 级报告，例如：

```json
{
  "report_version": 4,
  "cluster": {
    "world_size": 8,
    "node_count": 2,
    "communicators": 1
  },
  "collectives": {
    "all_reduce": {"calls": 120, "bytes": 987654321},
    "all_gather": {"calls": 30, "bytes": 123456789}
  },
  "links": [
    {
      "src": "node0",
      "dst": "node1",
      "bytes": 456789123,
      "avg_latency_us": 17.4
    }
  ],
  "ranks": [
    {
      "rank": 0,
      "node": "node0",
      "wait_time_ms": 321.5,
      "timeouts": 0
    }
  ]
}
```

建议关注的指标：

- 每类 collective 的调用次数与字节数
- 每个 rank 的等待时间、最长 barrier 卡顿
- 每条虚拟链路的累计流量和平均耗时
- 失败 / 超时 / abort 次数
- chunked transfer 的分片数量与回退次数

建议补充一条实现约束：

- Step 1 ~ Step 15 的 cluster report schema 可以标记为 experimental，只保证核心字段存在，不宜过早承诺长期稳定的 JSON 结构

## 16. 代码结构建议

建议新增如下文件组织：

```text
src/distributed/
├── cluster_config.hpp/.cpp
├── cluster_coordinator.hpp/.cpp
├── communicator.hpp/.cpp
├── transport.hpp
├── transport_local.cpp
├── transport_tcp.cpp
├── staging_buffer.hpp/.cpp
├── collective_executor.hpp
├── collective_allreduce.cpp
├── collective_broadcast.cpp
├── collective_allgather.cpp
├── collective_reducescatter.cpp
├── fault_injector.hpp/.cpp
└── topology_model.hpp/.cpp

src/nccl/
├── nccl_defs.hpp
├── nccl_stubs.cpp
├── nccl_passthrough.hpp/.cpp
└── nccl_mode_dispatch.hpp
```

现有模块建议的接入点：

- `src/core/backend_config.hpp`
  - 新增 distributed mode 配置读取
- `src/core/global_state.*`
  - 增加 rank / node / communicator 维度的 report hooks
- `src/monitor/monitor.cpp`
  - 扩展 cluster report 输出

## 17. 测试策略

### 17.1 单元测试

验证：

- cluster config 解析
- communicator 生命周期
- collective 参数一致性检查
- time model 计算
- fault 注入命中逻辑

### 17.2 直接 API 测试

建议新增：

- `verification/test_nccl_direct.cpp`
- `verification/test_collective_executor.py`

### 17.3 框架集成测试

建议新增：

- `test/test_ddp_multinode.py`
- `test/test_torch_distributed_barrier.py`
- `test/test_allreduce_correctness.py`

测试维度：

1. 单机 2 rank
2. 单机 4 rank
3. 单机模拟 2 节点 x 4 rank
4. `simulate + simulate`
5. `hybrid + simulate`
6. 故障注入下的超时与 abort

建议把测试门槛分开：

- Step 1 ~ Step 9 门槛：2 rank / 4 rank 单机通过
- Step 10 ~ Step 15 门槛：单机模拟 2 节点 x 4 rank 通过
- Step 16 ~ Step 18 门槛：拓扑参数变化能反映到报告结果

### 17.4 对比测试

在有真实 GPU 的机器上，可增加：

- 真实 NCCL vs FakeGPU `proxy`
- 对比 collective 输出是否一致
- 对比调用序列和通信量统计是否合理

## 18. 实施步骤规划

本文只保留一套执行逻辑：**按 Step 顺序推进**。

- Step 1 ~ Step 9 的目标是拿到 direct collective MVP
- Step 10 ~ Step 15 的目标是逐步补齐框架接入与 DDP 主路径
- Step 16 ~ Step 18 的目标是补齐拓扑、时间模型和大 tensor 支持
- Step 19 ~ Step 21 的目标是进入 hybrid、proxy 和真实多机扩展

### 18.1 详细实施步骤（每步都可验证）

下面的步骤按推荐开发顺序排列。  
原则是：每一步都应能独立合并、独立验证、独立回退。

#### Step 1：分布式配置读取与环境校验

实施内容：

- 在 `src/core/backend_config.hpp` 中加入 distributed mode 相关配置读取
- 统一解析 `FAKEGPU_DIST_MODE`、`FAKEGPU_CLUSTER_CONFIG`、`FAKEGPU_COORDINATOR_ADDR`
- 建立最小的配置对象，供后续 `nccl` / `distributed` 模块复用

代码产物：

- `BackendConfig` 增加 distributed 配置访问接口
- 新增 `src/distributed/cluster_config.hpp/.cpp` 的最小骨架

验证方式：

- 新增 `verification/test_cluster_config.py`
- 覆盖“环境变量存在 / 缺失 / 非法值”三类情况

通过标准：

- 合法配置时，测试返回 0，并打印解析后的 mode / coordinator 地址
- 非法 mode 或缺失关键配置时，测试返回非 0，并输出明确错误信息

#### Step 2：Cluster YAML 解析与拓扑合法性校验

实施内容：

- 解析 cluster YAML 中的 `nodes`、`ranks`、`gpus`、`fabric`
- 校验 rank 唯一性、world size 一致性、node 定义合法性
- 输出内存中的规范化 cluster model

代码产物：

- `src/distributed/cluster_config.hpp/.cpp`
- Cluster model 数据结构

验证方式：

- 新增 `verification/test_cluster_config_valid.py`
- 新增 `verification/test_cluster_config_invalid.py`
- 提供一份合法 YAML 和两份非法 YAML 示例

通过标准：

- 合法 YAML 能被完整解析并打印规范化拓扑摘要
- 非法 YAML 会在启动前失败，而不是运行期静默出错

#### Step 3：Coordinator 守护进程最小骨架

实施内容：

- 实现本地 coordinator 守护进程
- 支持启动、监听、健康检查、优雅退出
- 先不处理 collective，只提供 `ping` / `shutdown` 控制命令

代码产物：

- `src/distributed/cluster_coordinator.hpp/.cpp`
- `src/distributed/transport_local.cpp`
- 可执行入口，例如 `fakegpu-coordinator`

验证方式：

- 新增 `verification/test_coordinator_smoke.py`
- 启动 coordinator 后检查 Unix socket 是否创建
- 发送 `ping` 请求并校验响应

通过标准：

- coordinator 启动后 1 秒内可接受连接
- `ping` 返回固定版本/状态信息
- `shutdown` 后 socket 与进程都被正确清理

#### Step 4：控制面协议与 communicator 注册

实施内容：

- 定义控制面消息格式：`HELLO`、`INIT_COMM`、`DESTROY_COMM`
- coordinator 维护 communicator 表和 rank 注册表
- 引入 communicator `seqno=0` 的初始状态

代码产物：

- `src/distributed/communicator.hpp/.cpp`
- coordinator 内部 registry

验证方式：

- 新增 `verification/test_communicator_registry.py`
- 模拟 2 个和 4 个 mock rank 注册同一个 communicator

通过标准：

- 所有 rank 注册完成后，返回相同 communicator 标识
- duplicate rank、world size 不一致、缺失 rank 都能立即报错

#### Step 5：Fake NCCL 最小初始化路径

实施内容：

- 提供 `libnccl.so` 的最小 shim
- 实现 `ncclGetVersion`、`ncclGetUniqueId`、`ncclCommInitRank`、`ncclCommDestroy`
- 这些 API 先只打通到 coordinator，不做 collective

代码产物：

- `src/nccl/nccl_defs.hpp`
- `src/nccl/nccl_stubs.cpp`
- `src/nccl/nccl_mode_dispatch.hpp`

验证方式：

- 新增 `verification/test_nccl_direct.cpp`
- 只测试 init/destroy，不测试张量通信

通过标准：

- 2 rank / 4 rank 能完成 communicator 初始化与销毁
- 重复销毁、非法 rank、非法 world size 都返回显式错误码

#### Step 6：Shared Memory Staging Layer

实施内容：

- 实现 staging buffer 管理器
- 支持创建、映射、关闭、释放命名 shared memory
- metadata 中记录 dtype、bytes、shape、owner rank、staging_id

代码产物：

- `src/distributed/staging_buffer.hpp/.cpp`

验证方式：

- 新增 `verification/test_staging_buffer.py`
- 两个独立进程分别写入和读取同一 staging buffer

通过标准：

- reader 读到的内容与 writer 写入内容字节级一致
- 进程退出后 shared memory 能被回收，不残留悬挂段

#### Step 7：AllReduce 语义执行器

实施内容：

- 实现 CPU 参考版 `all_reduce(sum)` 语义
- coordinator 等到所有 rank 到齐后执行 reduce，再写回各自结果
- 先只支持 `float32` / `int32`

代码产物：

- `src/distributed/collective_executor.hpp`
- `src/distributed/collective_allreduce.cpp`

验证方式：

- 新增 `test/test_allreduce_correctness.py`
- 覆盖 2 rank / 4 rank、`float32` / `int32`

通过标准：

- 所有 rank 输出与 Python 参考实现一致
- 多次运行结果稳定，不出现随机 hang

#### Step 8：Broadcast 语义执行器

实施内容：

- 实现 root -> all ranks 的 broadcast
- root rank 与目标 rank 的 buffer 校验统一走 coordinator

代码产物：

- `src/distributed/collective_broadcast.cpp`

验证方式：

- 新增 `verification/test_broadcast_correctness.py`
- 覆盖 root=0、root=last rank 两种情况

通过标准：

- 所有非 root rank 的输出与 root buffer 完全一致
- root 配置不一致时在 coordinator 侧立即报错

#### Step 9：参数一致性校验与超时快失败

实施内容：

- 为 collective 请求增加严格校验：
  - collective 类型
  - dtype
  - count
  - root
  - reduce op
- 增加 join timeout 和 collective timeout

代码产物：

- coordinator 校验逻辑
- timeout / abort 状态机

验证方式：

- 新增 `verification/test_collective_mismatch.py`
- 新增 `verification/test_collective_timeout.py`

通过标准：

- 参数不一致时 5 秒内失败并返回明确错误
- 缺失 rank 时不会无限挂起

#### Step 10：Torch Distributed 最小 smoke test

实施内容：

- 在已有 direct collective MVP 之上，尝试接入 `torch.distributed.init_process_group`
- 跑一个最小 `all_reduce` 和 `broadcast`
- 该步骤应标记为 exploratory：如果发现 barrier / group / async error 等缺口，应把缺口回灌到后续步骤，而不是为了“通过 smoke test”临时堆补丁

代码产物：

- `test/test_ddp_multinode.py` 的最小版
- `test/run_multinode_sim.sh`

验证方式：

- 单机 2 rank
- 单机 4 rank

通过标准：

- 至少能稳定复现框架接入所需的缺口列表
- 如果初始化成功，结果可作为 Step 11 ~ Step 15 的前置验证收益，而不是反向提升 Step 1 ~ Step 9 的门槛

#### Step 11：Framework Barrier 支持

实施内容：

- 在 coordinator 中实现 framework 层需要的 barrier 语义
- barrier 不单独建复杂数据通道，只做控制面同步

代码产物：

- barrier 逻辑并入 communicator / coordinator

验证方式：

- 新增 `test/test_torch_distributed_barrier.py`
- 新增 `verification/test_coordinator_barrier.py`

通过标准：

- 所有 rank 均能通过 barrier
- 任一 rank 超时会触发整组失败，而不是其他 rank 永久等待

#### Step 12：GroupStart / GroupEnd 支持

实施内容：

- 在 rank 侧缓存 group 内操作
- `GroupEnd` 时批量提交到 coordinator
- coordinator 以批为单位做参数校验和顺序调度

代码产物：

- group request buffer
- batch dispatch 逻辑

验证方式：

- 新增 `verification/test_group_semantics.py`

通过标准：

- group 内多个 collective 能按提交顺序一致执行
- 中途参数不一致时，整批次明确失败

#### Step 13：AllGather 与 ReduceScatter

实施内容：

- 实现 `all_gather`
- 实现 `reduce_scatter`
- 引入逻辑张量视图与 slice plan，避免完全重复实现

代码产物：

- `src/distributed/collective_allgather.cpp`
- `src/distributed/collective_reducescatter.cpp`

验证方式：

- 新增 `verification/test_allgather_correctness.py`
- 新增 `verification/test_reducescatter_correctness.py`

通过标准：

- 输出与 Python 参考实现一致
- 4 rank 场景下稳定通过

#### Step 14：Cluster Report 基础版

实施内容：

- 增加 cluster 级 JSON 报告
- 统计 communicator 数量、collective 调用次数、每 rank 等待时间

代码产物：

- `src/monitor/monitor.cpp` 扩展
- `fake_gpu_cluster_report.json`

验证方式：

- 新增 `verification/check_cluster_report.py`

通过标准：

- 报告文件存在且 schema 合法
- collective 次数、world size、rank 等待时间字段非空

#### Step 15：DDP 主路径验证

实施内容：

- 用多进程版本的简单模型跑通 DDP
- 覆盖梯度同步主路径

代码产物：

- 扩展 `test/test_ddp_simple.py` 或新增 `test/test_ddp_multinode.py`

验证方式：

- 单机模拟 2 节点 x 4 rank

通过标准：

- 训练脚本能完成至少 1 个 epoch
- 所有 rank 正常退出
- 报告中出现 AllReduce / Broadcast / Barrier 统计

#### Step 16：拓扑配置接入

实施内容：

- 让 cluster config 中的 `fabric` 参数真正参与建模
- 至少区分 intra-node 与 inter-node

代码产物：

- `src/distributed/topology_model.hpp/.cpp`

验证方式：

- 新增 `verification/test_topology_model.py`
- 对同一 workload 提供两份不同 fabric 配置

通过标准：

- 报告中的链路统计和估算耗时随配置变化而变化
- 不同 fabric 的结果可解释

#### Step 17：时间模型与链路级统计

实施内容：

- 把 `cost = latency + bytes / bandwidth + contention_penalty` 接入 coordinator
- 记录每条虚拟链路的流量、平均耗时、拥塞惩罚

代码产物：

- time model 实现
- link stats report 扩展

验证方式：

- 新增 `verification/test_time_model.py`
- 新增 `verification/check_cluster_report.py --expect-links`

通过标准：

- 相同 tensor 在慢链路下耗时更高
- 报告中能看到 link 级别统计字段

#### Step 18：Chunked Staging

实施内容：

- 为大 tensor collective 增加分片传输
- coordinator 支持 chunk-by-chunk 的 reduce / gather

代码产物：

- staging buffer chunk 计划
- chunked collective 执行逻辑

验证方式：

- 新增 `verification/test_chunked_collective.py`

通过标准：

- 大于单块阈值的 tensor 不再因 staging 峰值过高失败
- chunked 与非 chunked 结果一致

#### Step 19：Hybrid + Simulate 混合运行

实施内容：

- 在 `hybrid` 计算模式下支持跨节点 collective 模拟
- GPU buffer 通过 `cudaMemcpyDtoH/HtoD` 做 host staging

代码产物：

- `simulate` / `hybrid` 共用的控制面
- hybrid staging adapter

验证方式：

- 新增 `test/test_hybrid_multinode.py`

通过标准：

- 本地算子仍能在真实 GPU 上运行
- collective 结果通过 FakeGPU 分布式层同步完成

#### Step 20：Proxy / Passthrough 试验版

实施内容：

- 引入实验性 `proxy` 模式
- 允许控制面仍由 FakeGPU 管理，但数据面转发到真实 NCCL
- 增加 `FAKEGPU_REAL_NCCL_PATH`，允许显式指定真实 `libnccl.so.2`

代码产物：

- `src/nccl/nccl_passthrough.hpp/.cpp`

验证方式：

- 新增 `verification/test_nccl_proxy.py`
- 在有 GPU + NCCL 的机器上运行
- 脚本按机器 GPU 数自动选择 `world_size=1/2`

通过标准：

- baseline NCCL 与 proxy 模式在 collective 输出上保持一致
- FakeGPU 仍能生成 cluster 级报告

当前边界：

- 单 GPU 机器默认只验证 `world_size=1`；`world_size=2` 需要至少 2 张 GPU 才能做 baseline 对比
- grouped collective 仍只在 `simulate` 模式实现，`proxy/passthrough` 目前只覆盖 ungrouped direct collective

#### Step 21：远端 Coordinator 与多机扩展

实施内容：

- 增加 TCP transport
- 支持 coordinator 与 rank 分布在不同机器

代码产物：

- `src/distributed/transport_tcp.cpp`

验证方式：

- 新增 `verification/test_remote_coordinator.py`
- 使用 TCP loopback 做 2 rank happy path 与断链失败验证
- 真实多机或 network namespace 环境留作部署侧复核

通过标准：

- rank 可通过 TCP 注册与执行 collective
- 网络异常时能返回明确错误并回收 communicator

当前边界：

- 目前验证范围是单机 `127.0.0.1` TCP；没有在两台物理机上重复此测试
- 数据面仍然是现有 direct collective / proxy 路径，Step 21 只扩展 coordinator transport

### 18.2 Step 完成定义

每个 step 在进入下一个 step 之前，至少要满足以下条件：

- 有对应代码产物
- 有最小验证脚本或测试
- 测试结果可重复
- 失败路径有明确错误，而不是 hang
- 文档中记录新增配置项、限制和已知边界

### 18.3 Codex 分批执行建议

如果要让 codex 按批次完成，建议优先采用下面这套切分。原则是：

- 每次只处理一组强耦合 Step
- 每次都要包含代码、测试和最小验收
- 不要跨越明显不同性质的工作边界，例如把配置解析、collective 语义、PyTorch 接入、多机扩展混在同一次里

推荐分成 9 次：

#### 第 1 次：Step 1

范围：

- [x] Step 1：分布式配置读取与环境校验

本次完成标准：

- [x] `BackendConfig` 增加 distributed 配置访问接口
- [x] 新增最小 `cluster_config` 骨架
- [x] 新增环境变量解析测试
- [x] 非法配置能明确失败

#### 第 2 次：Step 2

范围：

- [x] Step 2：Cluster YAML 解析与拓扑合法性校验

本次完成标准：

- [x] 解析 `nodes`、`ranks`、`gpus`、`fabric`
- [x] 校验 rank 唯一性和 world size 一致性
- [x] 新增合法/非法 YAML 测试样例
- [x] 非法配置在启动前失败

#### 第 3 次：Step 3 + Step 4

范围：

- [x] Step 3：Coordinator 守护进程最小骨架
- [x] Step 4：控制面协议与 communicator 注册

本次完成标准：

- [x] coordinator 能启动、响应 `ping`、优雅退出
- [x] 定义最小控制面消息格式
- [x] communicator registry 可处理 2 rank / 4 rank 注册
- [x] duplicate rank、缺失 rank、world size 不一致能快速报错

#### 第 4 次：Step 5 + Step 6

范围：

- [x] Step 5：Fake NCCL 最小初始化路径
- [x] Step 6：Shared Memory Staging Layer

本次完成标准：

- [x] `libnccl.so` 最小 shim 可完成 init/destroy
- [x] communicator 初始化与销毁测试通过
- [x] staging buffer 可跨进程读写
- [x] shared memory 生命周期可正确清理

#### 第 5 次：Step 7 + Step 8 + Step 9

范围：

- [x] Step 7：AllReduce 语义执行器
- [x] Step 8：Broadcast 语义执行器
- [x] Step 9：参数一致性校验与超时快失败

本次完成标准：

- [x] 2 rank / 4 rank `all_reduce(sum)` 语义正确
- [x] broadcast 在 root=0 和 root=last rank 下正确
- [x] mismatch 和 timeout 能在限定时间内失败
- [x] direct collective MVP 可稳定重复运行

#### 第 6 次：Step 10

范围：

- [x] Step 10：Torch Distributed 最小 smoke test

本次完成标准：

- [x] 尝试接入 `torch.distributed.init_process_group`
- [x] 记录 barrier / group / async error 等缺口
- [x] 不为了通过 smoke test 临时跨越到后续 Step
- [x] 输出一份明确的缺口清单或探索结果

当前探索结果：

- [x] 单机 2 rank / 4 rank smoke 脚本已接入并可稳定输出探索报告
- [x] 当前首个框架接入缺口为 `libtorch_cuda.so` 依赖的 `ncclCommAbort` 缺失，且已在 Step 11 中关闭
- [x] 探索报告输出到 `test/output/ddp_multinode_2r_gap_report.md` 和 `test/output/ddp_multinode_4r_gap_report.md`

#### 第 7 次：Step 11 + Step 12 + Step 13 + Step 14 + Step 15

范围：

- [x] Step 11：Framework Barrier 支持
- [x] Step 12：GroupStart / GroupEnd 支持
- [x] Step 13：AllGather 与 ReduceScatter
- [x] Step 14：Cluster Report 基础版
- [x] Step 15：DDP 主路径验证

本次完成标准：

- [x] barrier 能完成整组同步和超时失败
- [x] group 语义可批量提交并保持顺序
- [x] all_gather / reduce_scatter 语义正确
- [x] cluster report 生成并包含核心字段
- [x] 单机模拟 2 节点 x 4 rank 的 DDP 主路径可跑通

#### 第 8 次：Step 16 + Step 17 + Step 18

范围：

- [x] Step 16：拓扑配置接入
- [x] Step 17：时间模型与链路级统计
- [x] Step 18：Chunked Staging

本次完成标准：

- [x] fabric 配置可进入模型计算
- [x] link 级统计出现在报告中
- [x] 慢链路下耗时高于快链路
- [x] 大 tensor 在 chunked 模式下可完成传输且结果一致

#### 第 9 次：Step 19 + Step 20 + Step 21

范围：

- [x] Step 19：Hybrid + Simulate 混合运行
- [x] Step 20：Proxy / Passthrough 试验版
- [x] Step 21：远端 Coordinator 与多机扩展

本次完成标准：

- [x] `hybrid + simulate` 可完成 host staging 和 collective 同步
- [x] `proxy` 模式能与 baseline NCCL 做输出对比
- [x] TCP transport 可支持远端 coordinator
- [x] 网络异常时 communicator 可明确回收或失败

### 18.4 批次使用建议

如果你希望 codex 的每次任务更稳，可以遵循下面的投喂规则：

- [ ] 一次只给一个批次，不要同时给多个批次
- [ ] 明确写出“只做这些 Step，不要提前实现后续 Step”
- [ ] 明确写出本次必须补哪些测试
- [ ] 明确写出本次通过标准
- [ ] 要求优先修根因，不要为了过 smoke test 堆临时补丁

如果你希望风险更低，还可以把上面的 9 次进一步细化成 12 次，做法如下：

- [ ] 把“第 4 次”拆成 Step 5 和 Step 6 两次
- [ ] 把“第 7 次”拆成 Step 11 + Step 12、Step 13 + Step 14、Step 15 三次
- [ ] 把“第 9 次”拆成 Step 19、Step 20、Step 21 三次

## 19. 主要风险

### 19.1 PyTorch 对 NCCL 行为细节依赖较重

例如：

- async error
- group 语义
- stream 绑定
- communicator abort 时机

如果第一版行为过于粗糙，容易出现 hang 而不是显式错误。

### 19.2 Staging 可能成为内存瓶颈

大张量 allgather / reduce-scatter 时，coordinator 端内存压力会很高，必须尽早支持 chunking。

### 19.3 “时延模拟”容易与“语义执行”耦合

如果把 timing 直接写进 collective 逻辑里，后续会很难维护。建议从一开始就严格拆层。

### 19.4 多机版本的运维复杂度显著上升

建议先把单机多进程做扎实，尤其是超时、诊断日志和报告。

## 20. 开放问题

在真正进入开发前，建议先明确：

1. Step 1 ~ Step 9 是否只承诺 direct NCCL collective MVP，把 PyTorch 框架兼容性放到 Step 10 之后
2. coordinator 是否必须从 Step 3 起就是单独守护进程，还是先稳定 local coordinator 抽象
3. cluster report 是单文件聚合，还是每 rank 一个文件再统一归并
4. `proxy` 模式是否真的需要在 Step 20 之前就做，还是先把 `simulate` 做完整
5. 是否要把多节点拓扑也做成类似 `profiles/*.yaml` 的内建预设

## 21. 推荐结论

如果目标是尽快得到“可用的多节点模拟能力”，推荐路线是：

1. 先引入独立的 distributed mode，不污染现有 compute mode
2. 先做 Fake NCCL + local coordinator 抽象 + host staging
3. 先做控制面/数据面分离，再补复杂 collective
4. 先保证 collective 语义正确，再补 timing model
5. 先支持单机多进程模拟多节点，再扩成真正多机
6. `proxy/passthrough` 只在 Step 1 ~ Step 18 稳定后再引入
7. 先把 Step 1 ~ Step 9 的 direct NCCL collective MVP 做扎实，再进入 Step 10 之后的 PyTorch DDP 主路径

这条路线和 FakeGPU 当前已有的模式分发、内存跟踪、报告体系是兼容的，后续拆任务和验证也最直接。
