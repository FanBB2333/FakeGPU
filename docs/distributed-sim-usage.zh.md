# 分布式模拟使用说明

这份文档主要讲“怎么把分布式模拟跑起来”，尤其是最稳定的单机多进程路径。

如果你更关心实现思路和边界，请看 [分布式设计说明](multi-node-design.md)。

## 推荐模式组合

| 目标 | 推荐组合 | 说明 |
|---|---|---|
| 第一次把分布式路径跑通 | `simulate + simulate` | 最稳定、最推荐 |
| 本地算子走真实 GPU，通信继续虚拟化 | `hybrid + simulate` | 有真实 GPU 时很实用 |
| 真实 NCCL 做 collective，同时保留 FakeGPU 报告 | `hybrid + proxy` | 更偏对比验证 |
| 尽量薄地转发到真实 NCCL | `passthrough + passthrough` | 不建议作为第一条路径 |

如果只记一个起点，就用：

```bash
FAKEGPU_MODE=simulate
FAKEGPU_DIST_MODE=simulate
```

## 前置条件

至少需要这些产物：

- `build/libnccl.so.2`
- `build/fakegpu-coordinator`
- `./fgpu`

常用构建命令：

```bash
cmake -S . -B build
cmake --build build -j4
```

如果你要跑 `torchrun`：

- 当前 Python 环境里要能导入 `torch`
- `torchrun` 需要在 `PATH` 中可用

## 关键配置

| 变量 | 作用 |
|---|---|
| `FAKEGPU_MODE` | 计算模式 |
| `FAKEGPU_DIST_MODE` | 分布式模式 |
| `FAKEGPU_CLUSTER_CONFIG` | cluster YAML 路径 |
| `FAKEGPU_COORDINATOR_TRANSPORT` | `unix` 或 `tcp` |
| `FAKEGPU_COORDINATOR_ADDR` | 绝对 socket 路径或 `host:port` |
| `FAKEGPU_CLUSTER_REPORT_PATH` | cluster 报告输出路径 |
| `FAKEGPU_STAGING_CHUNK_BYTES` | staging chunk 阈值 |
| `FAKEGPU_STAGING_FORCE_SOCKET` | 强制跳过 shared memory，直接验证 socket fallback |
| `FAKEGPU_DEVICE_COUNT` | 暴露的 fake device 数量 |

这些参数也都能通过 `./fgpu` 传入：

```bash
./fgpu --mode simulate --dist-mode simulate --cluster-config ... --coordinator-transport unix --coordinator-addr ...
```

## 最小 cluster config

```yaml
version: 1
cluster:
  name: dev-cluster
  default_backend: nccl

nodes:
  - id: node0
    host: 127.0.0.1
    ranks: [0, 1]
    gpus:
      - profile: a100
      - profile: a100

  - id: node1
    host: 127.0.0.1
    ranks: [2, 3]
    gpus:
      - profile: h100
      - profile: h100

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
```

仓库自带样例在 `verification/data/` 下。

配置时要注意：

- rank 需要唯一且连续
- 每个 node 的 `ranks` 和 `gpus` 数量要对齐
- 使用 `unix` transport 时，`FAKEGPU_COORDINATOR_ADDR` 必须是绝对路径

## 最快的验证方式

优先用仓库自带脚本：

```bash
./test/run_multinode_sim.sh 2
./test/run_multinode_sim.sh 4
./test/run_ddp_multinode.sh 4
./test/run_hybrid_multinode.sh 2
```

这些脚本会自动：

- 启动 `fakegpu-coordinator`
- 预加载 `libnccl.so.2`
- 调用 `./fgpu`
- 把日志和报告写到 `test/output/`

建议顺序：

1. `./test/run_multinode_sim.sh 2`
2. `./test/run_multinode_sim.sh 4`
3. `./test/run_ddp_multinode.sh 4`
4. `./test/run_hybrid_multinode.sh 2`

## 手动启动 coordinator

### Unix socket 示例

```bash
SOCKET_PATH=/tmp/fakegpu-coordinator.sock
CLUSTER_CONFIG=$PWD/verification/data/cluster_valid.yaml

FAKEGPU_DIST_MODE=simulate \
FAKEGPU_CLUSTER_CONFIG="$CLUSTER_CONFIG" \
FAKEGPU_COORDINATOR_TRANSPORT=unix \
FAKEGPU_COORDINATOR_ADDR="$SOCKET_PATH" \
FAKEGPU_CLUSTER_REPORT_PATH=/tmp/fakegpu-cluster-report.json \
./build/fakegpu-coordinator --transport unix --address "$SOCKET_PATH"
```

### TCP 示例

```bash
COORD_ADDR=127.0.0.1:29591
CLUSTER_CONFIG=$PWD/verification/data/cluster_valid.yaml

FAKEGPU_DIST_MODE=simulate \
FAKEGPU_CLUSTER_CONFIG="$CLUSTER_CONFIG" \
FAKEGPU_COORDINATOR_TRANSPORT=tcp \
FAKEGPU_COORDINATOR_ADDR="$COORD_ADDR" \
FAKEGPU_CLUSTER_REPORT_PATH=/tmp/fakegpu-cluster-report.json \
./build/fakegpu-coordinator --transport tcp --address "$COORD_ADDR"
```

## 手动 `torchrun` 模板

```bash
SOCKET_PATH=/tmp/fakegpu-coordinator.sock
CLUSTER_CONFIG=$PWD/verification/data/cluster_valid.yaml

export LD_PRELOAD="$PWD/build/libnccl.so.2${LD_PRELOAD:+:$LD_PRELOAD}"

./fgpu \
  --mode simulate \
  --dist-mode simulate \
  --cluster-config "$CLUSTER_CONFIG" \
  --coordinator-transport unix \
  --coordinator-addr "$SOCKET_PATH" \
  --device-count 4 \
  torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --master_addr 127.0.0.1 \
  --master_port 29500 \
  your_training_script.py
```

几点说明：

- `./fgpu` 和 `python3 -m fakegpu` 是等价的启动方式
- `--device-count` 控制当前进程能看到多少 fake device
- `torchrun` 的 rendezvous 参数和 FakeGPU coordinator 地址不是一回事

## 如何看输出

当开启分布式并设置 `FAKEGPU_CLUSTER_REPORT_PATH` 后，FakeGPU 会写出 cluster 级报告，里面通常会有：

- world size、transport 等元信息
- 各类 collective 的调用次数、字节数、估算耗时
- 节点间 / 节点内链路统计
- 各 rank 的等待时间、超时次数、communicator 初始化次数

## 常见失败点

- 在非 `passthrough` 模式下没设置 `FAKEGPU_COORDINATOR_ADDR`
- 运行时环境里的 rank / world size 和 cluster config 对不上
- 分布式路径忘了 preload `libnccl.so.2`
- 还没把基础 `simulate + simulate` 跑通就直接尝试 `proxy` 或 `passthrough`
