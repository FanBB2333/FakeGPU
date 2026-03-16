# FakeGPU 分布式模拟使用说明

这份文档面向“怎么跑起来”。

- 架构和实现细节看 [多节点模拟设计文档](multi-node-design.md)
- 这里主要说明 `simulate` 分布式通信模式的使用方法

## 1. 适用范围

先按下面的目标选组合：

| 目标 | 推荐组合 | 说明 |
|---|---|---|
| 单机先把多 rank / 多节点控制流跑通 | `simulate + simulate` | 最稳定，优先推荐 |
| 真实 GPU 做本地算子，通信仍走虚拟集群 | `hybrid + simulate` | 适合验证“本地算力真实，跨节点通信模拟” |
| 真实 NCCL 跑 collective，同时保留 FakeGPU 统计/报告 | `hybrid + proxy` | 偏实验和对比用途 |
| 尽量薄地转发到真实 NCCL | `passthrough + passthrough` | 更接近纯透传，不适合作为第一条上手路径 |

当前最稳定、最推荐的分布式模拟组合是：

- `FAKEGPU_MODE=simulate`
- `FAKEGPU_DIST_MODE=simulate`

这表示：

- 本地 CUDA / NCCL 都由 FakeGPU 接管
- rank 之间通过 FakeGPU coordinator 协调 collective 和 p2p
- 数据面优先走 POSIX shared memory，必要时会回退到 socket streaming

另外两条已可用但更偏实验性质的路径是：

- `FAKEGPU_MODE=hybrid` + `FAKEGPU_DIST_MODE=simulate`
- `FAKEGPU_MODE=hybrid` + `FAKEGPU_DIST_MODE=proxy|passthrough`

如果只是想在单机上稳定模拟多 rank / 多节点，优先用第一种。

## 2. 前置条件

先确保仓库已经构建完成，至少需要这些产物：

- `build/libnccl.so.2`
- `build/fakegpu-coordinator`
- 根目录包装器 `./fgpu`

常用构建命令：

```bash
cmake -S . -B build
cmake --build build -j4
```

如果你要跑 `torchrun` 相关脚本，还需要：

- 当前 Python 环境能导入 `torch`
- 命令行里能找到 `torchrun`

## 3. 关键配置项

分布式模拟最常用的环境变量如下：

| 变量 | 说明 |
|---|---|
| `FAKEGPU_MODE` | 计算模式，常用 `simulate` / `hybrid` |
| `FAKEGPU_DIST_MODE` | 通信模式，常用 `simulate` / `proxy` / `passthrough` |
| `FAKEGPU_CLUSTER_CONFIG` | cluster YAML 路径 |
| `FAKEGPU_COORDINATOR_TRANSPORT` | `unix` 或 `tcp` |
| `FAKEGPU_COORDINATOR_ADDR` | coordinator 地址；`unix` 时是绝对路径，`tcp` 时是 `host:port` |
| `FAKEGPU_CLUSTER_REPORT_PATH` | cluster report JSON 输出路径 |
| `FAKEGPU_STAGING_CHUNK_BYTES` | 大张量 chunking 阈值 |
| `FAKEGPU_STAGING_FORCE_SOCKET` | 设为 `1` 时强制跳过 shared memory，直接验证 socket fallback |
| `FAKEGPU_DEVICE_COUNT` | 暴露的 fake device 数量 |

对应的 CLI 参数也已经接上，可以通过 `./fgpu` 传入：

```bash
./fgpu --mode simulate --dist-mode simulate --cluster-config ... --coordinator-transport unix --coordinator-addr ...
```

## 4. Cluster Config

最小 cluster config 示例：

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

仓库里已有可直接使用的样例：

- [`verification/data/cluster_valid.yaml`](https://github.com/FanBB2333/FakeGPU/blob/dev/verification/data/cluster_valid.yaml)
- [`verification/data/cluster_proxy_1r.yaml`](https://github.com/FanBB2333/FakeGPU/blob/dev/verification/data/cluster_proxy_1r.yaml)
- [`verification/data/cluster_proxy_2r.yaml`](https://github.com/FanBB2333/FakeGPU/blob/dev/verification/data/cluster_proxy_2r.yaml)

注意：

- rank 需要唯一且连续
- `ranks` 和 `gpus` 数量需要对应
- `unix` transport 下 `FAKEGPU_COORDINATOR_ADDR` 必须是绝对路径

## 5. 最快启动方式

### 5.1 直接跑仓库自带脚本

如果你只是想先确认“分布式模拟能跑”，优先用现成脚本：

2 rank / 4 rank smoke：

```bash
./test/run_multinode_sim.sh 2
./test/run_multinode_sim.sh 4
```

4 rank DDP 主路径：

```bash
./test/run_ddp_multinode.sh 4
```

2 rank hybrid + simulate：

```bash
./test/run_hybrid_multinode.sh 2
```

这些脚本会自动：

- 启动 `fakegpu-coordinator`
- 设置 `LD_PRELOAD`
- 调 `./fgpu`
- 输出日志和报告到 `test/output/`

### 5.2 先跑哪个脚本

建议按下面顺序上手：

1. `./test/run_multinode_sim.sh 2`
2. `./test/run_multinode_sim.sh 4`
3. `./test/run_ddp_multinode.sh 4`
4. 有真实 GPU 时再跑 `./test/run_hybrid_multinode.sh 2`

如果第 1 步就失败，先不要直接看 DDP，先回到第 9 节的自检命令。

## 6. 手动启动 Coordinator + Torchrun

如果你想控制自己的训练脚本，可以按下面的方式手动跑。

### 6.1 启动 coordinator

Unix socket 示例：

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

TCP 示例：

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

### 6.2 运行分布式程序

下面是通用模板：

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

- `./fgpu` 本质上等价于 `python3 -m fakegpu`
- `--device-count` 只是告诉 FakeGPU 暴露多少 fake device
- `torchrun` 的 `--master_addr/--master_port` 是框架 rendezvous；和 FakeGPU coordinator 不是一回事

### 6.3 `hybrid + simulate` 模板

如果你要让本地算子走真实 GPU，而 collective 仍然走 FakeGPU 模拟通信，可以用下面这类模板。

先启动 coordinator：

```bash
SOCKET_PATH=/tmp/fakegpu-hybrid.sock
CLUSTER_CONFIG=$PWD/verification/data/cluster_hybrid_2r.yaml

FAKEGPU_DIST_MODE=simulate \
FAKEGPU_CLUSTER_CONFIG="$CLUSTER_CONFIG" \
FAKEGPU_COORDINATOR_TRANSPORT=unix \
FAKEGPU_COORDINATOR_ADDR="$SOCKET_PATH" \
FAKEGPU_CLUSTER_REPORT_PATH=/tmp/fakegpu-hybrid-report.json \
./build/fakegpu-coordinator --transport unix --address "$SOCKET_PATH"
```

另一个终端里运行：

```bash
FAKEGPU_MODE=hybrid \
FAKEGPU_DEVICE_COUNT=1 \
FAKEGPU_DIST_MODE=simulate \
FAKEGPU_CLUSTER_CONFIG="$CLUSTER_CONFIG" \
FAKEGPU_COORDINATOR_TRANSPORT=unix \
FAKEGPU_COORDINATOR_ADDR="$SOCKET_PATH" \
python3 test/test_hybrid_multinode.py \
  --report-dir /tmp/fakegpu-hybrid-ranks \
  --world-size 2 \
  --python-bin "$(command -v python3)" \
  --nccl-lib "$PWD/build/libnccl.so.2"
```

如果只是想先确认这条路径能跑，直接用：

```bash
./test/run_hybrid_multinode.sh 2
```

### 6.4 `proxy / passthrough` 实验模板

这两条路径更适合做对比验证，不建议作为第一次接入时的默认方案。

如果本机有真实 GPU，并且你想让 FakeGPU 只保留控制面与统计，可以直接运行：

```bash
python3 verification/test_nccl_proxy.py
```

这个脚本会自动完成 baseline、`proxy`、grouped `proxy` 和 grouped `passthrough` 的结果对比。

单 GPU 机器上，它会自动退化到 `world_size=1`；至少 2 张 GPU 时才会走 `world_size=2`。

## 7. 只想给已有命令加 FakeGPU

如果你已经有一条现成命令，也可以只套一层：

```bash
./fgpu \
  --mode simulate \
  --dist-mode simulate \
  --cluster-config "$PWD/verification/data/cluster_valid.yaml" \
  --coordinator-transport unix \
  --coordinator-addr /tmp/fakegpu.sock \
  --device-count 4 \
  python your_script.py
```

或者直接用环境变量：

```bash
export FAKEGPU_MODE=simulate
export FAKEGPU_DIST_MODE=simulate
export FAKEGPU_CLUSTER_CONFIG="$PWD/verification/data/cluster_valid.yaml"
export FAKEGPU_COORDINATOR_TRANSPORT=unix
export FAKEGPU_COORDINATOR_ADDR=/tmp/fakegpu.sock
export LD_PRELOAD="$PWD/build/libnccl.so.2${LD_PRELOAD:+:$LD_PRELOAD}"

python your_script.py
```

## 8. 报告和产物

常见输出包括：

- rank 侧日志
- coordinator 日志
- cluster report JSON
- DDP / hybrid 的 markdown validation report

常见位置：

- `test/output/`
- 你自己设置的 `FAKEGPU_CLUSTER_REPORT_PATH`

如果要检查 cluster report schema，可以用：

```bash
python3 verification/check_cluster_report.py --path /path/to/report.json
```

### 8.1 大张量 chunking 与 socket fallback

默认情况下，数据面会优先走 shared memory；如果 shared memory 不可用，当前实现会自动回退到 socket streaming。

如果你想主动调小 chunk 大小，可以这样跑：

```bash
FAKEGPU_STAGING_CHUNK_BYTES=1048576 ./test/run_ddp_multinode.sh 4
```

这会把大张量按约 1 MiB 的 staging chunk 拆开提交。

如果你想强制验证 socket fallback：

```bash
FAKEGPU_STAGING_FORCE_SOCKET=1 python3 verification/test_socket_staging_fallback.py
```

这个开关主要用于验收和排障，不建议默认长期打开。

## 9. 常用自检命令

如果你怀疑环境或配置有问题，可以先跑这些：

```bash
python3 verification/test_cluster_config.py
python3 verification/test_coordinator_smoke.py
python3 verification/test_communicator_registry.py
python3 verification/test_remote_coordinator.py
python3 verification/test_socket_staging_fallback.py
./build/fakegpu_nccl_direct_test
```

## 10. 常见问题

### 10.1 `coordinator socket was not created`

通常是：

- `fakegpu-coordinator` 没有构建
- `FAKEGPU_COORDINATOR_ADDR` 不是绝对路径
- 目录无写权限

### 10.2 `Invalid FAKEGPU_DIST_MODE`

检查是否拼成了下面四个合法值之一：

- `disabled`
- `simulate`
- `proxy`
- `passthrough`

### 10.3 `rank/world_size` 或 cluster config 不一致

优先检查：

- `torchrun --nproc_per_node`
- `--device-count`
- cluster YAML 中的 `ranks`

这三者需要互相匹配。

### 10.4 想确认 socket fallback 是否生效

可以强制打开：

```bash
FAKEGPU_STAGING_FORCE_SOCKET=1 python3 verification/test_socket_staging_fallback.py
```

这个开关主要用于验证，不建议默认长期打开。

### 10.5 `proxy/passthrough` 该怎么理解

简单理解：

- `simulate`：FakeGPU 自己执行 collective
- `proxy`：真实 NCCL 执行 collective，同时 FakeGPU 记录控制面和 cluster report
- `passthrough`：更接近纯透传，FakeGPU 保留最轻量的包装

如果只是做分布式控制流模拟，不要从 `proxy/passthrough` 起步。

## 11. 推荐使用顺序

建议按这个顺序上手：

1. 先跑 `python3 verification/test_cluster_config.py`
2. 再跑 `python3 verification/test_coordinator_smoke.py`
3. 如果要用 TCP coordinator，再跑 `python3 verification/test_remote_coordinator.py`
4. 再跑 `./test/run_multinode_sim.sh 2`
5. 最后再接自己的 `torchrun` 或训练脚本

这样问题最好定位。
