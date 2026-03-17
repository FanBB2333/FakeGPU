# Distributed Simulation Usage

This page focuses on the path that is easiest to bring up in practice: distributed simulation on a single host with multiple ranks.

For implementation details and design boundaries, see [Distributed Design Notes](multi-node-design.md).

## Recommended mode pairs

| Goal | Recommended pair | Notes |
|---|---|---|
| Stable first distributed bring-up | `simulate + simulate` | Best maintained path |
| Real local compute, simulated communication | `hybrid + simulate` | Useful when a local GPU is available |
| Real NCCL collectives with FakeGPU reports | `hybrid + proxy` | Comparison-oriented and more experimental |
| Minimal wrapping around real NCCL | `passthrough + passthrough` | Not the best first step |

If you only need one answer for where to start, use:

```bash
FAKEGPU_MODE=simulate
FAKEGPU_DIST_MODE=simulate
```

## Prerequisites

Make sure you have built at least:

- `build/libnccl.so.2`
- `build/fakegpu-coordinator`
- `./fgpu`

Typical build command:

```bash
cmake -S . -B build
cmake --build build -j4
```

For `torchrun`-based validation you also need:

- a Python environment with `torch`
- `torchrun` available on `PATH`

## Important settings

| Variable | Meaning |
|---|---|
| `FAKEGPU_MODE` | Compute mode |
| `FAKEGPU_DIST_MODE` | Distributed mode |
| `FAKEGPU_CLUSTER_CONFIG` | Cluster YAML path |
| `FAKEGPU_COORDINATOR_TRANSPORT` | `unix` or `tcp` |
| `FAKEGPU_COORDINATOR_ADDR` | Absolute socket path or `host:port` |
| `FAKEGPU_CLUSTER_REPORT_PATH` | Cluster report output path |
| `FAKEGPU_STAGING_CHUNK_BYTES` | Chunk size threshold for staged transfers |
| `FAKEGPU_STAGING_FORCE_SOCKET` | Force socket fallback instead of shared memory |
| `FAKEGPU_DEVICE_COUNT` | Number of exposed fake devices |

The same knobs are available through `./fgpu` flags:

```bash
./fgpu --mode simulate --dist-mode simulate --cluster-config ... --coordinator-transport unix --coordinator-addr ...
```

## Minimal cluster config

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

Bundled examples live under `verification/data/`.

Important validation rules:

- ranks should be unique and contiguous
- each node should list the same number of `ranks` and `gpus`
- `FAKEGPU_COORDINATOR_ADDR` must be an absolute path when using `unix`

## Fastest way to verify the path

Use the bundled scripts first:

```bash
./test/run_multinode_sim.sh 2
./test/run_multinode_sim.sh 4
./test/run_ddp_multinode.sh 4
./test/run_hybrid_multinode.sh 2
```

Those scripts take care of:

- starting `fakegpu-coordinator`
- preloading `libnccl.so.2`
- launching `./fgpu`
- writing logs and reports under `test/output/`

Recommended order:

1. `./test/run_multinode_sim.sh 2`
2. `./test/run_multinode_sim.sh 4`
3. `./test/run_ddp_multinode.sh 4`
4. `./test/run_hybrid_multinode.sh 2`

## Manual coordinator startup

### Unix socket example

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

### TCP example

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

## Manual `torchrun` template

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

Notes:

- `./fgpu` and `python3 -m fakegpu` are equivalent launch styles.
- `--device-count` controls how many fake devices FakeGPU exposes to the process.
- `torchrun` rendezvous settings are separate from the FakeGPU coordinator address.

## Reading the output

When distributed mode is enabled and `FAKEGPU_CLUSTER_REPORT_PATH` is set, FakeGPU writes a cluster report with:

- world-size and coordinator metadata
- per-collective call counts, bytes, and estimated time
- intra-node and inter-node link statistics
- per-rank wait time, timeouts, and communicator-init counts

## Common failure cases

- `FAKEGPU_COORDINATOR_ADDR` missing while not in `passthrough`
- rank or world-size mismatch between runtime environment and cluster config
- forgetting to preload `libnccl.so.2` for the distributed path
- using proxy or passthrough modes before the basic simulate path is already known to work
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
