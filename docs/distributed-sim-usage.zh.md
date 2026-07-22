# 分布式模拟使用说明

这份文档先介绍最稳定的单机多进程路径，再说明如何在可信 TCP 主机之间
使用同一套模拟器。

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
| `FAKEGPU_COORDINATOR_TIMEOUT_MS` | rank 汇合及操作等待时间，默认 `1000` 毫秒 |
| `FAKEGPU_CLUSTER_REPORT_PATH` | cluster 报告输出路径 |
| `FAKEGPU_CLUSTER_REPORT_MARKDOWN_PATH` | 可选 Markdown 项目报告路径；默认与 JSON 位于同一目录 |
| `FAKEGPU_CLUSTER_REPORT_MAX_OPERATIONS` | coordinator 观测时间线最多保留的条目数，默认 `4096`；设为 `0` 可关闭保留 |
| `FAKEGPU_STAGING_CHUNK_BYTES` | staging chunk 阈值 |
| `FAKEGPU_STAGING_FORCE_SOCKET` | 强制跳过 shared memory，直接验证 socket fallback |
| `FAKEGPU_DEVICE_COUNT` | 暴露的 fake device 数量 |
| `FAKEGPU_NCCL_FAULT_RANK` | 可选：在 direct 模拟 collective 中失败的全局 rank |
| `FAKEGPU_NCCL_FAULT_SEQNO` | 注入故障的正整数 communicator 序号 |
| `FAKEGPU_NCCL_FAULT_OPERATION` | collective 选择器，默认为 `all_reduce` |

这些参数也都能通过 `./fgpu` 传入：

```bash
./fgpu --mode simulate --dist-mode simulate --cluster-config ... --coordinator-transport unix --coordinator-addr ...
```

## TCP 多节点模拟与带宽测试

下面的命令会监听指定的本地端口，创建两个逻辑节点，每个节点启动一个
rank，并让 all-reduce 的数据经过 TCP：

```bash
python3 -m fakegpu bandwidth \
  --listen 127.0.0.1:29591 \
  --nodes 2 \
  --ranks-per-node 1 \
  --size 4MiB \
  --warmup 2 \
  --iterations 10 \
  --json /tmp/fakegpu-bandwidth.json \
  --cluster-report /tmp/fakegpu-cluster.json
```

coordinator 使用 TCP 时，FakeGPU 会用 socket payload 传输 collective
的输入和输出，不再依赖 POSIX shared memory。因此，同一套通信路径也能
用于两台物理主机。命令输出的是端到端吞吐率，其中包含 TCP 传输、
coordinator reduction、内存复制和进程调度。cluster YAML 中的
`bandwidth_gbps` 是拓扑模型参数，两者会分开报告。

要在两台物理主机之间测试，先在可信的开发网络中启动 coordinator：

```bash
python3 -m fakegpu coordinator \
  --listen 0.0.0.0:29591 \
  --cluster-config verification/data/cluster_tcp_2r.yaml \
  --report /tmp/fakegpu-cluster.json \
  --markdown-report /tmp/fakegpu-project-communication.md
```

仓库中的这份拓扑把 rank 0 配置为 `rtx-pro-5000-blackwell`，rank 1
配置为 `rtx3090ti`。其中的 TCP fabric 数值只是模型输入；可以按计划使用的
网络单独修改，不会改变实测结果。

然后在两台主机上同时执行 rank 命令。两边的 endpoint 和 `--session`
必须相同：

```bash
# 主机 A
python3 -m fakegpu bandwidth \
  --connect coordinator-host:29591 \
  --world-size 2 \
  --ranks 0 \
  --session test-2026-07-20 \
  --size 16MiB

# 主机 B
python3 -m fakegpu bandwidth \
  --connect coordinator-host:29591 \
  --world-size 2 \
  --ranks 1 \
  --session test-2026-07-20 \
  --size 16MiB
```

两个 rank 完成后停止 coordinator：

```bash
python3 -m fakegpu coordinator --shutdown coordinator-host:29591
```

coordinator 协议目前没有认证。请绑定到 loopback、Tailscale 或其他可信
接口，不要直接暴露到公网。

### 当前维护的验证结果

2026-07-20 在两台物理主机之间完成了验证：RTX PRO 5000 负责
coordinator 和 rank 0，RTX 3090 Ti 负责 rank 1，主机间通过 Tailscale
连接。

| 消息大小 | 计时迭代次数 | 正确性 | 有效算法带宽 | 每个 rank 的双向 socket 负载 |
|---:|---:|---|---:|---:|
| 1 MiB | 10 | 两个 rank 的 all-reduce 样本均为 `[3, 3, 3]` | 0.196 Gbit/s | 0.391 Gbit/s |
| 16 MiB | 5 | 两个 rank 的 all-reduce 样本均为 `[3, 3, 3]` | 0.261 Gbit/s | 0.521 Gbit/s |

coordinator 报告记录了两个节点、18 次计时及预热 all-reduce、双向节点间
流量，且两个 rank 均没有超时。这组数值用于确认 TCP 模拟数据路径，
不代表 GPU 性能或底层网络的原始带宽。

### 物理多机 Hybrid DDP 检查

数值检查 worker 也可以验证“真实 CUDA 计算 + 模拟 TCP 通信”。先按前面的
命令启动 coordinator，再在两台 GPU 主机上同时执行下面的命令。第一台
设置 `NODE_RANK=0`，第二台设置 `NODE_RANK=1`：

```bash
export NODE_RANK=0
export COORDINATOR_HOST=coordinator-host
export FAKEGPU_MODE=hybrid
export FAKEGPU_DEVICE_COUNT=1
export FAKEGPU_DIST_MODE=simulate
export FAKEGPU_CLUSTER_CONFIG="$PWD/verification/data/cluster_tcp_2r.yaml"
export FAKEGPU_COORDINATOR_TRANSPORT=tcp
export FAKEGPU_COORDINATOR_ADDR="${COORDINATOR_HOST}:29591"
export FAKEGPU_COORDINATOR_TIMEOUT_MS=60000

LD_PRELOAD="$PWD/build/libnccl.so.2" \
python3 -m torch.distributed.run \
  --nnodes=2 \
  --nproc-per-node=1 \
  --node-rank="$NODE_RANK" \
  --master-addr="$COORDINATOR_HOST" \
  --master-port=29592 \
  --max-restarts=0 \
  test/test_ddp_hybrid_numerics.py \
  --report-dir="/tmp/fakegpu-cross-ddp-rank${NODE_RANK}"
```

当前维护的两机结果使用 RTX PRO 5000 上的 PyTorch 2.9.1/CUDA 12.8，
以及 RTX 3090 Ti 上的 PyTorch 2.12.1/CUDA 13.0。基础 DDP、全部 DDP
选项场景、FSDP 和 FSDP2 都得到预期梯度及一致的重建参数。DDP/FSDP/异常
场景的完整 cluster 报告记录 34 个成功通信操作、节点对总量 1,104 字节、
节点对单次峰值 128 字节，并记录一次预期的缺少 rank 超时。单独的 FSDP2
会话通过了 FP32/FP16/BF16 参数路径以及 FP16/BF16 梯度归约；其中低精度
会话保留 8 个操作，节点对总量 160 字节，单次峰值 32 字节，时间线能够
区分 `float16`/`bfloat16` 负载与 `sum`/`avg` 归约。

### 可重复执行的 SSH 控制脚本

`verification/run_physical_multihost.py` 将上述手动步骤整合为一条控制端
命令。脚本不会复制代码，也不会修改远端仓库；两台主机需要预先通过 Git
同步到相同 commit，并完成 native build。

`--node` 的顺序对应 rank。SSH 进入 Windows、而仓库和 GPU 环境位于 WSL
时，设置 `shell=wsl`：

```bash
python3 verification/run_physical_multihost.py \
  --node 'name=blackwell;ssh=gpu-a;repo=/home/user/repos/fakeGPU;python=/opt/fakegpu/bin/python;shell=posix' \
  --node 'name=ampere-wsl;ssh=user@gpu-b;repo=/home/user/repos/fakeGPU;python=/opt/torch/bin/python;shell=wsl' \
  --coordinator-host 100.x.y.z
```

all-to-all-v 默认每个分片单位包含一个 FP32 元素。需要验证更大的 TCP payload
时，可以通过参数放大同一组非均匀与稀疏分片，不需要修改代码：

```bash
python3 verification/run_physical_multihost.py \
  --node 'name=blackwell;ssh=gpu-a;repo=/home/user/repos/fakeGPU;python=/opt/fakegpu/bin/python;shell=posix' \
  --node 'name=ampere-wsl;ssh=user@gpu-b;repo=/home/user/repos/fakeGPU;python=/opt/torch/bin/python;shell=wsl' \
  --coordinator-host 100.x.y.z \
  --case alltoallv \
  --alltoallv-elements-per-unit 262144
```

默认包含以下场景：

- 异构两主机 Hybrid DDP 数值正确性
- DDP `no_sync`、不同 rank 使用不同分支时的未使用参数、静态图和 gradient bucket view
- FSDP 全分片、reduce-scatter、optimizer、完整参数和 state dict 正确性
- FSDP2/DTensor FP32 分片、optimizer 与完整张量重建
- FSDP2 FP16/BF16 参数配合 FP32 梯度归约
- FSDP2 FP16/BF16 参数 dtype 梯度归约
- 两台物理主机之间的 grouped 非均匀与稀疏 all-to-all-v
- collective reduction operator 不一致以及持续可见的 async error
- 确定性的 rank 2 All-Reduce 故障、持续 `ncclRemoteError`、三 rank `ncclCommShrink` 与恢复后的 All-Reduce
- 从第二台物理主机触发缺少 rank 的 communicator 超时

只执行恢复场景时使用 `--case fault-shrink`。控制脚本会把四个逻辑 rank
分布到两个 node 配置：rank 0、2 位于第一台主机，rank 1、3 位于第二台。
排除 rank 2 后，子 communicator 包含全局 ranks `[0, 1, 3]`，本地 rank
为 `[0, 1, 2]`。Cluster 报告会同时记录注入故障和恢复事件。

DeepSpeed 是可选场景，不在默认集合中。添加 `--case deepspeed-zero2` 可以验证
每台物理主机各运行一个 rank；维护中的异构实验已在 DeepSpeed 0.15.3 和
0.19.2 之间通过。ZeRO-3 的 collective 序列与 DeepSpeed 版本有关，因此
`--case deepspeed-zero3` 要求两端版本一致，版本不同时会在预检阶段直接报告。

MiB 级物理 all-to-all-v 用例已在 RTX PRO 5000 与 RTX 3090 Ti WSL 主机之间
通过。非均匀跨主机分片为 2 MiB/3 MiB，稀疏分片为 0 MiB/1 MiB，因此覆盖了
PRO 5000 → 3090 Ti 的零字节方向。合并报告记录了 2 次 all-to-all、12 MiB
逻辑数据、6 MiB 跨节点数据，节点对单次峰值为 5 MiB。rank 报告保存 payload
样本、SHA-256 和完整元素检查状态，不会将 MiB 级数组写入 JSON。

启动前会检查两端的 tracked Git 状态、精确 commit、Python/PyTorch/CUDA
信息和 native 构建产物。合并报告写入
`build/physical_multihost_validation/<session>/`，其中包含各 rank 结果、
完整 cluster JSON/Markdown 报告、节点对通信总量和异常观测。对应的单机
回归检查入口是 `./ftest distributed_resilience`。

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

优先用当前维护中的检查：

```bash
python3 verification/test_coordinator_smoke.py
python3 test/test_allreduce_correctness.py
python3 verification/test_allgather_correctness.py
python3 verification/test_group_semantics.py
./ftest tcp_bandwidth
./test/run_hybrid_multinode.sh 2
python3 verification/run_hybrid_ddp_numerics.py --variant all
python3 verification/run_hybrid_fsdp_numerics.py
python3 verification/run_hybrid_fsdp2_numerics.py --world-size 4 --precision bf16
python3 verification/run_hybrid_fsdp2_numerics.py --world-size 4 --precision bf16 --reduce-precision parameter
```

这些检查分别覆盖：

- coordinator 生命周期
- direct collective 语义
- grouped submission 语义
- hybrid 计算 + simulate 通信集成
- 真实 CUDA 上的 DDP 常见选项数值结果
- FSDP 分片、reduce-scatter、参数重建与 checkpoint 恢复
- 双/四 rank FSDP2/DTensor 分片、重建与混合精度通信

建议顺序：

1. `python3 verification/test_coordinator_smoke.py`
2. `python3 test/test_allreduce_correctness.py`
3. `python3 verification/test_allgather_correctness.py`
4. `python3 verification/test_group_semantics.py`
5. `./test/run_multinode_sim.sh 2`
6. `./test/run_multinode_sim.sh 4`
7. `./test/run_ddp_multinode.sh 4`
8. `./test/run_hybrid_multinode.sh 2`
9. 在真实 CUDA 主机上执行 `python3 verification/run_hybrid_ddp_numerics.py --variant all`
10. 在真实 CUDA 主机上执行 `python3 verification/run_hybrid_fsdp_numerics.py`
11. 在真实 CUDA 主机上执行 `python3 verification/run_hybrid_fsdp2_numerics.py ...`

上面的 DDP 脚本已经属于当前维护中的 simulate-mode 验证集。它们提供的是 ProcessGroupNCCL 的 smoke / 控制流覆盖，并不意味着已经达到真实 NCCL 的完整数值或协议等价。

真实 CUDA 检查会核对具体数值。DDP runner 覆盖基础梯度平均、`no_sync`
梯度累积、不同 rank 使用不同分支时的未使用参数、静态图和 gradient bucket
view。FSDP runner 会验证真实双 rank 全分片、reduce-scatter 梯度平均、
optimizer 更新、完整参数重建和完整 state dict 恢复。两个 runner 都支持
两个 rank 共用一张物理卡。FSDP2 runner 还支持四个 rank、DTensor 重建、
FP16/BF16 参数，以及 FP32 或参数 dtype 梯度归约；物理多主机控制器默认
让每台已同步的 SSH 主机各承担一个 rank。

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
FAKEGPU_CLUSTER_REPORT_MARKDOWN_PATH=/tmp/fakegpu-project-communication.md \
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
FAKEGPU_CLUSTER_REPORT_MARKDOWN_PATH=/tmp/fakegpu-project-communication.md \
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
- P2P 操作数、发送次数和字节数
- 节点间 / 节点内链路统计
- 全部不同节点的两两组合，包含 collective/P2P 操作分类、方向总量、双向总量、单次操作峰值、传输次数，以及模型平均/峰值吞吐
- 各 rank 的等待时间、超时次数、communicator 初始化次数，以及 collective/P2P 调用次数
- 故障与 communicator 恢复事件，包含全局 rank、操作、观测 ranks、尝试负载、排除/存活集合和恢复耗时
- 有大小限制的操作时间线，记录全局 ranks、collective 数据类型/归约运算、逻辑/socket 负载、汇合时间、coordinator 执行时间和模型时间

设置 JSON 路径后，默认会在同一目录生成 `.md` 报告。通过
`FAKEGPU_CLUSTER_REPORT_MARKDOWN_PATH` 或 coordinator 的
`--markdown-report` 参数，可以指定最终项目报告路径。
版本化 JSON 契约由仓库根目录的 `cluster_report.schema.json` 定义，
`verification/check_cluster_report.py` 默认会按该契约校验。communicator
split 会保留局部 rank 到集群全局 rank 的映射，因此 subgroup 流量只会
归到实际参与的节点。
coordinator 观测时间从完整请求进入 communicator registry 开始，到
coordinator 侧执行结束为止；它不包含客户端准备和最终响应送达时间，也不
等同于网卡抓包结果。

## 常见失败点

- 在非 `passthrough` 模式下没设置 `FAKEGPU_COORDINATOR_ADDR`
- 运行时环境里的 rank / world size 和 cluster config 对不上
- 两台主机使用了不同的 `--session`、`--size` 或 `--iterations`
- 防火墙拦截了选定的 TCP coordinator 端口
- 分布式路径忘了 preload `libnccl.so.2`
- 还没把基础 `simulate + simulate` 跑通就直接尝试 `proxy` 或 `passthrough`
