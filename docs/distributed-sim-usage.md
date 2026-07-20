# Distributed Simulation Usage

This page starts with the most established single-host multi-rank path, then
shows how to use the same simulator across trusted TCP-connected hosts.

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
| `FAKEGPU_COORDINATOR_TIMEOUT_MS` | Rank rendezvous and operation timeout (default: `1000`) |
| `FAKEGPU_CLUSTER_REPORT_PATH` | Cluster report output path |
| `FAKEGPU_CLUSTER_REPORT_MARKDOWN_PATH` | Optional Markdown project-report path; defaults beside the JSON report |
| `FAKEGPU_CLUSTER_REPORT_MAX_OPERATIONS` | Maximum retained coordinator-observed timeline entries (default: `4096`; `0` disables retention) |
| `FAKEGPU_STAGING_CHUNK_BYTES` | Chunk size threshold for staged transfers |
| `FAKEGPU_STAGING_FORCE_SOCKET` | Force socket fallback instead of shared memory |
| `FAKEGPU_DEVICE_COUNT` | Number of exposed fake devices |

The same knobs are available through `./fgpu` flags:

```bash
./fgpu --mode simulate --dist-mode simulate --cluster-config ... --coordinator-transport unix --coordinator-addr ...
```

## TCP multi-node simulation and bandwidth

The shortest self-contained check starts a coordinator on the exact loopback
port you choose, creates two logical nodes, launches one rank per node, and
moves all-reduce payloads through TCP:

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

With a TCP coordinator, FakeGPU uses socket payloads rather than POSIX shared
memory. This makes the same collective path usable when ranks live on
different physical hosts. The reported throughput is end-to-end: it includes
TCP transfer, coordinator reduction, host memory copies, and process
scheduling. The `bandwidth_gbps` values in the cluster YAML remain a topology
model and are reported separately.

For a physical two-host check, first start one coordinator on a trusted
development network:

```bash
python3 -m fakegpu coordinator \
  --listen 0.0.0.0:29591 \
  --cluster-config verification/data/cluster_tcp_2r.yaml \
  --report /tmp/fakegpu-cluster.json \
  --markdown-report /tmp/fakegpu-project-communication.md
```

The bundled topology assigns rank 0 to `rtx-pro-5000-blackwell` and rank 1 to
`rtx3090ti`. Its TCP fabric values are model inputs; edit them to represent
your intended network separately from the measured result.

Then start the two rank commands concurrently. The endpoint and `--session`
must match:

```bash
# host A
python3 -m fakegpu bandwidth \
  --connect coordinator-host:29591 \
  --world-size 2 \
  --ranks 0 \
  --session test-2026-07-20 \
  --size 16MiB

# host B
python3 -m fakegpu bandwidth \
  --connect coordinator-host:29591 \
  --world-size 2 \
  --ranks 1 \
  --session test-2026-07-20 \
  --size 16MiB
```

After both commands finish:

```bash
python3 -m fakegpu coordinator --shutdown coordinator-host:29591
```

The coordinator protocol has no authentication. Bind it to loopback,
Tailscale, or another trusted interface instead of exposing it to the public
internet.

### Maintained validation snapshot

On 2026-07-20, this path was checked between an RTX PRO 5000
(coordinator and rank 0) and an RTX 3090 Ti (rank 1) over Tailscale:

| Payload | Measured iterations | Correctness | Effective algorithmic bandwidth | Bidirectional socket payload per rank |
|---:|---:|---|---:|---:|
| 1 MiB | 10 | all-reduce sample `[3, 3, 3]` on both ranks | 0.196 Gbit/s | 0.391 Gbit/s |
| 16 MiB | 5 | all-reduce sample `[3, 3, 3]` on both ranks | 0.261 Gbit/s | 0.521 Gbit/s |

The coordinator report recorded two nodes, 18 measured-plus-warmup
all-reduces, inter-node traffic in both directions, and zero rank timeouts.
These values validate the TCP simulator data path; they are not a benchmark
of the GPUs or the underlying network.

### Physical-host Hybrid DDP check

The numerical worker can also verify real CUDA compute with simulated TCP
communication. Start the coordinator as shown above, then execute this command
concurrently on both GPU hosts. Set `NODE_RANK=0` on the first host and
`NODE_RANK=1` on the second:

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

The maintained two-host result used PyTorch 2.9.1/CUDA 12.8 on the RTX PRO
5000 and PyTorch 2.12.1/CUDA 13.0 on the RTX 3090 Ti. Both ranks produced
gradient `[1.5, 3.0]`, updated parameters approximately
`[0.85, -0.30]`, and identical all-gather results. The cluster report recorded
broadcast, all-reduce, and all-gather traffic across two inter-node links with
zero timeouts.

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

Use the maintained checks first:

```bash
python3 verification/test_coordinator_smoke.py
python3 test/test_allreduce_correctness.py
python3 verification/test_allgather_correctness.py
python3 verification/test_group_semantics.py
./ftest tcp_bandwidth
./test/run_hybrid_multinode.sh 2
```

The maintained checks above validate:

- coordinator lifecycle
- direct collective semantics
- grouped submission semantics
- hybrid compute + simulated communication integration

Recommended order:

1. `python3 verification/test_coordinator_smoke.py`
2. `python3 test/test_allreduce_correctness.py`
3. `python3 verification/test_allgather_correctness.py`
4. `python3 verification/test_group_semantics.py`
5. `./test/run_multinode_sim.sh 2`
6. `./test/run_multinode_sim.sh 4`
7. `./test/run_ddp_multinode.sh 4`
8. `./test/run_hybrid_multinode.sh 2`
9. `python3 verification/run_hybrid_ddp_numerics.py` on a real CUDA host

The DDP-oriented scripts above are part of the maintained simulate-mode validation set. They provide smoke and control-flow coverage for ProcessGroupNCCL, but they do not imply full numerical or protocol parity with a real NCCL stack.

The final real-CUDA check is narrower but numerical: two ranks share one
physical device, fake NCCL averages a known non-zero gradient, and the test
verifies the optimizer result and parameters gathered from both ranks.

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
FAKEGPU_CLUSTER_REPORT_MARKDOWN_PATH=/tmp/fakegpu-project-communication.md \
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
FAKEGPU_CLUSTER_REPORT_MARKDOWN_PATH=/tmp/fakegpu-project-communication.md \
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
- point-to-point operation, send, and byte totals
- intra-node and inter-node link statistics
- every distinct node pair, with collective/P2P operation breakdowns, directional totals, combined total, per-operation peak payload, transfer count, and modeled average/peak throughput
- per-rank wait time, timeouts, communicator-init counts, and collective/P2P call counts
- a bounded operation timeline with global ranks, logical/socket payloads, rendezvous time, coordinator execution time, and modeled time

The JSON path also produces a sibling `.md` report by default. Use
`FAKEGPU_CLUSTER_REPORT_MARKDOWN_PATH` or coordinator
`--markdown-report` to choose the final project-report path.
The versioned JSON contract is defined by the repository-root
`cluster_report.schema.json`; `verification/check_cluster_report.py` validates
it by default. Communicator splits retain local-to-global rank membership, so
subgroup traffic is attributed only to the participating cluster nodes.
The coordinator-observed duration starts after a complete request has reached
the communicator registry and ends after coordinator-side execution. It does
not include client preparation or final response delivery and is not a NIC
packet-capture measurement.

## Common failure cases

- `FAKEGPU_COORDINATOR_ADDR` missing while not in `passthrough`
- rank or world-size mismatch between runtime environment and cluster config
- different `--session`, `--size`, or `--iterations` values across physical hosts
- a firewall blocking the selected TCP coordinator port
- forgetting to preload `libnccl.so.2` for the distributed path
- using proxy or passthrough modes before the basic simulate path is already known to work
