# Distributed plans and experiment sweeps

[Documentation index](README.md) · [简体中文](../zh-CN/distributed.md) ·
[繁體中文](../zh-TW/distributed.md)

## Distributed training plans

`plan-training` reads DeepSpeed, Accelerate, FSDP, and FSDP2 configs and
estimates memory per rank.

For FSDP2 `full_shard`, `plan-training --parameter-shapes layout.json` accepts
`{"shapes": [[12, 8], [12], [3, 12], [3]]}` together with the matching total
`--parameter-bytes` and a uniform `--parameter-element-bytes`. It models the
native per-parameter dim-0 chunks and budgets the largest rank. Without shapes,
it falls back to a balanced-shard approximation. Custom placements are not
modeled.

The [versioned native validation matrix](../../tests/data/research_native_evidence.json)
records six FP32/FP16/BF16 LoRA and attention CPU_SIM cases, tiny Qwen3 MoE
routing and gradients on CPU and GPU, and two CPU ranks running native DDP and
FSDP2. Physical TP/PP/CP/EP, multi-GPU ZeRO/FSDP, and multi-node collectives
are out of scope. Reproduce it with `scripts/test.sh cpu` and
`tests/native/research_distributed.py --help`.

## Topology, traces, and bandwidth

- `simulate-topology` models collective routes and link contention.
- `replay-trace` summarizes compute, communication, wait, and memory timelines
  from PyTorch Profiler, NCCL-style, or FakeGPU traces.
- `bandwidth` checks simulated TCP payloads and reports throughput. With
  `--listen` and no `--cluster-config`, the generated topology gives each
  logical node the GPU-to-GPU bandwidth of its platform (`--platform`, default
  `auto`; see [Platforms and slots](platforms.md)): the slowest hop of the
  best ring through the node's GPUs, since a ring collective goes at the pace
  of its slowest hop. For example, an 8-GPU HGX A100 board gets NVLink at
  300 GB/s per direction (2400 Gbit/s), and DGX-1 its double NVLinks at
  50 GB/s.
- `coordinator` manages the distributed simulation coordinator.

Distributed timing includes coordinator work, memory copies, sockets, and
process scheduling. It is not an NCCL, NVLink, or RDMA benchmark.

## Experiment sweeps

A finite sweep uses an explicit argv array, a parameter grid, and a repetition
count. Runs are sequential and tied to stable run IDs. Each one keeps its
stdout, exit status, errors, artifact hashes, and numeric metrics.

```bash
python -m fakegpu plan-experiments --config examples/experiments/cpu-sweep.json \
  --workspace /tmp/research-sweep --json /tmp/research-plan.json
python -m fakegpu run-experiments --config examples/experiments/cpu-sweep.json \
  --workspace /tmp/research-sweep --json /tmp/research-results.json.gz
python -m fakegpu run-experiments --config examples/experiments/cpu-sweep.json \
  --workspace /tmp/research-sweep --resume --json /tmp/research-resumed.json
python -m fakegpu compare-experiments --workspace /tmp/research-sweep \
  --plot /tmp/research-sweep.svg --x size --metric seconds \
  --json /tmp/research-comparison.json
```

- `--resume` skips verified successes. `--resume --retry-failed` also retries
  failures and interruptions.
- Changed inputs, sources, or environments need a new workspace.
- Commands must stay in the foreground. Metric paths support nested JSON.
- Plotting needs Matplotlib. Bars show the observed minimum and maximum, not
  confidence intervals.
- The example measures CPU work, not GPU throughput.
