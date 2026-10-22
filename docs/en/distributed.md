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

DeepSpeed ZeRO bucket sizes count elements, with DeepSpeed's defaults:
`reduce_bucket_size` and `allgather_bucket_size` 5e8, `stage3_prefetch_bucket_size`
5e7, and `stage3_max_live_parameters` and `stage3_max_reuse_distance` 1e9.

- The reduce bucket takes its element count times the model dtype's bytes,
  however small the model is. With `contiguous_gradients` on (the default),
  stages 1 and 2 hold it during backward and free it before the optimizer step.
  `overlap_comm` doubles it. Stage 3 holds one bucket for the whole run.
- The allgather bucket writes into the parameters in place and takes no memory.
- The ZeRO-3 gathered window is about half of `stage3_max_reuse_distance` plus
  the prefetch bucket. A single rank gathers nothing.
- Sizes set to `auto` take the values the HF Trainer and Accelerate fill in
  when you pass `--hidden-size`: `hidden_size²` for the reduce bucket and
  `0.9·hidden_size²` for the prefetch bucket. Without it, the plan uses the
  DeepSpeed default and says so in its notes.

This follows DeepSpeed 0.15.3 and 0.19.8. FSDP has no bucket setting, so FSDP
plans keep FakeGPU's heuristic of `min(parameter bytes, 500 MB)` per
collective.

FSDP `SHARD_GRAD_OP` and FSDP2 `reshard_after_forward=False` (`shard_grad_op`)
keep each rank's parameter shard at rest. The plan adds the full unsharded
parameters from forward through backward and reshards them for the optimizer
step, as FSDP does after backward on a gradient-sync step.
`_HYBRID_SHARD_ZERO2` (`hybrid_shard_zero2`) does the same within the shard
group and replicates across groups. Like `HYBRID_SHARD`, the group defaults
to Accelerate's processes per machine.

The phases describe a steady-state iteration:

- Forward and backward already hold the optimizer state and master weights
  from earlier steps. ZeRO-3 also holds its gradient partition, and gradient
  accumulation keeps the gradients.
- At the optimizer step, foreach Adam (torch's CUDA default) adds a temporary
  the size of one moment.
- DeepSpeed ZeRO also builds an FP32 copy of the gradient partition. Stages 1
  and 2 free the 16-bit gradients before it.

Single-rank DeepSpeed 0.15.3 runs of a 0.6B BF16 model on an Ampere GPU (sm_80)
matched the predicted step peak within 0.2% for stages 1 to 3. That does not
show the same accuracy at other model sizes or rank counts.

The [versioned native validation matrix](../../tests/data/research_native_evidence.json)
records six FP32/FP16/BF16 LoRA and attention CPU_SIM cases, tiny Qwen3 MoE
routing and gradients on CPU and GPU, and two CPU ranks running native DDP and
FSDP2. Physical TP/PP/CP/EP, multi-GPU ZeRO/FSDP, and multi-node collectives
are out of scope. Reproduce it with `scripts/test.sh cpu` and
`tests/native/research_distributed.py --help`.

<a id="multi-process-training-on-fakecuda"></a>

## Multi-process training on FakeCUDA

FakeCUDA joins real processes into one group when they are launched as a
distributed job: workers of `torchrun` or `accelerate launch`, processes that
`mp.spawn` or `multiprocessing` starts, and scripts that pass an
`init_method` or `store` to `init_process_group`. `init_process_group` then
joins a real gloo group over the CPU tensors behind the simulated GPUs, so
every rank sees the values NCCL would produce:

- `get_rank()` and `get_world_size()` return the real rank and world size.
  `get_backend()` still reports the requested backend, such as `nccl`, and
  `get_backend(group)` the backend `new_group` was asked for.
- `all_reduce` (including `ReduceOp.AVG`), `broadcast`, `all_gather`,
  `all_gather_into_tensor`, `reduce_scatter`, `reduce_scatter_tensor`,
  `reduce`, `gather`, `scatter`, `all_to_all`, `all_to_all_single`,
  `barrier`, `all_gather_object` and `broadcast_object_list` exchange real
  values. `ReduceOp.AVG` on integer tensors rounds toward zero. As with NCCL,
  `BAND`, `BOR` and `BXOR` on CUDA tensors raise `ValueError` and `AVG` on
  boolean tensors raises `TypeError`, unless the group asked for gloo.
- `DistributedDataParallel` broadcasts rank 0's parameters and buffers when it
  is built (no buffers with `broadcast_buffers=False`, nothing with
  `init_sync=False`). Unless `broadcast_buffers=False`, it broadcasts rank 0's
  buffers again before the first forward and before every forward that
  follows a training forward outside `no_sync()`, as DDP does. When backward
  ends it averages the gradients across ranks; with
  `find_unused_parameters=True`, a parameter that one rank did not use counts
  as zeros there. Without it or `static_graph=True`, a trainable parameter
  that got no gradient makes the next forward raise DDP's "Expected to have
  finished reduction in the prior iteration" error, in a simulated process
  too. Inside `no_sync()` gradients accumulate locally, as with real DDP.
- Accelerate sees every process, its rank and the world size.

Any other process keeps the in-process simulation, where collectives use
only the local tensors: a single process that names a world size, or one that
finds `WORLD_SIZE` and `MASTER_ADDR` in a cluster pod's environment, so it
does not wait for peers that never start. Set `FAKEGPU_DIST_COLLECTIVES=real`
for launchers FakeGPU does not recognize, such as the DeepSpeed or Lightning
launchers, or `FAKEGPU_DIST_COLLECTIVES=simulate` to simulate in every
process.
FSDP across processes is not covered by tests yet. Peak memory does not include
the DDP gradient buckets or NCCL buffers a real GPU holds. To run a launcher
on simulated GPUs without changing the script, use
[`fakegpu run`](commands.md#run-a-command-on-simulated-gpus); if `torchrun`
hangs at rendezvous, see
[If torchrun hangs at rendezvous](commands.md#torchrun-hangs-at-rendezvous).

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
