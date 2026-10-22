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
  `get_backend(group)` the backend `new_group` was asked for. With no backend
  named, `get_backend()` reports `undefined`, as torch does.
- `all_reduce` (including `ReduceOp.AVG`), `broadcast`, `all_gather`,
  `all_gather_into_tensor`, `reduce_scatter`, `reduce_scatter_tensor`,
  `reduce`, `gather`, `scatter`, `all_to_all`, `all_to_all_single`,
  `barrier`, `all_gather_object` and `broadcast_object_list` exchange real
  values. `ReduceOp.AVG` on integer tensors rounds toward zero. As with NCCL,
  `BAND`, `BOR` and `BXOR` on CUDA tensors raise `ValueError` and `AVG` on
  boolean tensors raises `TypeError`, unless the group asked for gloo.
  `ReduceOp.PREMUL_SUM`, built by `_make_nccl_premul_sum` as FSDP does,
  multiplies each rank's tensor by its factor before summing, as NCCL does; a
  gloo group still rejects it.
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
- A hook registered with `DistributedDataParallel.register_comm_hook`, such as
  the fp16, bf16 or PowerSGD hooks, gets a bucket with `GradBucket`'s methods
  (`buffer()`, `gradients()`, `parameters()`, `index()`, `is_last()` and
  `set_buffer()`), and its result becomes the gradients. The buckets follow
  DDP's reducer: one bucket until the first synced backward (the second with
  `static_graph=True`), then buckets of `bucket_cap_mb` in gradient order;
  with `find_unused_parameters=True` they keep their first layout.
- `model.join()` and `Join([model])` handle uneven inputs: a rank that runs
  out of batches joins the gradient reductions of the ranks still training
  with zeros, and `divide_by_initial_world_size`,
  `throw_on_early_termination`, `join_device` and `join_process_group` behave
  as in DDP.
- Accelerate sees every process, its rank and the world size.

Any other process keeps the in-process simulation, where collectives use
only the local tensors: a single process that names a world size, or one that
finds `WORLD_SIZE` and `MASTER_ADDR` in a cluster pod's environment, so it
does not wait for peers that never start. Set `FAKEGPU_DIST_COLLECTIVES=real`
for launchers FakeGPU does not recognize, such as the DeepSpeed or Lightning
launchers, or `FAKEGPU_DIST_COLLECTIVES=simulate` to simulate in every
process. A simulated `gather` fills every slot of `gather_list` with the local
tensor, and `scatter` hands the simulated rank its own `scatter_list` entry. A
communication hook runs on the local bucket; since its collectives see only
local tensors, the stock `allreduce_hook` leaves the gradients divided by the
world size.

In both modes FakeCUDA checks devices and backends as torch does on GPUs:

- A group made with `nccl`, or without a backend, has no CPU backend, so a
  collective on CPU tensors raises `RuntimeError: No backend type associated
  with device type cpu`. Use a gloo group, or the `cpu:gloo,cuda:nccl`
  backend, for CPU tensors.
- `monitored_barrier` raises torch's `ValueError` unless the group uses gloo
  (from torch 2.14, unless the group has a CPU backend).
- Before `init_process_group` and after `destroy_process_group`,
  `get_rank()`, `get_world_size()` and `get_backend()` raise torch's "Default
  process group has not been initialized" `ValueError`, and so does building
  `DistributedDataParallel`.
- `fully_shard` without a mesh, and `FullyShardedDataParallel`, use the
  simulated CUDA devices, also on macOS where torch would otherwise pick MPS.
  `FullyShardedDataParallel(module, device_id=torch.device("cuda", 0))` moves
  a CPU module to that device. With `fully_shard`, gradients count toward
  memory, but the optimizer states created on its DTensor parameters do not
  yet, and those parameters report device `cpu`.

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
