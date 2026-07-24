# Calibration, Training, Kernel, Trace, and Topology Analysis

These commands turn new experiments into data files instead of requiring a
new FakeGPU runtime patch for each test.

| Command | Input | Result |
|---|---|---|
| `fakegpu calibrate compare` | predicted and observed JSON reports | phase errors, interval coverage, error attribution, and safety margin/factor |
| `fakegpu calibrate bundle` | comparison or measurement reports | exact-scope workload calibration bundle |
| `fakegpu plan-training` | DeepSpeed, Accelerate, FSDP, or FSDP2 config | normalized sharding/offload settings and rank-local memory timeline |
| `fakegpu analyze-kernel` | PTX, SASS text, or CUDA source | instructions, registers, shared memory, recognized FLOPs, and optional occupancy ceiling |
| `fakegpu replay-trace` | PyTorch Profiler/Chrome, NCCL-style, or FakeGPU timeline | rank compute/communication/wait, overlap, rank pairs, critical links, memory, and recovery |
| `fakegpu simulate-topology` | JSON/YAML rack/switch/NIC/rank graph | ring, tree, or hierarchical route and contention report |

## Compare prediction and observation

```bash
fakegpu calibrate compare \
  build/static-estimate.json \
  build/observed-memory.json \
  --workload tiny-sft \
  --json build/calibration-comparison.json

fakegpu calibrate bundle \
  build/calibration-comparison.json \
  --output build/workload-calibrations.json
```

The comparison matches phase names when both reports expose a
`memory_timeline`; otherwise it compares their canonical peaks. The output
contains signed/absolute error, percentage error, prediction-interval
coverage, possible missing components, and conservative preflight flags.
Apply those values only to the same workload signature, stack, dtype, shape,
and GPU profile.

## Import a training configuration

```bash
fakegpu plan-training deepspeed.json \
  --parameter-bytes 16000000000 \
  --activation-bytes 4000000000 \
  --world-size 8 \
  --optimizer adamw \
  --json build/training-plan.json
```

The normalizer reads ZeRO stage, parameter/optimizer offload, precision,
gradient accumulation, activation checkpointing, communication overlap, and
bucket sizes. Accelerate configs may contain nested DeepSpeed or FSDP
settings. FSDP/FSDP2 sharding strategy and prefetch settings are normalized to
the same report contract.

The memory model reports model-load, forward, backward, and optimizer phases.
It does not multiply gradient storage by gradient accumulation count. Backend
workspaces, allocator fragmentation, and module-granularity prefetch overlap
remain calibration inputs.

## Inspect generated and native kernels

```bash
fakegpu analyze-repo /path/to/project --json build/repository.json
fakegpu analyze-kernel kernel.ptx \
  --profile rtx4090 \
  --threads-per-block 256 \
  --json build/kernel.json
```

Repository analysis resolves common Python import/function aliases and
decorators. It detects Triton, CuPy RawKernel, Numba CUDA, NVRTC, embedded
CUDA/PTX strings, `cpp_extension`, `CUDAExtension`, CMake CUDA language/toolkit
settings, `nvcc`, custom operator registrations, native sources, and compiled
extensions.

PTX analysis counts static instructions and classes, declared registers,
static shared memory, entry points, recognized scalar/tensor arithmetic, and
profile resource limits. SASS input must be text disassembly. These are static
counts; branch and loop execution counts require a trace or profile.

## Replay traces with custom operator profiles

```bash
fakegpu replay-trace trace.json \
  --operator-profiles verification/data/operator_profiles.example.json \
  --topology verification/data/topology_leaf_spine.json \
  --json build/trace-replay.json
```

Supported roots are Chrome/PyTorch `traceEvents`, generic `events`, and
FakeGPU `operation_timeline.entries`. Event arguments may carry rank,
source/destination rank, bytes, dtype, shape, memory samples, links,
`fusion_id`, and restart generation. A fused profile suppresses child events
with the same `fusion_id`, avoiding launch/FLOP double counting.

The output includes, per rank:

- compute time;
- communication time and compute/communication overlap;
- exposed communication wait and explicit wait;
- total communication bytes and peak effective bandwidth;
- idle or untracked time.

It also includes every observed rank pair, link totals, memory samples suitable
for comparison with virtual `nvidia-smi`, and failure/retry/restart/
communicator events.

## Simulate a routed fabric

```bash
fakegpu simulate-topology \
  verification/data/topology_leaf_spine.json \
  --collective all-reduce \
  --algorithm hierarchical \
  --bytes-per-rank 100000000 \
  --json build/topology-report.json
```

Nodes may declare racks and one or more NICs. Links connect NICs, leaf
switches, spines, or other high-radix tiers with explicit bandwidth and
latency. Ranks can use all node NICs or pin one NIC; equal-cost paths use a
deterministic flow hash. Optional `nvlink_domain` groups add analytical
intra-domain links.

The simulator routes every logical transfer, serializes simultaneous traffic
per link, and reports rounds, paths, contention penalties, rank totals,
rank-pair totals, and critical links. It is not an NCCL, RDMA, or switch-buffer
protocol simulator.
