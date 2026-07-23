# Reports & Validation

This page summarizes the built-in test entry points and the report files FakeGPU emits at runtime.

## Maintained test entry points

| Command | What it checks |
|---|---|
| `./ftest smoke` | build, preload, fake device discovery, report schema, multi-architecture profiles, pointer-memory-type coverage |
| `./ftest cpu_sim` | CPU-backed cuBLAS/cuBLASLt correctness against CPU references |
| `./ftest python` | basic PyTorch CUDA device, tensor, and matmul flow |
| `./ftest preflight_oom` | fakecuda fit/OOM classification and report schema |
| `./ftest static_memory_validation` | fake-tensor ATen forward/backward storage liveness, optimizer memory, and optional real-CUDA allocator comparison |
| `./ftest real_gpu_calibration` | real/passthrough/Hybrid/fakecuda memory and result-signature calibration |
| `fakegpu estimate-llm ...` | header-only dense-decoder parameter, KV-cache, transient-memory, and matrix-FLOP estimate |
| `verification/compare_qwen_memory.py ...` | matching real-CUDA/FakeCUDA load, inference, virtual-SMI, token, and FLOP comparison |
| `verification/compare_qwen_sft_memory.py ...` | matching real-CUDA, FakeCUDA, and ATen static-graph peaks for full/LoRA/native-NF4 QLoRA Qwen3.5 SFT |
| `verification/summarize_qwen_sft_matrix.py ...` | full/LoRA/QLoRA, checkpointing, accumulation, and sequence-length SFT matrix summary |
| `verification/run_qwen_fsdp_sft_memory.py ...` | two-rank Hybrid FSDP Qwen SFT parameter/gradient/AdamW sharding, phase peaks, static projection, and collective traffic |
| `verification/run_qwen_fsdp2_lora_sft_memory.py ...` | two- or four-rank mixed-BF16/FP32 FSDP2 LoRA DTensor shards, phase peaks, byte-packed all-gathers, FP32 reduce-scatters, and static projection |
| `python3 verification/test_coordinator_smoke.py` | coordinator startup, request/response, and clean shutdown |
| `python3 test/test_allreduce_correctness.py` | direct all-reduce semantics |
| `python3 verification/test_allgather_correctness.py` | direct all-gather semantics |
| `python3 verification/test_group_semantics.py` | grouped collective submission semantics |
| `./ftest tcp_bandwidth` | chosen-port TCP payload correctness and end-to-end simulator throughput |
| `./ftest elastic_ddp` | active worker exit, full `torchrun` worker-group replacement, restart generation synchronization, resumed DDP numerics, SGD checkpoint recovery, and rank-remapped accumulated AdamW/multi-worker DataLoader recovery over Gloo |
| `./ftest elastic_ddp_checkpoint` | focused atomic checkpoint, completed-step, model parameter, SGD momentum, and continued-update recovery after worker replacement |
| `./ftest elastic_ddp_training_state` | focused AdamW moments, StepLR, replicated rank-state bundle, rank-local RNG, `DistributedSampler` cursor, two-worker DataLoader reconstruction, staged-batch/worker-RNG replay, optimizer-step, pending gradient, and continued accumulation recovery |
| `./ftest distributed_resilience` | deterministic collective failure, real worker exit, elastic DDP restart/checkpoint/training-state resume, collective-timeout inference, async-error propagation, communicator shrink/recovery, TCP mismatch, missing-peer timeout, and bounded report retention |
| `./test/run_hybrid_multinode.sh 2` | maintained multi-process validation with hybrid compute + simulated communication |
| `python3 verification/run_hybrid_ddp_numerics.py --variant all` | real-CUDA DDP basic, `no_sync`, unused-parameter, static-graph, bucket-view, optimizer, and cross-rank parameter checks |
| `python3 verification/run_hybrid_fsdp_numerics.py` | real-CUDA FSDP sharding, reduce-scatter gradients, optimizer result, full-parameter reconstruction, and state-dict restoration |
| `python3 verification/run_hybrid_fsdp2_numerics.py ...` | real-CUDA FSDP2/DeviceMesh/DTensor numerics with two/four ranks, FP32/FP16/BF16 parameters, and FP32 or parameter-dtype gradient reduction |
| `python3 verification/run_physical_multihost.py ...` | repeatable two-host Hybrid DDP/FSDP/FSDP2, fixed-size elastic restart/checkpoint/training-state recovery, mixed precision, injected failure and worker-exit recovery, mismatch, timeout, Git-revision, and report checks over SSH |
| `./ftest llm` | optional LLM smoke test when local model files are available |
| `python test/run_error_simulation_suite.py` | unified Python error simulation suite: cross-device, OOM, invalid device, dtype, checkpoint, and gradient |
| `python test/test_error_cross_device.py` | cross-device tensor operation guards |
| `python test/test_error_oom.py` | per-device OOM simulation |

The first three commands are the best baseline after a code or build change.

## `fake_gpu_report.json`

At process shutdown, FakeGPU writes `fake_gpu_report.json` unless `FAKEGPU_REPORT_PATH` overrides the location.

The report includes:

- runtime mode metadata
- one entry per fake device
- current and peak device memory usage
- IO counters for H2D, D2H, D2D, peer, and memset activity
- compute counters and FLOP estimates for maintained cuBLAS and cuBLASLt paths
- host-to-host copy counters

Example shape:

```json
{
  "report_version": "1.5.4",
  "mode": "simulate",
  "devices": [
    {
      "index": 0,
      "name": "Fake NVIDIA A100-SXM4-80GB",
      "used_memory_peak": 123456,
      "io": {
        "h2d": {"calls": 1, "bytes": 4096}
      },
      "compute": {
        "cublas_gemm": {"calls": 2, "flops": 8192}
      }
    }
  ]
}
```

## Cluster report

When distributed mode is enabled and `FAKEGPU_CLUSTER_REPORT_PATH` is set,
FakeGPU writes the cluster data to JSON and automatically creates a sibling
Markdown project report. Set `FAKEGPU_CLUSTER_REPORT_MARKDOWN_PATH` to choose
a different Markdown path, or set it to `off` to disable the companion.
The maintained DDP and Hybrid validation scripts also embed this complete
node-pair table directly in their final validation reports.

That report includes:

- cluster mode, world size, node count, and coordinator transport
- per-collective counts, bytes, and estimated time
- point-to-point operation, send, and byte totals
- directional link statistics for intra-node and inter-node paths
- every distinct node pair from the configured cluster, including zero-traffic pairs
- collective/P2P operation breakdowns, directional and combined byte totals, largest payload per operation, transfer counts, modeled average/peak throughput, estimated time, and contention
- per-rank wait time, timeout count, communicator init count, and collective/P2P call counts
- injected or collective-timeout-inferred failure and communicator-recovery events, including global rank, operation, observed ranks, attempted payload, exclusions, survivors, and recovery time
- a bounded recent-operation timeline containing global communicator ranks, collective data type/reduction operator, logical and socket payloads, rendezvous time, coordinator execution time, and topology-modeled time

The repository-root `cluster_report.schema.json` defines the
`cluster_report.v1` JSON contract. `verification/check_cluster_report.py`
validates it by default and can additionally require P2P traffic with
`--expect-point-to-point` or resilience events with `--expect-failure` and
`--expect-recovery`. The default validation also reconciles event counters and
all completed collective/P2P counters with retained plus discarded timeline entries. When the
timeline is complete, operation-specific calls and logical bytes must match
exactly. Directional link samples, bytes, peaks, modeled time, and throughput
must also match their node-pair direction.

The Markdown companion presents the node-pair data as a complete table:

| Node A | Node B | Combined total | Pair peak/op | Operations | Collective ops | P2P ops | Transfers | Avg est. Gbit/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `node0` | `node1` | 8.00 GiB | 128.00 MiB | 64 | 60 | 4 | 128 | 18.420 |
| `node0` | `node2` | 0 B | 0 B | 0 | 0 | 0 | 0 | 0.000 |

`peak/op` is the largest payload attributed to the direction or unordered
node pair during one completed communication operation. Throughput, time, and
contention values come from the configured topology model; they are not packet
captures or measured NIC/NCCL bandwidth. The JSON file retains exact integer
byte counters for automated processing.

The timeline's `coordinator_duration_us` begins when the first complete rank
request enters the registry and ends after coordinator-side execution. It does
not include client preparation or final response delivery. Retention defaults
to 4096 latest entries and can be changed with
`FAKEGPU_CLUSTER_REPORT_MAX_OPERATIONS`.

This report is useful for validating control flow, topology modeling, and broad communication-volume trends.

## Preflight report

The AI researcher preflight workflow adds a higher-level report on top of the existing runtime reports.

The runner writes `preflight_report.json`, with a Markdown companion `preflight_report.md`. It answers whether a user command reached a selected stage, whether it triggered OOM under a target GPU profile, and how much memory headroom remains. The JSON contract is defined in the repo-root `preflight_report.schema.json`, and `verification/check_preflight_report.py` validates reports against that schema by default.

Status values:

| Status | Meaning |
|---|---|
| `PASS_FIT` | The selected stage completed and no tracked OOM occurred. |
| `FAIL_OOM` | The run exceeded the target profile or raised an OOM. |
| `FAIL_RUNTIME` | Dependencies, data, model loading, code, or environment setup failed. |
| `WARN_INCOMPLETE_TRACKING` | The run completed, but memory tracking was not complete enough for a strong fit/no-fit judgment. |

Each device entry includes total memory, peak memory, headroom, allocation count, `current_bytes_by_category`, `peak_by_stage`, and `largest_allocations`. In fakecuda mode, top allocations include bytes, dtype, shape, stage, and a coarse category such as `parameter`, `buffer`, `gradient`, `optimizer_state`, `activation`, `temporary`, or `tensor`. Add `--allocation-stacks` to include short Python stack traces for those top allocations.

Static-memory validation reports retain forward, backward, and optimizer CUDA peaks separately. Workspace details distinguish total profiled bytes from the effective peak contribution: graph-phase persistent storage applies across the graph, while operator-local workspace is combined only with the live storage at its ATen node. Reports also list graph-modeled and unprofiled Attention operators.

The current real calibration target is a single NVIDIA RTX PRO 5000 72GB Blackwell (compute capability 12.0). The maintained suite covers seven controlled workloads, requires passthrough and Hybrid result signatures to match real CUDA, records Hybrid Driver allocation peaks, and verifies the PyTorch OOM surface under Hybrid clamp. It does not prove that a multi-node target cluster will fit or perform well.

See [AI Researcher Preflight](ai-researcher-preflight.md) for usage and current limitations.

## LLM inference estimate and virtual SMI

`fakegpu estimate-llm --json <path>` writes a
`fakegpu.llm_inference_estimate.v1` report without loading checkpoint payloads.
It includes parameter/checkpoint metadata, KV cache, prefill/decode transient
memory, tensor/process peaks, and per-step matrix FLOPs.

When `FAKEGPU_SMI_STATE_PATH` or `FAKEGPU_SMI_STATE_DIR` is set before the
FakeCUDA runtime starts, each process publishes a `fakegpu.smi_state.v1` JSON
file. `fakegpu nvidia-smi` displays current and peak tracked tensor memory plus
an optional empirically calibrated runtime overhead.

The Qwen validation worker separates model-load and inference peaks, records
NVML process memory in real mode, executes CPU-backed FakeCUDA with the physical
GPU hidden, and compares observed FLOPs with the shape estimator. See
[LLM Inference Estimation](llm-inference-estimation.md) for commands, measured
results, and the exact scope of the current model.

## Unified HTML test report

`test/report.html` is a self-contained HTML report with tab navigation covering:

- **Phase 1** — device discovery and profile exposure
- **Phase 2** — training flow (nanoGPT on fake GPU)
- **Phase 3** — MoE architecture validation plus torch_patch proof experiments
- **Phase 4** — error simulation experiments (23 tests across 6 categories)

Regenerate it with:

```bash
python test/run_error_simulation_suite.py
```

The report is designed for co-deployment with the mkdocs site at `/test/report.html`.

For the proof-oriented Markdown companion, see:

- `test/real_scene/nanoGPT/TORCH_PATCH_PROOF.md` — 520M / 1.0B load-only scaling, a100-1g OOM validation, and the current scope limitation that op-produced activations are not yet reflected in fakecuda terminal summary peaks.

## Stability guidance

Treat the following as the most stable paths:

- `smoke`
- `cpu_sim`
- `python`
- single-host `simulate + simulate`
- local chosen-port TCP collective and bandwidth validation
- direct NCCL verification plus simulate-mode DDP validation (`test_coordinator_smoke.py`, `test_allreduce_correctness.py`, `test_allgather_correctness.py`, `test_group_semantics.py`, `test_fault_injection_recovery.py`, `run_multinode_sim.sh`, `run_ddp_multinode.sh`)

Treat the following as more environment-sensitive or extended coverage:

- `hybrid` distributed runs
- physical multi-host TCP measurements
- `proxy` and `passthrough` distributed modes
- LLM smoke paths that depend on local model files and broader framework coverage

## Practical validation order

1. Build the repository.
2. Run `./ftest smoke`.
3. Run `./ftest cpu_sim`.
4. Run `./ftest python` if PyTorch is installed.
5. Run `python3 verification/test_coordinator_smoke.py`.
6. Run `python3 test/test_allreduce_correctness.py`.
7. Run `python3 verification/test_allgather_correctness.py`.
8. Run `python3 verification/test_group_semantics.py`.
9. Run `./ftest tcp_bandwidth`.
10. Run `./ftest distributed_resilience`.
11. Run `./test/run_multinode_sim.sh 2`.
12. Run `./test/run_multinode_sim.sh 4`.
13. Run `./test/run_ddp_multinode.sh 4`.
14. Move to `./test/run_hybrid_multinode.sh 2`.
15. On a real CUDA host, run `python3 verification/run_hybrid_ddp_numerics.py --variant all`.
16. On a real CUDA host, run `python3 verification/run_hybrid_fsdp_numerics.py`.
17. On a real CUDA host, run the FSDP2 matrix with `python3 verification/run_hybrid_fsdp2_numerics.py ...`.
18. Run `python3 verification/run_qwen_fsdp_sft_memory.py ...` with a matching static Qwen SFT report.
19. Run `python3 verification/run_qwen_fsdp2_lora_sft_memory.py ...` with a matching LoRA static report.
20. With two synchronized GPU hosts, run `python3 verification/run_physical_multihost.py ...`.
21. Run `python test/run_error_simulation_suite.py` for error simulation coverage.
