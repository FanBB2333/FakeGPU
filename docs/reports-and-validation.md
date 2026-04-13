# Reports & Validation

This page summarizes the built-in test entry points and the report files FakeGPU emits at runtime.

## Maintained test entry points

| Command | What it checks |
|---|---|
| `./ftest smoke` | build, preload, fake device discovery, report schema, multi-architecture profiles, pointer-memory-type coverage |
| `./ftest cpu_sim` | CPU-backed cuBLAS/cuBLASLt correctness against CPU references |
| `./ftest python` | basic PyTorch CUDA device, tensor, and matmul flow |
| `python3 verification/test_coordinator_smoke.py` | coordinator startup, request/response, and clean shutdown |
| `python3 test/test_allreduce_correctness.py` | direct all-reduce semantics |
| `python3 verification/test_allgather_correctness.py` | direct all-gather semantics |
| `python3 verification/test_group_semantics.py` | grouped collective submission semantics |
| `./test/run_hybrid_multinode.sh 2` | maintained multi-process validation with hybrid compute + simulated communication |
| `./ftest llm` | optional LLM smoke test when local model files are available |

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
  "report_version": 4,
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

When distributed mode is enabled and `FAKEGPU_CLUSTER_REPORT_PATH` is set, FakeGPU also writes a cluster-level report.

That report includes:

- cluster mode, world size, node count, and coordinator transport
- per-collective counts, bytes, and estimated time
- link statistics for intra-node and inter-node paths
- per-rank wait time, timeout count, communicator init count, and collective-call count

This report is useful for validating control flow, topology modeling, and broad communication-volume trends.

## Stability guidance

Treat the following as the most stable paths:

- `smoke`
- `cpu_sim`
- `python`
- single-host `simulate + simulate`
- direct NCCL verification plus simulate-mode DDP validation (`test_coordinator_smoke.py`, `test_allreduce_correctness.py`, `test_allgather_correctness.py`, `test_group_semantics.py`, `run_multinode_sim.sh`, `run_ddp_multinode.sh`)

Treat the following as more environment-sensitive or extended coverage:

- `hybrid` distributed runs
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
9. Run `./test/run_multinode_sim.sh 2`.
10. Run `./test/run_multinode_sim.sh 4`.
11. Run `./test/run_ddp_multinode.sh 4`.
12. Move to `./test/run_hybrid_multinode.sh 2`.
