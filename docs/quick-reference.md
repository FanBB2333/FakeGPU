# Quick Reference

## Build commands

```bash
cmake -S . -B build
cmake --build build
```

Enable verbose FakeGPU logging:

```bash
cmake -S . -B build -DENABLE_FAKEGPU_LOGGING=ON
cmake --build build
```

Disable CPU-backed cuBLAS/cuBLASLt simulation:

```bash
cmake -S . -B build -DENABLE_FAKEGPU_CPU_SIMULATION=OFF
cmake --build build
```

## Common run commands

```bash
./fgpu nvidia-smi
./fgpu python3 your_script.py
./fgpu --profile t4 --device-count 2 python3 your_script.py
./fgpu --devices "a100:4,h100:4" python3 your_script.py
./fgpu --mode hybrid --oom-policy clamp python3 your_script.py
```

Dynamic initialization inside Python:

```bash
python3 -c "import fakegpu; fakegpu.init(runtime='native'); import torch; print(torch.cuda.device_count())"
```

Python-level fake-CUDA routing:

```bash
python3 -c "import fakegpu; print(fakegpu.init(runtime='auto').runtime)"
```

Tiny Transformer training demo with `pytorch-fakegpu`:

```bash
python3 demo_usage.py --test transformer
python3 demo_usage.py --test transformer --quiet
```

This route uses `fakegpu.torch_patch.patch()` inside the demo and is meant for
fake-CUDA training smoke tests on CPU-only hosts.

## Preflight / OOM checks

Run a fakecuda preflight before submitting a Python training command:

```bash
fakegpu preflight \
  --runtime fakecuda \
  --devices a100-1g:1 \
  --stage forward \
  --report-dir preflight-a100-1g \
  --allocation-stacks \
  --strict \
  -- python3 train.py --small-config
```

The runner writes:

- `preflight_report.json`
- `preflight_report.md`
- `preflight_stdout.log`
- `preflight_stderr.log`

Use a small profile such as `a100-1g` to confirm OOM detection, then repeat with the target profile. For lightweight regression tests, `test-512m` is also available as a 512 MB fakecuda/native profile. The runner auto-initializes fakecuda for Python commands and reports `C2_torch_tensor_lifetime` confidence, including stage peaks, top allocations, optional allocation stack traces, coarse memory categories, shared-storage alias handling, basic logical-device attribution, and saved autograd tensors visible through PyTorch hooks. CUDA backend-internal workspaces can still be undercounted; use `--memory-safety-margin <bytes>` when real-GPU calibration shows a mostly fixed gap, and reserve `--memory-safety-factor <factor>` for gaps that scale with workload size.

`./ftest preflight_oom` now includes a profile matrix check: the same 560 MB allocation must fail on `test-512m` and pass on `a100`.

When `--strict` is set, skipped child tests are treated as `FAIL_RUNTIME` instead of a passing preflight.

For real-GPU calibration, run a reduced workload directly on the GPU and compare with passthrough or hybrid when available:

```bash
./ftest real_gpu_calibration
python3 train.py --small-config
./fgpu --mode passthrough python3 train.py --small-config
./fgpu --mode hybrid --oom-policy clamp python3 train.py --small-config
```

The calibration suite writes `build/real_gpu_calibration/calibration_real_gpu.json` and `.md`. It auto-selects `rtx-pro-5000-blackwell` for the current server and records an explicit skip reason when CUDA, PyTorch, or a matching profile is unavailable. The suite includes tensor, MLP, Tiny Transformer, gradient accumulation, gradient checkpointing, Hugging Face tiny GPT-2, and PEFT LoRA tiny GPT-2 workloads. By default, it performs one warmup followed by three measured trials and uses the largest observed peak as an empirical upper bound. Each trial retains PyTorch allocated/reserved/requested peaks and NVML samples. When NVML exposes the current PID, the report includes process memory; when WSL does not expose that mapping, the report explicitly marks process sampling unavailable and keeps device-memory deltas. Each workload runs on real CUDA, passthrough, Hybrid clamp, and fakecuda. Calibration evidence applies only to an exact GPU profile and workload signature under a comparable software stack.

Combine reports from multiple GPUs into an empirical lookup bundle, then let preflight select the matching profile:

```bash
python3 verification/aggregate_real_gpu_calibrations.py \
  reports/3090ti/calibration_real_gpu.json \
  reports/pro5000/calibration_real_gpu.json \
  --output build/calibration_bundle.json \
  --markdown build/calibration_bundle.md

python3 -m fakegpu preflight \
  --runtime fakecuda \
  --profile rtx3090ti \
  --memory-calibration build/calibration_bundle.json \
  --calibration-workload tiny_transformer_step \
  --report-dir preflight-report \
  -- python3 train.py
```

This path uses the observed real-CUDA upper bound instead of fitting a universal factor. If a workload name has more than one signature, select the full signature. Do not reuse a calibration after changing batch size, sequence length, or model shape.

See [AI Researcher Preflight](ai-researcher-preflight.md) for the current design and limitations.

## Test commands

```bash
./ftest smoke
./ftest cpu_sim
./ftest python
./ftest preflight_oom
./ftest real_gpu_calibration
./ftest all
```

```bash
./test/run_comparison.sh
python3 verification/test_coordinator_smoke.py
python3 test/test_allreduce_correctness.py
python3 verification/test_allgather_correctness.py
python3 verification/test_group_semantics.py
./test/run_multinode_sim.sh 2
./test/run_multinode_sim.sh 4
./test/run_ddp_multinode.sh 4
./test/run_hybrid_multinode.sh 2
```

These cover the maintained simulate-mode DDP path; they are smoke and control-flow checks, not a claim of full PyTorch/NCCL parity.

### Error simulation tests

```bash
python test/run_error_simulation_suite.py   # all 23 tests + HTML report
python test/test_error_cross_device.py      # E1: cross-device
python test/test_error_oom.py               # E2: OOM
python test/test_error_device_index.py      # E3: invalid device
python test/test_error_dtype_autocast.py    # E4: dtype / autocast
python test/test_error_checkpoint_load.py   # E5: checkpoint load
python test/test_error_gradient.py          # E7: gradient
```

## Manual preload

Using `./fgpu` is recommended. If you need manual control:

### Linux

```bash
LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH \
LD_PRELOAD=./build/libcublas.so.12:./build/libcudart.so.12:./build/libcuda.so.1:./build/libnvidia-ml.so.1 \
python3 your_script.py
```

### macOS

```bash
DYLD_LIBRARY_PATH=./build:$DYLD_LIBRARY_PATH \
DYLD_INSERT_LIBRARIES=./build/libcublas.dylib:./build/libcudart.dylib:./build/libcuda.dylib:./build/libnvidia-ml.dylib \
python3 your_script.py
```

Mode-specific preload behavior in the Python API:

| Compute mode | Fake libraries loaded by `fakegpu.init(runtime=\"native\")` / `fakegpu.env()` |
|---|---|
| `simulate` | cuBLAS + CUDA Runtime + CUDA Driver + NVML |
| `hybrid` | CUDA Runtime + CUDA Driver + NVML |
| `passthrough` | CUDA Runtime + CUDA Driver |

## Environment variables

### Compute and profiles

| Variable | Meaning |
|---|---|
| `FAKEGPU_MODE` | `simulate`, `hybrid`, or `passthrough` |
| `FAKEGPU_OOM_POLICY` | Hybrid oversubscription strategy |
| `FAKEGPU_PROFILE` | One preset ID for every fake device |
| `FAKEGPU_DEVICE_COUNT` | Number of fake devices to expose |
| `FAKEGPU_PROFILES` | Per-device preset spec such as `a100:4,h100:4` |
| `FAKEGPU_REAL_CUDA_LIB_DIR` | Override directory for real CUDA libraries |

### Distributed

| Variable | Meaning |
|---|---|
| `FAKEGPU_DIST_MODE` | `disabled`, `simulate`, `proxy`, or `passthrough` |
| `FAKEGPU_CLUSTER_CONFIG` | Cluster YAML path |
| `FAKEGPU_COORDINATOR_TRANSPORT` | `unix` or `tcp` |
| `FAKEGPU_COORDINATOR_ADDR` | Socket path or `host:port` |
| `FAKEGPU_CLUSTER_REPORT_PATH` | Output path for cluster-level JSON report |
| `FAKEGPU_STAGING_CHUNK_BYTES` | Chunk size for staged transfers |
| `FAKEGPU_STAGING_FORCE_SOCKET` | Set to `1` to skip shared memory and force socket fallback |

### Reporting and debugging

| Variable | Meaning |
|---|---|
| `FAKEGPU_REPORT_PATH` | Output path for `fake_gpu_report.json` |
| `PYTORCH_NO_CUDA_MEMORY_CACHING` | Useful when debugging allocation flow |
| `TORCH_SDPA_KERNEL=math` | Helpful for avoiding Flash Attention-specific paths |
| `CUDA_LAUNCH_BLOCKING=1` | Forces synchronous error surfacing |

### Error simulation

| Variable | Meaning |
|---|---|
| `FAKEGPU_CROSS_DEVICE_CHECK` | Cross-device operation guard; `0` to disable |
| `FAKEGPU_MEMORY_TRACKING` | Per-device memory tracking and OOM simulation; `0` to disable |
| `FAKEGPU_STRICT_COMPAT` | Strict dtype and architecture compatibility checks; `0` to disable |

## Troubleshooting

Reset a broken terminal:

```bash
reset
```

Inspect exported NVML symbols:

Linux:

```bash
nm -D ./build/libnvidia-ml.so.1 | grep ' T nvml'
```

macOS:

```bash
nm -gU ./build/libnvidia-ml.dylib | rg '\\bnvml'
```

Inspect dynamic-library dependencies:

Linux:

```bash
ldd ./build/libcuda.so.1
ldd ./build/libcudart.so.12
ldd ./build/libcublas.so.12
ldd ./build/libnvidia-ml.so.1
```

macOS:

```bash
otool -L ./build/libcuda.dylib
otool -L ./build/libcudart.dylib
otool -L ./build/libcublas.dylib
otool -L ./build/libnvidia-ml.dylib
```

## Related pages

- [Getting Started](getting-started.md)
- [Architecture](project-structure.md)
- [Reports & Validation](reports-and-validation.md)
- [Distributed Simulation Usage Guide](distributed-sim-usage.md)
- [Error Simulation](error-simulation.md)
