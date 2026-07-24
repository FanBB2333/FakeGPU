# Repository and Performance Analysis

FakeGPU provides three static checks before a workload is executed:

| Command | Question answered |
|---|---|
| `fakegpu analyze-repo` | Which entrypoints and GPU-specific dependencies need validation? |
| `fakegpu capabilities` | How does the native layer classify supported, simulated, and unsupported APIs? |
| `fakegpu estimate-roofline` | What latency interval follows from one GPU profile and an explicit workload model? |

These checks do not import repository code, execute a kernel, or allocate GPU
memory.

## Scan a repository

```bash
fakegpu analyze-repo /path/to/project \
  --entry train.py \
  --json build/repository-analysis.json
```

The scanner inventories Python, CUDA, PTX, compiled-extension, and
configuration files. It parses Python imports and selected call sites,
collects package dependencies, discovers likely entrypoints, and detects
PyTorch, Transformers, Accelerate, DeepSpeed, PEFT, TRL, Triton,
bitsandbytes, Flash Attention, xFormers, Apex, Lightning, and torchtune.

The readiness verdict has three useful levels:

| Verdict | Meaning |
|---|---|
| `preflight_candidate` | No statically visible GPU-only blocker was found |
| `requires_targeted_validation` | A path such as `torch.compile`, DeepSpeed, FSDP, or model parallelism needs a parameterized experiment |
| `requires_real_gpu_or_hybrid` | Native CUDA, Triton, or another compiled acceleration path must be checked on a matching real stack |
| `analysis_incomplete` | No runnable entrypoint was selected or a Python file could not be parsed |

The report includes suggested preflight and Hybrid experiments. Dynamic
imports, generated kernels, runtime tensor shapes, and data-dependent branches
remain outside a static scan.

## Audit native API behavior

```bash
fakegpu capabilities --source-root . --strict
fakegpu capabilities --source-root . --build-dir build --strict --json -
```

The versioned manifest covers CUDA Runtime, CUDA Driver, cuBLAS, NVML, and
NCCL behavior groups plus explicitly classified high-risk APIs. Source audit
checks that recognized compatibility stubs and unsupported-policy call sites
are declared. Build audit checks exported vendor symbols against an explicit
entry or a reviewed group rule. Strict mode returns a non-zero status when an
audit is incomplete.

`FAKEGPU_UNSUPPORTED_API=error` is the execution-time complement to this
audit: recognized no-op calls return `NotSupported` instead of silently
allowing a compatibility result.

## Estimate a roofline interval

```bash
fakegpu estimate-roofline \
  --profile a100 \
  --flops 2000000000000 \
  --memory-bytes 8000000000 \
  --launch-count 120 \
  --json build/roofline.json
```

The model derives a scalar FP32 ceiling from SM count, compute capability,
profile clock, and CUDA-core issue width. It derives profile memory bandwidth
from memory clock and bus width. For optimistic, expected, and conservative
efficiency assumptions, it computes:

```text
time = max(FLOPs / effective_compute_rate,
           bytes / effective_memory_bandwidth)
       + launch_count × launch_overhead
```

The report contains the arithmetic intensity, ridge point, predicted
bottleneck, all assumptions, and a lower/expected/upper interval. Matrix or
tensor-core acceleration is never guessed. Supply a documented or measured
factor explicitly:

```bash
fakegpu estimate-roofline \
  --profile a100 \
  --flops 2000000000000 \
  --memory-bytes 8000000000 \
  --compute-acceleration-factor 8
```

This is an analytical range, not a CUDA benchmark. Power limits, occupancy,
fusion, tensor shape eligibility, scheduling, cache reuse, and contention can
move real latency outside a poorly specified workload model.

The checkpoint estimator accepts `--target-profile` and uses the same model
with its estimated matrix FLOPs, memory traffic, and launch count:

```bash
fakegpu estimate-llm \
  --model-dir /models/decoder \
  --prompt-tokens 128 \
  --target-profile a100 \
  --json build/decoder-estimate.json
```
