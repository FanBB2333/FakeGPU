# AI Researcher Preflight

This page describes the intended preflight workflow for AI researchers who want to check a training or inference command before submitting it to a large GPU cluster.

The immediate goal is narrow: determine whether the command can reach a selected stage and whether the workload is likely to hit OOM under a target GPU profile. FakeGPU does not currently try to predict GPU utilization, step time, throughput, or cluster scheduling behavior.

## Local Hardware Assumption

The current real calibration machine is a single NVIDIA GeForce RTX 3090 Ti with 24GB of VRAM.

That machine is useful for:

- checking that the real CUDA / PyTorch / transformers environment works
- calibrating small and medium workloads that fit within 24GB
- comparing real PyTorch `torch.cuda.max_memory_allocated()` with FakeGPU reports
- validating 24GB OOM boundaries
- sanity-checking fakecuda memory tracking on controlled workloads

It cannot directly prove:

- that an 80GB A100 or H100 cluster run will fit
- that a multi-GPU NCCL, NVLink, RDMA, or InfiniBand path behaves correctly
- that cluster throughput or GPU utilization will be good
- that CUDA kernels produce numerically identical results in simulate or fakecuda mode

Treat the 3090 Ti as a calibration point, not as a substitute for the target cluster.

## What Preflight Should Answer

For a command such as:

```bash
python train.py --model qwen --batch-size 4 --seq-len 4096
```

the preflight workflow should answer:

- Did the command start and import dependencies correctly?
- Did model loading finish?
- Did forward pass finish?
- Did backward pass finish?
- Did one optimizer step finish?
- What was the peak memory per logical GPU?
- How much headroom was left under the selected target GPU profile?
- If it failed, was the failure an OOM or a runtime/configuration error?
- How complete is the memory tracking behind this conclusion?

## Current Target Workflow

The initial runner provides a single preflight entry point for Python commands:

```bash
fakegpu preflight \
  --runtime fakecuda \
  --devices a100:8 \
  --stage train_step \
  --steps 1 \
  --report-dir preflight-a100 \
  --strict \
  -- python train.py --model qwen --batch-size 4 --seq-len 4096
```

Expected outputs:

- `preflight_report.json`
- `preflight_report.md`
- `preflight_stdout.log`
- `preflight_stderr.log`

The report status is one of:

| Status | Meaning |
|---|---|
| `PASS_FIT` | The selected stage completed and no tracked OOM occurred. |
| `FAIL_OOM` | The workload exceeded the selected target profile or raised an OOM. |
| `FAIL_RUNTIME` | The run failed for dependencies, data, model loading, code errors, or environment problems. |
| `WARN_INCOMPLETE_TRACKING` | The command ran, but memory tracking was too incomplete for a strong fit/no-fit answer. |

The initial fakecuda runner auto-initializes Python commands before executing the script, module, or `-c` code. Non-Python commands can still be run through native, hybrid, or passthrough modes, but fakecuda cannot auto-patch them.

## Recommended Workflow

### 1. Check FakeGPU Baseline

```bash
./ftest smoke
./ftest cpu_sim
./ftest python
./ftest preflight_oom
```

These commands verify that the build, preload path, reports, GPU profiles, CPU-backed cuBLAS paths, and basic PyTorch CUDA surface are working.

### 2. Run A Fakecuda OOM Probe

Run the preflight command with a small profile:

```bash
fakegpu preflight \
  --runtime fakecuda \
  --devices a100-1g:1 \
  --stage forward \
  --report-dir preflight-a100-1g \
  --strict \
  -- python train.py --small-config
```

Then repeat with the intended target profile:

```bash
fakegpu preflight \
  --runtime fakecuda \
  --devices a100:8 \
  --stage forward \
  --report-dir preflight-a100 \
  --strict \
  -- python train.py --cluster-config
```

Important limitation: fakecuda preflight now tracks torch-level tensor lifetimes, stage peaks, top allocations, optional allocation stack traces, coarse categories for parameters, buffers, gradients, optimizer state, activations, and temporaries, shared-storage aliases, and basic logical-device attribution. Saved autograd activations still need more validation, so a pass should be treated as a preflight signal rather than proof that a full cluster run will fit.

### 3. Calibrate On The 3090 Ti

Run a reduced version of the workload directly on the real GPU:

```bash
python train.py --small-config
```

Record real PyTorch memory:

```python
import torch

print(torch.cuda.max_memory_allocated())
print(torch.cuda.mem_get_info())
```

Then compare with FakeGPU passthrough or hybrid runs, when your CUDA installation can be loaded by FakeGPU:

```bash
./fgpu --mode passthrough python train.py --small-config
./fgpu --mode hybrid --oom-policy clamp python train.py --small-config
```

The goal is not exact equality. The goal is to understand the error between real 3090 Ti memory and FakeGPU-reported memory on small controlled workloads. That error should be included in future preflight reports as calibration context.

## Stage Markers

The future runner should support optional Python markers:

```python
import fakegpu

with fakegpu.stage("model_load"):
    model = load_model()

with fakegpu.stage("forward"):
    outputs = model(**batch)
    loss = outputs.loss

with fakegpu.stage("backward"):
    loss.backward()

with fakegpu.stage("optimizer_step"):
    optimizer.step()
```

Without markers, FakeGPU can still report process status and peak memory, but failures may be mapped to `unknown_or_last_seen`.

## Report Requirements

The preflight report should include:

- command and working directory
- FakeGPU version and git commit
- runtime mode
- target GPU profiles
- calibration GPU, when present
- status
- stage reached
- per-device memory totals, peaks, and headroom
- current memory by category
- largest allocations
- peak memory by stage
- warnings
- errors
- stdout/stderr log paths
- tracking confidence

Suggested confidence levels:

| Level | Meaning |
|---|---|
| `C0_incomplete` | The command ran, but memory tracking is insufficient for OOM judgment. |
| `C1_weight_storage` | Mostly tracks weights and explicit fake-CUDA storage. |
| `C2_torch_tensor_lifetime` | Tracks torch tensor lifetimes well enough for fakecuda preflight. |
| `C3_native_cuda_allocations` | Tracks native CUDA allocations in simulate mode. |
| `C4_real_gpu_calibrated` | Has real 3090 Ti calibration data for this workload class. |

## Recommended Next Work

The next implementation should prioritize:

1. Saved autograd activation coverage beyond visible op outputs.
2. 3090 Ti calibration reports for small controlled workloads.
3. Small/large profile pass-fail matrix for the same workload.
4. More workload examples that attach `preflight_report.json` to Slurm submission notes.
5. Documentation that clearly separates fit/no-fit checks from performance prediction.
