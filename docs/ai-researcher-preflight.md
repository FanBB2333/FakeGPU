# AI Researcher Preflight

This page describes the intended preflight workflow for AI researchers who want to check a training or inference command before submitting it to a large GPU cluster.

The immediate goal is narrow: determine whether the command can reach a selected stage and whether the workload is likely to hit OOM under a target GPU profile. FakeGPU does not currently try to predict GPU utilization, step time, throughput, or cluster scheduling behavior.

## Local Hardware Assumption

The current real calibration machine is a single NVIDIA RTX PRO 5000 72GB Blackwell with compute capability 12.0.

That machine is useful for:

- checking that the real CUDA / PyTorch / transformers environment works
- calibrating workloads that fit within the GPU's measured memory limit
- comparing real PyTorch `torch.cuda.max_memory_allocated()` with FakeGPU reports
- validating real OOM behavior up to the GPU's available memory boundary
- sanity-checking fakecuda memory tracking on controlled workloads

It cannot directly prove:

- that a workload calibrated on this GPU will fit a different target profile
- that a multi-GPU NCCL, NVLink, RDMA, or InfiniBand path behaves correctly
- that cluster throughput or GPU utilization will be good
- that CUDA kernels produce numerically identical results in simulate or fakecuda mode

Treat the RTX PRO 5000 as a calibration point, not as a substitute for the target cluster.

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

Important limitation: fakecuda preflight now tracks torch-level tensor lifetimes, saved autograd tensors visible through PyTorch hooks, stage peaks, top allocations, optional allocation stack traces, coarse categories for parameters, buffers, gradients, optimizer state, activations, and temporaries, shared-storage aliases, and basic logical-device attribution. CUDA backend-internal workspaces and optimizer temporaries may still be invisible in fakecuda, especially for Transformer-heavy workloads. Treat `PASS_FIT` as a preflight signal rather than proof that a full cluster run will fit.

### 3. Calibrate On The Real GPU

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

For the built-in calibration suite, run:

```bash
./ftest real_gpu_calibration
```

The built-in suite includes a tensor allocation probe, a torch MLP train step, a torch Tiny Transformer train step, gradient accumulation, gradient checkpointing, a locally initialized Hugging Face tiny GPT-2 train step, and a PEFT LoRA tiny GPT-2 train step. It does not download model weights.

The goal is not exact equality. The goal is to understand the error between memory measured on the current real GPU and FakeGPU-reported memory on small controlled workloads. The suite auto-selects `rtx-pro-5000-blackwell` for the current server and writes `build/real_gpu_calibration/calibration_real_gpu.json` plus a Markdown report. By default, every real/native worker performs one warmup followed by three measured trials. The report keeps the full distribution of PyTorch allocated, reserved, and requested peaks, samples memory through NVML, and uses the largest measured peak as the empirical upper bound. Process memory is included when NVML exposes the current PID; WSL hosts that do not expose this mapping keep device-memory deltas and mark process sampling unavailable. Each workload executes on real CUDA, passthrough, Hybrid clamp, and fakecuda. Native result signatures must match real CUDA. A final oversized tensor verifies that Hybrid clamp raises `torch.cuda.OutOfMemoryError` without consuming the physical GPU capacity.

Reports from different calibration GPUs can be combined without fitting a universal factor:

```bash
python3 verification/aggregate_real_gpu_calibrations.py \
  reports/3090ti/calibration_real_gpu.json \
  reports/pro5000/calibration_real_gpu.json \
  --output build/calibration_bundle.json \
  --markdown build/calibration_bundle.md
```

For a known workload signature, preflight can use the matching profile's observed physical-memory upper bound:

```bash
fakegpu preflight \
  --runtime fakecuda \
  --profile rtx3090ti \
  --memory-calibration build/calibration_bundle.json \
  --calibration-workload tiny_transformer_step \
  --report-dir preflight-empirical \
  --strict \
  -- python train.py --small-config
```

This raises tracking confidence to `C4_real_gpu_calibrated` only when every target device has a matching profile observation. Physical peaks prefer NVML process memory, which includes CUDA context and backend allocations. When WSL cannot expose the process, the estimate uses the larger of the PyTorch allocator peak and NVML device delta and records that fallback source. It does not extrapolate across model shapes: changing batch size, sequence length, model dimensions, or optimizer configuration requires a new workload signature and new samples.

### 4. Static ATen Storage-Liveness Validation

`./ftest static_memory_validation` captures a fake-tensor ATen forward/backward graph without executing CUDA kernels. CUDA-enabled hosts trace fake CUDA tensors so device-dependent operators select the measured backend's ATen path. The estimator accounts for unique storage aliases, graph last-use lifetimes, parameters, buffers, gradients, and Adam/AdamW moment state. Graph and optimizer phases are compared separately. The eager single-tensor optimizer model follows parameter iteration order because two current-parameter intermediates can overlap the previous parameter's denominator. CUDA Flash Attention auxiliary storage is derived from query shape, dtype, and 64-token sequence tiles. A CUDA run adds one measured post-release backend-resident allocation for the current GPU/software profile and checks six parameterized MLP/Transformer FP32/BF16 workloads. The maintained threshold rejects a measured underestimate above 5%.

The current cross-machine evidence covers RTX 3090 Ti Ampere (PyTorch 2.12/CUDA 13.0) and RTX PRO 5000 Blackwell (PyTorch 2.9/CUDA 12.8). Across 13 workloads and 26 GPU observations, maximum allocator underestimation and maximum absolute error were 0.08%. MLP requested-byte estimates were exact. FP32 Efficient Attention shapes differed from requested peaks by at most 28 bytes, while three Flash Attention shapes differed by at most 260 bytes after backend-resident calibration. Static peak bytes matched across both hosts. Transformer graph fingerprints still differed across PyTorch versions despite equal byte estimates. This validates the current parameter grid, not arbitrary models or software stacks.

```bash
python3 verification/aggregate_static_memory_validations.py \
  reports/3090ti/static_memory_validation.json \
  reports/pro5000/static_memory_validation.json \
  --output build/static_memory_validation_bundle.json \
  --markdown build/static_memory_validation_bundle.md
```

Other unmatched backend workspaces, fused/foreach optimizer extras, allocator fragmentation, custom CUDA kernels, distributed buffers, and graph breaks still require additional modeling or empirical measurements.

To produce an individual preflight report for every maintained workload:

```bash
workloads=(
  tensor_256mb
  mlp_train_step
  tiny_transformer_step
  gradient_accumulation_step
  gradient_checkpointing_step
  hf_tiny_gpt2_step
  peft_lora_tiny_step
)
for workload in "${workloads[@]}"; do
  python3 -m fakegpu preflight \
    --runtime fakecuda \
    --profile rtx-pro-5000-blackwell \
    --device-count 1 \
    --stage "$workload" \
    --report-dir "build/preflight-$workload" \
    --strict \
    -- python3 verification/calibration_real_gpu.py \
      --worker fakecuda --workload "$workload" \
      --profile rtx-pro-5000-blackwell
done
```

Each generated report contains the workload stage, peak memory, and final status.

When no exact empirical workload match exists and the missing memory looks like a mostly fixed backend workspace gap, use an additive margin in preflight. For example, if calibration reports roughly 18 MiB missing at `after_backward`:

```bash
fakegpu preflight \
  --runtime fakecuda \
  --devices a100:8 \
  --stage optimizer_step \
  --memory-safety-margin 18MiB \
  --report-dir preflight-a100-margin \
  --strict \
  -- python train.py --cluster-config
```

Use `--memory-safety-factor` only when calibration shows the gap scales with workload size. Reports that use either safety option keep both the raw tracked peak and the estimated peak used for fit/OOM classification.

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
| `C4_real_gpu_calibrated` | Has calibration data from an identified real GPU for this workload class. |

## Recommended Next Work

The next implementation should prioritize:

1. Adding phase-local cuDNN/cuBLASLt and fused optimizer workspace profiles.
2. A manual large tensor OOM probe on the current real calibration GPU.
3. Small/large profile pass-fail matrix for more realistic HF and LoRA workloads.
4. More workload examples that attach `preflight_report.json` to Slurm submission notes.
5. Documentation that clearly separates fit/no-fit checks from performance prediction.
