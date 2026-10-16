# Calibrating against a real GPU

[Documentation index](README.md) · [简体中文](../zh-CN/calibration.md) ·
[繁體中文](../zh-TW/calibration.md)

FakeGPU's estimates are only trustworthy for a workload after they have been
compared with real measurements on the same stack. This page shows how to
collect those measurements and how to gate an estimate on them. Published
results are summarized in [Validation and reliability](validation.md).

## Static memory validation

On a CUDA host, regenerate the controlled ATen comparison:

```bash
python3 scripts/validation/static_memory_validation.py \
  --output build/static-memory-validation.json \
  --markdown build/static-memory-validation.md \
  --max-underestimate-percent 5
```

On a CPU-only host, `--static-only` checks the estimation path without
producing an accuracy measurement.

## Serving observations

### Record phase peaks by hand

To compare a `plan-serving` report with repeated real-GPU phase peaks:

```bash
python3 -m fakegpu calibrate observe-serving \
  build/speculative-serving-plan.json \
  --prefill-peak-bytes 24800000000,24900000000,25000000000 \
  --prefill-peak-bytes 24950000000,25050000000 \
  --decode-peak-bytes 25200000000,25300000000,25250000000 \
  --decode-peak-bytes 25350000000,25400000000 \
  --source torch.cuda.max_memory_reserved \
  --strict \
  --json build/serving-observation.json
```

Every sample is kept and each phase uses its maximum. Five samples per phase
are required by default, and `--strict` exits 1 when there are fewer.
`ready_for_comparison` only means there are enough samples. It does not mean
`GPU-validated`. The plan must set `--target-profile`.

### Collect repeated runs automatically

With a framework-specific runner:

```bash
python3 -m fakegpu calibrate collect-serving \
  build/speculative-serving-plan.json \
  --repetitions 5 \
  --timeout-seconds 900 \
  --strict \
  --json build/serving-observation.json \
  -- python3 benchmark_serving.py
```

Homogeneous, non-speculative Transformers workloads can use the built-in
runner. The plan must use a dynamic KV cache:

```bash
python3 -m fakegpu plan-serving \
  --model-dir /models/Qwen/Qwen3-0.6B \
  --active-sequences 4 \
  --max-batch-size 8 \
  --prompt-tokens 2048 \
  --generated-tokens 128 \
  --dtype bfloat16 \
  --attention-implementation sdpa \
  --kv-cache-strategy dynamic \
  --target-profile a100 \
  --json build/transformers-serving-plan.json

python3 -m fakegpu calibrate collect-serving \
  build/transformers-serving-plan.json \
  --repetitions 5 \
  --timeout-seconds 900 \
  --strict \
  --json build/transformers-serving-observation.json \
  -- python3 -m fakegpu calibrate sample-transformers
```

What `sample-transformers` does:

- It rechecks the local config and Safetensors headers against the plan, then
  measures a full prefill and token-by-token decode.
- It reports `torch.cuda.max_memory_reserved` by default. Use
  `--metric allocated` for tensor-only peaks.
- It rejects mixed requests, speculative decoding, paged or quantized caches,
  shared-prefix or chunked-prefill plans, quantized checkpoints, and adapters.
- PyTorch and Transformers must already be installed in the CUDA environment.

### Build a sample from your own measurements

vLLM or custom instrumentation with trustworthy phase peaks can build the
sample directly:

```bash
python3 -m fakegpu calibrate emit-serving-sample \
  build/vllm-serving-plan.json \
  --prefill-peak-bytes 24900000000 \
  --decode-peak-bytes 25300000000 \
  --metric nvml.process_family_peak_bytes \
  --framework vllm \
  --framework-version 0.10.2
```

The same adapter is available in Python as
`fakegpu.build_cuda_serving_sample(...)`. vLLM's cache-usage metric is a block
fraction, not a byte peak, so the benchmark must supply real phase bytes.

### Sample protocol

Each repetition runs in a new process and must print one
`fakegpu.serving_peak_sample.v1` object, either as its last line or on a line
prefixed with `FAKEGPU_SERVING_SAMPLE=`:

```json
{
  "schema_version": "fakegpu.serving_peak_sample.v1",
  "workload_signature": "<FAKEGPU_SERVING_WORKLOAD_SIGNATURE>",
  "run_index": 1,
  "metric": "torch.cuda.max_memory_reserved",
  "phases": {
    "prefill": {"peak_bytes": 24900000000},
    "decode": {"peak_bytes": 25300000000}
  },
  "environment": {
    "backend": "cuda",
    "simulated": false,
    "gpu_name": "NVIDIA A100-SXM4-80GB",
    "gpu_uuid": "GPU-...",
    "compute_capability": [8, 0],
    "total_memory_bytes": 85899345920,
    "software": {
      "framework": "transformers",
      "framework_version": "...",
      "cuda_version": "...",
      "torch_version": "..."
    }
  }
}
```

- FakeGPU passes the plan path, workload signature, target profile, compute
  capability, run index, and run count through `FAKEGPU_SERVING_*` variables.
  The built-in runner and the protocol builder fill in this contract for you,
  and neither framework becomes a FakeGPU dependency.
- Collection fails on simulated CUDA, a capacity outside a 2% profile
  tolerance, a different compute capability, inconsistent software metadata,
  a timeout, or a nonzero exit.
- Raw command arguments are not stored. Only the executable name and a command
  fingerprint are kept.
- `simulated: false` is supplied by the runner and is not hardware
  attestation. A `GPU-validated` claim still needs retained logs and passing
  calibration gates.

## Compare and gate

To compare any compatible prediction and observation, then gate the result:

```bash
python3 -m fakegpu calibrate compare \
  build/prediction.json \
  build/observation.json \
  --json build/calibration-comparison.json

python3 -m fakegpu calibrate verify \
  build/calibration-comparison.json \
  --max-underestimate-percent 5 \
  --max-absolute-percentage-error-percent 5 \
  --min-interval-coverage-percent 90 \
  --capacity-bytes 25769803776 \
  --json build/calibration-verification.json
```

`compare` reports per-phase signed and absolute error, interval coverage, and a
recommended safety margin. `verify` exits 1 when any gate fails. The gates are
underestimation, median/p95/max error, interval coverage, false-safe fits at
the given capacity, and workload consistency. Apply results only to the same
workload signature, shapes, dtype, software stack, and GPU profile.

## Phase memory calibration

Allocated tensor storage and allocator-reserved memory are tracked separately.
`calibrate phase-sample` collects both in one fresh CUDA process.
`phase-estimate` seals per-phase maxima from at least five processes, and
`phase-validate` checks a later independent run. Use a new output path for
every sample and budget.

```bash
# plan.json: a compatible local plan-serving or estimate-diffusion report
for i in 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=1 python -m fakegpu calibrate phase-sample plan.json \
    --json "/tmp/phase-cal-$i.json"
done
python -m fakegpu calibrate phase-estimate \
  --calibration-sample /tmp/phase-cal-1.json \
  --calibration-sample /tmp/phase-cal-2.json \
  --calibration-sample /tmp/phase-cal-3.json \
  --calibration-sample /tmp/phase-cal-4.json \
  --calibration-sample /tmp/phase-cal-5.json --json /tmp/phase-budget.json.gz
CUDA_VISIBLE_DEVICES=1 python -m fakegpu calibrate phase-sample plan.json \
  --json /tmp/phase-holdout.json
python -m fakegpu calibrate phase-validate --budget /tmp/phase-budget.json.gz \
  --holdout-sample /tmp/phase-holdout.json --max-absolute-error-percent 20 \
  --max-underestimate-percent 10 --json /tmp/phase-validation.json
```

- Sampling supports homogeneous Transformers dynamic-cache serving, and local
  SD/SDXL with eager or SDPA attention, attention slicing, VAE slicing, VAE
  tiling, and their combinations. Offload is not supported.
- For tiling, `--vae-tile-size` sets both the native sample and latent tile
  limits. Effective processors, VAE flags, and tile limits are part of the
  native identity.
- The plan, selected weights and config, GPU UUID, Torch/CUDA/cuDNN versions,
  allocator, and implementation must match. No shape transfer or hidden
  safety multiplier is applied, and kernel caches are uncontrolled.
- Default limits are 20% absolute error and 10% underestimation. The command
  exits 1 on failure. Offline estimation and validation need only the standard
  library.

Evidence so far is experimental and limited to the tested GPU, software, and
fixed inputs:

- [Phase-memory validation](../../tests/data/phase_memory_evidence.json):
  after adding the VAE upsampling convolution, four resolution cohorts show
  0.045%–1.172% VAE allocated error. Six conditions with five calibrations and
  five holdouts each (60 processes) pass both gates.
- [Broader memory validation](../../tests/data/memory_generalization_evidence.json):
  25 single-GPU conditions in 74 processes across SD 1.4/SDXL,
  Qwen3-0.6B/1.7B/Qwen2.5-0.5B, FP16/BF16/FP32, batches, rectangular images,
  guidance, eager attention, slicing, and tiling. Worst allocated errors are
  10.93% absolute and 3.58% underestimate (gates: 20%/10%). Exact-condition
  budgets match all twelve holdouts, and 68 cross-condition reuses are
  rejected. These are not `GPU-validated` rows, because 21 conditions have
  fewer than five observations and optimized modes used an archived native
  probe instead of the public CLI.
- [Public optimized-mode validation](../../tests/data/optimized_phase_memory_evidence.json):
  seven SD/SDXL conditions in 70 processes. All 210 holdout comparisons match
  their calibration maxima, with analytical allocated error at most 6.69%
  absolute and 3.38% underestimate. 42 cross-mode and 21 effective-mode
  identity mismatches are rejected.

None of these show transfer to new shapes, prediction intervals, or physical
OOM safety.
