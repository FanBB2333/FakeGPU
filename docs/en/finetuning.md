# Fine-tuning time and VRAM

[Documentation index](README.md) · [简体中文](../zh-CN/finetuning.md) ·
[繁體中文](../zh-TW/finetuning.md)

FakeGPU can estimate single-GPU fine-tuning from a config file, run the real
training framework to record what happened, and check later predictions
against independent runs. The supported frameworks are Hugging Face
Trainer/PEFT, LLaMA-Factory, and Axolotl.

Contents:

1. [Estimate from a config](#estimate-from-a-config)
2. [Profile the dataset](#profile-the-dataset)
3. [Record a native run](#record-a-native-run)
4. [Validate timing windows and events](#validate-timing-windows-and-events)
5. [Budget a complete pipeline](#budget-a-complete-pipeline)
6. [Random sampling and changed datasets](#random-sampling-and-changed-datasets)
7. [Length grouping and packing](#length-grouping-and-packing)
8. [Kernel caches](#kernel-caches)
9. [Native run setup](#native-run-setup)
10. [Probe with Hugging Face Trainer](#probe-with-hugging-face-trainer)
11. [Validation records](#validation-records)

## Estimate from a config

`estimate-finetune` estimates single-GPU full, LoRA, and 4-bit QLoRA memory and
**training-loop** time from local model metadata and a framework config. It
does not import the frameworks or load the weights.

- Samples are the processed training sequences after filtering, splitting,
  and packing. Sequence length includes template tokens and padding.
- `max_steps` overrides epochs. `--max-steps -1` clears a configured limit.
- Unpacked Axolotl configs convert epochs as Axolotl does:
  `ceil(samples * epochs / (micro_batch_size * gradient_accumulation_steps))`
  updates, which Axolotl passes to Trainer as `max_steps`.
- Axolotl passes `max_grad_norm` to Trainer only when it is set, so an unset
  value means Trainer's default of 1.0. An unset Axolotl `lora_r` has no
  default (PEFT rejects it); the estimate assumes 8 and says so in
  `assumptions`.
- Without `parameter_dtype` or `torch_dtype`, weight storage follows the
  framework. LLaMA-Factory keeps LoRA/QLoRA frozen weights in the dtype of the
  local checkpoint's `config.json` (`torch_dtype="auto"`) and full tuning in
  FP32. Axolotl loads weights in its precision flag's dtype (`bf16`, then
  `fp16` or `load_in_8bit`). HF Trainer configs and checkpoints without a
  declared dtype use FP32. `assumptions` names the rule that was applied.

```bash
python -m fakegpu estimate-finetune \
  --config examples/finetuning/huggingface-lora.yaml \
  --model /path/to/Qwen3.5-0.8B --text-only \
  --profile "$TARGET_GPU_PROFILE" --num-samples 10000 --epochs 3 \
  --sequence-length 128 --json estimate.json
```

- Config examples for all three frameworks are in
  [`examples/finetuning/`](../../examples/finetuning/). The config adaptation
  is CPU-tested but not benchmarked for every framework and runtime
  combination.
- Distributed or offloaded execution, unsupported optimizers, and unresolved
  adapter targets fail with an explicit error. Use `plan-training` for
  distributed plans.
- Qwen3.5 linear and full attention layers are counted separately. DeltaNet
  backward workspaces, fast kernels, and MoE dispatch are only partly modeled.
  `--text-only` selects the text backbone of a multimodal checkpoint.

## Profile the dataset

`profile-finetune-data` scans every row of local JSONL files and records a
content fingerprint, token lengths, truncation, padding, and sequence counts.
Raw `text` or `messages` rows use a local tokenizer. `--packing concatenate`
is a token-budget scenario and does not reproduce any framework's packing
algorithm.

For counts from a real training split, export its final `input_ids` rows
(optionally with `attention_mask` and `labels`) and pass `--processed`:

```bash
python -m fakegpu profile-finetune-data \
  --data /path/to/processed-train.jsonl --format input_ids --processed \
  --sequence-length 128 --json dataset.json
python -m fakegpu estimate-finetune \
  --config examples/finetuning/huggingface-lora.yaml \
  --model /path/to/Qwen3.5-0.8B --text-only \
  --profile "$TARGET_GPU_PROFILE" --dataset-report dataset.json \
  --epochs 3 --max-steps -1 --json estimate.json
```

`--dataset-report` fixes the sample count and the sequence cap. It replaces
the config's sample count, rejects conflicting overrides, and requires a
matching cap. `--source-framework` is a provenance label only. Packed training
needs the final processed sequence counts.

## Record a native run

`run-finetune` runs the installed HF Trainer/PEFT, LLaMA-Factory, or Axolotl
pipeline on one physical CUDA GPU. It supports full tuning, LoRA, and
bitsandbytes 4-bit QLoRA, with a local unquantized text checkpoint and unpacked
data unless noted below. Training uses the complete native config, and
relative paths resolve from the config directory. The run report records:

- native preprocessing, the effective dataset, actual batches, optimizer
  updates, allocator peaks, evaluation and checkpoint events, and final saving
- `observation.updates[].batches`, with each microbatch's shape, epoch
  position, token slots, nonpadding tokens, and supervised labels
- `observation.batch_profile`, with timings grouped by complete ordered
  accumulation window. A group is `sufficient` after two steady updates. This
  describes sample availability, not accuracy.
- `observation.phase_events`, the actual evaluation and save call tree. Only
  root intervals count toward phase totals, and post-save file inventories are
  not disk-write measurements.

Warmup and collection overhead are excluded from steady samples but included
in the total wall time.

## Validate timing windows and events

`validate-finetune-windows` compares timing predictions with a separately
recorded holdout run.

- It needs at least two independent calibration reports with the same model,
  data, software, and native settings.
- Every window needs two retained observations per calibration run, after
  excluding global warmup and the window's first occurrence. Uncovered windows
  are rejected.
- It only validates historical evidence. It does not predict shuffle order,
  warmup, total wall time, or dynamic VRAM. Native packing and MoE windows are
  not supported.

```bash
python -m fakegpu validate-finetune-windows \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --holdout-run holdout.json --json window-validation.json
```

`validate-finetune-events` validates evaluation and save subtotals the same
way. Initial, first, repeated, and final events stay separate. It currently
supports unpacked dense text models, loss-only evaluation, and local
model/adapter or resumable checkpoint saves. It does not predict event counts
or total pipeline time.

```bash
python -m fakegpu validate-finetune-events \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --holdout-run holdout.json --json event-validation.json
```

`--event-context` also matches the latest evaluation and checkpoint boundaries
before each window, including ordered calls, workloads, phases, and updates
since each boundary. Each combination of window and context needs two samples
per calibration run. This validates recurring windows after periodic events.
It does not cover recovery after the first save or the complete training time.

```bash
python -m fakegpu validate-finetune-windows --event-context \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --holdout-run holdout.json --json event-window-validation.json
```

`--epoch-context` also matches each microbatch's position within its epoch.
This separates windows that include epoch-start loading or worker startup. Use
it alone or together with `--event-context`:

```bash
python -m fakegpu validate-finetune-windows --event-context --epoch-context \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --holdout-run holdout.json --json epoch-window-validation.json
```

## Budget a complete pipeline

`plan-finetune-pipeline` projects future epochs or updates from a calibration
run's complete observed epoch. `estimate-finetune-pipeline` combines at least
two calibrations into a complete native budget. The budget covers setup,
warmup, ordered training windows, evaluation, checkpoints, Trainer gaps, and
finalization.

- The span ends after the native call and the final CUDA sync. Python startup,
  report construction, and CLI output are excluded.
- Time ranges are the extremes of per-process means, not confidence
  intervals. Missing components fail with an explicit error.
- Supported schedules are unpacked, sequential text datasets that keep the
  tail batch: HF Trainer (Transformers 5.6.0) and LLaMA-Factory 0.9.5 with
  `disable_shuffling: true`, and Axolotl 0.18.0 (Transformers 5.14.1) with
  `curriculum_sampling: true`.
- `--epochs` means the framework's own `num_epochs`. HF and LLaMA-Factory
  finish every epoch's short final accumulation window. Axolotl runs
  `ceil(rows * epochs / (micro_batch_size * gradient_accumulation_steps))`
  updates, the `max_steps` it passes to Trainer. The same rule applies to
  `plan-finetune-transfer` and `run-finetune --planning-epochs`. Packed
  Axolotl horizons need a step count (`--max-steps`, or
  `--planning-max-steps` for `run-finetune`), because Axolotl derives packed
  epochs from its multipack length estimate.
- Dataset, model, hardware, and execution settings must match the
  calibration, and a different row count needs a new calibration. Logging must
  be off or happen on every update. Checkpoint rotation and best-model reload
  are not supported.

```bash
python -m fakegpu plan-finetune-pipeline \
  --run-report calibration-1.json --epochs 30 --json future-plan.json
python -m fakegpu estimate-finetune-pipeline \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --plan future-plan.json --memory-budget-gib 24 --reserve-gib 1 --json budget.json
python -m fakegpu validate-finetune-pipeline \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --holdout-run holdout.json --json validation.json
```

Memory results are the observed allocated and reserved peaks over the full run,
on the same GPU and allocator. They do not carry over to smaller cards. The
optional reserve is a margin you choose, not proof of OOM safety.
`validate-finetune-pipeline` reports timing errors and memory underestimation
against an independent holdout.

## Random sampling and changed datasets

For default random sampling or a changed dataset, use the empirical transfer
workflow. It plans the CPU sampler and collator outputs from the native
prepared dataloader, and checks the consumed row indices and input tensor
hashes during calibration. Supported stacks are HF and LLaMA-Factory with
Accelerate 1.11.0, and Axolotl with Accelerate 1.13.0. Random sampling uses the
native data seed plus the epoch. (Axolotl 0.18.0 does not forward its YAML
`data_seed` to Trainer.)

```bash
python -m fakegpu calibrate-finetune \
  --config target-training.json --framework huggingface \
  --profile "$TARGET_GPU_PROFILE" --python /path/to/framework/python \
  --cuda-device 0 --workspace ./calibration-target \
  --calibration-steps 24 --warmup-steps 2 --epochs 3 \
  --timeout-seconds 600 --json calibration.json
python -m fakegpu budget-finetune-workspace \
  --workspace ./calibration-target --epochs 2 --json revised-budget.json
```

- `calibrate-finetune` runs exactly two fresh native processes with step and
  time limits, and with new empty kernel-cache directories by default. It keeps
  every attempt, seals reports with SHA256, and supports `--resume`. It never
  silently extends or reruns a calibration.
- Rebudgeting on the CPU needs no GPU or framework, but cannot go past the
  saved sampling horizon.
- Keep the workspace at its original absolute path and outside the recorded
  input directories. Nested workspaces are rejected. Only finite local datasets
  and POSIX lock semantics are supported. Prepared Axolotl and LLaMA-Factory
  datasets are isolated under the workspace's `prepared/` directory, and
  resume rejects changed raw inputs.
- `--reuse-calibration-run PATH` (repeatable) reuses compatible training
  evidence from another dataset. Two fresh calibrations on the target dataset
  still provide the startup and finalization costs, which are never scaled by
  sample count. Training costs across different token, mask, and supervision
  distributions are **approximate**. Missing coverage exits 1 with a list of
  gaps.
- `run-finetune --planning-epochs N` (or `--planning-max-steps N`) records the
  CPU horizon directly. `plan-finetune-transfer`, `estimate-finetune-transfer`,
  and `validate-finetune-transfer` expose the same steps without the workspace
  manager. Streaming, custom samplers, and automatic batch-size changes are not
  supported.

## Length grouping and packing

### Length grouping

Set `train_sampling_strategy: group_by_length` (HF or LLaMA-Factory) or
`group_by_length: true` (Axolotl), and pass `--length-grouping-seed 123` to
`calibrate-finetune` or `run-finetune`.

- It uses the native grouping algorithm with an independent CPU generator
  seeded by `123 + epoch`. Native grouping with the global RNG is not
  reproducible.
- A planning horizon is required. One finite evaluation dataset uses the same
  explicit seed without advancing the training sampler. Custom evaluation
  samplers and prepared loaders are rejected.
- Sampler plan v2 includes the grouping lengths and batch size in the
  calibration identity, so a different set of lengths needs its own evidence.

### Packing

- The [packing semantics audit](../../tests/native/finetune_packing_semantics.py)
  shows that the tested ordinary HF and LLaMA-Factory hybrid paths leak across
  block masks and reset positions in tiny Qwen3.5 models. It also shows that
  the tested Axolotl patch does not fit the Transformers 5.14.1 decoder
  interface.
- The opt-in [segment reference audit](../../tests/native/finetune_segment_reference.py)
  checks a project-owned Qwen3.5 interface that runs each linear-attention
  segment with empty convolution and recurrent state. It matches independent
  native sequences, including checkpoint recomputation and reload.
- Enable it with `--packing-seed 123` on `run-finetune` or
  `calibrate-finetune`. This needs dense `qwen3_5_text` LoRA, a planning
  horizon, native packing (`packing: true` plus `neat_packing: true` for
  LLaMA-Factory, or `sample_packing: true` for Axolotl), and evaluation turned
  off.
- HF processed rows also need `position_ids` and contiguous positive segment
  IDs in `attention_mask` (optional zero right padding). The first label of
  every later segment must be ignored.
- Sampler plan v3 records epoch bins, logical-sequence counts, tensor rows,
  block geometry, dtypes, and input hashes, so different data needs its own
  calibration. Native packing reports require the transfer commands.
- There is no claim about varlen-kernel throughput. Full tuning, QLoRA, and
  grouped packing are not covered. See also the
  [PEFT direct-forward audit](../../tests/native/finetune_segment_peft.py) and
  the [packed-budget evidence](../../tests/data/finetune_packing_budget_evidence.json).

## Kernel caches

`run-finetune --cache-regime empty_kernel_cache` requires five distinct,
existing, empty directories set explicitly: `TRITON_CACHE_DIR`,
`TORCHINDUCTOR_CACHE_DIR`, `CUDA_CACHE_PATH`, `TORCH_EXTENSIONS_DIR`, and
`PYTORCH_KERNEL_CACHE_PATH`. `populated_kernel_cache` records each file's
relative path, size, and SHA256. The runner never creates, clears, or warms
these caches.

Reports with the default `uncontrolled` regime make no claim about caches and
cannot be mixed with a controlled calibration. Triton metadata can refer to
the original cache path, so leave those files unchanged.

## Native run setup

- Unpacked HF runs (`--framework huggingface`) use a flat config and processed
  JSONL rows with exactly `input_ids`, `attention_mask`, and `labels`. These
  must be integer arrays of equal length, with a binary mask and between 2 and
  `sequence_length` tokens. No tokenization is applied, and the tokenizer needs
  a pad token. See [the native LoRA example](../../examples/finetuning/huggingface-native-lora.yaml).
- Gradient checkpointing in LLaMA-Factory uses
  `disable_gradient_checkpointing: false` with `use_reentrant_gc: true` or
  `false`. Axolotl uses `gradient_checkpointing: true` with
  `gradient_checkpointing_kwargs: {use_reentrant: true}` (or `false`). The
  effective checkpointing and quantization settings are part of the
  calibration identity.
- Native runs need a text checkpoint, and `--text-only` does not convert
  weights. For a local Qwen3.5, export a text checkpoint on the CPU and create
  synthetic Alpaca data for integration checks:

```bash
python tests/native/prepare_finetune_fixture.py \
  --model /path/to/Qwen3.5-0.8B \
  --output-dir /tmp/fakegpu-finetune/qwen35-text \
  --dataset-dir /tmp/fakegpu-finetune/data
```

The export needs a Transformers version with `Qwen3_5ForCausalLM`. The native
examples target LLaMA-Factory 0.9.5 (Transformers 5.6.0) and Axolotl 0.18.0
(Transformers 5.14.1). Install each one in its own environment, with
compatible CUDA PyTorch, PEFT, and dataset dependencies. Use a new `output_dir`
for every run. Axolotl's prepared cache lives in
`/tmp/fakegpu-finetune/axolotl-prepared`.

```bash
CUDA_VISIBLE_DEVICES=1 python -m fakegpu run-finetune \
  --config examples/finetuning/axolotl-native-lora.yaml --framework axolotl \
  --profile "$TARGET_GPU_PROFILE" --warmup-steps 2 --output run.json
python -m fakegpu estimate-finetune \
  --run-report run.json --epochs 3 --max-steps -1 --json estimate.json
```

- For LLaMA-Factory, use `--framework llamafactory` with
  [`llamafactory-native-lora.yaml`](../../examples/finetuning/llamafactory-native-lora.yaml).
- Reuse `--run-report` in the same Python environment. The CLI rechecks the
  local weights, model config, and software identity, and defaults to the
  report's GPU profile.
- This single-report `estimate-finetune` interface needs fixed full batches.
  A different `--num-samples` gives a training-loop scenario without
  preprocessing or loading costs. Python startup and downloads are excluded.

For 512-token NF4 QLoRA with checkpointing, use
[`llamafactory-native-qlora.yaml`](../../examples/finetuning/llamafactory-native-qlora.yaml)
or [`axolotl-native-qlora.yaml`](../../examples/finetuning/axolotl-native-qlora.yaml).
Both use double quantization and BF16, and need bitsandbytes. Create the longer
dataset from the exported tokenizer:

```bash
python tests/native/prepare_finetune_fixture.py \
  --model /tmp/fakegpu-finetune/qwen35-text --dataset-only \
  --dataset-dir /tmp/fakegpu-finetune/data-long \
  --sequence-length 1024 --num-samples 64
```

The fixture length is a lower bound. Each framework applies its configured
512- or 1024-token cutoff, and the run report records the sequences actually
used.

Collect independent calibration and holdout processes with
[`tests/native/finetune_framework_matrix.py`](../../tests/native/finetune_framework_matrix.py).
It reads the sequence length, microbatch size, accumulation, and seed from the
native config:

```bash
python tests/native/finetune_framework_matrix.py \
  --framework axolotl --config examples/finetuning/axolotl-native-qlora.yaml \
  --python /path/to/axolotl/venv/bin/python \
  --output-dir /tmp/fakegpu-finetune/axolotl-qlora-matrix \
  --gpu-index 1 --calibration-steps 6 --warmup-steps 2 \
  --holdout-runs 2 --holdout-steps 16
```

Step counts include warmup. The defaults are 6 calibration updates, 2 warmup
updates, and three 32-update holdouts. Every run uses a fresh output directory,
and predictions use the production estimator with the actual microstep counts.

## Probe with Hugging Face Trainer

For a measured estimate, run a short real-GPU probe and reuse its observation
in the **same Python environment**. The model, GPU, precision, adapter,
attention implementation, batch, sequence length, and accumulation must also
match.

```bash
CUDA_VISIBLE_DEVICES=1 python -m fakegpu probe-finetune \
  --config examples/finetuning/huggingface-lora.yaml \
  --model /path/to/Qwen3.5-0.8B --text-only \
  --profile "$TARGET_GPU_PROFILE" --warmup-steps 2 --measure-steps 5 \
  --output observation.json
python -m fakegpu estimate-finetune \
  --config examples/finetuning/huggingface-lora.yaml \
  --model /path/to/Qwen3.5-0.8B --text-only \
  --profile "$TARGET_GPU_PROFILE" --num-samples 10000 --epochs 3 \
  --observation observation.json --json calibrated.json
```

The probe runs Hugging Face Trainer on fixed-length synthetic text. It needs
Transformers, Accelerate, and CUDA PyTorch, plus PEFT for adapters and
bitsandbytes for QLoRA. Warmup is recorded separately, and both the startup
and the steady peaks are kept. Dataset preparation, model loading, evaluation,
and checkpoint I/O are excluded from the projected time. Checkpoint files are
hashed, so replacing the weights invalidates the calibration.

## Validation records

Exact workloads, versions, and limits are in [`refactor.md`](../../refactor.md).

- [Complete pipeline validation](../../tests/data/finetune_pipeline_validation_evidence.json):
  30 native processes across all three frameworks, including allocator OOM
  controls.
- [Sampler and transfer validation](../../tests/data/finetune_transfer_validation_evidence.json):
  18 Qwen3.5-0.8B processes on a Blackwell GPU, changing 25 to 73 training rows
  plus random sampling with periodic evaluation and saving. There is one
  holdout per condition.
- [Qwen3.5-2B record](../../tests/data/finetune_qwen2b_evidence.json):
  512/1024-token LoRA and 512-token NF4 QLoRA with checkpointing, two
  calibrations and two holdouts per condition. Total-time errors were
  0.054%–1.578%.
- [Native validation archive](../../tests/data/finetune_framework_validation_evidence.json):
  earlier 128-token LoRA runs. Maximum wall-time errors were 3.931%
  (LLaMA-Factory) and 1.125% (cached Axolotl). Reusing Axolotl's first-use
  calibration on later cached runs gave a 364.885% error (the first update took
  66.542 s), so cache and JIT costs cannot be reused from a model and software
  signature alone.
- [Extended validation archive](../../tests/data/finetune_extended_validation_evidence.json):
  six configurations at 512/1024 tokens and batch 1/2, including checkpointing
  and NF4 QLoRA, with 18 measured processes.
- [Qwen3.5-0.8B probe record](../../tests/data/finetune_validation_evidence.json):
  full, LoRA, and QLoRA calibration and holdout runs, reproducible with
  [`tests/native/finetune_matrix.py`](../../tests/native/finetune_matrix.py).

These short runs do not establish general accuracy or OOM safety.
