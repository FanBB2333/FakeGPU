# Diffusion training

[Documentation index](README.md) · [简体中文](../zh-CN/diffusion-training.md) ·
[繁體中文](../zh-TW/diffusion-training.md)

FakeGPU can plan, run, and budget cached-feature UNet training. Full tuning
and LoRA are supported, with optional `--ema-decay`. For generation memory, see
[Memory analysis and estimation](estimation.md#estimate-diffusion-generation-memory).

## How it works

- `plan-diffusion-training` hashes the local weights and the FP32 cached
  tensors.
- `run-diffusion-training` trains on one CUDA GPU with BF16 autocast, FP32
  trainable parameters, and AdamW.
- `estimate-diffusion-training` builds a budget from compatible native runs.
- `validate-diffusion-training` compares the budget with an independent run.

VAE and text-encoder training and raw-image preparation are not included.

## Example

The fixture below generates random features for integration tests:

```bash
python tests/native/prepare_diffusion_fixture.py \
  --model-dir /path/to/stable-diffusion-v1-4 --output-dir /tmp/diff-fixture \
  --num-samples 5 --epochs 3 --method lora --gradient-checkpointing
CUDA_VISIBLE_DEVICES=1 python -m fakegpu run-diffusion-training \
  --plan /tmp/diff-fixture/plan.json --output-dir /tmp/diff-cal-1 \
  --json /tmp/diff-cal-1.json
CUDA_VISIBLE_DEVICES=1 python -m fakegpu run-diffusion-training \
  --plan /tmp/diff-fixture/plan.json --output-dir /tmp/diff-cal-2 \
  --json /tmp/diff-cal-2.json
python -m fakegpu estimate-diffusion-training --plan /tmp/diff-fixture/plan.json \
  --epochs 5 --calibration-run /tmp/diff-cal-1.json \
  --calibration-run /tmp/diff-cal-2.json --json /tmp/diff-budget.json.gz
```

## Requirements and limits

- Activation and reserved peaks are unknown until there are native
  observations.
- A budget needs two independent processes and repeated samples for every
  future window. The first use of a shape is charged separately.
- `--cache-regime` behaves as in the
  [fine-tuning workflow](finetuning.md#kernel-caches), with `uncontrolled` as
  the default. Allocator, GPU UUID, software, inputs, and execution identity
  must all match.
- `validate-diffusion-training` compares an independent `--holdout-run`.
  `--max-total-time-error-percent` and `--max-memory-underestimate-percent`
  make it exit 1 when exceeded.

## Evidence

The [native diffusion record](../../tests/data/diffusion_native_evidence.json)
has SD/SDXL generation observations and twelve training processes. Six
training holdouts show 0.044%–3.185% total-time error, with no underestimate of
the allocator envelope. For generation, VAE allocated peaks were
underestimated by 12.0%–13.8% and reserved peaks by up to 31.6%.
