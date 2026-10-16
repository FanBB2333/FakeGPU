# Memory analysis and estimation

[Documentation index](README.md) · [简体中文](../zh-CN/estimation.md) ·
[繁體中文](../zh-TW/estimation.md)

FakeGPU can check memory in two ways. `preflight` runs your code on FakeCUDA up
to a chosen stage. The static estimators read model metadata and never load the
weights. For online serving, see [Online LLM serving](serving.md).

## Check memory before a run

Run a command to a target stage and write reports under an ignored build
directory:

```bash
python3 -m fakegpu preflight \
  --runtime fakecuda \
  --profile a100 \
  --stage forward \
  --report-dir build/preflight \
  --strict \
  -- python3 train.py
```

Preflight tracks the memory used on the path that actually ran and reports
whether the selected profile fits the workload.

## Analyze a repository

```bash
# Find GPU entry points, dependencies, native sources, and compatibility risks.
python3 -m fakegpu analyze-repo .

# Inspect instructions, FLOPs, registers, shared memory, and occupancy
# in a PTX, SASS, or CUDA source file.
python3 -m fakegpu analyze-kernel kernel.ptx --profile a100 --threads-per-block 256
```

## Estimate LLM memory

```bash
# Estimate checkpoint, KV-cache, transient, adapter, and MoE memory.
python3 -m fakegpu estimate-llm \
  --model-dir /models/example \
  --batch-size 1 \
  --prompt-tokens 128 \
  --generated-tokens 32 \
  --dtype bfloat16 \
  --kv-cache-strategy paged \
  --kv-cache-block-tokens 16 \
  --target-profile a100 \
  --json build/llm-estimate.json

# Match a generation runtime that projects only the last token to logits.
python3 -m fakegpu estimate-llm \
  --model-dir /models/example --prompt-tokens 128 --generated-tokens 32 \
  --logits-to-keep 1 --json build/last-token-estimate.json
```

The default, `--logits-to-keep 0`, projects all tokens. Use the setting your
runtime actually uses. It changes LM-head FLOPs and memory, not the token
counts of the other layers.

How the LLM estimator works:

- It reads safetensors headers only.
- `--kv-cache-strategy` accepts `dynamic`, `static`, `quantized`, or `paged`.
  The report separates logical storage, quantization savings, static
  reservation, paged overhead, and sliding-window limits.
- Quantized caches keep the 128 most recent tokens at the compute dtype by
  default. Change this with `--kv-cache-residual-tokens`.
- MHA, GQA, and MQA use separate key and value storage. Configs with
  `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, and `v_head_dim` are
  treated as MLA (compressed latent plus decoupled RoPE). MLA backend
  workspaces need calibration.
- For models with a vision encoder, checkpoint weights are counted but
  vision and projector activations and FLOPs are not. The report flags this.
  Decoder token counts must already include the visual tokens.

## Estimate diffusion generation memory

```bash
# Compare diffusion generation phases and memory optimizations.
python3 -m fakegpu estimate-diffusion --list-profiles
python3 -m fakegpu estimate-diffusion \
  --model-profile stable-diffusion-xl-base-1.0 \
  --height 1024 --width 1024 --batch-size 4 \
  --attention-backend sdpa --vae-slicing \
  --offload model --target-profile a100 \
  --json build/diffusion-estimate.json

# Inspect a local Diffusers pipeline without loading tensor payloads.
python3 -m fakegpu estimate-diffusion \
  --model-dir /models/pixart-or-flux \
  --height 1024 --width 1024 --text-tokens 300 \
  --dtype bfloat16 --attention-backend sdpa \
  --offload model --target-profile a100 \
  --json build/local-diffusion-estimate.json
```

How the diffusion estimator works:

- It separates text encoding, repeated denoising, and VAE decode.
  `--model-dir` reads only `model_index.json`, component `config.json` files,
  and selected safetensors headers. It runs no remote code and loads no tensor
  payloads.
- Checkpoint storage and runtime weight bytes are reported separately. Local
  Euler/PNDM scheduler configs set denoiser evaluation counts. Other
  schedulers report `denoiser_evaluations: null`.
- UNets, cross-attention transformers (DiT/PixArt), and SD3/Flux-style
  joint-attention transformers use different activation formulas. CFG doubles
  the denoiser batch only where both branches exist.
- `--weight-variant fp16|bf16|fp32` selects a checkpoint family. Built-in
  profiles include `pixart-sigma-xl-2-1024-ms` and two Stable Diffusion UNets.
- `--offload model`, `--attention-slicing`, `--vae-slicing`, and
  `--vae-tiling` model common Diffusers trade-offs. Without a matching
  real-GPU comparison, reports stay `Modeled` and `uncalibrated`, with no error
  percentage or prediction interval.

To train diffusion models, see [Diffusion training](diffusion-training.md).

## Other estimators

- `estimate-roofline` produces a profile-aware analytical latency interval.
  It is not a measured kernel latency.
- `plan-training` estimates rank-local memory for distributed training. See
  [Distributed plans and experiment sweeps](distributed.md).
- `estimate-finetune` estimates single-GPU fine-tuning time and VRAM. See
  [Fine-tuning time and VRAM](finetuning.md).

Large reports and plans can use `.json.gz` wherever a JSON path is accepted.
Compressed output is written atomically and reproducibly.
