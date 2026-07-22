# DeepSpeed validation

FakeGPU can run a maintained DeepSpeed training path in Hybrid mode: CUDA
compute executes on one physical GPU while FakeGPU simulates the NCCL traffic
of two or four logical ranks. The validator checks optimizer numerics and
communication, not just process startup.

## Quick checks

Use a Python environment that already contains PyTorch and DeepSpeed.

```bash
PYTHON=/path/to/python

$PYTHON verification/run_hybrid_deepspeed_numerics.py \
  --zero-stage all --precision bf16 --world-size 2 \
  --report-dir build/deepspeed-numerics
```

This compact matrix verifies:

- DeepSpeed Engine initialization
- ZeRO stages 0, 1, 2, and 3
- FP32 or BF16 parameters
- two-step gradient accumulation with exactly one optimizer update
- identical updated parameters on every rank
- broadcast, all-reduce, all-gather, and reduce-scatter reporting
- complete node-pair byte and per-operation peak statistics

The same command accepts `--world-size 4`. Use `--zero-stage 3` when only the
most communication-intensive stage is needed.

## Qwen LoRA SFT

The real-model path loads local Qwen3.5 weights, applies PEFT LoRA, performs
forward, backward, and AdamW update phases through DeepSpeed, then checks a
full LoRA parameter on every rank.

```bash
MODEL=/home/l1ght/models/Qwen/Qwen3.5-0.8B

$PYTHON verification/run_qwen_deepspeed_lora_sft.py \
  --model-dir "$MODEL" \
  --output-dir build/qwen-deepspeed-zero3 \
  --zero-stage 3 --world-size 2 \
  --dtype bfloat16 --sequence-length 16
```

Gradient accumulation and activation checkpointing are command-line axes; no
project edit is required:

```bash
$PYTHON verification/run_qwen_deepspeed_lora_sft.py \
  --model-dir "$MODEL" \
  --output-dir build/qwen-deepspeed-zero3-s64-gas2-gc \
  --zero-stage 3 --world-size 2 \
  --dtype bfloat16 --sequence-length 64 \
  --gradient-accumulation-steps 2 \
  --gradient-checkpointing
```

The DeepSpeed-specific default is reentrant checkpointing. DeepSpeed 0.15.3
ZeRO-3 leaves partition placeholders visible during non-reentrant PyTorch
recomputation for this Qwen/PEFT stack. The validator retains
`--checkpoint-implementation non-reentrant` as an explicit compatibility
probe, but it is not the maintained passing configuration.

## Maintained results

The compact numerical matrix passed on both current test systems:

| GPU | Software | Matrix |
|---|---|---|
| RTX PRO 5000 72GB Blackwell, CC 12.0 | PyTorch 2.8.0 + CUDA 12.8, DeepSpeed 0.15.3 | 2-rank FP32/BF16 ZeRO 0–3; 4-rank BF16 ZeRO-3 |
| GeForce RTX 3090 Ti Ampere, CC 8.6 | PyTorch 2.12.1 + CUDA 13.0, DeepSpeed 0.19.2 | 2-rank FP32 ZeRO-0/3 and BF16 ZeRO 0–3 |

Qwen3.5-0.8B uses batch size 1, BF16 base weights, LoRA rank 8, and two
logical ranks. Memory values below are the maximum PyTorch allocated peak of
one rank; both ranks share one physical GPU in this experiment.

| GPU / case | Peak per rank | Logical node-pair bytes | Result |
|---|---:|---:|---|
| PRO 5000, ZeRO-2, sequence 16 | 1.754 GiB | 2.884 GiB | finite loss and identical LoRA update |
| PRO 5000, ZeRO-3, sequence 16 | 2.899 GiB | 12.273 GiB | 2,259 all-gathers and 2 reduce-scatters passed |
| RTX 3090 Ti, ZeRO-3, sequence 16 | 2.904 GiB | 12.273 GiB | same training invariants under DeepSpeed 0.19.2 |
| PRO 5000, ZeRO-3, sequence 64, accumulation 2 | 3.052 GiB | 15.196 GiB | one optimizer update after two microsteps |
| PRO 5000, previous case + checkpointing | 2.409 GiB | 17.134 GiB | reentrant recomputation passed |

Checkpointing reduced the maintained sequence-64 allocated peak by 21.1% and
increased simulated communication by 12.8%. ZeRO-3 used more peak memory than
ZeRO-2 for this small model because parameter prefetch and gathering buffers
outweighed its shard savings. This is a measured property of the exact test,
not a general statement about larger models or physical multi-GPU runs.

Every run writes:

- `summary.json` and `summary.md`
- one detailed JSON report per rank
- `cluster-report.json` and `cluster-report.md`
- collective call counts and the complete node-pair communication table

## WSL without `nvcc`

DeepSpeed 0.19.2 probes optional CUDA operators during import. The current
3090 Ti WSL environment has CUDA runtime libraries but no `nvcc`, so the
pure-PyTorch optimizer tests use the official detection bypass:

```bash
DS_IGNORE_CUDA_DETECTION=1 $PYTHON \
  verification/run_hybrid_deepspeed_numerics.py \
  --zero-stage all --precision bf16 --world-size 2 \
  --report-dir build/deepspeed-numerics
```

Do not use that setting as evidence that JIT-compiled DeepSpeed operators are
available. Install a matching CUDA compiler before testing those operators.

## Current boundary

The following paths are verified:

- client PyTorch SGD and AdamW optimizers
- DeepSpeed Engine backward and step
- ZeRO 0–3 on two logical ranks, plus a four-rank ZeRO-3 case
- FP32 and BF16 communication
- gradient accumulation
- Transformers Qwen3.5 + PEFT LoRA SFT
- reentrant activation checkpointing
- logical communication reports for every node pair

The following paths still require dedicated experiments:

- DeepSpeed fused optimizers and other JIT-compiled CUDA operators
- CPU optimizer/parameter offload and NVMe offload
- ZeRO checkpoint save, restore, and consolidation
- pipeline, tensor, sequence, expert, and MoE parallelism
- DeepSpeed launch across multiple physical hosts
- Hugging Face `Trainer` with a DeepSpeed configuration object

Passing the maintained suite therefore means the tested DeepSpeed training
path works; it does not claim complete DeepSpeed API or performance parity.
