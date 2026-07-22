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

### CPU offload

The same runner exposes optimizer and parameter offload as command-line axes:

```bash
$PYTHON verification/run_hybrid_deepspeed_numerics.py \
  --zero-stage 2 --precision fp32 --world-size 2 \
  --offload optimizer \
  --report-dir build/deepspeed-offload-zero2

$PYTHON verification/run_hybrid_deepspeed_numerics.py \
  --zero-stage 3 --precision fp32 --world-size 2 \
  --offload optimizer-and-parameter \
  --report-dir build/deepspeed-offload-zero3
```

These checks use a client-provided PyTorch SGD optimizer and set
`zero_force_ds_cpu_optimizer=false`; they do not compile DeepSpeed CPUAdam.
The validator requires optimizer state to be on CPU and, for ZeRO-3 parameter
offload, requires the local parameter partition to be on CPU.

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

## Checkpoint save and resume

The checkpoint runner performs an optimizer step, saves from every rank,
constructs a fresh engine, restores model/optimizer/scheduler/client state,
continues training, and compares the result with an uninterrupted run. For
ZeRO-2/3 it also reconstructs an FP32 state dict.

```bash
$PYTHON verification/run_hybrid_deepspeed_checkpoint.py \
  --zero-stage 3 --precision fp32 --world-size 2 \
  --report-dir build/deepspeed-checkpoint
```

DeepSpeed 0.15.3 checkpoints loaded by PyTorch 2.8 require three legacy
DeepSpeed types to be registered with PyTorch's safe checkpoint loader. The
runner allowlists only the types reported by PyTorch for its own checkpoint;
it does not disable `weights_only` protection globally. DeepSpeed requires all
ranks to participate in checkpoint save operations; see the
[DeepSpeed checkpoint documentation](https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html).

## Hugging Face Trainer

The Trainer validator supports a self-contained tiny model and local Qwen
LoRA weights. It passes the DeepSpeed configuration through
`TrainingArguments(deepspeed=...)` and checks the actual adapter or model
update on every rank.

```bash
$PYTHON verification/run_hf_trainer_deepspeed.py \
  --workload tiny --zero-stage 2 --precision bf16 \
  --gradient-accumulation-steps 2 \
  --report-dir build/hf-trainer-deepspeed-tiny

$PYTHON verification/run_hf_trainer_deepspeed.py \
  --workload qwen-lora --model-dir "$MODEL" \
  --zero-stage 3 --precision bf16 --sequence-length 64 \
  --gradient-accumulation-steps 2 --gradient-checkpointing \
  --report-dir build/hf-trainer-deepspeed-qwen
```

The configuration follows the
[Transformers DeepSpeed integration](https://huggingface.co/docs/transformers/main/en/deepspeed),
including Trainer-resolved `auto` values.

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

Additional maintained checks passed on both GPU/software stacks:

| Path | Coverage | Result |
|---|---|---|
| ZeRO checkpoint | ZeRO-3 FP32; ZeRO-2/3 BF16 | save, fresh-engine restore, AdamW/StepLR/client-state resume, uninterrupted-result match, and FP32 consolidation |
| Hugging Face Trainer | tiny ZeRO-2/3 BF16; Qwen3.5-0.8B LoRA ZeRO-3 BF16 | finite loss, real parameter update, accumulation, rank consistency, and communication report |
| CPU offload | ZeRO-2 optimizer; ZeRO-3 optimizer + parameter, FP32 | exact analytical update; optimizer state and requested parameter partitions on CPU |

## Physical multi-host DeepSpeed

The SSH controller can place one DeepSpeed rank on each physical host:

```bash
python3 verification/run_physical_multihost.py \
  --node 'name=blackwell;ssh=gpu-a;repo=/home/user/repos/fakeGPU;python=/opt/torch/bin/python;shell=posix' \
  --node 'name=ampere-wsl;ssh=user@gpu-b;repo=/home/user/repos/fakeGPU;python=/opt/torch/bin/python;shell=wsl' \
  --coordinator-host 100.x.y.z \
  --case deepspeed-zero2
```

The maintained RTX PRO 5000 ↔ RTX 3090 Ti ZeRO-2 run completed seven TCP
collectives, recorded 176 node-pair bytes with a 32-byte per-operation peak,
and produced identical parameters `[0.775, -0.45]` on both ranks. This run
used DeepSpeed 0.15.3 and 0.19.2 respectively.

ZeRO-3 issues a version-dependent parameter-trace collective sequence. The
current two hosts use different DeepSpeed versions, and an unguarded run was
observed to issue an all-gather on 0.15.3 while 0.19.2 issued a broadcast at
the same sequence number. `--case deepspeed-zero3` therefore requires an
identical DeepSpeed version on both hosts and reports a preflight error before
training when they differ. ZeRO-3 remains verified separately on each GPU.

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
- ZeRO checkpoint save, restore, continued training, and FP32 consolidation
- Hugging Face `Trainer` with tiny and Qwen LoRA workloads
- CPU optimizer offload and ZeRO-3 CPU parameter offload
- physical two-host ZeRO-2 over TCP
- logical communication reports for every node pair

The following paths still require dedicated experiments:

- DeepSpeed fused optimizers and other JIT-compiled CUDA operators
- NVMe offload
- pipeline, tensor, sequence, expert, and MoE parallelism
- physical two-host ZeRO-3 with identical DeepSpeed versions

Passing the maintained suite therefore means the tested DeepSpeed training
path works; it does not claim complete DeepSpeed API or performance parity.
