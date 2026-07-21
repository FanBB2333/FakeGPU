# LLM SFT Memory Estimation

FakeGPU includes a reproducible Qwen3.5 text-SFT validation path that runs the
same random-token batch through real CUDA, CPU-backed FakeCUDA, and FakeTensor
ATen graph analysis. The worker supports full-parameter or LoRA training,
gradient checkpointing, and gradient accumulation while measuring forward,
backward, and a single-tensor AdamW update.

## Workload

The default workload uses `Qwen3_5ForCausalLM` without the vision tower or MTP
weights, BF16 SDPA, batch size 1, sequence length 16, `use_cache=False`, and
AdamW with `foreach=False`. A fixed seed generates valid token IDs and masks
the first half of the labels with `-100`, approximating instruction/response
loss masking.

Random tokens are not meaningful training data. Once the model branch is
fixed, however, memory is primarily determined by tensor shapes, dtypes,
trainable parameters, and optimizer behavior, so this is a useful memory test.

## Run the three backends

```bash
PYTHON=/home/l1ght/anaconda3/envs/torch/bin/python
MODEL=/home/l1ght/models/Qwen/Qwen3.5-0.8B

$PYTHON verification/qwen_sft_memory_worker.py \
  --mode real --model-dir "$MODEL" --sequence-length 16 \
  --output build/qwen-sft/0.8b-real.json

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
$PYTHON verification/qwen_sft_memory_worker.py \
  --mode fakecuda --profile rtx-pro-5000-blackwell \
  --model-dir "$MODEL" --sequence-length 16 \
  --output build/qwen-sft/0.8b-fake.json

$PYTHON verification/qwen_sft_memory_worker.py \
  --mode static --model-dir "$MODEL" --sequence-length 16 \
  --output build/qwen-sft/0.8b-static.json

$PYTHON verification/compare_qwen_sft_memory.py \
  --real build/qwen-sft/0.8b-real.json \
  --fakecuda build/qwen-sft/0.8b-fake.json \
  --static build/qwen-sft/0.8b-static.json \
  --output build/qwen-sft/0.8b-comparison.json
```

Use the Qwen3.5-2B model path and matching output names to repeat the 2B case.

The following options can be combined:

```bash
--training-method lora --lora-rank 8
--gradient-checkpointing
--gradient-accumulation-steps 2
```

Summarize completed worker reports as one matrix:

```bash
$PYTHON verification/summarize_qwen_sft_matrix.py \
  --input-dir build/qwen-sft-matrix \
  --output build/qwen-sft-matrix/matrix.json
```

## RTX PRO 5000 matrix

These measurements were collected on 2026-07-21 with an NVIDIA RTX PRO 5000
72GB Blackwell, PyTorch 2.8.0+cu128, CUDA 12.8, and Transformers 5.12.1. The
reference is `torch.cuda.max_memory_allocated()`, excluding CUDA context and
reserved-but-unused allocator memory.

| Case | Method | Checkpoint | Accum. | Seq. | Trainable parameters | Real peak | Static error | FakeCUDA error |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0.8B full | full | no | 2 | 16 | 752,393,024 | 6.626 GiB | 0.989% | 0.985% |
| 0.8B full | full | no | 1 | 128 | 752,393,024 | 6.687 GiB | 1.121% | 1.119% |
| 0.8B full | full | yes | 1 | 128 | 752,393,024 | 6.679 GiB | 1.006% | 1.006% |
| 0.8B LoRA | LoRA r8 | no | 1 | 16 | 5,411,328 | 1.756 GiB | 1.921% | 2.439% |
| 0.8B LoRA | LoRA r8 | no | 1 | 128 | 5,411,328 | 2.696 GiB | 1.271% | — |
| 0.8B LoRA | LoRA r8 | yes | 1 | 128 | 5,411,328 | 1.876 GiB | 1.506% | — |
| 2B full | full | no | 2 | 16 | 1,881,825,088 | 15.939 GiB | 0.102% | — |
| 2B LoRA | LoRA r8 | no | 1 | 16 | 8,409,600 | 3.868 GiB | 0.266% | 0.500% |
| 2B LoRA | LoRA r8 | no | 1 | 128 | 8,409,600 | 4.959 GiB | 0.359% | — |
| 2B LoRA | LoRA r8 | yes | 1 | 128 | 8,409,600 | 3.974 GiB | 0.281% | — |

All ten cases pass. Static overall errors range from 0.102% to 1.921%; cases
with FakeCUDA execution range from 0.500% to 2.439%. LoRA checkpointing reduces
the sequence-128 peak by 30.4% for 0.8B and 19.9% for 2B. Full-parameter AdamW
remains optimizer-update dominated, so checkpointing mostly changes its forward
peak rather than the overall peak.

## Ampere and Blackwell

The RTX 3090 Ti (compute capability 8.6, PyTorch 2.12.1+cu130) reproduces the
0.8B full sequence-16 peak at 6.630 GiB with 1.047% static error, and the LoRA
checkpointed sequence-128 peak at 1.876 GiB with 1.506% static error. These
allocator peaks are effectively identical to the RTX PRO 5000 Blackwell
results. New attention implementations and shapes outside the workspace
calibration envelope still require separate validation.

## First step versus steady state

AdamW moment tensors do not exist until the first optimizer update. The static
estimator therefore reports separate first-step graph/overall peaks and
steady-state graph/overall peaks. This captures activation aliases, operator
temporaries, and workspace lifetimes instead of relying only on a fixed
parameter-plus-gradient-plus-optimizer formula.

Gradient accumulation models eager in-place accumulation with one additional
largest trainable-parameter gradient, rather than duplicating all gradients.
Its accumulation-2 backward errors are 1.305% for 0.8B and 0.157% for 2B.

PyTorch checkpoint saved-tensor hooks cannot currently be nested directly in
the `torch.func.grad` capture. Full training retains a parameter-gradient graph
floor. LoRA removes checkpointed decoder activations while retaining model
storage, outputs, recognized loss temporaries, gradients, optimizer state, and
profiled workspaces. Unrecognized loss paths are explicitly reported as upper
bounds.

The FakeCUDA runtime tracker cannot observe every short-lived temporary inside
CPU backward kernels, so its execution-only backward value underestimates this
workload by about 21%–26%. The comparison uses the captured ATen
storage-liveness graph as the gating backward prediction and retains the
runtime value as a diagnostic.

These results cover single-GPU full-parameter and LoRA BF16 training,
single-tensor AdamW, gradient checkpointing, accumulation 2, and the
Transformers PyTorch fallback linear-attention path. QLoRA, Flash Linear
Attention, custom CUDA/Triton kernels, quantized optimizers, and sharded
training require separate validation.
