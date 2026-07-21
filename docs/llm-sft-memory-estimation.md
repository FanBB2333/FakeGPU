# LLM SFT Memory Estimation

FakeGPU includes a reproducible Qwen3.5 text-SFT validation path that runs the
same random-token batch through real CUDA, CPU-backed FakeCUDA, and FakeTensor
ATen graph analysis. The worker supports full-parameter, LoRA, and a packed-NF4
QLoRA reference path, gradient checkpointing, and gradient accumulation while
measuring forward, backward, and a single-tensor AdamW update.

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

Choose either LoRA or QLoRA; checkpointing and accumulation can then be
combined with that training method:

```bash
--training-method lora --lora-rank 8
--training-method qlora --lora-rank 8
--training-method qlora --lora-rank 8 --quantization-double-quantization
--gradient-checkpointing
--gradient-accumulation-steps 2
```

### Native NF4 QLoRA reference

`--training-method qlora` is a dependency-free reference backend for memory
validation. It replaces every eligible frozen linear weight except the output
embedding with two NF4 codes per byte and one FP32 absmax per 64 weights. LoRA
A/B matrices stay FP32, matching PEFT's default adapter promotion, and use
single-tensor AdamW.

`--quantization-double-quantization` enables nested scale storage. The worker
subtracts the mean of the first-level absmax values, encodes them with the
bitsandbytes signed dynamic 8-bit map, and stores an FP32 second-level absmax
per 256 values plus one FP32 offset. This follows the state layout in the
[official bitsandbytes implementation](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/functional.py)
and the [QLoRA paper](https://arxiv.org/abs/2305.14314). The paper reports an
average saving of about 0.37 bit per parameter from this step.

A custom autograd function dequantizes one frozen matrix for forward and
recomputes it for input-gradient calculation during backward. It therefore
does not retain all dequantized matrices. Static analysis captures the
equivalent PEFT LoRA ATen graph, substitutes exact packed storage, and adds the
largest implementation-specific dequantization workspace.

This path validates quantized storage, adapter gradients, optimizer state, and
dequantization lifetime without installing another package. It is not a
bitsandbytes kernel implementation: NF4 and nested-scale lookup use ordinary
PyTorch operations, and it has no fused 4-bit matmul, paged optimizer, or
bitsandbytes allocator behavior. In FakeCUDA reports, `runtime_memory_phases`
retains the raw CPU tracker values while `memory_phases` adds the known
per-layer reference dequantization workspace to compute peaks.

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

## Native NF4 QLoRA matrices

The same RTX PRO 5000 software stack produced the following results. All five
static predictions are analytical because the packed-storage substitution and
reference dequantization workspace are added to the captured LoRA graph.

### Direct FP32 scale storage

| Case | Checkpoint | Seq. | Trainable parameters | Real peak | Static peak | Static error | FakeCUDA error |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.8B QLoRA | no | 16 | 5,411,328 | 1.072 GiB | 1.079 GiB | 0.698% | 0.150% |
| 0.8B QLoRA | no | 128 | 5,411,328 | 2.007 GiB | 2.019 GiB | 0.628% | — |
| 0.8B QLoRA | yes | 128 | 5,411,328 | 1.192 GiB | 1.205 GiB | 1.085% | — |
| 2B QLoRA | no | 16 | 8,409,600 | 2.039 GiB | 2.067 GiB | 1.386% | — |
| 2B QLoRA | yes | 128 | 8,409,600 | 2.136 GiB | 2.172 GiB | 1.689% | — |

### Nested 8-bit scale storage

| Case | Checkpoint | Seq. | Trainable parameters | Real peak | Static peak | Static error | FakeCUDA error |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.8B QLoRA | no | 16 | 5,411,328 | 1.051 GiB | 1.058 GiB | 0.700% | 0.165% |
| 0.8B QLoRA | no | 128 | 5,411,328 | 1.986 GiB | 1.998 GiB | 0.629% | — |
| 0.8B QLoRA | yes | 128 | 5,411,328 | 1.171 GiB | 1.184 GiB | 1.094% | — |
| 2B QLoRA | no | 16 | 8,409,600 | 1.979 GiB | 2.008 GiB | 1.422% | — |
| 2B QLoRA | yes | 128 | 8,409,600 | 2.077 GiB | 2.113 GiB | 1.732% | — |

| Model | Quantized values | Direct storage | Direct bits/weight | Nested storage | Nested bits/weight | Storage saved |
|---|---:|---:|---:|---:|---:|---:|
| 0.8B | 497,614,848 | 280,098,816 B | 4.503 | 257,085,816 B | 4.133 | 23,013,000 B |
| 2B | 1,372,717,056 | 772,343,808 B | 4.501 | 708,524,040 B | 4.129 | 63,819,768 B |

Nested storage saves 0.370 bit/weight for 0.8B and 0.372 bit/weight for 2B.
Across the five fixed batches, the SFT loss changes by at most 0.018 compared
with direct FP32 scales. The short 0.8B case also completes the full FakeCUDA
optimizer step; all comparison checks pass without relaxing the default error
limits.

## Ampere and Blackwell

The RTX 3090 Ti (compute capability 8.6, PyTorch 2.12.1+cu130) reproduces the
0.8B full sequence-16 peak at 6.630 GiB with 1.047% static error, and the LoRA
checkpointed sequence-128 peak at 1.876 GiB with 1.506% static error. These
allocator peaks are effectively identical to the RTX PRO 5000 Blackwell
results. New attention implementations and shapes outside the workspace
calibration envelope still require separate validation.

For nested-scale NF4, both GPUs reproduce the 0.8B sequence-16 peak at 1.051
GiB and the checkpointed sequence-128 peak at 1.171 GiB byte-for-byte. Their
static peaks (1.058 GiB and 1.184 GiB), persistent storage, and random-batch
fingerprints are also identical across the two software stacks.

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

These results cover single-GPU full-parameter and LoRA BF16 training, the
PyTorch native-NF4 QLoRA reference with direct or nested scales, single-tensor
AdamW, gradient checkpointing, accumulation 2, and the Transformers PyTorch
fallback linear-attention path. External bitsandbytes fused kernels and
allocator behavior, paged/quantized optimizers, Flash Linear Attention, custom
CUDA/Triton kernels, and sharded training require separate validation.
