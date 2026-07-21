# LLM SFT Memory Estimation

FakeGPU includes a reproducible Qwen3.5 text-SFT validation path that runs the
same random-token batch through real CUDA, CPU-backed FakeCUDA, and FakeTensor
ATen graph analysis. The worker measures one full-parameter training step:
forward, backward, and a single-tensor AdamW update.

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

## RTX PRO 5000 results

These measurements were collected on 2026-07-21 with an NVIDIA RTX PRO 5000
72GB Blackwell, PyTorch 2.8.0+cu128, CUDA 12.8, and Transformers 5.12.1. The
reference is `torch.cuda.max_memory_allocated()`, excluding CUDA context and
reserved-but-unused allocator memory.

| Model | Trainable parameters | Real peak | FakeCUDA peak / error | Static first-step peak / error | Static backward error |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 752,393,024 | 6.630 GiB | 6.561 GiB / 1.045% | 6.560 GiB / 1.047% | 1.442% |
| Qwen3.5-2B | 1,881,825,088 | 15.939 GiB | 15.923 GiB / 0.101% | 15.923 GiB / 0.102% | 0.173% |

Both reports pass a 2% overall-peak limit and a 5% static backward-peak limit.
The FakeCUDA/real loss differences are 0.471% and 1.200%, respectively, due to
CPU versus CUDA BF16 kernel numerics; the model weights and batch fingerprints
match.

## First step versus steady state

AdamW moment tensors do not exist until the first optimizer update. The static
estimator therefore reports separate first-step graph/overall peaks and
steady-state graph/overall peaks. This captures activation aliases, operator
temporaries, and workspace lifetimes instead of relying only on a fixed
parameter-plus-gradient-plus-optimizer formula.

The FakeCUDA runtime tracker cannot observe every short-lived temporary inside
CPU backward kernels, so its execution-only backward value underestimates this
workload by about 21%–26%. The comparison uses the captured ATen
storage-liveness graph as the gating backward prediction and retains the
runtime value as a diagnostic.

These results cover single-GPU full-parameter BF16 training with single-tensor
AdamW and the Transformers PyTorch fallback linear-attention path. LoRA/QLoRA,
gradient checkpointing, accumulation, Flash Linear Attention, custom
CUDA/Triton kernels, quantized optimizers, and sharded training require their
own validation.
