# LLM Inference Estimation

FakeGPU offers three complementary levels for decoder-only inference:

| Level | GPU required | What it measures |
|---|---:|---|
| Checkpoint-only estimate | No | Dense/MoE parameter bytes, quantized storage, adapters, KV cache, transient tensors, expert traffic, matrix FLOPs, and an optional profile roofline interval |
| FakeCUDA execution | No | Real CPU execution through CUDA-shaped PyTorch code, tensor-lifetime memory, generated tokens, and observed matrix FLOPs |
| CUDA calibration | Yes | PyTorch allocator and NVML process memory for one exact GPU/software/operator path |

The first two levels can be used on a CPU-only host. A calibration makes
backend/context overhead visible, but should only be reused for a matching GPU,
PyTorch/CUDA stack, model shape, dtype, and attention implementation.

## Inspect a checkpoint without loading weights

```bash
python3 -m fakegpu estimate-llm \
  --model-dir /models/Qwen/Qwen3-8B \
  --batch-size 1 \
  --prompt-tokens 9 \
  --generated-tokens 2 \
  --dtype bfloat16 \
  --attention-implementation sdpa \
  --target-profile a100 \
  --json build/qwen-estimate.json
```

This command reads `config.json` and safetensors headers only. It does not read
tensor payloads, materialize the model, or create a CUDA context. The estimate
contains:

- exact checkpoint parameter count and storage metadata
- base and adapter parameter bytes at the selected runtime representation
- KV-cache bytes from layers, KV heads, head dimension, batch, and sequence
- eager- or SDPA-specific transient tensor estimates
- dense or active routed/shared-expert matrix FLOPs
- uniform expert-parallel dispatch/combine bytes when world size is specified
- lower/expected/upper memory-traffic and optional roofline latency intervals

Quantized base checkpoints use the exact safetensors payload size, including
scale tensors. Dense checkpoints and adapters use parameter count multiplied
by the selected runtime dtype. The compute model includes only active experts,
router matrix multiplication, shared experts, attention, and the LM head.

Pass adapters and expert-parallel size explicitly:

```bash
fakegpu estimate-llm \
  --model-dir /models/moe-decoder \
  --adapter-dir /models/adapters/domain-lora \
  --adapter-dir /models/adapters/style-lora \
  --expert-parallel-size 4 \
  --prompt-tokens 128 \
  --generated-tokens 16 \
  --target-profile h100 \
  --json build/moe-adapter-estimate.json
```

The expert traffic estimate assumes uniform placement and routing. It does not
predict hot experts, capacity-factor drops, all-to-all protocol overhead, or
fused quantization workspace.

## Execute the model with no visible GPU

The maintained worker loads the actual Transformers model into host memory and
executes the same forward/decode flow on CPU-backed FakeCUDA tensors:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
python3 verification/qwen_inference_memory_worker.py \
  --mode fakecuda \
  --model-dir /models/Qwen/Qwen3-8B \
  --profile rtx-pro-5000-blackwell \
  --prompt Hello \
  --generated-tokens 2 \
  --dtype bfloat16 \
  --attention-implementation sdpa \
  --smi-state /tmp/fakegpu-qwen.json \
  --output build/qwen-fakecuda.json
```

`CUDA_VISIBLE_DEVICES=''` is intentional: it proves that this execution path
does not fall through to a physical GPU. FakeCUDA does real CPU computation for
maintained PyTorch operators; it does not try to reproduce CUDA kernel timing.

For a normal application, set the state path before calling
`fakegpu.init(runtime="fakecuda")`. While the process is alive, inspect it from
a second terminal:

```bash
python3 -m fakegpu nvidia-smi --state /tmp/fakegpu-qwen.json
```

Use `FAKEGPU_SMI_STATE_DIR=/tmp/fakegpu-smi` when several processes should
publish separate files. The viewer aggregates their reported memory per
logical GPU. It shows host, profile, stage, tracking confidence, and both
current/peak simulated and raw tracked memory. For a bounded live view:

```bash
python3 -m fakegpu nvidia-smi \
  --state-dir /tmp/fakegpu-smi --loop 1 --count 10
```

The state directory is scanned again on every refresh so processes can appear
after the viewer starts. Looped `--json` output uses one JSON object per line.

## Calibrate against real CUDA

Run the same prompt, generation count, dtype, and attention implementation:

```bash
python3 verification/qwen_inference_memory_worker.py \
  --mode real \
  --model-dir /models/Qwen/Qwen3-8B \
  --prompt Hello \
  --generated-tokens 2 \
  --dtype bfloat16 \
  --attention-implementation sdpa \
  --output build/qwen-real.json

python3 verification/compare_qwen_memory.py \
  --real build/qwen-real.json \
  --fakecuda build/qwen-fakecuda.json \
  --output build/qwen-comparison.json \
  --markdown build/qwen-comparison.md
```

The comparison requires exact parameter counts and generated token IDs, less
than 1% load/inference memory error, less than 0.01% static FLOP error, and
exact FakeCUDA-versus-CUDA observed matrix FLOPs.

## Maintained Qwen3-8B result

The following result was measured on an NVIDIA RTX PRO 5000 72GB Blackwell,
PyTorch 2.9.1/CUDA 12.8, BF16 SDPA, one 9-token prompt, and two generated token
IDs:

| Comparison | Predicted | Observed | Absolute error |
|---|---:|---:|---:|
| FakeCUDA model load vs CUDA allocator | 16,381,470,976 B | 16,383,586,816 B | 0.012914% |
| FakeCUDA inference peak vs CUDA allocator | 16,385,992,936 B | 16,396,630,528 B | 0.064877% |
| Static inference peak vs CUDA allocator | 16,385,606,472 B | 16,396,630,528 B | 0.067234% |
| Virtual SMI process vs NVML process | 16,825,298,920 B | 16,835,936,256 B | 0.063182% |
| FakeCUDA observed matrix FLOPs vs CUDA | 151,415,620,864 | 151,415,620,864 | 0% |
| Static matrix FLOPs vs CUDA | 151,415,619,584 | 151,415,620,864 | 0.000001% |

The virtual-SMI comparison adds a `442,049,024`-byte overhead measured as NVML
process memory minus current CUDA allocator bytes in that same run. Do not use
that number as a universal CUDA-context constant.

## Accuracy boundary

This result supports a concrete statement: FakeGPU is accurate for this
maintained dense-Qwen inference envelope. MoE, quantized, and adapter support
is an analytical checkpoint model and does not inherit the same empirical
error bound.

Check or calibrate separately when a workload uses:

- nonuniform MoE routing, fused quantization kernels, adapter merge state, or mixed per-tensor runtime dtypes
- custom CUDA extensions, Triton kernels, or operators outside FakeCUDA coverage
- model-specific persistent buffers or dynamically changing control flow
- a different attention backend, PyTorch/CUDA release, allocator policy, or GPU
- exact throughput/latency prediction; profile roofline output is an analytical interval, not a benchmark

For an arbitrary repository, run
[`fakegpu analyze-repo`](repository-and-performance-analysis.md) first, then
use [`fakegpu preflight`](ai-researcher-preflight.md) or the ATen graph
estimator for the selected entrypoint and shapes.
