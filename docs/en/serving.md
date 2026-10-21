# Online LLM serving

[Documentation index](README.md) · [简体中文](../zh-CN/serving.md) ·
[繁體中文](../zh-TW/serving.md)

`plan-serving` estimates how many requests fit on a GPU before you deploy.
`benchmark-serving` measures a server that is already running.

<a id="plan-online-llm-serving-capacity"></a>

## Plan serving capacity

Estimate a pool of identical active requests without loading checkpoint
tensors:

```bash
python3 -m fakegpu plan-serving \
  --model-dir /models/example \
  --active-sequences 16 \
  --max-batch-size 64 \
  --prompt-tokens 4096 \
  --generated-tokens 256 \
  --prefill-chunk-tokens 512 \
  --shared-prefix-tokens 1024 \
  --kv-cache-strategy paged \
  --kv-cache-block-tokens 16 \
  --target-profile a100 \
  --memory-utilization 0.9 \
  --json build/serving-plan.json
```

The report separates weights, prefill and decode transients, shared and private
KV segments, runtime overhead, and usable headroom.

- Use `--device-memory-gib` instead of `--target-profile` for a custom
  capacity.
- Prefix sharing needs a dynamic or paged cache. Like vLLM, a paged cache
  shares only full prefix blocks; a partial last block stays private to
  each request.
- Capacity search never goes beyond `--max-batch-size`.

## Match vLLM's memory reservation

vLLM preallocates a paged KV pool instead of growing it with active requests.
Select its memory policy when you deploy on vLLM:

```bash
python3 -m fakegpu plan-serving \
  --model-dir /models/Qwen/Qwen3-0.6B \
  --active-sequences 16 \
  --max-batch-size 64 \
  --prompt-tokens 4096 \
  --generated-tokens 256 \
  --runtime vllm \
  --kv-cache-strategy paged \
  --kv-cache-block-tokens 16 \
  --target-profile "$TARGET_GPU_PROFILE" \
  --memory-utilization 0.92 \
  --json build/vllm-serving-plan.json
```

- Automatic sizing follows vLLM V1. It takes the requested executor memory,
  subtracts the profiled non-KV memory, and rounds down to whole blocks. The
  report shows `num_gpu_blocks`, cache token capacity, whether initialization
  fits, and concurrency at `max_model_len`, separately from the logical KV
  demand of the requests.
- Like vLLM, one block is held back as the null block. Initialization fails
  unless one `max_model_len` request fits in the remaining blocks.
- `--vllm-non-kv-cache-memory-bytes` replaces the modeled non-KV value with
  one from a vLLM startup profile (weights, peak activations, non-torch memory,
  and CUDA graphs).
- `--vllm-kv-cache-memory-bytes` sets an exact per-GPU cache size. Like vLLM,
  it then ignores `--memory-utilization`.
- `--memory-utilization` defaults to `0.92`, matching the
  [vLLM CacheConfig reference](https://docs.vllm.ai/en/latest/api/vllm/config/cache/).
  Set it explicitly for version-locked reports.
- The budget follows vLLM's
  [GPU-worker profiling path](https://docs.vllm.ai/en/latest/api/vllm/v1/worker/gpu_worker/).
  Free memory, other processes, tensor and pipeline parallelism, hybrid cache
  groups, fragmentation, and version-specific kernels still need an
  observation on the same stack.
- Speculative decoding is only available with the generic runtime model.

## Speculative decoding

Add an independent autoregressive draft checkpoint to estimate speculative
decoding memory:

```bash
python3 -m fakegpu plan-serving \
  --model-dir /models/target \
  --draft-model-dir /models/draft \
  --active-sequences 8 \
  --max-batch-size 32 \
  --prompt-tokens 4096 \
  --generated-tokens 256 \
  --speculative-tokens 5 \
  --speculative-acceptance-rate 0.7 \
  --kv-cache-strategy paged \
  --target-profile a100 \
  --json build/speculative-serving-plan.json
```

Target and draft weights and both KV caches stay resident, with conservative
lookahead slots. The acceptance rate only changes the expected output tokens
per target call. It does not change peak memory and it is not a speedup
estimate. Vocabulary sizes must match, but tokenizer identity is not checked.

Every serving report includes a path-independent `workload_signature`. It
covers architecture, weight storage, request shapes, KV and prefill settings,
and speculative settings, but not model paths or device capacity.
`calibrate verify` checks it and `target.profile` separately, so evidence from
another configuration or GPU is rejected.

## Mixed request lengths

For chat, RAG, completion, or summarization traffic with different lengths,
write an ordered request manifest:

```json
{
  "schema_version": "fakegpu.serving_requests.v1",
  "requests": [
    {
      "id": "chat-a",
      "prompt_tokens": 4096,
      "generated_tokens": 256,
      "prefix_group": "system-prompt",
      "shared_prefix_tokens": 1024
    },
    {
      "id": "chat-b",
      "prompt_tokens": 8192,
      "generated_tokens": 512,
      "prefix_group": "system-prompt",
      "shared_prefix_tokens": 1024
    },
    {
      "id": "completion",
      "prompt_tokens": 2048,
      "generated_tokens": 1024
    }
  ]
}
```

```bash
python3 -m fakegpu plan-serving \
  --model-dir /models/example \
  --requests serving-requests.json \
  --max-batch-size 64 \
  --prefill-chunk-tokens 512 \
  --prefill-concurrency 2 \
  --kv-cache-strategy paged \
  --target-profile a100 \
  --memory-utilization 0.9 \
  --json build/mixed-serving-plan.json
```

- Requests are evaluated in file order. Each prefix group is stored once, and
  the longest fitting prefix of the list is admitted without reordering.
- The report lists admitted and rejected IDs, per-request transients,
  concurrent-prefill bounds, grouped KV segments, and headroom.
- Members of a group must share the same positive `shared_prefix_tokens`.
  Named groups are treated as resident cache hits.
- Arrival timing, eviction, preemption, tensor parallelism, adaptive
  speculation, KV rollback, throughput, and latency are not modeled. Reports
  stay `Modeled` and `uncalibrated` until a matching observation is supplied.

To compare a plan with measured peaks, see
[Calibrating against a real GPU](calibration.md#serving-observations).

## Benchmark a running server

`benchmark-serving` measures an already running completion endpoint using
vLLM's `ignore_eos` and final streaming usage fields. A token-ID manifest sets
arrivals and output limits. The report keeps SSE events, failures, client queue
time, and the latency to the first nonempty text. It does not retry requests or
infer device phase peaks. Use your server's served model name.

```bash
python -m fakegpu benchmark-serving \
  --base-url http://127.0.0.1:8000/v1 --model local-model \
  --requests examples/serving/online-requests.json \
  --concurrency 4 --repetitions 3 --json build/online.json.gz
```

The [native serving record](../../tests/data/serving_native_evidence.json)
covers 20 8K/32K Transformers processes and 72 mixed vLLM requests with prefix
caching off and on. At 32K, allocated phase errors were 2.02% and 7.34%, but
reserved decode memory was underestimated by 24.07%. Tensor-storage estimates
alone do not establish allocator or OOM safety.
