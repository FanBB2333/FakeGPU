# 在线 LLM serving

[文档索引](README.md) · [English](../en/serving.md) ·
[繁體中文](../zh-TW/serving.md)

`plan-serving` 在部署前估算一张 GPU 能容纳多少请求。`benchmark-serving` 测量已经
在运行的服务。

<a id="plan-online-llm-serving-capacity"></a>

## 规划 serving 容量

以下命令不会加载 checkpoint tensor，用于估算一组相同的活跃请求：

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

报告分别列出模型权重、prefill/decode 临时显存、共享与私有 KV segment、runtime
开销和可用余量。

- 自定义容量时，用 `--device-memory-gib` 代替 `--target-profile`。
- Prefix 共享需要 dynamic 或 paged cache。与 vLLM 一致，paged cache 只共享
  填满的 prefix block，最后一个未填满的 block 由每个请求各自保存。
- 容量搜索不会超过 `--max-batch-size`。

## 对齐 vLLM 的显存预留

vLLM 会预先分配 paged KV 池，不会随活跃请求逐步增长。部署目标是 vLLM 时，选择
对应的显存策略：

```bash
python3 -m fakegpu plan-serving \
  --model-dir /models/Qwen/Qwen3-0.6B \
  --active-sequences 16 \
  --prompt-tokens 4096 \
  --generated-tokens 256 \
  --runtime vllm \
  --max-batch-size 256 \
  --vllm-max-num-batched-tokens 8192 \
  --target-profile "$TARGET_GPU_PROFILE" \
  --memory-utilization 0.92 \
  --json build/vllm-serving-plan.json
```

模型对应 vLLM 0.19.1（V1 引擎，单 GPU）。

- 自动模式用 executor 申请的显存（设备显存乘以 `--memory-utilization`）减去
  vLLM 启动 profile run 的非 KV 显存，再向下取整为完整 block。报告给出
  `num_gpu_blocks`、vLLM 日志中 "GPU KV cache size" 的 token 数、初始化能否
  放下，以及 `max_model_len` 下的并发量，并与请求实际需要的逻辑 KV 分开列出。
- profile run 是一次 `max_num_batched_tokens` 个 token 的 forward，加上每个请求
  一行、共 `min(max_num_batched_tokens, max_num_seqs)` 行的 sampler：模型 dtype
  的 logits、float32 logits、概率与采样噪声，以及 top-k/top-p workspace。再加上
  权重和建模的 112 MiB 非 torch 显存（库句柄与已加载的 kernel）。该值是在一张
  Ampere GPU（sm_80）上用 vLLM 0.19.1 测得的，会随软件栈变化。
- `--max-batch-size` 即 vLLM 的 `max_num_seqs`，`--vllm-max-num-batched-tokens`
  即 `max_num_batched_tokens`。不指定时取 vLLM 默认值：`vllm serve` 为 256 个
  序列、2048 个 token；离线 `LLM` 类（`--vllm-usage-context llm_class`）为 8192
  个 token。显存不少于 70 GiB 且不是 A100 的设备为 1024 个序列、8192 或 16384
  个 token。token 默认值不超过 `max_num_seqs` 乘以 `max_model_len`（最长请求）。
- 未指定 `--prefill-chunk-tokens` 时，prefill 按 `max_num_batched_tokens` 分块。
- `--vllm-kv-cache-dtype` 可选 `auto`（沿用模型 dtype）、`bfloat16`、`float16`、
  `fp8`、`fp8_e4m3` 和 `fp8_e5m2`。fp8 类型每个元素占 1 字节。
- sliding-window 模型使用 vLLM 的 hybrid KV cache group：每种 attention 类型的层
  分成大小相同的 group，最后一个 group 补齐；每个 block ID 在一个 group 的每层各
  占一页。sliding-window group 保留窗口加上本步调度的 token；`max_model_len` 下的
  并发量按一个请求在所有 group 中需要的 block 计算。
- MLA 模型缓存 latent 加 RoPE key。block 大小默认 16；compute capability 10.x 上
  vLLM 选择 FlashInfer MLA（32 个 token，要求 `qk_nope_head_dim` 为 64、128 或
  192），否则选择 CUTLASS MLA（128 个 token）。可用 `--kv-cache-block-tokens`
  覆盖。
- 与 vLLM 一致，有一个 block 作为 null block 保留不用。剩余 block 放不下一个
  `max_model_len` 请求时，初始化失败。
- paged prefix cache 命中只计完整 block，且最多 `prompt_tokens - 1` 个 token。
- 未建模 CUDA graph 显存：vLLM 0.19.1 在确定 KV cache 大小之后才 capture graph，
  只有设置 `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` 时才会扣除。
  `torch.compile` 对 forward 峰值的影响和 attention backend workspace 也未建模。
- `--vllm-non-kv-cache-memory-bytes` 用 vLLM 启动 profile 中的值（权重、activation
  峰值、非 torch 显存和 CUDA graph）替代建模的非 KV 值。
- `--vllm-kv-cache-memory-bytes` 指定每张 GPU 的精确 cache 大小。与 vLLM 一致，
  这时不再使用 `--memory-utilization`。
- 与 vLLM 的 `--dtype auto` 一致，float32 checkpoint 按 bfloat16 计算；compute
  capability 低于 8.0 时按 float16 计算。
- `--memory-utilization` 默认 `0.92`，与
  [vLLM CacheConfig 文档](https://docs.vllm.ai/en/latest/api/vllm/config/cache/)
  一致。需要锁定版本的报告建议显式填写。
- 预算对应 vLLM 的
  [GPU worker profile 流程](https://docs.vllm.ai/en/latest/api/vllm/v1/worker/gpu_worker/)。
  空闲显存、其他进程、tensor/pipeline parallelism、碎片和版本相关的 kernel，
  仍需在相同软件栈上观测。
- Speculative decoding 目前只适用于 generic 运行时模型。

## Speculative decoding

加上一个独立的自回归 draft checkpoint，即可估算 speculative decoding 显存：

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

target 与 draft 的权重和两套 KV cache 都常驻显存，并留出保守的 lookahead slot。
接受率只影响每次 target 调用的预期输出 token 数。它不改变显存峰值，也不是加速比
估计。两个模型的 vocabulary 大小必须相同，但不检查 tokenizer 是否一致。

每份 serving 报告都带有与路径无关的 `workload_signature`。它涵盖架构、权重存储、
请求 shape、KV 和 prefill 设置以及 speculative 设置，不含模型路径和设备容量。
`calibrate verify` 会检查签名，并单独检查 `target.profile`，其他配置或 GPU 的
证据会被拒绝。

## 混合请求长度

chat、RAG、completion 或 summarization 请求长度各不相同时，写一个按顺序排列的
请求清单：

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

- 按文件顺序逐个计算请求。每个 prefix 分组只保存一次。在不重排的前提下，接纳
  能放进显存的最长清单前缀。
- 报告列出已接纳和被拒绝的请求 ID、每个请求的临时显存、并发 prefill 上界、分组
  KV segment 和余量。
- 同组请求必须使用相同且大于零的 `shared_prefix_tokens`。命名分组视为已驻留的
  cache 命中。
- 请求到达时间、eviction、preemption、tensor parallelism、自适应 speculative 调度、
  KV 回退、吞吐量和延迟都没有建模。拿到匹配的观测之前，报告保持 `Modeled` 和
  `uncalibrated`。

要把计划与实测峰值对比，请看
[对照真实 GPU 校准](calibration.md#serving-observations)。

## 测试运行中的服务

`benchmark-serving` 测量已在运行的 completion 接口，依赖 vLLM 的 `ignore_eos` 和
最终流式 usage 字段。token-ID 清单指定到达时间和输出上限。报告保留 SSE 事件、
失败、客户端排队时间，以及到首个非空文本的延迟。它不重试请求，也不推断设备阶段
峰值。`--model` 填写服务端配置的模型名称。

```bash
python -m fakegpu benchmark-serving \
  --base-url http://127.0.0.1:8000/v1 --model local-model \
  --requests examples/serving/online-requests.json \
  --concurrency 4 --repetitions 3 --json build/online.json.gz
```

[真实 serving 记录](../../tests/data/serving_native_evidence.json)包含 20 个
8K/32K Transformers 进程，以及 prefix caching 关闭和开启时的 72 个混合 vLLM 请求。
32K 时 allocated 两个阶段的误差为 2.02% 和 7.34%，但 reserved decode 显存低估了
24.07%。只靠张量存储估算，不能保证 allocator 或 OOM 安全。
