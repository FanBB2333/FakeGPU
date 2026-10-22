# 線上 LLM serving

[文件索引](README.md) · [English](../en/serving.md) ·
[简体中文](../zh-CN/serving.md)

`plan-serving` 在部署前估算一張 GPU 能容納多少請求。`benchmark-serving` 則測量
已經在執行的服務。

<a id="plan-online-llm-serving-capacity"></a>

## 規劃 serving 容量

以下指令不會載入 checkpoint tensor，用於估算同構活躍請求池：

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

報告分別列出模型權重、prefill/decode 暫存記憶體、共享與私有 KV segment、runtime
開銷與可用餘量。

- 自訂容量可用 `--device-memory-gib` 取代 `--target-profile`。
- Prefix 共享需要 dynamic 或 paged cache。與 vLLM 一致，paged cache 只共享
  填滿的 prefix block，最後一個未填滿的 block 由每個請求各自保存。
- 容量搜尋不會超過 `--max-batch-size`。

## 對齊 vLLM 的記憶體預留

vLLM 會預先配置 paged KV 池，不會隨活躍請求逐步增長。部署目標為 vLLM 時可選擇
對應的記憶體策略：

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

模型對應 vLLM 0.19.1（V1 引擎，單 GPU）。

- 自動模式以 executor 申請的 GPU 記憶體（裝置記憶體乘以 `--memory-utilization`）
  減去 vLLM 啟動 profile run 的非 KV 記憶體，再向下取整為完整 block。報告會列出
  `num_gpu_blocks`、vLLM 日誌中 "GPU KV cache size" 的 token 數、初始化是否可行
  與 `max_model_len` 下的並行量，並與請求實際需要的邏輯 KV 分開呈現。
- profile run 是一次 `max_num_batched_tokens` 個 token 的 forward，加上每個請求
  一列、共 `min(max_num_batched_tokens, max_num_seqs)` 列的 sampler：模型 dtype
  的 logits、float32 logits、機率與取樣雜訊，以及 top-k/top-p workspace。再加上
  權重與建模的 112 MiB 非 PyTorch 記憶體（函式庫 handle 與已載入的 kernel）。
  此值是在一張 Ampere GPU（sm_80）上以 vLLM 0.19.1 量得，會隨軟體堆疊變化。
- `--max-batch-size` 即 vLLM 的 `max_num_seqs`，`--vllm-max-num-batched-tokens`
  即 `max_num_batched_tokens`。未指定時採用 vLLM 預設值：`vllm serve` 為 256 個
  序列、2048 個 token；離線 `LLM` 類別（`--vllm-usage-context llm_class`）為
  8192 個 token。記憶體不少於 70 GiB 且不是 A100 的裝置為 1024 個序列、8192 或
  16384 個 token。token 預設值不超過 `max_num_seqs` 乘以 `max_model_len`
  （最長請求）。
- 未指定 `--prefill-chunk-tokens` 時，prefill 依 `max_num_batched_tokens` 分塊。
- `--vllm-kv-cache-dtype` 可選 `auto`（沿用模型 dtype）、`bfloat16`、`float16`、
  `fp8`、`fp8_e4m3` 與 `fp8_e5m2`。fp8 類型每個元素占 1 位元組。
- sliding-window 模型使用 vLLM 的 hybrid KV cache group：每種 attention 類型的層
  分成大小相同的 group，最後一個 group 補齊；每個 block ID 在一個 group 的每層各
  占一頁。sliding-window group 保留視窗加上本步排程的 token；`max_model_len` 下的
  並行量依一個請求在所有 group 中需要的 block 計算。
- MLA 模型快取 latent 加 RoPE key。block 大小預設 16；compute capability 10.x 上
  vLLM 選擇 FlashInfer MLA（32 個 token，需 `qk_nope_head_dim` 為 64、128 或
  192），否則選擇 CUTLASS MLA（128 個 token）。可用 `--kv-cache-block-tokens`
  覆寫。
- 與 vLLM 一致，有一個 block 作為 null block 保留不用。剩餘 block 放不下一個
  `max_model_len` 請求時，初始化失敗。
- paged prefix cache 命中只計完整 block，且最多 `prompt_tokens - 1` 個 token。
- 未建模 CUDA graph 記憶體：vLLM 0.19.1 在決定 KV cache 大小之後才 capture
  graph，只有設定 `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` 時才會扣除。
  `torch.compile` 對 forward 峰值的影響與 attention backend workspace 也未建模。
- `--vllm-non-kv-cache-memory-bytes` 以 vLLM 啟動 profile（權重、activation 峰值、
  非 PyTorch 記憶體與 CUDA graph）取代建模的非 KV 值。
- `--vllm-kv-cache-memory-bytes` 指定每張 GPU 的精確 cache 大小。與 vLLM 一致，
  此時不再使用 `--memory-utilization`。
- 與 vLLM 的 `--dtype auto` 一致，float32 checkpoint 以 bfloat16 計算；compute
  capability 低於 8.0 時以 float16 計算。
- `--memory-utilization` 預設 `0.92`，與
  [vLLM CacheConfig 文件](https://docs.vllm.ai/en/latest/api/vllm/config/cache/)
  一致。固定軟體版本的報告建議明確填入。
- 預算對應 vLLM 的
  [GPU worker profile 流程](https://docs.vllm.ai/en/latest/api/vllm/v1/worker/gpu_worker/)。
  初始可用記憶體、其他程序、tensor/pipeline parallelism、記憶體碎片與版本相關的
  kernel，仍需在相同軟體堆疊上觀測。
- Speculative decoding 目前只適用於 generic 執行期模型。

## Speculative decoding

指定獨立的自迴歸 draft checkpoint，即可估算 speculative decoding 記憶體：

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

target 與 draft 權重和兩套 KV cache 都保持常駐，並加入保守的 lookahead slot。
接受率只影響每次 target 呼叫的預期輸出 token 數。它不改變記憶體峰值，也不代表
加速比。兩個模型的 vocabulary 大小必須相同，tokenizer identity 則不做檢查。

每份 serving 報告都包含與路徑無關的 `workload_signature`。它涵蓋架構、權重儲存、
請求 shape、KV 與 prefill 設定，以及 speculative 設定，不含模型路徑與裝置容量。
`calibrate verify` 會比較簽章，並單獨檢查 `target.profile`。其他設定或 GPU 的
證據不會通過驗證。

## 混合請求長度

對於長度不同的 chat、RAG、completion 或 summarization 流量，可以寫一份依準入
順序排列的請求清單：

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

- 依清單順序逐一計算請求。每個 prefix 群組只儲存一次，在不重排的前提下接納能放入
  GPU 記憶體的最長清單前綴。
- 報告包含已接納與被拒絕的請求 ID、每個請求的暫存記憶體、並行 prefill 上界、群組
  KV segment 與容量餘量。
- 同組請求必須使用相同且大於零的 `shared_prefix_tokens`。命名群組視為已駐留的
  cache 命中。
- 請求到達時間、eviction、preemption、tensor parallelism、自適應 speculative 排程、
  KV 回復、吞吐量與延遲均未建模。取得匹配觀測前，報告保持 `Modeled` 與
  `uncalibrated`。

要把計畫與實測峰值比較，請參閱
[對照實體 GPU 校準](calibration.md#serving-observations)。

## 測量執行中的服務

`benchmark-serving` 測量已在執行的 completion 端點，依賴 vLLM 的 `ignore_eos` 與
最終串流 usage 欄位。token-ID 清單指定到達時間與輸出上限。報告保留 SSE 事件、失敗、
用戶端排隊時間與首個非空文字的延遲。它不重試請求，也不推斷裝置階段峰值。`--model`
請填寫服務端設定的模型名稱。

```bash
python -m fakegpu benchmark-serving \
  --base-url http://127.0.0.1:8000/v1 --model local-model \
  --requests examples/serving/online-requests.json \
  --concurrency 4 --repetitions 3 --json build/online.json.gz
```

[原生 serving 記錄](../../tests/data/serving_native_evidence.json)包含 20 個
8K/32K Transformers 程序，以及 prefix caching 關閉與開啟時的 72 個混合 vLLM 請求。
32K 時 allocated 兩階段誤差為 2.02% 與 7.34%，但 reserved decode 記憶體低估了
24.07%。單靠張量儲存估算，無法保證 allocator 或 OOM 安全。
