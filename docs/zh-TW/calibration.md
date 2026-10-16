# 對照實體 GPU 校準

[文件索引](README.md) · [English](../en/calibration.md) ·
[简体中文](../zh-CN/calibration.md)

FakeGPU 的估算要先在相同軟體堆疊上和實測結果比較過，才適合用於該工作負載。本頁說明
如何採集實測資料，以及如何用它們對估算做門檻檢查。已公開的結果整理在
[驗證與可靠性](validation.md)。

## 靜態記憶體驗證

在真實 CUDA 主機上，可以重新執行專案維護的受控 ATen 比較：

```bash
python3 scripts/validation/static_memory_validation.py \
  --output build/static-memory-validation.json \
  --markdown build/static-memory-validation.md \
  --max-underestimate-percent 5
```

在無 GPU 主機上加入 `--static-only`，只檢查估算流程，不會產生準確性結果。

<a id="serving-observations"></a>

## Serving 觀測

### 手動記錄階段峰值

將 `plan-serving` 報告與多次實體 GPU 階段峰值比較：

```bash
python3 -m fakegpu calibrate observe-serving \
  build/speculative-serving-plan.json \
  --prefill-peak-bytes 24800000000,24900000000,25000000000 \
  --prefill-peak-bytes 24950000000,25050000000 \
  --decode-peak-bytes 25200000000,25300000000,25250000000 \
  --decode-peak-bytes 25350000000,25400000000 \
  --source torch.cuda.max_memory_reserved \
  --strict \
  --json build/serving-observation.json
```

此指令保留全部樣本，每個階段取最大值。預設每階段至少需要五次觀測，不足時
`--strict` 傳回 1。`ready_for_comparison` 只表示樣本數足夠，不代表
`GPU-validated`。Plan 必須設定 `--target-profile`。

### 自動採集多次執行

搭配框架對應的 runner：

```bash
python3 -m fakegpu calibrate collect-serving \
  build/speculative-serving-plan.json \
  --repetitions 5 \
  --timeout-seconds 900 \
  --strict \
  --json build/serving-observation.json \
  -- python3 benchmark_serving.py
```

同構、非 speculative 的 Transformers 工作負載可使用內建 runner。Plan 須使用
dynamic KV cache：

```bash
python3 -m fakegpu plan-serving \
  --model-dir /models/Qwen/Qwen3-0.6B \
  --active-sequences 4 \
  --max-batch-size 8 \
  --prompt-tokens 2048 \
  --generated-tokens 128 \
  --dtype bfloat16 \
  --attention-implementation sdpa \
  --kv-cache-strategy dynamic \
  --target-profile a100 \
  --json build/transformers-serving-plan.json

python3 -m fakegpu calibrate collect-serving \
  build/transformers-serving-plan.json \
  --repetitions 5 \
  --timeout-seconds 900 \
  --strict \
  --json build/transformers-serving-observation.json \
  -- python3 -m fakegpu calibrate sample-transformers
```

`sample-transformers` 會做這些事：

- 重新核對本機設定、Safetensors header 與 plan，再測量完整 prefill 和逐 token
  decode。
- 預設回報 `torch.cuda.max_memory_reserved`。要取張量配置峰值，請使用
  `--metric allocated`。
- 不接受混合請求、speculative decoding、paged 或 quantized cache、共享 prefix 或
  chunked prefill plan、量化 checkpoint 與 adapter。
- CUDA 環境中需要預先安裝 PyTorch 與 Transformers。

### 用自己的測量結果建構樣本

vLLM 或自訂測量程式已有可信的階段峰值時，可以直接建構樣本：

```bash
python3 -m fakegpu calibrate emit-serving-sample \
  build/vllm-serving-plan.json \
  --prefill-peak-bytes 24900000000 \
  --decode-peak-bytes 25300000000 \
  --metric nvml.process_family_peak_bytes \
  --framework vllm \
  --framework-version 0.10.2
```

Python 中對應的介面是 `fakegpu.build_cuda_serving_sample(...)`。vLLM 的 cache usage
指標是 block 使用比例，沒有位元組峰值，因此 benchmark 需要提供真實的階段位元組數。

### 樣本協定

每次重複都會啟動新程序，runner 需要輸出一個 `fakegpu.serving_peak_sample.v1`
物件。它可以放在最後一行，也可以放在以 `FAKEGPU_SERVING_SAMPLE=` 開頭的單行中：

```json
{
  "schema_version": "fakegpu.serving_peak_sample.v1",
  "workload_signature": "<FAKEGPU_SERVING_WORKLOAD_SIGNATURE>",
  "run_index": 1,
  "metric": "torch.cuda.max_memory_reserved",
  "phases": {
    "prefill": {"peak_bytes": 24900000000},
    "decode": {"peak_bytes": 25300000000}
  },
  "environment": {
    "backend": "cuda",
    "simulated": false,
    "gpu_name": "NVIDIA A100-SXM4-80GB",
    "gpu_uuid": "GPU-...",
    "compute_capability": [8, 0],
    "total_memory_bytes": 85899345920,
    "software": {
      "framework": "transformers",
      "framework_version": "...",
      "cuda_version": "...",
      "torch_version": "..."
    }
  }
}
```

- FakeGPU 透過 `FAKEGPU_SERVING_*` 環境變數提供 plan 路徑、工作負載簽章、目標
  profile、compute capability、目前次數與總次數。內建 runner 與協定建構器會自動
  產生這個物件，兩個框架都不會變成 FakeGPU 的相依套件。
- Runner 回報模擬 CUDA、容量超出 profile 2% 容差、compute capability 不符、
  軟體中繼資料不一致、逾時或非零退出時，採集會失敗。
- 報告不儲存原始指令參數，只記錄可執行檔名與指令指紋。
- `simulated: false` 由 runner 提供，不能當作硬體證明。聲明 `GPU-validated` 仍需
  保留日誌並通過校準門檻。

## 比較與門檻檢查

對任意相容的預測報告與觀測報告執行比較與門檻檢查：

```bash
python3 -m fakegpu calibrate compare \
  build/prediction.json \
  build/observation.json \
  --json build/calibration-comparison.json

python3 -m fakegpu calibrate verify \
  build/calibration-comparison.json \
  --max-underestimate-percent 5 \
  --max-absolute-percentage-error-percent 5 \
  --min-interval-coverage-percent 90 \
  --capacity-bytes 25769803776 \
  --json build/calibration-verification.json
```

`compare` 回報各階段的有號誤差與絕對誤差、區間涵蓋率與建議安全餘量。任一門檻未通過
時，`verify` 傳回 1。門檻包括最大低估、誤差中位數/95 百分位數/最大值、區間涵蓋率、
指定容量下的 false-safe 判斷，以及工作負載一致性。結果只適用於相同的工作負載簽章、
shape、dtype、軟體堆疊與 GPU profile。

## 階段記憶體校準

Allocated 張量儲存與 allocator reserved 記憶體分別統計。`calibrate phase-sample`
在一個全新 CUDA 程序中同時採集兩者。`phase-estimate` 以至少五個程序封存各階段
最大值，`phase-validate` 則對照之後的獨立執行。每份樣本與預算都使用新的輸出路徑。

```bash
# plan.json: a compatible local plan-serving or estimate-diffusion report
for i in 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=1 python -m fakegpu calibrate phase-sample plan.json \
    --json "/tmp/phase-cal-$i.json"
done
python -m fakegpu calibrate phase-estimate \
  --calibration-sample /tmp/phase-cal-1.json \
  --calibration-sample /tmp/phase-cal-2.json \
  --calibration-sample /tmp/phase-cal-3.json \
  --calibration-sample /tmp/phase-cal-4.json \
  --calibration-sample /tmp/phase-cal-5.json --json /tmp/phase-budget.json.gz
CUDA_VISIBLE_DEVICES=1 python -m fakegpu calibrate phase-sample plan.json \
  --json /tmp/phase-holdout.json
python -m fakegpu calibrate phase-validate --budget /tmp/phase-budget.json.gz \
  --holdout-sample /tmp/phase-holdout.json --max-absolute-error-percent 20 \
  --max-underestimate-percent 10 --json /tmp/phase-validation.json
```

- 採集支援同構 Transformers dynamic-cache serving，以及本機 SD/SDXL 的 eager 或
  SDPA attention、attention slicing、VAE slicing、VAE tiling 及其組合。不支援
  offload。
- Tiling 時，`--vae-tile-size` 同時設定原生 sample 與 latent tile 上限。實際的
  processor、VAE 開關與 tile 上限都納入原生身分。
- 計畫、選定的權重與設定、GPU UUID、Torch/CUDA/cuDNN 版本、allocator 與實作都須
  一致。不做形狀遷移，也不加隱含的安全係數。kernel cache 不受控。
- 預設門檻為絕對誤差 20%、低估 10%，未通過時傳回 1。離線估算與驗證只需要標準
  函式庫。

目前的證據仍屬 experimental，只適用於已測的 GPU、軟體與固定輸入：

- [階段記憶體驗收](../../tests/data/phase_memory_evidence.json)：補上 VAE 上採樣
  卷積後，四種解析度的 VAE allocated 誤差為 0.045%–1.172%。六個條件各五次校準、
  五次 holdout（共 60 個程序）皆通過兩項門檻。
- [擴展記憶體驗證](../../tests/data/memory_generalization_evidence.json)：25 個
  單卡條件、74 個程序，涵蓋 SD 1.4/SDXL、Qwen3-0.6B/1.7B/Qwen2.5-0.5B、
  FP16/BF16/FP32、不同 batch、矩形影像、guidance、eager attention、slicing 與
  tiling。allocated 最大絕對誤差 10.93%，最大低估 3.58%（門檻 20%/10%）。精確匹配
  條件的預算在十二次 holdout 中全部吻合，68 次跨條件重用被拒絕。這些資料沒有列為
  `GPU-validated`，原因是 21 個條件不足五次觀測，且最佳化模式使用的是封存的原生
  探針，沒有經過公開 CLI。
- [公開最佳化模式驗收](../../tests/data/optimized_phase_memory_evidence.json)：七個
  SD/SDXL 條件、70 個程序。210 項 holdout 比較皆與校準最大值一致，解析式 allocated
  最大絕對誤差 6.69%，最大低估 3.38%。42 次跨模式與 21 次實際模式身分不符被拒絕。

這些結果都不能證明可以遷移到新形狀、具備預測區間或保證實體 OOM 安全。
