<a id="readme-top"></a>

<div align="center">

# FakeGPU

**無需正式 GPU 叢集，即可驗證面向 CUDA 的應用程式、估算 GPU 記憶體並模擬分散式 GPU 工作流程。**

[![測試][test-shield]][test-url]
[![版本][release-shield]][release-url]
[![Python][python-shield]][python-url]
[![授權][license-shield]][license-url]

[English](README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md)

[回報問題](https://github.com/FanBB2333/FakeGPU/issues/new?labels=bug) ·
[提出功能建議](https://github.com/FanBB2333/FakeGPU/issues/new?labels=enhancement)

</div>

> [!IMPORTANT]
> FakeGPU 用於開發、相容性測試與容量規劃，無法讓任意 CUDA kernel 取得與實體
> GPU 相同的數值結果或效能。Passthrough、hybrid 與校準流程仍需要真實的
> CUDA 環境。

## 目錄

1. [專案介紹](#專案介紹)
   - [FakeGPU 能回答哪些問題](#fakegpu-能回答哪些問題)
   - [典型使用情境](#use-cases)
   - [實體 GPU 記憶體估算驗證](#memory-estimation-evidence)
   - [研究情境可靠性](#llm-reliability)
   - [運作方式](#運作方式)
   - [主要技術](#主要技術)
2. [快速開始](#快速開始)
   - [環境需求](#環境需求)
   - [安裝](#安裝)
   - [驗證安裝結果](#驗證安裝結果)
3. [使用方式](#使用方式)
   - [使用 FakeCUDA 執行 PyTorch](#使用-fakecuda-執行-pytorch)
   - [查看 FakeGPU 裝置與程序](#查看-fakegpu-裝置與程序)
   - [匯出監控指標](#export-monitoring-metrics)
   - [攔截原生 CUDA 函式庫](#攔截原生-cuda-函式庫)
   - [執行前檢查 GPU 記憶體](#執行前檢查-gpu-記憶體)
   - [規劃線上 LLM 服務容量](#plan-online-llm-serving-capacity)
   - [分析儲存庫或模型](#分析儲存庫或模型)
4. [指令參考](#指令參考)
5. [GPU profiles](#gpu-profiles)
6. [開發](#開發)
   - [建置](#建置)
   - [測試](#測試)
   - [可重複使用的腳本](#可重複使用的腳本)
7. [專案結構](#專案結構)
8. [限制](#限制)
9. [開發計畫](#開發計畫)
10. [參與貢獻](#參與貢獻)
11. [授權](#授權)
12. [致謝](#致謝)

## 專案介紹

FakeGPU 為開發、CI、相容性檢查與容量規劃模擬面向 CUDA 的執行環境。應用程式
可以偵測可設定的 NVIDIA 風格裝置；已維護的運算會在 CPU 上執行；模擬的 GPU
記憶體與通訊資料會被記錄。對於不應直接載入的工作負載，FakeGPU 也提供靜態
估算工具。

模擬與分析功能不需要實體 GPU。只有 passthrough、hybrid 與校準流程需要相容的
真實 CUDA 環境。

### FakeGPU 能回答哪些問題

| 問題 | 建議入口 | 需要實體 GPU |
|---|---|---:|
| PyTorch 程式碼能否依預期執行面向 CUDA 的控制流程？ | Python FakeCUDA runtime | 否 |
| 未修改的程序能否載入並呼叫 CUDA 系列動態函式庫？ | 原生函式庫攔截 | 否 |
| 某個工作負載能否放入選定的 GPU profile？ | Preflight 或靜態 GPU 記憶體估算器 | 否 |
| LLM 的 checkpoint、KV cache、adapter 或 MoE 需要多少 GPU 記憶體？ | LLM 估算器 | 否 |
| 給定 GPU 記憶體預算可以容納哪些同構或混合長度的線上 LLM 請求？ | Serving 規劃器 | 否 |
| Speculative decoding 的 target 與 draft 模型能否同時放入 GPU 記憶體？ | Serving 規劃器 | 否 |
| 解析度、batch、CFG、VAE tiling 與 offload 會如何影響 diffusion 生成所需的 GPU 記憶體？ | Diffusion 估算器 | 否 |
| 儲存庫中有哪些僅支援 GPU 的入口與相依套件？ | 儲存庫分析器 | 否 |
| 分散式訓練設定對應多少單 rank GPU 記憶體？ | 訓練規劃器 | 否 |
| Trace 中的運算、通訊、等待與 GPU 記憶體如何重疊？ | Trace 重播 | 否 |
| 估算結果與真實 CUDA 執行結果相差多少？ | Passthrough 或 hybrid 校準 | 是 |

<a id="use-cases"></a>

### 典型使用情境

| 適用情境 | FakeGPU 提供的能力 | 建議入口 |
|---|---|---|
| 在租用 GPU 或啟動長時間工作前選擇硬體 | 依 profile 估算 checkpoint、KV cache、activation、optimizer 與 workspace 的 GPU 記憶體 | `estimate-llm`、`preflight` |
| 部署 chat、RAG、completion、summarization 或 draft-assisted 服務前規劃容量 | 依請求估算 prompt/生成長度、continuous batch 準入、chunked prefill 暫存記憶體、共享 prefix 群組、speculative target/draft 記憶體與可用記憶體餘量 | `plan-serving`、`validate` |
| 在筆記型電腦或無 GPU 的 CI 中開發面向 CUDA 的 PyTorch 程式碼 | 讓程式看到 CUDA 裝置，同時在 CPU 上執行已維護的 tensor 運算 | `fakegpu.init(...)`、`demo`、`validate` |
| 比較完整參數微調、LoRA、QLoRA、checkpointing、offload 或分片方案 | 在配置叢集前估算各階段與單 rank GPU 記憶體 | `plan-training`、Python GPU 記憶體估算器 |
| 比較 UNet 與 diffusion Transformer 的生成 shape 和 GPU 記憶體最佳化選項 | 根據固定 profile 或本機 pipeline，依架構估算 text encoder、denoising 與 VAE decode 階段 | `estimate-diffusion`、`validate` |
| 檢查不熟悉的 GPU 儲存庫或原生擴充 | 統計 GPU 入口、相依套件、kernel 與不支援的 API | `analyze-repo`、`analyze-kernel`、`capabilities` |
| 設計或診斷分散式工作流程 | 分析 collective 路由、鏈路競爭、rank 等待、GPU 記憶體時間軸與 TCP payload | `simulate-topology`、`replay-trace`、`bandwidth` |
| 在 CI 或本機實驗環境中觀察模擬裝置與程序 | 提供有數量上限的 Prometheus 指標、exporter 健康狀態與短期記憶體歷史 | `nvidia-smi`、`metrics` |
| 將小規模實體 GPU 試驗用於後續重複工作 | 產生預測值與實測值的比較報告，並依工作負載簽章保存校準資料 | `calibrate`、`preflight --memory-calibration` |

<a id="memory-estimation-evidence"></a>

### 實體 GPU 記憶體估算驗證

> [!NOTE]
> 在以下已記錄的驗證範圍內，使用對應軟體堆疊校準後的靜態估算器在 26 個受控
> GPU 觀測上的誤差不超過 **0.08%**；十個 Qwen 完整參數/LoRA SFT 案例的
> 誤差不超過 **1.921%**。

絕對百分比誤差依
`|預測值 - 實測值| / 實測值 × 100%` 計算。表中的「一致度」等於
`100% - 誤差`，只是同一測量結果的直觀表示。

| 已驗證範圍 | 實體 GPU 參考環境 | 證據規模 | 絕對百分比誤差 | 一致度 |
|---|---|---:|---:|---:|
| [包含 backend 常駐 GPU 記憶體校準的受控 ATen MLP 與 Transformer 參數網格][validation-static] | Ampere GPU 與 Blackwell GPU；PyTorch/CUDA 2.12/13.0 和 2.9/12.8 | 13 個工作負載，26 個觀測 | **最大 0.08%** | **≥99.92%** |
| [Qwen3-8B BF16 SDPA 推論][validation-inference] | Blackwell GPU；PyTorch 2.9.1/CUDA 12.8 | 模型載入與推論峰值 | 載入 0.0129%；**峰值 0.0672%** | 99.9871%；**99.9328%** |
| [Qwen 0.8B/2B 完整參數與 LoRA SFT][validation-sft] | Blackwell GPU；PyTorch 2.8/CUDA 12.8 | 10 個訓練案例 | **0.102%–1.921%** | **98.079%–99.898%** |
| [Qwen 0.8B/2B 原生 NF4 QLoRA][validation-qlora] | Blackwell GPU；PyTorch 2.8/CUDA 12.8 | 10 個量化訓練案例 | **0.628%–1.732%** | **98.268%–99.372%** |

這些數字需要配合測量方式理解：

- Qwen 資料以 `torch.cuda.max_memory_allocated()` 為參考，不包含 CUDA context
  與 allocator 已保留但尚未使用的 GPU 記憶體。
- 受控 ATen 資料加入目前 GPU 與軟體堆疊的 backend 常駐 GPU 記憶體測量值，
  不能將該值用於其他環境。
- 區間表示所有案例中的最小與最大誤差，不是平均值。評估 OOM 風險時應特別注意
  最大低估值。
- `99.x%` 一致度不表示還有相同比例的可用 GPU 記憶體。容量規劃仍應加入與
  工作負載對應的安全餘量或係數。

這些數字來自固定工作負載，不能作為任意情境的通用準確率。模型、shape、
attention backend、量化 kernel、allocator、PyTorch/CUDA 版本或 GPU 改變時，
需要重新校準。表中連結指向固定版本的驗證快照，其中保留完整設定與實測位元組數。
CI 也會檢查目前儲存庫中的
[結構化證據摘要](https://github.com/FanBB2333/FakeGPU/blob/main/tests/data/memory_validation_evidence.json)
與 README 是否一致。

在真實 CUDA 主機上，可以重新執行專案維護的受控比較：

```bash
python3 scripts/validation/static_memory_validation.py \
  --output build/static-memory-validation.json \
  --markdown build/static-memory-validation.md \
  --max-underestimate-percent 5
```

在無 GPU 主機上加入 `--static-only` 可以檢查估算流程，但不會產生實體 GPU
準確性結果。對於 `plan-serving` 報告，可先將多次實體 GPU 階段峰值整理為標準
觀測報告：

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

此命令保留全部樣本，並採用每個階段的最大值。預設要求每個階段至少五次觀測；
樣本不足時，`--strict` 傳回狀態碼 1。`ready_for_comparison` 只表示樣本數量符合
要求，不代表結果已經是 `GPU-validated`。Serving plan 必須包含
`--target-profile`，觀測報告才能記錄一致的 GPU 身分。

需要自動執行多次測量時，可以接入與框架對應的 runner：

```bash
python3 -m fakegpu calibrate collect-serving \
  build/speculative-serving-plan.json \
  --repetitions 5 \
  --timeout-seconds 900 \
  --strict \
  --json build/serving-observation.json \
  -- python3 benchmark_serving.py
```

對於同質、非 speculative 的 Transformers 工作負載，FakeGPU 已內建 runner，
不需要另外維護 benchmark 檔案。Plan 必須採用相同的執行參數，並使用 dynamic
KV cache：

```bash
python3 -m fakegpu plan-serving \
  --model-dir /mnt/z/models/Qwen/Qwen3-0.6B \
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

模型路徑保留 Hugging Face 的 `owner/repository` 兩層結構。主機沒有
`/mnt/z` 時，只需替換 models 根目錄，並保留 `Qwen/Qwen3-0.6B`。

`sample-transformers` 會重新核對本機模型設定、Safetensors header 與 plan，
接著測量完整 prefill 和重用 KV cache 的逐 token decode。預設指標是
`torch.cuda.max_memory_reserved`；如需張量配置峰值，可使用
`--metric allocated`。目前不接受混合請求集、speculative decoding、paged 或
quantized cache、共享前綴、chunked prefill、量化 checkpoint 和 adapter，因為
這些執行路徑還沒有語意一致的測量實作。PyTorch 與 Transformers 仍是選用相依，
需要預先安裝在 CUDA benchmark 環境中。

如果 vLLM 或自訂測量程式已取得可信的階段峰值，可以用協定建構器補齊 CUDA
環境資訊：

```bash
python3 -m fakegpu calibrate emit-serving-sample \
  build/vllm-serving-plan.json \
  --prefill-peak-bytes 24900000000 \
  --decode-peak-bytes 25300000000 \
  --metric nvml.process_family_peak_bytes \
  --framework vllm \
  --framework-version 0.10.2
```

Python benchmark 也可以呼叫 `fakegpu.build_cuda_serving_sample(...)`。vLLM 會按
設定的 GPU 預算預留 engine 與 KV cache 記憶體，而 cache usage 指標表示 block
使用比例，不是程序記憶體位元組峰值。因此，vLLM benchmark 需要提供真實的階段
位元組數，FakeGPU 不會把 cache usage 標記成 prefill 或 decode 記憶體。

採集器每次都會啟動新的子程序。Runner 需要輸出一個
`fakegpu.serving_peak_sample.v1` JSON 物件。可以將它作為最後一行輸出；
如果框架也會向 stdout 寫入日誌，則使用 `FAKEGPU_SERVING_SAMPLE=`
作為單行 JSON 的前綴：

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

FakeGPU 會透過 `FAKEGPU_SERVING_*` 環境變數提供 plan 路徑、工作負載
簽章、目標 profile、compute capability、目前次數與總次數。這套協定可以
包裝 Transformers、vLLM 或自訂服務，無需將這些大型依賴加入 FakeGPU。
內建 Transformers runner 與 vLLM/自訂協定建構器會自動產生該物件。
Runner 報告模擬 CUDA、GPU 容量超出 profile 的 2% 容差、compute capability
不符、軟體環境變動、逾時或非零退出時，採集會失敗。報告會記錄實際容量
與 profile 容量的差值。報告不儲存原始命令參數，只記錄
可執行檔名與命令指紋。`simulated: false` 仍是 runner 提供的中繼資料，
不是獨立的硬體證明；聲明 `GPU-validated` 仍需保留 benchmark 日誌並通過校準
門檻檢查。

對於任意工作負載，可以對相容的預測報告與觀測報告執行：

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

比較報告包含各階段的有號誤差、絕對誤差、預測區間涵蓋情形，以及建議的 GPU
記憶體安全餘量與係數。`calibrate verify` 會檢查最大低估、絕對百分比誤差的
中位數/95 百分位數/最大值、預測區間涵蓋率、指定容量下的 false-safe 判斷，
以及工作負載參數是否一致；任一門檻未通過時傳回狀態碼 1。結果只適用於相同的
工作負載簽章、shape、dtype、軟體堆疊與 GPU profile。

<a id="llm-reliability"></a>

### 研究情境可靠性報告

FakeGPU 依工作負載與環境簽章報告可靠性。`GPU-validated` 結果只適用於記錄中的
模型 revision、shape、dtype、attention backend、allocator、軟體堆疊與 GPU。
`CPU-validated` 表示已維護的執行或分析行為在無實體 GPU 的環境中通過驗證。
`Modeled` 表示已有分析模型，但沒有對應的實體 GPU 資料；`Planned` 表示尚未
形成可公開聲明的準確率。
Serving 規劃報告提供與路徑無關的 `workload_signature`，避免校準資料在不同
服務設定之間被誤用。
可使用 `calibrate observe-serving` 保存重複的階段觀測，再執行比較與可靠性門檻檢查。
需要重複採集真實 CUDA 資料時，可將 `calibrate collect-serving` 與內建
`sample-transformers` runner 配合使用；vLLM 或自訂測量結果可先由
`build_cuda_serving_sample` 轉成統一協定。

#### 儲存庫驗證快照（2026-08-05）

目前儲存庫狀態於 2026-08-05 在 macOS 26.5 arm64、Python 3.11.9 與 PyTorch
2.9.1 CPU 環境中執行了 `scripts/test.sh` 的全部測試群組與兩個宣告式驗證
manifest。Transformers 取樣適配器也在 Blackwell GPU、PyTorch
2.11.0+cu128 與 Transformers 5.14.1 環境中完成實體 CUDA 驗證：

| 驗證層 | 已維護的檢查 | 結果 |
|---|---|---|
| Python runtime、估算器、CLI、schema 與 README 契約 | 完整 `pytest` 測試集 | **223 個通過** |
| 宣告式驗證矩陣 | 7 個 smoke 案例，加 39 個 research 架構、cache、generic/vLLM serving、訓練、校準與 diffusion 案例 | **46 個通過** |
| 原生函式庫攔截 | 建置、函式庫邊界、匯出符號、preload、GPU 記憶體類型、coordinator 與不支援 API 策略 | **通過** |
| FakeGPU-SMI 診斷 | 有上限的狀態發布、拓撲/NVLink/MIG 檢視、NVML peer/MIG 查詢、健康欄位與事件報告 | **通過** |
| 監控 exporter | Prometheus/JSON 快照、歷史與序列數量限制、異常狀態降級及 HTTP 介面 | **通過** |
| 原生能力清單 | 5 個能力群組、26 個明確 API、24 個強制執行策略的 API | **通過** |
| GPU profile 目錄 | 82 個 profile，涵蓋 15 種 compute capability | **通過** |
| CPU 數值模擬 | GEMM、cuBLASLt、批次 GEMM、BLAS1/2 與 FP16，共 8 組測試 | **通過** |
| CUDA 版 PyTorch 原生矩陣乘法 | 需要含 CUDA 的 PyTorch | **目前 CPU-only 主機未執行** |
| Transformers 實體 CUDA serving 取樣器 | Qwen/Qwen3-0.6B、BF16 SDPA、2 × 128-token prefill + 8 個生成 token，每階段 3 個獨立樣本 | **通過；`ready_for_comparison`** |

GitHub CI 也會在 Python 3.10–3.12 上執行 Python 測試，並在 Linux 與 macOS
上執行原生 smoke 與 CPU simulation。上文已發布的準確率來自對應的固定版本
驗證快照。新增的 Transformers 結果驗證了重複執行、CUDA 身分與報告採集；
由於尚未通過預測值與實測值的比較門檻，它不構成新的準確率聲明。

#### 已維護的研究工作負載矩陣

| 工作負載類型 | 已涵蓋變化 | 驗證依據 | 狀態 |
|---|---|---|---|
| 離線 decoder 推論 | Qwen3-8B、BF16、SDPA、模型載入、prefill 與 decode 峰值 | Blackwell GPU 預測值與實測值 | `GPU-validated` |
| 完整參數與 adapter SFT | Qwen 0.8B/2B 完整參數微調與 LoRA | 十個 Blackwell GPU 訓練案例 | `GPU-validated` |
| 量化 adapter SFT | Qwen 0.8B/2B 原生 NF4 QLoRA | 十個 Blackwell GPU 訓練案例 | `GPU-validated` |
| 通用 decoder 分析 | Dense/MoE 中繼資料；MHA、GQA、MQA 與壓縮 latent MLA；adapter、量化 checkpoint、eager/SDPA attention、KV cache 與 expert-parallel 通訊量 | 公式、fixture、CLI 迴歸測試與四種架構矩陣 | `CPU-validated` + `Modeled` |
| 分散式訓練規劃 | DeepSpeed、Accelerate、FSDP/FSDP2、分片、checkpointing 與 CPU/NVMe offload | 設定、位元組計算、拓撲與 trace 測試 | `CPU-validated` + `Modeled` |
| KV cache 配置 | Dynamic 增長、static 預留、2/4/8-bit quantized 儲存、paged block 取整與 sliding-window 上限 | 公式、API、`--kv-cache-strategy` CLI 與 `tests/data/research_validation.yaml` 矩陣測試 | `CPU-validated` + `Modeled` |
| 線上 serving 容量 | 面向同構與混合長度請求集的 continuous batching、依序準入、chunked prefill、群組 prefix caching、paged KV block、generic 或 vLLM 執行期預算、profile 或明確 KV 容量，以及準入餘量 | Checkpoint header、架構與 block 配置公式、`plan-serving --runtime vllm`、`--vllm-kv-cache-memory-bytes`、`--vllm-non-kv-cache-memory-bytes`、單元測試，以及 `tests/data/research_validation.yaml` 中的 generic/vLLM chat/RAG 案例 | `CPU-validated` + `Modeled` |
| Diffusion 圖像生成 | Stable Diffusion v1.5、SDXL Base 與 PixArt-Sigma；本機 UNet、交叉注意力 DiT、SD3/Flux 類聯合注意力 Transformer；CFG、offload、attention/VAE slicing 與 VAE tiling | 固定版本與本機元件 header、架構設定、階段公式、CLI/單元測試及 `tests/data/research_validation.yaml` 中的三種本機架構矩陣 | `CPU-validated` + `Modeled` |
| Speculative decoding | 獨立自迴歸 draft 模型權重與 KV cache、target lookahead KV、批次 verification 暫存記憶體、可選的固定接受率假設，以及同構/混合請求準入 | Checkpoint header、架構公式、`plan-serving --draft-model-dir`、`--speculative-tokens`、`--speculative-acceptance-rate` 與單元測試；尚無專案維護的實體 GPU 觀測 | `CPU-validated` + `Modeled` |
| 多 GPU LLM 執行 | TP、PP、CP、EP、MoE 負載不均衡，以及 FSDP/ZeRO 組合執行 | 目前只有分析拓撲與 coordinator 驗證 | `Modeled` |
| Diffusion 訓練 | Optimizer、gradient、activation、EMA 與參數高效微調 | 尚無專用估算器或實體 GPU 資料 | `Planned` |

<a id="research-scenario-effects"></a>

#### 架構感知估算比較

先前的「可重現的建模效果」只表示固定輸入能得到相同的公式結果，並不表示
估算值已接近實體 GPU 觀測。為避免混淆，本節改為「架構感知估算比較」。
下表由儲存庫中的公式產生，並由 README 契約測試檢查；GiB 使用二進位單位。
它用於檢查架構、shape 與最佳化選項造成的變化，不能取代相同設定的實體 GPU
校準。

| Research 情境 | 對照條件 | 建模結果 | 驗證方式 |
|---|---|---:|---|
| LLM 推論 KV cache | 32 層 GQA、8 個 KV head、head dim 128、batch 1、BF16；context 4K → 32K | **0.50 → 4.00 GiB**（cache 增加 8 倍） | 精確位元組公式 |
| Attention cache 架構 | 32 層、8 個活躍請求、4K BF16 paged cache；標準 head dim 128；MHA 32 個 KV head / GQA 8 個 / MQA 1 個 / MLA latent 512 + RoPE 64 | **16.00 / 4.00 / 0.50 / 1.125 GiB** | 對應架構的 cache layout 公式與四種 manifest 案例 |
| 線上 serving prefix cache | 相同 decoder、8 個活躍請求、4K prompt + 256 個生成 token、paged BF16 KV；獨立 cache → 共享 1K prefix | **4.25 → 3.38 GiB**（減少 20.6%） | 共享/私有 paged block 公式 |
| 混合 chat/completion serving | 相同 decoder；prompt 為 4K/8K/2K，生成長度為 256/512/1024；獨立 KV → 兩個 chat 請求共享 1K system prefix | **1.97 → 1.84 GiB** decode KV（減少 6.3%） | 異構請求集公式與 chat/RAG 矩陣 |
| Speculative target 呼叫 | 每次 verification 生成 5 個 draft token，並假設單 token 獨立接受率為 70% | 每個 target step 的預期輸出 token 為 **1.00 → 2.94**；此數值不表示吞吐提升 | 幾何接受率公式與雙模型記憶體測試 |
| LLM 訓練 | 8B BF16 參數、AdamW、4 張 GPU、activation checkpointing；replicated → full shard | **104.31 → 26.54 GiB**/rank（減少 74.6%） | 訓練階段模型 |
| Stable Diffusion v1.5 生成 | 512²、batch 1、FP16、CFG；所有權重常駐 → model offload | **2.30 → 1.65 GiB**（減少 28.1%） | 元件與階段模型 |
| SDXL 批次生成 | 1024²、batch 4、FP16、CFG；一般 VAE decode → VAE slicing | **11.49 → 7.74 GiB**（減少 32.7%） | 依序 VAE batch shape 模型 |
| SDXL 高解析度生成 | 2048²、batch 1、FP16、CFG；完整影像 VAE → 512² VAE tile | **11.49 → 7.27 GiB**（減少 36.7%） | 依序 VAE tile shape 模型 |
| PixArt-Sigma DiT 生成 | 1024²、batch 1、FP16、CFG、model offload；eager → SDPA 的 denoise 階段 | **2.32 → 1.28 GiB**（減少 44.7%） | patch token、交叉注意力與階段模型 |

Diffusion profile 中的元件參數量來自
[Stable Diffusion v1.5][diffusion-sd15] 與
[SDXL Base 1.0][diffusion-sdxl]、[PixArt-Sigma XL 2][diffusion-pixart]
的固定版本。上述數字不包含 runtime context、allocator 碎片、backend 私有
workspace 與傳輸重疊。
在取得匹配觀測值之前，架構感知報告保持為 `Modeled`，`accuracy.status`
為 `uncalibrated`。
本機架構檢查透過 `estimate-diffusion --model-dir` 使用。

Cache 公式參考
[Transformers cache strategies](https://huggingface.co/docs/transformers/kv_cache)
提供的工作負載形態。Serving 規劃器支援同構 shape 與依清單排列的異構活躍
請求集。Speculative 模式依照
[原始 speculative decoding 論文](https://arxiv.org/abs/2211.17192)與
[Transformers assisted decoding](https://huggingface.co/docs/transformers/assisted_decoding)描述的獨立
draft/target 流程建模：較小的模型提出多個 token，target 在一次 forward 中完成
驗證。請求到達分布、cache eviction、preemption、自適應 draft 長度、KV 回復
workspace、端到端吞吐量與延遲不在目前估算範圍內。相關術語參考
[vLLM serving](https://docs.vllm.ai/en/stable/)。CPU FakeCUDA 不執行二進位
CUDA 擴充或任意 kernel；這類工作負載需要分析結果，並透過 passthrough 或
hybrid 模式取得實體 GPU 觀測值。

後續新增或更新的公開驗證資料應包含：

- 至少五次獨立執行，以及其中最大的觀測峰值；
- 每個報告階段的預測位元組數與實測位元組數；
- 將最大低估作為主要 OOM 風險指標；標記為 `GPU-validated` 時，建議不超過
  5%；
- 絕對百分比誤差的中位數、95 分位數與最大值；
- 預測區間涵蓋率，以及 FakeGPU 判斷可執行但實際工作負載發生 OOM 的
  false-safe 次數；
- 模型 revision、完整指令、shape、dtype、backend、allocator 設定、GPU、
  driver、CUDA、PyTorch 與框架版本。

公開結果前可使用 `calibrate verify` 對機器可讀的比較報告執行這些門檻檢查。

未達到上述目標的資料繼續標記為 `Modeled` 或 experimental，不作為已驗證
準確率展示。「一致度」只作為輔助資訊，最大低估與 false-safe OOM 判斷更能反映
容量規劃風險。

### 運作方式

| 路徑 | 應用程式看到的內容 | 實際執行方式 |
|---|---|---|
| **Python FakeCUDA** | CUDA 裝置、CUDA 風格 tensor、GPU 記憶體 API 與常見訓練流程 | 已維護的 PyTorch 運算透過 `FakeCudaTensor` 在 CPU 上執行 |
| **原生函式庫攔截** | `libcuda`、`libcudart`、`libcublas`、`libnvidia-ml` 與 `libnccl` 入口 | 選定的運算使用主機記憶體或 CPU 計算；不支援的行為會被分類並寫入報告 |
| **分析與報告** | GPU 記憶體、FLOP、Roofline、拓撲與通訊報告 | 分析 ATen 圖、safetensors 中繼資料、執行 trace、校準資料與 coordinator 事件 |

### 主要技術

- [Python](https://www.python.org/) 3.10+：runtime、估算器、CLI 與報告
- C++17 與 [CMake](https://cmake.org/)：原生攔截函式庫與 coordinator
- [PyTorch](https://pytorch.org/)：CPU FakeCUDA 執行與 ATen 圖擷取
- YAML 與 JSON Schema：GPU profiles、驗證 manifest 與報告

<p align="right">(<a href="#readme-top">返回頂端</a>)</p>

## 快速開始

### 環境需求

- Linux 或 macOS
- Python 3.10 或更新版本
- CMake 3.14 或更新版本
- 支援 C++17 的編譯器
- Python FakeCUDA runtime 需要 PyTorch

Debian 或 Ubuntu 可安裝 `build-essential`，macOS 可安裝 Xcode Command Line
Tools。

### 安裝

複製儲存庫：

```bash
git clone https://github.com/FanBB2333/FakeGPU.git
cd FakeGPU
```

建置原生函式庫並安裝 Python 套件：

```bash
scripts/build.sh
FAKEGPU_BUILD_DIR="$PWD/build" python3 -m pip install .
```

直接從原始碼目錄開發：

```bash
python3 -m pip install pytest PyYAML jsonschema ruff
export PYTHONPATH="$PWD"
```

### 驗證安裝結果

```bash
python3 -m fakegpu doctor --list-profiles
python3 -m fakegpu demo --profile l4
```

`doctor` 檢查 profile 目錄、原生函式庫與 PyTorch 環境。`demo` 在 CPU 上完成
一個小型 forward、backward 與 optimizer step，同時讓程式看到 CUDA 裝置。

<p align="right">(<a href="#readme-top">返回頂端</a>)</p>

## 使用方式

### 使用 FakeCUDA 執行 PyTorch

請在匯入 PyTorch 前初始化 FakeGPU：

```python
import fakegpu

fakegpu.init(runtime="fakecuda", profile="a100", device_count=2)

import torch

device = torch.device("cuda:0")
model = torch.nn.Linear(8, 4).to(device)
x = torch.randn(2, 8, device=device)
loss = model(x).square().mean()
loss.backward()

print(torch.cuda.device_count())      # 2
print(torch.cuda.get_device_name(0))  # NVIDIA A100
print(loss.item())
```

已維護的運算會在 CPU 上執行，裝置放置、GPU 記憶體限制、訓練控制流程與錯誤
處理則使用模擬的 CUDA 介面。

### 查看 FakeGPU 裝置與程序

狀態發布功能需要明確啟用。請在工作負載呼叫 `fakegpu.init(...)` 或透過原生啟動器
執行前設定狀態目錄；`build/` 已被 Git 忽略：

```bash
export FAKEGPU_SMI_STATE_DIR=build/smi

# Python FakeCUDA 工作負載。
python3 your_script.py

# 未修改的原生 CUDA/NVML 工作負載。
python3 -m fakegpu --build-dir build ./your_native_workload
```

在另一個終端中查看正在執行的工作負載：

```bash
# 精簡的裝置與程序表；加入 "-l 1" 可每秒更新一次。
python3 -m fakegpu nvidia-smi --state-dir build/smi

# 裝置清單，以及 runtime、profile 與 allocator 詳細報告。
python3 -m fakegpu nvidia-smi --state-dir build/smi -L
python3 -m fakegpu nvidia-smi --state-dir build/smi -q

# 接近 NVIDIA 指令格式的建模拓撲與 NVLink 檢視。
python3 -m fakegpu nvidia-smi topo -m --state-dir build/smi
python3 -m fakegpu nvidia-smi nvlink -s --state-dir build/smi

# 建模故障、相容性警告、發布失敗與過期狀態。
python3 -m fakegpu nvidia-smi events --state-dir build/smi

# 建模 MIG GPU Instance 與 Compute Instance。
python3 -m fakegpu nvidia-smi mig -lgi --state-dir build/smi
python3 -m fakegpu nvidia-smi mig -lci --state-dir build/smi

# 適合腳本處理的 GPU 與程序查詢。
python3 -m fakegpu nvidia-smi --state-dir build/smi \
  --query-gpu=index,name,uuid,pci.bus_id,profile.id,compute_cap,memory.total,memory.used,memory.free,allocator.model,native.kernel_launches,native.gemm_calls,native.io_bytes \
  --format=csv
python3 -m fakegpu nvidia-smi --state-dir build/smi \
  --query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory,peak_gpu_memory,stage,status \
  --format=csv,noheader,nounits
python3 -m fakegpu nvidia-smi --state-dir build/smi \
  --query-runtime=pid,fakegpu.version,runtime.backend,runtime.mode,policy.oom,tracking.dispatch,catalog.profiles,catalog.native_apis,dispatch.calls,publisher.failed_writes,status \
  --format=json
```

詳細報告包含 FakeGPU 版本、runtime backend 與策略、Python/PyTorch/CUDA 版本、
狀態更新時間、profile 目錄與原生 API 涵蓋情形、模擬裝置識別、運算規格、GPU
記憶體分類、allocator 活動、dispatch 追蹤及各程序峰值。`-i` 可以依裝置索引、
UUID、PCI Bus ID 或 profile ID 篩選；`--json` 會輸出完整的標準化資料。目前
發布 state schema v2，同時仍可讀取 v1 狀態檔。
`--query-runtime` 可將相同的執行期、策略、軟體、追蹤、catalog、dispatch 與
publisher 健康資訊輸出為便於程式處理的 CSV 或 JSON。使用
`--help-query-runtime` 可查看全部欄位。

原生攔截也會發布 allocation 生命週期、傳輸量、kernel launch、GEMM 呼叫與
FLOP、相容性事件及 unsupported API 次數。程序執行時會定期更新狀態，結束時會
將狀態標記為 exited。`FAKEGPU_SMI_DETAIL_LIMIT` 用於限制保留的明細數量，
`FAKEGPU_SMI_MAX_STATE_BYTES` 用於限制單一狀態檔大小；`-q` 會顯示發布次數、
失敗次數、耗時與序列化大小。

NVLink 模型預設關閉。在工作負載啟動前設定
`FAKEGPU_NVLINK_GROUPS="0,1;2,3"`，可為分號分隔的每組裝置建立全連接關係。
`FAKEGPU_NVLINK_BANDWIDTH_GBPS` 是選用的頻寬參數，預設值為 `900`。狀態檔、
`topo -m`、`nvlink -s`、`topology.*`/`nvlink.*` 查詢欄位，以及原生 NVML
的鏈路狀態、capability、遠端裝置類型與遠端 PCI 查詢共用同一模型。設定無效時，
所有鏈路都會保持 inactive，狀態中會顯示設定錯誤。

MIG 模型預設關閉。請在工作負載啟動前使用
`FAKEGPU_MIG_LAYOUT="DEVICE:PROFILE:MEMORY_MIB[:COUNT];..."` 設定實例，例如
`FAKEGPU_MIG_LAYOUT="0:1g.10gb:10240:2;1:2g.20gb:20480"`。每個建模 GPU
Instance 對應一個 Compute Instance。狀態檔、`-L`、`-q`、`mig -lgi`、
`mig -lci`、`mig.*` 查詢欄位，以及原生 NVML 的 MIG mode、handle、UUID、
父裝置、實例 ID 與 GPU 記憶體容量查詢共用同一設定。單一裝置最多包含 8 個
實例與 8 個 slice，實例 GPU 記憶體總量不能超過父裝置；無效設定不會建立部分
實例。目前還無法將 runtime allocation 歸屬到特定 MIG 實例，因此狀態檔會將
實例 used/free GPU 記憶體標示為 `unobserved`。

故障注入預設同樣關閉。設定格式為
`FAKEGPU_FAULT_EVENTS="DEVICE:CODE:SEVERITY[:COUNT];..."`，例如
`FAKEGPU_FAULT_EVENTS="0:XID_79:critical;1:NVLINK_CRC:error:3"`。CODE 僅作為
標籤，嚴重度支援 `info`、`warning`、`error` 與 `critical`。`events` 檢視會
彙整設定的故障事件、原生 unsupported API 呼叫、狀態發布失敗、過期狀態與模型
設定錯誤。`health.*` 查詢欄位會顯示每個裝置的建模狀態、最高嚴重度、事件數，
並明確標註硬體健康狀態無法觀測。輸入無效時不會啟用任何故障，只會回報設定錯誤。

UUID 與 PCI Bus ID 是穩定的模擬識別。CPU runtime 無法觀測溫度、風扇轉速、
即時功耗與硬體 GPU 使用率，因此這些欄位顯示為 `N/A`；profile 功耗與時脈會
作為靜態規格分開顯示。拓撲關係、設定頻寬、故障代碼與健康狀態屬於建模輸入，
不是硬體測量結果，也不是觀測到的 ECC/Xid 資料。

<a id="export-monitoring-metrics"></a>

### 匯出監控指標

FakeGPU-SMI 的標準化狀態無需安裝監控相依套件即可匯出。單次指令預設輸出
Prometheus 文字，也可以輸出標準化 JSON 快照：

```bash
python3 -m fakegpu metrics --state-dir build/smi
python3 -m fakegpu metrics --state-dir build/smi --json
```

需要本機持續收集時，可啟動帶記憶體歷史上限的 exporter：

```bash
python3 -m fakegpu metrics --state-dir build/smi --serve \
  --host 127.0.0.1 --port 9400 --interval 1 \
  --history-size 300 --max-process-series 128

curl http://127.0.0.1:9400/metrics
curl http://127.0.0.1:9400/healthz
curl http://127.0.0.1:9400/api/v1/history
```

`/metrics` 提供最新的裝置、程序、runtime、MIG、拓撲、健康狀態與發布器指標；
`/healthz` 顯示資料來源與收集狀態；`/api/v1/history` 傳回近期的標準化樣本。
程序序列優先保留 GPU 記憶體使用量最高的程序，預設上限為 128，最大為 256；
`--max-process-series 0` 可以關閉程序序列。歷史預設保留 300 個樣本，最多
1,440 個。這些限制都在記憶體中執行，監控歷史不會寫入磁碟，也不會由 Git
管理。如需長期保留，請使用 Prometheus 或其他收集系統。

服務預設只監聽 `127.0.0.1`，本身不提供身分驗證。繫結非本機位址前，應在前方
設定帶身分驗證的 proxy 或網路存取策略。匯出的資料沿用 FakeGPU-SMI 對「建模值」
和「觀測值」的區分，不會把無法取得的硬體遙測資料表示為真實測量結果。

### 攔截原生 CUDA 函式庫

建置原生函式庫後，使用模組啟動器為未修改的指令設定 `LD_PRELOAD` 或
`DYLD_INSERT_LIBRARIES`：

```bash
python3 -m fakegpu --build-dir build --profile a100 \
  python3 your_script.py

python3 -m fakegpu --build-dir build --devices "a100:2,h100:2" \
  python3 your_script.py

python3 -m fakegpu --build-dir build \
  --mode simulate \
  --unsupported-api error \
  python3 your_script.py
```

不支援的原生呼叫可以記錄、顯示警告，或傳回 `cudaErrorNotSupported` 或
`CUDA_ERROR_NOT_SUPPORTED`。

### 執行前檢查 GPU 記憶體

將指令執行到指定階段，並把報告寫入 Git 忽略的建置目錄：

```bash
python3 -m fakegpu preflight \
  --runtime fakecuda \
  --profile a100 \
  --stage forward \
  --report-dir build/preflight \
  --strict \
  -- python3 train.py
```

Preflight 追蹤已執行路徑中的可見 GPU 記憶體，並判斷選定的 profile 能否容納
該工作負載。

<a id="plan-online-llm-serving-capacity"></a>

### 規劃線上 LLM 服務容量

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

報告分別列出模型權重、prefill/decode 暫存記憶體、共享與私有 KV segment、
runtime/scheduler 開銷和裝置可用餘量。自訂容量可以用
`--device-memory-gib` 取代 `--target-profile`。Prefix 共享支援 dynamic 和
paged cache；paged 的共享與私有 segment 會分別依 block 取整。容量搜尋不會
超過 `--max-batch-size`。

#### 對齊 vLLM 的執行期預留

vLLM 會預先配置 paged KV 池，不是只依目前活躍請求逐步增加 cache。部署目標為
vLLM 時，可以選擇對應的 GPU 記憶體策略：

```bash
python3 -m fakegpu plan-serving \
  --model-dir /mnt/z/models/Qwen/Qwen3-0.6B \
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

自動模式採用 vLLM V1 的計算方式：model executor 申請的 GPU 記憶體減去
profile 得到的非 KV 記憶體，再向下取整為完整 cache block。報告會區分請求
實際需要的 KV 與預留池，並列出 `num_gpu_blocks`、cache token 容量、block
取整餘量、初始化是否可行，以及 `max_model_len` 下的並行容量。若已取得相同
設定的 vLLM 啟動 profile，可用 `--vllm-non-kv-cache-memory-bytes` 傳入權重、
activation 峰值、非 PyTorch 配置與 CUDA graph 的合計值，取代 FakeGPU 的
非 KV 建模值。`--vllm-kv-cache-memory-bytes` 用於指定每張 GPU 的 cache
大小；與 vLLM 一致，該選項會使 `--memory-utilization` 不再參與 cache 大小
計算。

未指定 `--memory-utilization` 時，vLLM 模式使用 `0.92`，與目前的
[vLLM CacheConfig 文件](https://docs.vllm.ai/en/latest/api/vllm/config/cache/)
一致。需要固定軟體版本時，建議明確填入該值。預算公式對應 vLLM 文件中的
[GPU worker profile 流程](https://docs.vllm.ai/en/latest/api/vllm/v1/worker/gpu_worker/)；
初始可用 GPU 記憶體、其他程序、tensor/pipeline parallelism、hybrid cache
group、allocator 碎片與版本相關 kernel 仍需在相同軟體堆疊上觀測。Research
manifest 覆蓋自動計算與明確 cache 兩種模式。Speculative decoding 目前只適用
於 generic 執行期模型；vLLM 專用的 speculative scheduler 與 cache manager
尚未納入估算。

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

報告會同時保留 target 與 draft 權重和兩套 KV cache，加入保守的 lookahead
slot，並比較 target verification 與 draft proposal 的暫存記憶體。接受率用於
計算每次 target 呼叫的預期輸出 token 數，不會改變記憶體峰值，也不表示端到端
加速比。兩個模型必須具有相同的 vocabulary 大小；checkpoint header 無法證明
token ID 完全一致，因此 tokenizer identity 仍標記為未驗證。

每份同質或混合請求的 serving 報告也會產生與路徑無關的
`workload_signature`。此簽章涵蓋 target/draft 架構與權重儲存、請求 shape、
KV/prefill 參數和 speculative 設定，不包含絕對模型路徑與裝置容量。使用
`calibrate verify` 前，可將相同欄位寫入觀測報告；校準檢查會比較簽章，並將
`target.profile` 作為獨立維度檢查，因此 proposal 長度或 GPU profile 不一致的
觀測資料不會通過驗證。

對於長度不同的 chat、RAG、completion 或 summarization 請求，可提供一份依
準入順序排列的請求清單：

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

清單模式會分別計算每個請求，同名 prefix 群組只儲存一次，並在不改變請求順序
的前提下接納能夠放入 GPU 記憶體的最長清單前綴。報告包含已接納與被拒絕的
請求 ID、每個請求的暫存記憶體、並行 prefill 中各元件的記憶體上界、群組 KV
segment 與容量餘量。同組請求必須使用相同且大於零的
`shared_prefix_tokens`；共享 prefix 僅支援 dynamic 或 paged KV 儲存。命名
群組視為已命中並駐留於 cache；報告不預測 cache 填充、命中機率或 eviction
行為。

請求到達時間、cache eviction、preemption 或排程器重排、tensor parallelism、
自適應 speculative 排程、KV 回復 workspace、吞吐量與延遲會列入未建模項目。
取得相同設定的線上服務觀測前，`validation_status` 保持為 `Modeled`，
`accuracy.status` 保持為
`uncalibrated`。

### 分析儲存庫或模型

```bash
# 尋找 GPU 入口、相依套件、原生原始碼與相容性風險。
python3 -m fakegpu analyze-repo .

# 估算 checkpoint、KV cache、暫存 tensor、adapter 與 MoE GPU 記憶體。
python3 -m fakegpu estimate-llm \
  --model-dir /models/example \
  --batch-size 1 \
  --prompt-tokens 128 \
  --generated-tokens 32 \
  --dtype bfloat16 \
  --kv-cache-strategy paged \
  --kv-cache-block-tokens 16 \
  --target-profile a100 \
  --json build/llm-estimate.json

# 比較 diffusion 生成階段與 GPU 記憶體最佳化選項。
python3 -m fakegpu estimate-diffusion --list-profiles
python3 -m fakegpu estimate-diffusion \
  --model-profile stable-diffusion-xl-base-1.0 \
  --height 1024 --width 1024 --batch-size 4 \
  --attention-backend sdpa --vae-slicing \
  --offload model --target-profile a100 \
  --json build/diffusion-estimate.json

# 檢查本機 Diffusers pipeline，不載入 tensor payload。
python3 -m fakegpu estimate-diffusion \
  --model-dir /models/pixart-or-flux \
  --height 1024 --width 1024 --text-tokens 300 \
  --dtype bfloat16 --attention-backend sdpa \
  --offload model --target-profile a100 \
  --json build/local-diffusion-estimate.json

# 根據能力 manifest 檢查原始碼與已建置原生函式庫的匯出符號。
python3 -m fakegpu capabilities \
  --source-root . \
  --build-dir build \
  --strict
```

LLM 估算器只會讀取 safetensors header，不會把 checkpoint 權重載入記憶體。
`--kv-cache-strategy` 可選 `dynamic`、`static`、`quantized` 或 `paged`；
JSON 報告會分別列出邏輯儲存量、量化節省量、static 預留、paged block 額外占用
與可選的 sliding-window 上限。量化 cache 預設將最近 128 個 token 保留為計算
dtype，可透過 `--kv-cache-residual-tokens` 調整。
MHA、GQA 與 MQA 使用分離的 key/value 儲存。模型設定同時包含
`kv_lora_rank`、`qk_nope_head_dim`、`qk_rope_head_dim` 與 `v_head_dim`
時，估算器會辨識 MLA，並依壓縮 KV latent 與獨立 RoPE 分量計算 cache 位元組。
MLA backend workspace 與 projection absorption 取決於執行期，仍需校準。

Diffusion 估算器分別計算 text encoding、重複 denoising 與 VAE decode 階段。
使用 `--model-dir` 時，它只讀取 `model_index.json`、元件 `config.json` 與選定
safetensors 檔案的 header，不會匯入遠端自訂程式碼，也不會讀取 tensor payload。
checkpoint 儲存位元組與指定 dtype 下的執行期權重位元組會分別列出。
內建 `pixart-sigma-xl-2-1024-ms` profile 與兩個 Stable Diffusion UNet profile
一同提供固定版本的 Transformer 範例。

估算器會區分 UNet、含交叉注意力的 patch Transformer（DiT/PixArt），以及
SD3/Flux 類聯合注意力 Transformer。圖像 token 數會結合 VAE 縮放、patch size
與 Flux latent packing 計算；CFG 只在需要正負分支 batch 的架構上使 denoiser
batch 加倍。當本機目錄包含多套權重時，可用
`--weight-variant fp16|bf16|fp32` 指定要檢查的檔案族。

`--offload model`、`--attention-slicing`、`--vae-slicing` 與
`--vae-tiling` 對應 Diffusers 中常見的 GPU 記憶體選項。取得相同設定的實體
GPU 比較資料前，報告狀態保持為 `Modeled`，`accuracy.status` 為
`uncalibrated`，也不會產生缺乏觀測依據的誤差百分比或預測區間。

<p align="right">(<a href="#readme-top">返回頂端</a>)</p>

## 指令參考

### 微調總時間與 GPU 記憶體預估

`estimate-finetune` 讀取本機模型中繼資料，支援 Hugging Face Trainer/PEFT、
LLaMA-Factory 與 Axolotl 設定，無須載入模型權重或匯入訓練框架。支援單卡
全量、LoRA、4-bit QLoRA，輸出 optimizer step 數、梯度累積、可訓練/凍結參數、
activation/workspace 記憶體，以及給定資料量與 epochs 的**訓練迴圈總時間**。
`max_steps` 優先於 epochs。樣本數指過濾、切分與 packing 後的訓練序列；
序列長度包含模板 token 與 padding，不能直接以原始文件行數代替 packed 序列數。
可用 `--max-steps -1` 清除設定中的步數限制，按指定 epochs 估算。

```bash
python -m fakegpu estimate-finetune \
  --config examples/finetuning/huggingface-lora.yaml \
  --model /path/to/Qwen3.5-0.8B --text-only \
  --profile "$TARGET_GPU_PROFILE" --num-samples 10000 --epochs 3 \
  --sequence-length 128 --json estimate.json
```

三類框架的估算範例位於 [`examples/finetuning/`](examples/finetuning/)。設定轉換已通過
CPU 測試，不代表所有框架與執行環境組合都已完成實體 GPU 基準測試。分散式/offload 與尚未
涵蓋的 optimizer/adapter 策略會明確報錯；分散式記憶體規劃仍用 `plan-training`。
Qwen3.5 的線性/完整 attention 分層計算，DeltaNet backward workspace、可選快速
核心與 MoE dispatch 仍為部分解析涵蓋，未校準結果不能保證不會 OOM。
`--text-only` 明確選擇多模態 checkpoint 的文字骨幹。

`profile-finetune-data` 完整掃描本機 JSONL 的每筆資料，統計內容摘要、token 長度、
截斷、padding 與處理後的序列數。原始 `text` 或 `messages` 使用本機 tokenizer，
計數表示所選前處理情境；通用 `--packing concatenate` 僅估算串接後的 token 預算，
不重現框架的 packing 演算法。要使用實際訓練集計數，應匯出最終 `input_ids` 並加上
`--processed`；此模式保留序列邊界，拒絕超過長度上限的資料。可選的
`attention_mask` 與 `labels` 用於統計 padding 與監督 token。

```bash
python -m fakegpu profile-finetune-data \
  --data /path/to/processed-train.jsonl --format input_ids --processed \
  --sequence-length 128 --json dataset.json
python -m fakegpu estimate-finetune \
  --config examples/finetuning/huggingface-lora.yaml \
  --model /path/to/Qwen3.5-0.8B --text-only \
  --profile "$TARGET_GPU_PROFILE" --dataset-report dataset.json \
  --epochs 3 --max-steps -1 --json estimate.json
```

`--dataset-report` 同時綁定樣本數與序列上限，取代設定中的預估樣本數，並拒絕衝突的
明確覆寫；報告長度上限必須與訓練設定一致。`--source-framework` 僅記錄來源標籤，
不能證明框架已執行。Packed 訓練需要最終處理後的序列計數，不能使用原始筆數或
通用串接情境代替。

`run-finetune` 在一張實體 CUDA 顯示卡上呼叫已安裝的 HF Trainer/PEFT、LLaMA-Factory 或 Axolotl
原生流程，支援本機未量化文字 checkpoint、無 packing 的全量微調、LoRA 與
bitsandbytes 4-bit QLoRA。輸入應為
包含資料與輸出路徑的完整原生設定，相對路徑以設定檔目錄解析。報告記錄原生
前處理、有效訓練資料、實際 batch、optimizer 更新、allocator 峰值、評估/儲存事件與
最終模型儲存。Warmup 不計入穩態步驟樣本，但保留在總牆鐘時間中；此指令會執行
原生設定要求的訓練。

`observation.updates[].batches` 記錄每個實際訓練 microbatch 的形狀、從 1 開始的
epoch 位置、token 槽位、非 padding tokens，以及一般因果移位前後的有效 labels。
`observation.batch_profile` 核對總數，並依完整且有序的梯度累積窗口分組耗時；混合
形狀與不完整窗口分別保留。每組至少有兩個穩態 update 才標記 `sufficient`，這僅
表示樣本數量足夠，不代表精度認證。缺失或非二值 attention mask 的 token 數保持未知，
position IDs 的重置也不代表原始 packed 樣本邊界。統計開銷分別記錄並從 update
耗時中扣除，總牆鐘時間仍包含該開銷；同步觀測使用獨立校準身分。

`validate-finetune-windows` 使用獨立訓練報告驗證動態視窗耗時預測。至少提供兩份
模型、資料、軟體及原生執行設定相同的獨立校準報告；每個完整有序視窗，在每份
校準報告中排除全域 warmup 和該視窗首次出現後，仍須有兩個樣本。各程序均值
等權聚合，缺少覆蓋的視窗直接拒絕。新報告記錄 run UUID、PID 與 UTC 開始時間，
用於發現意外重複採樣。

只預測 holdout 已記錄的穩態視窗；holdout 耗時僅用於計算誤差，不參與校準。
該命令驗證歷史證據，不載入目前權重，也不認證目前環境。它不推斷未來 shuffle
順序、warmup、總牆鐘時間或動態 VRAM；視窗統計相同不代表完整輸入等價。
週期 evaluation/save 排程需要下方明確啟用的事件上下文模式；原生 packing 與 MoE
視窗校準仍未支援。
```bash
python -m fakegpu validate-finetune-windows \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --holdout-run holdout.json --json window-validation.json
```


`observation.phase_events` 記錄實際評估/儲存呼叫樹，包括訓練開始評估、週期
checkpoint 和框架最終呼叫；階段總量只累計根事件。帳本區分原始牆鐘時間與
事件統計開銷，記錄真實評估 batch/內容指紋和儲存後存在的檔案。檔案清單不代表
實際磁碟寫入量或持久化完成時間。update 的 `preceding_event_ids` 指向上次更新
之後發生的事件。

`validate-finetune-events` 用獨立訓練報告驗證評估/儲存子總時間。至少提供兩份
校準報告；開始、首次、重複和最終事件分別匹配，重複事件在每份校準報告中至少
需要兩個樣本。程序均值等權聚合，缺少覆蓋直接拒絕。首版限無 packing 的 dense
文字模型，支援 loss-only 且無自訂 metrics 的評估，以及本地模型/adapter 或
完整可恢復 checkpoint 儲存。
它不推斷未來事件次數或完整流程時間：儲存可能改變後續訓練的 allocator/cache
狀態。

```bash
python -m fakegpu validate-finetune-events \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --holdout-run holdout.json --json event-validation.json
```

`validate-finetune-windows --event-context` 也會匹配訓練視窗之前的實際事件歷史。
它分別保留最近的評估和 checkpoint 邊界，包括有序呼叫、工作量、首次/重複階段，
以及各邊界之後已完成的 optimizer updates。新的評估不會抹去之前儲存的上下文。
報告必須包含事件帳本且實際硬體一致；事件支援範圍與 `validate-finetune-events` 相同。

排除全域 warmup 和每個「視窗＋上下文」組合的首次出現後，每份校準報告仍須有
至少兩個樣本。這驗證週期事件後重複出現的視窗，不預測第一次儲存的恢復耗時，
也不證明更早歷史已無影響或完整訓練時間可預測。評估/儲存本身的耗時仍分別統計。

```bash
python -m fakegpu validate-finetune-windows --event-context \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --holdout-run holdout.json --json event-window-validation.json
```

增加 `--epoch-context` 可同時匹配每個 microbatch 在 epoch 內的實際位置，區分
包含新一輪取數或 worker 啟動的同形視窗。它使用完整有序、從 1 開始的 microstep
位置，不綁定絕對 epoch 編號。報告必須有 epoch 中繼資料；仍排除 warmup 和組合首次
出現，並要求足夠的獨立樣本。無週期事件時可單獨使用，有週期事件時與
`--event-context` 一起使用：

```bash
python -m fakegpu validate-finetune-windows --event-context --epoch-context \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --holdout-run holdout.json --json epoch-window-validation.json
```

`plan-finetune-pipeline` 從校準執行中完整觀測到的 epoch 推導未來輪次或更新次數，
在目標訓練啟動前產生計畫。`estimate-finetune-pipeline` 使用至少兩次獨立校準，
組合 setup、warmup、首次視窗、訓練視窗、評估、checkpoint、Trainer 間隙與收尾。
新執行報告提供互不重疊的時間帳本，分別保留原始牆鐘時間及已測統計開銷。計時止於
原生呼叫及最後 CUDA 同步，不包含 Python 程序啟動、報告建構或 CLI 寫檔。
時間區間由各部件程序均值的極值組合而成，不是信賴區間；任一部件缺少獨立覆蓋
都會明確拒絕預測。

未來排程首版支援無 packing、保留尾批次的循序文字資料：HF Trainer 5.6.0 和
LLaMA-Factory 0.9.5 / Transformers 5.6.0 要求 `disable_shuffling: true`；
Axolotl 0.18.0 / Transformers 5.14.1 要求 `curriculum_sampling: true`。
會核對實際 batch 順序、評估/儲存排程、小數輪次及末尾事件。資料、模型、硬體和
執行設定須符合校準；改變資料筆數需要新的匹配校準。日誌須關閉或每次 update
記錄；checkpoint 輪替刪除和最佳模型重新載入仍需要獨立支援。

```bash
python -m fakegpu plan-finetune-pipeline \
  --run-report calibration-1.json --epochs 30 --json future-plan.json
python -m fakegpu estimate-finetune-pipeline \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --plan future-plan.json --memory-budget-gib 24 --reserve-gib 1 --json budget.json
python -m fakegpu validate-finetune-pipeline \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --holdout-run holdout.json --json validation.json
```

預設隨機採樣與更換訓練資料可使用顯式的經驗遷移預算。程式讀取原生 prepared
dataloader，在 CPU 計畫目標輪次內的 sampler/collator 輸出，並於校準訓練中
逐批核對列號及輸入張量 SHA256。上述 HF/Factory 版本搭配 Accelerate 1.11.0，
Axolotl 搭配 Accelerate 1.13.0；採樣使用實際 data seed 加 epoch。
Axolotl 0.18.0 不會將 YAML 的 `data_seed` 傳給 Trainer。

```bash
python -m fakegpu calibrate-finetune \
  --config target-training.json --framework huggingface \
  --profile "$TARGET_GPU_PROFILE" --python /path/to/framework/python \
  --cuda-device 0 --workspace ./calibration-target \
  --calibration-steps 24 --warmup-steps 2 --epochs 3 \
  --timeout-seconds 600 --json calibration.json
python -m fakegpu budget-finetune-workspace \
  --workspace ./calibration-target --epochs 2 --json revised-budget.json
```

工作流程固定啟動兩次獨立原生程序，明確限制校準步數及執行時間，預設各自使用新的
空 kernel 快取目錄。保留所有嘗試，以 SHA256 封存報告與預測，中斷後可加 `--resume`。
重用前核驗成功產物，不隱式增加校準次數，也不因誤差較大而重跑。
純 CPU 重估預算不需 GPU 或框架環境，但不能超過已儲存的採樣計畫範圍。
工作區需保留原始絕對路徑，並放在被整體讀取的輸入目錄（如模型或本機資料目錄）之外；
巢狀路徑會在探測執行環境或訓練之前拒絕。
首版支援有限本機資料，工作區依賴 POSIX 程序及檔案鎖語義。

重複加入 `--reuse-calibration-run PATH` 可重用另一份資料的相容訓練證據。
啟動及收尾仍由目標資料的兩次新校準提供，不依樣本數比例放大。
訓練分組保留有序張量形狀、累積尾窗、epoch 邊界、warmup、形狀首次使用及事件上下文；
不同 token、mask、監督分布間的成本重用是**經驗近似**，報告保留完整描述符分布。
每組至少兩個獨立程序，重複組於每程序至少兩次；缺少覆蓋時回傳退出碼 1 及缺口清單，
不產生總預算。縮短時長亦可能引入未校準的終止位置，仍須重新檢查覆蓋。
GPU 記憶體獨立報告完整校準程序峰值包絡，不保證長跑組合或更小顯卡不會 OOM。

亦可用 `run-finetune --planning-epochs N` 或 `--planning-max-steps N` 儲存 CPU 計畫，
再分別呼叫 `plan-finetune-transfer`、`estimate-finetune-transfer`、
`validate-finetune-transfer` 產生計畫、預算與獨立 holdout 比較。
既有嚴格 pipeline 命令維持原匹配規則。packing、streaming、自訂採樣器及自動
batch size 調整仍未開放。

長度分組時，HF/LLaMA-Factory 設定 `train_sampling_strategy: group_by_length`，
Axolotl 設定 `group_by_length: true`，再向 `calibrate-finetune` 或 `run-finetune`
明確加入 `--length-grouping-seed 123`。它保留原生 Trainer 的分組演算法，以獨立 CPU
生成器使用 `123 + epoch` 種子；此選項會改變 sampler 的隨機流。依賴全域 RNG 的原生
長度分組不能直接用於可重現的未來計畫。必須指定規劃範圍。可提供一個有限評估資料集：
評估使用獨立生成器，每次呼叫均使用同一明確種子，persistent worker 同樣維持固定順序，
不推進訓練 sampler。評估分組策略與長度 SHA 納入校準身分；拒絕自訂評估 sampler
及提前準備的 dataloader。
sampler plan v2 將完整有序分組長度及分組 batch size 保留於校準身分中；不同長度序列
需有自己的證據。每組仍須有兩個獨立程序與足夠重複觀測，短校準可能無法覆蓋多種形狀。

[packing 語義審計](tests/native/finetune_packing_semantics.py) 在極小的真實 Qwen3.5 模型中
改變前一段，檢查後一段的輸出與梯度，並提供純 attention 對照。實測一般 HF/Factory
混合路徑於分塊 mask 及重設位置後仍有跨段影響；所測 Axolotl patch 與 Transformers
5.14.1 的 decoder 介面不相容。這是核心診斷，尚未開放 packed 訓練預算。版本與保留結果
見 [`refactor.md`](refactor.md)。

[分段參考審計](tests/native/finetune_segment_reference.py) 檢查專案自有的 Qwen3.5
分段介面：每段 linear attention 使用空的卷積與遞迴狀態，並檢查分塊 mask、padding
及 causal label 邊界。逐層輸出與梯度和獨立原生序列對照，同時檢查 checkpoint 重算
與模型儲存還原。這是顯式參考介面；原生 packed sampler 計數與訓練預算另行驗收。

記憶體結果為同一 GPU 和 allocator 上已覆蓋工作量的完整執行 allocated/reserved 峰值
包絡，不相加各視窗峰值，也不代表已投射到一張更小的實體卡。
`--reserve-gib` 是使用者指定的餘量；落在包絡內不保證不會 OOM。
`validate-finetune-pipeline` 分別報告 holdout 時間誤差和記憶體低估；實際記憶體分配失敗
透過另外的真實訓練驗證。

`run-finetune --cache-regime empty_kernel_cache` 要求明確設定五個互不重疊、已存在
且為空的目錄：`TRITON_CACHE_DIR`、`TORCHINDUCTOR_CACHE_DIR`、`CUDA_CACHE_PATH`、
`TORCH_EXTENSIONS_DIR`、`PYTORCH_KERNEL_CACHE_PATH`。
`populated_kernel_cache` 記錄其中全部檔案的相對路徑、大小和 SHA256；每次獨立
執行應複製同一份種子到新目錄。runner 不建立、清理或預熱快取。這僅控制指定
目錄，不代表 OS 檔案快取或所有函式庫快取均受控；預設 `uncontrolled` 不作快取狀態
承諾，也不能與受控校準混用。檔案清單不證明實際 cache 命中或外部相依完整：
Triton group 中繼資料可能仍透過絕對路徑引用原快取，必須保留並固定這些檔案；
本輪驗證另外凍結、核對了該相依。

HF 原生入口接收平鋪設定及每行恰好包含 `input_ids`、`attention_mask`、`labels` 的
processed JSONL：三個等長整數陣列、二值 attention mask，padding 前長度為 2 到
`sequence_length`；本機 tokenizer 須已有 pad token，不隱式斷詞或截斷。使用 [原生 LoRA 範例](examples/finetuning/huggingface-native-lora.yaml)
並向 `run-finetune` 傳入 `--framework huggingface`；既有定長 `probe-finetune` 仍可使用。

[完整流水線驗收紀錄](tests/data/finetune_pipeline_validation_evidence.json) 彙整三框架
30 個原生程序，包含獨立的 allocator OOM 對照。分階段誤差、快取條件及未涵蓋範圍
見 [`refactor.md`](refactor.md)。

[採樣與遷移驗收記錄](tests/data/finetune_transfer_validation_evidence.json) 另含 Blackwell GPU
上的 18 個 Qwen3.5-0.8B 原生程序：三框架的 25 → 73 筆資料遷移，以及隨機採樣與週期
eval/save 組合。每組僅一次獨立 holdout，保留分階段誤差與覆蓋不足記錄，不能視為通用
精度或 OOM 安全保證。

支援標準梯度檢查點。LLaMA-Factory 使用 `disable_gradient_checkpointing: false`，
並透過 `use_reentrant_gc: true` 或 `false` 選擇實作；Axolotl 使用
`gradient_checkpointing: true` 與 `gradient_checkpointing_kwargs: {use_reentrant: true}`
（也可將 `use_reentrant` 設為 `false`）。程式從載入後的模型核驗實際檢查點與量化設定，並納入校準身分；
變更這些設定後需要重新校準。

原生文字訓練需要文字 checkpoint，估算器的 `--text-only` 不會為原生訓練轉換原始
多模態權重。對於本機 Qwen3.5，先在 CPU 上匯出並驗證獨立文字 checkpoint，
同時產生用於框架整合檢查的合成 Alpaca 資料：

```bash
python tests/native/prepare_finetune_fixture.py \
  --model /path/to/Qwen3.5-0.8B \
  --output-dir /tmp/fakegpu-finetune/qwen35-text \
  --dataset-dir /tmp/fakegpu-finetune/data
```

匯出時使用支援 `Qwen3_5ForCausalLM` 的 Transformers 環境。原生範例針對
LLaMA-Factory 0.9.5 / Transformers 5.6.0 與 Axolotl 0.18.0 / Transformers 5.14.1。
分別在獨立環境安裝對應框架與相容的 CUDA PyTorch、PEFT、資料處理依賴。
設定使用上述路徑，輸出目錄必須為新目錄或空目錄；再次執行前選擇新的
`output_dir`。Axolotl 的前處理快取另行保留在 `/tmp/fakegpu-finetune/axolotl-prepared`。

```bash
CUDA_VISIBLE_DEVICES=1 python -m fakegpu run-finetune \
  --config examples/finetuning/axolotl-native-lora.yaml --framework axolotl \
  --profile "$TARGET_GPU_PROFILE" --warmup-steps 2 --output run.json
python -m fakegpu estimate-finetune \
  --run-report run.json --epochs 3 --max-steps -1 --json estimate.json
```

LLaMA-Factory 使用 `--framework llamafactory` 與
[`llamafactory-native-lora.yaml`](examples/finetuning/llamafactory-native-lora.yaml)。應在**同一 Python
環境**重用 `--run-report`：CLI 會核對目前本機權重、模型設定及軟體版本與執行紀錄
的身分一致性。有效資料內容摘要與實測形狀仍綁定在報告中，GPU profile 預設沿用
報告值。上述舊 `estimate-finetune` 單報告介面要求固定形狀的完整 batch；其原生
流程總耗時僅在相同資料與本機
快取狀態、無週期性評估或 checkpoint 儲存時，重用一次實測的啟動、warmup 與最終
儲存開銷。改變 `--num-samples` 僅提供訓練迴圈情境，不外推未測量的資料準備與
載入時間。這些情境不是信賴區間，也不包含 Python 程序啟動或下載。

512-token NF4 QLoRA 與梯度檢查點範例見
[`llamafactory-native-qlora.yaml`](examples/finetuning/llamafactory-native-qlora.yaml) 與
[`axolotl-native-qlora.yaml`](examples/finetuning/axolotl-native-qlora.yaml)。兩者啟用
double quantization 與 BF16 compute/storage，需要在對應框架環境安裝 bitsandbytes。
使用已匯出的 tokenizer 產生較長資料，無須再次匯出模型：

```bash
python tests/native/prepare_finetune_fixture.py \
  --model /tmp/fakegpu-finetune/qwen35-text --dataset-only \
  --dataset-dir /tmp/fakegpu-finetune/data-long \
  --sequence-length 1024 --num-samples 64
```

fixture 參數指定並檢查原始資料的最小長度；框架隨後依設定截斷至 512 或 1024
tokens，執行報告記錄實際採用的序列與 batch。

獨立校準與複測程序可透過
[`tests/native/finetune_framework_matrix.py`](tests/native/finetune_framework_matrix.py) 執行。
矩陣沿用原生設定的序列長度、microbatch、梯度累積與 seed；可分別指定校準與複測步數：

```bash
python tests/native/finetune_framework_matrix.py \
  --framework axolotl --config examples/finetuning/axolotl-native-qlora.yaml \
  --python /path/to/axolotl/venv/bin/python \
  --output-dir /tmp/fakegpu-finetune/axolotl-qlora-matrix \
  --gpu-index 1 --calibration-steps 6 --warmup-steps 2 \
  --holdout-runs 2 --holdout-steps 16
```

步數包含 warmup。預設仍為 6 次校準更新、2 次 warmup，以及三次各 32 次更新的複測。
每次訓練使用獨立輸出目錄；矩陣保留原 seed，複製設定前解析本機相對路徑。
時間預測呼叫正式估算器，並使用實際 microstep 計數，包含 epoch 尾端不完整的累積視窗。

[原生框架驗證歸檔](tests/data/finetune_framework_validation_evidence.json) 保留了
先前 128-token Qwen3.5-0.8B LoRA 在 Blackwell GPU 上的兩框架結果，各含三次獨立完整 epoch 複測。
整體時間最大誤差為 Factory **3.931%**、已有編譯快取的 Axolotl **1.125%**。
歸檔也保留首次執行校準外推至後續快取程序時 **364.885%** 的誤差：首次 update 耗時
**66.542 秒**。目前不會自動核驗 cache/JIT 狀態；模型與軟體簽名相同不能保證這部分
固定開銷可重用。精確設定、版本、GPU 記憶體結果及未涵蓋範圍見 [`refactor.md`](refactor.md)。

[擴充驗證歸檔](tests/data/finetune_extended_validation_evidence.json) 涵蓋兩框架的六組
設定，包括 512/1024 tokens、batch 1/2、梯度檢查點與原生 NF4 QLoRA。每組包含
一次 6-update 校準與兩次獨立 16-update 複測，每個程序前 2 次更新為 warmup，
共 18 個測量程序。各組均先短跑預熱，並保留共享快取。精確結果與設定適用範圍
見 [`refactor.md`](refactor.md)。

有實體顯示卡時，可在**同一 Python 環境**先取樣，再推估資料量與 epochs：

```bash
CUDA_VISIBLE_DEVICES=1 python -m fakegpu probe-finetune \
  --config examples/finetuning/huggingface-lora.yaml \
  --model /path/to/Qwen3.5-0.8B --text-only \
  --profile "$TARGET_GPU_PROFILE" --warmup-steps 2 --measure-steps 5 \
  --output observation.json
python -m fakegpu estimate-finetune \
  --config examples/finetuning/huggingface-lora.yaml \
  --model /path/to/Qwen3.5-0.8B --text-only \
  --profile "$TARGET_GPU_PROFILE" --num-samples 10000 --epochs 3 \
  --observation observation.json --json calibrated.json
```

內建 probe 使用 Hugging Face Trainer 與固定長度合成算術文字，需要 CUDA
PyTorch、支援模型架構的 Transformers 與 Accelerate；adapter 另需 PEFT，QLoRA
另需 bitsandbytes。觀測綁定模型、GPU、軟體版本、精度、adapter、attention、batch、
長度與累積設定；保留啟動/穩態峰值，計時包含 batch 讀取與 optimizer 更新。
它不驗證收斂，也不代替其他框架的實測。推估時間不含資料準備、模型載入、評估與
checkpoint 儲存；解析情境範圍和實測樣本極值都不是統計信賴區間。
採樣與重用觀測時會計算本機 checkpoint 檔案的內容摘要，同路徑替換權重會使
舊校準失效；大型模型因此會增加啟動時的檔案讀取開銷。

[Qwen3.5-0.8B 驗證紀錄](tests/data/finetune_validation_evidence.json) 保留了
Blackwell GPU 上全量、LoRA、bitsandbytes QLoRA 的獨立校準與複測，以及未校準
解析誤差。這些短跑不能證明更長或不同設定的精度；重現入口為
[`tests/native/finetune_matrix.py`](tests/native/finetune_matrix.py)。

完整功能驗收矩陣與後續階段見 [`refactor.md`](refactor.md)。

| 指令 | 用途 |
|---|---|
| `fakegpu doctor` | 檢查安裝、原生函式庫、PyTorch 與 profiles |
| `fakegpu demo` | 執行小型 CPU FakeCUDA 訓練步驟 |
| `fakegpu preflight` | 將工作負載執行到指定階段並判斷 fit 或 OOM |
| `fakegpu analyze-repo` | 統計儲存庫入口與僅支援 GPU 的風險 |
| `fakegpu analyze-kernel` | 檢查 CUDA、PTX 與 SASS 資源及運算 |
| `fakegpu estimate-llm` | 估算 decoder GPU 記憶體、通訊量與 FLOP |
| `fakegpu plan-serving` | 規劃同構或混合請求準入、chunked prefill 與群組共享 prefix KV 儲存 |
| `fakegpu estimate-diffusion` | 估算 diffusion 的 text encode、denoise 與 VAE decode 階段 GPU 記憶體 |
| `fakegpu estimate-roofline` | 產生與 profile 相關的分析延遲區間 |
| `fakegpu plan-training` | 統一分散式訓練設定並估算單 rank GPU 記憶體 |
| `fakegpu estimate-finetune` | 依框架設定估算單卡全量/LoRA/QLoRA 的步數、時間與 GPU 記憶體 |
| `fakegpu probe-finetune` | 在實體 CUDA GPU 蒐集固定形狀 Hugging Face Trainer 時間與記憶體 |
| `fakegpu profile-finetune-data` | 完整掃描本機 JSONL 的 token 長度、截斷、padding 與處理後序列數 |
| `fakegpu run-finetune` | 執行 HF Trainer/PEFT、LLaMA-Factory 或 Axolotl 原生訓練並記錄資料、CUDA 時間、記憶體及最終儲存 |
| `fakegpu validate-finetune-windows` | 使用獨立校準與 holdout 訓練驗證動態視窗耗時 |
| `fakegpu validate-finetune-events` | 使用獨立原生事件記錄驗證評估與 checkpoint 成本 |
| `fakegpu plan-finetune-pipeline` | 從校準循序資料推導未來原生訓練輪次或更新數 |
| `fakegpu estimate-finetune-pipeline` | 估算完整原生執行時間和已覆蓋記憶體包絡 |
| `fakegpu validate-finetune-pipeline` | 對照獨立原生訓練驗證完整時間和記憶體預測 |
| `fakegpu plan-finetune-transfer` | 計畫經核驗的隨機或循序原生輸入 |
| `fakegpu estimate-finetune-transfer` | 估算顯式經驗形狀遷移預算 |
| `fakegpu validate-finetune-transfer` | 對照獨立原生執行驗證遷移預算 |
| `fakegpu calibrate-finetune` | 於可恢復工作區執行限額原生校準 |
| `fakegpu budget-finetune-workspace` | 僅用 CPU 對儲存的證據重估預算 |
| `fakegpu simulate-topology` | 模擬 collective 路由與鏈路競爭 |
| `fakegpu replay-trace` | 彙整運算、通訊、等待與 GPU 記憶體時間軸 |
| `fakegpu calibrate` | 整理 serving 觀測、比較 GPU 記憶體報告並執行可靠性門檻檢查 |
| `fakegpu capabilities` | 列出或嚴格檢查原生 API 分類 |
| `fakegpu nvidia-smi` | 查看裝置、程序、建模拓撲/MIG、健康狀態與可靠性事件 |
| `fakegpu metrics` | 匯出有數量上限的 Prometheus/JSON 指標與短期記憶體歷史 |
| `fakegpu workspace-profiles` | 驗證並查看 workspace 估算 profiles |
| `fakegpu validate` | 執行 JSON、TOML 或 YAML 宣告式驗證矩陣 |
| `fakegpu coordinator` | 管理分散式模擬 coordinator |
| `fakegpu bandwidth` | 驗證模擬 TCP payload 並回報吞吐量 |

使用 `python3 -m fakegpu --help` 查看完整列表，使用
`python3 -m fakegpu <command> --help` 查看各指令的選項。

<p align="right">(<a href="#readme-top">返回頂端</a>)</p>

## GPU profiles

目錄包含 82 個 YAML profile，涵蓋從 Maxwell 到 Blackwell 的消費級、工作站、
資料中心與嵌入式 NVIDIA GPU。Python 與原生 runtime 共用這些 profiles。

```bash
python3 -m fakegpu doctor --list-profiles
python3 -m fakegpu demo --profile rtx4090
python3 -m fakegpu --build-dir build --devices "t4,a100:2,h100" \
  python3 your_script.py
python3 scripts/update_nvidia_gpu_catalog.py --check
```

設定 `FAKEGPU_PROFILE` 或傳入 `--profile` 可選擇一個 profile。使用 `--devices`
可設定異質裝置列表。

<p align="right">(<a href="#readme-top">返回頂端</a>)</p>

## 開發

### 建置

所有可重複使用的原生建置行為都由一個腳本提供：

```bash
scripts/build.sh
scripts/build.sh --release
scripts/build.sh --debug
scripts/build.sh --build-dir build-custom -- -DSOME_CMAKE_OPTION=value
```

建置目錄與編譯產物不會由 Git 管理。

### 測試

專案維護的回歸測試分為四組指令：

```bash
scripts/test.sh python
scripts/test.sh smoke
scripts/test.sh cpu
scripts/test.sh all
```

| 測試組 | 涵蓋內容 |
|---|---|
| `python` | 專案維護的 Python 回歸測試 |
| `smoke` | 原生函式庫載入、報告、能力檢查、SMI 拓撲/健康狀態與 coordinator |
| `cpu` | CPU cuBLAS 模擬 |
| `all` | 所有維護中的測試 |

需要時可以直接執行宣告式驗證 manifest：

```bash
python3 -m fakegpu validate \
  --manifest tests/data/validation_smoke.yaml \
  --report-dir build/validation-smoke \
  --strict

python3 -m fakegpu validate \
  --manifest tests/data/research_validation.yaml \
  --report-dir build/validation-research \
  --strict
```

### 可重複使用的腳本

| 路徑 | 用途 |
|---|---|
| `scripts/build.sh` | 設定並編譯原生目標 |
| `scripts/test.sh` | 執行維護中的測試組 |
| `scripts/update_nvidia_gpu_catalog.py` | 檢查或更新 profile 中繼資料 |
| `scripts/validation/` | 共用的報告與產物驗證工具 |
| `scripts/linux/` | Linux GPU 管理工具 |
| `scripts/macos/` | macOS 到 Linux VM 的輔助工具 |

<p align="right">(<a href="#readme-top">返回頂端</a>)</p>

## 專案結構

```text
FakeGPU/
├── fakegpu/    Python 套件、CLI、runtime 與估算器
├── profiles/   YAML GPU profile 目錄
├── schemas/    JSON 報告與驗證 schema
├── scripts/    可重複使用的建置、測試、平台與驗證工具
├── src/        原生 C++ 攔截與 coordinator 實作
└── tests/      維護中的回歸測試與最小原生測試 fixture
```

產生的建置目錄、編譯函式庫、測試報告、快取、本機環境、二進位資源與設計草稿
都由 `.gitignore` 排除。

<p align="right">(<a href="#readme-top">返回頂端</a>)</p>

## 限制

- 原生模擬無法執行任意 CUDA kernel。
- FakeCUDA 涵蓋專案維護的 Python 與 PyTorch 行為，不支援二進位 CUDA 擴充。
- 靜態分析無法解析所有動態 import、生成式 kernel、執行階段 shape 或依賴資料的
  分支。
- GPU 記憶體估算可能遺漏 backend 私有配置、自訂 operator、allocator 策略與未
  匹配的 workspace。
- `estimate-llm` 與 `plan-serving` 對視覺模型計入 checkpoint 權重，但未涵蓋
  vision/projector activation 與 FLOP；decoder token 數需要包含視覺 token，
  完整多模態記憶體仍需另外驗證。
- Diffusion profile 只描述固定的參考 pipeline。自訂 ControlNet、IP-Adapter、
  LoRA、safety checker、refiner、影片與 DiT 元件需要額外的元件資料及對應的
  實體 GPU 驗證。
- 線上 serving 規劃支援不同請求長度與獨立的自迴歸 draft 模型，但不估算請求
  到達分布、cache eviction、preemption 或排程器重排、self-speculative、
  Medusa、EAGLE、tokenizer 重映射、tensor parallelism、吞吐量與延遲。
- Roofline 輸出是分析區間，不是實測 kernel 延遲。
- 分散式耗時包含 coordinator、記憶體複製、socket 與程序排程，不能作為 NCCL、
  NVLink 或 RDMA benchmark。
- 建模故障事件僅用於檢查控制面報告，不會改變 CUDA 執行，也不代表 NVML ECC
  計數、實際觀測到的 Xid 或硬體故障。
- Hybrid 與 passthrough 模式需要相容的實體 CUDA 環境。
- macOS System Integrity Protection 可能移除系統程式的 `DYLD_*` 環境變數。
  原生攔截建議使用 Homebrew、conda 或 pyenv Python。

<p align="right">(<a href="#readme-top">返回頂端</a>)</p>

## 開發計畫

- [x] CPU PyTorch FakeCUDA runtime
- [x] 原生 CUDA、NVML、cuBLAS 與 NCCL 攔截
- [x] 可識別架構的 GPU profile 目錄
- [x] 執行階段、靜態、LLM 與分散式 GPU 記憶體分析
- [x] 分階段估算 Stable Diffusion v1.5 與 SDXL 生成所需的 GPU 記憶體
- [x] 儲存庫、kernel、拓撲與 trace 分析
- [x] FakeGPU-SMI 裝置、runtime、allocator 與程序詳細查詢
- [ ] 擴充可執行的原生 CUDA 運算與 cuBLAS 涵蓋範圍
- [x] 支援三類框架的單卡微調預估與相同設定 Hugging Face Trainer 實體 GPU 取樣
- [x] 完整資料掃描與 LLaMA-Factory/Axolotl 單卡文字全量微調、LoRA、4-bit QLoRA 原生執行報告
- [ ] 使用各自實際執行入口驗證 LLaMA-Factory 與 Axolotl
- [x] 透過 FakeGPU-SMI 發布原生 runtime 的即時狀態與活動
- [x] 加入建模拓撲、NVLink 檢視與 NVML peer 查詢
- [x] 加入建模故障注入、健康狀態與可靠性事件檢視
- [x] 加入建模 MIG 檢視與原生 NVML MIG handle 查詢
- [x] 為裝置、程序與 runtime 匯出有數量上限的指標及 Prometheus 記憶體歷史
- [x] 加入 continuous batching、混合請求長度、chunked prefill 與群組 prefix caching 的線上 serving 容量模型
- [x] 加入 draft-model speculative decoding 記憶體與準入模型
- [ ] 為長上下文與線上服務加入實體 GPU LLM 驗證
- [ ] 加入實體 GPU diffusion 驗證，以及訓練與 DiT GPU 記憶體模型
- [ ] 在更多 GPU 與軟體堆疊上驗證分散式與 MoE 估算

建議功能與已知限制請參閱
[GitHub Issues](https://github.com/FanBB2333/FakeGPU/issues)。

<p align="right">(<a href="#readme-top">返回頂端</a>)</p>

## 參與貢獻

歡迎提交問題報告、針對性測試案例、profile 修正、文件改進與程式碼修改。

1. Fork 儲存庫。
2. 建立分支：`git checkout -b feat/your-change`。
3. 為修改的行為新增或更新測試。
4. 執行 `scripts/test.sh all`。
5. 使用清楚的
   [Conventional Commit](https://www.conventionalcommits.org/) 訊息提交。
6. Push 分支並建立 pull request。

GPU 記憶體估算或相容性問題應附上完整指令、選定的 profile、軟體版本與產生的
報告。

<p align="right">(<a href="#readme-top">返回頂端</a>)</p>

## 授權

專案採用 MIT License，詳情請參閱 [LICENSE](LICENSE)。

<p align="right">(<a href="#readme-top">返回頂端</a>)</p>

## 致謝

- README 結構參考
  [Best-README-Template](https://github.com/othneildrew/Best-README-Template)
- 基於 [PyTorch](https://pytorch.org/) 驗證 CPU 框架行為
- 使用 [CMake](https://cmake.org/) 建置原生函式庫

<p align="right">(<a href="#readme-top">返回頂端</a>)</p>

[test-shield]: https://img.shields.io/github/actions/workflow/status/FanBB2333/FakeGPU/test.yml?branch=main&style=for-the-badge&label=tests
[test-url]: https://github.com/FanBB2333/FakeGPU/actions/workflows/test.yml
[release-shield]: https://img.shields.io/github/v/release/FanBB2333/FakeGPU?include_prereleases&sort=semver&style=for-the-badge
[release-url]: https://github.com/FanBB2333/FakeGPU/releases
[python-shield]: https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white
[python-url]: https://www.python.org/
[license-shield]: https://img.shields.io/github/license/FanBB2333/FakeGPU?style=for-the-badge
[license-url]: LICENSE
[validation-static]: https://github.com/FanBB2333/FakeGPU/blob/df254c21eebc0a5bbf13992f3f5a8e995cfa8708/docs/ai-researcher-preflight.md#4-static-aten-storage-liveness-validation
[validation-inference]: https://github.com/FanBB2333/FakeGPU/blob/df254c21eebc0a5bbf13992f3f5a8e995cfa8708/docs/llm-inference-estimation.md#maintained-qwen3-8b-result
[validation-sft]: https://github.com/FanBB2333/FakeGPU/blob/df254c21eebc0a5bbf13992f3f5a8e995cfa8708/docs/llm-sft-memory-estimation.md#blackwell-matrix
[validation-qlora]: https://github.com/FanBB2333/FakeGPU/blob/df254c21eebc0a5bbf13992f3f5a8e995cfa8708/docs/llm-sft-memory-estimation.md#native-nf4-qlora-matrices
[diffusion-sd15]: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/451f4fe16113bff5a5d2269ed5ad43b0592e9a14
[diffusion-sdxl]: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/462165984030d82259a11f4367a4eed129e94a7b
[diffusion-pixart]: https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS/tree/e102b3591cc82e97071b8b4cb90d834d0c487207
