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
   - [微調總時間與 GPU 記憶體預估](#微調總時間與-gpu-記憶體預估)
   - [線上 serving 基準測試](#線上-serving-基準測試)
   - [Diffusion 訓練](#diffusion-訓練)
   - [階段記憶體校準](#階段記憶體校準)
   - [分散式規劃與實驗批次執行](#分散式規劃與實驗批次執行)
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

範例中的 `TARGET_GPU_PROFILE` 由使用者透過
`python3 -m fakegpu doctor --list-profiles` 選擇後設定。公開實驗資料只披露 GPU
架構；支援目錄中的型號表示相容能力，不表示維護者持有該裝置。

FakeGPU 為開發、CI、相容性檢查與容量規劃模擬面向 CUDA 的執行環境：應用程式
可以偵測可設定的 NVIDIA 風格裝置，已維護的運算會在 CPU 上執行，模擬的 GPU
記憶體與通訊資料會被記錄。對於不應直接載入的工作負載，FakeGPU 也提供靜態
估算工具。

只有 passthrough、hybrid 與校準流程需要實體 GPU。

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

誤差依 `|預測值 - 實測值| / 實測值 × 100%` 計算，一致度為 `100% - 誤差`。

| 已驗證範圍 | 實體 GPU 參考環境 | 證據規模 | 絕對百分比誤差 | 一致度 |
|---|---|---:|---:|---:|
| [包含 backend 常駐 GPU 記憶體校準的受控 ATen MLP 與 Transformer 參數網格][validation-static] | Ampere GPU 與 Blackwell GPU；PyTorch/CUDA 2.12/13.0 和 2.9/12.8 | 13 個工作負載，26 個觀測 | **最大 0.08%** | **≥99.92%** |
| [Qwen3-8B BF16 SDPA 推論][validation-inference] | Blackwell GPU；PyTorch 2.9.1/CUDA 12.8 | 模型載入與推論峰值 | 載入 0.0129%；**峰值 0.0672%** | 99.9871%；**99.9328%** |
| [Qwen 0.8B/2B 完整參數與 LoRA SFT][validation-sft] | Blackwell GPU；PyTorch 2.8/CUDA 12.8 | 10 個訓練案例 | **0.102%–1.921%** | **98.079%–99.898%** |
| [Qwen 0.8B/2B 原生 NF4 QLoRA][validation-qlora] | Blackwell GPU；PyTorch 2.8/CUDA 12.8 | 10 個量化訓練案例 | **0.628%–1.732%** | **98.268%–99.372%** |

這些數字需要配合測量方式理解：

- Qwen 資料以 `torch.cuda.max_memory_allocated()` 為參考，不含 CUDA context
  與 allocator 已保留但未使用的 GPU 記憶體。
- 受控 ATen 資料加入目前軟體堆疊的 backend 常駐記憶體測量值，不能用於其他環境。
- 區間是案例中的最小/最大誤差，不是平均值；評估 OOM 風險應特別注意最大低估。
- `99.x%` 一致度不等於可用記憶體餘量，容量規劃仍需加入安全餘量。

這些數字來自固定工作負載，不是通用準確率；模型、shape、backend、kernel、
allocator、軟體版本或 GPU 改變時需要重新校準。表中連結指向保留完整設定與實測
位元組數的固定版本快照；CI 會檢查本表與
[結構化證據摘要](https://github.com/FanBB2333/FakeGPU/blob/main/tests/data/memory_validation_evidence.json)
是否一致。

在真實 CUDA 主機上，可以重新執行專案維護的受控比較：

```bash
python3 scripts/validation/static_memory_validation.py \
  --output build/static-memory-validation.json \
  --markdown build/static-memory-validation.md \
  --max-underestimate-percent 5
```

無 GPU 主機加入 `--static-only` 只檢查估算流程，不產生準確性結果。將
`plan-serving` 報告與多次實體 GPU 階段峰值比較：

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

此指令保留全部樣本，每個階段取最大值。預設每階段至少五次觀測（不足時 `--strict`
傳回 1）；`ready_for_comparison` 只表示樣本數足夠，不代表 `GPU-validated`。
Plan 必須包含 `--target-profile`。

以框架對應的 runner 自動執行多次測量：

```bash
python3 -m fakegpu calibrate collect-serving \
  build/speculative-serving-plan.json \
  --repetitions 5 \
  --timeout-seconds 900 \
  --strict \
  --json build/serving-observation.json \
  -- python3 benchmark_serving.py
```

同構、非 speculative 的 Transformers 工作負載可使用內建 runner，plan 須使用
dynamic KV cache：

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

主機沒有 `/mnt/z` 時只替換 models 根目錄，保留 `Qwen/Qwen3-0.6B` 結構。
`sample-transformers`：

- 重新核對本機設定、Safetensors header 與 plan，再測量完整 prefill 和逐 token
  decode；
- 預設指標為 `torch.cuda.max_memory_reserved`（`--metric allocated` 取張量配置峰值）；
- 不接受混合請求、speculative decoding、paged/quantized cache、共享前綴或
  chunked prefill plan、量化 checkpoint 和 adapter；
- 需要在 CUDA 環境中預先安裝 PyTorch 與 Transformers。

vLLM 或自訂測量程式已有可信階段峰值時，可直接建構樣本：

```bash
python3 -m fakegpu calibrate emit-serving-sample \
  build/vllm-serving-plan.json \
  --prefill-peak-bytes 24900000000 \
  --decode-peak-bytes 25300000000 \
  --metric nvml.process_family_peak_bytes \
  --framework vllm \
  --framework-version 0.10.2
```

Python 中對應 `fakegpu.build_cuda_serving_sample(...)`。vLLM 的 cache usage 指標
是 block 使用比例而非位元組峰值，因此 benchmark 需提供真實的階段位元組數。

每次重複都會啟動新程序，runner 需輸出一個 `fakegpu.serving_peak_sample.v1`
物件：放在最後一行，或以 `FAKEGPU_SERVING_SAMPLE=` 作為單行前綴：

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
  profile、compute capability、目前次數與總次數；內建 runner 與協定建構器會自動
  產生該物件，無需將這些框架加入 FakeGPU 相依。
- Runner 報告模擬 CUDA、容量超出 profile 2% 容差、compute capability 不符、
  軟體環境變動、逾時或非零退出時，採集失敗。
- 報告不儲存原始指令參數，只記錄可執行檔名與指令指紋。
- `simulated: false` 是 runner 提供的中繼資料，不是硬體證明；聲明 `GPU-validated`
  仍需保留 benchmark 日誌並通過校準門檻。

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

`compare` 報告各階段有號/絕對誤差、區間涵蓋率與建議安全餘量。`verify` 在任一
門檻未通過時傳回 1：最大低估、誤差中位數/95 百分位數/最大值、區間涵蓋率、指定
容量下的 false-safe 判斷或工作負載一致性。結果只適用於相同的工作負載簽章、
shape、dtype、軟體堆疊與 GPU profile。

<a id="llm-reliability"></a>

### 研究情境可靠性報告

FakeGPU 依工作負載與環境簽章報告可靠性：

- `GPU-validated`：只適用於記錄中的模型 revision、shape、dtype、attention
  backend、allocator、軟體堆疊與 GPU。
- `CPU-validated`：已維護的執行或分析行為在無 GPU 環境中通過驗證。
- `Modeled`：已有分析模型，但沒有對應的實體 GPU 資料。
- `Planned`：尚未形成可公開聲明的準確率。

Serving 規劃報告提供與路徑無關的 `workload_signature`，避免校準資料在不同服務
設定之間誤用。需要重複採集真實 CUDA 資料時，使用 `calibrate observe-serving`
或 `calibrate collect-serving`。

#### 儲存庫驗證快照（2026-08-05）

在 macOS 26.5 arm64、Python 3.11.9 與 PyTorch 2.9.1 CPU 環境中執行了
`scripts/test.sh` 全部測試群組與兩個宣告式驗證 manifest；Transformers 取樣器另在
Blackwell GPU（PyTorch 2.11.0+cu128、Transformers 5.14.1）上完成實體
CUDA 驗證：

| 驗證層 | 已維護的檢查 | 結果 |
|---|---|---|
| Python runtime、估算器、CLI、schema 與 README 契約 | 完整 `pytest` 測試集 | **223 個通過** |
| 宣告式驗證矩陣 | 7 個 smoke 案例，加 39 個 research 架構、cache、generic/vLLM serving、訓練、校準與 diffusion 案例 | **46 個通過** |
| 原生函式庫攔截 | 建置、函式庫邊界、匯出符號、preload、GPU 記憶體類型、coordinator 與不支援 API 策略 | **通過** |
| FakeGPU-SMI 診斷 | 有上限的狀態發布、拓撲/NVLink/MIG 檢視、NVML peer/MIG 查詢、健康欄位與事件報告 | **通過** |
| 監控 exporter | Prometheus/JSON 快照、歷史與序列數量限制、異常狀態降級及 HTTP 介面 | **通過** |
| 原生能力清單（2026-09-05 更新） | 5 個能力群組、104 個明確 API、102 個強制執行策略的 API | **通過** |
| GPU profile 目錄 | 82 個 profile，涵蓋 15 種 compute capability | **通過** |
| CPU 數值模擬 | GEMM、cuBLASLt、批次 GEMM、BLAS1/2 與 FP16，共 8 組測試 | **通過** |
| CUDA 版 PyTorch 原生矩陣乘法 | 需要含 CUDA 的 PyTorch | **目前 CPU-only 主機未執行** |
| Transformers 實體 CUDA serving 取樣器 | Qwen/Qwen3-0.6B、BF16 SDPA、2 × 128-token prefill + 8 個生成 token，每階段 3 個獨立樣本 | **通過；`ready_for_comparison`** |

GitHub CI 在 Python 3.10–3.12 上執行 Python 測試，並在 Linux 與 macOS 上執行
原生 smoke 與 CPU simulation。取樣器一列只驗證了重複執行與 CUDA 身分，不構成
準確率聲明。

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
| Diffusion 訓練 | 快取特徵 U-Net full/LoRA、FP32 AdamW/EMA、BF16 autocast 與實測 activation/allocator 包絡 | 12 個 SD/SDXL 原生程序、獨立 holdout 與重載檢查 | 已記錄場景為 `GPU-validated` |

<a id="research-scenario-effects"></a>

#### 架構感知估算比較

下表由儲存庫中的公式產生，並由 README 契約測試檢查。它們是公式結果，不是實體
GPU 觀測，不能取代相同設定的校準；GiB 使用二進位單位。

| Research 情境 | 對照條件 | 建模結果 | 驗證方式 |
|---|---|---:|---|
| LLM 推論 KV cache | 32 層 GQA、8 個 KV head、head dim 128、batch 1、BF16；context 4K → 32K | **0.50 → 4.00 GiB**（cache 增加 8 倍） | 精確位元組公式 |
| Attention cache 架構 | 32 層、8 個活躍請求、4K BF16 paged cache；標準 head dim 128；MHA 32 個 KV head / GQA 8 個 / MQA 1 個 / MLA latent 512 + RoPE 64 | **16.00 / 4.00 / 0.50 / 1.125 GiB** | 對應架構的 cache layout 公式與四種 manifest 案例 |
| 線上 serving prefix cache | 相同 decoder、8 個活躍請求、4K prompt + 256 個生成 token、paged BF16 KV；獨立 cache → 共享 1K prefix | **4.25 → 3.38 GiB**（減少 20.6%） | 共享/私有 paged block 公式 |
| 混合 chat/completion serving | 相同 decoder；prompt 為 4K/8K/2K，生成長度為 256/512/1024；獨立 KV → 兩個 chat 請求共享 1K system prefix | **1.97 → 1.84 GiB** decode KV（減少 6.3%） | 異構請求集公式與 chat/RAG 矩陣 |
| Speculative target 呼叫 | 每次 verification 生成 5 個 draft token，並假設單 token 獨立接受率為 70% | 每個 target step 的預期輸出 token 為 **1.00 → 2.94**；此數值不表示吞吐提升 | 幾何接受率公式與雙模型記憶體測試 |
| LLM 訓練 | 8B BF16 參數、AdamW、4 張 GPU、activation checkpointing；replicated → full shard | **104.31 → 26.54 GiB**/rank（減少 74.6%） | 訓練階段模型 |
| Stable Diffusion v1.5 生成 | 512²、batch 1、FP16、CFG；所有權重常駐 → model offload | **2.61 → 1.72 GiB**（減少 34.3%） | 元件與階段模型 |
| SDXL 批次生成 | 1024²、batch 4、FP16、CFG；一般 VAE decode → VAE slicing | **26.64 → 11.64 GiB**（減少 56.3%） | 依序 VAE batch shape 模型 |
| SDXL 高解析度生成 | 2048²、batch 1、FP16、CFG；完整影像 VAE → 512² VAE tile | **26.64 → 8.28 GiB**（減少 68.9%） | 依序 VAE tile shape 模型 |
| PixArt-Sigma DiT 生成 | 1024²、batch 1、FP16、CFG、model offload；eager → SDPA 的 denoise 階段 | **2.32 → 1.28 GiB**（減少 44.7%） | patch token、交叉注意力與階段模型 |

- Diffusion profile 的元件參數量來自 [Stable Diffusion v1.5][diffusion-sd15]、
  [SDXL Base 1.0][diffusion-sdxl] 與 [PixArt-Sigma XL 2][diffusion-pixart] 的固定
  版本，不含 runtime context、allocator 碎片、backend 私有 workspace 與傳輸重疊。
- 取得匹配觀測前，報告保持 `Modeled`，`accuracy.status` 為 `uncalibrated`。
  本機架構檢查透過 `estimate-diffusion --model-dir` 使用。
- Cache 公式參考
  [Transformers cache strategies](https://huggingface.co/docs/transformers/kv_cache)；
  Speculative 模式依照
  [原始 speculative decoding 論文](https://arxiv.org/abs/2211.17192)與
  [Transformers assisted decoding](https://huggingface.co/docs/transformers/assisted_decoding)
  描述的 draft/target 流程建模；術語參考 [vLLM serving](https://docs.vllm.ai/en/stable/)。
- 請求到達分布、cache eviction、preemption、自適應 draft 長度、KV 回復 workspace、
  吞吐量與延遲不在估算範圍內。二進位 CUDA 擴充與任意 kernel 需要透過 passthrough
  或 hybrid 模式取得實體 GPU 觀測。

後續新增或更新的公開驗證資料應包含：

- 至少五次獨立執行，以及其中最大的觀測峰值；
- 每個報告階段的預測位元組數與實測位元組數；
- 將最大低估作為主要 OOM 風險指標；標記為 `GPU-validated` 時不超過 5%；
- 絕對百分比誤差的中位數、95 分位數與最大值；
- 預測區間涵蓋率，以及 false-safe 次數（判斷可執行但實際 OOM）；
- 模型 revision、完整指令、shape、dtype、backend、allocator 設定、GPU、
  driver、CUDA、PyTorch 與框架版本。

公開結果前使用 `calibrate verify` 執行這些門檻檢查；未達標的資料繼續標記為
`Modeled` 或 experimental。

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

- `-q` 顯示 FakeGPU 版本、runtime 策略、軟體版本、狀態更新時間、profile 目錄與
  原生 API 涵蓋、裝置識別、GPU 記憶體分類、allocator 活動、dispatch 追蹤及各程序
  峰值。
- `-i` 依裝置索引、UUID、PCI Bus ID 或 profile ID 篩選；`--json` 輸出完整的標準化
  資料。目前寫入 state schema v2，仍可讀取 v1。
- `--query-runtime` 以 CSV 或 JSON 輸出相同的執行期資訊；`--help-query-runtime`
  列出全部欄位。
- 原生攔截也會發布 allocation 生命週期、傳輸量、kernel launch、GEMM 呼叫與 FLOP、
  相容性事件及 unsupported API 次數。`FAKEGPU_SMI_DETAIL_LIMIT` 與
  `FAKEGPU_SMI_MAX_STATE_BYTES` 分別限制保留明細與單一狀態檔大小。

NVLink、MIG 與故障模型預設關閉，需在工作負載啟動前設定；設定無效時不會啟用
任何內容，只回報設定錯誤。

- **NVLink**：`FAKEGPU_NVLINK_GROUPS="0,1;2,3"` 為每組裝置建立全連接關係；
  `FAKEGPU_NVLINK_BANDWIDTH_GBPS` 預設為 `900`。`topo -m`、`nvlink -s`、
  `topology.*`/`nvlink.*` 欄位與原生 NVML 鏈路查詢共用該模型。
- **MIG**：`FAKEGPU_MIG_LAYOUT="DEVICE:PROFILE:MEMORY_MIB[:COUNT];..."`，例如
  `FAKEGPU_MIG_LAYOUT="0:1g.10gb:10240:2;1:2g.20gb:20480"`。每個 GPU Instance
  對應一個 Compute Instance；超出父裝置記憶體、8 個 slice 或 8 個實例的設定會被
  拒絕。實例的 used/free GPU 記憶體標示為 `unobserved`。
- **故障**：`FAKEGPU_FAULT_EVENTS="DEVICE:CODE:SEVERITY[:COUNT];..."`，例如
  `FAKEGPU_FAULT_EVENTS="0:XID_79:critical;1:NVLINK_CRC:error:3"`，嚴重度支援
  `info`、`warning`、`error` 與 `critical`。`events` 檢視同時彙整 unsupported API
  呼叫、發布失敗與過期狀態；`health.*` 欄位顯示裝置的建模狀態。

UUID 與 PCI Bus ID 是穩定的模擬識別。溫度、風扇轉速、即時功耗與硬體使用率顯示
為 `N/A`；拓撲、頻寬、故障與健康狀態屬於建模輸入，不是硬體測量或 ECC/Xid 資料。

<a id="export-monitoring-metrics"></a>

### 匯出監控指標

無需安裝監控相依套件，即可將 FakeGPU-SMI 標準化狀態匯出為 Prometheus 文字
（預設）或 JSON：

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

- `/metrics` 提供最新的裝置、程序、runtime、MIG、拓撲、健康狀態與發布器指標；
  `/healthz` 顯示資料來源與收集狀態；`/api/v1/history` 傳回近期樣本。
- 程序序列優先保留 GPU 記憶體使用量最高的程序，預設 128 個、最多 256 個，
  `--max-process-series 0` 可關閉；歷史預設 300 個樣本、最多 1,440 個。
- 資料只保存在記憶體中；長期保留請使用 Prometheus 或其他收集系統。
- 服務預設只監聽 `127.0.0.1` 且不提供身分驗證；對外開放前應設定帶身分驗證的
  proxy。匯出資料沿用 FakeGPU-SMI 對建模值與觀測值的區分。

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

原生 hybrid 模式下，CUDA runtime 與 cuBLAS 保持原生。設定了
`CUDA_VISIBLE_DEVICES`（依索引或 UUID）時會保留實體 NVML 偵測，讓框架正確解析
該選擇；否則使用建模的 NVML 裝置清單。子環境切換模式時，`fakegpu.env()` 會移除
殘留的 FakeGPU preload。

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

報告分別列出模型權重、prefill/decode 暫存記憶體、共享與私有 KV segment、runtime
開銷與可用餘量。自訂容量可用 `--device-memory-gib` 取代 `--target-profile`。
Prefix 共享需要 dynamic 或 paged cache，容量搜尋不會超過 `--max-batch-size`。

#### 對齊 vLLM 的執行期預留

vLLM 會預先配置 paged KV 池，而不是隨活躍請求逐步增長。部署目標為 vLLM 時可選擇
對應的 GPU 記憶體策略：

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

- 自動模式採用 vLLM V1 的計算方式：executor 申請的 GPU 記憶體減去 profile 得到的
  非 KV 記憶體，再向下取整為完整 block。報告區分請求實際需要的 KV 與預留池，並列出
  `num_gpu_blocks`、cache token 容量、初始化是否可行與 `max_model_len` 下的並行量。
- `--vllm-non-kv-cache-memory-bytes` 以 vLLM 啟動 profile（權重 + activation 峰值
  + 非 PyTorch 配置 + CUDA graph）取代建模的非 KV 值。
- `--vllm-kv-cache-memory-bytes` 指定每張 GPU 的精確 cache 大小；與 vLLM 一致，
  此時不再使用 `--memory-utilization`。
- `--memory-utilization` 預設 `0.92`，與
  [vLLM CacheConfig 文件](https://docs.vllm.ai/en/latest/api/vllm/config/cache/)
  一致；固定軟體版本時建議明確填入。
- 預算對應 vLLM 的 [GPU worker profile 流程](https://docs.vllm.ai/en/latest/api/vllm/v1/worker/gpu_worker/)，
  但初始可用 GPU 記憶體、其他程序、tensor/pipeline parallelism、hybrid cache group、
  allocator 碎片與版本相關 kernel 仍需在相同軟體堆疊上觀測。Speculative decoding
  目前只適用於 generic 執行期模型。

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

報告同時保留 target 與 draft 權重和兩套 KV cache，並加入保守的 lookahead slot。
接受率只影響每次 target 呼叫的預期輸出 token 數，不改變記憶體峰值，也不表示加速比。
兩個模型的 vocabulary 大小必須相同；tokenizer identity 不做驗證。

每份 serving 報告都會產生與路徑無關的 `workload_signature`，涵蓋架構、權重儲存、
請求 shape、KV/prefill 與 speculative 設定，不含模型路徑與裝置容量。
`calibrate verify` 會比較簽章並單獨檢查 `target.profile`，其他設定或 GPU 的觀測
不會通過驗證。

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

- 依清單順序逐一計算請求，同名 prefix 群組只儲存一次，在不重排的前提下接納能放入
  GPU 記憶體的最長清單前綴。
- 報告包含已接納與被拒絕的請求 ID、每個請求的暫存記憶體、並行 prefill 上界、群組
  KV segment 與容量餘量。
- 同組請求必須使用相同且大於零的 `shared_prefix_tokens`；命名群組視為已駐留的
  cache 命中。
- 請求到達時間、eviction、preemption、tensor parallelism、自適應 speculative 排程、
  KV 回復、吞吐量與延遲均未建模；取得匹配觀測前，報告保持 `Modeled` 與
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

LLM 估算器：

- 只讀取 safetensors header。`--kv-cache-strategy` 可選 `dynamic`、`static`、
  `quantized` 或 `paged`；報告分別列出邏輯儲存量、量化節省量、static 預留、
  paged 額外占用與 sliding-window 上限。
- 量化 cache 預設將最近 128 個 token 保留為計算 dtype（`--kv-cache-residual-tokens`）。
- MHA、GQA 與 MQA 使用分離的 key/value 儲存；設定同時包含 `kv_lora_rank`、
  `qk_nope_head_dim`、`qk_rope_head_dim` 與 `v_head_dim` 時辨識為 MLA（壓縮
  latent + 獨立 RoPE），MLA backend workspace 仍需校準。

Diffusion 估算器：

- 分別計算 text encoding、重複 denoising 與 VAE decode。`--model-dir` 只讀取
  `model_index.json`、元件 `config.json` 與選定 safetensors header，不匯入遠端
  程式碼，也不讀取 tensor payload。
- checkpoint 儲存位元組與執行期權重位元組分別列出。本機 Euler/PNDM scheduler
  設定決定 denoiser 呼叫數，其他 scheduler 回報 `denoiser_evaluations: null`。
- UNet、交叉注意力 Transformer（DiT/PixArt）與 SD3/Flux 類聯合注意力
  Transformer 使用不同的 activation 公式；CFG 只在存在正負分支的架構上使 batch
  加倍。
- `--weight-variant fp16|bf16|fp32` 指定權重檔案族。內建 profile 包含
  `pixart-sigma-xl-2-1024-ms` 與兩個 Stable Diffusion UNet。
- `--offload model`、`--attention-slicing`、`--vae-slicing` 與 `--vae-tiling`
  對應 Diffusers 常見 GPU 記憶體選項。沒有相同設定的實體 GPU 比較時，報告保持
  `Modeled` 與 `uncalibrated`，不給出誤差百分比或預測區間。

<p align="right">(<a href="#readme-top">返回頂端</a>)</p>

## 指令參考

| 指令 | 用途 |
|---|---|
| `fakegpu doctor` | 檢查安裝、原生函式庫、PyTorch 與 profiles |
| `fakegpu demo` | 執行小型 CPU FakeCUDA 訓練步驟 |
| `fakegpu preflight` | 將工作負載執行到指定階段並判斷 fit 或 OOM |
| `fakegpu analyze-repo` | 統計儲存庫入口與僅支援 GPU 的風險 |
| `fakegpu analyze-kernel` | 檢查 CUDA、PTX 與 SASS 資源及運算 |
| `fakegpu estimate-llm` | 估算 decoder GPU 記憶體、通訊量與 FLOP |
| `fakegpu plan-serving` | 規劃同構或混合請求準入、chunked prefill 與群組共享 prefix KV 儲存 |
| `fakegpu benchmark-serving` | 測量混合串流請求，保留用量、延遲、SSE 事件與失敗 |
| `fakegpu plan-experiments` | 展開限額參數網格並檢查實際命令 |
| `fakegpu run-experiments` | 執行或明確恢復實驗，保留每次嘗試 |
| `fakegpu compare-experiments` | 核驗原始產物、比較重複觀測並可匯出圖表 |
| `fakegpu estimate-diffusion` | 估算 diffusion 的 text encode、denoise 與 VAE decode 階段 GPU 記憶體 |
| `fakegpu plan-diffusion-training` | 規劃快取特徵 U-Net 訓練與持久儲存 |
| `fakegpu run-diffusion-training` | 在單個實體 GPU 執行快取特徵 U-Net full/LoRA 訓練 |
| `fakegpu estimate-diffusion-training` | 以相容原生觀測預算快取特徵訓練 |
| `fakegpu validate-diffusion-training` | 對照獨立原生 diffusion 訓練 holdout |
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
`python3 -m fakegpu <command> --help` 查看各指令的選項。大型報告與計畫在所有接受
JSON 路徑的位置都可以使用 `.json.gz`，壓縮輸出採用原子替換且位元組可重現。

### 微調總時間與 GPU 記憶體預估

`estimate-finetune` 讀取本機模型中繼資料與 Hugging Face Trainer/PEFT、LLaMA-Factory
或 Axolotl 設定，無須匯入訓練框架或載入權重，即可估算單卡全量、LoRA、4-bit QLoRA
的 GPU 記憶體與**訓練迴圈總時間**。

- 樣本數指過濾、切分與 packing 後的訓練序列；序列長度包含模板 token 與 padding。
- `max_steps` 優先於 epochs；`--max-steps -1` 可清除設定中的步數限制。

```bash
python -m fakegpu estimate-finetune \
  --config examples/finetuning/huggingface-lora.yaml \
  --model /path/to/Qwen3.5-0.8B --text-only \
  --profile "$TARGET_GPU_PROFILE" --num-samples 10000 --epochs 3 \
  --sequence-length 128 --json estimate.json
```

- 三類框架的設定範例位於 [`examples/finetuning/`](examples/finetuning/)。設定轉換已
  通過 CPU 測試，不代表所有框架與執行環境組合都已完成實體 GPU 基準測試。
- 分散式/offload、未涵蓋的 optimizer 或無法解析的 adapter 目標會明確報錯；分散式
  規劃使用 `plan-training`。
- Qwen3.5 的線性/完整 attention 分層計算；DeltaNet backward workspace、快速核心與
  MoE dispatch 只有部分解析涵蓋。`--text-only` 選擇多模態 checkpoint 的文字骨幹。

`profile-finetune-data` 完整掃描本機 JSONL 的每筆資料，統計內容摘要、token 長度、
截斷、padding 與序列數。原始 `text`/`messages` 使用本機 tokenizer；
`--packing concatenate` 只是 token 預算情境，不重現框架的 packing 演算法。要使用
實際訓練集計數，請匯出最終 `input_ids`（可附帶 `attention_mask` 與 `labels`）並加上
`--processed`：

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

`--dataset-report` 同時綁定樣本數與序列上限：取代設定中的樣本數、拒絕衝突的覆寫，
並要求長度上限一致。`--source-framework` 只是來源標籤。Packed 訓練需要最終處理後的
序列計數。

`run-finetune` 在一張實體 CUDA 顯示卡上呼叫已安裝的 HF Trainer/PEFT、LLaMA-Factory
或 Axolotl 原生流程（全量、LoRA 或 bitsandbytes 4-bit QLoRA；本機未量化文字
checkpoint；除下文 packing 說明外為無 packing 資料）。它依完整原生設定執行訓練
（相對路徑以設定檔目錄解析），並記錄：

- 原生前處理、有效訓練資料、實際 batch、optimizer 更新、allocator 峰值、評估/儲存
  事件與最終儲存；
- `observation.updates[].batches`：每個 microbatch 的形狀、epoch 位置、token 槽位、
  非 padding token 與有效 labels；
- `observation.batch_profile`：依完整有序的梯度累積視窗分組耗時；每組有兩個穩態
  update 後才標記 `sufficient`（僅表示樣本足夠，不是精度認證）；
- `observation.phase_events`：實際評估/儲存呼叫樹，階段總量只累計根事件；儲存後的
  檔案清單不代表磁碟寫入量。

Warmup 與統計開銷不計入穩態樣本，但包含在總牆鐘時間中。

`validate-finetune-windows` 使用獨立記錄的 holdout 驗證耗時預測：

- 至少提供兩份模型、資料、軟體及原生執行設定相同的獨立校準報告。
- 排除全域 warmup 與視窗首次出現後，每個視窗在每份校準報告中仍須有兩個樣本；
  缺少涵蓋的視窗直接拒絕。
- 只驗證歷史證據，不預測 shuffle 順序、warmup、總牆鐘時間或動態 VRAM；原生
  packing 與 MoE 視窗尚未支援。

```bash
python -m fakegpu validate-finetune-windows \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --holdout-run holdout.json --json window-validation.json
```

`validate-finetune-events` 以同樣方式驗證評估/儲存子總時間，開始、首次、重複與最終
事件分別匹配。目前支援無 packing 的 dense 文字模型、loss-only 評估，以及本機
模型/adapter 或可恢復 checkpoint 儲存；不預測事件次數或完整流程時間。

```bash
python -m fakegpu validate-finetune-events \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --holdout-run holdout.json --json event-validation.json
```

`--event-context` 也會匹配視窗前最近的評估與 checkpoint 邊界（有序呼叫、工作量、
階段及各邊界後的 update 數）。每個「視窗＋上下文」組合在每份校準報告中需要兩個
樣本。它驗證週期事件後重複出現的視窗，不預測首次儲存的恢復耗時或完整訓練時間。

```bash
python -m fakegpu validate-finetune-windows --event-context \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --holdout-run holdout.json --json event-window-validation.json
```

`--epoch-context` 也會匹配每個 microbatch 在 epoch 內的位置，用來區分包含新一輪
取數或 worker 啟動的視窗；可單獨使用，也可與 `--event-context` 組合：

```bash
python -m fakegpu validate-finetune-windows --event-context --epoch-context \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --holdout-run holdout.json --json epoch-window-validation.json
```

`plan-finetune-pipeline` 從校準執行中完整觀測到的 epoch 推導未來輪次或更新次數。
`estimate-finetune-pipeline` 組合至少兩次獨立校準，得到完整原生預算：setup、warmup、
有序訓練視窗、評估、checkpoint、Trainer 間隙與收尾。

- 計時止於原生呼叫及最後 CUDA 同步，不含 Python 啟動、報告建構或 CLI 輸出。
- 時間區間是各程序均值的極值，不是信賴區間；任一部件缺少涵蓋都會明確報錯。
- 支援無 packing、保留尾批次的循序文字資料：HF Trainer（Transformers 5.6.0）與
  LLaMA-Factory 0.9.5 需 `disable_shuffling: true`；Axolotl 0.18.0
  （Transformers 5.14.1）需 `curriculum_sampling: true`。
- 資料、模型、硬體與執行設定須與校準一致，改變資料筆數需重新校準。日誌須關閉或
  每次 update 記錄；不支援 checkpoint 輪替刪除與最佳模型重新載入。

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

預設隨機採樣或更換訓練資料時，使用經驗遷移流程：讀取原生 prepared dataloader，在
CPU 上計畫 sampler/collator 輸出，並於校準中核對實際消費的列號與輸入張量 SHA256。
支援 HF/Factory 搭配 Accelerate 1.11.0、Axolotl 搭配 Accelerate 1.13.0；隨機採樣
使用原生 data seed 加 epoch（Axolotl 0.18.0 不會將 YAML 的 `data_seed` 傳給
Trainer）。

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

- `calibrate-finetune` 固定啟動兩次全新原生程序，限制步數與執行時間，預設各自使用
  新的空 kernel 快取目錄。保留所有嘗試，以 SHA256 封存報告，支援 `--resume`，不會
  隱式增加或重跑校準。
- 純 CPU 重估預算不需 GPU 或框架，但不能超過已儲存的採樣計畫範圍。
- 工作區需保留原始絕對路徑，並放在已記錄的輸入目錄之外（巢狀路徑會被拒絕）；只支援
  有限本機資料與 POSIX 鎖語義。Axolotl/Factory 的 prepared 資料隔離在工作區的
  `prepared/` 中，恢復時拒絕原始輸入變更。
- 重複的 `--reuse-calibration-run PATH` 可重用另一份資料的相容訓練證據；啟動與收尾
  仍由目標資料的兩次新校準提供，且不依樣本數放大。不同 token/mask/監督分布之間的
  訓練成本重用是**經驗近似**；缺少涵蓋時回傳 1 並列出缺口。
- `run-finetune --planning-epochs N`（或 `--planning-max-steps N`）直接儲存 CPU 計畫；
  `plan-finetune-transfer`、`estimate-finetune-transfer`、`validate-finetune-transfer`
  無需工作區管理即可完成相同步驟。streaming、自訂採樣器與自動 batch size 調整
  尚未支援。

記憶體結果為同一 GPU 與 allocator 上完整執行的 allocated/reserved 峰值包絡，不能
推算到更小的顯示卡；`--reserve-gib` 只是使用者指定的餘量，不證明不會 OOM。
`validate-finetune-pipeline` 對照獨立 holdout 回報時間誤差與記憶體低估。

長度分組：HF/Factory 設定 `train_sampling_strategy: group_by_length`，Axolotl 設定
`group_by_length: true`，並向 `calibrate-finetune` 或 `run-finetune` 傳入
`--length-grouping-seed 123`。

- 保留原生分組演算法，使用以 `123 + epoch` 為種子的獨立 CPU 生成器；依賴全域 RNG 的
  原生分組無法重現。
- 必須指定規劃範圍。一個有限評估資料集使用同一明確種子，不推進訓練 sampler；
  拒絕自訂評估 sampler 與提前準備的 dataloader。
- sampler plan v2 把分組長度與 batch size 納入校準身分，不同長度序列需要各自的證據。

Packing：

- [packing 語義審計](tests/native/finetune_packing_semantics.py) 顯示，在極小的
  Qwen3.5 模型中，所測一般 HF/Factory 混合路徑於分塊 mask 與重設位置後仍有跨段
  影響；所測 Axolotl patch 與 Transformers 5.14.1 的 decoder 介面不相容。
- 選用的 [分段參考審計](tests/native/finetune_segment_reference.py) 檢查專案自有的
  Qwen3.5 分段介面：每段 linear attention 使用空的卷積與遞迴狀態，逐層輸出與梯度
  與獨立原生序列一致，包括 checkpoint 重算與重新載入。
- 對 dense `qwen3_5_text` LoRA，向 `run-finetune` 或 `calibrate-finetune` 傳入
  `--packing-seed 123` 啟用此路徑；需指定規劃範圍、關閉評估並啟用原生 packing
  （Factory 設定 `packing: true` 與 `neat_packing: true`，Axolotl 設定
  `sample_packing: true`）。
- HF processed 列另需 `position_ids`，`attention_mask` 使用連續正整數 segment ID
  （右側 padding 為零）；後續各段的首個 label 必須忽略。
- sampler plan v3 記錄 epoch bins、邏輯序列數、張量列數、分段幾何、dtype 與輸入
  SHA；不同資料需要獨立校準。原生 packing 報告須使用 transfer 指令。
- 不承諾 varlen 核心吞吐；full/QLoRA packing 與 grouped packing 未涵蓋。另見
  [PEFT forward 審計](tests/native/finetune_segment_peft.py) 與
  [packing 預算證據](tests/data/finetune_packing_budget_evidence.json)。

`run-finetune --cache-regime empty_kernel_cache` 要求明確設定五個互不相同、已存在且
為空的目錄：`TRITON_CACHE_DIR`、`TORCHINDUCTOR_CACHE_DIR`、`CUDA_CACHE_PATH`、
`TORCH_EXTENSIONS_DIR`、`PYTORCH_KERNEL_CACHE_PATH`。`populated_kernel_cache` 記錄
其中每個檔案的相對路徑、大小與 SHA256。runner 不建立、清理或預熱快取；預設
`uncontrolled` 不作快取狀態承諾，也不能與受控校準混用。Triton 中繼資料可能透過
絕對路徑引用原快取，必須保持這些檔案不變。

原生執行準備：

- Unpacked HF 執行（`--framework huggingface`）使用平鋪設定，以及每行恰好包含
  `input_ids`、`attention_mask`、`labels` 的 processed JSONL：等長整數陣列、二值
  mask、長度 2 到 `sequence_length`。不隱式斷詞，tokenizer 須有 pad token。參見
  [原生 LoRA 範例](examples/finetuning/huggingface-native-lora.yaml)。
- 梯度檢查點：LLaMA-Factory 使用 `disable_gradient_checkpointing: false` 並以
  `use_reentrant_gc: true` 或 `false` 選擇實作；Axolotl 使用
  `gradient_checkpointing: true` 與 `gradient_checkpointing_kwargs: {use_reentrant: true}`
  （或 `false`）。實際檢查點與量化設定納入校準身分。
- 原生訓練需要文字 checkpoint，`--text-only` 不會轉換權重。對於本機 Qwen3.5，先在
  CPU 上匯出文字 checkpoint，並產生用於整合檢查的合成 Alpaca 資料：

```bash
python tests/native/prepare_finetune_fixture.py \
  --model /path/to/Qwen3.5-0.8B \
  --output-dir /tmp/fakegpu-finetune/qwen35-text \
  --dataset-dir /tmp/fakegpu-finetune/data
```

匯出需要支援 `Qwen3_5ForCausalLM` 的 Transformers。原生範例針對 LLaMA-Factory 0.9.5
（Transformers 5.6.0）與 Axolotl 0.18.0（Transformers 5.14.1），請分別在獨立環境
安裝對應框架及相容的 CUDA PyTorch、PEFT 與資料相依。每次執行使用新的 `output_dir`；
Axolotl 前處理快取位於 `/tmp/fakegpu-finetune/axolotl-prepared`。

```bash
CUDA_VISIBLE_DEVICES=1 python -m fakegpu run-finetune \
  --config examples/finetuning/axolotl-native-lora.yaml --framework axolotl \
  --profile "$TARGET_GPU_PROFILE" --warmup-steps 2 --output run.json
python -m fakegpu estimate-finetune \
  --run-report run.json --epochs 3 --max-steps -1 --json estimate.json
```

- LLaMA-Factory 使用 `--framework llamafactory` 與
  [`llamafactory-native-lora.yaml`](examples/finetuning/llamafactory-native-lora.yaml)。
- 在同一 Python 環境重用 `--run-report`：CLI 會核對本機權重、模型設定與軟體身分，
  GPU profile 預設沿用報告值。
- 此單報告 `estimate-finetune` 介面要求固定形狀的完整 batch。改變 `--num-samples`
  只提供訓練迴圈情境，不包含資料準備與載入時間，也不含 Python 啟動或下載。

512-token NF4 QLoRA 與梯度檢查點範例見
[`llamafactory-native-qlora.yaml`](examples/finetuning/llamafactory-native-qlora.yaml) 與
[`axolotl-native-qlora.yaml`](examples/finetuning/axolotl-native-qlora.yaml)（double
quantization、BF16，需要 bitsandbytes）。以已匯出的 tokenizer 產生較長資料：

```bash
python tests/native/prepare_finetune_fixture.py \
  --model /tmp/fakegpu-finetune/qwen35-text --dataset-only \
  --dataset-dir /tmp/fakegpu-finetune/data-long \
  --sequence-length 1024 --num-samples 64
```

fixture 長度只是下限；框架會依設定截斷至 512 或 1024 tokens，執行報告記錄實際
採用的序列。

以 [`tests/native/finetune_framework_matrix.py`](tests/native/finetune_framework_matrix.py)
執行獨立校準與複測程序，序列長度、microbatch、梯度累積與 seed 沿用原生設定：

```bash
python tests/native/finetune_framework_matrix.py \
  --framework axolotl --config examples/finetuning/axolotl-native-qlora.yaml \
  --python /path/to/axolotl/venv/bin/python \
  --output-dir /tmp/fakegpu-finetune/axolotl-qlora-matrix \
  --gpu-index 1 --calibration-steps 6 --warmup-steps 2 \
  --holdout-runs 2 --holdout-steps 16
```

步數包含 warmup（預設 6 次校準更新、2 次 warmup、三次各 32 次更新的複測）。每次
訓練使用獨立輸出目錄，時間預測呼叫正式估算器並使用實際 microstep 計數。

有實體顯示卡時，可在**同一 Python 環境**先短跑取樣，再推估資料量與 epochs；
模型、GPU、精度、adapter、attention、batch、長度與累積設定須保持一致：

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

內建 probe 以 Hugging Face Trainer 與固定長度的合成文字執行，需要 Transformers、
Accelerate 與 CUDA PyTorch；adapter 另需 PEFT，QLoRA 另需 bitsandbytes。warmup
另行記錄，並保留啟動/穩態峰值。推估時間不含資料準備、模型載入、評估與 checkpoint
I/O。checkpoint 檔案會計算內容摘要，同路徑替換權重會使舊校準失效。

驗證紀錄（精確負載、版本與適用範圍見 [`refactor.md`](refactor.md)）：

- [完整流水線驗收](tests/data/finetune_pipeline_validation_evidence.json)：三框架共
  30 個原生程序，含獨立的 allocator OOM 對照。
- [採樣與遷移驗收](tests/data/finetune_transfer_validation_evidence.json)：Blackwell GPU 上 18 個 Qwen3.5-0.8B 程序，包括 25 → 73 筆資料遷移以及隨機採樣搭配週期
  eval/save；每組一次獨立 holdout。
- [Qwen3.5-2B 驗收](tests/data/finetune_qwen2b_evidence.json)：512/1024-token LoRA
  與 512-token NF4 QLoRA（啟用梯度檢查點），每條件兩次校準、兩次 holdout；總時間
  誤差 0.054%–1.578%。
- [原生框架驗證歸檔](tests/data/finetune_framework_validation_evidence.json)：先前的
  128-token LoRA，最大牆鐘誤差 Factory 3.931%、已快取的 Axolotl 1.125%。把 Axolotl
  首次執行的校準用於後續快取程序時誤差為 364.885%（首次 update 66.542 秒），因此
  模型/軟體簽章相同不代表 cache/JIT 開銷可重用。
- [擴充驗證歸檔](tests/data/finetune_extended_validation_evidence.json)：兩框架六組
  設定，涵蓋 512/1024 tokens、batch 1/2、梯度檢查點與 NF4 QLoRA，共 18 個測量程序。
- [Qwen3.5-0.8B probe 紀錄](tests/data/finetune_validation_evidence.json)：全量、LoRA、
  QLoRA 的校準與複測，可用 [`tests/native/finetune_matrix.py`](tests/native/finetune_matrix.py)
  重現。

這些短跑結果不代表通用精度或 OOM 安全。

### 線上 serving 基準測試

`benchmark-serving` 測量已執行的 completion 服務，依賴 vLLM 的 `ignore_eos` 與最終
串流 usage 欄位。token-ID 清單指定到達時間與輸出上限；報告保留 SSE 事件、失敗、
用戶端排隊與首個非空文字延遲，不重試請求，也不推斷裝置階段峰值。`--model` 填寫
服務端設定的模型名稱。

```bash
python -m fakegpu benchmark-serving \
  --base-url http://127.0.0.1:8000/v1 --model local-model \
  --requests examples/serving/online-requests.json \
  --concurrency 4 --repetitions 3 --json build/online.json.gz
```

[真實 serving 驗收](tests/data/serving_native_evidence.json) 包含 20 個 8K/32K
Transformers 程序，以及 prefix cache 關閉/開啟時的 72 個混合 vLLM 請求。32K 時
allocated 兩階段誤差為 2.02%/7.34%，但 reserved decode 低估 24.07%：張量公式無法
保證 allocator 或 OOM 安全。

### Diffusion 訓練

快取特徵 U-Net 訓練支援 full、LoRA 與可選 `--ema-decay`。`plan-diffusion-training`
綁定本機權重與 FP32 快取張量的 SHA；`run-diffusion-training` 使用一個實體 CUDA
GPU、BF16 autocast、FP32 可訓練參數與 AdamW。VAE/text encoder 訓練與原始影像
前處理不在範圍內；以下 fixture 產生隨機特徵，僅供整合驗證：

```bash
python tests/native/prepare_diffusion_fixture.py \
  --model-dir /path/to/stable-diffusion-v1-4 --output-dir /tmp/diff-fixture \
  --num-samples 5 --epochs 3 --method lora --gradient-checkpointing
CUDA_VISIBLE_DEVICES=1 python -m fakegpu run-diffusion-training \
  --plan /tmp/diff-fixture/plan.json --output-dir /tmp/diff-cal-1 \
  --json /tmp/diff-cal-1.json
CUDA_VISIBLE_DEVICES=1 python -m fakegpu run-diffusion-training \
  --plan /tmp/diff-fixture/plan.json --output-dir /tmp/diff-cal-2 \
  --json /tmp/diff-cal-2.json
python -m fakegpu estimate-diffusion-training --plan /tmp/diff-fixture/plan.json \
  --epochs 5 --calibration-run /tmp/diff-cal-1.json \
  --calibration-run /tmp/diff-cal-2.json --json /tmp/diff-budget.json.gz
```

- 缺少原生觀測時，activation 與 reserved 峰值保持未知。
- 預算要求兩個獨立程序，且未來每類視窗都有重複樣本；首次形狀開銷另計。
- `--cache-regime` 與微調流程相同（預設 `uncontrolled`）。Allocator、GPU UUID、
  軟體、輸入內容與執行身分必須一致。
- `validate-diffusion-training` 對照獨立 `--holdout-run`；
  `--max-total-time-error-percent` 與 `--max-memory-underestimate-percent` 超限時
  回傳 1。

[真實 diffusion 記錄](tests/data/diffusion_native_evidence.json) 包含 SD/SDXL
generation 觀測與 12 個訓練程序。六次訓練 holdout 的總時間誤差為 0.044%–3.185%，
allocator 包絡未低估；generation VAE allocated 峰值低估 12.0%–13.8%，reserved
最多低估 31.6%。

### 階段記憶體校準

Allocated 張量儲存與 allocator reserved 分別統計。`calibrate phase-sample` 在一個
全新 CUDA 程序中同時採集兩者，`phase-estimate` 以至少五個獨立程序封存各階段最大值，
`phase-validate` 對照之後的獨立執行。每份樣本與預算都使用新的輸出路徑。

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

- 採集支援同構 Transformers dynamic-cache serving，以及本機 SD/SDXL 的 eager/SDPA、
  attention slicing、VAE slicing、VAE tiling 及其組合，要求無 offload。
- Tiling 時，`--vae-tile-size` 同時設定原生 sample 與 latent tile 上限；實際
  processor、VAE 開關與 tile 尺寸納入原生身分。
- 計畫、選定權重與設定、GPU UUID、Torch/CUDA/cuDNN 版本、allocator 與執行實作須
  一致；不做形狀遷移，也不加隱含安全係數；kernel cache 不受控。
- 預設門檻為絕對誤差 20%、低估 10%，未通過回傳 1。離線預算/驗證只依賴標準函式庫。

驗證證據（experimental，僅適用於已測 GPU、軟體與固定輸入）：

- [階段記憶體驗收](tests/data/phase_memory_evidence.json)：補上 VAE 上採樣卷積後，
  四種解析度的 VAE allocated 誤差為 0.045%–1.172%；六個條件各五次校準、五次
  holdout（共 60 個程序）皆通過兩項門檻。
- [擴展記憶體驗證](tests/data/memory_generalization_evidence.json)：25 個單卡條件、
  74 個程序，涵蓋 SD 1.4/SDXL、Qwen3-0.6B/1.7B/Qwen2.5-0.5B、FP16/BF16/FP32、
  不同 batch、矩形影像、guidance、eager attention、slicing 與 tiling。最大 allocated
  絕對誤差 10.93%、最大低估 3.58%（門檻 20%/10%）。精確匹配條件的預算在十二次
  holdout 中全部吻合，68 次跨條件重用被拒絕。這些不是 `GPU-validated` 列：21 個
  條件不足五次觀測，最佳化模式使用的是封存的原生探針而非公開 CLI。
- [公開最佳化模式驗收](tests/data/optimized_phase_memory_evidence.json)：七個 SD/SDXL
  條件、70 個程序，210 項 holdout 比較皆與校準最大值一致；解析式 allocated 最大
  絕對誤差 6.69%、最大低估 3.38%；42 次跨模式與 21 次實際模式不匹配被拒絕。

這些結果皆不代表可移轉至新形狀、具備預測區間或實體 OOM 安全。

### 分散式規劃與實驗批次執行

FSDP2 `full_shard` 可向 `plan-training` 提供 `--parameter-shapes layout.json`，內容為
`{"shapes": [[12, 8], [12], [3, 12], [3]]}`，並提供一致的 `--parameter-bytes` 與
統一的 `--parameter-element-bytes`。它依原生每參數第 0 維 chunk 計算並以最大 rank
做預算；未提供形狀時使用均衡切分近似，不涵蓋自訂 placement。

[版本化原生驗收矩陣](tests/data/research_native_evidence.json) 記錄六組
FP32/FP16/BF16 LoRA/attention CPU_SIM 案例、CPU 與 GPU 上 tiny Qwen3 MoE 的路由與
梯度，以及兩個 CPU rank 的原生 DDP/FSDP2。實體 TP/PP/CP/EP、多 GPU ZeRO/FSDP 與
多機 collective 不在目前範圍內。重現入口為 `scripts/test.sh cpu` 與
`tests/native/research_distributed.py --help`。

實驗批次執行使用明確 argv、參數網格與重複次數，依序執行並綁定穩定執行 ID，每次
嘗試保留日誌、退出狀態、錯誤、產物 SHA 與數值指標：

```bash
python -m fakegpu plan-experiments --config examples/experiments/cpu-sweep.json \
  --workspace /tmp/research-sweep --json /tmp/research-plan.json
python -m fakegpu run-experiments --config examples/experiments/cpu-sweep.json \
  --workspace /tmp/research-sweep --json /tmp/research-results.json.gz
python -m fakegpu run-experiments --config examples/experiments/cpu-sweep.json \
  --workspace /tmp/research-sweep --resume --json /tmp/research-resumed.json
python -m fakegpu compare-experiments --workspace /tmp/research-sweep \
  --plot /tmp/research-sweep.svg --x size --metric seconds \
  --json /tmp/research-comparison.json
```

`--resume` 跳過已核驗成功項，`--resume --retry-failed` 重試失敗或中斷的項目。輸入、
原始碼或環境改變需要新工作區。指令須在前景執行，指標路徑支援巢狀 JSON。繪圖需要
Matplotlib，誤差棒是觀測最小/最大值，並非信賴區間。範例測量 CPU 工作，不代表 GPU
吞吐。

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

[發佈回歸記錄](tests/data/research_release_evidence.json) 包含 1734 項通過的 Python
測試（略過一個 CPU-only build 分支）、原生 CPU 套件、104 個封裝模組、42 個指令說明
與 68 次離線 CLI 功能檢查；152 份歷史預測逐位元組保持一致。

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
- `estimate-llm` 與 `plan-serving` 對視覺模型只計入 checkpoint 權重，不含
  vision/projector activation 與 FLOP；decoder token 數需已包含視覺 token。
- Diffusion profile 只描述固定的參考 pipeline。自訂 ControlNet、IP-Adapter、
  LoRA、safety checker、refiner、影片與 DiT 元件需要額外的元件資料及實體 GPU 驗證。
- 線上 serving 規劃不估算請求到達分布、cache eviction、preemption 或排程器重排、
  self-speculative/Medusa/EAGLE、tokenizer 重映射、tensor parallelism、吞吐量與延遲。
- Roofline 輸出是分析區間，不是實測 kernel 延遲。
- 分散式耗時包含 coordinator、記憶體複製、socket 與程序排程，不能作為 NCCL、
  NVLink 或 RDMA benchmark。
- 建模故障事件只用於報告，不會改變 CUDA 執行，也不代表真實的 ECC/Xid 資料。
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
- [x] 加入 SD/SDXL GPU 觀測、快取特徵 U-Net full/LoRA/EMA 預算及依架構區分的 DiT 記憶體模型
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
[validation-static]: https://github.com/FanBB2333/FakeGPU/blob/6efed08ce20f9628481328ef31f90195df78c7fb/docs/ai-researcher-preflight.md#4-static-aten-storage-liveness-validation
[validation-inference]: https://github.com/FanBB2333/FakeGPU/blob/6efed08ce20f9628481328ef31f90195df78c7fb/docs/llm-inference-estimation.md#maintained-qwen3-8b-result
[validation-sft]: https://github.com/FanBB2333/FakeGPU/blob/6efed08ce20f9628481328ef31f90195df78c7fb/docs/llm-sft-memory-estimation.md#blackwell-gpu-matrix
[validation-qlora]: https://github.com/FanBB2333/FakeGPU/blob/6efed08ce20f9628481328ef31f90195df78c7fb/docs/llm-sft-memory-estimation.md#native-nf4-qlora-matrices
[diffusion-sd15]: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/451f4fe16113bff5a5d2269ed5ad43b0592e9a14
[diffusion-sdxl]: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/462165984030d82259a11f4367a4eed129e94a7b
[diffusion-pixart]: https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS/tree/e102b3591cc82e97071b8b4cb90d834d0c487207
