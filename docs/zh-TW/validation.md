# 驗證與可靠性

[文件索引](README.md) · [English](../en/validation.md) ·
[简体中文](../zh-CN/validation.md)

本頁整理 FakeGPU 記憶體估算背後的實體 GPU 證據，並說明每類工作負載如何標示可靠性。
想自己產生新證據，請參閱[對照實體 GPU 校準](calibration.md)。

公開實驗記錄只標示 GPU 架構。profile 目錄中支援的型號表示相容能力，與測試所用的
硬體無關。

<a id="memory-estimation-evidence"></a>

## 實體 GPU 記憶體估算證據

> [!NOTE]
> 在以下已記錄的驗證範圍內，依軟體堆疊校準後的靜態估算器在 26 個受控 GPU 觀測上
> 的誤差不超過 **0.08%**。十個 Qwen 完整參數/LoRA SFT 案例的誤差不超過 **1.921%**。

誤差依 `|預測值 - 實測值| / 實測值 × 100%` 計算，一致度為 `100% - 誤差`。

| 已驗證範圍 | 實體 GPU 參考環境 | 證據規模 | 絕對百分比誤差 | 一致度 |
|---|---|---:|---:|---:|
| [包含 backend 常駐 GPU 記憶體校準的受控 ATen MLP 與 Transformer 參數網格][validation-static] | Ampere GPU 與 Blackwell GPU，PyTorch/CUDA 2.12/13.0 和 2.9/12.8 | 13 個工作負載，26 個觀測 | **最大 0.08%** | **≥99.92%** |
| [Qwen3-8B BF16 SDPA 推論][validation-inference] | Blackwell GPU，PyTorch 2.9.1/CUDA 12.8 | 模型載入與推論峰值 | 載入 0.0129%；**峰值 0.0672%** | 99.9871%；**99.9328%** |
| [Qwen 0.8B/2B 完整參數與 LoRA SFT][validation-sft] | Blackwell GPU，PyTorch 2.8/CUDA 12.8 | 10 個訓練案例 | **0.102%–1.921%** | **98.079%–99.898%** |
| [Qwen 0.8B/2B 原生 NF4 QLoRA][validation-qlora] | Blackwell GPU，PyTorch 2.8/CUDA 12.8 | 10 個量化訓練案例 | **0.628%–1.732%** | **98.268%–99.372%** |

閱讀這些數字時請注意：

- Qwen 資料以 `torch.cuda.max_memory_allocated()` 為參考，不含 CUDA context 與
  allocator 已保留但未使用的記憶體。
- ATen 資料加入了在該軟體堆疊上實測的 backend 常駐記憶體。請勿套用到其他環境。
- 區間取各案例的最小與最大誤差，沒有取平均。評估 OOM 風險時，真正要注意的是低估。
- `99.x%` 一致度不等於可用的記憶體餘量。容量規劃仍需加入安全餘量。

這些數字來自固定的工作負載，不能當作通用準確率。模型、shape、backend、kernel、
allocator、軟體版本或 GPU 改變時，都需要重新校準。表中連結指向保留完整設定與實測
位元組數的固定版本快照。CI 會檢查本表與
[結構化證據摘要](../../tests/data/memory_validation_evidence.json)是否一致。

<a id="llm-reliability"></a>

## 研究工作負載可靠性

FakeGPU 依工作負載與環境簽章回報可靠性：

- `GPU-validated`：只適用於記錄中的模型 revision、shape、dtype、attention
  backend、allocator、軟體堆疊與 GPU。
- `CPU-validated`：已維護的執行或分析行為在無 GPU 環境中通過驗證。
- `Modeled`：已有分析模型，但沒有對應的實體 GPU 證據。
- `Planned`：目前還沒有可公開聲明的準確率。

Serving 規劃報告帶有與路徑無關的 `workload_signature`，校準證據因此不會在不知不覺間
套用到其他服務設定。需要重複採集真實 CUDA 資料時，使用 `calibrate observe-serving`
或 `calibrate collect-serving`。

### 驗證快照（2026-08-05）

在 macOS 26.5 arm64、Python 3.11.9 與 PyTorch 2.9.1 CPU 環境中執行了
`scripts/test.sh` 全部測試組與兩個驗證 manifest。Transformers 取樣器另在
Blackwell GPU（PyTorch 2.11.0+cu128、Transformers 5.14.1）上執行過。

| 驗證層 | 已維護的檢查 | 結果 |
|---|---|---|
| Python runtime、估算器、CLI、schema 與文件契約 | 完整 `pytest` 測試集 | **223 個通過** |
| 宣告式驗證矩陣 | 7 個 smoke 案例，加 39 個 research 架構、cache、generic/vLLM serving、訓練、校準與 diffusion 案例 | **46 個通過** |
| 原生函式庫攔截 | 建置、函式庫邊界、匯出符號、preload、GPU 記憶體類型、coordinator 與不支援 API 策略 | **通過** |
| FakeGPU-SMI 診斷 | 有上限的狀態發布、拓撲/NVLink/MIG 檢視、NVML peer/MIG 查詢、健康欄位與事件報告 | **通過** |
| 監控 exporter | Prometheus/JSON 快照、歷史與序列數量限制、異常狀態降級及 HTTP 介面 | **通過** |
| 原生能力清單（2026-10-21 更新） | 5 個能力群組、107 個明確 API、94 個強制執行策略的 API | **通過** |
| GPU profile 目錄 | 307 個 profile，涵蓋 16 種 compute capability（目錄更新於 2026-10-16） | **通過** |
| CPU 數值模擬 | GEMM、cuBLASLt、批次 GEMM、BLAS1/2 與 FP16，共 8 組測試 | **通過** |
| CUDA 版 PyTorch 原生矩陣乘法 | 需要含 CUDA 的 PyTorch | **這台 CPU-only 主機未執行** |
| Transformers 實體 CUDA serving 取樣器 | Qwen/Qwen3-0.6B、BF16 SDPA、2 × 128-token prefill + 8 個生成 token，每階段 3 個獨立樣本 | **通過；`ready_for_comparison`** |

GitHub CI 在 Python 3.10–3.12 上執行 Python 測試，並在 Linux 與 macOS 上執行
原生 smoke 與 CPU simulation。取樣器一列只檢查重複執行與 CUDA 身分，與準確率無關。

### 已維護的研究工作負載矩陣

| 工作負載類型 | 已涵蓋變化 | 驗證依據 | 狀態 |
|---|---|---|---|
| 離線 decoder 推論 | Qwen3-8B、BF16、SDPA、模型載入、prefill 與 decode 峰值 | Blackwell GPU 預測值與實測值 | `GPU-validated` |
| 完整參數與 adapter SFT | Qwen 0.8B/2B 完整參數微調與 LoRA | 十個 Blackwell GPU 訓練案例 | `GPU-validated` |
| 量化 adapter SFT | Qwen 0.8B/2B 原生 NF4 QLoRA | 十個 Blackwell GPU 訓練案例 | `GPU-validated` |
| 通用 decoder 分析 | Dense/MoE 中繼資料；MHA、GQA、MQA 與壓縮 latent MLA；adapter、量化 checkpoint、eager/SDPA attention、KV cache 與 expert-parallel 通訊量 | 公式、fixture、CLI 迴歸測試與四種架構矩陣 | `CPU-validated` + `Modeled` |
| 分散式訓練規劃 | DeepSpeed、Accelerate、FSDP/FSDP2、分片、checkpointing 與 CPU/NVMe offload | 設定、位元組計算、拓撲與 trace 測試 | `CPU-validated` + `Modeled` |
| KV cache 配置 | Dynamic 增長、static 預留、2/4/8-bit quantized 儲存、paged block 取整與 sliding-window 上限 | 公式、API、`--kv-cache-strategy` CLI 與 `tests/data/research_validation.yaml` 矩陣測試 | `CPU-validated` + `Modeled` |
| 線上 serving 容量 | 同構與混合長度請求集，涵蓋 continuous batching、依序準入、chunked prefill、群組 prefix caching、paged KV block、generic 或 vLLM 執行期預算、profile 或明確 KV 容量，以及準入餘量 | Checkpoint header、架構與 block 配置公式、`plan-serving --runtime vllm`、`--vllm-kv-cache-memory-bytes`、`--vllm-non-kv-cache-memory-bytes`、單元測試，以及 `tests/data/research_validation.yaml` 中的 generic/vLLM chat/RAG 案例 | `CPU-validated` + `Modeled` |
| Diffusion 圖像生成 | Stable Diffusion v1.5、SDXL Base 與 PixArt-Sigma；本機 UNet、交叉注意力 DiT、SD3/Flux 類聯合注意力 Transformer；CFG、offload、attention/VAE slicing 與 VAE tiling | 固定版本與本機元件 header、架構設定、階段公式、CLI/單元測試，以及 `tests/data/research_validation.yaml` 中的三種本機架構矩陣 | `CPU-validated` + `Modeled` |
| speculative decoding | 獨立自迴歸 draft 模型權重與 KV cache、target lookahead KV、批次 verification 暫存記憶體、可選的固定接受率假設，以及同構/混合請求準入 | Checkpoint header、架構公式、`plan-serving --draft-model-dir`、`--speculative-tokens`、`--speculative-acceptance-rate` 與單元測試，尚無專案維護的實體 GPU 觀測 | `CPU-validated` + `Modeled` |
| 多 GPU LLM 執行 | TP、PP、CP、EP、MoE 負載不均衡，以及 FSDP/ZeRO 組合執行 | 目前只有分析拓撲與 coordinator 驗證 | `Modeled` |
| Diffusion 訓練 | 快取特徵 UNet full/LoRA、FP32 AdamW/EMA 儲存、BF16 autocast 與原生 activation/allocator 包絡 | 12 個 SD/SDXL 原生程序、獨立 holdout 與重載檢查 | 已記錄的案例為 `GPU-validated` |

<a id="research-scenario-effects"></a>

### 架構感知估算比較

下表由儲存庫中的公式產生，並由文件契約測試檢查。這些是公式結果，沒有實體 GPU
觀測，不能取代相同設定的校準。GiB 使用二進位單位。

| Research 情境 | 對照條件 | 建模結果 | 驗證方式 |
|---|---|---:|---|
| LLM 推論 KV cache | 32 層 GQA、8 個 KV head、head dim 128、batch 1、BF16；context 4K → 32K | **0.50 → 4.00 GiB**（cache 增加 8 倍） | 精確位元組公式 |
| Attention cache 架構 | 32 層、8 個活躍請求、4K BF16 paged cache；標準 head dim 128；MHA 32 個 KV head / GQA 8 個 / MQA 1 個 / MLA latent 512 + RoPE 64 | **16.00 / 4.00 / 0.50 / 1.125 GiB** | 對應架構的 cache layout 公式與四種 manifest 案例 |
| 線上 serving prefix cache | 相同 decoder、8 個活躍請求、4K prompt + 256 個生成 token、paged BF16 KV；獨立 cache → 共享 1K prefix | **4.25 → 3.38 GiB**（減少 20.6%） | 共享/私有 paged block 公式 |
| 混合 chat/completion serving | 相同 decoder；prompt 為 4K/8K/2K，生成長度為 256/512/1024；獨立 KV → 兩個 chat 請求共享 1K system prefix | **1.97 → 1.84 GiB** decode KV（減少 6.3%） | 異構請求集公式與 chat/RAG 矩陣 |
| Speculative target 呼叫 | 每次 verification 生成 5 個 draft token，並假設單 token 獨立接受率為 70% | 每個 target step 的預期輸出 token 為 **1.00 → 2.94**，此數值不代表吞吐量提升 | 幾何接受率公式與雙模型記憶體測試 |
| LLM 訓練 | 8B BF16 參數、AdamW、4 張 GPU、activation checkpointing；replicated → full shard | **104.31 → 26.54 GiB**/rank（減少 74.6%） | 訓練階段模型 |
| Stable Diffusion v1.5 生成 | 512²、batch 1、FP16、CFG；所有權重常駐 → model offload | **2.61 → 1.72 GiB**（減少 34.3%） | 元件與階段模型 |
| SDXL 批次生成 | 1024²、batch 4、FP16、CFG；一般 VAE decode → VAE slicing | **26.64 → 11.64 GiB**（減少 56.3%） | 依序 VAE batch shape 模型 |
| SDXL 高解析度生成 | 2048²、batch 1、FP16、CFG；完整影像 VAE → 512² VAE tile | **26.64 → 8.28 GiB**（減少 68.9%） | 依序 VAE tile shape 模型 |
| PixArt-Sigma DiT 生成 | 1024²、batch 1、FP16、CFG、model offload；eager → SDPA 的 denoise 階段 | **2.32 → 1.28 GiB**（減少 44.7%） | patch token、交叉注意力與階段模型 |

- Diffusion profile 的元件參數量來自 [Stable Diffusion v1.5][diffusion-sd15]、
  [SDXL Base 1.0][diffusion-sdxl] 與 [PixArt-Sigma XL 2][diffusion-pixart] 的固定
  版本，不含 runtime context、記憶體碎片、私有 workspace 與傳輸重疊。
- 取得匹配觀測前，報告保持 `Modeled`，`accuracy.status` 為 `uncalibrated`。
  `estimate-diffusion --model-dir` 可用來檢查本機架構。
- Cache 公式參考
  [Transformers cache strategies](https://huggingface.co/docs/transformers/kv_cache)。
  Speculative 模式依照
  [原始 speculative decoding 論文](https://arxiv.org/abs/2211.17192)與
  [Transformers assisted decoding](https://huggingface.co/docs/transformers/assisted_decoding)
  描述的 draft/target 流程建模。serving 術語參考 [vLLM serving](https://docs.vllm.ai/en/stable/)。
- 請求到達分布、cache eviction、preemption、自適應 draft 長度、KV 回復 workspace、
  吞吐量與延遲不在估算範圍內。二進位 CUDA 擴充與任意 kernel 需要透過 passthrough
  或 hybrid 模式取得實體 GPU 觀測。

### 發布新的驗證資料

新增或更新的公開驗證資料應記錄：

- 至少五次獨立觀測，以及其中最大的觀測峰值
- 每個報告階段的預測位元組數與實測位元組數
- 將最大低估作為主要 OOM 風險指標，標記為 `GPU-validated` 時不超過 5%
- 絕對百分比誤差的中位數、95 百分位數與最大值
- 預測區間涵蓋率，以及 false-safe 判斷次數（預測可執行但實際 OOM）
- 模型 revision、指令、shape、dtype、backend、allocator 設定、GPU、driver、CUDA、
  PyTorch 與框架版本

公開結果前，請用 `calibrate verify` 執行這些門檻檢查。未達標的資料繼續標記為
`Modeled` 或 experimental。

[validation-static]: https://github.com/FanBB2333/FakeGPU/blob/cc7f221356b582e3e06618f226d9e05bd7da64ef/docs/ai-researcher-preflight.md#4-static-aten-storage-liveness-validation
[validation-inference]: https://github.com/FanBB2333/FakeGPU/blob/cc7f221356b582e3e06618f226d9e05bd7da64ef/docs/llm-inference-estimation.md#maintained-qwen3-8b-result
[validation-sft]: https://github.com/FanBB2333/FakeGPU/blob/cc7f221356b582e3e06618f226d9e05bd7da64ef/docs/llm-sft-memory-estimation.md#blackwell-gpu-matrix
[validation-qlora]: https://github.com/FanBB2333/FakeGPU/blob/cc7f221356b582e3e06618f226d9e05bd7da64ef/docs/llm-sft-memory-estimation.md#native-nf4-qlora-matrices
[diffusion-sd15]: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/451f4fe16113bff5a5d2269ed5ad43b0592e9a14
[diffusion-sdxl]: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/462165984030d82259a11f4367a4eed129e94a7b
[diffusion-pixart]: https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS/tree/e102b3591cc82e97071b8b4cb90d834d0c487207
