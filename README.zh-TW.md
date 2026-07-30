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
| 給定 GPU 記憶體預算可以容納多少個同構線上 LLM 請求？ | Serving 規劃器 | 否 |
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
| 部署線上 LLM 服務前規劃容量 | 估算 continuous batch 準入、chunked prefill 暫存記憶體、共享 prefix KV block 與可用記憶體餘量 | `plan-serving`、`validate` |
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
| [包含 backend 常駐 GPU 記憶體校準的受控 ATen MLP 與 Transformer 參數網格][validation-static] | RTX 3090 Ti 與 RTX PRO 5000；PyTorch/CUDA 2.12/13.0 和 2.9/12.8 | 13 個工作負載，26 個觀測 | **最大 0.08%** | **≥99.92%** |
| [Qwen3-8B BF16 SDPA 推論][validation-inference] | RTX PRO 5000；PyTorch 2.9.1/CUDA 12.8 | 模型載入與推論峰值 | 載入 0.0129%；**峰值 0.0672%** | 99.9871%；**99.9328%** |
| [Qwen 0.8B/2B 完整參數與 LoRA SFT][validation-sft] | RTX PRO 5000；PyTorch 2.8/CUDA 12.8 | 10 個訓練案例 | **0.102%–1.921%** | **98.079%–99.898%** |
| [Qwen 0.8B/2B 原生 NF4 QLoRA][validation-qlora] | RTX PRO 5000；PyTorch 2.8/CUDA 12.8 | 10 個量化訓練案例 | **0.628%–1.732%** | **98.268%–99.372%** |

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
準確性結果。對於自己的工作負載，可以對相容的預測報告與實測報告執行：

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

#### 目前儲存庫驗證

目前儲存庫狀態於 2026-07-30 在 macOS 26.5 arm64、Python 3.11.9 與 PyTorch
2.9.1 CPU 環境中執行了 `scripts/test.sh all` 與兩個宣告式驗證 manifest：

| 驗證層 | 已維護的檢查 | 結果 |
|---|---|---|
| Python runtime、估算器、CLI、schema 與 README 契約 | 完整 `pytest` 測試集 | **191 個通過** |
| 宣告式驗證矩陣 | 6 個 smoke 案例，加 26 個 research cache、serving、訓練、校準與 diffusion 案例 | **32 個通過** |
| 原生函式庫攔截 | 建置、函式庫邊界、匯出符號、preload、GPU 記憶體類型、coordinator 與不支援 API 策略 | **通過** |
| FakeGPU-SMI 診斷 | 有上限的狀態發布、拓撲/NVLink/MIG 檢視、NVML peer/MIG 查詢、健康欄位與事件報告 | **通過** |
| 監控 exporter | Prometheus/JSON 快照、歷史與序列數量限制、異常狀態降級及 HTTP 介面 | **通過** |
| 原生能力清單 | 5 個能力群組、26 個明確 API、24 個強制執行策略的 API | **通過** |
| GPU profile 目錄 | 82 個 profile，涵蓋 15 種 compute capability | **通過** |
| CPU 數值模擬 | GEMM、cuBLASLt、批次 GEMM、BLAS1/2 與 FP16，共 8 組測試 | **通過** |
| CUDA 版 PyTorch 原生矩陣乘法 | 需要含 CUDA 的 PyTorch | **目前 CPU-only 主機未執行** |

GitHub CI 也會在 Python 3.10–3.12 上執行 Python 測試，並在 Linux 與 macOS
上執行原生 smoke 與 CPU simulation。上文的實體 GPU 結果來自對應的固定版本
驗證快照；本次檢查驗證了結構化證據與計算公式，沒有在目前 CPU-only 主機上
重新測量。

#### 已維護的研究工作負載矩陣

| 工作負載類型 | 已涵蓋變化 | 驗證依據 | 狀態 |
|---|---|---|---|
| 離線 decoder 推論 | Qwen3-8B、BF16、SDPA、模型載入、prefill 與 decode 峰值 | RTX PRO 5000 預測值與實測值 | `GPU-validated` |
| 完整參數與 adapter SFT | Qwen 0.8B/2B 完整參數微調與 LoRA | 十個 RTX PRO 5000 訓練案例 | `GPU-validated` |
| 量化 adapter SFT | Qwen 0.8B/2B 原生 NF4 QLoRA | 十個 RTX PRO 5000 訓練案例 | `GPU-validated` |
| 通用 decoder 分析 | Dense/MoE 中繼資料、adapter、量化 checkpoint、eager/SDPA attention、KV cache 與 expert-parallel 通訊量 | 公式、fixture 與 CLI 迴歸測試 | `CPU-validated` + `Modeled` |
| 分散式訓練規劃 | DeepSpeed、Accelerate、FSDP/FSDP2、分片、checkpointing 與 CPU/NVMe offload | 設定、位元組計算、拓撲與 trace 測試 | `CPU-validated` + `Modeled` |
| KV cache 配置 | Dynamic 增長、static 預留、2/4/8-bit quantized 儲存、paged block 取整與 sliding-window 上限 | 公式、API、`--kv-cache-strategy` CLI 與 `tests/data/research_validation.yaml` 矩陣測試 | `CPU-validated` + `Modeled` |
| 線上 serving 容量 | 同構 continuous batching、chunked prefill、prefix caching、paged KV block、profile 或明確容量，以及準入餘量 | Checkpoint header、架構公式、`plan-serving`、單元測試與 `tests/data/research_validation.yaml` | `CPU-validated` + `Modeled` |
| Diffusion 圖像生成 | Stable Diffusion v1.5、SDXL Base 與 PixArt-Sigma；本機 UNet、交叉注意力 DiT、SD3/Flux 類聯合注意力 Transformer；CFG、offload、attention/VAE slicing 與 VAE tiling | 固定版本與本機元件 header、架構設定、階段公式、CLI/單元測試及 `tests/data/research_validation.yaml` | `CPU-validated` + `Modeled` |
| Speculative decoding | Draft 模型權重、draft KV cache、接受率與 target verification batch | 尚無專用估算器或專案維護的實體 GPU 資料 | `Planned` |
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
| 線上 serving prefix cache | 相同 decoder、8 個活躍請求、4K prompt + 256 個生成 token、paged BF16 KV；獨立 cache → 共享 1K prefix | **4.25 → 3.38 GiB**（減少 20.6%） | 共享/私有 paged block 公式 |
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
提供的工作負載形態。Serving 規劃器只估算同構活躍請求的記憶體準入；請求到達
分布、preemption、吞吐量、延遲與 speculative decoding 不在目前估算範圍內。
相關術語參考 [vLLM serving](https://docs.vllm.ai/en/stable/)。CPU FakeCUDA
不執行二進位 CUDA 擴充或任意 kernel；這類工作負載需要分析結果，並透過
passthrough 或 hybrid 模式取得實體 GPU 觀測值。

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
```

詳細報告包含 FakeGPU 版本、runtime backend 與策略、Python/PyTorch/CUDA 版本、
狀態更新時間、profile 目錄與原生 API 涵蓋情形、模擬裝置識別、運算規格、GPU
記憶體分類、allocator 活動、dispatch 追蹤及各程序峰值。`-i` 可以依裝置索引、
UUID、PCI Bus ID 或 profile ID 篩選；`--json` 會輸出完整的標準化資料。目前
發布 state schema v2，同時仍可讀取 v1 狀態檔。

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

目前模型假設所有活躍請求的長度相同。請求到達時間、混合長度、cache eviction、
preemption、tensor parallelism、speculative draft 模型、吞吐量與延遲會列入
未建模項目。取得相同設定的線上服務觀測前，`validation_status` 保持為
`Modeled`，`accuracy.status` 保持為 `uncalibrated`。

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

| 指令 | 用途 |
|---|---|
| `fakegpu doctor` | 檢查安裝、原生函式庫、PyTorch 與 profiles |
| `fakegpu demo` | 執行小型 CPU FakeCUDA 訓練步驟 |
| `fakegpu preflight` | 將工作負載執行到指定階段並判斷 fit 或 OOM |
| `fakegpu analyze-repo` | 統計儲存庫入口與僅支援 GPU 的風險 |
| `fakegpu analyze-kernel` | 檢查 CUDA、PTX 與 SASS 資源及運算 |
| `fakegpu estimate-llm` | 估算 decoder GPU 記憶體、通訊量與 FLOP |
| `fakegpu plan-serving` | 規劃 continuous batch 記憶體準入、chunked prefill 與共享 prefix KV 儲存 |
| `fakegpu estimate-diffusion` | 估算 diffusion 的 text encode、denoise 與 VAE decode 階段 GPU 記憶體 |
| `fakegpu estimate-roofline` | 產生與 profile 相關的分析延遲區間 |
| `fakegpu plan-training` | 統一分散式訓練設定並估算單 rank GPU 記憶體 |
| `fakegpu simulate-topology` | 模擬 collective 路由與鏈路競爭 |
| `fakegpu replay-trace` | 彙整運算、通訊、等待與 GPU 記憶體時間軸 |
| `fakegpu calibrate` | 比較 GPU 記憶體報告並執行可靠性門檻檢查 |
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
- Diffusion profile 只描述固定的參考 pipeline。自訂 ControlNet、IP-Adapter、
  LoRA、safety checker、refiner、影片與 DiT 元件需要額外的元件資料及對應的
  實體 GPU 驗證。
- 線上 serving 規劃採用同構活躍請求池，尚未建模請求到達、混合長度、cache
  eviction、preemption、speculative decoding、tensor parallelism、吞吐量與
  延遲。
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
- [x] 透過 FakeGPU-SMI 發布原生 runtime 的即時狀態與活動
- [x] 加入建模拓撲、NVLink 檢視與 NVML peer 查詢
- [x] 加入建模故障注入、健康狀態與可靠性事件檢視
- [x] 加入建模 MIG 檢視與原生 NVML MIG handle 查詢
- [x] 為裝置、程序與 runtime 匯出有數量上限的指標及 Prometheus 記憶體歷史
- [x] 加入 continuous batching、chunked prefill 與 prefix caching 的線上 serving 容量模型
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
[validation-sft]: https://github.com/FanBB2333/FakeGPU/blob/df254c21eebc0a5bbf13992f3f5a8e995cfa8708/docs/llm-sft-memory-estimation.md#rtx-pro-5000-matrix
[validation-qlora]: https://github.com/FanBB2333/FakeGPU/blob/df254c21eebc0a5bbf13992f3f5a8e995cfa8708/docs/llm-sft-memory-estimation.md#native-nf4-qlora-matrices
[diffusion-sd15]: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/451f4fe16113bff5a5d2269ed5ad43b0592e9a14
[diffusion-sdxl]: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/462165984030d82259a11f4367a4eed129e94a7b
[diffusion-pixart]: https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS/tree/e102b3591cc82e97071b8b4cb90d834d0c487207
