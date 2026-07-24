<a id="readme-top"></a>

<div align="center">

# FakeGPU

**無需正式 GPU 叢集，即可驗證 CUDA 應用程式、估算 GPU 記憶體並模擬分散式 GPU 工作流程。**

[![Test][test-shield]][test-url]
[![Release][release-shield]][release-url]
[![Python][python-shield]][python-url]
[![License][license-shield]][license-url]

[English](README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md)

[瀏覽文件](https://fanbb2333.github.io/FakeGPU/) ·
[回報問題](https://github.com/FanBB2333/FakeGPU/issues/new?labels=bug) ·
[提出功能建議](https://github.com/FanBB2333/FakeGPU/issues/new?labels=enhancement)

</div>

![FakeGPU 工作流程：使用 CPU 執行 PyTorch、切換 GPU 設定、記憶體預檢與靜態工作負載估算](docs/assets/readme/tldr-workflows.png)

> [!IMPORTANT]
> FakeGPU 用於開發、測試與容量規劃，無法讓任意 CUDA kernel 取得與實體
> GPU 相同的數值與效能。TCP 測試結果也不能取代 NCCL、NVLink 或 RDMA
> 效能測試。

## 目錄

1. [專案介紹](#專案介紹)
   - [FakeGPU 能回答哪些問題](#fakegpu-能回答哪些問題)
   - [運作方式](#運作方式)
   - [主要技術](#主要技術)
2. [快速開始](#快速開始)
   - [環境需求](#環境需求)
   - [安裝](#安裝)
   - [檢查安裝結果](#檢查安裝結果)
3. [使用方式](#使用方式)
   - [使用 FakeCUDA 執行 PyTorch](#使用-fakecuda-執行-pytorch)
   - [攔截原生 CUDA 函式庫](#攔截原生-cuda-函式庫)
   - [執行前檢查 GPU 記憶體](#執行前檢查-gpu-記憶體)
   - [分析程式碼儲存庫或模型](#分析程式碼儲存庫或模型)
   - [檢視虛擬 GPU 記憶體](#檢視虛擬-gpu-記憶體)
   - [透過 TCP 模擬多台主機](#透過-tcp-模擬多台主機)
4. [功能範圍](#功能範圍)
   - [指令速查](#指令速查)
   - [執行模式](#執行模式)
5. [GPU Profiles](#gpu-profiles)
6. [報告與驗證](#報告與驗證)
7. [架構](#架構)
8. [限制](#限制)
9. [開發計畫](#開發計畫)
10. [文件](#文件)
11. [參與貢獻](#參與貢獻)
12. [授權](#授權)
13. [致謝](#致謝)

## 專案介紹

FakeGPU 是一套 CUDA、CUDA Runtime、cuBLAS、NVML 與 NCCL 攔截及分析工具。
應用程式可以偵測可設定的 NVIDIA 風格裝置；已維護的運算會在 CPU 上執行；
記憶體、通訊與相容性事件會寫入結構化報告。對於不適合直接載入的工作負載，
FakeGPU 也提供靜態記憶體與運算量估算。

本專案適用於 CI、本機開發、相容性測試、容量規劃與可重複實驗。模擬及分析
功能不需要實體 GPU；passthrough、hybrid 與校準實驗需要真實 CUDA 環境。

### FakeGPU 能回答哪些問題

| 問題 | 建議入口 | 需要實體 GPU |
|---|---|---:|
| PyTorch 程式碼能否依預期執行 CUDA 相關控制流程？ | Python FakeCUDA runtime | 否 |
| 未修改的程式能否偵測並呼叫 CUDA 系列動態函式庫？ | 原生函式庫攔截 | 否 |
| 某個工作負載能否放入指定的 GPU profile？ | Preflight 或靜態記憶體估算 | 否 |
| LLM 的權重、KV cache、Adapter 或 MoE 記憶體大約是多少？ | LLM 估算器 | 否 |
| 儲存庫中有哪些 GPU 入口、相依套件與原生擴充？ | 儲存庫分析器 | 否 |
| 指定 profile 的運算/頻寬延遲範圍是多少？ | Roofline 估算器 | 否 |
| 多 rank 控制流程、通訊與復原是否符合預期？ | 分散式模擬器 | 否 |
| 估算值與真實 CUDA 執行結果相差多少？ | Passthrough/hybrid 校準 | 是 |

### 運作方式

FakeGPU 提供三條互相配合的路徑：

| 路徑 | 應用程式看到的內容 | 實際執行方式 |
|---|---|---|
| **Python FakeCUDA** | CUDA 裝置、CUDA 風格 tensor、記憶體 API 與常見訓練流程 | `FakeCudaTensor` 將已維護的 PyTorch 運算交給 CPU |
| **原生函式庫攔截** | `libcuda`、`libcudart`、`libcublas`、`libnvidia-ml` 與 `libnccl` 符號 | 已維護的呼叫使用主機記憶體或 CPU 運算；不支援的行為會被分類並記錄 |
| **分析與報告** | 記憶體、FLOP、Roofline、拓撲與通訊報告 | 分析 ATen graph、safetensors metadata、執行追蹤、校準資料與 coordinator 事件 |

### 主要技術

- Python 3.10+：runtime、估算器、CLI 與報告
- C++17 與 CMake：原生攔截函式庫和分散式 coordinator
- PyTorch：CPU FakeCUDA 執行與 ATen graph 擷取
- YAML/JSON Schema：GPU profiles、測試矩陣與報告

<p align="right">(<a href="#readme-top">返回頂端</a>)</p>

## 快速開始

### 環境需求

- Linux 或 macOS
- Python 3.10 或更新版本
- CMake 3.14 或更新版本
- 支援 C++17 的編譯器（Debian/Ubuntu 可安裝 `build-essential`，macOS
  可安裝 Xcode Command Line Tools）
- Python FakeCUDA 路徑需要 PyTorch

### 安裝

複製儲存庫：

```bash
git clone https://github.com/FanBB2333/FakeGPU.git
cd FakeGPU
```

建置並驗證原生函式庫：

```bash
cmake -S . -B build
cmake --build build -j
./ftest smoke
```

安裝包含原生函式庫的 Python 套件：

```bash
python3 -m pip install .
```

從原始碼目錄開發時，可安裝驗證相依套件並設定 `PYTHONPATH`：

```bash
python3 -m pip install PyYAML jsonschema pytest
export PYTHONPATH="$PWD"
```

### 檢查安裝結果

```bash
python3 -m fakegpu doctor --list-profiles
python3 -m fakegpu demo --profile l4
```

`doctor` 檢查 profile 目錄、原生函式庫與 PyTorch 環境。`demo` 在 CPU 上完成
一個小型 forward、backward 與 optimizer step，同時讓程式看到 CUDA 裝置。

<p align="right">(<a href="#readme-top">返回頂端</a>)</p>

## 使用方式

### 使用 FakeCUDA 執行 PyTorch

需要在匯入 PyTorch 前初始化 FakeGPU：

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

此路徑適合檢查裝置配置、訓練控制流程、例外處理、記憶體統計與框架相容性。
運算由 CPU 實際完成，大型模型的執行速度可能遠低於 CUDA。

### 攔截原生 CUDA 函式庫

`fgpu` 會為未修改的指令設定動態函式庫預載環境：

```bash
./fgpu --profile a100 --device-count 2 python3 your_script.py
./fgpu --devices "a100:2,h100:2" python3 your_script.py
./fgpu --mode simulate --unsupported-api error python3 your_script.py
```

不支援的原生呼叫有三種明確策略：

| 策略 | 行為 |
|---|---|
| `warn` | 每個不支援的 API 回報一次，並在可行時繼續執行 |
| `error` | 回傳 `cudaErrorNotSupported` 或 `CUDA_ERROR_NOT_SUPPORTED` |
| `allow` | 記錄事件，但不顯示警告 |

### 執行前檢查 GPU 記憶體

將指令執行到指定階段，並產生 JSON、Markdown、stdout 與 stderr 檔案：

```bash
python3 -m fakegpu preflight \
  --runtime fakecuda \
  --profile a100 \
  --stage forward \
  --report-dir preflight-report \
  --allocation-stacks \
  --strict \
  -- python3 train.py --small-config
```

Preflight 會統計已執行路徑中可見的 parameter、buffer、gradient、
optimizer state、activation、tensor alias、dispatch 建立的 storage、
saved tensor、快取配置器狀態與有上下界的 workspace。

不配置真實 CUDA 記憶體的 ATen graph 估算：

```python
from fakegpu import estimate_module_memory, require_workspace_coverage

report = estimate_module_memory(
    model,
    (example_input,),
    mode="training",
    optimizer="adamw",
    target_device="auto",
)

require_workspace_coverage(
    report,
    minimum_fraction=1.0,
    allow_extrapolated=False,
    require_upper_bound=True,
)
print(report["estimated_peak_interval_bytes"])
```

### 分析程式碼儲存庫或模型

```bash
# 尋找入口、訓練框架、CUDA 原始碼與驗證風險。
fakegpu analyze-repo .

# 估算權重、KV cache、暫存 tensor、Adapter 與 MoE 記憶體。
fakegpu estimate-llm \
  --model-dir /models/qwen \
  --batch-size 1 \
  --prompt-tokens 128 \
  --generated-tokens 32 \
  --dtype bfloat16 \
  --target-profile a100 \
  --json build/llm-estimate.json

# 估算與 profile 相關的分析延遲區間。
fakegpu estimate-roofline \
  --profile a100 \
  --flops 1000000000000 \
  --memory-bytes 4000000000 \
  --launch-count 100

# 根據能力清單檢查原始碼與已編譯的原生符號。
fakegpu capabilities --source-root . --build-dir build --strict
```

LLM 估算器只讀取 safetensors header，不會將 checkpoint 權重載入記憶體。
它支援 dense 與常見 MoE decoder metadata、量化 checkpoint 儲存、多個 PEFT
Adapter、expert-parallel 通訊、KV cache、eager/SDPA 暫存 tensor、矩陣 FLOP
以及選用的 Roofline 區間。

### 檢視虛擬 GPU 記憶體

在一個終端機中發布程序狀態：

```bash
FAKEGPU_SMI_STATE_DIR=/tmp/fakegpu-smi python3 your_inference.py
```

在另一個終端機中檢視：

```bash
fakegpu nvidia-smi --state-dir /tmp/fakegpu-smi
fakegpu nvidia-smi --state-dir /tmp/fakegpu-smi --loop 1 --count 10
```

表格會分別顯示請求的 tensor 位元組、allocator reserved 位元組與選用的校準
runtime overhead。這裡顯示的是 FakeGPU 狀態，不是主機 NVIDIA 驅動程式資料。

### 透過 TCP 模擬多台主機

在指定連接埠執行自包含的雙節點 loopback 測試：

```bash
fakegpu bandwidth \
  --listen 127.0.0.1:29591 \
  --nodes 2 \
  --ranks-per-node 1 \
  --size 4MiB \
  --iterations 10
```

FakeGPU 會啟動 coordinator、建立拓撲、透過 TCP 傳輸 collective payload、
驗證規約結果、回報端對端吞吐量，並記錄任意節點對之間的通訊資料。獨立部署的
coordinator 可用於多台實體主機。

`torchrun`、DDP、FSDP、DeepSpeed、拓撲與復原範例請參閱
[分散式模擬使用說明](docs/distributed-sim-usage.zh.md)。

<p align="right">(<a href="#readme-top">返回頂端</a>)</p>

## 功能範圍

| 領域 | 已實作範圍 | 限制 |
|---|---|---|
| FakeCUDA runtime | 常見 tensor 建立與操作、module、autograd、optimizer、裝置傳遞、記憶體 API、混合精度、checkpoint、DataLoader 與 dispatch storage 生命週期統計 | 二進位 CUDA 擴充與未涵蓋的 PyTorch operator 需要個別驗證 |
| 原生 CUDA stack | 部分 Driver、Runtime、NVML、cuBLAS/cuBLASLt 與 NCCL 符號；主機記憶體；CPU GEMM；可設定的不支援 API 策略 | native simulate 模式不執行任意 CUDA kernel |
| 記憶體估算 | 執行峰值、簡化 caching allocator、ATen graph 生命週期、optimizer 階段、workspace 區間、校準資料與 OOM 檢查 | backend 內部記憶體與未配對的自訂 kernel 需要校準 |
| LLM 分析 | Dense/MoE 推論、量化權重、PEFT Adapter、KV cache、暫存 tensor、expert 通訊、FLOP、SFT 參考與 FSDP/FSDP2 投影 | 不會自動推斷融合量化 kernel、expert 不均衡與任意模型架構 |
| 效能模型 | 純量運算與記憶體頻寬 Roofline，以及 lower/expected/upper 效率假設 | 只提供分析區間；Tensor Core 加速需要明確傳入 |
| 儲存庫分析 | Python 入口、import、訓練框架、設定、CUDA/PTX 原始碼、二進位擴充與建議實驗 | 動態 import、產生的 kernel 與資料相依分支需要執行驗證 |
| 分散式模擬 | TCP/Unix coordinator、collective、P2P、subgroup、異質拓撲、節點對報告、timeout、錯誤注入、communicator shrink 與固定規模 elastic restart | 不重現 NCCL 協定、NVLink 或 RDMA 時序 |
| 框架相容性 | 針對 Transformers、Accelerate、PEFT、TRL、DDP、FSDP/FSDP2、DeepSpeed ZeRO/Pipeline/AutoTP/AutoEP、torchtune、Lightning、LitGPT 與 nanoGPT 的測試 | 相容性只適用於文件記錄的版本與選項 |
| 監控與報告 | 虛擬 `nvidia-smi`、原生裝置報告、preflight、cluster matrix、驗證 manifest 與 JSON Schema 檢查 | 虛擬資料只反映已統計或已校準的狀態 |
| GPU 目錄 | 82 個 profile，涵蓋八種 NVIDIA 架構以及消費級、工作站、資料中心、嵌入式與測試裝置 | 硬體規格不能保證 kernel 層級的效能等價 |

範圍標記：

- **Maintained**：屬於常規迴歸測試範圍。
- **Validated**：在文件說明的模型、shape、軟體與架構範圍內完成專項數值或
  實體主機實驗。
- **Compatibility-tested**：完成針對特定框架工作流程的測試。
- **Experimental**：僅適用於對應原型範圍，不提供通用相容性保證。

### 指令速查

| 指令 | 用途 |
|---|---|
| `fakegpu doctor` | 檢查安裝、動態函式庫、PyTorch 與 GPU profiles |
| `fakegpu demo` | 執行小型 CPU FakeCUDA 訓練步驟 |
| `fakegpu preflight` | 將工作負載執行到指定階段並判斷 fit/OOM |
| `fakegpu analyze-repo` | 統計儲存庫入口與 GPU 相依風險 |
| `fakegpu estimate-llm` | 估算 decoder 權重、執行記憶體、通訊與 FLOP |
| `fakegpu estimate-roofline` | 產生與 profile 相關的分析延遲區間 |
| `fakegpu capabilities` | 檢視或嚴格檢查原生 API 分類 |
| `fakegpu nvidia-smi` | 顯示虛擬程序 GPU 記憶體 |
| `fakegpu workspace-profiles` | 檢查並列出 workspace profiles |
| `fakegpu validate` | 執行 JSON/TOML/YAML 宣告式測試矩陣 |
| `fakegpu coordinator` | 啟動、探測、停止分散式 coordinator 或產生報告 |
| `fakegpu bandwidth` | 驗證 TCP payload 並測量端對端吞吐量 |

### 執行模式

Python runtime：

| Runtime | 行為 |
|---|---|
| `fakecuda` | 為 PyTorch 加入 FakeCudaTensor 行為，並在 CPU 上執行已維護運算 |
| `native` | 在目前程序中載入 FakeGPU 原生動態函式庫 |
| `auto` | 可用時選擇 `fakecuda`，否則使用 `native` |

原生運算模式：

| `FAKEGPU_MODE` | 行為 | 需要實體 GPU |
|---|---|---:|
| `simulate` | 虛擬裝置身分與記憶體；已維護的 cuBLAS/cuBLASLt 路徑可使用 CPU | 否 |
| `passthrough` | 不注入 FakeGPU CUDA/NVML 的真實 CUDA 基準 | 是 |
| `hybrid` | 保留真實 CUDA 運算，同時虛擬化部分 Driver/NVML 並處理 OOM 策略 | 是 |

分散式模式：

| `FAKEGPU_DIST_MODE` | 行為 |
|---|---|
| `disabled` | 不安裝 FakeGPU 分散式層 |
| `simulate` | 使用 coordinator 管理 collective 與 point-to-point 語意 |
| `proxy` | 保留真實 NCCL 資料傳輸並加入 FakeGPU 控制面報告 |
| `passthrough` | 直接轉送到真實 NCCL |

<p align="right">(<a href="#readme-top">返回頂端</a>)</p>

## GPU Profiles

Profiles 位於 `profiles/<architecture>/<segment>/*.yaml`，由 Python 與原生
runtime 共同使用。

| 架構 | Profile 數量 | Compute capability | 產品範圍 |
|---|---:|---|---|
| Maxwell | 1 | 5.2 | GeForce GTX 900 系列 |
| Pascal | 9 | 6.0, 6.1 | GeForce GTX 10 與 Tesla P 系列 |
| Volta | 1 | 7.0 | Tesla V 系列 |
| Turing | 12 | 7.5 | GeForce RTX 20、Quadro RTX 與 T4 |
| Ampere | 22 | 8.0, 8.6, 8.7 | GeForce RTX 30、RTX A、A 系列加速卡與 Jetson |
| Ada | 17 | 8.9 | GeForce RTX 40、RTX Ada 與 L 系列加速卡 |
| Hopper | 2 | 9.0 | H 系列加速卡 |
| Blackwell | 18 | 10.0, 10.3, 11.0, 12.0, 12.1 | GeForce RTX 50、RTX PRO、B 系列、Jetson 與 GB10 |

每個 profile 都宣告架構與 compute capability；驗證器會拒絕不相符的組合。
YAML 檔案也會記錄規格來源與 measured/reference/synthetic 狀態。

```bash
fakegpu doctor --list-profiles
./fgpu --profile rtx4090 --device-count 2 python3 your_script.py
./fgpu --devices "t4,a100:2,h100" python3 your_script.py
python3 scripts/update_nvidia_gpu_catalog.py --check
```

資料來源與驗證規則請參閱 [profiles/README.md](profiles/README.md)。

<p align="right">(<a href="#readme-top">返回頂端</a>)</p>

## 報告與驗證

| 檔案 | 產生入口 | 主要內容 |
|---|---|---|
| `fake_gpu_report.json` | 原生 runtime | 單一裝置記憶體、IO、API 呼叫、不支援行為與已維護 GEMM FLOP |
| `cluster_report.json/.md` | 分散式 coordinator | Collective/P2P 總量、完整節點對矩陣、峰值、拓撲、時間軸、故障與復原 |
| `preflight_report.json/.md` | Preflight CLI | 階段進度、fit/OOM、記憶體類別、workspace 涵蓋率與可信度 |
| LLM estimate | `fakegpu estimate-llm` | 權重、KV cache、暫存 tensor、Adapter、MoE 通訊、FLOP 與 Roofline |
| 靜態記憶體報告 | `./ftest static_memory_validation` | graph 生命週期、optimizer 階段、workspace profiles 與選用的 CUDA 比較 |
| 宣告式驗證報告 | `fakegpu validate` | 展開的測試矩陣、前置條件、斷言、主機/Git 資訊、耗時與日誌 |
| Virtual SMI state | FakeCUDA runtime | 單一程序 requested、reserved、simulated 目前/峰值位元組、階段與可信度 |

常規本機檢查：

```bash
./ftest smoke
./ftest cpu_sim
./ftest static_memory_validation
python3 -m pytest -q
python3 -m fakegpu validate \
  --manifest verification/data/validation_smoke.yaml \
  --report-dir build/validation-smoke \
  --strict
```

目前迴歸基準為 425 個測試通過，1 個選用測試略過。原生 smoke、CPU 數值模擬、
嚴格能力檢查、wheel 安裝與嚴格 MkDocs 建置也已通過。精確度資料只適用於文件
記錄的工作負載與校準簽章。

完整數值、分散式、框架與跨架構結果請參閱
[報告與驗證](docs/reports-and-validation.zh.md)。

<p align="right">(<a href="#readme-top">返回頂端</a>)</p>

## 架構

```text
面向 GPU 的應用程式
├── Python runtime: fakegpu.init(runtime="fakecuda")
│   └── FakeCudaTensor + 策略 + 記憶體統計
│       └── 已維護的 PyTorch 運算在 CPU 上執行
│
├── 原生 runtime: ./fgpu 或 fakegpu.init(runtime="native")
│   └── libcuda / libcudart / libcublas / libnvidia-ml / libnccl
│       ├── profiles、allocation、stream 與指標
│       ├── 主機記憶體與 CPU 數值運算
│       └── hybrid 模式可轉送到真實 CUDA
│
└── 分析工具
    ├── 儲存庫與相依套件統計
    ├── ATen graph 與 safetensors 估算
    └── Roofline、校準與報告檢查

分散式 coordinator
└── 邏輯節點、TCP/Unix 傳輸、collective、故障與報告
```

檔案層級說明請參閱 [架構與專案結構](docs/project-structure.zh.md)。

<p align="right">(<a href="#readme-top">返回頂端</a>)</p>

## 限制

- Native simulate 模式不執行任意 CUDA kernel。相容性 no-op 會影響測試結論
  時，應使用 `FAKEGPU_UNSUPPORTED_API=error`。
- FakeCudaTensor 涵蓋已維護的 Python/PyTorch 行為，不支援二進位 CUDA 擴充。
- 靜態儲存庫分析無法解析所有動態 import、產生的 kernel、執行期 shape 與資料
  相依分支。
- 執行與靜態記憶體估算可能遺漏 backend 內部記憶體、自訂 operator、特定
  allocator 策略與未配對的 workspace。容量規劃應搭配相同環境的真實 GPU
  校準。
- LLM 估算器不重現融合量化 kernel、不推斷 expert 不均衡，也不能自動執行任意
  模型架構。
- Roofline 結果是分析區間，不是實測 kernel 延遲。
- 分散式耗時包含 coordinator、記憶體複製、socket 與程序排程，不能當作原始
  網路或 NCCL 效能。
- Hybrid 與 passthrough 模式需要相容的實體 CUDA 環境。
- macOS SIP 可能移除系統程式的 `DYLD_*` 環境變數。原生攔截建議使用
  Homebrew、conda 或 pyenv Python。

<p align="right">(<a href="#readme-top">返回頂端</a>)</p>

## 開發計畫

- [x] CPU PyTorch FakeCUDA runtime
- [x] 原生 CUDA/NVML/cuBLAS/NCCL 攔截
- [x] 可設定並檢查架構的 GPU profile 目錄
- [x] 執行、靜態、LLM、MoE、量化與 Adapter 記憶體估算
- [x] 嚴格原生 API 能力清單與匯出符號檢查
- [x] 儲存庫分析器與 profile Roofline 估算器
- [x] TCP 多節點模擬與完整節點對通訊報告
- [x] DDP、FSDP/FSDP2、DeepSpeed 與 elastic recovery 專項驗證
- [ ] 擴充可執行的原生 CUDA 與 cuBLAS 操作
- [ ] 增加更多軟體環境與工作負載的校準資料
- [ ] 加強產生 kernel 與自訂擴充偵測
- [ ] 增加分層與高基數網路拓撲模型

提議功能與已知限制請參閱
[GitHub Issues](https://github.com/FanBB2333/FakeGPU/issues)。

<p align="right">(<a href="#readme-top">返回頂端</a>)</p>

## 文件

- [入門指南（簡體中文）](docs/getting-started.zh.md)
- [快速參考（簡體中文）](docs/quick-reference.zh.md)
- [AI 工作負載 Preflight（簡體中文）](docs/ai-researcher-preflight.zh.md)
- [儲存庫與 Roofline 分析（簡體中文）](docs/repository-and-performance-analysis.zh.md)
- [LLM 推論估算（簡體中文）](docs/llm-inference-estimation.zh.md)
- [LLM SFT 記憶體估算（簡體中文）](docs/llm-sft-memory-estimation.zh.md)
- [分散式模擬使用說明（簡體中文）](docs/distributed-sim-usage.zh.md)
- [DeepSpeed 驗證（簡體中文）](docs/deepspeed-validation.zh.md)
- [錯誤模擬（簡體中文）](docs/error-simulation.zh.md)
- [報告與驗證（簡體中文）](docs/reports-and-validation.zh.md)
- [宣告式驗證 Manifest（簡體中文）](docs/validation-manifests.zh.md)
- [架構與專案結構（簡體中文）](docs/project-structure.zh.md)

在本機預覽文件：

```bash
python3 -m pip install -e ".[docs]"
mkdocs serve
```

<p align="right">(<a href="#readme-top">返回頂端</a>)</p>

## 參與貢獻

歡迎提交問題報告、最小測試案例、profile 資料修正、文件改進與程式碼修改。

1. Fork 儲存庫。
2. 建立分支：`git checkout -b feat/your-change`。
3. 為修改的行為新增或更新測試。
4. 執行對應的 `ftest` target 與 Python 測試。
5. 使用清楚的
   [Conventional Commit](https://www.conventionalcommits.org/) 訊息提交。
6. Push 分支並建立 pull request。

GPU 記憶體估算或相容性問題應附上完整指令、profile、軟體版本與產生的報告。

<p align="right">(<a href="#readme-top">返回頂端</a>)</p>

## 授權

本專案使用 MIT License，詳細內容請參閱 [LICENSE](LICENSE)。

<p align="right">(<a href="#readme-top">返回頂端</a>)</p>

## 致謝

- README 結構參考
  [Best-README-Template](https://github.com/othneildrew/Best-README-Template)
- GPU 型號與 compute capability 資料參考
  [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda/gpus) 與
  [舊版 GPU 清單](https://developer.nvidia.com/cuda/gpus/legacy)
- CPU 框架驗證以 [PyTorch](https://pytorch.org/) 為基礎

<p align="right">(<a href="#readme-top">返回頂端</a>)</p>

[test-shield]: https://github.com/FanBB2333/FakeGPU/actions/workflows/test.yml/badge.svg?branch=main
[test-url]: https://github.com/FanBB2333/FakeGPU/actions/workflows/test.yml
[release-shield]: https://img.shields.io/github/v/release/FanBB2333/FakeGPU?include_prereleases&sort=semver
[release-url]: https://github.com/FanBB2333/FakeGPU/releases
[python-shield]: https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white
[python-url]: https://www.python.org/
[license-shield]: https://img.shields.io/github/license/FanBB2333/FakeGPU
[license-url]: LICENSE
