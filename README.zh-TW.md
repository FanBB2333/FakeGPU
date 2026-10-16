<a id="readme-top"></a>

<div align="center">

# FakeGPU

**無需正式 GPU 叢集，即可驗證面向 CUDA 的應用程式、估算 GPU 記憶體並模擬分散式
GPU 工作流程。**

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
2. [快速開始](#快速開始)
3. [使用方式](#使用方式)
4. [文件](#文件)
5. [開發](#開發)
6. [限制](#限制)
7. [參與貢獻](#參與貢獻)
8. [授權](#授權)
9. [致謝](#致謝)

## 專案介紹

FakeGPU 讓沒有 GPU 的機器看起來像有 GPU。應用程式會看到可設定的 NVIDIA 風格裝置，
已支援的運算則實際在 CPU 上執行。FakeGPU 同時記錄記憶體用量與通訊資料，讓你在花錢
租用真實硬體前，先確認程式能不能跑、工作負載放不放得下。

它有三種使用方式：

- **Python FakeCUDA**：PyTorch 程式可以使用 `cuda` 裝置，tensor 則留在 CPU 上。
- **原生函式庫攔截**：取代 `libcuda`、`libcudart`、`libcublas`、`libnvidia-ml` 與
  `libnccl` 載入，未修改的程式也能執行。
- **靜態估算器**：讀取模型中繼資料與設定，不載入權重就能回報記憶體、FLOP 與通訊量。

只有 passthrough、hybrid 與校準流程需要實體 GPU。

<a id="use-cases"></a>

### 能做什麼

- **租用 GPU 前先選好硬體。** 用 `estimate-llm` 估算 checkpoint、KV cache、
  activation 與 optimizer 記憶體，或用 `preflight` 把工作負載執行到指定階段。
- **規劃 LLM 服務容量。** `plan-serving` 檢查記憶體能容納多少 chat、RAG 或
  speculative decoding 請求，也支援 vLLM 風格的預算。
- **在筆記型電腦或無 GPU 的 CI 中開發 CUDA 程式。** 在腳本中呼叫
  `fakegpu.init(...)`，或試試 `demo` 與 `validate`。
- **比較訓練方案。** `plan-training` 估算完整參數微調、LoRA、QLoRA、分片與 offload
  的單 rank 記憶體。
- **規劃 diffusion 生成。** `estimate-diffusion` 顯示解析度、batch size、CFG、
  VAE tiling 與 offload 對記憶體的影響。
- **檢查不熟悉的 GPU 儲存庫。** `analyze-repo`、`analyze-kernel` 與 `capabilities`
  列出 GPU 入口、kernel 與不支援的 API。
- **診斷分散式工作流程。** `simulate-topology`、`replay-trace` 與 `bandwidth`
  涵蓋 collective 路由、rank 等待與 payload 檢查。
- **觀察模擬裝置。** `nvidia-smi` 與 `metrics` 顯示裝置與程序，並匯出 Prometheus
  指標。
- **用實體 GPU 檢驗估算。** `calibrate` 比較預測值與實測峰值，並執行誤差門檻檢查。

在已記錄的驗證工作負載上，校準後的記憶體估算器在 26 個受控 GPU 觀測上誤差不超過
0.08%，在十個 Qwen SFT 案例上不超過 1.921%。這些結果只涵蓋固定的工作負載。完整證據
請參閱[驗證與可靠性](docs/zh-TW/validation.md)。

### 主要技術

- [Python](https://www.python.org/) 3.10+：runtime、估算器與 CLI
- C++17 與 [CMake](https://cmake.org/)：原生函式庫
- [PyTorch](https://pytorch.org/)：在 CPU 上執行 FakeCUDA

<p align="right">(<a href="#readme-top">返回頂端</a>)</p>

## 快速開始

需要 Linux 或 macOS、Python 3.10+、CMake 3.14+、支援 C++17 的編譯器，Python
runtime 另需 PyTorch。Debian 或 Ubuntu 請安裝 `build-essential`，macOS 請安裝
Xcode Command Line Tools。

```bash
git clone https://github.com/FanBB2333/FakeGPU.git
cd FakeGPU
scripts/build.sh
FAKEGPU_BUILD_DIR="$PWD/build" python3 -m pip install '.[torch]'
```

確認一切正常：

```bash
python3 -m fakegpu doctor --list-profiles
python3 -m fakegpu demo --profile l4
```

`doctor` 檢查 profile 目錄、原生函式庫與 PyTorch。`demo` 在 CPU 上執行一個小型
訓練步驟，同時讓程式看到 CUDA 裝置。

原生 macOS Conda 環境的安裝方式見
[macOS 環境說明](docs/zh-TW/development.md#macos-conda-環境)。

<p align="right">(<a href="#readme-top">返回頂端</a>)</p>

## 使用方式

### 在 FakeCUDA 上執行 PyTorch 程式

請在匯入 PyTorch 前初始化 FakeGPU：

```python
import fakegpu

fakegpu.init(runtime="fakecuda", profile="a100", device_count=2)

import torch

device = torch.device("cuda:0")
model = torch.nn.Linear(8, 4).to(device)
loss = model(torch.randn(2, 8, device=device)).square().mean()
loss.backward()

print(torch.cuda.device_count())      # 2
print(torch.cuda.get_device_name(0))  # NVIDIA A100
```

### 執行未修改的程式

啟動器會幫你設定 `LD_PRELOAD` 或 `DYLD_INSERT_LIBRARIES`：

```bash
python3 -m fakegpu --build-dir build --devices "a100:2,h100:2" \
  python3 your_script.py
```

更多選項請參閱[原生函式庫攔截](docs/zh-TW/native-interception.md)。

### 執行前檢查記憶體

```bash
# 將 train.py 執行到 forward 階段，並回報是否放得下。
python3 -m fakegpu preflight --runtime fakecuda --profile a100 \
  --stage forward --report-dir build/preflight -- python3 train.py

# 只讀取 safetensors header 來估算 LLM 推論記憶體。
python3 -m fakegpu estimate-llm --model-dir /models/example \
  --prompt-tokens 128 --generated-tokens 32 --target-profile a100

# 規劃線上 serving 請求池。
python3 -m fakegpu plan-serving --model-dir /models/example \
  --active-sequences 16 --prompt-tokens 4096 --generated-tokens 256 \
  --target-profile a100
```

### 查看模擬裝置

```bash
export FAKEGPU_SMI_STATE_DIR=build/smi
python3 your_script.py &
python3 -m fakegpu nvidia-smi --state-dir build/smi
```

macOS/Linux 可用 `fakegpu nvitop --state-dir build/smi` 查看狀態，安裝與雙終端範例見 [NVML 橋接說明](docs/zh-TW/fakegpu-smi.md#在-macos-或-linux-使用-pip-安裝的-nvitop)。

若想讓一台模擬機器在背景持續執行，安裝 `console` 擴充後執行 `fakegpu console`：
FakeCUDA 程序會接入這台機器，控制台顯示 GPU、程序與事件，詳見[控制台說明](docs/zh-TW/console.md)。

執行 `python3 -m fakegpu --help` 查看所有指令，`python3 -m fakegpu <command> --help`
查看各指令的選項。

<p align="right">(<a href="#readme-top">返回頂端</a>)</p>

## 文件

詳細說明位於 [`docs/`](docs/zh-TW/README.md)：

- [指令參考](docs/zh-TW/commands.md)
- [記憶體分析與估算](docs/zh-TW/estimation.md)：
  `preflight`、`analyze-repo`、`estimate-llm`、`estimate-diffusion`
- [線上 LLM serving](docs/zh-TW/serving.md)：`plan-serving`、vLLM 預算、
  speculative decoding、混合請求、`benchmark-serving`
- [微調時間與 GPU 記憶體](docs/zh-TW/finetuning.md)
- [Diffusion 訓練](docs/zh-TW/diffusion-training.md)
- [分散式規劃與實驗批次執行](docs/zh-TW/distributed.md)
- [FakeGPU-SMI 與監控](docs/zh-TW/fakegpu-smi.md)
- [原生函式庫攔截](docs/zh-TW/native-interception.md)
- [對照實體 GPU 校準](docs/zh-TW/calibration.md)
- [驗證與可靠性](docs/zh-TW/validation.md)
- [開發與測試](docs/zh-TW/development.md)
- [限制](docs/zh-TW/limitations.md)

GPU profiles 是 [`profiles/`](profiles/) 中的 YAML 檔。目錄包含從 Maxwell 到
Blackwell 的 306 個可選配置，涵蓋桌上型、筆記型、不同記憶體與板卡版本。
型號、參數和來源見[完整目錄](profiles/README.md)。用 `--profile` 或 `FAKEGPU_PROFILE` 選擇一款，或用
`--devices` 傳入混合列表。針對實體裝置的範例使用 `TARGET_GPU_PROFILE`，請設定為
`python3 -m fakegpu doctor --list-profiles` 列出的目錄 ID。

<p align="right">(<a href="#readme-top">返回頂端</a>)</p>

## 開發

```bash
scripts/build.sh          # 建置原生函式庫
scripts/test.sh python    # Python 測試
scripts/test.sh all       # 所有維護中的測試組
```

建置選項、測試組與專案結構請參閱[開發與測試](docs/zh-TW/development.md)。

<p align="right">(<a href="#readme-top">返回頂端</a>)</p>

## 限制

- 原生模擬無法執行任意 CUDA kernel，FakeCUDA 也不支援二進位 CUDA 擴充。
- 記憶體估算可能遺漏 backend 私有配置、自訂 operator 與 allocator 行為。正式依賴
  估算前，請先在真實硬體上校準。
- 模擬器的耗時結果不能當作 NCCL、NVLink 或 kernel benchmark。
- Hybrid 與 passthrough 模式需要可用的 CUDA 環境。

完整列表請參閱[限制](docs/zh-TW/limitations.md)。後續計畫記錄在
[GitHub Issues](https://github.com/FanBB2333/FakeGPU/issues)。

<p align="right">(<a href="#readme-top">返回頂端</a>)</p>

## 參與貢獻

歡迎提交問題報告、測試案例、profile 修正、文件改進與程式碼修改。

1. Fork 儲存庫並建立分支。
2. 為修改的行為新增或更新測試。
3. 執行 `scripts/test.sh all`。
4. 使用 [Conventional Commit](https://www.conventionalcommits.org/) 格式的訊息提交，
   並建立 pull request。

回報估算或相容性問題時，請附上指令、profile、軟體版本與產生的報告。

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
