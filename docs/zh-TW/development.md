# 開發與測試

[文件索引](README.md) · [English](../en/development.md) ·
[简体中文](../zh-CN/development.md)

## 直接從原始碼目錄開發

```bash
python3 -m pip install '.[torch,dev,monitoring]'
export PYTHONPATH="$PWD"
```

## macOS Conda 環境

如尚未安裝 Xcode Command Line Tools，先執行 `xcode-select --install`。使用原生
macOS Conda，在專案根目錄執行：

```bash
conda env create -f environment.macos.yml
conda activate fakegpu
python -m fakegpu doctor
python -m fakegpu demo --profile l4
FAKEGPU_PYTHON="$CONDA_PREFIX/bin/python" scripts/test.sh all
```

如果 `fakegpu` 環境已存在，將建立指令替換為
`conda env update -n fakegpu -f environment.macos.yml`。此環境包含 Python 3.12、
CMake、原生 macOS PyTorch 與開發依賴，包括 Hugging Face、PEFT、datasets 與繪圖
整合。Transformers 5.6.0 與 Accelerate 1.11.0 對應原生取樣規劃器已驗證的版本組合。
安裝 Python 套件時會編譯並封裝 macOS 動態函式庫及 coordinator，無需 Linux VM 或
CUDA 安裝。

macOS 上的 `fakegpu.init()` 預設使用由 CPU 支撐的 FakeCUDA，讓一般 PyTorch
CUDA 裝置程式碼能執行相容性測試，實際運算發生在 CPU。原生 runtime 另外模擬
CUDA/NVML/cuBLAS/NCCL API，不會為 macOS PyTorch 增加 NVIDIA CUDA 支援。

修改 Python 或 C++ 原始碼後，用 `python -m pip install '.[torch,dev,monitoring]'` 重新安裝。
可執行以下檢查，確認安裝後的套件能獨立於原始碼目錄運作：

```bash
FAKEGPU_CHECKOUT="$PWD"
(
  unset PYTHONPATH FAKEGPU_BUILD_DIR FAKEGPU_LIB_DIR FAKEGPU_PROFILE_DIR
  cd "${TMPDIR:-/tmp}"
  python -I "$FAKEGPU_CHECKOUT/tests/support/installed_package_smoke.py"
)
```

2026-10-16 已在 macOS arm64、Python 3.12.15、PyTorch 2.14.1 環境驗證：
1879 項 Python 測試全部通過，無略過；306 個 profile 檢查、原生 smoke 與 CPU 數值
測試全部通過；兩個嚴格驗證清單的 46 次執行全部通過。安裝後的 wheel 也通過上述
隔離檢查。依賴 CUDA 版 PyTorch 的選用原生測試在 macOS 上略過，由 CPU 支撐的
FakeCUDA 訓練另有實際驗證。


另已在原始碼目錄外驗證安裝後的 NVML 橋接，使用 `nvitop 1.7.1` 與
`nvidia-ml-py 13.595.45`：兩個發布程序、兩個模擬裝置、記憶體及程序查詢、釋放
記憶體與程序結束後的清理均通過；`--once` 與互動介面也通過，按 `q` 可正常結束。
用法見 [監控文件](fakegpu-smi.md#在-macos-或-linux-使用-pip-安裝的-nvitop)。


8 卡 A100 參考監控也通過安裝後驗證：真實 torch 配置讓每張模擬卡從 32–60 MiB
增至 96–124 MiB，再回到初始值與零。終端解析器確認同一個 nvitop 互動介面顯示
各階段。遙測數值使用有來源的公開閒置讀數與 CPU 活躍度模型，並非物理 GPU 實測。

## 建置

所有原生建置行為都由一個腳本提供：

```bash
scripts/build.sh
scripts/build.sh --release
scripts/build.sh --debug
scripts/build.sh --build-dir build-custom -- -DSOME_CMAKE_OPTION=value
```

建置目錄與編譯產物不會由 Git 管理。

## 測試

專案維護的測試分為四組：

```bash
scripts/test.sh python   # Python 回歸測試
scripts/test.sh smoke    # 原生載入、報告、能力檢查、SMI 與 coordinator
scripts/test.sh cpu      # CPU cuBLAS 模擬
scripts/test.sh all      # 以上全部
```

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

[發布回歸記錄](../../tests/data/research_release_evidence.json)包含 1734 項通過的
Python 測試（略過一個 CPU-only build 分支）、原生 CPU 套件、104 個封裝的 Python
模組、42 個指令說明檢查與 68 次離線 CLI 檢查。152 份歷史預測全部逐位元組保持一致。

部分測試會檢查 README 與這些文件是否和程式碼一致，例如指令列表與已公開的證據數字。
修改指令或證據檔案時，請同時更新英文、簡體中文與繁體中文文件。

## GPU profiles

目錄包含 304 個 YAML profile，涵蓋從 Maxwell 到 Blackwell 的消費級、工作站、
資料中心與嵌入式 NVIDIA GPU。Python 與原生 runtime 共用這些 profiles。
型號、參數來源、變體口徑和未知欄位見[完整目錄](../../profiles/README.md)。

```bash
python3 -m fakegpu doctor --list-profiles
python3 -m fakegpu demo --profile rtx4090
python3 -m fakegpu --build-dir build --devices "t4,a100:2,h100" \
  python3 your_script.py
python3 scripts/update_nvidia_gpu_catalog.py --check
python3 scripts/render_gpu_catalog.py --check
```

設定 `FAKEGPU_PROFILE` 或傳入 `--profile` 可選擇一個 profile。異質裝置列表使用
`--devices`。

## 腳本

- `scripts/build.sh`：設定並編譯原生目標。
- `scripts/test.sh`：執行維護中的測試組。
- `scripts/update_nvidia_gpu_catalog.py`：檢查或更新 NVIDIA compute capability 快照。
- `scripts/render_gpu_catalog.py`：從 YAML profile 重新產生公開目錄與官網統計。
- `scripts/validation/`：共用的報告與產物驗證工具。
- `scripts/linux/`：Linux GPU 管理工具。
- `scripts/macos/`：macOS 到 Linux VM 的輔助工具。

## 專案結構

```text
FakeGPU/
├── docs/       英文、簡體中文與繁體中文說明文件
├── examples/   微調、serving 與實驗設定範例
├── fakegpu/    Python 套件、CLI、runtime 與估算器
├── profiles/   YAML GPU profile 目錄
├── schemas/    JSON 報告與驗證 schema
├── scripts/    可重複使用的建置、測試、平台與驗證工具
├── src/        原生 C++ 攔截與 coordinator 實作
└── tests/      維護中的回歸測試與最小原生測試 fixture
```

產生的建置目錄、編譯函式庫、測試報告、快取、本機環境、二進位資源與設計草稿
都由 `.gitignore` 排除。
