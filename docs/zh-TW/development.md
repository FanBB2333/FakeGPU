# 開發與測試

[文件索引](README.md) · [English](../en/development.md) ·
[简体中文](../zh-CN/development.md)

## 直接從原始碼目錄開發

```bash
python3 -m pip install pytest PyYAML jsonschema ruff
export PYTHONPATH="$PWD"
```

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

目錄包含 306 個 YAML profile，涵蓋從 Maxwell 到 Blackwell 的消費級、工作站、
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
