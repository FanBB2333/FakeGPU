# 原生函式庫攔截

[文件索引](README.md) · [English](../en/native-interception.md) ·
[简体中文](../zh-CN/native-interception.md)

原生攔截讓未修改的程式載入 FakeGPU 版本的 CUDA 系列函式庫。無法修改程式碼，或程式
從 C++ 而非 PyTorch 呼叫 CUDA 時，這條路徑最合適。

## FakeGPU 如何執行你的程式

| 路徑 | 應用程式看到的內容 | 實際執行方式 |
|---|---|---|
| **Python FakeCUDA** | CUDA 裝置、CUDA 風格 tensor、GPU 記憶體 API 與常見訓練流程 | 已維護的 PyTorch 運算透過 `FakeCudaTensor` 在 CPU 上執行 |
| **原生函式庫攔截** | `libcuda`、`libcudart`、`libcublas`、`libnvidia-ml` 與 `libnccl` 入口 | 選定的運算使用主機記憶體或 CPU 計算。不支援的行為會被分類並寫入報告 |
| **分析與報告** | GPU 記憶體、FLOP、Roofline、拓撲與通訊報告 | 分析 ATen 圖、safetensors 中繼資料、執行 trace、校準資料與 coordinator 事件 |

## 啟動程式

先用 `scripts/build.sh` 建置原生函式庫，再由模組啟動器設定 `LD_PRELOAD` 或
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

## 不支援的呼叫

不支援的原生呼叫可以記錄、顯示警告，或傳回 `cudaErrorNotSupported` 或
`CUDA_ERROR_NOT_SUPPORTED`。

同一個策略 `FAKEGPU_UNSUPPORTED_API` 也涵蓋沒有數值實作的 cuBLAS 演算法、不支援的
GEMM/Lt dtype、layout 或 pointer mode，以及略過的 CPU 計算。設為 `warn` 時會記錄
警告。設為 `error` 時，呼叫會在寫入輸出前傳回 `CUBLAS_STATUS_NOT_SUPPORTED`。

想確認哪些 API 已涵蓋，可以根據能力 manifest 檢查原始碼與已建置函式庫的匯出符號：

```bash
python3 -m fakegpu capabilities --source-root . --build-dir build --strict
```

## Hybrid 模式

原生 hybrid 模式下，CUDA runtime 與 cuBLAS 保持原生。設定了
`CUDA_VISIBLE_DEVICES`（依索引或 UUID）時會保留實體 NVML 偵測，讓框架正確解析
該選擇。沒有設定時使用建模的 NVML 裝置清單。子環境切換模式時，`fakegpu.env()`
會移除殘留的 FakeGPU preload。

## macOS

macOS System Integrity Protection 可能移除系統程式的 `DYLD_*` 環境變數。原生攔截
建議使用 Homebrew、conda 或 pyenv Python。
