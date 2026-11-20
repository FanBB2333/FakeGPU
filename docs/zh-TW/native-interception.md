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

## 函式庫名稱與 CUDA 版本

Linux 上啟動器 preload `libcudart.so.12`、`libcublas.so.12`、`libcuda.so.1`、
`libnvidia-ml.so.1` 與 `libnccl.so.2`。FakeGPU 的 runtime 與 cuBLAS 遵循 CUDA 12
ABI，回報 CUDA 12.0，因此原生攔截以為 CUDA 12 建置的程式為目標，包括
`torch.version.cuda` 以 `12.` 開頭的 PyTorch。

CUDA 13 建置連結 `libcudart.so.13` 與 `libcublas.so.13`。FakeGPU 不提供這兩個名稱，
也不實作 CUDA 13 ABI。真實的 CUDA 13 函式庫仍會載入，與 preload 的 FakeGPU 函式
同名的呼叫由 FakeGPU 接手，其餘呼叫進入真實 runtime，因此這種執行不是受支援的模擬。
PyPI 上 Linux 預設的 `torch` 2.14 wheel 相依 CUDA 13.0。原生攔截請安裝 CUDA 12
版本的 PyTorch，或使用不依賴 CUDA 函式庫的 Python FakeCUDA。

建置目錄也提供 `libcudart.so`（nvitop 在 Linux 上依此名稱載入），以及 macOS 上的
`libnvidia-ml.so.1`（nvidia-ml-py，即 `pynvml`，在 Windows 以外的平台都依此名稱
載入）。wheel 中這些名稱是複製的檔案。副本會作為第二個函式庫載入，但與 FakeGPU
共用程序狀態，所以啟動器下的 `pynvml` 能看到模擬 GPU 及其記憶體。

## 裝置可見性

simulate 模式下，原生函式庫依 CUDA driver 的規則處理 `CUDA_VISIBLE_DEVICES`：CUDA
ordinal 跟隨這個遮罩，NVML 與 `nvidia-smi` 仍依實體索引列出所有 GPU。每一項可以是
索引、`nvidia-smi -L` 顯示的 `GPU-` UUID（或足以唯一對應一張 GPU 的前綴），也可以是
單一 `MIG-` id。清單在第一個找不到 GPU 的項目處結束，所以空字串或 `-1` 會隱藏所有
GPU；同一張 GPU 出現兩次時，`cuInit` 傳回 `CUDA_ERROR_INVALID_DEVICE`。
`fakegpu.init(runtime="fakecuda")` 對模擬的 GPU 套用同樣的遮罩；遮罩隱藏全部 GPU
時不再 patch `torch.cuda`。

`CUDA_DEVICE_ORDER=PCI_BUS_ID` 依 PCI 匯流排順序列舉 GPU。CUDA 的預設值
`FASTEST_FIRST` 使用 NVIDIA 未公開的啟發式規則，FakeGPU 不建模它，同樣依 PCI 匯流排
順序列舉。在混插不同 GPU 的真實機器上，設定 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 可以得到
相同順序。

每張模擬 GPU 只有一個 UUID 與一個 PCI bus id，CUDA（`cuDeviceGetUuid`、
`cudaDeviceGetPCIBusId`、`cuDeviceGetByPCIBusId`）、NVML、`nvidia-smi` 與
`torch.cuda.get_device_properties()` 回報的都是它們。

## Hybrid 模式

原生 hybrid 模式下，CUDA runtime 與 cuBLAS 保持原生。設定了
`CUDA_VISIBLE_DEVICES`（依索引或 UUID）時會保留實體 NVML 偵測，讓框架正確解析
該選擇，遮罩交給真實 driver 處理。沒有設定時使用建模的 NVML 裝置清單。子環境切換模式時，`fakegpu.env()`
會移除殘留的 FakeGPU preload。

## macOS

macOS System Integrity Protection 可能移除系統程式的 `DYLD_*` 環境變數。原生攔截
建議使用 Homebrew、conda 或 pyenv Python。
