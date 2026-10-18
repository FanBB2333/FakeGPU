# 框架相容性

[文件索引](README.md) · [English](../en/frameworks.md) ·
[简体中文](../zh-CN/frameworks.md)

支援範圍依執行路徑驗證。通過以下檢查，不代表所有 GPU 後端或任意 CUDA 核心都已實作。

| 路徑 | 支援的執行方式 | 維護中的數值檢查 |
|---|---|---|
| PyTorch FakeCUDA | CUDA 介面張量，資料與計算位於 CPU | 既有執行階段與訓練測試 |
| scikit-learn Array API + FakeCUDA | 部分估計器使用模擬 CUDA 張量 | StandardScaler、PCA、Ridge 與 NumPy 結果對照 |
| 一般 scikit-learn | 正常 CPU 執行 | 前處理、迴歸與分類 pipeline |
| TensorFlow CPU 適配器 | 明確的 GPU device scope 重新導向 CPU | eager 矩陣乘、梯度、圖訓練與 Keras 訓練 |
| CUDA C/C++ + 原生函式庫 | 主機記憶體支援的配置、拷貝、stream/event 與部分 CPU cuBLAS 運算 | 編譯後的 C++ GEMM 流程、記憶體釋放、模組註冊與主機執行緒啟動設定 |
| `nvcc` 裝置核心 | 支援註冊與啟動策略；不支援任意核心計算 | 編譯 `.cu` 程式，檢查啟動錯誤及略過核心後輸出不變 |

## scikit-learn 使用模擬 CUDA 張量

在啟動 Python 前開啟 SciPy 的 Array API：

```bash
SCIPY_ARRAY_API=1 python your_script.py
```

```python
import fakegpu
fakegpu.init(runtime="fakecuda", profile="a100", device_count=2)

import sklearn
import torch
from sklearn.preprocessing import StandardScaler

x = torch.asarray([[1., 2.], [3., 4.]], device="cuda:1")
with sklearn.config_context(array_api_dispatch=True):
    result = StandardScaler().fit_transform(x)
assert result.device == torch.device("cuda:1")
```

此處的 profile 描述模擬裝置。維護中的檢查使用 scikit-learn 1.8.0 與 PyTorch 2.9.1；
Array API 支援取決於具體估計器與求解器。一般 NumPy 輸入的估計器仍是 CPU 計算。

## TensorFlow GPU 作用域在 CPU 上執行

```python
import tensorflow as tf
from fakegpu.tensorflow import simulate

with simulate(device_count=2):
    with tf.device("/GPU:1"):
        x = tf.constant([[1., 2.]])
        result = tf.matmul(x, tf.transpose(x))
        print(result.numpy())  # [[5.]]
```

在建立或追蹤待模擬的操作之前進入適配器。它暫時替換整個程序的 `tf.device`，
應在啟動工作執行緒前進入；退出時恢復原函式，發生例外時也會恢復。作用域內追蹤的圖
保留 CPU 放置。TensorFlow 的裝置探索與張量裝置仍顯示實際 CPU，不模擬 TensorFlow
GPU 記憶體統計、NCCL、XLA CUDA 或自訂 GPU 核心。維護中的檢查使用 TensorFlow CPU 2.20.0。

## 執行驗證

在 Linux 測試環境中安裝選用驗證依賴（macOS 將 `tensorflow-cpu` 換為
`tensorflow==2.20.0`）：

```bash
python -m pip install 'tensorflow-cpu==2.20.0' 'scikit-learn==1.8.0' 'torch==2.9.1' PyYAML
```

在原始碼目錄中執行：

```bash
PYTHONPATH=. python -m fakegpu validate \
  --manifest tests/data/framework_compatibility.yaml \
  --report-dir build/validation-frameworks --strict
scripts/test.sh smoke
```

驗證矩陣實際執行數值斷言。缺少依賴會標記為略過，`--strict` 會將其判為失敗。
CI 安裝上述版本並執行全部三個框架案例。原生 smoke 在 Linux/macOS 編譯並執行 C++
流程；裝有 CUDA 12 工具包（`nvcc`，已驗證 12.4）時，額外以 `--cudart shared` 編譯
`tests/native/cuda_kernel_launch.cu`，驗證 `allow`、`warn`、`error` 三種策略。
沒有 `nvcc` 時明確回報略過。

驗證原生數值工作負載時使用 `--unsupported-api error`，讓不支援的核心啟動回傳錯誤，
避免把未改變的輸出當成正確計算。應用啟動方式見[原生函式庫攔截](native-interception.md)。
這些檢查不構成任意 CUDA 核心的數值或耗時一致性保證。
