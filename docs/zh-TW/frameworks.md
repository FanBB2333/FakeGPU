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
| 包含 PTX 或相容 cubin/SASS 的 CUDA 核心（Linux） | 透過原生 Driver/Runtime API 在 CPU 上解譯執行 | 包含或不含 PTX 的 CUDA C++、獨立 cubin，驗證雙精度、warp 指令、結構參數、共享記憶體、屏障、原子操作、全域變數與裝置函式 |

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
CI 執行三個框架案例以及 Linux/macOS 原生 C++ 流程。Linux 額外執行不需 CUDA
工具包的 `tests/native/ptx_execution.py`；有 `nvcc` 時編譯
`tests/native/cuda_kernel_launch.cu`，檢查三種不支援 API 策略下包含 PTX 與純 SASS 兩種編譯結果的數值。
`tests/native/cubin_execution.py` 額外驗證 Driver 載入、參數 ABI、全域變數重定位、
指標複製、CPU cuBLAS 互操作、映像選擇與錯誤路徑。直接 PTX 與 `nvcc -rdc=true` 連結後的 SASS 程式
均驗證父核心、子核心與孫核心的數值。
Linux CI 安裝 CUDA 編譯器，確保此路徑持續驗證。

## 在 CPU 上執行自訂 CUDA 核心

Linux 建置預設啟用固定版本的 [PantheonSim](https://github.com/pantheongpu/pantheonsim)
PTX 與 SASS 解譯器，按裝置指令執行，不依賴預設核心名稱。支援 Driver API 載入的
PTX/cubin 與 Runtime 註冊的 fatbin，包括 LZ4 與 Zstandard 壓縮映像。
Zstandard 映像需要系統提供 `libzstd.so.1`。fatbin 優先選擇目標版本最高的相容 PTX；
沒有相容 PTX 時選擇相容的 cubin。

程式可以保留 PTX，也可以只包含 GPU 機器碼：

```bash
# 保留 PTX。
nvcc -arch=sm_80 --cudart shared your_program.cu -o your_program
# 或為合成 Ampere profile 編譯純 SASS 程式。
nvcc -gencode arch=compute_80,code=sm_80 --cudart shared your_program.cu -o your_program
FAKEGPU_PROFILES=a100 python -m fakegpu --build-dir build --mode simulate --unsupported-api error ./your_program
```

cubin 必須匹配所選 profile：普通映像要求運算能力主版本相同，裝置次版本不低於映像；
架構專用映像要求精確匹配。不相容時回傳 CUDA 錯誤 209。上游 SASS 解碼器涵蓋 Turing
至 Blackwell；編譯數值回歸涵蓋合成 Turing、Ampere、Ada 與 Hopper 目標，編譯器支援時
還驗證 Blackwell（`sm_100`、`sm_120`）。不能據此認為所有指令與編譯器版本均已支援。早於 Turing 的 cubin 仍不支援，可使用相容 PTX。

核心在 CPU 上同步執行，全域變數、常數和裝置分配在多次啟動間保留。不支援的指令與
啟動特性在所有策略下都回傳錯誤，無效控制代碼與越界存取也會失敗；執行錯誤可能發生在
部分寫入之後。cubin 核心回傳的普通全域指標支援 CUDA 複製與填充；回傳的模組相對
常數、區域及共享記憶體指標不支援主機複製，會明確報錯。常數仍可透過 CUDA 符號 API 讀寫。

裝置端子核心透過同一後端執行，在主機啟動呼叫回傳前完成，已驗證巢狀啟動，但不代表
完整裝置執行階段 API 均已支援。此路徑不提供 GPU 效能或完整 CUDA 相容性。
CUDA Graph、協作式啟動與紋理仍不屬於已驗證的涵蓋範圍。上下文仍採用簡化模型，
裝置資源重設回傳不支援。
框架仍需要其依賴的其他 CUDA API 已受支援；上述 TensorFlow 配接器仍是獨立的 CPU 路徑。

PTX/SASS 執行目前要求 Linux；macOS 保留現有原生模擬與 Python 配接器，載入這些
模組會明確回傳不支援。建置要求見[開發與測試](development.md)，啟動方式見
[原生函式庫攔截](native-interception.md)。
