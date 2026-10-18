# 限制

[文件索引](README.md) · [English](../en/limitations.md) ·
[简体中文](../zh-CN/limitations.md)

- Linux 原生模擬可在 CPU 上執行受支援的 PTX 指令；純 SASS cubin、不支援的指令與
  啟動特性會明確失敗。macOS 尚不執行 PTX。詳見[框架相容性](frameworks.md)。
- FakeCUDA 涵蓋專案維護的 Python 與 PyTorch 行為，不支援二進位 CUDA 擴充。
- 靜態分析無法解析所有動態 import、生成式 kernel、執行階段 shape 或依賴資料的
  分支。
- GPU 記憶體估算可能遺漏 backend 私有配置、自訂 operator、allocator 策略與從未
  匹配過的 workspace。
- 原生 cuBLAS 中沒有數值實作的演算法、不支援的 GEMM/Lt dtype、layout 或 pointer
  mode，以及略過的 CPU 計算，都依 `FAKEGPU_UNSUPPORTED_API` 處理。設為 `warn` 時
  會記錄警告。設為 `error` 時，呼叫會在寫入輸出前傳回 `CUBLAS_STATUS_NOT_SUPPORTED`。
- 對帶有 vision encoder 的模型，`estimate-llm` 與 `plan-serving` 只計入 checkpoint
  權重，不含 vision/projector activation 與 FLOP，報告會標示這一點。decoder token
  數需已包含視覺 token。
- Diffusion profile 只描述固定的參考 pipeline。自訂 ControlNet、IP-Adapter、
  LoRA、safety checker、refiner、影片與 DiT 元件需要額外的中繼資料及實體 GPU 驗證。
- 線上 serving 規劃不估算請求到達分布、cache eviction、preemption 或重排、
  self-speculative/Medusa/EAGLE、tokenizer 重映射、tensor parallelism、吞吐量與延遲。
- Roofline 輸出是分析區間，不代表實測的 kernel 延遲。
- 分散式耗時包含 coordinator、記憶體複製、socket 與程序排程，不能當作 NCCL、
  NVLink 或 RDMA benchmark。
- 建模故障事件只出現在報告中，不會改變 CUDA 執行，也不代表真實的 ECC/Xid 資料。
- Hybrid 與 passthrough 模式需要相容的實體 CUDA 環境。
- macOS System Integrity Protection 可能移除系統程式的 `DYLD_*` 環境變數。
  原生攔截建議使用 Homebrew、conda 或 pyenv Python。

## 仍在規劃中

- 擴充可執行的原生 CUDA 運算與 cuBLAS 涵蓋範圍。
- 在更多 GPU 與軟體堆疊上驗證分散式與 MoE 估算。

其他建議與已知問題請參閱
[GitHub Issues](https://github.com/FanBB2333/FakeGPU/issues)。
