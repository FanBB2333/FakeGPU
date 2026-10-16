# 指令參考

[文件索引](README.md) · [English](../en/commands.md) ·
[简体中文](../zh-CN/commands.md)

所有指令都以 `python3 -m fakegpu <command>` 執行。用
`python3 -m fakegpu --help` 查看完整列表，用
`python3 -m fakegpu <command> --help` 查看各指令的選項。大型報告與計畫在所有接受
JSON 路徑的位置都可以使用 `.json.gz`。壓縮輸出採用原子寫入，位元組可重現。

## 設定與執行環境

| 指令 | 用途 |
|---|---|
| `fakegpu doctor` | 檢查安裝、原生函式庫、PyTorch 與 profiles |
| `fakegpu demo` | 執行小型 CPU FakeCUDA 訓練步驟，程式可看到 CUDA 裝置 |
| `fakegpu preflight` | 將工作負載執行到指定階段並判斷 fit 或 OOM |
| `fakegpu capabilities` | 列出或嚴格檢查原生 API 分類 |
| `fakegpu validate` | 執行 JSON、TOML 或 YAML 宣告式驗證矩陣 |

## 分析與估算

| 指令 | 用途 |
|---|---|
| `fakegpu analyze-repo` | 統計儲存庫入口與僅支援 GPU 的風險 |
| `fakegpu analyze-kernel` | 檢查 CUDA、PTX 與 SASS 資源及運算 |
| `fakegpu estimate-llm` | 估算 decoder GPU 記憶體、通訊量與 FLOP |
| `fakegpu estimate-diffusion` | 估算 diffusion 的 text encode、denoise 與 VAE decode 階段 GPU 記憶體 |
| `fakegpu estimate-roofline` | 產生與 profile 相關的分析延遲區間 |
| `fakegpu workspace-profiles` | 驗證並查看 workspace 估算 profiles |

## Serving

| 指令 | 用途 |
|---|---|
| `fakegpu plan-serving` | 規劃同構或混合請求準入、chunked prefill 與群組共享 prefix KV 儲存 |
| `fakegpu benchmark-serving` | 依排程測量混合串流請求，保留用量、延遲、SSE 事件與失敗 |

## 微調

| 指令 | 用途 |
|---|---|
| `fakegpu estimate-finetune` | 依框架設定估算單卡全量/LoRA/QLoRA 的步數、時間與 GPU 記憶體 |
| `fakegpu probe-finetune` | 在實體 CUDA GPU 蒐集固定形狀 Hugging Face Trainer 時間與記憶體 |
| `fakegpu profile-finetune-data` | 掃描本機 JSONL 的 token 長度、截斷、padding 與處理後序列數 |
| `fakegpu run-finetune` | 執行 HF Trainer/PEFT、LLaMA-Factory 或 Axolotl 原生訓練並記錄資料、CUDA 時間、記憶體及最終儲存 |
| `fakegpu validate-finetune-windows` | 使用獨立校準與 holdout 訓練驗證動態視窗耗時 |
| `fakegpu validate-finetune-events` | 使用獨立原生事件記錄驗證評估與 checkpoint 成本 |
| `fakegpu plan-finetune-pipeline` | 從校準過的循序資料推導未來原生訓練輪次或更新數 |
| `fakegpu estimate-finetune-pipeline` | 估算完整原生執行時間和已涵蓋的記憶體包絡 |
| `fakegpu validate-finetune-pipeline` | 對照獨立原生訓練驗證完整時間和記憶體預測 |
| `fakegpu plan-finetune-transfer` | 計畫經核驗的隨機或循序原生輸入 |
| `fakegpu estimate-finetune-transfer` | 估算顯式的經驗形狀遷移預算 |
| `fakegpu validate-finetune-transfer` | 對照獨立原生執行驗證遷移預算 |
| `fakegpu calibrate-finetune` | 在可恢復的工作區執行有限額的原生校準 |
| `fakegpu budget-finetune-workspace` | 只用 CPU 對已儲存的原生證據重估預算 |

## Diffusion 訓練

| 指令 | 用途 |
|---|---|
| `fakegpu plan-diffusion-training` | 檢查快取特徵 UNet 訓練計畫與持久儲存 |
| `fakegpu run-diffusion-training` | 在一張實體 GPU 上執行快取特徵 UNet full/LoRA 訓練 |
| `fakegpu estimate-diffusion-training` | 以相容的原生快取特徵訓練觀測編列預算 |
| `fakegpu validate-diffusion-training` | 對照獨立的原生 diffusion 訓練 holdout |

## 分散式與實驗

| 指令 | 用途 |
|---|---|
| `fakegpu plan-training` | 統一分散式訓練設定並估算單 rank GPU 記憶體 |
| `fakegpu simulate-topology` | 模擬 collective 路由與鏈路競爭 |
| `fakegpu replay-trace` | 彙整運算、通訊、等待與 GPU 記憶體時間軸 |
| `fakegpu coordinator` | 管理分散式模擬 coordinator |
| `fakegpu bandwidth` | 驗證模擬 TCP payload 並回報吞吐量 |
| `fakegpu plan-experiments` | 展開有上限的參數網格並檢查實際指令 |
| `fakegpu run-experiments` | 執行或明確恢復實驗，保留每次嘗試 |
| `fakegpu compare-experiments` | 核驗原始產物、比較重複觀測，可選擇匯出圖表 |

## 校準與監控

| 指令 | 用途 |
|---|---|
| `fakegpu calibrate` | 整理 serving 觀測、比較 GPU 記憶體報告並執行可靠性門檻檢查 |
| `fakegpu nvidia-smi` | 查看裝置、程序、建模拓撲/MIG、健康狀態與可靠性事件 |
| `fakegpu nvitop` | 透過原生 NVML 橋接，用已安裝的 nvitop 查看本機 FakeGPU 狀態（macOS/Linux） |
| `fakegpu console` | 開啟背景機器的終端控制台：GPU、程序、事件與設定（[說明](console.md)） |
| `fakegpu daemon` | 在背景執行一台模擬 GPU 機器，FakeCUDA 程序會自動接入；支援啟動、停止、狀態、事件與設定 |
| `fakegpu metrics` | 匯出有數量上限的 Prometheus/JSON 指標，並提供短期記憶體歷史 |
