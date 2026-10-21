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
| `fakegpu run` | 在模擬 GPU 上直接執行未修改的指令，並回報記憶體是否放得下（[說明](#run-a-command-on-simulated-gpus)） |
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
| `fakegpu daemon` | 在背景執行一台模擬 GPU 機器，FakeCUDA 程序會自動接入；支援啟動、停止、狀態、事件與設定匯入匯出（[說明](console.md)） |
| `fakegpu platform` | 列出主機平台、顯示裝置鏈路，並解釋拷貝路徑、瓶頸與建模耗時（[說明](platforms.md)） |
| `fakegpu metrics` | 匯出有數量上限的 Prometheus/JSON 指標，並提供短期記憶體歷史 |

<a id="run-a-command-on-simulated-gpus"></a>

## 在模擬 GPU 上執行指令

`fakegpu run` 在模擬 GPU 上直接啟動未修改的指令，並在指令結束時把記憶體摘要輸出到
stderr：

```bash
python3 -m fakegpu run --gpu a100x8 -- python3 train.py
python3 -m fakegpu run --gpu h100:4,a100:4 -- \
  torchrun --nproc_per_node 8 train.py
python3 -m fakegpu run --gpu a100x4 --report-dir build/run -- \
  accelerate launch --multi_gpu --num_processes 4 train.py
```

執行時會把一個 `sitecustomize` 模組放在 `PYTHONPATH` 最前面，因此指令中的每個
Python 程序第一次匯入 torch 時都會啟動 FakeCUDA，包括 `torchrun`、
`accelerate launch` 與 `mp.spawn` 啟動的 worker。環境中原有的 `sitecustomize` 仍會
執行。這些 Python 必須能匯入 FakeGPU；無法匯入的 Python 會在匯入 torch 時印出警告，
並在沒有模擬 GPU 的情況下執行。以 `-I`、`-E` 或 `-S` 啟動的 Python 會忽略
`PYTHONPATH`，同樣沒有模擬 GPU。`CUDA_VISIBLE_DEVICES` 會像在真實機器上一樣，把程序
限制在這次執行的部分 GPU 上，摘要也把它的記憶體計在這些 GPU 上。指令的 `PATH` 上的
`nvidia-smi`，以及已安裝時的 `nvitop`，會列出這次執行的 GPU，即使還沒有程序使用它們。程序之間的 collective
傳遞真實數值，見
[在 FakeCUDA 上進行多程序訓練](distributed.md#multi-process-training-on-fakecuda)。

| 選項 | 說明 |
|---|---|
| `--gpu SPEC` | 要模擬的 GPU：`a100x8`、`8xa100`、`a100:8`，或 `h100:4,a100:4` 這樣的列表。ID 來自 `fakegpu doctor --list-profiles`；未知 ID 會以狀態碼 2 結束並提示相近的 ID |
| `--report-dir DIR` | 保留 `run_report.json`，並把各程序記錄存放在 `DIR/records/`，取代上一次執行的記錄。未指定時，執行產生的檔案放在暫存目錄，結束後刪除 |
| `--no-summary` | 不輸出摘要 |

未指定 `--gpu` 時，依下列順序取第一個可用的來源決定 GPU：

1. 正在執行的背景機器（`fakegpu daemon start` 或 `fakegpu console`）。這次執行會
   接入該機器，因此 `fakegpu console` 會顯示它的程序。`FAKEGPU_MACHINE=off` 會跳過
   這一步；
2. `FAKEGPU_PROFILES`；
3. `FAKEGPU_PROFILE` 與 `FAKEGPU_DEVICE_COUNT`，預設為 `a100` 與 8，因此什麼都沒
   設定時會得到八張 A100 GPU。

自帶 GPU 配置的執行不會接入背景機器。腳本若向 `fakegpu.init()` 傳入自己的 profile，
會保留該 profile；摘要隨後列出腳本實際使用的 GPU，並加上一行 `Note` 說明。

### 結束狀態與 Ctrl-C

`fakegpu run` 以指令的狀態碼結束。被訊號終止的指令回傳 128 加上訊號編號，例如
Ctrl-C 後為 130。`fakegpu run` 本身在找不到指令時以 127 結束，指令無法執行時為
126，`--gpu` 錯誤或缺少指令時為 2。

Ctrl-C 會從終端直接送達指令。`fakegpu run` 會等待指令停止，並照常輸出摘要，內容涵蓋
到中斷為止的執行。送給 `fakegpu run` 的 SIGTERM 與 SIGHUP 會轉送給指令；執行不在
終端前景時，SIGINT 也會轉送。

### 閱讀摘要

以一個在 `cuda:0` 上配置一個 1 GiB tensor 的腳本為例：

```text
── FakeGPU run summary ────────────────────────────────────────────────
 Command   python3 alloc.py
 Result    exit 0 after 1.6 s; 1 process used simulated GPUs
 GPUs      2 × a100 (NVIDIA A100-SXM4-80GB, 80.00 GiB each)  [catalog]

  GPU  Peak memory          Capacity     Headroom     Processes
    0     1.00 GiB  [C3]     80.00 GiB    79.00 GiB           1
    1            —           80.00 GiB    80.00 GiB           0

 Verdict   Fits. Largest peak: 1.00 GiB on GPU 0, 1.2% of its capacity.
 [C3]      Peak reserved memory: operator outputs tracked at PyTorch
           dispatch on the CPU; not calibrated against a real GPU (C4).
 [catalog] Capacity from the FakeGPU profile catalog.
 Excludes  the CUDA context, library workspaces and NCCL buffers a real
           GPU adds.
───────────────────────────────────────────────────────────────────────
```

- **Peak memory** 是每張 GPU 的峰值保留記憶體。同一張 GPU 上各程序的峰值會相加，
  若這些程序在不同時間達到峰值，數字可能偏高。
- 峰值後的標籤是記憶體追蹤等級，從 `[C0]`（沒有可用的追蹤）、`[C2]`（tensor
  生命週期）、`[C3]`（運算子輸出或原生配置）到 `[C4]`（已對照真實 GPU 校準）。
  多個程序共用的 GPU 顯示其中最弱的等級。表格下方會解釋每個出現的標籤。
- `[catalog]` 標示取自 FakeGPU profile 目錄的容量。
- `Excludes` 列出這次執行沒有計入的記憶體：真實 GPU 在峰值之外還會占用的 CUDA
  context、函式庫 workspace 與 NCCL 緩衝區。
- 判定結果為 `Fits`、`Does not fit`（模擬的記憶體不足錯誤終止了指令，或峰值超過
  容量）、`Unresolved`（指令先失敗或被中斷；峰值涵蓋到那時為止的執行），或沒有使用
  GPU（沒有任何資料放到 `cuda` 上，或 FakeCUDA 沒有啟動）。`run_report.json` 分別
  記為 `fits`、`does_not_fit`、`unresolved` 與 `no_gpu_use`。指令若攔截模擬的記憶體
  不足錯誤並繼續執行，仍算放得下；判定結果會說明攔截了幾次。
- 超過 16 張 GPU 時，只列出有使用的 GPU。

<a id="limits-of-fakegpu-run"></a>

### `fakegpu run` 的限制

- 只有 Python 程序會得到模擬 GPU。原生 CUDA 程式仍需透過啟動器
  `python3 -m fakegpu --build-dir build <program>` 執行；見
  [原生函式庫攔截](native-interception.md)。
- 執行期間的 `nvidia-smi` 讀取已發布的狀態，可能比程序落後約 250 ms。
- 每個程序每秒以及收到 SIGTERM 或 SIGHUP 時都會重寫自己的記錄，因此某個 rank 失敗後
  被 `torchrun` 停止的其他 rank 仍會計入摘要。透過 `os._exit()` 或 SIGKILL 結束的程序
  可能缺少最後一秒的資料。
- 不計入 DDP 梯度 bucket。

<a id="torchrun-hangs-at-rendezvous"></a>

### torchrun 卡在 rendezvous 時

若主機名稱經由代理的 fake-IP DNS 解析，`torchrun` 可能在 rendezvous 永遠等待，
無論是否使用 FakeGPU。請傳入 loopback 位址與可用的連接埠：

```bash
python3 -m fakegpu run --gpu a100x2 -- torchrun --nproc_per_node 2 \
  --master_addr 127.0.0.1 --master_port 29500 train.py
```
