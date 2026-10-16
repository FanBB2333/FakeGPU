# FakeGPU-SMI 與監控

[文件索引](README.md) · [English](../en/fakegpu-smi.md) ·
[简体中文](../zh-CN/fakegpu-smi.md)

`fakegpu nvidia-smi` 以接近 NVIDIA 工具的格式顯示模擬裝置與正在使用它們的程序。
`fakegpu metrics` 則把相同的狀態匯出給 Prometheus。

## 從工作負載發布狀態

狀態發布功能需要明確啟用。請在工作負載呼叫 `fakegpu.init(...)` 或透過原生啟動器
執行前設定狀態目錄。`build/` 已被 Git 忽略。

```bash
export FAKEGPU_SMI_STATE_DIR=build/smi

# Python FakeCUDA 工作負載。
python3 your_script.py

# 未修改的原生 CUDA/NVML 工作負載。
python3 -m fakegpu --build-dir build ./your_native_workload
```

## 查看裝置與程序

在另一個終端中執行：

```bash
# 精簡的裝置與程序表。加入 "-l 1" 可每秒更新一次。
python3 -m fakegpu nvidia-smi --state-dir build/smi

# 裝置清單，以及 runtime、profile 與 allocator 詳細報告。
python3 -m fakegpu nvidia-smi --state-dir build/smi -L
python3 -m fakegpu nvidia-smi --state-dir build/smi -q

# 接近 NVIDIA 指令格式的建模拓撲與 NVLink 檢視。
python3 -m fakegpu nvidia-smi topo -m --state-dir build/smi
python3 -m fakegpu nvidia-smi nvlink -s --state-dir build/smi

# 建模故障、相容性警告、發布失敗與過期狀態。
python3 -m fakegpu nvidia-smi events --state-dir build/smi

# 建模 MIG GPU Instance 與 Compute Instance。
python3 -m fakegpu nvidia-smi mig -lgi --state-dir build/smi
python3 -m fakegpu nvidia-smi mig -lci --state-dir build/smi

# 適合腳本處理的 GPU 與程序查詢。
python3 -m fakegpu nvidia-smi --state-dir build/smi \
  --query-gpu=index,name,uuid,pci.bus_id,profile.id,compute_cap,memory.total,memory.used,memory.free,allocator.model,native.kernel_launches,native.gemm_calls,native.io_bytes \
  --format=csv
python3 -m fakegpu nvidia-smi --state-dir build/smi \
  --query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory,peak_gpu_memory,stage,status \
  --format=csv,noheader,nounits
python3 -m fakegpu nvidia-smi --state-dir build/smi \
  --query-runtime=pid,fakegpu.version,runtime.backend,runtime.mode,policy.oom,tracking.dispatch,catalog.profiles,catalog.native_apis,dispatch.calls,publisher.failed_writes,status \
  --format=json
```

- `-q` 顯示 FakeGPU 版本、runtime 策略、軟體版本、狀態更新時間、profile 目錄與
  原生 API 涵蓋、裝置識別、GPU 記憶體分類、allocator 活動、dispatch 追蹤及各程序
  峰值。
- `-i` 依裝置索引、UUID、PCI Bus ID 或 profile ID 篩選。`--json` 輸出完整的標準化
  資料。FakeGPU 寫入 state schema v2，仍可讀取 v1。
- `--query-runtime` 以 CSV 或 JSON 輸出相同的執行期資訊。`--help-query-runtime`
  列出全部欄位。
- 原生攔截也會發布 allocation 生命週期、傳輸量、kernel launch、GEMM 呼叫與 FLOP、
  相容性事件及 unsupported API 次數。`FAKEGPU_SMI_DETAIL_LIMIT` 與
  `FAKEGPU_SMI_MAX_STATE_BYTES` 分別限制保留的明細與狀態檔大小。

## 建模 NVLink、MIG 與故障

這些功能預設關閉（背景機器上的程序會取得機器的平台），需在工作負載啟動前設定。
設定無效時不會啟用任何內容，只回報設定錯誤。

- **平台與 NVLink**：`FAKEGPU_PLATFORM`（或背景機器的平台）把 GPU 放進 HGX/DGX、
  超級晶片或 PCIe 插槽，並連好 NVLink，見[平台與插槽](platforms.md)。也可以用
  `FAKEGPU_NVLINK_GROUPS="0,1;2,3"` 給組內每對裝置一條速率為
  `FAKEGPU_NVLINK_BANDWIDTH_GBPS`（預設 `900`）的鏈路。`topo -m`（NV#、PIX、PXB、
  PHB、NODE、SYS 與 NUMA affinity）、`nvlink -s`（每條鏈路每個方向的 GB/s）、
  `topology.*`、`nvlink.*`、`platform`、`slot.type`、`numa.node`、`pcie.link.*`、
  `host_link.bandwidth` 欄位，以及原生 NVML 鏈路查詢都使用這個模型。
- **MIG**：`FAKEGPU_MIG_LAYOUT="DEVICE:PROFILE:MEMORY_MIB[:COUNT];..."`，例如
  `FAKEGPU_MIG_LAYOUT="0:1g.10gb:10240:2;1:2g.20gb:20480"`。每個 GPU Instance
  對應一個 Compute Instance。超出父裝置記憶體、8 個 slice 或 8 個實例的設定會被
  拒絕。各實例的 GPU 記憶體用量不會被觀測。
- **故障**：`FAKEGPU_FAULT_EVENTS="DEVICE:CODE:SEVERITY[:COUNT];..."`，例如
  `FAKEGPU_FAULT_EVENTS="0:XID_79:critical;1:NVLINK_CRC:error:3"`。嚴重度支援
  `info`、`warning`、`error` 與 `critical`。`events` 檢視同時列出 unsupported API
  呼叫、發布失敗與過期狀態，`health.*` 欄位顯示裝置的建模狀態。

UUID 與 PCI Bus ID 是穩定的模擬值。溫度、風扇轉速、即時功耗與硬體使用率顯示為
`N/A`。拓撲、頻寬、故障與健康狀態都是建模輸入，不代表硬體測量或 ECC/Xid 資料。

<a id="export-monitoring-metrics"></a>

## 匯出監控指標

`fakegpu metrics` 將 FakeGPU-SMI 標準化狀態匯出為 Prometheus 文字（預設）或 JSON，
不需要安裝額外的監控相依套件：

```bash
python3 -m fakegpu metrics --state-dir build/smi
python3 -m fakegpu metrics --state-dir build/smi --json
```

需要在本機持續收集時，可啟動有記憶體上限的 exporter：

```bash
python3 -m fakegpu metrics --state-dir build/smi --serve \
  --host 127.0.0.1 --port 9400 --interval 1 \
  --history-size 300 --max-process-series 128

curl http://127.0.0.1:9400/metrics
curl http://127.0.0.1:9400/healthz
curl http://127.0.0.1:9400/api/v1/history
```

- `/metrics` 提供最新的裝置、程序、runtime、MIG、拓撲、健康狀態與發布器指標。
  `/healthz` 顯示資料來源與收集狀態，`/api/v1/history` 傳回近期樣本。
- 程序序列優先保留 GPU 記憶體用量最高的程序，預設 128 個、最多 256 個。
  `--max-process-series 0` 可關閉。歷史預設保留 300 個樣本，最多 1,440 個。
- 資料只保存在記憶體中。長期保留請使用 Prometheus 或其他收集系統。
- 服務只監聽 `127.0.0.1`，不提供身分驗證。對外開放前請放在帶身分驗證的 proxy
  後面。匯出資料沿用 FakeGPU-SMI 對建模值與觀測值的區分。

## 在 macOS 或 Linux 使用 pip 安裝的 nvitop

先從原始碼目錄安裝監控依賴，再於第一個終端保持工作負載執行：

```bash
conda activate fakegpu
python -m pip install '.[monitoring]'
FAKEGPU_SMI_STATE_DIR=/tmp/fakegpu-smi python - <<'PYTHON'
import time
import fakegpu
fakegpu.init(profile="l4", device_count=1)  # 模擬 profile。
import torch
x = torch.ones((2048, 2048), device="cuda:0")
time.sleep(600)
PYTHON
```

第二個終端執行：

```bash
conda activate fakegpu
fakegpu nvitop --state-dir /tmp/fakegpu-smi
# 只輸出一次，不進入互動介面：
fakegpu nvitop --state-dir /tmp/fakegpu-smi -- --once
```

在 nvitop 中按 `q` 結束；工作負載用 Ctrl+C 停止。另一個指令
`fakegpu nvidia-smi -l 1` 的更新迴圈要用 Ctrl+C 結束，`q` 不生效。

此入口執行已安裝、未修改的 nvitop，明確載入專案內附的
`libnvidia-ml.dylib`（Linux 為 `libnvidia-ml.so.1`），每 0.25 秒更新原生
NVML 快照。直接執行一般的 `nvitop` 不會啟用橋接。無需安裝 NVIDIA 驅動、替換系統
NVML 或設定 shell 別名；FakeCUDA 的運算仍由 CPU 執行。

裝置名稱、合成 UUID、總 GPU 記憶體、報告用量、PID 與各程序用量來自狀態檔。
報告用量包含配置器預留與設定的 runtime 開銷，因此釋放張量後，可能需要呼叫
`torch.cuda.empty_cache()` 才會看到用量下降。相同合成 GPU UUID 的多個程序會
彙總顯示。預設模式下，溫度、功耗、GPU 使用率及程序 GPU 使用率顯示 `N/A`；建模的 MIG、
NVLink 資訊仍透過 `fakegpu nvidia-smi` 查看，不透過此橋接呈現。

只讀取本機仍在執行且心跳有效的發布者，預設兩秒過期，可用
`--stale-after-seconds` 調整。已結束、已消失的程序與無效檔案不會參與統計；相同
PID 的重複檔案只保留最新快照。已發現的裝置保留穩定 NVML 句柄，最後一個程序
結束後顯示零用量。新增裝置或 profile 後需重新啟動 nvitop，因為它於啟動時選擇
裝置清單。沒有活動發布者時，入口提示如何啟動工作負載並回傳狀態碼 1，不會產生預設裝置。

其他 Python NVML 用戶端可在獨立監控程序中、載入 NVML 之前，使用
`fakegpu.nvml.NvmlStateBridge` 上下文管理器，並在上下文內正常初始化與關閉
pynvml。關閉橋接會停止更新並清空用量；動態函式庫選擇及句柄保留到程序結束。

端到端回歸啟動兩個真實 FakeCUDA 工作程序，檢查 NVML 記憶體、程序查詢與
v1/v2/v3 緩衝區邊界、nvitop API 和 `--once` 輸出、釋放記憶體、正常結束、強制
終止及空狀態提示。安裝 wheel 後，可在原始碼目錄外重現：

```bash
FAKEGPU_CHECKOUT="$PWD"
(cd /tmp && python -I "$FAKEGPU_CHECKOUT/tests/support/nvml_bridge_smoke.py" --tui)
```


## 8 卡 A100：真實 torch 配置與參考遙測

在原始碼目錄啟動第一個終端：

```bash
conda activate fakegpu
python examples/monitoring/torch_eight_a100.py
```

第二個終端：

```bash
conda activate fakegpu
fakegpu nvitop --state-dir /tmp/fakegpu-smi --telemetry reference
```

程式建立 8 張模擬 A100 SXM4 80GB，實際配置 CPU 支撐的 PyTorch 張量。每輪先在
各卡配置 32–60 MiB，再各增加 64 MiB，達到 96–124 MiB；接著釋放新增張量，
最後全部釋放。階段切換時呼叫 `empty_cache()`，預設每階段 8 秒，Ctrl+C 停止。
`--cycles 1` 只執行一輪。主機實際配置量約 1 GiB，不會配置 8 × 80 GiB。

`--telemetry reference` 明確啟用 A100 顯示模型。用於其他程式時，在
`fakegpu.init()` 前設定 `FAKEGPU_TELEMETRY=reference`，並初始化
`profile="a100", device_count=8`。觀察器記錄各裝置 CPU torch 算子的執行時間及
估算的張量 I/O 位元組數，透過 NVML 提供 GPU/記憶體引擎使用率、各程序活躍度、
溫度、功耗、功率上限、P-state 與時脈。僅持有張量不會被當作運算負載。

[參考設定](../../fakegpu/data/a100_monitor_reference.json)明確區分來源與假設：

- [NVIDIA 官方規格](https://www.nvidia.com/en-us/data-center/a100/)提供標準 SXM
  功率上限與 HBM 頻寬。
- [NVIDIA/nccl-tests 中使用者公開的實機快照](https://github.com/NVIDIA/nccl-tests/issues/228)
  提供 8 卡閒置溫度、功耗與 P0 讀數，範圍為 26–30°C、56–60W。
  只保留這些數值，不複製機器身分、UUID 或序號。
- 動態使用率以 **CPU 算子執行占空比作為模擬輸入**，功耗向規格上限插值，溫度
  依明確的熱響應模型變化。記憶體引擎使用率依張量 I/O 與參考頻寬計算，並非
  記憶體容量使用率。負載曲線屬於模型假設，不是 A100 實測或效能校準結果。

模型目前只匹配 `a100`（SXM4 80GB）；其他 profile 或缺少活躍度觀測的發布者仍
保留 `N/A`。GPU 風扇與公開快照一致，繼續顯示 `N/A`；CUDA 驅動及未支援的影片、
PCIe 遙測也不會虛構。預設 `--telemetry none` 維持原行為，macOS 無需 NVIDIA 驅動。

維護的 `tests/support/a100_monitor_smoke.py` 會啟動此 torch 程式與原版 nvitop
互動介面，以 pyte 還原終端更新，檢查 8 卡記憶體增加與釋放、模型遙測及程序結束
清理，輸出只包含數值摘要。
