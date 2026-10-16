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

這些功能預設關閉，需在工作負載啟動前設定。設定無效時不會啟用任何內容，只回報
設定錯誤。

- **NVLink**：`FAKEGPU_NVLINK_GROUPS="0,1;2,3"` 為每組裝置建立全連接關係。
  `FAKEGPU_NVLINK_BANDWIDTH_GBPS` 預設為 `900`。`topo -m`、`nvlink -s`、
  `topology.*` 與 `nvlink.*` 欄位，以及原生 NVML 鏈路查詢都使用這個模型。
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
