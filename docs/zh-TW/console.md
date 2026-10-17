# 背景機器與控制台

[文件索引](README.md) · [English](../en/console.md) ·
[简体中文](../zh-CN/console.md)

FakeGPU 可以在背景執行一台模擬 GPU 機器。機器執行期間啟動的 FakeCUDA 程序會自動接入。
終端控制台顯示這台機器的裝置、程序與事件，並可修改它的設定。離開控制台後，機器仍會繼續執行。在原始碼目錄中安裝 `console` 擴充：

```bash
FAKEGPU_BUILD_DIR="$PWD/build" python3 -m pip install '.[torch,console]'
fakegpu console
```

如果機器尚未執行，`fakegpu console` 會先啟動它。首次啟動使用 HGX A100 8-GPU 基板上的
八張 A100 SXM4 80GB，並啟用參考遙測模型；可以用 `--devices`（例如 `--devices h100:8`
或 `--devices rtx4090:2`）指定裝置配置，用 `--platform`、`--pcie-gen` 與
`--nvlink-bridges` 指定主機平台（見[平台與插槽](platforms.md)），用
`--telemetry none|reference` 指定遙測模型。這些選項同樣會更新正在執行的機器。

## 在機器上執行工作負載

機器執行期間，任何呼叫 `fakegpu.init()` 的程序都會把狀態發布到這台機器，並使用機器的裝置配置：

```bash
python - <<'PYTHON'
import time
import fakegpu
fakegpu.init()
import torch
x = torch.ones((4096, 4096), device="cuda:3")
time.sleep(600)
PYTHON
```

如果程序向 `fakegpu.init()` 傳入 `profile`、`devices` 或 `device_count`，或設定了
`FAKEGPU_PROFILE`、`FAKEGPU_PROFILES`、`FAKEGPU_DEVICE_COUNT`，它會保留自己的配置。
設定 `FAKEGPU_MACHINE=off` 可讓程序不接入機器。新配置只對修改之後啟動的程序生效；
已在執行的程序保留啟動時的裝置。

常駐程序執行時，`fakegpu nvitop` 與 `fakegpu nvidia-smi` 不帶參數就會讀取這台機器：
和真機一樣列出配置中的每張 GPU（包括閒置的），以及機器上的程序。

修改裝置配置或遙測模式後，請退出並重新開啟 `fakegpu nvitop`，其裝置清單與遙測模式
在啟動時確定；控制台的裝置清單會即時更新。明確指定這台機器的狀態目錄時，也會繼承
機器的遙測模式，除非用 `--telemetry` 覆寫。

## 控制台

| 分頁 | 內容 |
|---|---|
| `1` Overview | 彙總所有 GPU 的曲線（記憶體、使用率、功耗、溫度、鏈路 I/O）；每張 GPU 一張卡片（記憶體、使用率、功耗、溫度、時脈、P-state、近期記憶體與使用率），按 `v` 可切換為每張 GPU 一列；程序表與最新事件。點擊卡片或在某一列按 Enter，會在 Devices 分頁開啟這張 GPU |
| `2` Processes | 所有已接入程序的命令列、使用者、階段、各裝置記憶體與活動；最近結束的程序 |
| `3` Devices | 配置中的 GPU，可以逐張修改（見下文）；單卡詳情：插槽、建模的拷貝與鏈路吞吐、配置器計數、依類別統計的記憶體、最大的幾次配置，以及最近一分鐘的曲線（記憶體、使用率、溫度、功耗、PCIe/C2C/NVLink 吞吐） |
| `4` Events | 完整事件紀錄，可依文字與等級篩選 |
| `5` Machine | 設定表單（配置、平台、PCIe 插槽代數、NVLink 橋、遙測、背景模式、心跳逾時與記憶體警告門檻）、插槽圖與 GPU 鏈路矩陣、常駐程序與路徑資訊 |
| `6` Profiles | GPU profile 目錄；按 `a` 或 Enter 把選取的 profile 加入配置表單 |

控制台使用低飽和度的莫蘭迪配色，每種讀數在卡片、曲線與表格中使用固定的顏色：記憶體灰紫、
使用率灰綠、功耗陶土色、溫度沙色，鏈路流量為灰青（PCIe）或灰藍（NVLink）。豆沙紅數值、
芥末黃或豆沙紅卡片邊框表示警告。

| 按鍵 | 作用 |
|---|---|
| `1` 到 `6` | 切換分頁 |
| `v` | Overview：在 GPU 卡片與每卡一列之間切換 |
| `k` / `K` / `i` | 向選取的程序傳送 SIGTERM / SIGKILL / SIGINT（會先確認） |
| `b` | 開啟或關閉背景模式 |
| `q` | 離開控制台 |
| `Q` | 停止機器（會先確認） |
| `s` | 機器停止後重新啟動 |

背景模式開啟時（預設），按 `q` 離開後機器繼續執行，可以用 `fakegpu console` 重新進入。
背景模式關閉時，離開控制台也會停止機器。停止機器不會停止正在執行的工作負載。

### 在 Devices 分頁修改 GPU

| 按鍵 | 作用 |
|---|---|
| `e`、Enter 或點擊已選取的列 | 把選取的 GPU 換成另一個 profile |
| `E` | 替換全部 GPU：選擇一個 profile 與數量 |
| `a` | 在最後一張之後新增 GPU，預設使用選取 GPU 的 profile |
| `x` / Delete | 移除選取的 GPU（會先確認） |

表格上方的按鈕功能相同。profile 選擇框會隨輸入篩選目錄，並在套用前預覽新的配置、總記憶體與平台。
如果目前設定的平台放不下新配置（例如 HGX 基板只能放同一型號的 GPU），控制台會把平台設為
`auto`，沒有範本能容納時設為 `generic`，並移除不再適用的 NVLink 橋；預覽會列出這些變更。
Machine 分頁以文字形式編輯同一個配置。新程序使用新配置；執行中的程序保留原本的裝置，
這些裝置在 Devices 分頁中以 `*` 標示，直到對應程序結束。

## 事件

機器會記錄程序與裝置上發生的事情：

| 事件 | 意義 |
|---|---|
| `process.attached` | 一個 FakeCUDA 程序接入了機器 |
| `process.exited` | 程序發布最終狀態後結束 |
| `process.terminated` | 程序因 SIGTERM 結束 |
| `process.vanished` | 程序沒有送出最後一次心跳就結束了，例如被 SIGKILL 或當機 |
| `process.stalled` / `process.resumed` | 存活的程序停止或恢復發布狀態 |
| `process.signal_sent` | 控制台或其他用戶端傳送了訊號 |
| `memory.high` / `memory.normal` / `memory.oversubscribed` | 裝置記憶體越過警告門檻或超出容量 |
| `config.changed` | 配置或某項設定發生變化 |
| `state.invalid` | 某個狀態檔無法讀取 |
| `daemon.started` / `daemon.stopped` | 機器啟動或停止 |

因此，在 nvitop 中或用 `kill` 結束的程序會顯示為 `process.terminated`（SIGTERM）
或 `process.vanished`（SIGKILL）。最近 1,000 筆事件在機器重新啟動後仍會保留。

## 在 shell 中管理機器

```bash
fakegpu daemon start [--devices a100:8] [--telemetry reference] [--background on]
fakegpu daemon status [--json]
fakegpu daemon events [-n 50] [-f] [--json]
fakegpu daemon config [--devices a100:4] [--platform auto] [--pcie-gen 0] [--nvlink-bridges 0-1] [--json]
fakegpu daemon stop
```

`fakegpu daemon run` 在前景執行機器。機器檔案位於 `~/.fakegpu/machine`
（`config.json`、`events.jsonl`、`daemon.log` 與 `state/` 目錄）；設定 `FAKEGPU_HOME` 可改變位置。

## 控制協定

用戶端透過 Unix socket（機器目錄中的 `daemon.sock`）與機器通訊，每個連線送出一個 JSON
請求並收到一個 JSON 回應，兩者都以換行結尾。協定名稱為 `fakegpu.machine.v1`：

```text
request   {"v": 1, "id": 7, "method": "status", "params": {}}
response  {"v": 1, "id": 7, "ok": true, "result": {...}}
```

方法包括 `hello`、`status`、`events`、`config.get`、`config.update`、`process.signal`
與 `daemon.stop`，參數說明見 [service 模組](../../fakegpu/machine/service.py)。
`fakegpu.machine.client.MachineClient` 是現成的 Python 用戶端。所有請求都經過
`MachineService.dispatch`，因此其他傳輸方式（例如供網頁管理介面使用的 HTTP）可以直接沿用。
