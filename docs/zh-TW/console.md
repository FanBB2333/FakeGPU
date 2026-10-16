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

## 控制台

| 分頁 | 內容 |
|---|---|
| `1` Overview | 機器概要、每張 GPU 的卡片（記憶體、使用率、溫度、功耗、時脈、P-state）、程序表與最新事件 |
| `2` Processes | 所有已接入程序的命令列、使用者、階段、各裝置記憶體與活動；最近結束的程序 |
| `3` Devices | 單卡詳情：插槽、建模的拷貝與鏈路吞吐、配置器計數、依類別統計的記憶體、最大的幾次配置，以及歷史曲線（記憶體、感測器、PCIe/C2C/NVLink 吞吐） |
| `4` Events | 完整事件紀錄，可依文字與等級篩選 |
| `5` Machine | 設定表單（配置、平台、PCIe 插槽代數、NVLink 橋、遙測、背景模式、心跳逾時與記憶體警告門檻）、插槽圖與 GPU 鏈路矩陣、常駐程序與路徑資訊 |
| `6` Profiles | GPU profile 目錄；按 `a` 或 Enter 把選取的 profile 加入配置表單 |

| 按鍵 | 作用 |
|---|---|
| `k` / `K` / `i` | 向選取的程序傳送 SIGTERM / SIGKILL / SIGINT（會先確認） |
| `b` | 開啟或關閉背景模式 |
| `q` | 離開控制台 |
| `Q` | 停止機器（會先確認） |
| `s` | 機器停止後重新啟動 |

背景模式開啟時（預設），按 `q` 離開後機器繼續執行，可以用 `fakegpu console` 重新進入。
背景模式關閉時，離開控制台也會停止機器。停止機器不會停止正在執行的工作負載。

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
