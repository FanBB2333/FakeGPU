# 平台與插槽

[文件索引](README.md) · [English](../en/platforms.md) ·
[简体中文](../zh-CN/platforms.md)

真實的 GPU 裝在插槽裡，插槽決定了它如何連到 CPU 和其他 GPU。SXM 模組安裝在為某一代
GPU 設計的基板上，彼此透過 NVLink 通訊，通常經過 NVSwitch 晶片；PCIe 卡插在 PCIe
插槽裡，鏈路速率取卡和插槽中較低的一方，還可以用 NVLink 橋兩兩相連；Grace 超級晶片
透過 NVLink-C2C 把 GPU 接到 CPU。FakeGPU 用平台範本來建模這些差異。

## 平台範本

`fakegpu platform list` 列出所有範本；`fakegpu platform show` 把一個裝置配置放進
某個範本並顯示結果。

| 平台 | 插槽 | GPU 互連 |
|---|---|---|
| `hgx-a100-8`、`hgx-h100-8`、`hgx-b200-8`、`hgx-b300-8` | 8 個 SXM | NVSwitch：每張 GPU 以全部 NVLink 連到其他每張 GPU（A100 為 NV12，Hopper 與 Blackwell 為 NV18） |
| `hgx-a100-4`、`hgx-h100-4` | 4 個 SXM | 直連，每對 GPU 4 條（A100）或 6 條（H100） |
| `dgx-1-p100`、`dgx-1-v100` | 8 個 SXM2 | Hybrid cube-mesh，依 NVIDIA 公布的成對連線 |
| `dgx-2` | 16 個 SXM3 | 跨兩塊基板的 NVSwitch；兩級 PCIe switch |
| `gh200` | 1 個超級晶片 | 透過 NVLink-C2C 連到 Grace CPU，每個方向 450 GB/s |
| `gb200-nvl72-tray`、`gb300-nvl72-tray` | 4 張超級晶片 GPU | 兩顆 Grace CPU 以 C2C 相連；每張 GPU 有 18 條 NVLink 接入機櫃的 switch tray |
| `dgx-station-gb300` | 1 個超級晶片 + 1 個 PCIe Gen5 x16 | GB300 走 C2C；可額外插一張 RTX PRO 卡 |
| `dgx-spark` | 1 個超級晶片 | GB10 GPU 與 CPU 共用統一記憶體 |
| `pcie-server` | 8 個 PCIe Gen5 x16 | 雙路 CPU，四個 PCIe switch，每個帶兩個插槽 |
| `workstation` | 4 個 PCIe Gen5 x16 | 單路 CPU，每個插槽在獨立的 root complex 上 |
| `desktop` | 2 個 PCIe Gen5 | 一張卡時 x16，兩張卡時 x8/x8 |
| `laptop`、`jetson` | 1 | 筆電 PCIe 鏈路；Jetson 統一記憶體 |
| `generic` | 最多 64 個，任意 GPU | 不做約束：每張 GPU 以自身鏈路掛在同一個 switch 下，沒有 NVLink |

基板與超級晶片的配置依據 NVIDIA 的 DGX/HGX 指南、Fabric Manager 使用者指南與架構
白皮書；[`platforms.json`](../../fakegpu/data/platforms.json) 中每個範本都列出了來源。
PCIe 伺服器、工作站、桌上型電腦與筆電範本是代表性配置，不對應具體產品。

## 選擇平台

`auto`（預設值）依 GPU 的外形規格選擇範本：

- SXM 模組放到能裝這種模組、插槽數足夠的最小基板上：`a100:2` 用 4-GPU HGX A100
  基板，`a100:8` 用 8-GPU 基板，`v100:8` 用 DGX-1。
- 超級晶片放到對應的 Grace 平台；GB300 DGX Station 還可以多插一張 PCIe 卡。
- PCIe 卡放進桌上型電腦（最多兩張消費級卡）、工作站（最多四張，或含工作站卡）或
  PCIe 伺服器（資料中心卡或多於四張卡）。外形規格未知的 GPU 視為 PCIe 卡。

不可能存在的配置會被拒絕並給出提示：一塊基板只裝一種 GPU 型號，所以
`a100:4,h100:4` 會失敗；SXM 模組與 PCIe 卡不會出現在同一個平台上；每個平台的插槽數
是固定的。`generic` 接受任意配置。

## 鏈路如何建模

- **PCIe**：GPU 以卡和插槽中較低的代數與寬度建立鏈路。每個方向每條 lane 的可用頻寬：
  Gen3 為 0.985 GB/s、Gen4 為 1.97、Gen5 為 3.94（128b/130b 編碼），Gen6 為 7.375
  （每個 256 位元組 FLIT 含 236 位元組資料）。可以統一調高或調低平台所有 PCIe 插槽的
  代數。
- **CPU 鏈路**：主機記憶體與 GPU 之間的資料經 PCIe 鏈路傳輸；超級晶片走
  NVLink-C2C；DGX Spark 與 Jetson 在共用記憶體內拷貝（取記憶體頻寬的一半，因為拷貝要
  讀寫每個位元組）。較窄的上游鏈路會限制 CPU 鏈路，例如每張 B300 的 Gen6
  ConnectX-8 後面是 Gen5 x16 主機鏈路。
- **NVLink**：每張 GPU 擁有其 profile 記錄的 NVLink 埠數（A100 12 條、H100 18 條、
  B200 18 條、V100 6 條、RTX 3090 4 條），每條鏈路每個方向的速率為 NVLink 1 的
  20 GB/s、NVLink 2 至 4 的 25 GB/s、NVLink 5 的 50 GB/s，或由橋接頻寬推出的速率
  （GA102 卡為 14.0625 GB/s）。NVSwitch 基板把全部埠綁定到每個對端（`NV12`、
  `NV18`）；直連佈線則每對 GPU 各有自己的鏈路（`NV4`、`NV6`，DGX-1 上為
  `NV1`/`NV2`）。
- **Peer access**：透過 NVLink 相連時，`cudaDeviceCanAccessPeer` 與
  `torch.cuda.can_device_access_peer` 為真；透過 PCIe 時，兩張 GPU 在同一 CPU
  socket 下（`PIX`、`PXB`、`PHB` 或 `NODE`）為真。跨 socket（`SYS`）、GeForce
  RTX 4090（NVIDIA 關閉了它的 PCIe peer access），以及超級晶片平台上沒有 NVLink 的
  GPU 之間（NVIDIA 未說明支援 peer access）為假。

原生程式看到的是同一個模型。NVML 回報 PCIe 代數與寬度（目前值、最大值與 GPU 自身的
最大值）、每個 NVLink 埠及其對端（GPU 或 NVSwitch），並提供
`nvmlDeviceGetTopologyCommonAncestor` 與 `nvmlDeviceGetP2PStatus`。
`cudaDeviceCanAccessPeer`、`cuDeviceCanAccessPeer` 與 `cuDeviceGetP2PAttribute`
依拓撲作答；對沒有 peer access 的 GPU 對，`cudaDeviceEnablePeerAccess` 回傳
`cudaErrorPeerAccessUnsupported`。

`fakegpu nvidia-smi topo -m` 以 NVIDIA 的圖例顯示結果：

```text
      GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7  NUMA Affinity
GPU0  X     NV1   NV1   NV2   NV2   SYS   SYS   SYS   0
```

## 拷貝時間與吞吐

FakeGPU 的拷貝以主機記憶體的速度完成，平台不會拖慢程式。主機記憶體與 GPU 之間、兩張
GPU 之間的每次拷貝，都會依它在建模鏈路上所需的時間記錄下來：

- 主機到 GPU 及反方向走 GPU 的 CPU 鏈路：PCIe、NVLink-C2C，或 DGX Spark 與 Jetson
  上的共用記憶體（只計時間，不佔鏈路）。
- GPU 之間有 NVLink 時走 NVLink；有 PCIe peer access 時走 PCIe 點對點；否則經主機
  記憶體中轉，經過兩張 GPU 各自的 CPU 鏈路，取兩者中較慢的速率。
- 同一鏈路同一方向上的拷貝依序進行，一批拷貝會讓鏈路忙碌它們時間之和那麼久。吞吐是
  鏈路在最近一秒內送達的資料量。

FakeCUDA 記錄 `Tensor.to/.cuda/.cpu`、`Module.to/.cuda`、指定 CUDA 裝置的
`torch.tensor` 與 `torch.as_tensor`、跨裝置的 `copy_`，以及 `item`/`tolist`。原生
程式記錄 `cudaMemcpy*` 與 `cuMemcpy*` 傳輸。結果顯示在：

- `fakegpu nvidia-smi -q`：`Copies (modeled)` 給出每個方向的位元組數與建模時間，
  `Link throughput (modeled)` 給出每條鏈路的吞吐；查詢欄位 `copy.h2d`、`copy.d2h`、
  `copy.peer`（MiB）、`copy.modeled_time`（秒），以及 `pcie.throughput.tx/rx`、
  `nvlink.throughput.tx/rx`、`c2c.throughput.tx/rx`（KB/s）。
- NVML：`nvmlDeviceGetPcieThroughput`，以及 `nvmlDeviceGetFieldValues` 的 NVLink
  鏈路數與每條鏈路的資料計數器；原生程式直接可見，`fakegpu nvitop` 等 NVML 用戶端經
  state bridge 可見。
- 控制台的 `Link I/O` 欄，以及裝置詳情中的 `Copies`、`Links` 列與 PCIe、C2C、
  NVLink 趨勢圖。
- 每個程序結束時印出的報告（每張 GPU 一行 `Copies (modeled)`），以及原生 JSON 報告
  中每個裝置的 `transfer` 物件。

沒有平台時，拷貝走每張 GPU 自身的 PCIe 鏈路。

## 設定背景機器

背景機器（見[控制台說明](console.md)）把平台和裝置配置一起儲存：

```bash
fakegpu daemon config --devices rtx3090:2 --nvlink-bridges 0-1
fakegpu daemon config --devices l40s:8 --platform pcie-server --pcie-gen 4
fakegpu daemon config --devices h100:8 --platform auto
```

`--platform` 取 `auto` 或範本 ID；`--pcie-gen` 覆寫 PCIe 插槽的代數（`0` 表示沿用
範本）；`--nvlink-bridges` 以 NVLink 橋連接 PCIe 卡（`0-1,2-3`；H200 NVL 還支援
`0-1-2-3` 這樣的四路橋）。只有帶 NVLink 介面的卡才能接橋，且兩端必須是同一型號。
控制台的 Machine 頁提供同樣的設定，並顯示插槽圖與鏈路矩陣。使用機器配置的 FakeCUDA
程序會取得這個平台。

## 設定單一程序

在背景機器之外，程序預設沒有平台，除非明確指定：

```bash
fakegpu run --devices a100-pcie-80gb:4 --platform workstation ./app
FAKEGPU_PLATFORM=auto FAKEGPU_PROFILES=h100:8 python train.py
```

```python
import fakegpu
fakegpu.init(devices="v100:8", platform="auto")
```

`FAKEGPU_PCIE_GEN` 與 `FAKEGPU_NVLINK_BRIDGES` 分別設定 PCIe 插槽代數與橋接。沒有
平台時，每張 GPU 以自身的鏈路代數與寬度掛在同一個 PCIe switch 下，沒有 NVLink，任意
兩張 GPU 之間都有 peer access，與引入平台之前的行為一致。

平台透過三個環境變數傳給 FakeCUDA 程序；對於以其他方式啟動的程式，可以用
`fakegpu platform env --devices ...` 印出這些變數。原生程式若只設定了
`FAKEGPU_PLATFORM`，會回報設定錯誤，因為解析平台需要 Python：

| 變數 | 內容 |
|---|---|
| `FAKEGPU_PLATFORM` | 範本 ID |
| `FAKEGPU_SLOTS` | 每張 GPU 一項，以 `;` 分隔，欄位為 `type`、`pcie`（`GENxWIDTH` 或 `none`）、`gpu`（GPU 自身的最高鏈路）、`host`（較窄的上游鏈路）、`link`（`pcie`、`c2c` 或 `unified`）、`hostbw`（每個方向的 GB/s）、`numa`、`bridge`、`switch`（`a` 或 `a.b`）與 `p2p`（`0` 關閉 PCIe peer access） |
| `FAKEGPU_NVLINK_FABRIC` | 以 `;` 分隔的分組 `KIND:MEMBERS:LINKS:GBPS`：`mesh` 讓組內每對成員各有 LINKS 條直連鏈路，`switch` 讓每個成員有 LINKS 條鏈路接入 NVSwitch；GBPS 為每條鏈路每個方向的速率 |

明確設定的 `FAKEGPU_SLOTS`、`FAKEGPU_NVLINK_FABRIC` 或舊的
`FAKEGPU_NVLINK_GROUPS` 優先於平台。`FAKEGPU_NVLINK_GROUPS="0,1;2,3"` 仍然給組內每對
GPU 一條速率為 `FAKEGPU_NVLINK_BANDWIDTH_GBPS`（預設 900）的鏈路。

## 限制

- 部分模組沒有記錄 NVLink 埠數（例如 A800 與 H800 SXM、H20、V100 SXM3），它們沿用
  基板的埠數，`fakegpu platform show` 會註明。
- NVIDIA 公布了每個 GB200/GB300 超級晶片 900 GB/s 的 C2C 頻寬，但沒有說明兩張 GPU
  如何分享；每張 GPU 以每個方向 450 GB/s 建模。
- H200 4-GPU HGX 的佈線、GB200 NVL4 的橋接拓撲與 DGX Spark 的 C2C 速率沒有公開，因此
  沒有範本使用它們。
- RTX 4090 以外的 GeForce 卡的 PCIe peer access 依上面的 socket 規則處理；NVIDIA 沒有
  公布逐型號的策略。
- 一張 GPU 發往不同對端的拷貝共用它的 NVLink 時間線，因此經不同直連鏈路的拷貝在模型
  中依序進行。同一 PCIe switch 或 CPU socket 下的 GPU 不分攤其頻寬。
- NVML 把一張 GPU 的 NVLink 流量平均分到它的各個埠上。
