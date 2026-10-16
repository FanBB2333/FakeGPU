# 平台与插槽

[文档索引](README.md) · [English](../en/platforms.md) ·
[繁體中文](../zh-TW/platforms.md)

真实的 GPU 装在插槽里，插槽决定了它如何连到 CPU 和其他 GPU。SXM 模组安装在为某一代
GPU 设计的基板上，彼此通过 NVLink 通信，通常经过 NVSwitch 芯片；PCIe 卡插在 PCIe
插槽里，链路速率取卡和插槽中较低的一方，还可以用 NVLink 桥两两相连；Grace 超级芯片
通过 NVLink-C2C 把 GPU 接到 CPU。FakeGPU 用平台模板来建模这些差异。

## 平台模板

`fakegpu platform list` 列出所有模板；`fakegpu platform show` 把一个设备布局放进
某个模板并显示结果。

| 平台 | 插槽 | GPU 互连 |
|---|---|---|
| `hgx-a100-8`、`hgx-h100-8`、`hgx-b200-8`、`hgx-b300-8` | 8 个 SXM | NVSwitch：每块 GPU 用全部 NVLink 连到其他每块 GPU（A100 为 NV12，Hopper 和 Blackwell 为 NV18） |
| `hgx-a100-4`、`hgx-h100-4` | 4 个 SXM | 直连，每对 GPU 4 条（A100）或 6 条（H100） |
| `dgx-1-p100`、`dgx-1-v100` | 8 个 SXM2 | Hybrid cube-mesh，按 NVIDIA 公布的成对连线 |
| `dgx-2` | 16 个 SXM3 | 跨两块基板的 NVSwitch；两级 PCIe switch |
| `gh200` | 1 个超级芯片 | 通过 NVLink-C2C 连到 Grace CPU，每个方向 450 GB/s |
| `gb200-nvl72-tray`、`gb300-nvl72-tray` | 4 块超级芯片 GPU | 两颗 Grace CPU 通过 C2C 相连；每块 GPU 有 18 条 NVLink 接入机柜的 switch tray |
| `dgx-station-gb300` | 1 个超级芯片 + 1 个 PCIe Gen5 x16 | GB300 走 C2C；可额外插一张 RTX PRO 卡 |
| `dgx-spark` | 1 个超级芯片 | GB10 GPU 与 CPU 共享统一内存 |
| `pcie-server` | 8 个 PCIe Gen5 x16 | 双路 CPU，四个 PCIe switch，每个带两个插槽 |
| `workstation` | 4 个 PCIe Gen5 x16 | 单路 CPU，每个插槽在独立的 root complex 上 |
| `desktop` | 2 个 PCIe Gen5 | 一张卡时 x16，两张卡时 x8/x8 |
| `laptop`、`jetson` | 1 | 笔记本 PCIe 链路；Jetson 统一内存 |
| `generic` | 最多 64 个，任意 GPU | 不做约束：每块 GPU 以自身链路挂在同一个 switch 下，没有 NVLink |

基板和超级芯片的布局依据 NVIDIA 的 DGX/HGX 指南、Fabric Manager 用户指南和架构白皮书；
[`platforms.json`](../../fakegpu/data/platforms.json) 中每个模板都列出了来源。PCIe
服务器、工作站、台式机和笔记本模板是代表性布局，不对应具体产品。

## 选择平台

`auto`（默认值）按 GPU 的外形规格选择模板：

- SXM 模组放到能装这种模组、插槽数足够的最小基板上：`a100:2` 用 4-GPU HGX A100
  基板，`a100:8` 用 8-GPU 基板，`v100:8` 用 DGX-1。
- 超级芯片放到对应的 Grace 平台；GB300 DGX Station 还可以多插一张 PCIe 卡。
- PCIe 卡放进台式机（最多两张消费级卡）、工作站（最多四张，或含工作站卡）或 PCIe
  服务器（数据中心卡或多于四张卡）。外形规格未知的 GPU 按 PCIe 卡处理。

不可能存在的布局会被拒绝并给出提示：一块基板只装一种 GPU 型号，所以
`a100:4,h100:4` 会失败；SXM 模组和 PCIe 卡不会出现在同一个平台上；每个平台的插槽数
是固定的。`generic` 接受任意布局。

## 链路如何建模

- **PCIe**：GPU 以卡和插槽中较低的代数和宽度建立链路。每个方向每条 lane 的可用带宽：
  Gen3 为 0.985 GB/s、Gen4 为 1.97、Gen5 为 3.94（128b/130b 编码），Gen6 为 7.375
  （每个 256 字节 FLIT 含 236 字节数据）。可以统一调高或调低平台所有 PCIe 插槽的代数。
- **CPU 链路**：主机内存与 GPU 之间的数据经 PCIe 链路传输；超级芯片走 NVLink-C2C；
  DGX Spark 和 Jetson 在共享内存内拷贝（取显存带宽的一半，因为拷贝要读写每个字节）。
  更窄的上游链路会限制 CPU 链路，例如每块 B300 的 Gen6 ConnectX-8 后面是 Gen5 x16
  主机链路。
- **NVLink**：每块 GPU 拥有其 profile 记录的 NVLink 端口数（A100 12 条、H100 18 条、
  B200 18 条、V100 6 条、RTX 3090 4 条），每条链路每个方向的速率为 NVLink 1 的
  20 GB/s、NVLink 2 至 4 的 25 GB/s、NVLink 5 的 50 GB/s，或由桥接带宽推出的速率
  （GA102 卡为 14.0625 GB/s）。NVSwitch 基板把全部端口绑定到每个对端（`NV12`、
  `NV18`）；直连布线则每对 GPU 各有自己的链路（`NV4`、`NV6`，DGX-1 上为
  `NV1`/`NV2`）。
- **Peer access**：通过 NVLink 相连时，`cudaDeviceCanAccessPeer` 和
  `torch.cuda.can_device_access_peer` 为真；通过 PCIe 时，两块 GPU 在同一 CPU
  socket 下（`PIX`、`PXB`、`PHB` 或 `NODE`）为真。跨 socket（`SYS`）、GeForce
  RTX 4090（NVIDIA 关闭了它的 PCIe peer access），以及超级芯片平台上没有 NVLink 的
  GPU 之间（NVIDIA 未说明支持 peer access）为假。

原生程序看到的是同一个模型。NVML 报告 PCIe 代数和宽度（当前值、最大值和 GPU 自身的
最大值）、每个 NVLink 端口及其对端（GPU 或 NVSwitch），并提供
`nvmlDeviceGetTopologyCommonAncestor` 和 `nvmlDeviceGetP2PStatus`。
`cudaDeviceCanAccessPeer`、`cuDeviceCanAccessPeer` 和 `cuDeviceGetP2PAttribute`
按拓扑作答；对没有 peer access 的 GPU 对，`cudaDeviceEnablePeerAccess` 返回
`cudaErrorPeerAccessUnsupported`。

`fakegpu nvidia-smi topo -m` 用 NVIDIA 的图例显示结果：

```text
      GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7  NUMA Affinity
GPU0  X     NV1   NV1   NV2   NV2   SYS   SYS   SYS   0
```

## 配置后台机器

后台机器（见[控制台说明](console.md)）把平台和设备布局一起保存：

```bash
fakegpu daemon config --devices rtx3090:2 --nvlink-bridges 0-1
fakegpu daemon config --devices l40s:8 --platform pcie-server --pcie-gen 4
fakegpu daemon config --devices h100:8 --platform auto
```

`--platform` 取 `auto` 或模板 ID；`--pcie-gen` 覆盖 PCIe 插槽的代数（`0` 表示沿用
模板）；`--nvlink-bridges` 用 NVLink 桥连接 PCIe 卡（`0-1,2-3`；H200 NVL 还支持
`0-1-2-3` 这样的四路桥）。只有带 NVLink 接口的卡才能接桥，且两端必须是同一型号。
控制台的 Machine 页提供同样的设置，并显示插槽图和链路矩阵。使用机器布局的 FakeCUDA
进程会获得这个平台。

## 配置单个进程

在后台机器之外，进程默认没有平台，除非显式指定：

```bash
fakegpu run --devices a100-pcie-80gb:4 --platform workstation ./app
FAKEGPU_PLATFORM=auto FAKEGPU_PROFILES=h100:8 python train.py
```

```python
import fakegpu
fakegpu.init(devices="v100:8", platform="auto")
```

`FAKEGPU_PCIE_GEN` 和 `FAKEGPU_NVLINK_BRIDGES` 分别设置 PCIe 插槽代数和桥接。没有
平台时，每块 GPU 以自身的链路代数和宽度挂在同一个 PCIe switch 下，没有 NVLink，任意
两块 GPU 之间都有 peer access，与引入平台之前的行为一致。

平台通过三个环境变量传给 FakeCUDA 进程；对于以其他方式启动的程序，可以用
`fakegpu platform env --devices ...` 打印这些变量。原生程序如果只设置了
`FAKEGPU_PLATFORM`，会报告配置错误，因为解析平台需要 Python：

| 变量 | 内容 |
|---|---|
| `FAKEGPU_PLATFORM` | 模板 ID |
| `FAKEGPU_SLOTS` | 每块 GPU 一项，以 `;` 分隔，字段为 `type`、`pcie`（`GENxWIDTH` 或 `none`）、`gpu`（GPU 自身的最高链路）、`host`（更窄的上游链路）、`link`（`pcie`、`c2c` 或 `unified`）、`hostbw`（每个方向的 GB/s）、`numa`、`bridge`、`switch`（`a` 或 `a.b`）和 `p2p`（`0` 关闭 PCIe peer access） |
| `FAKEGPU_NVLINK_FABRIC` | 以 `;` 分隔的分组 `KIND:MEMBERS:LINKS:GBPS`：`mesh` 让组内每对成员各有 LINKS 条直连链路，`switch` 让每个成员有 LINKS 条链路接入 NVSwitch；GBPS 为每条链路每个方向的速率 |

显式设置的 `FAKEGPU_SLOTS`、`FAKEGPU_NVLINK_FABRIC` 或旧的
`FAKEGPU_NVLINK_GROUPS` 优先于平台。`FAKEGPU_NVLINK_GROUPS="0,1;2,3"` 仍然给组内每对
GPU 一条速率为 `FAKEGPU_NVLINK_BANDWIDTH_GBPS`（默认 900）的链路。

## 局限

- 部分模组没有记录 NVLink 端口数（例如 A800 和 H800 SXM、H20、V100 SXM3），它们沿用
  基板的端口数，`fakegpu platform show` 会注明。
- NVIDIA 公布了每个 GB200/GB300 超级芯片 900 GB/s 的 C2C 带宽，但没有说明两块 GPU
  如何分享；每块 GPU 按每个方向 450 GB/s 建模。
- H200 4-GPU HGX 的布线、GB200 NVL4 的桥接拓扑和 DGX Spark 的 C2C 速率没有公开，因此
  没有模板使用它们。
- RTX 4090 以外的 GeForce 卡的 PCIe peer access 按上面的 socket 规则处理；NVIDIA 没有
  公布逐型号的策略。
