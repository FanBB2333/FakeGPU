# 后台机器与控制台

[文档索引](README.md) · [English](../en/console.md) ·
[繁體中文](../zh-TW/console.md)

FakeGPU 可以在后台运行一台模拟 GPU 机器。机器运行期间启动的 FakeCUDA 进程会自动接入。
终端控制台显示这台机器的设备、进程和事件，并可以修改它的设置。离开控制台后，机器仍会继续运行。在源码目录中安装 `console` 扩展：

```bash
FAKEGPU_BUILD_DIR="$PWD/build" python3 -m pip install '.[torch,console]'
fakegpu console
```

如果机器尚未运行，`fakegpu console` 会先启动它。首次启动使用 HGX A100 8-GPU 基板上的
八张 A100 SXM4 80GB，并启用参考遥测模型；可以用 `--devices`（例如 `--devices h100:8`
或 `--devices rtx4090:2`）指定设备布局，用 `--platform`、`--pcie-gen` 和
`--nvlink-bridges` 指定主机平台（见[平台与插槽](platforms.md)），用
`--telemetry none|reference` 指定遥测模型。这些选项同样会更新正在运行的机器。

## 在机器上运行工作负载

机器运行期间，任何调用 `fakegpu.init()` 的进程都会把状态发布到这台机器，并使用机器的设备布局：

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

如果进程向 `fakegpu.init()` 传入了 `profile`、`devices` 或 `device_count`，或者设置了
`FAKEGPU_PROFILE`、`FAKEGPU_PROFILES`、`FAKEGPU_DEVICE_COUNT`，它会保留自己的布局。
设置 `FAKEGPU_MACHINE=off` 可以让进程不接入机器。新布局只对修改之后启动的进程生效；
已经在运行的进程保留启动时的设备。

`fakegpu nvitop` 和 `fakegpu nvidia-smi --state-dir ~/.fakegpu/machine/state`
读取的是同一份状态，所以机器上的进程也会出现在这两个工具里。

## 控制台

| 标签页 | 内容 |
|---|---|
| `1` Overview | 机器概要、每张 GPU 的卡片（显存、利用率、温度、功耗、频率、P-state）、进程表和最新事件 |
| `2` Processes | 所有已接入进程的命令行、用户、阶段、各设备显存和活动；最近结束的进程 |
| `3` Devices | 单卡详情：插槽、建模的拷贝与链路吞吐、分配器计数、按类别统计的显存、最大的几次分配，以及历史曲线（显存、传感器、PCIe/C2C/NVLink 吞吐） |
| `4` Events | 完整事件日志，可按文本和级别过滤 |
| `5` Machine | 设置表单（布局、平台、PCIe 插槽代数、NVLink 桥、遥测、后台模式、心跳超时和显存告警阈值）、插槽图和 GPU 链路矩阵、守护进程与路径信息 |
| `6` Profiles | GPU profile 目录；按 `a` 或 Enter 把选中的 profile 加入布局表单 |

| 按键 | 作用 |
|---|---|
| `k` / `K` / `i` | 向选中的进程发送 SIGTERM / SIGKILL / SIGINT（会先确认） |
| `b` | 开启或关闭后台模式 |
| `q` | 离开控制台 |
| `Q` | 停止机器（会先确认） |
| `s` | 机器停止后重新启动 |

后台模式开启时（默认），按 `q` 离开后机器继续运行，可以用 `fakegpu console` 重新进入。
后台模式关闭时，离开控制台也会停止机器。停止机器不会停止正在运行的工作负载。

## 事件

机器会记录进程和设备上发生的事情：

| 事件 | 含义 |
|---|---|
| `process.attached` | 一个 FakeCUDA 进程接入了机器 |
| `process.exited` | 进程发布了最终状态后退出 |
| `process.terminated` | 进程因 SIGTERM 结束 |
| `process.vanished` | 进程没有发出最后一次心跳就结束了，例如被 SIGKILL 或崩溃 |
| `process.stalled` / `process.resumed` | 存活的进程停止或恢复发布状态 |
| `process.signal_sent` | 控制台或其他客户端发送了信号 |
| `memory.high` / `memory.normal` / `memory.oversubscribed` | 设备显存越过告警阈值或超出容量 |
| `config.changed` | 布局或某项设置发生变化 |
| `state.invalid` | 某个状态文件无法读取 |
| `daemon.started` / `daemon.stopped` | 机器启动或停止 |

因此，在 nvitop 中或用 `kill` 结束的进程会显示为 `process.terminated`（SIGTERM）
或 `process.vanished`（SIGKILL）。最近 1,000 条事件在机器重启后仍会保留。

## 在 shell 中管理机器

```bash
fakegpu daemon start [--devices a100:8] [--telemetry reference] [--background on]
fakegpu daemon status [--json]
fakegpu daemon events [-n 50] [-f] [--json]
fakegpu daemon config [--devices a100:4] [--platform auto] [--pcie-gen 0] [--nvlink-bridges 0-1] [--json]
fakegpu daemon stop
```

`fakegpu daemon run` 在前台运行机器。机器文件位于 `~/.fakegpu/machine`
（`config.json`、`events.jsonl`、`daemon.log` 和 `state/` 目录）；设置 `FAKEGPU_HOME` 可以改变位置。

## 控制协议

客户端通过 Unix socket（机器目录中的 `daemon.sock`）与机器通信，每个连接发送一个 JSON
请求并收到一个 JSON 响应，二者都以换行结尾。协议名为 `fakegpu.machine.v1`：

```text
request   {"v": 1, "id": 7, "method": "status", "params": {}}
response  {"v": 1, "id": 7, "ok": true, "result": {...}}
```

方法包括 `hello`、`status`、`events`、`config.get`、`config.update`、`process.signal`
和 `daemon.stop`，参数说明见 [service 模块](../../fakegpu/machine/service.py)。
`fakegpu.machine.client.MachineClient` 是现成的 Python 客户端。所有请求都经过
`MachineService.dispatch`，因此其他传输方式（例如供网页管理界面使用的 HTTP）可以直接复用它。
