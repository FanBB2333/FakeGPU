# FakeGPU-SMI 与监控

[文档索引](README.md) · [English](../en/fakegpu-smi.md) ·
[繁體中文](../zh-TW/fakegpu-smi.md)

`fakegpu nvidia-smi` 以接近 NVIDIA 工具的格式显示模拟设备和使用它们的进程。
`fakegpu metrics` 把同样的状态导出给 Prometheus。

## 从工作负载发布状态

状态发布需要显式启用。请在工作负载调用 `fakegpu.init(...)` 或通过原生启动器运行
之前设置状态目录。`build/` 已被 Git 忽略。[后台机器](console.md)运行时，工作负载向它
发布状态；不指定状态来源的 `fakegpu nvidia-smi` 和 `fakegpu nvitop` 会显示这台机器，
包括空闲的 GPU。

```bash
export FAKEGPU_SMI_STATE_DIR=build/smi

# Python FakeCUDA 工作负载。
python3 your_script.py

# 未修改的原生 CUDA/NVML 工作负载。
python3 -m fakegpu --build-dir build ./your_native_workload
```

用 [`fakegpu run`](commands.md#run-a-command-on-simulated-gpus) 启动的命令不需要
状态目录：命令 `PATH` 上的 `nvidia-smi` 会为这次运行执行 `fakegpu nvidia-smi`，并
列出它的 GPU，即使还没有进程使用它们。

## 查看设备和进程

在另一个终端中：

```bash
# 紧凑的设备与进程表，加上 "-l 1" 可每秒刷新一次。
python3 -m fakegpu nvidia-smi --state-dir build/smi

# 设备列表，以及 runtime、profile 和 allocator 详细报告。
python3 -m fakegpu nvidia-smi --state-dir build/smi -L
python3 -m fakegpu nvidia-smi --state-dir build/smi -q

# 接近 NVIDIA 命令格式的建模拓扑和 NVLink 视图。
python3 -m fakegpu nvidia-smi topo -m --state-dir build/smi
python3 -m fakegpu nvidia-smi nvlink -s --state-dir build/smi

# 建模故障、兼容性警告、发布失败和过期状态。
python3 -m fakegpu nvidia-smi events --state-dir build/smi

# 建模 MIG GPU Instance 和 Compute Instance。
python3 -m fakegpu nvidia-smi mig -lgi --state-dir build/smi
python3 -m fakegpu nvidia-smi mig -lci --state-dir build/smi

# 适合脚本处理的 GPU 与进程查询。
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

- `-q` 显示 FakeGPU 版本、runtime 策略、软件版本、状态更新时间、profile 目录与
  原生 API 覆盖、设备标识、插槽、建模的拷贝与链路吞吐、显存分类、allocator 活动、
  dispatch 跟踪和各进程峰值。
- `-L` 输出 nvidia-smi 格式的 `GPU 0: NAME (UUID: GPU-...)` 行，MIG 设备缩进
  列在所属 GPU 下方。只有同时显示多台主机时，索引后才带主机名。profile、
  compute capability 和 PCI Bus ID 见 `-q` 与 `--json`。
- `-i` 按设备索引、UUID、PCI Bus ID 或 profile ID 筛选；没有匹配的 GPU 时和
  nvidia-smi 一样输出 `No devices were found` 并以状态 6 退出。`--json`
  输出完整的标准化数据。FakeGPU 写入 state schema v2，仍可读取 v1。
- `--query-runtime` 以 CSV 或 JSON 输出相同的运行时信息。`--help-query-runtime`
  列出全部字段。
- CSV 输出和 `nvidia-smi` 一样用 `, ` 分隔字段，只有包含逗号、引号或换行的单元格
  才加引号。`--format` 的 `csv`、`noheader`、`nounits` 可按任意顺序给出，
  默认输出 CSV。`--query-gpu=count` 在每一行重复 GPU 数量，符合 Accelerate 的预期；
  用 `-i` 选择部分 GPU 时仍计入全部 GPU。
- `--query-gpu` 与 `--query-compute-apps` 还接受 nvidia-smi 590 `--help-query-*`
  列出的全部字段与别名，列名与 nvidia-smi 一致。FakeGPU 未建模的字段（如
  `utilization.gpu`、`temperature.gpu`）输出 `[N/A]`。
- `memory.reserved` 在 nvidia-smi 中表示“NVIDIA 驱动和固件保留的显存”，
  FakeGPU 不建模，因此输出 `[N/A]`。此前 `memory.reserved` 报告的 caching
  allocator 保留显存改为 `memory.allocator_reserved`。
- 表格开头和 nvidia-smi 的标题行一样带有 `Driver Version:` 与 `CUDA Version:`。
  驱动版本是建模值 `570.195.03`，与 FakeGPU 原生 NVML 报告的版本一致；
  `--query-gpu=driver_version` 输出同一个值。在 workload 启动前设置
  `FAKEGPU_DRIVER_VERSION` 可建模其他版本。
- 原生拦截还会发布 allocation 生命周期、传输量、kernel launch、GEMM 调用与 FLOP、
  兼容性事件和 unsupported API 次数。`FAKEGPU_SMI_DETAIL_LIMIT` 与
  `FAKEGPU_SMI_MAX_STATE_BYTES` 分别限制保留的明细和状态文件大小。

## 建模 NVLink、MIG 和故障

这些功能默认关闭（后台机器上的进程会获得机器的平台），需要在工作负载启动前设置。
配置无效时不会激活任何内容，只报告配置错误。

- **平台与 NVLink**：`FAKEGPU_PLATFORM`（或后台机器的平台）把 GPU 放进 HGX/DGX、
  超级芯片或 PCIe 插槽，并连好 NVLink，见[平台与插槽](platforms.md)。也可以用
  `FAKEGPU_NVLINK_GROUPS="0,1;2,3"` 给组内每对设备一条速率为
  `FAKEGPU_NVLINK_BANDWIDTH_GBPS`（默认 `900`）的链路。`topo -m`（NV#、PIX、PXB、
  PHB、NODE、SYS 和 NUMA affinity）、`nvlink -s`（每条链路每个方向的 GB/s）、
  `topology.*`、`nvlink.*`、`platform`、`slot.type`、`numa.node`、`pcie.link.*`、
  `host_link.bandwidth` 字段以及原生 NVML 链路查询都基于这个模型。拷贝按建模的时间
  和链路吞吐记录（`copy.*` 和 `*.throughput.*` 字段），见
  [拷贝时间与吞吐](platforms.md#拷贝时间与吞吐)。
- **MIG**：`FAKEGPU_MIG_LAYOUT="DEVICE:PROFILE:MEMORY_MIB[:COUNT];..."`，例如
  `FAKEGPU_MIG_LAYOUT="0:1g.10gb:10240:2;1:2g.20gb:20480"`。每个 GPU Instance
  对应一个 Compute Instance。超出父设备显存、8 个 slice 或 8 个实例的配置会被
  拒绝。各实例的显存用量不做观测。
- **故障**：`FAKEGPU_FAULT_EVENTS="DEVICE:CODE:SEVERITY[:COUNT];..."`，例如
  `FAKEGPU_FAULT_EVENTS="0:XID_79:critical;1:NVLINK_CRC:error:3"`。严重度支持
  `info`、`warning`、`error` 和 `critical`。`events` 视图同时显示 unsupported API
  调用、发布失败和过期状态，`health.*` 字段给出设备的建模状态。

UUID 和 PCI Bus ID 是稳定的模拟值。温度、风扇转速、实时功耗和硬件利用率显示为
`N/A`。拓扑、带宽、故障和健康状态都是建模输入，不是硬件测量，也不是 ECC/Xid
数据。

<a id="export-monitoring-metrics"></a>

## 导出监控指标

`fakegpu metrics` 将 FakeGPU-SMI 的标准化状态导出为 Prometheus 文本（默认）或
JSON，不需要额外的监控依赖：

```bash
python3 -m fakegpu metrics --state-dir build/smi
python3 -m fakegpu metrics --state-dir build/smi --json
```

本地连续采集时，启动带内存上限的 exporter：

```bash
python3 -m fakegpu metrics --state-dir build/smi --serve \
  --host 127.0.0.1 --port 9400 --interval 1 \
  --history-size 300 --max-process-series 128

curl http://127.0.0.1:9400/metrics
curl http://127.0.0.1:9400/healthz
curl http://127.0.0.1:9400/api/v1/history
```

- `/metrics` 提供最新的设备、进程、runtime、MIG、拓扑、健康状态和发布器指标。
  `/healthz` 显示数据源与采集状态，`/api/v1/history` 返回近期样本。
- 进程序列优先保留显存占用最高的进程，默认 128 个，最多 256 个。
  `--max-process-series 0` 可以关闭。历史默认保留 300 个样本，最多 1,440 个。
- 数据只保存在内存中。长期留存请用 Prometheus 或其他采集系统。
- 服务只监听 `127.0.0.1`，不提供认证。对外开放前应放在带认证的代理后面。导出的
  数据沿用 FakeGPU-SMI 对建模值和观测值的区分。

## 在 macOS 或 Linux 上使用 pip 安装的 nvitop

先从源码目录安装监控依赖，然后在第一个终端保持工作负载运行：

```bash
conda activate fakegpu
python -m pip install '.[monitoring]'
FAKEGPU_SMI_STATE_DIR=/tmp/fakegpu-smi python - <<'PYTHON'
import time
import fakegpu
fakegpu.init(profile="l4", device_count=1)  # 模拟 profile。
import torch
x = torch.ones((2048, 2048), device="cuda:0")
time.sleep(600)
PYTHON
```

第二个终端执行：

```bash
conda activate fakegpu
fakegpu nvitop --state-dir /tmp/fakegpu-smi
# 只输出一次，不进入交互界面：
fakegpu nvitop --state-dir /tmp/fakegpu-smi -- --once
```

在 nvitop 中按 `q` 退出；工作负载用 Ctrl+C 停止。另一个命令
`fakegpu nvidia-smi -l 1` 的刷新循环需要用 Ctrl+C 退出，`q` 不生效。

这个入口运行已安装、未经修改的 nvitop，并显式加载项目自带的
`libnvidia-ml.dylib`（Linux 为 `libnvidia-ml.so.1`），每 0.25 秒更新原生
NVML 快照。直接运行普通 `nvitop` 不会启用桥接。无需安装 NVIDIA 驱动、替换系统
NVML 或设置 shell 别名；FakeCUDA 的计算仍由 CPU 执行。

设备名称、合成 UUID、总显存、报告显存、PID 和每个进程的占用均来自状态文件。
报告显存包含分配器预留及配置的 runtime 开销，因此释放张量后，可能需要调用
`torch.cuda.empty_cache()` 才会看到占用下降。相同合成 GPU UUID 的多个进程会
汇总显示。默认模式下，温度、功耗、GPU 利用率和进程 GPU 利用率显示 `N/A`；建模的 MIG、
NVLink 信息仍通过 `fakegpu nvidia-smi` 查看，不通过此桥接展示。

只读取本机仍在运行且心跳有效的发布者，默认两秒过期，可用
`--stale-after-seconds` 调整。已退出、已消失的进程及无效文件不会参与统计；同一
PID 的重复文件只保留最新快照。已发现的设备保留稳定的 NVML 句柄，最后一个进程
退出后显示零占用。新增设备或 profile 后需重启 nvitop，因为它在启动时选择设备
列表。没有活动发布者时，入口提示如何启动工作负载并返回状态码 1，不会生成默认设备。

其他 Python NVML 客户端可以在独立监控进程中、加载 NVML 之前，使用
`fakegpu.nvml.NvmlStateBridge` 上下文管理器，并在上下文内正常初始化及关闭
pynvml。关闭桥接会停止刷新并清空占用；动态库选择和句柄保留到该进程结束。

端到端回归会启动两个真实 FakeCUDA 工作进程，检查 NVML 显存、进程查询和
v1/v2/v3 缓冲区边界、nvitop API 与 `--once` 输出、释放显存、正常退出、强制
终止及空状态提示。安装 wheel 后，可在源码目录外复现：

```bash
FAKEGPU_CHECKOUT="$PWD"
(cd /tmp && python -I "$FAKEGPU_CHECKOUT/tests/support/nvml_bridge_smoke.py" --tui)
```


## 8 卡 A100：真实 torch 分配与参考遥测

在源码目录中启动第一个终端：

```bash
conda activate fakegpu
python examples/monitoring/torch_eight_a100.py
```

第二个终端：

```bash
conda activate fakegpu
fakegpu nvitop --state-dir /tmp/fakegpu-smi --telemetry reference
```

程序创建 8 张模拟 A100 SXM4 80GB，实际分配 CPU 支撑的 PyTorch 张量。每轮先在
各卡分配 32–60 MiB，再各增加 64 MiB，达到 96–124 MiB；随后释放新增张量，
最后全部释放。阶段切换时调用 `empty_cache()`，默认每阶段 8 秒，Ctrl+C 停止。
`--cycles 1` 只运行一轮。主机实际分配量约 1 GiB，不会分配 8 × 80 GiB。

`--telemetry reference` 显式启用 A100 显示模型，此时 nvitop 的标题栏显示
`Driver Version: FakeGPU (modeled)`。用于其他程序时，在
`fakegpu.init()` 前设置 `FAKEGPU_TELEMETRY=reference`，并初始化
`profile="a100", device_count=8`。观察器记录各设备 CPU torch 算子的执行时间及
估算的张量 I/O 字节数，通过 NVML 提供 GPU/显存引擎利用率、各进程活跃度、温度、
功耗、功率上限、P-state 和时钟。仅持有张量不会被当作计算负载。

[参考配置](../../fakegpu/data/a100_monitor_reference.json)明确区分来源和假设：

- [NVIDIA 官方规格](https://www.nvidia.com/en-us/data-center/a100/)提供标准 SXM
  功率上限和 HBM 带宽。
- [NVIDIA/nccl-tests 中用户公开的实机快照](https://github.com/NVIDIA/nccl-tests/issues/228)
  提供 8 卡空闲温度、功耗和 P0 读数，范围为 26–30°C、56–60W。
  只保留这些数值，不复制机器身份、UUID 或序列号。
- 动态利用率以 **CPU 算子执行占空比作为模拟输入**，功耗向规格上限插值，温度
  按显式热响应模型变化。显存引擎利用率按张量 I/O 与参考带宽计算，不是显存容量
  占用率。负载曲线属于模型假设，不是 A100 实测或性能校准结果。

实测的空闲读数只属于 `a100`（SXM4 80GB）。其他目录 profile 使用同样的活跃度输入和一条默认曲线：
空闲时 30°C、1 W；功耗线性升至该 profile 在目录中的功率上限（目录缺失时按 300 W）；温度沿用
同样的 45°C 温升和 10 秒热响应；负载时钟取自目录（缺失时按 1500 MHz）。这些默认值是显示用的
模型假设，不是测量值。缺少活跃度观测的发布者和目录之外的 profile 仍保留 `N/A`。GPU 风扇与公开快照一致，继续显示 `N/A`；CUDA 驱动及未支持的视频、
PCIe 遥测也不会虚构。默认 `--telemetry none` 保持原行为，macOS 上无需 NVIDIA 驱动。

维护的 `tests/support/a100_monitor_smoke.py` 会启动该 torch 程序及原版 nvitop
交互界面，用 pyte 还原终端刷新，检查 8 卡显存的增加与释放、模型遥测、进程退出
清理，输出仅含数值摘要。
