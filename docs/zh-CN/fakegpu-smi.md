# FakeGPU-SMI 与监控

[文档索引](README.md) · [English](../en/fakegpu-smi.md) ·
[繁體中文](../zh-TW/fakegpu-smi.md)

`fakegpu nvidia-smi` 以接近 NVIDIA 工具的格式显示模拟设备和使用它们的进程。
`fakegpu metrics` 把同样的状态导出给 Prometheus。

## 从工作负载发布状态

状态发布需要显式启用。请在工作负载调用 `fakegpu.init(...)` 或通过原生启动器运行
之前设置状态目录。`build/` 已被 Git 忽略。

```bash
export FAKEGPU_SMI_STATE_DIR=build/smi

# Python FakeCUDA 工作负载。
python3 your_script.py

# 未修改的原生 CUDA/NVML 工作负载。
python3 -m fakegpu --build-dir build ./your_native_workload
```

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
  原生 API 覆盖、设备标识、显存分类、allocator 活动、dispatch 跟踪和各进程峰值。
- `-i` 按设备索引、UUID、PCI Bus ID 或 profile ID 筛选。`--json` 输出完整的标准化
  数据。FakeGPU 写入 state schema v2，仍可读取 v1。
- `--query-runtime` 以 CSV 或 JSON 输出相同的运行时信息。`--help-query-runtime`
  列出全部字段。
- 原生拦截还会发布 allocation 生命周期、传输量、kernel launch、GEMM 调用与 FLOP、
  兼容性事件和 unsupported API 次数。`FAKEGPU_SMI_DETAIL_LIMIT` 与
  `FAKEGPU_SMI_MAX_STATE_BYTES` 分别限制保留的明细和状态文件大小。

## 建模 NVLink、MIG 和故障

这些功能默认关闭，需要在工作负载启动前设置。配置无效时不会激活任何内容，只报告
配置错误。

- **NVLink**：`FAKEGPU_NVLINK_GROUPS="0,1;2,3"` 为每组内的设备建立全连接关系。
  `FAKEGPU_NVLINK_BANDWIDTH_GBPS` 默认为 `900`。`topo -m`、`nvlink -s`、
  `topology.*` 和 `nvlink.*` 字段以及原生 NVML 链路查询都基于这个模型。
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
