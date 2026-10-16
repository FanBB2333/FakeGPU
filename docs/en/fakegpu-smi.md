# FakeGPU-SMI and monitoring

[Documentation index](README.md) · [简体中文](../zh-CN/fakegpu-smi.md) ·
[繁體中文](../zh-TW/fakegpu-smi.md)

`fakegpu nvidia-smi` shows simulated devices and the processes using them, in
the same style as NVIDIA's tool. `fakegpu metrics` exports the same state for
Prometheus.

## Publish state from a workload

State publishing is opt-in. Set a state directory before the workload calls
`fakegpu.init(...)` or starts through the native launcher. `build/` is ignored
by Git.

```bash
export FAKEGPU_SMI_STATE_DIR=build/smi

# Python FakeCUDA workload.
python3 your_script.py

# Unmodified native CUDA/NVML workload.
python3 -m fakegpu --build-dir build ./your_native_workload
```

## Inspect devices and processes

From another terminal:

```bash
# Compact device and process table; add "-l 1" to refresh every second.
python3 -m fakegpu nvidia-smi --state-dir build/smi

# Device inventory and detailed runtime/profile/allocator report.
python3 -m fakegpu nvidia-smi --state-dir build/smi -L
python3 -m fakegpu nvidia-smi --state-dir build/smi -q

# NVIDIA-style modeled topology and NVLink views.
python3 -m fakegpu nvidia-smi topo -m --state-dir build/smi
python3 -m fakegpu nvidia-smi nvlink -s --state-dir build/smi

# Modeled faults, compatibility warnings, publisher failures, and stale state.
python3 -m fakegpu nvidia-smi events --state-dir build/smi

# Modeled MIG GPU and compute instances.
python3 -m fakegpu nvidia-smi mig -lgi --state-dir build/smi
python3 -m fakegpu nvidia-smi mig -lci --state-dir build/smi

# Script-friendly GPU and process queries.
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

- `-q` shows the FakeGPU version, runtime policies, software versions, state
  freshness, catalog and native API coverage, device identity, memory
  categories, allocator activity, dispatch tracking, and per-process peaks.
- `-i` selects devices by index, UUID, PCI bus ID, or profile ID. `--json`
  prints the full normalized inventory. FakeGPU writes state schema v2 and can
  still read v1.
- `--query-runtime` exposes the same runtime details as CSV or JSON.
  `--help-query-runtime` lists every field.
- Native interception also publishes allocation lifetime, transfer volume,
  kernel launches, GEMM calls and FLOPs, compatibility events, and unsupported
  API counts. `FAKEGPU_SMI_DETAIL_LIMIT` and `FAKEGPU_SMI_MAX_STATE_BYTES` limit
  the retained detail and the state-file size.

## Model NVLink, MIG, and faults

These features are off by default. Set them before the workload starts. Invalid
input activates nothing and is reported as a configuration error.

- **NVLink**: `FAKEGPU_NVLINK_GROUPS="0,1;2,3"` creates full-mesh peers within
  each group. `FAKEGPU_NVLINK_BANDWIDTH_GBPS` defaults to `900`. This drives
  `topo -m`, `nvlink -s`, the `topology.*` and `nvlink.*` fields, and native
  NVML link queries.
- **MIG**: `FAKEGPU_MIG_LAYOUT="DEVICE:PROFILE:MEMORY_MIB[:COUNT];..."`, for
  example `FAKEGPU_MIG_LAYOUT="0:1g.10gb:10240:2;1:2g.20gb:20480"`. Each GPU
  instance owns one compute instance. Layouts that exceed the parent memory,
  eight slices, or eight instances are rejected. Per-instance memory use is not
  observed.
- **Faults**: `FAKEGPU_FAULT_EVENTS="DEVICE:CODE:SEVERITY[:COUNT];..."`, for
  example `FAKEGPU_FAULT_EVENTS="0:XID_79:critical;1:NVLINK_CRC:error:3"`.
  Severities are `info`, `warning`, `error`, and `critical`. The `events` view
  also shows unsupported API calls, publisher failures, and stale state, and
  the `health.*` fields expose modeled device status.

UUIDs and PCI bus IDs are stable simulated values. Temperature, fan speed, live
power, and hardware utilization show `N/A`. Topology, bandwidth, faults, and
health are modeled inputs. They are not hardware measurements or ECC/Xid data.

<a id="export-monitoring-metrics"></a>

## Export monitoring metrics

`fakegpu metrics` exports the normalized FakeGPU-SMI state as Prometheus text
(the default) or JSON, with no extra monitoring dependencies:

```bash
python3 -m fakegpu metrics --state-dir build/smi
python3 -m fakegpu metrics --state-dir build/smi --json
```

For local collection, start the bounded in-memory exporter:

```bash
python3 -m fakegpu metrics --state-dir build/smi --serve \
  --host 127.0.0.1 --port 9400 --interval 1 \
  --history-size 300 --max-process-series 128

curl http://127.0.0.1:9400/metrics
curl http://127.0.0.1:9400/healthz
curl http://127.0.0.1:9400/api/v1/history
```

- `/metrics` serves the latest device, process, runtime, MIG, topology,
  health, and publisher values. `/healthz` reports source and scrape status,
  and `/api/v1/history` returns recent samples.
- Process series keep the processes using the most memory: 128 by default and
  256 at most. `--max-process-series 0` turns them off. History keeps 300
  samples by default and 1,440 at most.
- Everything stays in memory. Use Prometheus or another scraper for long-term
  retention.
- The server binds to `127.0.0.1` without authentication. Put it behind an
  authenticated proxy before exposing it. Exported values keep FakeGPU-SMI's
  distinction between modeled and observed data.
