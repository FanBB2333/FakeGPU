# FakeGPU-SMI and monitoring

[Documentation index](README.md) · [简体中文](../zh-CN/fakegpu-smi.md) ·
[繁體中文](../zh-TW/fakegpu-smi.md)

`fakegpu nvidia-smi` shows simulated devices and the processes using them, in
the same style as NVIDIA's tool. `fakegpu metrics` exports the same state for
Prometheus.

## Publish state from a workload

State publishing is opt-in. Set a state directory before the workload calls
`fakegpu.init(...)` or starts through the native launcher. `build/` is ignored
by Git. With the [background machine](console.md) running, workloads publish
to it, and `fakegpu nvidia-smi` and `fakegpu nvitop` without a state source
show the machine, its idle GPUs included.

```bash
export FAKEGPU_SMI_STATE_DIR=build/smi

# Python FakeCUDA workload.
python3 your_script.py

# Unmodified native CUDA/NVML workload.
python3 -m fakegpu --build-dir build ./your_native_workload
```

A command started with [`fakegpu run`](commands.md#run-a-command-on-simulated-gpus)
needs no state directory: `nvidia-smi` on the command's `PATH` runs
`fakegpu nvidia-smi` for the run and lists its GPUs, even before a process
uses them.

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
  freshness, catalog and native API coverage, device identity, slot, modeled
  copies and link throughput, memory categories, allocator activity, dispatch
  tracking, and per-process peaks.
- `-L` prints nvidia-smi's `GPU 0: NAME (UUID: GPU-...)` lines, with MIG
  devices indented below their GPU. A host name follows the index only when
  several hosts are shown. Profile, compute capability and PCI bus ID are in
  `-q` and `--json`.
- `-i` selects devices by index, UUID, PCI bus ID, or profile ID. `--json`
  prints the full normalized inventory. FakeGPU writes state schema v2 and can
  still read v1.
- `--query-runtime` exposes the same runtime details as CSV or JSON.
  `--help-query-runtime` lists every field.
- CSV output separates fields with `, ` like `nvidia-smi` and quotes only
  cells that contain a comma, quote, or newline. `--query-gpu=count` repeats
  the number of GPUs on every row, as Accelerate expects, and still counts
  every GPU when `-i` selects some.
- `--query-gpu` and `--query-compute-apps` also accept every field and alias
  of nvidia-smi 590's `--help-query-*` lists, headed by nvidia-smi's column
  names. Fields FakeGPU does not model, such as `utilization.gpu` or
  `temperature.gpu`, print `[N/A]`.
- The table's first lines carry `Driver Version:` and `CUDA Version:` like
  nvidia-smi's banner. The driver version is a modeled `570.195.03`, the
  version FakeGPU's native NVML reports, and `--query-gpu=driver_version`
  prints the same value. Set `FAKEGPU_DRIVER_VERSION` before the workload
  starts to model another one.
- Native interception also publishes allocation lifetime, transfer volume,
  kernel launches, GEMM calls and FLOPs, compatibility events, and unsupported
  API counts. `FAKEGPU_SMI_DETAIL_LIMIT` and `FAKEGPU_SMI_MAX_STATE_BYTES` limit
  the retained detail and the state-file size.

## Model NVLink, MIG, and faults

These features are off by default, except that processes on the background
machine get its platform. Set them before the workload starts. Invalid input
activates nothing and is reported as a configuration error.

- **Platform and NVLink**: `FAKEGPU_PLATFORM` (or the background machine's
  platform) places the GPUs in HGX/DGX, superchip or PCIe slots and wires
  their NVLinks; see [Platforms and slots](platforms.md).
  `FAKEGPU_NVLINK_GROUPS="0,1;2,3"` instead gives each pair in a group one
  link at `FAKEGPU_NVLINK_BANDWIDTH_GBPS` (default `900`). This drives
  `topo -m` (NV#, PIX, PXB, PHB, NODE, SYS and NUMA affinity), `nvlink -s`
  (GB/s per link and direction), the `topology.*`, `nvlink.*`, `platform`,
  `slot.type`, `numa.node`, `pcie.link.*` and `host_link.bandwidth` fields,
  and native NVML link queries. Copies are recorded with their modeled time
  and link throughput (the `copy.*` and `*.throughput.*` fields); see
  [Copy time and throughput](platforms.md#copy-time-and-throughput).
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

## Use pip-installed nvitop on macOS or Linux

Install the optional monitoring dependencies from the checkout, then keep the
workload running in terminal 1:

```bash
conda activate fakegpu
python -m pip install '.[monitoring]'
FAKEGPU_SMI_STATE_DIR=/tmp/fakegpu-smi python - <<'PYTHON'
import time
import fakegpu
fakegpu.init(profile="l4", device_count=1)  # Synthetic profile.
import torch
x = torch.ones((2048, 2048), device="cuda:0")
time.sleep(600)
PYTHON
```

In terminal 2:

```bash
conda activate fakegpu
fakegpu nvitop --state-dir /tmp/fakegpu-smi
# Print once without entering the interactive monitor:
fakegpu nvitop --state-dir /tmp/fakegpu-smi -- --once
```

Press `q` to exit nvitop; stop the workload with Ctrl+C. The separate
`fakegpu nvidia-smi -l 1` loop exits with Ctrl+C, not `q`.

The wrapper runs the installed, unmodified nvitop. It explicitly loads bundled
`libnvidia-ml.dylib` on macOS (`libnvidia-ml.so.1` on Linux) and refreshes an
atomic native NVML snapshot every 0.25 seconds. Calling plain `nvitop` does not
enable this bridge. No NVIDIA driver, system NVML replacement, or shell alias
is installed. The workload still computes on the CPU with FakeCUDA.

Device names, synthetic UUIDs, total memory, reported memory, PIDs and per-process
memory come from the published state. Reported memory includes allocator
reservations and any configured runtime overhead, so freeing tensors may require
`torch.cuda.empty_cache()` before the displayed allocation decreases. Workers
with the same synthetic GPU UUID are aggregated. By default, temperature, power draw, GPU
utilization and process GPU utilization are unavailable (`N/A`). Modeled MIG and
NVLink views remain available through `fakegpu nvidia-smi`, not this bridge.

Only fresh, running publishers on this host are included; the default heartbeat
expiry is two seconds (`--stale-after-seconds`). Exited/dead processes and invalid
files are excluded. Duplicate files for one PID use the newest snapshot. Devices
already discovered keep stable NVML handles and show zero usage after their last
publisher exits. Restart nvitop after adding a new device/profile, because nvitop
selects its device list at startup. With no live publishers the wrapper prints
startup guidance and exits with status 1; it does not invent a device inventory.

Python NVML clients can use `fakegpu.nvml.NvmlStateBridge` as a context manager
in a dedicated monitor process, before loading NVML. Initialize/shutdown pynvml
normally inside that context. Closing the bridge stops refreshes and clears
usage; its library selection and handles remain process-local until exit.

The regression probe uses two real FakeCUDA worker processes and checks NVML
memory/process queries (including v1/v2/v3 buffer sizes), nvitop's API and
`--once` output, memory release, graceful exit, abrupt termination and empty-state
guidance. Run it outside the checkout after installing the wheel:

```bash
FAKEGPU_CHECKOUT="$PWD"
(cd /tmp && python -I "$FAKEGPU_CHECKOUT/tests/support/nvml_bridge_smoke.py" --tui)
```


## Eight A100s with reference telemetry and a real torch workload

From the checkout, run terminal 1:

```bash
conda activate fakegpu
python examples/monitoring/torch_eight_a100.py
```

In terminal 2:

```bash
conda activate fakegpu
fakegpu nvitop --state-dir /tmp/fakegpu-smi --telemetry reference
```

The program creates eight synthetic A100 SXM4 80GB devices and real CPU-backed
PyTorch tensors. Each cycle allocates 32–60 MiB per device, adds another 64 MiB
per device (96–124 MiB), releases the added tensors, then releases everything.
It calls `empty_cache()` at phase boundaries. The default phase lasts eight
seconds; Ctrl+C stops the loop. For one finite cycle use `--cycles 1`.
Actual host allocations stay around 1 GiB, not eight times 80 GiB.

`--telemetry reference` enables the opt-in A100 display model, and nvitop's
header then reads `Driver Version: FakeGPU (modeled)`. For another
workload, set `FAKEGPU_TELEMETRY=reference` before `fakegpu.init()` and initialize
with `profile="a100", device_count=8`. The observer records per-device CPU torch
operator duration and estimated tensor I/O bytes. The viewer exposes GPU and
memory-engine utilization, per-process activity, temperature, power, power cap,
P-state and clocks through NVML. Tensor residency alone does not imply activity.

The [reference configuration](../../fakegpu/data/a100_monitor_reference.json)
separates its sources and assumptions:

- [NVIDIA specifications](https://www.nvidia.com/en-us/data-center/a100/) supply
  the standard SXM power cap and HBM bandwidth.
- A [public user-reported snapshot in NVIDIA/nccl-tests](https://github.com/NVIDIA/nccl-tests/issues/228)
  supplies eight idle temperature/power observations (26–30°C, 56–60W, P0).
  Only those numbers are retained; no machine identity, UUID or serial is copied.
- Busy percentages use **CPU dispatch duty cycle as a simulation input**. Power
  is interpolated toward the specification cap; temperature follows an explicit
  thermal response model. Memory-engine utilization uses observed tensor I/O
  against reference bandwidth, not allocated-capacity percentage. Loaded curves
  are assumptions, not measured A100 behavior or a performance calibration.

The observed idle readings belong to the `a100` SXM4 80GB profile only. Every
other catalog profile follows a default curve with the same activity input: it
idles at 30°C and 1 W, power rises linearly to the profile's catalog power cap
(300 W when the catalog has none), temperature follows the same 45°C rise and
10 s thermal response, and loaded clocks come from the catalog (1500 MHz when
missing). These defaults are display assumptions, not measurements. Publishers
without activity observations, and profiles outside the catalog, retain `N/A`.
The GPU-local fan remains
`N/A`, matching the public snapshot; CUDA-driver and unsupported video/PCIe
telemetry also remain unavailable. The default `--telemetry none` retains the
original behavior. No NVIDIA CUDA driver is installed on macOS.

The maintained test `tests/support/a100_monitor_smoke.py` launches this torch
program and the real nvitop TUI, reconstructs terminal updates with pyte, checks
all eight memory values through growth and release, checks modeled sensors, and
verifies process cleanup. It prints only numeric summaries.
