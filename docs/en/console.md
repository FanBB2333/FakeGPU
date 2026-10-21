# Background machine and console

[Documentation index](README.md) · [简体中文](../zh-CN/console.md) ·
[繁體中文](../zh-TW/console.md)

FakeGPU can run a simulated GPU machine in the background. FakeCUDA processes
started while it runs join it automatically. A terminal console shows the
machine's devices, processes and events, and changes its settings. The machine
keeps running after you leave the console. Install the `console` extra from the
checkout:

```bash
FAKEGPU_BUILD_DIR="$PWD/build" python3 -m pip install '.[torch,console]'
fakegpu console
```

`fakegpu console` starts the machine if it is not running. The first start uses
eight A100 SXM4 80GB devices on an HGX A100 8-GPU baseboard with the reference
telemetry model; set another layout with `--devices` (for example
`--devices h100:8` or `--devices rtx4090:2`), the host platform with
`--platform`, `--pcie-gen` and `--nvlink-bridges` (see
[Platforms and slots](platforms.md)), and `--telemetry none|reference`. These
options also update a running machine.

## Run workloads on the machine

While the machine runs, any process that calls `fakegpu.init()` publishes its
state to the machine and uses the machine's device layout:

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

[`fakegpu run`](commands.md#run-a-command-on-simulated-gpus) without `--gpu`
joins the machine the same way, so every process of the command, torchrun
workers included, shows up in the console:

```bash
fakegpu run -- torchrun --nproc_per_node 8 train.py
```

A process keeps its own layout when it passes `profile`, `devices` or
`device_count` to `fakegpu.init()`, or sets `FAKEGPU_PROFILE`,
`FAKEGPU_PROFILES` or `FAKEGPU_DEVICE_COUNT`. Set `FAKEGPU_MACHINE=off` to keep
a process out of the machine. A new layout applies to processes started after
the change; running processes keep the devices they started with.

While the daemon runs, `fakegpu nvitop` and `fakegpu nvidia-smi` read the
machine without options. Like on a real host they list every GPU of the
layout, idle ones included, and the processes on the machine. With reference
telemetry, nvitop's header reads `Driver Version: FakeGPU (modeled)`.

After changing the device layout or sensor mode, quit and reopen `fakegpu
nvitop`: its device list and sensor mode are selected at startup. The console
updates its device list immediately. Selecting the machine's state directory
explicitly also inherits its sensor mode unless `--telemetry` overrides it.

## The console

| Tab | Shows |
|---|---|
| `1` Overview | Charts that combine all GPUs (memory, utilization, power, temperature, link I/O); a card per GPU (memory, utilization, power, temperature, clocks, P-state, recent memory and utilization) or, after `v`, one line per GPU; the process table and the latest events. Click a card, or press Enter on a line, to open that GPU on the Devices tab |
| `2` Processes | Every attached process with command line, user, stage, memory and activity per device; recently ended processes |
| `3` Devices | The GPUs of the layout, which you can change one by one (see below), and per-GPU detail: slot, modeled copies and link throughput, allocator counters, memory by category, largest allocations, and charts of the last minute (memory, utilization, temperature, power, PCIe/C2C/NVLink throughput) |
| `4` Events | The full event log with text and level filters |
| `5` Machine | Settings form (layout, platform, PCIe slot generation, NVLink bridges, telemetry, background, stale and memory warning thresholds), the slot map and GPU link matrix, daemon and path information |
| `6` Profiles | The GPU profile catalog; `a` or Enter adds the selected profile to the layout form |

The console uses a muted, low-saturation (Morandi) palette, and every reading
keeps one color in the cards, charts and tables: memory mauve, utilization
sage, power clay, temperature sand, and link traffic teal (PCIe) or dusty blue
(NVLink). Rose values and mustard or rose card borders mark warnings.

Chart titles name where the readings come from: `memory tracked, sensors
modeled`, or `memory tracked, sensors N/A` with telemetry off. Memory is
tracked from the processes' allocations; utilization, temperature and power
are modeled from CPU activity, not measured.

| Key | Action |
|---|---|
| `1` to `6` | Switch tabs |
| `v` | Overview: switch between GPU cards and one line per GPU |
| `k` / `K` / `i` | Send SIGTERM / SIGKILL / SIGINT to the selected process (asks first) |
| `b` | Turn background mode on or off |
| `q` | Leave the console |
| `Q` | Stop the machine (asks first) |
| `s` | Start the machine again after it stopped |

With background mode on (the default), `q` leaves the machine running; reopen
it with `fakegpu console`. With background mode off, leaving the console also
stops the machine. Stopping the machine does not stop running workloads.

### Change the GPUs on the Devices tab

| Key | Action |
|---|---|
| `e`, Enter, or a click on the selected row | Replace the selected GPU with another profile |
| `E` | Replace every GPU: pick one profile and a count |
| `a` | Add GPUs after the last one, starting from the selected GPU's profile |
| `x` / Delete | Remove the selected GPU (asks first) |

The buttons above the table do the same. The profile picker filters the
catalog as you type and previews the new layout, its total memory and its
platform before you apply it. When the configured platform cannot hold the new
layout (an HGX board holds one GPU model, for example), the console sets the
platform to `auto`, or to `generic` when no template fits, and drops NVLink
bridges that no longer apply; the preview lists these changes. The Machine tab
edits the same layout as text. New processes use the new layout; running
processes keep their devices, which the Devices tab marks with `*` until those
processes exit.

## Events

The machine records what happens to its processes and devices:

| Event | Meaning |
|---|---|
| `process.attached` | A FakeCUDA process joined the machine |
| `process.exited` | The process published its final state and exited |
| `process.terminated` | The process ended on SIGTERM |
| `process.vanished` | The process ended without a final heartbeat, for example after SIGKILL or a crash |
| `process.stalled` / `process.resumed` | A live process stopped or restarted publishing state |
| `process.signal_sent` | The console or another client sent a signal |
| `memory.high` / `memory.normal` / `memory.oversubscribed` | Device memory crossed the warning threshold or exceeded capacity |
| `config.changed` | The layout or a setting changed |
| `state.invalid` | A state file could not be read |
| `daemon.started` / `daemon.stopped` | The machine started or stopped |

A process killed from nvitop or with `kill` therefore shows up as
`process.terminated` (SIGTERM) or `process.vanished` (SIGKILL). The last
1,000 events survive a machine restart.

## Manage the machine from the shell

```bash
fakegpu daemon start [--devices a100:8] [--telemetry reference] [--background on]
fakegpu daemon status [--json]
fakegpu daemon events [-n 50] [-f] [--json]
fakegpu daemon config [--devices a100:4] [--platform auto] [--pcie-gen 0] [--nvlink-bridges 0-1] [--json]
fakegpu daemon stop
```

With reference telemetry, `fakegpu daemon status` adds a note under the GPU
table that Util, Temp and Power are modeled from CPU activity, not measured.
`fakegpu daemon run` runs the machine in the foreground. Machine files live in
`~/.fakegpu/machine` (`config.json`, `events.jsonl`, `daemon.log` and the
`state/` directory); set `FAKEGPU_HOME` to move them.

## Control protocol

Clients talk to the machine over a Unix socket (`daemon.sock` in the machine
directory) with one JSON request and one JSON response per connection, each
ending in a newline. The protocol is `fakegpu.machine.v1`:

```text
request   {"v": 1, "id": 7, "method": "status", "params": {}}
response  {"v": 1, "id": 7, "ok": true, "result": {...}}
```

Methods are `hello`, `status`, `events`, `config.get`, `config.update`,
`process.signal` and `daemon.stop`; the
[service module](../../fakegpu/machine/service.py) documents their parameters.
`fakegpu.machine.client.MachineClient` is a ready-made Python client. All
requests go through `MachineService.dispatch`, so another transport, such as
HTTP for a web management page, can reuse it unchanged.
