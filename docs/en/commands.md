# Command reference

[Documentation index](README.md) · [简体中文](../zh-CN/commands.md) ·
[繁體中文](../zh-TW/commands.md)

Every command runs as `python3 -m fakegpu <command>`. Use
`python3 -m fakegpu --help` for the full list and
`python3 -m fakegpu <command> --help` for a command's options. Large reports
and plans can use `.json.gz` wherever a JSON path is accepted. Compressed
output is written atomically and reproducibly.

## Setup and runtime

| Command | Purpose |
|---|---|
| `fakegpu run` | Run a command unchanged on simulated GPUs and report whether it fit in memory ([details](#run-a-command-on-simulated-gpus)) |
| `fakegpu doctor` | Check the installation, native libraries, PyTorch, and profiles |
| `fakegpu demo` | Run a small CPU-backed, CUDA-visible training step |
| `fakegpu preflight` | Execute a workload to a target stage and classify fit or OOM |
| `fakegpu capabilities` | List or strictly audit native API classifications |
| `fakegpu validate` | Run a declarative JSON, TOML, or YAML validation matrix |

## Analysis and estimation

| Command | Purpose |
|---|---|
| `fakegpu analyze-repo` | Inventory repository entry points and GPU-only risks |
| `fakegpu analyze-kernel` | Inspect CUDA, PTX, and SASS resources and operations |
| `fakegpu estimate-llm` | Estimate decoder memory, communication, and FLOPs |
| `fakegpu estimate-diffusion` | Estimate diffusion text-encode, denoise, and VAE-decode memory phases |
| `fakegpu estimate-roofline` | Produce a profile-aware analytical latency interval |
| `fakegpu workspace-profiles` | Validate and inspect workspace estimation profiles |

## Serving

| Command | Purpose |
|---|---|
| `fakegpu plan-serving` | Plan homogeneous or mixed-request admission, chunked prefill, and grouped shared-prefix KV storage |
| `fakegpu benchmark-serving` | Measure scheduled mixed streaming requests and keep usage, latency, SSE events, and failures |

## Fine-tuning

| Command | Purpose |
|---|---|
| `fakegpu estimate-finetune` | Estimate single-GPU full/LoRA/QLoRA training steps, time, and VRAM from framework configs |
| `fakegpu probe-finetune` | Collect fixed-shape Hugging Face Trainer timing and memory on a physical CUDA GPU |
| `fakegpu profile-finetune-data` | Scan local JSONL data for token lengths, truncation, padding, and processed sequence counts |
| `fakegpu run-finetune` | Run native HF Trainer/PEFT, LLaMA-Factory or Axolotl training and record data, CUDA timing, memory, and final saving |
| `fakegpu validate-finetune-windows` | Validate dynamic window timing using independent calibration and holdout runs |
| `fakegpu validate-finetune-events` | Validate evaluation and checkpoint costs from independent native event records |
| `fakegpu plan-finetune-pipeline` | Plan future native epochs/updates from a calibrated sequential dataset |
| `fakegpu estimate-finetune-pipeline` | Estimate complete native execution time and a covered memory envelope |
| `fakegpu validate-finetune-pipeline` | Compare full-span time and memory predictions against independent native runs |
| `fakegpu plan-finetune-transfer` | Plan verified random or sequential native inputs |
| `fakegpu estimate-finetune-transfer` | Estimate explicit empirical shape-transfer budgets |
| `fakegpu validate-finetune-transfer` | Compare transfer budgets with independent native runs |
| `fakegpu calibrate-finetune` | Run bounded native calibration in a resumable workspace |
| `fakegpu budget-finetune-workspace` | Rebudget saved native evidence using only the CPU |

## Diffusion training

| Command | Purpose |
|---|---|
| `fakegpu plan-diffusion-training` | Inspect cached-feature UNet training plans and persistent storage |
| `fakegpu run-diffusion-training` | Run cached-feature UNet full/LoRA training on one real GPU |
| `fakegpu estimate-diffusion-training` | Budget compatible native cached-feature training observations |
| `fakegpu validate-diffusion-training` | Compare an independent native diffusion training holdout |

## Distributed and experiments

| Command | Purpose |
|---|---|
| `fakegpu plan-training` | Normalize distributed training configs and estimate rank memory |
| `fakegpu simulate-topology` | Model collective routes and link contention |
| `fakegpu replay-trace` | Summarize compute, communication, wait, and memory timelines |
| `fakegpu coordinator` | Manage the distributed simulation coordinator |
| `fakegpu bandwidth` | Validate simulated TCP payloads and report throughput |
| `fakegpu plan-experiments` | Plan a bounded parameter sweep and inspect exact commands |
| `fakegpu run-experiments` | Run or explicitly resume experiments while keeping every attempt |
| `fakegpu compare-experiments` | Verify raw artifacts, compare repetitions and optionally export a plot |

## Calibration and monitoring

| Command | Purpose |
|---|---|
| `fakegpu calibrate` | Build serving observations, compare memory reports, and enforce reliability gates |
| `fakegpu nvidia-smi` | Inspect devices, processes, modeled topology/MIG, health status, and reliability events |
| `fakegpu nvitop` | Run installed nvitop against live local FakeGPU state through bundled NVML (macOS/Linux) |
| `fakegpu console` | Open the terminal console for the background machine: GPUs, processes, events, settings ([guide](console.md)) |
| `fakegpu daemon` | Run a simulated GPU machine in the background that FakeCUDA processes join; start, stop, status, events, config |
| `fakegpu platform` | List host platforms and show the slots, PCIe links and NVLinks a device layout gets ([guide](platforms.md)) |
| `fakegpu metrics` | Export bounded Prometheus/JSON metrics and serve short in-memory history |

<a id="run-a-command-on-simulated-gpus"></a>

## Run a command on simulated GPUs

`fakegpu run` starts a command unchanged on simulated GPUs and prints a memory
summary to stderr when it ends:

```bash
python3 -m fakegpu run --gpu a100x8 -- python3 train.py
python3 -m fakegpu run --gpu h100:4,a100:4 -- \
  torchrun --nproc_per_node 8 train.py
python3 -m fakegpu run --gpu a100x4 --report-dir build/run -- \
  accelerate launch --multi_gpu --num_processes 4 train.py
```

The run puts a `sitecustomize` module first on `PYTHONPATH`, so every Python
process of the command starts FakeCUDA when it first imports torch, including
the workers that `torchrun`, `accelerate launch` and `mp.spawn` start. An
existing `sitecustomize` still runs. That Python must be able to import
FakeGPU; one that cannot prints a warning when it imports torch and runs
without simulated GPUs. Python started with `-I`, `-E` or `-S` ignores
`PYTHONPATH` and also runs without them. `CUDA_VISIBLE_DEVICES` limits a
process to some of the run's GPUs, as on a real machine, and the summary
counts its memory on those GPUs. `nvidia-smi` on the command's `PATH`, and
`nvitop` when it is installed, list the run's GPUs, even before a process
uses them.
Collectives between the processes carry real values; see
[Multi-process training on FakeCUDA](distributed.md#multi-process-training-on-fakecuda).

| Option | Meaning |
|---|---|
| `--gpu SPEC` | GPUs to simulate: `a100x8`, `8xa100`, `a100:8`, or a list such as `h100:4,a100:4`. IDs come from `fakegpu doctor --list-profiles`; an unknown ID exits with status 2 and suggests close matches |
| `--report-dir DIR` | Keep `run_report.json` and the per-process records in `DIR/records/`, replacing an earlier run's records. Without it the run's files go to a temporary directory that is removed afterwards |
| `--no-summary` | Do not print the summary |

Without `--gpu` the run takes its GPUs from the first of:

1. the background machine, when it is running (`fakegpu daemon start` or
   `fakegpu console`). The run joins the machine, so `fakegpu console` shows
   its processes. `FAKEGPU_MACHINE=off` skips this step;
2. `FAKEGPU_PROFILES`;
3. `FAKEGPU_PROFILE` and `FAKEGPU_DEVICE_COUNT`, which default to `a100` and
   8, so a run with nothing set gets eight A100 GPUs.

A run with its own GPUs stays out of the background machine. A script that
passes its own profile to `fakegpu.init()` keeps it; the summary then lists the
GPUs the script used and adds a `Note` line saying so.

### Exit status and Ctrl-C

`fakegpu run` exits with the command's status. A command ended by a signal
gives 128 plus the signal number, for example 130 after Ctrl-C. The run itself
exits with 127 when the command is not found, 126 when it is not executable,
and 2 for a bad `--gpu` or a missing command.

Ctrl-C reaches the command from the terminal. `fakegpu run` waits for the
command to stop and still prints the summary, which then covers the run up to
that point. SIGTERM and SIGHUP sent to `fakegpu run` are passed on to the
command, and so is SIGINT when the run is not in the terminal's foreground.

### Read the summary

For a script that allocates one 1 GiB tensor on `cuda:0`:

```text
── FakeGPU run summary ────────────────────────────────────────────────
 Command   python3 alloc.py
 Result    exit 0 after 1.6 s; 1 process used simulated GPUs
 GPUs      2 × a100 (NVIDIA A100-SXM4-80GB, 80.00 GiB each)  [catalog]

  GPU  Peak memory          Capacity     Headroom     Processes
    0     1.00 GiB  [C3]     80.00 GiB    79.00 GiB           1
    1            —           80.00 GiB    80.00 GiB           0

 Verdict   Fits. Largest peak: 1.00 GiB on GPU 0, 1.2% of its capacity.
 [C3]      Peak reserved memory: operator outputs tracked at PyTorch
           dispatch on the CPU; not calibrated against a real GPU (C4).
 [catalog] Capacity from the FakeGPU profile catalog.
 Excludes  the CUDA context, library workspaces and NCCL buffers a real
           GPU adds.
───────────────────────────────────────────────────────────────────────
```

- **Peak memory** is the peak reserved memory on each GPU. The peaks of the
  processes on one GPU are added, so the figure can overstate a GPU whose
  processes peaked at different times.
- The label after a peak is the memory tracking level, from `[C0]` (no usable
  tracking) through `[C2]` (tensor lifetimes) and `[C3]` (operator outputs or
  native allocations) to `[C4]` (calibrated against a real GPU). A GPU shared
  by several processes shows the weakest level among them. Each label that
  appears is explained below the table.
- `[catalog]` marks capacities taken from the FakeGPU profile catalog.
- `Excludes` names memory the run does not count: the CUDA context, library
  workspaces and NCCL buffers that a real GPU adds on top of the peak.
- The verdict is `Fits`, `Does not fit` (a simulated out-of-memory error ended
  the command, or a peak exceeded capacity), `Unresolved` (the command failed
  or was interrupted first; the peaks cover the run until then), or no GPU use
  (nothing was put on `cuda`, or FakeCUDA did not start). `run_report.json`
  stores these as `fits`, `does_not_fit`, `unresolved` and `no_gpu_use`. A
  command that catches a simulated out-of-memory error and continues still
  fits; the verdict says how many it caught.
- With more than 16 GPUs, only the GPUs in use are listed.

<a id="limits-of-fakegpu-run"></a>

### Limits of `fakegpu run`

- Only Python processes get simulated GPUs. Native CUDA programs still need
  the launcher, `python3 -m fakegpu --build-dir build <program>`; see
  [Native interception](native-interception.md).
- `nvidia-smi` inside the run reads published state and can lag the
  processes by about 250 ms.
- Each process rewrites its record every second and on SIGTERM or SIGHUP,
  so the summary still covers ranks that `torchrun` stops after another rank
  fails. A process that ends through `os._exit()` or SIGKILL can miss its
  last second.
- DDP gradient buckets are not counted.

<a id="torchrun-hangs-at-rendezvous"></a>

### If torchrun hangs at rendezvous

On a host whose hostname resolves through a proxy's fake-IP DNS, `torchrun`
can wait forever at rendezvous, with or without FakeGPU. Pass a loopback
address and a free port:

```bash
python3 -m fakegpu run --gpu a100x2 -- torchrun --nproc_per_node 2 \
  --master_addr 127.0.0.1 --master_port 29500 train.py
```
