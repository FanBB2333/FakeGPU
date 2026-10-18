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
| `fakegpu daemon` | Run a simulated GPU machine in the background that FakeCUDA processes join; start, stop, status, events, config import/export ([guide](console.md)) |
| `fakegpu platform` | List host platforms, show device links, and explain copy paths, bottlenecks and modeled time ([guide](platforms.md)) |
| `fakegpu metrics` | Export bounded Prometheus/JSON metrics and serve short in-memory history |
