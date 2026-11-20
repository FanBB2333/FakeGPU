# FakeGPU documentation

[Back to the project README](../../README.md) · [简体中文](../zh-CN/README.md) ·
[繁體中文](../zh-TW/README.md)

Start with the [project README](../../README.md) for installation and a quick
tour. The pages below go into detail.

## Using FakeGPU

- [Command reference](commands.md): every command, grouped by task, and
  `fakegpu run` for running a command unchanged on simulated GPUs
- [Memory analysis and estimation](estimation.md): `preflight`,
  `analyze-repo`, `analyze-kernel`, `estimate-llm`, and `estimate-diffusion`
- [Online LLM serving](serving.md): serving capacity, vLLM budgets,
  speculative decoding, mixed request lengths, and `benchmark-serving`
- [Fine-tuning time and VRAM](finetuning.md): config-based estimates, native
  runs, calibration, and validation for HF Trainer, LLaMA-Factory, and Axolotl
- [Diffusion training](diffusion-training.md): cached-feature UNet training
  plans and budgets
- [Distributed plans and experiment sweeps](distributed.md): plans,
  multi-process training on FakeCUDA, topology, traces, and sweeps
- [Background machine and console](console.md): `fakegpu console`, the
  background machine, events, and `fakegpu daemon`
- [Platforms and slots](platforms.md): HGX/DGX baseboards, Grace superchips,
  PCIe slots and NVLink bridges, and how they set PCIe links, NVLink and peer
  access
- [FakeGPU-SMI and monitoring](fakegpu-smi.md): `nvidia-smi` views, NVLink,
  MIG, and fault modeling, and Prometheus metrics
- [Native interception](native-interception.md): running unmodified programs
  and handling unsupported APIs
- [Framework compatibility](frameworks.md): TensorFlow, scikit-learn and CUDA C/C++ validation

## Accuracy

- [Calibrating against a real GPU](calibration.md): collecting peaks and
  gating estimates
- [Validation and reliability](validation.md): published evidence, reliability
  labels, and modeled comparisons

## Project

- [Development and testing](development.md): build, tests, GPU profiles, and
  project layout
- [Limitations](limitations.md)
