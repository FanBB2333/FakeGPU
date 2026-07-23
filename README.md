# FakeGPU

FakeGPU is a CUDA, cuBLAS, NVML, and NCCL interception toolkit for validating GPU-facing code without depending on a production GPU cluster. It provides a Python fake-CUDA path, native shared-library interception, configurable GPU profiles, distributed control-flow simulation, and memory preflight reports.

[Getting started](docs/getting-started.md) · [Complete feature list](#complete-feature-list) · [Quick reference](docs/quick-reference.md) · [中文文档](docs/index.zh.md) · [Changelog](CHANGELOG.md)

## TL;DR

**FakeGPU lets PyTorch/CUDA programs see simulated NVIDIA GPUs.** It runs common test flows on CPU, switches GPU profiles, checks likely out-of-memory failures, and estimates training memory before you use a real GPU.

It checks code paths and memory plans; it does not predict GPU kernel speed.
Its TCP benchmark measures the simulator path, not production NCCL/RDMA.

```bash
fakegpu doctor --list-profiles       # inspect the installation and GPU catalog
fakegpu demo --profile l4            # run a tiny CUDA-visible training step on CPU
fakegpu estimate-llm --model-dir /models/qwen --prompt-tokens 128  # inspect VRAM/FLOPs without loading weights
fakegpu bandwidth --listen 127.0.0.1:29591 --nodes 2  # simulate two TCP nodes
```

![Four FakeGPU workflows: simulated PyTorch training, GPU profile switching, preflight OOM checks, and static VRAM estimation](docs/assets/readme/tldr-workflows.png)

_Real command output from the maintained v1.5.4 workflows._

> FakeGPU is a development and validation tool. It does not provide numerical or performance parity for arbitrary CUDA kernels.

## Choose a path

| Goal | GPU required | Recommended entry point |
|---|---:|---|
| Check the installation and list GPU profiles | No | `fakegpu doctor --list-profiles` |
| Run the smallest end-to-end PyTorch example | No | `fakegpu demo --profile a100` |
| Exercise PyTorch CUDA-style code on CPU | No | `fakegpu.init(runtime="fakecuda")` |
| Intercept CUDA, NVML, cuBLAS, or NCCL shared-library calls | No | `./fgpu --mode simulate ...` |
| Check whether a workload reaches a stage or exceeds a GPU profile | No | `python3 -m fakegpu preflight ...` |
| Estimate training memory from an ATen graph | No | `./ftest static_memory_validation` |
| Estimate dense-decoder inference memory/FLOPs from safetensors metadata | No | `fakegpu estimate-llm ...` |
| Display memory from running FakeCUDA processes | No | `fakegpu nvidia-smi --state-dir ...` |
| Compare an unmodified real-GPU baseline | Yes | `./fgpu --mode passthrough ...` |
| Keep real CUDA compute while virtualizing selected surfaces | Yes | `./fgpu --mode hybrid --oom-policy clamp ...` |
| Simulate multi-rank collective control flow | No | `FAKEGPU_DIST_MODE=simulate` |
| Validate deterministic multi-worker DataLoader reconstruction | No | `./ftest dataloader_replay` |
| Simulate logical machines over TCP and measure throughput | No | `fakegpu bandwidth --listen 127.0.0.1:29591 --nodes 2` |

## Quick start

### Check the installation and run the demo

From a checkout or installed package with PyTorch available:

```bash
python3 -m fakegpu doctor --list-profiles
python3 -m fakegpu demo --profile l4
```

`doctor` checks the profile catalog, native libraries, and PyTorch environment.
`demo` performs a small forward, backward, and optimizer step through the
CPU-backed fake-CUDA runtime. Neither command needs a physical GPU.

### Build the native libraries

Requirements: Linux or macOS, Python 3.10+, CMake 3.14+, and a C++17 compiler.

```bash
git clone https://github.com/FanBB2333/FakeGPU.git
cd FakeGPU

cmake -S . -B build
cmake --build build
./ftest smoke
```

The default build executes maintained cuBLAS/cuBLASLt operations on CPU. Generic CUDA kernel launches remain no-ops in native simulate mode.

### Run PyTorch code with fake CUDA semantics

From the repository root, with PyTorch installed:

```python
import fakegpu

fakegpu.init(runtime="fakecuda", profile="a100", device_count=2)

import torch

device = torch.device("cuda:0")
model = torch.nn.Linear(8, 4).to(device)
x = torch.randn(2, 8, device=device)
loss = model(x).square().mean()
loss.backward()

print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(loss.item())
```

This route uses FakeCudaTensor semantics and CPU execution. It is suitable for training-flow, device-placement, error-handling, and framework compatibility checks. Custom CUDA extensions and arbitrary CUDA kernels are outside its maintained scope.

### Run a native intercepted command

`./fgpu` selects the built libraries and sets the preload environment:

```bash
./fgpu --profile a100 --device-count 2 nvidia-smi  # when nvidia-smi is installed
./fgpu --mode simulate python3 your_script.py
./fgpu --devices "a100:2,h100:2" python3 your_script.py
```

The same native route can be enabled before importing CUDA-using libraries:

```python
import fakegpu

fakegpu.init(runtime="native", profile="a100", device_count=2)

import torch
```

## AI workload preflight

Use preflight to check whether a command reaches a target stage and whether tracked memory fits a selected profile:

```bash
python3 -m fakegpu preflight \
  --runtime fakecuda \
  --profile a100 \
  --device-count 1 \
  --stage forward \
  --report-dir preflight-report \
  --allocation-stacks \
  --strict \
  -- python3 train.py --small-config
```

The command writes:

- `preflight_report.json`
- `preflight_report.md`
- `preflight_stdout.log`
- `preflight_stderr.log`

Preflight tracks tensor lifetimes, stage peaks, parameters, buffers, gradients, optimizer state, activations, shared-storage aliases, and saved tensors visible through PyTorch hooks. A matching empirical calibration report can replace the raw fake-CUDA estimate with an observed physical-memory upper bound.

See [AI Researcher Preflight](docs/ai-researcher-preflight.md) for calibration inputs, confidence levels, and current limits.

## Static memory estimation

The static estimator captures a target-device ATen forward/backward graph without allocating real CUDA memory. It models:

- unique tensor storages and aliases
- storage lifetime through final graph use
- parameters, buffers, gradients, and optimizer state
- separate graph and optimizer phases
- matched CUDA Attention workspace profiles
- operator-local workspace at the corresponding graph-liveness point

Run the maintained workload grid:

```bash
./ftest static_memory_validation
```

Or call the API directly:

```python
from fakegpu import estimate_module_memory

report = estimate_module_memory(
    model,
    (example_input,),
    mode="training",
    optimizer="adamw",
    target_device="auto",
)
print(report["estimated_peak_bytes"])
```

CUDA-specific ATen traces require a CUDA-enabled PyTorch build, but the trace itself does not allocate physical GPU memory.

Backend context memory, allocator fragmentation, unmatched workspaces, custom CUDA operators, and fused/foreach optimizer temporaries require device- and software-specific calibration.

## LLM inference without a GPU

For a local dense decoder checkpoint, FakeGPU can inspect safetensors headers
without materializing the weights or creating a CUDA context:

```bash
python3 -m fakegpu estimate-llm \
  --model-dir /models/Qwen/Qwen3-8B \
  --batch-size 1 \
  --prompt-tokens 9 \
  --generated-tokens 2 \
  --dtype bfloat16 \
  --attention-implementation sdpa \
  --json build/qwen-estimate.json
```

The report derives parameter storage from checkpoint metadata, KV-cache size
from layer/KV-head dimensions, transient tensor storage from the selected
attention path, and matrix FLOPs from model shapes. It currently targets dense
decoder-only safetensors checkpoints; MoE, quantized weights, adapters, custom
CUDA operators, and backend-specific workspaces need a dedicated model or
empirical calibration.

A FakeCUDA process can also publish its live tracked memory for a second
terminal:

```bash
FAKEGPU_SMI_STATE_DIR=/tmp/fakegpu-smi python3 your_inference.py
python3 -m fakegpu nvidia-smi --state-dir /tmp/fakegpu-smi
```

Set `FAKEGPU_SMI_RUNTIME_OVERHEAD_BYTES` only when the same GPU/software path
has been calibrated against NVML. See
[LLM Inference Estimation](docs/llm-inference-estimation.md) for a complete
real-CUDA/FakeCUDA comparison and the current accuracy boundary.

For training, `verification/qwen_sft_memory_worker.py` can run the same
full-parameter, LoRA, or packed-NF4 QLoRA reference step on real CUDA,
CPU-backed FakeCUDA, or a static ATen graph. The maintained 0.8B and 2B BF16
matrix covers gradient checkpointing and accumulation while distinguishing
first-step from steady-state AdamW memory. The QLoRA reference supports direct
FP32 block scales and optional bitsandbytes-style nested 8-bit scale storage:

```bash
python3 verification/qwen_sft_memory_worker.py \
  --mode static --model-dir /models/Qwen3.5-0.8B \
  --training-method qlora --quantization-double-quantization \
  --output build/qwen-sft-qlora-static.json
```

For full-parameter FSDP, a second controller projects the same static ATen
graph onto per-unit FULL_SHARD storage and checks it against a real two-rank
optimizer step:

```bash
python3 verification/qwen_sft_memory_worker.py \
  --mode static --model-dir /models/Qwen3.5-0.8B \
  --output build/qwen-sft-static.json

python3 verification/run_qwen_fsdp_sft_memory.py \
  --model-dir /models/Qwen3.5-0.8B \
  --static-report build/qwen-sft-static.json \
  --output-dir build/qwen-fsdp-sft-memory
```

The maintained Ampere and Blackwell sequence-16/128 experiments keep graph
and overall peak errors below 0.76% while recording all-gather and
reduce-scatter traffic.

LoRA uses the FSDP2 controller because its frozen BF16 base and trainable FP32
adapters cannot be represented by one FSDP1 flat-parameter dtype:

```bash
python3 verification/qwen_sft_memory_worker.py \
  --mode static --model-dir /models/Qwen3.5-0.8B \
  --training-method lora --lora-rank 8 --sequence-length 16 \
  --output build/qwen-fsdp2-lora-static.json

python3 verification/run_qwen_fsdp2_lora_sft_memory.py \
  --model-dir /models/Qwen3.5-0.8B \
  --static-report build/qwen-fsdp2-lora-static.json \
  --sequence-length 16 --world-size 4 \
  --output-dir build/qwen-fsdp2-lora
```

This path models each DTensor shard and all-gather/reduce-scatter buffer
lifetime separately. Qwen3.5-0.8B sequence-16/64/128 runs with two or four
ranks on both the RTX PRO 5000 and RTX 3090 Ti keep every phase within 1.98%
while exercising mixed-dtype `uint8` all-gathers. Existing dimensions such as
sequence length, LoRA rank, dtype, and world size 2/4 are command parameters;
new tests along those dimensions do not require source changes.

The native NF4 path needs no external quantization package, but it does not
claim bitsandbytes fused-kernel equivalence; see
[LLM SFT Memory Estimation](docs/llm-sft-memory-estimation.md).

## Runtime model

FakeGPU separates the Python runtime path, native compute mode, and distributed mode.

### Python runtime

| Runtime | Behavior |
|---|---|
| `fakecuda` | Patches PyTorch with FakeCudaTensor behavior and executes maintained operations on CPU |
| `native` | Loads FakeGPU shared libraries into the current process |
| `auto` | Selects `fakecuda` when available, otherwise uses `native` |

### Native compute mode

| `FAKEGPU_MODE` | Behavior | Real GPU |
|---|---|---:|
| `simulate` | Virtual device identity and memory; maintained cuBLAS/cuBLASLt paths can execute on CPU | No |
| `passthrough` | Unmodified real-CUDA baseline with no FakeGPU CUDA/NVML injection | Yes |
| `hybrid` | Real CUDA compute with selected Driver/NVML virtualization and OOM policy handling | Yes |

Hybrid OOM policies:

```text
clamp | managed | mapped_host | spill_cpu
```

`clamp` is the maintained validation path. Oversubscription policies are experimental and do not guarantee numerical parity.

### Distributed mode

| `FAKEGPU_DIST_MODE` | Behavior |
|---|---|
| `disabled` | No FakeGPU distributed layer |
| `simulate` | Coordinator-managed collective and point-to-point semantics |
| `proxy` | Real NCCL data movement with FakeGPU control-plane reporting |
| `passthrough` | Thin forwarding to real NCCL |

Compute and distributed modes can be combined. The recommended CPU-only mode pair is:

```bash
FAKEGPU_MODE=simulate
FAKEGPU_DIST_MODE=simulate
```

The simulated distributed path also needs a coordinator endpoint and cluster configuration. For complete coordinator and `torchrun` examples, see [Distributed Simulation Usage](docs/distributed-sim-usage.md).

For a self-contained two-node loopback check on a chosen port:

```bash
fakegpu bandwidth \
  --listen 127.0.0.1:29591 \
  --nodes 2 \
  --ranks-per-node 1 \
  --size 4MiB \
  --iterations 10
```

This starts the coordinator, generates a two-node topology, moves collective
payloads through TCP, checks the all-reduce result, reports measured
end-to-end throughput, and shuts the coordinator down. A separately hosted
coordinator and per-host rank selection are available for physical multi-host
checks.

## Complete feature list

FakeGPU has three complementary paths. Choosing the right path is important
because they answer different questions:

| Path | Main question | What actually runs |
|---|---|---|
| Python FakeCUDA | Will this Python/PyTorch workload follow the expected CUDA-facing control flow? | Maintained tensor, module, autograd, optimizer, and framework operations execute on CPU while reporting `cuda` devices |
| Native interception | Can an unmodified process discover and call the expected CUDA-family libraries? | Fake shared libraries expose selected CUDA Driver, CUDA Runtime, NVML, cuBLAS/cuBLASLt, and NCCL entry points; maintained matrix operations can execute on CPU |
| Analysis and reporting | How much memory, matrix work, or communication should a workload require? | Runtime traces, ATen graphs, safetensors metadata, calibrated measurements, and distributed events are converted into structured reports |

Coverage labels used below:

- **Maintained** — included in the repository's regular regression surface.
- **Validated** — has dedicated numerical or physical-host experiments; the
  result applies to the documented model, shape, GPU, and software envelope.
- **Compatibility-tested** — an optional framework integration has focused
  smoke or workflow tests, not complete upstream API coverage.
- **Experimental** — usable for its documented prototype scope, but not a
  general compatibility promise.

### Commands and public entry points

| Entry point | Function | Coverage |
|---|---|---|
| `fakegpu doctor` | Checks profiles, native libraries, and PyTorch; can inspect one profile or list the complete catalog in text/JSON | Maintained |
| `fakegpu demo` | Runs a small CUDA-visible PyTorch forward, backward, and optimizer step on CPU | Maintained |
| `./fgpu ...` / `python3 -m fakegpu ...` | Launches an arbitrary command with mode, OOM policy, distributed mode, coordinator, and uniform or heterogeneous device settings | Maintained |
| `fakegpu preflight -- ...` | Runs an arbitrary command to a requested stage and produces fit/OOM, memory-category, confidence, stdout, and stderr reports | Maintained; accuracy is execution-path and calibration dependent |
| `fakegpu estimate-llm` | Estimates dense decoder checkpoint storage, KV cache, transient tensors, process memory, and matrix FLOPs without loading weights | Maintained for dense decoder-only safetensors checkpoints |
| `fakegpu nvidia-smi` | Displays current and peak tracked memory published by one or more FakeCUDA processes | Maintained; this is FakeGPU state rather than host-driver telemetry |
| `fakegpu coordinator` | Starts, probes, stops, and reports on the TCP distributed coordinator | Maintained |
| `fakegpu bandwidth` | Creates logical nodes or connects physical hosts, checks payload correctness, and measures simulator-path TCP throughput | Maintained and physically validated; not an NCCL/RDMA speed predictor |
| `fakegpu.init`, `patch_torch`, `env`, `run`, `library_dir` | Selects a runtime or embeds launch/preload behavior in Python | Maintained public Python API |
| `estimate_module_memory`, `analyze_graph_memory`, `estimate_decoder_inference`, `inspect_safetensors_checkpoint` | Exposes graph and checkpoint estimators as Python APIs | Maintained public Python API |
| `fakegpu.stage(...)`, `MatmulFlopCounterMode` | Annotates preflight stages and counts matrix-heavy FLOPs, including grouped-query SDPA | Maintained helper APIs |

### Runtime and native-library features

| Feature | What is covered | Status and boundary |
|---|---|---|
| Python FakeCUDA runtime | `torch.cuda` discovery, CUDA-looking tensors and modules, logical devices, CPU-backed execution, and runtime memory tracking | Maintained; it does not emulate binary CUDA extensions |
| Native simulate mode | Virtual NVIDIA device identity, host-backed allocations, selected native API behavior, and CPU-backed maintained GEMM/matmul paths | Maintained; arbitrary CUDA kernel launches remain no-ops |
| Passthrough mode | Runs an unmodified real-CUDA baseline without FakeGPU CUDA/NVML injection | Maintained comparison path; requires a real GPU |
| Hybrid mode | Keeps real CUDA compute while virtualizing selected Driver/NVML surfaces and recording reports | `clamp` is validated; `managed`, `mapped_host`, and `spill_cpu` are experimental |
| Distributed mode selection | `disabled`, coordinator-backed `simulate`, real-NCCL `proxy` with reporting, and thin `passthrough` | Simulate is maintained; proxy/passthrough require a matching real stack |
| CUDA Driver and Runtime | Device count/properties, current-device state, allocation/free, memory information and types, memset/copies including peer copies, streams/events, and selected Driver forwarding | Maintained subset; the full CUDA API is not implemented |
| NVML | Device identity, memory information, and common monitoring queries used by tools and framework probes | Maintained subset; unavailable telemetry is synthetic or explicitly unsupported |
| cuBLAS and cuBLASLt | Selected GEMM and matmul calls, typed operation accounting, FLOP reports, and CPU numerical execution | Maintained subset; unsupported algorithms or attributes may be stubs |
| Library preload | Linux `LD_PRELOAD`, macOS `DYLD_INSERT_LIBRARIES`, explicit build/library directories, and mode-specific library selection | Maintained; macOS SIP can strip `DYLD_*` from system binaries |
| PrivateUse1 `fgpu` device | `torch.device("fgpu")`, tensor/module conversion, CPU-backed forward/backward, one-step Adam, alias-preserving save/load, and disabled autocast | Experimental local-debug prototype; it is separate from CUDA semantics and distributed support |

### PyTorch and training-framework features

| Feature | What is covered | Status and boundary |
|---|---|---|
| Core PyTorch | Common tensor creation/manipulation, modules, device propagation, autograd, gradients, optimizers, memory APIs, pinned-memory DataLoader paths, and multithreaded use | Maintained regression surface; unsupported operators can still expose CPU/FakeTensor gaps |
| CUDA-facing training flow | Forward, loss, backward, gradient accumulation, optimizer updates, gradient checkpointing, autocast, GradScaler-facing behavior, and checkpoint round trips | Maintained for tested operations and dtypes |
| Multi-device correctness guards | Logical `cuda:N` placement, per-device profiles and memory, invalid ordinals, and cross-device operation checks | Maintained |
| `torch.compile` | Eager-backend compilation smoke for functions, modules, autograd, dynamic shapes, and fake-CUDA inputs | Compatibility-tested; arbitrary Inductor/Triton CUDA code generation is not claimed |
| Transformers and Accelerate | Model loading, Trainer train/eval/predict/checkpoint flows, CUDA selection, mixed precision, and device-map helpers | Compatibility-tested with focused tiny-model and local-model workflows |
| PEFT and TRL | LoRA, SFTTrainer, DPOTrainer, adapter save/load, and LLaMA-Factory-style LoRA SFT/DPO flows | Compatibility-tested; not every trainer option or quantization backend is covered |
| torchtune, Lightning Fabric, and LitGPT | Single-device fine-tuning/training-loop and import-level compatibility checks | Compatibility-tested optional integrations |
| nanoGPT and MoE components | Model construction, routing-related components, forward/loss, and training-wrapper helpers | Compatibility-tested components; not arbitrary production MoE kernels |
| DDP | Simulated control flow plus Hybrid numerical validation for gradient averaging, accumulation/`no_sync`, unused parameters, static graph, bucket views, checkpointing, and failure restart cases | Maintained/validated within the documented rank and option matrix |
| FSDP and FSDP2 | Basic fake-CUDA smoke, state-dict round trips, Hybrid sharding numerics, mixed-precision parameter/reduction cases, and Qwen memory projection | Validated on the documented two- and four-rank cases |
| DeepSpeed | ZeRO 1/2/3, CPU optimizer/parameter offload, checkpoint resume, Hugging Face Trainer, Qwen LoRA SFT, two-stage Pipeline Parallel, AutoTP, AutoEP, and variable all-to-all workflows | Targeted Hybrid validation; fused/JIT optimizers, NVMe offload, sequence parallelism, and combined AutoTP+AutoEP remain outside the validated set |
| Local LLM execution | Dense Hugging Face models can load and run tokenization/generation or training reference steps through CPU-backed FakeCUDA when all exercised operators are covered | Compatibility-tested; output comes from real CPU math, so it can be much slower than CUDA |

### Memory and compute estimation

| Feature | What is measured or modeled | Status and boundary |
|---|---|---|
| Runtime allocation tracking | Current/peak bytes per logical device, allocation/free events, storage aliases, coarse categories, stages, top allocations, and optional Python stack traces | Maintained; allocations hidden inside unobserved backend code need calibration |
| Profile-aware OOM simulation | Enforces each selected GPU profile's memory limit and raises `torch.cuda.OutOfMemoryError` | Maintained; failure timing can differ from a real allocator |
| Repository/workload preflight | Wraps any command, checks arrival at `import`, `model_load`, `forward`, `backward`, `optimizer_step`, or `n_steps`, and reports fit, headroom, failure class, and confidence | Maintained; only the code path and shapes actually executed are assessed |
| Static ATen graph estimator | Captures forward/backward without real CUDA allocation; deduplicates storage aliases, follows last-use lifetimes, and models parameters, buffers, gradients, optimizer state, and separate graph/optimizer phases | Maintained; target-device ATen traces need a CUDA-enabled PyTorch build for CUDA-specific dispatch |
| Workspace modeling | Adds graph-persistent or operator-local workspace at the correct liveness point, including maintained Flash/Efficient Attention profiles | Validated for the profiled operator, dtype, shape, PyTorch, CUDA, and GPU combinations |
| Real-GPU calibration | Compares real CUDA, passthrough, Hybrid clamp, FakeCUDA, allocator counters, and NVML; aggregates multiple GPUs into signature-matched calibration bundles | Validated on RTX 3090 Ti and RTX PRO 5000 datasets; calibration is not extrapolated across changed workload signatures |
| Dense-decoder inference estimator | Reads safetensors headers without materializing weights and derives parameter bytes, KV cache, eager/SDPA transients, runtime overhead, and prefill/decode matrix FLOPs | Maintained for dense decoder-only models; MoE, adapters, quantized checkpoints, custom kernels, and backend workspaces need dedicated models |
| SFT memory references | Full-parameter, LoRA, and native packed-NF4 QLoRA; BF16, checkpointing, accumulation, first/steady AdamW steps, and direct or nested scale storage | Validated for maintained Qwen3.5-0.8B/2B matrices; native NF4 does not claim bitsandbytes fused-kernel parity |
| Sharded training projection | Projects a static graph onto FSDP FULL_SHARD or FSDP2 DTensor parameter/gradient/optimizer storage and all-gather/reduce-scatter buffer lifetimes | Validated for documented Qwen full-parameter and mixed-dtype LoRA cases at world sizes 2/4 |
| FLOP accounting | Native maintained GEMM/cuBLASLt FLOPs, checkpoint-shape inference FLOPs, and PyTorch matrix-heavy operation counts including grouped-query attention | Maintained for recognized matrix operations; it is not a kernel latency model |
| Virtual `nvidia-smi` | Publishes per-process current/peak tracked memory, device/profile identity, tracking confidence, and optional calibrated runtime overhead | Maintained; similarity to physical NVML depends on matching calibration |

### Distributed communication and topology

| Feature | What is covered | Status and boundary |
|---|---|---|
| Logical cluster model | YAML nodes, ranks, per-rank GPU profiles, links, bandwidth/latency, heterogeneous devices, sub-communicator membership, and preserved global-rank identity | Maintained |
| Coordinator transports | Unix sockets for local runs and chosen-port TCP for logical or physical multi-host runs | Maintained; TCP payloads have been validated across two physical hosts |
| Collectives | All-Reduce, Reduce, Broadcast, All-Gather, Reduce-Scatter, All-to-All, grouped operations, chunked payloads, and zero-element cases | Maintained direct/native suite |
| Variable all-to-all | Nonuniform and sparse all-to-all-v through grouped send/receive | Validated over Unix, loopback TCP, and physical two-host TCP |
| Reductions and dtypes | `sum`, `prod`, `min`, `max`, `avg`, and pre-multiplied sum across maintained integer, FP16, BF16, FP32, and FP64 payload paths | Maintained direct/native suite; support varies by operation where NCCL semantics require it |
| Point-to-point and groups | Send/receive, grouped collective/P2P submission, subgroup isolation, communicator split/shrink, zero-byte payloads, and mismatch detection | Maintained subset of NCCL host semantics |
| Socket staging | Copies submitted payloads through coordinator TCP sockets and falls back to socket staging when shared memory is unavailable | Maintained and physically validated |
| Topology/time model | Accounts for modeled link bandwidth, latency, routes, contention, operation duration, and bounded coordinator-observed timelines | Maintained analytical model; it does not reproduce NCCL protocols, NVLink, or RDMA behavior |
| Communication reports | Per-collective and P2P calls/bytes, complete node-pair matrix including zero-traffic pairs, directional/per-operation peaks, links, ranks, communicators, timeline, failures, and recovery | Maintained `cluster_report.v1` JSON plus Markdown table |
| Bandwidth experiment | Validates reduction payloads and reports per-rank timings plus algorithmic and socket-payload throughput | Maintained simulator benchmark; values include coordinator, copies, and scheduling overhead |
| PyTorch distributed flow | ProcessGroupNCCL-facing symbols required by tested PyTorch stacks, `torchrun`, DDP, FSDP/FSDP2, and DeepSpeed workflows | Maintained/validated subset; not complete NCCL 2.29 protocol or device-API parity |
| Physical-host controller | Checks identical Git revisions, launches selected two-host cases over SSH, applies rank placement, collects artifacts, and validates report consistency | Maintained validation tooling for the documented Linux/WSL hosts |

### Error injection, resilience, and recovery

| Feature | Simulated behavior | Status and boundary |
|---|---|---|
| Cross-device errors (E1) | Rejects arithmetic, matmul, concatenation, linear, and module calls that mix logical CUDA devices | Maintained Python check |
| Out of memory (E2) | Fails allocations that exceed the selected profile and exposes PyTorch CUDA OOM errors | Maintained Python/native checks |
| Invalid device (E3) | Rejects device indices outside the configured inventory | Maintained Python/native checks |
| dtype/autocast compatibility (E4) | Rejects BF16 autocast on profiles below compute capability 8.0 | Maintained Python check |
| Checkpoint compatibility (E5) | Detects checkpoint GPU architecture/compute-capability mismatches | Maintained Python check |
| Distributed failure (E6) | Injects a deterministic rank/sequence/collective failure, persists async error state, and recovers survivors through explicit communicator shrink | Maintained direct-collective suite and physical four-rank validation |
| Gradient errors (E7) | Detects invalid non-leaf/detached gradient use in covered paths | Maintained Python check |
| Collective contract errors | Detects operation, dtype, reduction, root, count, missing-peer, and timeout mismatches | Maintained coordinator/direct suite |
| Real worker exit | Terminates a rank while its communicator is active, infers the absent rank from timeout, preserves `ncclSystemError`, and permits explicit shrink | Validated on local and physical multi-host fixed-rank cases |
| Elastic DDP restart | Uses `torchrun --max-restarts=1` to replace the fixed-size worker group and verifies resumed DDP numerics | Maintained local Gloo and physical-host validation |
| Checkpoint recovery | Atomically restores model parameters, completed steps, SGD momentum, and continued updates | Maintained two-worker validation |
| Mid-accumulation recovery | Restores AdamW moments, scheduler, pending gradients, rank-local RNG, sampler cursor, and rank-remapped state before completing the step | Maintained two-worker validation; not sharded ZeRO/FSDP optimizer recovery |
| DataLoader replay | Reconstructs deterministic shuffled map-style multi-worker DataLoaders across epochs, worker counts, prefetch depths, batch sizes, and PyTorch/Python/NumPy RNG | Maintained parameterized matrix; live queues, iterable datasets, external side effects, and nondeterministic transforms are outside scope |

### GPU catalog, monitoring, and reports

| Feature | What is included | Status and boundary |
|---|---|---|
| GPU profile catalog | 24 profiles across Maxwell, Pascal, Volta, Turing, Ampere, Ada, Hopper, and Blackwell; consumer, data-center, workstation, embedded, and test segments | Maintained; see [GPU profiles](#gpu-profiles) for every profile ID |
| Architecture validation | Every profile declares compute capability; Python and C++ validators reject architecture/capability mismatches | Maintained |
| Source provenance | Per-profile NVIDIA specification URLs and measured/reference/synthetic status, plus checked-in current/legacy NVIDIA model tables | Maintained; `scripts/update_nvidia_gpu_catalog.py --check` verifies the snapshot |
| Uniform and heterogeneous inventories | One profile repeated by device count or mixed specifications such as `a100:4,h100:4` | Maintained in Python and native runtimes |
| Native device report | Memory peaks, IO, calls, kernel-launch counts, compatibility events, and maintained GEMM statistics | Maintained `fake_gpu_report.json` |
| Preflight and estimator reports | JSON/Markdown preflight, calibration, static-memory, LLM inference, SFT, FSDP/FSDP2, and comparison artifacts | Maintained for their documented runners; see [Reports](#reports) |
| Schema and consistency checks | JSON Schema validation plus cross-field reconciliation of bytes, timelines, directions, ranks, topology, failures, and recovery events | Maintained validation tooling |
| Human-readable test evidence | Markdown companions, a unified HTML error/test report, focused suite summaries, and the checked-in [validation snapshot](#validation-snapshot) | Maintained documentation artifacts |

The list above describes implemented behavior, not a promise that every CUDA,
PyTorch, NCCL, or framework API is emulated. See [Limitations](#limitations)
for the boundaries that should be reviewed before using a result for capacity
planning or production decisions.

## GPU profiles

Profiles are stored under `profiles/<architecture>/<segment>/*.yaml`, loaded by
the Python runtime, and compiled into native builds. Catalog segments are
`consumer`, `datacenter`, `workstation`, `embedded`, and `test`:

```text
profiles/
├── ampere/
│   ├── consumer/rtx3090ti.yaml
│   ├── datacenter/a100.yaml
│   ├── embedded/jetson-agx-orin-64gb.yaml
│   └── test/test-512m.yaml
└── blackwell/
    ├── datacenter/b200.yaml
    ├── embedded/jetson-t5000.yaml
    └── workstation/rtx-pro-5000-blackwell.yaml
```

The catalog currently contains 24 profiles across 8 NVIDIA architectures and
15 compute capabilities:

| Architecture | Segment(s) | Compute capability | Profile IDs |
|---|---|---|---|
| Maxwell | Consumer | 5.2 | `gtx980` |
| Pascal | Data center | 6.0, 6.1 | `p100`, `p4` |
| Volta | Data center | 7.0 | `v100` |
| Turing | Data center | 7.5 | `t4` |
| Ampere | Consumer, data center, embedded, test | 8.0, 8.6, 8.7 | `a100`, `a100-1g`, `a30`, `a10`, `a40`, `rtx3090ti`, `jetson-agx-orin-64gb`, `test-512m` |
| Ada | Data center | 8.9 | `l4`, `l40s` |
| Hopper | Data center | 9.0 | `h100`, `h200` |
| Blackwell | Data center, embedded, workstation | 10.0, 10.3, 11.0, 12.0, 12.1 | `b100`, `b200`, `b300`, `jetson-t5000`, `rtx-pro-5000-blackwell`, `rtx-pro-6000-blackwell`, `gb10` |

Every profile declares `compute_major` and `compute_minor`; both the Python
catalog validator and native C++ loader reject an architecture/compute
capability mismatch. Model-to-capability mappings come from NVIDIA's
[current CUDA GPU table](https://developer.nvidia.com/cuda/gpus) and
[legacy table](https://developer.nvidia.com/cuda/gpus/legacy). Product
specification URLs and whether a profile is measured, reference, or synthetic
are recorded in each YAML file.

Refresh or verify the checked-in NVIDIA model snapshot with:

```bash
python3 scripts/update_nvidia_gpu_catalog.py
python3 scripts/update_nvidia_gpu_catalog.py --check
```

Jetson and GB10 profiles describe unified system memory, not a GPU-exclusive
allocation budget. See [profiles/README.md](profiles/README.md) for data
provenance, status meanings, and validation rules.

Select one uniform profile or a heterogeneous device list:

```bash
./fgpu --profile rtx3090ti --device-count 2 python3 your_script.py
./fgpu --devices "t4,a100:2,h100" python3 your_script.py
```

Equivalent environment variables:

```bash
FAKEGPU_PROFILE=a100
FAKEGPU_DEVICE_COUNT=8
FAKEGPU_PROFILES=a100:4,h100:4
```

## Reports

| Report | Produced by | Contents |
|---|---|---|
| `fake_gpu_report.json` | Native runtime | Per-device peak memory, IO, calls, and maintained GEMM FLOPs |
| `cluster_report.json/.md` | Distributed coordinator | Collective/P2P totals, communicator-aware node-pair traffic, failure/recovery events, directional/per-operation peaks, bounded coordinator-observed timeline, modeled throughput, and rank statistics |
| TCP bandwidth report | `fakegpu bandwidth --json ...` | Validated payload size, per-rank timings, and end-to-end socket throughput |
| `preflight_report.json/.md` | Preflight CLI | Stage status, fit/OOM result, memory categories, and confidence |
| Real-GPU calibration report | `./ftest real_gpu_calibration` | Real, passthrough, hybrid, fakecuda, allocator, and NVML observations |
| Static memory validation report | `./ftest static_memory_validation` | Graph liveness, optimizer phases, workspace profiles, and real-CUDA comparison |
| LLM inference estimate | `fakegpu estimate-llm --json ...` | Checkpoint storage, KV cache, transient tensors, process-memory estimate, and matrix FLOPs |
| Virtual SMI state | `FAKEGPU_SMI_STATE_PATH` / `FAKEGPU_SMI_STATE_DIR` | Per-process current/peak tracked bytes and optional calibrated runtime overhead |

Output paths can be configured with:

```bash
FAKEGPU_REPORT_PATH=/path/to/fake_gpu_report.json
FAKEGPU_CLUSTER_REPORT_PATH=/path/to/cluster_report.json
FAKEGPU_CLUSTER_REPORT_MARKDOWN_PATH=/path/to/project_communication.md
```

When only `FAKEGPU_CLUSTER_REPORT_PATH` is set, FakeGPU automatically writes
the Markdown report beside the JSON file. Every distinct node pair from the
cluster configuration appears in its table, including pairs with zero traffic.
The coordinator writes both reports even when a session completes without any
communication, preserving the configured topology and explicit zero counters.
The JSON contract is `cluster_report.v1`, defined by
`cluster_report.schema.json`; `verification/check_cluster_report.py` validates
it by default. Sub-communicators retain their global-rank membership, and
successful P2P sends are counted once rather than once at each endpoint.
Failure and recovery entries preserve global ranks across a shrunk
communicator and report attempted payload, exclusions, survivors, and recovery
time.

## Validation snapshot

The maintained cross-GPU static-memory grid was checked on:

| GPU | Compute capability | PyTorch / CUDA |
|---|---:|---|
| NVIDIA GeForce RTX 3090 Ti | 8.6 | PyTorch 2.12.1 / CUDA 13.0 |
| NVIDIA RTX PRO 5000 72GB Blackwell | 12.0 | PyTorch 2.9.1 / CUDA 12.8 |

Across 13 MLP/Transformer workloads and 26 GPU observations:

- static peak bytes matched across both hosts
- maximum allocated-byte absolute error was `0.077160%`
- maximum requested-byte absolute error was `0.001358%`
- FP32 Efficient Attention requested-byte differences were at most 28 bytes
- no maintained Attention workload used an unprofiled or extrapolated profile

These results validate the maintained parameter grid. They do not establish accuracy for arbitrary models, shapes, PyTorch releases, or CUDA backends.

The Qwen3-8B BF16 inference path was also checked on the RTX PRO 5000 with
SDPA, a 9-token prompt, and two generated token IDs:

| Comparison | Predicted | Real CUDA | Absolute error |
|---|---:|---:|---:|
| Model load: FakeCUDA tracked vs allocator | 16,381,470,976 B | 16,383,586,816 B | 0.012914% |
| Inference peak: FakeCUDA tracked vs allocator | 16,385,992,936 B | 16,396,630,528 B | 0.064877% |
| Inference peak: checkpoint-only estimate vs allocator | 16,385,606,472 B | 16,396,630,528 B | 0.067234% |
| Process memory: virtual SMI vs NVML | 16,825,298,920 B | 16,835,936,256 B | 0.063182% |
| Matrix FLOPs: FakeCUDA execution vs CUDA | 151,415,620,864 | 151,415,620,864 | 0% |
| Matrix FLOPs: shape estimate vs CUDA | 151,415,619,584 | 151,415,620,864 | 0.000001% |

The virtual-SMI row includes a `442,049,024`-byte runtime overhead measured in
that same CUDA run. That value is evidence for this GPU, PyTorch, CUDA, model,
and operator path—not a portable constant.

The distributed paths were also checked on the same two hosts:

| Check | Placement | Result |
|---|---|---|
| Hybrid DDP numerical check | Two ranks sharing the RTX PRO 5000, then two ranks sharing the RTX 3090 Ti | Averaged gradient `[1.5, 3.0]`, identical gathered parameters, and the expected SGD update on both CUDA 12.8 and CUDA 13.0 |
| Hybrid FSDP numerical check | Two ranks sharing each GPU | Full sharding, averaged reduce-scatter gradients, optimizer update, full-parameter reconstruction, and state-dict restoration passed on both CUDA stacks |
| Hybrid FSDP2/DTensor matrix | Two or four ranks sharing each GPU | FP32, FP16, and BF16 parameter paths passed; FP16/BF16 gradient reduction also passed with reconstructed DTensor parameters |
| Hybrid DeepSpeed ZeRO matrix | Two or four ranks sharing each GPU | ZeRO 0–3, FP32/BF16, gradient accumulation, optimizer updates, and cross-rank parameter consistency passed on DeepSpeed 0.15.3 and 0.19.2 |
| DeepSpeed Pipeline Parallel | Two stages sharing each GPU | FP32/BF16 direct-P2P, GAS=2, and checkpoint matrices plus an FP32 batched-P2P smoke passed on DeepSpeed 0.15.3 and 0.19.2; the report identifies the 0.19.2 non-last-stage gradient-scaling compatibility setting |
| DeepSpeed AutoTP | Two ranks sharing the RTX 3090 Ti | ZeRO 0–2 × FP32/BF16 passed on DeepSpeed 0.19.2 with sharded weights, numerical updates, all-reduce/all-gather, and communication reports |
| DeepSpeed AutoEP | Two ranks sharing the RTX 3090 Ti | ZeRO 0–2 × FP32/BF16 passed on DeepSpeed 0.19.2 with nonuniform expert routing, variable-split all-to-all, gradients, updates, and exact split-byte accounting |
| Qwen3.5 DeepSpeed LoRA SFT | Two ranks sharing each GPU | ZeRO-2/3 forward, backward, AdamW update, communication report, accumulation, and reentrant checkpointing passed with local Qwen3.5-0.8B weights |
| DeepSpeed checkpoint and offload | Two ranks sharing each GPU | ZeRO-2/3 save/restore/continued training/FP32 consolidation passed; ZeRO-2 optimizer and ZeRO-3 optimizer + parameter CPU offload placed the requested state on CPU |
| Hugging Face Trainer + DeepSpeed | Two ranks sharing each GPU | Tiny ZeRO-2/3 and Qwen3.5-0.8B LoRA ZeRO-3 completed real updates, gradient accumulation, checkpointing, and rank-consistency checks |
| Physical-host Hybrid DDP | One rank on the RTX PRO 5000 ↔ one rank on the RTX 3090 Ti | The same numerical result across PyTorch 2.8.0/CUDA 12.8 and PyTorch 2.12.1/CUDA 13.0; TCP broadcast, all-reduce, and all-gather completed with zero timeouts |
| Physical-host elastic DDP | One rank on each GPU over Tailscale | The 3090 Ti worker exited with code 86 while its communicator was active; `torchrun` replaced both PIDs, synchronized asymmetric local restart counts, created a second communicator, and reproduced gradient `[1.5, 3.0]` and parameters `[0.85, -0.30]` |
| Physical-host elastic DDP checkpoint recovery | One rank on each GPU with host-local checkpoints | Both replacement workers restored step 1 parameters `[0.85, -0.30]` and SGD momentum `[1.5, 3.0]`, then produced step 2 parameters `[0.565, -0.87]` and momentum `[2.85, 5.7]`; two repeated runs each reported 512 node-pair bytes and a 64-byte peak |
| Physical-host elastic DDP training-state recovery | One rank and two persistent DataLoader workers on each GPU host; failure after one of two accumulation micro-steps | Every host checkpoint replicated both rank-local states; recovery deliberately mapped `0 → 1` and `1 → 0`, rebuilt seeded spawn DataLoaders, exactly replayed staged samples `8` and `7` with worker RNG values `0.4750137329` and `0.1462690830`, replaced all worker PIDs, and produced parameters `[0.983838, -0.014662]`; two repeated runs each reported 25,960 node-pair bytes and a 25,232-byte peak |
| Cross-runtime DataLoader replay matrix | CPU DataLoaders on macOS, RTX PRO 5000 Linux, and RTX 3090 Ti WSL | Five shuffle/epoch/worker/prefetch/batch scenarios covered 12 rank cases and 52 fresh workers per runtime. Sample-order plus PyTorch/Python/NumPy RNG digests matched across PyTorch 2.8.0, 2.9.1, and 2.12.1; two physical repeats used disjoint worker PID sets |
| Physical-host Hybrid FSDP2 | One rank on the RTX PRO 5000 ↔ one rank on the RTX 3090 Ti | FP32/FP16/BF16 parameters and FP16/BF16 gradient reductions passed over TCP; the report identifies collective dtype and reduction operator |
| Physical-host Hybrid DeepSpeed | One rank per physical GPU over Tailscale | ZeRO-2 passed across DeepSpeed 0.15.3 ↔ 0.19.2 with identical parameters and 176 reported node-pair bytes; ZeRO-3 now rejects mismatched DeepSpeed versions during preflight |
| Physical-host TCP all-to-all-v | RTX PRO 5000 coordinator/rank 0 ↔ RTX 3090 Ti rank 1 over Tailscale | Nonuniform 2 MiB/3 MiB and sparse 0 MiB/1 MiB cross-host splits produced exact payloads; 2 calls reported 12 MiB logical bytes, 6 MiB inter-node bytes, and a 5 MiB node-pair peak |
| Physical-host TCP all-reduce | RTX PRO 5000 coordinator/rank 0 ↔ RTX 3090 Ti rank 1 over Tailscale | Correct 1 MiB and 16 MiB reductions, zero coordinator timeouts; 16 MiB × 5 measured about `0.261 Gbit/s` algorithmic and `0.521 Gbit/s` bidirectional socket payload per rank |
| Physical-host rank-failure recovery | Ranks 0/2 on the RTX PRO 5000 ↔ ranks 1/3 on the RTX 3090 Ti WSL host | Injected rank 2 failure reached all four ranks as persistent `ncclRemoteError`; global ranks `[0,1,3]` recovered through `ncclCommShrink`, and all three obtained the post-recovery sum `7.0` |
| Physical-host process-exit recovery | Ranks 0/2 on the RTX PRO 5000 ↔ ranks 1/3 on the RTX 3090 Ti WSL host | Rank 2 exited with code 86 after communicator initialization; ranks `[0,1,3]` inferred its absence from one AllReduce timeout, retained `ncclSystemError`, explicitly shrank the communicator, and all obtained the recovered sum `7.0` |

The TCP numbers are an end-to-end simulator measurement from this specific
test network. They are not raw link capacity or an NCCL/RDMA performance
prediction.

## Test suites

```bash
./ftest smoke
./ftest cpu_sim
./ftest python
./ftest preflight_oom
./ftest tcp_bandwidth
./ftest distributed_resilience
./ftest elastic_ddp
./ftest elastic_ddp_checkpoint
./ftest elastic_ddp_training_state
./ftest dataloader_replay
./ftest static_memory_validation
./ftest real_gpu_calibration
python3 -m pytest -q
```

| Suite | Purpose |
|---|---|
| `smoke` | Build, library boundaries, preload, profiles, and memory types |
| `cpu_sim` | CPU-backed cuBLAS/cuBLASLt correctness |
| `python` | Basic PyTorch native-interception path |
| `preflight_oom` | Fit/OOM classification and report validation |
| `tcp_bandwidth` | Two logical nodes, TCP payload correctness, and throughput reporting |
| `distributed_resilience` | Rank-failure injection, real worker exit, elastic DDP restart/checkpoint/training-state resume, collective-timeout inference, persistent async errors, communicator shrink/recovery, collective mismatch, missing-peer timeout, and bounded operation-timeline retention |
| `elastic_ddp` | Two-worker `torchrun --max-restarts=1` process exit, full-group PID replacement, restarted DDP numerics, SGD checkpoint resume, and accumulated AdamW training-state resume over local Gloo |
| `elastic_ddp_checkpoint` | Focused model parameter, SGD momentum, completed-step, atomic checkpoint, and resumed-update validation after worker-group replacement |
| `elastic_ddp_training_state` | Focused AdamW first/second moments, StepLR, replicated rank-state bundle, rank-local RNG, `DistributedSampler` cursor, two-worker DataLoader reconstruction, staged-batch/worker-RNG replay, optimizer-step, and partial gradient-accumulation recovery |
| `dataloader_replay` | Parameterized shuffled persistent-worker reconstruction across epochs, worker counts, prefetch depths, batch sizes, and PyTorch/Python/NumPy RNG sources |
| `static_memory_validation` | ATen graph memory estimation; optional real-CUDA comparison |
| `real_gpu_calibration` | Real/passthrough/hybrid/fakecuda comparison on a supported GPU |

Additional distributed and framework checks are listed in [Reports and Validation](docs/reports-and-validation.md).
On a real CUDA host, the maintained numerical commands are:

```bash
python3 verification/run_hybrid_ddp_numerics.py --variant all
python3 verification/run_hybrid_fsdp_numerics.py
python3 verification/run_hybrid_fsdp2_numerics.py --world-size 4 --precision bf16
python3 verification/run_hybrid_fsdp2_numerics.py --world-size 4 --precision bf16 --reduce-precision parameter
python3 verification/run_hybrid_deepspeed_numerics.py --zero-stage all --precision bf16
python3 verification/run_hybrid_deepspeed_numerics.py --zero-stage 3 --precision fp32 --offload optimizer-and-parameter
python3 verification/run_hybrid_deepspeed_checkpoint.py --zero-stage 3 --precision fp32
python3 verification/run_hf_trainer_deepspeed.py --workload tiny --zero-stage 3 --precision bf16
python3 verification/run_qwen_deepspeed_lora_sft.py --model-dir /path/to/Qwen3.5-0.8B --output-dir build/qwen-deepspeed --zero-stage 3
python3 verification/run_hybrid_deepspeed_pipeline.py --precision all --activation-checkpointing
python3 verification/run_hybrid_deepspeed_pipeline.py --precision all --activation-checkpointing --gradient-accumulation-steps 2
python3 verification/run_hybrid_deepspeed_autotp.py --zero-stage all --precision all
python3 verification/run_hybrid_deepspeed_autoep.py --zero-stage all --precision all
```

The DeepSpeed checks cover the native Engine and a Transformers/PEFT Qwen
model with PyTorch optimizers, ZeRO checkpoint resume, Hugging Face Trainer,
CPU optimizer/parameter offload, two-stage Pipeline Parallel, AutoTP, and
AutoEP. AutoTP and training AutoEP require the newer DeepSpeed 0.19.2 test
stack; the 0.15.3 stack reports those modules as unavailable. Fused/JIT
optimizers, NVMe offload, sequence parallelism, combined AutoTP+AutoEP, and
physical ZeRO-3 with matching DeepSpeed versions remain separate validation
targets. See
[DeepSpeed Validation](docs/deepspeed-validation.md) for commands, measured
results, and the WSL-without-`nvcc` setup.

The physical two-host controller in
`verification/run_physical_multihost.py` verifies that both repositories use
the same Git commit, launches Hybrid DDP (including common execution options),
FSDP/FSDP2 (including mixed-precision parameters and reductions), optional
DeepSpeed ZeRO-2/3 cases, fixed-size elastic DDP restart/checkpoint/training-state recovery, and TCP fault cases
including injected failure and real four-rank worker-exit/shrink recovery,
then collects JSON and Markdown reports on the control host.
The standalone DataLoader matrix is selected explicitly with
`--case dataloader-replay --case-timeout 600`; it is excluded from the default
physical case set because it creates 52 worker processes on each runtime.
Cluster report validation also reconciles collective/P2P
counters, operation-timeline retention, collective dtype/reduction metadata,
resilience-event counts, directional links, and node-pair totals.

## Architecture

```text
GPU-facing application
├── Python runtime: fakegpu.init(runtime="fakecuda")
│   └── FakeCudaTensor + FakeGPU policies
│       └── maintained PyTorch operations execute on CPU
│
└── Native runtime: ./fgpu or fakegpu.init(runtime="native")
    └── libcuda / libcudart / libcublas / libnvidia-ml / libnccl
        ├── GlobalState: profiles, allocations, streams, metrics
        ├── system RAM and CPU-backed maintained math paths
        ├── optional real CUDA forwarding in hybrid mode
        └── coordinator for simulated distributed operations

Reports: device JSON · cluster JSON · preflight · calibration · static memory
```

## Limitations

- Native simulate mode does not execute arbitrary CUDA kernels.
- FakeCudaTensor covers Python/PyTorch behavior, not binary CUDA extensions.
- The checkpoint-only LLM estimator supports dense decoder-only safetensors models; it does not infer arbitrary repository control flow, MoE routing, quantization state, or custom kernels.
- Supported cuBLAS/cuBLASLt operations can be numerically checked on CPU; unsupported operations may be stubs.
- Distributed simulation checks semantics and control flow. Its TCP result includes coordinator reduction, memory copies, and process scheduling, so it is not an exact NCCL/RDMA or raw-link measurement.
- The fixed-size elastic DDP checks rely on `torchrun` for worker supervision and rendezvous. Mid-accumulation recovery stores one atomic file per host and replicates every rank-local gradient, RNG state, sampler cursor, deterministic DataLoader construction seed, committed sample record, and one application-staged prefetched batch into each file, so a replacement can select state by logical rank after reassignment. The maintained two-worker case rebuilds a spawn DataLoader and replays deterministic map-style worker transforms; it does not serialize live multiprocessing queues or guarantee replay for iterable datasets, external side effects, or nondeterministic transforms. This approach keeps the world size fixed, has O(world-size) rank-state storage per host, assumes the local checkpoint survives, and does not restore sharded ZeRO/FSDP optimizer state. FakeGPU does not provide heartbeat-based detection or dynamic membership resizing. The direct-collective recovery path still requires survivors to call `ncclCommShrink` with an explicit exclusion list.
- The standalone DataLoader matrix validates exact reconstruction of an application-visible committed/staged prefix. Internal prefetch counters are diagnostics only; it drains the remaining epoch before worker shutdown for compatibility with PyTorch 2.8 and does not serialize live queues or in-flight dataset side effects.
- Static and runtime memory estimates can omit backend-internal allocations outside matched profiles.
- Hybrid mode requires a real GPU and remains limited to validated Driver/runtime surfaces.
- macOS system binaries can remove `DYLD_*` variables because of SIP; use a Homebrew, conda, or pyenv Python.

## Documentation

- [Getting Started](docs/getting-started.md)
- [Quick Reference](docs/quick-reference.md)
- [AI Researcher Preflight](docs/ai-researcher-preflight.md)
- [LLM Inference Estimation](docs/llm-inference-estimation.md)
- [LLM SFT Memory Estimation](docs/llm-sft-memory-estimation.md)
- [Architecture and Project Structure](docs/project-structure.md)
- [Torch Patch System](docs/phase2-custom-torch.md)
- [Reports and Validation](docs/reports-and-validation.md)
- [Distributed Simulation Usage](docs/distributed-sim-usage.md)
- [Distributed Design Notes](docs/multi-node-design.md)
- [cuBLASLt Compatibility Notes](docs/cublaslt-fix.md)
- [Error Simulation](docs/error-simulation.md)

Preview the documentation site:

```bash
python3 -m pip install -e ".[docs]"
mkdocs serve
```

## License

[MIT](LICENSE)
