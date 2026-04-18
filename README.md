# FakeGPU

A CUDA API interception library that simulates GPU devices in non-GPU environments, enabling basic operations for PyTorch and other deep learning frameworks, and supporting single-host multi-process distributed simulation for NCCL-style workloads.

## How It Works

FakeGPU makes applications believe they are running on real NVIDIA GPUs by intercepting every CUDA / NVML API call and redirecting GPU operations to system RAM. No physical GPU is needed. It provides two independent runtime paths:

### Overview

```
+-----------------------------------------------------------------------+
|                        User's GPU Application                         |
|                  (PyTorch, pynvml, nvidia-smi, ...)                   |
+----------------------------------+------------------------------------+
                                   |
                    CUDA / NVML API calls
                                   |
              +--------------------+--------------------+
              |                                         |
   [Path A: Native]                          [Path B: FakeCudaTensor]
   C-level interception                      Python-level monkeypatch
              |                                         |
              v                                         v
+----------------------------+            +----------------------------+
| Dynamic Linker intercepts  |            | fakegpu.patch_torch()      |
| symbols via LD_PRELOAD /   |            |                            |
| DYLD_INSERT_LIBRARIES      |            | Layer 1: FakeCudaTensor    |
|                            |            |   __torch_function__       |
| cuda_stubs.cpp             |            |   tensor.device = cuda:N   |
|   cudaMalloc -> malloc()   |            |   tensor.is_cuda = True    |
|   cudaFree   -> free()     |            |                            |
|   cudaMemcpy -> memcpy()   |            | Layer 2: FakeGPU additions |
|   cudaLaunchKernel -> noop |            |   GPU profiles & limits    |
|                            |            |   Memory tracking + OOM    |
| nvml_stubs.cpp             |            |   Cross-device guards      |
|   nvmlDeviceGetMemoryInfo  |            |   Autocast validation      |
|   nvmlDeviceGetName ...    |            +----------------------------+
|                            |                          |
| cublas_stubs.cpp           |               All compute runs on CPU
|   GEMM -> CPU simulation   |
+----------------------------+
              |
              v
+----------------------------+            +----------------------------+
|       GlobalState          |            |   _DeviceMemoryTracker     |
|  (singleton, thread-safe)  |            |   (Python per-device)      |
|                            |            |                            |
|  Device[] with GpuProfile  |            |  used / peak / total mem   |
|  allocations map: ptr ->   |            |  OOM enforcement           |
|    {size, device, kind}    |            |  weakref GC cleanup        |
|  per-device stats:         |            +----------------------------+
|    peak mem, IO, FLOPs     |                          |
+----------------------------+                          v
              |                           Terminal summary on exit
              v
+----------------------------+
|    Monitor (atexit)        |
|  fake_gpu_report.json      |
|  terminal summary          |
+----------------------------+
```

### Memory Lifecycle (Native Path)

```
cudaMalloc(devPtr, size)                    cudaFree(devPtr)
        |                                          |
        v                                          v
  malloc(size)  ──── real system RAM ────>   free(devPtr)
        |                                          |
        v                                          v
  GlobalState.register_allocation()       GlobalState.release_allocation()
  ├─ Check: used + size > total?          ├─ Look up ptr in allocations map
  │    Yes -> return OOM error            ├─ dev.used_memory -= size
  │    No  -> continue                    └─ Remove from allocations map
  ├─ dev.used_memory += size
  ├─ dev.used_memory_peak = max(peak, used)
  └─ Record {ptr -> (size, device)} in map
                    |
                    v  (on process exit)
           Monitor.dump_report()
           └─ Write fake_gpu_report.json
              ├─ per-device peak memory
              ├─ IO bytes (H2D / D2H / D2D)
              └─ compute FLOPs (cuBLAS GEMM)
```

### Why Can It Simulate a GPU?

1. **Symbol interception**: The dynamic linker resolves `cudaMalloc`, `nvmlDeviceGetName`, etc. to FakeGPU's `extern "C"` stubs before the real NVIDIA libraries are found. Applications see the exact same C API.
2. **RAM as VRAM**: All "device memory" is plain `malloc`'d system RAM. `cudaMemcpy` becomes `memcpy`. From the application's perspective, pointers are valid and memory operations succeed.
3. **Faithful bookkeeping**: Per-device memory limits, allocation tracking, peak usage, and OOM errors all behave like a real GPU driver — so code that checks `torch.cuda.mem_get_info()` or handles `OutOfMemoryError` works correctly.
4. **Profile-driven identity**: GPU profiles (name, compute capability, memory size, SM count) are loaded from YAML, so `torch.cuda.get_device_name()` returns "NVIDIA A100-SXM4-80GB" and `torch.cuda.get_device_capability()` returns `(8, 0)`.

## Documentation

This repository now ships a MkDocs + Material for MkDocs documentation site configuration.

Local preview:
```bash
python3 -m pip install -e ".[docs]"
mkdocs serve
```

GitHub Pages deployment:
- The site configuration lives in `mkdocs.yml`
- The published content lives under `docs/`
- `.github/workflows/docs.yml` builds and deploys the site through GitHub Pages
- The workflow auto-deploys on pushes to `main`
- `workflow_dispatch` remains available, so `dev` or any other branch can still be published manually from the Actions page

## Timeline

### Implemented Features
- [x] **CUDA Driver API** - Device management, memory allocation, kernel launch
- [x] **CUDA Runtime API** - cudaMalloc/Free, cudaMemcpy, Stream, Event
- [x] **cuBLAS/cuBLASLt** - Matrix operations (GEMM, PyTorch 2.x compatible)
- [x] **NVML API** - GPU information queries
- [x] **Distributed Communication Simulation** - Fake NCCL init, collective ops, point-to-point send/recv, communicator split, grouped submission
- [x] **Coordinator & Topology Model** - Single-host multi-process cluster simulation with Unix/TCP coordinator transport and YAML cluster config
- [x] **Cluster-Level Reporting** - Collective counts/bytes/timing plus inter-node and intra-node link statistics
- [x] **Python API Wrapper** - `import fakegpu; fakegpu.init(runtime="native")` enables FakeGPU from inside Python
- [x] **PyTorch Support** - Basic tensor ops, linear layers, neural networks
- [x] **Hybrid Multi-Process Validation Path** - Hybrid compute + simulated communication smoke coverage
- [x] **Real Scenario Testing** - LLM inference smoke test (Qwen2.5-0.5B-Instruct)
- [x] **GPU Tool Compatibility** - Compatible with existing GPU status monitoring tools (nvidia-smi, gpustat, etc.)
- [x] **Preset GPU Info** - Add more preset GPU hardware configurations
- [x] **Detailed Reporting** - Writes `fake_gpu_report.json` with per-device peak memory, IO bytes, and cuBLAS/cuBLASLt FLOPs
- [x] **Multi-Architecture & Data Types** - Support different GPU architectures and various data storage/memory types

### Planned Features
- [ ] **Deeper Protocol Fidelity** - Better overlap/ordering realism beyond semantic NCCL simulation
- [ ] **Broader Multi-Host Validation** - More real multi-machine coverage beyond current single-host and loopback transport validation
- [ ] **Broader Framework Coverage** - Extend maintained validation beyond the current PyTorch/DDP smoke paths into wider Transformers and model families
- [ ] **Enhanced Testing** - Optimize test suite with more languages and runtime environments

## Operation Modes

FakeGPU supports three compute modes, controlled by the `FAKEGPU_MODE` environment variable:

### Simulate Mode (Default)
```bash
FAKEGPU_MODE=simulate ./fgpu python your_script.py
```
- All CUDA APIs return fake data
- No real GPU required
- Device memory backed by system RAM
- Kernel launches are no-ops

### Passthrough Mode
```bash
FAKEGPU_MODE=passthrough ./fgpu python your_script.py
```
- Forwards all CUDA calls to real GPU libraries
- Results identical to running without FakeGPU
- Useful for parity testing and debugging
- Requires real GPU and CUDA installation

### Hybrid Mode
```bash
FAKEGPU_MODE=hybrid FAKEGPU_OOM_POLICY=clamp ./fgpu python your_script.py
```
- Device info is virtualized (can report different GPU specs)
- Compute operations use real GPU
- OOM safety policies prevent crashes when virtual memory exceeds real GPU capacity

**OOM Policies** (for Hybrid mode):
- `clamp` (default): Report memory clamped to real GPU capacity
- `managed`: Use `cudaMallocManaged` for oversubscription (relies on UVM)
- `mapped_host`: Use `cudaHostAllocMapped` for overflow allocations
- `spill_cpu`: Spill excess allocations to CPU memory

**Environment Variables:**
```bash
FAKEGPU_MODE={simulate,passthrough,hybrid}  # Operation mode
FAKEGPU_OOM_POLICY={clamp,managed,mapped_host,spill_cpu}  # OOM policy for hybrid mode
FAKEGPU_REAL_CUDA_LIB_DIR=/path/to/cuda/lib  # Custom CUDA library path
```

## Distributed Communication Modes

Distributed communication is controlled separately by `FAKEGPU_DIST_MODE` so compute mode and communication mode can be combined.

| `FAKEGPU_DIST_MODE` | Meaning |
|---|---|
| `disabled` | No FakeGPU distributed layer |
| `simulate` | FakeGPU coordinator executes collectives / p2p using simulated topology |
| `proxy` | Real NCCL executes collectives while FakeGPU records control-plane and cluster-report data |
| `passthrough` | Thin forwarding to real NCCL with minimal FakeGPU wrapping |

For first-time setup, the recommended mode pair is:

```bash
FAKEGPU_MODE=simulate
FAKEGPU_DIST_MODE=simulate
```

Useful distributed environment variables:

```bash
FAKEGPU_DIST_MODE={disabled,simulate,proxy,passthrough}
FAKEGPU_CLUSTER_CONFIG=/abs/path/to/cluster.yaml
FAKEGPU_COORDINATOR_TRANSPORT={unix,tcp}
FAKEGPU_COORDINATOR_ADDR=/tmp/fakegpu.sock        # or 127.0.0.1:29591 for tcp
FAKEGPU_CLUSTER_REPORT_PATH=/path/to/cluster-report.json
FAKEGPU_STAGING_CHUNK_BYTES=1048576
FAKEGPU_STAGING_FORCE_SOCKET=1
```

**Report Output (Hybrid mode):**
```json
{
  "report_version": "1.5.0",
  "mode": "hybrid",
  "oom_policy": "clamp",
  "hybrid_stats": {
    "real_alloc": {"count": 10, "bytes": 1073741824},
    "managed_alloc": {"count": 0, "bytes": 0},
    "spilled_alloc": {"count": 2, "bytes": 134217728}
  },
  "backing_gpus": [
    {"index": 0, "total_memory": 25769803776, "used_memory": 1073741824}
  ],
  ...
}
```


## Quick Start

### Build

```bash
cmake -S . -B build
cmake --build build
```

CPU-backed compute for supported cuBLAS/cuBLASLt operators is enabled by default (runs on CPU; no real GPU required).

Optional (disable CPU simulation and fall back to stub/no-op behavior):
```bash
cmake -S . -B build -DENABLE_FAKEGPU_CPU_SIMULATION=OFF
cmake --build build
```

Generated libraries:
- Linux:
  - `build/libcuda.so.1` - CUDA Driver API
  - `build/libcudart.so.12` - CUDA Runtime API
  - `build/libcublas.so.12` - cuBLAS/cuBLASLt API
  - `build/libnvidia-ml.so.1` - NVML API
  - `build/libnccl.so.2` - Fake NCCL shim for distributed simulation / proxy / passthrough
  - `build/fakegpu-coordinator` - Coordinator daemon for distributed communication
- macOS:
  - `build/libcuda.dylib` - CUDA Driver API
  - `build/libcudart.dylib` - CUDA Runtime API
  - `build/libcublas.dylib` - cuBLAS/cuBLASLt API
  - `build/libnvidia-ml.dylib` - NVML API

### Test

**Standardized test runner (recommended):**
```bash
./ftest smoke          # C + Python (no torch needed)
./ftest cpu_sim        # CPU simulation correctness (validates cuBLAS ops; runs a PyTorch matmul check if torch is installed)
./ftest python         # PyTorch tests (requires torch)
./ftest llm            # LLM inference smoke test (requires torch + transformers + local model files)
./ftest all            # smoke + python
```

**Comparison test (recommended):**
```bash
./test/run_comparison.sh
```
Runs identical tests on both real GPU and FakeGPU to verify correctness.

**PyTorch test:**
```bash
./fgpu python3 test/test_comparison.py --mode fake
```

**Distributed validation (maintained):**
```bash
python3 verification/test_coordinator_smoke.py
python3 test/test_allreduce_correctness.py
python3 verification/test_allgather_correctness.py
python3 verification/test_group_semantics.py
./test/run_multinode_sim.sh 2       # 2-rank ProcessGroupNCCL / DDP smoke
./test/run_multinode_sim.sh 4       # 4-rank single-host simulate smoke
./test/run_ddp_multinode.sh 4       # clustered DDP validation + cluster report checks
./test/run_hybrid_multinode.sh 2   # hybrid compute + simulated communication
```

The maintained paths above pass in the current tree. The DDP-oriented scripts provide smoke and validation coverage for the simulate-mode ProcessGroupNCCL path, but they should not be read as a claim of full PyTorch or NCCL protocol parity.

### Usage

```python
import torch

# All PyTorch CUDA operations are intercepted by FakeGPU
device = torch.device('cuda:0')
x = torch.randn(100, 100, device=device)
y = torch.randn(100, 100, device=device)
z = x @ y  # Matrix multiplication

# Simple neural network
model = torch.nn.Linear(100, 50).to(device)
output = model(x)
```

**Runtime requires preloading all libraries:**
Linux:
```bash
LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH \
LD_PRELOAD=./build/libcublas.so.12:./build/libcudart.so.12:./build/libcuda.so.1:./build/libnvidia-ml.so.1:./build/libnccl.so.2 \
python your_script.py
```

macOS:
```bash
DYLD_LIBRARY_PATH=./build:$DYLD_LIBRARY_PATH \
DYLD_INSERT_LIBRARIES=./build/libcublas.dylib:./build/libcudart.dylib:./build/libcuda.dylib:./build/libnvidia-ml.dylib:./build/libnccl.dylib \
python3 your_script.py
```

On macOS, prefer a Homebrew, conda, or pyenv-managed Python. SIP strips `DYLD_*` variables for protected system executables such as `/usr/bin/python3`.

**Python wrapper (no need to start Python with LD_PRELOAD):**
```python
import fakegpu

# Call early (before importing torch / CUDA-using libraries)
fakegpu.init(runtime="native")  # preload/native route, default: 8x A100
# Optional: fakegpu.init(runtime="native", profile="t4", device_count=2)
# Optional: fakegpu.init(runtime="native", devices="a100:4,h100:4")

import torch
```

For the Python-level fake-CUDA route:

```python
import fakegpu

fakegpu.init(runtime="auto")      # prefer torch.fakegpu when installed, else native
# or: fakegpu.init(runtime="fakecuda")
```

**Tiny Transformer training smoke with `pytorch-fakegpu`:**
```bash
python3 demo_usage.py --test transformer
python3 demo_usage.py --test transformer --quiet
```

This path uses `fakegpu.torch_patch.patch()` inside the demo so CUDA-style
training code can run as a fake-CUDA smoke test on CPU-only hosts.

**Shortcut runner:**
```bash
./fgpu python your_script.py
# Optional: ./fgpu --profile t4 --device-count 2 python your_script.py
# Optional: ./fgpu --devices 't4,h100' python your_script.py
# Optional: FAKEGPU_BUILD_DIR=/path/to/build ./fgpu python your_script.py
```

**Python runner (installs `fakegpu` console script):**
```bash
fakegpu python your_script.py
# Optional: fakegpu --profile t4 --device-count 2 python your_script.py
# Optional: fakegpu --devices 'a100:4,h100:4' python your_script.py
# or: python -m fakegpu python your_script.py
```

**Distributed runner example (single host, simulated multi-node):**
```bash
SOCKET_PATH=/tmp/fakegpu-coordinator.sock
CLUSTER_CONFIG=$PWD/verification/data/cluster_valid.yaml

FAKEGPU_DIST_MODE=simulate \
FAKEGPU_CLUSTER_CONFIG="$CLUSTER_CONFIG" \
FAKEGPU_COORDINATOR_TRANSPORT=unix \
FAKEGPU_COORDINATOR_ADDR="$SOCKET_PATH" \
FAKEGPU_CLUSTER_REPORT_PATH=/tmp/fakegpu-cluster-report.json \
./build/fakegpu-coordinator --transport unix --address "$SOCKET_PATH"
```

In another terminal:
```bash
SOCKET_PATH=/tmp/fakegpu-coordinator.sock
CLUSTER_CONFIG=$PWD/verification/data/cluster_valid.yaml

./fgpu \
  --mode simulate \
  --dist-mode simulate \
  --cluster-config "$CLUSTER_CONFIG" \
  --coordinator-transport unix \
  --coordinator-addr "$SOCKET_PATH" \
  --device-count 4 \
  torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --master_addr 127.0.0.1 \
  --master_port 29500 \
  test/test_ddp_multinode.py \
  --report-dir /tmp/fakegpu-rank-reports \
  --epochs 1
```

`./fgpu` now preloads the fake NCCL shim automatically, so no extra `LD_PRELOAD`/`DYLD_INSERT_LIBRARIES` export is required for this path.

For a more complete walkthrough, see `docs/distributed-sim-usage.md`.

**GPU tools (nvidia-smi)**
```bash
# FakeGPU-simulated devices via NVML stubs
./fgpu nvidia-smi
# Temperatures may show N/A because the TemperatureV struct is not fully emulated yet.
```

### Reporting

FakeGPU writes `fake_gpu_report.json` at program exit (also triggered by `nvmlShutdown()`), including:
- Per-device `used_memory_peak` (peak VRAM requirement)
- Per-device IO bytes/calls: H2D / D2H / D2D / peer copies + memset
- Per-device compute FLOPs/calls for GEMM/Matmul (`cuBLAS` / `cuBLASLt`)

When distributed mode is enabled and `FAKEGPU_CLUSTER_REPORT_PATH` is set, FakeGPU also writes a cluster-level JSON report with:
- Cluster/world-size metadata
- Collective counts, bytes, and estimated time
- Intra-node and inter-node link statistics
- Experimental topology/timing fields used by the distributed validation scripts

Notes:
- FLOPs are theoretical estimates (GEMM ≈ `2*m*n*k`, complex GEMM uses a larger factor); kernel launches are no-ops and not counted.
- `host_io.memcpy_*` tracks Host↔Host copies (e.g. `cudaMemcpyHostToHost`).
- Optional: set `FAKEGPU_REPORT_PATH=/path/to/report.json` to change the output location.

## Test Results

| Test | Status | Description |
|------|--------|-------------|
| Tensor creation | ✓ | Basic memory allocation |
| Element-wise ops | ✓ | Add, multiply, trigonometric |
| Matrix multiplication | ✓ | cuBLAS/cuBLASLt GEMM |
| Linear layer | ✓ | PyTorch nn.Linear |
| Neural network | ✓ | Multi-layer forward pass |
| Memory transfer | ✓ | CPU ↔ GPU data copy |

## Architecture

```
FakeGPU
├── src/
│   ├── core/          # Global state and device management
│   ├── cuda/          # CUDA Driver/Runtime API stubs
│   ├── cublas/        # cuBLAS/cuBLASLt API stubs
│   ├── distributed/   # Coordinator, topology config, communicator, staging, collective execution
│   ├── nccl/          # Fake NCCL shim plus proxy/passthrough dispatch
│   ├── nvml/          # NVML API stubs
│   └── monitor/       # Resource monitoring and reporting
└── test/              # Test scripts
```

**Core Design:**
- Uses `LD_PRELOAD` to intercept CUDA API calls
- Device memory backed by system RAM (malloc/free)
- By default, supported cuBLAS/cuBLASLt ops are executed on CPU (CPU simulation)
- Build with `-DENABLE_FAKEGPU_CPU_SIMULATION=OFF` to disable CPU simulation
- Kernel launches are no-ops (logging only)

### GPU Profiles

- Default build exposes eight `Fake NVIDIA A100-SXM4-80GB` devices to mirror common server nodes.
- GPU parameters are edited in YAML under `profiles/*.yaml`; CMake embeds these files at build time so no runtime file lookup is needed. Add or tweak a file, rerun `cmake -S . -B build`, and the new profiles are compiled in.
- Presets cover multiple compute capabilities (Maxwell→Blackwell) and feed the existing helpers (`GpuProfile::GTX980/P100/V100/T4/A40/A100/H100/L40S/B100/B200`), which now prefer the YAML data and fall back to code defaults if parsing fails.
- Select presets at runtime via environment variables:
  - `FAKEGPU_PROFILE=<id>` + `FAKEGPU_DEVICE_COUNT=<n>` (uniform devices)
  - `FAKEGPU_PROFILES=<spec>` (per-device spec, e.g. `a100:4,h100:4` or `t4,l40s`)
- Python wrapper passes the same settings (must be called before importing CUDA-using libs like torch for the preload route): `fakegpu.init(runtime="native", profile="t4", device_count=2)` or `fakegpu.init(runtime="native", devices="a100:4,h100:4")`.

## Limitations

- ❌ No real GPU execution (CUDA kernels are no-ops; supported cuBLAS/cuBLASLt ops run on CPU)
- ❌ Complex models (Transformers) may require additional APIs
- ⚠️ Distributed support is a semantic simulator, not a protocol-level recreation of NCCL/RDMA/NVLink internals
- ⚠️ The most validated distributed path is still single-host multi-process simulation; TCP coordinator support exists, but real multi-machine coverage is more limited
- ⚠️ Some proxy/passthrough and advanced NCCL behaviors remain experimental
- ⚠️ macOS: Official PyTorch wheels do not include CUDA, so FakeGPU only helps when running CUDA-enabled binaries (typically in Linux via Docker/VM).
- ⚠️ For testing and development environments only

## Use Cases

### Already Supported

- **CI/CD Pipeline Testing** — Run GPU-dependent tests on CPU-only CI runners (GitHub Actions, GitLab CI). FakeGPU's JSON report output and HTML test reports integrate naturally into CI artifact archival. No GPU runners or cloud GPU costs needed.

- **Multi-GPU Code Development** — Develop and validate multi-device parallelization (DataParallel, device placement) on a laptop. Configure `FAKEGPU_DEVICE_COUNT=8` or `FAKEGPU_PROFILES=a100:4,h100:4` to simulate heterogeneous multi-GPU setups. Cross-device operation guards catch real bugs.

- **OOM Debugging & Memory Planning** — Test whether a model fits on target hardware before committing to expensive GPU time. Use small-VRAM profiles (e.g., `a100-1g` with 1 GB) to trigger OOM quickly, or full profiles to estimate peak VRAM. `torch.cuda.mem_get_info()` and `OutOfMemoryError` behave like real CUDA.

- **Distributed Training Development** — Prototype DDP, NCCL collectives, and multi-rank communication on a single host. The full coordinator + topology + collective stack simulates allreduce, allgather, alltoall, broadcast, and reduce-scatter with configurable cluster topology (NVLink, InfiniBand, Ethernet bandwidth/latency).

- **Hardware Compatibility Testing** — Validate code across GPU architectures (Maxwell → Blackwell) using 11 built-in profiles. Catch dtype mismatches (bf16 requires compute capability >= 8.0), checkpoint portability issues, and autocast failures before deploying to specific cloud instances.

- **Education & Teaching** — Teach CUDA programming, distributed training, and GPU memory management on student laptops without GPU labs. Students write real `device="cuda"` code that executes, produces correct gradients, and shows realistic error messages.

- **Model Architecture Prototyping** — Validate that training converges (forward/backward pass, optimizer steps, loss decrease) before requesting GPU time. CPU-backed cuBLAS computes correct GEMM results. Tested with nanoGPT (10.65M params) and MoE architectures.

- **GPU Monitoring Tool Development** — Build and test GPU dashboards, alerting systems, or orchestration tools (nvidia-smi, gpustat, nvitop, custom Prometheus exporters) against NVML stubs without real hardware.

### Potential Additions

- **Cost Optimization Workflow** — Use hybrid mode (`FAKEGPU_MODE=hybrid`) to develop on smaller GPUs while simulating larger ones. OOM policies (`clamp`, `managed`, `spill_cpu`) let you test 80 GB A100 workloads on a 24 GB card.

- **Pre-submission Resource Estimation** — Before submitting jobs to Slurm/PBS clusters, run locally with FakeGPU to estimate per-GPU peak VRAM, IO volume, and compute FLOPs from `fake_gpu_report.json`. Use this to right-size GPU allocation requests.

- **Framework Migration Testing** — When upgrading PyTorch versions (verified 2.6.0 → 2.11.0) or switching frameworks, run your training pipeline under FakeGPU to catch API breakage without tying up real GPUs.

- **Reproducibility Validation** — Verify that training scripts, checkpoint loading, and data pipelines run correctly on a clean CPU-only environment before sharing code in papers or repositories.

## Dependencies

- CMake 3.14+
- C++17 compiler
- Python 3.10+ (for package, testing, and docs)
- PyTorch 2.x (optional, for testing)

## License

MIT License

## Documentation

- `mkdocs.yml` - MkDocs site config for local preview and GitHub Pages
- [Test Guide](test/README.md) - Detailed testing instructions
- [Torch Patch System](docs/phase2-custom-torch.md) - Vendored FakeCudaTensor backend with GPU profiles, memory tracking, and error simulation
- [Distributed Usage Guide](docs/distributed-sim-usage.md) - How to run single-host simulated multi-node workloads
- [Multi-Node Design](docs/multi-node-design.md) - Distributed design notes, implementation plan, and current boundaries
- [cuBLASLt Implementation](docs/cublaslt-fix.md) - cuBLASLt support details
