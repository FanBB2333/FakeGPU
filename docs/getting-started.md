# Getting Started

This guide is the shortest path from a fresh checkout to a working FakeGPU environment.

## 1. Prerequisites

- Python 3.10+
- CMake 3.14+
- A C++17-capable compiler
- Linux or macOS
- `torch` installed if you want to run the PyTorch or hybrid distributed checks

## 2. Build the project

```bash
cmake -S . -B build
cmake --build build
```

By default, FakeGPU enables CPU-backed execution for supported cuBLAS/cuBLASLt operators. That gives you better correctness coverage than pure stubs for the maintained math paths.

If you only want stub/no-op behavior:

```bash
cmake -S . -B build -DENABLE_FAKEGPU_CPU_SIMULATION=OFF
cmake --build build
```

## 3. Run the baseline checks

Start with the maintained smoke path:

```bash
./ftest smoke
./ftest cpu_sim
```

If your Python environment has PyTorch:

```bash
./ftest python
```

Recommended next distributed checks:

```bash
python3 verification/test_coordinator_smoke.py
python3 test/test_allreduce_correctness.py
./test/run_hybrid_multinode.sh 2
```

## 4. Run your first FakeGPU command

The wrapper is the easiest entry point because it sets the preload variables correctly:

```bash
./fgpu nvidia-smi
./fgpu python3 -c "import torch; print(torch.cuda.device_count())"
```

If `nvidia-smi` is installed, `./fgpu nvidia-smi` is the most visual smoke check. Example output on this repository's default profile set looks like:

```text
Tue Mar 17 21:57:16 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.211.01             Driver Version: 570.195.03     CUDA Version: 12.8     |
|=========================================+========================+======================|
|   0  Fake NVIDIA A100-SXM4-80GB     Off |   00000000:01:00.0 Off |                  Off |
| 50%   N/A    P0              1W /  300W |       0MiB /  81920MiB |     50%      Default |
+-----------------------------------------+------------------------+----------------------+
|   1  Fake NVIDIA A100-SXM4-80GB     Off |   00000000:02:00.0 Off |                  Off |
| 50%   N/A    P0              1W /  300W |       0MiB /  81920MiB |     50%      Default |
+-----------------------------------------------------------------------------------------+
```

The timestamp, driver line, and number of devices may vary. Temperature can show `N/A` because that part of the NVML surface is not fully modeled yet.

You can also choose profiles and modes from the command line:

```bash
./fgpu --profile t4 --device-count 2 python3 your_script.py
./fgpu --mode hybrid --oom-policy clamp python3 your_script.py
```

If you have the `fakegpu` Python environment plus a local `pytorch-fakegpu` install,
the repository also ships a tiny Transformer training demo that uses fake CUDA
semantics from inside Python instead of preload-based interception:

```bash
python3 demo_usage.py --test transformer
python3 demo_usage.py --test transformer --quiet
```

That path is useful on CPU-only hosts where you still want `device="cuda"` style
training code to run for smoke validation.

## 5. Enable FakeGPU inside Python

Use `fakegpu.init()` when you want to stay inside a normal Python process. Call it before importing `torch` or any CUDA-using library.

```python
import fakegpu

fakegpu.init(profile="a100", device_count=4)

import torch
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
```

You can also pass per-device profile specs:

```python
fakegpu.init(devices="a100:2,h100:2")
```

## 6. Compute and distributed modes

### Compute modes

| Mode | Meaning |
|---|---|
| `simulate` | Fake device info, fake memory, and stubbed compute paths; no real GPU required |
| `hybrid` | Fake device identity, but real CUDA execution where supported; OOM policy matters |
| `passthrough` | Forward to real CUDA libraries with minimal FakeGPU interference |

### Distributed modes

| Mode | Meaning |
|---|---|
| `disabled` | No FakeGPU distributed layer |
| `simulate` | FakeGPU coordinator executes collective and point-to-point semantics |
| `proxy` | Real NCCL does the data movement while FakeGPU records control-plane and report data |
| `passthrough` | Thin forwarding to real NCCL |

## 7. Key environment variables

```bash
FAKEGPU_MODE={simulate,passthrough,hybrid}
FAKEGPU_DIST_MODE={disabled,simulate,proxy,passthrough}
FAKEGPU_OOM_POLICY={clamp,managed,mapped_host,spill_cpu}
FAKEGPU_PROFILE=a100
FAKEGPU_DEVICE_COUNT=8
FAKEGPU_PROFILES=a100:4,h100:4
FAKEGPU_CLUSTER_CONFIG=/abs/path/to/cluster.yaml
FAKEGPU_COORDINATOR_TRANSPORT={unix,tcp}
FAKEGPU_COORDINATOR_ADDR=/tmp/fakegpu.sock
FAKEGPU_CLUSTER_REPORT_PATH=/path/to/cluster-report.json
FAKEGPU_REPORT_PATH=/path/to/fake_gpu_report.json
```

## 8. Preview the documentation locally

```bash
python3 -m pip install -e ".[docs]"
mkdocs serve
```

## 9. Read next

- [Quick Reference](quick-reference.md)
- [Architecture](project-structure.md)
- [Reports & Validation](reports-and-validation.md)
- [Distributed Simulation Usage Guide](distributed-sim-usage.md)
