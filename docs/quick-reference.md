# Quick Reference

## Build commands

```bash
cmake -S . -B build
cmake --build build
```

Enable verbose FakeGPU logging:

```bash
cmake -S . -B build -DENABLE_FAKEGPU_LOGGING=ON
cmake --build build
```

Disable CPU-backed cuBLAS/cuBLASLt simulation:

```bash
cmake -S . -B build -DENABLE_FAKEGPU_CPU_SIMULATION=OFF
cmake --build build
```

## Common run commands

```bash
./fgpu nvidia-smi
./fgpu python3 your_script.py
./fgpu --profile t4 --device-count 2 python3 your_script.py
./fgpu --devices "a100:4,h100:4" python3 your_script.py
./fgpu --mode hybrid --oom-policy clamp python3 your_script.py
```

Dynamic initialization inside Python:

```bash
python3 -c "import fakegpu; fakegpu.init(); import torch; print(torch.cuda.device_count())"
```

## Test commands

```bash
./ftest smoke
./ftest cpu_sim
./ftest python
./ftest all
```

```bash
./test/run_comparison.sh
./test/run_multinode_sim.sh 2
./test/run_ddp_multinode.sh 4
./test/run_hybrid_multinode.sh 2
```

## Manual preload

Using `./fgpu` is recommended. If you need manual control:

### Linux

```bash
LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH \
LD_PRELOAD=./build/libcublas.so.12:./build/libcudart.so.12:./build/libcuda.so.1:./build/libnvidia-ml.so.1 \
python3 your_script.py
```

### macOS

```bash
DYLD_LIBRARY_PATH=./build:$DYLD_LIBRARY_PATH \
DYLD_INSERT_LIBRARIES=./build/libcublas.dylib:./build/libcudart.dylib:./build/libcuda.dylib:./build/libnvidia-ml.dylib \
python3 your_script.py
```

Mode-specific preload behavior in the Python API:

| Compute mode | Fake libraries loaded by `fakegpu.init()` / `fakegpu.env()` |
|---|---|
| `simulate` | cuBLAS + CUDA Runtime + CUDA Driver + NVML |
| `hybrid` | CUDA Runtime + CUDA Driver + NVML |
| `passthrough` | CUDA Runtime + CUDA Driver |

## Environment variables

### Compute and profiles

| Variable | Meaning |
|---|---|
| `FAKEGPU_MODE` | `simulate`, `hybrid`, or `passthrough` |
| `FAKEGPU_OOM_POLICY` | Hybrid oversubscription strategy |
| `FAKEGPU_PROFILE` | One preset ID for every fake device |
| `FAKEGPU_DEVICE_COUNT` | Number of fake devices to expose |
| `FAKEGPU_PROFILES` | Per-device preset spec such as `a100:4,h100:4` |
| `FAKEGPU_REAL_CUDA_LIB_DIR` | Override directory for real CUDA libraries |

### Distributed

| Variable | Meaning |
|---|---|
| `FAKEGPU_DIST_MODE` | `disabled`, `simulate`, `proxy`, or `passthrough` |
| `FAKEGPU_CLUSTER_CONFIG` | Cluster YAML path |
| `FAKEGPU_COORDINATOR_TRANSPORT` | `unix` or `tcp` |
| `FAKEGPU_COORDINATOR_ADDR` | Socket path or `host:port` |
| `FAKEGPU_CLUSTER_REPORT_PATH` | Output path for cluster-level JSON report |
| `FAKEGPU_STAGING_CHUNK_BYTES` | Chunk size for staged transfers |
| `FAKEGPU_STAGING_FORCE_SOCKET` | Set to `1` to skip shared memory and force socket fallback |

### Reporting and debugging

| Variable | Meaning |
|---|---|
| `FAKEGPU_REPORT_PATH` | Output path for `fake_gpu_report.json` |
| `PYTORCH_NO_CUDA_MEMORY_CACHING` | Useful when debugging allocation flow |
| `TORCH_SDPA_KERNEL=math` | Helpful for avoiding Flash Attention-specific paths |
| `CUDA_LAUNCH_BLOCKING=1` | Forces synchronous error surfacing |

## Troubleshooting

Reset a broken terminal:

```bash
reset
```

Inspect exported NVML symbols:

Linux:

```bash
nm -D ./build/libnvidia-ml.so.1 | grep ' T nvml'
```

macOS:

```bash
nm -gU ./build/libnvidia-ml.dylib | rg '\\bnvml'
```

Inspect dynamic-library dependencies:

Linux:

```bash
ldd ./build/libcuda.so.1
ldd ./build/libcudart.so.12
ldd ./build/libcublas.so.12
ldd ./build/libnvidia-ml.so.1
```

macOS:

```bash
otool -L ./build/libcuda.dylib
otool -L ./build/libcudart.dylib
otool -L ./build/libcublas.dylib
otool -L ./build/libnvidia-ml.dylib
```

## Related pages

- [Getting Started](getting-started.md)
- [Architecture](project-structure.md)
- [Reports & Validation](reports-and-validation.md)
- [Distributed Simulation Usage Guide](distributed-sim-usage.md)
