# FakeGPU

A CUDA API interception library that simulates GPU devices in non-GPU environments, enabling basic operations for PyTorch and other deep learning frameworks.

## Timeline

### Implemented Features
- [x] **CUDA Driver API** - Device management, memory allocation, kernel launch
- [x] **CUDA Runtime API** - cudaMalloc/Free, cudaMemcpy, Stream, Event
- [x] **cuBLAS/cuBLASLt** - Matrix operations (GEMM, PyTorch 2.x compatible)
- [x] **NVML API** - GPU information queries
- [x] **Python API Wrapper** - `import fakegpu; fakegpu.init()` enables FakeGPU from inside Python
- [x] **PyTorch Support** - Basic tensor ops, linear layers, neural networks
- [x] **GPU Tool Compatibility** - Compatible with existing GPU status monitoring tools (nvidia-smi, gpustat, etc.)
- [x] **Preset GPU Info** - Add more preset GPU hardware configurations

### Planned Features
- [ ] **Detailed Reporting** - More comprehensive documentation and analysis reports
- [ ] **Multi-Node GPU Communication** - Simulate cross-node GPU communication (NCCL, etc.)
- [ ] **Enhanced Testing** - Optimize test suite with more languages and runtime environments
- [ ] **Multi-Architecture & Data Types** - Support different GPU architectures and various data storage/memory types

## Quick Start

### Build

```bash
cmake -S . -B build
cmake --build build
```

Optional (enable CPU-backed compute for supported operators, default: OFF):
```bash
cmake -S . -B build -DENABLE_FAKEGPU_CPU_SIMULATION=ON
cmake --build build
```

Generated libraries:
- Linux:
  - `build/libcuda.so.1` - CUDA Driver API
  - `build/libcudart.so.12` - CUDA Runtime API
  - `build/libcublas.so.12` - cuBLAS/cuBLASLt API
  - `build/libnvidia-ml.so.1` - NVML API
- macOS:
  - `build/libcuda.dylib` - CUDA Driver API
  - `build/libcudart.dylib` - CUDA Runtime API
  - `build/libcublas.dylib` - cuBLAS/cuBLASLt API
  - `build/libnvidia-ml.dylib` - NVML API

### Test

**Standardized test runner (recommended):**
```bash
./ftest smoke          # C + Python (no torch needed)
./ftest python         # PyTorch tests (requires torch)
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
LD_PRELOAD=./build/libcublas.so.12:./build/libcudart.so.12:./build/libcuda.so.1:./build/libnvidia-ml.so.1 \
python your_script.py
```

macOS:
```bash
DYLD_LIBRARY_PATH=./build:$DYLD_LIBRARY_PATH \
DYLD_INSERT_LIBRARIES=./build/libcublas.dylib:./build/libcudart.dylib:./build/libcuda.dylib:./build/libnvidia-ml.dylib \
python3 your_script.py
```

**Python wrapper (no need to start Python with LD_PRELOAD):**
```python
import fakegpu

# Call early (before importing torch / CUDA-using libraries)
fakegpu.init()  # default: 8x A100
# Optional: fakegpu.init(profile="t4", device_count=2)
# Optional: fakegpu.init(devices="a100:4,h100:4")

import torch
```

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

**GPU tools (nvidia-smi)**
```bash
# FakeGPU-simulated devices via NVML stubs
./fgpu nvidia-smi
# Temperatures may show N/A because the TemperatureV struct is not fully emulated yet.
```

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
│   ├── nvml/          # NVML API stubs
│   └── monitor/       # Resource monitoring and reporting
└── test/              # Test scripts
```

**Core Design:**
- Uses `LD_PRELOAD` to intercept CUDA API calls
- Device memory backed by system RAM (malloc/free)
- Matrix operations return random values (no actual computation)
- Kernel launches are no-ops (logging only)

### GPU Profiles

- Default build exposes eight `Fake NVIDIA A100-SXM4-80GB` devices to mirror common server nodes.
- GPU parameters are edited in YAML under `profiles/*.yaml`; CMake embeds these files at build time so no runtime file lookup is needed. Add or tweak a file, rerun `cmake -S . -B build`, and the new profiles are compiled in.
- Presets cover multiple compute capabilities (Maxwell→Blackwell) and feed the existing helpers (`GpuProfile::GTX980/P100/V100/T4/A40/A100/H100/L40S/B100/B200`), which now prefer the YAML data and fall back to code defaults if parsing fails.
- Select presets at runtime via environment variables:
  - `FAKEGPU_PROFILE=<id>` + `FAKEGPU_DEVICE_COUNT=<n>` (uniform devices)
  - `FAKEGPU_PROFILES=<spec>` (per-device spec, e.g. `a100:4,h100:4` or `t4,l40s`)
- Python wrapper passes the same settings (must be called before importing CUDA-using libs like torch): `fakegpu.init(profile="t4", device_count=2)` or `fakegpu.init(devices="a100:4,h100:4")`.

## Limitations

- ❌ No real GPU computation (kernels are no-ops)
- ❌ Complex models (Transformers) may require additional APIs
- ❌ No multi-GPU synchronization
- ⚠️ macOS: Official PyTorch wheels do not include CUDA, so FakeGPU only helps when running CUDA-enabled binaries (typically in Linux via Docker/VM).
- ⚠️ For testing and development environments only

## Use Cases

- ✅ Running GPU code tests in CI/CD environments
- ✅ Debugging deep learning code on machines without GPUs
- ✅ Validating CUDA API call logic
- ✅ Prototyping and unit testing

## Dependencies

- CMake 3.14+
- C++17 compiler
- Python 3.8+ (for testing)
- PyTorch 2.x (optional, for testing)

## License

MIT License

## Documentation

- [Test Guide](test/README.md) - Detailed testing instructions
- [cuBLASLt Implementation](docs/cublaslt-fix.md) - cuBLASLt support details
