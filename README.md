# FakeGPU

A CUDA API interception library that simulates GPU devices in non-GPU environments, enabling basic operations for PyTorch and other deep learning frameworks.

## Timeline

### Implemented Features
- [x] **CUDA Driver API** - Device management, memory allocation, kernel launch
- [x] **CUDA Runtime API** - cudaMalloc/Free, cudaMemcpy, Stream, Event
- [x] **cuBLAS/cuBLASLt** - Matrix operations (GEMM, PyTorch 2.x compatible)
- [x] **NVML API** - GPU information queries
- [x] **PyTorch Support** - Basic tensor ops, linear layers, neural networks
- [x] **GPU Tool Compatibility** - Compatible with existing GPU status monitoring tools (nvidia-smi, gpustat, etc.)

### Planned Features
- [ ] **Python API Wrapper** - Package as Python library for easier integration
- [ ] **Detailed Reporting** - More comprehensive documentation and analysis reports
- [ ] **Multi-Node GPU Communication** - Simulate cross-node GPU communication (NCCL, etc.)
- [ ] **Enhanced Testing** - Optimize test suite with more languages and runtime environments
- [ ] **Preset GPU Info** - Add more preset GPU hardware configurations
- [ ] **Multi-Architecture & Data Types** - Support different GPU architectures and various data storage/memory types

## Quick Start

### Build

```bash
cmake -S . -B build
cmake --build build
```

Generated libraries:
- `build/libcuda.so.1` - CUDA Driver API
- `build/libcudart.so.12` - CUDA Runtime API
- `build/libcublas.so.12` - cuBLAS/cuBLASLt API
- `build/libnvidia-ml.so.1` - NVML API

### Test

**Comparison test (recommended):**
```bash
./test/run_comparison.sh
```
Runs identical tests on both real GPU and FakeGPU to verify correctness.

**PyTorch test:**
```bash
LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH \
LD_PRELOAD=./build/libcublas.so.12:./build/libcudart.so.12:./build/libcuda.so.1:./build/libnvidia-ml.so.1 \
python3 test/test_comparison.py --mode fake
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
```bash
LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH \
LD_PRELOAD=./build/libcublas.so.12:./build/libcudart.so.12:./build/libcuda.so.1:./build/libnvidia-ml.so.1 \
python your_script.py
```

**Shortcut runner:**
```bash
./fgpu python your_script.py
# Optional: FAKEGPU_BUILD_DIR=/path/to/build ./fgpu python your_script.py
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

## Limitations

- ❌ No real GPU computation (kernels are no-ops)
- ❌ Complex models (Transformers) may require additional APIs
- ❌ No multi-GPU synchronization
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
