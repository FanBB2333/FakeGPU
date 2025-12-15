# Using FakeGPU with CUDA Runtime API

## Overview

FakeGPU now provides **three** fake libraries:

1. **libcudart.so.12** - CUDA Runtime API (cuda* functions) - **NEW!**
2. **libcuda.so.1** - CUDA Driver API (cu* functions)
3. **libnvidia-ml.so.1** - NVML API (nvml* functions)

## Quick Start

### Basic Test

Test the CUDA Runtime API directly:

```bash
python3 test/test_cudart_basic.py
```

### Using with PyTorch (Partial Support)

**Note**: PyTorch requires many additional CUDA Runtime functions that are not yet implemented. Basic functionality works, but full PyTorch support requires implementing more functions (see TODOs.md).

```bash
# Preload all three libraries
LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH \
LD_PRELOAD=./build/libcudart.so.12:./build/libcuda.so.1:./build/libnvidia-ml.so.1 \
python your_script.py
```

## What's Implemented

### Core Functions (✅ Working)

**Device Management**:
- `cudaGetDeviceCount()` - Get number of devices
- `cudaSetDevice()` - Set current device
- `cudaGetDevice()` - Get current device
- `cudaGetDeviceProperties()` - Get device properties
- `cudaDeviceReset()` - Reset device
- `cudaDeviceSynchronize()` - Synchronize device

**Memory Management**:
- `cudaMalloc()` - Allocate device memory
- `cudaFree()` - Free device memory
- `cudaMemcpy()` - Copy memory (all directions)
- `cudaMemcpyAsync()` - Async memory copy (simplified)
- `cudaMemset()` - Set memory
- `cudaMemsetAsync()` - Async memset (simplified)
- `cudaMemGetInfo()` - Get memory info

**Stream Management**:
- `cudaStreamCreate()` - Create stream
- `cudaStreamDestroy()` - Destroy stream
- `cudaStreamSynchronize()` - Synchronize stream
- `cudaStreamQuery()` - Query stream status

**Event Management**:
- `cudaEventCreate()` - Create event
- `cudaEventDestroy()` - Destroy event
- `cudaEventRecord()` - Record event
- `cudaEventSynchronize()` - Synchronize event
- `cudaEventQuery()` - Query event
- `cudaEventElapsedTime()` - Get elapsed time

**Error Handling**:
- `cudaGetLastError()` - Get and clear last error
- `cudaPeekAtLastError()` - Get last error without clearing
- `cudaGetErrorString()` - Get error description
- `cudaGetErrorName()` - Get error name

**Version Info**:
- `cudaRuntimeGetVersion()` - Get runtime version (returns 12000)
- `cudaDriverGetVersion()` - Get driver version (returns 12000)

**Kernel Launch** (stubs - no actual computation):
- `cudaLaunchKernel()` - Launch kernel (no-op)
- `cudaConfigureCall()` - Configure call (no-op)
- `cudaSetupArgument()` - Setup argument (no-op)
- `cudaLaunch()` - Launch (no-op)

**Host Memory**:
- `cudaMallocHost()` - Allocate pinned host memory
- `cudaFreeHost()` - Free pinned host memory
- `cudaHostAlloc()` - Allocate host memory with flags

**Peer Access**:
- `cudaDeviceCanAccessPeer()` - Check peer access
- `cudaDeviceEnablePeerAccess()` - Enable peer access
- `cudaDeviceDisablePeerAccess()` - Disable peer access

**Driver Entry Point (CUDA 12+)**:
- `cudaGetDriverEntryPointByVersion()` - Get driver function pointer
- `cudaGetDriverEntryPoint()` - Get driver function pointer

## What's NOT Implemented

PyTorch requires many additional functions that are not yet implemented:

- CUDA Graph API (`cudaGraphCreate`, `cudaGraphLaunch`, etc.)
- Texture/Surface API
- Advanced stream features
- Many other specialized functions

See [TODOs.md](TODOs.md) for the full list and implementation plan.

## Architecture

```
Your Application (e.g., PyTorch)
    ↓
libcudart.so.12 (Fake CUDA Runtime) ← You are here!
    ↓
libcuda.so.1 (Fake CUDA Driver)
    ↓
Fake GPU (simulated with system RAM)
```

The Runtime API (`cudart_stubs.cpp`) internally calls the Driver API (`cuda_driver_stubs.cpp`), which manages the fake GPU devices.

## Implementation Details

### Memory Management

- All "device" memory is actually system RAM allocated with `malloc()`
- Memory allocations are tracked in `GlobalState`
- Each device has a configurable memory limit (default: 16GB)

### Kernel Execution

- Kernel launches are **no-ops** (no actual computation)
- This is sufficient for:
  - Testing memory management
  - Testing data flow
  - Debugging GPU-dependent code structure
- This is **NOT** sufficient for:
  - Actual training/inference
  - Performance testing
  - Correctness testing of computations

### Error Handling

- Thread-local error tracking
- Errors are converted from Driver API (`CUresult`) to Runtime API (`cudaError_t`)
- `cudaGetLastError()` clears the error
- `cudaPeekAtLastError()` does not clear the error

## Examples

### Example 1: Basic Memory Operations

```python
import ctypes

# Load library
cudart = ctypes.CDLL('./build/libcudart.so.12')

# Get device count
count = ctypes.c_int()
cudart.cudaGetDeviceCount(ctypes.byref(count))
print(f"Devices: {count.value}")

# Allocate memory
ptr = ctypes.c_void_p()
cudart.cudaMalloc(ctypes.byref(ptr), 1024)
print(f"Allocated at: {hex(ptr.value)}")

# Free memory
cudart.cudaFree(ptr)
```

### Example 2: Using with PyTorch (Limited)

```python
import torch

# This will work:
print(torch.cuda.is_available())  # True
print(torch.cuda.device_count())  # 8

# This may fail due to missing functions:
# torch.cuda.set_device(0)  # May fail
# x = torch.randn(10, 10, device='cuda')  # May fail
```

## Troubleshooting

### "undefined symbol: cudaXXX"

This means PyTorch is trying to use a function that's not implemented yet. Options:

1. Implement the missing function in `cudart_stubs.cpp`
2. Use a simpler test that doesn't require that function
3. Check [TODOs.md](TODOs.md) for the implementation plan

### "CUDA error: initialization error"

This usually means the Driver API initialization failed. Check:

1. All three libraries are preloaded
2. `LD_LIBRARY_PATH` includes `./build`
3. Libraries are built successfully

### Segmentation Fault

This might happen if:

1. A function returns NULL when it shouldn't
2. A required function is missing
3. Memory corruption in the fake implementation

Enable debug logging by uncommenting `printf` statements in the source code.

## Performance

**Note**: This is a **simulation**, not an emulation. There is:

- ❌ No actual GPU computation
- ❌ No parallelism
- ❌ No GPU-specific optimizations
- ✅ Memory tracking
- ✅ API compatibility (partial)
- ✅ Useful for debugging and testing code structure

## Next Steps

To add full PyTorch support, see the implementation plan in [TODOs.md](TODOs.md), specifically:

- **Phase 1**: Implement core functions for PyTorch initialization (30 functions)
- **Phase 2**: Implement tensor operations (50 functions)
- **Phase 3**: Implement training support (20+ functions)

## Contributing

When adding new functions:

1. Add declaration to `src/cuda/cudart_defs.hpp`
2. Implement in `src/cuda/cudart_stubs.cpp`
3. Call corresponding Driver API functions when possible
4. Add test in `test/test_cudart_basic.py`
5. Update this README

## License

Same as the main FakeGPU project.
