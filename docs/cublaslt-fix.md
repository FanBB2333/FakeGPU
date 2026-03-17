# cuBLASLt Compatibility Notes

This page records the most common cuBLASLt bring-up issue and the runtime behavior that matters when debugging PyTorch linear and matmul paths.

## Typical failure symptom

If you manually preload FakeGPU libraries but omit `libcublas`, PyTorch 2.x can fail with errors such as:

```text
CUBLAS_STATUS_NOT_SUPPORTED
```

That usually appears around `cublasLtMatmulAlgoGetHeuristic(...)` or nearby linear-layer setup.

## Why it happens

Two conditions matter:

1. PyTorch 2.x expects cuBLASLt, not just the older cuBLAS surface.
2. Manual preload setups must include the FakeGPU `libcublas` library when you want FakeGPU's cuBLAS/cuBLASLt path.

## Recommended launch path

Prefer the wrapper:

```bash
./fgpu python3 your_script.py
```

It keeps the preload order correct and avoids the most common manual mistakes.

## Manual preload commands

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

## Behavior by compute mode

| Mode | cuBLAS/cuBLASLt source |
|---|---|
| `simulate` | FakeGPU `libcublas` with maintained CPU-backed math for supported paths |
| `hybrid` | Real cuBLAS/cuBLASLt, while FakeGPU still virtualizes device identity and reporting |
| `passthrough` | Real cuBLAS/cuBLASLt with minimal FakeGPU interference |

## What is currently covered

The maintained CPU-simulation validation includes:

- `cublasSgemm_v2`
- `cublasLtMatmul` for common matmul paths
- device-pointer mode checks
- strided batched GEMM
- batched GEMM
- several BLAS1 operations

That coverage is enough for basic PyTorch tensor, linear, and matmul smoke paths, but not a promise that every advanced kernel path used by every model is already implemented.

## Debugging checklist

1. Start with `./ftest cpu_sim` to confirm the maintained FakeGPU math path works in your build.
2. Use `./fgpu` before falling back to manual preload.
3. If a workload only fails in `simulate`, try `hybrid` to separate fake-device concerns from fake-cuBLAS concerns.
4. For framework-specific issues, compare a real-GPU run and a FakeGPU run with the same user script.
