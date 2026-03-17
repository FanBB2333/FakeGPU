# cuBLASLt 兼容性说明

这页记录的是最常见的 cuBLASLt 启动问题，以及在 PyTorch 线性层 / matmul 调试时真正需要关心的运行时行为。

## 常见报错现象

如果你手动 preload 了 FakeGPU 库，但漏掉了 `libcublas`，PyTorch 2.x 很容易出现类似下面的错误：

```text
CUBLAS_STATUS_NOT_SUPPORTED
```

常见触发点是 `cublasLtMatmulAlgoGetHeuristic(...)` 一类线性层初始化路径。

## 为什么会这样

核心原因通常有两点：

1. PyTorch 2.x 依赖的是 cuBLASLt，而不只是旧版 cuBLAS API。
2. 如果你想走 FakeGPU 的 cuBLAS / cuBLASLt 路径，手动 preload 时必须把 FakeGPU 的 `libcublas` 一起带上。

## 推荐启动方式

优先使用包装器：

```bash
./fgpu python3 your_script.py
```

这样最不容易把 preload 顺序或库列表写错。

## 手动 preload 命令

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

## 不同计算模式下的行为

| 模式 | cuBLAS / cuBLASLt 来源 |
|---|---|
| `simulate` | 使用 FakeGPU `libcublas`；已维护路径可以走 CPU-backed math |
| `hybrid` | 使用真实 cuBLAS / cuBLASLt，同时保留 FakeGPU 的设备虚拟化与报告 |
| `passthrough` | 使用真实 cuBLAS / cuBLASLt，FakeGPU 尽量少介入 |

## 当前已经覆盖的内容

当前维护的 CPU simulation 验证包括：

- `cublasSgemm_v2`
- 常见 `cublasLtMatmul`
- device pointer mode 检查
- strided batched GEMM
- batched GEMM
- 若干 BLAS1 操作

这足以覆盖基础 PyTorch tensor / linear / matmul smoke path，但并不意味着所有高级 kernel 路径、所有模型都已经完整支持。

## 调试建议

1. 先跑 `./ftest cpu_sim`，确认当前构建下维护的 FakeGPU math path 是通的。
2. 优先使用 `./fgpu`，只有在确实需要时再手动 preload。
3. 如果某个工作负载只在 `simulate` 下失败，可以切到 `hybrid`，先把“fake 设备问题”和“fake cuBLAS 问题”拆开。
4. 遇到框架特定问题时，最好用同一份脚本分别在真实 GPU 和 FakeGPU 下做对照。
