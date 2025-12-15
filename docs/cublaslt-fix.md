# FakeGPU cuBLASLt Support - Problem Resolution

## 问题描述

执行以下命令时报错：
```bash
LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH \
LD_PRELOAD=./build/libcudart.so.12:./build/libcuda.so.1:./build/libnvidia-ml.so.1 \
python test/test_transformers.py
```

报错信息：
```
RuntimeError: CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling
`cublasLtMatmulAlgoGetHeuristic(...)`
```

## 根本原因

1. **缺少 libcublas.so.12 预加载**
   - 原始命令没有包含 `libcublas.so.12`
   - PyTorch 的线性层需要 cuBLAS 库支持

2. **缺少 cuBLASLt API 实现**
   - PyTorch 2.x 使用现代的 cuBLASLt API 而非传统 cuBLAS API
   - FakeGPU 之前只实现了传统的 cuBLAS API
   - 缺少关键函数 `cublasLtMatmulAlgoGetHeuristic()`

## 解决方案

### 1. 添加 cuBLASLt API 定义

在 `src/cublas/cublas_defs.hpp` 中添加：
- cuBLASLt 句柄类型定义
- 描述符类型定义
- 矩阵布局类型定义
- 算法选择相关函数声明

### 2. 实现 cuBLASLt Stubs

在 `src/cublas/cublas_stubs.cpp` 中实现：
- `cublasLtCreate/Destroy` - 句柄管理
- `cublasLtMatmulDescCreate/Destroy` - 描述符管理
- `cublasLtMatrixLayoutCreate/Destroy` - 矩阵布局管理
- `cublasLtMatmulPreferenceCreate/Destroy` - 偏好设置管理
- `cublasLtMatmulAlgoGetHeuristic` - **关键函数**，返回算法建议
- `cublasLtMatmul` - 矩阵乘法执行

### 3. 正确的预加载命令

```bash
LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH \
LD_PRELOAD=./build/libcublas.so.12:./build/libcudart.so.12:./build/libcuda.so.1:./build/libnvidia-ml.so.1 \
python test/your_test.py
```

注意顺序：
1. `libcublas.so.12` - cuBLAS/cuBLASLt 库
2. `libcudart.so.12` - CUDA Runtime 库
3. `libcuda.so.1` - CUDA Driver 库
4. `libnvidia-ml.so.1` - NVML 库

## 测试结果

### 真实 GPU 测试
```
Real GPU: 6/6 tests passed
  ✓ Tensor creation
  ✓ Element-wise operations
  ✓ Basic matmul
  ✓ Linear layer
  ✓ Model forward
  ✓ Memory transfer
```

### FakeGPU 测试（修复后）
```
FakeGPU: 6/6 tests passed
  ✓ Tensor creation
  ✓ Element-wise operations
  ✓ Basic matmul
  ✓ Linear layer          ← 之前失败，现在通过
  ✓ Model forward         ← 之前失败，现在通过
  ✓ Memory transfer
```

## 重要说明

1. **测试代码没有问题** - 在真实 GPU 上 6/6 通过
2. **环境配置正确** - PyTorch 和依赖库都正常工作
3. **之前的失败是因为 FakeGPU 缺少 cuBLASLt 支持**

## 使用建议

### 快速测试
```bash
# 对比测试（真实GPU vs FakeGPU）
./test/run_comparison.sh

# 仅测试 FakeGPU
python3 test/test_comparison.py --mode fake
```

### 开发新功能
如果遇到新的错误：
1. 先在真实 GPU 上测试，确认代码正确
2. 在 FakeGPU 上测试，定位缺失的 API
3. 实现缺失的 API stubs

## 实现的关键点

### cublasLtMatmulAlgoGetHeuristic
这是最关键的函数，PyTorch 用它来选择矩阵乘法算法：

```cpp
cublasStatus_t cublasLtMatmulAlgoGetHeuristic(...) {
    // 返回一个默认算法
    if (algoCount > 0) {
        heuristicResultsArray[0].algo = 0;
        heuristicResultsArray[0].workspaceSize = 0;
        heuristicResultsArray[0].state = 0;
        heuristicResultsArray[0].wavesCount = 1.0f;
    }
    *returnAlgoCount = algoCount;
    return CUBLAS_STATUS_SUCCESS;
}
```

## 下一步

虽然基本的 PyTorch 操作现在可以工作，但对于更复杂的模型（如 transformers），
可能还需要实现：

1. 更多的 cuBLASLt 高级功能
2. Flash Attention 相关的 kernel stubs
3. 其他深度学习库的特定 API

可以使用相同的方法：
1. 在真实 GPU 测试
2. 定位错误的 API
3. 添加 stub 实现
