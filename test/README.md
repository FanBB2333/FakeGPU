# FakeGPU 测试脚本

## 当前状态

FakeGPU项目实现了完整的CUDA Driver API、CUDA Runtime API、NVML API和cuBLAS API拦截功能。PyTorch和Transformers已完全支持。

### 已实现的功能

- **CUDA Driver API** (libcuda.so.1):
  - 设备管理: cuInit, cuDeviceGet, cuDeviceGetCount, cuDeviceGetName, cuDeviceGetAttribute等
  - 上下文管理: cuCtxCreate, cuCtxSetCurrent, cuDevicePrimaryCtxRetain等
  - 内存管理: cuMemAlloc, cuMemFree, cuMemcpy, cuMemGetInfo等
  - 流和事件: cuStreamCreate, cuEventCreate等
  - 模块和函数: cuModuleLoad, cuLaunchKernel等
  - 动态符号查找: cuGetProcAddress (关键函数)

- **CUDA Runtime API** (libcudart.so.12):
  - 基础API: cudaMalloc, cudaFree, cudaMemcpy, cudaMemset等
  - 设备管理: cudaGetDeviceCount, cudaSetDevice, cudaGetDeviceProperties等
  - 异步操作: cudaMallocAsync, cudaFreeAsync, cudaMemcpyAsync等
  - 流管理: cudaStreamCreate, cudaStreamDestroy, cudaStreamSynchronize等
  - 事件管理: cudaEventCreate, cudaEventRecord, cudaEventElapsedTime等
  - 内存池: cudaMemPoolCreate, cudaMallocFromPoolAsync等 (10+ 函数)
  - CUDA Graph: cudaGraphCreate, cudaGraphInstantiate, cudaGraphLaunch等 (30+ 函数)
  - 纹理/表面: cudaCreateTextureObject, cudaCreateSurfaceObject等
  - 协作组: cudaLaunchCooperativeKernel等
  - 内部注册: __cudaRegisterFunction, __cudaRegisterVar等

- **cuBLAS API** (libcublas.so.12):
  - 句柄管理: cublasCreate, cublasDestroy, cublasSetStream等
  - BLAS Level 1: 向量操作 (dot, axpy, scal, nrm2等)
  - BLAS Level 2: 矩阵向量操作 (gemv, ger等)
  - BLAS Level 3: 矩阵矩阵操作 (gemm, trsm, symm等)
  - 批处理操作: gemmStridedBatched, gemmBatched等
  - 混合精度: GemmEx, SgemmEx等
  - 复数运算: Cgemm, Zgemm, Cdotu, Zdotu等
  - 线性代数: getrfBatched, getrsBatched, geqrfBatched, gelsBatched等
  - **总计**: 100+ cuBLAS函数已实现

- **NVML API** (libnvidia-ml.so.1):
  - nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetMemoryInfo等

### 支持的框架

- ✅ PyTorch 2.x (完全支持 - 包括矩阵运算)
- ✅ Transformers (完全支持 - 可用于模型加载和推理)
- ✅ 其他基于CUDA的深度学习框架

### 当前功能

FakeGPU + cuBLAS实现了完整的stub函数集合，可以成功:
- ✅ 检测GPU设备和属性
- ✅ 分配和管理内存
- ✅ 加载模型和权重
- ✅ 执行张量操作（element-wise和矩阵运算）
- ✅ 记录资源消耗模式

**重要**: FakeGPU返回随机值用于所有计算结果，不执行实际GPU计算。这是设计上的预期行为，主要用途是:
1. 在无GPU环境下测试模型加载和设备管理代码
2. 记录和分析硬件资源消耗模式
3. 指导未来的硬件资源分配决策
4. 开发和调试不需要实际计算结果的工具

如果需要实际的计算结果，请使用真实GPU或CPU模式。

## 测试脚本说明

### 1. CUDA Driver API直接测试 (test_cuda_direct.py) - 推荐

直接测试FakeGPU的CUDA Driver API实现:

```bash
python test/test_cuda_direct.py
```

测试内容:
- cuInit, cuDriverGetVersion
- cuDeviceGetCount, cuDeviceGet, cuDeviceGetName
- cuDeviceTotalMem, cuDevicePrimaryCtxRetain
- cuMemAlloc, cuMemFree, cuMemGetInfo

这个测试可以验证FakeGPU的核心功能是否正常工作。

### 2. PyTorch基础测试 (test_pytorch_basic.py)

测试PyTorch的基本CUDA功能:

```bash
# 使用正确的库路径和预加载顺序
LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH \
LD_PRELOAD=./build/libcudart.so.12:./build/libcuda.so.1:./build/libnvidia-ml.so.1 \
python test/test_pytorch_basic.py
```

预期输出:
- PyTorch成功检测到8个虚拟GPU (默认配置)
- 每个GPU显示为 "Fake NVIDIA A100-SXM4-80GB"
- 80GB显存容量
- 计算能力8.0 (Ampere架构)
- 成功创建张量并进行基本运算

测试内容:
- torch.cuda.is_available()
- torch.cuda.device_count()
- torch.cuda.get_device_properties()
- 张量创建和运算
- CPU-GPU数据传输
- 内存分配和追踪

### 3. Transformers测试 (test_transformers.py / test_transformers_simple.py)

测试transformers库的基本功能:

```bash
# 简单测试
LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH \
LD_PRELOAD=./build/libcudart.so.12:./build/libcuda.so.1:./build/libnvidia-ml.so.1 \
python test/test_transformers_simple.py --epochs 1 --batch-size 2 --num-samples 20
```

参数:
- `--epochs`: 训练轮数 (默认: 2)
- `--batch-size`: 批次大小 (默认: 4)
- `--lr`: 学习率 (默认: 5e-5)
- `--num-samples`: 数据集样本数 (默认: 100)
- `--log-interval`: 日志打印间隔 (默认: 10)

### 4. Transformers DDP测试 (test_transformers.py)

完整的DDP多卡并行训练测试:

单卡模式:
```bash
./test/run_transformers_test.sh
```

DDP多卡模式:
```bash
./test/run_transformers_test.sh --ddp 2
```

## 依赖安装

```bash
pip install torch transformers
```

## 预期输出

成功运行时会看到:
- FakeGPU的初始化日志
- CUDA API调用日志
- 训练进度信息
- 内存使用统计
- 生成的`fake_gpu_report.json`报告

## 故障排查

### 重要提示

FakeGPU现在完全支持PyTorch。所有CUDA Runtime API函数已实现为stub函数。

### 正确的环境变量设置

必须按以下顺序设置:

```bash
# 1. 库搜索路径 - 确保找到fake库
LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH

# 2. 预加载顺序 - 非常重要!
# 顺序: libcudart.so.12 -> libcuda.so.1 -> libnvidia-ml.so.1
LD_PRELOAD=./build/libcudart.so.12:./build/libcuda.so.1:./build/libnvidia-ml.so.1
```

预加载顺序说明:
- `libcudart.so.12` 必须首先加载 (PyTorch直接依赖)
- `libcuda.so.1` 其次 (Runtime API内部调用Driver API)
- `libnvidia-ml.so.1` 最后 (监控和管理功能)

### 常见问题

1. **"undefined symbol" 错误**
   - 确保已重新编译: `cmake --build build`
   - 检查是否有新的CUDA API被调用
   - 使用 `nm -D build/libcudart.so.12 | grep <symbol>` 检查符号

2. **PyTorch无法检测GPU**
   - 检查LD_PRELOAD顺序是否正确
   - 确认libcudart.so.12在预加载列表的第一位
   - 查看FakeGPU日志确认API调用

3. **段错误或崩溃**
   - 可能是某些stub函数参数处理不当
   - 检查FakeGPU的调试输出
   - 使用gdb调试: `gdb --args python test/test_pytorch_basic.py`

4. **检查基础环境**
   - 确保已构建项目: `cmake -S . -B build && cmake --build build`
   - 检查库文件: `ls -l build/libcudart.so.12 build/libcuda.so.1 build/libnvidia-ml.so.1`
   - 确保已安装依赖: `pip install torch transformers`
   - 查看详细日志以了解具体错误信息

### 调试工具

检查缺失的CUDA符号:
```bash
# 查看PyTorch依赖的CUDA符号
./find_missing_symbols.sh

# 查看特定库的符号
nm -D build/libcudart.so.12 | grep cudaMalloc
```

## 实现细节

### CUDA Runtime API Stub实现

所有CUDA Runtime API函数都实现为stub:
- 内存分配函数(cudaMalloc等)使用系统malloc
- 内核启动函数(cudaLaunchKernel等)仅打印日志，不执行实际计算
- 流和事件函数调用对应的Driver API
- 错误处理使用线程局部变量追踪

### 已实现的关键函数

总计200+个CUDA Runtime API函数，包括:
- 设备管理 (20+ 函数)
- 内存管理 (30+ 函数)
- 流管理 (15+ 函数)
- 事件管理 (10+ 函数)
- 内存池管理 (10+ 函数)
- CUDA Graph API (30+ 函数)
- 纹理/表面API (10+ 函数)
- 内核启动 (10+ 函数)
- 内部注册函数 (10+ 函数)

详细列表见 [src/cuda/cudart_defs.hpp](../src/cuda/cudart_defs.hpp)

## 开发建议

### 扩展FakeGPU功能

当前实现已满足PyTorch基本需求。如需进一步扩展:

1. **添加更多CUDA API**
   - 参考NVIDIA CUDA文档
   - 在cudart_defs.hpp中添加函数声明
   - 在cudart_stubs.cpp中实现stub
   - 大多数函数只需简单stub，返回成功状态

2. **改进内存管理**
   - 当前使用系统malloc，可以添加更精确的内存追踪
   - 实现真实的显存限制检查
   - 添加内存池的实际管理逻辑

3. **添加性能分析**
   - 追踪API调用频率
   - 记录内存分配模式
   - 生成详细的性能报告

4. **调试支持**
   - 使用环境变量控制日志级别
   - 添加更详细的函数调用追踪
   - 实现API调用时序图生成

### 代码贡献

欢迎贡献:
- 新的测试用例
- Bug修复
- 性能优化
- 文档改进

## 快速参考

### PyTorch测试一键命令

```bash
# 构建项目
cmake -S . -B build && cmake --build build

# 测试PyTorch基础功能
LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH \
LD_PRELOAD=./build/libcudart.so.12:./build/libcuda.so.1:./build/libnvidia-ml.so.1 \
python test/test_pytorch_basic.py
```

### 环境变量模板

添加到 ~/.bashrc 或测试脚本:

```bash
export FAKEGPU_ROOT=/path/to/fakeGPU
export LD_LIBRARY_PATH=$FAKEGPU_ROOT/build:$LD_LIBRARY_PATH
export LD_PRELOAD=$FAKEGPU_ROOT/build/libcudart.so.12:$FAKEGPU_ROOT/build/libcuda.so.1:$FAKEGPU_ROOT/build/libnvidia-ml.so.1
```
