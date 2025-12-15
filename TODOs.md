# FakeGPU TODOs and Known Issues

## 问题分析：为什么 PyTorch 无法使用 FakeGPU

### 现状

- ✅ **test_cuda_direct.py 可以正常工作**：直接使用 ctypes 加载 `libcuda.so.1`，所有调用都进入 fake 实现
- ❌ **test_transformers.py 失败**：PyTorch 调用 `torch.cuda.set_device(0)` 时报错 `cudaErrorInitializationError`

### 根本原因

PyTorch 使用**真实的 libcudart.so.12**（CUDA Runtime），而不是直接调用 Driver API：

```
PyTorch
  ↓
真实的 libcudart.so.12 (NVIDIA CUDA Runtime)
  ↓
libcuda.so.1 (我们的 fake Driver) ← 通过 LD_PRELOAD 拦截
```

**问题**：真实的 `libcudart.so.12` 在初始化时会：
1. 调用 `cuGetExportTable` 获取 NVIDIA 内部函数表（未公开的 API）
2. 我们的 fake 实现返回 `CUDA_ERROR_NOT_INITIALIZED`（返回 NULL 会导致 segfault）
3. 真实的 Runtime 检测到驱动无效，返回 `cudaErrorInitializationError`

**核心矛盾**：真实的 CUDA Runtime 需要与真实的 CUDA Driver 配合，无法与 fake Driver 一起工作。

### 测试对比

| 测试程序 | 加载方式 | CUDA Runtime | CUDA Driver | 结果 |
|---------|---------|-------------|------------|------|
| test_cuda_direct.py | ctypes.CDLL 直接加载 | 不使用 | fake (libcuda.so.1) | ✅ 成功 |
| test_transformers.py | PyTorch 导入 | 真实 (libcudart.so.12) | fake (libcuda.so.1) | ❌ 失败 |

## 解决方案

### 方案 1：创建 fake libcudart.so.12（推荐）

**目标**：创建一个 fake CUDA Runtime 库，替代真实的 `libcudart.so.12`

**优点**：
- 完全控制 CUDA Runtime 和 Driver API
- 不依赖真实的 NVIDIA 库
- 可以支持 PyTorch 和其他深度学习框架

**实现步骤**：

1. **创建 CUDA Runtime stubs**
   ```bash
   src/cuda/cudart_stubs.cpp      # 实现 CUDA Runtime API
   src/cuda/cudart_defs.hpp       # CUDA Runtime API 定义
   ```

2. **需要实现的主要函数**：
   - 设备管理：`cudaSetDevice`, `cudaGetDevice`, `cudaGetDeviceCount`, `cudaGetDeviceProperties`
   - 内存管理：`cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaMemcpyAsync`, `cudaMemset`
   - 流管理：`cudaStreamCreate`, `cudaStreamDestroy`, `cudaStreamSynchronize`
   - 事件管理：`cudaEventCreate`, `cudaEventDestroy`, `cudaEventRecord`, `cudaEventSynchronize`
   - 错误处理：`cudaGetLastError`, `cudaPeekAtLastError`, `cudaGetErrorString`
   - 版本信息：`cudaRuntimeGetVersion`, `cudaDriverGetVersion`
   - 内核启动：`cudaLaunchKernel`, `cudaConfigureCall`, `cudaSetupArgument`
   - 同步：`cudaDeviceSynchronize`, `cudaThreadSynchronize`

3. **修改 CMakeLists.txt**
   ```cmake
   # 添加新的库目标
   add_library(fake_cudart SHARED
       src/cuda/cudart_stubs.cpp
   )
   set_target_properties(fake_cudart PROPERTIES
       OUTPUT_NAME "cudart"
       VERSION "12.0.0"
       SOVERSION "12"
   )
   ```

4. **使用方式**
   ```bash
   # 同时 preload Runtime 和 Driver
   LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH \
   LD_PRELOAD=./build/libcudart.so.12:./build/libcuda.so.1:./build/libnvidia-ml.so.1 \
   python test/test_transformers.py
   ```

**工作量估计**：中等（需要实现约 100+ 个 Runtime API 函数）

**优先级**：高

---

### 方案 2：研究 cuGetExportTable 内部结构（困难）

**目标**：逆向工程 `cuGetExportTable` 的返回结构，提供一个有效的内部函数表

**挑战**：
- `cuGetExportTable` 是未公开的内部 API
- 需要逆向分析真实的 `libcudart.so.12` 和 `libcuda.so.1`
- 内部结构可能在不同 CUDA 版本间变化
- 需要提供大量内部函数指针

**实现步骤**：
1. 使用 IDA Pro / Ghidra 逆向分析 `libcudart.so.12`
2. 找到 `cuGetExportTable` 的调用点和返回结构
3. 分析内部函数表的布局和必需的函数指针
4. 在 fake Driver 中构造兼容的函数表

**工作量估计**：大（需要深入逆向工程）

**优先级**：低（不推荐，维护成本高）

---

### 方案 3：使用 CPU 模式（临时方案）

**目标**：如果只是为了测试训练逻辑，使用 CPU 模式

**实现**：
```python
# 在 test_transformers.py 中
device = torch.device('cpu')  # 不使用 CUDA
```

**优点**：
- 简单快速
- 不需要修改 FakeGPU

**缺点**：
- 无法测试 GPU 相关功能
- 性能较慢

**优先级**：低（仅作为临时方案）

---

### 方案 4：拦截 PyTorch 的 CUDA 初始化（实验性）

**目标**：在 Python 层面拦截 PyTorch 的 CUDA 初始化，绕过真实的 Runtime

**实现思路**：
```python
# 使用 ctypes 替换 torch._C._cuda_init
import ctypes
fake_cuda = ctypes.CDLL('./build/libcuda.so.1')
# 手动初始化...
```

**挑战**：
- 需要深入了解 PyTorch 内部结构
- 可能需要 monkey-patch 大量函数
- 维护成本高

**优先级**：低（实验性质）

---

## 当前实现状态

### 已完成 (2025-12-15)

#### CUDA Runtime API (libcudart.so.12) - ✅ 已实现
- ✅ **独立的 libcudart.so.12 库**：成功创建并编译
- ✅ **设备管理**：cudaGetDeviceCount, cudaSetDevice, cudaGetDevice, cudaGetDeviceProperties (20+ 函数)
- ✅ **内存管理**：cudaMalloc, cudaFree, cudaMallocAsync, cudaMemcpy 等 (30+ 函数)
- ✅ **流管理**：cudaStreamCreate, cudaStreamDestroy, cudaStreamSynchronize 等 (15+ 函数)
- ✅ **事件管理**：cudaEventCreate, cudaEventRecord, cudaEventElapsedTime 等 (10+ 函数)
- ✅ **内存池**：cudaMemPoolCreate, cudaMallocFromPoolAsync 等 (10+ 函数)
- ✅ **CUDA Graph API**：cudaGraphCreate, cudaGraphInstantiate, cudaGraphLaunch 等 (30+ 函数)
- ✅ **纹理/表面**：cudaCreateTextureObject, cudaCreateSurfaceObject 等 (10+ 函数)
- ✅ **协作组**：cudaLaunchCooperativeKernel 等
- ✅ **内部注册函数**：__cudaRegisterFunction, __cudaRegisterVar, __cudaPushCallConfiguration 等
- ✅ **错误处理**：cudaGetLastError, cudaPeekAtLastError, cudaGetErrorString
- ✅ **版本信息**：cudaRuntimeGetVersion, cudaDriverGetVersion
- ✅ **内核启动**：cudaLaunchKernel, cudaConfigureCall（stub，不执行实际计算）

**总计**: 200+ CUDA Runtime API 函数已实现为stub

#### PyTorch 集成状态
- ✅ PyTorch 成功检测到虚拟GPU (8个设备)
- ✅ 设备属性查询正常工作
- ✅ 内存分配和管理正常
- ✅ 张量创建成功
- ✅ 基本的element-wise操作（加法等）
- ❌ **cuBLAS矩阵运算失败** - 需要真实的cuBLAS库
- ❌ **cuDNN卷积操作失败** - 需要真实的cuDNN库

#### CUDA Driver API (libcuda.so.1)
- ✅ 设备管理：`cuInit`, `cuDeviceGet`, `cuDeviceGetCount`, `cuDeviceGetName`, `cuDeviceGetAttribute`
- ✅ 上下文管理：`cuCtxCreate`, `cuCtxDestroy`, `cuCtxSetCurrent`, `cuCtxGetCurrent`
- ✅ 内存管理：`cuMemAlloc`, `cuMemFree`, `cuMemcpy*`, `cuMemGetInfo`
- ✅ 流和事件：`cuStreamCreate`, `cuEventCreate`, `cuStreamSynchronize`
- ✅ 主上下文：`cuDevicePrimaryCtxRetain`, `cuDevicePrimaryCtxRelease`, `cuDevicePrimaryCtxGetState`
- ✅ 内存池：`cuDeviceGetDefaultMemPool`, `cuMemAllocAsync`, `cuMemFreeAsync`
- ✅ 动态查找：`cuGetProcAddress`, `cuGetProcAddress_v2`
- ⚠️ 内部 API：`cuGetExportTable`（返回错误，避免 segfault）

#### CUDA Runtime API (cuda* 函数，在 libcuda.so.1 中)
- ✅ 设备管理：`cudaGetDeviceCount`, `cudaSetDevice`, `cudaGetDevice`, `cudaGetDeviceProperties`
- ✅ 内存管理：`cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaMemset`
- ✅ 同步：`cudaDeviceSynchronize`, `cudaStreamSynchronize`
- ✅ 错误处理：`cudaGetLastError`, `cudaPeekAtLastError`, `cudaGetErrorString`
- ✅ 版本信息：`cudaRuntimeGetVersion`, `cudaDriverGetVersion`
- ✅ 内核启动：`cudaLaunchKernel`（stub，不执行实际计算）

#### NVML API (libnvidia-ml.so.1)
- ✅ 初始化：`nvmlInit`, `nvmlShutdown`
- ✅ 设备查询：`nvmlDeviceGetCount`, `nvmlDeviceGetHandleByIndex`
- ✅ 设备信息：`nvmlDeviceGetName`, `nvmlDeviceGetUUID`, `nvmlDeviceGetMemoryInfo`
- ✅ 时钟信息：`nvmlDeviceGetClockInfo`

### 缺失的功能（导致完整计算失败）

#### 关键限制
- ❌ **cuBLAS库**：矩阵乘法等线性代数运算需要真实的libcublas.so
- ❌ **cuDNN库**：卷积、池化等深度学习操作需要真实的libcudnn.so
- ❌ **实际GPU计算**：内核启动是stub，不执行真实计算

这些限制是**设计上的预期行为**。FakeGPU的目标是:
1. ✅ 模拟GPU设备检测
2. ✅ 模拟内存管理
3. ✅ 测试模型加载和设备代码
4. ❌ 不执行实际的GPU计算

#### 可选扩展（如需要）
要支持实际计算，需要额外实现:
- [ ] cuBLAS stub库 (libcublas.so.12)
- [ ] cuDNN stub库 (libcudnn.so.9)
- [ ] 使用CPU后端执行实际计算

**工作量**: 非常大，不推荐。建议使用真实GPU或CPU模式进行实际计算。

---

## 推荐实施计划

### ✅ 阶段 1：创建基础 fake libcudart.so.12（已完成）

**目标**：实现 PyTorch 初始化所需的最小 Runtime API 集合

**已完成任务**：
- [x] 创建 `src/cuda/cudart_stubs.cpp` 和 `src/cuda/cudart_defs.hpp`
- [x] 实现核心设备管理函数（20+ 个函数）
- [x] 实现核心内存管理函数（30+ 个函数）
- [x] 实现错误处理和版本查询（5+ 个函数）
- [x] 修改 CMakeLists.txt 构建 libcudart.so.12
- [x] 测试 PyTorch 能否成功初始化（`torch.cuda.is_available()`）

**验收标准**：✅ 已达成
```python
import torch
assert torch.cuda.is_available() == True  # ✅ 成功
assert torch.cuda.device_count() == 8     # ✅ 成功
torch.cuda.set_device(0)                  # ✅ 不报错
```

### ✅ 阶段 2：支持基础张量操作（已完成）

**目标**：支持 PyTorch 张量的创建和基本操作

**已完成任务**：
- [x] 实现流管理函数（15+ 个函数）
- [x] 实现事件管理函数（10+ 个函数）
- [x] 实现异步内存操作（5+ 个函数）
- [x] 实现内核启动相关函数（10+ 个函数）
- [x] 测试张量创建和数据传输

**验收标准**：✅ 部分达成
```python
import torch
x = torch.randn(10, 10, device='cuda')    # ✅ 成功
y = torch.randn(10, 10, device='cuda')    # ✅ 成功
z = x + y  # 基本element-wise操作        # ✅ 成功
z_cpu = z.cpu()  # 数据传输               # ✅ 成功
c = torch.matmul(x, y)  # 矩阵乘法        # ❌ 失败（需要cuBLAS）
```

### 🔶 阶段 3：支持完整计算（可选，不推荐）

**状态**: 暂不实施

**原因**：
1. 需要实现cuBLAS库的stub（工作量巨大）
2. 需要实现cuDNN库的stub（工作量巨大）
3. 即使实现stub，也需要CPU后端来执行实际计算
4. 性价比极低，不如直接使用CPU模式或真实GPU

**替代方案**：
- 使用 `device='cpu'` 进行实际计算
- 在有真实GPU的机器上运行
- 使用FakeGPU仅用于设备检测和内存管理测试

---

## 技术债务和改进

### 代码质量
- [ ] 添加更多的错误检查和边界条件处理
- [ ] 统一日志格式和调试输出
- [ ] 添加单元测试覆盖核心功能
- [ ] 改进内存泄漏检测

### 文档
- [ ] 完善 API 文档
- [ ] 添加架构设计文档
- [ ] 编写贡献指南
- [ ] 添加更多使用示例

### 性能
- [ ] 优化内存分配策略
- [ ] 减少不必要的日志输出
- [ ] 改进 GlobalState 的线程安全性

---

## 参考资料

### CUDA 文档
- [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/index.html)
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
- [NVML API](https://docs.nvidia.com/deploy/nvml-api/index.html)

### 相关项目
- [gpuless](https://github.com/ai-dock/gpuless) - 类似的 GPU 模拟项目
- [cuda-api-wrappers](https://github.com/eyalroz/cuda-api-wrappers) - CUDA API 的 C++ 封装

### 调试技巧
```bash
# 查看动态链接
ldd /path/to/pytorch/lib/libtorch_cuda.so

# 跟踪系统调用
strace -e trace=openat python test.py 2>&1 | grep cuda

# 查看符号
nm -D ./build/libcuda.so.1 | grep cuda

# 调试动态链接
LD_DEBUG=bindings,libs LD_PRELOAD=... python test.py
```

---

## 更新日志

### 2025-12-15 - 主要更新

**✅ 已完成的工作**：
1. ✅ 成功创建独立的 libcudart.so.12 库
2. ✅ 实现200+ CUDA Runtime API函数stub
3. ✅ PyTorch成功检测虚拟GPU并初始化
4. ✅ 支持基础张量操作（创建、传输、element-wise运算）
5. ✅ 更新文档说明当前限制

**❌ 已知限制**：
- cuBLAS矩阵运算失败（预期行为）
- cuDNN卷积操作失败（预期行为）
- 不支持实际GPU计算（设计决策）

**📝 文档更新**：
- test/README.md: 添加PyTorch测试说明和限制说明
- TODOs.md: 更新实施计划状态

**🎯 项目目标达成**：
FakeGPU已实现其核心目标：
- ✅ 模拟GPU设备检测
- ✅ 模拟内存管理
- ✅ 支持模型加载测试
- ✅ 支持设备管理代码调试
