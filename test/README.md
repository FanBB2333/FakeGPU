# FakeGPU 测试脚本

## 当前状态

FakeGPU项目实现了完整的CUDA Driver API和NVML API拦截功能。

### 已实现的功能

- **CUDA Driver API** (libcuda.so.1):
  - 设备管理: cuInit, cuDeviceGet, cuDeviceGetCount, cuDeviceGetName, cuDeviceGetAttribute等
  - 上下文管理: cuCtxCreate, cuCtxSetCurrent, cuDevicePrimaryCtxRetain等
  - 内存管理: cuMemAlloc, cuMemFree, cuMemcpy, cuMemGetInfo等
  - 流和事件: cuStreamCreate, cuEventCreate等
  - 模块和函数: cuModuleLoad, cuLaunchKernel等
  - 动态符号查找: cuGetProcAddress (关键函数)

- **CUDA Runtime API** (通过LD_PRELOAD拦截):
  - cudaMalloc, cudaFree, cudaMemcpy等
  - cudaGetDeviceCount, cudaSetDevice等
  - cudaDeviceSynchronize, cudaStreamSynchronize等

- **NVML API** (libnvidia-ml.so.1):
  - nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetMemoryInfo等

### 当前限制

PyTorch使用真正的libcudart.so.12，它内部会进行深层次的驱动检查。虽然我们的fake libcuda.so.1可以被加载，但libcudart内部可能调用一些我们尚未完全模拟的底层功能。

直接使用CUDA Driver API的程序可以正常工作。

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
LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH \
LD_PRELOAD=./build/libnvidia-ml.so.1 \
python test/test_pytorch_basic.py
```

注意: 由于PyTorch使用真正的libcudart.so.12，可能会遇到初始化错误。

### 3. Transformers测试 (test_transformers.py / test_transformers_simple.py)

测试transformers库的基本功能:

```bash
LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH \
LD_PRELOAD=./build/libnvidia-ml.so.1 \
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

### CUDA初始化错误

如果遇到"CUDA driver error: initialization error":

1. 这是预期的限制，PyTorch需要更完整的Driver API支持
2. 可以尝试使用CPU模式运行测试
3. 查看FakeGPU的日志输出，了解哪些API被调用
4. 根据需要扩展Driver API实现

### 其他问题

1. 确保已构建项目: `cmake -S . -B build && cmake --build build`
2. 检查库文件: `ls -l build/libnvidia-ml.so.1.0.0`
3. 确保已安装依赖: `pip install torch transformers`
4. 查看详细日志以了解具体错误信息

## 开发建议

要支持完整的PyTorch训练，可能需要:

1. 实现更多的CUDA Driver API函数
2. 添加CUDA Runtime API的完整错误处理
3. 实现CUDA Stream和Event管理
4. 添加更详细的调试日志
5. 考虑使用LD_DEBUG=all来追踪动态链接过程
