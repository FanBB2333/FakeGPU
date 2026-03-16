# 快速参考

## 常用命令

### 编译

```bash
cmake -S . -B build -DENABLE_FAKEGPU_LOGGING=ON
cmake --build build
```

```bash
cmake -S . -B build -DENABLE_FAKEGPU_LOGGING=OFF
cmake --build build
```

### 运行 Python 程序

```bash
./fgpu python3 your_script.py
```

或在 Python 进程内动态启用：

```bash
python3 -c "import fakegpu; fakegpu.init(); import torch; print(torch.cuda.device_count())"
```

### 测试

```bash
./ftest smoke
./ftest python
./ftest all
```

```bash
./test/run_comparison.sh
./test/run_multinode_sim.sh 2
./test/run_ddp_multinode.sh 4
```

## 环境变量

### Linux

```bash
LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH
LD_PRELOAD=./build/libcublas.so.12:./build/libcudart.so.12:./build/libcuda.so.1:./build/libnvidia-ml.so.1
```

### macOS

```bash
DYLD_LIBRARY_PATH=./build:$DYLD_LIBRARY_PATH
DYLD_INSERT_LIBRARIES=./build/libcublas.dylib:./build/libcudart.dylib:./build/libcuda.dylib:./build/libnvidia-ml.dylib
```

### 常用配置

```bash
FAKEGPU_PROFILE=a100
FAKEGPU_DEVICE_COUNT=8
FAKEGPU_PROFILES=a100:4,h100:4
FAKEGPU_MODE=simulate
FAKEGPU_DIST_MODE=simulate
PYTORCH_NO_CUDA_MEMORY_CACHING=1
TORCH_SDPA_KERNEL=math
CUDA_LAUNCH_BLOCKING=1
```

## 问题排查

### 终端光标消失

```bash
reset
```

### 查看 NVML 符号

Linux:

```bash
nm -D ./build/libnvidia-ml.so.1 | grep ' T nvml'
```

macOS:

```bash
nm -gU ./build/libnvidia-ml.dylib | rg '\\bnvml'
```

### 查看动态库依赖

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

## 相关文档

- [快速开始](getting-started.md)
- [项目结构](project-structure.md)
- [分布式模拟使用说明](distributed-sim-usage.md)
