# FakeGPU 项目结构说明

本仓库通过拦截 CUDA 相关动态库，在没有 GPU 的环境中模拟出可用的 CUDA/NVML/cuBLAS 接口，让 PyTorch 等框架能“看到”虚拟 GPU 并完成基本流程（不做真实计算）。

## 构建与产物
- 核心 CMake 目标：`libnvidia-ml.so.1`（NVML 拦截）、`libcuda.so.1`（Driver API 拦截）、`libcudart.so.12`（Runtime API 拦截）、`libcublas.so.12`（cuBLAS/cuBLASLt 拦截）。
- 默认提供 8 张 “Fake NVIDIA A100-SXM4-80GB” 虚拟设备，显存用系统内存模拟。
- 虚拟 GPU 参数集中保存在根目录 `profiles/*.yaml`，CMake 在编译阶段嵌入到二进制，运行时无需额外配置文件。

## 源码目录
- `src/core/`  
  - `global_state.*`：单例维护虚拟设备列表、当前设备、内存分配记录，并在库加载时自动初始化。  
  - `device.*`：虚拟设备元数据（名称/UUID/PCI/显存占用）。  
  - `logging.hpp`：可选调试日志宏。  
  - `dl_intercept.cpp`：拦截 `dlopen/dlsym/dlvsym`，对 GPU 相关库名返回假句柄并映射到本项目的 fake 符号。
- `src/cuda/`  
  - `cuda_defs.hpp`、`cuda_driver_defs.hpp`、`cudart_defs.hpp`：Driver/Runtime API 的类型与函数声明。  
  - `cuda_driver_stubs.cpp`、`cuda_stubs.cpp`：实现 CUDA Driver API（`cuInit/cuDevice*`、上下文、内存/流/事件等），返回模拟数据并记录内存占用。  
  - `cudart_stubs.cpp`：实现 CUDA Runtime API（设备/内存/流/事件/Graph/纹理等大量 stub），内部调用 Driver 层，内核启动为 no-op。
- `src/cublas/`  
  - `cublas_defs.hpp`：cuBLAS 与 cuBLASLt 的主要枚举/句柄/函数声明。  
  - `cublas_stubs.cpp`：句柄与流管理、GEMM/GemmEx/批量 GEMM 等函数的 stub，实现返回成功或填充随机结果，供 PyTorch 2.x 使用（含 cuBLASLt 关键接口）。
- `src/nvml/`  
  - `nvml_defs.hpp`：NVML 类型与返回码定义。  
  - `nvml_stubs.cpp`：`nvmlInit/Shutdown`、设备查询、显存/温度/功耗等信息的假实现，使用 `GlobalState` 设备数据。
- `src/monitor/`  
  - `monitor.*`：进程退出或显式调用时生成 `fake_gpu_report.json`，记录各虚拟设备的显存使用峰值，避免重复输出（原子标记）。
- `profiles/`  
  - 一组统一格式的 YAML 预设，覆盖不同 Compute Capability（Maxwell → Blackwell），默认使用 `a100.yaml` 生成 8 张虚拟卡。修改或新增文件后重新运行 `cmake -S . -B build` 即可生效。

## 测试与示例
- `test/`  
  - `test_cuda_direct.py`：直接通过 ctypes 调 Driver API 验证核心功能。  
  - `test_comparison.py`、`run_comparison.sh`：真实 GPU 与 FakeGPU 结果对比。  
  - `test_pytorch_basic.py`、`test_pytorch_with_cublas.py`：PyTorch 基础与 cuBLAS 相关路径。  
  - `test_transformers*.py`、`run_transformers_test.sh`、`test_load_qwen2_5.py`：Transformers/大模型加载与简易推理，含 DDP 示例。  
  - 其他脚本演示内存/流/事件/通信等路径。
- 根目录脚本：`build_debug.sh`、`build_release.sh`（构建），`run_test_clean.sh`（清理后跑 PyTorch 测试），`demo_usage.py`、`show_gpu_info.py`（快速演示）。

## 文档与规划
- `README.md`：项目简介、功能列表、构建/测试命令与架构示意。
- `USAGE_CUDART.md`：fake Runtime 的使用方式与已实现/未实现函数清单。
- `docs/cublaslt-fix.md`：cuBLASLt 兼容性问题分析与修复方案。
- `design.md`：初期方案思考。
- `TODOs.md`：实现进度、PyTorch 集成情况、后续计划与已知限制。
- `fake_gpu_report.json`：运行时资源监控输出（由 `src/monitor` 生成）。

## 其他
- `verification/`、`research_poc/`：验证性代码与早期拦截实验。
- 构建产物默认位于 `build/`，运行时需按 README/测试脚本中的顺序 `LD_PRELOAD` 对应的 fake 库。
